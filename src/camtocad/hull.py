"""Visual hull: voxels wegsnijden met silhouetten, daarna oppervlaktepunten met normalen.

Robuuste variant van space carving: een voxel verdwijnt pas als hij in minstens `min_bg`
foto's op *zekere* mat valt (masks.py). Onbekende pixels snijden niets weg. Werkt zonder
textuur op het object en is ongevoelig voor glans — precies de zwakke plekken van MVS
(ARCHITECTURE.md §5.3, [R3c]). Holtes die in geen enkel silhouet zichtbaar zijn (blinde
gaten, kamers) kan een visual hull niet zien; daarvoor is MVS nodig.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .calib import Pose, project
from .masks import ViewMasks


@dataclass
class Blob:
    """Een voorwerp op de mat in de grove visual hull (V12, v0.12)."""

    center: np.ndarray  # (x, y) mm
    size: np.ndarray  # (x, y) mm, omhullende rechthoek
    area: float  # mm², bovenaanzicht
    height: float  # mm
    at_edge: bool  # raakt de rand van de mat


@dataclass
class VoxelGrid:
    origin: np.ndarray  # middelpunt van voxel (0, 0, 0), mm
    voxel: float
    occ: np.ndarray  # bool (nx, ny, nz)
    chosen: Blob | None = None  # het gekozen voorwerp (grove hull)
    others: list = field(default_factory=list)  # andere voorwerpen van betekenis (Blob)

    def centers(self) -> np.ndarray:
        idx = np.indices(self.occ.shape).reshape(3, -1).T
        return self.origin + idx * self.voxel


def _grid(lo: np.ndarray, hi: np.ndarray, voxel: float) -> tuple[np.ndarray, tuple[int, int, int]]:
    shape = tuple(int(np.ceil((b - a) / voxel)) for a, b in zip(lo, hi))
    return lo + voxel / 2, shape


def carve(views: list[tuple[Pose, ViewMasks]], K: np.ndarray, origin: np.ndarray, shape, voxel: float,
          min_bg: int = 2, chunk: int = 1_500_000) -> tuple[np.ndarray, np.ndarray]:
    """Telt per voxel in hoeveel foto's hij op zekere mat resp. op het object valt."""
    n = int(np.prod(shape))
    count_bg = np.zeros(n, np.uint16)
    count_fg = np.zeros(n, np.uint16)
    idx_all = np.arange(n)
    for start in range(0, n, chunk):
        ids = idx_all[start:start + chunk]
        ijk = np.stack(np.unravel_index(ids, shape), axis=1)
        pts = origin + ijk * voxel
        for pose, m in views:
            uv, z = project(pts, pose, K)
            h, w = m.fg.shape
            ui = np.floor(uv[:, 0] + 0.5).astype(np.int64)
            vi = np.floor(uv[:, 1] + 0.5).astype(np.int64)
            ok = (z > 1.0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            sel = np.flatnonzero(ok)
            count_bg[ids[sel[m.bg[vi[sel], ui[sel]]]]] += 1
            count_fg[ids[sel[m.fg[vi[sel], ui[sel]]]]] += 1
    return count_bg.reshape(shape), count_fg.reshape(shape)


# V12 (v0.12): een ander voorwerp telt mee vanaf zoveel mm² in bovenaanzicht en zoveel van het gekozen voorwerp
OTHER_MIN_MM2, OTHER_MIN_FRAC = 60.0, 0.1


def _blob(mask3: np.ndarray, origin: np.ndarray, voxel: float) -> Blob:
    col = mask3.any(axis=2)
    ij = np.argwhere(col)
    lo, hi = ij.min(axis=0), ij.max(axis=0)
    k = np.argwhere(mask3)[:, 2].max()
    at_edge = bool(lo.min() == 0 or hi[0] == col.shape[0] - 1 or hi[1] == col.shape[1] - 1)
    return Blob(origin[:2] + (lo + hi) / 2 * voxel, (hi - lo + 1) * voxel, float(col.sum()) * voxel ** 2,
                float((k + 1) * voxel), at_edge)


def choose_object(occ: np.ndarray, origin: np.ndarray, voxel: float, aim: np.ndarray | None = None) \
        -> tuple[np.ndarray, Blob, list[Blob]]:
    """Welk voorwerp is het onderdeel (V12, v0.12)? Zonder meer het grootste. Liggen er meer voorwerpen van betekenis
    op de mat (een liniaal, een munt, een pen), dan het voorwerp waar de foto's recht van boven op gericht zijn (`aim`,
    (x, y) mm): een liniaal kan groter zijn dan het onderdeel. Geeft (masker, gekozen, de andere)."""
    labels, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    if n == 0:
        raise ValueError("Geen object gevonden: controleer of het object op de mat staat en in beeld is")
    blobs = [_blob(labels == k, origin, voxel) for k in range(1, n + 1)]
    big = max(b.area for b in blobs)
    keep = [k for k, b in enumerate(blobs) if b.area >= max(OTHER_MIN_MM2, OTHER_MIN_FRAC * big)]
    pick = max(keep, key=lambda k: blobs[k].area)
    if aim is not None and len(keep) > 1:
        def dist(b: Blob) -> float:  # afstand van het richtpunt tot de omhullende rechthoek
            return float(np.linalg.norm(np.maximum(np.abs(aim - b.center) - b.size / 2, 0.0)))
        pick = min(keep, key=lambda k: (dist(blobs[k]), -blobs[k].area))
    others = [blobs[k] for k in keep if k != pick]
    return labels == pick + 1, blobs[pick], others


def top_aim(poses, max_tilt_deg: float = 25.0) -> np.ndarray | None:
    """Waar de foto's recht van boven op gericht zijn: de mediaan van de punten waar hun optische as de mat raakt."""
    pts = []
    for pose in poses:
        axis = pose.R.T @ np.array([0.0, 0.0, 1.0])
        if axis[2] >= 0 or np.degrees(np.arccos(min(1.0, -axis[2]))) > max_tilt_deg:
            continue
        c = pose.center
        pts.append(c[:2] + (-c[2] / axis[2]) * axis[:2])
    return np.median(np.array(pts), axis=0) if len(pts) >= 2 else None


def _kept(cbg: np.ndarray, cfg: np.ndarray, min_bg: int, ratio: float) -> np.ndarray:
    """Een voxel blijft staan tenzij genoeg foto's hem op zekere mat zien, ook relatief:
    een paar verkeerd geclassificeerde pixels mogen geen gat in het object snijden."""
    carved = (cbg >= min_bg) & (cbg >= ratio * (cbg.astype(np.int32) + cfg))
    return ~carved


def reconstruct(views: list[tuple[Pose, ViewMasks]], K: np.ndarray, bounds_xy: tuple[float, float, float, float],
                z_max: float = 120.0, voxel: float = 0.5, coarse: float = 2.0, min_bg: int = 2,
                ratio: float = 0.15, fine: bool = True) -> VoxelGrid:
    """Grof-naar-fijn: eerst het object lokaliseren (2 mm), dan fijn uitsnijden rond het object."""
    x0, x1, y0, y1 = bounds_xy
    lo, hi = np.array([x0, y0, 0.0]), np.array([x1, y1, z_max])
    origin, shape = _grid(lo, hi, coarse)
    cbg, cfg = carve(views, K, origin, shape, coarse, min_bg)
    occ = _kept(cbg, cfg, min_bg, ratio) & (cfg >= 2)
    if not occ.any():
        raise ValueError("Geen object gevonden: controleer of het object op de mat staat en in beeld is")
    occ, chosen, others = choose_object(occ, origin, coarse, top_aim([p for p, _ in views]))
    if not fine:
        return VoxelGrid(origin, coarse, occ, chosen, others)
    ijk = np.argwhere(occ)
    flo = origin + ijk.min(axis=0) * coarse - 2.5 * coarse
    fhi = origin + ijk.max(axis=0) * coarse + 2.5 * coarse
    flo[2] = 0.0
    origin, shape = _grid(flo, fhi, voxel)
    cbg, cfg = carve(views, K, origin, shape, voxel, min_bg)
    # Vlak boven de mat projecteert een voxel in elke foto op (bijna) hetzelfde matpunt; boven egale
    # vakken wordt hij dus nooit weggesneden. Daar eisen we daarom ook bewijs van het object.
    z = origin[2] + voxel * np.arange(shape[2])
    near_floor = (z < 1.5)[None, None, :]
    occ = _kept(cbg, cfg, min_bg, ratio) & ((cfg >= 1) | ~near_floor)
    occ = ndimage.binary_fill_holes(choose_object(occ, origin, voxel, chosen.center)[0])
    return VoxelGrid(origin, voxel, occ, chosen, others)


def surface_points(grid: VoxelGrid, sigma_vox: float = 1.0, z_min: float = 0.6,
                   spacing: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Punten op het 0,5-niveau van de (gladgestreken) bezettingsgraad, met naar buiten wijzende normalen."""
    S = ndimage.gaussian_filter(grid.occ.astype(np.float32), sigma_vox, mode="nearest")
    # onderkant: de mat is geen objectoppervlak; bezetting loopt 'door' onder z=0 (mode=nearest)
    gx, gy, gz = np.gradient(S, grid.voxel)
    band = (S > 0.15) & (S < 0.85)
    ijk = np.argwhere(band)
    s = S[band]
    g = np.stack([gx[band], gy[band], gz[band]], axis=1)
    gn = np.linalg.norm(g, axis=1)
    ok = gn > 1e-3
    ijk, s, g, gn = ijk[ok], s[ok], g[ok], gn[ok]
    n = -g / gn[:, None]
    pts = grid.origin + ijk * grid.voxel + ((s - 0.5) / gn)[:, None] * n
    keep = pts[:, 2] > z_min
    pts, n = pts[keep], n[keep]
    spacing = spacing or grid.voxel * 0.5
    key = np.floor(pts / spacing).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return pts[first], n[first]
