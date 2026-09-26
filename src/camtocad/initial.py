"""Startmodel: hoogte en contour van het object uit de bovenaanzichten, met terugvalopties.

Van (bijna) recht boven is het silhouet van een prisma de omtrek van het bovenvlak. Alleen op
de hoogte van dat vlak vallen de terugprojecties uit verschillende standpunten samen; op elke
andere hoogte schuiven ze uit elkaar (parallax). Een zoektocht over de hoogte ('plane sweep')
vindt dus tegelijk het bovenvlak en een eerste hoogte, zonder de grove visual hull te
vertrouwen, want die is bij weinig schuine foto's vaak veel te hoog.

Het resultaat is gewogen: per rastercel telt alleen een foto mee waarin die cel in beeld is.
Lukt het niet, dan volgen terugvalopties (lagere drempel, schuinere bovenaanzichten, hele mat
doorzoeken, visual hull). Elke stap wordt vastgelegd voor het rapport en de debugbeelden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from .calib import Pose
from .hull import VoxelGrid
from .masks import ViewMasks
from .profile import Part2p5D, from_footprint, regularize_angles, remove_short_edges

Views = list[tuple[Pose, ViewMasks]]
Bounds = tuple[float, float, float, float]

MIN_AREA_MM2 = 6.0  # kleiner is geen onderdeel maar ruis


def tilt_deg(pose: Pose, point=None) -> float:
    """Hoek (graden) tussen loodrecht omlaag en de kijkrichting: naar `point` als dat gegeven is.

    Voor parallax telt de kijkrichting naar het object, niet de stand van de camera: een foto recht
    omlaag maar 60 mm naast het onderdeel kijkt er toch ~10° schuin op.
    """
    if point is None:
        d = pose.R[2]  # optische as in mat-coördinaten
    else:
        d = np.asarray(point, float) - pose.center
        d = d / (np.linalg.norm(d) + 1e-12)
    return math.degrees(math.acos(float(np.clip(-d[2], -1.0, 1.0))))


def select_top_views(views: Views, point=None, steep_deg: float = 8.0, max_deg: float = 25.0) -> Views:
    """Liefst echt loodrechte opnamen: bij schuin kijken verdwijnt de doorkijk door gaten."""
    steep = [v for v in views if tilt_deg(v[0], point) <= steep_deg]
    return steep if len(steep) >= 2 else [v for v in views if tilt_deg(v[0], point) <= max_deg]


def vote_map(views: Views, K: np.ndarray, height: float, bounds: Bounds, px: float, unknown: bool = False):
    """Aandeel foto's waarin een punt op het vlak z = height op het object valt.

    Geeft (fractie, oorsprong); NaN waar het punt in minder dan de helft van de foto's op de mat
    valt (buiten de mat is een object niet te zien). Rastercel (rij v, kolom u) ligt op mat-XY
    (x0 + px·u, y0 + px·v). Met `unknown` ook een masker van cellen die in de meeste foto's
    dubbelzinnig zijn (object en mat even donker of licht, en niet opgevuld): daar zegt de fractie
    niets.
    """
    x0, x1, y0, y1 = bounds
    w, h = max(int(np.ceil((x1 - x0) / px)), 1), max(int(np.ceil((y1 - y0) / px)), 1)
    T = np.array([[px, 0, x0], [0, px, y0], [0, 0, 1.0]])
    votes = np.zeros((h, w), np.float32)
    cover = np.zeros((h, w), np.float32)
    amb = np.zeros((h, w), np.float32)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    for pose, m in views:
        Hm = K @ np.column_stack([pose.R[:, 0], pose.R[:, 1], pose.R[:, 2] * height + pose.t]) @ T
        votes += cv2.warpPerspective(m.fg.astype(np.float32), Hm, (w, h), flags=flags, borderValue=0)
        cover += cv2.warpPerspective((m.valid | m.fg).astype(np.float32), Hm, (w, h), flags=flags, borderValue=0)
        if unknown and m.amb is not None:
            amb += cv2.warpPerspective((m.amb & ~m.fg).astype(np.float32), Hm, (w, h), flags=flags, borderValue=0)
    frac = np.full((h, w), np.nan, np.float32)
    ok = cover >= max(0.999, 0.5 * len(views))
    frac[ok] = votes[ok] / cover[ok]
    if unknown:
        return frac, np.array([x0, y0], float), ok & (amb >= 0.5 * np.maximum(cover, 1e-6))
    return frac, np.array([x0, y0], float)


def hull_roi(coarse: VoxelGrid, bounds: Bounds, px: float, grow_mm: float = 4.0) -> np.ndarray:
    """Bovenaanzicht van de grove visual hull (verruimd), op het raster van vote_map.

    De hull is een bovengrens van het object: wat erbuiten valt (ruis, andere dingen op de mat)
    hoort niet bij het object. Bewust de hele voetafdruk en niet de doorsnede op één hoogte: het
    'dak' van een te hoge hull is smaller dan het object en zou de contour afknippen.
    """
    x0, x1, y0, y1 = bounds
    w, h = max(int(np.ceil((x1 - x0) / px)), 1), max(int(np.ceil((y1 - y0) / px)), 1)
    v = coarse.voxel
    layer = coarse.occ.any(axis=2)
    r = int(math.ceil(grow_mm / v))
    if r > 0:
        layer = cv2.dilate(layer.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    xs = x0 + px * np.arange(w)
    ys = y0 + px * np.arange(h)
    i = np.round((xs - coarse.origin[0]) / v).astype(int)
    j = np.round((ys - coarse.origin[1]) / v).astype(int)
    oki = (i >= 0) & (i < layer.shape[0])
    okj = (j >= 0) & (j < layer.shape[1])
    out = np.zeros((h, w), bool)
    sub = layer[np.clip(i, 0, layer.shape[0] - 1)][:, np.clip(j, 0, layer.shape[1] - 1)].T  # (rij = y, kolom = x)
    out[:, :] = sub & okj[:, None] & oki[None, :]
    return out


@dataclass
class Sweep:
    heights: np.ndarray
    agreement: np.ndarray  # |fractie ≥ 0,75| / |fractie ≥ 0,25| binnen de ROI
    area: np.ndarray  # mm² met fractie ≥ 0,75
    best: float | None
    informative: bool


def sweep_height(top: Views, K: np.ndarray, bounds: Bounds, z_hi: float, roi=None, px: float = 0.5,
                 step: float = 0.5) -> Sweep:
    """Hoogte waarop de teruggeprojecteerde bovenaanzichten het best samenvallen."""
    hs = np.arange(step, max(z_hi, 2 * step) + 1e-9, step)
    agree, area = np.zeros(len(hs)), np.zeros(len(hs))
    for k, h in enumerate(hs):
        frac, _ = vote_map(top, K, float(h), bounds, px)
        if roi is not None:
            frac = np.where(roi(float(h), frac.shape), frac, np.nan)
        f = frac[~np.isnan(frac)]
        strong = np.count_nonzero(f >= 0.75)
        weak = np.count_nonzero(f >= 0.25)
        agree[k] = strong / weak if weak else 0.0
        area[k] = strong * px * px
    usable = (area >= max(MIN_AREA_MM2, 0.3 * area.max())) if area.max() > 0 else np.zeros(len(hs), bool)
    if not usable.any():
        return Sweep(hs, agree, area, None, False)
    score = np.where(usable, agree, -1.0)
    k = int(np.argmax(score))
    best = float(hs[k])
    if 0 < k < len(hs) - 1 and usable[k - 1] and usable[k + 1]:
        e0, e1, e2 = agree[k - 1], agree[k], agree[k + 1]
        denom = e0 - 2 * e1 + e2
        if denom < 0:
            best += 0.5 * step * (e0 - e2) / denom
    # informatief als de overeenstemming duidelijk piekt (alle bovenaanzichten vanaf één plek: vlak)
    informative = bool(agree[k] - np.median(agree[usable]) > 0.02)
    return Sweep(hs, agree, area, best, informative)


@dataclass
class Footprint:
    mask: np.ndarray  # bool, rij = y
    origin: np.ndarray
    px: float
    frac: np.ndarray  # stemfractie (NaN = niet in beeld)
    height: float
    method: str
    n_views: int
    threshold: float = 0.5
    unknown: np.ndarray | None = None  # dubbelzinnig in de meeste foto's (zie vote_map)


def _threshold_mask(frac: np.ndarray, threshold: float) -> np.ndarray:
    fp = np.nan_to_num(frac, nan=0.0) > threshold
    return cv2.morphologyEx(fp.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def footprint(top: Views, K: np.ndarray, height: float, bounds: Bounds, px: float = 0.25,
              threshold: float | None = None, roi: np.ndarray | None = None) -> Footprint:
    """Contour op hoogte `height`: cellen die in de meerderheid van de bovenaanzichten object zijn.

    Standaard (threshold None) de buitencontour bij meerderheid 0,5 en de gaten bij een strengere
    0,7: door een gat kijkt maar een deel van de foto's heen (parallax), dus bij 0,7 krijgt een gat
    zijn volle maat, terwijl de buitenrand bij 0,5 het rechtst blijft (bij 0,7 wordt bijvoorbeeld een
    binnenhoek rafelig). Zijn de maskers te rommelig voor 0,7, dan alles bij 0,5.
    """
    frac, origin, unknown = vote_map(top, K, height, bounds, px, unknown=True)
    if roi is not None:
        frac = np.where(roi, frac, np.nan)
        unknown &= roi
    if threshold is None:
        loose, strict = _threshold_mask(frac, 0.5), _threshold_mask(frac, 0.7)
        fp = loose
        if strict.sum() >= 0.85 * loose.sum():
            holes = ndimage.binary_fill_holes(strict) & ~strict
            fp = loose & ~holes
        threshold = 0.5
    else:
        fp = _threshold_mask(frac, threshold)
    return Footprint(fp, origin, px, frac, height, "", len(top), threshold, unknown)


def _usable(fp: Footprint) -> bool:
    """Een bruikbare contour: groot genoeg en niet afgesneden door de rand van het zoekgebied."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fp.mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return False
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] * fp.px ** 2 < MIN_AREA_MM2:
        return False
    x, y, w, h = stats[k, :4]
    H, W = fp.mask.shape
    return x > 0 and y > 0 and x + w < W and y + h < H


def _smooth(mask: np.ndarray, r: int) -> np.ndarray:
    """Morfologisch openen en sluiten met een schijf van straal r: uitsteeksels en inhammen van een
    paar rastercellen (schaduwrand, ruis) verdwijnen, de vorm en de gaten blijven."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k) > 0


def to_part(fp: Footprint, height: float) -> Part2p5D:
    # Geeft een rafelige rand een ongeldige omtrek (bijv. een lusje), dan eerst gladgestreken opnieuw
    # proberen (0,75 en 1,5 mm), in plaats van meteen een terugvaloptie met een lagere drempel, die
    # juist meer schaduw meeneemt.
    for r in (0, 3, 6):
        mask = fp.mask if r == 0 else _smooth(fp.mask, r)
        outer, holes, cutouts = from_footprint(mask, fp.origin, fp.px, lenient_holes=True, unknown=fp.unknown)
        outer, _ = regularize_angles(outer)
        outer = remove_short_edges(outer, 3 * fp.px)
        if outer.is_valid():
            return Part2p5D(height, outer, holes, cutouts)
    raise ValueError("contour levert geen geldige omtrek op")


def clipped_by_view(fp: Footprint) -> bool:
    """Raakt de contour gebied dat in de meeste bovenaanzichten buiten beeld valt?"""
    ring = cv2.dilate(fp.mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return bool(np.any(ring & np.isnan(fp.frac)))


@dataclass
class InitResult:
    part: Part2p5D
    footprint: Footprint
    bounds: Bounds
    sweep: Sweep | None
    z_top: float  # bovengrens van de hoogte (grove visual hull)
    notes: list[str] = field(default_factory=list)  # voor rapport en log
    top: Views = field(default_factory=list)  # gebruikte bovenaanzichten
    threshold: float | None = None  # None = automatisch (zie footprint)
    use_hull: bool = True

    def at_height(self, K: np.ndarray, coarse: VoxelGrid, height: float) -> Part2p5D:
        """Contour opnieuw bepalen op een andere hoogte, met dezelfde foto's en instellingen."""
        if not self.top:  # contour uit de visual hull: alleen de hoogte aanpassen
            p = self.part.copy()
            p.height = height
            return p
        roi = hull_roi(coarse, self.bounds, 0.25) if self.use_hull else None
        fp = footprint(self.top, K, height, self.bounds, threshold=self.threshold, roi=roi)
        if not _usable(fp):
            p = self.part.copy()
            p.height = height
            return p
        try:
            part = to_part(fp, height)
        except ValueError:
            p = self.part.copy()
            p.height = height
            return p
        fp.method = self.footprint.method
        self.footprint = fp
        return part


class InitError(ValueError):
    """Geen bruikbare objectcontour; `details` beschrijft wat er is geprobeerd."""

    def __init__(self, message: str, details: list[str]):
        super().__init__(message)
        self.details = details


def locate(views: Views, K: np.ndarray, coarse: VoxelGrid, board_bounds: Bounds, *,
           height: float | None = None, log=None) -> InitResult:
    """Startmodel uit de bovenaanzichten; `height` gegeven = geen hoogtezoektocht (herinitialisatie).

    Volgorde van pogingen: (1) steile bovenaanzichten binnen de visual hull, (2) lagere drempel,
    (3) alle bovenaanzichten tot 25°, (4) de hele mat zonder hull (als de hull iets anders
    gevonden heeft dan het object), (5) de doorsnede van de visual hull zelf.
    """
    log = log or (lambda m: None)
    tried: list[str] = []
    ijk = np.argwhere(coarse.occ)
    lo = coarse.origin + ijk.min(axis=0) * coarse.voxel
    hi = coarse.origin + ijk.max(axis=0) * coarse.voxel
    z_top = float(hi[2]) + coarse.voxel
    hull_bounds = (lo[0] - 6, hi[0] + 6, lo[1] - 6, hi[1] + 6)
    target = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0])

    steep = select_top_views(views, target)
    wide = [v for v in views if tilt_deg(v[0], target) <= 25.0]
    attempts = [("steile bovenaanzichten", steep, None, True, hull_bounds),
                ("steile bovenaanzichten, lage drempel", steep, 0.3, True, hull_bounds)]
    if len(wide) > len(steep):
        attempts.append(("alle bovenaanzichten tot 25°", wide, None, True, hull_bounds))
    attempts.append(("hele mat, zonder visual hull", wide, None, False, board_bounds))

    for name, top, thr, use_hull, bounds in attempts:
        if not top:
            continue
        roi_sweep = hull_roi(coarse, bounds, 0.5) if use_hull else None
        roi_fn = (lambda h, shape: roi_sweep) if use_hull else None
        sweep = None
        h = height
        if h is None:
            z_hi = z_top + 2.0 if use_hull else max(z_top, 60.0)
            sweep = sweep_height(top, K, bounds, z_hi, roi_fn)
            if sweep.best is None:
                tried.append(f"{name}: geen hoogte gevonden waarop de bovenaanzichten overeenkomen")
                continue
            h = sweep.best if sweep.informative else float(np.clip(0.6 * z_top, 2.0, sweep.best + 20.0))
        b = bounds
        for _ in range(3):  # zoekgebied vergroten als de contour tegen de rand loopt
            roi = hull_roi(coarse, b, 0.25) if use_hull else None
            fp = footprint(top, K, h, b, threshold=thr, roi=roi)
            if _usable(fp) or not fp.mask.any():
                break
            b = (b[0] - 15, b[1] + 15, b[2] - 15, b[3] + 15)
        fp.method = name
        if not _usable(fp):
            area = fp.mask.sum() * fp.px ** 2
            tried.append(f"{name} ({len(top)} foto's, hoogte {h:.1f} mm): "
                         + ("geen object" if area == 0 else f"alleen {area:.0f} mm² of afgesneden contour"))
            continue
        try:
            part = to_part(fp, h)
        except ValueError as e:
            tried.append(f"{name}: {e}")
            continue
        notes = [] if not tried else [f"startcontour via terugvaloptie '{name}'"]
        if tried:
            log("  eerdere pogingen: " + "; ".join(tried))
        if clipped_by_view(fp):
            notes.append("het object valt in de bovenaanzichten deels buiten beeld; de contour kan onvolledig zijn")
        return InitResult(part, fp, b, sweep, z_top, notes + tried, top, thr, use_hull)

    # laatste redmiddel: doorsnede van de grove visual hull (geen gaten, grove randen)
    col = coarse.occ.any(axis=2)
    if col.sum() * coarse.voxel ** 2 >= MIN_AREA_MM2 and height is None:
        px = 0.25
        b = hull_bounds
        w, hgt = int(np.ceil((b[1] - b[0]) / px)), int(np.ceil((b[3] - b[2]) / px))
        xs, ys = b[0] + px * np.arange(w), b[2] + px * np.arange(hgt)
        i = np.clip(np.round((xs - coarse.origin[0]) / coarse.voxel).astype(int), 0, col.shape[0] - 1)
        j = np.clip(np.round((ys - coarse.origin[1]) / coarse.voxel).astype(int), 0, col.shape[1] - 1)
        mask = col[i][:, j].T
        mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # hull is te ruim
        fp = Footprint(mask, np.array([b[0], b[2]]), px, mask.astype(np.float32), 0.6 * z_top,
                       "visual hull (grof)", 0)
        if _usable(fp):
            try:
                part = to_part(fp, max(2.0, 0.6 * z_top))
                tried.append("startcontour uit de grove visual hull: gaten en details kunnen ontbreken")
                log("  " + "; ".join(tried))
                return InitResult(part, fp, b, None, z_top, tried, [], None, True)
            except ValueError as e:
                tried.append(f"visual hull: {e}")
    raise InitError("Geen objectcontour gevonden in de bovenaanzichten", tried)
