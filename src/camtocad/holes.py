"""Gemiste gaten toevoegen na de fit (V4): waar de bovenaanzichten door het bovenvlak heen mat zien.

De fit kan de vorm van het model verfijnen, maar geen gat toevoegen: een gat dat in de startcontour
ontbrak (klein, of zijn rand viel weg tegen een even donker stuk mat), blijft dan weg. Hier wordt het
bovenvlak (z = hoogte) teruggeprojecteerd in de foto's recht van boven. Een plek waar in minstens twee
van die foto's, en in de meerderheid, zekere mat te zien is, is een kandidaatgat; een cluster van zulke
cellen wordt een cirkel (middelpunt en oppervlak). De pipeline houdt hem alleen als het model er na een
korte fit duidelijk beter door past.

Ook een niet-ronde uitsparing die wél rond blijkt (een gat met een rafelige rand in de startcontour),
wordt een gat.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .calib import project
from .initial import tilt_deg
from .profile import Hole, Part2p5D, fit_opening
from .silhouette import render

CELL_MM = 0.25
MIN_D_MM = 1.5


def _grid_mask(part: Part2p5D, cell: float, margin_mm: float = 1.0):
    """Cellen (middelpunten, mat-XY) binnen het bovenvlak: niet op of naast de rand, niet in een bestaand gat."""
    outline = part.outer.outline(4.0)
    pad = margin_mm + 2 * cell  # ruimte rondom, anders erodeert de marge niet vanaf de rand van het raster
    lo, hi = outline.min(axis=0) - pad, outline.max(axis=0) + pad
    w, h = math.ceil((hi[0] - lo[0]) / cell), math.ceil((hi[1] - lo[1]) / cell)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round((outline - lo) / cell - 0.5).astype(np.int32)], 1)
    for hl in part.holes:
        c = (np.array([hl.x, hl.y]) - lo) / cell - 0.5
        cv2.circle(mask, tuple(int(round(v)) for v in c), math.ceil((hl.d / 2 + margin_mm) / cell), 0, -1)
    for cu in list(part.cutouts) + [s.outline() for s in part.slots]:
        cv2.fillPoly(mask, [np.round((cu - lo) / cell - 0.5).astype(np.int32)], 0)
    r = max(1, round(margin_mm / cell))
    mask = cv2.erode(mask, np.ones((2 * r + 1, 2 * r + 1), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask > 0, lo


def candidates(part: Part2p5D, K: np.ndarray, vd: list, max_tilt_deg: float = 25.0, min_views: int = 2,
               cell: float = CELL_MM, min_d: float = MIN_D_MM) -> list[Hole]:
    """Kandidaatgaten: clusters in het bovenvlak waar de bovenaanzichten zekere mat zien.

    `vd`: de uitsneden van silhouette.prepare (object, zekere mat, ROI-verschuiving); alleen de foto's die
    onder hooguit `max_tilt_deg` op het midden van het bovenvlak kijken tellen mee.
    """
    inside, lo = _grid_mask(part, cell)
    if not inside.any():
        return []
    ys, xs = np.nonzero(inside)
    pts = np.column_stack([lo[0] + (xs + 0.5) * cell, lo[1] + (ys + 0.5) * cell, np.full(len(xs), part.height)])
    center = pts.mean(axis=0)
    top = [v for v in vd if tilt_deg(v.pose, center) <= max_tilt_deg]
    if len(top) < min_views:
        return []
    mat = np.zeros(len(pts), np.int32)
    seen = np.zeros(len(pts), np.int32)
    for v in top:
        uv, z = project(pts, v.pose, K)
        h, w = v.fg.shape
        ui = np.floor(uv[:, 0] - v.x0 + 0.5).astype(np.int64)
        vi = np.floor(uv[:, 1] - v.y0 + 0.5).astype(np.int64)
        ok = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        sel = np.flatnonzero(ok)
        bg = v.bg[vi[sel], ui[sel]]
        fg = v.fg[vi[sel], ui[sel]]
        mat[sel] += bg
        seen[sel] += bg | fg
    hit = (mat >= min_views) & (mat >= 0.5 * np.maximum(seen, 1))
    grid = np.zeros(inside.shape, np.uint8)
    grid[ys[hit], xs[hit]] = 1
    n, _, stats, cent = cv2.connectedComponentsWithStats(grid, connectivity=8)
    out = []
    min_area = math.pi * (min_d / 2) ** 2 / cell ** 2
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        # door parallax ziet een schuin bovenaanzicht maar een deel van het gat: de fit maakt hem op maat
        d = 2 * math.sqrt(area / math.pi) * cell
        cx, cy = lo + (cent[k] + 0.5) * cell
        out.append(Hole(float(cx), float(cy), float(d)))
    return out


# V14 (v0.12): een restcluster telt vanaf dit oppervlak en deze breedte (de grootste ingeschreven cirkel); niet binnen
# MARGIN_MM van de rand (daar zit de onzekerheid van de rand zelf), en in een foto pas TOL_PX buiten het modelsilhouet
CLUSTER_MM2, CLUSTER_WIDTH_MM, CLUSTER_MARGIN_MM, CLUSTER_TOL_PX = 2.0, 1.0, 0.75, 2


def residual_clusters(part: Part2p5D, K: np.ndarray, vd: list, max_tilt_deg: float = 25.0, min_views: int = 2,
                      share: float = 0.6, cell: float = CELL_MM, band_mm: float = 8.0) -> list[dict]:
    """Plekken waar de foto's van boven iets anders zien dan het model (V14, v0.12): zekere mat binnen het
    modelsilhouet (een gemiste inham of een gemist gat) of object erbuiten (een gemiste uitstulping, of iets dat
    tegen het onderdeel aan ligt). Het bovenvlak (z = hoogte) wordt per cel van `cell` mm in die foto's
    geprojecteerd; een cel telt als minstens `min_views` foto's en een aandeel `share` van de foto's die er iets
    zien het eens zijn. Geeft per cluster {soort: 'inham' of 'uitstulping', x, y (mm, mat), oppervlak (mm²),
    breedte (mm), fotos}, de grootste eerst."""
    outline = part.outer.outline(4.0)
    pad = band_mm + 2 * cell
    lo, hi = outline.min(axis=0) - pad, outline.max(axis=0) + pad
    w, h = math.ceil((hi[0] - lo[0]) / cell), math.ceil((hi[1] - lo[1]) / cell)
    inside = np.zeros((h, w), np.uint8)
    cv2.fillPoly(inside, [np.round((outline - lo) / cell - 0.5).astype(np.int32)], 1)
    d_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5) * cell
    d_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5) * cell
    zone = ((inside > 0) & (d_in > CLUSTER_MARGIN_MM)) | ((inside == 0) & (d_out > CLUSTER_MARGIN_MM)
                                                        & (d_out < band_mm))
    ys, xs = np.nonzero(zone)
    xy = np.column_stack([lo[0] + (xs + 0.5) * cell, lo[1] + (ys + 0.5) * cell])
    z = np.full(len(xy), part.height)
    for st in part.steps:  # bij een trede: de bovenkant daar
        z[(xy @ st.normal() > st.offset) & (z > st.height)] = st.height
    pts = np.column_stack([xy, z])
    center = np.array([*(outline.min(axis=0) + outline.max(axis=0)) / 2, part.height])
    top = [v for v in vd if tilt_deg(v.pose, center) <= max_tilt_deg]
    if len(top) < min_views:
        return []
    k = np.ones((2 * CLUSTER_TOL_PX + 1, 2 * CLUSTER_TOL_PX + 1), np.uint8)
    votes = {"inham": np.zeros(len(pts), np.int32), "uitstulping": np.zeros(len(pts), np.int32)}
    seen = np.zeros(len(pts), np.int32)
    for v in top:
        P = render(part, K, v)
        bump = v.fg & (cv2.dilate(P, k) == 0)
        notch = v.bg & (cv2.erode(P, k, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0)
        uv, depth = project(pts, v.pose, K)
        hh, ww = v.fg.shape
        ui = np.floor(uv[:, 0] - v.x0 + 0.5).astype(np.int64)
        vi = np.floor(uv[:, 1] - v.y0 + 0.5).astype(np.int64)
        sel = np.flatnonzero((depth > 0) & (ui >= 0) & (ui < ww) & (vi >= 0) & (vi < hh))
        votes["uitstulping"][sel] += bump[vi[sel], ui[sel]]
        votes["inham"][sel] += notch[vi[sel], ui[sel]]
        seen[sel] += v.fg[vi[sel], ui[sel]] | v.bg[vi[sel], ui[sel]]
    out = []
    for kind, n_vote in votes.items():
        hit = (n_vote >= min_views) & (n_vote >= share * np.maximum(seen, 1))
        grid = np.zeros((h, w), np.uint8)
        grid[ys[hit], xs[hit]] = 1
        n, labels, stats, cent = cv2.connectedComponentsWithStats(grid, connectivity=8)
        for c in range(1, n):
            area = stats[c, cv2.CC_STAT_AREA] * cell ** 2
            blob = (labels == c).astype(np.uint8)
            width = 2 * float(cv2.distanceTransform(np.pad(blob, 1), cv2.DIST_L2, 5).max()) * cell
            if area < CLUSTER_MM2 or width < CLUSTER_WIDTH_MM:
                continue
            x, y = lo + (cent[c] + 0.5) * cell
            in_cl = labels[ys, xs] == c
            out.append({"soort": kind, "x": round(float(x), 2), "y": round(float(y), 2), "oppervlak": round(area, 2),
                        "breedte": round(width, 2), "fotos": int(np.median(n_vote[in_cl]))})
    return sorted(out, key=lambda c: -c["oppervlak"])


def slot_cutouts(part: Part2p5D, px: float = 0.25, loose: bool = False) -> tuple[Part2p5D, int]:
    """Uitsparingen met de vorm van een sleuf of rechthoek worden een Slot (V15); geeft (model, aantal).
    `loose`: ook als de polygoon er maar ruw op lijkt (tot 1 mm of 30% van de breedte); de pipeline houdt
    het resultaat alleen als het model na een korte fit minstens even goed past."""
    keep, slots = [], list(part.slots)
    for cu in part.cutouts:
        s = fit_opening(cu, px)[0] if not loose else None
        if s is None and loose:
            w = min(np.ptp(np.asarray(cu, float), axis=0))
            s = fit_opening(cu, px, max_dev=max(1.0, 0.3 * w))[0]
        if s is None:
            keep.append(cu)
        else:
            slots.append(s)
    n = len(slots) - len(part.slots)
    if not n:
        return part, 0
    out = part.copy()
    out.slots, out.cutouts = slots, keep
    return out, n


def round_cutouts(part: Part2p5D, max_dev: float = 0.12) -> tuple[Part2p5D, int]:
    """Uitsparingen die (bijna) rond zijn, worden gaten; geeft (model, aantal omgezet)."""
    keep, holes, n = [], list(part.holes), 0
    for cu in part.cutouts:
        c = cu.mean(axis=0)
        r = np.linalg.norm(cu - c, axis=1)
        if len(cu) >= 8 and r.mean() > 0 and (r.max() - r.min()) / r.mean() < max_dev:
            area = abs(0.5 * float(np.dot(cu[:, 0], np.roll(cu[:, 1], -1)) - np.dot(cu[:, 1], np.roll(cu[:, 0], -1))))
            holes.append(Hole(float(c[0]), float(c[1]), 2 * math.sqrt(area / math.pi)))
            n += 1
        else:
            keep.append(cu)
    if not n:
        return part, 0
    out = part.copy()
    out.holes, out.cutouts = holes, keep
    return out, n
