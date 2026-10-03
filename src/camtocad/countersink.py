"""Verzinkingen herkennen (V16): een gat met een kegel aan het bovenvlak, voor een verzonken schroef.

In het silhouet verandert een verzinking bijna niets (alleen het doorgaande gat eronder is korter), maar in de
grijswaarden is ze een ring rond het gat: de kegel staat schuin en is anders belicht dan het bovenvlak. Per gat het
radiale profiel van de grijswaarden op het bovenvlak (in mm), in de foto's van boven, en daarin de sterkste rand
buiten het gat. Een verzinking is het alleen als die rand rondom ligt (in bijna alle richtingen) en in de meeste
foto's op dezelfde afstand: een kras, een vlek of glans geeft geen ring. De randfit zet de maat daarna precies
(edgefit.measure_inner).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .calib import project
from .profile import Part2p5D

MIN_ELEV_DEG = 60.0  # foto's van boven: de kegel is rondom te zien, en het gat eronder verschuift weinig
R_STEP_MM = 0.05
MIN_GAP_MM = 0.4  # de ring ligt minstens zo ver buiten het gat (de rand van het gat zelf is ook een rand)
MAX_WIDTH_MM = 6.0  # breedste verzinking buiten het gat (DIN 74: M10 heeft 4,7 mm)
# De rand is minstens zoveel sterker dan de rest van het profiel, en per richting minstens DIR_RATIO. Een gewoon gat
# heeft buiten zijn rand geen eigen piek (alleen de uitloper van die rand), ook niet in de zwarte stresstests; een
# verzinking op een zwart onderdeel boven een zwart vak gaf 2,4x
PEAK_RATIO, DIR_RATIO = 2.0, 1.5
MIN_ANGLES = 0.7  # in dit deel van de richtingen
MIN_VIEWS = 0.6  # in dit deel van de foto's op dezelfde afstand (± PEAK_TOL_MM)
PEAK_TOL_MM = 0.3


def _room(part: Part2p5D, i: int) -> float:
    """Hoeveel mm er rond gat i vrij bovenvlak is: tot de buitenrand en de andere gaten, min 1 mm."""
    h = part.holes[i]
    a = part.outer.outline()
    b = np.roll(a, -1, axis=0)
    ab = b - a
    t = np.clip(np.sum((np.array([h.x, h.y]) - a) * ab, axis=1) / np.maximum(np.sum(ab * ab, axis=1), 1e-12), 0, 1)
    near = a + t[:, None] * ab  # het dichtstbijzijnde punt op elke rand van de contour
    edge = float(np.min(np.hypot(near[:, 0] - h.x, near[:, 1] - h.y))) if len(a) else np.inf
    others = [math.hypot(o.x - h.x, o.y - h.y) - o.d / 2 for j, o in enumerate(part.holes) if j != i]
    others += [math.hypot(s.x - h.x, s.y - h.y) - s.width / 2 for s in part.slots]
    return float(min([edge] + others)) - 1.0


def radial_gradient(part: Part2p5D, K: np.ndarray, vd: list, i: int, n_angles: int = 72):
    """Voor gat i: de afstanden r (mm) en per foto van boven |dI/dr| (grijswaarden per mm) per richting en r."""
    h = part.holes[i]
    r_max = min(h.d / 2 + MAX_WIDTH_MM, _room(part, i))
    if r_max <= h.d / 2 + MIN_GAP_MM + 0.5:
        return None, []
    z = part.height_at(h.x, h.y) if part.steps else part.height
    r = np.arange(h.d / 2 + 0.1, r_max, R_STEP_MM)
    a = (np.arange(n_angles) + 0.5) / n_angles * 2 * math.pi
    X = h.x + np.outer(np.cos(a), r)
    Y = h.y + np.outer(np.sin(a), r)
    P = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    out = []
    for v in vd:
        g = getattr(v, "gray", None)
        if g is None:
            continue
        c = v.pose.center
        if math.degrees(math.atan2(c[2] - z, math.hypot(c[0] - h.x, c[1] - h.y))) < MIN_ELEV_DEG:
            continue
        uv, depth = project(P, v.pose, K)
        uv = uv - [v.x0, v.y0]
        hh, ww = g.shape
        if np.any(depth <= 0) or uv[:, 0].min() < 1 or uv[:, 1].min() < 1 or uv[:, 0].max() > ww - 2 \
                or uv[:, 1].max() > hh - 2:
            continue
        prof = ndimage.map_coordinates(g.astype(np.float32), [uv[:, 1], uv[:, 0]], order=1).reshape(X.shape)
        if not np.all(np.isfinite(prof)):
            continue
        prof = ndimage.gaussian_filter1d(prof, 1.0, axis=1)  # ~0,05 mm: tegen ruis, de rand blijft scherp
        out.append(np.abs(np.gradient(prof, R_STEP_MM, axis=1)))
    return r, out


def detect(part: Part2p5D, K: np.ndarray, vd: list, log=None) -> dict[int, float]:
    """Gaten met een verzinking: {index: diameter aan het bovenvlak (mm)}. Zie de moduletekst."""
    found = {}
    for i, h in enumerate(part.holes):
        if h.csk > 0:
            continue
        r, grads = radial_gradient(part, K, vd, i)
        if r is None or len(grads) < 2:
            if log:
                log(f"gat {i + 1}: geen verzinking te zoeken ({'te weinig ruimte rond het gat' if r is None else ''}"
                    f"{'' if r is None else f'{len(grads)} foto van boven'})")
            continue
        search = r >= h.d / 2 + MIN_GAP_MM
        per_view = [np.mean(gv, axis=0) for gv in grads]  # gemiddeld over de richtingen
        norm = [pv / max(float(np.median(pv[search])), 1e-6) for pv in per_view]
        mean = np.mean(norm, axis=0)
        idx = np.flatnonzero(search)
        # de sterkste eigen piek: een lokaal maximum met een dal tussen haar en de rand van het gat (de uitloper
        # van die rand is bij een donker onderdeel vaak sterker dan de verzinking zelf)
        tops = [j for j in idx[1:-1] if mean[j] >= mean[j - 1] and mean[j] >= mean[j + 1]
                and float(np.min(mean[:j])) <= 0.5 * mean[j]]
        peak = bool(tops)
        k = max(tops, key=lambda j: mean[j]) if tops else int(idx[np.argmax(mean[search])])
        strength = float(mean[k] / max(float(np.median(mean[search])), 1e-6))
        # in de meeste foto's op dezelfde afstand (voorbij het dal)
        jv = int(np.argmin(mean[:k])) if k > 0 else 0
        peaks = np.array([r[jv + int(np.argmax(pv[jv:]))] for pv in per_view])
        agree = float(np.mean(np.abs(peaks - r[k]) <= PEAK_TOL_MM))
        # rondom: per richting (over de foto's) een duidelijke rand bij r[k]
        near = np.abs(r - r[k]) <= 0.15
        dirs = np.mean([gv[:, near].max(axis=1) / max(float(np.median(gv[:, search])), 1e-6) for gv in grads],
                       axis=0)
        around = float(np.mean(dirs > DIR_RATIO))
        ok = peak and strength >= PEAK_RATIO and agree >= MIN_VIEWS and around >= MIN_ANGLES
        if log:
            log(f"gat {i + 1}: {'verzinking' if ok else 'geen verzinking'}; sterkste rand buiten het gat op "
                f"r = {r[k]:.2f} mm (Ø {2 * r[k]:.2f}){'' if peak else ' (geen eigen piek)'}, {strength:.1f}x zo sterk "
                f"als de rest, rondom {around:.0%}, in {agree:.0%} van {len(grads)} foto's op dezelfde plaats")
        if ok:
            found[i] = float(2 * r[k])
    return found
