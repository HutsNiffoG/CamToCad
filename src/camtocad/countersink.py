"""Verzinkingen herkennen (V16): een gat met een kegel aan het bovenvlak, voor een verzonken schroef.

In het silhouet verandert een verzinking bijna niets (alleen het doorgaande gat eronder is korter), maar in de
grijswaarden is ze een ring rond het gat: de kegel staat schuin en is anders belicht dan het bovenvlak. Per gat het
radiale profiel van de grijswaarden op het bovenvlak (in mm), in de foto's van boven, en daarin de sterkste rand
buiten het gat. Een verzinking is het alleen als die rand rondom ligt (in bijna alle richtingen) en in de meeste
foto's op dezelfde afstand: een kras, een vlek of glans geeft geen ring. De randfit zet de maat daarna precies
(edgefit.measure_inner).

Een smalle verzinking (een faas van een halve millimeter aan de gatrand, v0.11) heeft geen eigen piek: haar rand ligt
zo dicht bij die van het gat dat ze in elkaar overlopen. Daarvoor `narrow`: per veronderstelde breedte de rand meten
zoals de randfit dat doet, en kijken waar hij werkelijk ligt. Bij een faas ligt die gemeten rand steeds op dezelfde
plek, een halve millimeter buiten het gat; bij een gewoon gat vindt de meting alleen de uitloper van de gatrand zelf,
een pixel of wat buiten het gat. Die uitloper reikt bij een donker onderdeel tot een halve millimeter, net zo ver als
een faas: daarom moeten ook de silhouetten de faas steunen (in de schuine foto's kijk je langs een faas verder door
het gat). In de stresstests van v0.12 past een gewoon gat met een faas 7-25% slechter, en een gat met een faas tot 5%
beter of 0,4% slechter: met een faas mag het silhouet hooguit NARROW_SIL_TOL slechter passen. Tot v0.12 moest het
strikt beter, en dan besliste bij een zwart onderdeel één eenheid energie (229 tegen 230).
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
# smalle verzinking (narrow): veronderstelde breedtes (mm buiten het gat), en de gemeten rand moet minstens zoveel
# buiten het gat liggen (mm, en in pixels van de bovenaanzichten: de uitloper van een gewone gatrand meet 1,1-1,2 px,
# een faas van 0,5 mm op 0,25 mm per pixel 1,9-2,5 px), met zoveel metingen
NARROW_W = tuple(np.round(np.arange(0.45, 1.31, 0.05), 2))
NARROW_MIN_MM, NARROW_MIN_PX, NARROW_MIN_MEAS = 0.3, 1.5, 50
NARROW_TOP_DEG = 78.0  # het gat 'van boven' en de pixelmaat: alleen de echte bovenaanzichten
NARROW_SIL_TOL = 0.03  # met een faas mag de silhouetenergie rond het gat zoveel hoger zijn (zie de moduletekst)


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


def _elevation(part: Part2p5D, i: int, v) -> float:
    """Hoe steil de camera van foto `v` op gat i kijkt (graden boven het bovenvlak)."""
    h = part.holes[i]
    z = part.height_at(h.x, h.y) if part.steps else part.height
    c = v.pose.center
    return math.degrees(math.atan2(c[2] - z, math.hypot(c[0] - h.x, c[1] - h.y)))


def _top_view_hole(part: Part2p5D, K: np.ndarray, vd: list, i: int) -> float:
    """Diameter van gat i uit alleen de foto's van boven: daar begrenst het nauwste stuk de doorkijk, ook bij een faas.
    De schuine foto's kijken langs de faas en maken een gewoon gefit gat te groot."""
    from . import counterbore, silhouette

    h = part.holes[i]
    views = [v for v in counterbore._local_views(part, i, vd, K) if _elevation(part, i, v) >= NARROW_TOP_DEG]
    if len(views) < 2:
        return h.d
    fitted, _, _ = silhouette.refine(part, K, views, max_evals=60, only={f"hd{i}"})
    return fitted.holes[i].d


def narrow(part: Part2p5D, K: np.ndarray, vd: list, i: int) -> tuple[float | None, str]:
    """Een smalle verzinking (faas) aan gat i (zie de moduletekst): (diameter aan het bovenvlak of None, toelichting).

    Per breedte uit NARROW_W een verzinking verondersteld, de rand gemeten (edgefit.measure_inner), en per meting
    teruggerekend waar de rand werkelijk ligt (de modelrand min het residu, in mm). Een faas: die plek ligt over de
    breedtes heen vast, minstens NARROW_MIN_MM en NARROW_MIN_PX buiten het gat in de foto's van boven."""
    from dataclasses import replace

    from . import edgefit

    h = part.holes[i]
    d0 = _top_view_hole(part, K, vd, i)
    rims, mm_px, n_total = [], [], 0
    for w in NARROW_W:
        trial = part.copy()
        trial.holes[i] = replace(h, d=d0, csk=d0 + 2 * w, cb=0.0, cb_depth=0.0)
        if not trial.is_valid():
            continue
        lay = edgefit.layout(trial)
        ie = edgefit.inner_edges(trial, lay)
        sl = ie.groups.get(("verzinking", i))
        if sl is None:
            continue
        res = []
        for v, (idx, pos, nrm, _wt) in zip(vd, edgefit.measure_inner(trial, K, vd, lay)):
            sel = (idx >= sl.start) & (idx < sl.stop)
            if not sel.any():
                continue
            P = ie.P[idx[sel]]
            uv, _ = project(P, v.pose, K)
            uv2, _ = project(P + np.column_stack([0.1 * ie.D[idx[sel]], np.zeros(len(P))]), v.pose, K)
            px_per_mm = np.linalg.norm(uv2 - uv, axis=1) / 0.1
            r_px = np.sum((uv - [v.x0, v.y0] - pos[sel]) * nrm[sel], axis=1)
            res += list(r_px / np.maximum(px_per_mm, 1e-6))
            if _elevation(part, i, v) >= NARROW_TOP_DEG:
                mm_px += list(1.0 / np.maximum(px_per_mm, 1e-6))
        if len(res) >= NARROW_MIN_MEAS:
            rims.append(d0 / 2 + w - float(np.median(res)))
            n_total += len(res)
    if len(rims) < 3:
        return None, f"faas: te weinig metingen ({n_total})"
    rim = float(np.median(rims))
    need = max(NARROW_MIN_MM, NARROW_MIN_PX * float(np.median(mm_px)) if mm_px else NARROW_MIN_MM)
    width = rim - d0 / 2
    note = (f"faas: rand op Ø {2 * rim:.2f} ({width:.2f} mm buiten het gat Ø {d0:.2f} van boven; nodig {need:.2f}), "
            f"spreiding {float(np.std(rims)):.2f} mm over {len(rims)} breedtes, {n_total} metingen")
    if width < need or float(np.std(rims)) > 0.15:
        return None, note
    e_plain, e_csk = _silhouette_support(part, K, vd, i, 2 * rim)
    note += f"; silhouetten {e_plain:.0f} als gewoon gat, {e_csk:.0f} met faas"
    if not e_csk < (1.0 + NARROW_SIL_TOL) * e_plain:
        return None, note
    return 2 * rim, note


def _silhouette_support(part: Part2p5D, K: np.ndarray, vd: list, i: int, dk: float) -> tuple[float, float]:
    """Silhouetenergie rond gat i als gewoon gat en met een verzinking Ø `dk`, elk met de best passende gatdiameter
    (zie counterbore.detect)."""
    from dataclasses import replace

    from . import counterbore, silhouette

    h = part.holes[i]
    views = counterbore._local_views(part, i, vd, K)

    def energy(d: float, csk: float = 0.0) -> float:
        trial = part.copy()
        trial.holes[i] = replace(h, d=d, csk=csk, cb=0.0, cb_depth=0.0)
        return silhouette.energy(trial, K, views) if trial.is_valid() else math.inf

    ds = np.arange(h.d - 0.5, h.d + 0.201, 0.05)
    return (min(energy(float(d)) for d in ds),
            min((energy(float(d), dk) for d in ds if dk > d + 0.2), default=math.inf))


def detect(part: Part2p5D, K: np.ndarray, vd: list, log=None, weak: set | None = None) -> dict[int, float]:
    """Gaten met een verzinking: {index: diameter aan het bovenvlak (mm)}. Zie de moduletekst. `weak`: gaten zonder
    bewijs rond de rand (edgefit.evidence); daar is het gat van boven niet goed te zien, en dan lijkt zijn eigen rand
    op een faas: geen smalle verzinking."""
    found = {}
    for i, h in enumerate(part.holes):
        if h.csk > 0 or h.cb > 0 or h.blind:
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
        elif not (weak and i in weak):  # geen eigen ring: misschien een smalle verzinking (faas) tegen de gatrand aan
            dk, note = narrow(part, K, vd, i)
            if log:
                log(f"gat {i + 1}: {'smalle verzinking' if dk else 'geen smalle verzinking'}; {note}")
            if dk:
                found[i] = dk
    return found
