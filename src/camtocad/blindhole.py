"""Blinde gaten herkennen (v0.11): een gat dat niet door het onderdeel heen gaat.

In het silhouet is een blind gat er niet: in geen enkele foto is er de mat door te zien, en dus vonden de pixelfit en
holes.candidates het niet. In de grijswaarden wel. Zijn bovenrand ligt op het bovenvlak, en in de schuine foto's is
de wand aan de overkant een band die anders belicht is dan het bovenvlak. Per foto de sterkte van de randen
(gradiënt, in lineair licht) op het bovenvlak afgebeeld (een homografie per foto) en gemiddeld over de foto's: de
bovenrand van een gat ligt in elke foto op dezelfde plek en telt op; de onderrand van de wand ligt per foto ergens
anders (parallax) en vervaagt. Daarin cirkels (Hough), binnen het bovenvlak en niet bij een gat, sleuf, uitsparing of
trede die er al is.

Een kandidaat telt alleen als ook de onderrand van de wand te vinden is (edgefit.measure_inner, in de schuine foto's
door het gat heen): een opgedrukte of gekraste ring op het bovenvlak heeft geen wand. Die onderrand geeft ook de
diepte (op een rooster); de randfit zet daarna alles precies.
"""

from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np

from . import edgefit
from .calib import project
from .profile import Hole, Part2p5D

RES_MM = 0.1  # raster op het bovenvlak
MIN_ELEV_DEG = 30.0  # foto's waarin het bovenvlak goed te zien is
R_MIN_MM, R_MAX_MM = 1.0, 8.0  # straal van een blind gat
MARGIN_MM = 1.0  # zo ver van de buitenrand, een gat, sleuf of trede die er al is
RING_RATIO, RING_AROUND = 3.0, 0.6  # de ring is zoveel sterker dan zijn omgeving, in dit deel van de richtingen
DEPTHS = np.arange(0.1, 0.951, 0.05)  # diepte, als deel van de hoogte
MIN_FLOOR_MEAS, MIN_FLOOR_VIEWS = 60, 3  # metingen van de onderrand van de wand (en in zoveel foto's)


def _linear(g: np.ndarray, tone) -> np.ndarray:
    """Grijswaarden terug naar lineair licht (de toonkromme van de camera; tone.py)."""
    g = g.astype(np.float32)
    if np.isscalar(tone) and tone == 1.0:
        return g
    return 255.0 * np.clip(g / 255.0, 1e-6, 1.5) ** (1.0 / np.asarray(tone, np.float32))


def edge_map(part: Part2p5D, K: np.ndarray, vd: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gemiddelde randsterkte op het bovenvlak (zie de moduletekst): (kaart, oorsprong in mm, aantal foto's per
    rastercel). Rastercel (rij j, kolom i) ligt op (oorsprong + (i + ½, j + ½) · RES_MM, hoogte)."""
    ring = part.outer.outline()
    lo, hi = ring.min(axis=0) - 1.0, ring.max(axis=0) + 1.0
    W, H = (int(math.ceil(v)) for v in (hi - lo) / RES_MM)
    acc, cnt = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    z = part.height
    c0 = (lo + hi) / 2
    for v in vd:
        g = getattr(v, "gray", None)
        if g is None:
            continue
        cam = v.pose.center
        if math.degrees(math.atan2(cam[2] - z, math.hypot(cam[0] - c0[0], cam[1] - c0[1]))) < MIN_ELEV_DEG:
            continue
        gl = cv2.GaussianBlur(_linear(g, getattr(v, "tone", 1.0)), (0, 0), 0.7)
        mag = cv2.magnitude(cv2.Sobel(gl, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gl, cv2.CV_32F, 0, 1, ksize=3))
        inside = cv2.erode(v.fg.astype(np.uint8), np.ones((5, 5), np.uint8))
        if not inside.any():
            continue
        mag /= max(float(np.percentile(mag[inside > 0], 99)), 1e-3)
        R, t = v.pose.R, v.pose.t
        Hg = K @ np.column_stack([R[:, 0] * RES_MM, R[:, 1] * RES_MM,
                                  R @ np.array([lo[0] + RES_MM / 2, lo[1] + RES_MM / 2, z]) + t])
        T = np.array([[1.0, 0.0, -v.x0], [0.0, 1.0, -v.y0], [0.0, 0.0, 1.0]]) @ Hg
        flags = cv2.WARP_INVERSE_MAP
        warped = cv2.warpPerspective(mag, T, (W, H), flags=cv2.INTER_LINEAR | flags, borderValue=0)
        seen = cv2.warpPerspective(inside, T, (W, H), flags=cv2.INTER_NEAREST | flags, borderValue=0) > 0
        acc[seen] += warped[seen]
        cnt[seen] += 1
    out = np.where(cnt >= 3, acc / np.maximum(cnt, 1), 0.0).astype(np.float32)
    return out, lo, cnt


def _free_area(part: Part2p5D, lo: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Waar op het bovenvlak een blind gat kan liggen (zie MARGIN_MM), als masker op het raster van edge_map."""
    def px(P) -> np.ndarray:
        return np.round((np.asarray(P, float) - lo) / RES_MM - 0.5).astype(np.int32)

    m = np.zeros(shape, np.uint8)
    inset = MARGIN_MM + (part.top_edge.size if part.top_edge is not None else 0.0)
    top = part.outer.inset(inset) if part.outer.kind == "polygon" else None
    if part.outer.kind == "circle":
        c = px(part.outer.center)
        cv2.circle(m, (int(c[0]), int(c[1])), int((part.outer.radius - inset) / RES_MM), 1, -1)
    elif top is not None and top.is_valid():
        cv2.fillPoly(m, [px(top.outline())], 1)
    for h in part.holes:
        c = px([h.x, h.y])
        cv2.circle(m, (int(c[0]), int(c[1])), int((h.top_d / 2 + MARGIN_MM) / RES_MM), 0, -1)
    for s in part.slots:
        cv2.fillPoly(m, [px(s.outline())], 0)
    for poly in part.cutouts:
        cv2.fillPoly(m, [px(poly)], 0)
    m = cv2.erode(m, np.ones((3, 3), np.uint8)) if (part.slots or part.cutouts) else m
    for st in part.steps:  # de trede: haar vlak en de lijn erlangs
        q = part.step_crossings(st)
        if q is not None:
            a, b = px(q[0]), px(q[1])
            cv2.line(m, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 0, int(2 * MARGIN_MM / RES_MM) + 1)
    return m > 0


def _sample(M: np.ndarray, cx: float, cy: float, r: float, n: int = 72) -> np.ndarray:
    """De kaart op een cirkel (middelpunt en straal in rastercellen)."""
    a = (np.arange(n) + 0.5) / n * 2 * math.pi
    h, w = M.shape
    x = np.clip(cx + r * np.cos(a), 0, w - 1)
    y = np.clip(cy + r * np.sin(a), 0, h - 1)
    return cv2.remap(M, x.astype(np.float32)[None, :], y.astype(np.float32)[None, :], cv2.INTER_LINEAR)[0]


def _ring(M: np.ndarray, cx: float, cy: float, r: float, n: int = 72) -> tuple[float, float]:
    """(sterkte van de ring t.o.v. de omgeving, deel van de richtingen waar de ring duidelijk is), in rastercellen."""
    def sample(rr: float) -> np.ndarray:
        return _sample(M, cx, cy, rr, n)

    on = np.max([sample(r + d) for d in (-1.0, 0.0, 1.0)], axis=0)
    # de omgeving: alleen het bovenvlak erbuiten (binnen een blind gat liggen de randen van wand en bodem)
    off = np.median(np.concatenate([sample(r + d) for d in (6.0, 8.0, 10.0, 12.0)]))
    off = max(float(off), 1e-3)
    return float(np.median(on) / off), float(np.mean(on > 2.0 * off))


def _floor(part: Part2p5D, K: np.ndarray, vd: list, hole: Hole) -> tuple[float | None, int, int]:
    """De diepte uit de onderrand van de wand: per diepte op een rooster het aantal metingen van die rand dicht bij het
    model (|residu| < 1 px). Geeft (diepte of None, metingen, foto's)."""
    best = (None, 0, 0)
    for f in DEPTHS:
        trial = part.copy()
        trial.holes.append(replace(hole, depth=float(f * part.height)))
        if not trial.is_valid():
            continue
        i = len(trial.holes) - 1
        lay = edgefit.layout(trial)
        ie = edgefit.inner_edges(trial, lay)
        sl = ie.groups[("blind", i)]
        n = (sl.stop - sl.start) // 2
        n_meas, views = 0, 0
        for v, (idx, pos, nrm, _w) in zip(vd, edgefit.measure_inner(trial, K, vd, lay)):
            sel = (idx >= sl.start + n) & (idx < sl.stop)
            if not sel.any():
                continue
            uv, _ = project(ie.P[idx[sel]], v.pose, K)
            r = np.sum((uv - [v.x0, v.y0] - pos[sel]) * nrm[sel], axis=1)
            good = int(np.sum(np.abs(r) < 1.0))
            n_meas += good
            views += good > 0
        if n_meas > best[1]:
            best = (float(f * part.height), n_meas, views)
    return best


def detect(part: Part2p5D, K: np.ndarray, vd: list, log=None, maybe: list | None = None) -> list[Hole]:
    """Blinde gaten in het bovenvlak (zie de moduletekst), als Hole met diepte. Een duidelijke ring waarvan de
    onderrand van de wand niet te meten is (bij een zwart onderdeel is het contrast daar te klein), komt in `maybe`:
    [(x, y, diameter)], voor een waarschuwing."""
    if part.steps:  # het bovenvlak heeft dan meer hoogtes; (nog) niet
        return []
    M, lo, _ = edge_map(part, K, vd)
    free = _free_area(part, lo, M.shape)
    if not free.any():
        return []
    scale = float(np.percentile(M[free], 99.5)) if np.any(M[free] > 0) else 0.0
    if scale <= 0:
        return []
    # wat niet vrij is (de buitenrand, bestaande gaten): het niveau van het bovenvlak, niet nul (een rand van nul rond
    # een bestaand gat is zelf een cirkel)
    img = (255 * np.clip(np.where(free, M, float(np.median(M[free]))) / scale, 0, 1)).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=2 * R_MIN_MM / RES_MM, param1=60, param2=18,
                               minRadius=int(R_MIN_MM / RES_MM), maxRadius=int(R_MAX_MM / RES_MM))
    found: list[Hole] = []
    for cx, cy, r in ([] if circles is None else circles[0][:8]):
        # de straal fijner: waar de ring (de mediaan rondom) het sterkst is
        r = max(((float(np.median(_sample(M, cx, cy, rr))), rr) for rr in np.arange(r - 6, r + 6.01, 0.25)),
                key=lambda t: t[0])[1]
        ratio, around = _ring(M, cx, cy, r)
        x, y = lo + (np.array([cx, cy]) + 0.5) * RES_MM
        d = 2 * r * RES_MM
        if any(math.hypot(h.x - x, h.y - y) < (h.d + d) / 2 for h in found):
            continue
        inside = float(np.mean(_sample(free.astype(np.float32), cx, cy, r) > 0.5))  # niet over iets dat er al is
        if inside < 0.9 or ratio < RING_RATIO or around < RING_AROUND:
            if log:
                log(f"geen blind gat op ({x:.1f}, {y:.1f}) Ø {d:.1f}: ring {ratio:.1f}x zo sterk als de omgeving, "
                    f"rondom {around:.0%}")
            continue
        depth, n_meas, views = _floor(part, K, vd, Hole(float(x), float(y), float(d)))
        ok = depth is not None and n_meas >= MIN_FLOOR_MEAS and views >= MIN_FLOOR_VIEWS
        if log:
            log(f"{'blind gat' if ok else 'geen blind gat'} op ({x:.1f}, {y:.1f}) Ø {d:.2f}: ring {ratio:.1f}x zo "
                f"sterk als de omgeving, rondom {around:.0%}; onderrand van de wand {n_meas} keer gemeten in {views} "
                f"foto's" + (f", diepte ~{depth:.1f} mm" if depth is not None else ""))
        if ok:
            found.append(Hole(float(x), float(y), float(d), depth=depth))
        elif maybe is not None:
            maybe.append((float(x), float(y), float(d)))
    return found
