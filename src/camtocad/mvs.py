"""Stereo voor de vrije vorm (v0.17): per foto een dieptekaart uit de buurfoto's, en daaruit de lege ruimte.

De visual hull (freeform.py) is te vol waar geen foto de mat achter het onderdeel ziet. Er ligt dan een 'dak' boven
een vlakke of licht gebogen bovenkant, en een kuil of de spleet onder de arm van een nietmachine blijft dicht. Heeft
het onderdeel daar textuur, dan ziet een punt op het echte oppervlak er in de buurfoto's hetzelfde uit, en een punt
in het dak niet.

Per foto (de referentie) en per pixel op het object zoekt de stereo langs de kijkstraal, van de voorkant van de hull
naar achteren, de diepte waar het venster rond de pixel het best past bij de buurfoto's. De maat is de NCC; telt het
gemiddelde van de twee best passende van vier buren, dan telt een buur die het punt niet ziet niet mee. Alleen een
duidelijke diepte telt:
* het venster heeft textuur en ligt helemaal op het object;
* de NCC is hoog en het tweede beste piekje ligt duidelijk lager;
* een andere foto vindt op die plek dezelfde diepte.
De dieptekaarten gaan samen in een afstandsveld (TSDF, zoals bij 3D-scanners): per punt het gemiddelde over de
foto's van hoe ver het achter het gemeten oppervlak ligt, afgekapt op een paar mm. Waar dat negatief is, is lucht;
het nulniveau is het gemeten oppervlak, gemiddeld over de foto's (zie fused).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calib import Pose, project

WINDOW = 7  # px, venster van de NCC
NCC_MIN = 0.7  # zo goed moet het venster passen
PEAK_GAP = 0.07  # zoveel lager moet het beste piekje elders langs de straal liggen
TEXTURE_MIN = 3.0  # grijswaarden: spreiding in het venster, anders geen textuur (ruis is ~2)
STEP_MM = 0.5  # stap langs de kijkstraal
SEARCH_MM = 30.0  # zo ver achter de voorkant van de hull wordt gezocht
CONSISTENT_MM = 1.0  # een andere foto bevestigt de diepte binnen zoveel mm
TRUNC_MM = 2.0  # TSDF: afkappen op zoveel mm voor en achter het gemeten oppervlak
MIN_FUSE_VIEWS = 3  # zoveel foto's met een diepte moeten een punt zien


@dataclass
class DepthView:
    pose: Pose
    depth: np.ndarray  # float32 (h, w): z in de camera (mm) van het gemeten oppervlak; NaN zonder betrouwbare diepte
    score: np.ndarray  # float32 (h, w): de NCC bij die diepte


def neighbours(poses: list[Pose], center: np.ndarray, k: int = 4, min_deg: float = 6.0,
               max_deg: float = 40.0) -> list[list[int]]:
    """Per foto tot `k` buurfoto's: een kijkrichting op het onderdeel tussen `min_deg` en `max_deg` ervan, en onderling
    ook minstens `min_deg` uit elkaar (dan liggen ze niet allemaal aan dezelfde kant)."""
    dirs = np.array([(center - p.center) / np.linalg.norm(center - p.center) for p in poses])
    ang = np.degrees(np.arccos(np.clip(dirs @ dirs.T, -1.0, 1.0)))
    out = []
    for i in range(len(poses)):
        chosen: list[int] = []
        for j in np.argsort(ang[i]):
            if not min_deg <= ang[i, j] <= max_deg:
                continue
            if all(ang[j, c] >= min_deg for c in chosen):
                chosen.append(int(j))
            if len(chosen) == k:
                break
        out.append(chosen)
    return out


def hull_depths(V: np.ndarray, pose: Pose, K: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel de diepte van de voor- en achterkant van de hull (uit de hoekpunten van de mesh: de kleinste en
    grootste z per pixel, gaten gedicht met een minimum- en maximumfilter); inf / -inf buiten de hull."""
    h, w = shape
    uv, z = project(V, pose, K)
    ui, vi = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
    ok = (z > 1.0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    idx = vi[ok] * w + ui[ok]
    front = np.full(h * w, np.inf, np.float32)
    back = np.full(h * w, -np.inf, np.float32)
    np.minimum.at(front, idx, z[ok].astype(np.float32))
    np.maximum.at(back, idx, z[ok].astype(np.float32))
    k = np.ones((5, 5), np.uint8)
    return cv2.erode(front.reshape(h, w), k), cv2.dilate(back.reshape(h, w), k)


def _box(img: np.ndarray) -> np.ndarray:
    return cv2.boxFilter(img, -1, (WINDOW, WINDOW), normalize=True, borderType=cv2.BORDER_REFLECT)


def depth_map(ref: np.ndarray, ref_pose: Pose, nbrs: list[tuple[np.ndarray, Pose]], K: np.ndarray,
              mask: np.ndarray, front: np.ndarray, back: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """De dieptekaart van één foto (z in de camera, mm; NaN zonder betrouwbare diepte) en de NCC daarbij. `mask`:
    pixels waar gezocht wordt (op het object, het venster er helemaal binnen); `front`/`back`: de hull. `nbrs`: per
    buurfoto (beeld, pose).

    Geen filter op sprongen in de voorkant van de hull: die zitten ook waar de hull zelf fout is (een torentje of
    dak), en juist daar is de diepte nodig."""
    h, w = ref.shape
    depth = np.full((h, w), np.nan, np.float32)
    score = np.zeros((h, w), np.float32)
    search = mask & np.isfinite(front)
    if not search.any() or len(nbrs) < 2:
        return depth, score
    ys, xs = np.nonzero(search)
    pad = WINDOW
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
    R = ref[y0:y1, x0:x1].astype(np.float32)
    m_r = _box(R)
    s_r = np.sqrt(np.maximum(_box(R * R) - m_r * m_r, 0.0))
    sel = search[y0:y1, x0:x1] & (s_r >= TEXTURE_MIN)
    if not sel.any():
        return depth, score
    vv, uu = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    ray = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], axis=-1)
    # De zoekvlakken volgen de voorkant van de hull, maar glad: binnen een venster moeten alle pixels op (bijna)
    # dezelfde diepte liggen, anders past het venster nergens. Rond een fout in de hull (een smal torentje) springt
    # de voorkant; daar geeft het minimum in de buurt, vervaagd, toch een glad vlak, en wordt verder gezocht.
    fr = front[y0:y1, x0:x1]
    fin = np.isfinite(fr)
    fr_f = np.where(fin, fr, np.nanmax(np.where(fin, fr, np.nan)) if fin.any() else 0.0).astype(np.float32)
    k15 = np.ones((2 * WINDOW + 1, 2 * WINDOW + 1), np.uint8)
    f0 = np.minimum(cv2.GaussianBlur(cv2.erode(fr_f, k15), (0, 0), WINDOW / 2.0), fr_f) - 2 * STEP_MM
    f1 = np.minimum(np.where(np.isfinite(back[y0:y1, x0:x1]), back[y0:y1, x0:x1], 0.0), fr_f + SEARCH_MM)
    n_steps = int(np.ceil(max(float(np.max((f1 - f0)[sel])), 0.0) / STEP_MM)) + 1
    # per buur: X_n(d) = d · A + b, met A = R_n R^T · straal en b = t_n - R_n R^T t
    geo = []
    for img, pose in nbrs:
        Rn = pose.R @ ref_pose.R.T
        geo.append((img.astype(np.float32), ray @ Rn.T, pose.t - Rn @ ref_pose.t))
    best = np.full(R.shape, -1.0, np.float32)
    best_d = np.full(R.shape, np.nan, np.float32)
    scores = np.full((n_steps,) + R.shape, -1.0, np.float16)
    for k in range(n_steps):
        d = f0 + k * STEP_MM
        live = sel & (d <= f1)
        if not live.any():
            break
        nccs = []
        for img, A, b in geo:
            X = A * d[..., None] + b
            zn = X[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                un = (K[0, 0] * X[..., 0] / zn + K[0, 2]).astype(np.float32)
                vn = (K[1, 1] * X[..., 1] / zn + K[1, 2]).astype(np.float32)
            inside = (zn > 1.0) & (un >= 0) & (un <= img.shape[1] - 1) & (vn >= 0) & (vn <= img.shape[0] - 1)
            W = cv2.remap(img, un, vn, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            m_w = _box(W)
            s_w = np.sqrt(np.maximum(_box(W * W) - m_w * m_w, 1e-6))
            ncc = (_box(R * W) - m_r * m_w) / np.maximum(s_r * s_w, 1e-6)
            ncc[~inside] = -1.0
            nccs.append(ncc)
        stack = np.sort(np.stack(nccs), axis=0)
        sc = np.where(live, 0.5 * (stack[-1] + stack[-2]), -1.0).astype(np.float32)
        scores[k] = sc
        upd = sc > best
        best[upd] = sc[upd]
        best_d[upd] = d[upd]
    # duidelijk: het beste piekje elders (meer dan 3 stappen verder) ligt PEAK_GAP lager
    kbest = np.round((best_d - f0) / STEP_MM)
    ks = np.arange(n_steps, dtype=np.float32)[:, None, None]
    far = np.abs(ks - kbest[None]) > 3
    second = np.max(np.where(far, scores.astype(np.float32), -1.0), axis=0)
    good = sel & (best >= NCC_MIN) & (best - second >= PEAK_GAP)
    depth[y0:y1, x0:x1] = np.where(good, best_d, np.nan)
    score[y0:y1, x0:x1] = np.where(good, best, 0.0)
    return depth, score


def consistent(maps: list[DepthView], K: np.ndarray, others: list[list[int]]) -> list[DepthView]:
    """Houdt alleen de diepten die een andere foto (een van de buren) bevestigt: het punt valt daar op een pixel met
    een diepte die binnen CONSISTENT_MM klopt."""
    out = []
    for i, m in enumerate(maps):
        h, w = m.depth.shape
        ys, xs = np.nonzero(np.isfinite(m.depth))
        if not len(ys):
            out.append(m)
            continue
        z = m.depth[ys, xs]
        ray = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones(len(xs))], axis=1)
        X = (ray * z[:, None] - m.pose.t) @ m.pose.R  # mat-coördinaten
        ok = np.zeros(len(ys), bool)
        for j in others[i]:
            o = maps[j]
            uv, zo = project(X, o.pose, K)
            ui, vi = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
            inside = (ui >= 0) & (ui < o.depth.shape[1]) & (vi >= 0) & (vi < o.depth.shape[0])
            dz = np.full(len(ys), np.inf)
            dz[inside] = np.abs(o.depth[vi[inside], ui[inside]] - zo[inside])
            ok |= dz <= CONSISTENT_MM
        depth = np.full((h, w), np.nan, np.float32)
        depth[ys[ok], xs[ok]] = z[ok]
        out.append(DepthView(m.pose, depth, np.where(np.isfinite(depth), m.score, 0.0).astype(np.float32)))
    return out


def fill_small_holes(m: DepthView, mask: np.ndarray, radius: int = 4, min_cover: float = 0.6) -> DepthView:
    """Kleine gaatjes in een dieptekaart (een pixel zonder textuur of een twijfelgeval tussen goede diepten) vullen
    met het gewogen gemiddelde van de diepten eromheen, als die rondom liggen (minstens `min_cover` van het venster)
    en niet meer dan 2 mm uiteenlopen. Alleen op het object (`mask`). Zonder dat blijft daar de hull staan, als een
    smal torentje boven het gemeten oppervlak."""
    ok = np.isfinite(m.depth)
    if not ok.any():
        return m
    k = 2 * radius + 1
    w = cv2.boxFilter(ok.astype(np.float32), -1, (k, k), normalize=True)
    d0 = np.where(ok, m.depth, 0.0).astype(np.float32)
    mean = cv2.boxFilter(d0, -1, (k, k), normalize=True) / np.maximum(w, 1e-6)
    big = np.where(ok, m.depth, -1e6).astype(np.float32)
    small = np.where(ok, m.depth, 1e6).astype(np.float32)
    kk = np.ones((k, k), np.uint8)
    spread = cv2.dilate(big, kk) - cv2.erode(small, kk)
    fill = ~ok & mask & (w >= min_cover) & (spread <= 2.0)
    if not fill.any():
        return m
    depth = m.depth.copy()
    depth[fill] = mean[fill]
    return DepthView(m.pose, depth, m.score)


def fused(pts: np.ndarray, maps: list[DepthView], K: np.ndarray, no_evidence: float,
          trunc: float = TRUNC_MM, min_views: int = MIN_FUSE_VIEWS) -> np.ndarray:
    """De dieptekaarten samen (TSDF, zoals bij 3D-scanners): per punt en per foto met een diepte op die pixel
    u = z - diepte (positief achter het gemeten oppervlak, negatief ervoor), afgekapt op ±`trunc`; ligt het punt meer
    dan `trunc` achter het oppervlak, dan telt die foto niet (het punt kan erachter verborgen liggen). Het gemiddelde
    is positief binnen het onderdeel; het nulniveau is het gemeten oppervlak, gemiddeld over de foto's. Robuust: in
    een tweede ronde tellen foto's die ver van het eerste gemiddelde afliggen minder mee (een foute diepte in een
    van de weinige foto's die een plek zien, mag daar geen kuil maken). Met minder dan `min_views` foto's:
    `no_evidence`."""
    samples = []
    for m in maps:
        uv, z = project(pts, m.pose, K)
        ui, vi = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
        h, w = m.depth.shape
        inside = (z > 1.0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        idx = np.flatnonzero(inside)
        d = m.depth[vi[idx], ui[idx]]
        u = z[idx] - d
        use = np.isfinite(u) & (u <= trunc)
        samples.append((idx[use], np.maximum(u[use], -trunc).astype(np.float32)))
    total = np.zeros(len(pts), np.float32)
    count = np.zeros(len(pts), np.float32)
    for idx, u in samples:
        np.add.at(total, idx, u)
        np.add.at(count, idx, 1.0)
    mean = total / np.maximum(count, 1.0)
    total[:] = 0.0
    weight = np.zeros(len(pts), np.float32)
    for idx, u in samples:  # tweede ronde: gewicht 1 / (1 + (afwijking / 1 mm)²)
        w = 1.0 / (1.0 + (u - mean[idx]) ** 2)
        np.add.at(total, idx, w * u)
        np.add.at(weight, idx, w)
    return np.where(count >= min_views, total / np.maximum(weight, 1e-6), no_evidence).astype(np.float32)


def seen(pts: np.ndarray, maps: list[DepthView], K: np.ndarray, tol: float = TRUNC_MM,
         min_views: int = MIN_FUSE_VIEWS) -> np.ndarray:
    """Per punt: zien minstens `min_views` dieptekaarten op die plek een oppervlak binnen `tol` mm (bool)?"""
    count = np.zeros(len(pts), np.int32)
    for m in maps:
        uv, z = project(pts, m.pose, K)
        ui, vi = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
        h, w = m.depth.shape
        inside = (z > 1.0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        d = np.full(len(pts), np.nan)
        d[inside] = m.depth[vi[inside], ui[inside]]
        count += np.abs(z - d) <= tol
    return count >= min_views
