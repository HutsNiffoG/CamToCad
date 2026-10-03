"""Toonkromme, onscherpte en verscherping van de camera (V2, v0.10-v0.11).

Een telefoon slaat niet het licht zelf op: de sensor integreert per pixel, de optiek vervaagt, daarna komt een
toonkromme (sRGB, ~1/2,2) en tot slot verscherping (unsharp masking op de gecodeerde grijswaarden), en het geheel wordt
afgekapt op 0-255. Voor randen op subpixelniveau (alpha is een mengverhouding van licht) en voor maskers die de mat
goed voorspellen, moet dat terug. Per scan en per foto:

* **Verscherping** (`camera_sharpening`), een eigenschap van de camera: uit de randen tussen de zwarte en witte vakken
  van de mat (de slanted-edge-meting van ISO 12233). Per rand alle ruwe pixels in de vrije strook langs de rand (de
  stippen en de marker beginnen pas verderop), met hun afstand tot de rand uit het cameramodel en de pose: zonder
  interpolatie, dus zonder gecodeerde waarden te mengen. Daarop een 1D-model: een stap, pixelintegratie (een trapezium
  langs de normaal), Gauss-vervaging, de kromme, unsharp masking (k, σ) en afkappen; zwart en wit per foto, per rand
  alleen de belichting en een kleine verschuiving (de posefout verschuift een rand, maar verandert zijn vorm niet).
  De verscherping (k, σ) komt daar goed uit (stresstest: 0,43 en 0,99 bij 0,5 en 1,0; σ 1,4-1,6 bij 1,5); de kromme
  niet (de vorm van de vervaging past niet precies), die komt daarna.
* **Kromme en onscherpte** (`estimate`), per foto: de ruwe foto eerst ontscherpt, dan masks._tone_exponent op de ruwe
  foto en de voorspelde mat in de ruwe geometrie. Niet op de ontvervormde foto: de interpolatie bij het ontvervormen
  mengt gecodeerde grijswaarden, en dat maakte een sRGB-kromme ~0,05 te vlak (§3k).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from . import calib, masks
from .mat import MatSpec, make_board

# vrije strook langs een vakrand (mm): de stippen in het zwarte vak beginnen op ~1,7 mm, de marker in het witte op
# 2,5 mm; met de vervaging daarvan nog wat ruimte
STRIP_BLACK_MM, STRIP_WHITE_MM, STRIP_ALONG_MM = 1.0, 1.6, 6.0
MAX_EDGES = 24
STEP = 0.1  # px, raster van het 1D-randmodel
GRID = np.arange(-12.0, 12.0001, STEP)
SHARP_MIN = 0.05  # kleinere k: geen verscherping
# Een kleinere straal is geen verscherping van de camera maar een restje van het model: zonder verscherping, met een
# vage foto en een sRGB-kromme, vindt de fit soms k ~0,2 op de ondergrens van σ (0,5 px)
SHARP_MIN_PX = 0.75


@dataclass
class Sharpening:
    k: float = 0.0  # sterkte van de unsharp masking (0 = geen)
    sigma: float = 1.5  # straal (px)
    n_photos: int = 0  # op hoeveel foto's gemeten

    def to_dict(self) -> dict:
        return {"k": round(self.k, 3), "sigma_px": round(self.sigma, 2), "fotos": self.n_photos}


def desharpen(img: np.ndarray, sh: Sharpening) -> np.ndarray:
    """Zie masks.desharpen."""
    return masks.desharpen(img, sh.k if sh.k >= SHARP_MIN else 0.0, sh.sigma)


@lru_cache(maxsize=4)
def _inverse_map_cached(key: tuple) -> tuple[np.ndarray, np.ndarray]:
    K = np.array(key[0]).reshape(3, 3)
    dist = np.array(key[1])
    w, h = key[2]
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([uu.ravel(), vv.ravel()], 1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K).reshape(h, w, 2)
    return np.ascontiguousarray(und[..., 0]), np.ascontiguousarray(und[..., 1])


def inverse_map(cam) -> tuple[np.ndarray, np.ndarray]:
    """Per ruwe pixel waar hij in het ontvervormde beeld ligt (voor cv2.remap van ontvervormd naar ruw)."""
    return _inverse_map_cached((tuple(np.asarray(cam.K, float).ravel()), tuple(np.asarray(cam.dist, float).ravel()),
                                (int(cam.width), int(cam.height))))


def to_raw(img: np.ndarray, cam, nearest: bool = False) -> np.ndarray:
    """Een ontvervormd beeld (zoals de voorspelde mat) in de geometrie van de ruwe foto."""
    if not np.any(np.abs(cam.dist) > 0):
        return img
    mx, my = inverse_map(cam)
    return cv2.remap(img, mx, my, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------- verscherping uit de randen

def edge_profiles(gray: np.ndarray, pose, cam, ids, spec: MatSpec, max_edges: int = MAX_EDGES, seed: int = 0):
    """De randen tussen zwarte en witte vakken (zie de moduletekst): per rand (afstand tot de rand in px, + = naar het
    witte vak; pixelwaarde; |cos| en |sin| van de normaal in beeld, voor de pixelintegratie)."""
    from .preflight import _fine_raster

    board = make_board(spec)
    chess = np.asarray(board.getChessboardCorners(), float)[:, :2]
    nx = spec.squares_x - 1
    raster = _fine_raster(spec)
    rk, rm = raster.px_per_mm, raster.margin_mm
    A, a = calib.board_to_mat_transform(spec)
    A_inv = np.linalg.inv(A)
    rvec = cv2.Rodrigues(pose.R)[0]
    C = -pose.R.T @ pose.t

    def value(xy):
        u, v = int(round(xy[0] * rk + rm * rk)), int(round(xy[1] * rk + rm * rk))
        return raster.image[min(max(v, 0), raster.image.shape[0] - 1), min(max(u, 0), raster.image.shape[1] - 1)]

    def to_mat(xy):
        return np.column_stack([xy, np.zeros(len(xy))]) @ A.T + a

    have = {int(i) for i in ids}
    pairs = [(c, nb) for c in sorted(have) for nb in (c + 1 if (c % nx) + 1 < nx else None, c + nx)
             if nb is not None and nb in have]
    order = np.random.default_rng(seed).permutation(len(pairs))
    h, w = gray.shape
    out = []
    for j in order:
        if len(out) >= max_edges:
            break
        c, nb = pairs[j]
        p0, p1 = chess[c], chess[nb]
        mid = (p0 + p1) / 2
        t = (p1 - p0) / np.linalg.norm(p1 - p0)
        nrm = np.array([-t[1], t[0]])
        if value(mid + nrm) < value(mid - nrm):
            nrm = -nrm
        box = np.array([mid - STRIP_ALONG_MM * t - STRIP_BLACK_MM * nrm, mid + STRIP_ALONG_MM * t - STRIP_BLACK_MM * nrm,
                        mid + STRIP_ALONG_MM * t + STRIP_WHITE_MM * nrm, mid - STRIP_ALONG_MM * t + STRIP_WHITE_MM * nrm])
        uv = cv2.projectPoints(to_mat(box), rvec, pose.t, cam.K, cam.dist)[0].reshape(-1, 2)
        x0, y0 = np.floor(uv.min(axis=0)).astype(int) - 1
        x1, y1 = np.ceil(uv.max(axis=0)).astype(int) + 1
        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        pix = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64).reshape(-1, 1, 2)
        nrm_pts = cv2.undistortPoints(pix, cam.K, cam.dist).reshape(-1, 2)
        rays = pose.R.T @ np.vstack([nrm_pts.T, np.ones(len(nrm_pts))])
        s = -C[2] / rays[2]
        board_xy = (np.column_stack([C[0] + s * rays[0], C[1] + s * rays[1], np.zeros(len(s))]) - a) @ A_inv.T
        rel = board_xy[:, :2] - mid
        along, across = rel @ t, rel @ nrm
        keep = (np.abs(along) <= STRIP_ALONG_MM) & (across >= -STRIP_BLACK_MM) & (across <= STRIP_WHITE_MM)
        if np.count_nonzero(keep) < 80:
            continue
        vals = gray[ys.ravel()[keep], xs.ravel()[keep]].astype(np.float64)
        if vals.std() < 15.0:  # geen contrast: onder het object, in de schaduw of buiten beeld
            continue
        q = cv2.projectPoints(to_mat(np.array([mid, mid + 0.5 * nrm])), rvec, pose.t, cam.K, cam.dist)[0].reshape(2, 2)
        dn = (q[1] - q[0]) / 0.5
        # afstand loodrecht op de rand in beeld: across (mm) maal de component van de geprojecteerde normaal
        # loodrecht op de beeldrand
        q2 = cv2.projectPoints(to_mat(np.array([mid, mid + 0.5 * t])), rvec, pose.t, cam.K, cam.dist)[0].reshape(2, 2)
        te = (q2[1] - q2[0]) / max(float(np.linalg.norm(q2[1] - q2[0])), 1e-9)
        ne = np.array([-te[1], te[0]])
        scale = abs(float(dn @ ne))
        out.append((across[keep] * scale, vals, abs(float(ne[0])), abs(float(ne[1]))))
    return out


def _box(width: float) -> np.ndarray:
    m = max(1, int(round(width / STEP)))
    return np.ones(m) / m


def _gauss(sig: float) -> np.ndarray:
    x = np.arange(-int(4 * sig / STEP) - 1, int(4 * sig / STEP) + 2) * STEP
    k = np.exp(-0.5 * (x / sig) ** 2)
    return k / k.sum()


_NG = 6  # globaal: g, vervaging, k, σ_k, zwart, wit (lineair licht); per rand: belichting en verschuiving


def _steps(edges: list) -> np.ndarray:
    """Per rand een stap met de pixelintegratie (twee blokjes langs de normaal), op GRID: (randen, punten)."""
    step = (GRID >= 0).astype(float)
    out = []
    for _dd, _vals, cx, cy in edges:
        foot = np.convolve(_box(cx), _box(cy)) if min(cx, cy) > STEP else _box(max(cx, cy))
        n = len(foot) // 2
        out.append(np.convolve(np.pad(step, n, mode="edge"), foot / foot.sum(), mode="same")[n:n + len(step)])
    return np.array(out)


def _edge_model(params: np.ndarray, edges: list, steps: np.ndarray) -> np.ndarray:
    g, sb, k, sk, black, white = params[:_NG]
    c = params[_NG::2][:, None]
    delta = params[_NG + 1::2]
    prof = ndimage.convolve1d(steps, _gauss(sb), axis=1, mode="nearest")
    E = 255.0 * np.clip(c * (black + (white - black) * prof) / 255.0, 1e-6, 1.0) ** g
    S = np.clip(E + k * (E - ndimage.convolve1d(E, _gauss(sk), axis=1, mode="nearest")), 0.0, 255.0)
    return np.concatenate([np.interp(dd - delta[e], GRID, S[e]) - vals for e, (dd, vals, _cx, _cy) in enumerate(edges)])


def fit_sharpening(edges: list) -> tuple[float, float] | None:
    """(k, σ) uit de randprofielen van één foto (zie de moduletekst), of None met te weinig randen."""
    if len(edges) < 6:
        return None
    allv = np.concatenate([e[1] for e in edges])
    g0 = 0.6
    lo, hi = np.percentile(allv, 3), np.percentile(allv, 97)
    p0 = [g0, 0.6, 0.3, 1.5, 255.0 * (lo / 255.0) ** (1 / g0), 255.0 * (hi / 255.0) ** (1 / g0)] + [1.0, 0.0] * len(edges)
    lb = [0.25, 0.15, 0.0, 0.5, 0.0, 10.0] + [0.5, -3.0] * len(edges)
    ub = [1.6, 4.0, 3.0, 4.0, 200.0, 400.0] + [1.5, 3.0] * len(edges)
    m = sum(len(e[0]) for e in edges)
    J = lil_matrix((m, _NG + 2 * len(edges)), dtype=int)
    r0 = 0
    for e, (dd, *_rest) in enumerate(edges):
        J[r0:r0 + len(dd), :_NG] = 1
        J[r0:r0 + len(dd), _NG + 2 * e: _NG + 2 * e + 2] = 1
        r0 += len(dd)
    try:
        r = least_squares(_edge_model, np.clip(p0, lb, ub), bounds=(lb, ub), args=(edges, _steps(edges)),
                          loss="soft_l1", f_scale=3.0, x_scale="jac", jac_sparsity=J, max_nfev=80)
    except (ValueError, np.linalg.LinAlgError):
        return None
    k, sig = float(r.x[2]), float(r.x[3])
    return (k if sig >= SHARP_MIN_PX else 0.0), sig


def camera_sharpening(photos: list, cam, spec: MatSpec, max_photos: int = 3) -> Sharpening:
    """De verscherping van de camera: de mediaan over een paar foto's (`photos`: (grijsbeeld, pose, hoek-ID's)),
    verspreid over de scan. Zonder bruikbare randen: geen verscherping."""
    if not photos:
        return Sharpening()
    pick = [photos[i] for i in np.linspace(0, len(photos) - 1, min(max_photos, len(photos))).round().astype(int)]
    found = []
    for gray, pose, ids in pick:
        res = fit_sharpening(edge_profiles(gray, pose, cam, ids, spec))
        if res is not None:
            found.append(res)
    if not found:
        return Sharpening()
    k, sig = (float(np.median([f[i] for f in found])) for i in range(2))
    return Sharpening(k if k >= SHARP_MIN else 0.0, sig, len(found))


# ----------------------------------------------------------------------------- kromme en onscherpte per foto

# Lokale toonbewerking (HDR, v0.11): een telefoon past de kromme per stuk beeld aan. Per tegel (TILE_N x TILE_N) een
# eigen g, alleen als dat de residuen op de overgangen van de mat minstens TILE_MIN_GAIN kleiner maakt en de tegels
# minstens TILE_MIN_RANGE verschillen; anders één kromme voor de hele foto
TILE_N, TILE_MIN_GAIN, TILE_MIN_RANGE = 3, 0.05, 0.04


def tile_grid(raw: np.ndarray, p: np.ndarray, v: np.ndarray, g0: float, blur: float) -> np.ndarray | None:
    """g per tegel (zie TILE_N), of None als één kromme volstaat. `raw`: de ontscherpte ruwe foto, `p`, `v`: de
    voorspelde mat in de ruwe geometrie."""
    ps = cv2.GaussianBlur(p, (0, 0), blur) if blur > 0 else p
    h, w = raw.shape
    grid = np.full((TILE_N, TILE_N), g0, float)
    base = better = 0.0
    used = 0
    for i in range(TILE_N):
        for j in range(TILE_N):
            sl = (slice(i * h // TILE_N, (i + 1) * h // TILE_N), slice(j * w // TILE_N, (j + 1) * w // TILE_N))
            idx = np.flatnonzero(v[sl].ravel())[::5]
            if len(idx) < 3000:
                continue
            y = np.clip(raw[sl].ravel()[idx].astype(np.float64) / 255.0, 0.0, 1.0)
            pv = ps[sl].ravel()[idx].astype(np.float64)
            lo, hi = np.percentile(pv, [5, 95])
            trans = (pv > lo + 0.15 * (hi - lo)) & (pv < hi - 0.15 * (hi - lo))
            if np.count_nonzero(trans) < 400:
                continue
            every = np.ones(len(y), bool)

            def spread(g: float) -> float:
                ol = 255.0 * y ** (1.0 / g)
                a, b, _ = masks._robust_affine(ol, pv, every)
                return float(np.median(np.abs(ol - (a * pv + b))[trans]) / max(abs(a), 1e-3))

            coarse = np.arange(max(0.25, g0 - 0.15), g0 + 0.1501, 0.05)
            gc = float(coarse[int(np.argmin([spread(g) for g in coarse]))])
            fine = np.arange(max(0.25, gc - 0.04), gc + 0.0401, 0.01)
            sf = [spread(g) for g in fine]
            k = int(np.argmin(sf))
            e0 = spread(g0)
            grid[i, j] = round(float(fine[k]), 2)
            base += e0 * np.count_nonzero(trans)
            better += sf[k] * np.count_nonzero(trans)
            used += 1
    if used < 4 or better > (1.0 - TILE_MIN_GAIN) * base or np.ptp(grid) < TILE_MIN_RANGE:
        return None
    return grid


def estimate(gray_raw: np.ndarray, pred: np.ndarray, valid: np.ndarray, cam, sh: Sharpening) -> masks.Tone:
    """Toonkromme en extra onscherpte van één foto (zie de moduletekst), met de verscherping van de camera, en bij
    lokale toonbewerking een kromme per tegel (tile_grid). `pred`, `valid`: de voorspelde mat (ontvervormd, zoals
    masks.predict_background)."""
    raw = desharpen(gray_raw, sh)
    p = to_raw(pred.astype(np.float32), cam)
    v = to_raw(valid.astype(np.uint8), cam, nearest=True) > 0
    g, blur = masks._tone_exponent(raw, p, v)
    grid = tile_grid(raw, p, v, g, blur)
    return masks.Tone(g=g, blur=blur, sharpen=sh.k if sh.k >= SHARP_MIN else 0.0, sharpen_px=sh.sigma, grid=grid)


def describe(tones: list) -> str:
    """Korte samenvatting voor het log: mediaan en spreiding van g en de onscherpte, en hoe vaak een kromme per tegel."""
    gs = np.array([t.g for t in tones])
    bl = np.array([t.blur for t in tones])
    n_grid = sum(t.grid is not None for t in tones)
    return (f"toonkromme g {np.median(gs):.2f} ({gs.min():.2f}-{gs.max():.2f})"
            + (f", in {n_grid} foto's per stuk beeld anders (lokale toonbewerking)" if n_grid else "")
            + f", onscherpte {np.median(bl):.2f} px (extra t.o.v. de voorspelde mat)")

