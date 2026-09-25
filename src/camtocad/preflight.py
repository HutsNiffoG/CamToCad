"""Fotocontrole vooraf: foto's beoordelen zodra ze binnen zijn, vóór de (trage) verwerking.

Per foto (~0,2 s): is de kalibratiemat te vinden, hoe scherp en hoe belicht is de foto, en
onder welke hoek kijkt hij? Per scan: een dekkingskaart (van welke kanten en hoogtes er foto's
zijn, hoeveel recht van boven) en concrete aanwijzingen, zoals "nog 2 foto's recht boven het
onderdeel".

Alles wordt gemeten aan de mat zelf:
* onscherpte: de spreiding σ (pixels op werkresolutie) van de zwart-witranden tussen de vakken,
  gefit met een vervaagde stap (foutfunctie);
* belichting: papierwit en zwart naast de schaakbordhoeken, en het aandeel uitgebeten wit;
* kijkhoek: per foto grof uit de homografie van de mat, per scan na een snelle zelfkalibratie;
* waar het onderdeel ligt: de hoeken en markers die in beeld horen te zijn maar steeds
  ontbreken (het onderdeel ligt erop of ervoor).
"""

from __future__ import annotations

import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

from . import calib
from .mat import MatSpec, board_to_mat, get_spec, make_board, rasterize_board

WORK_SIDE = 2000  # dezelfde werkresolutie als de pipeline
MIN_CORNERS = 12
BLUR_WARN, BLUR_BAD = 1.8, 3.0  # onscherpte σ in pixels
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
# richting van de camera gezien vanaf het onderdeel; boven = kant met de titel, onder = meetlijn X
SECTORS = ("rechts", "rechtsboven", "boven", "linksboven", "links", "linksonder", "onder", "rechtsonder")
BANDS = (("hoog", 50.0, 80.0, 60), ("laag", 20.0, 50.0, 35))  # naam, elevatie van-tot, richtwaarde (graden)
STEEP_DEG, TOP_DEG = 8.0, 25.0  # zoals initial.select_top_views


@dataclass
class PhotoCheck:
    name: str
    width: int = 0
    height: int = 0
    mat: str | None = None
    corners: int = 0
    mat_fraction: float = 0.0  # gevonden / alle schaakbordhoeken
    blur_px: float | None = None
    white: float | None = None  # grijswaarde van het papier
    contrast: float | None = None  # papier min zwart
    clipped: float | None = None  # aandeel van de mat dat wit uitgebeten is
    f_px: float | None = None  # brandpuntsafstand uit de homografie (grof)
    tilt_deg: float | None = None  # optische as t.o.v. loodrecht (grof)
    verdict: str = "onbruikbaar"  # goed | matig | onbruikbaar
    notes: list[str] = field(default_factory=list)
    ids: list[int] = field(default_factory=list)
    points: list[list[float]] = field(default_factory=list)
    marker_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PhotoCheck":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def public(self) -> dict:
        """Zonder de detectiegegevens (voor de telefoonpagina)."""
        return {k: v for k, v in self.to_dict().items() if k not in ("ids", "points", "marker_ids")}


# ----------------------------------------------------------------------------- beelden

def read_gray(path: str | Path) -> np.ndarray | None:
    """Leest een foto als grijswaarden, zonder EXIF-rotatie (sensorformaat, zoals de pipeline).
    Via imdecode: werkt ook met niet-ASCII-tekens in het pad (Windows)."""
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)


def to_work(img: np.ndarray, max_side: int = WORK_SIDE) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = max_side / max(img.shape)
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


# ----------------------------------------------------------------------------- per foto

def _detect_any(gray: np.ndarray, spec: MatSpec | None, name: str):
    if spec is not None:
        det = calib.detect(gray, make_board(spec), name)
        if det is not None and len(det.ids) >= MIN_CORNERS:
            return det, spec
    found, _ = calib.identify_mat([gray])
    if found is None or (spec is not None and found.name == spec.name):
        return None, spec
    det = calib.detect(gray, make_board(found), name)
    return (det, found) if det is not None and len(det.ids) >= MIN_CORNERS else (None, spec)


def _marker_squares(spec: MatSpec) -> set[tuple[int, int]]:
    s = spec.square_mm
    return {(int(c.reshape(4, 3)[:, 0].mean() // s), int(c.reshape(4, 3)[:, 1].mean() // s))
            for c in make_board(spec).getObjPoints()}


def _sample(gray: np.ndarray, uv: np.ndarray) -> np.ndarray:
    m = uv.astype(np.float32).reshape(1, -1, 2)
    return cv2.remap(gray.astype(np.float32), m[..., 0], m[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).ravel()


def _photometry(gray: np.ndarray, H: np.ndarray, det, spec: MatSpec) -> tuple[float, float, float]:
    """Papierwit en zwart op 1,25 mm diagonaal naast elke hoek (wit: marge rond de marker; zwart:
    stipvrije hoekzone), en het aandeel van het bord dat op de maximale grijswaarde zit."""
    s, nx = spec.square_mm, spec.squares_x - 1
    markers = _marker_squares(spec)
    pts, is_white = [], []
    for cid in det.ids:
        i, j = int(cid) % nx + 1, int(cid) // nx + 1
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            x, y = i * s + 1.25 * dx, j * s + 1.25 * dy
            pts.append((x, y))
            is_white.append((int(x // s), int(y // s)) in markers)
    uv = cv2.perspectiveTransform(np.array(pts, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)
    vals = _sample(gray, uv)
    is_white = np.array(is_white)
    white, black = float(np.median(vals[is_white])), float(np.median(vals[~is_white]))
    w, h = spec.board_w_mm, spec.board_h_mm
    outline = cv2.perspectiveTransform(np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(-1, 1, 2), H)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(np.clip(outline.reshape(-1, 2), -1e5, 1e5)).astype(np.int32), 1)
    inside = mask.astype(bool)
    clipped = float((gray[inside] >= 250).mean()) if inside.any() else 0.0
    return white, white - black, clipped


@lru_cache(maxsize=8)
def _fine_raster(spec: MatSpec):
    """Het bord in board-mm op 20 px/mm, zonder anti-aliasing: randen liggen op 0,05 mm nauwkeurig."""
    return rasterize_board(spec.nominal(), px_per_mm=20.0, margin_mm=3.0, supersample=1)


def _warp_patch(raster, H: np.ndarray, center: np.ndarray, half: int, shift=(0.0, 0.0)) -> np.ndarray:
    """Voorspelde (scherpe) matpatch rond `center` (beeld-px), met pixelintegratie via 2x supersampling."""
    k, m = raster.px_per_mm, raster.margin_mm
    to_raster = np.array([[k, 0.0, m * k - 0.5], [0.0, k, m * k - 0.5], [0.0, 0.0, 1.0]])
    n = 2 * half + 1
    x0, y0 = center[0] - half + shift[0], center[1] - half + shift[1]
    # beeld-px (x0 + u, y0 + v) → 2x-subpixels (2u + 0,5, 2v + 0,5)
    to_patch = np.array([[2.0, 0.0, 0.5 - 2 * x0], [0.0, 2.0, 0.5 - 2 * y0], [0.0, 0.0, 1.0]])
    M = to_patch @ H @ np.linalg.inv(to_raster)
    big = cv2.warpPerspective(raster.image, M, (2 * n, 2 * n), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return cv2.resize(big.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA)


def _edge_blur(gray: np.ndarray, H: np.ndarray, det, spec: MatSpec, n_patches: int = 40, half: int = 15,
               sig_max: float = 6.0) -> float | None:
    """Onscherpte σ (px): hoeveel Gauss-vervaging maakt de voorspelde, scherpe mat gelijk aan de foto?

    Patches rond de middens van de vakranden (rand, witte marge, markerrand, stippen) worden uit het
    matraster voorspeld via de homografie, per patch uitgelijnd (lensvervorming verschuift ze iets)
    en met één σ voor de hele foto vervaagd; per patch een eigen niveau en contrast (belichting).
    De slechtste 20% (object erop, schaduw, glans) telt niet mee. Pixelintegratie zit al in de
    voorspelling: σ is de extra onscherpte van lens, focus en beweging."""
    nx = spec.squares_x - 1
    chess = np.asarray(make_board(spec).getChessboardCorners(), float)[:, :2]
    have = {int(i) for i in det.ids}
    mids = [(chess[c] + chess[nb]) / 2 for c in sorted(have)
            for nb in (c + 1 if (c % nx) + 1 < nx else None, c + nx) if nb is not None and nb in have]
    if len(mids) < 4:
        return None
    mids = np.array(mids)[np.random.default_rng(0).permutation(len(mids))[:n_patches]]
    centers = cv2.perspectiveTransform(mids.reshape(-1, 1, 2), H).reshape(-1, 2)
    raster = _fine_raster(spec)
    pad = int(math.ceil(3 * sig_max))
    big_half = half + pad
    h, w = gray.shape
    img = gray.astype(np.float32)
    obs, preds = [], []
    for c in centers:
        cx, cy = int(round(c[0])), int(round(c[1]))
        if cx - half - 2 < 0 or cy - half - 2 < 0 or cx + half + 3 > w or cy + half + 3 > h:
            continue
        center = np.array([cx, cy], float)
        o_ext = img[cy - half - 2:cy + half + 3, cx - half - 2:cx + half + 3]
        if o_ext.std() < 15.0:  # geen contrast: onder het object of buiten de mat
            continue
        # uitlijnen: kruiscorrelatie met de licht vervaagde voorspelling, subpixel via een parabool
        tmpl = cv2.GaussianBlur(_warp_patch(raster, H, center, half), (0, 0), 1.0)
        resp = cv2.matchTemplate(o_ext, tmpl, cv2.TM_CCOEFF_NORMED)
        iy, ix = np.unravel_index(int(np.argmax(resp)), resp.shape)
        if resp[iy, ix] < 0.5:
            continue

        def sub(r0, r1, r2):
            den = r0 - 2 * r1 + r2
            return 0.0 if abs(den) < 1e-9 else float(np.clip(0.5 * (r0 - r2) / den, -0.5, 0.5))

        dx = ix - 2 + (sub(resp[iy, ix - 1], resp[iy, ix], resp[iy, ix + 1]) if 0 < ix < resp.shape[1] - 1 else 0.0)
        dy = iy - 2 + (sub(resp[iy - 1, ix], resp[iy, ix], resp[iy + 1, ix]) if 0 < iy < resp.shape[0] - 1 else 0.0)
        # het beeld is verschoven t.o.v. de voorspelling: voorspel op de verschoven plek
        preds.append(_warp_patch(raster, H, center, big_half, shift=(-dx, -dy)))
        obs.append(img[cy - half:cy + half + 1, cx - half:cx + half + 1])
    if len(obs) < 4:
        return None
    ob = np.array(obs).reshape(len(obs), -1)
    Oc = ob - ob.mean(axis=1, keepdims=True)
    o_var = np.maximum((Oc * Oc).sum(axis=1), 1.0)

    def cost(sig: float) -> float:
        B = np.array([cv2.GaussianBlur(p, (0, 0), sig)[pad:-pad, pad:-pad] for p in preds]).reshape(len(preds), -1)
        Bc = B - B.mean(axis=1, keepdims=True)
        b = (Bc * Oc).sum(axis=1) / np.maximum((Bc * Bc).sum(axis=1), 1e-9)
        err = ((Oc - b[:, None] * Bc) ** 2).sum(axis=1) / o_var
        return float(np.sort(err)[: max(3, int(0.8 * len(err)))].sum())

    return float(minimize_scalar(cost, bounds=(0.05, sig_max), method="bounded", options={"xatol": 0.01}).x)


def measure_blur(gray: np.ndarray, det, spec: MatSpec) -> float | None:
    """Onscherpte σ (px) van een foto met een gevonden mat (zie _edge_blur)."""
    chess = np.asarray(make_board(spec).getChessboardCorners(), float)[:, :2]
    H, _ = cv2.findHomography(chess[det.ids], det.corners)
    return None if H is None else _edge_blur(gray, H, det, spec.nominal())


def _focal_and_tilt(H: np.ndarray, w: int, h: int) -> tuple[float | None, float]:
    """Brandpuntsafstand (hoofdpunt in het midden, vierkante pixels) en kanteling uit één homografie.
    Bij een (bijna) loodrechte foto is f niet te bepalen; de kanteling is dan ook zonder f klein."""
    T = np.array([[1.0, 0.0, -(w - 1) / 2], [0.0, 1.0, -(h - 1) / 2], [0.0, 0.0, 1.0]])
    Hc = T @ H
    Hc = Hc / np.linalg.norm(Hc[:, 0])
    h1, h2 = Hc[:, 0], Hc[:, 1]
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    inv_f2 = -float(a @ b) / max(float(a @ a), 1e-30)
    f = 1.0 / math.sqrt(inv_f2) if inv_f2 > 0 else None
    if f is not None and not 0.4 * max(w, h) < f < 2.5 * max(w, h):
        f = None
    fk = f or 0.8 * max(w, h)
    r1 = np.array([h1[0] / fk, h1[1] / fk, h1[2]])
    r2 = np.array([h2[0] / fk, h2[1] / fk, h2[2]])
    r3 = np.cross(r1, r2)
    r3 /= np.linalg.norm(r3) + 1e-12
    return f, math.degrees(math.acos(min(1.0, abs(float(r3[2])))))


def check_image(name: str, img: np.ndarray, spec: MatSpec | None = None) -> PhotoCheck:
    """Beoordeelt één foto. `spec`: de mat van de scan, als die al bekend is."""
    gray = to_work(img)
    h, w = gray.shape
    chk = PhotoCheck(name, w, h)
    det, spec = _detect_any(gray, spec, name)
    if det is None:
        chk.white = float(np.percentile(gray, 95))
        chk.notes.append("kalibratiemat niet gevonden: zorg dat de mat grotendeels en scherp in beeld is")
        if chk.white < 60:
            chk.notes.append("de foto is erg donker")
        return chk
    chess = np.asarray(make_board(spec).getChessboardCorners(), float)[:, :2]
    chk.mat, chk.corners = spec.name, int(len(det.ids))
    chk.mat_fraction = round(len(det.ids) / len(chess), 3)
    chk.ids = [int(i) for i in det.ids]
    chk.points = det.corners.round(3).tolist()
    chk.marker_ids = [] if det.marker_ids is None else [int(i) for i in det.marker_ids]
    H, _ = cv2.findHomography(chess[det.ids], det.corners)
    if H is None:
        chk.notes.append("mat gevonden, maar de hoeken zijn niet consistent")
        return chk
    white, contrast, clipped = _photometry(gray, H, det, spec)
    chk.white, chk.contrast, chk.clipped = round(white, 1), round(contrast, 1), round(clipped, 4)
    blur = _edge_blur(gray, H, det, spec)
    chk.blur_px = None if blur is None else round(blur, 2)
    f, tilt = _focal_and_tilt(H, w, h)
    chk.f_px, chk.tilt_deg = (None if f is None else round(f, 1)), round(tilt, 1)

    bad, weak = False, False
    if blur is not None and blur > BLUR_BAD:
        bad = True
        chk.notes.append(f"te onscherp (σ {blur:.1f} px): houd de telefoon stil, meer licht, tik om scherp te stellen")
    elif blur is not None and blur > BLUR_WARN:
        weak = True
        chk.notes.append(f"onscherp (σ {blur:.1f} px): houd de telefoon stil en zorg voor meer licht")
    if white < 45:
        bad = True
        chk.notes.append("veel te donker: meer (diffuus) licht")
    elif white < 70:
        weak = True
        chk.notes.append("donker: meer (diffuus) licht")
    if contrast < 50 and white >= 45:
        weak = True
        chk.notes.append("weinig contrast tussen wit en zwart (wazig, of licht dat op de mat schittert)")
    if clipped > 0.05:
        weak = True
        chk.notes.append(f"{100 * clipped:.0f}% van de mat is wit uitgebeten (glans of overbelichting): "
                         "vermijd direct licht")
    if chk.mat_fraction < 0.25:
        weak = True
        chk.notes.append(f"maar {100 * chk.mat_fraction:.0f}% van de mat in beeld: neem meer van de mat mee")
    chk.verdict = "onbruikbaar" if bad else "matig" if weak else "goed"
    return chk


def check_file(path: str | Path, spec: MatSpec | None = None) -> PhotoCheck:
    img = read_gray(path)
    if img is None:
        return PhotoCheck(Path(path).name, notes=["geen leesbare foto (JPG of PNG)"])
    return check_image(Path(path).name, img, spec)


def check_folder(folder: str | Path, mat: str | None = None, workers: int = 4) -> list[PhotoCheck]:
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXT)
    spec = None if not mat or mat.lower() == "auto" else get_spec(mat)
    if spec is None and paths:  # de mat één keer herkennen, niet per foto
        sample = [im for im in (read_gray(p) for p in paths[:: max(1, len(paths) // 6)][:6]) if im is not None]
        spec, _ = calib.identify_mat([to_work(im) for im in sample])
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:  # OpenCV geeft de GIL vrij
        return list(pool.map(lambda p: check_file(p, spec), paths))


# ----------------------------------------------------------------------------- per scan

def _detection(c: PhotoCheck) -> calib.BoardDetection:
    return calib.BoardDetection(c.name, c.width, c.height, np.array(c.points, float), np.array(c.ids, np.int32),
                                len(c.marker_ids), np.array(c.marker_ids, np.int32))


def _camera(checks: list[PhotoCheck], spec: MatSpec, size: tuple[int, int]) -> calib.CameraModel:
    """Snelle zelfkalibratie op hooguit 20 foto's; anders een camera uit de homografieën."""
    w, h = size
    same = [c for c in checks if (c.width, c.height) == size]
    if len(same) >= 6:
        pick = same[:: max(1, len(same) // 20)][:20]
        try:
            res = calib.calibrate([_detection(c) for c in pick], spec)
            if res.camera.rms_px < 3.0:
                return res.camera
        except (ValueError, cv2.error):
            pass
    fs = [c.f_px for c in checks if c.f_px]
    f = float(np.median(fs)) if fs else 0.8 * max(w, h)
    K = np.array([[f, 0.0, (w - 1) / 2], [0.0, f, (h - 1) / 2], [0.0, 0.0, 1.0]])
    return calib.CameraModel(K, np.zeros(5), w, h)


def _pose(c: PhotoCheck, spec: MatSpec, cam: calib.CameraModel) -> calib.Pose | None:
    if (c.width, c.height) == (cam.width, cam.height):
        return calib.solve_pose(_detection(c), spec, cam)
    if (c.height, c.width) == (cam.width, cam.height):  # staand opgeslagen: benadering met gespiegelde K
        K = cam.K.copy()
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = cam.K[1, 1], cam.K[0, 0], cam.K[1, 2], cam.K[0, 2]
        return calib.solve_pose(_detection(c), spec, calib.CameraModel(K, np.zeros(5), c.width, c.height))
    return None


def _probes(spec: MatSpec) -> tuple[np.ndarray, int]:
    board = make_board(spec)
    corners = board_to_mat(np.asarray(board.getChessboardCorners(), float), spec)
    centers = board_to_mat(np.array([np.asarray(c, float).reshape(4, 3).mean(axis=0) for c in board.getObjPoints()]),
                           spec)
    return np.vstack([corners, centers]), len(corners)


def locate_object(poses: dict, checks: dict, spec: MatSpec, cams: dict) -> tuple[np.ndarray, bool]:
    """Waar ligt het onderdeel? Punten van de mat (hoeken, markermiddens) die in beeld horen te zijn
    maar steeds ontbreken, liggen onder of vlak achter het onderdeel. Bovenaanzichten tellen zwaarder:
    daar valt de verstopte zone precies samen met het onderdeel."""
    probes, n_corners = _probes(spec)
    marker_ids = spec.marker_ids
    seen, hidden = np.zeros(len(probes)), np.zeros(len(probes))
    for name, pose in poses.items():
        c, cam = checks[name], cams[name]
        uv, z = calib.project(probes, pose, cam.K)
        m = 0.03 * max(c.width, c.height)
        inview = (z > 0) & (uv[:, 0] > m) & (uv[:, 0] < c.width - m) & (uv[:, 1] > m) & (uv[:, 1] < c.height - m)
        found = np.concatenate([np.isin(np.arange(n_corners), c.ids), np.isin(marker_ids, c.marker_ids)])
        wt = 1.0 if c.tilt_deg is not None and c.tilt_deg <= 30.0 else 0.3
        seen += wt * inview
        hidden += wt * (inview & ~found)
    ratio = hidden / np.maximum(seen, 1e-9)
    sel = (ratio >= 0.5) & (seen >= 1.5)
    if not sel.any():
        return np.array([spec.size_mm[0] / 2, spec.size_mm[1] / 2, 0.0]), False
    wts = ratio[sel] * seen[sel]
    return np.array([*(probes[sel, :2] * wts[:, None]).sum(axis=0) / wts.sum(), 0.0]), True


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _photos(n: int) -> str:
    return _plural(n, "foto", "foto's")


def coverage(poses: dict, point, verdicts: dict | None = None) -> dict:
    """Dekking rond het onderdeel (`point`, mat-mm): hoeveel foto's recht van boven, en per
    hoogteband uit welke richtingen (8 sectoren van 45°), met aanwijzingen voor wat ontbreekt."""
    point = np.asarray(point, float)
    out = {"recht_van_boven": 0, "bovenaanzicht": 0, "dekking": {b[0]: [0] * 8 for b in BANDS},
           "sectoren": list(SECTORS), "punten": [], "advies": []}
    min_tilt = 90.0
    for name, pose in poses.items():
        d = point - pose.center
        tilt = math.degrees(math.acos(float(np.clip(-d[2] / (np.linalg.norm(d) + 1e-12), -1, 1))))
        el = 90.0 - tilt
        az = math.degrees(math.atan2(pose.center[1] - point[1], pose.center[0] - point[0])) % 360.0
        out["recht_van_boven"] += tilt <= STEEP_DEG
        out["bovenaanzicht"] += tilt <= TOP_DEG
        min_tilt = min(min_tilt, tilt)
        for band, lo, hi, _ in BANDS:
            if lo <= el < hi:
                out["dekking"][band][int(((az + 22.5) % 360.0) // 45.0)] += 1
        out["punten"].append({"naam": name, "azimut": round(az, 1), "elevatie": round(el, 1),
                              "oordeel": (verdicts or {}).get(name, "goed")})
    advice = out["advies"]
    steep = out["recht_van_boven"]
    if steep < 4:
        need = 4 - steep
        extra = f" (de steilste foto kijkt nu {min_tilt:.0f}° schuin)" if steep == 0 and poses else ""
        advice.append(f"Nog {_photos(need)} recht boven het onderdeel: telefoon evenwijdig aan de mat, onderdeel "
                      f"midden in beeld, hele mat zichtbaar{extra}.")
    for band, _, _, deg in BANDS:
        missing = [SECTORS[k] for k in range(8) if out["dekking"][band][k] == 0]
        if len(missing) == 8:
            advice.append(f"Nog geen foto's {band} rondom (~{deg}° boven de mat): loop rond het onderdeel en maak om "
                          "de ~45° een foto.")
        elif missing:
            advice.append(f"{band.capitalize()} rondom (~{deg}°) ontbreken nog foto's van: {', '.join(missing)} "
                          "(boven = kant met de titel van de mat).")
    best_ring = max(sum(1 for k in out["dekking"][b[0]] if k) for b in BANDS)
    out["compleet"] = steep >= 3 and best_ring >= 6
    return out


def summarize(checks: list[PhotoCheck], spec: MatSpec | None = None) -> dict:
    """Dekking van de scan en concrete aanwijzingen voor ontbrekende of slechte foto's."""
    counts = Counter(c.verdict for c in checks)
    found = [c for c in checks if c.mat and c.verdict != "onbruikbaar"]
    mats = Counter(c.mat for c in found)
    if mats:
        spec = get_spec(mats.most_common(1)[0][0])
    out = {"fotos": len(checks), "oordelen": {k: counts.get(k, 0) for k in ("goed", "matig", "onbruikbaar")},
           "mat": spec.label if spec else None, "bruikbaar": 0, "recht_van_boven": 0, "bovenaanzicht": 0,
           "dekking": {b[0]: [0] * 8 for b in BANDS}, "sectoren": list(SECTORS), "object_mm": None,
           "punten": [], "klaar": False, "advies": []}
    advice: list[str] = out["advies"]
    if spec is None:
        if checks:
            advice.append("De kalibratiemat is nog in geen enkele foto gevonden: zorg dat de mat grotendeels, scherp "
                          "en recht van boven of schuin in beeld is.")
        return out
    found = [c for c in found if c.mat == spec.name]
    sizes = Counter((c.width, c.height) for c in found)
    size = sizes.most_common(1)[0][0]
    other_size = [c for c in found if (c.width, c.height) not in (size, size[::-1])]
    cam = _camera(found, spec, size)
    poses, cams = {}, {}
    for c in found:
        if (c.width, c.height) not in (size, size[::-1]):
            continue
        p = _pose(c, spec, cam)
        if p is not None:
            poses[c.name], cams[c.name] = p, cam
    by_name = {c.name: c for c in found}
    n_ok = len(poses)
    if n_ok < 6:
        advice.append(f"Te weinig bruikbare foto's ({n_ok}): minimaal 6 voor de kalibratie, beter 30-60.")
    elif n_ok < 20:
        advice.append(f"Maak meer foto's: nu {n_ok} bruikbaar, mik op 30-60.")
    located = False
    if poses:
        point, located = locate_object(poses, by_name, spec, cams)
        cov = coverage(poses, point, {n: c.verdict for n, c in by_name.items()})
        advice += cov.pop("advies")
        complete = cov.pop("compleet")
        out.update(cov, object_mm=[round(float(point[0]), 1), round(float(point[1]), 1)])
        out["klaar"] = n_ok >= 15 and complete
    out.update(bruikbaar=n_ok, brandpunt_px=round(float(cam.K[0, 0]), 1))
    if poses and not located:
        advice.append("Het onderdeel is nog niet gevonden op de mat: leg het midden op het geblokte deel en maak "
                      "foto's recht van boven.")
    blurry = sum(1 for c in checks if c.blur_px and c.blur_px > BLUR_WARN)
    if blurry:
        verb = "is" if blurry == 1 else "zijn"
        advice.append(f"{_photos(blurry)} {verb} onscherp: houd de telefoon stil, zorg voor meer licht en tik op het "
                      "scherm om scherp te stellen.")
    no_mat = sum(1 for c in checks if not c.mat)
    if no_mat:
        advice.append(f"In {_photos(no_mat)} is de mat niet gevonden: die worden overgeslagen.")
    glare = sum(1 for c in checks if c.clipped and c.clipped > 0.05)
    if glare:
        advice.append(f"Glans of overbelichting in {_photos(glare)}: vermijd direct licht of glanzend papier.")
    if other_size:
        verb = "heeft" if len(other_size) == 1 else "hebben"
        advice.append(f"{_photos(len(other_size))} {verb} een ander beeldformaat (andere camera, zoom of bijgesneden) "
                      "en worden niet gebruikt: gebruik één camera zonder zoom.")
    if out["klaar"] and not advice:
        advice.append("Deze fotoset ziet er goed uit: klaar om te verwerken.")
    return out


def report_lines(checks: list[PhotoCheck], summary: dict) -> list[str]:
    """Leesbaar overzicht voor de opdrachtregel."""
    head = f"{'foto':28s} {'oordeel':12s} {'mat':8s} {'hoeken':>6s} {'σ px':>5s} {'wit':>4s} {'kant.':>5s}"
    lines = [head + "  opmerkingen"]
    for c in checks:
        blur = f"{c.blur_px:5.2f}" if c.blur_px is not None else "    -"
        white = f"{c.white:4.0f}" if c.white is not None else "   -"
        tilt = f"{c.tilt_deg:4.0f}°" if c.tilt_deg is not None else "    -"
        lines.append(f"{c.name[:28]:28s} {c.verdict:12s} {(c.mat or '-'):8s} {c.corners:6d} {blur} {white} {tilt}  "
                     + "; ".join(c.notes))
    s = summary
    lines += ["", (f"mat: {s['mat'] or 'niet gevonden'}; bruikbaar: {s['bruikbaar']} van {s['fotos']}; recht van "
                   f"boven (≤ 8°): {s['recht_van_boven']}; bovenaanzicht (≤ 25°): {s['bovenaanzicht']}")]
    for band, _, _, deg in BANDS:
        cells = "  ".join(f"{name}:{n}" for name, n in zip(SECTORS, s["dekking"][band]))
        lines.append(f"{band} (~{deg}°): {cells}")
    lines.append("")
    lines += [f"- {a}" for a in s["advies"]] or ["- geen aanwijzingen"]
    lines.append("klaar om te verwerken" if s["klaar"] else "nog niet compleet (verwerken kan wel)")
    return lines
