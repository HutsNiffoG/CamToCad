"""Mat als tracker: detectie van het ChArUco-bord, zelfkalibratie en cameraposes.

Alle foto's van een scan tonen dezelfde, vlakke mat met bekende maten. Daarmee is de
camera zelf te kalibreren (methode van Zhang: brandpuntsafstand, hoofdpunt, vervorming) en
volgt per foto een metrische camerapose in mat-coördinaten. De schaal komt rechtstreeks
uit de mat: er is geen VIO (ARCore) en geen schaaldrift nodig.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from .mat import PRESETS, MatSpec, board_to_mat_transform, make_board


@dataclass
class BoardDetection:
    name: str
    width: int
    height: int
    corners: np.ndarray  # (N, 2) subpixel-hoeken van het schaakbord
    ids: np.ndarray  # (N,) hoek-ID's
    n_markers: int
    marker_ids: np.ndarray | None = None  # ID's van de gevonden markers


@dataclass
class CameraModel:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    rms_px: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "K": self.K.tolist(),
            "dist": self.dist.ravel().tolist(),
            "width": self.width,
            "height": self.height,
            "rms_px": self.rms_px,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        return cls(np.array(d["K"], float), np.array(d["dist"], float), int(d["width"]), int(d["height"]),
                   float(d.get("rms_px", float("nan"))))

    def scaled(self, factor: float) -> "CameraModel":
        """Model voor een beeld dat met `factor` is geschaald (vervormingscoëfficiënten blijven gelijk)."""
        K = self.K.copy()
        K[:2] *= factor
        return CameraModel(K, self.dist.copy(), int(round(self.width * factor)), int(round(self.height * factor)),
                           self.rms_px * factor)


@dataclass
class Pose:
    """Camerapose als transformatie mat → camera: X_cam = R @ X_mat + t (mm)."""

    name: str
    R: np.ndarray
    t: np.ndarray
    rms_px: float = float("nan")
    n_corners: int = 0

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def to_dict(self) -> dict:
        return {"name": self.name, "R": self.R.tolist(), "t": self.t.tolist(),
                "rms_px": self.rms_px, "n_corners": self.n_corners}

    @classmethod
    def from_dict(cls, d: dict) -> "Pose":
        return cls(d["name"], np.array(d["R"], float), np.array(d["t"], float),
                   float(d.get("rms_px", float("nan"))), int(d.get("n_corners", 0)))


@dataclass
class CalibrationResult:
    camera: CameraModel
    poses: dict[str, Pose]
    rejected: dict[str, str] = field(default_factory=dict)  # foto → reden


def make_detector(board) -> "cv2.aruco.CharucoDetector":
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    charuco = cv2.aruco.CharucoParameters()
    charuco.tryRefineMarkers = True
    return cv2.aruco.CharucoDetector(board, charuco, params)


def detect(gray: np.ndarray, board, name: str = "", detector=None, bias=None) -> BoardDetection | None:
    """Zoekt het bord in een grijswaardenbeeld; None als er te weinig hoeken zijn.

    `bias`: systematische verschuiving van de detector (zie `detector_bias`), wordt afgetrokken.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    detector = detector or make_detector(board)
    corners, ids, _, marker_ids = detector.detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None
    corners = corners.reshape(-1, 2).astype(np.float64)
    if bias is not None:
        corners = corners - np.asarray(bias, float)
    return BoardDetection(
        name=name, width=gray.shape[1], height=gray.shape[0],
        corners=corners, ids=ids.ravel().astype(np.int32),
        n_markers=0 if marker_ids is None else len(marker_ids),
        marker_ids=None if marker_ids is None else np.asarray(marker_ids).ravel().astype(np.int32),
    )


ALL_MARKERS = "DICT_5X5_1000"  # bevat DICT_5X5_250 (v1) als eerste 250 markers: één zoektocht voor alle matten


def identify_mat(images, candidates=None, max_images: int = 8) -> tuple[MatSpec | None, dict[str, int]]:
    """Welke mat staat op de foto's (formaat en versie)?

    Zoekt markers met een woordenboek dat alle matten dekt en houdt de matten over waarvan
    markers gezien zijn. v2-matten hebben elk een eigen ID-bereik, dus dat is meestal één mat.
    De v1-matten A4 en A3 delen ID's: dan beslist het aantal schaakbordhoeken dat met de indeling
    van elke mat klopt. `images`: beelden of (naam, beeld)-paren. Geeft (mat of None, {mat: score}).
    """
    candidates = list(candidates or PRESETS.values())
    sample = images[:: max(1, len(images) // max_images)][:max_images]
    sample = [im[1] if isinstance(im, tuple) else im for im in sample]
    sample = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im for im in sample]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ALL_MARKERS))
    finder = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    seen: set[int] = set()
    for im in sample:
        _, ids, _ = finder.detectMarkers(im)
        if ids is not None:
            seen.update(int(i) for i in ids.ravel())
    hits = {s.name: len(seen & set(s.marker_ids.tolist())) for s in candidates}
    plausible = [s for s in candidates if hits[s.name] >= 2]
    if len(plausible) <= 1:
        return (plausible[0] if plausible else None), {s.name: hits[s.name] for s in plausible}
    corners = {}
    for s in plausible:
        board = make_board(s)
        det = make_detector(board)
        found = (detect(im, board, detector=det) for im in sample)
        corners[s.name] = sum(len(d.ids) for d in found if d is not None)
    best = max(plausible, key=lambda s: corners[s.name])
    return (best if corners[best.name] > 0 else None), corners


_BIAS: dict[tuple, np.ndarray] = {}


def detector_bias(spec: MatSpec) -> np.ndarray:
    """Systematische verschuiving (px) van de ChArUco-hoeken in de geïnstalleerde OpenCV-versie.

    OpenCV 4.x legt de hoeken gemiddeld ~0,5 px naar rechtsonder (een andere pixelconventie), 5.x
    niet. Voor de kalibratie maakt dat weinig uit (het hoofdpunt schuift mee), maar dan valt de
    voorspelde mat (masks.py) een halve pixel naast de foto en worden silhouetten en maten
    onzuiver. Gemeten op synthetische beelden van de mat met bekende geometrie, één keer per proces.
    """
    spec = spec.nominal()  # een eigenschap van de detector, niet van de printschaal
    key = (spec, cv2.__version__)
    if key not in _BIAS:
        _BIAS[key] = _measure_bias(spec)
    return _BIAS[key]


def _measure_bias(spec: MatSpec) -> np.ndarray:
    from .mat import rasterize_board
    from .render import look_at

    board = make_board(spec)
    detector = make_detector(board)
    raster = rasterize_board(spec, 10.0, 3.0)
    A, a = board_to_mat_transform(spec)
    chess = np.asarray(board.getChessboardCorners(), float).reshape(-1, 3) @ A.T + a
    w, h, f, s = 1600, 1200, 1300.0, 2
    K = np.array([[f, 0.0, (w - 1) / 2], [0.0, f, (h - 1) / 2], [0.0, 0.0, 1.0]])
    Ks = K.copy()
    Ks[:2, :2] *= s
    Ks[:2, 2] = s * K[:2, 2] + 0.5 * (s - 1)  # supersampling met dezelfde pixelconventie
    to_raster = np.linalg.inv(raster.mat_to_pixel_matrix())
    target = np.array([spec.board_w_mm / 2, spec.board_h_mm / 2, 0.0])
    distance = 1.4 * f * spec.board_w_mm / w
    offsets = []
    for tilt, az in ((3, 0), (30, 40), (30, 160), (30, 280), (45, 100), (45, 220)):
        ti, az_ = np.radians(tilt), np.radians(az)
        center = target + distance * np.array([np.sin(ti) * np.cos(az_), np.sin(ti) * np.sin(az_), np.cos(ti)])
        R, t = look_at(center, target)
        Hm = Ks @ np.column_stack([R[:, 0], R[:, 1], t]) @ to_raster
        img = cv2.warpPerspective(raster.image.astype(np.float32), Hm, (w * s, h * s), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) * 0.85 + 15.0
        det = detect(np.clip(img, 0, 255).astype(np.uint8), board, detector=detector)
        if det is None:
            continue
        uv, _ = project(chess[det.ids], Pose("", R, t), K)
        offsets.append(det.corners - uv)
    if not offsets:
        return np.zeros(2)
    bias = np.median(np.vstack(offsets), axis=0)
    return bias if np.all(np.abs(bias) < 1.5) else np.zeros(2)


def _object_image_points(det: BoardDetection, board, spec: MatSpec) -> tuple[np.ndarray, np.ndarray]:
    """Hoeken als (werkelijke board-mm, pixels): de printschaal zit in de objectpunten, zodat
    kalibratie en poses kloppen met de mat zoals die geprint is."""
    obj, img = board.matchImagePoints(det.corners.reshape(-1, 1, 2).astype(np.float32), det.ids.reshape(-1, 1))
    obj = obj.reshape(-1, 3) * np.array([spec.scale_x, spec.scale_y, 1.0])
    return obj.reshape(-1, 1, 3).astype(np.float32), img.reshape(-1, 1, 2).astype(np.float32)


def _to_mat_pose(name: str, rvec, tvec, spec: MatSpec, rms: float, n: int) -> Pose:
    A, a = board_to_mat_transform(spec)
    A = A @ np.diag([1.0 / spec.scale_x, 1.0 / spec.scale_y, 1.0])  # objectpunten zijn al geschaald
    R_cb, _ = cv2.Rodrigues(np.asarray(rvec, float))
    t_cb = np.asarray(tvec, float).ravel()
    return Pose(name, R_cb @ A, t_cb - R_cb @ A @ a, rms, n)


def _view_rms(obj, img, rvec, tvec, K, dist) -> float:
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1))))


def calibrate(
    detections: list[BoardDetection], spec: MatSpec, *, min_corners: int = 12, fix_k3: bool = True,
    camera: CameraModel | None = None,
) -> CalibrationResult:
    """Zelfkalibratie over alle foto's (of alleen poses als `camera` gegeven is)."""
    board = make_board(spec)
    rejected: dict[str, str] = {}
    sizes = Counter((d.width, d.height) for d in detections)
    if not sizes:
        raise ValueError("Geen enkele foto met een herkenbare kalibratiemat")
    (w, h), _ = sizes.most_common(1)[0]

    usable: list[tuple[BoardDetection, np.ndarray, np.ndarray]] = []
    for d in detections:
        if (d.width, d.height) != (w, h):
            rejected[d.name] = f"afwijkend formaat {d.width}x{d.height}"
        elif len(d.ids) < min_corners:
            rejected[d.name] = f"te weinig mathoeken ({len(d.ids)})"
        elif board.checkCharucoCornersCollinear(d.ids.reshape(-1, 1)):
            rejected[d.name] = "mathoeken liggen op één lijn"
        else:
            obj, img = _object_image_points(d, board, spec)
            usable.append((d, obj, img))

    if camera is None:
        if len(usable) < 6:
            raise ValueError(
                f"Te weinig bruikbare foto's voor kalibratie ({len(usable)}; minimaal 6 met ≥ {min_corners} hoeken)"
            )
        flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
        for round_ in range(4):  # opnieuw zonder uitschieters, tot er geen meer zijn
            objs = [u[1] for u in usable]
            imgs = [u[2] for u in usable]
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, (w, h), None, None, flags=flags)
            errs = [_view_rms(o, i, r, t, K, dist) for o, i, r, t in zip(objs, imgs, rvecs, tvecs)]
            limit = max(3.0 * float(np.median(errs)), 2.0)
            keep = [k for k, e in enumerate(errs) if e <= limit]
            # stoppen zolang rvecs/tvecs/errs nog precies bij `usable` horen (anders krijgen foto's
            # de pose van hun buurman)
            if len(keep) == len(usable) or len(keep) < 6 or round_ == 3:
                break
            for k, e in enumerate(errs):
                if e > limit:
                    rejected[usable[k][0].name] = f"reprojectiefout {e:.2f} px"
            usable = [usable[k] for k in keep]
        cam = CameraModel(K, dist.ravel(), w, h, float(rms))
        poses = {
            u[0].name: _to_mat_pose(u[0].name, r, t, spec, e, len(u[0].ids))
            for u, r, t, e in zip(usable, rvecs, tvecs, errs)
        }
    else:
        cam = camera
        poses = {}
        for d, _, _ in usable:
            pose = solve_pose(d, spec, cam)
            if pose is None:
                rejected[d.name] = "pose niet te bepalen"
            else:
                poses[d.name] = pose
    return CalibrationResult(cam, poses, rejected)


def solve_pose(det: BoardDetection, spec: MatSpec, cam: CameraModel) -> Pose | None:
    """Pose van één foto bij een bekende camera (IPPE voor vlakke doelen + LM-verfijning)."""
    board = make_board(spec)
    obj, img = _object_image_points(det, board, spec)
    if len(obj) < 6:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, cam.dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, cam.K, cam.dist, rvec, tvec)
    rms = _view_rms(obj, img, rvec, tvec, cam.K, cam.dist)
    return _to_mat_pose(det.name, rvec, tvec, spec, rms, len(obj))


def project(points_mat: np.ndarray, pose: Pose, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole-projectie (zonder vervorming) van mat-punten; geeft (N, 2) pixels en diepte."""
    pc = points_mat @ pose.R.T + pose.t
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (pc[:, :2] / z[:, None]) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv, z


def undistort(image: np.ndarray, cam: CameraModel) -> np.ndarray:
    """Verwijdert lensvervorming; het resultaat is een pinholebeeld met dezelfde K."""
    if not np.any(np.abs(cam.dist) > 0):
        return image
    return cv2.undistort(image, cam.K, cam.dist)
