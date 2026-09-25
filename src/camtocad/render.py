"""Synthetische scans: rendert een CAD-object op de kalibratiemat vanuit bekende camera's.

Gebruikt door de tests en door `camtocad demo`: de 'render-and-reconstruct'-aanpak uit
ARCHITECTURE.md §6.12 in het klein. Eenvoudige software-rasterizer (z-buffer, vlakke
Lambert-belichting), 2x supersampling tegen aliasing, optioneel lensvervorming en ruis.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calib import CameraModel, Pose
from .mat import BoardRaster, MatSpec, rasterize_board


@dataclass
class SyntheticView:
    name: str
    image: np.ndarray  # uint8 grijswaarden, met vervorming als het cameramodel die heeft
    mask: np.ndarray  # bool, object zichtbaar (grondwaarheid, zonder vervorming)
    pose: Pose


def look_at(center, target, up=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Rotatie en translatie (mat → camera) voor een camera in `center` die naar `target` kijkt."""
    center, target = np.asarray(center, float), np.asarray(target, float)
    z = target - center
    z /= np.linalg.norm(z)
    up = np.asarray(up, float)
    if abs(np.dot(z, up)) > 0.98:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ center


def scan_poses(target, distance: float, rings=((35.0, 14), (55.0, 14), (72.0, 12)), top_views: int = 6,
               rng: np.random.Generator | None = None) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Camera's in ringen rond het object plus een paar (bijna) loodrechte bovenaanzichten."""
    rng = rng or np.random.default_rng(0)
    target = np.asarray(target, float)
    out = []
    for elev, count in rings:
        offset = rng.uniform(0, 360.0 / count)
        for k in range(count):
            az = np.radians(offset + 360.0 * k / count)
            el = np.radians(elev + rng.uniform(-3, 3))
            d = distance * rng.uniform(0.95, 1.05)
            c = target + d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            R, t = look_at(c, target + rng.normal(0, 3.0, 3))
            out.append((f"ring{int(elev):02d}_{k:02d}", R, t))
    for k in range(top_views):
        lateral = rng.normal(0, 12.0, 2)
        c = target + np.array([lateral[0], lateral[1], distance])
        R, t = look_at(c, target + np.array([lateral[0] * 0.5, lateral[1] * 0.5, 0.0]))
        out.append((f"top_{k:02d}", R, t))
    return out


def tessellate(shape, tolerance: float = 0.05, angular: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Driehoeksmesh (vertices in mm, driehoeken) van een CadQuery-object."""
    solid = shape.val() if hasattr(shape, "val") else shape
    verts, tris = solid.tessellate(tolerance, angular)
    V = np.array([[v.x, v.y, v.z] for v in verts], float)
    F = np.array(tris, np.int64)
    return V, F


def _rasterize_mesh(V, F, K, R, t, width, height, light, albedo, ambient):
    img = np.zeros((height, width), np.float32)
    zbuf = np.full((height, width), np.inf, np.float32)
    pc = V @ R.T + t
    z = pc[:, 2]
    uv = pc[:, :2] / z[:, None] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    light = np.asarray(light, float) / np.linalg.norm(light)
    tri_n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    tri_n /= np.linalg.norm(tri_n, axis=1, keepdims=True) + 1e-12
    shade = albedo * 255.0 * (ambient + (1 - ambient) * np.abs(tri_n @ light))
    for f, s in zip(F, shade):
        if np.any(z[f] <= 1e-3):
            continue
        (x0, y0), (x1, y1), (x2, y2) = uv[f]
        xmin, xmax = int(max(np.floor(min(x0, x1, x2)), 0)), int(min(np.ceil(max(x0, x1, x2)), width - 1))
        ymin, ymax = int(max(np.floor(min(y0, y1, y2)), 0)), int(min(np.ceil(max(y0, y1, y2)), height - 1))
        if xmin > xmax or ymin > ymax:
            continue
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-9:
            continue
        xs, ys = np.meshgrid(np.arange(xmin, xmax + 1, dtype=np.float64), np.arange(ymin, ymax + 1, dtype=np.float64))
        w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
        w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        zz = 1.0 / (w0 / z[f[0]] + w1 / z[f[1]] + w2 / z[f[2]])
        sub_z = zbuf[ymin:ymax + 1, xmin:xmax + 1]
        upd = inside & (zz < sub_z)
        sub_z[upd] = zz[upd]
        img[ymin:ymax + 1, xmin:xmax + 1][upd] = s
    return img, np.isfinite(zbuf)


def _distort(img: np.ndarray, cam: CameraModel, border: int) -> np.ndarray:
    """Past lensvervorming toe: voor elke vervormde pixel wordt de ideale bronpixel opgezocht."""
    h, w = img.shape[:2]
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([us.ravel(), vs.ravel()], axis=1).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(pts, cam.K, cam.dist, P=cam.K).reshape(h, w, 2)
    return cv2.remap(img, ideal[..., 0], ideal[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def render_view(raster: BoardRaster, cam: CameraModel, R: np.ndarray, t: np.ndarray, mesh=None, *,
                albedo: float = 0.55, light=(0.35, -0.45, 0.82), ambient: float = 0.35,
                noise: float = 1.5, blur: float = 0.5, table: int = 105, gradient: float = 0.08,
                rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rendert één view; geeft (grijswaardenbeeld uint8, objectmasker bool)."""
    rng = rng or np.random.default_rng()
    s = 2  # supersampling
    Ks = cam.K.copy()
    Ks[:2, :2] *= s
    Ks[0, 2] = s * cam.K[0, 2] + 0.5 * (s - 1)
    Ks[1, 2] = s * cam.K[1, 2] + 0.5 * (s - 1)
    W, H = cam.width * s, cam.height * s
    plane = Ks @ np.column_stack([R[:, 0], R[:, 1], t])
    Hm = plane @ np.linalg.inv(raster.mat_to_pixel_matrix())
    bg = cv2.warpPerspective(raster.image, Hm, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=table).astype(np.float32)
    # papier met iets witbalans/belichtingsverloop, zoals in een echte foto
    bg = bg * 0.86 + 12.0
    if mesh is not None:
        obj, cover = _rasterize_mesh(mesh[0], mesh[1], Ks, R, t, W, H, light, albedo, ambient)
        img = np.where(cover, obj, bg)
    else:
        cover = np.zeros((H, W), bool)
        img = bg
    img = cv2.resize(img, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(cover.astype(np.float32), (cam.width, cam.height), interpolation=cv2.INTER_AREA) > 0.5
    if gradient:
        yy, xx = np.mgrid[0:cam.height, 0:cam.width].astype(np.float32)
        img *= 1.0 + gradient * ((xx / cam.width - 0.5) + 0.5 * (yy / cam.height - 0.5))
    if np.any(cam.dist):
        img = _distort(img, cam, table)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise:
        img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8), mask


def place(shape, spec: MatSpec, angle_deg: float = 0.0, offset=(0.0, 0.0)):
    """Zet een CadQuery-object (ontworpen rond de oorsprong, z ≥ 0) midden op de mat, gedraaid om Z."""
    return (shape.rotate((0, 0, 0), (0, 0, 1), angle_deg)
            .translate((spec.board_w_mm / 2 + offset[0], spec.board_h_mm / 2 + offset[1], 0)))


def default_camera(width: int = 1600, height: int = 1200, f: float = 1300.0,
                   dist=(-0.06, 0.02, 0.0, 0.0, 0.0)) -> CameraModel:
    K = np.array([[f, 0, (width - 1) / 2 + 3.0], [0, f, (height - 1) / 2 - 2.0], [0, 0, 1]], float)
    return CameraModel(K, np.array(dist, float), width, height)


def render_scan(shape, spec: MatSpec, cam: CameraModel | None = None, *, distance: float = 330.0,
                rings=((35.0, 14), (55.0, 14), (72.0, 12)), top_views: int = 6, seed: int = 0,
                raster: BoardRaster | None = None) -> list[SyntheticView]:
    """Rendert een volledige synthetische scan van `shape` (al geplaatst in mat-coördinaten)."""
    rng = np.random.default_rng(seed)
    cam = cam or default_camera()
    raster = raster or rasterize_board(spec, px_per_mm=10.0, margin_mm=3.0)
    mesh = tessellate(shape) if shape is not None else None
    if mesh is not None:
        lo, hi = mesh[0].min(axis=0), mesh[0].max(axis=0)
        target = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] / 2])
    else:
        target = np.array([spec.board_w_mm / 2, spec.board_h_mm / 2, 0.0])
    views = []
    for name, R, t in scan_poses(target, distance, rings, top_views, rng):
        img, mask = render_view(raster, cam, R, t, mesh, rng=rng)
        views.append(SyntheticView(name, img, mask, Pose(name, R, t)))
    return views
