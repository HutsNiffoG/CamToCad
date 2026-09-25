"""Objectmaskers via de bekende mat-achtergrond.

Met de camerapose is precies te voorspellen hoe de mat er in elke foto uitziet. Pixels die
daarvan afwijken horen bij het object. Drie klassen per pixel:

* `fg`    – objectpixel (wijkt af van de voorspelde mat);
* `bg`    – *zeker* mat: klopt met de voorspelling én de voorspelling heeft daar textuur;
* overig  – onbekend (bijv. egale zwarte vakken, of buiten de mat).

Alleen zekere mat-pixels mogen voxels wegsnijden (hull.py). Zo veroorzaakt een donker
object op een zwart vak geen gat in het model: daar is het simpelweg 'onbekend'.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calib import Pose
from .mat import BoardRaster


@dataclass
class ViewMasks:
    fg: np.ndarray  # objectpixels
    bg: np.ndarray  # zekere mat, met een veiligheidsmarge rond het object (voor het uitsnijden)
    valid: np.ndarray  # pixel valt op de mat
    edge_bg: np.ndarray | None = None  # zekere mat zonder marge (voor de modelverfijning)
    sigma: float = 0.0  # ruisniveau van het residu (grijswaarden)


def predict_background(raster: BoardRaster, K: np.ndarray, pose: Pose, size: tuple[int, int]):
    """Voorspelde grijswaarde van de mat en een masker van pixels die op de mat vallen."""
    w, h = size
    s = 2
    Ks = K.copy()
    Ks[:2, :2] *= s
    Ks[0, 2], Ks[1, 2] = s * K[0, 2] + 0.5 * (s - 1), s * K[1, 2] + 0.5 * (s - 1)
    Hm = Ks @ np.column_stack([pose.R[:, 0], pose.R[:, 1], pose.t]) @ np.linalg.inv(raster.mat_to_pixel_matrix())
    pred = cv2.warpPerspective(raster.image.astype(np.float32), Hm, (w * s, h * s), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside = cv2.warpPerspective(np.full(raster.image.shape, 255, np.uint8), Hm, (w * s, h * s),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(inside, (w, h), interpolation=cv2.INTER_AREA) >= 255
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return pred, valid


def _robust_affine(o: np.ndarray, p: np.ndarray, sel: np.ndarray) -> tuple[float, float, float]:
    """Robuuste fit o ≈ a·p + b; geeft (a, b, sigma)."""
    ov, pv = o[sel][::7], p[sel][::7]
    a, b, thr = 1.0, float(np.median(ov - pv)), 80.0
    sigma = 10.0
    for _ in range(4):
        r = ov - (a * pv + b)
        keep = np.abs(r) < thr
        if keep.sum() < 50:
            break
        A = np.column_stack([pv[keep], np.ones(keep.sum())])
        (a, b), *_ = np.linalg.lstsq(A, ov[keep], rcond=None)
        r = ov[keep] - (a * pv[keep] + b)
        sigma = 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-3
        thr = max(4.0 * sigma, 6.0)
    return float(a), float(b), sigma


def _refine_boundary(fg: np.ndarray, o: np.ndarray, bgv: np.ndarray, valid: np.ndarray, tau: float) -> np.ndarray:
    """Herclassificeert pixels vlak bij de objectrand met de 50%-regel.

    Een lage drempel op een (vervaagde) rand legt de grens te ver naar buiten. Hier telt een
    randpixel als object als hij dichter bij de lokale objectgrijswaarde ligt dan bij de
    voorspelde mat: de grens ligt dan op het halve contrast, dus op de echte rand.
    """
    fg8 = fg.astype(np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    band = (cv2.dilate(fg8, k5) > 0) & ~(cv2.erode(fg8, k5) > 0) & valid
    interior = cv2.erode(fg8, k5).astype(np.float32)
    num = cv2.boxFilter(o * interior, -1, (11, 11), normalize=False)
    den = cv2.boxFilter(interior, -1, (11, 11), normalize=False)
    obj = np.where(den > 0, num / np.maximum(den, 1e-6), 0.0)
    contrast = np.abs(obj - bgv)
    usable = band & (den > 0) & (contrast > 2 * tau)
    decision = np.abs(o - bgv) > 0.5 * contrast
    out = fg.copy()
    out[usable] = decision[usable]
    return out


def classify(observed: np.ndarray, pred: np.ndarray, valid: np.ndarray, *, k_sigma: float = 6.0,
             tau_min: float = 14.0, texture_min: float = 18.0, min_area_frac: float = 2e-4,
             misreg_px: float = 0.4) -> ViewMasks:
    """Deelt een (ontvervormd) grijswaardenbeeld in: object, zekere mat, onbekend."""
    if observed.ndim == 3:
        observed = cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY)
    o = cv2.GaussianBlur(observed.astype(np.float32), (0, 0), 0.8)
    p = cv2.GaussianBlur(pred.astype(np.float32), (0, 0), 0.8)
    a, b, sigma = _robust_affine(o, p, valid)

    # traag verlopende belichtingsverschillen wegwerken (genormaliseerde convolutie over de mat)
    r = o - (a * p + b)
    w = (valid & (np.abs(r) < max(4 * sigma, 8.0))).astype(np.float32)
    num = cv2.GaussianBlur(r * w, (0, 0), 35.0)
    den = cv2.GaussianBlur(w, (0, 0), 35.0)
    o = o - np.where(den > 0.05, num / np.maximum(den, 1e-6), 0.0)

    # tolerantie voor een kleine posefout: evenredig met de lokale gradiënt van de voorspelling
    # (een vast 3x3-min/max-venster is te ruim: objectpixels op patroonranden zouden dan 'mat' lijken)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    tol = misreg_px * abs(a) * np.sqrt(gx * gx + gy * gy)
    res = np.maximum(np.abs(o - (a * p + b)) - tol, 0.0)
    k3 = np.ones((3, 3), np.uint8)
    _, _, sigma = _robust_affine(o, p, valid)
    tau = max(k_sigma * sigma, tau_min)
    fg = (valid & (res > tau)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    min_area = min_area_frac * fg.size
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    fg = keep[labels]
    fg = _refine_boundary(fg, o, a * p + b, valid, tau)

    mean = cv2.blur(p, (15, 15))
    texture = np.sqrt(np.maximum(cv2.blur(p * p, (15, 15)) - mean * mean, 0.0)) * abs(a)
    edge_bg = valid & ~fg & (texture > texture_min)
    near_fg = cv2.dilate(fg.astype(np.uint8), k3) > 0
    return ViewMasks(fg=fg, bg=edge_bg & ~near_fg, valid=valid, edge_bg=edge_bg, sigma=sigma)
