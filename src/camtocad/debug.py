"""Diagnose-uitvoer: beelden en cijfers die laten zien wat de pipeline in de foto's heeft gezien.

Staat in `<uitvoer>/debug/`, ook als de verwerking halverwege stopt:

* `masker_<foto>.jpg`   – objectmasker over de foto (oranje = object, blauw = zekere mat, paars =
                          object en mat daar even donker of licht: geen bewijs);
* `lokalisatie.png`     – grove visual hull van boven over de mat (lichter = hoger), met zoekgebied;
* `bovenaanzicht.png`   – stemmen van de bovenaanzichten op de gekozen hoogte (geel = allemaal
                          object), met de startcontour in cyaan;
* `diagnose.json`       – per foto: bovenaanzicht ja/nee, kanteling, objectaandeel, ruis, hoeken.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .hull import VoxelGrid
from .initial import Footprint, tilt_deg
from .masks import ViewMasks


def mask_overlay(img: np.ndarray, m: ViewMasks) -> np.ndarray:
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32) * 0.6
    base[m.bg] = base[m.bg] * 0.7 + np.array([255, 120, 0]) * 0.3  # zekere mat: blauw
    base[m.fg] = base[m.fg] * 0.5 + np.array([0, 140, 255]) * 0.5  # object: oranje
    if m.amb is not None:
        base[m.amb] = base[m.amb] * 0.5 + np.array([200, 50, 160]) * 0.5  # niet te onderscheiden: paars
    return np.clip(base, 0, 255).astype(np.uint8)


def _upscale(img: np.ndarray, min_width: int = 700) -> np.ndarray:
    f = max(1, int(np.ceil(min_width / max(img.shape[1], 1))))
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) if f > 1 else img


def hull_image(coarse: VoxelGrid, board_bounds, bounds=None) -> np.ndarray:
    """Bovenaanzicht van de grove hull: per kolom de hoogste bezette voxel (mat-Y omhoog)."""
    occ = coarse.occ
    z = np.where(occ.any(axis=2), occ.shape[2] - np.argmax(occ[:, :, ::-1], axis=2), 0).astype(np.float32)
    img = (z.T[::-1] / max(z.max(), 1) * 255).astype(np.uint8)  # rijen = -Y
    col = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    col[img == 0] = (40, 40, 40)
    col = _upscale(col, 720)
    if bounds is not None:
        s = col.shape[1] / occ.shape[0]
        x0 = int((bounds[0] - coarse.origin[0] + coarse.voxel / 2) / coarse.voxel * s)
        x1 = int((bounds[1] - coarse.origin[0] + coarse.voxel / 2) / coarse.voxel * s)
        y0 = col.shape[0] - int((bounds[3] - coarse.origin[1] + coarse.voxel / 2) / coarse.voxel * s)
        y1 = col.shape[0] - int((bounds[2] - coarse.origin[1] + coarse.voxel / 2) / coarse.voxel * s)
        cv2.rectangle(col, (x0, y0), (x1, y1), (255, 255, 255), 1)
    return col


def footprint_image(fp: Footprint) -> np.ndarray:
    """Stemfractie van de bovenaanzichten (geel = alle foto's zien object), startcontour in cyaan."""
    frac = np.nan_to_num(fp.frac, nan=-1.0)
    img = (np.clip(frac, 0, 1) * 255).astype(np.uint8)[::-1]  # rijen = -Y
    col = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    col[frac[::-1] < 0] = (70, 70, 70)  # buiten beeld
    contours, _ = cv2.findContours(fp.mask[::-1].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    f = max(1, int(np.ceil(700 / max(col.shape[1], 1))))
    col = _upscale(col, 700)
    cv2.drawContours(col, [c * f + f // 2 for c in contours], -1, (255, 255, 0), 1 if f > 1 else 2)
    return col


def view_table(views: list, dets: dict, top_names: set[str], blur: dict | None = None,
               placement: dict | None = None) -> list[dict]:
    rows = []
    blur = blur or {}
    placement = placement or {}
    for pose, m in views:
        n_valid = int(m.valid.sum())
        d = dets.get(pose.name)
        rows.append({
            "foto": pose.name,
            "kanteling_graden": round(tilt_deg(pose), 1),
            "bovenaanzicht_gebruikt": pose.name in top_names,
            "object_pct_van_mat": round(100.0 * float(m.fg.sum()) / max(n_valid, 1), 2),
            "zekere_mat_pct": round(100.0 * float(m.bg.sum()) / max(n_valid, 1), 1),
            "ruis_grijswaarden": round(float(m.sigma), 2),
            "mathoeken": int(len(d.ids)) if d is not None else 0,
            "reprojectiefout_px": round(float(pose.rms_px), 3),
            "camerahoogte_mm": round(float(pose.center[2]), 0),
            "onscherpte_px": None if blur.get(pose.name) is None else round(float(blur[pose.name]), 2),
            "ligging": placement.get(pose.name),
        })
    return rows


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
