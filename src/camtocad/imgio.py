"""Foto's lezen en beelden schrijven, ook met niet-ASCII-tekens in het pad.

`cv2.imread`/`cv2.imwrite` gaan op Windows mis met paden als `C:\\Users\\Jörg\\scans`. Via een
bytebuffer (`imdecode`/`imencode` met numpy) werkt het overal.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_gray(path: str | Path) -> np.ndarray | None:
    """Leest een foto als grijswaarden, zonder EXIF-rotatie (sensorformaat, zoals de pipeline)."""
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)


def imwrite(path: str | Path, img: np.ndarray, params: list[int] | None = None) -> bool:
    """Schrijft een beeld; het formaat volgt uit de extensie (.png, .jpg, ...)."""
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img, params or [])
    if ok:
        buf.tofile(str(path))
    return bool(ok)
