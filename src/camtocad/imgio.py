"""Foto's lezen (ook HEIC, en de EXIF-gegevens van de camera) en beelden schrijven.

`cv2.imread`/`cv2.imwrite` gaan op Windows mis met paden als `C:\\Users\\Jörg\\scans`. Via een
bytebuffer (`imdecode`/`imencode` met numpy) werkt het overal.

HEIC/HEIF (het standaardformaat van iPhones) leest OpenCV niet. Met de optionele extra `heic`
(`pip install "camtocad[heic]"`, pi-heif: libheif en libde265, LGPL-3.0, dynamisch gelinkt) kan dat wel.

Uit de EXIF-gegevens komt welke camera, lens en zoom een foto maakte (V10). Telefoons wisselen soms
ongemerkt van lens (een iPhone schakelt dichtbij naar de macrostand van de ultragroothoek) of zoomen
digitaal bij, met hetzelfde beeldformaat; zulke foto's passen niet in één cameramodel.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

HEIF_EXT = {".heic", ".heif"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"} | HEIF_EXT

# EXIF-tags (basis-IFD en Exif-IFD)
_MAKE, _MODEL, _EXIF_IFD = 0x010F, 0x0110, 0x8769
_FOCAL, _ZOOM, _FOCAL35, _LENS_MAKE, _LENS_MODEL = 0x920A, 0xA404, 0xA405, 0xA433, 0xA434


@dataclass
class PhotoInfo:
    """Camera, lens en zoom volgens de EXIF-gegevens; None waar een gegeven ontbreekt."""

    make: str | None = None
    model: str | None = None
    lens: str | None = None
    focal_mm: float | None = None
    focal35_mm: float | None = None
    zoom: float | None = None  # digitale zoom (1 = geen)

    def key(self) -> tuple | None:
        """Wat voor één cameramodel gelijk moet zijn: toestel, lens, brandpuntsafstand en zoom. None zonder
        cameragegevens (bijv. een PNG of een bewerkte foto)."""
        if self.model is None and self.lens is None and self.focal_mm is None:
            return None
        focal = None if self.focal_mm is None else round(self.focal_mm, 2)
        zoom = round(self.zoom, 2) if self.zoom and self.zoom > 0 else 1.0
        return (self.make, self.model, self.lens, focal, zoom)

    def label(self) -> str:
        """Leesbaar, bijv. 'Apple iPhone 14 Pro, 6,9 mm (24 mm-equivalent)' of '..., zoom 2x'."""
        parts = [" ".join(p for p in (self.make, self.model) if p) or "onbekende camera"]
        if self.focal_mm:
            f = f"{self.focal_mm:.1f} mm".replace(".", ",")
            if self.focal35_mm:
                f += f" ({self.focal35_mm:.0f} mm-equivalent)"
            parts.append(f)
        elif self.lens:
            parts.append(self.lens)
        if self.zoom and self.zoom > 1.001:
            parts.append(f"zoom {self.zoom:g}x".replace(".", ","))
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def _text(v) -> str | None:
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    v = str(v).strip().strip("\x00").strip() if v is not None else ""
    return v or None


def heif_supported() -> bool:
    """Is er een HEIC-decoder (pi-heif of pillow-heif)?"""
    return _heif_module() is not None


def _heif_module():
    for name in ("pi_heif", "pillow_heif"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        module.register_heif_opener()
        return module
    return None


def read_info(path: str | Path) -> PhotoInfo:
    """Camera, lens en zoom uit de EXIF-gegevens (alleen de kop van het bestand wordt gelezen)."""
    try:
        from PIL import Image
    except ImportError:
        return PhotoInfo()
    if Path(path).suffix.lower() in HEIF_EXT and _heif_module() is None:
        return PhotoInfo()
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            sub = exif.get_ifd(_EXIF_IFD)
    except Exception:  # noqa: BLE001 - een foto zonder (leesbare) EXIF is geen fout
        return PhotoInfo()
    return PhotoInfo(make=_text(exif.get(_MAKE)), model=_text(exif.get(_MODEL)),
                     lens=_text(sub.get(_LENS_MODEL)) or _text(sub.get(_LENS_MAKE)),
                     focal_mm=_num(sub.get(_FOCAL)), focal35_mm=_num(sub.get(_FOCAL35)), zoom=_num(sub.get(_ZOOM)))


def read_gray(path: str | Path) -> np.ndarray | None:
    """Leest een foto als grijswaarden, zonder EXIF-rotatie (sensorformaat, zoals de pipeline). HEIC via
    pi-heif, als dat geïnstalleerd is; None als de foto niet te lezen is."""
    path = Path(path)
    if path.suffix.lower() in HEIF_EXT:
        if _heif_module() is None:
            return None
        try:
            from PIL import Image
            with Image.open(path) as img:
                return np.asarray(img.convert("L"))
        except Exception:  # noqa: BLE001
            return None
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
