"""Telefoonfoto's (V10): HEIC lezen, camera, lens en zoom uit de EXIF, en foto's van een andere lens apart."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from camtocad import imgio, pipeline
from camtocad.imgio import PhotoInfo

DATA = Path(__file__).parent / "data"


def jpeg_with_exif(path: Path, model="Pixel 8", focal=6.9, zoom=None, lens=None) -> Path:
    exif = Image.Exif()
    exif[0x010F] = "Google"
    exif[0x0110] = model
    sub = exif.get_ifd(0x8769)
    sub[0x920A] = focal
    sub[0xA405] = 25
    if zoom is not None:
        sub[0xA404] = zoom
    if lens is not None:
        sub[0xA434] = lens
    img = np.tile(np.linspace(30, 220, 80, dtype=np.uint8), (60, 1))
    Image.fromarray(img).convert("RGB").save(path, exif=exif, quality=92)
    return path


def test_heic_is_read_with_its_camera_data():
    pytest.importorskip("pi_heif")
    path = DATA / "klein.heic"  # 96 x 64: verloop 40 -> 190 grijs, licht vlak 230; iPhone 14 Pro-EXIF
    gray = imgio.read_gray(path)
    assert gray.shape == (64, 96) and abs(int(gray[5, 2]) - 43) < 6 and abs(int(gray[30, 40]) - 230) < 6
    info = imgio.read_info(path)
    assert (info.make, info.model, info.focal_mm, info.focal35_mm) == ("Apple", "iPhone 14 Pro", 6.86, 24.0)
    assert info.label() == "Apple iPhone 14 Pro, 6,9 mm (24 mm-equivalent)"


def test_heic_is_read_in_sensor_layout_like_jpeg():
    """libheif zet een HEIC-foto bij het lezen rechtop; de pijplijn wil het sensorformaat, zoals bij JPEG (zonder
    EXIF-rotatie): staand opgeslagen foto's draait ze zelf terug, en ondersteboven geen andere camera-as."""
    pytest.importorskip("pi_heif")
    path = DATA / "gedraaid.heic"  # 90 x 60 met links een lichte strook, opgeslagen met EXIF-oriëntatie 6
    for img in (imgio.read_gray(path), imgio.read_color(path)):
        assert img.shape[:2] == (60, 90) and img[:, :10].mean() > 100 and img[:, 40:].mean() < 60


def test_heic_without_decoder_gives_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.setattr(imgio, "_heif_module", lambda: None)
    for k in range(3):
        (tmp_path / f"IMG_{k}.HEIC").write_bytes((DATA / "klein.heic").read_bytes())
    assert imgio.read_gray(tmp_path / "IMG_0.HEIC") is None and not imgio.heif_supported()
    with pytest.raises(pipeline.ScanError, match="camtocad\\[heic\\]"):
        pipeline.load_images(tmp_path, log=lambda m: None)


def test_jpeg_exif_gives_camera_lens_and_zoom(tmp_path):
    info = imgio.read_info(jpeg_with_exif(tmp_path / "a.jpg", zoom=2.0, lens="Pixel 8 back camera"))
    assert info.model == "Pixel 8" and info.focal_mm == pytest.approx(6.9) and info.zoom == 2.0
    assert info.label() == "Google Pixel 8, 6,9 mm (25 mm-equivalent), zoom 2x"
    assert info.key() == ("Google", "Pixel 8", "Pixel 8 back camera", 6.9, 2.0)
    # zonder EXIF (een render of bewerkte foto): geen sleutel
    Image.fromarray(np.zeros((10, 10), np.uint8)).save(tmp_path / "b.png")
    assert imgio.read_info(tmp_path / "b.png").key() is None


def test_photos_from_another_lens_are_set_aside(tmp_path):
    """Een iPhone schakelt dichtbij naar de macrolens (ultragroothoek), met hetzelfde beeldformaat: die foto's
    en een foto met digitale zoom horen niet in het cameramodel van de rest; een foto zonder EXIF wel."""
    main = PhotoInfo("Apple", "iPhone 14 Pro", "back camera 6.86mm", 6.86, 24.0, 1.0)
    macro = PhotoInfo("Apple", "iPhone 14 Pro", "back camera 2.22mm", 2.22, 13.0, 1.0)
    zoomed = PhotoInfo("Apple", "iPhone 14 Pro", "back camera 6.86mm", 6.86, 48.0, 2.0)
    infos = {f"f{k}": main for k in range(8)} | {"m1": macro, "m2": macro, "z1": zoomed, "render": PhotoInfo()}
    key, other = pipeline.camera_groups(list(infos), infos)
    assert key == main.key() and set(other) == {"m1", "m2", "z1"}
    assert "2,2 mm" in other["m1"] and "zoom 2x" in other["z1"] and "6,9 mm" in other["m1"]
    # alleen foto's zonder cameragegevens: niets apart
    assert pipeline.camera_groups(["a", "b"], {"a": PhotoInfo(), "b": PhotoInfo()}) == (None, {})


def test_load_images_collects_camera_data(tmp_path):
    for k in range(3):
        jpeg_with_exif(tmp_path / f"p{k}.jpg")
    jpeg_with_exif(tmp_path / "p9.jpg", model="Pixel 8", focal=2.0)  # ultragroothoek
    infos = {}
    images = pipeline.load_images(tmp_path, log=lambda m: None, infos=infos)
    assert [n for n, _ in images] == ["p0.jpg", "p1.jpg", "p2.jpg", "p9.jpg"] and len(infos) == 4
    _, other = pipeline.camera_groups([n for n, _ in images], infos)
    assert list(other) == ["p9.jpg"]


def test_load_images_keeps_the_color_apart(tmp_path):
    """Kleur (V8): de pijplijn rekent in grijs en krijgt de kleur apart, op halve resolutie; een foto zonder kleur
    (hier een grijs verloop als JPEG, of de HEIC-testfoto) krijgt geen chroma."""
    jpeg_with_exif(tmp_path / "a.jpg")
    img = np.full((60, 80, 3), 120, np.uint8)
    img[20:40, 30:60] = (160, 90, 30)  # BGR: blauw vlak
    Image.fromarray(img[:, :, ::-1]).save(tmp_path / "b.jpg", quality=92)
    chroma = {}
    images = pipeline.load_images(tmp_path, log=lambda m: None, chroma=chroma)
    assert [n for n, _ in images] == ["a.jpg", "b.jpg"] and all(g.ndim == 2 for _, g in images)
    assert list(chroma) == ["b.jpg"] and chroma["b.jpg"].shape == (30, 40, 2)
    rg, yb = chroma["b.jpg"][15, 22].astype(float)  # midden in het blauwe vlak: R - G < 0, (R + G)/2 - B < 0
    assert rg < -40 and yb < -60 and np.abs(chroma["b.jpg"][2, 2].astype(float)).max() < 3
    if imgio.heif_supported():
        assert imgio.split_chroma(imgio.read_color(DATA / "klein.heic"))[1] is None
