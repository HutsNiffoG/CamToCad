"""Fotocontrole vooraf: oordeel per foto, onscherpte, dekking en aanwijzingen."""

import cv2
import numpy as np
import pytest

from camtocad import mat, preflight, render
from camtocad.calib import Pose


@pytest.fixture(scope="module")
def box_scan():
    import cadquery as cq

    spec = mat.PRESETS["A4"]
    box = render.place(cq.Workplane("XY").box(50, 30, 10, centered=(True, True, False)), spec, angle_deg=10,
                       offset=(30, 15))
    views = render.render_scan(box, spec, render.default_camera(), rings=((35.0, 8), (60.0, 6)), top_views=4, seed=1)
    return spec, views


def test_good_photo_is_recognized_and_measured(box_scan):
    spec, views = box_scan
    top = next(v for v in views if v.name.startswith("top"))
    chk = preflight.check_image(top.name, top.image)
    assert chk.verdict == "goed" and chk.mat == "A4" and not chk.notes
    assert chk.corners >= 40 and 0.4 < chk.blur_px < 1.0  # render: 0,5 px vervaging plus bemonstering
    assert chk.tilt_deg < 8 and 180 < chk.white < 250 and chk.contrast > 150


def test_blur_is_measured_in_pixels(box_scan):
    spec, views = box_scan
    img = cv2.GaussianBlur(views[3].image, (0, 0), 2.0)
    chk = preflight.check_image("wazig", img, spec)
    assert chk.blur_px == pytest.approx(np.hypot(2.0, 0.65), abs=0.2)
    assert chk.verdict == "matig" and "onscherp" in chk.notes[0]


def test_unusable_photos_get_a_reason(box_scan):
    spec, views = box_scan
    blank = preflight.check_image("leeg", np.full((1200, 1600), 128, np.uint8), spec)
    assert blank.verdict == "onbruikbaar" and "niet gevonden" in blank.notes[0]
    dark = preflight.check_image("donker", (views[0].image * 0.15).astype(np.uint8), spec)
    assert dark.verdict in ("matig", "onbruikbaar") and any("donker" in n for n in dark.notes)


def test_old_mat_is_still_recognized():
    old = mat.PRESETS["A4-V1"]
    view = render.render_scan(None, old, render.default_camera(), rings=((45.0, 1),), top_views=0, seed=2)[0]
    chk = preflight.check_image(view.name, view.image)
    assert chk.mat == "A4-v1" and chk.verdict == "goed"


def test_summary_locates_the_object_and_asks_for_missing_photos(box_scan):
    spec, views = box_scan
    checks = [preflight.check_image(v.name, v.image, spec) for v in views]
    s = preflight.summarize(checks)
    assert s["bruikbaar"] == len(views) and s["recht_van_boven"] == 4
    assert np.allclose(s["object_mm"], [150.0, 95.0], atol=6.0)  # middelpunt van de doos
    assert s["klaar"] is True

    no_top = preflight.summarize([c for c in checks if not c.name.startswith("top")])
    assert no_top["klaar"] is False
    assert any("recht boven" in a for a in no_top["advies"])


def test_coverage_names_the_missing_directions():
    target = np.array([120.0, 80.0, 0.0])
    poses = {}
    for name, R, t in render.scan_poses(target, 330.0, rings=((35.0, 8),), top_views=0):
        center = -R.T @ t
        if center[1] - target[1] > 50:  # alles aan de kant van de titel ('boven') weglaten
            continue
        poses[name] = Pose(name, R, t)
    cov = preflight.coverage(poses, target)
    assert cov["recht_van_boven"] == 0 and cov["compleet"] is False
    laag = next(a for a in cov["advies"] if a.startswith("Laag"))
    assert "boven" in laag and "onder," not in laag
    assert any(a.startswith("Nog geen foto's hoog") for a in cov["advies"])


def test_photos_in_paths_with_non_ascii_characters(tmp_path):
    from camtocad.imgio import imwrite, read_gray

    folder = tmp_path / "Jörg ø"
    folder.mkdir()
    img = (np.arange(600).reshape(20, 30) % 256).astype(np.uint8)
    assert imwrite(folder / "foto.png", img)
    assert np.array_equal(read_gray(folder / "foto.png"), img)
    assert read_gray(folder / "bestaat-niet.jpg") is None


def test_summary_warns_about_photos_from_another_lens(box_scan):
    """V10: foto's met een andere lens of zoom (EXIF) worden apart gezet, met een aanwijzing."""
    spec, views = box_scan
    checks = [preflight.check_image(v.name, v.image, spec) for v in views]
    main = {"make": "Apple", "model": "iPhone 14 Pro", "focal_mm": 6.86, "lens": "back camera 6.86mm"}
    macro = {"make": "Apple", "model": "iPhone 14 Pro", "focal_mm": 2.22, "lens": "back camera 2.22mm"}
    for k, c in enumerate(checks):
        c.camera = dict(macro if k in (1, 2) else main)
    s = preflight.summarize(checks)
    advice = [a for a in s["advies"] if "andere camera, lens of zoom" in a]
    assert len(advice) == 1 and advice[0].startswith("2 foto's zijn") and "macrolens" in advice[0]
    assert s["bruikbaar"] == len(views) - 2  # die twee tellen niet mee in de dekking


def test_a_small_mat_in_the_picture_is_reported(box_scan, tmp_path):
    """v0.16.3: hoe groot de mat in beeld is (pixels per mm). Met de groothoeklens (0,6×), of van ver weg, is de mat
    klein, en dan ziet de verwerking de stippen van de mat en een donker onderdeel op de zwarte vakken niet."""
    from PIL import Image

    spec, views = box_scan
    top = next(v for v in views if v.name.startswith("top"))
    chk = preflight.check_image(top.name, top.image, spec)
    assert chk.px_per_mm == pytest.approx(1300 / 330, rel=0.08) and chk.verdict == "goed"  # brandpunt / afstand
    half = [(v.name, cv2.resize(v.image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)) for v in views]
    small = preflight.check_image("klein", dict(half)[top.name], spec)  # half zo groot in beeld: ~2 px per mm
    assert small.px_per_mm == pytest.approx(0.5 * 1300 / 330, rel=0.1) and small.verdict == "matig"
    assert any(n.startswith("mat klein in beeld") and n.endswith("kom dichterbij") for n in small.notes)
    # met de groothoeklens (EXIF: 13 mm-equivalent) de gewone lens aanraden
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = "samsung", "SM-S921B"
    sub = exif.get_ifd(0x8769)
    sub[0x920A], sub[0xA405] = 2.2, 13
    path = tmp_path / "groothoek.jpg"
    Image.fromarray(dict(half)[top.name]).convert("RGB").save(path, exif=exif, quality=95)
    wide = preflight.check_file(path, spec)
    assert any("groothoeklens (13 mm-equivalent), gebruik de gewone lens (1×)" in n for n in wide.notes), wide.notes
    checks = [preflight.check_image(n, img, spec) for n, img in half]
    for c in checks:
        c.camera = {"make": "samsung", "model": "SM-S921B", "focal_mm": 2.2, "focal35_mm": 13.0}
    s = preflight.summarize(checks)
    assert s["groothoek"] and s["px_per_mm"] < preflight.LOW_PX_MM
    advice = [a for a in s["advies"] if a.startswith("De mat is klein in beeld")]
    assert len(advice) == 1 and "groothoeklens" in advice[0] and "gewone lens (1×)" in advice[0] and not s["klaar"]
    assert not preflight.summarize([preflight.check_image(v.name, v.image, spec) for v in views])["groothoek"]


def test_summary_warns_about_photos_taken_too_close(box_scan):
    """V28 (v0.12): een foto van dichterbij dan 15 cm boven de mat krijgt een aanwijzing (scherptediepte, macrolens)."""
    import cadquery as cq

    spec, views = box_scan
    checks = [preflight.check_image(v.name, v.image, spec) for v in views]
    assert not any("dichterbij" in a for a in preflight.summarize(checks)["advies"])
    box = render.place(cq.Workplane("XY").box(50, 30, 10, centered=(True, True, False)), spec, angle_deg=10,
                       offset=(30, 15))
    R, t = render.look_at([110.0, 60.0, 120.0], [110.0, 62.0, 0.0])
    img, _ = render.render_view(render.rasterize_board(spec, 10.0, 3.0), render.default_camera(), R, t,
                                render.tessellate(box), rng=np.random.default_rng(0))
    close = preflight.check_image("dichtbij", img, spec)
    assert close.mat and close.corners >= 12
    advice = [a for a in preflight.summarize(checks + [close])["advies"] if "dichterbij" in a]
    assert len(advice) == 1 and advice[0].startswith("1 foto is")


def test_close_is_measured_along_the_view_not_as_height():
    """v0.13: 'te dichtbij' is de afstand langs de kijkrichting (waarop de telefoon scherpstelt), niet de hoogte boven
    de mat: een lage foto op 30 cm hangt maar 13 cm boven de mat."""
    R, t = render.look_at([0.0, 0.0, 127.0], [272.0, 0.0, 0.0])  # 25° boven de mat, 30 cm van het midden
    assert preflight.view_distance(Pose("laag", R, t)) == pytest.approx(300.0, abs=1)
    R, t = render.look_at([110.0, 60.0, 120.0], [110.0, 62.0, 0.0])
    assert preflight.view_distance(Pose("dichtbij", R, t)) == pytest.approx(120.0, abs=1)
