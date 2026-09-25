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
