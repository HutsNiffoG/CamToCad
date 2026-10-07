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


# ----------------------------------------------------------------------------- live begeleiding (V25, v0.13)

LIVE_F = 700.0  # brandpuntsafstand (px) van de gerenderde livebeelden: 960 x 540, zoals een verkleind videobeeld


@pytest.fixture(scope="module")
def live_frames():
    """Livebeelden van een doosje midden op de mat, vanuit een positie (azimut, elevatie) rond het doosje."""
    import cadquery as cq

    spec = mat.PRESETS["A4"]
    mesh = render.tessellate(render.place(cq.Workplane("XY").box(50, 30, 10, centered=(True, True, False)), spec))
    raster = mat.rasterize_board(spec, 4.0, 3.0)
    cam = render.default_camera(960, 540, LIVE_F, dist=(0, 0, 0, 0, 0))
    target = np.array([spec.board_w_mm / 2, spec.board_h_mm / 2, 0.0])

    def frame(az, el, d=320.0, blur=0.5, aim=(0.0, 0.0, 0.0)):
        a, e = np.radians(az), np.radians(el)
        c = target + d * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        R, t = render.look_at(c, target + np.asarray(aim))
        return render.render_view(raster, cam, R, t, mesh, noise=1.0, blur=blur, rng=np.random.default_rng(0))[0]

    return spec, frame


def test_live_guidance_starts_with_the_views_from_straight_above(live_frames):
    spec, frame = live_frames
    f_rel = LIVE_F / 960
    r = preflight.live_check(frame(270, 60), spec, None, f_rel)
    assert r["mat"] == "A4" and r["status"] == "verplaats" and not r["opnemen"] and r["doelvak"] == "boven"
    assert "recht boven het onderdeel" in r["aanwijzing"]
    assert r["positie"]["azimut"] == pytest.approx(270, abs=2) and r["positie"]["elevatie"] == pytest.approx(60, abs=2)
    assert r["positie"]["afstand_mm"] == pytest.approx(320, rel=0.04)
    assert r["f_rel"] == pytest.approx(f_rel, rel=0.03)  # uit de homografie, voor de volgende beelden
    assert np.allclose(r["doel"], [0.5, 0.5], atol=0.02) and len(r["omtrek"]) == 4  # het onderdeel midden in beeld
    top = preflight.live_check(frame(270, 89), spec, None, f_rel)
    assert top["vak"] == "boven" and top["opnemen"] and top["status"] == "opnemen"
    assert top["nodig"]["boven"] == preflight.LIVE_TOP and not top["gedekt"]


def test_live_guidance_walks_around_the_part(live_frames):
    """Na de bovenaanzichten: twee foto's per richting en hoogte, en de weg naar het dichtstbijzijnde vak dat nog
    foto's mist. Naar rechts lopen (gezien vanaf de telefoon) is rechtsom: de azimut gaat omhoog."""
    spec, frame = live_frames
    f_rel = LIVE_F / 960
    img = frame(0, 60)
    ov = {"recht_van_boven": 5}
    r = preflight.live_check(img, spec, ov, f_rel)
    assert r["vak"] == "hoog-0" and r["opnemen"] and "ontbreekt nog" in r["aanwijzing"]
    near = {**ov, "punten": [{"naam": "foto_0005.jpg", "azimut": 2.0, "elevatie": 58.0, "oordeel": "goed"}]}
    r = preflight.live_check(img, spec, near, f_rel)
    assert not r["opnemen"] and r["status"] == "verplaats" and "al een foto" in r["aanwijzing"]
    r = preflight.live_check(img, spec, {**ov, "dekking": {"hoog": [2, 0, 0, 0, 0, 0, 0, 0], "laag": [0] * 8}}, f_rel)
    assert r["doelvak"] == "hoog-1" and "~45° naar rechts" in r["aanwijzing"] and not r["opnemen"]  # gelijk: rechtsom
    r = preflight.live_check(img, spec, {**ov, "dekking": {"hoog": [2, 2] + [0] * 6, "laag": [2] * 8}}, f_rel)
    assert r["doelvak"] == "hoog-7" and "~45° naar links" in r["aanwijzing"]
    r = preflight.live_check(img, spec, {**ov, "dekking": {"hoog": [2] * 8, "laag": [0] * 8}}, f_rel)
    assert r["doelvak"] == "laag-0" and "lager, ~35°" in r["aanwijzing"] and "loop" not in r["aanwijzing"]
    r = preflight.live_check(img, spec, {**ov, "dekking": {"hoog": [2] * 8, "laag": [2] * 8}}, f_rel)
    assert r["status"] == "klaar" and r["gedekt"] and not r["opnemen"]


def test_live_guidance_checks_the_frame_before_a_photo(live_frames):
    spec, frame = live_frames
    f_rel = LIVE_F / 960
    ov = {"recht_van_boven": 5}
    r = preflight.live_check(frame(0, 60, blur=2.2), spec, ov, f_rel)
    assert r["scherpte_px"] > preflight.LIVE_BLUR_MAX and not r["opnemen"] and "stil" in r["aanwijzing"]
    r = preflight.live_check(frame(0, 60, d=130), spec, ov, f_rel)
    assert not r["opnemen"] and r["aanwijzing"].startswith("Te dichtbij")
    r = preflight.live_check(frame(0, 60, aim=(90.0, 0.0, 0.0)), spec, ov, f_rel)
    assert not r["opnemen"] and "op het onderdeel" in r["aanwijzing"]
    r = preflight.live_check(frame(0, 60) // 6, spec, ov, f_rel)
    assert not r["opnemen"] and "donker" in r["aanwijzing"]
    r = preflight.live_check(np.full((540, 960), 120, np.uint8), spec, ov)
    assert r["status"] == "zoek" and r["mat"] is None and "kalibratiemat" in r["aanwijzing"]
