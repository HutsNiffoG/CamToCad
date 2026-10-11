"""Live begeleiding bij het fotograferen (V25: v0.13, uitgebreid in v0.16): waar staat de camera, waar ligt het
onderdeel, welke richting ontbreekt nog, en is dit een goed moment voor een foto?"""

import cv2
import numpy as np
import pytest

from camtocad import live, mat, render

# ----------------------------------------------------------------------------- live begeleiding (V25, v0.13)

LIVE_F = 700.0  # brandpuntsafstand (px) van de gerenderde livebeelden: 960 x 540, zoals een verkleind videobeeld


@pytest.fixture(scope="module")
def live_frames():
    """Livebeelden van een doosje (50 x 30 x 10 mm) midden op de mat, vanuit een positie (azimut, elevatie) rond het
    midden. `scene`: de voorwerpen als (dx, dy, lengte, breedte, hoogte) vanaf het midden van de mat; `f`: de
    brandpuntsafstand (een andere lens); `center`: de camerapositie zelf (mat-mm); `pose`: ook R en de positie terug."""
    import cadquery as cq

    spec = mat.PRESETS["A4"]
    raster = mat.rasterize_board(spec, 4.0, 3.0)
    target = np.array([spec.board_w_mm / 2, spec.board_h_mm / 2, 0.0])
    meshes = {}

    def mesh_of(scene):
        if scene not in meshes:
            shape = None
            for dx, dy, lx, ly, lz in scene:
                b = cq.Workplane("XY").box(lx, ly, lz, centered=(True, True, False)).translate((dx, dy, 0))
                shape = b if shape is None else shape.union(b)
            meshes[scene] = render.tessellate(render.place(shape, spec))
        return meshes[scene]

    def frame(az, el, d=320.0, blur=0.5, aim=(0.0, 0.0, 0.0), albedo=0.55, scene=((0, 0, 50, 30, 10),), f=LIVE_F,
              center=None, pose=False):
        a, e = np.radians(az), np.radians(el)
        c = (target + d * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]) if center is None
             else np.asarray(center, float))
        R, t = render.look_at(c, target + np.asarray(aim))
        cam = render.default_camera(960, 540, f, dist=(0, 0, 0, 0, 0))
        img = render.render_view(raster, cam, R, t, mesh_of(tuple(scene)), albedo=albedo, noise=1.0, blur=blur,
                                 rng=np.random.default_rng(0))[0]
        return (img, R, c) if pose else img

    return spec, frame


def test_live_guidance_starts_with_the_views_from_straight_above(live_frames):
    spec, frame = live_frames
    f_rel = LIVE_F / 960
    r = live.live_check(frame(270, 60), spec, None, f_rel)
    assert r["mat"] == "A4" and r["status"] == "verplaats" and not r["opnemen"] and r["doelvak"] == "boven"
    assert "recht boven het onderdeel" in r["aanwijzing"] and r["icoon"] == "boven"
    assert r["positie"]["azimut"] == pytest.approx(270, abs=2) and r["positie"]["elevatie"] == pytest.approx(60, abs=2)
    assert r["positie"]["afstand_mm"] == pytest.approx(320, rel=0.04)
    assert r["f_rel"] == pytest.approx(f_rel, rel=0.03)  # uit de homografie, voor de volgende beelden
    assert np.allclose(r["doel"], [0.5, 0.5], atol=0.02) and len(r["omtrek"]) == 4  # het onderdeel midden in beeld
    top = live.live_check(frame(270, 89), spec, None, f_rel)
    assert top["vak"] == "boven" and top["opnemen"] and top["status"] == "opnemen" and top["icoon"] == "opnemen"
    assert top["nodig"]["boven"] == live.LIVE_TOP and not top["gedekt"]


def test_live_guidance_walks_around_the_part(live_frames):
    """Na de bovenaanzichten: twee foto's per richting en hoogte, en de weg naar het dichtstbijzijnde vak dat nog
    foto's mist. Naar rechts lopen (gezien vanaf de telefoon) is rechtsom: de azimut gaat omhoog."""
    spec, frame = live_frames
    f_rel = LIVE_F / 960
    img = frame(0, 60)
    ov = {"recht_van_boven": 5}
    r = live.live_check(img, spec, ov, f_rel)
    assert r["vak"] == "hoog-0" and r["opnemen"] and "ontbreekt nog" in r["aanwijzing"]
    near = {**ov, "punten": [{"naam": "foto_0005.jpg", "azimut": 2.0, "elevatie": 58.0, "oordeel": "goed"}]}
    r = live.live_check(img, spec, near, f_rel)
    assert not r["opnemen"] and r["status"] == "verplaats" and "al een foto" in r["aanwijzing"]
    r = live.live_check(img, spec, {**ov, "dekking": {"hoog": [2, 0, 0, 0, 0, 0, 0, 0], "laag": [0] * 8}}, f_rel)
    assert r["doelvak"] == "hoog-1" and "~45° naar rechts" in r["aanwijzing"] and not r["opnemen"]  # gelijk: rechtsom
    assert r["spraak"] == "Naar rechts." and r["icoon"] == "rechts"
    r = live.live_check(img, spec, {**ov, "dekking": {"hoog": [2, 2] + [0] * 6, "laag": [2] * 8}}, f_rel)
    assert r["doelvak"] == "hoog-7" and "~45° naar links" in r["aanwijzing"]
    r = live.live_check(img, spec, {**ov, "dekking": {"hoog": [2] * 8, "laag": [0] * 8}}, f_rel)
    assert r["doelvak"] == "laag-0" and "lager, ~35°" in r["aanwijzing"] and "loop" not in r["aanwijzing"]
    assert r["spraak"] == "Lager." and r["icoon"] == "lager" and r["doel_elevatie"] == 35
    r = live.live_check(img, spec, {**ov, "dekking": {"hoog": [2] * 8, "laag": [2] * 8}}, f_rel)
    assert r["status"] == "klaar" and r["gedekt"] and not r["opnemen"] and r["icoon"] == "klaar"


def test_live_guidance_checks_the_frame_before_a_photo(live_frames):
    spec, frame = live_frames
    f_rel = LIVE_F / 960
    ov = {"recht_van_boven": 5}
    r = live.live_check(frame(0, 60, blur=2.2), spec, ov, f_rel)
    assert r["scherpte_px"] > live.LIVE_BLUR_MAX and not r["opnemen"] and "stil" in r["aanwijzing"]
    assert r["icoon"] == "stil"
    r = live.live_check(frame(0, 60, d=130), spec, ov, f_rel)
    assert not r["opnemen"] and r["aanwijzing"].startswith("Te dichtbij") and r["icoon"] == "weg"
    r = live.live_check(frame(0, 60, aim=(90.0, 0.0, 0.0)), spec, ov, f_rel)
    assert not r["opnemen"] and "op het onderdeel" in r["aanwijzing"] and r["icoon"] == "richt"
    r = live.live_check(frame(0, 60) // 6, spec, ov, f_rel)
    assert not r["opnemen"] and "donker" in r["aanwijzing"] and r["icoon"] == "donker"
    r = live.live_check(np.full((540, 960), 120, np.uint8), spec, ov)
    assert r["status"] == "zoek" and r["mat"] is None and "kalibratiemat" in r["aanwijzing"] and r["icoon"] == "mat"


def test_live_guidance_from_above_names_the_side_of_the_mat(live_frames):
    """Recht van boven zegt de azimut weinig: de volgende richting heet naar de kant van de mat, te beginnen onder
    (de kant van meetlijn X)."""
    spec, frame = live_frames
    r = live.live_check(frame(30, 88), spec, {"recht_van_boven": 5}, LIVE_F / 960)
    assert r["vak"] == "boven" and r["doelvak"] == "hoog-6" and not r["opnemen"]
    assert "aan de kant onder van de mat" in r["aanwijzing"] and "loop" not in r["aanwijzing"]
    assert r["spraak"] == "Lager, aan de kant onder." and r["icoon"] == "lager"
    full = {"recht_van_boven": 5, "dekking": {"hoog": [0] * 6 + [2, 0], "laag": [0] * 8}}
    r = live.live_check(frame(30, 88), spec, full, LIVE_F / 960)
    assert r["doelvak"] == "hoog-7" and "rechtsonder" in r["aanwijzing"]  # rechtsom verder


# ----------------------------------------------------------------------------- v0.16

def test_the_part_is_found_in_the_live_frames(live_frames):
    """Waar het onderdeel ligt, uit de livebeelden zelf: waar het patroon van de mat ontbreekt. Grijs, zwart en wit,
    na drie beelden van boven; schuine beelden (die ook de mat achter het onderdeel bedekken) schuiven het niet op.
    Een wit onderdeel is op de witte vakken niet te zien: zijn midden ligt een paar mm minder precies."""
    spec, frame = live_frames
    mid = (spec.board_w_mm / 2, spec.board_h_mm / 2)
    for albedo in (0.55, 0.12, 0.92):
        s = live.LiveSession()
        for az in (0, 120, 240):
            r = live.live_check(frame(az, 86, albedo=albedo), spec, None, LIVE_F / 960, session=s)
        part = r["onderdeel"]
        assert part and part["bron"] == "live" and not part["andere"], (albedo, part)
        assert np.hypot(part["midden_mm"][0] - mid[0], part["midden_mm"][1] - mid[1]) < 5, (albedo, part)
        assert part["straal_mm"] == pytest.approx(np.hypot(25, 15), abs=4), (albedo, part)
        assert part["rand_mm"] == pytest.approx(mid[1] - 15, abs=4)
        for az, el in ((30, 60), (200, 35)):
            r = live.live_check(frame(az, el, albedo=albedo), spec, None, LIVE_F / 960, session=s)
        part = r["onderdeel"]
        assert np.hypot(part["midden_mm"][0] - mid[0], part["midden_mm"][1] - mid[1]) < 5, (albedo, part)
        outline = np.array(part["omtrek"])  # de omtrek over het beeld, rond het richtpunt
        assert len(outline) >= 4 and np.allclose(outline.mean(axis=0), r["doel"], atol=0.05)


def test_a_second_object_and_a_part_at_the_edge_are_reported(live_frames):
    spec, frame = live_frames
    s = live.LiveSession()
    scene = ((0, 0, 50, 30, 10), (75, 45, 20, 20, 2))  # en een klein plat ding rechtsboven
    for az in (0, 120, 240):
        r = live.live_check(frame(az, 86, scene=scene), spec, None, LIVE_F / 960, session=s)
    assert len(r["onderdeel"]["andere"]) == 1 and "ander voorwerp op de mat: haal het weg" in r["opmerkingen"]
    x, y = r["onderdeel"]["andere"][0]
    assert abs(x - (spec.board_w_mm / 2 + 75)) < 4 and abs(y - (spec.board_h_mm / 2 + 45)) < 4
    # in beeld een cirkel om het voorwerp heen (v0.16.2); de omtrek van de mat geeft waar het voorwerp in beeld ligt
    bw, bh = spec.board_w_mm, spec.board_h_mm
    Hn = cv2.getPerspectiveTransform(np.float32([[0, bh], [bw, bh], [bw, 0], [0, 0]]), np.float32(r["omtrek"]))
    u, v = cv2.perspectiveTransform(np.float32([[[x, y]]]), Hn)[0, 0]
    circle = np.float32(r["onderdeel"]["andere_beeld"][0])
    assert len(r["onderdeel"]["andere_beeld"]) == 1 and len(circle) == 25
    assert cv2.pointPolygonTest(circle, (float(u), float(v)), False) > 0
    assert cv2.pointPolygonTest(circle, tuple(map(float, r["doel"])), False) < 0  # niet om het onderdeel zelf
    assert all(len(n) <= 60 for n in r["opmerkingen"])  # kort: één regel in beeld
    s = live.LiveSession()
    for az in (0, 120, 240):
        r = live.live_check(frame(az, 86, scene=((-85, 0, 50, 30, 10),), aim=(-85, 0, 0)), spec, None, LIVE_F / 960,
                            session=s)
    assert r["onderdeel"]["rand_mm"] == pytest.approx(spec.board_w_mm / 2 - 110, abs=4)
    assert "onderdeel dicht bij de rand: leg het meer naar het midden" in r["opmerkingen"]


def test_the_ring_and_the_arrow_point_the_way(live_frames):
    """Een ring op de mat rond het onderdeel (binnen ~60°, buiten ~35°, per vak 45°) en een pijl langs de ring naar
    het volgende vak. Het vak aan de kant van de camera ligt onderaan in beeld, en rechtsom lopen (naar rechts) is
    in het beeld naar rechts."""
    spec, frame = live_frames
    ov = {"recht_van_boven": 5, "dekking": {"hoog": [2, 0, 0, 0, 0, 0, 0, 0], "laag": [0] * 8}}
    r = live.live_check(frame(0, 60), spec, ov, LIVE_F / 960)
    assert r["doelvak"] == "hoog-1" and r["doel_elevatie"] == 60
    ring = r["ring"]
    assert len(ring["hoog"]) == 8 and len(ring["laag"]) == 8
    assert ring["straal_mm"]["laag"] > ring["straal_mm"]["hoog"] >= live.RING_MIN_MM
    near, far = np.array(ring["hoog"][0]), np.array(ring["hoog"][4])
    assert near[:, 1].mean() > r["doel"][1] > far[:, 1].mean()
    arrow = np.array(r["pijl"])
    assert np.allclose(arrow[0], near[3], atol=0.02)  # vanaf de camera (azimut 0: het midden van vak 0)
    assert ring["band"] == "hoog" and np.allclose(ring["hier"], arrow[0], atol=0.005)
    assert arrow[-1, 0] > arrow[0, 0] + 0.05  # naar rechts


def test_photos_from_above_show_where_to_shift_the_phone(live_frames):
    """Voor de foto's recht van boven: de richting in het beeld waarin de telefoon moet schuiven. Wie die volgt, komt
    dichter boven het onderdeel."""
    spec, frame = live_frames
    img, R, c = frame(30, 70, pose=True)
    r = live.live_check(img, spec, None, LIVE_F / 960)
    assert r["doelvak"] == "boven" and r["boven"]["hoek"] == pytest.approx(20, abs=2) and r["doel_elevatie"] == 90
    move = R.T @ np.array([*r["boven"]["richting"], 0.0])  # van beeld naar mat
    move[2] = 0.0
    r2 = live.live_check(frame(30, 70, center=c + 40.0 * move / np.linalg.norm(move)), spec, None, LIVE_F / 960)
    assert r2["boven"]["hoek"] < r["boven"]["hoek"] - 5


def test_a_moving_phone_or_another_lens_gets_no_photo(live_frames):
    """Hoe snel de mat door het beeld beweegt (de homografie van het vorige beeld tegen die van nu) en of de
    beeldhoek plots anders is (een andere lens of zoom): dan geen foto."""
    spec, frame = live_frames
    ov = {"recht_van_boven": 5}
    s = live.LiveSession()
    r = live.live_check(frame(0, 60), spec, ov, session=s, t=0.0)
    assert r["opnemen"] and r["stil"] is None  # het eerste beeld: nog geen snelheid
    r = live.live_check(frame(4, 60), spec, ov, session=s, t=0.1)  # 4° verder in 0,1 s: ~100 px/s
    assert r["snelheid_px_s"] > live.STILL_PX_S and r["stil"] is False and not r["opnemen"]
    assert "beweegt" in r["aanwijzing"] and not any("rustiger" in n for n in r["opmerkingen"])
    r = live.live_check(frame(24, 60), spec, ov, session=s, t=0.2)  # 20° in 0,1 s: te snel
    assert r["snelheid_px_s"] > live.FAST_PX_S and any("rustiger" in n for n in r["opmerkingen"])
    r = live.live_check(frame(24, 60), spec, ov, session=s, t=0.3)
    assert r["stil"] and r["opnemen"]
    for az in (30, 60, 90, 120, 150):  # schuine beelden: de brandpuntsafstand van deze lens
        live.live_check(frame(az, 45), spec, ov, session=s)
    r = live.live_check(frame(0, 45, f=470.0), spec, ov, session=s)  # een groothoek: een derde kortere f
    assert not r["opnemen"] and "lens" in r["aanwijzing"] and r["spraak"].startswith("Andere lens")


def test_a_small_mat_asks_to_come_closer_or_to_switch_to_the_normal_lens(live_frames):
    """v0.16.3: met de groothoeklens (0,6×) is de mat op 30 cm klein in beeld (~2,5 px per mm in de foto, de
    verwerking rekent op 3,5): eerst de gewone lens of dichterbij. Kiest de pagina een andere camera, dan begint de
    brandpuntsafstand opnieuw (geen melding 'andere lens')."""
    spec, frame = live_frames
    ov = {"recht_van_boven": 5}
    s = live.LiveSession()
    for az in (0, 40, 80, 120, 160):  # schuine beelden: de brandpuntsafstand van deze lens
        r = live.live_check(frame(az, 45, d=300, f=365.0), spec, ov, session=s, lens="achter-2")
    assert r["f_rel"] == pytest.approx(365 / 960, rel=0.05)
    assert not r["opnemen"] and r["icoon"] == "lens" and r["spraak"] == "Groothoeklens: kies 1×."
    assert r["px_per_mm"] < live.LOW_PX_MM
    r = live.live_check(frame(0, 60, d=200, f=365.0), spec, ov, session=s, lens="achter-2")  # dichterbij: wel goed
    assert r["px_per_mm"] >= live.LOW_PX_MM and r["icoon"] != "lens" and r["opnemen"]
    r = live.live_check(frame(0, 60), spec, ov, session=s, lens="achter-0")  # de gewone lens gekozen
    assert r["opnemen"] and not r["aanwijzing"].startswith("De telefoon wisselde van lens")
    # zonder brandpuntsafstand uit de beelden, maar met foto's van de groothoeklens (EXIF): ook verder dan ~55 cm
    r = live.live_check(frame(0, 60, d=560), spec, {**ov, "groothoek": True}, LIVE_F / 960)
    assert r["icoon"] == "lens" and not r["opnemen"]
    r = live.live_check(frame(0, 60, d=560), spec, ov, LIVE_F / 960)
    assert r["icoon"] == "dichterbij" and r["spraak"] == "Dichterbij."


def test_steps_and_progress():
    need = live.missing_views({"recht_van_boven": 2})
    step, progress = live._progress(need)
    assert step == {"nummer": 1, "naam": "recht van boven", "gedaan": 2, "totaal": 5}
    assert progress == {"gedaan": 2, "totaal": live.LIVE_TOTAL}
    need = live.missing_views({"recht_van_boven": 7, "dekking": {"hoog": [2] * 8, "laag": [1] + [0] * 7}})
    step, progress = live._progress(need)
    assert step["nummer"] == 2 and step["gedaan"] == 17 and progress["gedaan"] == 22
    # één hoogte tegelijk rondom: ook vanaf de andere kant eerst het laatste vak op ~60°
    assert live._next_cell(180.0, 40.0, live.missing_views(
        {"recht_van_boven": 5, "dekking": {"hoog": [2] * 7 + [0], "laag": [0] * 8}})) == "hoog-7"
