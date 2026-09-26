"""Maskers voor onderdelen in de kleur van de mat: zwart op zwart, wit op wit (V8)."""

import numpy as np
import pytest
from scipy import ndimage

from camtocad import mat, masks, render
from camtocad.calib import Pose


def test_fill_bridges_invisible_strips_but_not_real_holes():
    """Opvullen binnen de sluiting van het object, behalve waar matbewijs aan grenst (een echt gat)."""
    fg = np.zeros((120, 160), bool)
    fg[20:100, 20:140] = True
    amb = np.zeros_like(fg)
    ev = ~fg.copy()  # rond het object: mat
    fg[40:56, 40:60], amb[40:56, 40:60] = False, True  # A: onzichtbare strook binnen het object
    fg[40:54, 90:104], amb[40:54, 90:104] = False, True  # B: echt gat, in het midden zichtbare mat
    amb[45:49, 95:99], ev[45:49, 95:99] = False, True
    fg[20:30, 70:80], amb[20:30, 70:80] = False, True  # C: inham aan de bovenrand
    ev[5:20, 60:90], amb[5:20, 60:90] = False, True  # D: onzichtbare strook buiten het object, tegen C aan
    o = np.where(fg, 20.0, 12.0)  # object iets lichter dan de mat eronder; onzichtbare stukken: gemiddeld
    o[amb] = 16.0
    domain = masks._closure(fg, 8)
    out, mat_like = masks._fill_ambiguous(fg, amb, ev, domain, o, np.full(o.shape, 12.0), np.full(o.shape, 20.0))
    assert out[40:56, 40:60].all()  # A
    assert not (out & amb)[40:54, 90:104].any()  # B blijft open
    # C: de mat net buiten de rechte rand zegt niets over de inham; alleen de boog van de sluiting over
    # de opening (straal 8 over 10 px: ~2 px) blijft open
    assert out[22:30, 70:80].all()
    assert not out[5:20, 60:90].any()  # D ligt buiten de sluiting
    assert domain[22:100, 20:140].all() and not domain[5:18, 60:90].any() and not mat_like.any()
    # een stuk dat als geheel duidelijk op de mat lijkt (een schaduw: 16 grijswaarden donkerder dan het
    # object, per pixel binnen de ruis), wordt niet opgevuld; zwart op zwart (8 verschil) wel
    o[40:56, 40:60] = 12.5
    out, mat_like = masks._fill_ambiguous(fg, amb, ev, domain, o, np.full(o.shape, 12.0), np.full(o.shape, 28.0))
    assert not out[40:56, 40:60].any() and mat_like[40:56, 40:60].all()
    out, mat_like = masks._fill_ambiguous(fg, amb, ev, domain, o, np.full(o.shape, 12.0), np.full(o.shape, 20.0))
    assert out[40:56, 40:60].all() and not mat_like.any()


@pytest.fixture(scope="module")
def dark_top_view():
    """Recht bovenaanzicht van een zwart plaatje (albedo 0,08) met een gat, op mat v2."""
    import cadquery as cq

    spec = mat.PRESETS["A4"]
    part = (cq.Workplane("XY").box(40, 25, 6, centered=(True, True, False)).edges("|Z").fillet(3)
            .faces(">Z").workplane().pushPoints([(10, 0)]).hole(6.0))
    mesh = render.tessellate(render.place(part, spec, angle_deg=12.0, offset=(10, -5)))
    cam = render.default_camera(dist=(0, 0, 0, 0, 0))
    raster = mat.rasterize_board(spec, 10.0, 3.0)
    lo, hi = mesh[0].min(axis=0), mesh[0].max(axis=0)
    c = (lo + hi) / 2
    R, t = render.look_at([c[0] + 15, c[1] - 10, 330.0], [c[0], c[1], 0.0])
    img, truth = render.render_view(raster, cam, R, t, mesh, albedo=0.08, rng=np.random.default_rng(3))
    pred, valid = masks.predict_background(raster, cam.K, Pose("top", R, t), (cam.width, cam.height))
    return img, pred, valid, truth, cam.K[0, 0] / 330.0


def test_dark_part_is_solid_and_mat_does_not_leak_in(dark_top_view):
    img, pred, valid, truth, ppm = dark_top_view
    obj = truth & valid
    core = obj & ~(ndimage.binary_dilation(~truth, iterations=3))  # meer dan 3 px binnen de rand
    old = masks.classify(img, pred, valid, px_per_mm=ppm, fill_mm=0.0)
    new = masks.classify(img, pred, valid, px_per_mm=ppm)
    # zonder opvulling blijft een flink deel onbekend: zwarte stukken mat zonder stippen onder het object
    assert np.count_nonzero(old.fg & core) < 0.9 * np.count_nonzero(core)
    assert np.count_nonzero(new.fg & core) > 0.99 * np.count_nonzero(core)
    # aan de rand boven een even donker stuk mat is de rand in deze foto niet te zien (dat beslissen
    # de andere foto's), maar er valt wel veel minder weg
    band = obj & ~core
    assert np.count_nonzero(new.fg & band) > np.count_nonzero(old.fg & band) + 0.1 * np.count_nonzero(band)
    assert np.count_nonzero(new.fg & ~truth & valid) < 0.01 * np.count_nonzero(obj)
    # 'zekere mat' lekt niet meer een halve vensterbreedte het zwarte object in
    assert np.count_nonzero(new.bg & obj) < 0.25 * np.count_nonzero(old.bg & obj)
    # het gat blijft open: daar is mat te zien
    hole = ndimage.binary_fill_holes(truth) & ~truth
    assert np.count_nonzero(new.fg & hole) < 0.5 * np.count_nonzero(hole)
    # waar object en mat even donker zijn, is een pixel geen bewijs: de fit negeert die
    assert new.amb is not None and np.count_nonzero(new.amb & obj) > 0.05 * np.count_nonzero(obj)
