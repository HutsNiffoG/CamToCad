"""Startmodel uit de bovenaanzichten, ook als de grove visual hull veel te hoog is.

Regressie voor 'Geen objectcontour gevonden': bij echte scans met weinig schuine foto's is de
grove hull vaak veel hoger dan het object. De oude start (terugprojecteren op 0,6 x die hoogte)
liet de bovenaanzichten dan uit elkaar vallen; de hoogtezoektocht vindt het bovenvlak zelf.
"""

import numpy as np
import pytest

from camtocad import initial, mat, render
from camtocad.calib import Pose
from camtocad.hull import VoxelGrid
from camtocad.masks import ViewMasks

HEIGHT = 5.0


def _plate():
    import cadquery as cq
    return (cq.Workplane("XY").box(30, 20, HEIGHT, centered=(True, True, False)).edges("|Z").fillet(2)
            .faces(">Z").workplane().pushPoints([(-8, 0)]).hole(4.5))


@pytest.fixture(scope="module")
def top_scan():
    spec = mat.PRESETS["A4"]
    offset = np.array([-30.0, 20.0])
    mesh = render.tessellate(render.place(_plate(), spec, angle_deg=25.0, offset=tuple(offset)))
    cam = render.default_camera(dist=(0, 0, 0, 0, 0))
    raster = mat.rasterize_board(spec, 10.0, 3.0)
    rng = np.random.default_rng(4)
    target = np.array([spec.board_w_mm / 2 + offset[0], spec.board_h_mm / 2 + offset[1], 0.0])
    views = []
    for k in range(5):  # bovenaanzichten vanaf verschillende plekken, zoals met de hand
        lateral = rng.normal(0, 40.0, 2)
        R, t = render.look_at(target + [lateral[0], lateral[1], 330.0],
                              target + [0.8 * lateral[0], 0.8 * lateral[1], 0.0])
        _, mask = render.render_view(raster, cam, R, t, mesh, rng=rng)
        views.append((Pose(f"top{k}", R, t), ViewMasks(fg=mask, bg=~mask, valid=np.ones_like(mask), edge_bg=~mask)))
    # grove hull op de goede plek, maar 60 mm hoog (object: 5 mm)
    coarse = VoxelGrid(np.array([target[0] - 21, target[1] - 17, 1.0]), 2.0, np.ones((22, 18, 30), bool))
    board = (0.0, spec.board_w_mm, 0.0, spec.board_h_mm)
    return cam, views, coarse, board


def test_top_views_are_selected_by_tilt(top_scan):
    _, views, _, _ = top_scan
    assert all(initial.tilt_deg(p) < 8.0 for p, _ in views)
    assert len(initial.select_top_views(views)) == 5


def test_height_sweep_finds_top_face(top_scan):
    cam, views, coarse, _ = top_scan
    bounds = (coarse.origin[0] - 7, coarse.origin[0] + 50, coarse.origin[1] - 7, coarse.origin[1] + 42)
    sweep = initial.sweep_height(views, cam.K, bounds, 60.0)
    assert sweep.informative
    # piek iets onder het bovenvlak (de wand aan de camerakant is ook zichtbaar); startwaarde voor fit_height
    assert HEIGHT - 2.0 < sweep.best < HEIGHT + 0.5


def test_locate_with_far_too_tall_hull(top_scan):
    cam, views, coarse, board = top_scan
    res = initial.locate(views, cam.K, coarse, board)
    part = res.part
    assert HEIGHT - 2.0 < part.height < HEIGHT + 0.5  # startwaarde; fit_height (alle foto's) verfijnt
    expected = 30 * 20 - (4 - np.pi) * 2 ** 2
    assert part.outer.n == 4
    assert abs(part.outer.area() - expected) < 0.04 * expected
    assert len(part.holes) == 1 and abs(part.holes[0].d - 4.5) < 0.6
    # dezelfde foto's en instellingen op de (door fit_height gevonden) echte hoogte
    again = res.at_height(cam.K, coarse, HEIGHT)
    assert again.outer.n == 4
    assert abs(again.outer.area() - expected) < 0.015 * expected
    assert len(again.holes) == 1 and abs(again.holes[0].d - 4.5) < 0.25


def test_no_object_gives_explained_error(top_scan):
    cam, views, coarse, board = top_scan
    empty = [(p, ViewMasks(fg=np.zeros_like(m.fg), bg=m.valid, valid=m.valid, edge_bg=m.valid)) for p, m in views]
    with pytest.raises(initial.InitError) as err:
        initial.locate(empty, cam.K, coarse, board)
    assert err.value.details and "bovenaanzichten" in str(err.value)
