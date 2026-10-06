"""De grove visual hull kiest het onderdeel als er meer voorwerpen op de mat liggen (V12, v0.12)."""

import numpy as np

from camtocad import hull, mat, render
from camtocad.calib import Pose
from camtocad.masks import ViewMasks


def test_the_object_is_the_one_the_top_views_aim_at_not_the_largest():
    occ = np.zeros((60, 40, 10), bool)
    occ[5:45, 2:8, :1] = True  # een liniaal: 80 x 12 mm, 2 mm hoog
    occ[20:35, 20:30, :5] = True  # het onderdeel: 30 x 20 mm, 10 mm hoog
    occ[50:52, 30:32, :1] = True  # een kruimel van 16 mm²: telt niet mee
    origin, voxel = np.zeros(3), 2.0
    mask, chosen, others = hull.choose_object(occ, origin, voxel)  # zonder richtpunt: het grootste
    assert mask[10, 4, 0] and not mask[25, 25, 0]
    assert np.allclose(chosen.size, [80, 12]) and len(others) == 1
    mask, chosen, others = hull.choose_object(occ, origin, voxel, aim=np.array([50.0, 45.0]))
    assert mask[25, 25, 0] and not mask[10, 4, 0] and not mask[51, 31, 0]
    assert np.allclose(chosen.center, [54, 49]) and np.allclose(chosen.size, [30, 20])
    assert chosen.area == 600 and chosen.height == 10 and not chosen.at_edge
    assert len(others) == 1 and np.allclose(others[0].center, [49, 9]) and others[0].area == 960
    occ = np.zeros((60, 40, 10), bool)
    occ[0:10, 20:30, :3] = True  # tegen de rand van de mat
    _, chosen, others = hull.choose_object(occ, origin, voxel)
    assert chosen.at_edge and not others


def test_a_ruler_next_to_the_part_is_reported_and_left_out():
    """Gerenderde maskers: een plaatje midden op de mat en een langere liniaal ernaast; de foto's zijn op het
    plaatje gericht. De grove hull kiest het plaatje en meldt de liniaal."""
    import cadquery as cq
    spec = mat.PRESETS["A4"]
    w, h = spec.board_w_mm, spec.board_h_mm
    plate = cq.Workplane("XY").box(40, 30, 6, centered=(True, True, False)).translate((w / 2, h / 2, 0))
    ruler = cq.Workplane("XY").box(150, 20, 2, centered=(True, True, False)).translate((w / 2, h / 2 + 55, 0))
    mesh = render.tessellate(plate.union(ruler))
    cam = render.default_camera(800, 600, 650.0, dist=(0, 0, 0, 0, 0))
    raster = mat.rasterize_board(spec, 4.0, 3.0)
    rng = np.random.default_rng(3)
    views = []
    for name, R, t in render.scan_poses((w / 2, h / 2, 3.0), 330.0, ((45.0, 8), (70.0, 6)), 4, rng):
        _, m = render.render_view(raster, cam, R, t, mesh, noise=0, rng=rng)
        views.append((Pose(name, R, t), ViewMasks(fg=m, bg=~m, valid=np.ones_like(m), edge_bg=~m)))
    grid = hull.reconstruct(views, cam.K, (0.0, w, 0.0, h), fine=False)
    assert np.allclose(grid.chosen.center, [w / 2, h / 2], atol=3) and np.allclose(grid.chosen.size, [40, 30], atol=5)
    assert len(grid.others) == 1 and np.allclose(grid.others[0].center, [w / 2, h / 2 + 55], atol=3)
    assert np.allclose(grid.others[0].size, [150, 20], atol=6) and not grid.chosen.at_edge
    ijk = np.argwhere(grid.occ)
    assert np.ptp(ijk[:, 1]) * grid.voxel < 36  # alleen het plaatje in de hull
