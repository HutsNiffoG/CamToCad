"""Verplaatsingscontrole: foto's van een onderdeel dat tussendoor is verschoven of omgelegd."""

import math

import cv2
import numpy as np

from camtocad import placement, render
from camtocad.calib import Pose, project
from camtocad.masks import ViewMasks

W, H = 1600, 1200
K = np.array([[1300.0, 0, W / 2], [0, 1300.0, H / 2], [0, 0, 1]])
BOUNDS = (0.0, 240.0, 0.0, 162.0)


def _box(cx, cy, angle_deg, sx, sy, h):
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    xy = np.array([[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]]) / 2 @ R.T + [cx, cy]
    return np.vstack([np.column_stack([xy, np.zeros(4)]), np.column_stack([xy, np.full(4, h)])])


def _views(box, prefix, n_ring=8, n_top=3, seed=0):
    """Silhouetten van een balk (bolle vorm: omhullende van de hoekpunten) vanuit een ring en van boven."""
    rng = np.random.default_rng(seed)
    target = np.array([*box[:, :2].mean(axis=0), 0.0])
    poses = []
    for k in range(n_ring):
        az, el = 2 * math.pi * (k + rng.uniform(0, 0.5)) / n_ring, math.radians(rng.uniform(40, 65))
        poses.append(render.look_at(target + 320 * np.array([math.cos(el) * math.cos(az),
                                                              math.cos(el) * math.sin(az), math.sin(el)]), target))
    for k in range(n_top):
        c = target + [*rng.normal(0, 25, 2), 320.0]
        poses.append(render.look_at(c, [c[0], c[1], 0.0]))
    out = []
    for k, (R, t) in enumerate(poses):
        pose = Pose(f"{prefix}{k:02d}", R, t)
        uv, _ = project(box, pose, K)
        fg = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(fg, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
        fg = fg > 0
        near = cv2.dilate(fg.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        out.append((pose, ViewMasks(fg=fg, bg=~near, valid=np.ones_like(fg))))
    return out


def test_an_object_that_stays_put_is_one_placement():
    views = _views(_box(120, 80, 15, 60, 30, 10), "a", n_ring=10, n_top=4)
    res = placement.find(views, K, BOUNDS)
    assert len(res.groups) == 1 and len(res.groups[0]) == len(views)
    assert not res.outliers and min(res.scores.values()) > 0.9


def test_moved_and_turned_on_its_side_are_separate_placements():
    flat = _views(_box(120, 80, 15, 60, 30, 10), "plat", seed=1)
    moved = _views(_box(150, 95, 60, 60, 30, 10), "schuif", seed=2)
    side = _views(_box(115, 85, -20, 60, 10, 30), "kant", seed=3)
    for other in (moved, side):
        res = placement.find(flat + other, K, BOUNDS)
        groups = sorted(sorted(g) for g in res.groups)
        assert groups == sorted([sorted(p.name for p, _ in flat), sorted(p.name for p, _ in other)]), res.groups
        assert not res.outliers


def test_a_few_odd_photos_are_outliers_not_a_placement():
    views = _views(_box(120, 80, 15, 60, 30, 10), "a", n_ring=12, n_top=4)
    odd = _views(_box(60, 40, 0, 40, 20, 8), "b", n_ring=2, n_top=0, seed=4)  # iets anders, twee keer
    empty = (Pose("leeg", views[0][0].R, views[0][0].t),
             ViewMasks(fg=np.zeros((H, W), bool), bg=np.ones((H, W), bool), valid=np.ones((H, W), bool)))
    res = placement.find(views + odd + [empty], K, BOUNDS)
    assert len(res.groups) == 1 and len(res.groups[0]) == len(views)
    assert sorted(res.outliers) == ["b00", "b01"]
    assert res.unjudged == ["leeg"]


def test_the_message_names_the_groups_in_photo_order():
    order = [f"IMG_{k:03d}.jpg" for k in range(30)]
    res = placement.Placements([order[:15], order[15:27]], [order[27]], order[28:])
    text = placement.moved_message(res, order)
    assert "2 groepen" in text and "IMG_000.jpg t/m IMG_014.jpg" in text and "IMG_015.jpg t/m IMG_026.jpg" in text
    assert "1 foto past nergens bij" in text and "aparte scan" in text
