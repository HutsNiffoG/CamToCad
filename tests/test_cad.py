import math

import cv2
import numpy as np
import pytest

from camtocad import cadmodel, profile, silhouette, snapping
from camtocad.profile import Hole, Part2p5D, Profile


def rect_profile(w, h, r=0.0, center=(0.0, 0.0)):
    c = np.array(center, float)
    angles = np.array([-np.pi / 2, 0.0, np.pi / 2, np.pi])  # onder, rechts, boven, links (buitennormalen)
    offsets = np.array([h / 2, w / 2, h / 2, w / 2])
    return Profile("polygon", c, angles, offsets, np.full(4, r))


# ----------------------------------------------------------------------------- snappen

def test_snap_iso273_clearance_hole():
    s = snapping.snap("gat", 6.63, 0.03, snapping.hole_candidates(6.63))
    assert s.snapped and s.value == 6.6 and "M6" in s.reason


def test_snap_whole_mm_and_refuse_when_uncertain():
    assert snapping.snap("l", 79.94, 0.05, snapping.length_candidates(79.94)).value == 80.0
    s = snapping.snap("l", 79.80, 0.3, snapping.length_candidates(79.80))
    assert not s.snapped and s.value == 79.80


def test_snap_standard_radius():
    s = snapping.snap("r", 2.93, 0.08, snapping.radius_candidates(2.93))
    assert s.snapped and s.value == 3.0


# ----------------------------------------------------------------------------- profiel

def test_rectangle_with_fillets_area_and_vertices():
    p = rect_profile(80, 40, r=3.0)
    assert np.allclose(p.vertices(), [[-40, -20], [40, -20], [40, 20], [-40, 20]])
    expected = 80 * 40 - (4 - math.pi) * 9
    assert abs(p.area() - expected) / expected < 1e-3
    assert p.is_valid()


def test_oversized_fillets_are_invalid():
    assert not rect_profile(10, 10, r=6.0).is_valid()


def test_regularize_snaps_nearly_orthogonal_edges():
    p = rect_profile(50, 30)
    p.angles = p.angles + np.radians([0.8, -0.5, 1.2, 0.0]) + np.radians(20)
    q, theta0 = profile.regularize_angles(p)
    rel = np.degrees(q.angles - theta0) % 90
    assert np.allclose(np.minimum(rel, 90 - rel), 0, atol=1e-9)


def test_footprint_to_profile_l_shape_with_hole():
    px = 0.25
    img = np.zeros((400, 400), np.uint8)
    # L-vorm 60 x 50 mm met inham 30 x 25, plus gat Ø 8 (in pixels van 0,25 mm)
    pts = (np.array([[0, 0], [60, 0], [60, 25], [30, 25], [30, 50], [0, 50]]) + 10) / px
    cv2.fillPoly(img, [pts.astype(np.int32)], 1)
    rows, cols = np.mgrid[0:400, 0:400]
    img[(cols * px - 25.0) ** 2 + (rows * px - 22.0) ** 2 < 4.0 ** 2] = 0  # gat Ø 8 (pixelmiddens)
    outer, holes, cutouts = profile.from_footprint(img > 0, (0.0, 0.0), px)
    assert outer.kind == "polygon" and outer.n == 6
    assert len(holes) == 1 and abs(holes[0].d - 8.0) < 0.15
    assert abs(holes[0].x - 25.0) < 0.05 and abs(holes[0].y - 22.0) < 0.05
    assert abs(outer.area() - (60 * 50 - 30 * 25)) < 60 * 50 * 0.01


# ----------------------------------------------------------------------------- CAD

def test_script_rebuilds_the_same_model():
    part = Part2p5D(12.0, rect_profile(80, 40, 3.0, center=(40, 20)),
                    [Hole(10, 20, 6.6), Hole(70, 20, 6.6)])
    snaps = [snapping.Snap("hoogte", 12.03, 0.03, 12.0, True, "hele mm", 0.95)]
    model = cadmodel.build(part)
    assert model.val().isValid()
    expected = (80 * 40 - (4 - math.pi) * 9 - 2 * math.pi * 3.3 ** 2) * 12
    assert abs(model.val().Volume() - expected) < 1.0
    ns = {"__name__": "test"}
    exec(compile(cadmodel.script(part, snaps), "model.py", "exec"), ns)
    assert abs(ns["model"].val().Volume() - model.val().Volume()) < 1e-3


def test_l_shape_with_reflex_fillet_is_valid():
    angles = np.array([-np.pi / 2, 0.0, np.pi / 2, 0.0, np.pi / 2, np.pi])
    # randen: onder (y=0), rechts (x=60), boven-rechts (y=25), inham (x=30), boven (y=50), links (x=0)
    offsets = np.array([0.0, 60.0, 25.0, 30.0, 50.0, 0.0])
    p = Profile("polygon", np.zeros(2), angles, offsets, np.array([2.0, 2.0, 2.0, 4.0, 2.0, 2.0]))
    assert p.is_valid()
    model = cadmodel.build(Part2p5D(8.0, p))
    assert model.val().isValid()
    assert abs(model.val().Volume() - p.area() * 8.0) < 2.0


def test_part_frame_puts_datum_at_origin():
    part = Part2p5D(5.0, rect_profile(30, 20, 1.0, center=(120, 80)), [Hole(125, 82, 4.0)])
    part = part.transformed(math.radians(33), np.array([3.0, -2.0]))
    pf, angle, shift = cadmodel.to_part_frame(part)
    V = pf.outer.vertices()
    assert np.allclose(V.min(axis=0), [0, 0], atol=1e-9) and np.allclose(V.max(axis=0), [30, 20], atol=1e-9)
    assert abs(pf.holes[0].x - 20.0) < 1e-9 and abs(pf.holes[0].y - 12.0) < 1e-9


# ----------------------------------------------------------------------------- rasterisatie

def test_shrink_makes_fillpoly_exact():
    for lo, hi in [(10.25, 50.25), (10.5, 50.5), (3.3, 40.9)]:
        m = np.zeros((64, 64), np.uint8)
        q = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]])
        pts = silhouette._to_fixed(silhouette._shrink(q), 0, 0)
        cv2.fillPoly(m, [pts], 1, cv2.LINE_8, silhouette.SHIFT)
        inside = np.arange(64)
        n = np.count_nonzero((inside > lo) & (inside < hi))
        assert m.sum() == n * n


@pytest.mark.parametrize("value, sigma", [(12.4, 0.05), (3.3, 0.02)])
def test_uncertainty_is_reported(value, sigma):
    s = snapping.snap("x", value, sigma, snapping.length_candidates(value))
    assert s.u95 == pytest.approx(2 * sigma)
