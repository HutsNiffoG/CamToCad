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


def test_chamfer_does_not_tilt_the_part_frame():
    """Eén schuine rand (afschuining 30°) mag de hoofdrichting niet scheef trekken."""
    # rechthoek 80 x 40 met één hoek schuin afgesneden (17,3 x 10 mm), dan 17° gedraaid
    angles = np.array([-np.pi / 2, 0.0, np.radians(60.0), np.pi / 2, np.pi])
    p = Profile("polygon", np.zeros(2), angles, np.zeros(5), np.zeros(5))
    corners = [(0, 0), (80, 0), (80, 30), (62.68, 40), (0, 40)]
    for k, a in enumerate(angles):  # rand k loopt van hoek k naar hoek k+1
        n = np.array([np.cos(a), np.sin(a)])
        p.offsets[k] = n @ np.array(corners[k], float)
    assert p.is_valid()
    rot = Part2p5D(5.0, p).transformed(math.radians(17.0), np.array([100.0, 50.0])).outer
    assert abs(math.degrees(profile.dominant_angle(rot)) - 17.0) < 1e-6


def test_hole_with_mask_blemish_stays_a_hole():
    """Een gat met een 'staart' (maskerfout) moet een rond gat blijven, geen uitsparing."""
    px = 0.25
    rows, cols = np.mgrid[0:240, 0:320]
    img = np.zeros((240, 320), np.uint8)
    img[20:220, 20:300] = 1
    img[(cols * px - 30.0) ** 2 + (rows * px - 30.0) ** 2 < 3.3 ** 2] = 0  # gat Ø 6,6
    img[(rows * px > 32.0) & (rows * px < 35.2) & (np.abs(cols * px - 29.5) < 0.9)] = 0  # staart onder het gat
    outer, holes, cutouts = profile.from_footprint(img > 0, (0.0, 0.0), px, lenient_holes=True)
    assert not cutouts and len(holes) == 1
    assert abs(holes[0].d - 6.6) < 0.3 and abs(holes[0].x - 30.0) < 0.2 and abs(holes[0].y - 30.0) < 0.2


def test_short_edges_are_removed():
    """Een trapje van 0,01 mm in een hoek (twee extra randen) verdwijnt; de rechthoek blijft."""
    # rechthoek 30 x 20; onderrand in twee stukken op y = 0 en y = 0,01 met een mini-rand ertussen
    angles = np.array([-np.pi / 2, np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
    offsets = np.array([0.0, -0.3, 0.01, 30.0, 20.0, 0.0])  # rand 1: x = 0,3 (verticaal stukje van 0,01 mm)
    p = Profile("polygon", np.zeros(2), angles, offsets, np.zeros(6))
    q = profile.remove_short_edges(p, 0.75)
    assert q.n == 4 and q.is_valid()
    V = q.vertices()
    assert np.allclose(V.min(axis=0), [0.0, 0.0], atol=0.02) and np.allclose(V.max(axis=0), [30.0, 20.0], atol=0.02)


def test_slightly_slanted_edge_does_not_bias_the_frame():
    """Een korte rand die maar ~6° afwijkt (binnen het zoekvenster) mag de hoofdrichting niet verschuiven."""
    corners = [(0, 0), (80, 0), (80, 20), (79, 30), (79, 40), (0, 40)]
    n = len(corners)
    angles, offsets = np.zeros(n), np.zeros(n)
    for k in range(n):  # rand k loopt van hoek k naar hoek k+1; buitennormaal rechts van de looprichting
        (x0, y0), (x1, y1) = corners[k], corners[(k + 1) % n]
        normal = np.array([y1 - y0, -(x1 - x0)], float)
        normal /= np.linalg.norm(normal)
        angles[k] = math.atan2(normal[1], normal[0])
        offsets[k] = normal @ np.array([x0, y0], float)
    p = Profile("polygon", np.zeros(2), angles, offsets, np.zeros(n))
    assert p.is_valid()
    rot = Part2p5D(5.0, p).transformed(math.radians(-25.0), np.array([100.0, 50.0])).outer
    assert abs(math.degrees(profile.dominant_angle(rot)) + 25.0) < 1e-6


def _from_vertices(V, fillets=None):
    V = np.asarray(V, float)
    d = np.roll(V, -1, axis=0) - V
    n = np.column_stack([d[:, 1], -d[:, 0]]) / np.linalg.norm(d, axis=1, keepdims=True)  # buitennormaal (CCW)
    c = V.mean(axis=0)
    f = np.zeros(len(V)) if fillets is None else np.asarray(fillets, float)
    return Profile("polygon", c, np.arctan2(n[:, 1], n[:, 0]), np.einsum("ij,ij->i", n, V - c), f)


def test_merge_at_vertex_straightens_a_kink():
    """Twee randen met een knik van 4° worden één rechte rand; de andere hoeken en afrondingen blijven."""
    p = _from_vertices([(0, 0), (60, 0), (60, 40), (30, 41.05), (0, 40)], fillets=[1.0, 2.0, 3.0, 0.5, 4.0])
    q = profile.merge_at_vertex(p, 3)
    assert q is not None and q.n == 4 and q.is_valid()
    V = q.vertices()
    assert np.allclose(V[[0, 1]], [(0, 0), (60, 0)], atol=1e-6)
    assert np.allclose(V[2:, 1], 40.525, atol=0.01)  # gemiddelde hoogte van de twee delen, horizontaal
    assert np.allclose(q.fillets, [1.0, 2.0, 3.0, 4.0])


def test_hole_next_to_an_invisible_area_is_fitted_on_its_visible_rim():
    """Een rond gat naast een vlak dat in de bovenaanzichten niets zegt (zwart op zwart): de opening loopt
    door tot in dat vlak, maar de cirkel volgt alleen de zichtbare rand."""
    px = 0.25
    rows, cols = np.mgrid[0:240, 0:320]
    img = np.zeros((240, 320), np.uint8)
    img[20:220, 20:300] = 1
    hole = (cols * px - 30.0) ** 2 + (rows * px - 30.0) ** 2 < 3.3 ** 2  # gat Ø 6,6
    # zwart vlak rechts tegen het gat aan: bedekt ~45% van de rand
    square = (cols * px > 30.5) & (cols * px < 36.5) & (rows * px > 26.0) & (rows * px < 34.0)
    unknown = square & ~hole
    img[hole | square] = 0
    _, holes, cutouts = profile.from_footprint(img > 0, (0.0, 0.0), px, lenient_holes=True)
    assert cutouts or abs(holes[0].d - 6.6) > 0.5  # zonder die kennis: uitsparing of een veel te groot gat
    _, holes, cutouts = profile.from_footprint(img > 0, (0.0, 0.0), px, lenient_holes=True, unknown=unknown)
    assert not cutouts and len(holes) == 1
    assert abs(holes[0].d - 6.6) < 0.3 and abs(holes[0].x - 30.0) < 0.2 and abs(holes[0].y - 30.0) < 0.2
