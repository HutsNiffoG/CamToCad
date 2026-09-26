"""Randfit (V2), onzekerheid per maat (V3) en gemiste gaten (V4) op scherpe synthetische silhouetten."""

import math

import numpy as np
import pytest

from camtocad import cadmodel, edgefit, holes, pipeline, silhouette, uncertainty
from camtocad.calib import Pose
from camtocad.masks import ViewMasks
from camtocad.profile import Hole, Part2p5D, Profile
from camtocad.render import look_at

W, H = 1600, 1200
K = np.array([[1300.0, 0.0, 799.5], [0.0, 1300.0, 599.5], [0.0, 0.0, 1.0]])


def plate(hole: bool = True) -> Part2p5D:
    """50 x 30 mm, 6 mm dik, R2, gat Ø6 (zelfde plaat als in test_silhouette)."""
    prof = Profile("polygon", np.array([120.0, 80.0]), np.array([-np.pi / 2, 0, np.pi / 2, np.pi]),
                   np.array([15.0, 25.0, 15.0, 25.0]), np.full(4, 2.0))
    return Part2p5D(6.0, prof, [Hole(110.0, 80.0, 6.0)] if hole else [])


def exact_mask(part: Part2p5D, v: silhouette.ViewData, s: int = 5) -> np.ndarray:
    """Silhouet met pixelmiddens binnen het model, via s x s supersampling (de snelle modelrenderer van
    de pixelfit tekent in schuine aanzichten ~0,15 px te ruim: dunne wandvierhoeken)."""
    Ks = K.copy()
    Ks[:2, :2] *= s
    Ks[:2, 2] = s * K[:2, 2] + 0.5 * (s - 1)
    h, w = v.fg.shape
    big = silhouette.ViewData(v.pose, np.zeros((s * h, s * w), bool), None, None, s * v.x0, s * v.y0)
    return silhouette.render(part, Ks, big).reshape(h, s, w, s).mean(axis=(1, 3)) >= 0.5


def scan(truth: Part2p5D, n_ring: int = 8, n_top: int = 3) -> list[silhouette.ViewData]:
    """ROI-uitsneden met het silhouet van `truth`: object waar het model is, zekere mat overal daarbuiten."""
    poses = []
    for k in range(n_ring):
        az = 2 * np.pi * (k + 0.3) / n_ring
        poses.append(look_at([120 + 200 * np.cos(az), 80 + 200 * np.sin(az), 250.0], [120.0, 80.0, 0.0]))
    for k in range(n_top):
        c = np.array([112.0 + 8 * k, 76.0 + 6 * (k % 2), 320.0])
        poses.append(look_at(c, [c[0], c[1], 0.0]))
    empty = np.zeros((H, W), bool)
    views = [(Pose(f"v{k}", R, t), ViewMasks(fg=empty, bg=~empty, valid=~empty)) for k, (R, t) in enumerate(poses)]
    vd = silhouette.prepare(views, K, truth)
    for v in vd:
        v.fg = exact_mask(truth, v)
        v.bg = ~v.fg
    return vd


@pytest.fixture(scope="module")
def plate_scan():
    return scan(plate())


@pytest.fixture(scope="module")
def plate_fit(plate_scan):
    start = plate()
    start.height += 0.3
    start.outer.offsets = start.outer.offsets + np.array([0.3, -0.2, 0.25, -0.3])
    start.outer.fillets[:] = 2.3
    start.holes[0] = Hole(110.2, 79.8, 6.3)
    return edgefit.fit(start, K, plate_scan)


# ----------------------------------------------------------------------------- randfit

def test_signed_distance_is_zero_on_the_pixel_edge():
    m = np.zeros((9, 9), bool)
    m[:, :4] = True  # kolommen 0-3 object: de rand ligt op x = 3,5
    sd = edgefit.signed_dist(m)
    assert sd[4, 3] == pytest.approx(-0.5) and sd[4, 4] == pytest.approx(0.5)
    assert sd[4, 0] < 0 < sd[4, 8]


def test_edge_fit_recovers_the_part_to_a_fraction_of_a_pixel(plate_fit):
    """Vanuit een model dat 0,2-0,3 mm verkeerd staat komt de fit op de ware maten (0,25 mm per pixel)."""
    truth, ef = plate(), plate_fit
    assert ef.accepted, ef.note
    assert ef.part.height == pytest.approx(truth.height, abs=0.04)
    assert np.allclose(ef.part.outer.offsets, truth.outer.offsets, atol=0.04)
    assert np.allclose(ef.part.outer.fillets, truth.outer.fillets, atol=0.25)
    h = ef.part.holes[0]
    assert (h.x, h.y, h.d) == pytest.approx((110.0, 80.0, 6.0), abs=0.05)


def test_edge_fit_stays_inside_the_trust_region(plate_scan):
    """Een model dat 2 mm te groot is, ligt buiten het vertrouwensgebied: dan blijft de pixelfit staan."""
    start = plate()
    start.outer.offsets = start.outer.offsets + 2.0
    ef = edgefit.fit(start, K, plate_scan)
    assert not ef.accepted and "vertrouwensgebied" in ef.note
    assert ef.part is start


def test_parameters_that_would_break_the_contour_stay_fixed():
    """Een afronding die net op zijn randen past: een stap van 0,1 mm maakt de contour ongeldig."""
    part = plate()
    part.outer.fillets[:] = [2.0, 2.0, 2.0, 27.95]  # hoek 4 en hoek 1 vullen de korte rand (30 mm) bijna
    assert part.outer.is_valid()
    free = [p.name for p in edgefit._free_params(part, part.outer.angles)]
    assert "fil3" not in free
    assert {"h", "rot", "fil1", "hd0"} <= set(free)


def test_jackknife_gives_a_small_positive_covariance(plate_fit):
    C = edgefit.jackknife(plate_fit)
    n = len(plate_fit.x)
    assert C.shape == (n, n) and np.allclose(C, C.T)
    assert np.linalg.eigvalsh(C).min() > -1e-12
    s_h = math.sqrt(C[plate_fit.names.index("h"), plate_fit.names.index("h")])
    assert 0 < s_h < 0.05


# ----------------------------------------------------------------------------- onzekerheid per maat

def _height_model(part: Part2p5D):
    def build(x):
        p = part.copy()
        p.height = float(x[0])
        p.holes = [Hole(h.x, h.y, float(x[1])) for h in p.holes]
        return p
    return build


def test_sensitivity_propagates_the_covariance_to_each_dimension():
    part = plate()
    C = np.diag([0.02 ** 2, 0.05 ** 2])
    sens = uncertainty.Sensitivity(_height_model(part), np.array([6.0, 6.0]), C, 0.3, np.array([5.0, -2.0]))
    assert sens.sigma(lambda p: p.height) == pytest.approx(0.02, rel=1e-3)
    assert sens.sigma(lambda p: p.holes[0].d) == pytest.approx(0.05, rel=1e-3)
    assert sens.sigma(lambda p: p.height + p.holes[0].d) == pytest.approx(math.hypot(0.02, 0.05), rel=1e-3)
    assert sens.sigma(lambda p: float(p.outer.offsets[0])) == pytest.approx(0.0, abs=1e-9)


def test_edge_position_along_the_axes():
    part = plate()
    # buitennormalen: onder (-y), rechts (+x), boven (+y), links (-x); middelpunt (120, 80), 50 x 30
    assert uncertainty.edge_position(part, 1, "x") == pytest.approx(145.0)
    assert uncertainty.edge_position(part, 3, "x") == pytest.approx(95.0)
    assert uncertainty.edge_position(part, 0, "y") == pytest.approx(65.0)
    # een iets scheve rand: de ligging op de hoogte van een gat, zodat een draaiing van het hele onderdeel
    # de gatpositie ten opzichte van die rand niet verandert
    tilted = part.transformed(math.radians(2.0), np.zeros(2))
    hole = tilted.holes[0]
    o, k = tilted.outer, 3
    n = np.array([math.cos(o.angles[k]), math.sin(o.angles[k])])
    x_at = uncertainty.edge_position(tilted, k, "x", at=hole.y)
    assert n @ [x_at, hole.y] == pytest.approx(n @ o.center + o.offsets[k])
    assert hole.x - x_at == pytest.approx((110.0 - 95.0) / math.cos(math.radians(2.0)), abs=1e-9)


def test_budget_adds_systematics_and_print_scale():
    sens = uncertainty.Sensitivity(_height_model(plate()), np.array([6.0, 6.0]), np.diag([0.02 ** 2, 0.0]), 0.0,
                                   np.zeros(2))
    b = uncertainty.Budget(sens, mm_per_px=0.25, scale_rel=1e-3)
    rel = b.rel(lambda p: p.height, "hoogte")
    assert rel == pytest.approx(math.hypot(0.02, uncertainty.SYS_PX["hoogte"] * 0.25))
    assert b.total(rel, 6.0) == pytest.approx(math.hypot(rel, 0.006))
    assert uncertainty.Budget(None, 0.25, 1e-3).rel(lambda p: p.height, "hoogte") is None


def test_snap_part_with_a_budget_gives_each_dimension_its_own_u95():
    """Een onzekere hoogte krijgt een ruime U95, een goed bepaalde lengte een krappe; de printschaal telt
    in de U95 maar niet bij het snappen."""
    prof = Profile("polygon", np.array([40.0, 20.0]), np.array([-np.pi / 2, 0, np.pi / 2, np.pi]),
                   np.array([20.0, 40.02, 20.0, 40.0]), np.zeros(4))
    part = Part2p5D(12.2, prof, [Hole(20.0, 20.0, 6.6)])
    sens = uncertainty.Sensitivity(_height_model(part), np.array([12.2, 6.6]), np.diag([0.2 ** 2, 0.01 ** 2]), 0.0,
                                   np.zeros(2))
    unc = cadmodel.estimate_uncertainty(0.25, 40, 5)
    snapped, snaps = cadmodel.snap_part(part, unc, budget=uncertainty.Budget(sens, 0.25, scale_rel=0.01))
    by = {s.name: s for s in snaps}
    sys_h, sys_l = uncertainty.SYS_PX["hoogte"] * 0.25, uncertainty.SYS_PX["lengte"] * 0.25
    assert by["hoogte"].u95 == pytest.approx(2 * math.hypot(0.2, sys_h, 0.01 * 12.2), rel=1e-3)
    length = by["x-maat rand 2"]
    assert length.snapped and length.value == 80.0  # 80,02 ± 0,03 (zonder printschaal)
    assert length.u95 == pytest.approx(2 * math.hypot(sys_l, 0.01 * 80.02), rel=1e-3)
    assert snapped.outer.vertices()[:, 0].max() == pytest.approx(80.0)


# ----------------------------------------------------------------------------- gemiste gaten

def test_a_hole_missing_from_the_model_is_found_in_the_top_views(plate_scan):
    found = holes.candidates(plate(hole=False), K, plate_scan)
    assert len(found) == 1
    h = found[0]
    assert math.hypot(h.x - 110.0, h.y - 80.0) < 0.6 and 4.5 < h.d < 6.5


def test_no_hole_candidates_where_the_model_already_has_the_hole(plate_scan):
    assert holes.candidates(plate(), K, plate_scan) == []


def test_round_cutouts_become_holes():
    part = plate(hole=False)
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    circle = np.column_stack([130 + 3 * np.cos(a), 80 + 3 * np.sin(a)])
    square = np.array([[100.0, 75.0], [104.0, 75.0], [104.0, 79.0], [100.0, 79.0]])
    part.cutouts = [circle, square]
    out, n = holes.round_cutouts(part)
    assert n == 1 and len(out.cutouts) == 1 and np.allclose(out.cutouts[0], square)
    h = out.holes[0]
    assert (h.x, h.y) == pytest.approx((130.0, 80.0), abs=1e-6) and h.d == pytest.approx(6.0, abs=0.05)


def test_pipeline_adds_a_missed_hole_only_when_the_model_fits_better(plate_scan):
    messages = []
    start = plate(hole=False)
    e0 = silhouette.energy(start, K, plate_scan)
    part, e = pipeline._add_missed_holes(start, K, plate_scan, e0, log=messages.append)
    assert len(part.holes) == 1 and e < 0.5 * e0
    h = part.holes[0]
    assert (h.x, h.y, h.d) == pytest.approx((110.0, 80.0, 6.0), abs=0.3)
    assert any("gat toegevoegd" in m for m in messages)
    # met het gat al in het model verandert er niets
    same, e_same = pipeline._add_missed_holes(plate(), K, plate_scan, silhouette.energy(plate(), K, plate_scan),
                                              log=messages.append)
    assert len(same.holes) == 1 and e_same == silhouette.energy(plate(), K, plate_scan)
