"""Randfit (V2), onzekerheid per maat (V3) en gemiste gaten (V4) op scherpe synthetische silhouetten."""

import math

import cv2
import numpy as np
import pytest

from camtocad import cadmodel, edgefit, holes, pipeline, silhouette, uncertainty
from camtocad.calib import Pose, project
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
    """Silhouet als een gerenderd masker: een pixel is object als meer dan de helft ervan het model bedekt,
    via s x s supersampling (de modelrenderer zelf neemt het pixelmidden; tot v0.7 tekende hij in schuine
    aanzichten ~0,15 px te ruim, en daarom was dit het ijkpunt, V29)."""
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


def without_mat_around_hole(vd, part: Part2p5D, sector=None, r_mm: float = 6.0) -> list[silhouette.ViewData]:
    """Geen zekere mat meer rond het gat (binnen r_mm van het middelpunt, onder en boven), of alleen in de
    hoeksector (graden, tegen de klok in vanaf +x): zoals een zwart onderdeel boven een zwart vak."""
    h = part.holes[0]
    a = np.radians(np.arange(0.0, 360.0, 0.5))
    if sector is not None:
        a = a[(np.degrees(a) - sector[0]) % 360 < (sector[1] - sector[0]) % 360]
    rr = np.linspace(0.0, r_mm, 25)
    disc = np.column_stack([h.x + np.outer(rr, np.cos(a)).ravel(), h.y + np.outer(rr, np.sin(a)).ravel()])
    return without_mat_at(vd, part, disc)


def without_mat_at(vd, part: Part2p5D, pts: np.ndarray) -> list[silhouette.ViewData]:
    """Geen zekere mat meer bij deze punten (2D, onder en boven): zoals een zwart onderdeel boven een zwart vak."""
    out = []
    for v in vd:
        hh, ww = v.fg.shape
        blank = np.zeros((hh, ww), np.uint8)
        for z in (0.0, part.height):
            uv, _ = project(np.column_stack([pts, np.full(len(pts), z)]), v.pose, K)
            uv = np.round(uv - [v.x0, v.y0]).astype(int)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < ww) & (uv[:, 1] >= 0) & (uv[:, 1] < hh)
            blank[uv[ok, 1], uv[ok, 0]] = 1
        bg = v.bg & ~(cv2.dilate(blank, np.ones((5, 5), np.uint8)) > 0)
        out.append(silhouette.ViewData(v.pose, v.fg, bg, ~(v.fg | bg), v.x0, v.y0))
    return out


def test_evidence_all_around_a_hole_and_on_half_of_it(plate_scan):
    """Met zekere mat rondom telt het systematische deel gewoon; met bewijs aan één kant hangen plaats en maat
    samen en is het ~2,3x zo groot; zonder bewijs is het gat 'zwak'."""
    truth = plate()
    for views, lo, hi in ((plate_scan, 1.0, 1.0), (without_mat_around_hole(plate_scan, truth, (0, 180)), 1.8, 3.0)):
        prob = edgefit._Problem(truth, K, views)
        prob.set_status(truth)
        e = edgefit.evidence(prob, prob.x_of(truth))[("gat", 0)]
        assert lo - 1e-6 <= e.amp_size <= hi + 1e-6 and lo - 1e-6 <= e.amp_pos <= hi + 1e-6 and not e.weak
    prob = edgefit._Problem(truth, K, without_mat_around_hole(plate_scan, truth))
    prob.set_status(truth)
    e = edgefit.evidence(prob, prob.x_of(truth))[("gat", 0)]
    assert e.weak and e.fraction < edgefit.EVIDENCE_MIN and e.amp_size == e.amp_pos == edgefit.AMP_MAX


def test_a_hole_without_evidence_stays_put_and_gets_a_wide_u95(plate_scan):
    """Zwart op zwart rond het gat: het gat blijft staan (anders drijft het naar de rand van het
    vertrouwensgebied en blijft voor alles de pixelfit staan), de rest wordt gewoon gefit, en maat en plaats
    van het gat krijgen een ruime U95 in plaats van alleen de systematiek van een gat met bewijs rondom."""
    truth = plate()
    views = without_mat_around_hole(plate_scan, truth)
    start = truth.copy()
    start.outer.offsets = start.outer.offsets + np.array([0.3, -0.2, 0.25, -0.3])
    start.holes[0] = Hole(110.2, 79.8, 6.3)
    ef = edgefit.fit(start, K, views)
    assert ef.accepted, ef.note
    assert ef.extra["evidence"][("gat", 0)].weak
    assert (ef.part.holes[0].x, ef.part.holes[0].y, ef.part.holes[0].d) == (110.2, 79.8, 6.3)
    assert np.allclose(ef.part.outer.offsets, truth.outer.offsets, atol=0.05)
    part_pf, angle, shift = cadmodel.to_part_frame(ef.part)
    budget = uncertainty.Budget(uncertainty.Sensitivity(ef.extra["problem"].build, ef.x, edgefit.jackknife(ef),
                                                        angle, shift), 0.25, 0.0)
    unc = cadmodel.estimate_uncertainty(0.25, len(views), 3)
    _, snaps = cadmodel.snap_part(part_pf, unc, budget=budget, evidence=ef.extra["evidence"])
    by = {s.name: s for s in snaps}
    assert by["gat Ø (gat 1)"].sigma >= 0.99 * edgefit.AMP_MAX * uncertainty.SYS_PX["gat"] * 0.25
    assert by["gat 1 x"].sigma >= 0.99 * edgefit.AMP_MAX * uncertainty.SYS_PX["positie"] * 0.25
    assert not by["gat Ø (gat 1)"].snapped


def test_equal_holes_are_averaged_by_how_well_each_is_seen():
    """Twee gaten die even groot lijken: het gat met bewijs aan maar een deel van de rand telt minder mee."""
    prof = Profile("polygon", np.array([40.0, 20.0]), np.array([-np.pi / 2, 0, np.pi / 2, np.pi]),
                   np.array([20.0, 40.0, 20.0, 40.0]), np.zeros(4))
    part = Part2p5D(12.0, prof, [Hole(10.0, 20.0, 6.6), Hole(70.0, 20.0, 6.45)])
    unc = cadmodel.estimate_uncertainty(0.25, 40, 5)
    ev = {("gat", 0): edgefit.Evidence(1.0, 1.0, 1.0), ("gat", 1): edgefit.Evidence(0.4, 4.0, 4.0)}
    _, snaps = cadmodel.snap_part(part, unc, evidence=ev)
    d = next(s for s in snaps if s.name == "gat Ø (2x)")
    assert d.measured == pytest.approx((6.6 + 6.45 / 16) / (1 + 1 / 16))
    by = {s.name: s for s in snaps}
    assert by["gat 2 x"].sigma > 3.9 * by["gat 1 x"].sigma


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


def test_a_nudge_between_photos_is_measured_found_and_undone():
    """Na de zevende foto is de plaat 0,4 mm opzij geschoven en 0,3° gedraaid (V27, v0.12). Per foto de verschuiving
    van het model: ~0 ervoor, de beweging erna; de sprong is een duwtje, en met de gecorrigeerde poses past het model
    weer in elke foto."""
    from camtocad import placement

    truth = plate()
    vd = scan(truth, n_ring=10, n_top=4)
    oc = truth.outer.outline()
    c = (oc.min(axis=0) + oc.max(axis=0)) / 2
    d = np.array([0.4, -0.25, math.radians(0.3)])
    nudged = edgefit.moved(truth, c, d)
    for v in vd[7:]:
        v.fg = exact_mask(nudged, v)
        v.bg = ~v.fg
    D, c2 = edgefit.view_offsets(truth, K, vd)
    radius = float(np.max(np.linalg.norm(oc - c, axis=1)))
    rim = [1.0, 1.0, radius]  # de draaiing als verplaatsing aan de rand (mm)
    assert np.allclose(c2, c)
    assert np.all(np.abs(D[:7] * rim) < 0.05)  # pixelmaskers: 0,15-0,25 mm per pixel
    assert np.all(np.abs((D[7:] - d) * rim) < 0.05)
    names = [v.pose.name for v in vd]
    info = {}
    nd = placement.find_nudge(D, names, names, c, radius, info)
    assert nd is not None and nd.before == "v6" and nd.moved == names[7:], info
    assert np.allclose(nd.motion, d, atol=0.03) and 0.45 < nd.size_mm < 0.75
    for v, (p, _) in zip(vd, placement.apply_nudge([(v.pose, None) for v in vd], nd)):
        v.pose = p
    D2, _ = edgefit.view_offsets(truth, K, vd)
    assert np.all(np.abs(D2 * rim) < 0.07)  # ruis, plus de fout in de geschatte beweging
    assert placement.find_nudge(D2, names, names, c, radius) is None
    # zonder duwtje: geen melding (de toets op de verschuivingen van een stilliggend onderdeel)
    assert placement.find_nudge(D[:7], names[:7], names, c, radius) is None
    # een flinke duw (2,5 mm en 1°, in beeld meer dan 15 px): de verschuiving per foto volgt die ook, al is er zoals
    # in echte maskers alleen zekere mat vlak langs het silhouet (egale mat telt niet)
    big = np.array([2.0, 1.5, math.radians(1.0)])
    for v in vd[7:]:
        v.fg = exact_mask(edgefit.moved(truth, c, big), v)
        v.bg = ~v.fg & (cv2.dilate(v.fg.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0)
    D3, _ = edgefit.view_offsets(truth, K, vd)
    assert np.all(np.abs((D3[7:] - big) * rim) < 0.05)


def test_a_round_part_has_no_measurable_rotation():
    """Een ring met het gat in het midden (V27): een draaiing verandert in geen foto iets, dus per foto alleen een
    verschuiving; die volgt een duw zoals bij elk ander onderdeel."""
    ring = Part2p5D(8.0, Profile("circle", np.array([120.0, 80.0]), radius=12.5), [Hole(120.0, 80.0, 8.0)])
    vd = scan(ring)
    for v in vd[5:]:
        v.fg = exact_mask(edgefit.moved(ring, np.array([120.0, 80.0]), [0.3, -0.2, 0.0]), v)
        v.bg = ~v.fg
    D, _ = edgefit.view_offsets(ring, K, vd)
    assert np.all(D[:, 2] == 0.0)
    assert np.all(np.abs(D[:5, :2]) < 0.05) and np.all(np.abs(D[5:, :2] - [0.3, -0.2]) < 0.05)


def test_a_missed_notch_and_tab_are_residual_clusters(plate_scan):
    """De foto's van boven zien een inham van 3 x 2,5 mm in de bovenrand en een lipje van 3 x 2,5 mm aan de
    onderrand, die het model mist (V14, v0.12): twee restclusters op de goede plek. Zonder die afwijkingen geen."""
    part = plate()
    assert holes.residual_clusters(part, K, plate_scan) == []
    vd = scan(part, n_top=4)

    def box(x0, x1, y0, y1):
        return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (0.0, part.height)])

    notch, tab = box(118.0, 121.0, 92.5, 95.0), box(128.0, 131.0, 62.5, 65.0)
    for v in vd[8:]:  # de bovenaanzichten
        def poly(P):
            uv, _ = project(P, v.pose, K)
            return cv2.convexHull(np.round(uv - [v.x0, v.y0]).astype(np.int32))
        top, bottom = np.zeros(v.fg.shape, np.uint8), np.zeros(v.fg.shape, np.uint8)
        cv2.fillConvexPoly(top, poly(notch[notch[:, 2] > 0]), 1)
        cv2.fillConvexPoly(bottom, poly(notch[notch[:, 2] == 0]), 1)
        through = (top & bottom) > 0  # door de inham heen: binnen de projectie van boven- en onderkant
        lip = np.zeros_like(top)
        cv2.fillConvexPoly(lip, poly(tab), 1)
        v.fg = (v.fg & ~through) | (lip > 0)
        v.bg = ~v.fg
    found = holes.residual_clusters(part, K, vd)
    assert sorted(c["soort"] for c in found) == ["inham", "uitstulping"], found
    by = {c["soort"]: c for c in found}
    assert abs(by["inham"]["x"] - 119.5) < 0.5 and 92.5 < by["inham"]["y"] < 94.5
    assert abs(by["uitstulping"]["x"] - 129.5) < 0.5 and 62.5 < by["uitstulping"]["y"] < 64.5
    assert all(c["fotos"] >= 3 for c in found)


def test_an_edge_the_model_does_not_follow_is_a_misfit(plate_scan):
    """De rechterrand ligt in de foto's 0,6 mm verder naar buiten dan in het model (V14, v0.12): één stuk rand dat
    het model niet volgt, langs die hele rand. Met het goede model niets."""
    assert edgefit.contour_misfit(plate(), K, plate_scan, 0.2) == []
    wrong = plate()
    wrong.outer.offsets = wrong.outer.offsets - np.array([0.0, 0.6, 0.0, 0.0])
    found = edgefit.contour_misfit(wrong, K, plate_scan, 0.2)
    assert len(found) == 1, found
    m = found[0]
    assert m["afwijking"] == pytest.approx(0.6, abs=0.1) and m["lengte"] > 20 and m["fotos"] >= 4
    assert m["x"] == pytest.approx(144.4, abs=0.5) and m["y"] == pytest.approx(80.0, abs=2.0)


def test_edge_fit_recovers_a_slot():
    """Een sleuf (V15) in de plaat: middelpunt, breedte en lengte uit de randen van de doorkijk."""
    from camtocad.profile import Slot

    truth = plate()
    truth.slots = [Slot(130.0, 80.0, 14.0, 5.0, math.radians(90.0), 0.0, "sleuf")]
    views = scan(truth)
    start = truth.copy()
    start.slots[0] = Slot(130.3, 79.7, 13.6, 4.7, math.radians(88.5), 0.0, "sleuf")
    ef = edgefit.fit(start, K, views)
    assert ef.accepted, ef.note
    s = ef.part.slots[0]
    assert (s.x, s.y, s.width) == pytest.approx((130.0, 80.0, 5.0), abs=0.05)
    assert s.length == pytest.approx(14.0, abs=0.12) and math.degrees(s.angle) == pytest.approx(90.0, abs=0.3)


def test_a_rectangular_cutout_with_one_side_without_evidence_stays_put():
    """Een rechthoekige uitsparing waarvan één lange zijde nergens zekere mat naast zich heeft (zwart op zwart): de
    breedte is dan een eigen parameter die de rest van de rand niet vastlegt, dus de uitsparing blijft staan (v0.11,
    sleuf_donker: anders liep die zijde weg). Met bewijs rondom vindt de randfit haar gewoon."""
    from dataclasses import replace

    from camtocad.profile import Slot

    truth = plate(hole=False)
    truth.slots = [Slot(120.0, 80.0, 14.0, 8.0, 0.0, 1.5, "rechthoek")]
    views = scan(truth)
    s = truth.slots[0]
    start = truth.copy()
    start.slots[0] = replace(s, y=s.y + 0.15, width=s.width + 0.3)
    ef = edgefit.fit(start, K, views)
    assert ef.accepted, ef.note
    assert not ef.extra["evidence"][("sleuf", 0)].weak
    assert (ef.part.slots[0].y, ef.part.slots[0].width) == pytest.approx((80.0, 8.0), abs=0.05)
    xs, ys = np.meshgrid(np.linspace(s.x - 8.0, s.x + 8.0, 81), np.linspace(83.0 - 1.5, 83.0 + 1.5, 16))
    bare = without_mat_at(views, truth, np.column_stack([xs.ravel(), ys.ravel()]))
    prob = edgefit._Problem(truth, K, bare)
    prob.set_status(truth)
    e = edgefit.evidence(prob, prob.x_of(truth))[("sleuf", 0)]
    assert e.weak and e.fraction > edgefit.EVIDENCE_MIN  # de rest van de rand heeft bewijs genoeg
    ef = edgefit.fit(start, K, bare)
    assert ef.accepted, ef.note
    assert ef.part.slots[0] == start.slots[0]


def test_edge_distance_from_alpha_is_subpixel_even_where_the_mask_is_off():
    """V2-open: de afstand tot de rand komt uit de zachte objectfractie, lineair over de rand (dus ook tussen twee
    pixels), ook waar het binaire masker een pixel te krap is; zonder alpha de BETA-regel op dat masker."""
    from scipy.special import ndtr

    h, w, edge = 30, 60, 30.3
    x = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w))
    alpha = ndtr((edge - x) / 1.2).astype(np.float16)
    aw = np.full((h, w), 4000.0, np.float16)
    fg, bg = x < 29, x > 31  # masker 1,3 px te krap, strook zonder bewijs tot x = 31,5
    sf, sb = edgefit.signed_dist(fg), edgefit.signed_dist(bg)
    d = edgefit.edge_distance(sf, sf + sb, alpha, aw)
    assert np.interp(edge, x[0], d[15]) == pytest.approx(0.0, abs=0.06)
    assert d[15, 29] == pytest.approx(29 - edge, abs=0.15) and d[15, 31] == pytest.approx(31 - edge, abs=0.15)
    assert np.all(np.diff(d[15]) > 0)  # verder weg (alpha verzadigd) beslist het masker, maar steeds oplopend
    assert edgefit.edge_blur(alpha) == pytest.approx(1.2, abs=0.1)
    beta = edgefit.edge_distance(sf, sf + sb, None, None)
    assert np.interp(edge, x[0], beta[15]) == pytest.approx(1.8 - edgefit.BETA * 1.5, abs=0.01)


def soft_views(truth: Part2p5D, vd: list[silhouette.ViewData], erode_px: int = 1, blur: float = 1.0):
    """Zoals een foto: de zachte bedekking van `truth`, vervaagd, als alpha; het binaire masker `erode_px` te krap
    (een drempel die net verkeerd ligt), zekere mat pas 1 px buiten de ware rand."""
    out = []
    for v in vd:
        s = 5
        Ks = K.copy()
        Ks[:2, :2] *= s
        Ks[:2, 2] = s * K[:2, 2] + 0.5 * (s - 1)
        hh, ww = v.fg.shape
        big = silhouette.ViewData(v.pose, np.zeros((s * hh, s * ww), bool), None, None, s * v.x0, s * v.y0)
        cover = silhouette.render(truth, Ks, big).reshape(hh, s, ww, s).mean(axis=(1, 3)).astype(np.float32)
        true = cover >= 0.5
        k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        fg = cv2.erode(true.astype(np.uint8), k) > 0
        bg = ~(cv2.dilate(true.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        nv = silhouette.ViewData(v.pose, fg, bg, ~(fg | bg), v.x0, v.y0)
        nv.alpha = cv2.GaussianBlur(cover, (0, 0), blur).astype(np.float16)
        nv.alpha_w = np.full((hh, ww), 2000.0, np.float16)
        out.append(nv)
    return out


def test_edge_fit_finds_the_soft_edge_where_the_mask_threshold_is_off(plate_scan, monkeypatch):
    """Het masker ligt overal een pixel binnen de rand: met alpha vindt de randfit toch de ware maten, met alleen
    de BETA-regel (de v0.8-fit) niet."""
    truth = plate()
    vd = soft_views(truth, plate_scan)
    start = plate()
    start.outer.offsets = start.outer.offsets + np.array([0.2, -0.15, 0.1, -0.2])
    start.holes[0] = Hole(110.15, 79.9, 6.2)
    ef = edgefit.fit(start, K, vd)
    assert ef.accepted, ef.note
    assert np.allclose(ef.part.outer.offsets, truth.outer.offsets, atol=0.05)
    assert ef.part.holes[0].d == pytest.approx(6.0, abs=0.06)
    monkeypatch.setattr(edgefit, "EDGE_MODEL", "beta")
    old = edgefit.fit(start, K, vd)
    assert np.mean(truth.outer.offsets - old.part.outer.offsets) > 0.1  # te klein: het masker is te krap


# ----------------------------------------------------------------------------- verzinkingen (V16)

def top_face_gray(part: Part2p5D, v: silhouette.ViewData, true_csk: float, blur: float = 1.0) -> np.ndarray:
    """Grijswaarden zoals een foto: het bovenvlak 150, binnen de verzinking (de kegel) 90, het gat zelf 40, en de
    mat 200; per pixel het snijpunt van de kijkstraal met het bovenvlak (de rand van de verzinking ligt daar)."""
    hh, ww = v.fg.shape
    u, w_ = np.meshgrid(np.arange(ww) + v.x0, np.arange(hh) + v.y0)
    rays = np.linalg.inv(K) @ np.vstack([u.ravel(), w_.ravel(), np.ones(u.size)])
    R, t = v.pose.R, v.pose.t
    d = R.T @ rays
    c = -R.T @ t
    s = (part.height - c[2]) / d[2]
    X, Y = c[0] + s * d[0], c[1] + s * d[1]
    h = part.holes[0]
    r = np.hypot(X - h.x, Y - h.y).reshape(hh, ww)
    g = np.where(v.fg, 150.0, 200.0)
    g = np.where(v.fg & (r < true_csk / 2), 90.0, g)
    g = np.where(r < h.d / 2, 40.0, g)
    return cv2.GaussianBlur(g.astype(np.float32), (0, 0), blur)


def test_a_countersink_is_found_in_the_top_views_and_fitted_on_its_inner_edge(plate_scan):
    """Een ring rond het gat in de grijswaarden wordt een verzinking (Ø 11 x 90°), de randfit zet de maat op die
    rand; zonder ring geen verzinking."""
    from camtocad import countersink

    truth = plate()
    vd = [silhouette.ViewData(v.pose, v.fg, v.bg, v.unk, v.x0, v.y0) for v in plate_scan]
    for v in vd:
        v.gray = top_face_gray(truth, v, 11.0)
    found = countersink.detect(truth, K, vd)
    assert list(found) == [0] and found[0] == pytest.approx(11.0, abs=0.2)
    start = plate()
    start.holes[0] = Hole(110.0, 80.0, 6.0, 10.7)
    ef = edgefit.fit(start, K, vd)
    assert ef.accepted, ef.note
    assert ef.part.holes[0].csk == pytest.approx(11.0, abs=0.05)
    assert ef.part.holes[0].d == pytest.approx(6.0, abs=0.05)
    for v in vd:  # zonder ring: een gewoon gat
        v.gray = top_face_gray(truth, v, 0.0)
    assert countersink.detect(truth, K, vd) == {}


def test_a_small_chamfer_on_a_hole_is_a_narrow_countersink(plate_scan):
    """v0.11: een faas van 0,5 mm (Ø 7 op een gat Ø 6) heeft in het radiale profiel geen eigen piek: haar rand loopt
    in die van het gat over. countersink.narrow meet de rand zoals de randfit: bij een faas ligt hij vast op Ø 7, bij
    een gewoon gat vindt de meting alleen de uitloper van de gatrand zelf (~1 px buiten het gat). En de silhouetten
    moeten de faas steunen: met alleen een ring in de grijswaarden (bijv. een kras) blijft het een gewoon gat."""
    from camtocad import countersink

    truth = plate()
    sunk = plate()
    sunk.holes[0] = Hole(110.0, 80.0, 6.0, csk=7.0)
    vd = scan(sunk)
    for v in vd:
        v.gray = top_face_gray(truth, v, 7.0)
    found = countersink.detect(truth, K, vd)
    assert list(found) == [0] and found[0] == pytest.approx(7.0, abs=0.2)
    plain = [silhouette.ViewData(v.pose, v.fg, v.bg, v.unk, v.x0, v.y0) for v in plate_scan]
    for v in plain:  # dezelfde ring, maar de silhouetten van een gewoon gat
        v.gray = top_face_gray(truth, v, 7.0)
    assert countersink.narrow(truth, K, plain, 0)[0] is None
    for v in plain:  # zonder faas
        v.gray = top_face_gray(truth, v, 0.0)
    assert countersink.narrow(truth, K, plain, 0)[0] is None and countersink.detect(truth, K, plain) == {}


def test_the_inner_edge_of_a_top_chamfer_is_fitted_from_the_top_views(plate_scan):
    """V16 (v0.10): de binnenrand van een afschuining van de bovenrand ligt in de foto's van boven binnen het silhouet,
    maar het schuine vlak is anders belicht dan het bovenvlak. Met die rand komt de maat van de afschuining uit de
    randfit, ook vanuit een startmodel dat 0,25 mm verkeerd staat."""
    from camtocad.profile import TopEdge

    truth = plate(hole=False)
    truth.top_edge = TopEdge("afschuining", 1.0)
    vd = []
    for v in plate_scan:
        nv = silhouette.ViewData(v.pose, exact_mask(truth, v), None, None, v.x0, v.y0)
        nv.bg = ~cv2.dilate(nv.fg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        hh, ww = nv.fg.shape
        u, w_ = np.meshgrid(np.arange(ww) + v.x0, np.arange(hh) + v.y0)
        rays = np.linalg.inv(K) @ np.vstack([u.ravel(), w_.ravel(), np.ones(u.size)])
        d, c = v.pose.R.T @ rays, -v.pose.R.T @ v.pose.t
        s = (truth.height - c[2]) / d[2]
        X, Y = (c[0] + s * d[0]).reshape(hh, ww), (c[1] + s * d[1]).reshape(hh, ww)
        top = truth.outer.inset(1.0)  # het bovenvlak: 1 mm binnen de buitencontour
        inside = np.all([(np.cos(a) * (X - top.center[0]) + np.sin(a) * (Y - top.center[1])) <= off
                         for a, off in zip(top.angles, top.offsets)], axis=0)
        gray = np.where(nv.fg, np.where(inside, 150.0, 90.0), 200.0)
        nv.gray = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
        vd.append(nv)
    start = truth.copy()
    start.top_edge = TopEdge("afschuining", 0.75)
    start.height = truth.height - 0.25 + 0.0  # zelfde schouder (de pixelfit zet die goed)
    ef = edgefit.fit(start, K, vd)
    assert ef.accepted, ef.note
    assert ef.part.top_edge.size == pytest.approx(1.0, abs=0.08)
    assert ef.part.height == pytest.approx(truth.height, abs=0.08)


# ----------------------------------------------------------------------------- kamerboringen (v0.10)

def bore_scan(truth: Part2p5D) -> list[silhouette.ViewData]:
    """Silhouetten van `truth` op ~51° en ~70° boven de mat (8 + 8) en drie van boven: door een gat met een kamer of
    verzinking kijk je in de schuine foto's verder dan door een gewoon gat."""
    poses = []
    for k in range(8):
        az = 2 * np.pi * (k + 0.3) / 8
        poses.append(look_at([120 + 200 * np.cos(az), 80 + 200 * np.sin(az), 250.0], [120.0, 80.0, 0.0]))
        poses.append(look_at([120 + 100 * np.cos(az + 0.4), 80 + 100 * np.sin(az + 0.4), 280.0], [120.0, 80.0, 0.0]))
    for k in range(3):
        c = np.array([112.0 + 8 * k, 76.0 + 6 * (k % 2), 320.0])
        poses.append(look_at(c, [c[0], c[1], 0.0]))
    empty = np.zeros((H, W), bool)
    views = [(Pose(f"v{k}", R, t), ViewMasks(fg=empty, bg=~empty, valid=~empty)) for k, (R, t) in enumerate(poses)]
    vd = silhouette.prepare(views, K, truth)
    for v in vd:
        v.fg = exact_mask(truth, v)
        v.bg = ~v.fg
    return vd


def bore_gray(part: Part2p5D, v: silhouette.ViewData, s: int = 3) -> np.ndarray:
    """Grijswaarden van een plaat met een kamer, per subpixel de kijkstraal door het model: bovenvlak en bodem 150
    (allebei vlak), de wand 90, het gat 40, de zijkant 120 en de mat 200; vervaagd met σ 1 px."""
    hh, ww = v.fg.shape
    u, w_ = np.meshgrid((np.arange(ww * s) + 0.5) / s - 0.5 + v.x0, (np.arange(hh * s) + 0.5) / s - 0.5 + v.y0)
    rays = np.linalg.inv(K) @ np.vstack([u.ravel(), w_.ravel(), np.ones(u.size)])
    d, c = v.pose.R.T @ rays, -v.pose.R.T @ v.pose.t
    h, o = part.holes[0], part.outer

    def hit(z):  # snijpunt met het vlak op hoogte z
        t = (z - c[2]) / d[2]
        return c[0] + t * d[0], c[1] + t * d[1]

    X, Y = hit(part.height)
    Xf, Yf = hit(part.height - h.cb_depth)
    inside = np.all([(np.cos(a) * (X - o.center[0]) + np.sin(a) * (Y - o.center[1])) <= off
                     for a, off in zip(o.angles, o.offsets)], axis=0)
    r_top, r_floor = np.hypot(X - h.x, Y - h.y), np.hypot(Xf - h.x, Yf - h.y)
    fg = np.repeat(np.repeat(v.fg, s, axis=0), s, axis=1).ravel()
    g = np.where(fg, 120.0, 200.0)
    g = np.where(inside & (r_top >= h.cb / 2), 150.0, g)
    in_cb = inside & (r_top < h.cb / 2)
    g = np.where(in_cb, np.where(r_floor > h.cb / 2, 90.0, np.where(r_floor >= h.d / 2, 150.0, 40.0)), g)
    g = g.reshape(hh, s, ww, s).mean(axis=(1, 3))
    return cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)


def test_a_counterbore_is_found_from_the_silhouettes_and_fitted(plate_scan):
    """Een kamer Ø10 x 3 diep in de plaat: door het gat kijk je in de schuine foto's veel verder dan door een gewoon
    gat. counterbore.detect vindt haar uit de silhouetten, de randfit zet diameter en diepte; een gewoon gat krijgt
    geen kamer."""
    from camtocad import counterbore

    truth = plate()
    truth.holes[0] = Hole(110.0, 80.0, 6.0, cb=10.0, cb_depth=3.0)
    vd = bore_scan(truth)
    found = counterbore.detect(plate(), K, vd)
    assert list(found) == [0]
    dk, t, d = found[0]
    assert dk == pytest.approx(10.0, abs=0.5) and t == pytest.approx(3.0, abs=0.5) and d == pytest.approx(6.0, abs=0.15)
    start = plate()
    start.holes[0] = Hole(110.1, 79.9, 6.1, cb=dk + 0.3, cb_depth=t - 0.3)
    ef = edgefit.fit(start, K, vd)
    assert ef.accepted, ef.note
    h = ef.part.holes[0]
    assert h.d == pytest.approx(6.0, abs=0.05) and (h.x, h.y) == pytest.approx((110.0, 80.0), abs=0.05)
    assert h.cb == pytest.approx(10.0, abs=0.1) and h.cb_depth == pytest.approx(3.0, abs=0.15)
    assert counterbore.detect(plate(), K, plate_scan) == {}


def test_a_countersink_seen_only_in_the_silhouettes_is_not_a_counterbore():
    """v0.11: ook door een verzonken gat kijk je in de schuine foto's verder. Zonder ring in de grijswaarden werd een
    verzinking in v0.10 een ondiepe kamer; nu wint de verzinking (een maat minder), en gaat ze terug naar de pijplijn."""
    from camtocad import counterbore

    truth = plate()
    truth.holes[0] = Hole(110.0, 80.0, 6.0, csk=11.0)
    sunk = {}
    assert counterbore.detect(plate(), K, bore_scan(truth), sunk=sunk) == {}
    assert list(sunk) == [0] and sunk[0] == pytest.approx((11.0, 6.0), abs=0.2)


def test_the_floor_edge_of_a_counterbore_sets_its_depth():
    """v0.11: de onderrand van de wand van een kamer (waar ze de bodem raakt) is in de schuine foto's een rand in de
    grijswaarden, als de camera door de kamer heen de bodem ziet. Daarmee komt de diepte uit de randfit, ook vanuit
    een start die 0,6 mm te ondiep is; zonder grijswaarden alleen uit het silhouet."""
    truth = plate()
    truth.holes[0] = Hole(110.0, 80.0, 6.0, cb=10.0, cb_depth=3.0)
    vd = bore_scan(truth)
    for v in vd:
        v.gray = bore_gray(truth, v)
    lay = edgefit.layout(truth)
    n_rim = lay["cb"][0]
    meas = edgefit.measure_inner(truth, K, vd, lay)
    assert sum(int(np.sum(m[0] >= n_rim)) for m in meas) > 100  # de bodemrand is in de meeste foto's gemeten
    start = plate()
    start.holes[0] = Hole(110.1, 79.9, 6.1, cb=10.3, cb_depth=2.4)
    ef = edgefit.fit(start, K, vd)
    assert ef.accepted, ef.note
    h = ef.part.holes[0]
    assert h.cb_depth == pytest.approx(3.0, abs=0.05) and h.cb == pytest.approx(10.0, abs=0.05)
