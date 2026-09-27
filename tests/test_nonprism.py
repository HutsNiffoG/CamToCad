"""V17: een afgeschuinde of afgeronde bovenrand rondom en een rechte trede als model."""

import math
import runpy

import numpy as np
import pytest

from camtocad import cadmodel, edgefit, pipeline, prismcheck, report, silhouette, validate
from camtocad.profile import Hole, Part2p5D, Profile, Step, TopEdge

from test_edgefit import K
from test_prismcheck import block, scan


def l_profile() -> Profile:
    """L-vorm 40 x 30 met een hap van 20 x 15 rechtsboven; de holle hoek is scherp."""
    V = np.array([(0, 0), (40, 0), (40, 15), (20, 15), (20, 30), (0, 30)], float)
    d = np.roll(V, -1, axis=0) - V
    nrm = np.column_stack([d[:, 1], -d[:, 0]]) / np.linalg.norm(d, axis=1)[:, None]
    c = np.array([15.0, 12.0])
    return Profile("polygon", c, np.arctan2(nrm[:, 1], nrm[:, 0]), np.einsum("ij,ij->i", nrm, V - c),
                   np.array([2.0, 2.0, 2.0, 0.0, 2.0, 2.0]))


# ----------------------------------------------------------------------------- geometrie

def test_inset_moves_edges_in_and_changes_the_fillets():
    p = l_profile()
    q = p.inset(1.5)
    assert np.allclose(q.offsets, p.offsets - 1.5)
    assert list(p.convex_corners()) == [True, True, True, False, True, True]
    assert np.allclose(q.fillets, [0.5, 0.5, 0.5, 0.0, 0.5, 0.5])  # bol: kleiner; scherp hol: blijft scherp
    assert np.allclose(p.inset(3.0).fillets[[0, 1, 2, 4, 5]], 0.0)  # kleiner dan nul wordt scherp


def test_a_step_splits_the_part_in_cells():
    p = Part2p5D(8.0, l_profile(), steps=[Step.from_line(0.0, 30.0, 3.0, (30.0, 7.0))])
    assert p.is_valid()
    (main, h_main), (low, h_low) = p.cells()
    assert (h_main, h_low) == (8.0, 3.0)
    assert main.area() + low.area() == pytest.approx(p.outer.area(), rel=1e-6)
    assert low.area() == pytest.approx(10 * 15 - (4 - math.pi) * 4 / 2, rel=0.01)  # 10 x 15 met twee R2-hoeken
    assert p.height_at(35.0, 5.0) == 3.0 and p.height_at(10.0, 20.0) == 8.0
    q = p.step_crossings(p.steps[0])
    assert np.allclose(sorted(q[:, 1]), [0.0, 15.0], atol=1e-6) and np.allclose(q[:, 0], 30.0)
    # een lijn die de contour vier keer snijdt (x + y = 38: door beide benen van de L) is geen rechte trede
    diagonal = Step.from_line(math.pi / 4, 38.0 / math.sqrt(2.0), 3.0, (20.0, 18.0))
    assert not Part2p5D(8.0, l_profile(), steps=[diagonal]).is_valid()


def test_step_turns_about_its_pivot_and_moves_with_the_part():
    st = Step.from_line(0.0, 60.0, 4.0, (60.0, 20.0))
    assert st.offset == pytest.approx(60.0) and st.pivot == (60.0, 20.0)
    turned = Step(st.angle + math.radians(1.0), st.dist, st.height, st.pivot)
    assert abs(turned.normal() @ [60.0, 20.0] - turned.offset) < 1e-9  # het draaipunt blijft op de lijn
    p = Part2p5D(8.0, Profile("polygon", np.array([40.0, 20.0]), np.array([-np.pi / 2, 0, np.pi / 2, np.pi]),
                              np.array([20.0, 40.0, 20.0, 40.0]), np.zeros(4)), steps=[st])
    angle, shift = 0.4, np.array([5.0, -3.0])
    q = p.transformed(angle, shift)
    R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    for pt in ([70.0, 5.0], [50.0, 35.0]):
        assert q.height_at(*(R @ pt + shift)) == p.height_at(*pt)
    assert [c.area() for c, _ in q.cells()] == pytest.approx([c.area() for c, _ in p.cells()])


def test_top_edge_levels():
    assert TopEdge("afschuining", 2.0).levels(10.0) == [(0.0, 0.0), (8.0, 0.0), (10.0, 2.0)]
    lv = TopEdge("afronding", 2.0).levels(10.0)
    assert lv[1] == (8.0, 0.0) and lv[-1] == pytest.approx((10.0, 2.0))
    assert all(math.hypot(z - 8.0, d - 2.0) == pytest.approx(2.0) for z, d in lv[1:])  # op de boog
    p = block(95, 145, 65, 95, 8.0, 2.0)
    p.top_edge = TopEdge("afschuining", 7.0)  # groter dan 0,8 x de hoogte
    assert not p.is_valid()


# ----------------------------------------------------------------------------- silhouet

def test_render_of_a_chamfer_matches_thin_layers():
    """Het silhouet van een afgeschuinde plaat is (op een halve pixel langs de rand na) dat van een stapel dunne
    prisma's met dezelfde doorsnede."""
    part = block(95, 145, 65, 95, 8.0, 2.0)
    part.top_edge = TopEdge("afschuining", 2.0)
    layers = [block(95 + d, 145 - d, 65 + d, 95 - d, 6.0 + d, max(2.0 - d, 0.0)) for d in np.linspace(0, 2, 41)]
    vd = scan([part], elevations=(35.0, 72.0), n_top=1)
    for v in vd:
        a = silhouette.render(part, K, v).astype(bool)
        b = np.logical_or.reduce([silhouette.render(p, K, v).astype(bool) for p in layers])
        assert np.count_nonzero(a ^ b) < 0.02 * np.count_nonzero(b)


def test_rounded_top_edge_is_not_drawn_too_wide():
    """De banden van een afronding zijn in beeld vaak smaller dan een pixel. De snelle renderer tekent de wanden
    daarom vanaf de onderrand: dan past de waarheid even goed als bij een prisma (eerder 2x zoveel pixels fout)."""
    energies = []
    for kind in (None, "afronding"):
        truth = block(95, 145, 65, 95, 8.0, 2.0)
        truth.top_edge = TopEdge(kind, 2.0) if kind else None
        energies.append(silhouette.energy(truth, K, scan([truth], elevations=(35.0, 55.0, 72.0))))
    assert energies[1] < 1.2 * energies[0]


# ----------------------------------------------------------------------------- fits

@pytest.fixture(scope="module")
def chamfer_scan():
    truth = block(95, 145, 65, 95, 8.0, 2.0)
    truth.top_edge = TopEdge("afschuining", 2.0)
    return scan([truth])


def _prism_fit(vd, h0):
    start = block(95.3, 144.8, 65.2, 95.3, h0, 2.0)
    part, energy, _ = silhouette.refine(start, K, vd, max_evals=800)
    return part, energy, edgefit.fit(part, K, vd, mm_per_px=0.23)


def test_chamfer_all_round_is_modelled(chamfer_scan):
    """Het prisma komt op de schouder uit (~6 mm); met de afschuining als model worden hoogte en maat goed en
    verdwijnt de melding van de vormtoets."""
    vd = chamfer_scan
    part, energy, ef = _prism_fit(vd, 7.0)
    shape = pipeline._prism_check(ef, part)
    assert shape.trend < prismcheck.TREND_LOW_MM and part.height < 6.6
    alt = pipeline._non_prism(part, K, vd, energy, shape, mm_per_px=0.23, log=lambda m: None)
    assert alt is not None and alt[0].top_edge.kind == "afschuining"
    ef2 = edgefit.fit(alt[0], K, vd, mm_per_px=0.23)
    assert ef2.accepted
    fitted = ef2.part
    assert fitted.height == pytest.approx(8.0, abs=0.12) and fitted.top_edge.size == pytest.approx(2.0, abs=0.15)
    assert pipeline._prism_check(ef2, fitted).issues == []


def test_a_prism_gets_no_chamfer():
    """Een prisma: geen afwijkend stuk bovenrand, en een afschuining levert te weinig op (hier ~3%, door de
    renderer, V29; een afschuining van 1 mm gaf 10%) en is kleiner dan 2 px."""
    vd = scan([block(95, 145, 65, 95, 8.0, 2.0)])
    part, energy, ef = _prism_fit(vd, 7.5)
    shape = pipeline._prism_check(ef, part)
    assert not shape.runs_mat and abs(shape.trend) < 0.03
    info = {}
    assert pipeline._non_prism(part, K, vd, energy, shape, 0.23, log=lambda m: None, info=info) is None
    cand, e = silhouette.fit_top_edge(part, K, vd)
    assert e > energy * (1 - pipeline.NON_PRISM_GAIN) and cand.top_edge.size < 2 * 0.23


def test_a_step_is_modelled():
    """De rechter 15 mm van de plaat is maar 3 mm hoog: de vormtoets vindt het lage stuk, de trede wordt het
    model, en de randfit legt de lijn en beide hoogtes vast."""
    vd = scan([block(95, 145, 65, 95, 3.0), block(95, 130, 65, 95, 6.0)])
    part, energy, ef = _prism_fit(vd, 5.5)
    shape = pipeline._prism_check(ef, part)
    assert [r["soort"] for r in shape.runs_mat] == ["lager"]
    alt = pipeline._non_prism(part, K, vd, energy, shape, 0.23, log=lambda m: None)
    assert alt is not None and len(alt[0].steps) == 1 and alt[1] < 0.3 * energy
    ef2 = edgefit.fit(alt[0], K, vd, mm_per_px=0.23)
    assert ef2.accepted
    p = ef2.part
    st = p.steps[0]
    assert st.offset == pytest.approx(130.0, abs=0.2) and abs(math.degrees(st.angle)) < 0.5
    assert st.height == pytest.approx(3.0, abs=0.1) and p.height == pytest.approx(6.0, abs=0.1)
    check = pipeline._prism_check(ef2, p)
    assert check.issues == [] and check.details["stukken"] == []
    assert edgefit.jackknife(ef2) is not None


def test_a_small_high_part_is_a_step_too():
    """Alleen de rechter 15 mm is 6 mm hoog, de rest 3 mm. Het prisma komt dan tussenin uit, en de vormtoets vindt
    het lage deel rond drie zijden: als één stuk, of als meer stukken als een deel van de linkerzijde bij dit
    compromis net niet afwijkt (op de ene machine een gat van 9 mm, op een andere van 11). Samen geven ze de start
    van de trede: van boven tot onder langs x = 130, en de trede ligt links."""
    vd = scan([block(95, 145, 65, 95, 3.0), block(130, 145, 65, 95, 6.0)])
    part, energy, ef = _prism_fit(vd, 4.0)
    shape = pipeline._prism_check(ef, part)
    assert shape.runs_mat and all(r["soort"] == "lager" for r in shape.runs_mat)
    start = shape.start
    assert start["soort"] == "lager" and abs(start["van"][0] - 130) < 2 and abs(start["tot"][0] - 130) < 2
    assert abs(start["van"][1] - start["tot"][1]) > 25
    alt = pipeline._non_prism(part, K, vd, energy, shape, 0.23, log=lambda m: None)
    assert alt is not None
    ef2 = edgefit.fit(alt[0], K, vd, mm_per_px=0.23)
    st = ef2.part.steps[0]
    assert st.normal() @ [-1.0, 0.0] > 0.9999 and -st.offset == pytest.approx(130.0, abs=0.2)
    assert st.height == pytest.approx(3.0, abs=0.1) and ef2.part.height == pytest.approx(6.0, abs=0.1)


# ----------------------------------------------------------------------------- CAD, rapport, validatie

@pytest.mark.parametrize("kind, size, corner", [("afschuining", 2.0, 1.0), ("afschuining", 1.5, 3.0),
                                                ("afronding", 2.0, 1.0)])
def test_top_edge_builds_and_the_script_matches(tmp_path, kind, size, corner):
    """Ook een afschuining die groter is dan de hoekafronding (daar faalt een gewone CAD-afschuining)."""
    prof = Profile("polygon", np.array([40.0, 20.0]), np.array([-np.pi / 2, 0, np.pi / 2, np.pi]),
                   np.array([20.0, 40.0, 20.0, 40.0]), np.full(4, corner))
    part = Part2p5D(12.0, prof, [Hole(10.0, 20.0, 6.6)], top_edge=TopEdge(kind, size))
    snapped, snaps = cadmodel.snap_part(cadmodel.to_part_frame(part)[0], cadmodel.Uncertainty(0.03, 0.03, 0.03,
                                                                                            0.03, 0.1))
    assert snapped.top_edge == TopEdge(kind, size)
    assert any(s.name == cadmodel.top_edge_name(part.top_edge) and s.value == size for s in snaps)
    model = cadmodel.build(snapped)
    assert model.val().isValid()
    # volume: de doorsnede per hoogte (de contour steeds verder naar binnen), min het gat
    zs = np.linspace(12.0 - size, 12.0, 201)
    inset = zs - (12.0 - size) if kind == "afschuining" else size - np.sqrt(size ** 2 - (zs - 12.0 + size) ** 2)
    areas = [prof.inset(d).area() for d in inset]
    volume = prof.area() * (12.0 - size) + np.trapezoid(areas, zs) - math.pi * 3.3 ** 2 * 12.0
    assert model.val().Volume() == pytest.approx(volume, rel=2e-3)
    path = tmp_path / "model.py"
    path.write_text(cadmodel.script(snapped, snaps).replace('if __name__ == "__main__":', "if False:"))
    assert runpy.run_path(str(path))["model"].val().Volume() == pytest.approx(model.val().Volume(), rel=1e-9)


def test_step_is_dimensioned_from_the_datum_and_built(tmp_path):
    prof = Profile("polygon", np.array([40.0, 20.0]), np.array([-np.pi / 2, 0, np.pi / 2, np.pi]),
                   np.array([20.0, 40.0, 20.0, 40.0]), np.zeros(4))
    part = Part2p5D(12.03, prof, [Hole(70.0, 20.0, 6.6)], steps=[Step.from_line(0.004, 60.04, 5.98, (60.0, 20.0))])
    part = part.transformed(0.3, np.array([100.0, 60.0]))  # zoals op de mat
    pf, _, _ = cadmodel.to_part_frame(part)
    snapped, snaps = cadmodel.snap_part(pf, cadmodel.Uncertainty(0.03, 0.03, 0.03, 0.03, 0.1))
    by = {s.name: s for s in snaps}
    assert by["trede 1 positie"].measured == pytest.approx(60.04, abs=0.01) and by["trede 1 positie"].value == 60.0
    assert by["trede 1 hoogte"].value == 6.0 and by["hoogte"].value == 12.0
    st = snapped.steps[0]
    assert st.offset == pytest.approx(60.0) and st.angle == pytest.approx(0.0, abs=1e-12)
    model = cadmodel.build(snapped)
    volume = 80 * 40 * 12 - 20 * 40 * 6 - math.pi * 3.3 ** 2 * 6
    assert model.val().Volume() == pytest.approx(volume, rel=1e-6)
    path = tmp_path / "model.py"
    code = cadmodel.script(snapped, snaps)
    assert "trede1_x = 60.0" in code
    path.write_text(code.replace('if __name__ == "__main__":', "if False:"))
    assert runpy.run_path(str(path))["model"].val().Volume() == pytest.approx(volume, rel=1e-6)
    geo = snapped.to_dict()
    assert geo["treden"][0]["hoogte"] == 6.0 and geo["bovenrand"] is None


def test_report_draws_the_top_edge_and_the_step():
    prof = Profile("polygon", np.array([40.0, 20.0]), np.array([-np.pi / 2, 0, np.pi / 2, np.pi]),
                   np.array([20.0, 40.0, 20.0, 40.0]), np.full(4, 3.0))
    chamfered = Part2p5D(12.0, prof, top_edge=TopEdge("afschuining", 1.5))
    stepped = Part2p5D(12.0, prof, steps=[Step.from_line(0.0, 60.0, 6.0, (60.0, 20.0))])
    assert report._svg_top_view(chamfered).count('class="edge"') == 1  # binnenrand van de afschuining
    svg = report._svg_top_view(stepped)
    assert svg.count('class="edge"') == 1 and "trede 6.00" in svg


def test_validation_compares_top_edge_and_step():
    geo = Part2p5D(12.0, Profile("polygon", np.array([40.0, 20.0]), np.array([-np.pi / 2, 0, np.pi / 2, np.pi]),
                                 np.array([20.0, 40.0, 20.0, 40.0]), np.zeros(4)),
                   top_edge=TopEdge("afschuining", 1.9)).to_dict()
    geo["contour"] = {"soort": "polygoon", "hoekpunten": [[0, 0, 0], [80, 0, 0], [80, 40, 0], [0, 40, 0]]}
    geo["treden"] = [{"hoogte": 5.9}]
    unc = {"edge": 0.03, "height": 0.03, "hole_d": 0.03, "hole_xy": 0.03, "fillet": 0.1, "scale_rel": 5e-4}
    report = {"geometry": {"gefit": geo, "gesnapt": geo}, "uncertainty_model": unc}
    ref = {"maten": {"bovenrand": [2.0], "trede": [6.0]}}
    rows, _ = validate.compare(ref, report)
    got = {r.soort: r for r in rows}
    assert got["bovenrand"].fout == pytest.approx(-0.1) and got["bovenrand"].binnen_u95
    assert got["trede"].fout == pytest.approx(-0.1)
