"""Validatie (Fase 0): schuifmaatmetingen koppelen aan het model en de fouten samenvatten."""

import json

import numpy as np
import pytest

from camtocad import validate
from camtocad.profile import Hole, Part2p5D, Profile


def bracket(dx: float = 0.0, d: float = 6.6) -> Part2p5D:
    """Rechthoek 80 x 40 (plus dx) met R3-hoeken en twee gaten, in het werkassenstelsel."""
    w, h = 80.0 + dx, 40.0
    prof = Profile("polygon", np.array([w / 2, h / 2]), np.array([-np.pi / 2, 0, np.pi / 2, np.pi]),
                   np.array([h / 2, w / 2, h / 2, w / 2]), np.full(4, 3.0))
    return Part2p5D(12.02, prof, [Hole(10.0, 20.0, d), Hole(70.0 + dx, 20.0, d + 0.02)])


UNC = {"edge": 0.05, "height": 0.05, "hole_d": 0.04, "hole_xy": 0.04, "fillet": 0.2, "scale_rel": 5e-4}


def write_part(folder, fitted: Part2p5D, snapped: Part2p5D, reference: dict):
    (folder / "fotos").mkdir(parents=True)
    (folder / "fotos" / "a.jpg").write_bytes(b"x")
    (folder / "maten.json").write_text(json.dumps(reference), encoding="utf-8")
    (folder / "resultaat").mkdir()
    report = {"summary": {"betrouwbaarheid": "normaal"}, "uncertainty_model": UNC,
              "geometry": {"gefit": fitted.to_dict(), "gesnapt": snapped.to_dict()}}
    (folder / "resultaat" / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_reference_accepts_singular_plural_and_lists(tmp_path):
    p = tmp_path / "maten.json"
    p.write_text(json.dumps({"maten": {"Lengte": 80.02, "gat": 6.6, "gaten": [6.61], "hartafstanden": [60]},
                             "meetlijn": 99.8}), encoding="utf-8")
    ref = validate.read_reference(p)
    assert ref["maten"] == {"lengte": [80.02], "gat": [6.6, 6.61], "hartafstand": [60.0]}
    assert ref["meetlijn"] == [99.8, 99.8] and ref["mat"] == "auto"
    p.write_text(json.dumps({"maten": {"lengte": 80, "dikte": 3}}), encoding="utf-8")
    with pytest.raises(ValueError, match="dikte"):
        validate.read_reference(p)


def test_model_measures_match_the_caliper_view():
    m = validate.model_measures(bracket().to_dict(), UNC)
    assert m["lengte"][0][0] == pytest.approx(80.0, abs=1e-6)  # afrondingen verkorten de buitenmaat niet
    assert m["breedte"][0][0] == pytest.approx(40.0, abs=1e-6)
    assert sorted(v for v, _ in m["gat"]) == pytest.approx([6.6, 6.62])
    assert m["hartafstand"][0][0] == pytest.approx(60.0)
    assert [v for v, _ in m["afronding"]] == pytest.approx([3.0] * 4)
    assert m["lengte"][0][1] == pytest.approx(2 * np.hypot(0.05 * np.sqrt(2), 5e-4 * 80))


def test_validation_of_a_folder_with_cached_results(tmp_path):
    ref = {"naam": "beugel", "maten": {"lengte": 80.1, "breedte": 40.0, "hoogte": 12.0, "gaten": [6.6, 6.6, 5.0],
                                       "hartafstanden": [60.0], "afrondingen": [3.0], "diameter": 20.0}}
    write_part(tmp_path / "beugel", bracket(), bracket(d=6.6), ref)
    ref2 = {"naam": "plaat", "maten": {"lengte": 79.6, "breedte": 40.0, "hoogte": 12.0}}
    write_part(tmp_path / "plaat", bracket(dx=0.3), bracket(dx=0.3), ref2)
    (tmp_path / "leeg").mkdir()  # zonder maten.json: geen onderdeel

    results, summary = validate.validate(tmp_path, run=lambda *a, **k: pytest.fail("niet opnieuw verwerken"),
                                         log=lambda m: None)
    assert [r.naam for r in results] == ["beugel", "plaat"]
    rows = {(c.soort, c.referentie): c for c in results[0].vergelijkingen}
    assert rows[("lengte", 80.1)].fout == pytest.approx(-0.1) and rows[("lengte", 80.1)].binnen_u95 is True
    assert rows[("gat", 5.0)].gefit is None and "ontbreekt" in rows[("gat", 5.0)].opmerking
    assert rows[("diameter", 20.0)].opmerking == "model is niet rond"
    assert rows[("hoogte", 12.0)].fout == pytest.approx(0.02)
    plaat = {c.soort: c for c in results[1].vergelijkingen}
    assert plaat["lengte"].fout == pytest.approx(0.7) and plaat["lengte"].binnen_u95 is False
    assert summary["scans_ok"] == 2 and summary["niet_gevonden"] == 2
    assert summary["per_soort"]["lengte"]["n"] == 2
    lines = validate.report_lines(results, summary)
    assert any("BUITEN U95" in line for line in lines)
    paths = validate.write_report(results, summary, tmp_path)
    assert "beugel" in paths["html"].read_text(encoding="utf-8")
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["samenvatting"]["maten"] == summary["maten"]


def test_systematic_length_error_points_at_the_print_scale(tmp_path):
    """Alle lengtes 0,4% te groot: de samenvatting wijst op de meetlijnen (printschaal)."""
    for i, (w, s) in enumerate([(80.0, 1.004), (50.0, 1.004), (120.0, 1.004)]):
        fitted = bracket(dx=w * s - 80.0)
        ref = {"maten": {"lengte": w, "breedte": 40.0}}
        write_part(tmp_path / f"p{i}", fitted, fitted, ref)
    _, summary = validate.validate(tmp_path, run=lambda *a, **k: None, log=lambda m: None)
    assert summary["schaal_bias_pct"] == pytest.approx(0.2, abs=0.01)  # lengtes +0,4%, breedtes kloppen
    assert any("printschaal" in n for n in summary["opmerkingen"])
