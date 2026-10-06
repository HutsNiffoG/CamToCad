"""Het stresstest-harnas (V26, v0.12): scenario's, de waarheid per maat, het vergelijken van twee runs, en één kleine
stresstest van begin tot eind."""

import json
import math

import pytest

from camtocad import stresstest


def test_every_scenario_has_an_object_with_a_truth():
    for name, cfg in stresstest.SCENARIOS.items():
        make, truth = stresstest.OBJECTS[cfg.get("object", "bracket")]
        assert callable(make) and "h" in truth, name
        assert stresstest.describe(name).startswith(cfg.get("object", "bracket"))
    assert set(stresstest.GRAY_CONTROLS) <= set(stresstest.SCENARIOS)


def test_the_truth_of_a_dimension_from_the_report():
    t = stresstest.SLOTTED
    assert stresstest.truth_of("hoogte", 12.01, t) == 12.0
    assert stresstest.truth_of("x-maat rand 2", 79.98, t) == 80.0
    assert stresstest.truth_of("x-maat rand 2", 77.0, t) is None  # meer dan 2 mm ernaast: een andere maat
    assert stresstest.truth_of("gat 1 x", 10.05, t) == 10.0 and stresstest.truth_of("gat 1 x", 69.9, t) == 70.0
    assert stresstest.truth_of("gat Ø (2x)", 6.61, t) == 6.6
    assert stresstest.truth_of("uitsparing 1 x", 69.98, t) == 70.0
    assert stresstest.truth_of("sleuf 1 hartafstand", 15.97, t) == 16.0
    assert stresstest.truth_of("afronding R (4x)", 3.1, t) == 3.0
    # een verzinking, kamer of afschuining die er niet is: de fout is de hele maat
    assert stresstest.truth_of("verzinking Ø (gat 1)", 12.3, t) == 0.0
    assert stresstest.truth_of("afschuining bovenrand", 0.4, t) == 0.0
    csk = stresstest.OBJECTS["verzonken"][1]
    assert stresstest.truth_of("verzinking Ø (2x)", 12.38, csk) == 12.4
    assert stresstest.truth_of("gat 1 x", 70.0, stresstest.OBJECTS["afrondbeugel"][1]) is None
    assert [stresstest.kind_of(n) for n in ("hoogte", "y-maat rand 3", "gat Ø (2x)", "gat 2 y", "uitsparing 1 x",
                                            "uitsparing 1 hoekstraal", "sleuf 1 breedte", "trede 1 hoogte",
                                            "kamerboring diepte (2x)", "blind gat Ø (gat 2)")] == [
        "hoogte", "lengte", "gat", "positie", "positie", "afronding", "sleuf", "trede", "kamerboring", "blind gat"]


def _fake_run(root, measured: dict, snapped: dict, warnings=()):
    """Een run met één scenario ('basis') zoals stresstest.run hem achterlaat: stress.json en report.json."""
    d = root / "basis"
    (d / "resultaat").mkdir(parents=True)
    (d / "stress.json").write_text(json.dumps({"scenario": "basis", "status": "OK", "time": 1.0,
                                               "truth": {"h": 12.0, "x": 80.0, "y": 40.0}}), encoding="utf-8")
    scale = 0.003
    dims = []
    for name, (value, u95) in measured.items():
        sigma = math.hypot(u95 / 2, scale * value)  # in het rapport zit de printschaal in sigma
        nominal = snapped.get(name)
        dims.append({"name": name, "measured": value, "value": nominal if nominal is not None else value,
                     "snapped": nominal is not None, "sigma": sigma, "u95": 2 * sigma})
    report = {"warnings": list(warnings), "dimensions": dims,
              "uncertainty_model": {"scale_rel": scale, "methode": "test"}}
    (d / "resultaat" / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_two_runs_are_compared_against_the_truth(tmp_path):
    """De dekking telt zonder de printschaal (een gerenderde mat is exact); een snap naar de verkeerde waarde telt
    als fout; een run met een waarschuwing 'onbetrouwbaar' telt apart."""
    _fake_run(tmp_path / "oud", {"hoogte": (12.03, 0.05), "x-maat rand 1": (80.09, 0.06)}, {"hoogte": 12.0})
    _fake_run(tmp_path / "nieuw", {"hoogte": (12.01, 0.05), "x-maat rand 1": (80.02, 0.06),
                                   "gat 1 x": (70.3, 0.08)}, {"hoogte": 12.0, "gat 1 x": 70.5})
    lines = []
    s = stresstest.compare(tmp_path / "oud", tmp_path / "nieuw", log=lines.append)
    assert s["alle"]["n"] == 3 and s["alle"]["inside"] == 2  # het gat is 0,3 mm ernaast bij een U95 van 0,08
    assert s["alle"]["snapped"] == 2 and s["alle"]["wrong_snaps"] == 1
    assert s["zonder waarschuwing"]["n"] == 3
    assert any("buiten U95" in line for line in lines)
    rows = {r["name"]: r for r in s["rows"]}
    assert rows["x-maat rand 1"]["err"] == pytest.approx(0.02) and rows["x-maat rand 1"]["u95"] == pytest.approx(0.06)
    assert rows["x-maat rand 1"]["err_old"] == pytest.approx(0.09)
    _fake_run(tmp_path / "gemeld", {"hoogte": (12.3, 0.05)}, {}, warnings=["onbetrouwbaar: test"])
    s = stresstest.compare(tmp_path / "oud", tmp_path / "gemeld", log=lambda m: None)
    assert s["alle"]["n"] == 1 and "zonder waarschuwing" not in s


@pytest.mark.slow
@pytest.mark.stress
def test_a_small_stress_scan_lands_within_its_u95(tmp_path):
    """Eén stresstest van begin tot eind, kleiner gemaakt (20 foto's in plaats van 46): ruis, JPEG, vignettering en
    verscherping; alle maten binnen hun U95, buitenmaten en gaten binnen 0,1 mm."""
    cfg = dict(stresstest.BASE, rings=((40.0, 8), (65.0, 8)), top_views=4)
    summary = stresstest.run("realistisch", tmp_path / "run", cfg=cfg, log=lambda m: None)
    assert summary["status"] == "OK", summary["log"][-3:]
    s = stresstest.compare(tmp_path / "geen", tmp_path / "run", log=lambda m: None)
    assert s["alle"]["inside"] == s["alle"]["n"] >= 9 and s["alle"]["wrong_snaps"] == 0
    for r in s["rows"]:
        if stresstest.kind_of(r["name"]) in ("lengte", "gat", "hoogte"):
            assert abs(r["err"]) < 0.1, r
