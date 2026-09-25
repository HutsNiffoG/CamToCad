"""End-to-end: synthetische foto's van een beugel op de mat → CAD-model met de juiste maten."""

import runpy

import numpy as np
import pytest

from camtocad import mat, pipeline, render


@pytest.mark.slow
def test_bracket_scan_end_to_end(tmp_path):
    spec = mat.PRESETS["A4"]
    placed = render.place(pipeline.demo_part(), spec, angle_deg=17.0, offset=(5, -8))
    views = render.render_scan(placed, spec, render.default_camera(), seed=5)
    result = pipeline.run_scan([(v.name, v.image) for v in views], tmp_path, pipeline.ScanOptions(),
                               log=lambda m: None, scan_name="test")
    part = result["part"]
    fitted = result["part_fitted"]

    # gefitte (ongesnapte) maten liggen dicht bij de waarheid
    assert abs(fitted.height - 12.0) < 0.15
    V = fitted.outer.vertices()
    size = V.max(axis=0) - V.min(axis=0)
    assert np.allclose(sorted(size), [40.0, 80.0], atol=0.2)
    assert len(fitted.holes) == 2
    for h in fitted.holes:
        assert abs(h.d - 6.6) < 0.2
    assert np.allclose(sorted([(round(h.x), round(h.y)) for h in fitted.holes]), [(10, 20), (70, 20)])

    # gesnapt model: hoofdmaten exact
    assert part.height == 12.0
    Vs = part.outer.vertices()
    assert np.allclose(Vs.max(axis=0) - Vs.min(axis=0), [80.0, 40.0])
    assert np.allclose(part.outer.fillets, 3.0)

    # uitvoer: geldige STEP, en het script bouwt hetzelfde model
    assert (tmp_path / "model.step").stat().st_size > 10_000
    assert (tmp_path / "report.html").exists()
    ns = runpy.run_path(str(tmp_path / "model.py"), run_name="test")
    assert ns["model"].val().isValid()
