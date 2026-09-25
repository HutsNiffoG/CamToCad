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
    images = [(v.name, v.image) for v in views]
    # twee foto's zoals sommige apps ze opslaan: pixels gedraaid (staand) in plaats van een EXIF-vlag
    images[3] = (images[3][0], np.rot90(images[3][1]).copy())
    images[20] = (images[20][0], np.rot90(images[20][1], -1).copy())
    result = pipeline.run_scan(images, tmp_path, pipeline.ScanOptions(), log=lambda m: None, scan_name="test")
    assert result["summary"]["foto's gebruikt"].startswith(f"{len(views)} van {len(views)}")
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

    # gesnapt model: hoofdmaten en gaten exact; de vier (gelijke) afrondingen als één groep. Een
    # afronding bepaalt maar een klein stukje silhouet: binnen ~0,3 mm, snappen alleen als het zeker is.
    assert part.height == 12.0
    Vs = part.outer.vertices()
    assert np.allclose(Vs.max(axis=0) - Vs.min(axis=0), [80.0, 40.0])
    assert np.allclose(part.outer.fillets, part.outer.fillets[0])
    assert abs(part.outer.fillets[0] - 3.0) < 0.35
    # gaten: gelijk gegroepeerd; 6,6 (ISO 273 M6) of, als 6,5 even aannemelijk is, ongesnapt dichtbij
    assert part.holes[0].d == part.holes[1].d and abs(part.holes[0].d - 6.6) < 0.1, [h.d for h in part.holes]
    assert result["summary"]["betrouwbaarheid"] == "normaal"

    # uitvoer: geldige STEP, en het script bouwt hetzelfde model
    assert (tmp_path / "model.step").stat().st_size > 10_000
    assert (tmp_path / "report.html").exists()
    ns = runpy.run_path(str(tmp_path / "model.py"), run_name="test")
    assert ns["model"].val().isValid()


def test_wrong_mat_size_is_named(tmp_path):
    views = render.render_scan(None, mat.PRESETS["A3"], render.default_camera(), rings=((50.0, 6),), top_views=0,
                               distance=450.0, seed=2)
    with pytest.raises(pipeline.ScanError, match="A3-mat"):
        pipeline.run_scan([(v.name, v.image) for v in views], tmp_path, pipeline.ScanOptions(mat="A4"),
                          log=lambda m: None)


def test_scaled_part_scales_every_dimension():
    from camtocad.profile import Hole, Part2p5D, Profile

    prof = Profile("polygon", np.array([10.0, 5.0]), np.array([-np.pi / 2, 0, np.pi / 2, np.pi]),
                   np.array([5.0, 10.0, 5.0, 10.0]), np.full(4, 1.0))
    part = Part2p5D(4.0, prof, [Hole(12.0, 5.0, 3.0)])
    big = part.scaled(1.03)
    assert big.height == pytest.approx(4.12) and big.holes[0].d == pytest.approx(3.09)
    assert big.outer.area() == pytest.approx(part.outer.area() * 1.03 ** 2, rel=1e-3)
