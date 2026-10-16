"""Vlakken, cilinders en bollen in de vrije vorm (v0.18): herkennen, maten, referentiegeometrie."""

import numpy as np
import pytest

from camtocad import freeform, primitives


def sdf_mesh(fn, lo, hi, h=0.5, noise=0.03, seed=0) -> freeform.Mesh:
    """Een mesh zoals de vrije vorm hem maakt (surface nets, op het nulniveau gezet, licht gladgestreken) van een
    analytisch veld (positief binnen), met wat ruis langs de normaal zoals van stereo."""
    xs = [np.arange(lo[k], hi[k] + h / 2, h) for k in range(3)]
    P = np.stack(np.meshgrid(*xs, indexing="ij"), axis=-1)
    fld = freeform.Field(np.array(lo, float), h, np.minimum(fn(P), P[..., 2]).astype(np.float32))  # de mat sluit af
    mesh = freeform.smooth(freeform.snap_to_surface(freeform.surface_nets(fld), fld), iters=2)
    if noise:
        n = primitives.vertex_normals(mesh)
        lift = mesh.V[:, 2] > 1e-6  # op de mat blijft op de mat
        V = mesh.V + lift[:, None] * noise * np.random.default_rng(seed).standard_normal(len(mesh.V))[:, None] * n
        mesh = freeform.Mesh(V, mesh.F)
    return mesh


def box(P, lo, hi):
    return np.min(np.concatenate([P - np.array(lo), np.array(hi) - P], axis=-1), axis=-1)


def cylinder(P, p, a, r, t0, t1):
    a = np.array(a, float) / np.linalg.norm(a)
    q = P - np.array(p)
    t = q @ a
    radial = np.linalg.norm(q - t[..., None] * a, axis=-1)
    return np.minimum(r - radial, np.minimum(t - t0, t1 - t))


def by_kind(shapes, base):
    return [s for s in shapes if primitives.base_name(s) == base]


def test_a_plate_with_a_hole():
    """Plaat 40 x 30 x 8 met een gat Ø10: zes vlakken (de onderkant op de mat), een boring langs Z op de goede plek,
    de dikte 8 en de afstanden 40 en 30 tussen de zijkanten, binnen een paar honderdste."""
    mesh = sdf_mesh(lambda P: np.minimum(box(P, (0, 0, 0), (40, 30, 8)), -cylinder(P, (20, 15, 0), (0, 0, 1), 5.0,
                                                                                    -5, 20)),
                    (-3, -3, -3), (43, 33, 11))
    shapes = primitives.detect(mesh)
    planes = by_kind(shapes, "vlak")
    assert len(planes) == 6
    assert sorted(primitives.axis_label(s.n) for s in planes) == ["X", "X", "Y", "Y", "Z", "Z"]
    assert sum(primitives.describe(s) == "vlak op de mat (de onderkant)" for s in planes) == 1
    (hole,) = by_kind(shapes, "boring")
    assert hole.r == pytest.approx(5.0, abs=0.05) and primitives.axis_label(hole.a) == "Z"
    p = hole.p - (hole.p @ hole.a) * hole.a
    assert p[:2] == pytest.approx([20.0, 15.0], abs=0.05)
    meas = {m["name"]: m["value"] for m in primitives.measurements(shapes, mesh)}
    assert meas["boring 1 Ø"] == pytest.approx(10.0, abs=0.1)
    assert meas["dikte 1"] == pytest.approx(8.0, abs=0.05)
    assert sorted(v for k, v in meas.items() if k.startswith("afstand")) == pytest.approx([30.0, 40.0], abs=0.05)
    assert all(primitives.NAME_RE.match(k) for k in meas)


def test_a_lying_shaft_is_a_cylinder_not_a_row_of_facets():
    """Een liggende as Ø20 x 50: één cilinder langs X (geen rij smalle vlakjes erop), en de twee kopse vlakken op
    50 mm van elkaar."""
    mesh = sdf_mesh(lambda P: cylinder(P, (0, 20, 10), (1, 0, 0), 10.0, 5, 55), (2, 7, -3), (58, 33, 23))
    shapes = primitives.detect(mesh)
    (shaft,) = by_kind(shapes, "as")
    assert 2 * shaft.r == pytest.approx(20.0, abs=0.1) and primitives.axis_label(shaft.a) == "X"
    assert shaft.t1 - shaft.t0 == pytest.approx(50.0, abs=1.5)
    planes = by_kind(shapes, "vlak")
    assert all(primitives.axis_label(s.n) == "X" for s in planes) and len(planes) == 2
    meas = {m["name"]: m["value"] for m in primitives.measurements(shapes, mesh)}
    assert meas["afstand 1"] == pytest.approx(50.0, abs=0.05)


def test_a_ball():
    """Een bol Ø30: ook zonder stereo betrouwbaar, want een dubbel gekromd oppervlak maken de silhouetten niet zelf."""
    mesh = sdf_mesh(lambda P: 15.0 - np.linalg.norm(P - np.array([20.0, 20.0, 16.0]), axis=-1), (2, 2, -2),
                    (38, 38, 34))
    shapes = primitives.detect(mesh, stereo=np.zeros(len(mesh.V), bool))
    (ball,) = by_kind(shapes, "bol")
    assert 2 * ball.r == pytest.approx(30.0, abs=0.1) and ball.p == pytest.approx([20.0, 20.0, 16.0], abs=0.05)
    assert ball.reliable


def test_a_gently_curved_top_gives_no_facets():
    """Een blok met een flauw gebogen bovenkant (straal ~90 mm en meer): binnen 0,3 mm passen daar stukken vlak op,
    maar dat zijn facetten van een gebogen oppervlak, geen vlakken. Alleen de zijkanten en de onderkant zijn vlak."""
    def fn(P):
        top = 10.0 + 4.0 * np.sin(np.pi * P[..., 0] / 60.0)
        return np.minimum(box(P, (0, 0, 0), (60, 30, 20)), top - P[..., 2])

    mesh = sdf_mesh(fn, (-3, -3, -3), (63, 33, 17))
    shapes = primitives.detect(mesh)
    planes = by_kind(shapes, "vlak")
    assert planes and all(abs(s.n[2]) < 0.05 or primitives.describe(s) == "vlak op de mat (de onderkant)"
                          for s in planes), [primitives.describe(s) for s in planes]
    assert not by_kind(shapes, "as") and not by_kind(shapes, "boring")


def test_a_rounded_corner_is_a_fillet():
    """Een blok met verticale hoeken afgerond (R 6): een afronding van een kwart cirkel, geen boring of as."""
    def fn(P):  # rechthoek 50 x 30 met hoeken R 6, 10 hoog
        q = np.abs(P[..., :2] - [25.0, 15.0]) - [19.0, 9.0]
        d2 = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0) - 6.0
        return np.minimum(-d2, 10.0 - P[..., 2])

    mesh = sdf_mesh(fn, (-3, -3, -3), (53, 33, 13))
    shapes = primitives.detect(mesh)
    fillets = by_kind(shapes, "afronding")
    assert len(fillets) >= 3
    assert all(f.r == pytest.approx(6.0, abs=0.4) and primitives.axis_label(f.a) == "Z" for f in fillets)
    meas = [m for m in primitives.measurements(shapes, mesh) if m["name"].startswith("afronding")]
    assert meas and all(m["name"].endswith(" R") for m in meas)


def test_reliability_needs_stereo():
    """Zonder stereo telt alleen de onderkant op de mat: de rest kan een stuk van de omhullende van de silhouetten zijn
    (die bestaat zelf uit vlakke stukken) of een 'dak' boven een vlakke bovenkant."""
    mesh = sdf_mesh(lambda P: box(P, (0, 0, 0), (40, 30, 8)), (-3, -3, -3), (43, 33, 11))
    shapes = primitives.detect(mesh, stereo=np.zeros(len(mesh.V), bool))
    assert len(shapes) == 6
    assert [primitives.describe(s) for s in shapes if s.reliable] == ["vlak op de mat (de onderkant)"]
    meas = primitives.measurements(shapes, mesh)
    assert meas and not any(m["reliable"] for m in meas)
    shapes = primitives.detect(mesh, stereo=np.ones(len(mesh.V), bool))
    assert all(s.reliable for s in shapes) and all(m["reliable"] for m in primitives.measurements(shapes, mesh))


def test_reference_geometry_as_step(tmp_path):
    """De vormen als STEP: begrensde vlakken (met het gat erin), het cilindervlak met zijn as, te lezen door
    CadQuery."""
    cq = pytest.importorskip("cadquery")
    mesh = sdf_mesh(lambda P: np.minimum(box(P, (0, 0, 0), (40, 30, 8)), -cylinder(P, (20, 15, 0), (0, 0, 1), 5.0,
                                                                                    -5, 20)),
                    (-3, -3, -3), (43, 33, 11))
    shapes = primitives.detect(mesh)
    path = tmp_path / "referentie.step"
    assert primitives.reference_step(shapes, mesh, path) == len(shapes)
    shape = cq.importers.importStep(str(path)).val()
    kinds = [f.geomType() for f in shape.Faces()]
    assert kinds.count("PLANE") == 6 and kinds.count("CYLINDER") >= 1
    top = max((f for f in shape.Faces() if f.geomType() == "PLANE"), key=lambda f: f.Center().z)
    assert top.Center().z == pytest.approx(8.0, abs=0.05) and len(top.innerWires()) == 1  # het gat
    assert top.Area() == pytest.approx(40 * 30 - np.pi * 25, rel=0.05)  # tot een halve mm voor de rand
    assert any(e.geomType() == "LINE" and abs(e.Length() - 8.0) > 1.0 for e in shape.Edges())  # de as
