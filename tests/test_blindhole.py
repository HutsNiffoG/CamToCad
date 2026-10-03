"""Blinde gaten (v0.11): niet in het silhouet, wel in de grijswaarden; als model, in de randfit, herkend in de foto's
en in het CAD-model."""

import math
import runpy

import cv2
import numpy as np
import pytest
from test_edgefit import K, bore_scan, plate

from camtocad import blindhole, cadmodel, edgefit, silhouette
from camtocad.profile import Hole


def blind_plate(depth: float = 3.5) -> "plate":
    """De plaat (50 x 30 x 6) zonder doorgaand gat, met een blind gat Ø6 op (110, 80)."""
    p = plate(hole=False)
    p.holes = [Hole(110.0, 80.0, 6.0, depth=depth)]
    return p


def blind_gray(part, v: silhouette.ViewData, s: int = 3) -> np.ndarray:
    """Grijswaarden met een blind gat, per subpixel de kijkstraal door het model: bovenvlak en bodem 150 (allebei
    vlak), de wand 90, de zijkant 120 en de mat 200; vervaagd met σ 1 px."""
    hh, ww = v.fg.shape
    u, w_ = np.meshgrid((np.arange(ww * s) + 0.5) / s - 0.5 + v.x0, (np.arange(hh * s) + 0.5) / s - 0.5 + v.y0)
    rays = np.linalg.inv(K) @ np.vstack([u.ravel(), w_.ravel(), np.ones(u.size)])
    d, c = v.pose.R.T @ rays, -v.pose.R.T @ v.pose.t
    h, o = part.holes[0], part.outer

    def hit(z):
        t = (z - c[2]) / d[2]
        return c[0] + t * d[0], c[1] + t * d[1]

    X, Y = hit(part.height)
    Xb, Yb = hit(part.height - h.depth)
    inside = np.all([(np.cos(a) * (X - o.center[0]) + np.sin(a) * (Y - o.center[1])) <= off
                     for a, off in zip(o.angles, o.offsets)], axis=0)
    r_top, r_bottom = np.hypot(X - h.x, Y - h.y), np.hypot(Xb - h.x, Yb - h.y)
    fg = np.repeat(np.repeat(v.fg, s, axis=0), s, axis=1).ravel()
    g = np.where(fg, 120.0, 200.0)
    g = np.where(inside, np.where(r_top >= h.d / 2, 150.0, np.where(r_bottom > h.d / 2, 90.0, 150.0)), g)
    g = g.reshape(hh, s, ww, s).mean(axis=(1, 3))
    return cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)


@pytest.fixture(scope="module")
def blind_scan():
    truth = blind_plate()
    vd = bore_scan(truth)
    for v in vd:
        v.gray = blind_gray(truth, v)
    return truth, vd


def test_a_blind_hole_is_valid_only_inside_the_part_and_without_a_chamber():
    assert blind_plate(3.5).is_valid() and blind_plate(3.5).holes[0].blind and not plate().holes[0].blind
    assert not blind_plate(5.9).is_valid()  # bijna door het onderdeel heen
    p = blind_plate(3.5)
    p.holes[0] = Hole(110.0, 80.0, 6.0, csk=8.0, depth=3.5)
    assert not p.is_valid()


def test_a_blind_hole_is_not_in_the_silhouette(blind_scan):
    """Door een blind gat is geen mat te zien: het silhouet is dat van de plaat zonder gat."""
    truth, vd = blind_scan
    for v in vd[::4]:
        assert np.array_equal(silhouette.render(truth, K, v), silhouette.render(plate(hole=False), K, v))


def test_a_blind_hole_is_found_and_fitted_from_its_edges(blind_scan):
    """blindhole.detect vindt het gat in de grijswaarden (bovenrand en de onderrand van de wand), met een diepte op
    het rooster; de randfit zet maat, plaats en diepte daarna precies. Zonder gat: niets."""
    truth, vd = blind_scan
    found = blindhole.detect(plate(hole=False), K, vd)
    assert len(found) == 1
    h = found[0]
    assert (h.x, h.y) == pytest.approx((110.0, 80.0), abs=0.15) and h.d == pytest.approx(6.0, abs=0.25)
    assert h.depth == pytest.approx(3.5, abs=0.4)
    start = plate(hole=False)
    start.holes = [h]
    ef = edgefit.fit(start, K, vd)
    assert ef.accepted, ef.note
    fit = ef.part.holes[0]
    assert (fit.x, fit.y) == pytest.approx((110.0, 80.0), abs=0.05) and fit.d == pytest.approx(6.0, abs=0.08)
    assert fit.depth == pytest.approx(3.5, abs=0.1)
    ev = ef.extra["evidence"][("gat", 0)]
    assert not ev.weak and ev.fraction > 0.9
    no_hole = plate(hole=False)
    no_hole.holes = [Hole(110.0, 80.0, 6.0, depth=1e-6)]  # geen wand: overal bovenvlak
    flat = []
    for v in vd:
        nv = silhouette.ViewData(v.pose, v.fg, v.bg, v.unk, v.x0, v.y0)
        nv.gray = blind_gray(no_hole, v)
        flat.append(nv)
    assert blindhole.detect(plate(hole=False), K, flat) == []


def test_a_blind_hole_in_the_cad_model_and_the_script(tmp_path):
    part = blind_plate(3.5)
    part.holes.append(Hole(130.0, 80.0, 5.0))
    snapped, snaps = cadmodel.snap_part(cadmodel.to_part_frame(part)[0],
                                        cadmodel.Uncertainty(0.03, 0.03, 0.03, 0.03, 0.1))
    names = [s.name for s in snaps]
    assert "blind gat Ø (gat 1)" in names and "blind gat diepte (gat 1)" in names and "gat Ø (gat 2)" in names
    assert snapped.holes[0].depth == 3.5 and not snapped.holes[1].blind
    model = cadmodel.build(snapped)
    assert model.val().isValid()
    box = model.val().BoundingBox()
    assert box.zlen == pytest.approx(6.0)
    area = 50 * 30 - (4 - math.pi) * 2.0 ** 2
    volume = area * 6.0 - math.pi * 3.0 ** 2 * 3.5 - math.pi * 2.5 ** 2 * 6.0
    assert model.val().Volume() == pytest.approx(volume, rel=2e-3)
    path = tmp_path / "model.py"
    code = cadmodel.script(snapped, snaps)
    assert "blinde_gaten = [" in code and "blind_gat(cq" in code
    path.write_text(code.replace('if __name__ == "__main__":', "if False:"))
    assert runpy.run_path(str(path))["model"].val().Volume() == pytest.approx(model.val().Volume(), rel=1e-9)
