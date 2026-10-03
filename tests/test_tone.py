"""Het toonmodel van de camera (tone.py, v0.11): verscherping uit schuine randen, ontscherpen, de kromme per foto en
een kromme per tegel bij lokale toonbewerking (HDR)."""

import cv2
import numpy as np
import pytest

from camtocad import masks, tone
from camtocad.calib import CameraModel


def _usm(img: np.ndarray, k: float, sigma: float) -> np.ndarray:
    return img + k * (img - cv2.GaussianBlur(img, (0, 0), sigma))


def _pattern(h: int = 360, w: int = 480) -> np.ndarray:
    """Een mat in het klein: vakken met stippen, vervaagd door de optiek (lineair licht, 15-240)."""
    yy, xx = np.mgrid[0:h, 0:w]
    squares = ((xx // 41) + (yy // 33)) % 2 == 0
    dots = ((xx % 11) - 5) ** 2 + ((yy % 11) - 5) ** 2 < 5
    white = squares ^ (dots & ((xx // 41) % 3 == 0))
    return cv2.GaussianBlur(np.where(white, 240.0, 15.0).astype(np.float32), (0, 0), 1.2)


def _slanted_edges(g: float, k: float, sk: float, sb: float = 0.9, n: int = 8, seed: int = 0) -> list:
    """Randen tussen zwart en wit zoals edge_profiles ze geeft, maar uit een eigen simulatie: per pixel het bedekte
    oppervlak (8 x 8 subpixels), vervaging in lineair licht, de kromme, unsharp masking, ruis en afronden."""
    rng = np.random.default_rng(seed)
    size, ss = 40, 8
    ys, xs = (np.mgrid[0:size * ss, 0:size * ss] + 0.5) / ss - 0.5
    yy, xx = np.mgrid[0:size, 0:size]
    edges = []
    for i in range(n):
        th = rng.uniform(0, np.pi / n) + i * np.pi / n
        nrm = np.array([np.cos(th), np.sin(th)])
        c0 = size / 2 + rng.uniform(-0.5, 0.5, 2)
        cover = ((xs - c0[0]) * nrm[0] + (ys - c0[1]) * nrm[1] >= 0).reshape(size, ss, size, ss).mean(axis=(1, 3))
        lin = cv2.GaussianBlur((12.0 + 213.0 * rng.uniform(0.9, 1.1) * cover).astype(np.float32), (0, 0), sb)
        S = _usm(255.0 * (lin / 255.0) ** g, k, sk)
        S = np.clip(np.round(S + rng.normal(0, 1.0, S.shape)), 0, 255)
        dd = (xx - c0[0]) * nrm[0] + (yy - c0[1]) * nrm[1]
        keep = (np.abs(dd) < 7) & (np.minimum(xx, yy) > 8) & (np.maximum(xx, yy) < size - 8)
        edges.append((dd[keep], S[keep].astype(float), abs(nrm[0]), abs(nrm[1])))
    return edges


def test_desharpen_undoes_unsharp_masking():
    img = _pattern()
    sharp = _usm(img, 0.6, 1.4)
    assert np.abs(sharp - img).max() > 20.0
    back = masks.desharpen(sharp, 0.6, 1.4)
    assert np.abs(back - img)[8:-8, 8:-8].max() < 1.0
    assert np.array_equal(masks.desharpen(sharp, 0.0, 1.4), sharp)  # k = 0: niets te doen


@pytest.mark.parametrize("g", [1 / 2.2, 1.0])
def test_sharpening_is_measured_on_slanted_edges(g):
    """k en σ van de verscherping komen uit de randen, ook met een sRGB-kromme."""
    k, sigma = tone.fit_sharpening(_slanted_edges(g, 0.5, 1.5))
    assert k == pytest.approx(0.5, abs=0.1)
    assert sigma == pytest.approx(1.5, abs=0.3)


@pytest.mark.parametrize("sb", [0.5, 1.4])
def test_no_sharpening_is_found_where_there_is_none(sb):
    """Zonder verscherping vindt de fit hooguit een 'verscherping' met een straal van een halve pixel (een restje van
    het model bij een vage foto en een sRGB-kromme): dat telt niet."""
    for seed in (0, 1):
        k, _ = tone.fit_sharpening(_slanted_edges(1 / 2.2, 0.0, 1.5, sb=sb, seed=seed))
        assert k == 0.0
    assert tone.fit_sharpening(_slanted_edges(1 / 2.2, 0.5, 1.5)[:5]) is None  # te weinig randen


def test_curve_per_photo_with_the_cameras_sharpening():
    """De kromme komt uit de ontscherpte foto; de verscherping zelf gaat mee in het resultaat voor de maskers."""
    rng = np.random.default_rng(2)
    p = _pattern()
    cam = CameraModel(np.array([[500.0, 0, 240], [0, 500.0, 180], [0, 0, 1]]), np.zeros(5), 480, 360)
    raw = _usm(255.0 * ((0.9 * p + 8.0) / 255.0) ** (1 / 2.2), 0.5, 1.5) + rng.normal(0, 1.0, p.shape)
    raw = np.clip(raw, 0, 255).astype(np.float32)
    valid = np.ones(p.shape, bool)
    tn = tone.estimate(raw, p, valid, cam, tone.Sharpening(0.5, 1.5, 3))
    assert tn.g == pytest.approx(1 / 2.2, abs=0.04)
    assert (tn.sharpen, tn.sharpen_px, tn.grid) == (0.5, 1.5, None)
    assert tone.estimate(raw, p, valid, cam, tone.Sharpening(0.03, 1.5, 3)).sharpen == 0.0  # te weinig: geen


def test_local_tone_mapping_gets_a_curve_per_tile():
    """Lokale toonbewerking (HDR): de exponent loopt over het beeld. Dan een kromme per tegel, met de helling in de
    goede richting; bij één kromme voor het hele beeld geen raster."""
    rng = np.random.default_rng(4)
    p = _pattern(540, 720)
    valid = np.ones(p.shape, bool)
    lin = (0.9 * p + 8.0) / 255.0
    gx = np.linspace(-1.0, 1.0, p.shape[1], dtype=np.float32)[None, :]
    for spread, expect_grid in ((0.08, True), (0.0, False)):
        o = 255.0 * lin ** (1 / 2.2 + spread * gx) + rng.normal(0, 1.0, p.shape)
        o = np.clip(o, 0, 255).astype(np.float32)
        g0 = masks._tone_exponent(o, p, valid)[0]
        grid = tone.tile_grid(o, p, valid, g0, 0.0)
        assert (grid is not None) == expect_grid
        if expect_grid:
            assert grid[:, 2].mean() - grid[:, 0].mean() == pytest.approx(2 * 0.08 * 2 / 3, abs=0.04)
            tn = masks.Tone(g=g0, grid=grid)
            gm = tn.g_map(p.shape)
            assert gm.shape == p.shape and gm[:, -1].mean() > gm[:, 0].mean()
