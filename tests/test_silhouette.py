"""Silhouetfit: energie over alle foto's en afbreken als het model niet past."""

import numpy as np

from camtocad import silhouette
from camtocad.calib import Pose
from camtocad.profile import Hole, Part2p5D, Profile
from camtocad.render import look_at

K = np.array([[1300.0, 0.0, 799.5], [0.0, 1300.0, 599.5], [0.0, 0.0, 1.0]])


def plate() -> Part2p5D:
    prof = Profile("polygon", np.array([120.0, 80.0]), np.array([-np.pi / 2, 0, np.pi / 2, np.pi]),
                   np.array([15.0, 25.0, 15.0, 25.0]), np.full(4, 2.0))
    return Part2p5D(6.0, prof, [Hole(110.0, 80.0, 6.0)])


def views_of(part: Part2p5D, n: int = 8, fg_from_model: bool = True) -> list[silhouette.ViewData]:
    out = []
    for k in range(n):
        az = 2 * np.pi * k / n
        center = np.array([120 + 200 * np.cos(az), 80 + 200 * np.sin(az), 250.0])
        R, t = look_at(center, [120.0, 80.0, 0.0])
        v = silhouette.ViewData(Pose(f"v{k}", R, t), np.zeros((1200, 1600), bool), np.ones((1200, 1600), bool),
                                np.zeros((1200, 1600), bool), 0, 0)
        if fg_from_model:
            v.fg = silhouette.render(part, K, v).astype(bool)
            v.bg = ~v.fg
        out.append(v)
    return out


def test_energy_counts_every_view_once():
    part = plate()
    views = views_of(part)
    moved = part.copy()
    moved.outer.offsets = moved.outer.offsets + 0.8
    serial = 0.0
    for v in views:
        P = silhouette.render(moved, K, v).astype(bool)
        serial += np.count_nonzero(P & v.bg) + 0.25 * np.count_nonzero(P & v.unk) + np.count_nonzero(~P & v.fg)
    assert silhouette.energy(moved, K, views) == serial > 0
    assert silhouette.energy(part, K, views) == 0


def test_refine_gives_up_when_nothing_fits():
    """Geen objectpixels, alles mat: na 300 evaluaties stopt de zoektocht in plaats van 1500 te doen."""
    messages = []
    views = views_of(plate(), n=4, fg_from_model=False)
    _, _, evals = silhouette.refine(plate(), K, views, max_evals=1500, log=messages.append)
    assert 300 <= evals < 700
    assert any("afgebroken" in m for m in messages)


def test_fillet_probe_finds_a_large_fillet_from_a_sharp_corner():
    """Vanuit een scherpe hoek levert een kleine afronding bijna niets op; de proef vindt 4 mm wel."""
    truth = plate()
    truth.outer.fillets[:] = 4.0
    views = views_of(truth, n=4)
    start = truth.copy()
    start.outer.fillets[:] = 0.0
    e0 = silhouette.energy(start, K, views)
    found, e, changed = silhouette.probe_fillets(start, K, views, e0)
    assert changed and e < 0.2 * e0
    assert np.allclose(found.outer.fillets, 4.0)
