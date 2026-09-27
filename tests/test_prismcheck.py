"""Is het onderdeel wel 2,5D? (V19): een trede of een afschuining rondom moet een melding geven."""

import math

import numpy as np

from camtocad import cadmodel, edgefit, prismcheck, silhouette
from camtocad.calib import Pose
from camtocad.masks import ViewMasks
from camtocad.profile import Part2p5D, Profile
from camtocad.render import look_at

from test_edgefit import H, K, W, exact_mask


def block(x0, x1, y0, y1, h, r=0.0) -> Part2p5D:
    prof = Profile("polygon", np.array([(x0 + x1) / 2, (y0 + y1) / 2]), np.array([-np.pi / 2, 0, np.pi / 2, np.pi]),
                   np.array([(y1 - y0) / 2, (x1 - x0) / 2, (y1 - y0) / 2, (x1 - x0) / 2]), np.full(4, r))
    return Part2p5D(h, prof)


def scan(pieces: list[Part2p5D], elevations=(35.0, 55.0, 72.0), n_ring: int = 8, n_top: int = 3):
    """Silhouetten van een onderdeel dat uit prisma's is opgebouwd (vereniging): een trede of een afschuining
    in lagen. Ringen op de gegeven hoogtes (graden boven de mat) en een paar bovenaanzichten."""
    poses = []
    for e in elevations:
        for k in range(n_ring):
            az = 2 * np.pi * (k + 0.3 + 0.5 * (e > 45)) / n_ring
            d = 300.0
            c = [120 + d * math.cos(math.radians(e)) * np.cos(az), 80 + d * math.cos(math.radians(e)) * np.sin(az),
                 d * math.sin(math.radians(e))]
            poses.append(look_at(c, [120.0, 80.0, 0.0]))
    for k in range(n_top):
        c = np.array([112.0 + 8 * k, 76.0 + 6 * (k % 2), 320.0])
        poses.append(look_at(c, [c[0], c[1], 0.0]))
    empty = np.zeros((H, W), bool)
    views = [(Pose(f"v{k}", R, t), ViewMasks(fg=empty, bg=~empty, valid=~empty)) for k, (R, t) in enumerate(poses)]
    vd = silhouette.prepare(views, K, max(pieces, key=lambda p: p.height))
    for v in vd:
        v.fg = np.logical_or.reduce([exact_mask(p, v) for p in pieces])
        v.bg = ~v.fg
    return vd


def run_check(pieces, start, accepted: bool | None = None):
    """De randfit zoals in de pijplijn (0,23 mm per pixel), dan de vormtoets. Wordt de randfit niet aangenomen,
    dan toetst de vormtoets het startmodel (zoals de pijplijn de pixelfit); `accepted` eist het een of het ander."""
    vd = scan(pieces)
    ef = edgefit.fit(start, K, vd, mm_per_px=0.23)
    if accepted is not None:
        assert ef.accepted == accepted, ef.note
    _, angle, shift = cadmodel.to_part_frame(ef.part)
    return prismcheck.check(ef, angle, shift)


def test_a_prism_passes():
    plate = block(95, 145, 65, 95, 6.0, 2.0)
    res = run_check([plate], plate)
    assert res.issues == [] and res.details["stukken"] == []
    assert abs(res.details["kijkhoek_verschil_mm"]) < 0.03


def test_a_step_is_found_where_it_is():
    """De rechter 15 mm van de plaat is maar 3 mm hoog in plaats van 6: de bovenrand daar ligt lager."""
    low, high = block(95, 145, 65, 95, 3.0), block(95, 130, 65, 95, 6.0)
    res = run_check([low, high], block(95, 145, 65, 95, 6.0))
    lower = [r for r in res.details["stukken"] if r["soort"] == "lager"]
    assert len(lower) == 1 and lower[0]["afwijking_mm"] > 0.5 and lower[0]["lengte_mm"] > 20
    xs = [lower[0]["van"][0], lower[0]["tot"][0]]
    assert min(xs) > 30  # werkcoördinaten: de trede begint op x = 35
    assert any("trede, afschuining of ronding" in m for m in res.issues)


def test_a_chamfer_all_round_shows_in_the_low_photos():
    """Rondom een afschuining van 2 mm (vier lagen): het silhouet past op een prisma van ~4 mm, maar de
    lage foto's zien de afschuining boven het model uitsteken."""
    layers = [block(95 + d, 145 - d, 65 + d, 95 - d, 4.0 + d) for d in (0.0, 0.5, 1.0, 1.5, 2.0)]
    # De randfit komt op ~4,04 uit. Vanuit 4,5 lag dat 0,04 mm binnen het vertrouwensgebied; op sommige
    # CI-machines net erbuiten, en dan gaf het startmodel +0,02 in plaats van -0,18.
    res = run_check(layers, block(95, 145, 65, 95, 4.2), accepted=True)
    assert res.details["kijkhoek_verschil_mm"] < prismcheck.TREND_LOW_MM
    assert any("afgeschuind of afgerond" in m for m in res.issues)


def test_the_span_of_several_runs_skips_only_the_largest_gap():
    """De start van een trede (V17) bij meer stukken van dezelfde soort: alles behalve het grootste gat, ook als
    het stuk over het begin van de gesloten rij loopt."""
    assert prismcheck._span([np.arange(2, 6), np.arange(10, 13)], 20).tolist() == list(range(2, 13))
    assert prismcheck._span([np.array([18, 19, 0, 1]), np.array([5, 6])], 20).tolist() == [18, 19, 0, 1, 2, 3, 4, 5, 6]
    assert prismcheck._span([np.arange(3, 9)], 20).tolist() == list(range(3, 9))
