"""Kamerboringen herkennen (V16, v0.10): een gat met een cilindrische kamer aan het bovenvlak, voor een
cilinderkopschroef (ISO 4762).

Van boven is een kamer nauwelijks te zien: haar bodem staat even schuin als het bovenvlak en is dus even licht, en
de wand staat loodrecht (alleen aan de overkant, als smalle band). Maar het doorgaande gat eronder is veel korter,
en dat zie je in de schuine foto's: je kijkt verder door het gat heen dan bij een gewoon gat, en in steile foto's
bijna recht door de kamer. Per gat dus de silhouetenergie (zie silhouette.energy) met een kamer: de diepte op een
rooster, de diameter uit DIN 974 bij de maat van het gat (en ruimer en krapper). Een kamer alleen als die de energie
duidelijk verlaagt. De randfit zet diameter en diepte daarna precies, met de rand van de kamer aan het bovenvlak als
extra niveau van het gat (edgefit.rims) en de boven- en onderrand van de wand in de grijswaarden
(edgefit.inner_edges).

Sinds v0.11 eerlijker vergeleken (in v0.10 werd een gewoon gat in een HDR-scan een kamer van 0,3 mm diep): het gewone
gat krijgt ook een nieuwe diameter (de winst kwam vooral daarvan), een kamer is minstens CB_MIN_DEPTH diep, en ze moet
ook een verzinking verslaan. Ook een verzonken gat laat in de schuine foto's meer doorkijken; een verzinking die niet
aan haar ring herkend werd (countersink.py), werd anders een ondiepe kamer. Past een verzinking het best, dan gaat ze
terug naar de pijplijn (`sunk`).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from . import silhouette
from .profile import Part2p5D
from .snapping import DIN_974, ISO_273

DEPTHS = (0.2, 0.35, 0.5, 0.65, 0.8)  # diepte van de kamer, als deel van de hoogte
# Een kamer moet de energie minstens zoveel verlagen: per foto waarin het gat schuin te zien is, en als deel van de
# energie rond het gat bij een gewoon gat (zie detect)
MIN_GAIN_PX, MIN_GAIN_FRAC = 30.0, 0.15
# Een kamer of verzinking is minstens zo diep (mm, en als deel van de hoogte); minder is een afgeschuinde gatrand
# (countersink.py), geen kamer
CB_MIN_DEPTH, CB_MIN_FRAC = 1.0, 0.08


def _diameters(d: float) -> list[float]:
    """Kandidaten voor de diameter van de kamer bij een gat van `d` mm: DIN 974 bij de dichtstbijzijnde
    doorgangsmaat (ISO 273), en 1,5 en 2 keer het gat."""
    size = min(ISO_273, key=lambda s: min(abs(v - d) for v in ISO_273[s]))
    out = [1.5 * d, 2.0 * d]
    if size in DIN_974:
        out.append(DIN_974[size][0])
    return sorted(c for c in set(round(c, 2) for c in out) if c > d + 0.5)


def _local_views(part: Part2p5D, i: int, vd: list, K: np.ndarray, margin_mm: float = 3.0) -> list:
    """De foto's met alleen de uitsnede rond gat i (de energie van de rest verandert niet mee)."""
    from .calib import project

    h = part.holes[i]
    r = h.d + margin_mm  # de ruimste kamer (2 x het gat) plus een marge
    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = np.column_stack([h.x + r * np.cos(a), h.y + r * np.sin(a)])
    P = np.vstack([np.column_stack([ring, np.zeros(len(a))]), np.column_stack([ring, np.full(len(a), part.height)])])
    out = []
    for v in vd:
        uv, z = project(P, v.pose, K)
        if np.any(z <= 0):
            continue
        uv = uv - [v.x0, v.y0]
        hh, ww = v.fg.shape
        x0, y0 = int(max(np.floor(uv[:, 0].min()), 0)), int(max(np.floor(uv[:, 1].min()), 0))
        x1, y1 = int(min(np.ceil(uv[:, 0].max()) + 1, ww)), int(min(np.ceil(uv[:, 1].max()) + 1, hh))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        sl = (slice(y0, y1), slice(x0, x1))
        out.append(silhouette.ViewData(v.pose, v.fg[sl], v.bg[sl], v.unk[sl], v.x0 + x0, v.y0 + y0))
    return out


def detect(part: Part2p5D, K: np.ndarray, vd: list, log=None,
           sunk: dict | None = None) -> dict[int, tuple[float, float, float]]:
    """Gaten met een kamerboring: {index: (diameter van de kamer, diepte, diameter van het gat)}. Zie de moduletekst.

    Het gat zelf is opnieuw bepaald: als gewoon gat gefit is het te groot (de schuine foto's kijken er verder door),
    en een te groot gat vraagt een te ondiepe kamer. Daarom na het rooster om beurten de diepte, het gat en de kamer
    fijner (twee rondes). Een gat met een verzinking telt ook mee: past een kamer veel beter, dan was het geen
    verzinking (de ring van een kamer lijkt er in de foto's van boven op). Gaten waar een verzinking het best past (en
    die er nog geen hebben), komen in `sunk`: {index: (diameter aan het bovenvlak, diameter van het gat)}."""
    found = {}
    min_depth = max(CB_MIN_DEPTH, CB_MIN_FRAC * part.height)
    for i, h in enumerate(part.holes):
        views = _local_views(part, i, vd, K)
        if len(views) < 3:
            continue
        plain = part.copy()
        plain.holes[i] = replace(h, csk=0.0, cb=0.0, cb_depth=0.0)
        e_now = silhouette.energy(part, K, views) if (h.csk > 0 or h.cb > 0) else math.inf

        def energy_of(d: float, dk: float = 0.0, t: float = 0.0, csk: float = 0.0) -> float:
            trial = plain.copy()
            trial.holes[i] = replace(plain.holes[i], d=d, cb=dk, cb_depth=t, csk=csk)
            return silhouette.energy(trial, K, views) if trial.is_valid() else math.inf

        def refit_d(e: float, d: float, **kw) -> tuple[float, float]:
            for d2 in np.arange(d - 0.6, d + 0.2001, 0.1):
                e2 = energy_of(float(d2), **kw)
                if e2 < e:
                    e, d = e2, float(d2)
            return e, d

        # een gewoon gat, ook met een nieuwe diameter (de kamer en de verzinking krijgen die ook)
        e0, d0 = refit_d(energy_of(h.d), h.d)
        e_plain = min(e0, e_now)
        # de kamer: diepte en diameter op een rooster, dan om beurten fijner
        e1, d, dk, t = min((energy_of(h.d, dk, f * part.height), h.d, dk, f * part.height)
                           for dk in _diameters(h.d) for f in DEPTHS)
        if not math.isfinite(e1):
            continue
        for _ in range(2):
            for t2 in np.arange(t - 0.1 * part.height, t + 0.1001 * part.height, 0.025 * part.height):
                e = energy_of(d, dk, float(t2))
                if e < e1:
                    e1, t = e, float(t2)
            e1, d = refit_d(e1, d, dk=dk, t=t)
            # in schuine foto's begrenst de rand van de kamer de doorkijk
            for dk2 in np.arange(dk - 1.0, dk + 1.001, 0.25):
                e = energy_of(d, float(dk2), t)
                if e < e1:
                    e1, dk = e, float(dk2)
        # de verzinking (90°): de diameter aan het bovenvlak op een rooster, dan het gat en die diameter fijner
        tops = [c for c in np.arange(h.d + 2 * min_depth, 2.2 * h.d + 2.0, 0.5)]
        e2, D = min(((energy_of(h.d, csk=float(c)), float(c)) for c in tops), default=(math.inf, 0.0))
        d_csk = h.d
        if math.isfinite(e2):
            e2, d_csk = refit_d(e2, h.d, csk=D)
            for c in np.arange(D - 0.5, D + 0.501, 0.1):
                if c >= d_csk + 2 * min_depth:
                    e = energy_of(d_csk, csk=float(c))
                    if e < e2:
                        e2, D = e, float(c)
        need = max(MIN_GAIN_PX * len(views) / 10.0, MIN_GAIN_FRAC * e0)
        # een kamer: duidelijk beter dan een gewoon gat, diep genoeg, en ook beter dan een verzinking (die heeft een
        # maat minder: bij gelijke energie wint zij)
        ok = e_plain - e1 > need and t >= min_depth and e1 < e2 - MIN_GAIN_PX * len(views) / 10.0
        csk_ok = not ok and h.csk <= 0 and e_plain - e2 > need
        if log:
            log(f"gat {i + 1}: {'kamerboring' if ok else 'verzinking' if csk_ok else 'geen kamerboring'}; "
                f"Ø {dk:.1f} x {t:.1f} diep (gat Ø {d:.2f}) geeft energie {e1:.0f}, een verzinking Ø {D:.1f} "
                f"(gat Ø {d_csk:.2f}) {e2:.0f}, als gewoon gat (Ø {d0:.2f}) {e0:.0f}"
                + (f" en {e_now:.0f} met {'verzinking' if h.csk > 0 else 'kamer'}" if math.isfinite(e_now) else "")
                + f" (nodig: {need:.0f} minder) in {len(views)} foto's")
        if ok:
            found[i] = (dk, t, d)
        elif csk_ok and sunk is not None:
            sunk[i] = (D, d_csk)
    return found
