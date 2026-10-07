"""Is het onderdeel wel 2,5D? (V19)

Het model is een prisma: één contour, één hoogte, loodrechte wanden. Een trede, een afschuining of
afronding van de bovenrand, of een liggend draaideel past daar niet in. De fit zoekt dan een compromis:
bij een beugel met rondom een afschuining van 2 mm kwam de hoogte op de onderkant van die afschuining
uit (10,07 in plaats van 12 mm), terwijl het silhouet in elke foto goed paste (IoU 0,998). Zo'n stille
fout moet een melding worden. Twee toetsen op de afstanden tussen modelrand en maskerrand na de randfit:

* **Plaatselijk:** langs de buitencontour, per punt van de bovenrand, de mediaan over de foto's. Een
  aaneengesloten stuk van ≥ 8 mm dat meer dan 0,3 mm afwijkt, is een trede of afschuining (het model
  steekt uit: het onderdeel is daar lager) of een verhoging (het onderdeel steekt uit). Op gerenderde
  prisma's blijft dit binnen 0,11 mm; bij een trede van 6 mm is het tot 2 mm over de hele trede.
* **Kijkhoek:** per foto de mediaan over de bovenrand. Bij een prisma hangt die niet af van hoe schuin
  de foto is; bij een afschuining of afronding rondom zien de lage foto's (< 45°) de bovenrand boven het
  model uitsteken, de hoge niet. Op gerenderde prisma's is dat verschil 0,00 tot +0,04 mm (de lage
  foto's net iets ruimer; een zwart onderdeel met een telefoonkromme tot +0,11 mm), bij de afschuining
  −0,06 mm en bij een liggende cilinder +0,40 mm. Een afschuining van ~1 mm valt hierbinnen en wordt niet
  herkend. Zonder foto's onder 45° geen toets.

Een punt telt mee waar er bewijs is: zekere mat vlakbij, of het punt ligt voorbij de strook zonder
bewijs al in zekere mat (dan steekt het model zeker uit). Twee stukken van dezelfde soort met minder dan
10 mm ertussen (vaak een hoek zonder bewijs) zijn één stuk.

Wat de toets vindt, is ook de start voor V17: de pijplijn probeert dan een rechte trede (door de uiteinden
van het stuk) of een afschuining of afronding van de bovenrand rondom als model, en toetst daarna opnieuw.
Vindt de toets meer stukken van dezelfde soort, dan begint de trede bij het kleinste deel van de contour dat
ze allemaal bevat (`PrismCheck.start`): een rechte trede heeft één aaneengesloten laag deel, en een stuk
daarvan kan in de fit met een compromis-prisma net niet afwijken (v0.8: op de ene machine een gat van 9 mm,
op een andere van 11 mm, en dan begon de trede schuin).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import edgefit
from .calib import project

LOCAL_MM = 0.3  # plaatselijke afwijking van de bovenrand
MIN_RUN_MM = 8.0  # over minstens zoveel contour
MERGE_GAP_MM = 10.0  # twee stukken met een kleiner gat ertussen (vaak een hoek zonder bewijs) zijn één stuk
# verschil tussen lage en hoge foto's: prisma's 0,00 tot +0,04, een zwart onderdeel met een telefoonkromme tot +0,11
# (stresstest srgb_donker, v0.14); een liggende cilinder +0,40
TREND_LOW_MM, TREND_HIGH_MM = -0.04, 0.15
LOW_DEG, HIGH_DEG = 45.0, 60.0
MIN_VIEWS = 3


@dataclass
class PrismCheck:
    issues: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)  # voor diagnose.json
    # de afwijkende stukken in matcoördinaten (begin, midden, eind), als start voor een trede (V17)
    runs_mat: list[dict] = field(default_factory=list)
    # de start voor een trede: de stukken van de soort met de meeste lengte samen (zie de moduletekst)
    start: dict | None = None

    @property
    def trend(self) -> float | None:
        return self.details.get("kijkhoek_verschil_mm")


def _deviations(prob, x: np.ndarray):
    """Per modelpunt (rij in edgefit.rims) en foto de afstand modelrand -> maskerrand in mm (+: het model
    steekt buiten het object uit), en per foto de kijkhoek (graden boven de mat, vanuit het midden van het
    model)."""
    p = prob.build(x)
    rim = edgefit.rims(p, prob.lay)
    P = rim.P
    oc = p.outer.outline()
    center = np.array([*(oc.min(axis=0) + oc.max(axis=0)) / 2, p.height / 2])
    f = float(prob.K[0, 0])
    per_point: list[list[float]] = [[] for _ in range(len(P))]
    per_view: list[tuple[float, np.ndarray, np.ndarray]] = []
    for i, v in enumerate(prob.vd):
        C = v.pose.center
        elev = math.degrees(math.atan2(C[2] - center[2], math.hypot(C[0] - center[0], C[1] - center[1])))
        on = np.flatnonzero(prob.status[i])
        if not len(on):
            per_view.append((elev, on, np.zeros(0)))
            continue
        uv, z = project(P[on], v.pose, prob.K)
        h, w = v.fg.shape
        uv = np.clip(uv - [v.x0, v.y0], 0, [w - 1, h - 1])
        dist, band = prob.fields[i]  # afstand tot de rand (px, + = buiten; zie edgefit.edge_distance)
        r = edgefit._bilinear(dist, uv)
        g = edgefit._bilinear(band, uv)
        ok = (g < 4.0) | (r > g + 1.0)
        mm = r * z / f
        for j, val in zip(on[ok], mm[ok]):
            per_point[j].append(float(val))
        per_view.append((elev, on[ok], mm[ok]))
    return p, rim, per_point, per_view


def _smooth(a: np.ndarray, k: int = 5) -> np.ndarray:
    """Gemiddelde over k opeenvolgende punten van een gesloten contour (NaN telt niet mee)."""
    h = k // 2
    vals, wts = np.where(np.isnan(a), 0.0, a), (~np.isnan(a)).astype(float)
    num = np.convolve(np.r_[vals[-h:], vals, vals[:h]], np.ones(k), "valid")
    den = np.convolve(np.r_[wts[-h:], wts, wts[:h]], np.ones(k), "valid")
    return np.where(den >= 3, num / np.maximum(den, 1.0), np.nan)


def _close_gaps(mask: np.ndarray, step: np.ndarray, max_gap: float) -> np.ndarray:
    """Vult in een gesloten rij korte stukken False tussen twee stukken True op (lengte langs de contour)."""
    if mask.all() or not mask.any():
        return mask
    out = mask.copy()
    for gap in _runs(~mask):
        if float(step[gap].sum()) < max_gap:
            out[gap] = True
    return out


def _runs(mask: np.ndarray) -> list[np.ndarray]:
    """Aaneengesloten stukken True in een gesloten rij, als indexlijsten in volgorde."""
    n = len(mask)
    if mask.all():
        return [np.arange(n)]
    start = int(np.flatnonzero(~mask)[0])  # begin na een False, zodat geen stuk over het einde loopt
    order = (np.arange(n) + start) % n
    out, cur = [], []
    for j in order:
        if mask[j]:
            cur.append(j)
        elif cur:
            out.append(np.array(cur))
            cur = []
    if cur:
        out.append(np.array(cur))
    return out


def _span(runs: list[np.ndarray], n: int) -> np.ndarray:
    """Het kleinste stuk van de gesloten rij (n punten) dat alle stukken bevat: alles behalve het grootste gat."""
    keep = np.zeros(n, bool)
    for r in runs:
        keep[r] = True
    gaps = _runs(~keep)
    if gaps:
        keep[:] = True
        keep[max(gaps, key=len)] = False
    return _runs(keep)[0]


def check(ef: edgefit.EdgeFit, angle: float = 0.0, shift=(0.0, 0.0)) -> PrismCheck:
    """Toetst of het gefitte prisma bij de foto's past (zie de moduletekst). `angle`, `shift`: het
    werkassenstelsel (cadmodel.to_part_frame), voor de plaatsaanduiding in de melding."""
    out = PrismCheck()
    prob = ef.extra.get("problem")
    if prob is None or not prob.status:
        return out
    x = ef.x if ef.accepted else prob.x_of(ef.part)
    part, rim, per_point, per_view = _deviations(prob, x)
    no = edgefit.outer_count(prob.lay)
    # de bovenrand van de buitencontour: per contourpunt de hogere niveaus samen (per foto vormt er hooguit één
    # de silhouetrand; bij een prisma is dat alleen de bovenrand)
    upper = (rim.point >= 0) & (rim.point < no) & (rim.level > 0)
    vals: list[list[float]] = [[] for _ in range(no)]
    for r in np.flatnonzero(upper):
        vals[rim.point[r]] += per_point[r]
    top = np.array([np.median(v) if len(v) >= MIN_VIEWS else np.nan for v in vals])
    p2 = edgefit.points2d(part, prob.lay)[0][:no]
    c, s = math.cos(angle), math.sin(angle)
    q = p2 @ np.array([[c, s], [-s, c]]) + np.asarray(shift, float)  # werkcoördinaten
    step = np.linalg.norm(np.roll(p2, -1, axis=0) - p2, axis=1)
    sm = _smooth(top)
    runs = []
    found: dict[str, list[tuple[np.ndarray, float]]] = {"lager": [], "hoger": []}
    for sign, what in ((1.0, "lager"), (-1.0, "hoger")):
        # een stuk dat bij een hoek even geen bewijs heeft (of net onder de drempel komt), blijft één stuk
        for run in _runs(_close_gaps(np.nan_to_num(sign * sm, nan=-1.0) > LOCAL_MM, step, MERGE_GAP_MM)):
            length = float(step[run[:-1]].sum()) if len(run) > 1 else 0.0
            if length < MIN_RUN_MM:
                continue
            found[what].append((run, length))
            dev = float(np.nanmax(np.where(np.isnan(sm[run]), -np.inf, sign * sm[run])))
            a, b = q[run[0]], q[run[-1]]
            runs.append({"soort": what, "van": a.round(1).tolist(), "tot": b.round(1).tolist(),
                         "lengte_mm": round(length, 1), "afwijking_mm": round(dev, 2)})
            out.runs_mat.append({"soort": what, "van": p2[run[0]], "tot": p2[run[-1]],
                                 "midden": p2[run[len(run) // 2]], "afwijking_mm": dev})
            if what == "lager":
                out.issues.append(
                    f"langs de rand van ({a[0]:.0f}, {a[1]:.0f}) tot ({b[0]:.0f}, {b[1]:.0f}) (over {length:.0f} mm) ligt "
                    f"de bovenkant lager dan het model (afwijking in beeld tot {dev:.1f} mm): een trede, afschuining "
                    "of ronding die niet in een 2,5D-model met één hoogte past")
            else:
                out.issues.append(
                    f"langs de rand van ({a[0]:.0f}, {a[1]:.0f}) tot ({b[0]:.0f}, {b[1]:.0f}) (over {length:.0f} mm) "
                    f"steekt het onderdeel boven het model uit (tot {dev:.1f} mm in beeld): het is daar hoger of heeft "
                    "een ronde bovenkant")
    kind = max(found, key=lambda k: sum(length for _, length in found[k]))
    if found[kind]:
        span = _span([r for r, _ in found[kind]], no)
        dev = float(max(out_run["afwijking_mm"] for out_run in out.runs_mat if out_run["soort"] == kind))
        out.start = {"soort": kind, "van": p2[span[0]], "tot": p2[span[-1]], "midden": p2[span[len(span) // 2]],
                     "afwijking_mm": dev, "stukken": len(found[kind])}
    # kijkhoek: per foto de mediaan over de bovenrand van de buitencontour
    rows = []
    for elev, idx, mm in per_view:
        sel = upper[idx]
        if sel.sum() >= 10:
            rows.append((elev, float(np.median(mm[sel]))))
    trend = None
    if rows:
        e = np.array([r[0] for r in rows])
        m = np.array([r[1] for r in rows])
        low, high = m[e < LOW_DEG], m[e >= HIGH_DEG]
        if len(low) >= 4 and len(high) >= 4:
            trend = float(np.median(low) - np.median(high))
            same_side = np.mean(np.sign(low - np.median(high)) == np.sign(trend))
            if (trend < TREND_LOW_MM or trend > TREND_HIGH_MM) and same_side >= 0.7 and not runs:
                if trend < 0:
                    out.issues.append(
                        "in de lage foto's steekt de bovenrand rondom iets boven het model uit: de bovenrand is "
                        f"waarschijnlijk afgeschuind of afgerond, en dan geldt de hoogte ({part.height:.2f} mm) voor de "
                        "onderkant daarvan")
                else:
                    out.issues.append("in de lage foto's ligt de bovenrand rondom binnen het model: de wanden staan "
                                      "mogelijk niet loodrecht op de mat (tapse wanden of een ronde vorm)")
    out.details = {"stukken": runs, "kijkhoek_verschil_mm": None if trend is None else round(trend, 3),
                   "bovenrand_mediaan_mm": None if np.isnan(sm).all() else round(float(np.nanmedian(sm)), 3)}
    return out
