"""Randfit (V2): het 2,5D-model fitten op subpixel-afstanden tot de silhouetrand, met kleinste kwadraten.

De pixelfit (silhouette.py) telt pixels die niet kloppen. Dat is robuust, maar een trapfunctie:
geen afgeleiden, dus geen covariantie, en afrondingen en hoogte dwalen een paar tiende millimeter.
Deze fit verfijnt het resultaat daarna.

* Modelpunten op de onder- en bovenrand van de contour en van de gaten (~1 per mm) worden per foto
  geprojecteerd. Per punt vormt de onderrand de silhouetrand als de wand naar de camera kijkt, anders
  de bovenrand; punten die in het modelsilhouet verborgen liggen tellen niet. Een afgeschuinde of
  afgeronde bovenrand (V17) geeft de buitencontour niveaus tussen onder- en bovenrand (de schouder, de
  boog), en dan telt het niveau dat in beeld het verst naar buiten ligt. Een trede (V17) geeft de
  bovenrand per stuk een eigen hoogte, plus punten op de verticale randen van de trede (zie `rims`).
* Residu = afstand (px, subpixel via bilineaire interpolatie) van het punt tot de rand van het
  objectmasker, alleen waar ook zekere mat vlakbij is (bewijs). De ware rand ligt in de strook zonder
  bewijs tussen object en zekere mat: de maskerrand ligt gemiddeld iets naar binnen (vooral waar een
  zichtbare wand boven een zwart vak de rand vormt). Daarom telt de rand op een fractie `BETA` van die
  strook. BETA is afgesteld op gerenderde scans met zuivere silhouetten (ROUTE-A-VERBETERPUNTEN §3e).
* Kleinste kwadraten met een Cauchy-verlies (schaal 0,5 px): een uitschieter (schaduw, een hap uit het
  masker) telt nauwelijks mee, en de fit gedraagt zich meer als een mediaan dan als een gemiddelde.
* Alleen binnen een vertrouwensgebied rond de pixelfit (±0,5 mm, ±0,3°): zonder bewijs rond een gat
  (zwart op zwart) kan een parameter anders wegdrijven. Raakt de oplossing de rand van dat gebied, dan
  blijft de pixelfit staan. Ruimer waar de pixelfit zelf onnauwkeurig is: de as van een sleuf (±2°), de
  maat van een afschuining of afronding van de bovenrand (±1 mm), de lijn van een trede (±1 mm, ±2°), en
  een afronding mag tot 5 px groeien (onscherpte maakt van een scherpe hoek een kleine afronding).
* Een parameter die de contour bij een kleine stap (0,1 mm) ongeldig maakt, blijft staan: een afronding
  die net past, de randjes van een hap uit het masker. Zijn afgeleide is onbruikbaar (een sprong naar de
  strafwaarde) en zou de hele oplossing vastzetten.

Bij de oplossing hoort een Jacobiaan; met een jackknife over groepen foto's (`jackknife`) levert dat de
covariantie van de parameters voor een eerlijke U95 (V3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from . import silhouette
from .cadhelpers import afgeronde_hoeken
from .calib import project
from .profile import Part2p5D

BETA = 0.2  # waar in de strook zonder bewijs de rand ligt (0 = maskerrand, 1 = begin zekere mat)
BAND_MAX = 1.5  # px: breder telt als gebrek aan bewijs, niet als menging
LOSS, F_SCALE = "cauchy", 0.5  # px
DENSITY = 1.0  # punten per mm contour
TRUST_MM, TRUST_DEG = 0.5, 0.3
TRUST_SLOT_DEG = 2.0  # de as van een korte sleuf is in de pixelfit minder precies dan de hele contour
# de maat van een afschuining of afronding van de bovenrand (V17): alleen de lage foto's zien haar, en de
# pixelfit laat haar samen met de schouderhoogte schuiven (vooral bij een afronding)
TRUST_TOP_MM = 1.0
# een trede (V17): de plaats van de lijn is in de pixelfit onnauwkeuriger (alleen de verticale randen van de
# trede in zijaanzichten zeggen er iets over); punten per mm op die randen
TRUST_STEP_MM, STEP_VERT_DENSITY = 1.0, 4.0
# een scherpe hoek komt door onscherpte als een afronding van 3,5-4 px uit de randfit (§3e), vaak meer dan
# TRUST_MM boven de pixelfit; afrondingen mogen daarom tot zoveel pixels groeien (de pijplijn maakt alles onder
# 4,5 px daarna weer scherp)
TRUST_FILLET_PX = 5.0
FREEZE_PROBE_MM = 0.1


def signed_dist(mask: np.ndarray) -> np.ndarray:
    """Negatief binnen, positief buiten; nul op de pixelrand (px)."""
    m8 = mask.astype(np.uint8)
    d_out = cv2.distanceTransform(1 - m8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_in = cv2.distanceTransform(m8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(mask, 0.5 - d_in, d_out - 0.5).astype(np.float32)


def _bilinear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    return ndimage.map_coordinates(img, [uv[:, 1], uv[:, 0]], order=1, mode="nearest")


def _arc(t1, t2, c, r, m, n: int) -> np.ndarray:
    a1 = math.atan2(t1[1] - c[1], t1[0] - c[0])
    a2 = math.atan2(t2[1] - c[1], t2[0] - c[0])
    am = math.atan2(m[1] - c[1], m[0] - c[0])
    span = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
    if abs((am - a1 + math.pi) % (2 * math.pi) - math.pi) > abs(span) + 1e-6:
        span = span - math.copysign(2 * math.pi, span)
    a = a1 + (np.arange(n) + 0.5) / n * span
    return np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])


def _polygon_layout(table, density: float, min_edge: int) -> dict:
    hoeken = afgeronde_hoeken(table)
    n = len(hoeken)
    edges, arcs = [], []
    for k in range(n):
        _, m, t2, _, r = hoeken[k]
        arcs.append(max(2, int(abs(r) * math.pi / 2 * density) + 1) if m is not None else 2)
        nxt = hoeken[(k + 1) % n][0]
        length = math.hypot(nxt[0] - t2[0], nxt[1] - t2[1])
        # een rand van (bijna) nul lang, zoals de korte zijde van een sleuf, krijgt geen punten
        edges.append(0 if (min_edge == 0 and length < 0.5) else max(min_edge, int(length * density)))
    return {"edges": edges, "arcs": arcs}


def layout(part: Part2p5D, density: float = DENSITY) -> dict:
    """Aantal punten per rand, boog, gat en sleuf; vast tijdens een oplossing (vaste lengte van de residuen)."""
    o = part.outer
    out = {"holes": [max(16, int(math.pi * h.d * density)) for h in part.holes],
           "slots": [_polygon_layout(s.corner_table(), density, 0) for s in part.slots]}
    if o.kind == "circle":
        out["circle"] = max(48, int(2 * math.pi * o.radius * density))
    else:
        out.update(_polygon_layout(o.corner_table(), density, 4))
    if part.steps:  # treden (V17): punten langs de lijn en op de twee verticale randen van de trede
        out["steps"] = []
        for st in part.steps:
            q = part.step_crossings(st)
            length = float(np.linalg.norm(q[1] - q[0])) if q is not None else 0.0
            out["steps"].append({"line": max(4, int(length * density)),
                                 "vert": max(4, int(STEP_VERT_DENSITY * (part.height - st.height)))})
    return out


def cell_of(part: Part2p5D, p2: np.ndarray) -> np.ndarray:
    """Per punt de trede waar het boven ligt (index), of -1 voor het hoge deel."""
    out = np.full(len(p2), -1)
    for k in reversed(range(len(part.steps))):
        st = part.steps[k]
        out[p2 @ st.normal() > st.offset] = k
    return out


def _polygon_points(table, normals: np.ndarray, lay: dict, sign: float, pts: list, nrm: list, use: list) -> None:
    """Punten op een contour met afrondingen; `sign` -1 voor een uitsparing (het materiaal ligt erbuiten)."""
    hoeken = afgeronde_hoeken(table)
    n = len(hoeken)
    for k in range(n):
        t1, m, t2, c, r = hoeken[k]
        na = lay["arcs"][k]
        if m is None:
            pts.append(np.repeat([t1], na, axis=0))
            nrm.append(np.repeat([sign * normals[k]], na, axis=0))
            use.append(np.zeros(na, bool))
        else:
            arc = _arc(t1, t2, c, r, m, na)
            d = (arc - np.asarray(c)) / r
            # bolle hoek: middelpunt in het materiaal, normaal ervan af; holle hoek: andersom
            convex = float(np.dot(np.asarray(m) - np.asarray(c), normals[k - 1] + normals[k])) > 0
            pts.append(arc)
            nrm.append(sign * (d if convex else -d))
            use.append(np.ones(na, bool))
        nxt = np.asarray(hoeken[(k + 1) % n][0], float)
        ne = lay["edges"][k]
        s = (np.arange(ne) + 0.5) / max(ne, 1)
        pts.append(np.asarray(t2, float) + s[:, None] * (nxt - np.asarray(t2, float)))
        nrm.append(np.repeat([sign * normals[k]], ne, axis=0))
        use.append(np.ones(ne, bool))


def points2d(part: Part2p5D, lay: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contourpunten (2D) met de buitennormaal van het materiaal; `use` is False voor scherpe hoekpunten
    (die liggen op twee randen tegelijk en zeggen niets extra's)."""
    o = part.outer
    pts, nrm, use = [], [], []
    if o.kind == "circle":
        a = (np.arange(lay["circle"]) + 0.5) / lay["circle"] * 2 * math.pi
        d = np.column_stack([np.cos(a), np.sin(a)])
        pts.append(o.center + o.radius * d)
        nrm.append(d)
        use.append(np.ones(len(a), bool))
    else:
        _polygon_points(o.corner_table(), o.normals(), lay, 1.0, pts, nrm, use)
    for h, nh in zip(part.holes, lay["holes"]):
        a = (np.arange(nh) + 0.5) / nh * 2 * math.pi
        d = np.column_stack([np.cos(a), np.sin(a)])
        pts.append(np.array([h.x, h.y]) + h.d / 2 * d)
        nrm.append(-d)  # het materiaal ligt buiten het gat
        use.append(np.ones(nh, bool))
    for s, ls in zip(part.slots, lay["slots"]):  # sleuven: het materiaal ligt buiten de uitsparing
        prof = s.profile()
        _polygon_points(prof.corner_table(), prof.normals(), ls, -1.0, pts, nrm, use)
    return np.vstack(pts), np.vstack(nrm), np.concatenate(use)


def outer_count(lay: dict) -> int:
    """Aantal punten op de buitencontour (de eerste in points2d)."""
    return lay["circle"] if "circle" in lay else sum(lay["arcs"]) + sum(lay["edges"])


def _outer_points(part: Part2p5D, lay: dict, d: float) -> tuple[np.ndarray, np.ndarray]:
    """Punten van de buitencontour d mm naar binnen, met dezelfde opbouw als bij d = 0 (een afronding die
    daar scherp wordt, geeft herhaalde hoekpunten); en per punt of het meetelt."""
    o = part.outer
    if o.kind == "circle":
        a = (np.arange(lay["circle"]) + 0.5) / lay["circle"] * 2 * math.pi
        return o.center + (o.radius - d) * np.column_stack([np.cos(a), np.sin(a)]), np.ones(len(a), bool)
    pts, nrm, use = [], [], []
    _polygon_points(o.inset(d).corner_table(), o.normals(), lay, 1.0, pts, nrm, use)
    return np.vstack(pts), np.concatenate(use)


@dataclass
class Rims:
    """Alle modelpunten in 3D. Per contourpunt (index in points2d) een onderrand (niveau 0) en een bovenrand;
    een afgeschuinde of afgeronde bovenrand (V17) geeft de buitencontour er niveaus tussen: de schouder en de
    boog, elk naar binnen verschoven."""

    P: np.ndarray  # (R, 3)
    point: np.ndarray  # (R,) contourpunt
    level: np.ndarray  # (R,) 0 = onderrand
    use: np.ndarray  # (R,) False: scherp hoekpunt (ligt op twee randen tegelijk)


def rims(part: Part2p5D, lay: dict) -> Rims:
    """Rij voor rij: de onderrand van alle contouren, de tussenniveaus van de buitencontour en de bovenrand van
    alle contouren. Een prisma geeft dus eerst alle onder- en dan alle bovenranden.

    Met treden ligt de bovenrand per punt op de hoogte van het stuk waar het bij hoort; die indeling staat per
    ronde van de fit vast (`lay["cells"]`), anders springt de hoogte van een punt als de lijn eroverheen
    schuift. Daarna per trede punten langs de lijn (bovenrand van het hoge deel) en op de twee verticale randen
    waar de trede de contour snijdt: die vormen in zijaanzichten de silhouetrand en leggen de lijn vast."""
    p2, _, use = points2d(part, lay)
    n, no = len(p2), outer_count(lay)
    levels = part.outer_levels()
    P, point, level, us = [np.column_stack([p2, np.zeros(n)])], [np.arange(n)], [np.zeros(n, int)], [use]
    for j, (z, d) in enumerate(levels[1:], start=1):
        q, u = (p2[:no], use[:no]) if d == 0 else _outer_points(part, lay, d)
        if j < len(levels) - 1:  # tussenniveau: alleen de buitencontour
            P.append(np.column_stack([q, np.full(no, z)]))
            point.append(np.arange(no))
            level.append(np.full(no, j))
            us.append(u)
        else:
            zt = np.full(n, z)
            if part.steps:
                cells = lay.get("cells")
                cells = cell_of(part, p2) if cells is None or len(cells) != n else cells
                for k, st in enumerate(part.steps):
                    zt[cells == k] = st.height
            P.append(np.column_stack([np.vstack([q, p2[no:]]), zt]))
            point.append(np.arange(n))
            level.append(np.full(n, j))
            us.append(np.concatenate([u, use[no:]]))
    for st, ls in zip(part.steps, lay.get("steps", [])):
        q = part.step_crossings(st)
        if q is None:  # ongeldig model (de fit geeft dan een strafwaarde): punten op één plek
            q = np.repeat([st.offset * st.normal()], 2, axis=0)
        t = (np.arange(ls["line"]) + 0.5) / ls["line"]
        zs = st.height + (np.arange(ls["vert"]) + 0.5) / ls["vert"] * (part.height - st.height)
        extra = [np.column_stack([q[0] + t[:, None] * (q[1] - q[0]), np.full(len(t), part.height)])]
        extra += [np.column_stack([np.repeat([q[i]], len(zs), axis=0), zs]) for i in (0, 1)]
        e = np.vstack(extra)
        P.append(e)
        point.append(np.full(len(e), -1))
        level.append(np.full(len(e), -1))
        us.append(np.ones(len(e), bool))
    return Rims(np.vstack(P), np.concatenate(point), np.concatenate(level), np.concatenate(us))


def points3d(part: Part2p5D, lay: dict) -> np.ndarray:
    """De modelpunten in 3D (zie rims)."""
    return rims(part, lay).P


def boundary_status(part: Part2p5D, K: np.ndarray, vd: list, lay: dict, tol: float = 1.0) -> list[np.ndarray]:
    """Per foto welke modelpunten de silhouetrand vormen (zie de moduletekst). Kijkt de wand bij een punt van
    de camera weg, dan vormt een van de hogere niveaus de rand: het niveau dat in beeld het verst naar buiten
    ligt (bij een afschuining in lage foto's de bovenrand, in hoge de schouder)."""
    p2, nrm, _ = points2d(part, lay)
    rim = rims(part, lay)
    n, R = len(p2), len(rim.P)
    n_lev = int(rim.level.max()) + 1
    idx = np.full((n, n_lev), -1)
    contour = rim.point >= 0  # de punten van een trede hebben geen keuze: ze doen altijd mee
    idx[rim.point[contour], rim.level[contour]] = np.flatnonzero(contour)
    upper = idx[:, 1:]
    out = []
    for v in vd:
        facing = np.sum((v.pose.center[:2] - p2) * nrm, axis=1) > 0
        sd = signed_dist(silhouette.render(part, K, v).astype(bool))
        uv, z = project(rim.P, v.pose, K)
        uv = uv - [v.x0, v.y0]
        h, w = v.fg.shape
        ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
        d = np.full(R, -np.inf)
        d[ok] = _bilinear(sd, uv[ok])
        if upper.shape[1] == 1:
            top = upper[:, 0]
        else:
            du = np.where(upper >= 0, d[np.maximum(upper, 0)], -np.inf)
            top = upper[np.arange(n), np.argmax(du, axis=1)]
        pick = ~contour
        pick[np.where(facing, idx[:, 0], top)] = True
        on = pick & rim.use & ok
        on[on] = np.abs(d[on]) < tol
        out.append(on)
    return out


@dataclass
class EdgeFit:
    part: Part2p5D
    names: list[str]  # parameters (zie silhouette._params)
    x: np.ndarray
    jac: np.ndarray  # Jacobiaan van de (verliesgewogen) residuen bij de oplossing
    residuals: np.ndarray
    view_of: np.ndarray  # per residu de index van de foto
    accepted: bool  # False: de pixelfit bleef staan (rand van het vertrouwensgebied, of een fout)
    note: str = ""
    extra: dict = field(default_factory=dict)


def _free_params(part: Part2p5D, base: np.ndarray) -> list[silhouette.Param]:
    """De parameters die de fit mag verzetten (zie de moduletekst); rotatie is star en blijft altijd vrij."""
    free = []
    for p in silhouette._params(part):
        v = silhouette._get(part, base, p.name)
        ok = True
        if p.name != "rot":
            for step in (-FREEZE_PROBE_MM, FREEZE_PROBE_MM):
                if v + step < p.lower:
                    continue
                q = silhouette._set(part, base, p.name, v + step)
                if not q.is_valid():
                    ok = False
        if ok:
            free.append(p)
    return free


class _Problem:
    """De residuen als functie van de parametervector, met vaste punten en vaste randstatus."""

    def __init__(self, part: Part2p5D, K: np.ndarray, vd: list, density: float = DENSITY,
                 mm_per_px: float | None = None):
        self.K, self.vd = K, vd
        self.fillet_trust = max(TRUST_MM, TRUST_FILLET_PX * mm_per_px) if mm_per_px else TRUST_MM
        self.base = part.outer.angles.copy()
        self.params = _free_params(part, self.base)
        self.frozen = [p.name for p in silhouette._params(part) if p.name not in {q.name for q in self.params}]
        self.template = part
        self.lay = layout(part, density)
        self.fields = []
        for v in vd:
            sf, sb = signed_dist(v.fg), signed_dist(v.bg)
            self.fields.append((sf, sf + sb))  # afstand tot de maskerrand; breedte van de strook zonder bewijs
        self.status: list[np.ndarray] = []

    def build(self, x: np.ndarray) -> Part2p5D:
        p = self.template
        for prm, val in zip(self.params, x):
            p = silhouette._set(p, self.base, prm.name, float(val))
        return p

    def x_of(self, part: Part2p5D) -> np.ndarray:
        return np.array([silhouette._get(part, self.base, p.name) for p in self.params])

    def set_status(self, part: Part2p5D) -> None:
        if part.steps:  # welk punt bij welk stuk hoort, ligt per ronde vast (zie rims)
            self.lay["cells"] = cell_of(part, points2d(part, self.lay)[0])
        self.status = boundary_status(part, self.K, self.vd, self.lay)

    def view_index(self, views: list[int] | None = None) -> np.ndarray:
        views = range(len(self.vd)) if views is None else views
        return np.concatenate([np.full(int(self.status[i].sum()), i) for i in views])

    def residuals(self, x: np.ndarray, views: list[int] | None = None) -> np.ndarray:
        views = range(len(self.vd)) if views is None else views
        n_on = sum(int(self.status[i].sum()) for i in views)
        p = self.build(x)
        if not p.is_valid():
            return np.full(n_on, 20.0)
        P = points3d(p, self.lay)
        out = []
        for i in views:
            v, (sf, band), on = self.vd[i], self.fields[i], self.status[i]
            uv, _ = project(P[on], v.pose, self.K)
            h, w = v.fg.shape
            uv = np.clip(uv - [v.x0, v.y0], 0, [w - 1, h - 1])
            r = _bilinear(sf, uv)
            g = _bilinear(band, uv)
            wgt = np.clip((4.0 - g) / 2.0, 0.0, 1.0)  # geen zekere mat binnen ~3 px: geen bewijs
            # verder dan ~1,5 px is de strook geen menging van object en mat meer, maar gebrek aan bewijs
            out.append(wgt * np.clip(r - BETA * np.clip(g, 0.0, BAND_MAX), -15.0, 15.0))
        return np.concatenate(out) if out else np.zeros(0)

    def bounds(self, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = [], []
        for p, v in zip(self.params, x0):
            d = (math.radians(TRUST_DEG) if p.name == "rot"
                 else math.radians(TRUST_SLOT_DEG) if p.name.startswith(("sa", "ta"))
                 else TRUST_TOP_MM if p.name == "top" else TRUST_STEP_MM if p.name.startswith("to") else TRUST_MM)
            up = self.fillet_trust if p.name.startswith(("fil", "sr")) else d
            lo.append(max(v - d, p.lower))
            hi.append(v + up)
        return np.array(lo), np.array(hi)


def _solve(prob: _Problem, x0: np.ndarray, lb, ub, views=None, max_nfev: int = 60):
    x0 = np.clip(x0, lb + 1e-9, ub - 1e-9)
    return least_squares(lambda x: prob.residuals(x, views), x0, jac="2-point", bounds=(lb, ub), loss=LOSS,
                         f_scale=F_SCALE, diff_step=2e-4, x_scale="jac", max_nfev=max_nfev)


def fit(part: Part2p5D, K: np.ndarray, vd: list, rounds: int = 3, log=None, mm_per_px: float | None = None) -> EdgeFit:
    """Verfijnt de pixelfit `part` op de randafstanden (zie de moduletekst). `mm_per_px`: resolutie op het
    object, voor het vertrouwensgebied van de afrondingen (TRUST_FILLET_PX)."""
    prob = _Problem(part, K, vd, mm_per_px=mm_per_px)
    x_pix = prob.x_of(part)
    lb, ub = prob.bounds(x_pix)
    cur, res = part, None
    try:
        for _ in range(rounds):
            prob.set_status(cur)
            x_start = prob.x_of(cur)
            res = _solve(prob, x_start, lb, ub)
            cur = prob.build(res.x)
            if np.max(np.abs(res.x - x_start)) < 0.005:
                break
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as e:
        return EdgeFit(part, [p.name for p in prob.params], x_pix, np.zeros((0, len(x_pix))), np.zeros(0),
                       np.zeros(0, int), False, f"randfit mislukt: {e}")
    at_edge = np.flatnonzero((np.abs(res.x - lb) < 1e-6) | (np.abs(res.x - ub) < 1e-6))
    # een ondergrens die al in de pixelfit gold (een afronding van 0) telt niet als 'weggedreven'
    at_edge = [i for i in at_edge if not (abs(res.x[i] - prob.params[i].lower) < 1e-6)]
    names = [p.name for p in prob.params]
    ef = EdgeFit(cur, names, res.x, res.jac, res.fun, prob.view_index(), True)
    ef.extra["problem"] = prob
    if at_edge:
        ef.part, ef.accepted = part, False
        ef.note = "randfit liep tegen de grens van het vertrouwensgebied (" + ", ".join(names[i] for i in at_edge) + ")"
    if log:
        frozen = f"; vast (contour anders ongeldig): {', '.join(prob.frozen)}" if prob.frozen else ""
        log(("randfit: " if ef.accepted else "randfit niet gebruikt: ") + (ef.note or
            f"{len(res.fun)} randpunten over {len(vd)} foto's") + frozen)
    return ef


def jackknife(ef: EdgeFit, groups: int = 6, exact: bool = True, max_nfev: int = 20) -> np.ndarray | None:
    """Covariantie van de parameters: steeds één groep foto's weglaten.

    Fouten die per foto samenhangen (een iets verkeerde pose, een masker dat een wand net mist) maken
    de formele covariantie uit de Jacobiaan te optimistisch: honderden randpunten in één foto zijn niet
    onafhankelijk. De spreiding over de weggelaten groepen vangt dat wel.

    Per groep wordt opnieuw opgelost, vanuit de gezamenlijke oplossing (~5-20 s voor 6 groepen). Zonder
    `exact` alleen één Gauss-Newtonstap: zonder groep g verschuift het optimum over (J₋ᵍᵀJ₋ᵍ)⁻¹ Jᵍᵀ rᵍ.
    Dat is snel, maar onderschat de spreiding (afrondingen tot ~20x): de fit is niet lineair genoeg.
    """
    prob: _Problem | None = ef.extra.get("problem")
    if prob is None or not ef.accepted or len(ef.residuals) == 0:
        return None
    n_views = len(prob.vd)
    groups = min(groups, n_views)
    if groups < 3:
        return None
    thetas = []
    if exact:
        lb, ub = prob.bounds(ef.x)
        lb, ub = np.minimum(lb, ef.x - 1e-6), np.maximum(ub, ef.x + 1e-6)
        for g in range(groups):
            keep = [i for i in range(n_views) if i % groups != g]
            thetas.append(_solve(prob, ef.x, lb, ub, views=keep, max_nfev=max_nfev).x)
        # zonder een groep foto's verschuift het optimum altijd iets; blijft een oplossing staan, dan zat
        # de solver vast en zegt de spreiding niets
        if sum(np.max(np.abs(t - ef.x)) > 1e-9 for t in thetas) < groups - 1:
            return None
    else:
        J = ef.jac
        z = (ef.residuals / F_SCALE) ** 2
        r = ef.residuals / np.sqrt(1.0 + z)  # Cauchy: residu gewogen met sqrt(ρ'), zoals de Jacobiaan
        grp = ef.view_of % groups
        for g in range(groups):
            out = grp == g
            A = J[~out].T @ J[~out]
            b = J[out].T @ r[out]
            try:
                thetas.append(ef.x + np.linalg.lstsq(A, b, rcond=None)[0])
            except np.linalg.LinAlgError:
                return None
    T = np.array(thetas)
    d = T - T.mean(axis=0)
    return (groups - 1) / groups * d.T @ d
