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
* Residu = afstand (px, subpixel via bilineaire interpolatie) van het punt tot de rand, alleen waar ook
  zekere mat vlakbij is (bewijs). Sinds v0.9 (V2-open) komt die afstand uit de grijswaarden zelf: de zachte
  objectfractie rond de rand (masks._soft_alpha) geeft per pixel de afstand tot de rand (edge_distance), ook
  tussen twee pixels in. Waar die niets zegt (geen contrast, verzadigd), de afstand tot de maskerrand, met de
  rand op een fractie `BETA` van de strook zonder bewijs tussen object en zekere mat (de regel van v0.5-v0.8,
  afgesteld op grijze onderdelen; bij donkere en gekleurde paste hij niet, ROUTE-A-VERBETERPUNTEN §3h-§3i).
* Randen binnen het object (V16, v0.9): de bovenrand van een verzinking is in het silhouet niet te zien, maar
  in de grijswaarden wel. Per ronde gemeten waar die rand in elke foto ligt (measure_inner), daarna als vaste
  doelen in dezelfde kleinste kwadraten.
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
from scipy.special import ndtri

from . import silhouette
from .cadhelpers import afgeronde_hoeken
from .calib import project
from .profile import Part2p5D

BETA = 0.2  # waar in de strook zonder bewijs de rand ligt (0 = maskerrand, 1 = begin zekere mat)
BAND_MAX = 1.5  # px: breder telt als gebrek aan bewijs, niet als menging
# De rand uit de zachte objectfractie (V2-open, v0.9; zie edge_distance), per pixel gemengd met de maskerrand naar
# hun informatie; de onzekerheid (1σ) van de maskerrand met de BETA-regel
EDGE_MODEL = "alpha"  # "alpha" of "beta" (de regel van v0.5-v0.8)
MASK_SIGMA_PX = 0.3
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
# Bewijs rond een gat of sleuf (zie `evidence`): minder dan dit deel van de randpunten met zekere mat ernaast,
# of een systematische fout die hierdoor minstens AMP_MAX keer groter is dan met bewijs rondom: 'zonder bewijs'
EVIDENCE_MIN, AMP_MAX = 0.10, 8.0
# Randen binnen het object (V16, zie measure_inner): zo ver (px) langs de normaal gezocht, en alleen bij zoveel
# contrast (grijswaarden) tussen de kegel van een verzinking en het bovenvlak
INNER_SEARCH_PX, INNER_MIN_CONTRAST = 5.0, 10.0


def signed_dist(mask: np.ndarray) -> np.ndarray:
    """Negatief binnen, positief buiten; nul op de pixelrand (px)."""
    m8 = mask.astype(np.uint8)
    d_out = cv2.distanceTransform(1 - m8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_in = cv2.distanceTransform(m8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(mask, 0.5 - d_in, d_out - 0.5).astype(np.float32)


def edge_blur(alpha: np.ndarray, default: float = 1.2) -> float:
    """Onscherpte σ (px) van de randen in een foto, uit de zachte objectfractie: bij een Gauss-vervaagde rand is
    |∇alpha| op de rand 1 / (σ √(2π))."""
    ok = np.isfinite(alpha)
    a = np.where(ok, alpha, 0.0).astype(np.float32)
    gy, gx = np.gradient(a)
    sel = ok & (cv2.erode(ok.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (np.abs(a - 0.5) < 0.15)
    if np.count_nonzero(sel) < 50:
        return default
    return float(np.clip(0.3989 / max(float(np.median(np.hypot(gx, gy)[sel])), 1e-3), 0.5, 4.0))


def edge_distance(sf: np.ndarray, band: np.ndarray, alpha: np.ndarray | None,
                  alpha_w: np.ndarray | None) -> np.ndarray:
    """Afstand tot de rand per pixel (px, + = buiten het object): het residu van de randfit (V2-open, v0.9).

    Uit de zachte objectfractie (masks._soft_alpha): een Gauss-vervaagde rand op afstand d buiten een pixel geeft
    daar alpha = Φ(−d/σ), dus d = −σ Φ⁻¹(alpha), met σ de onscherpte van de foto (edge_blur). Tussen twee pixels
    is die afstand lineair, dus ook de bilineaire interpolatie: een subpixelrand, zonder de trappen en hapjes van
    het binaire masker. Per pixel gemengd met de afstand tot de maskerrand volgens de BETA-regel (de rand op een
    vaste fractie van de strook zonder bewijs), naar hun informatie: alpha telt waar ze de randplaats bepaalt
    (1/variantie × de helling van Φ in het kwadraat: op de rand veel, verzadigd nauwelijks), het masker waar
    alpha niets zegt (geen contrast, verzadigd, of geen alpha)."""
    mask_d = sf - BETA * np.clip(band, 0.0, BAND_MAX)
    if EDGE_MODEL != "alpha" or alpha is None or alpha_w is None:
        return mask_d
    a = alpha.astype(np.float32)
    ok = np.isfinite(a)
    if np.count_nonzero(ok) < 50:
        return mask_d
    sv = edge_blur(a)
    z = ndtri(np.clip(np.where(ok, a, 0.5), 0.02, 0.98)).astype(np.float32)
    info = np.where(ok, alpha_w.astype(np.float32) * np.exp(-z * z) / (2 * np.pi) / sv ** 2, 0.0)
    lam = info / (info + 1.0 / MASK_SIGMA_PX ** 2)
    return (lam * (-sv * z) + (1.0 - lam) * mask_d).astype(np.float32)


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
           "slots": [_polygon_layout(s.corner_table(), density, 0) for s in part.slots],
           "csk": [max(24, int(math.pi * h.csk * density)) if h.csk > 0 else 0 for h in part.holes]}
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
            start = no
            for h, nh in zip(part.holes, lay["holes"]):  # verzinking (V16): het doorgaande gat eindigt eronder
                zt[start:start + nh] -= h.csk_depth
                start += nh
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


def csk_points(part: Part2p5D, lay: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randen binnen het object (V16): punten op de bovenrand van elke verzinking (3D, op het bovenvlak), hun
    richting naar buiten (2D, radiaal) en per punt het gat."""
    pts, dirs, owner = [], [], []
    for i, (h, n) in enumerate(zip(part.holes, lay.get("csk", []))):
        if n == 0 or h.csk <= 0:
            continue
        a = (np.arange(n) + 0.5) / n * 2 * math.pi
        d = np.column_stack([np.cos(a), np.sin(a)])
        z = part.height_at(h.x, h.y) if part.steps else part.height
        pts.append(np.column_stack([h.x + h.csk / 2 * d[:, 0], h.y + h.csk / 2 * d[:, 1], np.full(n, z)]))
        dirs.append(d)
        owner.append(np.full(n, i))
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0, int)
    return np.vstack(pts), np.vstack(dirs), np.concatenate(owner)


def measure_inner(part: Part2p5D, K: np.ndarray, vd: list, lay: dict,
                  search_px: float = INNER_SEARCH_PX) -> list[tuple]:
    """Waar liggen de randen binnen het object in de foto's (V16)? Per foto (puntindex, gemeten randplaats in de ROI,
    normaal in beeld, gewicht), voor de punten van csk_points.

    Een verzinking is in de grijswaarden een ring: de kegel staat schuin en is dus anders belicht dan het bovenvlak.
    Per punt het profiel langs de normaal (in beeld), tot `search_px` en hooguit 80% van de breedte van de kegel in
    beeld aan beide kanten; de rand ligt waar het profiel halverwege de kegel en het bovenvlak is, het dichtst bij
    het model. Zonder contrast (INNER_MIN_CONTRAST grijswaarden) of met een stuk zonder grijswaarden geen meting."""
    P, D, owner = csk_points(part, lay)
    empty = (np.zeros(0, int), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
    if not len(P):
        return [empty for _ in vd]
    Q = P.copy()  # de onderkant van de kegel langs dezelfde richting: daar begint het doorgaande gat
    for i in np.unique(owner):
        h, sel = part.holes[i], owner == i
        Q[sel, :2] = np.array([h.x, h.y]) + h.d / 2 * D[sel]
        Q[sel, 2] = P[sel, 2] - h.csk_depth
    s = np.arange(-search_px, search_px + 1e-9, 0.25)
    out = []
    for v in vd:
        g = getattr(v, "gray", None)
        if g is None:
            out.append(empty)
            continue
        off = np.array([v.x0, v.y0], float)
        uv0, z0 = project(P, v.pose, K)
        uv1, _ = project(P + np.column_stack([0.3 * D, np.zeros(len(P))]), v.pose, K)
        uvq, _ = project(Q, v.pose, K)
        nrm = uv1 - uv0
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        uv0 = uv0 - off
        reach = np.minimum(search_px, 0.8 * np.sum((uv0 + off - uvq) * nrm, axis=1))
        hh, ww = g.shape
        m = search_px + 1
        ok = (z0 > 0) & (reach >= 2.0) & (uv0[:, 0] > m) & (uv0[:, 1] > m) & (uv0[:, 0] < ww - m - 1) \
            & (uv0[:, 1] < hh - m - 1)
        idx = np.flatnonzero(ok)
        if not len(idx):
            out.append(empty)
            continue
        q = uv0[idx, None, :] + s[None, :, None] * nrm[idx, None, :]
        prof = ndimage.map_coordinates(g.astype(np.float32), [q[..., 1].ravel(), q[..., 0].ravel()],
                                       order=1).reshape(len(idx), len(s))
        keep, pos, wts = [], [], []
        for k, j in enumerate(idx):
            p, r = prof[k], reach[j]
            outside, inside = (s >= r / 2) & (s <= r), (s <= -r / 2) & (s >= -r)
            span = np.abs(s) <= r
            if not np.all(np.isfinite(p[span])):
                continue
            l_out, l_in = float(np.median(p[outside])), float(np.median(p[inside]))
            c = l_in - l_out
            if abs(c) < INNER_MIN_CONTRAST:
                continue
            a = (p - l_out) / c - 0.5
            cross = np.flatnonzero(span[:-1] & span[1:] & (np.sign(a[:-1]) != np.sign(a[1:])))
            if not len(cross):
                continue
            t = cross[np.argmin(np.abs(s[cross]))]
            se = s[t] + a[t] / (a[t] - a[t + 1]) * (s[t + 1] - s[t])
            keep.append(j)
            pos.append(uv0[j] + se * nrm[j])
            wts.append(min(1.0, (abs(c) - INNER_MIN_CONTRAST) / INNER_MIN_CONTRAST))
        if not keep:
            out.append(empty)
            continue
        keep = np.array(keep)
        out.append((keep, np.array(pos), nrm[keep], np.array(wts)))
    return out


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


@dataclass
class Evidence:
    """Bewijs rond een gat of sleuf. `fraction`: welk deel van de randpunten op de silhouetrand (over alle foto's)
    zekere mat naast zich heeft. `amp_size`, `amp_pos`: hoeveel groter de systematische fout van de maat en van de
    positie daardoor is dan met bewijs rondom (1 = rondom; zie `evidence`)."""

    fraction: float
    amp_size: float
    amp_pos: float

    @property
    def weak(self) -> bool:
        return max(self.amp_size, self.amp_pos) >= AMP_MAX


def features(part: Part2p5D, lay: dict) -> list[tuple[str, int, np.ndarray, list[str], list[dict], list[dict]]]:
    """Gaten en sleuven: (soort, index, hun punten in points2d, parameters, maatgrootheden, positiegrootheden). Een
    grootheid is een lineaire combinatie van parameters, bijv. de hartafstand van een sleuf {sl: 1, sw: -1}."""
    out, start = [], outer_count(lay)
    for i, n in enumerate(lay["holes"]):
        out.append(("gat", i, np.arange(start, start + n), [f"hx{i}", f"hy{i}", f"hd{i}"], [{f"hd{i}": 1.0}],
                    [{f"hx{i}": 1.0}, {f"hy{i}": 1.0}]))
        start += n
    for i, (s, ls) in enumerate(zip(part.slots, lay["slots"])):
        n = sum(ls["arcs"]) + sum(ls["edges"])
        names = [f"sx{i}", f"sy{i}", f"sl{i}", f"sw{i}", f"sa{i}"] + ([f"sr{i}"] if s.kind == "rechthoek" else [])
        sizes = [{f"sw{i}": 1.0}, {f"sl{i}": 1.0, f"sw{i}": -1.0} if s.kind == "sleuf" else {f"sl{i}": 1.0}]
        out.append(("sleuf", i, np.arange(start, start + n), names, sizes, [{f"sx{i}": 1.0}, {f"sy{i}": 1.0}]))
        start += n
    return out


def evidence(prob: "_Problem", x: np.ndarray) -> dict[tuple[str, int], Evidence]:
    """Hoe goed is elk gat en elke sleuf bepaald? Per soort en index (zie `features`).

    Een gat in een zwart onderdeel boven een zwart vak laat geen mat zien: de randfit heeft er dan geen bewijs,
    en de maat en de plaats komen uit de pixelfit (analyse van de stresstest 'donker', v0.8). Ligt er alleen aan
    één kant zekere mat naast de rand, dan hangen maat en plaats samen: een fout in de maskerrand daar schuift
    het gat en verandert de maat. De jackknife ziet dat niet (alle foto's zien dezelfde mat onder het gat).

    Maat: per randpunt het gewicht van het bewijs, opgeteld over de foto's (zoals in de residuen), en per punt de
    verschuiving van de rand langs de normaal per eenheid van elke parameter van de vorm (in het vlak). Daarmee
    de informatiematrix met bewijs en die met bewijs overal waar het punt op de silhouetrand ligt, elk per
    randpunt genormeerd: de verhouding van de varianties van een maat is het kwadraat van de vergroting. Met
    bewijs rondom is die 1; met een halve ring ongeveer 2,3."""
    part = prob.build(x)
    lay = prob.lay
    p2, nrm, _ = points2d(part, lay)
    rim = rims(part, lay)
    n_pts = len(p2)
    W, N, E = np.zeros(n_pts), np.zeros(n_pts), np.zeros(n_pts)
    for i, v in enumerate(prob.vd):
        on = prob.status[i] & (rim.point >= 0)
        if not on.any():
            continue
        uv, _ = project(rim.P[on], v.pose, prob.K)
        h, w = v.fg.shape
        uv = np.clip(uv - [v.x0, v.y0], 0, [w - 1, h - 1])
        wgt = np.clip((4.0 - _bilinear(prob.fields[i][1], uv)) / 2.0, 0.0, 1.0)
        pt = rim.point[on]
        np.add.at(W, pt, wgt ** 2)
        np.add.at(N, pt, 1.0)
        np.add.at(E, pt, (wgt > 0.5).astype(float))
    out = {}
    eps = 1e-4
    for kind, i, idx, names, sizes, positions in features(part, lay):
        names = [nm for nm in names if nm in {p.name for p in prob.params} or nm in prob.frozen]
        cols = []
        for nm in names:
            q = silhouette._set(part, prob.base, nm, silhouette._get(part, prob.base, nm) + eps)
            cols.append(np.sum((points2d(q, lay)[0][idx] - p2[idx]) * nrm[idx], axis=1) / eps)
        D = np.column_stack(cols)
        w_i, n_i = W[idx], N[idx]
        frac = float(E[idx].sum() / n_i.sum()) if n_i.sum() > 0 else 0.0

        def amp(qs: list[dict]) -> float:
            if w_i.sum() <= 0 or n_i.sum() <= 0:
                return AMP_MAX
            Aw, Af = D.T @ (w_i[:, None] * D), D.T @ (n_i[:, None] * D)
            worst = 1.0
            for qd in qs:
                g = np.array([qd.get(nm, 0.0) for nm in names])
                try:
                    sw = float(g @ np.linalg.solve(Aw, g)) * w_i.sum()
                    sf = float(g @ np.linalg.solve(Af, g)) * n_i.sum()
                except np.linalg.LinAlgError:
                    return AMP_MAX
                if not (np.isfinite(sw) and sf > 0) or sw < 0:
                    return AMP_MAX
                worst = max(worst, math.sqrt(sw / sf))
            return min(worst, AMP_MAX)

        if frac < EVIDENCE_MIN:  # een paar punten met bewijs zeggen niets over de vorm van het bewijs
            out[(kind, i)] = Evidence(frac, AMP_MAX, AMP_MAX)
        else:
            out[(kind, i)] = Evidence(frac, amp(sizes), amp(positions))
    return out


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
            band = sf + sb  # breedte van de strook zonder bewijs
            self.fields.append((edge_distance(sf, band, getattr(v, "alpha", None), getattr(v, "alpha_w", None)),
                                band))
        self.status: list[np.ndarray] = []
        self.inner: list[tuple] = []  # randen binnen het object (V16), per foto; zie measure_inner

    def freeze(self, names: list[str]) -> None:
        """Deze parameters niet meer fitten (ze houden de waarde van het startmodel)."""
        self.params = [p for p in self.params if p.name not in names]
        self.frozen += [n for n in names if n not in self.frozen]

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
        # randen binnen het object: per ronde gemeten waar ze in de foto's liggen, dan als vaste doelen (V16)
        self.inner = measure_inner(part, self.K, self.vd, self.lay) if any(self.lay.get("csk", [])) else []

    def n_residuals(self, i: int) -> int:
        """Aantal residuen van foto i: punten op de silhouetrand en gemeten randen binnen het object."""
        return int(self.status[i].sum()) + (len(self.inner[i][0]) if self.inner else 0)

    def view_index(self, views: list[int] | None = None) -> np.ndarray:
        views = range(len(self.vd)) if views is None else views
        return np.concatenate([np.full(self.n_residuals(i), i) for i in views])

    def residuals(self, x: np.ndarray, views: list[int] | None = None) -> np.ndarray:
        views = range(len(self.vd)) if views is None else views
        n_on = sum(self.n_residuals(i) for i in views)
        p = self.build(x)
        if not p.is_valid():
            return np.full(n_on, 20.0)
        P = points3d(p, self.lay)
        P_in = csk_points(p, self.lay)[0] if self.inner else None
        out = []
        for i in views:
            v, (sf, band), on = self.vd[i], self.fields[i], self.status[i]
            uv, _ = project(P[on], v.pose, self.K)
            h, w = v.fg.shape
            uv = np.clip(uv - [v.x0, v.y0], 0, [w - 1, h - 1])
            r = _bilinear(sf, uv)
            g = _bilinear(band, uv)
            wgt = np.clip((4.0 - g) / 2.0, 0.0, 1.0)  # geen zekere mat binnen ~3 px: geen bewijs
            out.append(wgt * np.clip(r, -15.0, 15.0))
            if P_in is not None and len(self.inner[i][0]):
                # afstand (px, + = naar buiten) van het modelpunt tot de gemeten rand, langs de normaal in beeld
                idx, e, nrm, wt = self.inner[i]
                uvi, _ = project(P_in[idx], v.pose, self.K)
                out.append(wt * np.clip(np.sum((uvi - [v.x0, v.y0] - e) * nrm, axis=1), -15.0, 15.0))
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
    # een gat of sleuf zonder bewijs rondom blijft staan: zijn parameters zouden anders wegdrijven tot de rand
    # van het vertrouwensgebied, en dan bleef voor het hele onderdeel de pixelfit staan
    prob.set_status(part)
    ev = evidence(prob, prob.x_of(part))
    frozen_invalid = list(prob.frozen)
    weak = [(kind, i, names) for (kind, i, _, names, _, _) in features(part, prob.lay) if ev[(kind, i)].weak]
    # met het gat ook zijn verzinking (V16): wat er rond zo'n gat in de foto's te zien is, is te zwak om op te fitten
    prob.freeze([nm for kind, i, names in weak for nm in names
                 + ([f"hk{i}"] if kind == "gat" and part.holes[i].csk > 0 else [])])
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
                       np.zeros(0, int), False, f"randfit mislukt: {e}", {"evidence": ev})
    at_edge = np.flatnonzero((np.abs(res.x - lb) < 1e-6) | (np.abs(res.x - ub) < 1e-6))
    # een ondergrens die al in de pixelfit gold (een afronding van 0) telt niet als 'weggedreven'
    at_edge = [i for i in at_edge if not (abs(res.x[i] - prob.params[i].lower) < 1e-6)]
    names = [p.name for p in prob.params]
    ef = EdgeFit(cur, names, res.x, res.jac, res.fun, prob.view_index(), True)
    ef.extra["problem"] = prob
    if at_edge:
        ef.part, ef.accepted = part, False
        ef.note = "randfit liep tegen de grens van het vertrouwensgebied (" + ", ".join(names[i] for i in at_edge) + ")"
    # het bewijs bij het resultaat (voor de U95 per maat); zonder randfit dat bij de pixelfit
    ef.extra["evidence"] = evidence(prob, res.x) if ef.accepted else ev
    if log:
        frozen = f"; vast (contour anders ongeldig): {', '.join(frozen_invalid)}" if frozen_invalid else ""
        if weak:
            frozen += "; vast (geen bewijs rond de rand): " + ", ".join(f"{k} {i + 1}" for k, i, _ in weak)
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
