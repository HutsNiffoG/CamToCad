"""Model-gebaseerde verfijning: het 2,5D-model direct fitten op de silhouetten in alle foto's.

Analysis-by-synthesis (ARCHITECTURE.md §6.8): voor een kandidaatmodel rendert dit module per
foto het verwachte silhouet en telt de pixels die niet kloppen met de maskers:

    E = |model ∩ zekere mat| + w · |model ∩ onbekend| + |niet-model ∩ object|

Over tientallen foto's en duizenden randpixels levert dat maten met subpixel-nauwkeurigheid,
zonder dat het object textuur nodig heeft. Een visual hull dient alleen als startpunt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .calib import Pose, project
from .masks import ViewMasks
from .cadhelpers import afgeronde_hoeken
from .profile import Hole, Part2p5D, Step, TopEdge, circle_polygon

SHIFT = 4  # subpixel-coördinaten voor cv2.fillPoly (1/16 pixel)


@dataclass
class ViewData:
    pose: Pose
    fg: np.ndarray  # uitsnede (ROI) van het objectmasker
    bg: np.ndarray  # zekere mat
    unk: np.ndarray  # onbekend (op de mat, maar zonder textuur)
    x0: int
    y0: int


def prepare(views: list[tuple[Pose, ViewMasks]], K: np.ndarray, part: Part2p5D, margin_mm: float = 8.0,
            z_max: float | None = None, margin_px: int = 12) -> list[ViewData]:
    """Snijdt per foto een ROI uit rond het (verwachte) object."""
    outline = part.outer.outline()
    lo, hi = outline.min(axis=0) - margin_mm, outline.max(axis=0) + margin_mm
    z_top = z_max if z_max is not None else part.height * 1.5 + margin_mm
    box = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (0.0, z_top)])
    out = []
    for pose, m in views:
        uv, z = project(box, pose, K)
        if np.any(z <= 0):
            continue
        h, w = m.fg.shape
        x0 = int(max(np.floor(uv[:, 0].min()) - margin_px, 0))
        y0 = int(max(np.floor(uv[:, 1].min()) - margin_px, 0))
        x1 = int(min(np.ceil(uv[:, 0].max()) + margin_px, w))
        y1 = int(min(np.ceil(uv[:, 1].max()) + margin_px, h))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        sl = (slice(y0, y1), slice(x0, x1))
        bg = (m.edge_bg if m.edge_bg is not None else m.bg)[sl]
        fg = m.fg[sl]
        unk = m.valid[sl] & ~fg & ~bg
        if m.amb is not None:
            # Waar het object dezelfde grijswaarde heeft als de mat (zwart op zwart), telt een pixel
            # nergens mee, ook niet als het masker hem voor de startcontour heeft opgevuld: zo trekt hij
            # een gat niet groter of een buitenrand niet naar binnen, en bepaalt het bewijs eromheen de vorm.
            amb = m.amb[sl]
            fg, unk = fg & ~amb, unk & ~amb
        out.append(ViewData(pose, fg, bg, unk, x0, y0))
    return out


def _to_fixed(uv: np.ndarray, x0: int, y0: int) -> np.ndarray:
    return np.round((uv - [x0, y0]) * (1 << SHIFT)).astype(np.int32)


def _shrink(uv: np.ndarray, delta: float = 0.5) -> np.ndarray | None:
    """Verkleint een gesloten polygoon (beeldcoördinaten) met `delta` pixel (miter-offset).

    cv2.fillPoly vult ook de pixels waar de rand doorheen loopt: het resultaat is aan elke kant
    0,5 px groter dan 'pixelmidden binnen de polygoon'. Zonder deze correctie zou de optimizer het
    model 0,5 px te klein fitten. Geeft None voor (bijna) gedegenereerde polygonen.
    """
    nxt = np.concatenate([uv[1:], uv[:1]])
    area = 0.5 * float(np.sum(uv[:, 0] * nxt[:, 1] - uv[:, 1] * nxt[:, 0]))
    if abs(area) < 1.0:
        return None
    d = nxt - uv
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    s = 1.0 if area > 0 else -1.0
    n = s * np.column_stack([-d[:, 1], d[:, 0]])  # naar binnen gerichte normaal van rand i (van i naar i+1)
    n_prev = np.concatenate([n[-1:], n[:-1]])
    denom = np.maximum(1.0 + np.sum(n * n_prev, axis=1), 0.25)  # miterlengte begrenzen
    return uv + delta * (n + n_prev) / denom[:, None]


def _shrink_quads(q: np.ndarray, delta: float = 0.5) -> np.ndarray:
    """Gevectoriseerde _shrink voor een stapel vierhoeken (Q, 4, 2); gedegenereerde vallen weg."""
    x, y = q[..., 0], q[..., 1]
    area = 0.5 * (np.sum(x * np.roll(y, -1, axis=1), axis=1) - np.sum(y * np.roll(x, -1, axis=1), axis=1))
    keep = np.abs(area) >= 1.0
    q, area = q[keep], area[keep]
    d = np.roll(q, -1, axis=1) - q
    d /= np.linalg.norm(d, axis=2, keepdims=True) + 1e-12
    s = np.where(area > 0, 1.0, -1.0)[:, None]
    n = np.stack([-d[..., 1] * s, d[..., 0] * s], axis=2)
    n_prev = np.roll(n, 1, axis=1)
    denom = np.maximum(1.0 + np.sum(n * n_prev, axis=2), 0.25)
    return q + delta * (n + n_prev) / denom[..., None]


def level_rings(prof, insets, max_step_deg: float = 12.0) -> list[np.ndarray]:
    """Omtrekken van de contour, telkens `inset` mm naar binnen, met punt voor punt dezelfde opbouw (zodat
    de band tussen twee niveaus uit vierhoeken bestaat). Voor inset 0 gelijk aan prof.outline(max_step_deg)."""
    if prof.kind == "circle":
        k = max(24, int(360 / max_step_deg))
        return [circle_polygon(prof.center, prof.radius - d, k) for d in insets]
    tables = [afgeronde_hoeken(prof.inset(d).corner_table()) if d else afgeronde_hoeken(prof.corner_table())
              for d in insets]
    nrm = prof.normals()
    counts = []
    for k in range(prof.n):
        turn = math.degrees(math.acos(float(np.clip(nrm[k - 1] @ nrm[k], -1.0, 1.0))))
        rounded = any(tab[k][1] is not None for tab in tables)
        counts.append(max(2, int(turn / max_step_deg) + 1) + 1 if rounded else 1)
    rings = []
    for tab in tables:
        pts = []
        for k, (t1, m, t2, c, r) in enumerate(tab):
            if counts[k] == 1:
                pts.append(t1)
            elif m is None:
                pts.extend([t1] * counts[k])
            else:
                a1 = math.atan2(t1[1] - c[1], t1[0] - c[0])
                a2 = math.atan2(t2[1] - c[1], t2[0] - c[0])
                am = math.atan2(m[1] - c[1], m[0] - c[0])
                span = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
                if abs((am - a1 + math.pi) % (2 * math.pi) - math.pi) > abs(span) + 1e-6:
                    span = span - math.copysign(2 * math.pi, span)
                for s in np.linspace(0, 1, counts[k]):
                    pts.append((c[0] + r * math.cos(a1 + s * span), c[1] + r * math.sin(a1 + s * span)))
        rings.append(np.array(pts, float))
    return rings


def solids(part: Part2p5D) -> list[np.ndarray]:
    """Het model als stapels ringen (L, N, 3) van onder naar boven: een prisma is er één met twee niveaus, een
    afgeschuinde of afgeronde bovenrand geeft meer niveaus, een trede een stapel per stuk (V17)."""
    out = []
    if part.steps:
        for prof, h in part.cells():
            ring = prof.outline(12.0)
            out.append(np.stack([np.column_stack([ring, np.zeros(len(ring))]),
                                 np.column_stack([ring, np.full(len(ring), h)])]))
        return out
    levels = part.outer_levels()
    rings = level_rings(part.outer, [d for _, d in levels])
    out.append(np.stack([np.column_stack([r, np.full(len(r), z)]) for r, (z, _) in zip(rings, levels)]))
    return out


def render(part: Part2p5D, K: np.ndarray, v: ViewData, geom: list[np.ndarray] | None = None) -> np.ndarray:
    """Silhouet (uint8 0/1) van het model binnen de ROI van een foto. `geom`: solids(part), als die voor
    meerdere foto's al berekend is."""
    h, w = v.fg.shape
    mask = np.zeros((h, w), np.uint8)
    geom = solids(part) if geom is None else geom

    def fill(poly, target, convex=False):
        shrunk = _shrink(poly)
        if shrunk is None:
            return
        pts = _to_fixed(shrunk, v.x0, v.y0)
        if convex:
            cv2.fillConvexPoly(target, pts, 1, cv2.LINE_8, SHIFT)
        else:
            cv2.fillPoly(target, [pts], 1, cv2.LINE_8, SHIFT)

    for stack in geom:
        n_lev, n = stack.shape[:2]
        uv, _ = project(stack.reshape(-1, 3), v.pose, K)
        uv = uv.reshape(n_lev, n, 2)
        fill(uv[0], mask)
        fill(uv[-1], mask)
        # Wanden: vierhoeken van de onderrand naar elk niveau, niet van niveau naar niveau. De banden van een
        # afschuining of afronding zijn in beeld vaak smaller dan een pixel; zo'n vierhoek kan niet 0,5 px
        # kleiner (hij klapt om) en wordt dan te breed getekend. Een band ligt in doorsnede in de driehoek
        # onderrand-niveau-volgend niveau, dus binnen de twee vierhoeken vanaf de onderrand; die liggen
        # binnen het onderdeel (het profiel van de bovenrand is bol) en zijn zo breed als de wand.
        lo = uv[0]
        for hi in uv[1:]:
            quads = np.stack([lo, np.roll(lo, -1, axis=0), np.roll(hi, -1, axis=0), hi], axis=1)
            for q in _to_fixed(_shrink_quads(quads), v.x0, v.y0):
                cv2.fillConvexPoly(mask, q, 1, cv2.LINE_8, SHIFT)
    # doorkijk door gaten: binnen de projectie van zowel de boven- als de onderrand
    tmp_t = np.zeros_like(mask)
    tmp_b = np.zeros_like(mask)
    rings = [circle_polygon((hl.x, hl.y), hl.d / 2, 48) for hl in part.holes]
    rings += list(part.cutouts)
    rings += [s.outline() for s in part.slots]
    for r2 in rings:
        m2 = len(r2)
        c2 = r2.mean(axis=0)
        z_top = part.height_at(float(c2[0]), float(c2[1])) if part.steps else part.height
        uv2, _ = project(np.vstack([np.column_stack([r2, np.full(m2, z_top)]),
                                    np.column_stack([r2, np.zeros(m2)])]), v.pose, K)
        tmp_t[:] = 0
        tmp_b[:] = 0
        fill(uv2[:m2], tmp_t)
        fill(uv2[m2:], tmp_b)
        mask[(tmp_t & tmp_b) > 0] = 0
    return mask


def _mismatch(part: Part2p5D, K: np.ndarray, v: ViewData, geom: list[np.ndarray]) -> tuple[int, int, int]:
    P = render(part, K, v, geom).astype(bool)
    return np.count_nonzero(P & v.bg), np.count_nonzero(P & v.unk), np.count_nonzero(~P & v.fg)


def energy(part: Part2p5D, K: np.ndarray, views: list[ViewData], w_unknown: float = 0.25) -> float:
    # Serieel: per foto is het vooral Python-werk aan kleine arrays; threads maakten het twee keer trager.
    geom = solids(part)
    bg = unk = fg = 0
    for v in views:
        a, b, c = _mismatch(part, K, v, geom)
        bg, unk, fg = bg + a, unk + b, fg + c
    return bg + w_unknown * unk + fg


# ----------------------------------------------------------------------------- initialisatie

def is_top_view(pose: Pose, max_tilt_deg: float = 25.0) -> bool:
    """Kijkt de camera (bijna) loodrecht omlaag? De optische as in mat-coördinaten is R[2]."""
    return float(pose.R[2, 2]) < -math.cos(math.radians(max_tilt_deg))


# ----------------------------------------------------------------------------- parameters

@dataclass
class Param:
    name: str
    step: float  # beginstap
    min_step: float
    lower: float = -math.inf


def _params(part: Part2p5D) -> list[Param]:
    ps = [Param("h", 0.5, 0.01, 0.5)]
    if part.outer.kind == "circle":
        ps += [Param("cx", 0.3, 0.01), Param("cy", 0.3, 0.01), Param("R", 0.3, 0.01, 0.5)]
    else:
        ps.append(Param("rot", math.radians(0.5), math.radians(0.01)))
        ps += [Param(f"off{k}", 0.4, 0.01) for k in range(part.outer.n)]
        ps += [Param(f"fil{k}", 0.4, 0.02, 0.0) for k in range(part.outer.n)]
    for i in range(len(part.holes)):
        ps += [Param(f"hx{i}", 0.3, 0.01), Param(f"hy{i}", 0.3, 0.01), Param(f"hd{i}", 0.3, 0.01, 0.3)]
    for i, s in enumerate(part.slots):  # sleuven en rechthoekige uitsparingen (V15)
        ps += [Param(f"sx{i}", 0.3, 0.01), Param(f"sy{i}", 0.3, 0.01), Param(f"sl{i}", 0.3, 0.01, 0.5),
               Param(f"sw{i}", 0.3, 0.01, 0.5), Param(f"sa{i}", math.radians(1.0), math.radians(0.02))]
        if s.kind == "rechthoek":
            ps.append(Param(f"sr{i}", 0.3, 0.02, 0.0))
    if part.top_edge is not None:
        # Afschuining of afronding van de bovenrand (V17). "h" is dan de hoogte van de schouder eronder en
        # "top" de maat erboven: de hoge foto's zien vooral de schouder, alleen de lage de bovenrand. Met de
        # totale hoogte als parameter liggen hoogte en maat in een smal dal (samen omhoog houdt de schouder
        # op zijn plaats), waar de kompaszoektocht in blijft steken.
        ps.append(Param("top", 0.3, 0.02, 0.05))
    for k in range(len(part.steps)):  # treden (V17): plaats en richting van de lijn (om haar draaipunt), hoogte
        ps += [Param(f"to{k}", 0.4, 0.01), Param(f"ta{k}", math.radians(1.0), math.radians(0.02)),
               Param(f"th{k}", 0.4, 0.01, 0.3)]
    return ps


def _get(part: Part2p5D, base_angles: np.ndarray, name: str) -> float:
    o = part.outer
    if name == "h":  # met een afgeschuinde of afgeronde bovenrand: de hoogte van de schouder eronder
        return part.height - (part.top_edge.size if part.top_edge is not None else 0.0)
    if name == "cx":
        return float(o.center[0])
    if name == "cy":
        return float(o.center[1])
    if name == "R":
        return o.radius
    if name == "rot":
        return float(o.angles[0] - base_angles[0]) if o.n else 0.0
    if name.startswith("off"):
        return float(o.offsets[int(name[3:])])
    if name.startswith("fil"):
        return float(o.fillets[int(name[3:])])
    if name == "top":
        return part.top_edge.size
    if name[0] == "t":
        st = part.steps[int(name[2:])]
        return {"to": st.dist, "ta": st.angle, "th": st.height}[name[:2]]
    if name[0] == "s":
        s = part.slots[int(name[2:])]
        return {"sx": s.x, "sy": s.y, "sl": s.length, "sw": s.width, "sa": s.angle, "sr": s.r}[name[:2]]
    i = int(name[2:])
    return {"hx": part.holes[i].x, "hy": part.holes[i].y, "hd": part.holes[i].d}[name[:2]]


def _set(part: Part2p5D, base_angles: np.ndarray, name: str, value: float) -> Part2p5D:
    p = part.copy()
    o = p.outer
    if name == "h":
        p.height = value + (p.top_edge.size if p.top_edge is not None else 0.0)
    elif name == "cx":
        o.center[0] = value
    elif name == "cy":
        o.center[1] = value
    elif name == "R":
        o.radius = value
    elif name == "rot":
        o.angles = base_angles + value
    elif name.startswith("off"):
        o.offsets[int(name[3:])] = value
    elif name.startswith("fil"):
        o.fillets[int(name[3:])] = value
    elif name == "top":  # de schouder blijft staan (zie _params)
        p.height += value - p.top_edge.size
        p.top_edge = replace(p.top_edge, size=value)
    elif name[0] == "t":
        k = int(name[2:])
        p.steps[k] = replace(p.steps[k], **{{"to": "dist", "ta": "angle", "th": "height"}[name[:2]]: value})
    elif name[0] == "s":
        i = int(name[2:])
        field_name = {"sx": "x", "sy": "y", "sl": "length", "sw": "width", "sa": "angle", "sr": "r"}[name[:2]]
        p.slots[i] = replace(p.slots[i], **{field_name: value})
    else:
        i = int(name[2:])
        h = p.holes[i]
        p.holes[i] = Hole(value if name[:2] == "hx" else h.x, value if name[:2] == "hy" else h.y,
                          value if name[:2] == "hd" else h.d)
    return p


def fit_height(part: Part2p5D, K: np.ndarray, views: list[ViewData], h_max: float,
               step: float = 1.0, h_min: float | None = None) -> Part2p5D:
    """1D-zoektocht naar de hoogte (het bovenvlak is uit een visual hull niet te halen)."""
    start = max(step, h_min if h_min is not None else step)
    hs = np.arange(start, max(h_max, start + step) + step, step)
    es = [energy(_set(part, part.outer.angles, "h", float(h)), K, views) for h in hs]
    k = int(np.argmin(es))
    best = float(hs[k])
    if 0 < k < len(hs) - 1:  # parabool door de drie laagste punten
        e0, e1, e2 = es[k - 1], es[k], es[k + 1]
        denom = e0 - 2 * e1 + e2
        if denom > 0:
            best += 0.5 * step * (e0 - e2) / denom
    return _set(part, part.outer.angles, "h", best)


def refine(part: Part2p5D, K: np.ndarray, views: list[ViewData], max_evals: int = 1500,
           log=None, abort_iou: float = 0.8, only: set[str] | None = None) -> tuple[Part2p5D, float, int]:
    """Kompaszoektocht per parameter met halverende stappen; ongeldige geometrie wordt overgeslagen.

    Twee aanvullingen tegen te vroeg stoppen in een smalle vallei (bijv. hoogte en randen die
    elkaar in schuine foto's compenseren): na elke ronde een patroonstap (Hooke-Jeeves: de hele
    verplaatsing van die ronde nog eens), en na convergentie een herstart met grotere stappen.
    Past het model na 300 evaluaties nog steeds slecht (IoU-mediaan < `abort_iou`), dan stopt de
    zoektocht: de kwaliteitspoort keurt het toch af, en doorzoeken kost dan minuten; past het matig
    (< 0,95), dan volgt nog hooguit één blok van 300. Ook bij ruisige
    maskers (bijv. een zwart onderdeel) blijven er piepkleine 'verbeteringen' te vinden: levert de
    laatste 200 evaluaties samen minder dan 0,1% op, dan is de fit klaar.
    """
    base = part.outer.angles.copy()
    params = [p for p in _params(part) if only is None or p.name in only]  # `only`: alleen deze parameters
    steps = {p.name: p.step for p in params}
    best, e_best = part, energy(part, K, views)
    evals = 1

    def valid(cand: Part2p5D) -> bool:
        return cand.is_valid()

    restarts, e_restart = 0, e_best
    next_check = 300
    history = [e_best]  # beste energie per evaluatie
    while evals < max_evals:
        history += [e_best] * (evals - len(history) + 1)
        if evals >= 400 and history[evals - 200] - e_best < 1e-3 * e_best:
            if log:
                log(f"verfijning: geen noemenswaardige verbetering meer na {evals} evaluaties")
            break
        if abort_iou and evals >= next_check:
            next_check += 300
            iou = view_stats(best, K, views)["iou_median"]
            if iou < abort_iou:
                if log:
                    log(f"verfijning afgebroken na {evals} evaluaties: het model past niet bij de foto's")
                break
            if iou < 0.95:  # past matig: nog één blok, dan is het 'onbetrouwbaar' toch al duidelijk
                max_evals = min(max_evals, evals + 300)
        start = {p.name: _get(best, base, p.name) for p in params}
        moved_any = False
        for p in params:
            if steps[p.name] < p.min_step:
                continue
            x = _get(best, base, p.name)
            moved = False
            for sign in (1.0, -1.0):
                val = x + sign * steps[p.name]
                if val < p.lower:
                    continue
                cand = _set(best, base, p.name, val)
                if not valid(cand):
                    continue
                e = energy(cand, K, views)
                evals += 1
                if e < e_best:
                    best, e_best, moved = cand, e, True
                    break
            if moved:
                moved_any = True
            else:
                steps[p.name] *= 0.5
        if moved_any and evals < max_evals:
            cand = best
            for p in params:
                x = _get(best, base, p.name)
                if x != start[p.name] and x + (x - start[p.name]) >= p.lower:
                    cand = _set(cand, base, p.name, x + (x - start[p.name]))
            if cand is not best and valid(cand):
                e = energy(cand, K, views)
                evals += 1
                if e < e_best:
                    best, e_best = cand, e
        if not moved_any and all(steps[p.name] < p.min_step for p in params):
            if restarts >= 2 or e_best >= e_restart:
                break
            restarts, e_restart = restarts + 1, e_best
            for p in params:
                steps[p.name] = 0.25 * p.step
    if log:
        log(f"verfijning: {evals} evaluaties, E = {e_best:.0f}")
    return best, e_best, evals


def probe_fillets(part: Part2p5D, K: np.ndarray, views: list[ViewData], e_part: float | None = None,
                  radii=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)) -> tuple[Part2p5D, float, bool]:
    """Probeert per hoek grote stappen in de afrondingsstraal; geeft (model, energie, veranderd).

    Vanuit een scherpe hoek levert een kleine afronding bijna niets op (het weggesneden stukje groeit
    met R²). De kompaszoektocht vindt een afronding van 3 mm dan niet, en de fit stopt omdat er te
    weinig verbetert, ook al past die afronding veel beter.
    """
    e_best = energy(part, K, views) if e_part is None else e_part
    if part.outer.kind != "polygon":
        return part, e_best, False
    changed = False
    for k in range(part.outer.n):
        current = float(part.outer.fillets[k])
        for r in radii:
            if abs(r - current) < 0.25:
                continue
            cand = part.copy()
            cand.outer.fillets[k] = r
            if not cand.outer.is_valid():
                continue
            e = energy(cand, K, views)
            if e < e_best:
                part, e_best, changed = cand, e, True
    return part, e_best, changed


def fit_top_edge(part: Part2p5D, K: np.ndarray, views: list[ViewData], log=None,
                 sizes=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)) -> tuple[Part2p5D | None, float]:
    """Het prisma `part` met een afschuining of afronding van de bovenrand (V17); geeft (model, energie), of
    (None, inf) als geen enkele maat past. De pixelfit van een prisma legt de hoogte bij een afschuining op de
    schouder (de hoge foto's zien alleen die), bij een afronding iets erboven. Per soort daarom eerst een raster
    van maat en schouderhoogte, dan kort alleen die twee verfijnen; de beste soort daarna helemaal."""
    per_kind = []
    for kind in ("afschuining", "afronding"):
        best, e_best = None, math.inf
        for s in sizes:
            for below in (0.0, 0.25, 0.5):
                cand = part.copy()
                cand.top_edge = TopEdge(kind, float(s))
                cand.height = part.height + (1.0 - below) * s
                if not cand.is_valid():
                    continue
                e = energy(cand, K, views)
                if e < e_best:
                    best, e_best = cand, e
        if best is not None:
            best, e_best, _ = refine(best, K, views, max_evals=100, only={"h", "top"}, abort_iou=0.0)
            per_kind.append((e_best, best))
            if log:
                log(f"bovenrand als {kind}: {best.top_edge.size:.2f} mm, energie {e_best:.0f}")
    if not per_kind:
        return None, math.inf
    _, best = min(per_kind, key=lambda t: t[0])
    best, e_best, _ = refine(best, K, views, max_evals=300, abort_iou=0.0)
    return best, e_best


def fit_step(part: Part2p5D, K: np.ndarray, views: list[ViewData], a, b, mid, lower: bool = True,
             log=None) -> tuple[Part2p5D | None, float]:
    """Het prisma `part` met een trede (V17); geeft (model, energie), of (None, inf).

    Start: de lijn door a en b, de uiteinden van het stuk bovenrand dat de vormtoets (prismcheck) vond; `mid`
    ligt op dat stuk. `lower`: daar ligt de bovenkant lager dan het model (dat stuk wordt de trede), anders
    hoger (dat stuk wordt het hoge deel). De hoogtes eerst via een raster, dan lijn en hoogtes samen verfijnen,
    dan alles."""
    a, b, mid = (np.asarray(v, float) for v in (a, b, mid))
    d = b - a
    if np.linalg.norm(d) < 2.0:
        return None, math.inf
    n = np.array([d[1], -d[0]]) / np.linalg.norm(d)
    if n @ (mid - a) < 0:
        n = -n
    if not lower:  # het stuk is het hoge deel: de trede ligt aan de andere kant
        n = -n
    angle, offset = math.atan2(n[1], n[0]), float(n @ a)
    pivot = (a + b) / 2
    grid = ([(part.height, f * part.height) for f in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)] if lower else
            [(part.height + dh, part.height) for dh in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)])
    best, e_best = None, math.inf
    for h_main, h_step in grid:
        cand = part.copy()
        cand.height = h_main
        cand.steps = [Step.from_line(angle, offset, h_step, pivot)]
        if not cand.is_valid():
            continue
        e = energy(cand, K, views)
        if e < e_best:
            best, e_best = cand, e
    if best is None:
        return None, math.inf
    best, e_best, _ = refine(best, K, views, max_evals=150, only={"h", "to0", "ta0", "th0"}, abort_iou=0.0)
    best, e_best, _ = refine(best, K, views, max_evals=300, abort_iou=0.0)
    if log:
        st = best.steps[0]
        log(f"trede: hoogte {st.height:.2f} van {best.height:.2f} mm, energie {e_best:.0f}")
    return best, e_best


def view_stats(part: Part2p5D, K: np.ndarray, views: list[ViewData]) -> dict:
    """IoU van model-silhouet en objectmasker per foto (alleen pixels met zekere klasse)."""
    ious = []
    for v in views:
        P = render(part, K, v).astype(bool)
        known = v.fg | v.bg
        inter = np.count_nonzero(P & v.fg)
        union = np.count_nonzero((P & known) | v.fg)
        if union:
            ious.append(inter / union)
    return {"views": len(ious), "iou_median": float(np.median(ious)) if ious else float("nan"),
            "iou_min": float(np.min(ious)) if ious else float("nan")}
