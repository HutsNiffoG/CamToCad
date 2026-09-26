"""Model-gebaseerde verfijning: het 2,5D-model direct fitten op de silhouetten in alle foto's.

Analysis-by-synthesis (ARCHITECTURE.md §6.8): voor een kandidaatmodel rendert dit module per
foto het verwachte silhouet en telt de pixels die niet kloppen met de maskers:

    E = |model ∩ zekere mat| + w · |model ∩ onbekend| + |niet-model ∩ object|

Over tientallen foto's en duizenden randpixels levert dat maten met subpixel-nauwkeurigheid,
zonder dat het object textuur nodig heeft. Een visual hull dient alleen als startpunt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .calib import Pose, project
from .masks import ViewMasks
from .profile import Hole, Part2p5D, circle_polygon

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


def render(part: Part2p5D, K: np.ndarray, v: ViewData, ring: np.ndarray | None = None) -> np.ndarray:
    """Silhouet (uint8 0/1) van het model binnen de ROI van een foto."""
    h, w = v.fg.shape
    mask = np.zeros((h, w), np.uint8)
    ring = part.outer.outline(12.0) if ring is None else ring
    n = len(ring)
    both = np.vstack([np.column_stack([ring, np.zeros(n)]), np.column_stack([ring, np.full(n, part.height)])])
    uv, _ = project(both, v.pose, K)
    ub, ut = uv[:n], uv[n:]

    def fill(poly, target, convex=False):
        shrunk = _shrink(poly)
        if shrunk is None:
            return
        pts = _to_fixed(shrunk, v.x0, v.y0)
        if convex:
            cv2.fillConvexPoly(target, pts, 1, cv2.LINE_8, SHIFT)
        else:
            cv2.fillPoly(target, [pts], 1, cv2.LINE_8, SHIFT)

    fill(ub, mask)
    fill(ut, mask)
    quads = np.stack([ub, np.roll(ub, -1, axis=0), np.roll(ut, -1, axis=0), ut], axis=1)
    for q in _to_fixed(_shrink_quads(quads), v.x0, v.y0):  # één omzetting voor alle wanden
        cv2.fillConvexPoly(mask, q, 1, cv2.LINE_8, SHIFT)
    # doorkijk door gaten: binnen de projectie van zowel de boven- als de onderrand
    tmp_t = np.zeros_like(mask)
    tmp_b = np.zeros_like(mask)
    rings = [circle_polygon((hl.x, hl.y), hl.d / 2, 48) for hl in part.holes]
    rings += list(part.cutouts)
    for r2 in rings:
        m2 = len(r2)
        uv2, _ = project(np.vstack([np.column_stack([r2, np.full(m2, part.height)]),
                                    np.column_stack([r2, np.zeros(m2)])]), v.pose, K)
        tmp_t[:] = 0
        tmp_b[:] = 0
        fill(uv2[:m2], tmp_t)
        fill(uv2[m2:], tmp_b)
        mask[(tmp_t & tmp_b) > 0] = 0
    return mask


def _mismatch(part: Part2p5D, K: np.ndarray, v: ViewData, ring: np.ndarray) -> tuple[int, int, int]:
    P = render(part, K, v, ring).astype(bool)
    return np.count_nonzero(P & v.bg), np.count_nonzero(P & v.unk), np.count_nonzero(~P & v.fg)


def energy(part: Part2p5D, K: np.ndarray, views: list[ViewData], w_unknown: float = 0.25) -> float:
    # Serieel: per foto is het vooral Python-werk aan kleine arrays; threads maakten het twee keer trager.
    ring = part.outer.outline(12.0)
    bg = unk = fg = 0
    for v in views:
        a, b, c = _mismatch(part, K, v, ring)
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
    return ps


def _get(part: Part2p5D, base_angles: np.ndarray, name: str) -> float:
    o = part.outer
    if name == "h":
        return part.height
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
    i = int(name[2:])
    return {"hx": part.holes[i].x, "hy": part.holes[i].y, "hd": part.holes[i].d}[name[:2]]


def _set(part: Part2p5D, base_angles: np.ndarray, name: str, value: float) -> Part2p5D:
    p = part.copy()
    o = p.outer
    if name == "h":
        p.height = value
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
           log=None, abort_iou: float = 0.8) -> tuple[Part2p5D, float, int]:
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
    params = _params(part)
    steps = {p.name: p.step for p in params}
    best, e_best = part, energy(part, K, views)
    evals = 1

    def valid(cand: Part2p5D) -> bool:
        return cand.outer.kind != "polygon" or cand.outer.is_valid()

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
