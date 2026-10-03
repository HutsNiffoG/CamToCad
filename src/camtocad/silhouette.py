"""Model-gebaseerde verfijning: het 2,5D-model direct fitten op de silhouetten in alle foto's.

Analysis-by-synthesis (ARCHITECTURE.md §6.8): voor een kandidaatmodel rendert dit module per
foto het verwachte silhouet en telt de pixels die niet kloppen met de maskers:

    E = |model ∩ zekere mat| + w · |model ∩ onbekend| + |niet-model ∩ object|

Over tientallen foto's en duizenden randpixels levert dat maten met subpixel-nauwkeurigheid,
zonder dat het object textuur nodig heeft. Een visual hull dient alleen als startpunt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .calib import Pose, project
from .masks import ViewMasks
from .cadhelpers import afgeronde_hoeken
from .profile import Part2p5D, Step, TopEdge, circle_polygon


@dataclass
class ViewData:
    pose: Pose
    fg: np.ndarray  # uitsnede (ROI) van het objectmasker
    bg: np.ndarray  # zekere mat
    unk: np.ndarray  # onbekend (op de mat, maar zonder textuur)
    x0: int
    y0: int
    _sums: tuple | None = field(default=None, repr=False, compare=False)  # zie sums()
    # zachte objectfractie en haar gewicht in de ROI (masks._soft_alpha; NaN en 0 zonder bewijs), voor de randfit
    alpha: np.ndarray | None = field(default=None, repr=False, compare=False)
    alpha_w: np.ndarray | None = field(default=None, repr=False, compare=False)
    # grijswaarden in de ROI (NaN buiten de uitsnede van het masker): randen binnen het object (V16)
    gray: np.ndarray | None = field(default=None, repr=False, compare=False)
    tone: float = 1.0  # toonkromme van de camera in deze foto (masks._tone_exponent), voor `gray`

    def sums(self) -> tuple:
        """Per rij de cumulatieve aantallen zekere-mat-, onbekende en objectpixels (met een 0-kolom vooraan),
        en het totaal aantal objectpixels: zo telt de energie per reeks pixels in plaats van per pixel. Opnieuw
        berekend als een van de maskers vervangen is."""
        key = (id(self.fg), id(self.bg), id(self.unk))
        if self._sums is None or self._sums[0] != key:
            def cum(m):
                out = np.zeros((m.shape[0], m.shape[1] + 1), np.int32)
                np.cumsum(m, axis=1, dtype=np.int32, out=out[:, 1:])
                return out
            self._sums = (key, cum(self.bg), cum(self.unk), cum(self.fg), int(np.count_nonzero(self.fg)))
        return self._sums


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
        vd = ViewData(pose, fg, bg, unk, x0, y0)
        if m.alpha is not None:
            vd.alpha = _crop(m.alpha, m.alpha_at, y0, y1, x0, x1, np.nan)
            vd.alpha_w = _crop(m.alpha_w, m.alpha_at, y0, y1, x0, x1, 0.0)
        if m.gray is not None:
            vd.gray = _crop(m.gray, m.alpha_at, y0, y1, x0, x1, np.nan)
            vd.tone = m.tone
        out.append(vd)
    return out


def _crop(a: np.ndarray, at: tuple[int, int], y0: int, y1: int, x0: int, x1: int, fill: float) -> np.ndarray:
    """Een uitsnede van het beeld (vanaf `at`, rij en kolom) in de ROI; `fill` daarbuiten."""
    out = np.full((y1 - y0, x1 - x0), fill, a.dtype)
    ay, ax = at
    h, wd = a.shape
    ty0, tx0, ty1, tx1 = max(y0, ay), max(x0, ax), min(y1, ay + h), min(x1, ax + wd)
    if ty1 > ty0 and tx1 > tx0:
        out[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0] = a[ty0 - ay:ty1 - ay, tx0 - ax:tx1 - ax]
    return out


def _scan_quads(q: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixelmiddens binnen convexe vierhoeken (Q, 4, 2), in pixelcoördinaten van de ROI (middens op gehele
    getallen, zoals OpenCV): per vierhoek en rij de kolommen c0..c1. Exact, ook voor een vierhoek die smaller
    is dan een pixel (V29)."""
    empty = np.zeros(0, np.int64)
    if not len(q):
        return empty, empty, empty
    ya, yb = q[..., 1], q[:, [1, 2, 3, 0], 1]
    xa, xb = q[..., 0], q[:, [1, 2, 3, 0], 0]
    dy = yb - ya
    flat = dy == 0  # een horizontale rand snijdt geen rij (de andere randen dekken hem)
    lo = np.where(flat, np.inf, np.minimum(ya, yb))
    hi = np.where(flat, -np.inf, np.maximum(ya, yb))
    slope = (xb - xa) / np.where(flat, 1.0, dy)
    x0 = xa - ya * slope  # x op rij 0
    r0 = np.maximum(np.ceil(ya.min(axis=1)), 0).astype(np.int64)
    r1 = np.minimum(np.floor(ya.max(axis=1)), h - 1).astype(np.int64)
    n = np.maximum(r1 - r0 + 1, 0)
    total = int(n.sum())
    if not total:
        return empty, empty, empty
    qi = np.repeat(np.arange(len(q)), n)
    rows = np.repeat(r0, n) + (np.arange(total) - np.repeat(np.cumsum(n) - n, n))
    y = rows[:, None].astype(float)
    cross = (lo[qi] <= y) & (y <= hi[qi])
    x = x0[qi] + y * slope[qi]
    xl = np.where(cross, x, np.inf).min(axis=1)
    xr = np.where(cross, x, -np.inf).max(axis=1)
    c0 = np.maximum(np.ceil(xl - 1e-9), 0)
    c1 = np.minimum(np.floor(xr + 1e-9), w - 1)
    ok = c1 >= c0  # ook False als de rij geen rand snijdt (inf)
    return rows[ok], c0[ok].astype(np.int64), c1[ok].astype(np.int64)


def _scan_polys(polys: list[np.ndarray], h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Als _scan_quads, voor willekeurige (ook holle) polygonen: per polygoon en rij de snijpunten met de rij
    (halfopen regel op de hoekpunten), gesorteerd en per paar een reeks kolommen (even-oneven)."""
    empty = np.zeros(0, np.int64)
    polys = [p for p in polys if len(p) >= 3]
    if not polys:
        return empty, empty, empty
    sizes = np.array([len(p) for p in polys])
    P = np.vstack(polys).astype(float)
    first = np.repeat(np.cumsum(sizes) - sizes, sizes)
    nxt = np.arange(len(P)) + 1
    nxt = np.where(nxt - first >= np.repeat(sizes, sizes), first, nxt)
    ya, yb, xa, xb = P[:, 1], P[nxt, 1], P[:, 0], P[nxt, 0]
    r0 = np.maximum(np.ceil(np.minimum(ya, yb)), 0).astype(np.int64)  # halfopen: laag <= r < hoog
    r1 = np.minimum(np.ceil(np.maximum(ya, yb)) - 1, h - 1).astype(np.int64)
    n = np.maximum(r1 - r0 + 1, 0)
    total = int(n.sum())
    if not total:
        return empty, empty, empty
    e = np.repeat(np.arange(len(P)), n)
    rows = np.repeat(r0, n) + (np.arange(total) - np.repeat(np.cumsum(n) - n, n))
    x = xa[e] + (rows - ya[e]) / (yb[e] - ya[e]) * (xb[e] - xa[e])
    order = np.lexsort((x, rows, np.repeat(np.arange(len(polys)), sizes)[e]))
    x, rows = x[order], rows[order]
    c0 = np.maximum(np.ceil(x[0::2] - 1e-9), 0)
    c1 = np.minimum(np.floor(x[1::2] + 1e-9), w - 1)
    ok = c1 >= c0
    return rows[0::2][ok], c0[ok].astype(np.int64), c1[ok].astype(np.int64)


def _fill_runs(mask: np.ndarray, rows: np.ndarray, c0: np.ndarray, c1: np.ndarray, value: int = 1) -> None:
    """Zet de pixels (rij, c0..c1) op `value`: lange reeksen per rij (snel), korte in één keer."""
    length = c1 - c0 + 1
    long_ = length > 48
    for r, a, b in zip(rows[long_].tolist(), c0[long_].tolist(), c1[long_].tolist()):
        mask[r, a:b + 1] = value
    rows, c0, length = rows[~long_], c0[~long_], length[~long_]
    total = int(length.sum())
    if total:
        idx = np.repeat(rows * mask.shape[1] + c0 - (np.cumsum(length) - length), length) + np.arange(total)
        mask.ravel()[idx] = value


_EMPTY = np.zeros(0, np.int64)


def _cover(groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    """Reeksen (rij, c0, c1) van een paar groepen samengevoegd tot segmenten met per groep de dekking (hoeveel
    reeksen van die groep het segment bedekken). Geeft (rij, c0, c1, dekking (S, groepen))."""
    sizes = [len(g[0]) for g in groups]
    total = sum(sizes)
    if not total:
        return _EMPTY, _EMPTY, _EMPTY, np.zeros((0, len(groups)), np.int32)
    rows = np.concatenate([g[0] for g in groups])
    c0 = np.concatenate([g[1] for g in groups])
    c1 = np.concatenate([g[2] for g in groups])
    gid = np.repeat(np.arange(len(groups)), sizes)
    er = np.concatenate([rows, rows])
    ec = np.concatenate([c0, c1 + 1])  # begin en einde (exclusief)
    step = np.zeros((2 * total, len(groups)), np.int32)
    step[np.arange(total), gid] = 1
    step[total + np.arange(total), gid] = -1
    order = np.lexsort((ec, er))
    er, ec, step = er[order], ec[order], step[order]
    cov = np.cumsum(step, axis=0)  # per rij telt alles op tot 0, dus de rijen storen elkaar niet
    keep = (er[:-1] == er[1:]) & (ec[1:] > ec[:-1])
    return er[:-1][keep], ec[:-1][keep], ec[1:][keep] - 1, cov[:-1][keep]


def _union(rows: np.ndarray, c0: np.ndarray, c1: np.ndarray):
    """Vereniging van reeksen per rij: gesorteerd per rij en begin, dan een lopend maximum van het einde;
    een nieuwe reeks begint waar het begin voorbij dat maximum (+1) ligt."""
    if not len(rows):
        return rows, c0, c1
    order = np.lexsort((c0, rows))
    rows, c0, c1 = rows[order], c0[order], c1[order]
    span = int(c1.max()) + 2
    reach = np.maximum.accumulate(rows * span + c1) - rows * span  # lopend maximum binnen de rij
    new = np.ones(len(rows), bool)
    new[1:] = (rows[1:] != rows[:-1]) | (c0[1:] > reach[:-1] + 1)
    first = np.flatnonzero(new)
    last = np.r_[first[1:] - 1, len(rows) - 1]
    return rows[first], c0[first], reach[last]


def _silhouette_runs(part: Part2p5D, K: np.ndarray, v: "ViewData", geom: list[np.ndarray]):
    """Het silhouet als losse reeksen (rij, c0, c1) per rij: de vereniging van onder- en bovenvlak en de wanden,
    min de doorkijk door gaten, sleuven en uitsparingen. Alles volgens de pixelmiddenregel (zie render)."""
    h, w = v.fg.shape
    off = np.array([v.x0, v.y0], float)
    cam = v.pose.center
    faces = []
    for stack in geom:
        # Alleen de vlakken die naar de camera kijken: de eerste snijding van een kijkstraal met het onderdeel
        # ligt altijd op zo'n vlak, dus hun vereniging is precies het silhouet. Het ondervlak en de achterkant
        # vallen zo weg. De wanden en de banden van een afschuining of afronding tussen twee niveaus zijn
        # (vlakke) vierhoeken; de ringen lopen tegen de klok in, dus (C - A) x (D - B) wijst naar buiten.
        n_lev, n = stack.shape[:2]
        uv, _ = project(stack.reshape(-1, 3), v.pose, K)
        uv = uv.reshape(n_lev, n, 2) - off
        if cam[2] > stack[-1, 0, 2]:
            faces.append(_scan_polys([uv[-1]], h, w))
        nxt = np.r_[1:n, 0]
        for j in range(n_lev - 1):
            a, d = stack[j], stack[j + 1]
            u, t = d[nxt] - a, d - a[nxt]  # diagonalen C - A en D - B
            nrm = np.column_stack([u[:, 1] * t[:, 2] - u[:, 2] * t[:, 1], u[:, 2] * t[:, 0] - u[:, 0] * t[:, 2],
                                   u[:, 0] * t[:, 1] - u[:, 1] * t[:, 0]])
            front = np.flatnonzero(((cam - a) * nrm).sum(axis=1) > 0)
            if len(front):
                lo, hi = uv[j], uv[j + 1]
                quads = np.stack([lo[front], lo[nxt[front]], hi[nxt[front]], hi[front]], axis=1)
                faces.append(_scan_quads(quads, h, w))
    if not faces:
        return _EMPTY, _EMPTY, _EMPTY
    solid = _union(*(np.concatenate(parts) for parts in zip(*faces)))
    # doorkijk: binnen de projectie van zowel de boven- als de onderrand van een gat; bij een verzinking (V16) is
    # de bovenrand van het doorgaande gat de onderkant van de kegel (de kegel zelf wordt naar boven toe breder), bij
    # een kamerboring de bodem van de kamer, en dan moet een kijkstraal ook door de kamer zelf (haar rand aan het
    # bovenvlak; in schuine foto's begrenst die de doorkijk aan de kant van de camera)
    rings = [circle_polygon((hl.x, hl.y), hl.d / 2, 48) for hl in part.holes]
    below = [hl.bore_depth for hl in part.holes]
    chamber = [circle_polygon((hl.x, hl.y), hl.cb / 2, 48) if hl.cb > 0 else None for hl in part.holes]
    rings += list(part.cutouts)
    rings += [s.outline() for s in part.slots]
    through = []
    for k, r2 in enumerate(rings):
        c2 = r2.mean(axis=0)
        z_top = part.height_at(float(c2[0]), float(c2[1])) if part.steps else part.height
        loops = [(r2, z_top - (below[k] if k < len(below) else 0.0)), (r2, 0.0)]
        if k < len(chamber) and chamber[k] is not None:
            loops.append((chamber[k], z_top))
        uv2, _ = project(np.vstack([np.column_stack([r, np.full(len(r), z)]) for r, z in loops]), v.pose, K)
        uv2 = uv2 - off
        cuts = np.cumsum([0] + [len(r) for r, _ in loops])
        rt, a, b, cov = _cover([_scan_polys([uv2[cuts[j]:cuts[j + 1]]], h, w) for j in range(len(loops))])
        inside = np.all(cov > 0, axis=1)
        through.append((rt[inside], a[inside], b[inside]))
    if not through:
        return solid
    rt, a, b, cov = _cover([solid, tuple(np.concatenate(p) for p in zip(*through))])
    keep = (cov[:, 0] > 0) & (cov[:, 1] == 0)
    return rt[keep], a[keep], b[keep]


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
    """Silhouet (uint8 0/1) van het model binnen de ROI van een foto: een pixel hoort erbij als zijn midden
    binnen de projectie van het onderdeel ligt. `geom`: solids(part), als die voor meerdere foto's al
    berekend is.

    Het silhouet is de vereniging van de projecties van onder- en bovenvlak en de wanden, alles exact volgens
    die pixelmiddenregel gerasterd (V29). Tot v0.7 deed cv2.fillPoly dat, met elke polygoon 0,5 px gekrompen:
    gemiddeld goed voor een groot vlak (±0,15 px per vlak), maar een wand die in beeld smaller is dan een pixel
    kan niet krimpen, en de vereniging van vlakken met afwijkingen naar beide kanten neemt alleen de te ruime
    mee. Het silhouet werd zo in schuine foto's ~0,12 px te ruim, en de pixelfit ~0,05 mm te klein."""
    mask = np.zeros(v.fg.shape, np.uint8)
    _fill_runs(mask, *_silhouette_runs(part, K, v, solids(part) if geom is None else geom))
    return mask


def _mismatch(part: Part2p5D, K: np.ndarray, v: ViewData, geom: list[np.ndarray]) -> tuple[int, int, int]:
    """(model ∩ zekere mat, model ∩ onbekend, object buiten het model) in pixels, geteld per reeks via de
    cumulatieve sommen per rij: zonder masker."""
    rows, c0, c1 = _silhouette_runs(part, K, v, geom)
    _, s_bg, s_unk, s_fg, n_fg = v.sums()
    e = c1 + 1
    bg = int((s_bg[rows, e] - s_bg[rows, c0]).sum())
    unk = int((s_unk[rows, e] - s_unk[rows, c0]).sum())
    fg_in = int((s_fg[rows, e] - s_fg[rows, c0]).sum())
    return bg, unk, n_fg - fg_in


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
    for i, hl in enumerate(part.holes):
        ps += [Param(f"hx{i}", 0.3, 0.01), Param(f"hy{i}", 0.3, 0.01), Param(f"hd{i}", 0.3, 0.01, 0.3)]
        if hl.csk > 0:  # verzinking (V16): de diameter aan het bovenvlak
            ps.append(Param(f"hk{i}", 0.3, 0.01, 0.5))
        if hl.cb > 0:  # kamerboring (V16, v0.10): diameter en diepte van de kamer
            ps += [Param(f"hc{i}", 0.3, 0.01, 0.5), Param(f"hz{i}", 0.4, 0.02, 0.2)]
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
    h = part.holes[i]
    return {"hx": h.x, "hy": h.y, "hd": h.d, "hk": h.csk, "hc": h.cb, "hz": h.cb_depth}[name[:2]]


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
        p.holes[i] = replace(p.holes[i], **{{"hx": "x", "hy": "y", "hd": "d", "hk": "csk", "hc": "cb",
                                              "hz": "cb_depth"}[name[:2]]: value})
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


def _top_edge_grid(part: Part2p5D, K: np.ndarray, views: list[ViewData], kind: str,
                   sizes) -> tuple[Part2p5D | None, float]:
    """Het prisma met een afschuining of afronding van de bovenrand, over een raster van maat en schouderhoogte
    (zie fit_top_edge): het beste model en zijn energie."""
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
    return best, e_best


def top_edge_probe(part: Part2p5D, K: np.ndarray, views: list[ViewData], sizes=(0.5, 1.0, 1.5, 2.0, 3.0)) -> float:
    """Hoeveel beter past het prisma `part` met een afschuining of afronding van de bovenrand, zonder te fitten?
    Relatieve energiewinst van het beste rastermodel (negatief: slechter). Goedkoop (~30 energieën): de vormtoets
    kan een afschuining bij een zwart onderdeel missen (v0.8, ROUTE-A-VERBETERPUNTEN §3h)."""
    e0 = energy(part, K, views)
    e = min(_top_edge_grid(part, K, views, kind, sizes)[1] for kind in ("afschuining", "afronding"))
    return (e0 - e) / e0 if math.isfinite(e) and e0 > 0 else -math.inf


def fit_top_edge(part: Part2p5D, K: np.ndarray, views: list[ViewData], log=None,
                 sizes=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)) -> tuple[Part2p5D | None, float]:
    """Het prisma `part` met een afschuining of afronding van de bovenrand (V17); geeft (model, energie), of
    (None, inf) als geen enkele maat past. De pixelfit van een prisma legt de hoogte bij een afschuining op de
    schouder (de hoge foto's zien alleen die), bij een afronding iets erboven. Per soort daarom eerst een raster
    van maat en schouderhoogte, dan kort alleen die twee verfijnen; de beste soort daarna helemaal."""
    per_kind = []
    for kind in ("afschuining", "afronding"):
        best, e_best = _top_edge_grid(part, K, views, kind, sizes)
        if best is not None:
            best, e_best, _ = refine(best, K, views, max_evals=100, only={"h", "top"}, abort_iou=0.0)
            per_kind.append((e_best, best))
            if log:
                log(f"bovenrand als {kind} geprobeerd: {best.top_edge.size:.2f} mm, energie {e_best:.0f}")
    if not per_kind:
        return None, math.inf
    _, best = min(per_kind, key=lambda t: t[0])
    best, e_best, _ = refine(best, K, views, max_evals=300, abort_iou=0.0, only=_shape_params(best))
    return best, e_best


def _shape_params(part: Part2p5D) -> set[str]:
    """De parameters van de buitenvorm (hoogte, contour, bovenrand, treden), zonder gaten en sleuven: die liggen
    al goed uit de pixelfit, en een gat zonder bewijs rondom (zwart op zwart) dwaalt anders af, waarna de randfit
    tegen zijn vertrouwensgebied loopt."""
    return {p.name for p in _params(part) if not p.name.startswith(("hx", "hy", "hd", "sx", "sy", "sl", "sw", "sa",
                                                                     "sr"))}


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
    best, e_best, _ = refine(best, K, views, max_evals=300, abort_iou=0.0, only=_shape_params(best))
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
