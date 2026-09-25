"""Parametrisch 2,5D-model: een extrusie van een 2D-contour met doorgaande gaten.

De buitencontour is een cirkel of een polygoon van lijnen (buitennormaal + afstand tot een
referentiepunt) met per hoek een afrondingsstraal. Die parametrisatie past bij ontwerpintentie:
randen blijven recht, hoeken zijn scherp of afgerond, en verfijning verschuift hele randen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .cadhelpers import afgeronde_hoeken


def circle_polygon(center, radius: float, n: int) -> np.ndarray:
    """Regelmatige n-hoek met dezelfde oppervlakte als de cirkel.

    Een ingeschreven n-hoek is kleiner dan de cirkel (30 hoeken: 0,4% in straal); een model dat
    zo getekend wordt, fit dan net zoveel te groot. Hoekpunten iets buiten de cirkel heffen dat op.
    """
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = radius * math.sqrt(2 * math.pi / (n * math.sin(2 * math.pi / n)))
    return np.asarray(center, float) + r * np.column_stack([np.cos(a), np.sin(a)])


@dataclass
class Hole:
    x: float
    y: float
    d: float


@dataclass
class Profile:
    kind: str = "polygon"  # "polygon" of "circle"
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angles: np.ndarray = field(default_factory=lambda: np.zeros(0))  # buitennormaal van rand k (rad)
    offsets: np.ndarray = field(default_factory=lambda: np.zeros(0))  # n_k · (p - center) = offset_k
    fillets: np.ndarray = field(default_factory=lambda: np.zeros(0))  # straal van hoek k (tussen rand k-1 en k)
    radius: float = 0.0  # alleen voor een cirkel (center = middelpunt)

    def copy(self) -> "Profile":
        return Profile(self.kind, self.center.copy(), self.angles.copy(), self.offsets.copy(),
                       self.fillets.copy(), self.radius)

    @property
    def n(self) -> int:
        return len(self.angles)

    def normals(self) -> np.ndarray:
        return np.column_stack([np.cos(self.angles), np.sin(self.angles)])

    def vertices(self) -> np.ndarray:
        nrm = self.normals()
        out = np.empty((self.n, 2))
        for k in range(self.n):
            A = np.array([nrm[k - 1], nrm[k]])
            out[k] = np.linalg.solve(A, [self.offsets[k - 1], self.offsets[k]])
        return out + self.center

    def corner_table(self) -> list[tuple[float, float, float]]:
        return [(float(x), float(y), float(r)) for (x, y), r in zip(self.vertices(), self.fillets)]

    def outline(self, max_step_deg: float = 7.5) -> np.ndarray:
        """Dichte polylijn (tegen de klok in) van de contour, inclusief bogen."""
        if self.kind == "circle":
            k = max(24, int(360 / max_step_deg))
            return circle_polygon(self.center, self.radius, k)
        pts = []
        for t1, m, t2, c, r in afgeronde_hoeken(self.corner_table()):
            if m is None:
                pts.append(t1)
                continue
            a1 = math.atan2(t1[1] - c[1], t1[0] - c[0])
            a2 = math.atan2(t2[1] - c[1], t2[0] - c[0])
            am = math.atan2(m[1] - c[1], m[0] - c[0])
            span = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi  # kortste hoek, teken = draairichting
            if abs(((am - a1 + math.pi) % (2 * math.pi) - math.pi)) > abs(span) + 1e-6:
                span = span - math.copysign(2 * math.pi, span)
            steps = max(2, int(abs(math.degrees(span)) / max_step_deg) + 1)
            for s in np.linspace(0, 1, steps + 1):
                a = a1 + s * span
                pts.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
        return np.array(pts)

    def is_valid(self) -> bool:
        try:
            return self._is_valid()
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            return False

    def _is_valid(self) -> bool:
        if self.kind == "circle":
            return self.radius > 0
        if np.any(self.fillets < 0):
            return False
        try:
            V = self.vertices()
        except np.linalg.LinAlgError:
            return False
        if _signed_area(V) <= 0:
            return False
        # afrondingen moeten op de randen passen
        hoeken = afgeronde_hoeken(self.corner_table())
        for k in range(self.n):
            t2_prev = np.array(hoeken[k - 1][2])
            t1 = np.array(hoeken[k][0])
            edge = V[k] - V[k - 1]
            length = np.linalg.norm(edge)
            if length < 1e-6:
                return False
            if np.dot(t1 - t2_prev, edge) < -1e-6:
                return False
        return _is_simple(V)

    def area(self) -> float:
        return abs(_signed_area(self.outline()))


@dataclass
class Part2p5D:
    height: float
    outer: Profile
    holes: list[Hole] = field(default_factory=list)
    cutouts: list[np.ndarray] = field(default_factory=list)  # niet-ronde doorgaande uitsparingen (polygonen)

    def copy(self) -> "Part2p5D":
        return Part2p5D(self.height, self.outer.copy(), [Hole(h.x, h.y, h.d) for h in self.holes],
                        [c.copy() for c in self.cutouts])

    def scaled(self, factor: float) -> "Part2p5D":
        """Alle maten x factor (om de oorsprong), bijv. voor een mat die niet op 100% is geprint."""
        out = self.copy()
        out.height = self.height * factor
        o = out.outer
        o.center, o.offsets, o.fillets, o.radius = o.center * factor, o.offsets * factor, o.fillets * factor, \
            o.radius * factor
        out.holes = [Hole(h.x * factor, h.y * factor, h.d * factor) for h in self.holes]
        out.cutouts = [c * factor for c in self.cutouts]
        return out

    def transformed(self, angle: float, shift) -> "Part2p5D":
        """Starre 2D-transformatie p -> R(angle) p + shift."""
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        shift = np.asarray(shift, float)
        out = self.copy()
        out.outer.center = R @ self.outer.center + shift
        out.outer.angles = self.outer.angles + angle
        out.holes = [Hole(*(R @ [h.x, h.y] + shift), h.d) for h in self.holes]
        out.cutouts = [(R @ cu.T).T + shift for cu in self.cutouts]
        return out


def dominant_angle(profile: Profile, window_deg: float = 3.0) -> float:
    """Hoofdrichting (mod 90°) van de randen: de lengtegewogen modus, verfijnd over de randen binnen
    ±window_deg daarvan. Een schuine rand (afschuining) trekt de hoofdrichting zo niet scheef."""
    if profile.kind != "polygon" or profile.n == 0:
        return 0.0
    V = profile.vertices()
    lengths = np.linalg.norm(np.roll(V, -1, axis=0) - V, axis=1)  # rand k loopt van V[k] naar V[k+1]
    quarter = math.pi / 2

    def spread(center: float) -> np.ndarray:  # hoekafstand modulo 90°
        return np.abs((profile.angles - center + quarter / 2) % quarter - quarter / 2)

    win = math.radians(window_deg)
    candidates = np.radians(np.arange(0.0, 90.0, 0.25))
    support = [lengths[spread(c) < win].sum() for c in candidates]
    mode = float(candidates[int(np.argmax(support))])
    # binnen het venster de lengtegewogen mediaan (één schuine rand in het venster trekt niet mee),
    # dan het gemiddelde over de randen die daar werkelijk bij horen
    sel = spread(mode) < win
    dev = (profile.angles[sel] - mode + quarter / 2) % quarter - quarter / 2
    w = lengths[sel]
    order = np.argsort(dev)
    cum = np.cumsum(w[order])
    med = mode + float(dev[order][np.searchsorted(cum, 0.5 * cum[-1])])
    mad = float(np.sum(w * np.abs(dev - (med - mode))) / max(w.sum(), 1e-12))
    sel = spread(med) < max(math.radians(0.5), 2.0 * mad)
    z = np.sum(lengths[sel] * np.exp(4j * profile.angles[sel]))
    return float(np.angle(z) / 4)


def regularize_angles(profile: Profile, tol_deg: float = 3.0) -> tuple[Profile, float]:
    """Zet randen die bijna evenwijdig of loodrecht zijn exact op de hoofdrichting (ontwerpintentie)."""
    if profile.kind != "polygon":
        return profile, 0.0
    theta0 = dominant_angle(profile)
    out = profile.copy()
    for k, a in enumerate(profile.angles):
        m = round((a - theta0) / (math.pi / 2))
        target = theta0 + m * math.pi / 2
        if abs(a - target) < math.radians(tol_deg):
            # de rand draaien om zijn eigen middelpunt: offset aanpassen zodat dat punt op de rand blijft
            V = profile.vertices()
            mid = 0.5 * (V[k] + V[(k + 1) % profile.n]) - profile.center
            out.angles[k] = target
            out.offsets[k] = float(np.array([math.cos(target), math.sin(target)]) @ mid)
    # buren die nu precies evenwijdig zijn hebben geen snijpunt meer: de kortste vervalt
    V = profile.vertices()
    length = list(np.linalg.norm(np.roll(V, -1, axis=0) - V, axis=1))  # rand k loopt van V[k] naar V[k+1]
    k = 0
    while out.n > 3 and k < out.n:
        if math.cos(out.angles[k] - out.angles[k - 1]) > math.cos(math.radians(0.01)):
            drop = k if length[k] <= length[k - 1] else (k - 1) % out.n
            out.angles = np.delete(out.angles, drop)
            out.offsets = np.delete(out.offsets, drop)
            out.fillets = np.delete(out.fillets, drop)
            length.pop(drop)
            k = 0
        else:
            k += 1
    return out, theta0


def remove_short_edges(profile: Profile, min_len: float) -> Profile:
    """Randen korter dan min_len (ruis van de contour, bijv. een trapje van 0,01 mm in een hoek)
    weglaten: de buren snijden elkaar, of worden één rand als ze evenwijdig zijn."""
    out = profile.copy()
    while out.kind == "polygon" and out.n > 3:
        try:
            V = out.vertices()
        except np.linalg.LinAlgError:
            break
        lengths = np.linalg.norm(np.roll(V, -1, axis=0) - V, axis=1)  # rand k: V[k] -> V[k+1]
        k = int(np.argmin(lengths))
        if lengths[k] >= min_len:
            break
        # draaien zodat de korte rand index 1 heeft (buren 0 en 2); hoekpunt i ligt tussen rand i-1 en i
        shift = 1 - k
        a, off, fil = (np.roll(x, shift) for x in (out.angles, out.offsets, out.fillets))
        L = np.roll(lengths, shift)
        cand = out.copy()
        if math.cos(a[2] - a[0]) > math.cos(math.radians(0.5)):  # evenwijdige buren: samenvoegen
            off[0] = (L[0] * off[0] + L[2] * off[2]) / max(L[0] + L[2], 1e-9)
            drop = [1, 2]
            cand.angles, cand.offsets, cand.fillets = np.delete(a, drop), np.delete(off, drop), np.delete(fil, drop)
        else:
            fil[2] = max(fil[1], fil[2])
            cand.angles, cand.offsets, cand.fillets = np.delete(a, 1), np.delete(off, 1), np.delete(fil, 1)
        if cand.n < 3 or not cand.is_valid():
            break
        out = cand
    return out


def _signed_area(P: np.ndarray) -> float:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _segments_intersect(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _is_simple(V: np.ndarray) -> bool:
    n = len(V)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_intersect(V[i], V[(i + 1) % n], V[j], V[(j + 1) % n]):
                return False
    return True


# ----------------------------------------------------------------------------- fitten

def fit_line(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Totale-kleinste-kwadratenlijn: (eenheidsnormaal, offset, rms) met n·p = offset."""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c)
    n = vt[1]
    rms = float(np.sqrt(np.mean(((points - c) @ n) ** 2)))
    return n, float(n @ c), rms


def fit_circle(points: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraïsche cirkelfit (Kåsa) + Gauss-Newton; geeft (cx, cy, r, rms)."""
    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    (a0, a1, a2), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = a0 / 2, a1 / 2
    r = math.sqrt(max(a2 + cx * cx + cy * cy, 1e-12))
    for _ in range(10):
        dx, dy = x - cx, y - cy
        d = np.sqrt(dx * dx + dy * dy) + 1e-12
        J = np.column_stack([-dx / d, -dy / d, -np.ones_like(d)])
        res = d - r
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
        if np.linalg.norm(step) < 1e-9:
            break
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return float(cx), float(cy), float(abs(r)), rms


def _contour_mm(contour, origin, px) -> np.ndarray:
    c = contour.reshape(-1, 2).astype(float)
    return np.column_stack([origin[0] + c[:, 0] * px, origin[1] + c[:, 1] * px])


def polygon_from_contour(P: np.ndarray, px: float, eps_mm: float | None = None,
                         min_fillet: float = 1.0) -> Profile:
    """Contour (tegen de klok in, mm) → lijnen met afrondingen. Randen worden per stuk gefit."""
    n_pts = len(P)
    perim = float(np.sum(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)))
    eps = eps_mm or max(1.2 * px, 0.004 * perim)
    approx = cv2.approxPolyDP(P.astype(np.float32).reshape(-1, 1, 2), eps, True).reshape(-1, 2)
    index = {tuple(np.round(p, 5)): i for i, p in enumerate(P.astype(np.float32))}
    idx = sorted(index[tuple(np.round(a, 5))] for a in approx.astype(np.float32))

    # rand per paar opeenvolgende benaderingspunten: fit op de binnenste contourpunten
    edges = []
    for a, b in zip(idx, idx[1:] + [idx[0] + n_pts]):
        seg = P[np.arange(a, b + 1) % n_pts]
        m = max(2, int(0.15 * len(seg)))
        core = seg[m:-m] if len(seg) > 2 * m + 2 else seg
        nrm, off, _ = fit_line(core)
        d = seg[-1] - seg[0]
        outward = np.array([d[1], -d[0]])  # rechts van de looprichting = buiten (tegen de klok in)
        if nrm @ outward < 0:
            nrm, off = -nrm, -off
        edges.append([nrm, off, np.linalg.norm(d)])

    def intersect(e1, e2):
        A = np.array([e1[0], e2[0]])
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        return np.linalg.solve(A, [e1[1], e2[1]])

    # Een afgeronde hoek wordt door approxPolyDP een keten van korte randen die elk een déél van de
    # draaihoek nemen. Zo'n rand is kort ten opzichte van zijn buren en draait naar beide kanten
    # minder dan ~70°: weglaten en de buren laten snijden (de afronding volgt hieronder).
    def turn(e1, e2) -> float:
        return math.degrees(math.acos(float(np.clip(e1[0] @ e2[0], -1, 1))))

    changed = True
    while changed and len(edges) > 3:
        changed = False
        order = sorted(range(len(edges)), key=lambda i: edges[i][2])
        for i in order:
            e = edges[i]
            prev, nxt = edges[i - 1], edges[(i + 1) % len(edges)]
            # vergelijk met de langste buur: bij een hoek van meerdere koorden is de andere buur ook kort
            if e[2] > 0.35 * max(prev[2], nxt[2]) and e[2] > 2 * px:
                continue
            t1, t2 = turn(prev, e), turn(e, nxt)
            if t1 > 85 or t2 > 85 or t1 + t2 > 120:  # een echte trede draait 2 x 90°
                continue
            X = intersect(prev, nxt)
            if X is None:
                continue
            edges.pop(i)
            changed = True
            break

    # parallelle opeenvolgende randen (bijv. na te fijne benadering) samenvoegen
    k = 0
    while k < len(edges) and len(edges) > 3:
        if edges[k - 1][0] @ edges[k][0] > 0.9995:
            edges.pop(k)
        else:
            k += 1

    center = P.mean(axis=0)
    angles = np.array([math.atan2(e[0][1], e[0][0]) for e in edges])
    offsets = np.array([e[1] - e[0] @ center for e in edges])
    prof = Profile("polygon", center, angles, offsets, np.zeros(len(edges)))

    # afrondingsstraal per hoek uit de afstand hoekpunt → contour
    V = prof.vertices()
    fillets = np.zeros(len(edges))
    for k2, v in enumerate(V):
        n1, n2 = prof.normals()[k2 - 1], prof.normals()[k2]
        alpha = math.pi - math.acos(float(np.clip(n1 @ n2, -1, 1)))  # binnenhoek tussen de randen
        if alpha < 1e-3:  # teruglopende randen (piek in een rommelige contour): geen afronding
            continue
        delta = float(np.min(np.linalg.norm(P - v, axis=1))) - 0.7 * px
        factor = 1.0 / math.sin(alpha / 2) - 1.0
        r = max(delta, 0.0) / factor if factor > 1e-6 else 0.0
        fillets[k2] = r if r >= min_fillet else 0.0
    prof.fillets = fillets
    # afrondingen die niet passen verkleinen
    for _ in range(20):
        if prof.is_valid():
            break
        prof.fillets *= 0.8
    return prof


def _circle_with_blemish(contour: np.ndarray, shape, origin, px: float) -> tuple[float, float, float] | None:
    """Een rond gat met een kleine storing aan de rand (bijv. een 'staart' door een maskerfout)?

    Zoekt de grootste ingeschreven cirkel en fit daarna een cirkel op alleen de contourpunten die
    daarbij horen. Geeft (x, y, d) in mm, of None als de opening echt niet rond is (sleuf, hoek).
    """
    region = np.zeros(shape, np.uint8)
    cv2.drawContours(region, [contour], -1, 1, thickness=-1)
    cv2.drawContours(region, [contour], -1, 0, thickness=1)  # de contour zelf hoort bij het object
    dt = cv2.distanceTransform(region, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    r_in = float(dt.max())
    if r_in < 2.0:
        return None
    cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)
    pts = contour.reshape(-1, 2).astype(float)
    dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    near = np.abs(dist - r_in) < max(1.5, 0.15 * r_in)
    # de passende punten moeten rondom liggen (een lange staart heeft veel randpunten, maar het gat
    # is goed bepaald zolang de cirkel over het grootste deel van de omtrek zichtbaar is)
    ang = np.arctan2(pts[near, 1] - cy, pts[near, 0] - cx)
    coverage = len(np.unique(np.floor((ang + math.pi) / (2 * math.pi) * 36).astype(int))) / 36
    if coverage < 0.6 or region.sum() > 1.8 * math.pi * r_in ** 2:
        return None
    hx, hy, hr, hrms = fit_circle(_contour_mm(pts[near].reshape(-1, 1, 2), origin, px))
    if hrms > max(0.6 * px, 0.05 * hr) or abs(hr - r_in * px) > max(2 * px, 0.2 * hr):
        return None
    return hx, hy, 2 * hr - px


def from_footprint(fp: np.ndarray, origin, px: float, *, min_hole_d: float = 1.5, min_fillet: float = 1.0,
                   lenient_holes: bool = False) -> tuple[Profile, list[Hole], list[np.ndarray]]:
    """Bovenaanzicht (bool-raster, rijen = y) → buitencontour, ronde gaten en overige uitsparingen.

    `lenient_holes`: ook onregelmatig begrensde, ongeveer ronde openingen als gat behandelen. Nodig
    bij een footprint uit bovenaanzichten: de doorkijk door een gat is door parallax lensvormig.
    """
    img = fp.astype(np.uint8)
    contours, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Geen objectcontour gevonden")
    hier = hier[0]
    outer_i = max((i for i in range(len(contours)) if hier[i][3] < 0), key=lambda i: cv2.contourArea(contours[i]))
    P = _contour_mm(contours[outer_i], origin, px)
    if _signed_area(P) < 0:
        P = P[::-1]

    cx, cy, r, rms = fit_circle(P)
    area = abs(_signed_area(P))
    if rms < max(0.6 * px, 0.01 * r) and abs(area - math.pi * r * r) < 0.04 * math.pi * r * r:
        outer = Profile("circle", np.array([cx, cy]), radius=r + 0.5 * px)
    else:
        outer = polygon_from_contour(P, px, min_fillet=min_fillet)
        outer.offsets = outer.offsets + 0.5 * px  # contour loopt door pixelmiddens, rand ligt een halve pixel verder

    holes: list[Hole] = []
    cutouts: list[np.ndarray] = []
    for i in range(len(contours)):
        if hier[i][3] != outer_i:
            continue
        Q = _contour_mm(contours[i], origin, px)
        if len(Q) < 8 or abs(_signed_area(Q)) < math.pi * (min_hole_d / 2) ** 2:
            continue
        hx, hy, hr, hrms = fit_circle(Q)
        q_area = abs(_signed_area(Q))
        roundish = abs(q_area - math.pi * hr * hr) < 0.25 * math.pi * hr * hr and hrms < max(2 * px, 0.12 * hr)
        if hrms < max(0.6 * px, 0.03 * hr) or (lenient_holes and roundish):
            # de gatcontour loopt door de objectpixels rond het gat: een halve pixel buiten de rand
            holes.append(Hole(hx, hy, 2 * hr - px))
        elif lenient_holes and (circ := _circle_with_blemish(contours[i], img.shape, origin, px)) is not None:
            holes.append(Hole(*circ))
        else:
            approx = cv2.approxPolyDP(Q.astype(np.float32).reshape(-1, 1, 2), max(1.2 * px, 0.5), True)
            poly = approx.reshape(-1, 2).astype(float)
            if _signed_area(poly) < 0:
                poly = poly[::-1]
            cutouts.append(poly)
    return outer, holes, cutouts
