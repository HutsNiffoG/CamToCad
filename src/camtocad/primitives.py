"""Vlakken, cilinders en bollen in de vrije vorm (v0.18): exacte referentiegeometrie en maten uit de mesh.

Een mesh is in een CAD-programma lastig om mee verder te werken: er zijn geen vlakken om op te schetsen en geen assen
om op te meten. Gefreesde en gedraaide onderdelen bestaan echter grotendeels uit vlakken en cilinders. Die zoekt dit
module in de mesh van de vrije vorm (ARCHITECTURE.md §6.5, 'Stap C', en §6.9: uitvoerniveau 0 is de mesh plus gefitte
vlakken, assen en cilinders als referentiegeometrie), met RANSAC zoals Schnabel e.a. (2007):

1. per hoekpunt de normaal (gewogen naar het oppervlak van de driehoeken eromheen, en ook een over twee ringen
   gemiddelde normaal om kandidaten mee te maken);
2. elke ronde kandidaten van alle soorten: een vlak uit één hoekpunt met zijn normaal, een cilinder uit twee hoekpunten
   vlak bij elkaar (de as staat loodrecht op beide normalen, de normaallijnen snijden elkaar op de as) en een bol uit
   dezelfde twee (het punt waar de normaallijnen elkaar het dichtst naderen). Elke kandidaat krijgt als score het
   oppervlak binnen een kleine afstand met een normaal die erbij past (op een steekproef van de hoekpunten). De beste
   wint; zo eet een rij smalle facetten geen cilinder op, want de cilinder heeft meer oppervlak;
3. de winnaar groeit tot het samenhangende stuk mesh, wordt opnieuw gefit (kleinste kwadraten, robuust) en opnieuw
   gegroeid, en moet dan een toets doorstaan. Een vlak mag niet doorbuigen (een stuk van een flauw gebogen oppervlak
   past binnen de afstand ook in een vlak) en het oppervlak mag buiten zijn rand niet vloeiend doorlopen (dan is het
   een facet van een gebogen oppervlak, geen vlak met randen). Een cilinder moet een flink deel van de omtrek beslaan;
   een bol moet in twee richtingen gebogen zijn (anders is het een cilinder);
4. herhalen tot er geen vorm van betekenis meer is; daarna worden vlakken die in elkaars verlengde liggen samengevoegd.

Uit de vormen volgen de maten: per boring of as de diameter, per afronding de straal, en tussen evenwijdige vlakken die
tegenover elkaar liggen de afstand (een dikte als het materiaal ertussen zit en de vlakken breed zijn ten opzichte van
de afstand, een opening als er lucht tussen zit). Alles in het assenstelsel van het onderdeel (freeform.part_frame).

Een vorm uit een deel van de mesh dat alleen uit de silhouetten komt, is niet altijd echt: de omhullende van de
silhouetten bestaat zelf uit vlakke en kegelvormige stukken (elk recht stuk silhouetrand geeft een vlak stuk), en
boven een vlakke bovenkant zonder textuur laten de silhouetten een 'dak' staan. Een vorm telt daarom alleen als het
grootste deel ervan met stereo gemeten is, of als hij op de mat ligt (de onderkant), of als het een grote bol is (een
dubbel gekromd oppervlak kunnen de silhouetten niet zelf maken; daar liggen de randen van veel foto's dicht bij elkaar);
anders heet hij onzeker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .freeform import Mesh

TOL_MM = 0.3  # zo ver mag een hoekpunt van het vlak of de cilinder liggen
MAX_ANGLE_DEG = 15.0  # en zoveel mag zijn normaal afwijken
MIN_PLANE_MM2 = 40.0  # kleinere vlakken tellen niet
MIN_CURVED_MM2 = 25.0  # kleinere cilinders en bollen tellen niet
R_MIN_MM, R_MAX_MM = 1.5, 100.0  # straal van een cilinder of bol
R_MIN_PARTIAL_MM = 2.0  # een afronding met een kleinere straal is de gladmaking van een scherpe rand
MIN_COVERAGE_DEG = 60.0  # een cilinder moet minstens zo'n deel van de omtrek beslaan
FULL_COVERAGE_DEG = 270.0  # vanaf hier een boring of as, daaronder een afronding
SAG_MM = 0.25  # zoveel mag het binnenste van een vlak doorbuigen
SMOOTH_FRACTION = 0.3  # loopt het oppervlak langs meer dan dit deel van de rand vloeiend door: een facet
SPHERE_SPREAD_DEG = 10.0  # de normalen van een bol moeten in beide richtingen zo ver uiteenlopen (spreiding)
MAX_SHAPES = 40
MAX_MISSES = 12  # zoveel rondes achter elkaar zonder nieuwe vorm: klaar
MAX_ROUNDS = 150
SAMPLE = 10000  # zoveel hoekpunten per ronde om de kandidaten te scoren
N_CANDIDATES = 96  # zoveel kandidaten per soort per ronde
TOP_K = 6  # zoveel van de beste kandidaten proberen per ronde
CURVED_PENALTY = 1.15  # een cilinder of bol moet zoveel meer oppervlak hebben dan het beste vlak
PARALLEL_DEG = 3.0  # zo evenwijdig moeten twee vlakken zijn voor een afstand
MERGE_DEG = 1.5  # vlakken die zo evenwijdig zijn en in elkaars verlengde liggen, worden één vlak
AXIS_SNAP_DEG = 3.0  # een richting die zo dicht bij X, Y of Z ligt, heet langs die as
FACE_GROW_MM = 0.5  # een vlak in de STEP loopt zoveel verder dan zijn stuk mesh (dat stopt waar de rand afrondt)
MIN_STEREO = 0.5  # zo'n deel van een vorm moet met stereo gemeten zijn, anders is hij onzeker
MIN_BALL_MM2 = 200.0  # een bol (dubbel gekromd) van minstens dit oppervlak telt ook zonder stereo


@dataclass
class Shape:
    kind: str  # "vlak", "cilinder" of "bol"
    idx: np.ndarray  # hoekpunten van de mesh
    area: float  # mm²
    rms: float  # mm, afstand van de hoekpunten tot de vorm
    stereo: float = float("nan")  # deel van het oppervlak gemeten met stereo
    reliable: bool = True
    # vlak: n · x = d, n naar buiten
    n: np.ndarray | None = None
    d: float = 0.0
    # cilinder: as door p met richting a, straal r; bol: middelpunt p, straal r. hole: hol (boring, holle afronding,
    # kom) in plaats van bol (as, afronding, bol); t0..t1: axiaal bereik; coverage_deg: deel van de omtrek
    p: np.ndarray | None = None
    a: np.ndarray | None = None
    r: float = 0.0
    hole: bool = False
    t0: float = 0.0
    t1: float = 0.0
    coverage_deg: float = 0.0
    name: str = ""
    c: np.ndarray | None = None  # het midden van het stuk mesh (gewogen naar oppervlak)
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- de mesh

def vertex_normals(mesh: Mesh) -> np.ndarray:
    fn = np.cross(mesh.V[mesh.F[:, 1]] - mesh.V[mesh.F[:, 0]], mesh.V[mesh.F[:, 2]] - mesh.V[mesh.F[:, 0]])
    n = np.zeros_like(mesh.V)
    for k in range(3):
        np.add.at(n, mesh.F[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def vertex_areas(mesh: Mesh) -> np.ndarray:
    a = 0.5 * np.linalg.norm(np.cross(mesh.V[mesh.F[:, 1]] - mesh.V[mesh.F[:, 0]],
                                      mesh.V[mesh.F[:, 2]] - mesh.V[mesh.F[:, 0]]), axis=1)
    w = np.zeros(len(mesh.V))
    for k in range(3):
        np.add.at(w, mesh.F[:, k], a / 3.0)
    return w


def adjacency(mesh: Mesh):
    from scipy.sparse import coo_matrix
    E = np.concatenate([mesh.F[:, [0, 1]], mesh.F[:, [1, 2]], mesh.F[:, [2, 0]]])
    n = len(mesh.V)
    A = coo_matrix((np.ones(2 * len(E), np.float32), (np.concatenate([E[:, 0], E[:, 1]]),
                                                      np.concatenate([E[:, 1], E[:, 0]]))), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _component_with(adj, sel: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Van de geselecteerde hoekpunten (bool) het samenhangende stuk met de meeste `seeds` erin."""
    from scipy.sparse.csgraph import connected_components
    idx = np.flatnonzero(sel)
    if not len(idx):
        return idx
    sub = adj[idx][:, idx]
    _, lab = connected_components(sub, directed=False)
    pos = np.full(len(sel), -1)
    pos[idx] = np.arange(len(idx))
    s = pos[seeds]
    s = s[s >= 0]
    if not len(s):
        return np.zeros(0, np.int64)
    best = np.bincount(lab[s]).argmax()
    return idx[lab == best]


def _basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e1 = _unit(np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0]))
    return e1, np.cross(n, e1)


# ----------------------------------------------------------------------------- fitten

def _fit_plane(V: np.ndarray, w: np.ndarray, n_hint: np.ndarray) -> tuple[np.ndarray, float]:
    c = (V * w[:, None]).sum(axis=0) / w.sum()
    X = (V - c) * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    n = vt[2]
    if n @ n_hint < 0:
        n = -n
    return n, float(n @ c)


def _axis_dist(V: np.ndarray, p: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Afstand tot de as, en de eenheidsvector van de as naar het punt."""
    q = V - p
    radial = q - np.outer(q @ a, a)
    dist = np.linalg.norm(radial, axis=1)
    return dist, radial / np.maximum(dist, 1e-12)[:, None]


def _fit_cylinder(V: np.ndarray, w: np.ndarray, p0: np.ndarray, a0: np.ndarray, r0: float):
    """Kleinste kwadraten (robuust): as (punt, richting) en straal."""
    from scipy.optimize import least_squares
    a0 = _unit(a0)
    e1, e2 = _basis(a0)
    sw = np.sqrt(w / w.mean())

    def unpack(x):
        return p0 + x[2] * e1 + x[3] * e2, _unit(a0 + x[0] * e1 + x[1] * e2), x[4]

    def res(x):
        p, a, r = unpack(x)
        return (_axis_dist(V, p, a)[0] - r) * sw

    sol = least_squares(res, np.array([0.0, 0.0, 0.0, 0.0, r0]), loss="soft_l1", f_scale=TOL_MM, max_nfev=60)
    p, a, r = unpack(sol.x)
    # het punt op de as in het midden van het stuk
    return p + ((V - p) @ a).mean() * a, a, float(r)


def _fit_sphere(V: np.ndarray, w: np.ndarray, c0: np.ndarray, r0: float):
    from scipy.optimize import least_squares
    sw = np.sqrt(w / w.mean())
    sol = least_squares(lambda x: (np.linalg.norm(V - x[:3], axis=1) - x[3]) * sw, np.append(c0, r0),
                        loss="soft_l1", f_scale=TOL_MM, max_nfev=60)
    return sol.x[:3], float(sol.x[3])


def _inliers(kind: str, prm: tuple, V: np.ndarray, N: np.ndarray, tol: float, cos_t: float) -> np.ndarray:
    if kind == "vlak":
        n, d = prm
        return (np.abs(V @ n - d) < tol) & (N @ n > cos_t)
    if kind == "cilinder":
        c, a, r, hole = prm
        dist, u = _axis_dist(V, c, a)
    else:
        c, r, hole = prm
        q = V - c
        dist = np.linalg.norm(q, axis=1)
        u = q / np.maximum(dist, 1e-12)[:, None]
    s = -1.0 if hole else 1.0
    return (np.abs(dist - r) < tol) & (s * np.einsum("ij,ij->i", N, u) > cos_t)


def _refit(kind: str, prm: tuple, V: np.ndarray, N: np.ndarray, W: np.ndarray) -> tuple:
    if kind == "vlak":
        return _fit_plane(V, W, prm[0])
    if kind == "cilinder":
        c, a, r = _fit_cylinder(V, W, prm[0], prm[1], prm[2])
        _, u = _axis_dist(V, c, a)
        return c, a, abs(r), bool(np.average(np.einsum("ij,ij->i", N, u), weights=W) < 0)
    c, r = _fit_sphere(V, W, prm[0], prm[1])
    u = _unit(V - c)
    return c, abs(r), bool(np.average(np.einsum("ij,ij->i", N, u), weights=W) < 0)


def _residual(kind: str, prm: tuple, V: np.ndarray) -> np.ndarray:
    if kind == "vlak":
        return V @ prm[0] - prm[1]
    if kind == "cilinder":
        return _axis_dist(V, prm[0], prm[1])[0] - prm[2]
    return np.linalg.norm(V - prm[0], axis=1) - prm[1]


def _score_curved(kind: str, prms: list[tuple], V: np.ndarray, N: np.ndarray, W: np.ndarray, tol: float,
                  cos_t: float, chunk: int = 16) -> np.ndarray:
    """Het oppervlak binnen de afstand (met een passende normaal) voor een rij cilinders of bollen tegelijk."""
    C = np.array([p[0] for p in prms])
    if kind == "cilinder":
        A = np.array([p[1] for p in prms])
        R = np.array([p[2] for p in prms])
        S = np.array([-1.0 if p[3] else 1.0 for p in prms])
    else:
        R = np.array([p[1] for p in prms])
        S = np.array([-1.0 if p[2] else 1.0 for p in prms])
    out = np.zeros(len(prms))
    for k in range(0, len(prms), chunk):
        sl = slice(k, k + chunk)
        q = V[None, :, :] - C[sl, None, :]
        if kind == "cilinder":
            t = np.einsum("ksj,kj->ks", q, A[sl])
            q = q - t[..., None] * A[sl, None, :]
        dist = np.linalg.norm(q, axis=2)
        cosn = np.einsum("ksj,sj->ks", q, N) / np.maximum(dist, 1e-12)
        ok = (np.abs(dist - R[sl, None]) < tol) & (S[sl, None] * cosn > cos_t)
        out[sl] = ok @ W
    return out


def _coverage(ang_deg: np.ndarray) -> float:
    """Welk deel van de omtrek (graden) de hoeken beslaan: 360 min het grootste gat ertussen."""
    a = np.sort(np.mod(ang_deg, 360.0))
    if len(a) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    return float(360.0 - gaps.max())


# ----------------------------------------------------------------------------- kandidaten

def _cylinder_from_pair(vi, ni, vj, nj) -> tuple | None:
    a = np.cross(ni, nj)
    if np.linalg.norm(a) < 1e-6:
        return None
    a = _unit(a)
    pj = vj - ((vj - vi) @ a) * a  # in het vlak door i loodrecht op a
    ts, *_ = np.linalg.lstsq(np.column_stack([ni, -nj]), pj - vi, rcond=None)
    r = abs(ts[0])
    if not R_MIN_MM <= r <= R_MAX_MM or abs(abs(ts[1]) - r) > 0.2 * r:
        return None
    return vi + ts[0] * ni, a, float(r), bool(ts[0] > 0)  # de as aan de buitenkant: hol


def _sphere_from_pair(vi, ni, vj, nj) -> tuple | None:
    ts, *_ = np.linalg.lstsq(np.column_stack([ni, -nj]), vj - vi, rcond=None)
    c = 0.5 * (vi + ts[0] * ni + vj + ts[1] * nj)
    r = float(np.linalg.norm(vi - c))
    if not R_MIN_MM <= r <= R_MAX_MM or abs(abs(ts[0]) - abs(ts[1])) > 0.2 * r:
        return None
    return c, r, bool(ts[0] > 0)


class _Detector:
    def __init__(self, mesh: Mesh, seed: int, tol: float):
        from scipy.spatial import cKDTree
        self.V = mesh.V
        self.N, self.W, self.adj = vertex_normals(mesh), vertex_areas(mesh), adjacency(mesh)
        # de normaal over twee ringen gemiddeld: rustiger om kandidaten mee te maken
        self.Ns = _unit(self.adj @ (self.adj @ (self.N * self.W[:, None]) + self.N * self.W[:, None]))
        self.flat = np.einsum("ij,ij->i", self.N, self.Ns) > np.cos(np.radians(5.0))
        self.tree = cKDTree(self.V)
        self.rng = np.random.default_rng(seed)
        self.tol, self.cos_t = tol, np.cos(np.radians(MAX_ANGLE_DEG))
        n = len(self.V)
        self.free = np.ones(n, bool)
        self.plane_ok = np.ones(n, bool)  # mag nog een vlak beginnen (geen afgewezen facet)
        self.curved_ok = np.ones(n, bool)

    # -- kandidaten met een score op een steekproef van de vrije hoekpunten
    def candidates(self) -> list[tuple[float, str, tuple, int]]:
        V, N, Ns, W, rng = self.V, self.N, self.Ns, self.W, self.rng
        cand = np.flatnonzero(self.free)
        S = cand if len(cand) <= SAMPLE else rng.choice(cand, SAMPLE, replace=False)
        scale = W[cand].sum() / max(W[S].sum(), 1e-12)
        VS, NS, WS = V[S], N[S], W[S]
        out = []
        pool = np.flatnonzero(self.free & self.flat & self.plane_ok)
        if len(pool):
            sd = rng.choice(pool, min(N_CANDIDATES, len(pool)), replace=False)
            n = Ns[sd]
            d = np.einsum("ij,ij->i", V[sd], n)
            ok = (np.abs(VS @ n.T - d) < self.tol) & (NS @ n.T > self.cos_t)
            score = (WS @ ok) * scale
            out += [(float(score[k]), "vlak", (n[k], float(d[k])), int(sd[k])) for k in range(len(sd))]
        pool = np.flatnonzero(self.free & self.curved_ok)
        if len(pool):
            # genoeg paren om ook een kleine vorm in wat er nog over is te raken (ongeveer de helft van de paren
            # met een punt op de vorm is goed): bij 25 mm² op 1800 mm² vrij oppervlak zo'n 200
            n_pairs = int(np.clip(4.0 * W[cand].sum() / MIN_CURVED_MM2, N_CANDIDATES, 4 * N_CANDIDATES))
            I = rng.choice(pool, min(n_pairs, len(pool)), replace=False)
            cyl, sph = [], []
            for i, near in zip(I, self.tree.query_ball_point(V[I], 6.0)):
                near = np.asarray(near, np.int64)
                near = near[self.free[near] & (Ns[near] @ Ns[i] < np.cos(np.radians(4.0)))]
                if not len(near):
                    continue
                j = int(near[rng.integers(len(near))])
                c = _cylinder_from_pair(V[i], Ns[i], V[j], Ns[j])
                if c is not None:
                    cyl.append((c, int(i)))
                b = _sphere_from_pair(V[i], Ns[i], V[j], Ns[j])
                if b is not None:
                    sph.append((b, int(i)))
            for kind, lst in (("cilinder", cyl), ("bol", sph)):
                if lst:
                    score = _score_curved(kind, [c for c, _ in lst], VS, NS, WS, self.tol, self.cos_t) * scale
                    out += [(float(sc) / CURVED_PENALTY, kind, c, i) for sc, (c, i) in zip(score, lst)]
        out.sort(key=lambda c: -c[0])
        return out

    # -- groeien en opnieuw fitten
    def grow(self, kind: str, prm: tuple, seed: int) -> tuple[np.ndarray, tuple]:
        V, N, W = self.V, self.N, self.W
        region = np.zeros(0, np.int64)
        free = np.flatnonzero(self.free)

        def select(prm):
            sel = np.zeros(len(V), bool)
            sel[free[_inliers(kind, prm, V[free], N[free], self.tol, self.cos_t)]] = True
            return sel

        for _ in range(3):
            region = _component_with(self.adj, select(prm), np.array([seed]) if not len(region) else region)
            if len(region) < 8:
                return np.zeros(0, np.int64), prm
            prm = _refit(kind, prm, V[region], N[region], W[region])
        return _component_with(self.adj, select(prm), region), prm

    # -- de toetsen
    def plane_check(self, region: np.ndarray, n: np.ndarray, d: float) -> str:
        """'' als het een vlak is, anders waarom niet."""
        from scipy.spatial import cKDTree
        V, N, W = self.V, self.N, self.W
        lo, hi = V[region].min(axis=0) - 3.0, V[region].max(axis=0) + 3.0
        mask = np.zeros(len(V), bool)
        mask[region] = True
        near = np.flatnonzero(np.all((V >= lo) & (V <= hi), axis=1) & ~mask)
        # het binnenste (verder dan 1 mm van de rand) mag niet doorbuigen
        inner = region
        if len(near):
            dist_in, _ = cKDTree(V[near]).query(V[region], distance_upper_bound=1.0)
            inner = region[~np.isfinite(dist_in)]
        if W[inner].sum() >= 0.3 * W[region].sum() and len(inner) >= 30:
            sag = _sag(V[inner], W[inner], n, d)
            if sag > SAG_MM:
                return f"buigt {sag:.2f} mm door"
        # buiten de rand (1-3 mm) vloeiend verder in hetzelfde vlak: een facet van een gebogen oppervlak
        if len(near):
            dist_out, _ = cKDTree(V[region]).query(V[near], distance_upper_bound=3.0)
            band = near[(dist_out > 1.0) & np.isfinite(dist_out)]
            if W[band].sum() > 0:
                smooth = (np.abs(V[band] @ n - d) < 0.6) & (N[band] @ n > np.cos(np.radians(8.0)))
                frac = float(W[band][smooth].sum() / W[band].sum())
                if frac > SMOOTH_FRACTION:
                    return f"loopt langs {frac:.0%} van de rand vloeiend door"
        return ""

    def make_shape(self, kind: str, region: np.ndarray, prm: tuple) -> tuple[Shape | None, str]:
        shape, why = self._make_shape(kind, region, prm)
        if shape is not None:
            shape.c = np.average(self.V[region], axis=0, weights=self.W[region])
        return shape, why

    def _make_shape(self, kind: str, region: np.ndarray, prm: tuple) -> tuple[Shape | None, str]:
        V, N, W = self.V, self.N, self.W
        area = float(W[region].sum())
        rms = float(np.sqrt(np.average(_residual(kind, prm, V[region]) ** 2, weights=W[region])))
        if kind == "vlak":
            if area < MIN_PLANE_MM2:
                return None, "te klein"
            why = self.plane_check(region, *prm)
            return (None, why) if why else (Shape("vlak", region, area, rms, n=prm[0], d=prm[1]), "")
        if area < MIN_CURVED_MM2:
            return None, "te klein"
        if kind == "cilinder":
            c, a, r, hole = prm
            if not R_MIN_MM <= r <= R_MAX_MM:
                return None, f"straal {r:.1f} mm"
            _, u = _axis_dist(V[region], c, a)
            ref = u[0]
            cov = _coverage(np.degrees(np.arctan2(u @ np.cross(a, ref), u @ ref)))
            tt = (V[region] - c) @ a
            if cov < MIN_COVERAGE_DEG or (cov < FULL_COVERAGE_DEG and r < R_MIN_PARTIAL_MM) or np.ptp(tt) < 1.5:
                return None, f"omtrek {cov:.0f}°"
            return Shape("cilinder", region, area, rms, p=c, a=a, r=r, hole=hole, t0=float(tt.min()),
                         t1=float(tt.max()), coverage_deg=cov), ""
        c, r, hole = prm
        if not R_MIN_MM <= r <= R_MAX_MM:
            return None, f"straal {r:.1f} mm"
        # in twee richtingen gebogen: de normalen lopen in beide richtingen uiteen
        Nr = N[region]
        m = _unit(np.average(Nr, axis=0, weights=W[region]))
        e1, e2 = _basis(m)
        P = np.column_stack([Nr @ e1, Nr @ e2])
        ev = np.linalg.eigvalsh(np.cov(P.T, aweights=W[region]))
        if np.sqrt(max(ev[0], 0.0)) < np.sin(np.radians(SPHERE_SPREAD_DEG)):
            return None, "maar in één richting gebogen"
        return Shape("bol", region, area, rms, p=c, r=r, hole=hole), ""

    def run(self) -> list[Shape]:
        shapes: list[Shape] = []
        misses = rounds = 0
        # de kandidaten zijn willekeurig: pas na een reeks rondes zonder vorm is er niets meer te vinden
        while len(shapes) < MAX_SHAPES and misses < MAX_MISSES and rounds < MAX_ROUNDS:
            if self.W[self.free].sum() < MIN_CURVED_MM2:
                break
            rounds += 1
            found = None
            for score, kind, prm, seed in self.candidates()[:TOP_K]:
                if score < MIN_CURVED_MM2:
                    break
                if not (self.plane_ok if kind == "vlak" else self.curved_ok)[seed]:
                    continue  # uit een stuk dat in deze ronde al is afgewezen
                region, prm2 = self.grow(kind, prm, seed)
                if len(region) < 8:
                    (self.plane_ok if kind == "vlak" else self.curved_ok)[seed] = False
                    continue
                shape, why = self.make_shape(kind, region, prm2)
                if shape is None:
                    # dit stuk mesh begint geen vorm van deze soort meer (een afgewezen facet, een halve cilinder)
                    (self.plane_ok if kind == "vlak" else self.curved_ok)[region] = False
                    if kind != "vlak" and why.startswith(("straal", "omtrek")) and self.W[region].sum() > 200.0:
                        # een groot, flauw gebogen stuk dat geen cilinder of bol is (een vrij gevormd oppervlak):
                        # rest, anders komen er steeds weer kandidaten op uit
                        self.free[region] = False
                    continue
                found = shape
                break
            if found is None:
                misses += 1
                continue
            misses = 0
            shapes.append(found)
            self.free[found.idx] = False
        return _merge_planes(shapes, self.V, self.W)


def _sag(V: np.ndarray, w: np.ndarray, n: np.ndarray, d: float) -> float:
    """Hoeveel het stuk doorbuigt: het kwadratische deel van een tweedegraads oppervlak door de punten (top tot top),
    zonder het lineaire deel."""
    e1, e2 = _basis(n)
    u, v, r = V @ e1, V @ e2, V @ n - d
    u, v = u - np.average(u, weights=w), v - np.average(v, weights=w)
    sw = np.sqrt(w)
    A = np.column_stack([u * u, u * v, v * v, u, v, np.ones_like(u)])
    coef, *_ = np.linalg.lstsq(A * sw[:, None], r * sw, rcond=None)
    q = A[:, :3] @ coef[:3]
    B = A[:, 3:]
    lin, *_ = np.linalg.lstsq(B * sw[:, None], q * sw, rcond=None)
    q = q - B @ lin
    return float(np.ptp(q))


def _merge_planes(shapes: list[Shape], V: np.ndarray, W: np.ndarray) -> list[Shape]:
    """Vlakken die in elkaars verlengde liggen (bijv. een zijkant die door een naad in twee stukken viel) worden één:
    bijna evenwijdig, het midden van het kleinste ligt vlak bij het grootste (de richting van een klein stuk is minder
    zeker), en samen passen ze nog in één vlak."""
    out: list[Shape] = []
    for s in shapes:
        if s.kind == "vlak":
            for t in out:
                if t.kind != "vlak" or t.n @ s.n < np.cos(np.radians(MERGE_DEG)):
                    continue
                small, big = (s, t) if s.area <= t.area else (t, s)
                if abs(small.c @ big.n - big.d) > 2 * TOL_MM:
                    continue
                idx = np.concatenate([t.idx, s.idx])
                n, d = _fit_plane(V[idx], W[idx], t.n)
                rms = float(np.sqrt(np.average((V[idx] @ n - d) ** 2, weights=W[idx])))
                if rms > max(t.rms, s.rms) * 1.3 + 0.02:
                    continue
                t.idx, t.n, t.d, t.rms, t.area = idx, n, d, rms, float(W[idx].sum())
                t.c = np.average(V[idx], axis=0, weights=W[idx])
                break
            else:
                out.append(s)
        else:
            out.append(s)
    return out


def detect(mesh: Mesh, stereo: np.ndarray | None = None, seed: int = 0, tol: float = TOL_MM) -> list[Shape]:
    """De vlakken, cilinders en bollen in de mesh (assenstelsel van het onderdeel, de mat op z = 0), grootste eerst."""
    if len(mesh.V) < 50:
        return []
    det = _Detector(mesh, seed, tol)
    shapes = det.run()
    W = det.W
    for s in shapes:
        if stereo is not None and len(stereo) == len(mesh.V):
            s.stereo = float(np.average(stereo[s.idx], weights=W[s.idx]))
        on_mat = is_bottom(s)
        # de omhullende van de silhouetten bestaat uit vlakke en kegelvormige stukken (in één richting gebogen);
        # een groot bol stuk kan ze niet maken: dat liggen de randen van veel foto's dicht op elkaar, en daar klopt hij
        ball = s.kind == "bol" and not s.hole and s.area >= MIN_BALL_MM2
        s.reliable = bool(on_mat or ball or not np.isfinite(s.stereo) or s.stereo >= MIN_STEREO)
    shapes.sort(key=lambda s: (s.kind != "vlak", -s.area))
    _name(shapes)
    return shapes


# ----------------------------------------------------------------------------- beschrijving en maten

AXES = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}
NAME_RE = re.compile(r"^(vlak|boring|as|afronding|holle afronding|bol|kom|dikte|afstand|opening) \d+( Ø| R)?$")


def is_bottom(s: Shape) -> bool:
    """Is dit het vlak op de mat (de onderkant van het onderdeel)?"""
    return s.kind == "vlak" and s.n[2] < -0.98 and abs(float(s.c[2]) if s.c is not None else -s.d) < 0.5


def axis_label(v: np.ndarray) -> str | None:
    """'X', 'Y' of 'Z' als de richting (op het teken na) daar binnen AXIS_SNAP_DEG bij ligt."""
    for k, e in AXES.items():
        if abs(v @ e) > np.cos(np.radians(AXIS_SNAP_DEG)):
            return k
    return None


def base_name(s: Shape) -> str:
    if s.kind == "vlak":
        return "vlak"
    if s.kind == "bol":
        return "kom" if s.hole else "bol"
    if s.coverage_deg < FULL_COVERAGE_DEG:
        return "holle afronding" if s.hole else "afronding"
    return "boring" if s.hole else "as"


def _name(shapes: list[Shape]) -> None:
    counts: dict[str, int] = {}
    for s in shapes:
        base = base_name(s)
        counts[base] = counts.get(base, 0) + 1
        s.name = f"{base} {counts[base]}"


def _fmt(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}".replace(".", ",")


def describe(s: Shape) -> str:
    """Eén regel over de vorm, in het assenstelsel van het onderdeel."""
    if s.kind == "vlak":
        ax = axis_label(s.n)
        if ax is not None:
            k = "XYZ".index(ax)
            # waar het vlak ligt: bij zijn midden (een vlak dat een fractie scheef staat, snijdt de as elders)
            pos = float(s.c[k]) if s.c is not None else s.d / s.n[k]
            if is_bottom(s):
                return "vlak op de mat (de onderkant)"
            return f"vlak loodrecht op {ax}, {ax.lower()} = {_fmt(pos)} mm"
        tilt = np.degrees(np.arccos(min(1.0, abs(s.n[2]))))
        return f"schuin vlak, {_fmt(tilt, 1)}° uit de horizontaal"
    if s.kind == "bol":
        what = "holle bol (kom)" if s.hole else "bol"
        return f"{what} Ø {_fmt(2 * s.r)} mm, middelpunt ({_fmt(s.p[0], 1)}, {_fmt(s.p[1], 1)}, {_fmt(s.p[2], 1)})"
    ax = axis_label(s.a)
    where = ""
    if ax is not None:
        # de plaats van de as: de twee andere coördinaten
        a = AXES[ax]
        p = s.p - (s.p @ a) * a
        where = ", as op (" + ", ".join(f"{c.lower()} {_fmt(p['XYZ'.index(c)], 1)}" for c in "XYZ" if c != ax) + ")"
        direction = f"langs {ax}"
    else:
        direction = "schuin (richting " + ", ".join(_fmt(v, 3) for v in s.a) + ")"
    length = s.t1 - s.t0
    base = base_name(s)
    size = f"R {_fmt(s.r)} mm" if base.endswith("afronding") else f"Ø {_fmt(2 * s.r)} mm"
    return (f"{base} {size} {direction}{where}, lengte {_fmt(length, 1)} mm"
            + (f", {s.coverage_deg:.0f}° van de omtrek" if s.coverage_deg < 350 else ""))


def to_dict(s: Shape) -> dict:
    """De vorm voor report.json (mm, assenstelsel van het onderdeel)."""
    d = {"naam": s.name, "soort": s.kind, "beschrijving": describe(s), "oppervlak_mm2": round(s.area, 1),
         "rms_mm": round(s.rms, 3), "stereo": round(s.stereo, 3) if np.isfinite(s.stereo) else None,
         "betrouwbaar": s.reliable}
    if s.kind == "vlak":
        d.update(normaal=[round(float(v), 6) for v in s.n], afstand_tot_oorsprong_mm=round(s.d, 4),
                 midden_mm=[round(float(v), 3) for v in s.c] if s.c is not None else None)
    elif s.kind == "cilinder":
        d.update(as_punt_mm=[round(float(v), 4) for v in s.p], as_richting=[round(float(v), 6) for v in s.a],
                 straal_mm=round(s.r, 4), hol=s.hole, lengte_mm=round(s.t1 - s.t0, 3),
                 omtrek_graden=round(s.coverage_deg, 1))
    else:
        d.update(middelpunt_mm=[round(float(v), 4) for v in s.p], straal_mm=round(s.r, 4), hol=s.hole)
    return d


def _extent_2d(P: np.ndarray, n: np.ndarray) -> float:
    """De kleinste breedte van een stuk vlak (over de hoofdassen van de punten in het vlak)."""
    e1, e2 = _basis(n)
    uv = np.column_stack([P @ e1, P @ e2])
    uv -= uv.mean(axis=0)
    _, _, vt = np.linalg.svd(uv, full_matrices=False)
    r = uv @ vt.T
    return float(np.min(np.percentile(r, 98, axis=0) - np.percentile(r, 2, axis=0)))


def _facing(P: np.ndarray, wP: np.ndarray, Q: np.ndarray, n: np.ndarray, cell: float = 1.0) -> float:
    """Welk deel (naar oppervlak) van stuk P recht tegenover stuk Q ligt (langs de normaal n geprojecteerd)."""
    e1, e2 = _basis(n)
    q = np.column_stack([Q @ e1, Q @ e2])
    p = np.column_stack([P @ e1, P @ e2])
    lo = np.minimum(q.min(axis=0), p.min(axis=0)) - cell
    grid = np.zeros(tuple(np.ceil((np.maximum(q.max(axis=0), p.max(axis=0)) - lo) / cell).astype(int) + 3), bool)
    iq = np.floor((q - lo) / cell).astype(int)
    for dx in (-1, 0, 1):  # een cel speling
        for dy in (-1, 0, 1):
            grid[iq[:, 0] + 1 + dx, iq[:, 1] + 1 + dy] = True
    ip = np.floor((p - lo) / cell).astype(int)
    inside = grid[ip[:, 0] + 1, ip[:, 1] + 1]
    return float(wP[inside].sum() / max(wP.sum(), 1e-12))


def measurements(shapes: list[Shape], mesh: Mesh) -> list[dict]:
    """Maten uit de vormen: per boring, as en bol de diameter, per afronding de straal, en tussen evenwijdige vlakken
    die tegenover elkaar liggen de afstand: een dikte (materiaal ertussen, de vlakken breed ten opzichte van de
    afstand), een afstand (materiaal ertussen, smalle vlakken: bijv. de lengte over twee kopse kanten) of een opening
    (lucht ertussen). Elk met 'name', 'value' (mm), 'reliable', 'text' en 'axis' (X, Y, Z of None)."""
    W = vertex_areas(mesh)
    out = []
    for s in shapes:
        if s.kind == "vlak":
            continue
        partial = s.kind == "cilinder" and s.coverage_deg < FULL_COVERAGE_DEG
        out.append({"name": f"{s.name} {'R' if partial else 'Ø'}", "value": round((1 if partial else 2) * s.r, 3),
                    "reliable": s.reliable, "text": describe(s),
                    "axis": axis_label(s.a) if s.kind == "cilinder" else None, "shapes": [s.name]})
    planes = [s for s in shapes if s.kind == "vlak"]
    counts = {"dikte": 0, "afstand": 0, "opening": 0}
    for i, p in enumerate(planes):
        for q in planes[i + 1:]:
            if p.n @ q.n > -np.cos(np.radians(PARALLEL_DEG)):
                continue
            small, big = (p, q) if p.area <= q.area else (q, p)
            # de afstand van het midden van het kleinste vlak tot het grootste (de vlakken staan niet precies
            # evenwijdig: dan geldt de afstand daar waar ze tegenover elkaar liggen)
            c = np.average(mesh.V[small.idx], axis=0, weights=W[small.idx])
            gap = abs(float(c @ big.n) - big.d)
            if gap < 0.3:
                continue
            if _facing(mesh.V[small.idx], W[small.idx], mesh.V[big.idx], p.n) < 0.5:
                continue
            # materiaal ertussen: q ligt aan de binnenkant van p (tegen de normaal van p in)
            material = (mesh.V[q.idx].mean(axis=0) - mesh.V[p.idx].mean(axis=0)) @ p.n < 0
            if not material:
                kind = "opening"
            else:
                kind = "dikte" if gap < 0.5 * _extent_2d(mesh.V[small.idx], small.n) else "afstand"
            counts[kind] += 1
            ax = axis_label(p.n)
            out.append({"name": f"{kind} {counts[kind]}", "value": round(gap, 3),
                        "reliable": p.reliable and q.reliable,
                        "text": f"tussen {p.name} en {q.name}" + (f", langs {ax}" if ax else ""), "axis": ax,
                        "shapes": [p.name, q.name]})
    return out


# ----------------------------------------------------------------------------- kleuren en STEP

PALETTE_PLANES = [(0.85, 0.62, 0.35), (0.55, 0.80, 0.45), (0.80, 0.50, 0.75), (0.45, 0.75, 0.80), (0.90, 0.78, 0.45),
                  (0.60, 0.60, 0.90), (0.50, 0.85, 0.75), (0.85, 0.70, 0.60)]  # BGR, 0-1
COLOR_CYLINDER = (0.30, 0.55, 0.95)  # oranje
COLOR_SPHERE = (0.75, 0.40, 0.70)  # paars
COLOR_REST = (0.78, 0.78, 0.78)  # grijs


def colors(mesh: Mesh, shapes: list[Shape]) -> np.ndarray:
    """Per hoekpunt een kleur (BGR, 0-1): elk vlak een eigen tint, cilinders oranje, bollen paars, de rest grijs."""
    c = np.tile(np.array(COLOR_REST), (len(mesh.V), 1))
    k = 0
    for s in shapes:
        if s.kind == "vlak":
            c[s.idx] = PALETTE_PLANES[k % len(PALETTE_PLANES)]
            k += 1
        else:
            c[s.idx] = COLOR_CYLINDER if s.kind == "cilinder" else COLOR_SPHERE
    return c


def reference_step(shapes: list[Shape], mesh: Mesh, path) -> int:
    """De vormen als referentiegeometrie in een STEP-bestand (mm, assenstelsel van het onderdeel): elk vlak als
    begrensd vlak (de rand van zijn stuk mesh), elke boring, as en afronding als cilindervlak (zo ver als het stuk
    mesh) met de as als lijn, elke bol als bol. Om in een CAD-programma op te schetsen en te meten, naast de mesh.
    Geeft het aantal vormen erin."""
    import cadquery as cq
    parts, n = [], 0
    for s in shapes:
        try:
            if s.kind == "vlak":
                face = _plane_face(s, mesh)
                if face is None:
                    continue
                parts.append(face)
            elif s.kind == "cilinder":
                length = max(s.t1 - s.t0, 0.5)
                base = s.p + s.t0 * s.a
                solid = cq.Solid.makeCylinder(s.r, length, cq.Vector(*base), cq.Vector(*s.a))
                parts += [f for f in solid.Faces() if f.geomType() == "CYLINDER"]
                ext = 0.15 * length + 2.0
                parts.append(cq.Edge.makeLine(cq.Vector(*(base - ext * s.a)), cq.Vector(*(base + (length + ext) * s.a))))
            else:
                parts += cq.Solid.makeSphere(s.r, cq.Vector(*s.p), angleDegrees1=-90, angleDegrees2=90).Faces()
            n += 1
        except Exception:  # noqa: BLE001 - één vorm die niet lukt, mag de rest niet tegenhouden
            s.extra["niet in STEP"] = True
    if not parts:
        return 0
    cq.exporters.export(cq.Compound.makeCompound(parts), str(path), exportType="STEP")
    return n


def _plane_face(s: Shape, mesh: Mesh):
    """Het vlak, begrensd door de rand van zijn stuk mesh (buitenrand en gaten), in het vlak geprojecteerd."""
    import cadquery as cq
    import cv2
    e1, e2 = _basis(s.n)
    origin = s.d * s.n

    def to3(q):
        return [cq.Vector(*(origin + a * e1 + b * e2)) for a, b in q]

    # de rand als contour van het stuk in een raster van 0,25 mm (gesloten, met gaten)
    res = 0.25
    grow = round(FACE_GROW_MM / res)
    inside = np.isin(mesh.F, s.idx).all(axis=1)
    T = mesh.V[mesh.F[inside]]
    if not len(T):
        return None
    uv = np.stack([T @ e1, T @ e2], axis=-1)  # (driehoeken, 3, 2)
    lo = uv.reshape(-1, 2).min(axis=0) - 2 * res
    size = np.ceil((uv.reshape(-1, 2).max(axis=0) - lo) / res).astype(int) + 3
    img = np.zeros((size[1], size[0]), np.uint8)
    for q in np.round((uv - lo) / res * 16).astype(np.int32):
        cv2.fillConvexPoly(img, q, 1, cv2.LINE_8, 4)
    # het stuk mesh houdt op waar de (gladgestreken) rand begint af te ronden, zo'n halve mm voor de echte rand
    img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    contours, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    faces = []
    for k_out, c in enumerate(contours):
        if hier[0][k_out][3] != -1 or cv2.contourArea(c) * res * res < 4.0:
            continue
        outer = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2) * res + lo
        if len(outer) < 3:
            continue
        holes = [cv2.approxPolyDP(h, 1.0, True).reshape(-1, 2) * res + lo for k, h in enumerate(contours)
                 if hier[0][k][3] == k_out and cv2.contourArea(h) * res * res > 2.0]
        wire = cq.Wire.makePolygon(to3(outer), close=True)
        inner = [cq.Wire.makePolygon(to3(h), close=True) for h in holes if len(h) >= 3]
        faces.append(cq.Face.makeFromWires(wire, inner))
    if not faces:
        return None
    return faces[0] if len(faces) == 1 else cq.Compound.makeCompound(faces)
