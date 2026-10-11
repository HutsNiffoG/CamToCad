"""Vrije vorm (v0.17): de vorm van een onderdeel dat niet plat is, als mesh uit de silhouetten en stereo.

Route A fit een parametrisch 2,5D-model. Voor een onderdeel dat niet plat is (een nietmachine, een liggend draaideel,
een beugel met een opstaande flens) bestaat zo'n model niet. Dan levert de scan uitvoerniveau 0 uit
ARCHITECTURE.md §6.9: een mesh (STL en 3MF) met de buitenmaten.

Eerst de visual hull, met subpixelnauwkeurige randen. Per foto de afstand (in pixels) tot de rand van de zekere mat,
met een teken: positief waar de pixel geen zekere mat is (object, of geen bewijs), negatief op zekere mat. Een punt in
de ruimte ligt buiten het onderdeel als het in minstens drie foto's op zekere mat valt. De derde kleinste afstand over
de foto's, omgerekend naar mm op de diepte van het punt, is daarom een afstandsveld met de hull als nulniveau. Een
paar foto's met een mislukt masker snijden zo niets weg, en losse vlekjes 'zekere mat' op een onderdeel met textuur
tellen niet. De mat zelf (z = 0) sluit de onderkant af.

De hull is te vol waar geen foto de mat achter het onderdeel ziet: een 'dak' boven een vlakke of licht gebogen
bovenkant, een kuil, de spleet onder de arm van een nietmachine. Daar helpt stereo (mvs.py), als het onderdeel er
textuur heeft: per foto een dieptekaart uit de buurfoto's, samen een tweede afstandsveld (TSDF). Het onderdeel is waar
beide velden 'binnen' zeggen.

Het oppervlak komt uit het veld met 'surface nets': een hoekpunt per cel waar het veld van teken wisselt, en een
vierhoek over elke ribbe met een tekenwisseling. Daarna alleen het grootste samenhangende stuk, en de hoekpunten met
een paar Newtonstappen precies op het nulniveau. Eerst een grof rooster over het hele zoekgebied, dan een fijn rooster
alleen vlak bij het oppervlak.

Wat de scan niet kan zien: een holte die in geen enkele foto als mat of met textuur te zien is (een spleet die alleen
van opzij en heel laag te zien is, de ruimte onder een overhang vlak bij de mat, een blind gat). Daar is de mesh te
vol; het rapport toont welke delen alleen uit de silhouetten komen.
"""

from __future__ import annotations

import io
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calib import Pose, project
from .masks import ViewMasks

NO_EVIDENCE = 1.0e4  # px: buiten de foto, of geen enkele zekere mat in de foto
MIN_VIEWS_OUT = 3  # zoveel foto's moeten een punt op zekere mat zien voordat het wegvalt
MIN_MAT_PX = 40  # kleinere losse stukjes zekere mat tellen niet (vlekjes op een onderdeel met textuur)
FLOOR_MM = 1.5  # vlak boven de mat beslissen de foto's recht van boven mee (zie _field_at)
TOP_TILT_DEG = 25.0  # zo schuin mag een foto 'recht van boven' zijn


@dataclass
class Mesh:
    V: np.ndarray  # (n, 3) mm
    F: np.ndarray  # (m, 3) hoekpuntnummers, tegen de klok in van buiten gezien

    def normals(self) -> np.ndarray:
        n = np.cross(self.V[self.F[:, 1]] - self.V[self.F[:, 0]], self.V[self.F[:, 2]] - self.V[self.F[:, 0]])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def area(self) -> float:
        n = np.cross(self.V[self.F[:, 1]] - self.V[self.F[:, 0]], self.V[self.F[:, 2]] - self.V[self.F[:, 0]])
        return float(0.5 * np.linalg.norm(n, axis=1).sum())

    def volume(self) -> float:
        a, b, c = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.V.min(axis=0), self.V.max(axis=0)

    def transformed(self, R: np.ndarray, t: np.ndarray) -> Mesh:
        return Mesh(self.V @ R.T + t, self.F.copy())


@dataclass
class Field:
    origin: np.ndarray  # mm: roosterpunt (0, 0, 0)
    h: float  # roosterafstand, mm
    values: np.ndarray  # (nx, ny, nz) float32, mm; positief binnen het onderdeel

    def sample(self, pts: np.ndarray) -> np.ndarray:
        """Trilineair geïnterpoleerd; buiten het rooster de rand."""
        g = (pts - self.origin) / self.h
        hi = np.array(self.values.shape) - 1
        g = np.clip(g, 0, hi - 1e-6)
        i0 = np.floor(g).astype(np.int64)
        f = g - i0
        v = self.values
        out = np.zeros(len(pts), np.float64)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    w = ((f[:, 0] if di else 1 - f[:, 0]) * (f[:, 1] if dj else 1 - f[:, 1])
                         * (f[:, 2] if dk else 1 - f[:, 2]))
                    out += w * v[i0[:, 0] + di, i0[:, 1] + dj, i0[:, 2] + dk]
        return out

    def gradient(self, pts: np.ndarray) -> np.ndarray:
        e = 0.25 * self.h
        return np.stack([(self.sample(pts + d) - self.sample(pts - d)) / (2 * e)
                         for d in (np.array([e, 0, 0]), np.array([0, e, 0]), np.array([0, 0, e]))], axis=1)


def tilt_deg(pose: Pose) -> float:
    """Hoe schuin de camera kijkt: de hoek tussen de optische as en recht naar beneden."""
    axis = pose.R.T @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(-axis[2], -1.0, 1.0))))


def signed_distance(m: ViewMasks) -> np.ndarray:
    """Per pixel de afstand (px) tot de rand van de zekere mat: positief waar de pixel geen zekere mat is (object, of
    geen bewijs), negatief op zekere mat. De rand ligt halverwege tussen de pixelmiddens."""
    mat = (m.edge_bg if m.edge_bg is not None else m.bg).astype(np.uint8)
    if not mat.any():
        return np.full(mat.shape, NO_EVIDENCE, np.float32)
    # losse vlekjes 'zekere mat' (een paar pixels) midden op een onderdeel met textuur: weg. Een echt gat of de mat
    # rond het onderdeel is veel groter
    _, lab, stats, _ = cv2.connectedComponentsWithStats(mat, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < MIN_MAT_PX
    small[0] = False
    if small.any():
        mat = np.where(small[lab], 0, mat).astype(np.uint8)
    on_mat = cv2.distanceTransform(mat, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    off_mat = cv2.distanceTransform(1 - mat, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(mat > 0, 0.5 - on_mat, off_mat - 0.5).astype(np.float32)


def _sample_image(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilineair (cv2.remap, in stukken van 4096 punten breed); buiten het beeld NO_EVIDENCE."""
    n = len(uv)
    width = 4096
    rows = -(-n // width)
    pad = rows * width - n
    u = np.concatenate([uv[:, 0], np.full(pad, -10.0)]).astype(np.float32).reshape(rows, width)
    v = np.concatenate([uv[:, 1], np.full(pad, -10.0)]).astype(np.float32).reshape(rows, width)
    u[~np.isfinite(u)] = -10.0
    v[~np.isfinite(v)] = -10.0
    out = np.empty((rows, width), np.float32)
    for r0 in range(0, rows, 4000):  # cv2.remap: hooguit 32767 rijen
        out[r0:r0 + 4000] = cv2.remap(img, u[r0:r0 + 4000], v[r0:r0 + 4000], cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=NO_EVIDENCE)
    return out.ravel()[:n]


def _kth_smallest(values, n: int, order: int) -> np.ndarray:
    """De `order`-de kleinste waarde per punt over een reeks arrays (zonder ze allemaal tegelijk te bewaren)."""
    best = np.full((order, n), NO_EVIDENCE, np.float32)
    for d in values:
        for k in range(order):  # invoegen in de gesorteerde rij van de kleinste waarden
            lower = d < best[k]
            d, best[k] = np.where(lower, best[k], d), np.where(lower, d, best[k])
    return best[order - 1]


def object_distance(m: ViewMasks) -> np.ndarray:
    """Als signed_distance, maar tot de rand van het objectmasker (positief op het object); kleine gaatjes in het
    masker (donkere vlekjes op een onderdeel met textuur) eerst dicht."""
    fg = m.fg.astype(np.uint8)
    _, lab, stats, _ = cv2.connectedComponentsWithStats(1 - fg, connectivity=4)
    small = stats[:, cv2.CC_STAT_AREA] < 5 * MIN_MAT_PX
    small[0] = False  # label 0: het object zelf
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    small[edge] = False  # wat de rand van de foto raakt, is geen gaatje
    fg = np.where(small[lab] & (fg == 0), 1, fg).astype(np.uint8)
    if not fg.any():
        return np.full(fg.shape, -NO_EVIDENCE, np.float32)
    on = cv2.distanceTransform(fg, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    off = cv2.distanceTransform(1 - fg, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(fg > 0, on - 0.5, 0.5 - off).astype(np.float32)


def _field_at(pts: np.ndarray, views: list[tuple[Pose, np.ndarray]], K: np.ndarray, order: int,
              depth_maps: list | None = None, floor: list | None = None) -> np.ndarray:
    """Het veld (mm, positief binnen) in losse punten: de `order`-de kleinste afstand tot de rand van de zekere mat
    over de foto's; met dieptekaarten (mvs.py) ook de afstand tot het gemeten oppervlak (TSDF). `floor`: foto's recht
    van boven (pose, object_distance); vlak boven de mat (FLOOR_MM) moet een punt in minstens één daarvan op het object
    vallen. Daar ziet een schuine foto een punt vlak naast de voet op (bijna) dezelfde plek als de voet zelf, in de
    twijfelstrook langs de rand van het masker, en zou er een randje blijven staan."""
    f = 0.5 * (K[0, 0] + K[1, 1])

    def silhouettes():
        for pose, D in views:
            uv, z = project(pts, pose, K)
            d = _sample_image(D, uv)
            yield np.where(d >= NO_EVIDENCE * 0.5, NO_EVIDENCE, d * (np.maximum(z, 1.0) / f)).astype(np.float32)

    out = _kth_smallest(silhouettes(), len(pts), order)
    if depth_maps:
        from .mvs import fused
        out = np.minimum(out, fused(pts, depth_maps, K, NO_EVIDENCE))
    low = np.flatnonzero(pts[:, 2] < FLOOR_MM) if floor else np.zeros(0, np.int64)
    if len(low):
        best = np.full(len(low), -NO_EVIDENCE, np.float32)
        seen = np.zeros(len(low), bool)
        for pose, D in floor:
            uv, z = project(pts[low], pose, K)
            d = _sample_image(D, uv)
            ok = np.abs(d) < NO_EVIDENCE * 0.5
            best = np.where(ok, np.maximum(best, d * (np.maximum(z, 1.0) / f)), best)
            seen |= ok
        out[low] = np.where(seen, np.minimum(out[low], best), out[low])
    return np.minimum(out, pts[:, 2].astype(np.float32))  # de mat sluit de onderkant af


def hull_field(views: list[tuple[Pose, ViewMasks]], K: np.ndarray, lo, hi, h: float = 0.5,
               coarse_factor: int = 4, order: int = MIN_VIEWS_OUT, chunk: int = 1_000_000,
               depth_maps: list | None = None) -> Field:
    """Het afstandsveld van de visual hull op een rooster met afstand `h` (mm) tussen `lo` en `hi` (mat-mm), met
    dieptekaarten (mvs.py) ook de lucht vóór het gemeten oppervlak eruit. Eerst op een grof rooster; het fijne rooster
    alleen in de grove cellen vlak bij het oppervlak, elders het teken van het grove veld."""
    sd = [(pose, signed_distance(m)) for pose, m in views]
    floor = [(pose, object_distance(m)) for pose, m in views if tilt_deg(pose) <= TOP_TILT_DEG]
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    lo = lo.copy()
    lo[2] = -2.0 * h  # een laag onder de mat: daar is het veld negatief, de onderkant sluit
    H = h * coarse_factor
    shape_c = tuple(int(np.ceil((b - a) / H)) + 1 for a, b in zip(lo, hi))
    idx = np.indices(shape_c).reshape(3, -1).T
    coarse = np.concatenate([_field_at(lo + idx[s:s + chunk] * H, sd, K, order, depth_maps, floor)
                             for s in range(0, len(idx), chunk)]).reshape(shape_c)
    shape = tuple((n - 1) * coarse_factor + 1 for n in shape_c)
    # grof → fijn: trilineair, en waar het oppervlak dichtbij kan liggen opnieuw rekenen
    cf = Field(lo, H, coarse)
    n = int(np.prod(shape))

    def points(ids: np.ndarray) -> np.ndarray:  # roosterpunten per stuk: het hele rooster als indices is te groot
        return lo + np.stack(np.unravel_index(ids, shape), axis=1) * h

    values = np.empty(n, np.float32)
    near = np.zeros(n, bool)
    for s in range(0, n, chunk):
        v = cf.sample(points(np.arange(s, min(s + chunk, n))))
        values[s:s + chunk] = v
        near[s:s + chunk] = np.abs(v) < 1.8 * H  # het veld is (bijna) een afstand: verder weg verandert het teken niet
    sel = np.flatnonzero(near)
    for s in range(0, len(sel), chunk):
        ids = sel[s:s + chunk]
        values[ids] = _field_at(points(ids), sd, K, order, depth_maps, floor)
    values = values.reshape(shape)
    # de rand van het rooster ligt buiten het onderdeel: zo is het oppervlak altijd gesloten
    values[[0, -1], :, :] = np.minimum(values[[0, -1], :, :], -h)
    values[:, [0, -1], :] = np.minimum(values[:, [0, -1], :], -h)
    values[:, :, [0, -1]] = np.minimum(values[:, :, [0, -1]], -h)
    return Field(lo, h, values)


def surface_nets(fld: Field) -> Mesh:
    """Het nulniveau van het veld als driehoeksmesh (naïeve surface nets), naar buiten gericht."""
    F = fld.values
    inside = F > 0
    nx, ny, nz = F.shape
    cells = (nx - 1, ny - 1, nz - 1)
    acc = np.zeros(cells + (3,), np.float32)
    cnt = np.zeros(cells, np.float32)
    for axis in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis], b[axis] = slice(0, -1), slice(1, None)
        v0, v1 = F[tuple(a)], F[tuple(b)]
        cross = (v0 > 0) != (v1 > 0)
        t = np.where(cross, v0 / np.where(cross, v0 - v1, 1.0), 0.0).astype(np.float32)
        # het snijpunt in roostercoördinaten, ten opzichte van het begin van de ribbe
        pos = np.zeros(t.shape + (3,), np.float32)
        pos[..., axis] = t
        ii = np.indices(t.shape, dtype=np.float32)
        for d in range(3):
            pos[..., d] += ii[d]
        pos *= cross[..., None]
        # een ribbe langs `axis` hoort bij de vier cellen eromheen (de andere twee richtingen: -1 en 0)
        o1, o2 = [d for d in range(3) if d != axis]
        for s1 in (0, 1):
            for s2 in (0, 1):
                src = [slice(None)] * 3
                for d, s in ((o1, s1), (o2, s2)):
                    n = F.shape[d]
                    src[d] = slice(s, n - 1 + s)
                src[axis] = slice(None)
                acc += pos[tuple(src)]
                cnt += cross[tuple(src)]
    active = cnt > 0
    vid = np.full(cells, -1, np.int64)
    vid[active] = np.arange(int(active.sum()))
    V = acc[active] / cnt[active][:, None]
    V = fld.origin + V.astype(np.float64) * fld.h
    quads = []
    for axis in range(3):
        o1, o2 = [d for d in range(3) if d != axis]
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis], b[axis] = slice(0, -1), slice(1, None)
        cross = inside[tuple(a)] != inside[tuple(b)]
        out_dir = inside[tuple(a)] & cross  # binnen aan het begin van de ribbe: normaal in +axis
        # alleen ribben met vier cellen eromheen
        inner = [slice(None)] * 3
        inner[o1], inner[o2] = slice(1, -1), slice(1, -1)
        idx = np.argwhere(cross[tuple(inner)])
        if not len(idx):
            continue
        idx[:, o1] += 1
        idx[:, o2] += 1
        pos_dir = out_dir[tuple(idx.T)]

        def cell(d1: int, d2: int, idx=idx, o1=o1, o2=o2) -> np.ndarray:
            c = idx.copy()
            c[:, o1] += d1
            c[:, o2] += d2
            return vid[tuple(c.T)]

        # volgorde (o1, o2): (-1,-1), (0,-1), (0,0), (-1,0) geeft een normaal langs +axis als (o1, o2, axis)
        # rechtshandig is; anders andersom
        q = np.stack([cell(-1, -1), cell(0, -1), cell(0, 0), cell(-1, 0)], axis=1)
        right_handed = (o1, o2) in ((1, 2), (2, 0), (0, 1)) and axis == 3 - o1 - o2
        flip = pos_dir != right_handed
        q[flip] = q[flip][:, ::-1]
        quads.append(q)
    Q = np.concatenate(quads) if quads else np.zeros((0, 4), np.int64)
    # elke vierhoek in twee driehoeken, over de kortste diagonaal
    d02 = np.linalg.norm(V[Q[:, 0]] - V[Q[:, 2]], axis=1)
    d13 = np.linalg.norm(V[Q[:, 1]] - V[Q[:, 3]], axis=1)
    use02 = d02 <= d13
    T = np.concatenate([np.stack([Q[use02, 0], Q[use02, 1], Q[use02, 2]], 1),
                        np.stack([Q[use02, 0], Q[use02, 2], Q[use02, 3]], 1),
                        np.stack([Q[~use02, 1], Q[~use02, 2], Q[~use02, 3]], 1),
                        np.stack([Q[~use02, 1], Q[~use02, 3], Q[~use02, 0]], 1)])
    return Mesh(V, T)


def largest_piece(mesh: Mesh) -> Mesh:
    """Alleen het grootste samenhangende stuk: losse brokjes (een restje van het 'dak' waar geen stereo was, een
    holte binnenin) vallen weg."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if not len(mesh.F):
        return mesh
    E = np.concatenate([mesh.F[:, [0, 1]], mesh.F[:, [1, 2]], mesh.F[:, [2, 0]]])
    A = coo_matrix((np.ones(len(E), np.int8), (E[:, 0], E[:, 1])), shape=(len(mesh.V), len(mesh.V)))
    n, lab = connected_components(A, directed=False)
    if n == 1:
        return mesh
    big = np.argmax(np.bincount(lab[mesh.F[:, 0]], minlength=n))
    F = mesh.F[lab[mesh.F[:, 0]] == big]
    used = np.unique(F)
    remap = np.full(len(mesh.V), -1, np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(mesh.V[used], remap[F])


def smooth(mesh: Mesh, iters: int = 4, lam: float = 0.5, mu: float = -0.53, rough: np.ndarray | None = None,
           rough_iters: int = 12) -> Mesh:
    """Licht gladstrijken (Taubin: om en om een stap naar het gemiddelde van de buren en een kleinere terug, zodat de
    vorm niet krimpt): de ruis van de stereo eruit. Hoekpunten op de mat blijven op de mat. `rough`: hoekpunten die
    daarna nog `rough_iters` keer extra gladgestreken worden (wat alleen uit de silhouetten komt: daar zijn de randen
    van de maskers van foto tot foto net anders, en dat geeft een ruw oppervlak)."""
    from scipy.sparse import coo_matrix
    E = np.concatenate([mesh.F[:, [0, 1]], mesh.F[:, [1, 2]], mesh.F[:, [2, 0]]])
    n = len(mesh.V)
    A = coo_matrix((np.ones(2 * len(E)), (np.concatenate([E[:, 0], E[:, 1]]), np.concatenate([E[:, 1], E[:, 0]]))),
                   shape=(n, n)).tocsr()
    A.data[:] = 1.0  # dubbele ribben één keer
    deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1.0)[:, None]
    floor = mesh.V[:, 2] < 0.05
    V = mesh.V.copy()
    for _ in range(iters):
        for f in (lam, mu):
            V = V + f * (A @ V / deg - V)
            V[floor, 2] = 0.0
    if rough is not None and rough.any():
        move = rough[:, None].astype(float)
        for _ in range(rough_iters):
            for f in (lam, mu):
                V = V + f * move * (A @ V / deg - V)
                V[floor, 2] = 0.0
    return Mesh(V, mesh.F)


def snap_to_surface(mesh: Mesh, fld: Field, steps: int = 3) -> Mesh:
    """Newtonstappen op het nulniveau van het veld (hooguit een halve roostercel per stap)."""
    V = mesh.V.copy()
    for _ in range(steps):
        f = fld.sample(V)
        g = fld.gradient(V)
        gg = np.maximum((g * g).sum(axis=1), 1e-6)
        step = -(f / gg)[:, None] * g
        n = np.linalg.norm(step, axis=1, keepdims=True)
        V += step * np.minimum(1.0, 0.5 * fld.h / np.maximum(n, 1e-12))
    return Mesh(V, mesh.F)


def silhouette(mesh: Mesh, pose: Pose, K: np.ndarray, size: tuple[int, int], ss: int = 4) -> np.ndarray:
    """Het silhouet van de mesh in een foto (bool, h × w): een pixel hoort erbij als meer dan de helft bedekt is.
    Alleen de driehoeken die naar de camera kijken (bij een gesloten mesh is dat genoeg), elk apart: cv2.fillPoly
    met alle driehoeken tegelijk laat overlappende driehoeken elkaar opheffen. cv2.fillConvexPoly kleurt ook de
    pixels op de rand van een driehoek; daarom `ss` × fijner tekenen (alleen rond de mesh) en terugschalen."""
    w, h = size
    uv, z = project(mesh.V, pose, K)
    tri = uv[mesh.F]
    ok = (z[mesh.F] > 1.0).all(axis=1)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    # naar de camera: de normaal wijst naar de camera (negatieve z), en de projectie houdt de oriëntatie van x, y
    ok &= (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) < 0
    img = np.zeros((h, w), bool)
    if not ok.any():
        return img
    used = tri[ok].reshape(-1, 2)
    x0, y0 = np.maximum(np.floor(used.min(axis=0)).astype(int) - 1, 0)
    x1, y1 = np.minimum(np.ceil(used.max(axis=0)).astype(int) + 2, [w, h])
    if x1 <= x0 or y1 <= y0:
        return img
    # pixel (x, y) bedekt [x - 0,5, x + 0,5]: in het fijne beeld de pixels ss·x .. ss·x + ss - 1
    fine = np.round(((tri[ok] - [x0 - 0.5, y0 - 0.5]) * ss - 0.5) * 16).astype(np.int32)
    canvas = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
    for t in fine:
        cv2.fillConvexPoly(canvas, t, 1, cv2.LINE_8, 4)
    cover = cv2.resize(canvas.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    img[y0:y1, x0:x1] = cover > 0.5
    return img


def view_iou(mesh: Mesh, views: list[tuple[Pose, ViewMasks]], K: np.ndarray) -> list[float]:
    """Per foto de IoU van het silhouet van de mesh en het objectmasker, alleen waar de klasse zeker is."""
    out = []
    for pose, m in views:
        P = silhouette(mesh, pose, K, (m.fg.shape[1], m.fg.shape[0]))
        mat = m.edge_bg if m.edge_bg is not None else m.bg
        known = m.fg | mat
        union = np.count_nonzero((P & known) | m.fg)
        out.append(np.count_nonzero(P & m.fg) / union if union else float("nan"))
    return out


def reconstruct(views: list[tuple[Pose, ViewMasks]], K: np.ndarray, lo, hi, h: float = 0.5, images=None,
                info: dict | None = None, workers: int = 4) -> Mesh:
    """De buitenvorm van het onderdeel tussen `lo` en `hi` (mat-mm) als gesloten mesh. Met `images` (naam → het
    ontvervormde grijsbeeld) ook de stereo (mvs.py): de lucht vóór het gemeten oppervlak gaat eruit. `info` krijgt
    wat er gebeurd is (voor diagnose.json)."""
    info = {} if info is None else info
    fld = hull_field(views, K, lo, hi, h)
    mesh = snap_to_surface(largest_piece(surface_nets(fld)), fld)
    info["hull_volume_mm3"] = round(mesh.volume(), 1)
    if images is None:
        return smooth(mesh, rough=np.ones(len(mesh.V), bool))
    from concurrent.futures import ThreadPoolExecutor

    from . import mvs
    poses = [p for p, _ in views]
    nbr = mvs.neighbours(poses, mesh.V.mean(axis=0))
    r = mvs.WINDOW // 2 + 1

    def hull_at(i: int):
        pose, m = views[i]
        return mvs.hull_depths(mesh.V, pose, K, m.fg.shape)

    def one(i: int):
        pose, m = views[i]
        # waar het venster helemaal binnen het silhouet ligt (geen zekere mat), vlak bij het object: het objectmasker
        # zelf heeft bij een onderdeel met textuur gaatjes (donkere vlekjes zijn 'onbekend'), en die zouden zo vensters
        # groot worden; 'geen zekere mat' alleen is ook de tafel naast de mat
        near = cv2.dilate(m.fg.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
        mask = (signed_distance(m) >= r) & near
        nb = [(images(views[j][0].name), views[j][0]) for j in nbr[i]]
        depth, score = mvs.depth_map(images(pose.name), pose, nb, K, mask, hulls[i][0], hulls[i][1])
        return mvs.DepthView(pose, depth, score)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        hulls = list(pool.map(hull_at, range(len(views))))
        maps = list(pool.map(one, range(len(views))))
    raw = sum(int(np.isfinite(m.depth).sum()) for m in maps)
    maps = mvs.consistent(maps, K, nbr)
    maps = [mvs.fill_small_holes(m, v.fg) for m, (_, v) in zip(maps, views)]
    kept = sum(int(np.isfinite(m.depth).sum()) for m in maps)
    info.update({"stereo_pixels": raw, "stereo_na_controle": kept})
    if kept == 0:
        return smooth(mesh, rough=np.ones(len(mesh.V), bool))
    fld = hull_field(views, K, lo, hi, h, depth_maps=maps)
    mesh = snap_to_surface(largest_piece(surface_nets(fld)), fld)
    # welke hoekpunten de stereo heeft gezien (genoeg foto's met een diepte vlakbij), voor het rapport; de rest komt
    # alleen uit de silhouetten en wordt extra gladgestreken
    seen = mvs.seen(mesh.V, maps, K)
    mesh = smooth(mesh, rough=~seen)
    info["volume_mm3"] = round(mesh.volume(), 1)
    info["stereo_per_vertex"] = seen
    return mesh


# ----------------------------------------------------------------------------- uitvoer

def write_stl(mesh: Mesh, path: str | Path, name: str = "camtocad") -> None:
    """Binaire STL (mm)."""
    n = mesh.normals().astype(np.float32)
    tri = mesh.V[mesh.F].astype(np.float32)
    rec = np.zeros(len(mesh.F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as fh:
        fh.write(name.encode("ascii", "replace")[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(mesh.F)))
        fh.write(rec.tobytes())


def write_3mf(mesh: Mesh, path: str | Path, name: str = "camtocad") -> None:
    """3MF (mm): het formaat van de meeste slicers, met de eenheid erin."""
    buf = io.StringIO()
    buf.write('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="nl-NL" '
              'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
              f' <metadata name="Title">{name}</metadata>\n <resources>\n  <object id="1" type="model">\n'
              '   <mesh>\n    <vertices>\n')
    buf.write("".join(f'     <vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>\n' for x, y, z in mesh.V))
    buf.write("    </vertices>\n    <triangles>\n")
    buf.write("".join(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>\n' for a, b, c in mesh.F))
    buf.write('    </triangles>\n   </mesh>\n  </object>\n </resources>\n <build>\n  <item objectid="1"/>\n'
              " </build>\n</model>\n")
    types = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
             'content-types">\n <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
             'relationships+xml"/>\n <Default Extension="model" ContentType="application/vnd.ms-package.'
             '3dmanufacturing-3dmodel+xml"/>\n</Types>\n')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/'
            '2006/relationships">\n <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.'
            'microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n</Relationships>\n')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", buf.getvalue())


# ----------------------------------------------------------------------------- assenstelsel en aanzichten

def part_frame(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Een assenstelsel voor het onderdeel: om Z gedraaid zodat de omhullende rechthoek van het bovenaanzicht zo klein
    mogelijk is, met de lange zijde langs X, en de oorsprong in de hoek linksonder op de mat. Geeft (R, t), met
    x_onderdeel = R · x_mat + t."""
    xy = mesh.V[:, :2].astype(np.float32)
    pts = xy[cv2.convexHull(xy, returnPoints=False).ravel()].astype(np.float64)

    def extent(a: float) -> np.ndarray:
        c, s = np.cos(a), np.sin(a)
        r = pts @ np.array([[c, -s], [s, c]]).T
        return r.max(axis=0) - r.min(axis=0)

    coarse = np.radians(np.arange(0.0, 90.0, 0.5))
    a0 = min(coarse, key=lambda a: float(np.prod(extent(a))))
    a = min(a0 + np.radians(np.arange(-0.5, 0.5, 0.02)), key=lambda a: float(np.prod(extent(a))))
    if extent(a)[1] > extent(a)[0]:
        a += np.pi / 2
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    lo, _ = robust_extent(Mesh(mesh.V @ R.T, mesh.F))
    return R, np.array([-lo[0], -lo[1], 0.0])


COLOR_STEREO = (0.86, 0.74, 0.60)  # BGR: gemeten met stereo (blauwgrijs)
COLOR_HULL = (0.45, 0.72, 0.96)  # BGR: alleen uit de silhouetten (oranje)
def robust_extent(mesh: Mesh, q: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Het `q`%- en (100 - `q`)%-punt van het oppervlak langs elke as (hoekpunten gewogen naar hun oppervlak): de
    buitenmaten zonder losse uitschieters (een piekje ruis, een randje op de mat)."""
    a = np.linalg.norm(np.cross(mesh.V[mesh.F[:, 1]] - mesh.V[mesh.F[:, 0]], mesh.V[mesh.F[:, 2]] - mesh.V[mesh.F[:, 0]]),
                       axis=1) / 6.0
    w = np.zeros(len(mesh.V))
    for k in range(3):
        np.add.at(w, mesh.F[:, k], a)
    lo, hi = np.empty(3), np.empty(3)
    for d in range(3):
        order = np.argsort(mesh.V[:, d])
        cw = np.cumsum(w[order]) / max(w.sum(), 1e-12)
        lo[d] = mesh.V[order[np.searchsorted(cw, q / 100.0)], d]
        hi[d] = mesh.V[order[min(np.searchsorted(cw, 1.0 - q / 100.0), len(order) - 1)], d]
    return lo, hi


VIEWS = {  # richting van het onderdeel naar de kijker, in het assenstelsel van het onderdeel
    "schuin": (1.0, -1.4, 1.1), "voor": (0.0, -1.0, 0.0), "zij": (1.0, 0.0, 0.0), "boven": (0.0, 0.0, 1.0),
}


def render_preview(mesh: Mesh, toward=(1.0, -1.4, 1.1), size: tuple[int, int] = (480, 360),
                   colors: np.ndarray | None = None, base=(0.86, 0.74, 0.60)) -> np.ndarray:
    """Een aanzicht van de mesh (BGR, uint8): parallelle projectie, belicht vanuit de kijker en iets van boven.
    `colors`: per hoekpunt (BGR, 0-1); anders `base`."""
    from .render import look_at
    w, h = size
    d = np.asarray(toward, float)
    d /= np.linalg.norm(d)
    c = 0.5 * (mesh.V.min(axis=0) + mesh.V.max(axis=0))
    R, _ = look_at(c + d, c, (0.0, 1.0, 0.0) if abs(d[2]) > 0.98 else (0.0, 0.0, 1.0))
    P = (mesh.V - c) @ R.T
    span = np.maximum(P[:, :2].max(axis=0) - P[:, :2].min(axis=0), 1e-6)
    ss = 2
    scale = 0.9 * min(w / span[0], h / span[1]) * ss
    uv = P[:, :2] * scale + np.array([w * ss / 2, h * ss / 2])
    n = mesh.normals()
    front = n @ d > 0
    light = d + np.array([0.0, 0.0, 0.6])
    light /= np.linalg.norm(light)
    shade = 0.3 + 0.7 * np.clip(n @ light, 0.0, 1.0)
    col = (colors[mesh.F].mean(axis=1) if colors is not None else np.tile(np.asarray(base, float), (len(mesh.F), 1)))
    col = np.clip(col * shade[:, None] * 255.0, 0, 255)
    order = np.flatnonzero(front)
    order = order[np.argsort(-P[mesh.F[order]][:, :, 2].mean(axis=1))]  # ver weg eerst
    img = np.full((h * ss, w * ss, 3), 255, np.uint8)
    tri = np.round(uv[mesh.F[order]] * 16).astype(np.int32)
    for t, cc in zip(tri, col[order]):
        cv2.fillConvexPoly(img, t, (float(cc[0]), float(cc[1]), float(cc[2])), cv2.LINE_8, 4)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
