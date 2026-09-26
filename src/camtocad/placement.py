"""Ligt het onderdeel in alle foto's op dezelfde plek? Consensus van de objectmaskers in 3D.

De visual hull en de silhouetfit gaan ervan uit dat het onderdeel stilligt. Wordt het tussendoor
verschoven, gedraaid of op een andere kant gelegd, dan spreken de silhouetten elkaar tegen: de
hull wordt een grillig restje en het startmodel of de fit mislukt zonder aanwijsbare reden.
Deze controle vindt dat vooraf, en zegt welke foto's bij welke ligging horen.

Per voxel (3 mm) tellen we in hoeveel foto's hij op het object valt (`fg`) en in hoeveel op de
mat; de consensus is fg / (fg + mat). Langs de kijkstraal van een objectpixel ligt het object
zelf, dus daar is de consensus ~1 als het stilligt: de andere foto's 'steunen' de pixel. Omgekeerd
mag een foto de consensus-hull niet op de mat zien. Is het onderdeel verplaatst, dan steunen
alleen de foto's met dezelfde ligging elkaar en zakt de consensus naar hun aandeel in de serie.
Groeperen gaat iteratief: consensus over de leden, houd de foto's die er het best bij passen
(Otsu-drempel, of de slechtste 10% afpellen) tot de groep onderling klopt, en herhaal dat voor
de foto's die overblijven.

Een klein duwtje (een paar mm bij een onderdeel van 80 mm) valt hier niet op: de liggingen
overlappen dan grotendeels. Dat zie je later aan matig passende silhouetten.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from .calib import Pose, project
from .masks import ViewMasks

Views = list[tuple[Pose, ViewMasks]]

VOXEL_MM = 3.0
SCORE_OK = 0.75  # vanaf deze score hoort een foto bij een groep (_agreement: steun, en weinig wegsnijden)
MIN_PIXELS = 20  # minder objectpixels (op 1/4 schaal): niet te beoordelen


@dataclass
class Placements:
    groups: list[list[str]]  # fotonamen per ligging, grootste groep eerst
    outliers: list[str]  # wel beoordeeld, maar passen bij geen enkele groep
    unjudged: list[str]  # te weinig object in beeld om iets te zeggen
    scores: dict[str, float] = field(default_factory=dict)  # score t.o.v. de grootste groep

    @property
    def judged(self) -> int:
        return sum(len(g) for g in self.groups) + len(self.outliers)

    def to_dict(self) -> dict:
        return {"groepen": self.groups, "passen_nergens_bij": self.outliers, "niet_beoordeeld": self.unjudged,
                "scores": {k: round(v, 3) for k, v in self.scores.items()}}


def _opinions(views: Views, K: np.ndarray, origin: np.ndarray, shape: tuple[int, int, int], voxel: float):
    """Per foto en voxel: valt hij op het object (`fg`), of duidelijk op de mat?

    'Mat' is hier ruimer dan de zekere mat van de hull: elke pixel die met de voorspelde mat klopt,
    ook op egale stukken (bij mat v1 is zekere mat schaars: alleen langs de vakranden). Niet als het
    object daar onzichtbaar zou zijn (`amb`), en niet binnen een halve voxel van het objectmasker.
    """
    pts = origin + np.indices(shape).reshape(3, -1).T * voxel
    center = origin + (np.array(shape) - 1) * voxel / 2
    fg = np.zeros((len(views), len(pts)), bool)
    mat = np.zeros((len(views), len(pts)), bool)
    for k, (pose, m) in enumerate(views):
        uv, z = project(pts, pose, K)
        h, w = m.fg.shape
        ui = np.floor(uv[:, 0] + 0.5).astype(np.int64)
        vi = np.floor(uv[:, 1] + 0.5).astype(np.int64)
        sel = np.flatnonzero((z > 1.0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h))
        r = round(0.5 * voxel * K[0, 0] / max(float((pose.R @ center + pose.t)[2]), 1.0))
        near = cv2.dilate(m.fg.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
        free = m.valid & ~near
        if m.amb is not None:
            free &= ~m.amb
        fg[k, sel] = m.fg[vi[sel], ui[sel]]
        mat[k, sel] = free[vi[sel], ui[sel]]
    return fg, mat


def _rays(pose: Pose, m: ViewMasks, K: np.ndarray, origin: np.ndarray, shape: tuple[int, int, int], voxel: float,
          z_max: float, scale: float = 0.25, max_px: int = 1500):
    """Voxelindices (-1 buiten het raster) langs de kijkstralen van objectpixels, met per straal het
    nummer van het maskerdeel: alleen delen van minstens 30% van het grootste tellen mee (een losse
    vlek van schittering of schaduw is geen object)."""
    small = cv2.resize(m.fg.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) > 0.5
    inner = cv2.erode(small.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # randpixels zijn half object
    if inner.sum() >= 0.3 * small.sum():
        small = inner
    n, labels, stats, _ = cv2.connectedComponentsWithStats(small.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = 1 + np.flatnonzero(areas >= max(0.3 * areas.max(), MIN_PIXELS / 2))
    keep = np.isin(labels, big)
    vs, us = np.nonzero(keep)
    if len(us) < MIN_PIXELS:
        return None
    if len(us) > max_px:
        pick = np.random.default_rng(0).choice(len(us), max_px, replace=False)
        vs, us = vs[pick], us[pick]
    part = labels[vs, us]
    u = (us + 0.5) / scale - 0.5
    v = (vs + 0.5) / scale - 0.5
    d = (np.linalg.inv(K) @ np.stack([u, v, np.ones_like(u)])).T @ pose.R  # kijkrichting in matcoördinaten
    C = pose.center
    zs = np.arange(0.25 * voxel, z_max, 0.5 * voxel)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (zs[None, :] - C[2]) / d[:, 2:3]
    X = C[None, None, :] + lam[..., None] * d[:, None, :]
    ijk = np.floor((X - (origin - voxel / 2)) / voxel).astype(np.int64)
    ok = np.all((ijk >= 0) & (ijk < np.array(shape)), axis=-1) & (lam > 0)
    flat = np.ravel_multi_index(tuple(np.clip(ijk, 0, np.array(shape) - 1).transpose(2, 0, 1)), shape)
    return np.where(ok, flat, -1).astype(np.int32), part


def _consensus(fg: np.ndarray, mat: np.ndarray, members: list[int]) -> np.ndarray:
    """fg / (fg + mat) over de leden; -1 waar te weinig leden er iets over zeggen."""
    nf = fg[members].sum(axis=0, dtype=np.int32)
    cnt = nf + mat[members].sum(axis=0, dtype=np.int32)
    c = np.full(fg.shape[1] + 1, -1.0, np.float32)  # laatste element: buiten het raster
    ok = cnt >= max(3, len(members) // 2)
    c[:-1][ok] = nf[ok] / cnt[ok]
    return c


def _hull(c: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """De kern van de consensus-hull: c ≥ 0,9, in x en y één voxel ingekrompen. Een randvoxel valt in een
    scherende foto voor de helft op de mat; zonder inkrimpen 'snijdt' elke lage foto zo'n 10% weg. Bij
    een dun onderdeel (een plaat op zijn kant) blijft er na inkrimpen te weinig over: dan de hele hull."""
    h = (c[:-1] >= 0.9).reshape(shape)
    core = ndimage.binary_erosion(h, structure=np.ones((3, 3, 1), bool))
    return (core if core.sum() >= 0.5 * h.sum() else h).reshape(-1)


def _agreement(c: np.ndarray, hull: np.ndarray, fg: np.ndarray, mat: np.ndarray, rays, i: int) -> float:
    """Klopt foto `i` met de consensus `c`? Beide kanten op:

    * steun: 25e percentiel over de objectpixels van de hoogste consensus langs de kijkstraal (per
      maskerdeel; het best gesteunde deel telt);
    * wegsnijden: welk deel van de consensus-hull (`_hull`) de foto op de mat ziet. Zonder die toets
      past bijvoorbeeld een schuine foto van een plat liggend onderdeel ook bij dezelfde balk op zijn
      kant: dat silhouet valt binnen het hogere, maar snijdt de bovenkant weg.
    """
    idx, part = rays
    best = c[idx].max(axis=1)
    support = max(float(np.percentile(best[part == k], 25)) for k in np.unique(part))
    nf = np.count_nonzero(fg[i] & hull)
    nm = np.count_nonzero(mat[i] & hull)
    return min(support, 1.0 - 2.0 * nm / max(nf + nm, 1))


def _otsu(values: np.ndarray) -> float:
    v = np.sort(values)
    best, thr = -1.0, float(v[0])
    for i in range(1, len(v)):
        a, b = v[:i], v[i:]
        between = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if between > best:
            best, thr = between, float(v[i - 1] + v[i]) / 2
    return thr


def find(views: Views, K: np.ndarray, bounds_xy: tuple[float, float, float, float], z_max: float = 120.0,
         voxel: float = VOXEL_MM, score_ok: float = SCORE_OK) -> Placements:
    """Deelt de foto's in naar ligging van het onderdeel (zie de moduletekst)."""
    x0, x1, y0, y1 = bounds_xy
    shape = (int(np.ceil((x1 - x0) / voxel)), int(np.ceil((y1 - y0) / voxel)), int(np.ceil(z_max / voxel)))
    origin = np.array([x0, y0, 0.0]) + voxel / 2
    names = [p.name for p, _ in views]
    rays = {i: r for i, (p, m) in enumerate(views) if (r := _rays(p, m, K, origin, shape, voxel, z_max)) is not None}
    unjudged = [names[i] for i in range(len(views)) if i not in rays]
    if len(rays) < 3:
        return Placements([], [], unjudged)
    fg, mat = _opinions(views, K, origin, shape, voxel)
    remaining, groups = list(rays), []
    while len(remaining) >= 3:
        members = list(remaining)
        while len(members) >= 3:
            c = _consensus(fg, mat, members)
            hull = _hull(c, shape)
            s = np.array([_agreement(c, hull, fg, mat, rays[i], i) for i in members])
            if np.mean(s >= score_ok) >= 0.8:
                break
            # duidelijk twee niveaus: de onderste groep in één keer weg; anders (bijv. twee even grote
            # liggingen, iedereen ~0,5) de laagste 10% afpellen: toeval breekt de gelijkstand en
            # daarna wint de grootste ligging vanzelf
            thr = _otsu(s)
            hi, lo = s[s >= thr], s[s < thr]
            if len(hi) >= 3 and len(lo) and hi.mean() - lo.mean() > 0.25:
                members = [i for i, si in zip(members, s) if si >= thr]
            else:
                order = np.argsort(s, kind="stable")
                drop = set(order[:max(1, len(members) // 10)].tolist())
                members = [i for k, i in enumerate(members) if k not in drop]
        c = _consensus(fg, mat, members)
        hull = _hull(c, shape)
        scores = {i: _agreement(c, hull, fg, mat, rays[i], i) for i in remaining}
        group = [i for i in remaining if scores[i] >= score_ok]
        if len(group) < 3:
            break
        groups.append((group, scores))
        remaining = [i for i in remaining if i not in group]
    groups.sort(key=lambda g: len(g[0]), reverse=True)
    first = {names[i]: s for i, s in groups[0][1].items()} if groups else {}
    return Placements([[names[i] for i in g] for g, _ in groups], [names[i] for i in remaining], unjudged, first)


def _names(group: list[str], order: list[str]) -> str:
    """'foto a t/m b' als de groep een aaneengesloten stuk van de serie is, anders een paar namen."""
    g = sorted(group, key=order.index)
    idx = [order.index(n) for n in g]
    if idx[-1] - idx[0] + 1 == len(g) and len(g) > 2:
        return f"{len(g)} foto's: {g[0]} t/m {g[-1]}"
    shown = ", ".join(g[:4]) + (", ..." if len(g) > 4 else "")
    return f"{len(g)} foto's: {shown}"


def moved_message(res: Placements, order: list[str]) -> str:
    """Uitleg voor de gebruiker als de foto's elkaar tegenspreken (`order`: alle fotonamen)."""
    n_odd = len(res.outliers)
    odd = "1 foto past nergens bij" if n_odd == 1 else f"{n_odd} foto's passen nergens bij"
    advice = ("Een scan heeft één vaste ligging nodig. Leg het onderdeel neer en maak alle foto's zonder het aan te "
              "raken (wil je de mat draaien, draai dan mat en onderdeel samen). Wil je ook een andere kant scannen, "
              "maak daar dan een aparte scan van.")
    if len(res.groups) > 1:
        top = sorted(res.groups[:3], key=lambda g: min(order.index(n) for n in g))  # in de volgorde van de serie
        shown = "; ".join(_names(g, order) for g in top)
        if len(res.groups) > 3:
            rest = sum(len(g) for g in res.groups[3:])
            shown += f"; en nog {len(res.groups) - 3} kleinere groepen met samen {rest} foto's"
        return ("Het onderdeel ligt niet in alle foto's op dezelfde plek. De foto's vallen uiteen in "
                f"{len(res.groups)} groepen die elk onderling kloppen (" + shown + ")" + (f", en {odd}" if n_odd else "")
                + ". Waarschijnlijk is het tussendoor verschoven, gedraaid of op een andere kant gelegd. " + advice)
    return (f"De foto's spreken elkaar tegen: maar {len(res.groups[0])} van de {res.judged} foto's passen bij elkaar "
            f"({_names(res.groups[0], order)}). Waarschijnlijk is het onderdeel tussendoor verplaatst; het kan ook "
            "aan de objectmaskers liggen (harde schaduwen, glans, een onderdeel dat nauwelijks afsteekt tegen de "
            "mat: kijk in debug/masker_*.jpg). " + advice)
