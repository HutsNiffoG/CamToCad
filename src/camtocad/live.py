"""Live begeleiding bij het fotograferen (V25: v0.13, uitgebreid in v0.16).

De telefoonpagina stuurt ~8 beelden per seconde van de livecamera naar de pc, verkleind en zonder ze te bewaren. Per
beeld bepaalt de pc:

* de mat, en daarmee waar de camera staat ten opzichte van het onderdeel (azimut, elevatie, afstand);
* waar het onderdeel op de mat ligt, uit de livebeelden zelf (v0.16). Waar het patroon van de mat ontbreekt, ligt iets.
  Over beelden uit verschillende richtingen blijft alleen het grondvlak van het onderdeel over: een visual hull in het
  vlak van de mat. Daarmee richt de begeleiding al vóór de eerste foto op het echte onderdeel, en meldt ze een
  onderdeel aan de rand van de mat of een tweede voorwerp;
* welke richtingen nog foto's missen: eerst recht van boven, dan rondom op ~60° en daarna op ~35°. Eén aanwijzing in
  tekst, een korte zin die in beeld staat en uitgesproken kan worden, en een pictogram (v0.16.2);
* of dit een goed moment is voor een foto: het beeld is scherp (onscherpte aan de mat), de mat beweegt niet door het
  beeld (v0.16), en de telefoon wisselde niet van lens of zoom (v0.16);
* de geometrie voor de overlay: de omtrek van de mat en van het onderdeel, een ring op de mat rond het onderdeel met
  per richting en hoogte de dekking, een pijl langs de ring naar het volgende doel, en voor de foto's recht van boven
  de richting waarin de telefoon moet schuiven (v0.16), en een cirkel rond een ander voorwerp op de mat (v0.16.2).

De positie is grof (een paar graden): genoeg om de weg te wijzen. De foto zelf wordt na het uploaden gewoon
gecontroleerd (preflight.py). Wat er tussen de beelden onthouden wordt (brandpuntsafstand, mat, voetafdruk, het vorige
beeld), staat per scan in een `LiveSession`.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import calib
from .mat import MatSpec, make_board, rasterize_board
from .preflight import (BANDS, CLOSE_MM, LOW_PX_MM, SECTORS, STEEP_DEG, WORK_SIDE, _detect_any, _edge_blur,
                        _focal_and_tilt, _photometry, mat_scale, to_work, view_distance)

LIVE_SIDE = 960  # beelden van de livecamera: kleiner dan de werkresolutie, ~0,05 s per beeld
LIVE_BLUR_MAX = 1.5  # σ (px op LIVE_SIDE) waarboven het beeld beweegt (~3 px op de werkresolutie, BLUR_BAD)
LIVE_TOP, LIVE_PER_CELL = 5, 2  # doel: zoveel foto's recht van boven, en per hoogteband en richting
LIVE_TOTAL = LIVE_TOP + 2 * 8 * LIVE_PER_CELL  # 37 foto's
LIVE_SEP_DEG, LIVE_SEP_TOP_DEG = 5.0, 2.0  # een nieuwe foto minstens zo ver (gezien vanaf het onderdeel) van de vorige
FAR_MM = 600.0  # verder weg: het onderdeel wordt klein in beeld
LIVE_F_REL = 0.8  # brandpuntsafstand / lange zijde als de homografie hem niet geeft (zoals _focal_and_tilt)
LIVE_HIGH_DEG = 75.0  # hoger dan dit zegt de azimut weinig: de weg wijzen met de kanten van de mat

# v0.16
STILL_PX_S = 40.0  # de mat beweegt langzamer door het beeld (px/s op LIVE_SIDE): stil genoeg voor een foto
FAST_PX_S = 400.0  # sneller: "rustiger bewegen" (bij 1/60 s belichting ~7 px vegen)
LENS_JUMP = 0.12  # brandpuntsafstand meer dan 12% anders dan in de vorige beelden: een andere lens of zoom
LENS_MIN_SAMPLES = 5
FOOT_PX_MM, FOOT_CELL = 2.0, 4  # matraster voor de voetafdruk: 2 px/mm, cellen van 4 px (2 mm)
FOOT_SIGMA_MM, FOOT_WIN_MM = 1.0, 8.0  # vervaging en venster voor de vergelijking met het patroon
FOOT_DECAY = 0.985  # per beeld telt het bewijs iets minder: na ~6 s de helft (het onderdeel kan verschoven zijn)
FOOT_MIN_CELLS = 20  # kleinste voorwerp: 80 mm²
RING_MARGIN_MM, RING_MIN_MM = 15.0, 25.0  # de ring ligt zo ver buiten het onderdeel
EDGE_MM = 15.0  # het onderdeel dichter bij de rand van het bord: melden
# v0.16.3: een brandpuntsafstand (/ lange zijde) hieronder is een groothoeklens (0,6×: ~0,38; de gewone lens ~0,7)
WIDE_F_REL = 0.5


# ----------------------------------------------------------------------------- richtingen en vakken

def _angle(az1: float, el1: float, az2: float, el2: float) -> float:
    """Hoek (graden) tussen twee kijkrichtingen vanaf het onderdeel."""
    a1, e1, a2, e2 = map(math.radians, (az1, el1, az2, el2))
    c = math.sin(e1) * math.sin(e2) + math.cos(e1) * math.cos(e2) * math.cos(a1 - a2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _cell(az: float, el: float) -> str | None:
    """Het vak van een camerapositie: 'boven' (recht van boven), '<band>-<k>' (hoogteband, sector k), of None
    (tussen de banden, of te laag)."""
    if el >= 90.0 - STEEP_DEG:
        return "boven"
    for band, lo, hi, _ in BANDS:
        if lo <= el < hi:
            return f"{band}-{int(((az + 22.5) % 360.0) // 45.0)}"
    return None


def _cell_center(cell: str) -> tuple[float, float]:
    if cell == "boven":
        return 0.0, 90.0
    band, k = cell.split("-")
    return 45.0 * int(k), float(next(b[3] for b in BANDS if b[0] == band))


def missing_views(overview: dict | None) -> dict[str, int]:
    """Hoeveel foto's de live begeleiding per vak nog vraagt: {'boven': n, 'hoog-0': n, ..., 'laag-7': n}.
    `overview`: de samenvatting van de foto's tot nu toe (preflight.summarize)."""
    ov = overview or {}
    need = {"boven": max(0, LIVE_TOP - int(ov.get("recht_van_boven") or 0))}
    for band, *_ in BANDS:
        counts = (ov.get("dekking") or {}).get(band) or [0] * 8
        for k in range(8):
            need[f"{band}-{k}"] = max(0, LIVE_PER_CELL - int(counts[k]))
    return need


def _next_cell(az: float, el: float, need: dict[str, int]) -> str | None:
    """Het vak waar de volgende foto heen moet: eerst recht van boven (daarmee vindt de scan het onderdeel), dan
    rondom één hoogte tegelijk (eerst ~60°, dan ~35°: twee keer een rondje), en daarbinnen het dichtstbijzijnde vak
    dat nog foto's mist; bij gelijke afstand rechtsom, zodat de aanwijzing niet heen en weer springt."""
    open_cells = [c for c, n in need.items() if n > 0]
    if not open_cells:
        return None
    if "boven" in open_cells:
        return "boven"
    for band, *_ in BANDS:
        in_band = [c for c in open_cells if c.startswith(band + "-")]
        if in_band:
            open_cells = in_band
            break

    if el >= LIVE_HIGH_DEG:  # (bijna) recht van boven is elke kant even ver: begin aan de onderkant van de mat
        az, el = 270.0, 60.0

    def cost(cell: str) -> float:
        caz, cel = _cell_center(cell)
        right = (caz - az) % 360.0 < 180.0
        return _angle(az, el, caz, cel) - (5.0 if right else 0.0)

    return min(open_cells, key=cost)


def _band_of(el: float) -> str | None:
    return next((b[0] for b in BANDS if b[1] <= el < b[2]), None)


def _direction(az: float, el: float, cell: str) -> tuple[str, str, str]:
    """Eén aanwijzing om van de huidige positie naar `cell` te gaan: als tekst, als korte zin (om uit te spreken, en
    in beeld) en als pictogram ('boven', een richting zoals 'rechts' of 'links-lager', of 'opschuiven'). Azimut groeit
    als je rechtsom loopt (gezien vanaf de telefoon: naar rechts)."""
    if cell == "boven":
        return ("Volgende foto's: recht boven het onderdeel. Schuif de telefoon in de richting van de pijl, "
                "evenwijdig aan de mat.", "Recht boven het onderdeel.", "boven")
    caz, cel = _cell_center(cell)
    band, k = cell.split("-")
    if el >= LIVE_HIGH_DEG:  # van bovenaf: de kant van de mat noemen, want de azimut zegt hier weinig
        return (f"Volgende foto: houd de telefoon lager, ~{cel:.0f}° boven de mat, aan de kant {SECTORS[int(k)]} van "
                "de mat (boven = kant met de titel).", f"Lager, aan de kant {SECTORS[int(k)]}.", "lager")
    moves, said, icon = [], [], []
    daz = (caz - az + 180.0) % 360.0 - 180.0
    if abs(daz) >= 10.0:
        side = "rechts" if daz > 0 else "links"
        moves.append(f"loop ~{5 * round(abs(daz) / 5):.0f}° naar {side} om het onderdeel")
        said.append(f"naar {side}")
        icon.append(side)
    if _band_of(el) != band:
        up = "hoger" if cel > el else "lager"
        moves.append(f"houd de telefoon {up}, ~{cel:.0f}° boven de mat")
        said.append(up)
        icon.append(up)
    if not moves:
        moves.append("een klein stukje verder dan de vorige foto")
        said.append("een stukje verder")
        icon.append("opschuiven")
    text, speech = " en ".join(moves), " en ".join(said)
    return f"Volgende foto: {text[0].lower()}{text[1:]}.", f"{speech[0].upper()}{speech[1:]}.", "-".join(icon)


def _tell(out: dict, text: str, speech: str, icon: str) -> None:
    """De aanwijzing: de hele zin (`aanwijzing`), de korte zin die in beeld staat en uitgesproken wordt (`spraak`) en
    het pictogram ervoor (`icoon`, v0.16.2)."""
    out["aanwijzing"], out["spraak"], out["icoon"] = text, speech, icon


def _progress(need: dict[str, int]) -> tuple[dict, dict]:
    """De stap (1: recht van boven, 2: rondom op twee hoogtes, 3: klaar) en hoeveel van de 37 foto's er zijn."""
    done = LIVE_TOTAL - sum(need.values())
    if need["boven"] > 0:
        step = {"nummer": 1, "naam": "recht van boven", "gedaan": LIVE_TOP - need["boven"], "totaal": LIVE_TOP}
    elif any(need.values()):
        step = {"nummer": 2, "naam": "rondom, op ~60° en ~35°", "gedaan": done - LIVE_TOP,
                "totaal": LIVE_TOTAL - LIVE_TOP}
    else:
        step = {"nummer": 3, "naam": "klaar", "gedaan": LIVE_TOTAL, "totaal": LIVE_TOTAL}
    return step, {"gedaan": done, "totaal": LIVE_TOTAL}


# ----------------------------------------------------------------------------- het onderdeel in het livebeeld (v0.16)

def _quad_fit(x: np.ndarray, y: np.ndarray, v: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Robuust tweedegraadsvlak v(x, y) (belichtingsverloop), geëvalueerd op (xs, ys). Uitschieters (het onderdeel,
    een schaduw) tellen na de eerste ronde niet meer mee."""
    A = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    keep = np.ones(len(v), bool)
    c = np.zeros(6)
    for _ in range(3):
        if keep.sum() < 12:
            break
        c, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
        r = v - A @ c
        keep = np.abs(r) < 3.0 * 1.4826 * (np.median(np.abs(r[keep])) + 1e-6) + 2.0
    return c[0] + c[1] * xs + c[2] * ys + c[3] * xs * xs + c[4] * xs * ys + c[5] * ys * ys


@dataclass
class Footprint:
    """Het onderdeel op de mat (mat-mm): midden, straal (het verste punt vanaf het midden), afstand tot de rand van
    het bord, de omtrek, en de andere voorwerpen (midden, oppervlak in mm² en straal)."""

    center: tuple[float, float]
    radius_mm: float
    edge_mm: float
    outline: list[tuple[float, float]]
    others: list[tuple[float, float, float, float]] = field(default_factory=list)


class FootprintGrid:
    """Waar ligt het onderdeel op de mat, uit de livebeelden zelf (v0.16).

    Elk beeld wordt naar het vlak van de mat getransformeerd (de homografie van de mat) en per venster van 8 mm
    vergeleken met het patroon van de mat. Waar de mat ligt, klopt het patroon (een normalized cross-correlation, NCC,
    die ook onder een schaduw hoog blijft); waar iets ligt, is het patroon weg, ook bij een zwart of wit onderdeel.
    Vlakke stukken van het patroon (de witte rand rond een marker) vergelijken de helderheid met wit en zwart van de
    mat (een robuust vlak over het hele bord: het belichtingsverloop). Per cel van 2 mm tellen de beelden: hoe vaak in
    beeld, en hoe vaak bedekt. Een schuin beeld bedekt ook de mat achter het onderdeel (zijn wand en bovenvlak); in de
    beelden uit andere richtingen ligt daar gewoon mat, dus wat in de meeste beelden bedekt is, is het grondvlak.
    Beelden van boven tellen zwaarder."""

    def __init__(self, spec: MatSpec):
        self.spec = spec
        r = rasterize_board(spec, FOOT_PX_MM, 0.0).image.astype(np.float32) / 255.0
        self.R = cv2.GaussianBlur(r, (0, 0), FOOT_SIGMA_MM * FOOT_PX_MM)
        self.win = (int(round(FOOT_WIN_MM * FOOT_PX_MM)) | 1,) * 2
        self.mR = cv2.boxFilter(self.R, -1, self.win)
        self.vR = cv2.boxFilter(self.R * self.R, -1, self.win) - self.mR ** 2
        hh, ww = self.R.shape
        self.cells = (hh // FOOT_CELL, ww // FOOT_CELL)
        self.seen = np.zeros(self.cells, np.float32)
        self.hidden = np.zeros(self.cells, np.float32)
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        self.xn, self.yn = xx / ww - 0.5, yy / hh - 0.5
        self.white, self.black = self.R > 0.85, self.R < 0.15
        # board-mm → rasterpixel (het midden van pixel (0, 0) ligt op 0,25 mm)
        self.M0 = np.array([[FOOT_PX_MM, 0.0, -0.5], [0.0, FOOT_PX_MM, -0.5], [0.0, 0.0, 1.0]])
        self.frames = 0

    def objectness(self, gray: np.ndarray, H: np.ndarray) -> np.ndarray | None:
        """Per cel 0 (mat) .. 1 (iets anders dan de mat); NaN buiten beeld. None als er te weinig mat in beeld is."""
        hh, ww = self.R.shape
        M = self.M0 @ np.linalg.inv(H)
        warped = cv2.warpPerspective(gray.astype(np.float32), M, (ww, hh), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
        valid = warped >= 0
        I = cv2.GaussianBlur(np.where(valid, warped, 0.0).astype(np.float32), (0, 0), FOOT_SIGMA_MM * FOOT_PX_MM)
        sub = (slice(None, None, 3), slice(None, None, 3))
        levels = []
        for m in (self.white, self.black):
            sel = (m & valid)[sub]
            if sel.sum() < 200:
                return None
            levels.append(_quad_fit(self.xn[sub][sel], self.yn[sub][sel], I[sub][sel], self.xn, self.yn))
        W, B = levels
        In = (I - B) / np.maximum(W - B, 10.0)  # 0 = zwart van de mat, 1 = wit
        mI = cv2.boxFilter(In, -1, self.win)
        vI = cv2.boxFilter(In * In, -1, self.win) - mI * mI
        cov = cv2.boxFilter(In * self.R, -1, self.win) - mI * self.mR
        ncc = cov / np.sqrt(np.maximum(self.vR, 1e-6) * np.maximum(vI, 1e-6))
        textured = self.vR > 0.02
        # patroon: weg (ncc laag) of veel minder contrast dan de mat; vlak: helderheid anders dan voorspeld
        obj = np.where(textured,
                       0.5 * np.clip(1.0 - ncc, 0.0, 1.0) + 0.5 * np.clip(1.0 - vI / np.maximum(self.vR, 1e-6), 0, 1),
                       np.clip(np.abs(mI - self.mR) / 0.4, 0.0, 1.0))
        inside = cv2.erode(valid.astype(np.uint8), np.ones(self.win, np.uint8)).astype(bool)
        obj = np.where(inside, obj, np.nan)
        h2, w2 = self.cells[0] * FOOT_CELL, self.cells[1] * FOOT_CELL
        return obj[:h2, :w2].reshape(self.cells[0], FOOT_CELL, self.cells[1], FOOT_CELL).mean(axis=(1, 3))

    def add(self, gray: np.ndarray, H: np.ndarray, weight: float) -> bool:
        cells = self.objectness(gray, H)
        if cells is None:
            return False
        self.seen *= FOOT_DECAY
        self.hidden *= FOOT_DECAY
        valid = ~np.isnan(cells)
        self.seen[valid] += weight
        self.hidden[valid & (np.nan_to_num(cells) > 0.5)] += weight
        self.frames += 1
        return True

    def result(self) -> Footprint | None:
        """Het grootste voorwerp (in de meeste beelden bedekt, en minstens één keer goed gezien) en de andere."""
        mask = ((self.hidden >= 0.5 * self.seen) & (self.seen >= 1.0)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= FOOT_MIN_CELLS]
        if not comps:
            return None
        main = max(comps, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        mm = FOOT_CELL / FOOT_PX_MM  # mm per cel
        bw, bh = self.spec.board_w_mm, self.spec.board_h_mm

        def to_mat(cx: float, cy: float) -> tuple[float, float]:  # cel → mat-mm (nominaal: Y = bordhoogte - y)
            return (cx + 0.5) * mm, bh - (cy + 0.5) * mm

        def extent(i: int) -> tuple[float, float, float, np.ndarray, np.ndarray]:  # midden en straal (mm)
            ii, jj = np.nonzero(labels == i)
            X, Y = (jj + 0.5) * mm, bh - (ii + 0.5) * mm
            cx, cy = float(X.mean()), float(Y.mean())
            return cx, cy, float(np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).max()) + 0.5 * mm, X, Y

        cx, cy, radius, X, Y = extent(main)
        edge = float(min(X.min(), (bw - X).min(), Y.min(), (bh - Y).min())) - 0.5 * mm
        cont, _ = cv2.findContours((labels == main).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = cv2.approxPolyDP(max(cont, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
        outline = [to_mat(float(x), float(y)) for x, y in poly]
        others = [(*to_mat(*cents[i]), float(stats[i, cv2.CC_STAT_AREA] * mm * mm), extent(i)[2])
                  for i in comps if i != main]
        return Footprint((round(cx, 1), round(cy, 1)), round(radius, 1), round(edge, 1), outline, others)


# ----------------------------------------------------------------------------- per scan

class LiveSession:
    """Wat de live begeleiding per scan onthoudt tussen de beelden (v0.16): de brandpuntsafstand van de vorige beelden
    (de video heeft een andere beeldhoek dan de foto's), de mat (scheelt het zoeken met alle markers), de voetafdruk van
    het onderdeel en het vorige beeld (hoe snel de mat door het beeld beweegt)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.f: list[float] = []
        self.spec: MatSpec | None = None
        self.foot: FootprintGrid | None = None
        self.last: tuple[float, str, np.ndarray] | None = None  # (tijd in s, mat, homografie)
        self.lens: str | None = None  # welke camera de pagina gebruikt (v0.16.3)

    def f_rel(self) -> float | None:
        return float(np.median(self.f)) if self.f else None


def _to_image(points_mat, H: np.ndarray, spec: MatSpec, w: int, h: int) -> list | None:
    """Punten in het vlak van de mat (mat-mm, z = 0) naar genormaliseerde beeldcoördinaten, via de homografie: exact
    voor het vlak, zonder brandpuntsafstand. None als een punt achter de camera ligt."""
    p = np.asarray(points_mat, float).reshape(-1, 2)
    q = np.column_stack([p[:, 0], spec.board_h_mm - p[:, 1], np.ones(len(p))]) @ H.T
    if np.any(q[:, 2] <= 1e-9):
        return None
    uv = q[:, :2] / q[:, 2:] / [w, h]
    return uv.round(4).tolist()


def _arc(center, radius: float, a0: float, a1: float, n: int) -> np.ndarray:
    a = np.radians(np.linspace(a0, a1, n))
    return np.column_stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)])


def _image_speed(last, now: float | None, spec: MatSpec, H: np.ndarray) -> float | None:
    """Hoe snel de mat door het beeld beweegt (px/s), uit de homografie van het vorige beeld en die van nu."""
    if last is None or now is None or last[1] != spec.name:
        return None
    dt = now - last[0]
    if not 0.02 <= dt <= 1.0:
        return None
    bw, bh = spec.board_w_mm, spec.board_h_mm
    g = np.array([[x, y] for x in (0.25 * bw, 0.5 * bw, 0.75 * bw) for y in (0.25 * bh, 0.5 * bh, 0.75 * bh)])
    a = cv2.perspectiveTransform(g.reshape(-1, 1, 2), last[2]).reshape(-1, 2)
    b = cv2.perspectiveTransform(g.reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.median(np.linalg.norm(b - a, axis=1))) / dt


def live_check(img: np.ndarray, spec: MatSpec | None = None, overview: dict | None = None,
               f_rel: float | None = None, hint: MatSpec | None = None, session: LiveSession | None = None,
               t: float | None = None, lens: str | None = None) -> dict:
    """Beoordeelt één beeld van de livecamera (V25): is de mat in beeld, waar ligt het onderdeel, waar staat de
    camera ten opzichte daarvan, is dat een richting die nog ontbreekt, en is het beeld stil en scherp genoeg? Geeft
    één aanwijzing (`aanwijzing`; kort, in beeld en om uit te spreken: `spraak`; het pictogram: `icoon`), korte
    `opmerkingen`, `opnemen` (dit is een goed moment voor een foto) en de geometrie voor de overlay, in
    genormaliseerde beeldcoördinaten.

    `spec`: de mat van de scan (als die bekend is); `overview`: de samenvatting van de foto's tot nu toe
    (preflight.summarize: dekking, recht_van_boven, punten, object_mm); `session`: wat de begeleiding van de vorige
    beelden onthoudt; `t`: het moment van het beeld (s, klok van de telefoon), voor de snelheid; `lens`: welke camera
    de pagina gebruikt (v0.16.3; kiest de gebruiker een andere, dan begint de brandpuntsafstand opnieuw). Zonder
    sessie: `f_rel` (brandpuntsafstand / lange zijde) en `hint` (de mat van het vorige beeld)."""
    gray = to_work(img, LIVE_SIDE)
    h, w = gray.shape
    need = missing_views(overview)
    step, progress = _progress(need)
    out = {"breedte": w, "hoogte": h, "mat": None, "hoeken": 0, "mat_fractie": 0.0, "omtrek": None, "doel": None,
           "scherpte_px": None, "wit": None, "f_rel": None, "positie": None, "vak": None, "doelvak": None,
           "opnemen": False, "status": "zoek", "aanwijzing": "", "spraak": "", "icoon": "mat", "opmerkingen": [],
           "nodig": need, "gedekt": not any(need.values()), "stap": step, "voortgang": progress, "onderdeel": None,
           "ring": None, "pijl": None, "boven": None, "doel_elevatie": None, "snelheid_px_s": None, "stil": None,
           "px_per_mm": None}
    if session is not None:
        with session.lock:
            if lens is not None and lens != session.lens:  # een andere camera gekozen: een andere brandpuntsafstand
                session.f, session.last, session.lens = [], None, lens
            f_rel = f_rel or session.f_rel()
            hint = hint or session.spec
    det, found = _detect_any(gray, spec or hint, "live")
    if det is None:
        white = float(np.percentile(gray, 95))
        out["wit"] = round(white, 1)
        if white < 60:
            _tell(out, "Te donker: zorg voor meer (diffuus) licht.", "Te donker.", "donker")
        else:
            _tell(out, "Richt de camera op de kalibratiemat (grotendeels in beeld) en houd de telefoon stil.",
                  "Richt op de mat.", "mat")
        return out
    board = make_board(found)
    chess = np.asarray(board.getChessboardCorners(), float)[:, :2]
    out.update(mat=found.name, hoeken=int(len(det.ids)), mat_fractie=round(len(det.ids) / len(chess), 3))
    H, _ = cv2.findHomography(chess[det.ids], det.corners)
    if H is None:
        _tell(out, "Houd de mat in beeld.", "Houd de mat in beeld.", "mat")
        return out
    bw, bh = found.board_w_mm, found.board_h_mm
    out["omtrek"] = _to_image([[0, bh], [bw, bh], [bw, 0], [0, 0]], H, found, w, h)
    # hoe groot de mat in de foto komt (pixels per mm, op de werkresolutie van de verwerking; v0.16.3): uit de
    # homografie zelf, dus ook zonder brandpuntsafstand
    scale = mat_scale(H, found)
    out["px_per_mm"] = None if scale is None else round(scale * WORK_SIDE / max(w, h), 2)
    white, _, clipped = _photometry(gray, H, det, found)
    blur = _edge_blur(gray, H, det, found.nominal(), n_patches=16)
    out.update(wit=round(white, 1), scherpte_px=None if blur is None else round(blur, 2))
    f, tilt = _focal_and_tilt(H, w, h)
    if f is not None and tilt >= 20.0:  # alleen bij een schuine blik is f goed te bepalen
        out["f_rel"] = round(f / max(w, h), 4)
    notes = out["opmerkingen"]

    # wat de sessie onthoudt: de lens (brandpuntsafstand), de mat, de voetafdruk, de snelheid
    foot = None
    lens_jump = False
    if session is not None:
        with session.lock:
            if out["f_rel"]:
                ref = session.f_rel()
                if ref is not None and len(session.f) >= LENS_MIN_SAMPLES and abs(out["f_rel"] / ref - 1) > LENS_JUMP:
                    lens_jump = True  # niet in de geschiedenis: die blijft bij de lens van de scan
                else:
                    session.f = (session.f + [out["f_rel"]])[-25:]
            if session.spec is None or session.spec.name != found.name:
                session.spec, session.foot = found, None
            speed = _image_speed(session.last, t, found, H)
            if t is not None:
                session.last = (t, found.name, H)
            if session.foot is None:
                session.foot = FootprintGrid(found.nominal())
            if white >= 45:
                session.foot.add(gray, H, 1.0 if tilt <= 30.0 else 0.4)
            foot = session.foot.result()
        if speed is not None:
            out["snelheid_px_s"] = round(speed)
            out["stil"] = speed <= STILL_PX_S

    fk = (f_rel or LIVE_F_REL) * max(w, h)
    K = np.array([[fk, 0.0, (w - 1) / 2], [0.0, fk, (h - 1) / 2], [0.0, 0.0, 1.0]])
    pose = calib.solve_pose(det, found.nominal(), calib.CameraModel(K, np.zeros(5), w, h))
    if spec is not None and found.name != spec.name:
        _tell(out, f"Dit is mat {found.label}, maar de foto's van deze scan zijn op mat {spec.label} gemaakt.",
              f"Verkeerde mat: neem {spec.label}.", "let_op")
        return out
    if white < 45:
        _tell(out, "Te donker: zorg voor meer (diffuus) licht.", "Te donker.", "donker")
        return out
    if pose is None:
        _tell(out, "Houd de mat in beeld.", "Houd de mat in beeld.", "mat")
        return out

    # waar ligt het onderdeel: de voetafdruk uit de livebeelden, anders uit de foto's, anders het midden van de mat
    ov = overview or {}
    radius = None
    if foot is not None:
        obj, radius = foot.center, foot.radius_mm
        # de andere voorwerpen: in beeld een cirkel op de mat eromheen (v0.16.2), dan hoeft de opmerking niet te
        # zeggen waar ze liggen
        circles = [_to_image(_arc((x, y), r + 6.0, 0.0, 360.0, 25), H, found, w, h) for x, y, _, r in foot.others]
        out["onderdeel"] = {"midden_mm": list(foot.center), "straal_mm": foot.radius_mm, "rand_mm": foot.edge_mm,
                            "omtrek": _to_image(foot.outline, H, found, w, h), "bron": "live",
                            "andere": [[round(x), round(y)] for x, y, *_ in foot.others],
                            "andere_beeld": [c for c in circles if c is not None]}
        if len(foot.others) == 1:
            notes.append("ander voorwerp op de mat: haal het weg")
        elif foot.others:
            notes.append(f"{len(foot.others)} andere voorwerpen op de mat: haal ze weg")
        if foot.edge_mm < EDGE_MM:
            notes.append("onderdeel dicht bij de rand: leg het meer naar het midden")
        if foot.radius_mm > 0.4 * min(bw, bh):
            notes.append("onderdeel groot voor deze mat" + ("" if found.base_name == "A3" else ": neem A3"))
    elif ov.get("object_mm"):
        obj = tuple(float(v) for v in ov["object_mm"])
        out["onderdeel"] = {"midden_mm": list(obj), "straal_mm": None, "rand_mm": None, "omtrek": None,
                            "bron": "fotos", "andere": [], "andere_beeld": []}
    else:
        obj = (found.size_mm[0] / 2, found.size_mm[1] / 2)
    point = np.array([obj[0], obj[1], 0.0])
    d = point - pose.center
    dist = float(np.linalg.norm(d))
    el = 90.0 - math.degrees(math.acos(float(np.clip(-d[2] / (dist + 1e-12), -1, 1))))
    az = math.degrees(math.atan2(pose.center[1] - point[1], pose.center[0] - point[0])) % 360.0
    cell = _cell(az, el)
    out.update(positie={"azimut": round(az, 1), "elevatie": round(el, 1), "afstand_mm": round(dist),
                        "hoogte_mm": round(float(pose.center[2]))}, vak=cell)
    aim = _to_image([obj], H, found, w, h)
    out["doel"] = aim[0] if aim else None
    # de ring op de mat rond het onderdeel: binnen ~60° (hoog), buiten ~35° (laag); per vak 45°
    r1 = max(RING_MIN_MM, (radius or 30.0) + RING_MARGIN_MM)
    rings = {"hoog": r1, "laag": r1 + max(10.0, 0.25 * r1)}
    ring = {"straal_mm": {b: round(r, 1) for b, r in rings.items()}}
    for band, r in rings.items():
        arcs = [_to_image(_arc(obj, r, 45.0 * k - 22.5, 45.0 * k + 22.5, 7), H, found, w, h) for k in range(8)]
        ring[band] = arcs if all(a is not None for a in arcs) else None
    ring["band"] = _band_of(el)  # de hoogte waarop de camera nu is (None: daartussen, of recht van boven)
    here = _to_image([_arc(obj, rings[ring["band"] or "hoog"], az, az, 1)[0]], H, found, w, h)
    ring["hier"] = here[0] if here and el < LIVE_HIGH_DEG else None  # de kant van de camera
    out["ring"] = ring
    if white < 70:
        notes.append("donker: meer licht")
    if clipped > 0.05:
        notes.append("glans op de mat: geen direct licht")
    if out["snelheid_px_s"] is not None and out["snelheid_px_s"] > FAST_PX_S:
        notes.append("beweeg rustiger")
    out["status"] = "let_op"
    if lens_jump:
        _tell(out, "De telefoon wisselde van lens of zoom (de beeldhoek is anders): houd 25-35 cm afstand en zoom "
                   "niet; foto's van een andere lens passen niet in de scan.", "Andere lens. Houd meer afstand.",
              "let_op")
        return out
    if view_distance(pose) < CLOSE_MM:
        _tell(out, "Te dichtbij: houd de telefoon 25-35 cm van het onderdeel.", "Verder weg.", "weg")
        return out
    if dist > FAR_MM:
        _tell(out, "Te ver weg: kom dichterbij, tot 25-35 cm van het onderdeel.", "Dichterbij.", "dichterbij")
        return out
    # v0.16.3: de mat klein in beeld, bijvoorbeeld met de groothoeklens: dan ziet de verwerking de stippen van de mat
    # en een donker onderdeel niet
    wide = (f_rel is not None and f_rel < WIDE_F_REL) or bool(ov.get("groothoek"))
    if out["px_per_mm"] is not None and out["px_per_mm"] < LOW_PX_MM:
        if wide:
            _tell(out, "De telefoon gebruikt de groothoeklens (0,6×), en dan is de mat klein in beeld: kies de gewone "
                       "lens (1×) met de knop Lens, of kom dichterbij.", "Groothoeklens: kies 1×.", "lens")
        else:
            _tell(out, "De mat is klein in beeld: kom dichterbij, met de mat grotendeels in beeld.", "Dichterbij.",
                  "dichterbij")
        return out
    if cell is not None and need.get(cell, 0) > 0 and (cell == "boven" or need["boven"] == 0):
        out["doelvak"] = cell
        out["doel_elevatie"] = _cell_center(cell)[1]
        margin = 0.3 if cell == "boven" else 0.2  # het midden van het onderdeel niet aan de rand van het beeld
        u, v = (out["doel"] or (-1.0, -1.0))
        sep = LIVE_SEP_TOP_DEG if cell == "boven" else LIVE_SEP_DEG
        near = [p for p in ov.get("punten") or [] if p.get("oordeel") != "onbruikbaar"
                and _angle(az, el, p["azimut"], p["elevatie"]) < sep]
        if blur is None or blur > LIVE_BLUR_MAX:  # eerst: een bewogen beeld mist ook mathoeken
            _tell(out, "Houd de telefoon stil: het beeld is onscherp.", "Stil houden.", "stil")
        elif out["stil"] is False:
            _tell(out, "Houd de telefoon stil: het beeld beweegt nog.", "Stil houden.", "stil")
        elif not (margin <= u <= 1 - margin and margin <= v <= 1 - margin):
            _tell(out, "Richt de camera op het onderdeel: midden in beeld.", "Richt op het onderdeel.", "richt")
        elif out["mat_fractie"] < 0.25:
            _tell(out, "Neem meer van de mat mee in beeld.", "Meer mat in beeld.", "mat")
        elif near:
            out["status"] = "verplaats"
            _tell(out, "Hier is al een foto: schuif de telefoon een paar centimeter op.", "Een stukje verder.",
                  "opschuiven")
        else:
            out["opnemen"], out["status"] = True, "opnemen"
            where = "recht van boven" if cell == "boven" else "deze richting ontbreekt nog"
            _tell(out, f"Goed, {where}: houd stil voor de foto.", "Goed, stil houden.", "opnemen")
        return out
    target = _next_cell(az, el, need)
    if target is None:
        out["status"] = "klaar"
        _tell(out, "Alle richtingen zijn gedekt: je kunt de scan verwerken (meer foto's mag).",
              "Klaar. Alle richtingen zijn gedekt.", "klaar")
        return out
    out["status"] = "verplaats"
    out["doelvak"] = target
    out["doel_elevatie"] = _cell_center(target)[1]
    _tell(out, *_direction(az, el, target))
    if target == "boven":
        # schuif de telefoon (evenwijdig aan de mat) tot recht boven het onderdeel: de richting in het beeld
        shift = pose.R @ np.array([point[0] - pose.center[0], point[1] - pose.center[1], 0.0])
        n = float(np.hypot(shift[0], shift[1]))
        if n > 1e-6:
            out["boven"] = {"hoek": round(90.0 - el, 1), "richting": [round(shift[0] / n, 3), round(shift[1] / n, 3)]}
    else:
        caz, _ = _cell_center(target)
        daz = (caz - az + 180.0) % 360.0 - 180.0
        if abs(daz) >= 10.0 and el < LIVE_HIGH_DEG:  # een pijl langs de ring van de doelhoogte, van hier naar het doel
            band = target.split("-")[0]
            out["pijl"] = _to_image(_arc(obj, rings[band], az, az + daz, max(4, int(abs(daz) // 8) + 2)), H, found, w, h)
    return out
