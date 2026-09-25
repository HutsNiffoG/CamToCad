"""Kalibratiemat: specificatie, vectorgeometrie, PDF/PNG-export en het bijbehorende OpenCV-board.

De mat is een ChArUco-bord (schaakbord met ArUco-markers in de witte vakken). In de zwarte
vakken staan witte stippen: extra textuur voor het uitsnijden van silhouetten (masks.py)
zonder de detectie van schaakbordhoeken te hinderen (stippen blijven weg van de hoeken).

Twee versies:
* *v1*: drie losse stippen per zwart vak, markers 0.. uit DICT_5X5_250 (A4 en A3 delen ID's);
* *v2*: een dicht stippenraster in elk zwart vak, zodat ook een donker onderdeel op een zwart
  vak te zien is (de stippen verdwijnen), en een eigen marker-ID-bereik per formaat uit
  DICT_5X5_1000. Het formaat (en de versie) is daardoor aan de foto's te herkennen.

Twee assenstelsels:
* *board* (OpenCV): oorsprong linksboven, x naar rechts, y naar beneden, in mm;
* *mat* (pipeline): oorsprong linksonder, X naar rechts, Y omhoog, Z uit het papier.

Printschaal: een printer drukt zelden exact op 100%, en niet altijd in beide richtingen
gelijk. `scale_x`/`scale_y` (gemeten / nominale lengte van de meetlijnen) maken alle geometrie
in werkelijke millimeters; de tekening zelf (PDF, raster) blijft nominaal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .pdf import PdfCanvas

Rect = tuple[float, float, float, float, float]  # x, y, breedte, hoogte, grijswaarde (0 = zwart)


@dataclass(frozen=True)
class MatSpec:
    name: str
    page_w_mm: float
    page_h_mm: float
    squares_x: int
    squares_y: int
    square_mm: float = 20.0
    marker_mm: float = 15.0
    dictionary: str = "DICT_5X5_250"
    dots_per_square: int = 3  # v1: losse stippen per zwart vak
    dot_mm: float = 1.6
    seed: int = 7
    version: int = 1
    first_id: int = 0  # eerste marker-ID; v2 heeft per formaat een eigen bereik
    dot_pitch_mm: float = 0.0  # v2: stippen op een verspringend raster met deze steek
    scale_x: float = 1.0  # printschaal: gemeten / nominale lengte van de meetlijn langs X
    scale_y: float = 1.0  # idem langs Y

    @property
    def board_w_mm(self) -> float:
        """Nominale breedte van het bord (zoals getekend)."""
        return self.squares_x * self.square_mm

    @property
    def board_h_mm(self) -> float:
        return self.squares_y * self.square_mm

    @property
    def size_mm(self) -> tuple[float, float]:
        """Werkelijke afmetingen van het geprinte bord (mm), met de printschaal."""
        return self.board_w_mm * self.scale_x, self.board_h_mm * self.scale_y

    @property
    def n_markers(self) -> int:
        return self.squares_x * self.squares_y // 2

    @property
    def marker_ids(self) -> np.ndarray:
        return np.arange(self.first_id, self.first_id + self.n_markers, dtype=np.int32)

    @property
    def base_name(self) -> str:
        return self.name.split("-")[0]

    @property
    def label(self) -> str:
        return self.name if self.version >= 2 else f"{self.base_name} (v1)"

    @property
    def mat_id(self) -> str:
        return f"CTC-{self.base_name}-{self.seed}" if self.version < 2 else f"CTC{self.version}-{self.base_name}"

    @property
    def is_scaled(self) -> bool:
        return (self.scale_x, self.scale_y) != (1.0, 1.0)

    def with_scale(self, sx: float, sy: float | None = None) -> "MatSpec":
        return replace(self, scale_x=float(sx), scale_y=float(sx if sy is None else sy))

    def nominal(self) -> "MatSpec":
        return replace(self, scale_x=1.0, scale_y=1.0)

    def board_origin_on_page(self) -> tuple[float, float]:
        """Linkeronderhoek van het bord op de pagina (mm); iets omhoog voor de onderste meetlijn."""
        return (self.page_w_mm - self.board_w_mm) / 2.0, (self.page_h_mm - self.board_h_mm) / 2.0 + 4.0

    def to_dict(self) -> dict:
        return asdict(self)


_V2 = dict(dictionary="DICT_5X5_1000", version=2, dots_per_square=0, dot_mm=1.0, dot_pitch_mm=2.5)

PRESETS: dict[str, MatSpec] = {s.name.upper(): s for s in (
    MatSpec("A4", 297.0, 210.0, 12, 8, first_id=250, **_V2),
    MatSpec("Letter", 279.4, 215.9, 11, 8, first_id=300, **_V2),
    MatSpec("A3", 420.0, 297.0, 18, 12, first_id=350, **_V2),
    MatSpec("A4-v1", 297.0, 210.0, 12, 8),
    MatSpec("A3-v1", 420.0, 297.0, 18, 12),
)}
PRINTABLE = ("A4", "A3", "Letter")  # formaten om te printen; v1 blijft alleen herkend voor oude scans


def get_spec(spec: str | MatSpec | dict) -> MatSpec:
    if isinstance(spec, MatSpec):
        return spec
    if isinstance(spec, dict):
        return MatSpec(**spec)
    try:
        return PRESETS[spec.upper()]
    except KeyError:
        names = ", ".join(s.name for s in PRESETS.values())
        raise ValueError(f"Onbekende mat '{spec}'; kies uit {names}") from None


def make_board(spec: MatSpec) -> "cv2.aruco.CharucoBoard":
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    return cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y), spec.square_mm, spec.marker_mm, dictionary, spec.marker_ids
    )


def board_to_mat_transform(spec: MatSpec) -> tuple[np.ndarray, np.ndarray]:
    """(A, a) zodat X_mat = A @ X_board + a, voor nominale board-coördinaten (met de printschaal)."""
    return np.diag([spec.scale_x, -spec.scale_y, -1.0]), np.array([0.0, spec.scale_y * spec.board_h_mm, 0.0])


def board_to_mat(points_board: np.ndarray, spec: MatSpec) -> np.ndarray:
    """Zet nominale board-coördinaten (y omlaag, z het papier in) om naar mat-coördinaten (Y en Z omhoog)."""
    A, a = board_to_mat_transform(spec)
    return np.asarray(points_board, dtype=float).reshape(-1, 3) @ A.T + a


# ----------------------------------------------------------------------------- geometrie

def _texture_v1(spec: MatSpec, black_squares) -> list[Rect]:
    rng = np.random.default_rng(spec.seed)
    s = spec.square_mm
    lo, hi = 0.25 * s, 0.75 * s - spec.dot_mm
    rects = []
    for i, j in black_squares:
        for _ in range(spec.dots_per_square):
            dx, dy = rng.uniform(lo, hi, size=2)
            rects.append((i * s + dx, j * s + dy, spec.dot_mm, spec.dot_mm, 1.0))
    return rects


def _texture_v2(spec: MatSpec, black_squares, clear_edge: float = 1.5, clear_corner: float = 4.0,
                jitter: float = 0.3) -> list[Rect]:
    """Dicht stippenraster per zwart vak. Vrij blijven: een strook langs de randen (voor schone
    randprofielen, zie preflight.py) en een zone rond de hoeken, waar OpenCV de schaakbordhoeken
    verfijnt (venster hooguit ~3,5 mm)."""
    s, d, pitch = spec.square_mm, spec.dot_mm, spec.dot_pitch_mm
    m = int((s - 2 * clear_edge - d - 2 * jitter) // pitch) + 1
    base = s / 2 + (np.arange(m) - (m - 1) / 2) * pitch
    corners = np.array([[0.0, 0.0], [0.0, s], [s, 0.0], [s, s]])
    rng = np.random.default_rng(spec.seed)
    rects = []
    for i, j in black_squares:
        for cx in base:
            for cy in base:
                c = np.array([cx, cy]) + rng.uniform(-jitter, jitter, 2)
                if np.min(np.linalg.norm(corners - c, axis=1)) < clear_corner + d / np.sqrt(2):
                    continue
                rects.append((i * s + c[0] - d / 2, j * s + c[1] - d / 2, d, d, 1.0))
    return rects


def board_rects(spec: MatSpec, dots: bool = True) -> list[Rect]:
    """Alle te tekenen rechthoeken in board-coördinaten (mm, oorsprong linksboven, y omlaag)."""
    board = make_board(spec)
    s = spec.square_mm
    corners = [np.asarray(c, dtype=float).reshape(4, 3) for c in board.getObjPoints()]
    ids = board.getIds().ravel()
    marker_squares = {(int(c[:, 0].mean() // s), int(c[:, 1].mean() // s)) for c in corners}

    rects: list[Rect] = []
    black_squares = []
    for j in range(spec.squares_y):
        for i in range(spec.squares_x):
            if (i, j) not in marker_squares:
                rects.append((i * s, j * s, s, s, 0.0))
                black_squares.append((i, j))

    dictionary = board.getDictionary()
    n = dictionary.markerSize + 2  # inclusief zwarte rand van één bit
    for marker_id, c in zip(ids, corners):
        bits = dictionary.generateImageMarker(int(marker_id), n)
        cell = spec.marker_mm / n
        x0, y0 = c[0, 0], c[0, 1]
        for r in range(n):
            row = bits[r] < 128
            col = 0
            while col < n:
                if row[col]:
                    start = col
                    while col < n and row[col]:
                        col += 1
                    rects.append((x0 + start * cell, y0 + r * cell, (col - start) * cell, cell, 0.0))
                else:
                    col += 1

    if dots and spec.dot_pitch_mm > 0:
        rects += _texture_v2(spec, black_squares)
    elif dots and spec.dots_per_square > 0:
        rects += _texture_v1(spec, black_squares)
    return rects


def ruler_rects(spec: MatSpec) -> tuple[list[Rect], list[tuple[float, float, str, float]]]:
    """Twee loodrechte meetlijnen van 100,0 mm (paginacoördinaten, y omhoog) plus labels."""
    ox, oy = spec.board_origin_on_page()
    lw = 0.25
    rects: list[Rect] = []
    texts: list[tuple[float, float, str, float]] = []

    yr = oy - 10.0  # horizontale meetlijn onder het bord
    rects.append((ox - lw / 2, yr - lw / 2, 100.0 + lw, lw, 0.0))
    for k in range(11):
        length = 4.5 if k % 5 == 0 else 2.5
        rects.append((ox + 10.0 * k - lw / 2, yr - length, lw, length, 0.0))
    texts += [(ox - 0.8, yr - 8.0, "0", 7.0), (ox + 48.6, yr - 8.0, "50", 7.0), (ox + 97.0, yr - 8.0, "100 mm", 7.0)]
    if spec.version >= 2:
        texts.append((ox + 108.0, yr - 1.2, "Meetlijn X = 100,0 mm, meetlijn Y links: meet beide na met een "
                                            "schuifmaat en geef afwijkingen op", 7.0))
    else:
        texts.append((ox + 108.0, yr - 1.2, "Meetlijn 100,0 mm - controleer met een schuifmaat of liniaal", 7.0))

    xr = ox - 10.0  # verticale meetlijn links van het bord
    rects.append((xr - lw / 2, oy - lw / 2, lw, 100.0 + lw, 0.0))
    for k in range(11):
        length = 4.5 if k % 5 == 0 else 2.5
        rects.append((xr - length, oy + 10.0 * k - lw / 2, length, lw, 0.0))
    texts += [(xr - 7.0, oy - 1.0, "0", 7.0), (xr - 9.5, oy + 49.0, "50", 7.0), (xr - 11.0, oy + 99.0, "100", 7.0)]
    if spec.version >= 2:
        texts.append((xr - 1.6, oy + 104.0, "Y", 8.0))
        texts.append((ox + 102.0, yr - 1.4, "X", 8.0))
    return rects, texts


def page_rects(spec: MatSpec) -> tuple[list[Rect], list[tuple[float, float, str, float]]]:
    """Alle rechthoeken en teksten van de pagina (mm, oorsprong linksonder, y omhoog)."""
    ox, oy = spec.board_origin_on_page()
    h = spec.board_h_mm
    rects = [(ox + x, oy + (h - y - hh), w, hh, g) for x, y, w, hh, g in board_rects(spec)]
    rr, texts = ruler_rects(spec)
    rects += rr
    top = spec.page_h_mm
    square = f"{spec.square_mm:.1f}".replace(".", ",")
    marker = f"{spec.marker_mm:.1f}".replace(".", ",")
    print_note = ("Print op 100% (werkelijke grootte), niet 'aanpassen aan pagina'. "
                  "Gebruik mat papier en leg de mat vlak.")
    layout = f"{spec.squares_x} x {spec.squares_y} vakken van {square} mm, markers {marker} mm"
    if spec.version >= 2:
        ids = spec.marker_ids
        title = f"Cam-to-CAD kalibratiemat {spec.name} (v{spec.version}) - ID {spec.mat_id}"
        texts.append((ox, top - 9.0, title, 10.0))
        texts.append((ox, top - 14.5, f"{layout} ({spec.dictionary}, ID {ids[0]}-{ids[-1]}).", 7.0))
        texts.append((ox, top - 18.0, print_note, 7.0))
    else:
        subtitle = f"{layout} ({spec.dictionary}). " + print_note
        texts.append((ox, top - 9.0, f"Cam-to-CAD kalibratiemat {spec.base_name} - ID {spec.mat_id}", 10.0))
        texts.append((ox, top - 14.5, subtitle, 7.0))
    return rects, texts


# ----------------------------------------------------------------------------- export

def write_pdf(spec: MatSpec, path: str | Path) -> None:
    canvas = PdfCanvas(spec.page_w_mm, spec.page_h_mm)
    rects, texts = page_rects(spec)
    for x, y, w, h, g in rects:
        canvas.rect(x, y, w, h, gray=g)
    for x, y, t, size in texts:
        canvas.text(x, y, t, size_pt=size)
    canvas.save(path)


def _fill_rects(img: np.ndarray, rects, to_px) -> None:
    for rect in rects:
        c0, r0, c1, r1 = to_px(rect)
        img[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)] = int(round(rect[4] * 255))


@dataclass(frozen=True)
class BoardRaster:
    """Rasterversie van het bord met een witte rand, en de afbeelding mat-mm → pixel."""

    image: np.ndarray
    px_per_mm: float
    margin_mm: float
    spec: MatSpec

    def mat_to_pixel_matrix(self) -> np.ndarray:
        """3x3-matrix die homogene mat-coördinaten (X, Y, 1) afbeeldt op (kolom, rij, 1).

        Pixelconventie van OpenCV: het midden van pixel (0, 0) ligt op (0, 0). Een vak dat bij het
        rasteren de pixels c0 .. c1-1 vult, loopt dus van c0 - 0,5 tot c1 - 0,5 (vandaar de -0,5).
        Het raster is nominaal getekend; mat-coördinaten zijn werkelijke mm (printschaal).
        """
        k, m, h = self.px_per_mm, self.margin_mm, self.spec.board_h_mm
        sx, sy = self.spec.scale_x, self.spec.scale_y
        return np.array([[k / sx, 0.0, k * m - 0.5], [0.0, -k / sy, k * (h + m) - 0.5], [0.0, 0.0, 1.0]])


def rasterize_board(
    spec: MatSpec, px_per_mm: float = 10.0, margin_mm: float = 3.0, supersample: int = 2, dots: bool = True
) -> BoardRaster:
    """Rastert het bord (anti-aliased via supersampling) voor rendering en achtergrondvoorspelling."""
    k = px_per_mm * supersample
    w = int(round((spec.board_w_mm + 2 * margin_mm) * k))
    h = int(round((spec.board_h_mm + 2 * margin_mm) * k))
    img = np.full((h, w), 255, np.uint8)

    def to_px(rect):
        x, y, rw, rh, _ = rect
        return (
            int(round((x + margin_mm) * k)),
            int(round((y + margin_mm) * k)),
            int(round((x + rw + margin_mm) * k)),
            int(round((y + rh + margin_mm) * k)),
        )

    _fill_rects(img, board_rects(spec, dots=dots), to_px)
    if supersample > 1:
        out_size = (int(round(w / supersample)), int(round(h / supersample)))
        img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
    return BoardRaster(img, px_per_mm, margin_mm, spec)


def rasterize_page(spec: MatSpec, px_per_mm: float = 5.0) -> np.ndarray:
    """Voorbeeldweergave (PNG) van de hele pagina; de PDF is de printversie."""
    w, h = int(round(spec.page_w_mm * px_per_mm)), int(round(spec.page_h_mm * px_per_mm))
    img = np.full((h, w), 255, np.uint8)
    rects, texts = page_rects(spec)

    def to_px(rect):
        x, y, rw, rh, _ = rect
        return (
            int(round(x * px_per_mm)),
            int(round((spec.page_h_mm - y - rh) * px_per_mm)),
            int(round((x + rw) * px_per_mm)),
            int(round((spec.page_h_mm - y) * px_per_mm)),
        )

    _fill_rects(img, rects, to_px)
    for x, y, t, size in texts:
        scale = size * 0.3528 * px_per_mm / 22.0  # punt → mm → pixels; Hershey-font is ~22 px hoog bij schaal 1
        cv2.putText(img, t, (int(x * px_per_mm), int((spec.page_h_mm - y) * px_per_mm)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 1, cv2.LINE_AA)
    return img


def descriptor(spec: MatSpec) -> dict:
    ox, oy = spec.board_origin_on_page()
    ids = spec.marker_ids
    return {
        "type": "camtocad-mat",
        "version": spec.version,
        "mat_id": spec.mat_id,
        "spec": spec.to_dict(),
        "marker_ids": [int(ids[0]), int(ids[-1])],
        "board_size_mm": [spec.board_w_mm, spec.board_h_mm],
        "board_origin_on_page_mm": [ox, oy],
        "mat_frame": "oorsprong linksonder van het bord, X rechts, Y omhoog over het papier, Z uit het papier (mm)",
        "check_lines_mm": {"horizontal": [ox, oy - 10.0, 100.0], "vertical": [ox - 10.0, oy, 100.0]},
        "opencv": cv2.__version__,
        "camtocad": __version__,
    }


def write_mat(spec: str | MatSpec, out_dir: str | Path) -> dict[str, Path]:
    """Schrijft PDF (printversie), PNG (voorbeeld) en JSON (beschrijving) naar out_dir."""
    spec = get_spec(spec)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"kalibratiemat_{spec.name}"
    paths = {"pdf": out / f"{stem}.pdf", "png": out / f"{stem}.png", "json": out / f"{stem}.json"}
    write_pdf(spec, paths["pdf"])
    cv2.imwrite(str(paths["png"]), rasterize_page(spec))
    paths["json"].write_text(json.dumps(descriptor(spec), indent=2), encoding="utf-8")
    return paths
