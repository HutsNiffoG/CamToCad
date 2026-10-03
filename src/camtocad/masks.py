"""Objectmaskers via de bekende mat-achtergrond.

Met de camerapose is precies te voorspellen hoe de mat er in elke foto uitziet. Pixels die
daarvan afwijken horen bij het object. Klassen per pixel:

* `fg`    – objectpixel (wijkt af van de voorspelde mat);
* `bg`    – *zeker* mat: klopt met de voorspelling én de voorspelling heeft daar textuur;
* `amb`   – dubbelzinnig: het object heeft hier dezelfde grijswaarde als de mat (zwart op een
            zwart stuk zonder stippen, wit op wit), dus de foto zegt hier niets. Binnen het object
            is zo'n stuk voor de startcontour opgevuld (ook `fg`); de fit negeert het;
* overig  – onbekend (bijv. egale stukken mat, of buiten de mat).

Alleen zekere mat-pixels mogen voxels wegsnijden (hull.py). Zo veroorzaakt een donker
object op een zwart vak geen gat in het model: daar is het niet 'mat'.

Kleur (V8): de mat is zwart-wit. Een gekleurd object (blauw geanodiseerd, rood kunststof) is ook
zichtbaar waar het even donker of licht is als de mat eronder; zie `_chroma_evidence`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calib import Pose
from .mat import BoardRaster


@dataclass
class Tone:
    """Hoe de camera het licht vastlegt (tone.py): toonkromme T(x) = 255 · (x / 255)^g, extra onscherpte van de foto
    t.o.v. de voorspelde mat (σ in px, in lineair licht) en verscherping (unsharp masking: k, σ in px)."""

    g: float = 1.0
    blur: float = 0.0
    sharpen: float = 0.0
    sharpen_px: float = 1.5
    grid: np.ndarray | None = None  # lokale toonbewerking (HDR): g per tegel (n x n over het beeld), of None

    def to_dict(self) -> dict:
        out = {"g": round(self.g, 3), "onscherpte_px": round(self.blur, 2), "verscherping_k": round(self.sharpen, 3),
               "verscherping_px": round(self.sharpen_px, 2)}
        if self.grid is not None:
            out["g_per_tegel"] = np.round(self.grid, 3).tolist()
        return out

    def g_map(self, shape: tuple[int, int]):
        """g per pixel (een vloeiende overgang tussen de tegelmiddens), of één getal zonder raster."""
        if self.grid is None:
            return self.g
        return cv2.resize(self.grid.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _curved(g) -> bool:
    """Is er een toonkromme (g ≠ 1, of g per pixel)?"""
    return not np.isscalar(g) or g != 1.0


def desharpen(img: np.ndarray, k: float, sigma: float, iters: int = 6) -> np.ndarray:
    """Maakt unsharp masking ongedaan: E uit S = (1 + k) E - k G(E), met vaste-puntiteratie (convergeert: k/(1+k) < 1).
    Dat is een laagdoorlaatfilter: ruis en JPEG-randjes worden er kleiner van."""
    o = img.astype(np.float32)
    if k <= 0:
        return o
    E = o.copy()
    for _ in range(iters):
        E = (o + k * cv2.GaussianBlur(E, (0, 0), sigma)) / (1.0 + k)
    return np.clip(E, 0.0, 255.0)


@dataclass
class ViewMasks:
    fg: np.ndarray  # objectpixels
    bg: np.ndarray  # zekere mat, met een veiligheidsmarge rond het object (voor het uitsnijden)
    valid: np.ndarray  # pixel valt op de mat
    edge_bg: np.ndarray | None = None  # zekere mat zonder marge (voor de modelverfijning)
    sigma: float = 0.0  # ruisniveau van het residu (grijswaarden)
    amb: np.ndarray | None = None  # object zou hier onzichtbaar zijn (zelfde grijs als de mat): geen bewijs
    # zachte objectfractie rond het object (V2, v0.9; zie _soft_alpha), als uitsnede vanaf `alpha_at` (rij, kolom)
    alpha: np.ndarray | None = None
    alpha_w: np.ndarray | None = None  # gewicht: 1 / variantie van alpha, 0 zonder bewijs
    alpha_at: tuple[int, int] = (0, 0)
    # de grijswaarden (vervaagd, belichting gecorrigeerd) in dezelfde uitsnede: randen binnen het object (V16)
    gray: np.ndarray | None = None
    # in dezelfde uitsnede: afgekapt (0 of 255) of vlak ernaast (_clip_zone); daar geen bewijs uit alpha (v0.11)
    clip: np.ndarray | None = None
    tone: float = 1.0  # exponent van de toonkromme van de camera (V2, v0.10; zie _tone_exponent); 1 = lineair
    tone_params: Tone | None = None  # alles wat er over de camera bekend is (v0.11; tone.py)
    tone_map: np.ndarray | None = None  # g per pixel in de uitsnede van `gray`, bij lokale toonbewerking (v0.11)


def predict_background(raster: BoardRaster, K: np.ndarray, pose: Pose, size: tuple[int, int]):
    """Voorspelde grijswaarde van de mat en een masker van pixels die op de mat vallen."""
    w, h = size
    s = 2
    Ks = K.copy()
    Ks[:2, :2] *= s
    Ks[0, 2], Ks[1, 2] = s * K[0, 2] + 0.5 * (s - 1), s * K[1, 2] + 0.5 * (s - 1)
    Hm = Ks @ np.column_stack([pose.R[:, 0], pose.R[:, 1], pose.t]) @ np.linalg.inv(raster.mat_to_pixel_matrix())
    pred = cv2.warpPerspective(raster.image.astype(np.float32), Hm, (w * s, h * s), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside = cv2.warpPerspective(np.full(raster.image.shape, 255, np.uint8), Hm, (w * s, h * s),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(inside, (w, h), interpolation=cv2.INTER_AREA) >= 255
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return pred, valid


def _robust_affine(o: np.ndarray, p: np.ndarray, sel: np.ndarray) -> tuple[float, float, float]:
    """Robuuste fit o ≈ a·p + b; geeft (a, b, sigma)."""
    ov, pv = o[sel][::7], p[sel][::7]
    a, b, thr = 1.0, float(np.median(ov - pv)), 80.0
    sigma = 10.0
    for _ in range(4):
        r = ov - (a * pv + b)
        keep = np.abs(r) < thr
        if keep.sum() < 50:
            break
        A = np.column_stack([pv[keep], np.ones(keep.sum())])
        (a, b), *_ = np.linalg.lstsq(A, ov[keep], rcond=None)
        r = ov[keep] - (a * pv[keep] + b)
        sigma = 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-3
        thr = max(4.0 * sigma, 6.0)
    return float(a), float(b), sigma


def _tone_exponent(o: np.ndarray, p: np.ndarray, sel: np.ndarray) -> tuple[float, float]:
    """Toonkromme van de camera (V2, v0.10): de exponent g in foto = T(a · p + b) met T(x) = 255 · (x / 255)^g,
    met p de voorspelde mat, lineair in het licht (de print vervaagd door de optiek), en a, b belichting en het
    licht dat ook zwart nog terugkaatst.

    Een telefoon slaat niet het licht zelf op maar een kromme ervan (sRGB, ~1/2,2, met nog een S-bocht). Zwart en
    wit van de mat passen bij elke kromme op een rechte lijn; alleen de overgangen ertussen, waar de optiek ze
    mengt, laten de kromme zien. Per kandidaat g de foto terug naar lineair licht, T⁻¹(foto), met een robuuste
    lineaire fit op p; de juiste g maakt de residuen op die overgangen het kleinst. Drie dingen lijken ook op een
    kromme en tellen daarom mee (§3j):
    - de kromme hoort bij het licht, dus na b: een kromme op p zelf kromt ook de zwartwaarde (sRGB werd ~0,7);
    - `o` is de foto zelf, niet vervaagd: vervagen mengt gecodeerde grijswaarden, en dan lijken de overgangen
      rechter dan ze zijn (sRGB werd ~0,6);
    - de foto is vaak iets vager dan de voorspelling (optiek, ontvervormen), en dan zijn vooral de stippen van de
      mat minder diep: g wordt samen met een extra vervaging van de voorspelling geschat (TONE_BLUR_PX).
    Op gerenderde scans tot op ~0,04: verscherping in de telefoon (op de gecodeerde grijswaarden) maakt g iets te
    klein, zonder verscherping is hij iets te groot. Geeft (g, extra vervaging in px); g = 1,0 als de kromme weinig
    uitmaakt."""
    idx = np.flatnonzero(sel.ravel())[::11]
    if len(idx) < 2000:
        return 1.0, 0.0
    y = np.clip(o.ravel()[idx].astype(np.float64) / 255.0, 0.0, 1.0)
    p = p.astype(np.float32)
    blurred: dict[float, np.ndarray] = {}

    def at(s: float) -> np.ndarray:
        s = round(max(s, 0.0), 2)
        if s not in blurred:
            blurred[s] = (cv2.GaussianBlur(p, (0, 0), s) if s > 0 else p).ravel()[idx].astype(np.float64)
        return blurred[s]

    pv = at(0.0)
    lo, hi = np.percentile(pv, [5, 95])
    trans = (pv > lo + 0.15 * (hi - lo)) & (pv < hi - 0.15 * (hi - lo))
    if np.count_nonzero(trans) < 500:
        return 1.0, 0.0
    every = np.ones(len(y), bool)
    linear: dict[float, np.ndarray] = {}

    def spread(g: float, s: float) -> float:
        if g not in linear:
            linear[g] = 255.0 * y ** (1.0 / g)
        ol, ps = linear[g], at(s)
        a, b, _ = _robust_affine(ol, ps, every)
        return float(np.median(np.abs(ol - (a * ps + b))[trans]) / max(abs(a), 1e-3))

    def best(gs, ss) -> tuple[float, float, float]:
        return min((spread(g, s), float(g), float(s)) for g in gs for s in ss)

    e0, g0, s0 = best(np.arange(0.30, 1.3001, 0.1), TONE_BLUR_PX[:4])
    if s0 >= TONE_BLUR_PX[3]:  # een bewogen of onscherpe foto: ook ruimer
        e1, g1, s1 = best(np.arange(0.30, 1.3001, 0.1), TONE_BLUR_PX[4:])
        if e1 < e0:
            g0, s0 = g1, s1
    ds = 0.15 if s0 < 1.5 else 0.3
    sp, g, s = best(np.arange(max(0.25, g0 - 0.1), g0 + 0.1001, 0.02), (s0 - ds, s0, s0 + ds))
    s = max(s, 0.0)
    if abs(g - 1.0) < TONE_MIN_DEV or sp > (1.0 - TONE_MIN_GAIN) * spread(1.0, s):
        return 1.0, round(s, 2)
    return round(g, 2), round(s, 2)


def _linear(v: np.ndarray, g: float) -> np.ndarray:
    """Een gecodeerde grijswaarde terug naar lineair licht: T⁻¹ van de toonkromme (_tone_exponent)."""
    return 255.0 * np.clip(v / 255.0, 1e-6, 1.5) ** (1.0 / g)


def _linear_color(chroma: np.ndarray, lum: np.ndarray, g: float) -> tuple[np.ndarray, np.ndarray]:
    """Kleur in lineair licht (V2, v0.10): bij een toonkromme zijn de kleurverschillen van de gecodeerde kanalen geen
    mengverhoudingen meer, en klopt het model van de mat (kleurzweem evenredig met het grijs) niet. Uit grijs en de
    twee kleurverschillen volgen R, G en B per pixel (het grijs eerst even vaag als de kleur, die op halve resolutie
    is opgeslagen); die per kanaal terug naar lineair licht, en daaruit opnieuw de kleurverschillen. Geeft (kleur,
    grijs) in lineair licht."""
    y = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 0.7)
    c1, c2 = chroma[..., 0].astype(np.float32), chroma[..., 1].astype(np.float32)
    green = y - 0.356 * c1 + 0.114 * c2  # grijs = 0,299 R + 0,587 G + 0,114 B (OpenCV)
    red, blue = green + c1, green + 0.5 * c1 - c2
    rl, gl, bl = _linear(red, g), _linear(green, g), _linear(blue, g)
    return np.dstack([rl - gl, 0.5 * (rl + gl) - bl]).astype(np.float32), _linear(lum, g).astype(np.float32)


def _refine_boundary(fg: np.ndarray, o: np.ndarray, bgv: np.ndarray, valid: np.ndarray, tau: float,
                     color: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Herclassificeert pixels vlak bij de objectrand met de 50%-regel.

    Een lage drempel op een (vervaagde) rand legt de grens te ver naar buiten. Hier telt een
    randpixel als object als hij dichter bij de lokale objectgrijswaarde ligt dan bij de
    voorspelde mat: de grens ligt dan op het halve contrast, dus op de echte rand. Zonder bruikbaar
    grijscontrast beslist de kleur, met dezelfde regel (`color`: bruikbaar, object; zie _chroma_evidence).
    """
    fg8 = fg.astype(np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    band = (cv2.dilate(fg8, k5) > 0) & ~(cv2.erode(fg8, k5) > 0) & valid
    interior = cv2.erode(fg8, k5).astype(np.float32)
    num = cv2.boxFilter(o * interior, -1, (11, 11), normalize=False)
    den = cv2.boxFilter(interior, -1, (11, 11), normalize=False)
    obj = np.where(den > 0, num / np.maximum(den, 1e-6), 0.0)
    contrast = np.abs(obj - bgv)
    usable = band & (den > 0) & (contrast > 2 * tau)
    decision = np.abs(o - bgv) > 0.5 * contrast
    out = fg.copy()
    out[usable] = decision[usable]
    if color is not None:
        by_color = band & ~usable & color[0]
        out[by_color] = color[1][by_color]
        usable = usable | by_color
    # Randpixels zonder bruikbaar contrast (de voorspelde mat is daar toevallig even grijs als het
    # object, bijv. op een vervaagde stiprand) zeggen niets: die volgen de meerderheid van de
    # beslisbare pixels in een 5x5-omgeving, in plaats van standaard 'mat' te zijn.
    unsure = band & ~usable & (den > 0)
    if unsure.any():
        sure = (valid & ~unsure).astype(np.float32)
        n_fg = cv2.boxFilter(out.astype(np.float32) * sure, -1, (5, 5), normalize=False)
        n_sure = cv2.boxFilter(sure, -1, (5, 5), normalize=False)
        fill = unsure & (n_sure > 0)
        out[fill] = (n_fg > 0.5 * n_sure)[fill]
    return out


def _chroma_evidence(chroma: np.ndarray, lum: np.ndarray, ref: np.ndarray, valid: np.ndarray, radius: float,
                     min_area: float, k_sigma: float = 6.0, tau_min: float = 6.0):
    """Kleur als bewijs (V8): geeft (object, bruikbaar, fractie objectkleur, kleurverschil met de mat (2 kanalen),
    ruisvariantie van dat verschil per pixel) of None.

    `chroma`: R - G en (R + G)/2 - B per pixel (imgio.split_chroma, terug op volle resolutie), `lum` de
    grijswaarde, `ref` zekere mat. De mat is zwart-wit, maar het licht en de witbalans van de camera geven
    haar een kleurzweem die met de helderheid meeschaalt: per kanaal chroma ≈ k·grijs, met k lokaal gemeten op
    de mat (zo blijft een schaduw op de mat grijs: hij verlaagt beide evenveel). Wat daar boven uitsteekt is
    kleur van het object. De drempel volgt de ruis op de mat, plus kleurranden bij scherpe grijsovergangen
    (demosaicing, kleurschifting), evenredig met de gradiënt. Kleine stukjes tellen niet.

    Rond het gevonden object de 50%-regel op kleur, zoals voor grijs in _refine_boundary: een pixel is
    object als zijn kleur (langs die van het object in de buurt) meer dan half zo sterk is. Bruikbaar waar
    het object duidelijk gekleurd is (twee keer de drempel). Een grijs of zwart object geeft geen kleur, en
    dan verandert er niets."""
    if np.count_nonzero(ref) < 1000:
        return None
    ch = chroma.astype(np.float32)  # al vaag genoeg: halve resolutie (en in JPEG of HEIC zelf ook)
    lum = lum.astype(np.float32)
    w = ref.astype(np.float32)
    den = _normconv(lum * lum, w, 40.0, float(np.mean(lum[ref] ** 2)), min_weight=0.002)
    d = np.empty_like(ch)
    for c in range(2):
        k_all = float(np.sum(ch[..., c][ref] * lum[ref]) / max(np.sum(lum[ref] ** 2), 1e-6))
        k = _normconv(ch[..., c] * lum, w, 40.0, k_all * float(np.mean(lum[ref] ** 2)), min_weight=0.002) \
            / np.maximum(den, 1e-6)
        d[..., c] = ch[..., c] - k * lum
    mag = np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)
    sig = max(1.4826 * float(np.median(np.abs(d[..., c][ref][::5] - np.median(d[..., c][ref][::5]))))
              for c in range(2))
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = np.sqrt(gx * gx + gy * gy)
    steep = ref & (grad > 15.0)
    fringe = float(np.percentile(mag[steep] / grad[steep], 95)) if np.count_nonzero(steep) > 200 else 0.2
    thr = max(k_sigma * sig, tau_min) + fringe * grad
    # waar is kleur? (lage drempel: door de onscherpte van de kleur tot een paar pixels buiten de rand)
    seen = (valid & (cv2.GaussianBlur(mag, (0, 0), 1.0) > thr)).astype(np.uint8)
    seen = _drop_small(cv2.morphologyEx(seen, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0, min_area)
    inner = cv2.erode(seen.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if np.count_nonzero(inner) < 50:
        return None
    wi = inner.astype(np.float32)
    level = np.dstack([_normconv(d[..., c], wi, max(4.0, radius), 0.0) for c in range(2)])
    near = _normconv(wi, np.ones_like(wi), max(4.0, radius), 0.0) > 0.02
    strength = np.sqrt(level[..., 0] ** 2 + level[..., 1] ** 2)
    usable = valid & near & (strength > 2.0 * thr)
    frac = (d[..., 0] * level[..., 0] + d[..., 1] * level[..., 1]) / np.maximum(strength ** 2, 1e-6)
    # het object volgens de kleur: de 50%-regel (de rand op het halve kleurcontrast, dus op de echte rand)
    obj = _drop_small(usable & (frac > 0.5) & (seen > 0), min_area)
    return obj, usable, frac, d, sig * sig + (fringe * grad) ** 2


# Systematische onzekerheid (1σ) van de zachte objectfractie, bovenop ruis en posefout: verscherping en afkappen op
# zwart, en voor kleur ook de halve resolutie, verschuiven de halve-contrastrand een beetje (§3i). Zo beslist de kleur
# alleen waar grijs weinig zegt (blauw boven zwart), niet ook waar grijs een scherpe rand geeft.
ALPHA_SYS_GRAY, ALPHA_SYS_COLOR = 0.02, 0.05
# Toonkromme (_tone_exponent): alleen als de exponent minstens zoveel van 1 afwijkt en de residuen op de overgangen
# van de mat er minstens zoveel kleiner door worden. Verscherping geeft op lineaire foto's 0,92-1,02, een mat die
# niet vlak ligt tot 0,90 (§3j); een telefoon (sRGB) zit rond 0,45. Eén foto met een kromme die er niet is,
# veranderde in een stresstest de startcontour
TONE_MIN_DEV, TONE_MIN_GAIN = 0.15, 0.05
# extra vervaging (σ, px) van de voorspelling die _tone_exponent probeert; tot 3 px voor een bewogen of onscherpe foto
TONE_BLUR_PX = (0.0, 0.35, 0.7, 1.05, 1.5, 2.1, 3.0)
# voorkennis voor de versterking van de mat vlak buiten de rand (_soft_alpha): pas bij een patroon met meer dan
# ~5 grijswaarden spreiding telt de gemeten versterking, op een egaal vak blijft het een verschuiving
MAT_GAIN_PRIOR = 25.0
# Afgekapte grijswaarden (v0.11): waar de foto op 0 of 255 staat is het echte contrast onbekend, en dan ligt de
# halve-contrastrand ernaast verkeerd. Een stuk telt als afgekapt als minstens CLIP_FRAC van de pixels in een venster
# van 5 x 5 px op CLIP_LO of CLIP_HI staat (losse ruispixels niet); alpha uit grijs niet binnen CLIP_REACH px daarvan
# (de mat vlak buiten de rand wordt tot 7,5 px ver gemeten)
CLIP_LO, CLIP_HI, CLIP_FRAC, CLIP_REACH = 1, 254, 0.3, 8


def _soft_alpha(fg: np.ndarray, o: np.ndarray, bgv: np.ndarray, valid: np.ndarray, sigma: float,
                mis: np.ndarray, color=None, reach: int = 5, tone=1.0,
                clipped: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Zachte objectfractie rond de rand (V2, v0.9): per pixel welk deel ervan object is, uit de grijswaarde zelf.

    alpha = (foto − mat) / (object − mat), met de voorspelde mat `bgv` en de grijswaarde van het object vlak
    binnen de rand (3,5-7,5 px binnen het masker, σ 3 px: een wand in beeld heeft zijn eigen grijs), en de mat zoals
    gemeten vlak buiten de rand (3,5-7,5 px erbuiten, als correctie op de voorspelling). Een onscherpe
    rand gaat daarin geleidelijk van 1 naar 0, en ligt waar alpha 0,5 is: ook tussen twee pixels in, en zonder de
    strook zonder bewijs met een vaste fractie te verdelen (die fractie paste niet bij donkere en gekleurde
    onderdelen, ROUTE-A-VERBETERPUNTEN §3h). Variantie: ruis `sigma` en posefout `mis` (grijswaarden: posefout ×
    helling van de mat, groot aan patroonranden), gedeeld door het contrast². Met kleur (`color`: bruikbaar,
    kleurverschil met de mat, ruisvariantie; zie _chroma_evidence) ook de fractie objectkleur, met een eigen
    kleurniveau vlak binnen de rand, en beide gewogen naar hun variantie. Alleen binnen `reach` px van de
    maskerrand en waar het object in de buurt is. Geeft (alpha, gewicht = 1/variantie) als float16; NaN en 0
    zonder bewijs.

    `tone`: de exponent van de toonkromme van de camera (_tone_exponent). Alpha is een mengverhouding van licht,
    dus dan in lineair licht: L = T⁻¹(grijs). In de gecodeerde grijswaarden ligt de halve-contrastrand bij een
    sRGB-kromme 0,2-0,35 px naast de rand, afhankelijk van object en mat (§3j).

    `clipped`: waar de foto afgekapt is, met CLIP_REACH px eromheen (_clip_zone). Daar geen alpha uit grijs: met een
    afgekapte mat of een afgekapt object is het contrast te klein, en komt de rand te ver naar de andere kant (een
    wit vak naast een zwart onderdeel dat met lokaal contrast (HDR) boven 255 komt: gaten tot 0,3 mm te groot, §3k)."""
    fg8 = fg.astype(np.uint8)
    k = np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)
    band = (cv2.dilate(fg8, k) > 0) & ~(cv2.erode(fg8, k) > 0) & valid
    ring = (cv2.erode(fg8, np.ones((7, 7), np.uint8)) > 0) & ~(cv2.erode(fg8, np.ones((15, 15), np.uint8)) > 0)
    inner = cv2.erode(fg8, np.ones((5, 5), np.uint8)) > 0  # voor smalle stukken zonder ring

    def level(values, where):
        near = np.full(values.shape[:2], False)
        out = np.zeros_like(values, dtype=np.float32)
        for src in (ring & where, inner & where):
            wgt = src.astype(np.float32)
            den = cv2.GaussianBlur(wgt, (0, 0), 3.0)
            fill = ~near & (den > 0.02)
            if values.ndim == 2:
                out[fill] = (cv2.GaussianBlur(values * wgt, (0, 0), 3.0) / np.maximum(den, 1e-6))[fill]
            else:
                for c in range(values.shape[2]):
                    out[..., c][fill] = (cv2.GaussianBlur(values[..., c] * wgt, (0, 0), 3.0)
                                         / np.maximum(den, 1e-6))[fill]
            near |= fill
        return out, near

    lev, near = level(o.astype(np.float32), np.ones_like(fg))
    # De mat vlak buiten de rand (3,5-7,5 px) zoals gemeten: de voorspelde mat wijkt daar een paar grijswaarden af
    # (verscherping maakt het contrast van het matpatroon groter dan in de voorspelling, en de belichtingscorrectie
    # is grof), en dat schuift de halve-contrastrand bij een contrast van 100-200 al 0,05-0,1 px op (§3i). Daarom
    # daar per pixel een lineaire aanpassing foto ≈ a · voorspelling + b (gewogen, σ 3 px), met a naar 1 getrokken
    # waar de mat egaal is (dan alleen een verschuiving).
    outer = (cv2.dilate(fg8, np.ones((15, 15), np.uint8)) > 0) & ~(cv2.dilate(fg8, np.ones((7, 7), np.uint8)) > 0)
    b32, o32 = bgv.astype(np.float32), o.astype(np.float32)
    mat_like = outer & valid & (np.abs(o32 - b32) < np.maximum(4.0 * sigma, 0.25 * np.abs(lev - b32)))
    wm = mat_like.astype(np.float32)
    s0 = cv2.GaussianBlur(wm, (0, 0), 3.0)
    have = s0 > 0.02
    s0 = np.maximum(s0, 1e-6)
    m_b, m_o = cv2.GaussianBlur(wm * b32, (0, 0), 3.0) / s0, cv2.GaussianBlur(wm * o32, (0, 0), 3.0) / s0
    var_b = cv2.GaussianBlur(wm * b32 * b32, (0, 0), 3.0) / s0 - m_b * m_b
    cov = cv2.GaussianBlur(wm * o32 * b32, (0, 0), 3.0) / s0 - m_b * m_o
    gain = np.clip((cov + MAT_GAIN_PRIOR) / (np.maximum(var_b, 0.0) + MAT_GAIN_PRIOR), 0.8, 1.25)
    bgv = np.where(have, m_o + gain * (b32 - m_b), bgv)
    slope = 1.0
    if _curved(tone):
        slope = (_linear(o + 1.0, tone) - _linear(o - 1.0, tone)) / 2.0  # lineair licht per grijswaarde (variantie)
        o, bgv, lev = _linear(o, tone), _linear(bgv, tone), _linear(lev, tone)
    c = lev - bgv
    c = np.where(np.abs(c) < 1e-3, 1e-3, c)
    a = (o - bgv) / c
    var = (sigma * sigma + mis * mis) * slope * slope / (c * c) + ALPHA_SYS_GRAY ** 2
    ok = band & near & (var < 0.25)
    if clipped is not None:
        ok &= ~clipped
    if color is not None:
        usable, d, nvar = color
        lev_c, near_c = level(d, usable)
        strength2 = np.maximum(lev_c[..., 0] ** 2 + lev_c[..., 1] ** 2, 1e-6)
        frac = (d[..., 0] * lev_c[..., 0] + d[..., 1] * lev_c[..., 1]) / strength2
        var_c = nvar / strength2 + ALPHA_SYS_COLOR ** 2
        okc = band & near_c & usable & (var_c < 0.25)
        wg = np.where(ok, 1.0 / np.maximum(var, 1e-6), 0.0)
        wc = np.where(okc, 1.0 / np.maximum(var_c, 1e-6), 0.0)
        wsum = wg + wc
        a = np.where(wsum > 0, (wg * a + wc * frac) / np.maximum(wsum, 1e-12), a)
        var = np.where(wsum > 0, 1.0 / np.maximum(wsum, 1e-12), var)
        ok = ok | okc
    alpha = np.where(ok, np.clip(a, -1.0, 2.0), np.nan).astype(np.float16)
    w = np.where(ok, 1.0 / np.maximum(var, 1e-4), 0.0).astype(np.float16)
    return alpha, w


def _clip_zone(gray: np.ndarray) -> np.ndarray:
    """Waar de foto afgekapt is (zie CLIP_FRAC: minstens zoveel pixels in een venster van 5 x 5 op 0 of 255), met
    CLIP_REACH px eromheen."""
    at = ((gray <= CLIP_LO) | (gray >= CLIP_HI)).astype(np.float32)
    area = (cv2.blur(at, (5, 5)) >= CLIP_FRAC).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CLIP_REACH + 1, 2 * CLIP_REACH + 1))
    return cv2.dilate(area, k) > 0


def _drop_small(mask: np.ndarray, min_area: float) -> np.ndarray:
    """Alleen samenhangende stukken van minstens `min_area` pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def _object_level(fg: np.ndarray, o: np.ndarray, radius_px: float):
    """Lokale grijswaarde van het object (gemiddelde over het binnenste van het masker in de buurt).

    Geeft (grijswaarde, in de buurt), of None zonder object. Alleen in de buurt van het object (binnen
    ~2 x de straal van zijn binnenste) is de schatting iets waard: verder weg zou een algemene waarde,
    vervuild door een valse vlek object in een schaduw, de hele mat 'even donker als het object' maken.
    """
    inner = cv2.erode(fg.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if np.count_nonzero(inner) < 50:
        return None
    sigma = max(4.0, radius_px)
    wgt = inner.astype(np.float32)
    level = _normconv(o, wgt, sigma, float(np.median(o[inner])))
    near = _normconv(wgt, np.ones_like(wgt), sigma, 0.0) > 0.02
    return level, near


def _closure(fg: np.ndarray, radius_px: float) -> np.ndarray:
    """Morfologische sluiting met een schijf: overbrugt stroken en inhammen tot 2 x `radius_px` breed,
    maar groeit niet over een rechte of bolle rand heen."""
    r = max(int(round(radius_px)), 1)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, disk, borderType=cv2.BORDER_CONSTANT,
                            borderValue=0) > 0


def _fill_ambiguous(fg: np.ndarray, ambiguous: np.ndarray, evidence: np.ndarray, domain: np.ndarray,
                    o: np.ndarray, bgv: np.ndarray, level: np.ndarray,
                    min_contrast: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Vult dubbelzinnige stukken binnen het object op.

    Een zwart onderdeel op een zwart stuk mat zonder textuur (de stipvrije stroken van mat v2, de
    zwarte vlakken van een marker), of een wit onderdeel op een wit stuk, is daar per pixel
    onzichtbaar (`ambiguous`). Zonder opvulling krijgt het masker gaten die aan de mat vastzitten
    (z = 0), dus in elk bovenaanzicht op dezelfde plek: nepgaten in de startcontour en een te lage
    hoogte.

    Opgevuld wordt een samenhangend dubbelzinnig stuk als het binnen `domain` ligt (de sluiting van
    het object: die overbrugt een strook, maar groeit niet over een rechte buitenrand heen),
    daarbinnen nergens grenst aan bewijs voor mat (`evidence`), en als geheel niet duidelijk op de mat
    lijkt. Dat laatste is een toets op het gemiddelde van het stuk: per pixel is een slagschaduw op wit
    (door de schaduwcorrectie verwacht op ~45%) vaak binnen de ruis even grijs als een grijs object,
    maar over honderden pixels is het verschil duidelijk. Alleen als object en mat daar samen minstens
    `min_contrast` grijswaarden verschillen: zwart op zwart (~5) is ook als geheel niet te scheiden van
    een flauw belichtingsverloop, en wordt dus opgevuld. De mat net buiten een rechte rand zegt niets
    over een inham in die rand. Een echt gat laat stippen, randen of wit zien en blijft dus open; een
    gat boven een egaal stuk in precies de kleur van het object is in die foto onzichtbaar, en daar
    beslissen de andere foto's.

    Geeft (opgevuld masker, stukken die als geheel op de mat lijken).
    """
    none = np.zeros_like(fg)
    cand = (ambiguous & domain).astype(np.uint8)
    n, labels = cv2.connectedComponents(cand, connectivity=8)
    if n <= 1:
        return fg, none
    idx = labels.ravel()
    count = np.maximum(np.bincount(idx, minlength=n), 1)
    m_o, m_b, m_l = (np.bincount(idx, weights=x.ravel(), minlength=n) / count for x in (o, bgv, level))
    mat_like = (np.abs(m_l - m_b) > min_contrast) & (np.abs(m_o - m_b) < 0.5 * np.abs(m_o - m_l))
    mat_like[0] = False
    touched = cv2.dilate((evidence & domain).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    blocked = mat_like.copy()
    blocked[np.unique(labels[touched & (cand > 0)])] = True
    blocked[0] = True
    return fg | ~blocked[labels], mat_like[labels]


def _resolve_ambiguity(fg, specks, o, bgv, valid, res, mat_seen, tau: float, radius: float, window: int,
                       min_area: float, contrast: np.ndarray, color_obj=None, color_mat=None):
    """Zwart op zwart, wit op wit: geeft (objectmasker, dubbelzinnig, zekere-mat-toegestaan), of None.

    Waar de mat dezelfde grijswaarde heeft als het object ernaast, is een pixel zelf geen bewijs, en ook
    het textuurvenster niet vlak bij de rand. Een venster tot een halve vensterbreedte binnen het object
    ziet de stippen van de mat ernaast nog ('mat gezien'), en een venster tot een halve vensterbreedte
    buiten het object mist ze al ('textuur ontbreekt'). Daar telt alleen de kern van zo'n gebied: de rand
    ligt ertussen, en de fit bepaalt hem uit het bewijs rondom. De rest is dubbelzinnig: de fit negeert
    het, en binnen het object wordt het voor de startcontour opgevuld (_fill_ambiguous).

    Kleur (V8) beslist waar grijs het niet kan: een pixel in de kleur van het object is geen lek, en een pixel
    die duidelijk de kleur van de mat heeft (`color_mat`) is bewijs voor mat en wordt niet opgevuld.
    """
    found = _object_level(fg, o, radius)
    if found is None:
        return None
    level, near = found
    domain = _closure(fg, radius)
    # losse vlekjes object binnen de sluiting horen erbij (een witte stip onder een zwart onderdeel; vlak
    # naast een gat vaak het enige bewijs waar het object ophoudt); daarbuiten zijn ze ruis
    support = fg | (specks & domain)
    diff = np.abs(level - bgv)
    # 'even donker' binnen 2τ, maar nooit meer dan een kwart van het zwart-witcontrast van de mat: bij een
    # ruisige foto zou anders alles op elkaar lijken
    same = valid & near & (diff < np.minimum(2.0 * tau, 0.25 * contrast))
    kw = np.ones((window, window), np.uint8)
    mat_ok = mat_seen & (~same | (cv2.erode(mat_seen.astype(np.uint8), kw) > 0))
    leak = support & same & (res <= tau) & (cv2.dilate((valid & ~support).astype(np.uint8), kw) > 0)
    free = valid & ~support
    sure = free & mat_ok & (res <= tau)
    unseen = free & same & ~sure
    evidence = sure | (free & ~same & (np.abs(o - bgv) < 0.3 * diff))
    if color_obj is not None:
        leak &= ~color_obj
        unseen &= ~color_mat
        evidence |= free & color_mat
    filled, mat_like = _fill_ambiguous(support, unseen, evidence, domain, o, bgv, level)
    unseen &= ~mat_like  # als geheel duidelijk mat (schaduw, doorkijk): gewoon 'onbekend', geen opvulling
    # Wat binnen de sluiting nog open is, grenst aan matbewijs (bijv. een zwart vlak naast een echt gat):
    # elke pixel gaat naar het dichtstbijzijnde bewijs, object of mat, zodat de grens halverwege komt en
    # een gat rond blijft in plaats van de vorm van het zwarte vlak te krijgen.
    rest = unseen & domain & ~filled
    if rest.any():
        d_obj = cv2.distanceTransform((~(support & ~leak)).astype(np.uint8), cv2.DIST_L2, 3)
        d_mat = cv2.distanceTransform((~evidence).astype(np.uint8), cv2.DIST_L2, 3)
        filled = filled | (rest & (d_obj < d_mat))
    return _drop_small(filled, min_area) | (specks & domain), unseen | leak, mat_ok


def _black_lift(o: np.ndarray, p: np.ndarray, model: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Hoeveel lichter het zwart van de mat is dan het versterkingsmodel zegt (glans, strooilicht).

    Glans van een lamp op de toner maakt de zwarte vakken lichter en laat wit bijna gelijk; een lokale
    versterking kan dat niet beschrijven. Gemeten op zwarte pixels ruim binnen een vak (vlak bij een
    rand mengen onscherpte en posefout de kleuren) waarvan het venster het matpatroon herhaalt, fijn
    (8 px) waar genoeg bronnen zijn en grof (40 px) daartussen. Afwijkingen tot 3 grijswaarden zijn ruis.
    """
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    black = sources & (p < 40.0) & (np.sqrt(gx * gx + gy * gy) < 4.0)
    if np.count_nonzero(black) < 200:
        return np.zeros_like(o)
    r = o - model
    wgt = black.astype(np.float32)
    coarse = _normconv(r, wgt, 40.0, 0.0, min_weight=0.002)
    fine = _normconv(r, wgt, 8.0, 0.0, min_weight=0.0)
    density = _normconv(wgt, np.ones_like(wgt), 8.0, 0.0)
    alpha = np.clip(density / 0.05, 0.0, 1.0)
    lift = alpha * fine + (1.0 - alpha) * coarse
    return (np.sign(lift) * np.maximum(np.abs(lift) - 3.0, 0.0)).astype(np.float32)


def _local_stats(o: np.ndarray, p: np.ndarray, k: int):
    """Gemiddelden, varianties en covariantie van o en p in vensters van k x k pixels."""
    box = lambda x: cv2.boxFilter(x, cv2.CV_32F, (k, k))  # noqa: E731
    mo, mp = box(o), box(p)
    vo = np.maximum(box(o * o) - mo * mo, 0.0)
    vp = np.maximum(box(p * p) - mp * mp, 0.0)
    return mo, mp, vo, vp, box(o * p) - mo * mp


def _normconv(values: np.ndarray, weight: np.ndarray, sigma: float, fallback: float, min_weight: float = 0.05):
    """Genormaliseerde convolutie: gewogen gemiddelde van `values` in een Gauss-venster."""
    f = int(max(1, min(8, sigma // 4)))  # brede vensters op lagere resolutie: veel sneller, zelfde uitkomst
    h, w = values.shape
    vw, ww = (values * weight).astype(np.float32), weight.astype(np.float32)
    if f > 1:
        size = (max(w // f, 1), max(h // f, 1))
        vw, ww = cv2.resize(vw, size, interpolation=cv2.INTER_AREA), cv2.resize(ww, size, interpolation=cv2.INTER_AREA)
    num = cv2.GaussianBlur(vw, (0, 0), sigma / f)
    den = cv2.GaussianBlur(ww, (0, 0), sigma / f)
    out = np.where(den > min_weight, num / np.maximum(den, 1e-6), fallback).astype(np.float32)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR) if f > 1 else out


def classify(observed: np.ndarray, pred: np.ndarray, valid: np.ndarray, *, k_sigma: float = 6.0,
             tau_min: float = 14.0, texture_min: float = 10.0, min_area_frac: float = 2e-4,
             misreg_px: float = 0.4, ncc_mat: float = 0.75, window: int = 7, use_gain: bool = True,
             use_texture_missing: bool = True, px_per_mm: float = 4.0, fill_mm: float = 3.5,
             blur_px: float | None = None, chroma: np.ndarray | None = None, estimate_tone: bool = True,
             tone_params: Tone | None = None) -> ViewMasks:
    """Deelt een (ontvervormd) grijswaardenbeeld in: object, zekere mat, onbekend.

    Twee soorten bewijs:
    * *grijswaarde*: wijkt de pixel af van de voorspelde mat (na belichtingscorrectie)?
    * *textuur*: herhaalt het venster rond de pixel het voorspelde matpatroon (lokale correlatie)?

    Alleen pixels waarvan het venster het patroon herhaalt, zijn *zekere* mat. Een donker object
    op een zwart vak lijkt per pixel op de mat, maar mist de stippen en randen eromheen: dat is
    geen zekere mat meer (en waar textuur verwacht wordt maar ontbreekt, is het object). De
    correlatie is ongevoelig voor versterking: een schaduw op de mat blijft mat, en de lokale
    versterking die daaruit volgt, corrigeert ook de egale vlakken in de schaduw.

    Echte foto's wijken op nog drie manieren af van de voorspelde mat, en daar past het masker zich
    per foto op aan: onscherpte (`blur_px`, gemeten in preflight.py; de voorspelling wordt even
    onscherp gemaakt), een kleine posefout (de tolerantie aan patroonranden wordt gemeten aan de mat
    zelf, in plaats van vast 0,4 px) en glans (zwart dat lichter is dan de versterking zegt).

    `chroma` (V8): de kleur van de foto (imgio.split_chroma, op volle resolutie en ontvervormd), of een
    kleurbeeld als `observed`. Kleur is dan een derde soort bewijs (_chroma_evidence): voor het object, voor
    de rand waar grijs geen contrast heeft, en direct buiten de buitenrand ook voor de mat.
    """
    if observed.ndim == 3:
        if chroma is None:
            from .imgio import split_chroma
            _, half = split_chroma(observed)
            if half is not None:
                chroma = cv2.resize(half.astype(np.float32), (observed.shape[1], observed.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
        observed = cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY)
    # Hoe de camera het licht vastlegt (tone.py, v0.11). Voor de maskers blijft de foto zoals hij is, en gaat de
    # voorspelde mat vooruit door hetzelfde: even vaag, de toonkromme, de verscherping en het afkappen op 0-255. Zo
    # blijven ook de kleine stippen in de zwarte vakken (het bewijs naast een zwart onderdeel) even fel als in de foto;
    # met de foto ontscherpt waren ze dat niet, en dan gingen zwarte onderdelen mis (§3k). Voor alpha, de kleur en de
    # grijswaarden (lineair licht) is de verscherping wel ongedaan gemaakt (`lum_ds`, `p_ds`). Zonder `tone_params`
    # (losse aanroep): alleen de kromme, geschat op het ontvervormde beeld, en de onscherpte uit `blur_px` (v0.10).
    tp = tone_params
    o = cv2.GaussianBlur(observed.astype(np.float32), (0, 0), 0.8)
    lum = o.copy()  # grijswaarde zonder belichtingscorrectie: de kleurzweem van de mat schaalt daarmee
    p = pred.astype(np.float32)
    if tp is not None:
        if tp.blur > 0:
            p = cv2.GaussianBlur(p, (0, 0), tp.blur)
    else:
        if blur_px is not None and blur_px > 1.0:
            # een bewogen of onscherpe foto: de voorspelling (zelf ~0,5 px vaag) even vaag maken, anders geeft
            # elke zwart-witrand van de mat aan weerszijden een afwijking die op object lijkt
            p = cv2.GaussianBlur(p, (0, 0), float(np.sqrt(blur_px ** 2 - 1.0)))
        tp = Tone(g=_tone_exponent(observed, p, valid)[0] if estimate_tone else 1.0)
    sharp = tp.sharpen > 0
    lum_ds = cv2.GaussianBlur(desharpen(observed, tp.sharpen, tp.sharpen_px), (0, 0), 0.8) if sharp else lum
    tone = tp.g_map(o.shape)
    p_ds = p
    if _curved(tone) or sharp:
        # lineair licht la · p + lb (belichting, en het licht dat zwart terugkaatst), dan de kromme en de verscherping;
        # terug naar 0-255 van zwart tot wit, zodat p / 255 de fractie wit blijft (_black_lift, de helling hieronder)
        la, lb, _ = _robust_affine(_linear(lum_ds, tone) if _curved(tone) else lum_ds, p, valid)
        if la > 0:
            def encode(v):
                return 255.0 * np.clip(v / 255.0, 0.0, 1.0) ** tone if _curved(tone) else v

            E = encode(la * p + lb)
            lo_t, hi_t = encode(np.float32(lb)), encode(np.float32(la * 255.0 + lb))
            if np.min(hi_t - lo_t) > 10.0:
                p_ds = (255.0 * (E - lo_t) / (hi_t - lo_t)).astype(np.float32)
                if sharp:
                    E = np.clip(E + tp.sharpen * (E - cv2.GaussianBlur(E, (0, 0), tp.sharpen_px)), 0.0, 255.0)
                p = (255.0 * (E - lo_t) / (hi_t - lo_t)).astype(np.float32)
            else:
                tone = 1.0
        else:
            tone = 1.0
        if not _curved(tone) and tp.g != 1.0:
            tp = Tone(g=1.0, blur=tp.blur, sharpen=tp.sharpen, sharpen_px=tp.sharpen_px)
    # zoals de foto: vervaagd in gecodeerde grijswaarden, dus na de kromme
    p = cv2.GaussianBlur(p, (0, 0), 0.8)
    p_ds = cv2.GaussianBlur(p_ds, (0, 0), 0.8) if p_ds is not p else p
    a, b, sigma = _robust_affine(o, p, valid)

    # traag verlopende belichtingsverschillen wegwerken (genormaliseerde convolutie over de mat)
    r = o - (a * p + b)
    w = (valid & (np.abs(r) < max(4 * sigma, 8.0))).astype(np.float32)
    corr = _normconv(r, w, 35.0, 0.0)
    o = o - corr

    # textuurbewijs: lokale correlatie tussen foto en voorspelling
    mo, mp, vo, vp, cov = _local_stats(o, p, window)
    ncc = cov / np.sqrt(vo * vp + 1e-6)
    s_pred = np.sqrt(vp) * abs(a)  # verwacht lokaal contrast (grijswaarden van de foto)
    s_obs = np.sqrt(vo)
    textured = valid & (s_pred > texture_min)
    mat_seen = textured & (ncc > ncc_mat) & (s_obs > 0.25 * s_pred) & (s_obs < 2.5 * s_pred)
    # Bron voor de lokale versterking: alleen vensters waar niveau en contrast dezelfde versterking
    # geven, zoals bij mat in schaduw. Een objectrand die toevallig met het patroon correleert (bijv. de
    # rand van een gat langs een vakrand) geeft twee verschillende 'versterkingen' en zou anders het
    # object in de buurt als beschaduwde mat laten doorgaan. (Glans maakt zwart lichter en laat wit
    # bijna gelijk: dat is geen versterking, en daarvoor is er de zwart-optilling hieronder.)
    g_level = mo / np.maximum(a * mp + b, 1.0)
    g_contrast = s_obs / np.maximum(s_pred, 1e-3)
    gain_src = mat_seen & (np.abs(np.log(np.maximum(g_level, 1e-3) / np.maximum(g_contrast, 1e-3))) < 0.25)
    texture_missing = textured & (s_pred > 1.5 * texture_min) & (ncc < 0.3) & (s_obs < 0.35 * s_pred)
    if not use_texture_missing:
        texture_missing = np.zeros_like(textured)

    # lokale versterking (schaduw): verhouding van lokale sommen van foto en voorspelling, alleen over
    # pixels waarvan het venster het matpatroon herhaalt (objectpixels tellen dus niet mee). Tweede ronde
    # zonder pixels die niet bij die versterking passen (bijv. een objectrand met toevallig hoge
    # correlatie). Kleine afwijkingen (< 3%) zijn ruis en worden 1.
    den_img = a * p + b
    gain = np.ones_like(o)
    lift = np.zeros_like(o)
    if use_gain:
        # Niet vlak naast bewijs voor het object: textuur die verwacht wordt maar ontbreekt (binnen een
        # halve vensterbreedte plus één pixel). Anders kan een objectrand die samenvalt met een vakrand
        # (grijs object naast het donkere gat, op de plek van een zwart-witovergang) als 'mat in
        # schaduw' de versterking omlaag trekken, en valt het object ernaast weg als beschaduwd wit.
        # Niet ruimer: een slagschaduw ligt direct naast het object en heeft zijn bronnen juist daar.
        # (Een afwijking op een egaal stuk mat telt niet als bewijs: dat kan net zo goed schaduw zijn.)
        near_obj = cv2.dilate(texture_missing.astype(np.uint8), np.ones((window + 2, window + 2), np.uint8)) > 0
        gain_src &= ~near_obj
        src = gain_src.astype(np.float32)
        # Waar in de buurt geen bronnen zijn (de witte rand van het papier, midden in een groot vak), het
        # grove verloop volgen in plaats van 'geen schaduw': een schaduw van hand of telefoon valt ook
        # over de rand. Niet binnen ~10 mm van ontbrekende textuur: daar ontbreken de bronnen juist
        # door het object, en een doorgetrokken schaduw zou stukken object als beschaduwde mat laten
        # doorgaan (gaatjes midden in het object).
        reach = int(10.0 * px_per_mm)
        obj_zone = cv2.dilate(texture_missing.astype(np.uint8), np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)) > 0
        for _ in range(2):
            num = cv2.GaussianBlur(o * src, (0, 0), 6.0)
            den = cv2.GaussianBlur(den_img * src, (0, 0), 6.0)
            wsum = cv2.GaussianBlur(src, (0, 0), 6.0)
            coarse = np.clip(_normconv(o, src, 40.0, 1.0, min_weight=0.002)
                             / np.maximum(_normconv(den_img, src, 40.0, 1.0, min_weight=0.002), 1e-3), 0.2, 2.5)
            coarse = np.where(obj_zone, 1.0, coarse)
            gain = np.where(wsum > 0.05, np.clip(num / np.maximum(den, 1e-3), 0.2, 2.5), coarse).astype(np.float32)
            fit = np.abs(o - gain * den_img) < np.maximum(3.0 * sigma, 0.08 * gain * den_img)
            src = (gain_src & fit).astype(np.float32)
        dev = gain - 1.0
        gain = (1.0 + np.sign(dev) * np.maximum(np.abs(dev) - 0.03, 0.0)).astype(np.float32)
        lift = _black_lift(o, p, gain * den_img, mat_seen & ~near_obj)
    q = p / 255.0
    bgv = gain * den_img + lift * (1.0 - q)
    contrast = np.maximum(gain * abs(a) * 255.0 - lift, 1.0)  # lokaal verschil tussen wit en zwart van de mat

    # tolerantie voor een kleine posefout: evenredig met de lokale gradiënt van de voorspelling
    # (een vast 3x3-min/max-venster is te ruim: objectpixels op patroonranden zouden dan 'mat' lijken)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gmag = contrast / 255.0 * np.sqrt(gx * gx + gy * gy)  # helling van de voorspelde mat (grijswaarden/pixel)
    # Posefout per foto, gemeten aan de mat zelf: de afwijking aan duidelijke patroonranden gedeeld door
    # de helling daar is de verschuiving in pixels. Een synthetische scan blijft op 0,4 px; bij een echte
    # foto (onscherpe kalibratiefoto's, een niet helemaal vlakke mat) is het vaak 0,5-1,5 px.
    misreg = misreg_px
    strong = mat_seen & (gmag > 20.0)
    if np.count_nonzero(strong) > 500:
        misreg = float(np.clip(np.percentile(np.abs(o - bgv)[strong] / gmag[strong], 75), misreg_px, 2.5))
    res = np.maximum(np.abs(o - bgv) - misreg * gmag, 0.0)
    # ruis over alle zekere mat, randen inbegrepen: alleen de vlakke stukken geeft een lagere drempel,
    # en dan telt de rand van een slagschaduw (die de versterking niet helemaal volgt) als object
    sel = valid & mat_seen if mat_seen.sum() > 1000 else valid
    sigma = 1.4826 * float(np.median(np.abs((o - bgv)[sel][::7]))) + 1e-3
    tau = max(k_sigma * sigma, tau_min)
    k3 = np.ones((3, 3), np.uint8)
    min_area = min_area_frac * observed.size
    radius = fill_mm * px_per_mm
    color = None  # (object, bruikbaar, fractie objectkleur): kleur als bewijs (V8)
    if chroma is not None:
        lum_c = lum
        if _curved(tone):  # kleur in lineair licht (V2, v0.10), met de ontscherpte grijswaarde (v0.11)
            chroma, lum_c = _linear_color(chroma, lum_ds, tone)
        color = _chroma_evidence(chroma, lum_c, mat_seen & valid, valid, radius, min_area)
    fg = valid & ((res > tau) | texture_missing)
    if color is not None:
        fg |= color[0]
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, k3)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k3) > 0
    large = _drop_small(fg, min_area)
    fg, specks = large, fg & ~large
    color_obj = color_mat = None
    if color is not None:
        color_obj, color_mat = color[1] & (color[2] > 0.5), color[1] & (color[2] < 0.3)
    fg = _refine_boundary(fg, o, bgv, valid, tau, None if color is None else (color[1], color_obj))
    mat_ok, amb = mat_seen, None
    if fill_mm > 0 and fg.any():
        # alleen rond het object: daarbuiten verandert niets, en zo kost het weinig rekentijd
        x, y, w, h = cv2.boundingRect(fg.astype(np.uint8))
        m = int(radius) + window + 2
        sl = (slice(max(y - m, 0), y + h + m), slice(max(x - m, 0), x + w + m))
        out = _resolve_ambiguity(fg[sl], specks[sl], o[sl], bgv[sl], valid[sl], res[sl], mat_seen[sl], tau,
                                 radius, window, min_area, contrast[sl],
                                 None if color is None else color_obj[sl], None if color is None else color_mat[sl])
        if out is not None:
            fg, amb, mat_ok = np.zeros_like(fg), np.zeros_like(fg), mat_seen.copy()
            fg[sl], amb[sl], mat_ok[sl] = out

    # Zekere mat voor het uitsnijden (hull): het eigen venster herhaalt het matpatroon. Voor de fit
    # ook de pixels tot aan de objectrand: hun venster overlapt het object, maar een venster er vlak
    # naast (binnen de vensterstraal) is wel geverifieerd. Zonder die randstrook zou het model
    # goedkoop over de rand kunnen groeien ('onbekend' kost minder dan 'mat').
    # Alleen pixels die duidelijk op de mat lijken en niet op het object ernaast: bij een fijn
    # matpatroon heeft de voorspelde mat vlak naast de rand vaak tussenwaarden (vervaagde stippen),
    # en dan lijkt ook een objectpixel in de eerste ringen op 'mat'. Zulke twijfelpixels blijven
    # 'onbekend' in plaats van de fit naar binnen te duwen.
    match = valid & ~fg & (res <= tau)
    mean15 = cv2.blur(p, (15, 15))
    texture15 = np.sqrt(np.maximum(cv2.blur(p * p, (15, 15)) - mean15 * mean15, 0.0)) * (contrast / 255.0)
    near_seen = cv2.dilate(mat_seen.astype(np.uint8), np.ones((window, window), np.uint8)) > 0
    inner = cv2.erode(fg.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    obj_den = cv2.boxFilter(inner, -1, (11, 11), normalize=False)
    obj = cv2.boxFilter(o * inner, -1, (11, 11), normalize=False) / np.maximum(obj_den, 1e-6)
    clearly_mat = (obj_den <= 0) | (np.abs(o - bgv) < 0.3 * np.abs(obj - bgv))
    edge_bg = match & near_seen & (texture15 > 1.8 * texture_min) & clearly_mat
    if amb is not None:  # waar het object onzichtbaar zou zijn, zegt 'lijkt op de mat' niets
        edge_bg &= ~amb
    if color is not None:
        # Kleur (V8): een pixel direct buiten de buitenrand met duidelijk de kleur van de mat is zekere mat voor
        # de fit, ook waar grijs en textuur niets zeggen (een blauw onderdeel naast een zwart vak). Niet
        # binnen een ingesloten gebied: glans op het object is ook kleurloos, en een gat laat de mat alleen
        # zien als grijs of textuur dat bevestigen.
        n, lab = cv2.connectedComponents((~fg).astype(np.uint8), connectivity=4)
        edge = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        outside = np.isin(lab, np.unique(edge)) & ~fg
        ring = (cv2.dilate(fg.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & outside
        edge_bg |= match & color_mat & ring
    near_fg = cv2.dilate(fg.astype(np.uint8), k3) > 0
    out = ViewMasks(fg=fg, bg=match & mat_ok & ~near_fg, valid=valid, edge_bg=edge_bg, sigma=sigma, amb=amb,
                    tone=float(np.median(tone)) if _curved(tone) else 1.0, tone_params=tp)
    if fg.any():  # zachte objectfractie rond het object (V2), als uitsnede: dat scheelt geheugen
        x, y, w, h = cv2.boundingRect(fg.astype(np.uint8))
        y0, x0 = max(y - 12, 0), max(x - 12, 0)
        sl = (slice(y0, y + h + 12), slice(x0, x + w + 12))
        # de foto zelf, ontscherpt en zonder de belichtingscorrectie (een verschuiving in gecodeerde grijswaarden, die
        # vóór de toonkromme de mengverhoudingen zou veranderen), en de voorspelde mat daarbij (ook zonder verscherping)
        bgv_ds = bgv if p_ds is p else gain * (a * p_ds + b) + lift * (1.0 - q)
        raw, bg_raw = lum_ds[sl], bgv_ds[sl] + corr[sl]
        tone_sl = tone[sl] if not np.isscalar(tone) else tone
        clip = _clip_zone(observed[sl])
        out.alpha, out.alpha_w = _soft_alpha(fg[sl], raw, bg_raw, valid[sl], sigma, misreg * gmag[sl],
                                             None if color is None else (color[1][sl], color[3][sl], color[4][sl]),
                                             tone=tone_sl, clipped=clip)
        out.alpha_at = (y0, x0)
        out.clip = clip
        out.gray = raw.astype(np.float16)
        if not np.isscalar(tone_sl):
            out.tone_map = tone_sl.astype(np.float16)
    return out
