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
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calib import Pose
from .mat import BoardRaster


@dataclass
class ViewMasks:
    fg: np.ndarray  # objectpixels
    bg: np.ndarray  # zekere mat, met een veiligheidsmarge rond het object (voor het uitsnijden)
    valid: np.ndarray  # pixel valt op de mat
    edge_bg: np.ndarray | None = None  # zekere mat zonder marge (voor de modelverfijning)
    sigma: float = 0.0  # ruisniveau van het residu (grijswaarden)
    amb: np.ndarray | None = None  # object zou hier onzichtbaar zijn (zelfde grijs als de mat): geen bewijs


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


def _refine_boundary(fg: np.ndarray, o: np.ndarray, bgv: np.ndarray, valid: np.ndarray, tau: float) -> np.ndarray:
    """Herclassificeert pixels vlak bij de objectrand met de 50%-regel.

    Een lage drempel op een (vervaagde) rand legt de grens te ver naar buiten. Hier telt een
    randpixel als object als hij dichter bij de lokale objectgrijswaarde ligt dan bij de
    voorspelde mat: de grens ligt dan op het halve contrast, dus op de echte rand.
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
                       min_area: float, contrast: np.ndarray):
    """Zwart op zwart, wit op wit: geeft (objectmasker, dubbelzinnig, zekere-mat-toegestaan), of None.

    Waar de mat dezelfde grijswaarde heeft als het object ernaast, is een pixel zelf geen bewijs, en ook
    het textuurvenster niet vlak bij de rand. Een venster tot een halve vensterbreedte binnen het object
    ziet de stippen van de mat ernaast nog ('mat gezien'), en een venster tot een halve vensterbreedte
    buiten het object mist ze al ('textuur ontbreekt'). Daar telt alleen de kern van zo'n gebied: de rand
    ligt ertussen, en de fit bepaalt hem uit het bewijs rondom. De rest is dubbelzinnig: de fit negeert
    het, en binnen het object wordt het voor de startcontour opgevuld (_fill_ambiguous).
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
             blur_px: float | None = None) -> ViewMasks:
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
    """
    if observed.ndim == 3:
        observed = cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY)
    o = cv2.GaussianBlur(observed.astype(np.float32), (0, 0), 0.8)
    p = pred.astype(np.float32)
    if blur_px is not None and blur_px > 1.0:
        # een bewogen of onscherpe foto: de voorspelling (zelf ~0,5 px vaag) even vaag maken, anders geeft
        # elke zwart-witrand van de mat aan weerszijden een afwijking die op object lijkt
        p = cv2.GaussianBlur(p, (0, 0), float(np.sqrt(blur_px ** 2 - 1.0)))
    p = cv2.GaussianBlur(p, (0, 0), 0.8)
    a, b, sigma = _robust_affine(o, p, valid)

    # traag verlopende belichtingsverschillen wegwerken (genormaliseerde convolutie over de mat)
    r = o - (a * p + b)
    w = (valid & (np.abs(r) < max(4 * sigma, 8.0))).astype(np.float32)
    o = o - _normconv(r, w, 35.0, 0.0)

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
    fg = (valid & ((res > tau) | texture_missing)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k3) > 0
    large = _drop_small(fg, min_area)
    fg, specks = large, fg & ~large
    fg = _refine_boundary(fg, o, bgv, valid, tau)
    mat_ok, amb = mat_seen, None
    radius = fill_mm * px_per_mm
    if fill_mm > 0 and fg.any():
        # alleen rond het object: daarbuiten verandert niets, en zo kost het weinig rekentijd
        x, y, w, h = cv2.boundingRect(fg.astype(np.uint8))
        m = int(radius) + window + 2
        sl = (slice(max(y - m, 0), y + h + m), slice(max(x - m, 0), x + w + m))
        out = _resolve_ambiguity(fg[sl], specks[sl], o[sl], bgv[sl], valid[sl], res[sl], mat_seen[sl], tau,
                                 radius, window, min_area, contrast[sl])
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
    near_fg = cv2.dilate(fg.astype(np.uint8), k3) > 0
    return ViewMasks(fg=fg, bg=match & mat_ok & ~near_fg, valid=valid, edge_bg=edge_bg, sigma=sigma, amb=amb)
