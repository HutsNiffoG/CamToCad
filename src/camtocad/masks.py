"""Objectmaskers via de bekende mat-achtergrond.

Met de camerapose is precies te voorspellen hoe de mat er in elke foto uitziet. Pixels die
daarvan afwijken horen bij het object. Drie klassen per pixel:

* `fg`    – objectpixel (wijkt af van de voorspelde mat);
* `bg`    – *zeker* mat: klopt met de voorspelling én de voorspelling heeft daar textuur;
* overig  – onbekend (bijv. egale zwarte vakken, of buiten de mat).

Alleen zekere mat-pixels mogen voxels wegsnijden (hull.py). Zo veroorzaakt een donker
object op een zwart vak geen gat in het model: daar is het simpelweg 'onbekend'.
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
             use_texture_missing: bool = True) -> ViewMasks:
    """Deelt een (ontvervormd) grijswaardenbeeld in: object, zekere mat, onbekend.

    Twee soorten bewijs:
    * *grijswaarde*: wijkt de pixel af van de voorspelde mat (na belichtingscorrectie)?
    * *textuur*: herhaalt het venster rond de pixel het voorspelde matpatroon (lokale correlatie)?

    Alleen pixels waarvan het venster het patroon herhaalt, zijn *zekere* mat. Een donker object
    op een zwart vak lijkt per pixel op de mat, maar mist de stippen en randen eromheen: dat is
    geen zekere mat meer (en waar textuur verwacht wordt maar ontbreekt, is het object). De
    correlatie is ongevoelig voor versterking: een schaduw op de mat blijft mat, en de lokale
    versterking die daaruit volgt, corrigeert ook de egale vlakken in de schaduw.
    """
    if observed.ndim == 3:
        observed = cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY)
    o = cv2.GaussianBlur(observed.astype(np.float32), (0, 0), 0.8)
    p = cv2.GaussianBlur(pred.astype(np.float32), (0, 0), 0.8)
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
    # geven, zoals bij mat in schaduw of glans. Een objectrand die toevallig met het patroon
    # correleert (bijv. de rand van een gat langs een vakrand) geeft twee verschillende 'versterkingen'
    # en zou anders het object in de buurt als beschaduwde mat laten doorgaan.
    g_level = mo / np.maximum(a * mp + b, 1.0)
    g_contrast = s_obs / np.maximum(s_pred, 1e-3)
    gain_src = mat_seen & (np.abs(np.log(np.maximum(g_level, 1e-3) / np.maximum(g_contrast, 1e-3))) < 0.25)

    # tolerantie voor een kleine posefout: evenredig met de lokale gradiënt van de voorspelling
    # (een vast 3x3-min/max-venster is te ruim: objectpixels op patroonranden zouden dan 'mat' lijken)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = misreg_px * abs(a) * np.sqrt(gx * gx + gy * gy)
    texture_missing = textured & (s_pred > 1.5 * texture_min) & (ncc < 0.3) & (s_obs < 0.35 * s_pred)
    if not use_texture_missing:
        texture_missing = np.zeros_like(textured)
    k3 = np.ones((3, 3), np.uint8)
    min_area = min_area_frac * observed.size

    def foreground(gain):
        bgv = gain * (a * p + b)
        res = np.maximum(np.abs(o - bgv) - gain * grad, 0.0)
        sel = valid & mat_seen if mat_seen.sum() > 1000 else valid
        sig = 1.4826 * float(np.median(np.abs((o - bgv)[sel][::7]))) + 1e-3
        tau = max(k_sigma * sig, tau_min)
        fg = (valid & ((res > tau) | texture_missing)).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k3)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        return keep[labels], bgv, res, tau, sig

    # lokale versterking (schaduw, glans): verhouding van lokale sommen van foto en voorspelling,
    # alleen over pixels waarvan het venster het matpatroon herhaalt (objectpixels tellen dus niet
    # mee). Tweede ronde zonder pixels die niet bij die versterking passen (bijv. een objectrand
    # met toevallig hoge correlatie). Kleine afwijkingen (< 3%) zijn ruis en worden 1.
    den_img = a * p + b
    gain = np.ones_like(o)
    if use_gain:
        # Niet vlak naast bewijs voor het object: textuur die verwacht wordt maar ontbreekt (binnen een
        # halve vensterbreedte plus één pixel). Anders kan een objectrand die samenvalt met een vakrand
        # (grijs object naast het donkere gat, op de plek van een zwart-witovergang) als 'mat in
        # schaduw' de versterking omlaag trekken, en valt het object ernaast weg als beschaduwd wit.
        # Niet ruimer: een slagschaduw ligt direct naast het object en heeft zijn bronnen juist daar.
        # (Een afwijking op een egaal stuk mat telt niet als bewijs: dat kan net zo goed schaduw zijn.)
        evidence = texture_missing.astype(np.uint8)
        near_obj = cv2.dilate(evidence, np.ones((window + 2, window + 2), np.uint8)) > 0
        gain_src &= ~near_obj
        src = gain_src.astype(np.float32)
        for _ in range(2):
            num = cv2.GaussianBlur(o * src, (0, 0), 6.0)
            den = cv2.GaussianBlur(den_img * src, (0, 0), 6.0)
            wsum = cv2.GaussianBlur(src, (0, 0), 6.0)
            gain = np.where(wsum > 0.05, np.clip(num / np.maximum(den, 1e-3), 0.2, 2.5), 1.0).astype(np.float32)
            fit = np.abs(o - gain * den_img) < np.maximum(3.0 * sigma, 0.08 * gain * den_img)
            src = (gain_src & fit).astype(np.float32)
        dev = gain - 1.0
        gain = (1.0 + np.sign(dev) * np.maximum(np.abs(dev) - 0.03, 0.0)).astype(np.float32)
    fg, bgv, res, tau, sigma = foreground(gain)
    fg = _refine_boundary(fg, o, bgv, valid, tau)

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
    texture15 = np.sqrt(np.maximum(cv2.blur(p * p, (15, 15)) - mean15 * mean15, 0.0)) * abs(a)
    near_seen = cv2.dilate(mat_seen.astype(np.uint8), np.ones((window, window), np.uint8)) > 0
    inner = cv2.erode(fg.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    obj_den = cv2.boxFilter(inner, -1, (11, 11), normalize=False)
    obj = cv2.boxFilter(o * inner, -1, (11, 11), normalize=False) / np.maximum(obj_den, 1e-6)
    clearly_mat = (obj_den <= 0) | (np.abs(o - bgv) < 0.3 * np.abs(obj - bgv))
    edge_bg = match & near_seen & (texture15 > 1.8 * texture_min) & clearly_mat
    near_fg = cv2.dilate(fg.astype(np.uint8), k3) > 0
    return ViewMasks(fg=fg, bg=match & mat_seen & ~near_fg, valid=valid, edge_bg=edge_bg, sigma=sigma)
