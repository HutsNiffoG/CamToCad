"""Route A-pipeline: foto's van een onderdeel op de kalibratiemat → STEP, CadQuery-script en meetrapport.

Stappen: mat herkennen → camera zelf kalibreren en poses bepalen → objectmaskers → grove visual
hull (lokaliseren) → startmodel uit bovenaanzichten → hoogte zoeken → model fitten op alle
silhouetten → werkassenstelsel en snappen → CAD-model, export en rapport.

Objectklasse v0.1: 2,5D-onderdelen die plat op de mat liggen (extrusie met rechte of afgeronde
contour, of een cirkel), met doorgaande gaten.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from . import (__version__, blindhole, cadmodel, calib, counterbore, countersink, debug, edgefit, holes, hull, initial,
               masks, placement, preflight, prismcheck, profile, report, silhouette, tone, uncertainty)
from .cadhelpers import afgeronde_hoeken
from .imgio import IMAGE_EXT, PhotoInfo, heif_supported, imwrite, read_color, read_gray, read_info, split_chroma
from .mat import MatSpec, get_spec, rasterize_board
from .profile import Hole, Slot, dominant_angle
WORKERS = max(1, min(4, os.cpu_count() or 1))  # parallelle foto's (geheugen: ~250 MB per maskerberekening)
HDR_WARN_FRAC = 0.25  # een waarschuwing over lokale toonbewerking als minstens dit deel van de foto's een tegelraster heeft


class ScanError(ValueError):
    """Fout met een voor de gebruiker begrijpelijke uitleg."""


@dataclass
class ScanOptions:
    mat: str = "auto"  # "auto": herkend aan de markers; of een matnaam (A4, A3, Letter, A4-v1, A3-v1)
    max_side: int = 2000  # werkresolutie: langste zijde in pixels
    snap_threshold: float = 0.8
    imperial: bool = False
    max_evals: int = 1500
    debug_images: bool = True
    # printschaal: gemeten / nominale lengte van de meetlijnen; één getal (beide richtingen) of (X, Y).
    # None: niet gemeten (dan 1,0, maar met de onzekerheid van een onbekende printschaal in U95)
    mat_scale: float | tuple[float, float] | None = None

    @property
    def scale_xy(self) -> tuple[float, float]:
        s = self.mat_scale
        if s is None:
            return 1.0, 1.0
        if isinstance(s, (int, float)):
            return float(s), float(s)
        sx, sy = s
        return float(sx), float(sy)

    @property
    def scale_measured(self) -> bool:
        return self.mat_scale is not None


HEIC_HELP = ("HEIC-foto's (iPhone) zijn alleen te lezen met de extra 'heic': pip install \"camtocad[heic]\". "
             "Of zet de foto's om naar JPG, of stel de iPhone in op 'Meest compatibel' (Instellingen > Camera > "
             "Formaten).")


def load_images(folder: str | Path, max_side: int = 2000, log=print, infos: dict[str, PhotoInfo] | None = None,
                chroma: dict[str, np.ndarray] | None = None) -> list[tuple[str, np.ndarray]]:
    """Leest alle foto's als grijswaarden, zonder EXIF-rotatie (één consistent sensorformaat). `infos` krijgt
    per foto de camera, lens en zoom uit de EXIF-gegevens (V10); `chroma` de kleur op halve resolutie, voor
    foto's in kleur (V8, imgio.split_chroma)."""
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXT)
    heic = [p for p in paths if p.suffix.lower() in {".heic", ".heif"}]
    if heic and not heif_supported():
        log(f"  {len(heic)} HEIC-foto('s) overgeslagen: " + HEIC_HELP)
        if len(heic) == len(paths):
            raise ScanError("Alle foto's zijn HEIC en kunnen niet gelezen worden. " + HEIC_HELP)
    out = []
    for p in paths:
        img = read_gray(p) if chroma is None else read_color(p)  # ook met niet-ASCII-tekens in het pad (Windows)
        if img is None:
            if p not in heic or heif_supported():
                log(f"  overgeslagen (onleesbaar): {p.name}")
            continue
        scale = max_side / max(img.shape[:2])
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if chroma is not None:
            img, ch = split_chroma(img)
            if ch is not None:
                chroma[p.name] = ch
        out.append((p.name, img))
        if infos is not None:
            infos[p.name] = read_info(p)
    return out


def camera_groups(names: list[str], infos: dict[str, PhotoInfo]) -> tuple[tuple | None, dict[str, str]]:
    """Welke foto's horen bij het meest gebruikte toestel, lens en zoom (V10)? Geeft (sleutel van die groep,
    {foto: reden} voor de foto's die er niet bij horen). Foto's zonder cameragegevens horen er altijd bij:
    daarover zegt de EXIF niets, en de kalibratie gooit een foto die niet past er zelf uit."""
    keys = {n: infos[n].key() for n in names if n in infos and infos[n].key() is not None}
    if not keys:
        return None, {}
    counts: dict[tuple, int] = {}
    for k in keys.values():
        counts[k] = counts.get(k, 0) + 1
    main = max(counts, key=counts.get)
    label = {k: infos[next(n for n, kk in keys.items() if kk == k)].label() for k in counts}
    return main, {n: f"andere camera, lens of zoom ({label[k]}; de meeste foto's: {label[main]})"
                  for n, k in keys.items() if k != main}


def _object_distance(poses, center) -> float:
    return float(np.median([np.linalg.norm(p.center - center) for p in poses]))


F_STD_WARN = 0.003  # relatieve onzekerheid (1σ) van de brandpuntsafstand waarboven de scan waarschuwt


NO_CONTOUR_HELP = (
    "Mogelijke oorzaken: het object steekt weinig af tegen de mat (donker object op de zwarte "
    "vakken, of wit op wit), harde schaduwen, het object ligt (deels) naast het geblokte deel van "
    "de mat, of de foto's recht van boven zijn scheef of te dichtbij. Tips: diffuus licht, het "
    "object midden op de mat, 4-6 foto's recht van boven met de hele mat in beeld. Kijk in "
    "{debug} naar masker_*.jpg (oranje = object, blauw = mat) en bovenaanzicht.png."
)


def choose_mat(images, requested: str | None, warnings: list[str], log=print) -> MatSpec:
    """De mat op de foto's (herkend aan de markers); een afwijkende keuze wordt gemeld en overruled."""
    asked = None if not requested or requested.lower() == "auto" else get_spec(requested)
    found, _ = calib.identify_mat(images)
    if found is None:  # niets herkend: de detectie hieronder geeft de uitleg
        return asked or get_spec("A4")
    if asked is not None and found.name != asked.name:
        warnings.append(f"gekozen mat {asked.label}, maar de foto's tonen mat {found.label}: die is gebruikt")
        log(f"LET OP: de foto's tonen mat {found.label}, niet {asked.label}; mat {found.label} gebruikt")
    return found


def _advice(cov: dict) -> str:
    return (" Voor een nieuwe fotoset: " + " ".join(cov["advies"])) if cov.get("advies") else ""


def unseen_holes(part, K: np.ndarray, top_views: list, min_px: int = 20) -> list[int]:
    """Gaten waardoor in geen enkel bovenaanzicht zekere mat te zien is: mogelijk spookgaten (bijv. een
    wit onderdeel op de witte marge, waar object en mat niet te onderscheiden zijn). Per foto de doorkijk:
    binnen de projectie van zowel de boven- als de onderrand van het gat."""
    from .profile import circle_polygon

    out = []
    for i, h in enumerate(part.holes):
        if h.blind:  # een blind gat (v0.11) laat nooit mat zien
            continue
        ring = circle_polygon((h.x, h.y), h.d / 2, 48)
        seen = judged = False
        for pose, m in top_views:
            uv = []
            for z in (part.height_at(h.x, h.y), 0.0):  # bij een trede: de bovenkant daar
                p, depth = calib.project(np.column_stack([ring, np.full(len(ring), z)]), pose, K)
                if np.any(depth <= 0):
                    break
                uv.append(p)
            if len(uv) < 2:
                continue
            lo = np.floor(np.min(np.vstack(uv), axis=0)).astype(int) - 1
            hi = np.ceil(np.max(np.vstack(uv), axis=0)).astype(int) + 2
            H, W = m.bg.shape
            if lo[0] < 0 or lo[1] < 0 or hi[0] > W or hi[1] > H:
                continue
            a = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.uint8)
            b = np.zeros_like(a)
            cv2.fillPoly(a, [np.round(uv[0] - lo).astype(np.int32)], 1)
            cv2.fillPoly(b, [np.round(uv[1] - lo).astype(np.int32)], 1)
            through = (a & b).astype(bool)
            if through.sum() < min_px:
                continue
            judged = True
            if m.bg[lo[1]:hi[1], lo[0]:hi[0]][through].mean() > 0.1:
                seen = True
                break
        if judged and not seen:
            out.append(i)
    return out


def _turns_deg(outer) -> np.ndarray:
    """Richtingsverandering (graden) bij elk hoekpunt k, tussen rand k-1 en rand k."""
    return np.degrees(np.abs((outer.angles - np.roll(outer.angles, 1) + np.pi) % (2 * np.pi) - np.pi))


# V14/V15 (v0.12): een inham van hooguit zoveel mm breed en diep, in een rechte rand, wordt ook als rechthoek geprobeerd
NOTCH_MAX_MM, NOTCH_STRAIGHT_DEG = 12.0, 10.0


def _u_notch(o, t: int):
    """De V-vormige inham met de punt in hoekpunt `t` als rechthoekige inham: de twee schuine randen worden een wand,
    een bodem en een wand, haaks op de rand eromheen. Bodem op de diepte van de (afgeronde) punt, breedte die van de
    V op halve diepte. Geeft een Profile, of None als het geen V-inham in een rechte rand is."""
    n = o.n
    V = o.vertices()
    if n < 5:
        return None
    d = np.roll(V, -1, axis=0) - V  # rand k: hoekpunt k -> k+1
    prev = np.roll(d, 1, axis=0)
    cross = prev[:, 0] * d[:, 1] - prev[:, 1] * d[:, 0]  # < 0: holle hoek (tegen de klok in)
    a, b = (t - 1) % n, (t + 1) % n
    if not (cross[t] < 0 and cross[a] > 0 and cross[b] > 0):
        return None
    if max(np.linalg.norm(d[a]), np.linalg.norm(d[t])) > NOTCH_MAX_MM:
        return None
    before, after = (t - 2) % n, (t + 1) % n  # de rand voor en na de inham
    gap = abs((o.angles[before] - o.angles[after] + np.pi) % (2 * np.pi) - np.pi)
    if np.degrees(gap) > NOTCH_STRAIGHT_DEG:
        return None
    ang = float(o.angles[before] + ((o.angles[after] - o.angles[before] + np.pi) % (2 * np.pi) - np.pi) / 2)
    n_o = np.array([math.cos(ang), math.sin(ang)])  # buitennormaal van de rand
    tau = np.array([-n_o[1], n_o[0]])  # looprichting (tegen de klok in)
    hoeken = afgeronde_hoeken(o.corner_table())
    apex = np.asarray(hoeken[t][1] if hoeken[t][1] is not None else V[t], float)  # diepste punt van de punt
    mouth = (V[a] + V[b]) / 2
    depth_v = float(n_o @ (mouth - V[t]))  # diepte van de scherpe V
    depth = float(n_o @ (mouth - apex))
    if depth_v <= 0 or depth <= 0:
        return None
    width = float(np.linalg.norm(V[a] - V[b])) * (1.0 - 0.5 * depth / depth_v)
    c = o.center
    u = float(tau @ (apex - c))
    angles = [ang + math.pi / 2, ang, ang - math.pi / 2]  # wand (normaal = looprichting), bodem, wand
    offsets = [u - width / 2, float(n_o @ (apex - c)), -u - width / 2]
    # randen in de volgorde van de contour, met de punt op plaats 2: [.., rand voor, wand, bodem, wand, rand na, ..]
    order = [(t - 2 + k) % n for k in range(n)]
    ang_r, off_r, fil_r = o.angles[order], o.offsets[order], o.fillets[order]  # rand 0 = rand voor de inham
    new_angles = np.concatenate([[ang_r[0]], angles, ang_r[3:]])
    new_offsets = np.concatenate([[off_r[0]], offsets, off_r[3:]])
    # hoekpunt k ligt tussen rand k-1 en k: de ingang houdt zijn afronding, de bodemhoeken beginnen scherp
    new_fillets = np.concatenate([[fil_r[0], fil_r[1], 0.0, 0.0, fil_r[3]], fil_r[4:]])
    out = profile.Profile("polygon", c.copy(), np.unwrap(new_angles), new_offsets, new_fillets)
    return out if out.is_valid() else None


def _notch_alternatives(part, K: np.ndarray, vd: list, energy: float, log=print):
    """Een kleine rechthoekige inham in de buitenrand komt uit de startcontour vaak als V met een afgeronde punt
    (de hoeken van de bodem zijn een paar pixels groot). Per V-inham in een rechte rand ook de rechthoek proberen
    (_u_notch, kort gefit); past die duidelijk beter (de energie daalt meer dan de straf voor een hoekpunt extra),
    dan verder met de rechthoek. Geeft (model, energie, de vier hoekpunten (mat) van elke aangenomen inham)."""
    notches: list[np.ndarray] = []
    if part.outer.kind != "polygon":
        return part, energy, notches
    for _ in range(3):
        best = None
        for t in range(part.outer.n):
            outer = _u_notch(part.outer, t)
            if outer is None:
                continue
            cand = part.copy()
            cand.outer = outer
            cand, e, _ = silhouette.refine(cand, K, vd, max_evals=400)
            penalty = max(0.005 * min(e, energy), float(len(vd)))
            log(f"inham bij hoek {t + 1} als rechthoek geprobeerd: energie {e:.0f} tegen {energy:.0f}"
                + ("; verder met de rechthoek" if e + penalty < energy else ""))
            if e + penalty < energy and (best is None or e < best[1]):
                best = (cand, e, t)
        if best is None:
            break
        part, energy, _ = best
        notches.append(part.outer.vertices()[1:5])  # _u_notch zet de rand voor de inham op plaats 0
    return part, energy, notches


def _simplify_outline(part, K: np.ndarray, vd: list, energy: float, max_turn_deg: float = 10.0,
                      max_edge_mm: float = 5.0, max_failures: int = 6, log=print):
    """Haalt hoekpunten weg die het model niet nodig heeft: een knik van een paar graden in een rechte
    rand, of een korte rand (bijv. een afschuining) op een hoek.

    De fit kan geen hoekpunten weghalen. Zo'n hoekpunt in de startcontour komt vaak van een stuk rand
    zonder bewijs (zwart op zwart bij een hoek) en is dan geen kenmerk van het onderdeel. Per kandidaat
    volgt een korte fit zonder dat hoekpunt; past het model dan even goed (de energie stijgt minder dan
    0,5% of één pixel per foto), dan blijft het weg. Een echt kenmerk, of een schaduw die het masker
    verkeerd laat lopen, heeft bewijs in de foto's: dan blijft het staan (en markeert de
    kwaliteitspoort een schaduw). Geeft (model, energie, aantal weggehaalde hoekpunten).

    Alleen mislukte pogingen zijn beperkt (`max_failures`): een rommelige contour bij een hoek zonder bewijs
    kan tien hoekpunten te veel hebben, en tot v0.8 stopte het na acht pogingen, geslaagd of niet. Elke
    kandidaat krijgt 400 evaluaties: met 200 was hij bij zo'n contour vaak nog niet uitgefit, en bleef er een
    cluster korte randen staan waarop geen afschuining van de bovenrand meer paste (v0.8,
    ROUTE-A-VERBETERPUNTEN §3h). Een schone contour heeft geen kandidaten en kost dus niets extra.
    """
    removed, failures = 0, 0
    while part.outer.kind == "polygon" and part.outer.n > 3 and failures < max_failures:
        o = part.outer
        turn = _turns_deg(o)
        V = o.vertices()
        lengths = np.linalg.norm(np.roll(V, -1, axis=0) - V, axis=1)  # rand k: hoekpunt k -> k+1
        tol = max(0.005 * energy, float(len(vd)))
        options = [("knik", k, profile.merge_at_vertex(o, k)) for k in range(o.n) if turn[k] < max_turn_deg]
        options += [("korte rand", k, profile.drop_edge(o, k)) for k in range(o.n) if lengths[k] < max_edge_mm]
        cands = []
        for what, k, outer in options:
            if outer is None:
                continue
            outer, _ = profile.regularize_angles(outer)  # weer haaks op de andere randen, zoals de startcontour
            if not outer.is_valid():
                continue
            cand = part.copy()
            cand.outer = outer
            cands.append((silhouette.energy(cand, K, vd), what, k, cand))
        changed = False
        for e_raw, what, k, cand in sorted(cands, key=lambda c: c[0]):
            # Zonder bijstellen past een samengevoegde rand nog niet (hij ligt op het gemiddelde); kansloos
            # is een kandidaat pas als hij veel slechter past: dan is er bewijs voor dit hoekpunt.
            if e_raw > energy + max(20 * tol, 0.5 * energy) or failures >= max_failures:
                break
            cand, e_cand, _ = silhouette.refine(cand, K, vd, max_evals=400)
            if e_cand > energy + tol:
                failures += 1
            else:
                detail = f"knik van {turn[k]:.1f}°" if what == "knik" else f"korte rand van {lengths[k]:.1f} mm"
                log(f"{detail} uit de contour gehaald: het model past zonder even goed (energie {energy:.0f} → "
                    f"{e_cand:.0f})")
                part, energy, removed, changed = cand, e_cand, removed + 1, True
                break
        if not changed:
            break
    return part, energy, removed


COARSE_EPS = 0.01  # V30: tolerantie van de grovere startcontour, als deel van de omtrek
COARSE_RECT = 0.85  # een rechthoek als start als de contour minstens dit deel van zijn kleinste rechthoek vult


def _coarse_starts(part) -> list[tuple[str, profile.Profile]]:
    """V30: grovere buitencontouren naast een rommelige startcontour.

    Waar een zwart onderdeel op een zwart vak ligt, is er in de bovenaanzichten geen bewijs voor de rand, en dan
    rafelt de startcontour (een trap van korte randen). Vanuit zo'n start vindt de kompaszoektocht vaak geen goede
    vorm meer: hij kan geen hoekpunten weghalen, en welke rafels er zijn hangt af van kleinigheden (zelfs van het
    aantal rekenthreads). Daarom ook een grove benadering van dezelfde contour (tolerantie 1% van de omtrek, de
    randen haaks op de hoofdrichting) en, als de contour zijn kleinste omhullende rechthoek grotendeels vult, die
    rechthoek. De silhouetenergie van alle foto's kiest daarna (zie _refine_start)."""
    o = part.outer
    if o.kind != "polygon" or o.n <= 4:
        return []
    P = o.outline()
    perim = float(np.sum(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)))
    out = []
    coarse = profile.polygon_from_contour(P, 0.2, eps_mm=COARSE_EPS * perim)
    coarse, _ = profile.regularize_angles(coarse)
    if coarse.is_valid() and coarse.n <= o.n - 2:
        out.append(("grovere contour", coarse))
    (cx, cy), (w, h), a = cv2.minAreaRect(P.astype(np.float32))
    if w * h > 0 and o.area() / (w * h) >= COARSE_RECT and not (out and out[0][1].n == 4):
        t = math.radians(a)
        r = float(np.median(o.fillets[o.fillets > 0])) if np.any(o.fillets > 0) else 0.0
        rect = profile.Profile("polygon", np.array([cx, cy]), t + np.arange(4) * math.pi / 2,
                               np.array([w / 2, h / 2, w / 2, h / 2]), np.full(4, min(r, 0.25 * min(w, h))))
        if rect.is_valid():
            out.append(("rechthoek", rect))
    return out


def _refine_start(part, K: np.ndarray, vd: list, max_evals: int, log=print):
    """De hoofdverfijning, met bij een rommelige startcontour ook de grovere starten van _coarse_starts (V30).

    Elke start wordt volledig verfijnd (de rommelige kort: daaruit wordt het toch zelden wat); de laagste energie
    wint, met per hoekpunt een kleine straf (zoals in _simplify_outline: 0,5% of één pixel per foto). Een echt
    kenmerk dat de grove start mist, kost in elke foto veel meer dan die straf. Geeft (model, energie, evaluaties van
    de winnende start)."""
    n_features = ((part.outer.n if part.outer.kind == "polygon" else 1) + len(part.holes) + len(part.cutouts)
                  + len(part.slots))
    alts = _coarse_starts(part)
    first = max_evals
    if n_features > 20:  # rommelige startcontour: alleen kort fitten (de grovere start komt hieronder)
        first = min(max_evals, 300)
        log(f"rommelige startcontour ({n_features} randen, gaten en uitsparingen): korte verfijning")
    best, e_best, evals = silhouette.refine(part, K, vd, max_evals=first, log=log)
    if not alts:
        return best, e_best, evals

    def corners(p) -> int:
        return p.outer.n if p.outer.kind == "polygon" else 1

    for what, outer in alts:
        cand = part.copy()
        cand.outer = outer
        if not cand.is_valid():
            continue
        cand, e, n = silhouette.refine(cand, K, vd, max_evals=max_evals)
        better = e + max(0.005 * min(e, e_best), float(len(vd))) * (corners(cand) - corners(best)) < e_best
        log(f"startcontour met {part.outer.n} randen; ook als {what} ({outer.n} randen) geprobeerd: energie {e:.0f} "
            f"tegen {e_best:.0f}" + ("; verder met de " + what if better else ""))
        if better:
            best, e_best, evals = cand, e, n
    return best, e_best, evals


# V27: een duwtje wordt alleen gecorrigeerd als het gemiddelde verlies van de randfit daardoor minstens zoveel kleiner wordt;
# hooguit zoveel rondes (een nieuwe sprong, of dezelfde verder verfijnd)
NUDGE_GAIN, NUDGE_ROUNDS = 0.9, 4


def _fit_cost(ef) -> float:
    """Het gemiddelde verlies per randpunt (Cauchy, zoals in de randfit); oneindig zonder randfit."""
    if not ef.accepted or len(ef.residuals) == 0:
        return math.inf
    return float(np.mean(np.log1p((ef.residuals / edgefit.F_SCALE) ** 2)))


def _find_nudge(part, K: np.ndarray, vd: list, ef, order: list[str], info: dict):
    """V27: per foto de verschuiving van het model (edgefit.view_offsets) en de toets op een duwtje
    (placement.find_nudge). `info` krijgt de toets en de verschuivingen (voor diagnose.json)."""
    D, c = edgefit.view_offsets(part, K, vd, prob=ef.extra.get("problem"))
    radius = float(np.max(np.linalg.norm(part.outer.outline() - c, axis=1)))
    return placement.find_nudge(D, [v.pose.name for v in vd], order, c, radius, info)


HOLE_PARAMS = ("hx", "hy", "hd", "hk", "hc", "hz", "hp")  # maat, plaats, verzinking, kamer en diepte van een gat


def _edge_fit(part, K: np.ndarray, vd: list, mm_per_px: float, log=None, retries: int = 2):
    """edgefit.fit. Stuit alleen een gat op de grens van het vertrouwensgebied (zijn maat, plaats, verzinking, kamer of
    diepte), dan zoekt de randfit vanaf die grens verder, hooguit `retries` keer: de pixelfit zette dat gat te ver
    weg, en anders bleef voor het hele onderdeel de pixelfit staan (v0.11; in v0.10 alleen voor een kamer). Een gat
    zonder bewijs rondom staat in de randfit vast en kan dus niet zo weglopen."""
    ef = edgefit.fit(part, K, vd, log=log, mm_per_px=mm_per_px)
    for _ in range(retries):
        edge = ef.extra.get("at_edge", [])
        if ef.accepted or not edge or not all(n[:2] in HOLE_PARAMS for n in edge):
            break
        ef = edgefit.fit(ef.extra["moved"], K, vd, log=log, mm_per_px=mm_per_px)
    return ef


def _effective_uncertainty(unc, snaps: list, scale_rel: float):
    """Samenvatting per soort maat (voor report.json en `camtocad valideer`) uit de σ per maat: het
    grootste toevallige deel per soort, zonder printschaal (die staat apart in scale_rel)."""
    def rel(s) -> float:
        return math.sqrt(max(s.sigma ** 2 - (scale_rel * abs(s.measured)) ** 2, 0.0))

    def worst(pred, default):
        vals = [rel(s) for s in snaps if pred(s.name)]
        return max(vals) if vals else default

    def is_position(n: str) -> bool:
        return n.startswith("steekcirkel") or (n.startswith(("gat ", "sleuf ", "uitsparing ")) and n.endswith((" x", " y")))

    def is_step(n: str, what: str) -> bool:
        return n.startswith("trede ") and n.endswith(what)

    def is_inner_size(n: str) -> bool:  # gaten, verzinkingen, kamers, sleuven en uitsparingen: binnenvormen
        return n.startswith(("gat Ø", "blind gat Ø", "verzinking Ø", "kamerboring Ø")) or (
            n.startswith(("sleuf ", "uitsparing ")) and n.endswith(("breedte", "hartafstand", "lengte")))

    return cadmodel.Uncertainty(
        edge=worst(lambda n: n.startswith(("x-maat", "y-maat", "diameter")) or is_step(n, "positie"),
                   unc.edge * math.sqrt(2)) / math.sqrt(2),
        height=worst(lambda n: n == "hoogte" or is_step(n, "hoogte"), unc.height),
        hole_d=worst(is_inner_size, unc.hole_d),
        hole_xy=worst(is_position, unc.hole_xy),
        fillet=worst(lambda n: n.startswith("afronding") or n.endswith(("hoekstraal", "bovenrand")), unc.fillet),
        scale_rel=scale_rel)


def _slots_or_holes(part, K: np.ndarray, vd: list, energy: float, log=print):
    """Een korte sleuf die een gat even goed verklaart, wordt een gat. In schuine bovenaanzichten is de
    doorkijk door een gat lensvormig, en dan lijkt hij in de startcontour op een korte sleuf; de pixelfit
    rekent die parallax wel exact door."""
    for i in reversed(range(len(part.slots))):
        s = part.slots[i]
        if s.kind != "sleuf" or s.length > 2.0 * s.width:
            continue
        trial = part.copy()
        trial.slots.pop(i)
        trial.holes.append(Hole(s.x, s.y, 0.5 * (s.width + s.length)))
        j = len(trial.holes) - 1
        trial, e_trial, _ = silhouette.refine(trial, K, vd, max_evals=150, only={f"hx{j}", f"hy{j}", f"hd{j}"})
        if e_trial <= energy * 1.002 + 20:
            log(f"korte sleuf ({s.length:.1f} x {s.width:.1f} mm) is een gat Ø {trial.holes[j].d:.1f}: het model past "
                f"even goed (energie {energy:.0f} → {e_trial:.0f})")
            part, energy = trial, e_trial
    return part, energy


def _holes_or_pockets(part, K: np.ndarray, vd: list, energy: float, log=print):
    """Een gat dat eigenlijk een rechthoekige uitsparing of een sleuf is. In de bovenaanzichten ziet de
    doorkijk er door parallax en onscherpte vaak ronder uit dan hij is, en dan wordt hij in de startcontour
    een gat. Per gat een rechthoek met afgeronde hoeken proberen, gericht als de contour; alleen als die
    duidelijk beter past. (Een ronde rechthoek met afronding = halve breedte is een cirkel: bij een echt gat
    wordt hij niet beter.)"""
    theta = dominant_angle(part.outer) if part.outer.kind == "polygon" else 0.0
    for i in reversed(range(len(part.holes))):
        h = part.holes[i]
        if h.d < 3.0:
            continue
        trial = part.copy()
        trial.holes.pop(i)
        trial.slots.append(Slot(h.x, h.y, h.d, 0.9 * h.d, theta, 0.35 * h.d, "rechthoek"))
        j = len(trial.slots) - 1
        trial, e_trial, _ = silhouette.refine(trial, K, vd, max_evals=200,
                                              only={f"sx{j}", f"sy{j}", f"sl{j}", f"sw{j}", f"sa{j}", f"sr{j}"})
        if e_trial < energy - (0.002 * energy + 20):
            s = trial.slots[j]
            if s.r >= 0.45 * s.width:  # zo rond afgerond: een sleuf
                trial.slots[j] = replace(s, kind="sleuf", r=0.0)
            log(f"gat Ø {h.d:.1f} is een {'sleuf' if trial.slots[j].kind == 'sleuf' else 'uitsparing'} "
                f"{s.length:.1f} x {s.width:.1f} mm: het model past duidelijk beter (energie {energy:.0f} → "
                f"{e_trial:.0f})")
            part, energy = trial, e_trial
    return part, energy


def _add_missed_holes(part, K: np.ndarray, vd: list, energy: float, log=print):
    """V4: ronde uitsparingen worden gaten, en een gat dat de startcontour miste wordt toegevoegd waar de
    bovenaanzichten door het bovenvlak heen mat zien (holes.py). Alleen als het model er duidelijk beter
    door past: eerst wordt alleen het nieuwe gat op maat gebracht, daarna kort alles. Ook de andere
    features (V15): een gat dat een uitsparing is, een korte sleuf die een gat is, en een ruwe polygoon
    die een sleuf of rechthoek is."""
    part, energy = _holes_or_pockets(part, K, vd, energy, log)
    part, energy = _slots_or_holes(part, K, vd, energy, log)
    part2, n_round = holes.round_cutouts(part)
    if n_round:
        part2, e2, _ = silhouette.refine(part2, K, vd, max_evals=200)
        if e2 <= energy * 1.01:
            log(f"{n_round} ronde uitsparing(en) verder als gat")
            part, energy = part2, e2
    for loose in (False, True):
        part2, n_slot = holes.slot_cutouts(part, loose=loose)
        if n_slot:
            part2, e2, _ = silhouette.refine(part2, K, vd, max_evals=300)
            if e2 <= energy * 1.01:
                log(f"{n_slot} uitsparing(en) verder als sleuf of rechthoek")
                part, energy = part2, e2
    for cand in holes.candidates(part, K, vd):
        trial = part.copy()
        trial.holes.append(cand)
        i = len(trial.holes) - 1
        trial, _, _ = silhouette.refine(trial, K, vd, max_evals=150, only={f"hx{i}", f"hy{i}", f"hd{i}"})
        trial, e_trial, _ = silhouette.refine(trial, K, vd, max_evals=150)
        if e_trial < energy - (0.002 * energy + 20):
            h = trial.holes[i]
            log(f"gat toegevoegd: Ø {h.d:.1f} mm op ({h.x:.1f}, {h.y:.1f}); in de bovenaanzichten is daar mat te "
                f"zien (energie {energy:.0f} → {e_trial:.0f})")
            part, energy = trial, e_trial
    return part, energy


# V17: een vorm die geen prisma is als model proberen, als de vormtoets (V19) erom vraagt
# kijkhoekverschil (prismcheck) waaronder een afgeschuinde of afgeronde bovenrand geprobeerd wordt; prisma's gaven
# +0,003 tot +0,10 mm, een afschuining van 1,5 mm op een zwart onderdeel −0,013 mm (de meldgrens is −0,04 mm). Dat
# laatste hangt van de pixelfit af (v0.8: +0,002 of +0,034): daarom ook een proef zonder fit (TOP_PROBE_GAIN)
TOP_TRY_MM = 0.0
TOP_PROBE_GAIN = 0.01  # of: de proef met een afschuining of afronding past zonder fit al 1% beter (§3h)
NON_PRISM_GAIN = 0.05  # zoveel lager moet de energie worden (een prisma past met een afschuining 1-3% beter)


def _prism_check(ef, part):
    """V19 op de randfit (of de pixelfit, als de randfit niet gebruikt is)."""
    _, angle, shift = cadmodel.to_part_frame(ef.part if ef.accepted else part)
    return prismcheck.check(ef, angle, shift)


def _trend_asks(shape) -> bool:
    """Vraagt het kijkhoekverschil van de vormtoets om een afschuining of afronding van de bovenrand?"""
    return shape.trend is not None and shape.trend < TOP_TRY_MM


def _probe_asks(part, K: np.ndarray, vd: list, info: dict, log=print) -> bool:
    """Past het prisma met een afschuining of afronding van de bovenrand al zonder fit merkbaar beter? Bij een zwart
    onderdeel zegt het kijkhoekverschil te weinig: een afschuining van 1,5 mm gaf −0,013, +0,002 of +0,034 mm (per
    versie van de pixelfit), gewone prisma's +0,010 tot +0,051 (ROUTE-A-VERBETERPUNTEN §3h)."""
    gain = silhouette.top_edge_probe(part, K, vd)
    info["proef_bovenrand"] = round(gain, 4)
    if gain >= TOP_PROBE_GAIN:
        log(f"proef: met een afschuining of afronding van de bovenrand past het prisma {100 * gain:.1f}% beter")
    return gain >= TOP_PROBE_GAIN


def _non_prism(part, K: np.ndarray, vd: list, energy: float, shape, mm_per_px: float, log=print,
               info: dict | None = None):
    """V17: past een prisma niet (de vormtoets vond een stuk bovenrand dat lager of hoger ligt, of een
    kijkhoekverschil zoals bij een afschuining), dan een trede of een afgeschuinde of afgeronde bovenrand
    rondom als model proberen, vanuit de pixelfit. Alleen als het model er duidelijk beter door past (energie
    minstens 5% lager); geeft dan (model, energie), anders None. `info` (voor diagnose.json) krijgt wat er
    geprobeerd is."""
    info = {} if info is None else info
    goal = energy * (1.0 - NON_PRISM_GAIN)
    runs = shape.runs_mat
    info.update({"kijkhoek_verschil_mm": shape.trend, "energie_prisma": round(float(energy), 1)})
    if runs and part.outer.kind == "polygon":
        r = shape.start or max(runs, key=lambda r: np.linalg.norm(np.asarray(r["tot"]) - np.asarray(r["van"])))
        cand, e = silhouette.fit_step(part, K, vd, r["van"], r["tot"], r["midden"], lower=r["soort"] == "lager")
        info.update({"geprobeerd": "trede", "energie": None if cand is None else round(float(e), 1),
                     "aangenomen": bool(cand is not None and e < goal)})
        if cand is not None and e < goal:
            st = cand.steps[0]
            log(f"trede gemodelleerd: voorbij een rechte lijn is het deel {st.height:.2f} in plaats van "
                f"{cand.height:.2f} mm hoog; het model past duidelijk beter (energie {energy:.0f} → {e:.0f})")
            return cand, e
        log("geen trede gemodelleerd: " + (
            "geen geldige trede gevonden" if cand is None else
            f"het model past er niet duidelijk beter door (energie {energy:.0f} → {e:.0f})"))
    elif not runs and (_trend_asks(shape) or _probe_asks(part, K, vd, info, log)):
        cand, e = silhouette.fit_top_edge(part, K, vd, log=log)
        ok = cand is not None and e < goal and cand.top_edge.size >= max(0.3, 2.0 * mm_per_px)
        info.update({"geprobeerd": "bovenrand", "energie": None if cand is None else round(float(e), 1),
                     "soort": None if cand is None else cand.top_edge.kind,
                     "maat_mm": None if cand is None else round(cand.top_edge.size, 3), "aangenomen": ok})
        if ok:
            te = cand.top_edge
            log(f"bovenrand gemodelleerd: {te.kind} van {te.size:.2f} mm rondom, hoogte {cand.height:.2f} mm; het "
                f"model past duidelijk beter (energie {energy:.0f} → {e:.0f})")
            return cand, e
        log("geen afschuining of afronding van de bovenrand gemodelleerd: " + (
            "geen geldige vorm gevonden" if cand is None else
            f"het model past er niet duidelijk beter door (energie {energy:.0f} → {e:.0f}, "
            f"{cand.top_edge.kind} {cand.top_edge.size:.2f} mm)"))
    return None


def _quality_issues(part, stats: dict, evals: int, max_evals: int, mm_per_px: float = 0.25,
                    notches: list | None = None) -> list[str]:
    """Signalen dat het model niet klopt, ook al is er een model uitgekomen. `notches`: de hoekpunten van rechthoekige
    inhammen die de fit heeft aangenomen (_notch_alternatives): hun korte wanden zijn geen teken van een schaduw."""
    issues = []
    if part.outer.kind == "polygon":
        # een uitstulping of inham van een paar pixels (schaduw, rommelig masker) geeft korte randen
        V = part.outer.vertices()
        short_mm = max(2.0, 10.0 * mm_per_px)
        is_short = np.linalg.norm(np.roll(V, -1, axis=0) - V, axis=1) < short_mm  # rand k: hoekpunt k -> k+1

        def in_notch(k: int) -> bool:
            ends = (V[k], V[(k + 1) % len(V)])
            return any(all(np.min(np.linalg.norm(nt - e, axis=1)) < 0.5 for e in ends) for nt in notches or [])
        short = sum(1 for k in np.flatnonzero(is_short) if not in_notch(k))
        if short >= 2:
            issues.append(f"{short} zeer korte randen (< {short_mm:.1f} mm): mogelijk een uitstulping of inham die "
                          "er niet is")
        # een knik van een paar graden in een rechte rand (schaduw langs die rand), vaak met een enorme
        # 'afronding' die de knik gladstrijkt
        o = part.outer
        turn = _turns_deg(o)
        if np.any(turn < 10.0):
            issues.append(f"een rand heeft een knik van {float(turn.min()):.1f}°: waarschijnlijk een schaduw of een "
                          "rommelig masker langs die rand")
        lengths = np.linalg.norm(np.roll(V, -1, axis=0) - V, axis=1)  # rand k: hoekpunt k -> k+1
        shorter = np.minimum(lengths, np.roll(lengths, 1))  # de randen aan weerszijden van hoekpunt k
        if np.any(o.fillets > shorter):
            issues.append("een afronding is groter dan de randen eromheen: onwaarschijnlijke vorm")
    if stats["iou_median"] < 0.98:
        issues.append(f"silhouetten passen matig (IoU mediaan {stats['iou_median']:.3f}; goed is > 0,98)")
    if stats["iou_min"] < 0.95:
        issues.append(f"in minstens één foto past het model slecht (IoU {stats['iou_min']:.3f})")
    if evals >= max_evals:
        issues.append("de fit is niet uitgeconvergeerd")
    if part.outer.kind == "polygon" and part.outer.n > 12:
        issues.append(f"ongewoon veel randen ({part.outer.n}): waarschijnlijk een rommelige contour")
    if part.cutouts:
        issues.append(f"{len(part.cutouts)} niet-ronde uitsparing(en): controleer of dat klopt")
    return issues


# V8 (v0.12): een slagschaduw naast het onderdeel telt als de mediaan over de foto's van masks._cast_shadow minstens
# zo groot is
SHADOW_WARN = 0.10
# V28: foto's met een onscherpte boven CALIB_BLUR_MAX (px, gemeten aan de mat) niet in de kalibratie, als er
# minstens CALIB_MIN_SHARP scherpe overblijven; een camera dichter dan CLOSE_MM bij de mat (langs de kijkrichting) geeft
# een waarschuwing
CALIB_BLUR_MAX, CALIB_MIN_SHARP, CLOSE_MM = preflight.BLUR_BAD, 8, preflight.CLOSE_MM


def calibrate_sharp(dets: list, blur: dict, spec, log=print) -> calib.CalibrationResult:
    """Zelfkalibratie op de scherpe foto's; de pose van de onscherpe daarna uit die camera (V28). Bewogen of niet
    scherpgestelde foto's hebben onnauwkeurige mathoeken, en die trokken de camera mee (de eerste echte fotoset had een
    reprojectiefout van 1,6 px, vooral door zulke foto's). Voor de maskers zijn ze vaak nog bruikbaar: die passen zich
    aan de onscherpte aan."""
    soft = [d for d in dets if blur.get(d.name) is not None and blur[d.name] > CALIB_BLUR_MAX]
    if not soft or len(dets) - len(soft) < CALIB_MIN_SHARP:
        return calib.calibrate(dets, spec)
    cal = calib.calibrate([d for d in dets if d not in soft], spec)
    extra = calib.calibrate(soft, spec, camera=cal.camera)
    cal.poses.update(extra.poses)
    cal.rejected.update(extra.rejected)
    log(f"{len(soft)} onscherpe foto('s) (σ > {CALIB_BLUR_MAX:.0f} px) niet in de kalibratie; hun pose komt uit de "
        f"camera van de andere {len(dets) - len(soft)}")
    return cal


def run_scan(images, out_dir: str | Path, opts: ScanOptions | None = None, log=print, scan_name: str = "") -> dict:
    """Verwerkt een scan; `images` is een map of een lijst (naam, grijswaardenbeeld)."""
    opts = opts or ScanOptions()
    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sx, sy = opts.scale_xy
    for axis, s in (("X", sx), ("Y", sy)):
        if not 0.9 < s < 1.1:
            raise ScanError(f"Meetlijn {axis} {100 * s:.1f} mm wijkt te veel af van 100 mm: print de mat opnieuw op "
                            "100% (werkelijke grootte)")
    infos: dict[str, PhotoInfo] = {}
    chroma: dict[str, np.ndarray] = {}  # kleur op halve resolutie (V8), per foto in kleur
    if not isinstance(images, list):
        scan_name = scan_name or Path(images).name
        images = load_images(images, opts.max_side, log, infos, chroma)
    else:  # een lijst mag ook kleurbeelden (BGR) bevatten
        split = [(n, split_chroma(img)) for n, img in images]
        chroma = {n: ch for n, (_, ch) in split if ch is not None}
        images = [(n, gray) for n, (gray, _) in split]
    if chroma:
        log(f"{len(chroma)} foto('s) in kleur: kleur telt mee als bewijs voor het object")
    if len(images) < 6:
        raise ScanError(f"Te weinig foto's ({len(images)}); maak er minstens 20, rondom en recht van boven")
    warnings: list[str] = []
    spec = choose_mat(images, opts.mat, warnings, log).with_scale(sx, sy)
    log(f"kalibratiemat: {spec.label}" + (f", printschaal X {sx:.4f}, Y {sy:.4f}" if spec.is_scaled else ""))

    dbg = out / "debug"
    if opts.debug_images:
        dbg.mkdir(exist_ok=True)

    # 1-2. mat herkennen, zelfkalibratie en poses
    board = calib.make_board(spec)
    detector = calib.make_detector(board)
    bias = calib.detector_bias(spec)
    if np.any(np.abs(bias) > 0.05):
        log(f"OpenCV {cv2.__version__}: hoekdetectie gecorrigeerd voor een vaste verschuiving van "
            f"({bias[0]:+.2f}, {bias[1]:+.2f}) px")
    dets, lookup = [], dict(images)
    for name, img in images:
        d = calib.detect(img, board, name, detector, bias)
        if d is None:
            warnings.append(f"{name}: kalibratiemat niet gevonden")
        else:
            dets.append(d)
    log(f"mat gevonden in {len(dets)} van {len(images)} foto's")
    # V10: één cameramodel per toestel, lens en zoom; foto's van een andere lens (bijv. de macrostand van een
    # iPhone, die dichtbij vanzelf inschakelt) of met digitale zoom horen er niet bij
    _, other_lens = camera_groups([d.name for d in dets], infos)
    if other_lens:
        dets = [d for d in dets if d.name not in other_lens]
        log(f"{len(other_lens)} foto('s) van een andere camera, lens of zoom niet gebruikt: "
            + next(iter(other_lens.values())))
    # V28: de onscherpte per foto, gemeten aan de mat (preflight.py); onscherpe foto's niet in de kalibratie
    blur = {d.name: preflight.measure_blur(lookup[d.name], d, spec) for d in dets}
    try:
        cal = calibrate_sharp(dets, blur, spec, log)
    except ValueError as e:
        raise ScanError(str(e)) from e
    cal.rejected.update(other_lens)
    cam = cal.camera
    used = [infos[n] for n in cal.poses if n in infos and infos[n].key() is not None]
    camera_label = used[0].label() if used else ""
    # foto's die staand in plaats van liggend (of andersom) zijn opgeslagen: terugdraaien naar het
    # sensorformaat; de draairichting is die waarbij de pose met deze camera het best klopt
    n_turned = 0
    for name, img in images:
        if name in cal.poses or name in other_lens or img.shape != (cam.width, cam.height):
            continue
        best = None
        for code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            turned = cv2.rotate(img, code)
            d = calib.detect(turned, board, name, detector, bias)
            pose = calib.solve_pose(d, spec, cam) if d is not None and len(d.ids) >= 12 else None
            if pose is not None and (best is None or pose.rms_px < best[0].rms_px):
                best = (pose, turned, d, code)
        if best is not None and best[0].rms_px < 1.5:
            cal.poses[name], lookup[name] = best[0], best[1]
            if name in chroma:
                chroma[name] = np.ascontiguousarray(np.rot90(chroma[name], -1 if best[3] == cv2.ROTATE_90_CLOCKWISE
                                                             else 1))
            cal.rejected.pop(name, None)
            dets.append(best[2])
            blur[name] = preflight.measure_blur(best[1], best[2], spec)
            n_turned += 1
    if n_turned:
        log(f"{n_turned} foto('s) waren gedraaid opgeslagen en zijn teruggedraaid")
    for name, reason in cal.rejected.items():
        if reason.startswith("afwijkend formaat"):
            reason += " (andere camera/lens, zoom of bijgesneden?)"
        warnings.append(f"{name}: niet gebruikt ({reason})")
    log(f"camera gekalibreerd: f = {cam.K[0, 0]:.1f} px, reprojectiefout {cam.rms_px:.3f} px, "
        f"{len(cal.poses)} poses")
    close = sorted(n for n, pose in cal.poses.items() if preflight.view_distance(pose) < CLOSE_MM)
    if close:  # V28
        warnings.append(f"{len(close)} foto('s) van dichterbij dan {CLOSE_MM / 10:.0f} cm ("
                        + ", ".join(close[:6]) + (" ..." if len(close) > 6 else "") + "): de scherptediepte is "
                        "dan klein en een telefoon schakelt soms naar de macrolens. Houd 25-35 cm aan")
    if cam.f_std_rel > F_STD_WARN:  # V10: de brandpuntsafstand is slecht bepaald (weinig verschillende hoeken)
        warnings.append(f"brandpuntsafstand onzeker (σ {100 * cam.f_std_rel:.2f}%, goed is < {100 * F_STD_WARN:.1f}%): "
                        "maak foto's van meer verschillende hoeken en hoogtes, met de mat steeds grotendeels in beeld")
    blur = {n: b for n, b in blur.items() if n in cal.poses}
    blurry = sorted(n for n, b in blur.items() if b is not None and b > preflight.BLUR_WARN)
    if blurry:
        warnings.append(f"{len(blurry)} foto('s) onscherp (σ > {preflight.BLUR_WARN:.1f} px): "
                        + ", ".join(blurry[:6]) + (" ..." if len(blurry) > 6 else ""))

    # 3. objectmaskers
    raster = rasterize_board(spec, 10.0, 3.0)
    # hoe de camera het licht vastlegt (tone.py, v0.11): de verscherping één keer per scan, uit de randen van de
    # matvakken in een paar foto's; de toonkromme en de onscherpte daarna per foto, op de ruwe foto
    sharp = tone.camera_sharpening([(lookup[d.name], cal.poses[d.name], d.ids) for d in dets if d.name in cal.poses],
                                   cam, spec)

    def view_masks(pose):
        img = calib.undistort(lookup[pose.name], cam)
        pred, valid = masks.predict_background(raster, cam.K, pose, (cam.width, cam.height))
        depth = float((pose.R @ np.array([spec.size_mm[0] / 2, spec.size_mm[1] / 2, 0.0]) + pose.t)[2])
        ch = chroma.get(pose.name)
        if ch is not None:  # kleur (V8): terug naar volle resolutie en dezelfde ontvervorming als het grijsbeeld
            ch = calib.undistort(cv2.resize(ch.astype(np.float32), (img.shape[1], img.shape[0]),
                                            interpolation=cv2.INTER_LINEAR), cam)
        tn = tone.estimate(lookup[pose.name], pred, valid, cam, sharp)
        return pose, masks.classify(img, pred, valid, px_per_mm=cam.K[0, 0] / max(depth, 1.0),
                                    blur_px=blur.get(pose.name), chroma=ch, tone_params=tn)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:  # grote beeldbewerkingen: OpenCV en numpy geven de GIL vrij
        views = list(pool.map(view_masks, cal.poses.values()))
    tones = [m.tone_params for _, m in views if m.tone_params is not None]
    if tones:
        log(f"camera: {tone.describe(tones)}; verscherping k {sharp.k:.2f} (σ {sharp.sigma:.1f} px, "
            f"{sharp.n_photos} foto's)")
        n_hdr = sum(t.grid is not None for t in tones)
        if n_hdr >= HDR_WARN_FRAC * len(tones):
            # de kromme per tegel volgt een telefoon die per stuk beeld bewerkt; lokaal contrast (halo's) niet
            warnings.append(f"in {n_hdr} van de {len(tones)} foto's is de toonkromme per stuk beeld anders (lokale "
                            "toonbewerking, HDR). Die kromme wordt gevolgd, maar het lokale contrast dat er vaak bij hoort "
                            "niet. Waar wit daardoor afgekapt is (255), telt de rand niet als bewijs: een gat daar krijgt "
                            "maat en plaats uit de pixelfit, met een ruime U95. Zet HDR (Smart HDR, auto-HDR) uit voor de "
                            "beste nauwkeurigheid")
    top_views: list = []
    all_views, lig_of = views, {}  # ook de foto's die niet bij de rest passen staan in diagnose.json
    diag = {"camtocad": __version__, "opencv": cv2.__version__, "detector_bias_px": bias.tolist(),
            "camera": cam.to_dict(), "geweigerd": cal.rejected, "verscherping": sharp.to_dict()}

    def write_debug(extra: dict | None = None) -> None:
        if not opts.debug_images:
            return
        diag.update(extra or {})
        diag["fotos"] = debug.view_table(all_views, {d.name: d for d in dets}, {p.name for p, _ in top_views}, blur,
                                         lig_of, {n: i.label() for n, i in infos.items() if i.key() is not None})
        debug.write_json(dbg / "diagnose.json", diag)

    def write_overlays() -> None:
        if not opts.debug_images:
            return
        others = [v for v in views if all(v[0] is not t[0] for t in top_views)]
        for pose, m in top_views[:6] + others[:: max(1, len(others) // 4)][:4]:
            imwrite(dbg / f"masker_{Path(pose.name).stem}.jpg",
                        debug.mask_overlay(calib.undistort(lookup[pose.name], cam), m))

    # 3b. ligt het onderdeel in alle foto's op dezelfde plek? (placement.py)
    board_bounds = (0.0, spec.size_mm[0], 0.0, spec.size_mm[1])
    lig = placement.find(views, cam.K, board_bounds)
    diag["ligging"] = lig.to_dict()
    lig_of = {n: f"groep {k + 1}" for k, g in enumerate(lig.groups) for n in g}
    lig_of |= {n: "past nergens bij" for n in lig.outliers} | {n: "niet beoordeeld" for n in lig.unjudged}
    if lig.groups:
        if len(lig.groups[0]) < 0.75 * lig.judged:  # een kwart of meer past er niet bij: niet stilletjes weglaten
            write_overlays()
            write_debug()
            raise ScanError(placement.moved_message(lig, [n for n, _ in images]))
        odd = {n for g in lig.groups[1:] for n in g} | set(lig.outliers)
        if odd:  # een paar foto's die niet kloppen: onderdeel even aangeraakt, hand in beeld, mislukt masker
            views = [v for v in views if v[0].name not in odd]
            names = sorted(odd, key=[n for n, _ in images].index)
            log(f"{len(odd)} foto('s) passen niet bij de rest en worden niet gebruikt: " + ", ".join(names))
            warnings.append(f"{len(odd)} foto('s) niet gebruikt omdat ze niet bij de rest passen (onderdeel "
                            "verschoven of aangeraakt, hand of ander voorwerp in beeld, of mislukt masker): "
                            + ", ".join(names[:6]) + (" ..." if len(names) > 6 else ""))

    # 4. grove visual hull: waar staat het object ongeveer? (alleen lokaliseren en een bovengrens)
    try:
        coarse = hull.reconstruct(views, cam.K, board_bounds, fine=False)
    except ValueError as e:
        top_views = initial.select_top_views(views)
        cov = preflight.coverage(cal.poses, [board_bounds[1] / 2, board_bounds[3] / 2, 0.0])
        write_overlays()
        write_debug({"dekking": cov})
        raise ScanError(f"{e}. " + NO_CONTOUR_HELP.format(debug=dbg) + _advice(cov)) from e
    ijk = np.argwhere(coarse.occ)
    lo = coarse.origin + ijk.min(axis=0) * coarse.voxel
    hi = coarse.origin + ijk.max(axis=0) * coarse.voxel
    z_top = float(hi[2]) + coarse.voxel
    target = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0])
    top_views = initial.select_top_views(views, target)
    n_top = sum(initial.tilt_deg(p, target) <= 25.0 for p, _ in views)
    cov = preflight.coverage(cal.poses, target)  # welke foto's ontbreken er rond het object?
    diag["dekking"] = cov
    write_overlays()
    log(f"object gelokaliseerd: {hi[0] - lo[0]:.0f} x {hi[1] - lo[1]:.0f} mm, hoogte ≤ {z_top:.0f} mm")
    # V12 (v0.12): meer voorwerpen op de mat, of een onderdeel tegen of over de rand van de mat
    for b in coarse.others:
        log(f"nog een voorwerp op de mat bij ({b.center[0]:.0f}, {b.center[1]:.0f}) mm, ~{b.size[0]:.0f} x "
            f"{b.size[1]:.0f} mm: niet verwerkt")
    if coarse.others:
        where = "; ".join(f"bij ({b.center[0]:.0f}, {b.center[1]:.0f}) mm, ~{b.size[0]:.0f} x {b.size[1]:.0f} mm"
                          for b in coarse.others[:3])
        warnings.append(f"er {'ligt' if len(coarse.others) == 1 else 'liggen'} nog {len(coarse.others)} "
                        f"voorwerp{'' if len(coarse.others) == 1 else 'en'} op de mat ({where}). Alleen het onderdeel "
                        "waar de foto's van boven op gericht zijn is verwerkt; haal de rest van de mat, want vlak "
                        "naast het onderdeel verstoort een ander voorwerp de rand.")
    if coarse.chosen is not None and coarse.chosen.at_edge:
        warnings.append("het onderdeel ligt tegen of over de rand van de mat: daarbuiten ziet de verwerking geen mat "
                        "en dus geen rand. Leg het onderdeel midden op de mat, met minstens 2 cm mat eromheen.")
    if opts.debug_images:
        imwrite(dbg / "lokalisatie.png",
                    debug.hull_image(coarse, board_bounds, (lo[0] - 6, hi[0] + 6, lo[1] - 6, hi[1] + 6)))

    # 5. startmodel uit de bovenaanzichten (hoogte waarop ze samenvallen, met terugvalopties)
    if n_top == 0:
        write_debug()
        tilts = sorted(initial.tilt_deg(p, target) for p, _ in views)
        raise ScanError("Geen bovenaanzichten gevonden: maak ook 4-6 foto's recht boven het object "
                        f"(de steilste foto kijkt nu onder {tilts[0]:.0f}° naar het object; nodig is ≤ 25°)."
                        + _advice(cov))
    log(f"bovenaanzichten: {len(top_views)} gebruikt (kijkhoek "
        + ", ".join(f"{initial.tilt_deg(p, target):.0f}°" for p, _ in top_views) + ")")
    try:
        init = initial.locate(views, cam.K, coarse, board_bounds, log=log)
    except initial.InitError as e:
        if opts.debug_images:
            fp = initial.footprint(top_views, cam.K, max(2.0, 0.3 * z_top),
                                   (lo[0] - 6, hi[0] + 6, lo[1] - 6, hi[1] + 6))
            imwrite(dbg / "bovenaanzicht.png", debug.footprint_image(fp))
        write_debug({"startmodel": {"fout": str(e), "pogingen": e.details}})
        raise ScanError("Geen objectcontour gevonden in de foto's recht van boven. "
                        + NO_CONTOUR_HELP.format(debug=dbg) + _advice(cov)) from e
    warnings += init.notes
    sweep = init.sweep
    if sweep is not None and sweep.best is not None:
        log(f"startcontour: {init.footprint.method}, bovenvlak op ~{init.part.height:.1f} mm"
            + ("" if sweep.informative else " (bovenaanzichten te gelijk voor een hoogteschatting)"))

    # 6. hoogte zoeken en contour bijwerken tot het stabiel is, daarna fitten op alle silhouetten
    part = init.part
    h_max = max(z_top, 1.5 * part.height + 5.0)
    for it in range(3):
        vd = silhouette.prepare(views, cam.K, part, z_max=h_max + 3)
        h_old = part.height
        if it == 0:
            part = silhouette.fit_height(part, cam.K, vd, h_max=h_max)
        else:
            part = silhouette.fit_height(part, cam.K, vd, h_max=h_max, h_min=max(0.5, h_old - 4), step=0.5)
        part = init.at_height(cam.K, coarse, part.height)
        if abs(part.height - h_old) < 0.5:
            break
    if opts.debug_images:
        imwrite(dbg / "bovenaanzicht.png", debug.footprint_image(init.footprint))
    write_debug({"startmodel": {
        "methode": init.footprint.method, "hoogte_mm": round(part.height, 3), "pogingen": init.notes,
        "hoogtezoektocht": None if sweep is None else {
            "hoogtes_mm": sweep.heights.round(2).tolist(), "overeenstemming": sweep.agreement.round(4).tolist(),
            "oppervlak_mm2": sweep.area.round(1).tolist(), "beste_mm": sweep.best, "informatief": sweep.informative},
    }})
    part0 = part  # het startmodel; na een duwtje (V27) begint de fit hier opnieuw

    def fit_shape(views):
        """De pixelfit vanaf het startmodel: vorm, afrondingen, overbodige hoekpunten weg, gemiste gaten erbij."""
        vd = silhouette.prepare(views, cam.K, part0, z_max=part0.height * 1.4 + 4)
        part, energy, evals = _refine_start(part0, cam.K, vd, opts.max_evals, log=log)
        probed, e_probed, changed = silhouette.probe_fillets(part, cam.K, vd, energy)
        if changed:  # een afronding die de kompaszoektocht vanuit een (bijna) scherpe hoek niet vond
            probed, e_probed, _ = silhouette.refine(probed, cam.K, vd, max_evals=200)
            log(f"afrondingen opnieuw bepaald: energie {energy:.0f} → {e_probed:.0f}")
            part, energy = probed, e_probed
        part, energy, _ = _simplify_outline(part, cam.K, vd, energy, log=log)
        part, energy, notches = _notch_alternatives(part, cam.K, vd, energy, log=log)
        part, energy = _add_missed_holes(part, cam.K, vd, energy, log=log)
        return vd, part, energy, evals, notches

    vd, part, energy, evals, notches = fit_shape(views)
    oc = part.outer.outline()
    center = np.array([*(oc.min(axis=0) + oc.max(axis=0)) / 2, part.height / 2])
    mm_per_px = _object_distance(cal.poses.values(), center) / cam.K[0, 0]
    # randfit (V2): subpixel-verfijning op de randafstanden, met de covariantie voor de U95 (V3)
    t_fit = time.time()

    def fit_log(m):
        log(f"{m} ({time.time() - t_fit:.0f} s)")

    ef = _edge_fit(part, cam.K, vd, mm_per_px, log=fit_log)
    # V27 (v0.12): het onderdeel tussendoor even aangestoten? Dan liggen de foto's erna net iets anders dan die ervoor.
    # De poses van de foto's erna verschuiven dan met het onderdeel mee. Een nieuw duwtje: de hele fit opnieuw (de
    # eerste fit was een compromis, met bijvoorbeeld een spookgat waar de ene groep foto's mat zag). Daarna wordt
    # dezelfde sprong nog verfijnd: het compromis vervormde het model, en dan meet de eerste ronde maar een deel
    nudges: dict[str, float] = {}  # per sprong (de laatste foto ervoor) de grootte, opgeteld over de rondes
    diag["duwtje"] = []
    order = [n for n, _ in images]
    for _ in range(NUDGE_ROUNDS):
        info: dict = {}
        t_nudge = time.time()
        nd = _find_nudge(ef.part if ef.accepted else part, cam.K, vd, ef, order, info)
        diag["duwtje"].append(info)
        if nd is None:
            log(f"duwtje: geen (toets {info.get('toets', 0):.1f}, sprong {info.get('grootte_mm', 0):.3f} mm; "
                f"{time.time() - t_nudge:.0f} s)")
            break
        again = nd.before in nudges
        log(f"onderdeel verschoven na foto {nd.before}: {nd.motion[0]:+.2f}, {nd.motion[1]:+.2f} mm, "
            f"{math.degrees(nd.motion[2]):+.2f}° (toets {nd.t:.0f}); de poses van de "
            f"{sum(p.name in set(nd.moved) for p, _ in views)} foto's erna "
            + ("verder gecorrigeerd" if again else "gecorrigeerd, alles opnieuw gefit"))
        before = (views, top_views, vd, part, energy, evals, notches, ef)
        top_names = {p.name for p, _ in top_views}
        views = placement.apply_nudge(views, nd)
        top_views = [v for v in views if v[0].name in top_names]
        if again:
            start = ef.part if ef.accepted else part
            vd = silhouette.prepare(views, cam.K, start, z_max=start.height * 1.4 + 4)
            part, energy, _ = silhouette.refine(start, cam.K, vd, max_evals=max(300, opts.max_evals // 2))
        else:
            vd, part, energy, evals, notches = fit_shape(views)
        t_fit = time.time()
        ef = _edge_fit(part, cam.K, vd, mm_per_px, log=fit_log)
        # alleen als de fit er duidelijk beter door past (een verfijning: niet slechter): een sprong in de
        # verschuivingen die van iets anders komt (een maskerfout in een reeks foto's) maakt de fit niet beter
        c0, c1 = _fit_cost(before[-1]), _fit_cost(ef)
        e0 = before[4]  # energie
        info["verlies"] = [round(c, 4) if math.isfinite(c) else None for c in (c0, c1)]
        info["energie"] = [round(e0), round(energy)]
        gain = 1.0 if again else NUDGE_GAIN
        better = c1 < gain * c0 if math.isfinite(c0) or math.isfinite(c1) else energy < gain * e0  # zonder randfit
        if not better:
            log(f"duwtje niet gecorrigeerd: de fit past daarmee niet beter (verlies {c0:.4f} → {c1:.4f}, energie "
                f"{e0:.0f} → {energy:.0f})")
            views, top_views, vd, part, energy, evals, notches, ef = before
            break
        log(f"na de correctie: verlies {c0:.4f} → {c1:.4f}, energie {e0:.0f} → {energy:.0f}")
        nudges[nd.before] = nudges.get(nd.before, 0.0) + nd.size_mm
    diag["duwtjes"] = {k: round(v, 3) for k, v in nudges.items()}
    if nudges:
        warnings.append("het onderdeel is tijdens het fotograferen verschoven (aangestoten?): " + "; ".join(
            f"na foto {k} ~{v:.1f} mm" for k, v in nudges.items()) + ". De foto's erna zijn daarvoor "
            "gecorrigeerd; controleer de maten, of maak de foto's opnieuw zonder het onderdeel aan te raken.")
    # V19: past een prisma wel? Een trede of afschuining geeft anders een stil compromis (vooral in de hoogte).
    # V17: zo'n vorm dan als model proberen, en de randfit en de toets opnieuw
    shape = _prism_check(ef, part) if ef.extra.get("problem") is not None else None
    if shape is not None:
        diag["vormmodel"] = {}
        alt = _non_prism(part, cam.K, vd, energy, shape, mm_per_px, log=log, info=diag["vormmodel"])
        if alt is not None:
            part, energy = alt
            t_fit = time.time()
            ef = _edge_fit(part, cam.K, vd, mm_per_px, log=fit_log)
            shape = _prism_check(ef, part) if ef.extra.get("problem") is not None else None
    # V16: een verzinking is een ring rond een gat in de foto's van boven; dan als model, en de randfit opnieuw. Ook
    # als de randfit niet gebruikt is: een verzonken gat laat in schuine foto's meer doorkijken dan een gewoon gat,
    # en dan loopt de diameter van dat gat in de randfit tegen de grens van het vertrouwensgebied
    found_on = ef.part if ef.accepted else part
    if found_on.holes:
        notes: list[str] = []
        weak = {i for (kind, i), e in ef.extra.get("evidence", {}).items() if kind == "gat" and e.weak}
        sunk = countersink.detect(found_on, cam.K, vd, log=notes.append, weak=weak)
        if notes:
            diag["verzinkingen"] = notes
        trial = found_on.copy()
        for i, dk in sunk.items():
            trial.holes[i] = replace(trial.holes[i], csk=dk)
        if sunk and trial.is_valid():
            t_fit = time.time()
            ef_csk = _edge_fit(trial, cam.K, vd, mm_per_px, log=fit_log)
            if ef_csk.accepted:
                ef, part = ef_csk, trial
                log("verzinking herkend: " + ", ".join(f"gat {i + 1} Ø {ef.part.holes[i].csk:.2f} x "
                                                      f"{profile.CSK_ANGLE_DEG:.0f}°" for i in sorted(sunk)))
            else:
                log(f"verzinking niet gebruikt: {ef_csk.note}")
    # V16 (v0.10): door een gat met een kamerboring kijk je in de schuine foto's veel verder dan door een gewoon gat
    # (het doorgaande gat is korter). Per gat de silhouetten met en zonder kamer vergeleken (counterbore.detect), dan
    # als model, en de randfit opnieuw
    found_on = ef.part if ef.accepted else part
    if found_on.holes:
        notes = []
        sunk2: dict[int, tuple[float, float]] = {}  # verzinkingen die alleen de silhouetten zien (v0.11)
        bored = counterbore.detect(found_on, cam.K, vd, log=notes.append, sunk=sunk2)
        if notes:
            diag["kamerboringen"] = notes
        trial = found_on.copy()
        for i, (dk, t, d) in bored.items():
            trial.holes[i] = replace(trial.holes[i], d=d, csk=0.0, cb=dk, cb_depth=t)
        if bored and trial.is_valid():
            t_fit = time.time()
            # de diepte uit het silhouet staat soms meer dan het vertrouwensgebied verkeerd (een gat met weinig
            # bewijs): dan vanaf die grens verder (_edge_fit)
            ef_cb = _edge_fit(trial, cam.K, vd, mm_per_px, log=fit_log)
            # een kamer die in de randfit (bijna) verdwijnt, was er geen
            min_depth = max(counterbore.CB_MIN_DEPTH, counterbore.CB_MIN_FRAC * trial.height)
            shallow = [i for i in bored if ef_cb.accepted and ef_cb.part.holes[i].cb_depth < min_depth]
            if shallow:
                log("kamerboring niet gebruikt: " + ", ".join(
                    f"gat {i + 1} maar {ef_cb.part.holes[i].cb_depth:.2f} mm diep" for i in shallow))
                for i in shallow:
                    bored.pop(i)
                trial = found_on.copy()
                for i, (dk, t, d) in bored.items():
                    trial.holes[i] = replace(trial.holes[i], d=d, csk=0.0, cb=dk, cb_depth=t)
                ef_cb = _edge_fit(trial, cam.K, vd, mm_per_px, log=fit_log) if bored else ef_cb
            if bored and ef_cb.accepted:
                ef, part = ef_cb, ef_cb.part
                log("kamerboring herkend: " + ", ".join(f"gat {i + 1} Ø {ef.part.holes[i].cb:.2f} x "
                                                      f"{ef.part.holes[i].cb_depth:.2f} diep" for i in sorted(bored)))
            elif bored:
                log(f"kamerboring niet gebruikt: {ef_cb.note}")
        # een verzinking die niet aan haar ring herkend werd, maar de doorkijk in de schuine foto's beter verklaart dan
        # een gewoon gat of een kamer: als verzinking, niet als ondiepe kamer
        found_on = ef.part if ef.accepted else part
        trial = found_on.copy()
        for i, (dk, d) in sunk2.items():
            if trial.holes[i].cb <= 0:
                trial.holes[i] = replace(trial.holes[i], d=d, csk=dk)
        if sunk2 and trial.is_valid():
            t_fit = time.time()
            ef_csk = _edge_fit(trial, cam.K, vd, mm_per_px, log=fit_log)
            if ef_csk.accepted:
                ef, part = ef_csk, trial
                log("verzinking herkend aan de doorkijk: " + ", ".join(
                    f"gat {i + 1} Ø {ef.part.holes[i].csk:.2f} x {profile.CSK_ANGLE_DEG:.0f}°" for i in sorted(sunk2)))
            else:
                log(f"verzinking (doorkijk) niet gebruikt: {ef_csk.note}")
    # v0.11: blinde gaten. In het silhouet zijn ze er niet, in de grijswaarden wel (blindhole.py); dan als model en de
    # randfit opnieuw (hun maten komen alleen uit de randen in de grijswaarden)
    found_on = ef.part if ef.accepted else part
    notes, maybe = [], []
    blind = blindhole.detect(found_on, cam.K, vd, log=notes.append, maybe=maybe)
    if notes:
        diag["blinde_gaten"] = notes
    for x, y, d in maybe:
        warnings.append(f"mogelijk een blind gat op ({x:.1f}, {y:.1f}) mm (mat), Ø ~{d:.1f}: in de grijswaarden is de "
                        "bovenrand te zien, maar de bodem niet (te weinig contrast); niet gemodelleerd")
    trial = found_on.copy()
    trial.holes += blind
    if blind and trial.is_valid():
        t_fit = time.time()
        ef_b = _edge_fit(trial, cam.K, vd, mm_per_px, log=fit_log)
        n0 = len(found_on.holes)
        if ef_b.accepted:
            ef, part = ef_b, ef_b.part
            log("blind gat herkend: " + ", ".join(f"gat {n0 + k + 1} Ø {h.d:.2f} x {h.depth:.2f} diep"
                                                  for k, h in enumerate(ef.part.holes[n0:])))
        else:
            log(f"blind gat niet gebruikt: {ef_b.note}")
    if ef.accepted:
        part = ef.part
        energy = silhouette.energy(part, cam.K, vd)
    # Onscherpte in de foto's en het masker ronden ook scherpe hoeken af (ARCHITECTURE.md §7.3): op
    # gerenderde scans komt een scherpe hoek uit de randfit als een afronding van 3,5-4 pixels
    # (ROUTE-A-VERBETERPUNTEN §3e). Afrondingen onder 4,5 pixels zijn dus niet te onderscheiden van scherp.
    r_min = max(0.8, 4.5 * mm_per_px)
    small_slot_r = [i for i, s in enumerate(part.slots) if s.kind == "rechthoek" and 0 < s.r < r_min]
    if (part.outer.kind == "polygon" and np.any((part.outer.fillets > 0) & (part.outer.fillets < r_min))) \
            or small_slot_r:
        if part.outer.kind == "polygon":
            part.outer.fillets[part.outer.fillets < r_min] = 0.0
        for i in small_slot_r:
            part.slots[i] = replace(part.slots[i], r=0.0)
        warnings.append(f"afrondingen kleiner dan {r_min:.1f} mm zijn bij deze resolutie niet te "
                        "onderscheiden van een scherpe hoek en als scherp gemodelleerd")
    stats_fit = silhouette.view_stats(part, cam.K, vd)
    log(f"model gefit: hoogte {part.height:.3f} mm, {len(part.holes)} gat(en), "
        f"silhouet-IoU mediaan {stats_fit['iou_median']:.4f}")
    issues = _quality_issues(part, stats_fit, evals, opts.max_evals, mm_per_px, notches)
    if shape is not None:
        diag["prisma"] = shape.details
        issues += [f"geen 2,5D-vorm? {m}" for m in shape.issues]
    ghosts = unseen_holes(part, cam.K, top_views)
    if ghosts:
        issues.append(f"{len(ghosts)} gat(en) waardoor in geen bovenaanzicht mat te zien is: mogelijk spookgaten "
                      "(controleer ze; bij een oude mat v1 kan dit ook een echt gat boven een egaal zwart vak zijn)")
    # V14 (v0.12): plekken waar de foto's van boven iets anders zien dan het model: een gemiste inham, uitstulping of
    # gat, of iets dat tegen het onderdeel aan ligt
    clusters = holes.residual_clusters(part, cam.K, vd)
    diag["restclusters"] = clusters
    for c in clusters[:3]:
        what = ("mat waar het model materiaal heeft (een gemiste inham of een gemist gat)" if c["soort"] == "inham"
                else "materiaal buiten het model (een gemiste uitstulping, of iets dat tegen het onderdeel aan ligt)")
        issues.append(f"de foto's van boven zien op ({c['x']:.1f}, {c['y']:.1f}) mm (mat) {what}, ~{c['oppervlak']:.0f} "
                      f"mm² in {c['fotos']} foto's")
    # en stukken buitenrand die het model niet volgt (een vorm die het mist of anders heeft, zoals een kleine
    # rechthoekige inham die als V-vorm in het model kwam)
    misfit_info: dict = {}
    misfits = edgefit.contour_misfit(part, cam.K, vd, mm_per_px, prob=ef.extra.get("problem"), info=misfit_info)
    diag["randafwijkingen"] = misfits
    diag["randafwijking_max"] = misfit_info
    for m in misfits[:3]:
        issues.append(f"de rand bij ({m['x']:.1f}, {m['y']:.1f}) mm (mat) ligt in de foto's over {m['lengte']:.1f} mm "
                      f"tot {abs(m['afwijking']):.2f} mm verder naar {'buiten' if m['afwijking'] > 0 else 'binnen'} dan "
                      "in het model: een vorm die het model mist of anders heeft")
    if stats_fit["iou_median"] < 0.9:
        write_debug({"kwaliteit": issues})
        raise ScanError("Het gevonden model past niet bij de foto's (silhouet-IoU mediaan "
                        f"{stats_fit['iou_median']:.2f}). " + NO_CONTOUR_HELP.format(debug=dbg))
    if issues:
        log("LET OP, resultaat onbetrouwbaar: " + "; ".join(issues))
        warnings[:0] = [f"onbetrouwbaar: {i}" for i in issues]
    write_debug({"kwaliteit": issues})  # ook bij een gelukte scan: de vormtoets (prisma) en de redenen
    if not cov["compleet"]:
        warnings.append("fotoset onvolledig: " + " ".join(cov["advies"]))

    # 7. onzekerheid, werkassenstelsel, snappen (de printschaal zit al in de poses)
    # V8 (v0.12): een slagschaduw naast het onderdeel in de meeste foto's (masks._cast_shadow)
    shadow = float(np.median([m.shadow for _, m in views])) if views else 0.0
    cast_shadow = shadow >= SHADOW_WARN
    if cast_shadow:
        warnings.append(f"slagschaduw naast het onderdeel (in de meeste foto's {100 * shadow:.0f}% van de mat er vlak "
                        "omheen duidelijk donkerder): de rand aan de schaduwkant kan iets te ver naar buiten liggen, "
                        "buitenmaten tot ~0,2 mm te groot. De U95 is daarvoor ruimer. Gebruik diffuus licht (geen "
                        "lamp of zon recht op de mat) voor de beste nauwkeurigheid")
    unc = cadmodel.estimate_uncertainty(mm_per_px, len(vd), n_top)
    # printschaal: zonder gemeten meetlijnen is de schaal van de print niet bekend (printers wijken 0,1-1% af)
    unc.scale_rel = uncertainty.SCALE_REL_MEASURED if opts.scale_measured else uncertainty.SCALE_REL_ASSUMED
    scale_note = ("printschaal gemeten aan de meetlijnen (0,05%)" if opts.scale_measured else
                  "printschaal niet gemeten, meetlijnen niet opgegeven (0,3%)")
    part_pf, angle, shift = cadmodel.to_part_frame(part)
    budget, unc_method = None, "indicatief (resolutie en aantal foto's), " + scale_note
    if ef.accepted:  # V3: σ per maat uit de jackknife van de randfit, plus systematiek en printschaal
        t_jk = time.time()
        C = edgefit.jackknife(ef)
        log(f"onzekerheid per maat: jackknife over groepen foto's ({time.time() - t_jk:.0f} s)" if C is not None
            else "onzekerheid per maat: jackknife mislukt, indicatieve U95")
        if C is not None:
            prob = ef.extra["problem"]
            budget = uncertainty.Budget(uncertainty.Sensitivity(prob.build, ef.x, C, angle, shift), mm_per_px,
                                        unc.scale_rel)
            unc_method = "per maat: jackknife over groepen foto's (randfit), systematiek, " + scale_note
            if cast_shadow:
                budget.extra_px.update(uncertainty.SHADOW_PX)
                unc_method += ", slagschaduw"
    # bewijs rond gaten en sleuven (edgefit.evidence): zonder bewijs rondom zijn maat en plaats onzekerder
    evidence = ef.extra.get("evidence") or {}
    slot_label = [n for n, _ in cadmodel.slot_names(part_pf)]
    ev_diag = {}
    for (kind, i), e in sorted(evidence.items()):
        label = f"gat {i + 1}" if kind == "gat" else slot_label[i]
        ev_diag[label] = {"randpunten met bewijs": round(e.fraction, 3), "vergroting maat": round(e.amp_size, 2),
                          "vergroting plaats": round(e.amp_pos, 2), "zonder bewijs": e.weak}
        if e.weak:
            warnings.append(f"{label}: te weinig bewijs rond de rand (zwart op zwart, of wit dat in de foto's is "
                            "afgekapt: langs de rand is bijna nergens bruikbare mat te zien): maat en plaats komen uit "
                            "de pixelfit, met een ruime U95. Controleer ze, of leg het onderdeel anders op de mat")
    if ev_diag:
        write_debug({"bewijs binnenvormen": ev_diag})
    snapped, snaps = cadmodel.snap_part(part_pf, unc, threshold=opts.snap_threshold, imperial=opts.imperial,
                                        budget=budget, evidence=evidence)
    if budget is not None:
        unc = _effective_uncertainty(unc, snaps, budget.scale_rel)
    if snapped.outer.kind == "polygon" and not snapped.outer.is_valid():
        warnings.append("gesnapte contour was ongeldig; ongesnapte maten gebruikt")
        snapped, snaps = part_pf, [s.__class__(**{**s.__dict__, "value": s.measured, "snapped": False})
                                   for s in snaps]
    # controle: past het gesnapte model nog bij de foto's?
    c, s_ = math.cos(-angle), math.sin(-angle)
    back = snapped.transformed(-angle, -(np.array([[c, -s_], [s_, c]]) @ shift))
    stats_snap = silhouette.view_stats(back, cam.K, vd)
    if stats_snap["iou_median"] < stats_fit["iou_median"] - 0.01:
        warnings.append("het gesnapte model wijkt merkbaar af van de silhouetten; controleer de gesnapte maten")

    # 8. CAD-model en export
    model = cadmodel.build(snapped)
    if not model.val().isValid():
        warnings.append("CAD-kernel meldt een ongeldige solid")
    cadmodel.export(model, str(out / "model"))
    script_path = out / "model.py"
    script_path.write_text(cadmodel.script(snapped, snaps, {"scan": scan_name}), encoding="utf-8")

    # 9. rapport
    elapsed = time.time() - t_start
    te = snapped.top_edge
    summary = {
        "objectklasse": "2,5D (extrusie met doorgaande gaten)" + (
            f", bovenrand rondom {'afgeschuind' if te.kind == 'afschuining' else 'afgerond'}" if te else "")
        + (f", {len(snapped.steps)} trede" if snapped.steps else "")
        + (f", {n_sunk} verzonken gat(en)" if (n_sunk := sum(h.csk > 0 for h in snapped.holes)) else "")
        + (f", {n_cb} kamerboring(en)" if (n_cb := sum(h.cb > 0 for h in snapped.holes)) else "")
        + (f", {n_bl} blind(e) gat(en)" if (n_bl := sum(h.blind for h in snapped.holes)) else ""),
        "contour": "cirkel" if snapped.outer.kind == "circle" else f"polygoon, {snapped.outer.n} randen",
        "gaten": len(snapped.holes),
        "foto's gebruikt": f"{len(views)} van {len(images)} (waarvan {n_top} bovenaanzicht)",
        "kleur": (f"{len(chroma)} foto('s) in kleur: kleur telt mee als bewijs voor het object" if chroma
                  else "grijswaarden (geen kleur)"),
        "camera": (f"{camera_label}: " if camera_label else "") + f"f = {cam.K[0, 0]:.1f} px"
                  + (f" (σ {100 * cam.f_std_rel:.2f}%)" if np.isfinite(cam.f_std_rel) else "")
                  + f", reprojectiefout {cam.rms_px:.3f} px",
        "betrouwbaarheid": "laag: " + "; ".join(issues) if issues else "normaal",
        "resolutie op het object": f"{mm_per_px:.3f} mm/pixel",
        "schaalbron": f"kalibratiemat {spec.label} ({spec.mat_id})" + (
            (f", printschaal gecorrigeerd (meetlijn X {100 * sx:.2f} mm, Y {100 * sy:.2f} mm)" if spec.is_scaled
             else ", meetlijnen gemeten: 100,0 mm (geen correctie nodig)") if opts.scale_measured
            else " (meetlijnen niet opgegeven: aangenomen 100,0 mm)"),
        "silhouet-IoU (gefit)": f"mediaan {stats_fit['iou_median']:.4f}, minimum {stats_fit['iou_min']:.4f}",
        "silhouet-IoU (gesnapt)": f"mediaan {stats_snap['iou_median']:.4f}",
        "geldige solid": "ja" if model.val().isValid() else "nee",
        "volume": f"{model.val().Volume():.1f} mm³",
        "rekentijd": f"{elapsed:.0f} s",
    }
    data = {
        "title": f"Scan '{scan_name}' — camtocad {__version__}",
        "camtocad": __version__,
        "summary": summary,
        "dimensions": [s.to_dict() for s in snaps],
        "geometry": {"gefit": part_pf.to_dict(), "gesnapt": snapped.to_dict(),
                     "assenstelsel": "werkassenstelsel (datum linksonder), mm"},
        "uncertainty_model": {**unc.to_dict(), "methode": unc_method},
        "warnings": warnings,
        "files": {"step": "model.step", "stl": "model.stl", "script": "model.py", "json": "report.json"},
        "frame": {"angle_rad": angle, "shift_mm": shift.tolist(), "printschaal": [sx, sy],
                  "beschrijving": "werkcoördinaten = R(angle) · mat-XY + shift; mat-XY in werkelijke mm "
                                  "(printschaal al verwerkt)"},
        "mat": {"naam": spec.name, "versie": spec.version, "mat_id": spec.mat_id},
        "quality": {"status": "onbetrouwbaar" if issues else "normaal", "issues": issues},
        "camera": cam.to_dict(),
        "poses": {n: p.to_dict() for n, p in cal.poses.items()},
        "fit": {"energy": energy, "evaluations": evals, **stats_fit},
    }
    report.write(out, data, snapped)
    log(f"klaar in {elapsed:.0f} s: {out / 'model.step'}")
    return {"summary": summary, "dimensions": data["dimensions"], "warnings": warnings,
            "part": snapped, "part_fitted": part_pf, "out_dir": str(out)}


def demo_part():
    """Demonstratieonderdeel: beugel 80 x 40 x 12 mm, R3-hoeken, twee M6-doorgangsgaten (Ø 6,6)."""
    import cadquery as cq
    return (cq.Workplane("XY").box(80, 40, 12, centered=(True, True, False)).edges("|Z").fillet(3)
            .faces(">Z").workplane().pushPoints([(-30, 0), (30, 0)]).hole(6.6))


def run_demo(out_dir: str | Path, log=print, seed: int = 5) -> dict:
    """Rendert een synthetische scan van het demo-onderdeel en verwerkt die met run_scan."""
    from .render import default_camera, place, render_scan

    out = Path(out_dir)
    photos = out / "fotos"
    photos.mkdir(parents=True, exist_ok=True)
    spec = get_spec("A4")
    log("synthetische scan renderen (beugel 80 x 40 x 12 mm, 2 x Ø 6,6, R3) ...")
    views = render_scan(place(demo_part(), spec, angle_deg=17.0, offset=(5, -8)), spec, default_camera(), seed=seed)
    for v in views:
        imwrite(photos / f"{v.name}.png", v.image)
    result = run_scan(photos, out / "resultaat", ScanOptions(mat="A4"), log=log, scan_name="demo")
    # de werkelijke maten in het formaat van `camtocad valideer` (validate.py)
    truth = {"naam": "demo: beugel 80 x 40 x 12", "mat": "A4",
             "maten": {"lengte": 80.0, "breedte": 40.0, "hoogte": 12.0, "gaten": [6.6, 6.6],
                       "hartafstanden": [60.0], "afrondingen": [3.0, 3.0, 3.0, 3.0]}}
    (out / "maten.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
