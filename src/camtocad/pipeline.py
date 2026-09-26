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
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import __version__, cadmodel, calib, debug, hull, initial, masks, preflight, profile, report, silhouette
from .imgio import imwrite, read_gray
from .mat import MatSpec, get_spec, rasterize_board

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
WORKERS = max(1, min(4, os.cpu_count() or 1))  # parallelle foto's (geheugen: ~250 MB per maskerberekening)


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
    # printschaal: gemeten / nominale lengte van de meetlijnen; één getal (beide richtingen) of (X, Y)
    mat_scale: float | tuple[float, float] = 1.0

    @property
    def scale_xy(self) -> tuple[float, float]:
        s = self.mat_scale
        if isinstance(s, (int, float)):
            return float(s), float(s)
        sx, sy = s
        return float(sx), float(sy)


def load_images(folder: str | Path, max_side: int = 2000, log=print) -> list[tuple[str, np.ndarray]]:
    """Leest alle foto's als grijswaarden, zonder EXIF-rotatie (één consistent sensorformaat)."""
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXT)
    out = []
    for p in paths:
        img = read_gray(p)  # ook met niet-ASCII-tekens in het pad (Windows)
        if img is None:
            log(f"  overgeslagen (onleesbaar): {p.name}")
            continue
        scale = max_side / max(img.shape)
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out.append((p.name, img))
    return out


def _object_distance(poses, center) -> float:
    return float(np.median([np.linalg.norm(p.center - center) for p in poses]))


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
        ring = circle_polygon((h.x, h.y), h.d / 2, 48)
        seen = judged = False
        for pose, m in top_views:
            uv = []
            for z in (part.height, 0.0):
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


def _simplify_outline(part, K: np.ndarray, vd: list, energy: float, max_turn_deg: float = 10.0,
                      max_edge_mm: float = 5.0, max_tries: int = 8, log=print):
    """Haalt hoekpunten weg die het model niet nodig heeft: een knik van een paar graden in een rechte
    rand, of een korte rand (bijv. een afschuining) op een hoek.

    De fit kan geen hoekpunten weghalen. Zo'n hoekpunt in de startcontour komt vaak van een stuk rand
    zonder bewijs (zwart op zwart bij een hoek) en is dan geen kenmerk van het onderdeel. Per kandidaat
    volgt een korte fit zonder dat hoekpunt; past het model dan even goed (de energie stijgt minder dan
    0,5% of één pixel per foto), dan blijft het weg. Een echt kenmerk, of een schaduw die het masker
    verkeerd laat lopen, heeft bewijs in de foto's: dan blijft het staan (en markeert de
    kwaliteitspoort een schaduw). Geeft (model, energie, aantal weggehaalde hoekpunten).
    """
    removed, tries = 0, 0
    while part.outer.kind == "polygon" and part.outer.n > 3 and tries < max_tries:
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
            if e_raw > energy + max(20 * tol, 0.5 * energy) or tries >= max_tries:
                break
            tries += 1
            cand, e_cand, _ = silhouette.refine(cand, K, vd, max_evals=200)
            if e_cand <= energy + tol:
                detail = f"knik van {turn[k]:.1f}°" if what == "knik" else f"korte rand van {lengths[k]:.1f} mm"
                log(f"{detail} uit de contour gehaald: het model past zonder even goed (energie {energy:.0f} → "
                    f"{e_cand:.0f})")
                part, energy, removed, changed = cand, e_cand, removed + 1, True
                break
        if not changed:
            break
    return part, energy, removed


def _quality_issues(part, stats: dict, evals: int, max_evals: int, mm_per_px: float = 0.25) -> list[str]:
    """Signalen dat het model niet klopt, ook al is er een model uitgekomen."""
    issues = []
    if part.outer.kind == "polygon":
        # een uitstulping of inham van een paar pixels (schaduw, rommelig masker) geeft korte randen
        V = part.outer.vertices()
        short_mm = max(2.0, 10.0 * mm_per_px)
        short = int(np.sum(np.linalg.norm(np.roll(V, -1, axis=0) - V, axis=1) < short_mm))
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
    if not isinstance(images, list):
        scan_name = scan_name or Path(images).name
        images = load_images(images, opts.max_side, log)
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
    try:
        cal = calib.calibrate(dets, spec)
    except ValueError as e:
        raise ScanError(str(e)) from e
    cam = cal.camera
    # foto's die staand in plaats van liggend (of andersom) zijn opgeslagen: terugdraaien naar het
    # sensorformaat; de draairichting is die waarbij de pose met deze camera het best klopt
    n_turned = 0
    for name, img in images:
        if name in cal.poses or img.shape != (cam.width, cam.height):
            continue
        best = None
        for code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            turned = cv2.rotate(img, code)
            d = calib.detect(turned, board, name, detector, bias)
            pose = calib.solve_pose(d, spec, cam) if d is not None and len(d.ids) >= 12 else None
            if pose is not None and (best is None or pose.rms_px < best[0].rms_px):
                best = (pose, turned, d)
        if best is not None and best[0].rms_px < 1.5:
            cal.poses[name], lookup[name] = best[0], best[1]
            cal.rejected.pop(name, None)
            dets.append(best[2])
            n_turned += 1
    if n_turned:
        log(f"{n_turned} foto('s) waren gedraaid opgeslagen en zijn teruggedraaid")
    for name, reason in cal.rejected.items():
        if reason.startswith("afwijkend formaat"):
            reason += " (andere camera/lens, zoom of bijgesneden?)"
        warnings.append(f"{name}: niet gebruikt ({reason})")
    log(f"camera gekalibreerd: f = {cam.K[0, 0]:.1f} px, reprojectiefout {cam.rms_px:.3f} px, "
        f"{len(cal.poses)} poses")
    # onscherpte per foto, gemeten aan de mat (preflight.py)
    blur = {d.name: preflight.measure_blur(lookup[d.name], d, spec) for d in dets if d.name in cal.poses}
    blurry = sorted(n for n, b in blur.items() if b is not None and b > preflight.BLUR_WARN)
    if blurry:
        warnings.append(f"{len(blurry)} foto('s) onscherp (σ > {preflight.BLUR_WARN:.1f} px): "
                        + ", ".join(blurry[:6]) + (" ..." if len(blurry) > 6 else ""))

    # 3. objectmaskers
    raster = rasterize_board(spec, 10.0, 3.0)

    def view_masks(pose):
        img = calib.undistort(lookup[pose.name], cam)
        pred, valid = masks.predict_background(raster, cam.K, pose, (cam.width, cam.height))
        depth = float((pose.R @ np.array([spec.size_mm[0] / 2, spec.size_mm[1] / 2, 0.0]) + pose.t)[2])
        return pose, masks.classify(img, pred, valid, px_per_mm=cam.K[0, 0] / max(depth, 1.0))

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:  # grote beeldbewerkingen: OpenCV en numpy geven de GIL vrij
        views = list(pool.map(view_masks, cal.poses.values()))
    top_views: list = []
    diag = {"camtocad": __version__, "opencv": cv2.__version__, "detector_bias_px": bias.tolist(),
            "camera": cam.to_dict(), "geweigerd": cal.rejected}

    def write_debug(extra: dict | None = None) -> None:
        if not opts.debug_images:
            return
        diag.update(extra or {})
        diag["fotos"] = debug.view_table(views, {d.name: d for d in dets}, {p.name for p, _ in top_views}, blur)
        debug.write_json(dbg / "diagnose.json", diag)

    def write_overlays() -> None:
        if not opts.debug_images:
            return
        others = [v for v in views if all(v[0] is not t[0] for t in top_views)]
        for pose, m in top_views[:6] + others[:: max(1, len(others) // 4)][:4]:
            imwrite(dbg / f"masker_{Path(pose.name).stem}.jpg",
                        debug.mask_overlay(calib.undistort(lookup[pose.name], cam), m))

    # 4. grove visual hull: waar staat het object ongeveer? (alleen lokaliseren en een bovengrens)
    board_bounds = (0.0, spec.size_mm[0], 0.0, spec.size_mm[1])
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
    vd = silhouette.prepare(views, cam.K, part, z_max=part.height * 1.4 + 4)
    max_evals = opts.max_evals
    n_features = (part.outer.n if part.outer.kind == "polygon" else 1) + len(part.holes) + len(part.cutouts)
    if n_features > 20:  # rommelige startcontour: het resultaat wordt toch 'onbetrouwbaar'; niet minutenlang fitten
        max_evals = min(max_evals, 300)
        log(f"rommelige startcontour ({n_features} randen, gaten en uitsparingen): korte verfijning")
    part, energy, evals = silhouette.refine(part, cam.K, vd, max_evals=max_evals, log=log)
    probed, e_probed, changed = silhouette.probe_fillets(part, cam.K, vd, energy)
    if changed:  # een afronding die de kompaszoektocht vanuit een (bijna) scherpe hoek niet vond
        probed, e_probed, _ = silhouette.refine(probed, cam.K, vd, max_evals=200)
        log(f"afrondingen opnieuw bepaald: energie {energy:.0f} → {e_probed:.0f}")
        part, energy = probed, e_probed
    part, energy, _ = _simplify_outline(part, cam.K, vd, energy, log=log)
    oc = part.outer.outline()
    center = np.array([*(oc.min(axis=0) + oc.max(axis=0)) / 2, part.height / 2])
    mm_per_px = _object_distance(cal.poses.values(), center) / cam.K[0, 0]
    # Onscherpte in de foto's rondt ook scherpe hoeken een fractie af (ARCHITECTURE.md §7.3):
    # afrondingen onder ~3 pixels zijn niet te onderscheiden van scherp en worden scherp.
    r_min = max(0.8, 3.0 * mm_per_px)
    if part.outer.kind == "polygon" and np.any((part.outer.fillets > 0) & (part.outer.fillets < r_min)):
        part.outer.fillets[part.outer.fillets < r_min] = 0.0
        warnings.append(f"afrondingen kleiner dan {r_min:.1f} mm zijn bij deze resolutie niet te "
                        "onderscheiden van een scherpe hoek en als scherp gemodelleerd")
    stats_fit = silhouette.view_stats(part, cam.K, vd)
    log(f"model gefit: hoogte {part.height:.3f} mm, {len(part.holes)} gat(en), "
        f"silhouet-IoU mediaan {stats_fit['iou_median']:.4f}")
    issues = _quality_issues(part, stats_fit, evals, opts.max_evals, mm_per_px)
    ghosts = unseen_holes(part, cam.K, top_views)
    if ghosts:
        issues.append(f"{len(ghosts)} gat(en) waardoor in geen bovenaanzicht mat te zien is: mogelijk spookgaten "
                      "(controleer ze; bij een oude mat v1 kan dit ook een echt gat boven een egaal zwart vak zijn)")
    if stats_fit["iou_median"] < 0.9:
        write_debug({"kwaliteit": issues})
        raise ScanError("Het gevonden model past niet bij de foto's (silhouet-IoU mediaan "
                        f"{stats_fit['iou_median']:.2f}). " + NO_CONTOUR_HELP.format(debug=dbg))
    if issues:
        log("LET OP, resultaat onbetrouwbaar: " + "; ".join(issues))
        warnings[:0] = [f"onbetrouwbaar: {i}" for i in issues]
    if not cov["compleet"]:
        warnings.append("fotoset onvolledig: " + " ".join(cov["advies"]))

    # 7. onzekerheid, werkassenstelsel, snappen (de printschaal zit al in de poses)
    unc = cadmodel.estimate_uncertainty(mm_per_px, len(vd), n_top)
    part_pf, angle, shift = cadmodel.to_part_frame(part)
    snapped, snaps = cadmodel.snap_part(part_pf, unc, threshold=opts.snap_threshold, imperial=opts.imperial)
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
    summary = {
        "objectklasse": "2,5D (extrusie met doorgaande gaten)",
        "contour": "cirkel" if snapped.outer.kind == "circle" else f"polygoon, {snapped.outer.n} randen",
        "gaten": len(snapped.holes),
        "foto's gebruikt": f"{len(cal.poses)} van {len(images)} (waarvan {n_top} bovenaanzicht)",
        "camera": f"f = {cam.K[0, 0]:.1f} px, reprojectiefout {cam.rms_px:.3f} px",
        "betrouwbaarheid": "laag: " + "; ".join(issues) if issues else "normaal",
        "resolutie op het object": f"{mm_per_px:.3f} mm/pixel",
        "schaalbron": f"kalibratiemat {spec.label} ({spec.mat_id})" + (
            f", printschaal gecorrigeerd (meetlijn X {100 * sx:.2f} mm, Y {100 * sy:.2f} mm)" if spec.is_scaled
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
        "uncertainty_model": unc.to_dict(),
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
