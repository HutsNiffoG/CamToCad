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
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import __version__, cadmodel, calib, debug, hull, initial, masks, report, silhouette
from .mat import get_spec, rasterize_board

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


class ScanError(ValueError):
    """Fout met een voor de gebruiker begrijpelijke uitleg."""


@dataclass
class ScanOptions:
    mat: str = "A4"
    max_side: int = 2000  # werkresolutie: langste zijde in pixels
    snap_threshold: float = 0.8
    imperial: bool = False
    max_evals: int = 1500
    debug_images: bool = True
    mat_scale: float = 1.0  # gemeten / nominale lengte van de meetlijn (printschaal van de mat)


def load_images(folder: str | Path, max_side: int = 2000, log=print) -> list[tuple[str, np.ndarray]]:
    """Leest alle foto's als grijswaarden, zonder EXIF-rotatie (één consistent sensorformaat)."""
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXT)
    out = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
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


def _check_other_mats(images, spec) -> None:
    """Weinig mat gevonden: is het misschien een ander matformaat dan gekozen?"""
    from .mat import PRESETS

    sample = images[:: max(1, len(images) // 6)][:6]
    for other in PRESETS.values():
        if other.name == spec.name:
            continue
        board = calib.make_board(other)
        det = calib.make_detector(board)
        hits = sum(calib.detect(img, board, name, det) is not None for name, img in sample)
        if hits >= max(2, len(sample) // 2):
            raise ScanError(f"De foto's tonen de {other.name}-mat, niet de {spec.name}-mat: kies mat {other.name} "
                            f"(opdrachtregel: --mat {other.name})")


def _quality_issues(part, stats: dict, evals: int, max_evals: int) -> list[str]:
    """Signalen dat het model niet klopt, ook al is er een model uitgekomen."""
    issues = []
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
    spec = get_spec(opts.mat)
    if not isinstance(images, list):
        scan_name = scan_name or Path(images).name
        images = load_images(images, opts.max_side, log)
    if len(images) < 6:
        raise ScanError(f"Te weinig foto's ({len(images)}); maak er minstens 20, rondom en recht van boven")
    warnings: list[str] = []

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
    if len(dets) < 6:
        _check_other_mats(images, spec)
    if not 0.9 < opts.mat_scale < 1.1:
        raise ScanError(f"Meetlijn {100 * opts.mat_scale:.1f} mm wijkt te veel af van 100 mm: print de mat "
                        "opnieuw op 100% (werkelijke grootte)")
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

    # 3. objectmaskers
    raster = rasterize_board(spec, 10.0, 3.0)
    views = []
    for name, pose in cal.poses.items():
        img = calib.undistort(lookup[name], cam)
        pred, valid = masks.predict_background(raster, cam.K, pose, (cam.width, cam.height))
        views.append((pose, masks.classify(img, pred, valid)))
    top_views: list = []
    diag = {"camtocad": __version__, "opencv": cv2.__version__, "detector_bias_px": bias.tolist(),
            "camera": cam.to_dict(), "geweigerd": cal.rejected}

    def write_debug(extra: dict | None = None) -> None:
        if not opts.debug_images:
            return
        diag.update(extra or {})
        diag["fotos"] = debug.view_table(views, {d.name: d for d in dets}, {p.name for p, _ in top_views})
        debug.write_json(dbg / "diagnose.json", diag)

    def write_overlays() -> None:
        if not opts.debug_images:
            return
        others = [v for v in views if all(v[0] is not t[0] for t in top_views)]
        for pose, m in top_views[:6] + others[:: max(1, len(others) // 4)][:4]:
            cv2.imwrite(str(dbg / f"masker_{Path(pose.name).stem}.jpg"),
                        debug.mask_overlay(calib.undistort(lookup[pose.name], cam), m))

    # 4. grove visual hull: waar staat het object ongeveer? (alleen lokaliseren en een bovengrens)
    board_bounds = (0.0, spec.board_w_mm, 0.0, spec.board_h_mm)
    try:
        coarse = hull.reconstruct(views, cam.K, board_bounds, fine=False)
    except ValueError as e:
        top_views = initial.select_top_views(views)
        write_overlays()
        write_debug()
        raise ScanError(f"{e}. " + NO_CONTOUR_HELP.format(debug=dbg)) from e
    ijk = np.argwhere(coarse.occ)
    lo = coarse.origin + ijk.min(axis=0) * coarse.voxel
    hi = coarse.origin + ijk.max(axis=0) * coarse.voxel
    z_top = float(hi[2]) + coarse.voxel
    target = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0])
    top_views = initial.select_top_views(views, target)
    n_top = sum(initial.tilt_deg(p, target) <= 25.0 for p, _ in views)
    write_overlays()
    log(f"object gelokaliseerd: {hi[0] - lo[0]:.0f} x {hi[1] - lo[1]:.0f} mm, hoogte ≤ {z_top:.0f} mm")
    if opts.debug_images:
        cv2.imwrite(str(dbg / "lokalisatie.png"),
                    debug.hull_image(coarse, board_bounds, (lo[0] - 6, hi[0] + 6, lo[1] - 6, hi[1] + 6)))

    # 5. startmodel uit de bovenaanzichten (hoogte waarop ze samenvallen, met terugvalopties)
    if n_top == 0:
        write_debug()
        tilts = sorted(initial.tilt_deg(p, target) for p, _ in views)
        raise ScanError("Geen bovenaanzichten gevonden: maak ook 4-6 foto's recht boven het object "
                        f"(de steilste foto kijkt nu onder {tilts[0]:.0f}° naar het object; nodig is ≤ 25°)")
    log(f"bovenaanzichten: {len(top_views)} gebruikt (kijkhoek "
        + ", ".join(f"{initial.tilt_deg(p, target):.0f}°" for p, _ in top_views) + ")")
    try:
        init = initial.locate(views, cam.K, coarse, board_bounds, log=log)
    except initial.InitError as e:
        if opts.debug_images:
            fp = initial.footprint(top_views, cam.K, max(2.0, 0.3 * z_top),
                                   (lo[0] - 6, hi[0] + 6, lo[1] - 6, hi[1] + 6))
            cv2.imwrite(str(dbg / "bovenaanzicht.png"), debug.footprint_image(fp))
        write_debug({"startmodel": {"fout": str(e), "pogingen": e.details}})
        raise ScanError("Geen objectcontour gevonden in de foto's recht van boven. "
                        + NO_CONTOUR_HELP.format(debug=dbg)) from e
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
        cv2.imwrite(str(dbg / "bovenaanzicht.png"), debug.footprint_image(init.footprint))
    write_debug({"startmodel": {
        "methode": init.footprint.method, "hoogte_mm": round(part.height, 3), "pogingen": init.notes,
        "hoogtezoektocht": None if sweep is None else {
            "hoogtes_mm": sweep.heights.round(2).tolist(), "overeenstemming": sweep.agreement.round(4).tolist(),
            "oppervlak_mm2": sweep.area.round(1).tolist(), "beste_mm": sweep.best, "informatief": sweep.informative},
    }})
    vd = silhouette.prepare(views, cam.K, part, z_max=part.height * 1.4 + 4)
    part, energy, evals = silhouette.refine(part, cam.K, vd, max_evals=opts.max_evals, log=log)
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
    issues = _quality_issues(part, stats_fit, evals, opts.max_evals)
    if stats_fit["iou_median"] < 0.9:
        write_debug({"kwaliteit": issues})
        raise ScanError("Het gevonden model past niet bij de foto's (silhouet-IoU mediaan "
                        f"{stats_fit['iou_median']:.2f}). " + NO_CONTOUR_HELP.format(debug=dbg))
    if issues:
        log("LET OP, resultaat onbetrouwbaar: " + "; ".join(issues))
        warnings[:0] = [f"onbetrouwbaar: {i}" for i in issues]

    # 7. onzekerheid, werkassenstelsel, printschaal, snappen
    unc = cadmodel.estimate_uncertainty(mm_per_px * opts.mat_scale, len(vd), n_top)
    part_pf, angle, shift = cadmodel.to_part_frame(part)
    if opts.mat_scale != 1.0:  # mat niet op precies 100% geprint: alles schaalt mee (gelijkvormig)
        part_pf = part_pf.scaled(opts.mat_scale)
    snapped, snaps = cadmodel.snap_part(part_pf, unc, threshold=opts.snap_threshold, imperial=opts.imperial)
    if snapped.outer.kind == "polygon" and not snapped.outer.is_valid():
        warnings.append("gesnapte contour was ongeldig; ongesnapte maten gebruikt")
        snapped, snaps = part_pf, [s.__class__(**{**s.__dict__, "value": s.measured, "snapped": False})
                                   for s in snaps]
    # controle: past het gesnapte model nog bij de foto's?
    c, s_ = math.cos(-angle), math.sin(-angle)
    back = snapped.scaled(1.0 / opts.mat_scale).transformed(-angle, -(np.array([[c, -s_], [s_, c]]) @ shift))
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
        "resolutie op het object": f"{mm_per_px * opts.mat_scale:.3f} mm/pixel",
        "schaalbron": f"kalibratiemat {spec.mat_id}" + (
            f", printschaal gecorrigeerd (meetlijn {100 * opts.mat_scale:.2f} mm)" if opts.mat_scale != 1.0
            else " (meetlijn niet opgegeven: aangenomen 100,0 mm)"),
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
        "uncertainty_model": unc.to_dict(),
        "warnings": warnings,
        "files": {"step": "model.step", "stl": "model.stl", "script": "model.py", "json": "report.json"},
        "frame": {"angle_rad": angle, "shift_mm": shift.tolist(), "mat_scale": opts.mat_scale,
                  "beschrijving": "werkcoördinaten = mat_scale · (R(angle) · mat-XY + shift)"},
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
        cv2.imwrite(str(photos / f"{v.name}.png"), v.image)
    result = run_scan(photos, out / "resultaat", ScanOptions(mat="A4"), log=log, scan_name="demo")
    truth = {"hoogte": 12.0, "x-maat": 80.0, "y-maat": 40.0, "gat Ø": 6.6, "afronding R": 3.0}
    (out / "waarheid.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return result
