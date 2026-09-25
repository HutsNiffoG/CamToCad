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

from . import __version__, cadmodel, calib, hull, masks, report, silhouette
from .mat import get_spec, rasterize_board
from .profile import Part2p5D, from_footprint, regularize_angles

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


def _debug_overlay(img: np.ndarray, m: masks.ViewMasks) -> np.ndarray:
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32) * 0.6
    base[m.fg] = base[m.fg] * 0.5 + np.array([0, 140, 255]) * 0.5  # object: oranje
    base[m.bg] = base[m.bg] * 0.7 + np.array([255, 120, 0]) * 0.3  # zekere mat: blauw
    return np.clip(base, 0, 255).astype(np.uint8)


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

    # 1-2. mat herkennen, zelfkalibratie en poses
    board = calib.make_board(spec)
    detector = calib.make_detector(board)
    dets, lookup = [], dict(images)
    for name, img in images:
        d = calib.detect(img, board, name, detector)
        if d is None:
            warnings.append(f"{name}: kalibratiemat niet gevonden")
        else:
            dets.append(d)
    log(f"mat gevonden in {len(dets)} van {len(images)} foto's")
    try:
        cal = calib.calibrate(dets, spec)
    except ValueError as e:
        raise ScanError(str(e)) from e
    for name, reason in cal.rejected.items():
        warnings.append(f"{name}: niet gebruikt ({reason})")
    cam = cal.camera
    log(f"camera gekalibreerd: f = {cam.K[0, 0]:.1f} px, reprojectiefout {cam.rms_px:.3f} px, "
        f"{len(cal.poses)} poses")

    # 3. objectmaskers
    raster = rasterize_board(spec, 10.0, 3.0)
    views = []
    for name, pose in cal.poses.items():
        img = calib.undistort(lookup[name], cam)
        pred, valid = masks.predict_background(raster, cam.K, pose, (cam.width, cam.height))
        views.append((pose, masks.classify(img, pred, valid)))
    if opts.debug_images:
        dbg = out / "debug"
        dbg.mkdir(exist_ok=True)
        for (pose, m) in views[:: max(1, len(views) // 4)][:4]:
            cv2.imwrite(str(dbg / f"masker_{Path(pose.name).stem}.jpg"),
                        _debug_overlay(calib.undistort(lookup[pose.name], cam), m))

    # 4. grove visual hull: waar staat het object en hoe hoog is het ongeveer?
    try:
        coarse = hull.reconstruct(views, cam.K, (0, spec.board_w_mm, 0, spec.board_h_mm), fine=False)
    except ValueError as e:
        raise ScanError(str(e)) from e
    ijk = np.argwhere(coarse.occ)
    lo = coarse.origin + ijk.min(axis=0) * coarse.voxel
    hi = coarse.origin + ijk.max(axis=0) * coarse.voxel
    bounds = (lo[0] - 6, hi[0] + 6, lo[1] - 6, hi[1] + 6)
    z_top = float(hi[2]) + coarse.voxel
    log(f"object gelokaliseerd: {hi[0] - lo[0]:.0f} x {hi[1] - lo[1]:.0f} mm, hoogte ≤ {z_top:.0f} mm")

    # 5-6. startmodel uit bovenaanzichten, hoogte zoeken, fitten op alle silhouetten
    n_top = sum(silhouette.is_top_view(p) for p, _ in views)
    if n_top == 0:
        raise ScanError("Geen bovenaanzichten gevonden: maak ook 3-6 foto's recht van boven het object")

    def initial(height: float) -> Part2p5D:
        fp = silhouette.footprint_from_top_views(views, cam.K, height, bounds)
        outer, holes, cutouts = from_footprint(fp[0], fp[1], 0.25, lenient_holes=True)
        outer, _ = regularize_angles(outer)
        return Part2p5D(height, outer, holes, cutouts)

    part = initial(max(2.0, 0.6 * z_top))
    vd = silhouette.prepare(views, cam.K, part, z_max=z_top + 3)
    part = silhouette.fit_height(part, cam.K, vd, h_max=z_top)
    part = initial(part.height)
    vd = silhouette.prepare(views, cam.K, part, z_max=part.height * 1.4 + 4)
    part, energy, evals = silhouette.refine(part, cam.K, vd, max_evals=opts.max_evals, log=log)
    center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, part.height / 2])
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

    # 7. onzekerheid, werkassenstelsel, snappen
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
        "resolutie op het object": f"{mm_per_px:.3f} mm/pixel",
        "schaalbron": f"kalibratiemat {spec.mat_id}",
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
        "frame": {"angle_rad": angle, "shift_mm": shift.tolist(),
                  "beschrijving": "werkcoördinaten = R(angle) · mat-XY + shift"},
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
