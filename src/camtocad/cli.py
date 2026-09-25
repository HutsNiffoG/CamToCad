"""Opdrachtregel: `camtocad mat | controleer | scan | valideer | demo | server`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .mat import PRESETS, PRINTABLE

MATS = ["AUTO"] + list(PRESETS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="camtocad",
                                     description="Cam-to-CAD route A: lokaal en open source van foto's naar CAD.")
    parser.add_argument("--version", action="version", version=f"camtocad {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("mat", help="printbare kalibratiemat maken (PDF, PNG-voorbeeld, JSON)")
    p.add_argument("--formaat", type=str.upper, choices=[n.upper() for n in PRINTABLE], default="A4",
                   help="A4, A3 of Letter")
    p.add_argument("--uit", default=".", help="uitvoermap (standaard: huidige map)")

    p = sub.add_parser("scan", help="map met foto's van een onderdeel op de mat verwerken")
    p.add_argument("fotos", help="map met foto's (JPG/PNG)")
    p.add_argument("--uit", help="uitvoermap (standaard: <fotos>_cad)")
    p.add_argument("--mat", type=str.upper, choices=MATS, default="AUTO",
                   help="kalibratiemat; standaard herkend aan de markers in de foto's")
    p.add_argument("--max-zijde", type=int, default=2000, help="werkresolutie, langste zijde in pixels")
    p.add_argument("--snapdrempel", type=float, default=0.8, help="minimale kans om een maat te snappen (0-1)")
    p.add_argument("--inch", action="store_true", help="snappen naar inchmaten in plaats van mm")
    p.add_argument("--meetlijn", type=float, nargs="+", metavar="MM", default=[100.0],
                   help="gemeten lengte (mm) van de 100 mm-meetlijnen op de geprinte mat: X (onder) en eventueel "
                        "Y (links); één waarde geldt voor beide. Corrigeert de printschaal")

    p = sub.add_parser("controleer", help="fotoset snel controleren vóór het verwerken (mat, scherpte, dekking)")
    p.add_argument("fotos", help="map met foto's (JPG/PNG)")
    p.add_argument("--mat", type=str.upper, choices=MATS, default="AUTO")

    p = sub.add_parser("valideer", help="scans van echte onderdelen vergelijken met schuifmaatmetingen (maten.json)")
    p.add_argument("map", help="map met per onderdeel een submap met foto's en maten.json (of één zo'n map)")
    p.add_argument("--opnieuw", action="store_true",
                   help="alle scans opnieuw verwerken, ook als er al een resultaat is")
    p.add_argument("--eis-dekking", type=float, default=None, metavar="FRACTIE",
                   help="foutcode als minder dan deze fractie van de fouten binnen U95 valt (bijv. 0.9)")

    p = sub.add_parser("demo", help="synthetische testscan renderen en verwerken (zonder camera)")
    p.add_argument("--uit", default="camtocad-demo")

    p = sub.add_parser("server", help="lokale webserver: foto's uploaden vanaf je telefoon via wifi")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--poort", type=int, default=8000)
    p.add_argument("--data", default=str(Path.home() / "camtocad-data"))
    p.add_argument("--token", default=None, help="toegangscode (standaard: willekeurig gegenereerd)")

    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows-console (cp1252) kent o.a. '≤' niet
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        if args.cmd == "mat":
            from .mat import write_mat
            paths = write_mat(args.formaat, args.uit)
            print(f"Kalibratiemat {args.formaat} geschreven:")
            for kind, path in paths.items():
                print(f"  {kind}: {path}")
            print("Print de PDF op 100% (werkelijke grootte) en meet beide 100 mm-lijnen (X en Y) na.")
        elif args.cmd == "scan":
            from .pipeline import ScanOptions, run_scan
            out = args.uit or f"{Path(args.fotos).resolve()}_cad"
            opts = ScanOptions(mat=args.mat, max_side=args.max_zijde, snap_threshold=args.snapdrempel,
                               imperial=args.inch, mat_scale=_ruler_scale(args.meetlijn))
            result = run_scan(args.fotos, out, opts)
            _print_result(result)
        elif args.cmd == "controleer":
            from .preflight import check_folder, report_lines, summarize
            checks = check_folder(args.fotos, args.mat)
            if not checks:
                raise ValueError(f"Geen foto's (JPG/PNG) gevonden in {args.fotos}")
            print("\n".join(report_lines(checks, summarize(checks))))
        elif args.cmd == "valideer":
            from .validate import report_lines, validate, write_report
            results, summary = validate(args.map, rerun=args.opnieuw)
            root = Path(args.map)
            paths = write_report(results, summary, root if root.is_dir() else root.parent)
            print("\n".join(report_lines(results, summary)))
            print(f"\nRapport: {paths['html']}")
            if args.eis_dekking is not None and (summary["binnen_u95"] or 0.0) < args.eis_dekking:
                print(f"Dekking onder de eis van {100 * args.eis_dekking:.0f}%", file=sys.stderr)
                return 1
        elif args.cmd == "demo":
            from .pipeline import run_demo
            result = run_demo(args.uit)
            _print_result(result)
            print("Werkelijke maten: 80 x 40 x 12 mm, R3, 2 x Ø 6,6 op (10, 20) en (70, 20).")
            print(f"Vergelijken met de werkelijke maten: camtocad valideer {args.uit}")
        elif args.cmd == "server":
            from .server.app import serve
            serve(args.host, args.poort, Path(args.data), args.token)
    except ValueError as e:  # ScanError en andere invoerfouten: nette melding, geen traceback
        print(f"Fout: {e}", file=sys.stderr)
        return 1
    return 0


def _ruler_scale(values: list[float]) -> tuple[float, float]:
    if len(values) > 2:
        raise ValueError("--meetlijn: geef één waarde (X en Y gelijk) of twee (X en Y)")
    x = values[0]
    y = values[1] if len(values) > 1 else x
    return x / 100.0, y / 100.0


def _print_result(result: dict) -> None:
    print("\nMaten (werkassenstelsel, mm):")
    for d in result["dimensions"]:
        flag = "gesnapt" if d["snapped"] else "gemeten"
        print(f"  {d['name']:24s} {d['value']:9.3f}  ({flag}; gemeten {d['measured']:.3f} ± {d['u95']:.3f})")
    if result["warnings"]:
        print("\nWaarschuwingen:")
        for w in result["warnings"][:15]:
            print(f"  - {w}")
    print(f"\nUitvoer: {result['out_dir']} (model.step, model.stl, model.py, report.html)")


if __name__ == "__main__":
    raise SystemExit(main())
