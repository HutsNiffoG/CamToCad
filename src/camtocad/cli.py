"""Opdrachtregel: `camtocad mat | scan | demo | server`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="camtocad",
                                     description="Cam-to-CAD route A: lokaal en open source van foto's naar CAD.")
    parser.add_argument("--version", action="version", version=f"camtocad {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("mat", help="printbare kalibratiemat maken (PDF, PNG-voorbeeld, JSON)")
    p.add_argument("--formaat", choices=["A4", "A3"], default="A4")
    p.add_argument("--uit", default=".", help="uitvoermap (standaard: huidige map)")

    p = sub.add_parser("scan", help="map met foto's van een onderdeel op de mat verwerken")
    p.add_argument("fotos", help="map met foto's (JPG/PNG)")
    p.add_argument("--uit", help="uitvoermap (standaard: <fotos>_cad)")
    p.add_argument("--mat", choices=["A4", "A3"], default="A4")
    p.add_argument("--max-zijde", type=int, default=2000, help="werkresolutie, langste zijde in pixels")
    p.add_argument("--snapdrempel", type=float, default=0.8, help="minimale kans om een maat te snappen (0-1)")
    p.add_argument("--inch", action="store_true", help="snappen naar inchmaten in plaats van mm")

    p = sub.add_parser("demo", help="synthetische testscan renderen en verwerken (zonder camera)")
    p.add_argument("--uit", default="camtocad-demo")

    p = sub.add_parser("server", help="lokale webserver: foto's uploaden vanaf je telefoon via wifi")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--poort", type=int, default=8000)
    p.add_argument("--data", default=str(Path.home() / "camtocad-data"))
    p.add_argument("--token", default=None, help="toegangscode (standaard: willekeurig gegenereerd)")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "mat":
            from .mat import write_mat
            paths = write_mat(args.formaat, args.uit)
            print(f"Kalibratiemat {args.formaat} geschreven:")
            for kind, path in paths.items():
                print(f"  {kind}: {path}")
            print("Print de PDF op 100% (werkelijke grootte) en meet beide 100 mm-lijnen na.")
        elif args.cmd == "scan":
            from .pipeline import ScanOptions, run_scan
            out = args.uit or f"{Path(args.fotos).resolve()}_cad"
            opts = ScanOptions(mat=args.mat, max_side=args.max_zijde, snap_threshold=args.snapdrempel,
                               imperial=args.inch)
            result = run_scan(args.fotos, out, opts)
            _print_result(result)
        elif args.cmd == "demo":
            from .pipeline import run_demo
            result = run_demo(args.uit)
            _print_result(result)
            print("Werkelijke maten: 80 x 40 x 12 mm, R3, 2 x Ø 6,6 op (10, 20) en (70, 20).")
        elif args.cmd == "server":
            from .server.app import serve
            serve(args.host, args.poort, Path(args.data), args.token)
    except ValueError as e:  # ScanError en andere invoerfouten: nette melding, geen traceback
        print(f"Fout: {e}", file=sys.stderr)
        return 1
    return 0


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
