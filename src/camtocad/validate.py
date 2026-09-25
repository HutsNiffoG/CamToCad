"""Validatie (Fase 0): scans van echte onderdelen vergelijken met schuifmaatmetingen.

Per onderdeel een map met foto's en een `maten.json` met wat je met de schuifmaat hebt gemeten:

    validatie/
      beugel-alu/
        maten.json
        fotos/*.jpg          (of de foto's direct in de map)
      plaatje-zwart/
        ...

`camtocad valideer validatie/` verwerkt elke scan (of hergebruikt een eerder resultaat), koppelt
elke referentiemaat aan de overeenkomende maat van het model en rapporteert de fout, of die
binnen de opgegeven U95 valt, en per soort maat de gemiddelde afwijking (bias) en spreiding.
Samen laten die zien of de onzekerheid eerlijk is (±95% binnen U95) en of er iets systematisch
misgaat, zoals een printschaal die niet klopt.

`maten.json`:

    {
      "naam": "beugel aluminium",
      "mat": "auto",                   (optioneel: A4, A3, Letter, A4-v1, A3-v1)
      "meetlijn": [100.05, 99.95],     (optioneel: gemeten meetlijnen X en Y)
      "maten": {
        "lengte": 80.02,               (grootste buitenmaat)
        "breedte": 40.01,              (kleinste buitenmaat)
        "hoogte": 12.03,
        "diameter": 25.0,              (ronde onderdelen)
        "gaten": [6.62, 6.60],         (diameters)
        "hartafstanden": [60.01],      (tussen gatmiddens)
        "afrondingen": [3.0]           (stralen van afgeronde hoeken)
      }
    }
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cadhelpers import afgeronde_hoeken

REFERENCE = "maten.json"
KINDS = {  # soort → (enkelvoud/meervoud in maten.json, lengte-achtig voor de schaalcontrole)
    "lengte": ("lengte", True), "breedte": ("breedte", True), "hoogte": ("hoogte", False),
    "diameter": ("diameter", True), "gat": ("gaten", False), "hartafstand": ("hartafstanden", True),
    "afronding": ("afrondingen", False),
}
ALIASES = {"gat": "gat", "gaten": "gat", "hartafstand": "hartafstand", "hartafstanden": "hartafstand",
           "afronding": "afronding", "afrondingen": "afronding", "lengte": "lengte", "breedte": "breedte",
           "hoogte": "hoogte", "diameter": "diameter"}


@dataclass
class Comparison:
    soort: str
    referentie: float
    gefit: float | None = None  # maat van het ongesnapte model
    gesnapt: float | None = None  # maat van het gesnapte (CAD-)model
    u95: float | None = None
    opmerking: str = ""

    @property
    def fout(self) -> float | None:
        return None if self.gefit is None else self.gefit - self.referentie

    @property
    def fout_gesnapt(self) -> float | None:
        return None if self.gesnapt is None else self.gesnapt - self.referentie

    @property
    def binnen_u95(self) -> bool | None:
        return None if self.fout is None or self.u95 is None else abs(self.fout) <= self.u95

    def to_dict(self) -> dict:
        return {**asdict(self), "fout": self.fout, "fout_gesnapt": self.fout_gesnapt, "binnen_u95": self.binnen_u95}


@dataclass
class ScanValidation:
    naam: str
    map: str
    status: str  # ok | fout | geen foto's
    fout: str = ""
    betrouwbaarheid: str = ""
    vergelijkingen: list[Comparison] = field(default_factory=list)
    extra_gaten: int = 0  # gaten in het model die niet in de referentie staan

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vergelijkingen"] = [c.to_dict() for c in self.vergelijkingen]
        return d


# ----------------------------------------------------------------------------- referentie

def read_reference(path: str | Path) -> dict:
    """Leest maten.json; maten als lijsten per soort (enkelvoud of meervoud, getal of lijst)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("maten")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: 'maten' ontbreekt of is leeg")
    measures: dict[str, list[float]] = {}
    for key, value in raw.items():
        kind = ALIASES.get(key.lower())
        if kind is None:
            raise ValueError(f"{path}: onbekende maat '{key}' (bekend: {', '.join(sorted(set(ALIASES)))})")
        values = value if isinstance(value, list) else [value]
        try:
            measures.setdefault(kind, []).extend(float(v) for v in values)
        except (TypeError, ValueError):
            raise ValueError(f"{path}: '{key}' moet een getal of een lijst getallen zijn") from None
    rulers = data.get("meetlijn")
    if rulers is not None:
        rulers = [float(r) for r in (rulers if isinstance(rulers, list) else [rulers])]
        if len(rulers) not in (1, 2):
            raise ValueError(f"{path}: 'meetlijn' is één getal of [X, Y]")
        rulers = rulers * 2 if len(rulers) == 1 else rulers
    return {"naam": str(data.get("naam", Path(path).parent.name)), "mat": str(data.get("mat", "auto")),
            "meetlijn": rulers, "maten": measures}


# ----------------------------------------------------------------------------- modelmaten

def _outline(contour: dict) -> np.ndarray:
    if contour["soort"] == "cirkel":
        c, r = np.array(contour["middelpunt"]), contour["diameter"] / 2
        a = np.linspace(0, 2 * math.pi, 720, endpoint=False)
        return c + r * np.column_stack([np.cos(a), np.sin(a)])
    pts = []
    for t1, m, t2, c, r in afgeronde_hoeken([tuple(p) for p in contour["hoekpunten"]]):
        if m is None:
            pts.append(t1)
            continue
        a1, a2 = math.atan2(t1[1] - c[1], t1[0] - c[0]), math.atan2(t2[1] - c[1], t2[0] - c[0])
        span = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        pts += [(c[0] + r * math.cos(a1 + s * span), c[1] + r * math.sin(a1 + s * span)) for s in np.linspace(0, 1, 16)]
    return np.array(pts, float)


def model_measures(geometry: dict, unc: dict) -> dict[str, list[tuple[float, float]]]:
    """Per soort maat de waarden van het model met hun U95: {soort: [(waarde, u95), ...]}."""
    scale = float(unc.get("scale_rel", 5e-4))

    def u95(base: float, value: float) -> float:
        return 2.0 * math.hypot(base, scale * abs(value))

    out: dict[str, list[tuple[float, float]]] = {k: [] for k in KINDS}
    h = float(geometry["hoogte"])
    out["hoogte"].append((h, u95(unc["height"], h)))
    contour = geometry["contour"]
    if contour["soort"] == "cirkel":
        d = float(contour["diameter"])
        out["diameter"].append((d, u95(unc["edge"] * math.sqrt(2), d)))
        ext = np.array([d, d])
    else:
        pts = _outline(contour)
        ext = pts.max(axis=0) - pts.min(axis=0)
        for r in (float(p[2]) for p in contour["hoekpunten"]):
            if r > 0:
                out["afronding"].append((r, 2.0 * float(unc["fillet"])))
    big, small = float(max(ext)), float(min(ext))
    out["lengte"].append((big, u95(unc["edge"] * math.sqrt(2), big)))
    out["breedte"].append((small, u95(unc["edge"] * math.sqrt(2), small)))
    holes = geometry.get("gaten", [])
    for g in holes:
        out["gat"].append((float(g["d"]), u95(unc["hole_d"], g["d"])))
    for a, b in combinations(holes, 2):
        dist = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
        out["hartafstand"].append((dist, u95(unc["hole_xy"] * math.sqrt(2), dist)))
    return out


def _assign(ref: list[float], model: list[float]) -> list[int | None]:
    """Koppelt referentiewaarden aan modelwaarden (minimale totale afwijking); None = geen partner."""
    if not ref or not model:
        return [None] * len(ref)
    cost = np.abs(np.subtract.outer(np.asarray(ref), np.asarray(model)))
    rows, cols = linear_sum_assignment(cost)
    out: list[int | None] = [None] * len(ref)
    for r, c in zip(rows, cols):
        out[r] = int(c)
    return out


def compare(reference: dict, report: dict) -> tuple[list[Comparison], int]:
    """Vergelijkt de referentiematen met het gefitte en het gesnapte model uit report.json."""
    geo, unc = report["geometry"], report["uncertainty_model"]
    fitted, snapped = model_measures(geo["gefit"], unc), model_measures(geo["gesnapt"], unc)
    rows: list[Comparison] = []
    for kind, refs in reference["maten"].items():
        model_vals = [v for v, _ in fitted[kind]]
        for r, m in zip(refs, _assign(refs, model_vals)):
            if m is None:
                note = {"gat": "gat ontbreekt in het model", "diameter": "model is niet rond",
                        "afronding": "hoek is scherp in het model"}.get(kind, "niet in het model")
                rows.append(Comparison(kind, r, opmerking=note))
                continue
            snap_val = snapped[kind][m][0] if m < len(snapped[kind]) else None
            rows.append(Comparison(kind, r, fitted[kind][m][0], snap_val, fitted[kind][m][1]))
    extra = max(0, len(fitted["gat"]) - len(reference["maten"].get("gat", [])))
    return rows, extra


# ----------------------------------------------------------------------------- scans

def find_parts(root: str | Path) -> list[Path]:
    root = Path(root)
    if (root / REFERENCE).exists():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / REFERENCE).exists())


def _photo_dir(part: Path) -> Path:
    return part / "fotos" if (part / "fotos").is_dir() else part


def validate_part(part: Path, rerun: bool = False, log=print, run=None) -> ScanValidation:
    from .pipeline import IMAGE_EXT, ScanOptions, run_scan

    run = run or run_scan
    ref = read_reference(part / REFERENCE)
    out = ScanValidation(ref["naam"], str(part), "ok")
    photos = _photo_dir(part)
    if not any(p.suffix.lower() in IMAGE_EXT for p in photos.iterdir()):
        out.status, out.fout = "geen foto's", f"geen foto's in {photos}"
        return out
    result_dir = part / "resultaat"
    report_path = result_dir / "report.json"
    stale = not report_path.exists() or any(
        p.stat().st_mtime > report_path.stat().st_mtime for p in photos.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if rerun or stale:
        rulers = ref["meetlijn"] or [100.0, 100.0]
        opts = ScanOptions(mat=ref["mat"], mat_scale=(rulers[0] / 100.0, rulers[1] / 100.0))
        log(f"== {part.name}: scan verwerken ...")
        try:
            run(photos, result_dir, opts, log=lambda m: log(f"   {m}"), scan_name=part.name)
        except ValueError as e:  # ScanError: meenemen in het overzicht, doorgaan met de volgende
            out.status, out.fout = "fout", str(e)
            report_path.unlink(missing_ok=True)
            return out
    else:
        log(f"== {part.name}: eerder resultaat gebruikt (--opnieuw om opnieuw te verwerken)")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if "geometry" not in report:
        out.status, out.fout = "fout", "report.json is van een oudere versie: verwerk opnieuw (--opnieuw)"
        return out
    out.betrouwbaarheid = str(report.get("summary", {}).get("betrouwbaarheid", ""))
    out.vergelijkingen, out.extra_gaten = compare(ref, report)
    return out


def summarize(results: list[ScanValidation]) -> dict:
    """Totalen per soort: bias, spreiding, grootste fout, aandeel binnen U95; plus de schaalcontrole."""
    rows = [c for r in results for c in r.vergelijkingen if c.fout is not None]
    per_kind = {}
    for kind in KINDS:
        errs = np.array([c.fout for c in rows if c.soort == kind])
        if len(errs) == 0:
            continue
        inside = [c.binnen_u95 for c in rows if c.soort == kind]
        per_kind[kind] = {"n": len(errs), "bias": float(errs.mean()), "rms": float(np.sqrt((errs ** 2).mean())),
                          "max_abs": float(np.abs(errs).max()), "binnen_u95": float(np.mean(inside))}
    rel = [c.fout / c.referentie for c in rows if KINDS[c.soort][1] and c.referentie > 20.0]
    missing = sum(1 for r in results for c in r.vergelijkingen if c.fout is None)
    out = {"scans": len(results), "scans_ok": sum(r.status == "ok" for r in results), "maten": len(rows),
           "niet_gevonden": missing, "extra_gaten": sum(r.extra_gaten for r in results),
           "binnen_u95": float(np.mean([c.binnen_u95 for c in rows])) if rows else None,
           "per_soort": per_kind, "schaal_bias_pct": 100.0 * float(np.mean(rel)) if len(rel) >= 3 else None}
    notes = []
    if out["binnen_u95"] is not None and len(rows) >= 10:
        if out["binnen_u95"] < 0.9:
            notes.append(f"Maar {100 * out['binnen_u95']:.0f}% van de fouten valt binnen U95 (verwacht ~95%): de "
                         "opgegeven onzekerheid is te optimistisch.")
        elif out["binnen_u95"] > 0.99 and len(rows) >= 30:
            notes.append("Alle fouten vallen ruim binnen U95: de onzekerheid is mogelijk te ruim opgegeven.")
    if out["schaal_bias_pct"] is not None and abs(out["schaal_bias_pct"]) > 0.15:
        notes.append(f"Lengtes wijken gemiddeld {out['schaal_bias_pct']:+.2f}% af: controleer de meetlijnen van de "
                     "mat (printschaal) of geef ze op in maten.json.")
    for kind, s in per_kind.items():
        if s["n"] >= 3 and abs(s["bias"]) > 2 * s["rms"] / math.sqrt(s["n"]) and abs(s["bias"]) > 0.05:
            notes.append(f"{kind}: systematisch {s['bias']:+.3f} mm (rms {s['rms']:.3f} mm over {s['n']} maten).")
    out["opmerkingen"] = notes
    return out


def validate(root: str | Path, rerun: bool = False, log=print, run=None) -> tuple[list[ScanValidation], dict]:
    parts = find_parts(root)
    if not parts:
        raise ValueError(f"Geen {REFERENCE} gevonden in {root} of in de mappen daaronder")
    results = [validate_part(p, rerun, log, run) for p in parts]
    return results, summarize(results)


# ----------------------------------------------------------------------------- uitvoer

def _fmt(v: float | None, nd: int = 3, sign: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def report_lines(results: list[ScanValidation], summary: dict) -> list[str]:
    lines = []
    for r in results:
        head = f"{r.naam} ({r.status}" + (f", betrouwbaarheid {r.betrouwbaarheid}" if r.betrouwbaarheid else "") + ")"
        lines.append(head)
        if r.fout:
            lines.append(f"   {r.fout}")
        for c in r.vergelijkingen:
            flag = "" if c.binnen_u95 is None else ("ok" if c.binnen_u95 else "BUITEN U95")
            lines.append(f"   {c.soort:12s} ref {c.referentie:8.3f}  model {_fmt(c.gefit):>8s}  "
                         f"fout {_fmt(c.fout, sign=True):>7s}  U95 {_fmt(c.u95):>6s}  gesnapt {_fmt(c.gesnapt):>8s}  "
                         f"{flag} {c.opmerking}".rstrip())
        if r.extra_gaten:
            lines.append(f"   {r.extra_gaten} extra gat(en) in het model")
    s = summary
    lines += ["", f"Totaal: {s['scans_ok']} van {s['scans']} scans verwerkt, {s['maten']} maten vergeleken"
                  + (f", {100 * s['binnen_u95']:.0f}% binnen U95" if s["binnen_u95"] is not None else "")
                  + (f", {s['niet_gevonden']} niet in het model" if s["niet_gevonden"] else "")
                  + (f", {s['extra_gaten']} extra gaten" if s["extra_gaten"] else "")]
    for kind, k in s["per_soort"].items():
        lines.append(f"   {kind:12s} n {k['n']:3d}  bias {k['bias']:+.3f}  rms {k['rms']:.3f}  max {k['max_abs']:.3f}"
                     f"  binnen U95 {100 * k['binnen_u95']:.0f}%")
    if s["schaal_bias_pct"] is not None:
        lines.append(f"   lengtes gemiddeld {s['schaal_bias_pct']:+.3f}% (printschaal)")
    lines += [f"- {n}" for n in s["opmerkingen"]]
    return lines


def write_report(results: list[ScanValidation], summary: dict, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    data = {"samenvatting": summary, "scans": [r.to_dict() for r in results]}
    jpath = out / "validatie.json"
    jpath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    body = []
    for r in results:
        rows = "".join(
            f"<tr><td>{html.escape(c.soort)}</td><td>{c.referentie:.3f}</td><td>{_fmt(c.gefit)}</td>"
            f"<td>{_fmt(c.fout, sign=True)}</td><td>{_fmt(c.u95)}</td><td>{_fmt(c.gesnapt)}</td>"
            f"<td class=\"{'' if c.binnen_u95 is None else 'ok' if c.binnen_u95 else 'bad'}\">"
            f"{'-' if c.binnen_u95 is None else 'ja' if c.binnen_u95 else 'nee'}</td>"
            f"<td>{html.escape(c.opmerking)}</td></tr>" for c in r.vergelijkingen)
        link = (Path(r.map) / "resultaat" / "report.html").resolve()
        rel = f'<a href="{html.escape(link.relative_to(out.resolve()).as_posix())}">rapport</a>' if (
            link.exists() and link.is_relative_to(out.resolve())) else ""
        body.append(f"<h2>{html.escape(r.naam)} <small>{html.escape(r.status)} {rel}</small></h2>"
                    + (f"<p class=bad>{html.escape(r.fout)}</p>" if r.fout else "")
                    + (f"<p class=muted>betrouwbaarheid: {html.escape(r.betrouwbaarheid)}</p>"
                       if r.betrouwbaarheid else "")
                    + (f"<div class=wrap><table><thead><tr><th>Maat</th><th>Referentie</th><th>Model</th><th>Fout</th>"
                       f"<th>U95</th><th>Gesnapt</th><th>Binnen U95</th><th></th></tr></thead><tbody>{rows}"
                       "</tbody></table></div>" if rows else ""))
    kinds = "".join(f"<tr><td>{html.escape(k)}</td><td>{v['n']}</td><td>{v['bias']:+.3f}</td><td>{v['rms']:.3f}</td>"
                    f"<td>{v['max_abs']:.3f}</td><td>{100 * v['binnen_u95']:.0f}%</td></tr>"
                    for k, v in summary["per_soort"].items())
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in summary["opmerkingen"]) or "<li>geen</li>"
    cover = "-" if summary["binnen_u95"] is None else f"{100 * summary['binnen_u95']:.0f}%"
    page = f"""<!doctype html>
<html lang="nl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cam-to-CAD validatie</title>
<style>
:root {{ --bg:#fff; --fg:#1d232b; --muted:#5b6673; --line:#d8dde3; --ok:#2e7d4f; --bad:#b3261e; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#15191e; --fg:#e6e9ed; --muted:#9aa4af; --line:#2c333b;
  --ok:#7cc79a; --bad:#f2a39c; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:980px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:1.5rem; margin:0 0 4px; }} h2 {{ font-size:1.1rem; margin:28px 0 8px; }}
small, .muted {{ color:var(--muted); font-weight:normal; }} .ok {{ color:var(--ok); }} .bad {{ color:var(--bad); }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
.wrap {{ overflow-x:auto; }}
</style></head><body><main>
<h1>Cam-to-CAD validatie</h1>
<p class="muted">{summary['scans_ok']} van {summary['scans']} scans verwerkt · {summary['maten']} maten ·
binnen U95: {cover}</p>
<h2>Per soort maat</h2>
<div class="wrap"><table><thead><tr><th>Maat</th><th>n</th><th>Bias (mm)</th><th>RMS (mm)</th><th>Max (mm)</th>
<th>Binnen U95</th></tr></thead><tbody>{kinds}</tbody></table></div>
<h2>Opmerkingen</h2><ul>{notes}</ul>
{''.join(body)}
<p class="muted">Fout = model (ongesnapt) min referentie. U95 is de onzekerheid die de scan zelf opgeeft; bij een
eerlijke onzekerheid valt ongeveer 95% van de fouten daarbinnen.</p>
</main></body></html>"""
    hpath = out / "validatie.html"
    hpath.write_text(page, encoding="utf-8")
    return {"json": jpath, "html": hpath}
