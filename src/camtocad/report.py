"""Meetrapport: JSON voor verdere verwerking en een zelfstandige HTML-pagina met bovenaanzicht."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from .profile import Part2p5D


def _svg_top_view(part: Part2p5D, width: int = 560) -> str:
    outline = part.outer.outline(3.0)
    pts = [outline] + [np.column_stack([h.x + h.d / 2 * np.cos(a), h.y + h.d / 2 * np.sin(a)])
                       for h in part.holes for a in [np.linspace(0, 2 * np.pi, 72)]]
    pts += [s.outline(5.0) for s in part.slots] + [np.asarray(c, float) for c in part.cutouts]
    allp = np.vstack(pts)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    span = max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
    pad = 0.08 * span
    scale = width / (span + 2 * pad)
    h_px = (hi[1] - lo[1] + 2 * pad) * scale

    def tr(p):
        return (p[:, 0] - lo[0] + pad) * scale, h_px - (p[:, 1] - lo[1] + pad) * scale

    def path(p):
        x, y = tr(p)
        return "M " + " L ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y)) + " Z"

    a = np.linspace(0, 2 * np.pi, 72)
    body = [f'<path d="{path(outline)}" class="part"/>']
    labels: list[str] = []  # boven op de rest
    te = part.top_edge
    if te is not None:  # binnenrand van een afschuining of afronding rondom (V17)
        inner = part.outer.inset(te.size)
        ring = inner.outline(3.0) if inner.kind == "polygon" else np.column_stack(
            [inner.center[0] + inner.radius * np.cos(a), inner.center[1] + inner.radius * np.sin(a)])
        body.append(f'<path d="{path(ring)}" class="edge"/>')
    for st in part.steps:  # lijn van een trede, met de hoogte aan de lage kant
        q = part.step_crossings(st)
        if q is None:
            continue
        x, y = tr(q)
        body.append(f'<path d="M {x[0]:.1f},{y[0]:.1f} L {x[1]:.1f},{y[1]:.1f}" class="edge"/>')
        n = st.normal()
        # naast de lijn, aan de lage kant en dicht bij een einde (minder kans op een gat); SVG: y omlaag
        lx, ly = tr((0.85 * q[1] + 0.15 * q[0])[None, :])
        lx, ly = lx[0] + 6.0 * n[0], ly[0] - 14.0 * n[1] + 4.0
        anchor = "start" if n[0] > 0.5 else "end" if n[0] < -0.5 else "middle"
        labels.append(f'<text x="{lx:.1f}" y="{ly:.1f}" class="lbl" style="text-anchor:{anchor}">'
                      f'trede {st.height:.2f}</text>')
    for h in part.holes:
        body.append(f'<path d="{path(np.column_stack([h.x + h.d / 2 * np.cos(a), h.y + h.d / 2 * np.sin(a)]))}" '
                    f'class="hole"/>')
        x, y = tr(np.array([[h.x, h.y]]))
        body.append(f'<text x="{x[0]:.1f}" y="{y[0] - h.d / 2 * scale - 6:.1f}" class="lbl">'
                    f'Ø {h.d:.2f}</text>')
    for s in part.slots:
        body.append(f'<path d="{path(s.outline(5.0))}" class="hole"/>')
        x, y = tr(np.array([[s.x, s.y]]))
        label = (f"sleuf {s.width:.2f} × {s.length:.2f}" if s.kind == "sleuf"
                 else f"{s.length:.2f} × {s.width:.2f}, R{s.r:.1f}")
        body.append(f'<text x="{x[0]:.1f}" y="{y[0]:.1f}" class="lbl">{label}</text>')
    for c in part.cutouts:
        body.append(f'<path d="{path(np.asarray(c, float))}" class="hole"/>')
    ox, oy = tr(np.array([[0.0, 0.0]]))
    body.append(f'<circle cx="{ox[0]:.1f}" cy="{oy[0]:.1f}" r="4" class="datum"/>')
    body += labels
    return (f'<svg viewBox="0 0 {width} {h_px:.0f}" width="100%" role="img" '
            f'aria-label="Bovenaanzicht van het model">{"".join(body)}</svg>')


def write(out_dir: Path, data: dict, part: Part2p5D) -> dict[str, Path]:
    out_dir = Path(out_dir)
    jpath = out_dir / "report.json"
    jpath.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    rows = []
    for s in data.get("dimensions", []):
        status = "gesnapt" if s["snapped"] else "gemeten"
        rows.append(
            f"<tr><td>{html.escape(s['name'])}</td><td>{s['measured']:.3f}</td><td>± {s['u95']:.3f}</td>"
            f"<td><b>{s['value']:.3f}</b></td><td>{status}</td><td>{html.escape(s['reason'])}</td></tr>"
        )
    info = data.get("summary", {})
    method = str(data.get("uncertainty_model", {}).get("methode", "indicatief (resolutie en aantal foto's)"))
    info_rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
                        for k, v in info.items())
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in data.get("warnings", [])) or "<li>geen</li>"
    files = "".join(f'<li><a href="{html.escape(v)}">{html.escape(v)}</a></li>'
                    for v in data.get("files", {}).values())
    page = f"""<!doctype html>
<html lang="nl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cam-to-CAD meetrapport</title>
<style>
:root {{ --bg:#fff; --fg:#1d232b; --muted:#5b6673; --line:#d8dde3; --part:#dfe9f5; --stroke:#2b5d9b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#15191e; --fg:#e6e9ed; --muted:#9aa4af; --line:#2c333b;
  --part:#1f3350; --stroke:#8fb8ea; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:980px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:1.5rem; margin:0 0 4px; }} h2 {{ font-size:1.1rem; margin:28px 0 8px; }}
.muted {{ color:var(--muted); }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
.wrap {{ overflow-x:auto; }}
svg .part {{ fill:var(--part); stroke:var(--stroke); stroke-width:1.5; }}
svg .hole {{ fill:var(--bg); stroke:var(--stroke); stroke-width:1.2; }}
svg .edge {{ fill:none; stroke:var(--stroke); stroke-width:1; stroke-dasharray:4 3; }}
svg .lbl {{ fill:var(--fg); font-size:12px; text-anchor:middle; }}
svg .datum {{ fill:#d9534f; }}
</style></head><body><main>
<h1>Cam-to-CAD meetrapport</h1>
<p class="muted">{html.escape(data.get('title', ''))}</p>
<h2>Bovenaanzicht (werkassenstelsel)</h2>
{_svg_top_view(part)}
<p class="muted">Rode stip: datum (oorsprong). Maten in mm.</p>
<h2>Maten</h2>
<div class="wrap"><table><thead><tr><th>Maat</th><th>Gemeten</th><th>U95</th><th>Model</th><th>Status</th>
<th>Reden</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>Samenvatting</h2>
<div class="wrap"><table>{info_rows}</table></div>
<h2>Waarschuwingen</h2><ul>{warnings}</ul>
<h2>Bestanden</h2><ul>{files}</ul>
<p class="muted">U95 = onzekerheid met 95% dekking; {html.escape(method)}. Getoetst op gerenderde scans; valideer
kritieke maten met een schuifmaat.</p>
</main></body></html>"""
    hpath = out_dir / "report.html"
    hpath.write_text(page, encoding="utf-8")
    return {"json": jpath, "html": hpath}
