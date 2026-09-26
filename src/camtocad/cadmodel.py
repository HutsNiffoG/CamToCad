"""Van gefit 2,5D-model naar CAD: werkassenstelsel, snappen, CadQuery-solid en parametrisch script.

Het werkassenstelsel legt de hoofdrichting van de randen langs X/Y en de oorsprong (datum) op
de linker- en onderrand. Maten worden vanaf die datum gesnapt, zoals een ontwerper maatvoert.
Het gegenereerde script gebruikt dezelfde bouwfunctie (cadhelpers.py) als de pipeline, dus het
levert exact hetzelfde model op.
"""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass
from datetime import date

import cadquery as cq
import numpy as np

from . import __version__, cadhelpers
from .profile import Hole, Part2p5D, Profile, dominant_angle
from .snapping import Snap, hole_candidates, length_candidates, radius_candidates, snap


@dataclass
class Uncertainty:
    """1σ-onzekerheden (mm) per soort maat; indicatief, afgeleid van resolutie en aantal foto's."""

    edge: float
    height: float
    hole_d: float
    hole_xy: float
    fillet: float
    scale_rel: float = 5e-4  # relatieve schaalonzekerheid (geverifieerde mat)

    def total(self, base: float, value: float) -> float:
        return math.hypot(base, self.scale_rel * abs(value))

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def estimate_uncertainty(mm_per_px: float, n_views: int, n_top: int, systematic: float | None = None) -> Uncertainty:
    """1σ per soort maat. Het toevallige deel neemt af met het aantal foto's; het systematische deel
    (maskerrand, fit, pose: in de stresstests ~0,2 px per rand, allemaal dezelfde kant op) niet.
    Nog te kalibreren op echte metingen (Fase 0)."""
    n_views, n_top = max(n_views, 1), max(n_top, 1)
    if systematic is None:
        systematic = max(0.2 * mm_per_px, 0.03)
    edge = math.hypot(0.35 * mm_per_px / math.sqrt(n_views), systematic)
    return Uncertainty(
        edge=edge, height=edge,
        hole_d=math.hypot(0.5 * mm_per_px / math.sqrt(n_top), systematic),
        hole_xy=math.hypot(0.35 * mm_per_px / math.sqrt(n_top), systematic),
        # een afronding bepaalt maar een klein stukje silhouet en de fout (onscherpte, rastering in de
        # hoek) is in alle foto's dezelfde kant op: niet kleiner met meer foto's (stresstests: ±0,8 px)
        fillet=math.hypot(0.8 * mm_per_px, 0.1),
    )


# ----------------------------------------------------------------------------- werkassenstelsel

def edge_axes(profile: Profile, tol: float = 1e-6) -> list[tuple[str | None, float]]:
    """Per rand ('x', positie) voor verticale randen, ('y', positie) voor horizontale, anders (None, 0)."""
    out = []
    for a, off in zip(profile.angles, profile.offsets):
        n = np.array([math.cos(a), math.sin(a)])
        c = float(n @ profile.center + off)  # n · p = c
        if abs(abs(n[0]) - 1) < tol:
            out.append(("x", c / n[0]))
        elif abs(abs(n[1]) - 1) < tol:
            out.append(("y", c / n[1]))
        else:
            out.append((None, 0.0))
    return out


def _set_edge_position(profile: Profile, k: int, axis: str, pos: float) -> None:
    n = np.array([math.cos(profile.angles[k]), math.sin(profile.angles[k])])
    comp = n[0] if axis == "x" else n[1]
    profile.offsets[k] = comp * pos - float(n @ profile.center)


def bolt_circle(holes: list[Hole], center=(0.0, 0.0), tol_r: float = 0.25,
                tol_deg: float = 1.5) -> tuple[float, float, int] | None:
    """Herkent ≥ 3 gelijke gaten op gelijke afstand van `center` en met gelijke hoekverdeling.

    Geeft (straal, hoek van het eerste gat, aantal) of None.
    """
    if len(holes) < 3 or max(h.d for h in holes) - min(h.d for h in holes) > 0.3:
        return None
    c = np.asarray(center, float)
    rel = np.array([[h.x, h.y] for h in holes]) - c
    radii = np.linalg.norm(rel, axis=1)
    if radii.max() - radii.min() > 2 * tol_r or radii.mean() < 1.0:
        return None
    ang = np.sort(np.arctan2(rel[:, 1], rel[:, 0]) % (2 * math.pi))
    n = len(holes)
    steps = np.diff(np.append(ang, ang[0] + 2 * math.pi))
    if np.max(np.abs(steps - 2 * math.pi / n)) > math.radians(tol_deg):
        return None
    start = float(np.angle(np.mean(np.exp(1j * n * ang))) / n)  # gemiddelde patroonfase
    return float(radii.mean()), start, n


def to_part_frame(part: Part2p5D) -> tuple[Part2p5D, float, np.ndarray]:
    """Draait en verschuift het model naar het werkassenstelsel; geeft (model, hoek, verschuiving)."""
    if part.outer.kind == "circle":
        # ronde delen: oorsprong in het middelpunt; een gatenpatroon begint op 0°
        pattern = bolt_circle(part.holes, part.outer.center)
        angle = -pattern[1] if pattern else 0.0
        c, s = math.cos(angle), math.sin(angle)
        shift = -(np.array([[c, -s], [s, c]]) @ part.outer.center)
        return part.transformed(angle, shift), angle, shift
    theta0 = dominant_angle(part.outer)
    # kies de hoek zo dat de langste rand horizontaal ligt
    rotated = part.transformed(-theta0, np.zeros(2))
    axes = edge_axes(rotated.outer, tol=1e-4)
    V = rotated.outer.vertices()
    lengths = np.linalg.norm(np.roll(V, -1, axis=0) - V, axis=1)
    longest = int(np.argmax(lengths))
    angle = -theta0
    if axes[longest][0] == "x":
        angle -= math.pi / 2
        rotated = part.transformed(angle, np.zeros(2))
        axes = edge_axes(rotated.outer, tol=1e-4)
    outline = rotated.outer.outline()
    xs = [p for a, p in axes if a == "x"]
    ys = [p for a, p in axes if a == "y"]
    x0 = min(xs) if xs else float(outline[:, 0].min())
    y0 = min(ys) if ys else float(outline[:, 1].min())
    shift = -np.array([x0, y0])
    return part.transformed(angle, shift), angle, shift


# ----------------------------------------------------------------------------- snappen

def _groups(values: list[float], tol: float) -> list[list[int]]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups: list[list[int]] = []
    for i in order:
        if groups and abs(values[i] - np.mean([values[j] for j in groups[-1]])) <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def snap_part(part: Part2p5D, unc: Uncertainty, *, threshold: float = 0.8,
              imperial: bool = False) -> tuple[Part2p5D, list[Snap]]:
    """Snapt hoogte, randposities, diameters, straal en gatposities (in het werkassenstelsel)."""
    out = part.copy()
    snaps: list[Snap] = []
    s = snap("hoogte", part.height, unc.total(unc.height, part.height),
             length_candidates(part.height, imperial), threshold=threshold)
    out.height = s.value
    snaps.append(s)

    o = out.outer
    if o.kind == "circle":
        d = 2 * o.radius
        s = snap("diameter", d, unc.total(unc.edge * math.sqrt(2), d), length_candidates(d, imperial),
                 threshold=threshold)
        o.radius = s.value / 2
        snaps.append(s)
    else:
        for axis in ("x", "y"):
            axes = edge_axes(o)
            idx = [k for k, (a, _) in enumerate(axes) if a == axis]
            for k in idx:
                pos = axes[k][1]
                if abs(pos) < 1e-9:
                    continue  # de datumrand zelf
                s = snap(f"{axis}-maat rand {k + 1}", pos, unc.total(unc.edge * math.sqrt(2), pos),
                         length_candidates(pos, imperial), threshold=threshold)
                _set_edge_position(o, k, axis, s.value)
                snaps.append(s)
        radii = [float(r) for r in o.fillets]
        nz = [i for i, r in enumerate(radii) if r > 0]
        for g in _groups([radii[i] for i in nz], 2.5 * unc.fillet):
            members = [nz[j] for j in g]
            r = float(np.mean([radii[i] for i in members]))
            label = f"{len(members)}x" if len(members) > 1 else f"hoek {members[0] + 1}"  # geen twee dezelfde namen
            s = snap(f"afronding R ({label})", r, unc.fillet / math.sqrt(len(members)) + 0.03,
                     radius_candidates(r), threshold=threshold)
            for i in members:
                o.fillets[i] = s.value
            snaps.append(s)

    diam = [h.d for h in part.holes]
    # 3σ: het verschil tussen een gat en het groepsgemiddelde is zelf ook onzeker (~1,15σ)
    for g in _groups(diam, 3.0 * unc.hole_d):
        d = float(np.mean([diam[i] for i in g]))
        label = f"{len(g)}x" if len(g) > 1 else f"gat {g[0] + 1}"
        s = snap(f"gat Ø ({label})", d, unc.hole_d / math.sqrt(len(g)) + 0.02, hole_candidates(d),
                 threshold=threshold)
        for i in g:
            out.holes[i] = Hole(out.holes[i].x, out.holes[i].y, s.value)
        snaps.append(s)

    # gatenpatroon op een steekcirkel (ronde delen): steekcirkeldiameter snappen, gaten exact verdelen
    patterned: set[int] = set()
    if o.kind == "circle":
        pattern = bolt_circle(out.holes, (0.0, 0.0))
        if pattern:
            r, start, n = pattern
            s = snap(f"steekcirkel Ø ({n}x)", 2 * r, unc.total(unc.hole_xy, 2 * r) / math.sqrt(n) + 0.02,
                     length_candidates(2 * r, imperial), threshold=threshold)
            snaps.append(s)
            ang0 = 0.0 if abs(start) < math.radians(1.5) else start
            half = math.pi / n  # een gat op 359,8° hoort vooraan, bij 0°
            order = np.argsort([(math.atan2(h.y, h.x) - ang0 + half) % (2 * math.pi) for h in out.holes])
            for k, i in enumerate(order):
                a = ang0 + 2 * math.pi * k / n
                out.holes[i] = Hole(s.value / 2 * math.cos(a), s.value / 2 * math.sin(a), out.holes[i].d)
                patterned.add(int(i))
    for i, h in enumerate(out.holes):
        if i in patterned:
            continue
        sx = snap(f"gat {i + 1} x", h.x, unc.total(unc.hole_xy, h.x), length_candidates(h.x, imperial),
                  threshold=threshold)
        sy = snap(f"gat {i + 1} y", h.y, unc.total(unc.hole_xy, h.y), length_candidates(h.y, imperial),
                  threshold=threshold)
        out.holes[i] = Hole(sx.value, sy.value, h.d)
        snaps += [sx, sy]
    return out, snaps


# ----------------------------------------------------------------------------- bouwen

def build(part: Part2p5D) -> cq.Workplane:
    """CadQuery-solid: extrusie van de contour, doorgaande gaten en uitsparingen."""
    o = part.outer
    if o.kind == "circle":
        model = cq.Workplane("XY").center(*o.center).circle(o.radius).extrude(part.height)
    else:
        model = cadhelpers.bouw_contour(cq, o.corner_table()).extrude(part.height)
    for h in part.holes:
        cutter = cq.Workplane("XY").workplane(offset=-1).center(h.x, h.y).circle(h.d / 2).extrude(part.height + 2)
        model = model.cut(cutter)
    for poly in part.cutouts:
        cutter = cq.Workplane("XY").workplane(offset=-1).polyline([tuple(p) for p in poly]).close() \
            .extrude(part.height + 2)
        model = model.cut(cutter)
    return model


def _fmt(v: float) -> str:
    v = 0.0 if abs(v) < 5e-4 else v
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"


def _comment(s: Snap | None) -> str:
    if s is None:
        return ""
    return f"  # gemeten {s.measured:.3f} ± {s.u95:.3f} (U95) — {s.reason}"


def _safe_text(value) -> str:
    """Tekst voor in de docstring van het script: geen aanhalingstekens, backslashes of regeleinden."""
    return re.sub(r"[^\w .,()+-]", "_", str(value))[:80]


def script(part: Part2p5D, snaps: list[Snap], meta: dict | None = None) -> str:
    """Leesbaar CadQuery-script met benoemde maten; levert hetzelfde model als build()."""
    meta = meta or {}
    by_name = {s.name: s for s in snaps}
    lines = [
        '"""Cam-to-CAD — parametrisch model (CadQuery).',
        "",
        f"Gegenereerd door camtocad {__version__} op {date.today().isoformat()}"
        + (f" uit scan '{_safe_text(meta['scan'])}'" if meta.get("scan") else "") + ".",
        "Eenheden: mm. Oorsprong: datum (linker- en onderrand), Z omhoog vanaf het contactvlak.",
        "Per maat: ontwerpwaarde, met in het commentaar de gemeten waarde, U95 en de reden.",
        '"""',
        "import math",
        "",
        "import cadquery as cq",
        "",
        "# --- Maten ---",
        f"hoogte = {_fmt(part.height)}{_comment(by_name.get('hoogte'))}",
    ]
    o = part.outer
    if o.kind == "circle":
        lines.append(f"diameter = {_fmt(2 * o.radius)}{_comment(by_name.get('diameter'))}")
    else:
        axes = edge_axes(o)
        names: dict[int, str] = {}
        for axis in ("x", "y"):
            positions = sorted({round(p, 6) for a, p in axes if a == axis})
            for j, p in enumerate(positions):
                var = f"{axis}{j}"
                ks = [k for k, (a, q) in enumerate(axes) if a == axis and round(q, 6) == p]
                snap_rec = next((by_name.get(f"{axis}-maat rand {k + 1}") for k in ks
                                 if by_name.get(f"{axis}-maat rand {k + 1}")), None)
                note = "  # datum" if abs(p) < 1e-9 else _comment(snap_rec)
                lines.append(f"{var} = {_fmt(p)}{note}")
                for k in ks:
                    names[k] = var
        radius_vars: dict[float, str] = {}
        for r in sorted({round(float(r), 6) for r in o.fillets if r > 0}):
            var = f"r{len(radius_vars) + 1}"
            radius_vars[r] = var
            rec = next((s for s in snaps if s.name.startswith("afronding") and abs(s.value - r) < 1e-6), None)
            lines.append(f"{var} = {_fmt(r)}{_comment(rec)}")

        V = o.vertices()
        lines += ["", "# Buitencontour: hoekpunten (x, y, afrondingsstraal), tegen de klok in", "contour = ["]
        n = o.n
        for k in range(n):
            # hoekpunt k ligt op rand k-1 en rand k
            ex, ey = None, None
            for e in (k - 1, k):
                a, _ = axes[e % n]
                if a == "x":
                    ex = names[e % n]
                elif a == "y":
                    ey = names[e % n]
            xs = ex if ex else _fmt(V[k][0])
            ys = ey if ey else _fmt(V[k][1])
            r = round(float(o.fillets[k]), 6)
            rs = radius_vars.get(r, "0.0")
            note = "" if ex and ey else "  # schuine rand: coördinaat berekend"
            lines.append(f"    ({xs}, {ys}, {rs}),{note}")
        lines.append("]")

    pattern = bolt_circle(part.holes) if o.kind == "circle" else None
    if pattern and abs(pattern[1]) < 1e-6:
        r, _, n = pattern
        rec = next((s for s in snaps if s.name.startswith("steekcirkel")), None)
        drec = next((s for s in snaps if s.name.startswith("gat Ø")), None)
        lines += ["", "# Gatenpatroon op een steekcirkel (eerste gat op 0°)",
                  f"steekcirkel_d = {_fmt(2 * r)}{_comment(rec)}",
                  f"aantal_gaten = {n}",
                  f"gat_d1 = {_fmt(part.holes[0].d)}{_comment(drec)}",
                  "gaten = [(steekcirkel_d / 2 * math.cos(2 * math.pi * k / aantal_gaten),",
                  "          steekcirkel_d / 2 * math.sin(2 * math.pi * k / aantal_gaten), gat_d1)",
                  "         for k in range(aantal_gaten)]"]
    elif part.holes:
        dvars: dict[float, str] = {}
        lines += ["", "# Doorgaande gaten"]
        for h in part.holes:
            key = round(h.d, 6)
            if key not in dvars:
                dvars[key] = f"gat_d{len(dvars) + 1}"
                rec = next((s for s in snaps if s.name.startswith("gat Ø") and abs(s.value - h.d) < 1e-6), None)
                lines.append(f"{dvars[key]} = {_fmt(h.d)}{_comment(rec)}")
        lines.append("gaten = [  # (x, y, diameter)")
        for i, h in enumerate(part.holes):
            sx, sy = by_name.get(f"gat {i + 1} x"), by_name.get(f"gat {i + 1} y")
            note = ""
            if sx and sy:
                note = f"  # gemeten ({sx.measured:.3f}, {sy.measured:.3f}) ± {max(sx.u95, sy.u95):.3f}"
            lines.append(f"    ({_fmt(h.x)}, {_fmt(h.y)}, {dvars[round(h.d, 6)]}),{note}")
        lines.append("]")
    if part.cutouts:
        lines += ["", "# Overige doorgaande uitsparingen (polygonen)", "uitsparingen = ["]
        for poly in part.cutouts:
            lines.append("    [" + ", ".join(f"({_fmt(x)}, {_fmt(y)})" for x, y in poly) + "],")
        lines.append("]")

    helper = inspect.getsource(cadhelpers).split("\n", 2)[2]  # zonder 'import math'
    lines += ["", "", "# --- Bouwfuncties (identiek aan camtocad.cadhelpers) ---", helper.strip(), "", "",
              "# --- Model ---"]
    if o.kind == "circle":
        lines.append("model = cq.Workplane(\"XY\").circle(diameter / 2).extrude(hoogte)")
    else:
        lines.append("model = bouw_contour(cq, contour).extrude(hoogte)")
    if part.holes:
        lines += ["for x, y, d in gaten:",
                  "    model = model.cut(cq.Workplane(\"XY\").workplane(offset=-1).center(x, y)"
                  ".circle(d / 2).extrude(hoogte + 2))"]
    if part.cutouts:
        lines += ["for punten in uitsparingen:",
                  "    model = model.cut(cq.Workplane(\"XY\").workplane(offset=-1).polyline(punten).close()"
                  ".extrude(hoogte + 2))"]
    lines += ["", "", 'if __name__ == "__main__":',
              '    cq.exporters.export(model, "model.step")',
              '    print("model.step geschreven")', ""]
    return "\n".join(lines)


def export(model: cq.Workplane, stem) -> dict[str, str]:
    paths = {"step": f"{stem}.step", "stl": f"{stem}.stl"}
    cq.exporters.export(model, paths["step"])
    cq.exporters.export(model, paths["stl"], tolerance=0.02, angularTolerance=0.1)
    return paths
