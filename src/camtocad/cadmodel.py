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
from dataclasses import dataclass, replace
from datetime import date

import cadquery as cq
import numpy as np

from . import __version__, cadhelpers
from .profile import Hole, Part2p5D, Profile, Step, dominant_angle
from .snapping import Snap, hole_candidates, length_candidates, radius_candidates, snap
from .uncertainty import Budget, edge_position


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


def step_axis(st) -> str | None:
    """'x' voor een trede langs een verticale lijn (x = ...), 'y' voor een horizontale, anders None."""
    c, s = abs(math.cos(st.angle)), abs(math.sin(st.angle))
    return "x" if c > 1 - 1e-9 else "y" if s > 1 - 1e-9 else None


def step_point(st) -> tuple[float, float]:
    """Het punt van de lijn van een trede het dichtst bij de oorsprong."""
    x, y = st.offset * st.normal()
    return float(x), float(y)


def top_edge_name(te) -> str:
    """Naam van de maat van de bovenrand in het rapport."""
    return "afschuining bovenrand" if te.kind == "afschuining" else "afronding bovenrand"


def slot_names(part: Part2p5D) -> list[tuple[str, str]]:
    """(naam in het rapport, variabele in het script) per sleuf of uitsparing, per soort genummerd."""
    out, count = [], {"sleuf": 0, "rechthoek": 0}
    for s in part.slots:
        count[s.kind] += 1
        k = count[s.kind]
        out.append((f"sleuf {k}", f"sleuf{k}") if s.kind == "sleuf" else (f"uitsparing {k}", f"uitsparing{k}"))
    return out


def snap_part(part: Part2p5D, unc: Uncertainty, *, threshold: float = 0.8,
              imperial: bool = False, budget: Budget | None = None,
              evidence: dict | None = None) -> tuple[Part2p5D, list[Snap]]:
    """Snapt hoogte, randposities, diameters, straal en gatposities (in het werkassenstelsel).

    Met een `budget` (uncertainty.py) krijgt elke maat zijn eigen σ uit de covariantie van de randfit
    plus systematiek; zonder budget de indicatieve σ per soort maat uit `unc`. Snappen gebeurt zonder
    de printschaal (een schaalfout verschuift alle maten samen; een ontwerp in hele mm blijft dan het
    aannemelijkst), het rapport (U95) telt hem wel mee. `evidence`: het bewijs rond elk gat en elke sleuf
    (edgefit.evidence); met bewijs aan maar een deel van de rand is het systematische deel groter.
    """
    out = part.copy()
    snaps: list[Snap] = []
    scale_rel = budget.scale_rel if budget is not None else unc.scale_rel
    evidence = evidence or {}

    def amps(kind: str, i: int) -> tuple[float, float]:
        e = evidence.get((kind, i))
        return (e.amp_size, e.amp_pos) if e is not None else (1.0, 1.0)

    def do_snap(name, value, base, candidates, fn=None, kind="lengte", amp=1.0):
        """`base`: indicatieve σ zonder printschaal; `fn`: de maat als functie van het model (budget)."""
        rel = budget.rel(fn, kind, amp) if (budget is not None and fn is not None) else None
        sigma = base * amp if rel is None else rel
        s = snap(name, value, sigma, candidates, threshold=threshold)
        s.sigma = math.hypot(sigma, scale_rel * abs(value))
        return s

    # met een afgeschuinde of afgeronde bovenrand zien alleen de lage foto's de bovenkant (V17)
    s = do_snap("hoogte", part.height, unc.height, length_candidates(part.height, imperial),
                lambda p: p.height, "hoogte" if part.top_edge is None else "hoogte bovenrand")
    out.height = s.value
    snaps.append(s)
    if part.top_edge is not None:
        te = part.top_edge
        s = do_snap(top_edge_name(te), te.size, unc.fillet, radius_candidates(te.size),
                    lambda p: p.top_edge.size, "bovenrand")
        out.top_edge = replace(te, size=min(s.value, 0.75 * out.height))
        snaps.append(s)

    o = out.outer
    datum: dict[str, int] = {}
    length_kind = "lengte" if part.top_edge is None else "lengte bovenrand"
    if o.kind == "circle":
        d = 2 * o.radius
        s = do_snap("diameter", d, unc.edge * math.sqrt(2), length_candidates(d, imperial),
                    lambda p: 2 * p.outer.radius, length_kind)
        o.radius = s.value / 2
        snaps.append(s)
    else:
        axes0 = edge_axes(o)
        for axis in ("x", "y"):
            cand = [(abs(pos), k) for k, (a, pos) in enumerate(axes0) if a == axis]
            if cand:
                datum[axis] = min(cand)[1]
        for axis in ("x", "y"):
            axes = edge_axes(o)
            idx = [k for k, (a, _) in enumerate(axes) if a == axis]
            for k in idx:
                pos = axes[k][1]
                if abs(pos) < 1e-9:
                    continue  # de datumrand zelf
                fn = (lambda p, k=k, axis=axis: edge_position(p, k, axis) - edge_position(p, datum[axis], axis)) \
                    if axis in datum else None
                s = do_snap(f"{axis}-maat rand {k + 1}", pos, unc.edge * math.sqrt(2),
                            length_candidates(pos, imperial), fn, length_kind)
                _set_edge_position(o, k, axis, s.value)
                snaps.append(s)
        radii = [float(r) for r in o.fillets]
        nz = [i for i, r in enumerate(radii) if r > 0]
        for g in _groups([radii[i] for i in nz], 2.5 * unc.fillet):
            members = [nz[j] for j in g]
            r = float(np.mean([radii[i] for i in members]))
            label = f"{len(members)}x" if len(members) > 1 else f"hoek {members[0] + 1}"  # geen twee dezelfde namen
            s = do_snap(f"afronding R ({label})", r, unc.fillet / math.sqrt(len(members)) + 0.03,
                        radius_candidates(r), lambda p, m=members: float(np.mean(p.outer.fillets[m])), "afronding")
            for i in members:
                o.fillets[i] = s.value
            snaps.append(s)

    diam = [h.d for h in part.holes]
    # 3σ: het verschil tussen een gat en het groepsgemiddelde is zelf ook onzeker (~1,15σ)
    for g in _groups(diam, 3.0 * unc.hole_d):
        # gewogen: een gat met bewijs aan maar een deel van de rand telt minder mee (1/vergroting²)
        wts = np.array([1.0 / amps("gat", i)[0] ** 2 for i in g])
        wts /= wts.sum()
        d = float(wts @ [diam[i] for i in g])
        amp = 1.0 / math.sqrt(float(np.mean([1.0 / amps("gat", i)[0] ** 2 for i in g])))
        label = f"{len(g)}x" if len(g) > 1 else f"gat {g[0] + 1}"
        s = do_snap(f"gat Ø ({label})", d, unc.hole_d / math.sqrt(len(g)) + 0.02, hole_candidates(d),
                    lambda p, g=g, wts=wts: float(wts @ [p.holes[i].d for i in g]), "gat", amp)
        for i in g:
            out.holes[i] = Hole(out.holes[i].x, out.holes[i].y, s.value)
        snaps.append(s)

    def origin(p: Part2p5D, axis: str, at: float) -> float:
        """Datum voor gatposities: het middelpunt, of de datumrand ter hoogte van het gat."""
        if p.outer.kind == "circle":
            return float(p.outer.center[0 if axis == "x" else 1])
        return edge_position(p, datum[axis], axis, at) if axis in datum else 0.0

    # gatenpatroon op een steekcirkel (ronde delen): steekcirkeldiameter snappen, gaten exact verdelen
    patterned: set[int] = set()
    if o.kind == "circle":
        pattern = bolt_circle(out.holes, (0.0, 0.0))
        if pattern:
            r, start, n = pattern

            def fn(p: Part2p5D) -> float:
                c = p.outer.center
                return 2 * float(np.mean([math.hypot(h.x - c[0], h.y - c[1]) for h in p.holes]))

            amp = 1.0 / math.sqrt(float(np.mean([1.0 / amps("gat", i)[1] ** 2 for i in range(len(part.holes))])))
            s = do_snap(f"steekcirkel Ø ({n}x)", 2 * r, unc.hole_xy / math.sqrt(n) + 0.02,
                        length_candidates(2 * r, imperial), fn, "positie", amp)
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
        amp = amps("gat", i)[1]
        sx = do_snap(f"gat {i + 1} x", h.x, unc.hole_xy, length_candidates(h.x, imperial),
                     lambda p, i=i: p.holes[i].x - origin(p, "x", p.holes[i].y), "positie", amp)
        sy = do_snap(f"gat {i + 1} y", h.y, unc.hole_xy, length_candidates(h.y, imperial),
                     lambda p, i=i: p.holes[i].y - origin(p, "y", p.holes[i].x), "positie", amp)
        out.holes[i] = Hole(sx.value, sy.value, h.d)
        snaps += [sx, sy]

    # sleuven en rechthoekige uitsparingen (V15), zoals een ontwerper ze maatvoert: een sleuf met breedte
    # (vaak een doorgangsmaat) en hartafstand, een rechthoek met lengte, breedte en hoekstraal
    for i, (sl, (name, _)) in enumerate(zip(out.slots, slot_names(out))):
        a90 = round(sl.angle / (math.pi / 2)) * (math.pi / 2)
        angle = a90 if abs(sl.angle - a90) < math.radians(2.0) else sl.angle  # haaks op de datum als het bijna zo is
        base = unc.hole_d + 0.02
        r = sl.r
        a_size, a_pos = amps("sleuf", i)
        if sl.kind == "sleuf":
            sw = do_snap(f"{name} breedte", sl.width, base, hole_candidates(sl.width),
                         lambda p, i=i: p.slots[i].width, "gat", a_size)
            c2c = sl.length - sl.width
            sc = do_snap(f"{name} hartafstand", c2c, base, length_candidates(c2c, imperial),
                         lambda p, i=i: p.slots[i].length - p.slots[i].width, "gat", a_size)
            width, length = sw.value, sc.value + sw.value
            snaps += [sw, sc]
        else:
            sL = do_snap(f"{name} lengte", sl.length, base, length_candidates(sl.length, imperial),
                         lambda p, i=i: p.slots[i].length, "gat", a_size)
            sW = do_snap(f"{name} breedte", sl.width, base, length_candidates(sl.width, imperial),
                         lambda p, i=i: p.slots[i].width, "gat", a_size)
            length, width = sL.value, sW.value
            snaps += [sL, sW]
            if sl.r > 0:
                sr = do_snap(f"{name} hoekstraal", sl.r, unc.fillet, radius_candidates(sl.r),
                             lambda p, i=i: p.slots[i].r, "afronding", a_size)
                r = min(sr.value, width / 2)
                snaps.append(sr)
        sx = do_snap(f"{name} x", sl.x, unc.hole_xy, length_candidates(sl.x, imperial),
                     lambda p, i=i: p.slots[i].x - origin(p, "x", p.slots[i].y), "positie", a_pos)
        sy = do_snap(f"{name} y", sl.y, unc.hole_xy, length_candidates(sl.y, imperial),
                     lambda p, i=i: p.slots[i].y - origin(p, "y", p.slots[i].x), "positie", a_pos)
        out.slots[i] = replace(sl, x=sx.value, y=sy.value, length=max(length, width), width=width, angle=angle, r=r)
        snaps += [sx, sy]

    # treden (V17): hoogte, en de plaats van de lijn vanaf de datum; haaks op de datum als het bijna zo is
    for k, st in enumerate(part.steps):
        a90 = round(st.angle / (math.pi / 2)) * (math.pi / 2)
        angle = a90 if abs(st.angle - a90) < math.radians(2.0) else st.angle
        sh = do_snap(f"trede {k + 1} hoogte", st.height, unc.height, length_candidates(st.height, imperial),
                     lambda p, k=k: p.steps[k].height, "hoogte")
        n = np.array([math.cos(angle), math.sin(angle)])
        axis = "x" if abs(n[0]) > 0.99 else "y" if abs(n[1]) > 0.99 else None
        if axis is None:
            sp = do_snap(f"trede {k + 1} positie", st.offset, unc.edge * math.sqrt(2),
                         length_candidates(st.offset, imperial), lambda p, k=k: p.steps[k].offset, "lengte")
            offset = sp.value
        else:
            j = 0 if axis == "x" else 1

            def coord(p: Part2p5D, k=k, j=j, axis=axis) -> float:
                """Plaats van de lijn langs de as, vanaf de datumrand (op de hoogte van de oorsprong)."""
                s_ = p.steps[k]
                c = s_.offset / s_.normal()[j]
                return c - (edge_position(p, datum[axis], axis) if axis in datum else 0.0)

            pos = st.offset / n[j]
            sp = do_snap(f"trede {k + 1} positie", pos, unc.edge * math.sqrt(2), length_candidates(pos, imperial),
                         coord, "lengte")
            offset = sp.value * n[j]
        out.steps[k] = Step.from_line(angle, float(offset), min(sh.value, out.height - 0.3), st.pivot)
        snaps += [sp, sh]
    return out, snaps


# ----------------------------------------------------------------------------- bouwen

def build(part: Part2p5D) -> cq.Workplane:
    """CadQuery-solid: extrusie van de contour (met een afgeschuinde of afgeronde bovenrand en treden, V17),
    doorgaande gaten en uitsparingen."""
    o = part.outer
    if o.kind == "circle":
        def sketch():
            return cq.Workplane("XY").center(*o.center).circle(o.radius)
    else:
        table = o.corner_table()

        def sketch():
            return cadhelpers.bouw_contour(cq, table)
    te = part.top_edge
    model = cadhelpers.extrudeer(sketch, part.height, te.kind if te else None, te.size if te else 0.0)
    for st in part.steps:
        x, y = st.offset * st.normal()
        model = cadhelpers.trede(cq, model, x, y, math.degrees(st.angle), st.height, part.height)
    for h in part.holes:
        cutter = cq.Workplane("XY").workplane(offset=-1).center(h.x, h.y).circle(h.d / 2).extrude(part.height + 2)
        model = model.cut(cutter)
    for poly in part.cutouts:
        cutter = cq.Workplane("XY").workplane(offset=-1).polyline([tuple(p) for p in poly]).close() \
            .extrude(part.height + 2)
        model = model.cut(cutter)
    for s in part.slots:
        table = cadhelpers.sleuf_hoeken(s.x, s.y, s.length, s.width, math.degrees(s.angle), s.rad)
        model = model.cut(cadhelpers.bouw_contour(cq, table).extrude(part.height + 2).translate((0, 0, -1)))
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

    te = part.top_edge
    if te is not None:
        var = "afschuining" if te.kind == "afschuining" else "afronding_boven"
        lines += ["", "# Bovenrand, rondom: " + ("afschuining onder 45° (benen)" if te.kind == "afschuining"
                                                  else "afronding (straal)"),
                  f"{var} = {_fmt(te.size)}{_comment(by_name.get(top_edge_name(te)))}"]
    if part.steps:
        lines += ["", "# Treden: voorbij de lijn door (x, y), in de richting 'hoek' (graden, 0 = +X), is het deel "
                  "lager"]
        rows = []
        for k, st in enumerate(part.steps, start=1):
            x, y = step_point(st)
            axis = step_axis(st)
            lines.append(f"trede{k}_{axis or 'positie'} = {_fmt(x if axis != 'y' else y) if axis else _fmt(st.offset)}"
                         f"{_comment(by_name.get(f'trede {k} positie'))}")
            lines.append(f"trede{k}_hoogte = {_fmt(st.height)}{_comment(by_name.get(f'trede {k} hoogte'))}")
            if axis == "x":
                px, py = f"trede{k}_x", "0.0"
            elif axis == "y":
                px, py = "0.0", f"trede{k}_y"
            else:
                px = f"trede{k}_positie * math.cos(math.radians({_fmt(math.degrees(st.angle))}))"
                py = f"trede{k}_positie * math.sin(math.radians({_fmt(math.degrees(st.angle))}))"
            rows.append(f"    ({px}, {py}, {_fmt(math.degrees(st.angle))}, trede{k}_hoogte),")
        lines += ["treden = [  # (x, y) op de lijn, hoek naar het lage deel, hoogte"] + rows + ["]"]

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
    if part.slots:
        lines += ["", "# Sleuven en rechthoekige uitsparingen"]
        rows = []
        for s, (name, var) in zip(part.slots, slot_names(part)):
            if s.kind == "sleuf":
                lines.append(f"{var}_breedte = {_fmt(s.width)}{_comment(by_name.get(f'{name} breedte'))}")
                lines.append(f"{var}_hartafstand = {_fmt(s.length - s.width)}"
                             f"{_comment(by_name.get(f'{name} hartafstand'))}")
                length, r = f"{var}_hartafstand + {var}_breedte", f"{var}_breedte / 2"
            else:
                lines.append(f"{var}_lengte = {_fmt(s.length)}{_comment(by_name.get(f'{name} lengte'))}")
                lines.append(f"{var}_breedte = {_fmt(s.width)}{_comment(by_name.get(f'{name} breedte'))}")
                lines.append(f"{var}_hoekstraal = {_fmt(s.r)}{_comment(by_name.get(f'{name} hoekstraal'))}")
                length, r = f"{var}_lengte", f"{var}_hoekstraal"
            sx, sy = by_name.get(f"{name} x"), by_name.get(f"{name} y")
            note = f"  # gemeten ({sx.measured:.3f}, {sy.measured:.3f}) ± {max(sx.u95, sy.u95):.3f}" if sx and sy else ""
            rows.append(f"    ({_fmt(s.x)}, {_fmt(s.y)}, {length}, {var}_breedte, {_fmt(math.degrees(s.angle))}, {r}),"
                        f"{note}")
        lines += ["sleuven = [  # (x, y, lengte, breedte, hoek in graden, hoekstraal)"] + rows + ["]"]
    if part.cutouts:
        lines += ["", "# Overige doorgaande uitsparingen (polygonen)", "uitsparingen = ["]
        for poly in part.cutouts:
            lines.append("    [" + ", ".join(f"({_fmt(x)}, {_fmt(y)})" for x, y in poly) + "],")
        lines.append("]")

    helper = inspect.getsource(cadhelpers).split("\n", 2)[2]  # zonder 'import math'
    lines += ["", "", "# --- Bouwfuncties (identiek aan camtocad.cadhelpers) ---", helper.strip(), "", "",
              "# --- Model ---"]
    sketch = ("lambda: cq.Workplane(\"XY\").circle(diameter / 2)" if o.kind == "circle"
              else "lambda: bouw_contour(cq, contour)")
    if te is not None:
        lines.append(f"model = extrudeer({sketch}, hoogte, \"{te.kind}\", "
                     f"{'afschuining' if te.kind == 'afschuining' else 'afronding_boven'})")
    else:
        lines.append(f"model = extrudeer({sketch}, hoogte)")
    if part.steps:
        lines += ["for x, y, hoek, h in treden:", "    model = trede(cq, model, x, y, hoek, h, hoogte)"]
    if part.holes:
        lines += ["for x, y, d in gaten:",
                  "    model = model.cut(cq.Workplane(\"XY\").workplane(offset=-1).center(x, y)"
                  ".circle(d / 2).extrude(hoogte + 2))"]
    if part.cutouts:
        lines += ["for punten in uitsparingen:",
                  "    model = model.cut(cq.Workplane(\"XY\").workplane(offset=-1).polyline(punten).close()"
                  ".extrude(hoogte + 2))"]
    if part.slots:
        lines += ["for x, y, lengte, breedte, hoek, r in sleuven:",
                  "    uitsparing = bouw_contour(cq, sleuf_hoeken(x, y, lengte, breedte, hoek, r))",
                  "    model = model.cut(uitsparing.extrude(hoogte + 2).translate((0, 0, -1)))"]
    lines += ["", "", 'if __name__ == "__main__":',
              '    cq.exporters.export(model, "model.step")',
              '    print("model.step geschreven")', ""]
    return "\n".join(lines)


def export(model: cq.Workplane, stem) -> dict[str, str]:
    paths = {"step": f"{stem}.step", "stl": f"{stem}.stl"}
    cq.exporters.export(model, paths["step"])
    cq.exporters.export(model, paths["stl"], tolerance=0.02, angularTolerance=0.1)
    return paths
