"""Ontwerpintentie: gemeten maten Bayesiaans 'snappen' naar standaardwaarden.

Voor een maat x ± σ wordt elke kandidaat c met prior π afgewogen tegen de hypothese
'vrije maat' (ARCHITECTURE.md §6.6):

    P(c | x) = π_c N(x; c, σ²) / (π_vrij + Σ π_j N(x; c_j, σ²))

Snappen gebeurt alleen bij een posterior ≥ drempel én |x − c| ≤ 2,5σ. Bij een grote
onzekerheid blijft de gemeten waarde dus staan: eerlijker dan altijd afronden.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

# ISO 273 doorgangsgaten (fijn, middel, grof) en tapboormaten voor metrisch grof schroefdraad
ISO_273 = {
    "M2": (2.2, 2.4, 2.6), "M2.5": (2.7, 2.9, 3.1), "M3": (3.2, 3.4, 3.6), "M4": (4.3, 4.5, 4.8),
    "M5": (5.3, 5.5, 5.8), "M6": (6.4, 6.6, 7.0), "M8": (8.4, 9.0, 10.0), "M10": (10.5, 11.0, 12.0),
    "M12": (13.0, 13.5, 14.5),
}
TAP_DRILL = {"M2": 1.6, "M2.5": 2.05, "M3": 2.5, "M4": 3.3, "M5": 4.2, "M6": 5.0, "M8": 6.8, "M10": 8.5,
             "M12": 10.2}
STANDARD_RADII = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0)


@dataclass
class Snap:
    name: str
    measured: float
    sigma: float
    value: float
    snapped: bool
    reason: str
    posterior: float

    @property
    def u95(self) -> float:
        return 2.0 * self.sigma

    def to_dict(self) -> dict:
        d = asdict(self)
        d["u95"] = self.u95
        return d


def _grid(x: float, step: float, prior: float, reason: str, exclude_step: float | None = None):
    base = round(x / step)
    out = []
    for k in range(base - 2, base + 3):
        c = round(k * step, 6)
        if c <= 0:
            continue
        if exclude_step and abs(c / exclude_step - round(c / exclude_step)) < 1e-6:
            continue
        out.append((c, prior, reason))
    return out


def length_candidates(x: float, imperial: bool = False) -> list[tuple[float, float, str]]:
    if imperial:
        return (_grid(x, 25.4 / 16, 0.5, "inch (1/16\")")
                + _grid(x, 25.4 / 32, 0.2, "inch (1/32\")", exclude_step=25.4 / 16))
    return _grid(x, 1.0, 0.5, "hele mm") + _grid(x, 0.5, 0.2, "halve mm", exclude_step=1.0)


def hole_candidates(d: float) -> list[tuple[float, float, str]]:
    out = []
    for size, (fine, medium, coarse) in ISO_273.items():
        out += [(fine, 0.12, f"ISO 273 doorgangsgat {size} (fijn)"),
                (medium, 0.2, f"ISO 273 doorgangsgat {size} (middel)"),
                (coarse, 0.08, f"ISO 273 doorgangsgat {size} (grof)")]
    out += [(v, 0.15, f"tapboor {size} (voor schroefdraad {size})") for size, v in TAP_DRILL.items()]
    out = [c for c in out if abs(c[0] - d) < 2.0]
    return out + [(c, p * 0.6, r) for c, p, r in length_candidates(d)]


def radius_candidates(r: float) -> list[tuple[float, float, str]]:
    return [(c, 0.5, "standaardstraal") for c in STANDARD_RADII if abs(c - r) < 3.0] + \
        _grid(r, 0.5, 0.1, "halve mm", exclude_step=None)


def snap(name: str, x: float, sigma: float, candidates, *, free_prior: float = 0.2,
         threshold: float = 0.8, max_sigmas: float = 2.5) -> Snap:
    """Kiest de meest waarschijnlijke kandidaat; snapt alleen bij voldoende zekerheid."""
    sigma = max(float(sigma), 1e-4)
    merged: dict[float, tuple[float, str]] = {}
    for c, p, reason in candidates:  # dezelfde waarde uit meerdere bronnen: priors optellen
        key = round(c, 6)
        prev = merged.get(key)
        merged[key] = (p + (prev[0] if prev else 0.0), prev[1] if prev and prev[0] >= p else reason)
    weights = {c: p * math.exp(-0.5 * ((x - c) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
               for c, (p, _) in merged.items()}
    total = free_prior + sum(weights.values())
    if not weights:
        return Snap(name, x, sigma, x, False, "geen kandidaten", 0.0)
    best = max(weights, key=weights.get)
    post = weights[best] / total
    if post >= threshold and abs(x - best) <= max_sigmas * sigma:
        return Snap(name, x, sigma, best, True, merged[best][1], post)
    return Snap(name, x, sigma, x, False, f"niet gesnapt (beste kandidaat {best:g}, p = {post:.2f})", post)
