"""Onzekerheid per maat (V3): de covariantie van de randfit, doorgerekend naar elke maat, plus systematiek.

* Toevallig deel: de spreiding van de modelparameters over een jackknife van groepen foto's
  (edgefit.jackknife). Fouten die per foto samenhangen (pose, een masker dat een wand net mist) tellen
  zo mee; de formele covariantie uit de Jacobiaan is daarvoor veel te optimistisch.
* Doorrekenen: elke maat is een functie van het model in het werkassenstelsel (vaste draaiing en
  verschuiving, zodat de datum mee kan bewegen); σ² = gᵀ C g met g de numerieke gradiënt.
* Systematisch deel, per soort maat, in pixels op het object: wat de fit op gerenderde scans met
  zuivere silhouetten nog verkeerd doet (maskerrand, restbias van BETA). Afgesteld zodat daar ~95% van
  de fouten binnen U95 valt (ROUTE-A-VERBETERPUNTEN §3e); op echte foto's te toetsen met V1.
* Printschaal: relatief, 0,05% met gemeten meetlijnen, anders 0,3%. Die telt in de gerapporteerde U95,
  maar niet bij het snappen: een schaalfout verschuift alle maten samen, en een ontwerp in hele mm blijft
  dan het aannemelijkst.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .profile import Part2p5D

# systematische 1σ in pixels op het object (zie de moduletekst; ijking in ROUTE-A-VERBETERPUNTEN §3e)
SYS_PX = {"lengte": 0.12, "hoogte": 0.10, "gat": 0.30, "positie": 0.15, "afronding": 0.80}
SCALE_REL_MEASURED, SCALE_REL_ASSUMED = 5e-4, 3e-3


@dataclass
class Sensitivity:
    """Rekent maten van het model om naar een 1σ (toevallig deel), via de parametercovariantie."""

    build: Callable[[np.ndarray], Part2p5D]  # parameters -> model in matcoördinaten
    x: np.ndarray
    C: np.ndarray
    angle: float  # werkassenstelsel van het gefitte model (vast)
    shift: np.ndarray
    eps: float = 1e-4
    _cache: dict = field(default_factory=dict)

    def _parts(self) -> list[Part2p5D]:
        if "parts" not in self._cache:
            parts = [self.build(self.x).transformed(self.angle, self.shift)]
            for j in range(len(self.x)):
                x = self.x.copy()
                x[j] += self.eps
                parts.append(self.build(x).transformed(self.angle, self.shift))
            self._cache["parts"] = parts
        return self._cache["parts"]

    def sigma(self, fn: Callable[[Part2p5D], float]) -> float:
        parts = self._parts()
        base = fn(parts[0])
        g = np.array([(fn(p) - base) / self.eps for p in parts[1:]])
        return float(math.sqrt(max(float(g @ self.C @ g), 0.0)))


def edge_position(part: Part2p5D, k: int, axis: str, at: float = 0.0) -> float:
    """Ligging van rand k langs de as ('x' of 'y') in werkcoördinaten, ook als hij iets scheef staat:
    x van de randlijn op hoogte y = `at` (of y bij x = `at`). Voor een gat telt de rand ter hoogte van het
    gat; dan verandert de maat niet als het hele onderdeel een fractie draait."""
    o = part.outer
    n = np.array([math.cos(o.angles[k]), math.sin(o.angles[k])])
    c = float(n @ o.center + o.offsets[k])
    return (c - n[1] * at) / n[0] if axis == "x" else (c - n[0] * at) / n[1]


@dataclass
class Budget:
    """Alles wat snap_part nodig heeft voor σ per maat."""

    sens: Sensitivity | None
    mm_per_px: float
    scale_rel: float
    datum: dict = field(default_factory=dict)  # as -> index van de datumrand

    def sys(self, kind: str) -> float:
        return SYS_PX[kind] * self.mm_per_px

    def rel(self, fn: Callable[[Part2p5D], float], kind: str) -> float | None:
        """1σ voor snappen (zonder printschaal), of None als er geen covariantie is."""
        if self.sens is None:
            return None
        return math.hypot(self.sens.sigma(fn), self.sys(kind))

    def total(self, sigma_rel: float, value: float) -> float:
        """1σ voor het rapport: met de printschaal."""
        return math.hypot(sigma_rel, self.scale_rel * abs(value))
