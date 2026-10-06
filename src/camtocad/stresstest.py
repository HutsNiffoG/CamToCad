"""Stresstests (V26, v0.12): synthetische scans met realistische storingen door de hele pijplijn, en de maten
vergeleken met de waarheid.

Per scenario een gerenderde scan van een bekend onderdeel op de mat (46 foto's van 1600 x 1200 in drie ringen en zes
van boven), met ruis, JPEG, vignettering en verscherping, en per scenario bijvoorbeeld een slagschaduw, een
toonkromme zoals een telefoon (ook lokale toonbewerking), kleur, een zwart onderdeel of weinig lage foto's. De
uitvoer per scenario: de uitvoer van de pijplijn (`resultaat/`), een paar foto's en `stress.json` met de maten, de
waarheid en het log. `vergelijk` zet twee runs naast elkaar: per maat de fout, of die binnen de U95 valt (zonder de
printschaal: een gerenderde mat heeft een exacte schaal) en of een snap klopt; onderaan de samenvatting per soort
maat, voor alle scans en voor die zonder waarschuwing ("onbetrouwbaar").

Opdrachtregel: `camtocad stresstest [scenario ...]` en `camtocad stresstest --vergelijk OUD NIEUW`; zie de README
en ROUTE-A-VERBETERPUNTEN.md voor de uitkomsten per versie. Eén scenario kost 1,5-5 minuten en ~4-5 GB geheugen.
"""

from __future__ import annotations

import json
import math
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from .mat import get_spec, rasterize_board
from .pipeline import ScanOptions, demo_part, run_scan
from .render import default_camera, look_at, place, render_view, tessellate


# ----------------------------------------------------------------------------- de onderdelen en hun waarheid

def small_plate():
    """Plaatje 30 x 20 x 5 mm, R2, gat Ø4,5 op 8 mm links van het midden."""
    import cadquery as cq
    return (cq.Workplane("XY").box(30, 20, 5, centered=(True, True, False)).edges("|Z").fillet(2)
            .faces(">Z").workplane().pushPoints([(-8, 0)]).hole(4.5))


def washer():
    """Ring Ø25 x 8 mm met een gat Ø8."""
    import cadquery as cq
    return cq.Workplane("XY").circle(12.5).circle(4.0).extrude(8)


def slot_bracket():
    """Beugel met gat Ø6,6, sleuf 6,6 (hartafstand 16) en uitsparing 10 x 8 R1,5 (V15)."""
    import cadquery as cq
    part = cq.Workplane("XY").box(80, 40, 12, centered=(True, True, False)).edges("|Z").fillet(3)
    part = part.faces(">Z").workplane(centerOption="ProjectedOrigin").pushPoints([(-30, 0)]).hole(6.6)
    part = part.cut(cq.Workplane("XY").center(8, 0).slot2D(22.6, 6.6, 0).extrude(12))
    return part.cut(cq.Workplane("XY").center(30, 0).rect(10, 8).extrude(12).edges("|Z").fillet(1.5))


def chamfer_bracket():
    """Beugel met rondom een afschuining van 1,5 mm op de bovenrand en twee gaten Ø6,6 (V17)."""
    import cadquery as cq
    part = cq.Workplane("XY").box(80, 40, 12, centered=(True, True, False)).edges("|Z").fillet(3)
    return part.faces(">Z").edges().chamfer(1.5).faces(">Z").workplane().pushPoints([(-30, 0), (30, 0)]).hole(6.6)


def round_bracket():
    """Beugel zonder gaten met rondom een afronding R2 van de bovenrand (V17)."""
    import cadquery as cq
    return cq.Workplane("XY").box(80, 40, 12, centered=(True, True, False)).edges("|Z").fillet(3) \
        .faces(">Z").edges().fillet(2.0)


def step_block():
    """Beugel met een trede: de rechter 20 mm is 6 mm hoog; gaten Ø6,6 in het hoge en in het lage deel (V17)."""
    import cadquery as cq
    part = cq.Workplane("XY").box(80, 40, 12, centered=(True, True, False)).edges("|Z").fillet(3)
    part = part.cut(cq.Workplane("XY").box(20, 44, 6, centered=(False, True, False)).translate((20, 0, 6)))
    return part.cut(cq.Workplane("XY").pushPoints([(-30, 0), (30, 0)]).circle(3.3).extrude(12))


def csk_bracket():
    """Beugel met twee verzonken gaten Ø6,6 met verzinking Ø12,4 x 90° (DIN 74-1 A, M6; V16)."""
    import cadquery as cq
    return (cq.Workplane("XY").box(80, 40, 12, centered=(True, True, False)).edges("|Z").fillet(3)
            .faces(">Z").workplane().pushPoints([(-30, 0), (30, 0)]).cskHole(6.6, 12.4, 90))


def chamfered_holes():
    """Beugel met twee gaten Ø6,6 met een faas van 0,5 x 45° aan de bovenkant (Ø7,6; v0.11)."""
    import cadquery as cq
    return (cq.Workplane("XY").box(80, 40, 12, centered=(True, True, False)).edges("|Z").fillet(3)
            .faces(">Z").workplane().pushPoints([(-30, 0), (30, 0)]).cskHole(6.6, 7.6, 90))


def blind_bracket():
    """Beugel met een doorgaand gat Ø6,6 (links) en een blind gat Ø6,6 x 6 diep (rechts; v0.11)."""
    import cadquery as cq
    part = cq.Workplane("XY").box(80, 40, 12, centered=(True, True, False)).edges("|Z").fillet(3)
    part = part.faces(">Z").workplane(centerOption="ProjectedOrigin").pushPoints([(-30, 0)]).hole(6.6)
    return part.faces(">Z").workplane(centerOption="ProjectedOrigin").pushPoints([(30, 0)]).hole(6.6, depth=6.0)


def cbore_bracket():
    """Beugel met twee kamerboringen M6: gat Ø6,6, kamer Ø11 x 6,4 diep (DIN 974-1/2, ISO 4762; v0.10)."""
    import cadquery as cq
    return (cq.Workplane("XY").box(80, 40, 12, centered=(True, True, False)).edges("|Z").fillet(3)
            .faces(">Z").workplane().pushPoints([(-30, 0), (30, 0)]).cboreHole(6.6, 11.0, 6.4))


def notched_bracket():
    """Beugel met twee gaten Ø6,6 en een inham van 3 x 2 mm midden in een lange zijde (V14, v0.12)."""
    import cadquery as cq
    return demo_part().cut(cq.Workplane("XY").box(3, 2, 12, centered=(True, False, False)).translate((0, 18, 0)))


# De waarheid in het assenstelsel van het onderdeel (zoals in het rapport). Een lijst: elk van die waarden kan het
# zijn (de beugel kan in het rapport ook gespiegeld liggen, en dan zit het gat op 10 in plaats van 70)
BRACKET = {"h": 12.0, "x": [80.0], "y": [40.0], "R": 3.0, "d": 6.6, "hx": [10.0, 70.0], "hy": [20.0]}
PLATE = {"h": 5.0, "x": [30.0], "y": [20.0], "R": 2.0, "d": 4.5, "hx": [7.0, 23.0], "hy": [10.0]}
WASHER = {"h": 8.0, "D": 25.0, "d": 8.0, "hx": [0.0], "hy": [0.0]}
SLOTTED = dict(BRACKET, **{"sleuf breedte": 6.6, "sleuf hartafstand": 16.0, "sleuf x": [48.0, 32.0],
                           "sleuf y": [20.0], "uitsparing lengte": 10.0, "uitsparing breedte": 8.0,
                           "uitsparing hoekstraal": 1.5, "uitsparing x": [70.0, 10.0], "uitsparing y": [20.0]})

OBJECTS = {
    "bracket": (demo_part, BRACKET),
    "kamerbeugel": (cbore_bracket, dict(BRACKET, cb=11.0, cb_depth=6.4)),
    "verzonken": (csk_bracket, dict(BRACKET, csk=12.4)),
    "afschuinbeugel": (chamfer_bracket, dict(BRACKET, **{"afschuining bovenrand": 1.5})),
    "faasbeugel": (chamfered_holes, dict(BRACKET, csk=7.6)),
    "blindbeugel": (blind_bracket, dict(BRACKET, blind_d=6.6, blind_depth=6.0)),
    "afrondbeugel": (round_bracket, {"h": 12.0, "x": [80.0], "y": [40.0], "R": 3.0, "afronding bovenrand": 2.0,
                                     "hx": [], "hy": []}),
    "tredeblok": (step_block, dict(BRACKET, **{"trede 1 positie": 60.0, "trede 1 hoogte": 6.0})),
    "sleufbeugel": (slot_bracket, SLOTTED),
    # de inham: wanden op x = 38,5 en 41,5, bodem op y = 38 (of 2, als het rapport de beugel gespiegeld neerlegt)
    "inhambeugel": (notched_bracket, dict(BRACKET, x=[80.0, 41.5, 38.5], y=[40.0, 38.0, 2.0])),
    "plate": (small_plate, PLATE),
    "washer": (washer, WASHER),
}


# ----------------------------------------------------------------------------- de scenario's

BASE = {"noise": 2.0, "jpeg": 88, "vignette": 0.25, "sharpen": 0.5}
SCENARIOS = {
    "basis": {},
    "realistisch": dict(BASE),
    "schaduw": dict(BASE, shadow=0.55),
    "donker": dict(BASE, albedo=0.08),
    "licht": dict(BASE, albedo=0.95),
    "warp": dict(BASE, warp=1.0),
    "gamma": dict(BASE, gamma=0.75),
    # toonkromme zoals een telefoon (sRGB, ~1/2,2), op grijs, zwart en blauw (V2, v0.10)
    "srgb": dict(BASE, gamma=1 / 2.2),
    "srgb_donker": dict(BASE, gamma=1 / 2.2, albedo=0.08),
    "srgb_blauw": dict(BASE, gamma=1 / 2.2, color=(0.04, 0.08, 0.30)),
    # lokale toonbewerking (HDR, v0.11): de kromme loopt over het beeld (~0,38 tot ~0,53), plus lokaal contrast
    "srgb_hdr": dict(BASE, gamma=1 / 2.2, gamma_spread=0.08, local_contrast=0.25),
    "srgb_hdr_donker": dict(BASE, gamma=1 / 2.2, gamma_spread=0.08, local_contrast=0.25, albedo=0.08),
    # de twee delen van srgb_hdr apart: alleen de lopende exponent, alleen het lokale contrast
    "srgb_spreid": dict(BASE, gamma=1 / 2.2, gamma_spread=0.08),
    "srgb_lokaal": dict(BASE, gamma=1 / 2.2, local_contrast=0.25),
    "hand": dict(BASE, hand_shadow=0.5),
    "top_spreid": dict(BASE, top_spread=45.0, top_tilt=10.0),
    "weinig_laag": dict(BASE, rings=((60.0, 12), (75.0, 8)), top_views=5, top_spread=40.0),
    "klein": dict(BASE, object="plate", offset=(-30, 20), top_spread=40.0),
    "klein_steil": dict(BASE, object="plate", rings=((60.0, 12), (75.0, 8)), top_views=5, top_spread=40.0,
                        top_tilt=6.0),
    "ring": dict(BASE, object="washer", offset=(40, 25), rings=((55.0, 12), (72.0, 10)), top_spread=40.0),
    "zwaar": dict(BASE, shadow=0.55, warp=1.0, hand_shadow=0.5, top_spread=40.0, top_tilt=8.0,
                  rings=((50.0, 12), (70.0, 10)), top_views=5),
    "zwaar_klein": dict(BASE, object="plate", shadow=0.55, warp=1.0, hand_shadow=0.5, top_spread=40.0, top_tilt=8.0,
                        rings=((50.0, 12), (70.0, 10)), top_views=5, albedo=0.2),
    # sleuven en uitsparingen (V15)
    "sleuf": dict(BASE, object="sleufbeugel"),
    "sleuf_donker": dict(BASE, object="sleufbeugel", albedo=0.08),
    "sleuf_licht": dict(BASE, object="sleufbeugel", albedo=0.95),
    "sleuf_schaduw": dict(BASE, object="sleufbeugel", shadow=0.55),
    "sleuf_weinig": dict(BASE, object="sleufbeugel", rings=((60.0, 12), (75.0, 8)), top_views=5, top_spread=40.0),
    "sleuf_gedraaid": dict(BASE, object="sleufbeugel", angle=-40.0, offset=(-6, 4)),
    # afgeschuinde of afgeronde bovenrand en trede (V17)
    "afschuining": dict(BASE, object="afschuinbeugel"),
    "afschuining_donker": dict(BASE, object="afschuinbeugel", albedo=0.08),
    "afschuining_steil": dict(BASE, object="afschuinbeugel", rings=((60.0, 12), (75.0, 8)), top_views=5,
                              top_spread=40.0),
    "afronding": dict(BASE, object="afrondbeugel"),
    "trede": dict(BASE, object="tredeblok"),
    "trede_schaduw": dict(BASE, object="tredeblok", shadow=0.55),
    # kleur als objectbewijs (V8): donkerblauw geanodiseerd (grijs ~0,09, zoals 'donker'), rood kunststof, en
    # warm licht (witbalans niet helemaal goed); de mat blijft zwart-wit
    "blauw": dict(BASE, color=(0.04, 0.08, 0.30)),
    "blauw_warm": dict(BASE, color=(0.04, 0.08, 0.30), tint=(1.0, 0.93, 0.80)),
    "rood": dict(BASE, color=(0.60, 0.10, 0.08)),
    "grijs_kleur": dict(BASE, albedo=0.08, color=(0.08, 0.08, 0.08), tint=(1.0, 0.93, 0.80)),
    "sleuf_blauw": dict(BASE, object="sleufbeugel", color=(0.04, 0.08, 0.30)),
    # verzonken gaten (V16): grijs, zwart, en met alleen steile foto's van boven
    "verzinking": dict(BASE, object="verzonken"),
    "verzinking_donker": dict(BASE, object="verzonken", albedo=0.08),
    "verzinking_steil": dict(BASE, object="verzonken", rings=((60.0, 12), (75.0, 8)), top_views=5, top_spread=40.0),
    # kamerboringen (V16, v0.10): grijs, zwart, en met alleen steile foto's van boven
    "kamerboring": dict(BASE, object="kamerbeugel"),
    "kamerboring_donker": dict(BASE, object="kamerbeugel", albedo=0.08),
    "kamerboring_steil": dict(BASE, object="kamerbeugel", rings=((60.0, 12), (75.0, 8)), top_views=5,
                              top_spread=40.0),
    # een kleine faas aan de gaten (v0.11): als verzinking Ø7,6
    "faas": dict(BASE, object="faasbeugel"),
    "faas_donker": dict(BASE, object="faasbeugel", albedo=0.08),
    # een blind gat (v0.11): in geen silhouet te zien, alleen in de grijswaarden
    "blind": dict(BASE, object="blindbeugel"),
    "blind_donker": dict(BASE, object="blindbeugel", albedo=0.08),
    # het onderdeel tussendoor even aangestoten (V27, v0.12): (dx, dy mm, draaiing °, vanaf foto)
    "duwtje": dict(BASE, nudge=(0.5, -0.3, 0.3, 20)),
    "duw_groot": dict(BASE, nudge=(2.0, 1.5, 1.0, 30)),
    # een kleine inham in de buitenrand (V14, v0.12): in het model, of gemeld
    "inham": dict(BASE, object="inhambeugel"),
}
# kleurscenario's die ook als grijsbeelden draaien (controle: wat geeft kleur extra, V8)
GRAY_CONTROLS = ("blauw", "grijs_kleur")


def describe(name: str) -> str:
    """Een scenario in één regel: het onderdeel en wat er anders is dan bij 'realistisch'."""
    cfg = SCENARIOS[name]
    parts = [cfg.get("object", "bracket")]
    if not cfg:
        parts.append("zonder ruis, JPEG, vignettering of verscherping")
    parts += [f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in cfg.items()
              if k != "object" and BASE.get(k) != v]
    return ", ".join(parts)


# ----------------------------------------------------------------------------- de scan renderen

def poses(target, cfg, rng) -> list:
    """Drie ringen schuine foto's en een paar foto's van boven, met een willekeurige afstand, richting en draaiing om
    de optische as (zoals uit de hand)."""
    out = []
    dist = cfg.get("distance", 330.0)
    for elev, count in cfg.get("rings", ((35.0, 14), (55.0, 14), (72.0, 12))):
        off = rng.uniform(0, 360.0 / count)
        for k in range(count):
            az = math.radians(off + 360.0 * k / count)
            el = math.radians(elev + rng.uniform(-4, 4))
            d = dist * rng.uniform(0.9, 1.1)
            c = target + d * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            R, t = look_at(c, target + rng.normal(0, cfg.get("aim_noise", 3.0), 3))
            out.append((f"ring{int(elev):02d}_{k:02d}", R, t))
    n_top = cfg.get("top_views", 6)
    for k in range(n_top):
        lateral = rng.normal(0, cfg.get("top_spread", 12.0), 2)
        tilt = math.radians(cfg.get("top_tilt", 0.0) + rng.uniform(0, 3))
        az = rng.uniform(0, 2 * math.pi)
        h = cfg.get("top_distance", dist) * rng.uniform(0.9, 1.1)
        c = target + np.array([lateral[0], lateral[1], h])
        aim = c + np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), -math.cos(tilt)]) * h
        aim[2] = target[2]
        R, t = look_at(c, aim)
        roll = rng.uniform(0, 2 * math.pi)  # telefoon willekeurig gedraaid om de optische as
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
        out.append((f"top_{k:02d}", Rz @ R, Rz @ t))
    return out


def shadow_raster(raster, mesh, light, strength: float = 0.5, soft_mm: float = 1.5):
    """De slagschaduw van het onderdeel op de mat (langs het licht op z = 0 geprojecteerd), met een zachte rand."""
    V, F = mesh
    L = np.asarray(light, float) / np.linalg.norm(light)
    P = V - (V[:, 2:3] / L[2]) * L
    M = raster.mat_to_pixel_matrix()
    uv = (np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))]) @ M.T)[:, :2]
    mask = np.zeros(raster.image.shape, np.uint8)
    for f in F:
        cv2.fillConvexPoly(mask, np.round(uv[f] * 16).astype(np.int32), 1, cv2.LINE_8, 4)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), soft_mm * raster.px_per_mm)
    img = raster.image.astype(np.float32) * (1.0 - strength * soft)
    return raster.__class__(np.clip(img, 0, 255).astype(np.uint8), raster.px_per_mm, raster.margin_mm, raster.spec)


def warp_field(img, amp, rng, scale: float = 250.0):
    """Een mat die niet vlak ligt, nagebootst met een vloeiende vervorming van het beeld (tot `amp` px)."""
    h, w = img.shape[:2]
    fx = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), scale)
    fy = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), scale)
    fx *= amp / (np.abs(fx).max() + 1e-9)
    fy *= amp / (np.abs(fy).max() + 1e-9)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, xs + fx, ys + fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def post(img, name, cfg, rng):
    """Nabewerking als in een telefoon; grijs (h, w) of kleur (h, w, 3, BGR). Kleur: dezelfde velden voor alle
    kanalen, ruis per kanaal, JPEG met 4:2:0-chromasubsampling (standaard in OpenCV en telefoons)."""
    f = img.astype(np.float32)
    h, w = f.shape[:2]
    bc = (lambda x: x[..., None]) if f.ndim == 3 else (lambda x: x)
    if cfg.get("vignette"):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        f *= bc(1.0 - cfg["vignette"] * r2)
    if cfg.get("hand_shadow") and name.startswith("top"):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = rng.uniform(0, w), rng.choice([rng.uniform(-100, 150), rng.uniform(h - 150, h + 100)])
        blob = np.exp(-(((xx - cx) / 380) ** 2 + ((yy - cy) / 300) ** 2))
        f *= bc(1.0 - cfg["hand_shadow"] * blob)
    if cfg.get("gamma"):
        g = cfg["gamma"]
        if cfg.get("gamma_spread"):  # lokale toonbewerking (HDR): de exponent loopt over het beeld
            gx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
            gy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
            g = bc(g + cfg["gamma_spread"] * (0.7 * gx + 0.3 * gy))
        f = 255.0 * (np.clip(f, 0, 255) / 255.0) ** g
        if cfg.get("local_contrast"):  # en lokaal contrast (grote straal), zoals HDR-bewerking op een telefoon
            f = f + cfg["local_contrast"] * (f - cv2.GaussianBlur(f, (0, 0), 40.0))
    if cfg.get("warp"):
        f = warp_field(f, cfg["warp"], rng)
    if cfg.get("sharpen"):
        f = f + cfg["sharpen"] * (f - cv2.GaussianBlur(f, (0, 0), 1.5))
    if cfg.get("noise"):
        f = f + rng.normal(0, cfg["noise"], f.shape)
    out = np.clip(f, 0, 255).astype(np.uint8)
    if cfg.get("jpeg"):
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(cfg["jpeg"])])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR if out.ndim == 3 else cv2.IMREAD_GRAYSCALE)
    return out


def make_scan(cfg: dict, seed: int = 1, mat: str = "A4") -> tuple[list, dict]:
    """De foto's van een scenario: [(naam, beeld)], en de waarheid."""
    spec = get_spec(mat)
    rng = np.random.default_rng(seed)
    make, truth = OBJECTS[cfg.get("object", "bracket")]
    angle, offset = cfg.get("angle", 17.0), cfg.get("offset", (5, -8))
    shape = place(make(), spec, angle_deg=angle, offset=offset)
    mesh = tessellate(shape)
    nudge = cfg.get("nudge")  # vanaf foto nudge[3] ligt het onderdeel iets anders (V27)
    moved = None if not nudge else tessellate(place(make(), spec, angle_deg=angle + nudge[2],
                                                    offset=(offset[0] + nudge[0], offset[1] + nudge[1])))
    raster = rasterize_board(spec, 10.0, 3.0)
    light = cfg.get("light", (0.35, -0.45, 0.82))
    if cfg.get("shadow"):
        raster = shadow_raster(raster, mesh, light, cfg["shadow"])
    cam = default_camera()
    lo, hi = mesh[0].min(axis=0), mesh[0].max(axis=0)
    target = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] / 2])
    images = []
    color = cfg.get("color")  # albedo per kanaal (R, G, B): een gekleurd onderdeel (V8)
    tint = np.asarray(cfg.get("tint", (1.0, 1.0, 1.0)), np.float32)  # kleur van het licht (R, G, B), na witbalans
    for k, (name, R, t) in enumerate(poses(target, cfg, rng)):
        m = moved if nudge and k >= nudge[3] else mesh
        if color is None:
            img, _ = render_view(raster, cam, R, t, m, albedo=cfg.get("albedo", 0.55), light=light,
                                 noise=0.0, gradient=cfg.get("gradient", 0.08), rng=rng)
        else:  # per kanaal renderen (lineair in de albedo), dan het licht erover; BGR zoals OpenCV
            chans = [render_view(raster, cam, R, t, m, albedo=a, light=light, noise=0.0,
                                 gradient=cfg.get("gradient", 0.08), rng=rng)[0].astype(np.float32) * k
                     for a, k in zip(color, tint)]
            img = np.clip(np.dstack(chans[::-1]), 0, 255).astype(np.uint8)
        images.append((name, post(img, name, cfg, rng)))
    return images, dict(truth)


# ----------------------------------------------------------------------------- een scenario draaien

def run(name: str, out_dir: str | Path, seed: int = 1, cfg: dict | None = None, cache_dir: str | Path | None = None,
        gray: bool = False, mat: str = "A4", log=print) -> dict:
    """Een scenario renderen (of uit `cache_dir` halen) en door de pijplijn halen. Schrijft `stress.json` in
    `out_dir/<naam>` (met `_<zaad>` als het zaad niet 1 is) en geeft die samenvatting."""
    t0 = time.time()
    cfg = SCENARIOS[name] if cfg is None else cfg
    label = name if seed == 1 else f"{name}_{seed}"
    cache = Path(cache_dir) / f"scan_{name}_{mat}_{seed}.pkl" if cache_dir else None
    if cache is not None and cache.exists():
        with open(cache, "rb") as fh:
            images, truth = pickle.load(fh)
    else:
        images, truth = make_scan(cfg, seed, mat)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as fh:
                pickle.dump((images, truth), fh)
    truth = dict(truth, **OBJECTS[cfg.get("object", "bracket")][1])  # oudere caches: een deel, met x en y als getal
    if gray:  # kleurscan als grijsbeelden (vergelijking zonder kleurbewijs, V8)
        images = [(n, cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) if i.ndim == 3 else i) for n, i in images]
    out = Path(out_dir) / label
    out.mkdir(parents=True, exist_ok=True)
    for n, img in images[:: max(1, len(images) // 6)]:
        cv2.imwrite(str(out / f"foto_{n}.jpg"), img)
    lines: list[str] = []
    try:
        res = run_scan(images, out / "resultaat", ScanOptions(mat=mat, max_evals=cfg.get("max_evals", 1500)),
                       log=lines.append, scan_name=name)
        dims = {d["name"]: round(float(d["measured"]), 3) for d in res["dimensions"]}
        status = "OK"
    except Exception as e:  # noqa: BLE001 - een stresstest meldt elke fout, en gaat door met de volgende
        dims = {}
        status = f"FOUT {type(e).__name__}: {e}"
        lines.append(traceback.format_exc(limit=3))
    summary = {"scenario": label, "status": status, "truth": truth, "dims": dims, "log": lines,
               "time": round(time.time() - t0, 1), "config": cfg, "seed": seed, "mat": mat, "gray": gray}
    (out / "stress.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"== {label}: {status}  ({summary['time']} s)")
    for line in lines:
        if not line.startswith("Traceback"):
            log("    " + line)
    if dims:
        log("    maten: " + str({k: v for k, v in dims.items() if not k.startswith("gat ") or "Ø" in k}))
    return summary


def _run_quiet(args: tuple) -> tuple[str, dict]:
    """Voor de parallelle uitvoering: het log pas aan het eind, in één stuk."""
    lines: list[str] = []
    summary = run(*args, log=lines.append)
    return "\n".join(lines), summary


def run_many(names: list[str], out_dir: str | Path, cache_dir: str | Path | None = None, gray: bool = False,
             mat: str = "A4", parallel: int = 1, log=print) -> list[dict]:
    """Meerdere scenario's (`naam` of `naam:zaad`), eventueel `parallel` tegelijk (elk in een eigen proces)."""
    jobs = []
    for n in names:
        name, _, seed = n.partition(":")
        if name not in SCENARIOS:
            raise ValueError(f"onbekend scenario '{name}'; kies uit: {', '.join(SCENARIOS)}")
        jobs.append((name, out_dir, int(seed) if seed else 1, None, cache_dir, gray, mat))
    if parallel <= 1:
        return [run(*j, log=log) for j in jobs]
    out = []
    with ProcessPoolExecutor(max_workers=parallel) as pool:
        for fut in as_completed([pool.submit(_run_quiet, j) for j in jobs]):
            text, summary = fut.result()
            log(text)
            out.append(summary)
    return out


# ----------------------------------------------------------------------------- twee runs vergelijken

def truth_of(name: str, value: float, t: dict) -> float | None:
    """De ware waarde van een maat uit het rapport (None: geen waarheid, bijv. een maat die er niet hoort)."""
    if name in t:  # bovenrand, trede
        return t[name]
    if name.startswith("verzinking"):
        return t.get("csk", 0.0)  # een verzinking waar er geen is: de fout is de hele maat
    if name.startswith("blind gat diepte"):
        return t.get("blind_depth", 0.0)
    if name.startswith("blind gat Ø"):
        return t.get("blind_d", 0.0)
    if name.startswith("kamerboring diepte"):
        return t.get("cb_depth", 0.0)
    if name.startswith("kamerboring"):
        return t.get("cb", 0.0)
    if name.endswith("bovenrand"):
        return 0.0  # geen afschuining in de waarheid
    for kind in ("sleuf", "uitsparing"):
        if name.startswith(kind + " "):
            what = name.split()[-1]
            tv = t.get(f"{kind} {what}")
            if isinstance(tv, list):
                c = min(tv, key=lambda v: abs(v - value))
                return c if abs(c - value) < 2.0 else None
            return tv
    if name == "hoogte":
        return t["h"]
    if name.startswith(("x-maat", "y-maat")):
        c = min(t[name[0]], key=lambda v: abs(v - value))
        return c if abs(c - value) < 2.0 else None
    if name == "diameter":
        return t["D"]
    if name.startswith("afronding"):
        return t["R"]
    if name.startswith("gat Ø"):
        return t["d"]
    if name.startswith("gat ") and name.endswith((" x", " y")):
        options = t["h" + name[-1]]
        if not options:
            return None
        c = min(options, key=lambda v: abs(v - value))
        return c if abs(c - value) < 2.0 else None
    return None


def kind_of(name: str) -> str:
    """Soort maat voor de samenvatting."""
    if name.startswith("verzinking"):
        return "verzinking"
    if name.startswith("kamerboring"):
        return "kamerboring"
    if name.startswith("blind gat"):
        return "blind gat"
    if name.endswith("bovenrand"):
        return "bovenrand"
    if name.startswith("trede "):
        return "trede"
    if name.startswith(("sleuf ", "uitsparing ")):
        return {"x": "positie", "y": "positie", "hoekstraal": "afronding"}.get(name.split()[-1], "sleuf")
    for prefix, kind in (("hoogte", "hoogte"), ("x-maat", "lengte"), ("y-maat", "lengte"), ("diameter", "lengte"),
                         ("afronding", "afronding"), ("gat Ø", "gat"), ("gat ", "positie")):
        if name.startswith(prefix):
            return kind
    return "overig"


KINDS = ("hoogte", "lengte", "gat", "verzinking", "kamerboring", "blind gat", "sleuf", "positie", "afronding",
         "bovenrand", "trede")


def _load(run_dir: Path, sc: str) -> tuple[dict | None, dict | None]:
    s, f = run_dir / sc / "stress.json", run_dir / sc / "resultaat" / "report.json"
    if not s.exists():
        return None, None
    st = json.loads(s.read_text(encoding="utf-8"))
    rep = json.loads(f.read_text(encoding="utf-8")) if f.exists() and st["status"] == "OK" else None
    return st, rep


def _truth(st: dict) -> dict:
    """De waarheid van een run: die van het onderdeel van het scenario (oudere runs bewaarden maar een deel ervan,
    met x en y als getal in plaats van als lijst), anders wat de run zelf bewaarde."""
    t = dict(st.get("truth") or {})
    name = st["scenario"]
    base = name if name in SCENARIOS else name.rsplit("_", 1)[0]  # naam_zaad
    if base in SCENARIOS:
        t.update(OBJECTS[SCENARIOS[base].get("object", "bracket")][1])
    return t


def compare(old: str | Path, new: str | Path, log=print) -> dict:
    """Twee runs naast elkaar (zie de moduletekst). Geeft de samenvatting: per groep ('alle', 'zonder waarschuwing')
    het aantal maten, de dekking door U95, de fouten en de snaps, en de rijen per maat."""
    old, new = Path(old), Path(new)
    rows = []
    method_shown = False
    for d in sorted(p for p in new.iterdir() if p.is_dir()):
        sc = d.name
        st_n, rep_n = _load(new, sc)
        if st_n is None:
            continue
        st_o, rep_o = _load(old, sc) if old.exists() else (None, None)
        t = _truth(st_n)
        warn = [w for w in (rep_n or {}).get("warnings", []) if w.startswith("onbetrouwbaar")]
        log(f"== {sc}: {st_n['status'][:70]} ({st_n['time']} s; oud {st_o['time'] if st_o else '-'} s)"
            + (f"  [{len(warn)} x onbetrouwbaar]" if warn else ""))
        if rep_n is None:
            continue
        old_dims = {x["name"]: x for x in (rep_o or {}).get("dimensions", [])}
        scale = float(rep_n["uncertainty_model"].get("scale_rel", 0.0))
        if not method_shown:
            log("   (U95-methode: " + str(rep_n["uncertainty_model"].get("methode", "-")) + ")")
            method_shown = True
        for x in rep_n["dimensions"]:
            tv = truth_of(x["name"], x["measured"], t)
            o = old_dims.get(x["name"])
            err = x["measured"] - tv if tv is not None else float("nan")
            e_old = o["measured"] - tv if (o and tv is not None) else float("nan")
            # een gerenderde scan heeft een exacte printschaal: de dekking zonder de printschaalterm
            u95 = 2 * math.sqrt(max(x["sigma"] ** 2 - (scale * abs(x["measured"])) ** 2, 0.0))
            inside = abs(err) <= u95 if tv is not None else None
            snap_ok = abs(x["value"] - tv) < 1e-6 if tv is not None else None
            old_snap_ok = (abs(o["value"] - tv) < 1e-6) if (o and o["snapped"] and tv is not None) else None
            rows.append({"scenario": sc, "name": x["name"], "err": err, "u95": u95, "err_old": e_old,
                         "u95_old": o["u95"] if o else float("nan"), "inside": inside, "warned": bool(warn),
                         "snapped": x["snapped"], "snap_ok": snap_ok, "old_snap_ok": old_snap_ok})
            flag = "" if inside in (None, True) else "  <-- buiten U95"
            snap = ("gesnapt " + ("goed" if snap_ok else "FOUT")) if x["snapped"] else "niet gesnapt"
            log(f"   {x['name']:22s} {x['measured']:8.3f} fout {err:+.3f} ± {u95:.3f} (rapport ± {x['u95']:.3f})"
                f"  | oud {e_old:+.3f} ± {(o['u95'] if o else float('nan')):.3f}  | {snap}{flag}")
        for w in warn:
            log("      " + w[:150])
    known = [r for r in rows if r["inside"] is not None]
    summary = {"rows": rows}
    for label, S in (("alle", known), ("zonder waarschuwing", [r for r in known if not r["warned"]])):
        if not S:
            continue
        e_new = np.array([abs(r["err"]) for r in S])
        e_old = np.array([abs(r["err_old"]) for r in S if not math.isnan(r["err_old"])])
        snaps = [r for r in S if r["snapped"]]
        old_in = [abs(r["err_old"]) <= r["u95_old"] for r in S
                  if not (math.isnan(r["err_old"]) or math.isnan(r["u95_old"]))]
        old_snaps = [r for r in S if r["old_snap_ok"] is not None]
        res = {"n": len(S), "inside": int(sum(r["inside"] for r in S)), "median": float(np.median(e_new)),
               "max": float(e_new.max()), "snapped": len(snaps), "wrong_snaps": sum(1 for r in snaps if not r["snap_ok"])}
        summary[label] = res
        old_txt = (f"{np.median(e_old):.3f}", f"{e_old.max():.3f}") if len(e_old) else ("-", "-")
        if old_in:
            log(f"\n{label}: oude U95 dekt {np.mean(old_in):.0%} van {len(old_in)} oude maten")
            log(f"{label}: oud gesnapt {len(old_snaps)}, waarvan fout "
                f"{sum(1 for r in old_snaps if not r['old_snap_ok'])}")
        else:
            log("")
        log(f"{label}: {res['n']} maten, binnen U95 {res['inside']} ({res['inside'] / res['n']:.0%}); |fout| mediaan "
            f"{res['median']:.3f} (oud {old_txt[0]}), max {res['max']:.3f} (oud {old_txt[1]}); gesnapt "
            f"{res['snapped']}, waarvan fout {res['wrong_snaps']}")
    clean = [r for r in known if not r["warned"]]
    log("\nper soort (zonder waarschuwing): |fout| mediaan/max nieuw vs oud, U95 (zonder printschaal) mediaan "
        "nieuw vs oud, dekking")
    for kind in KINDS:
        S = [r for r in clean if kind_of(r["name"]) == kind]
        if not S:
            continue
        en = np.array([abs(r["err"]) for r in S])
        eo = np.array([abs(r["err_old"]) for r in S if not math.isnan(r["err_old"])])
        un = np.array([r["u95"] for r in S])
        uo = np.array([r["u95_old"] for r in S if not math.isnan(r["u95_old"])])
        o_txt = f"{np.median(eo):.3f}/{eo.max():.3f}" if len(eo) else "-"
        uo_txt = f"{np.median(uo):.3f}" if len(uo) else "-"
        log(f"   {kind:11s} n {len(S):3d}: {np.median(en):.3f}/{en.max():.3f} vs {o_txt}; U95 {np.median(un):.3f} vs "
            f"{uo_txt}; binnen U95 {np.mean([r['inside'] for r in S]):.0%}")
    return summary
