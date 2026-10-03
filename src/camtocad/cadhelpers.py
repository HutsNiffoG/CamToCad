import math


def afgeronde_hoeken(punten):
    """Geometrie van een gesloten contour met afgeronde hoeken.

    punten: lijst van (x, y, r) tegen de klok in; r is de afrondingsstraal van die hoek (0 = scherp).
    Geeft per hoek (T1, M, T2, C, r): raakpunt op de inkomende rand, boogmidden, raakpunt op de
    uitgaande rand en boogmiddelpunt. Bij een scherpe hoek zijn T1 = T2 = de hoek en M = C = None.
    """
    n = len(punten)
    hoeken = []
    for k in range(n):
        x0, y0, _ = punten[k - 1]
        x1, y1, r = punten[k]
        x2, y2, _ = punten[(k + 1) % n]
        l1 = math.hypot(x1 - x0, y1 - y0)
        l2 = math.hypot(x2 - x1, y2 - y1)
        if r <= 0 or l1 == 0 or l2 == 0:
            hoeken.append(((x1, y1), None, (x1, y1), None, 0.0))
            continue
        d1x, d1y = (x1 - x0) / l1, (y1 - y0) / l1
        d2x, d2y = (x2 - x1) / l2, (y2 - y1) / l2
        phi = math.atan2(d1x * d2y - d1y * d2x, d1x * d2x + d1y * d2y)  # draaihoek, + = linksom
        if abs(phi) < 1e-9 or abs(phi) > math.pi - 1e-6:  # rechtdoor of (bijna) omkeren: geen boog
            hoeken.append(((x1, y1), None, (x1, y1), None, 0.0))
            continue
        t = r * math.tan(abs(phi) / 2)
        t1 = (x1 - t * d1x, y1 - t * d1y)
        t2 = (x1 + t * d2x, y1 + t * d2y)
        s = 1.0 if phi > 0 else -1.0
        c = (t1[0] - s * r * d1y, t1[1] + s * r * d1x)
        vx, vy = x1 - c[0], y1 - c[1]
        lv = math.hypot(vx, vy) or 1e-12
        m = (c[0] + r * vx / lv, c[1] + r * vy / lv)
        hoeken.append((t1, m, t2, c, r))
    return hoeken


def sleuf_hoeken(x, y, lengte, breedte, hoek, r):
    """Hoekpunten (x, y, r) van een sleuf of rechthoekige uitsparing, tegen de klok in.

    lengte langs de as (onder `hoek` graden), breedte dwars erop, r de hoekafronding; r = breedte / 2
    is een sleuf (langgat) met twee halve cirkels.
    """
    c, s = math.cos(math.radians(hoek)), math.sin(math.radians(hoek))
    punten = []
    for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        px, py = u * lengte / 2, v * breedte / 2
        punten.append((x + c * px - s * py, y + s * px + c * py, r))
    return punten


def bouw_contour(cq, punten, vlak="XY"):
    """Bouwt een gesloten CadQuery-contour (lijnen en bogen) uit hoekpunten met afrondingsstralen."""
    hoeken = afgeronde_hoeken(punten)
    start = hoeken[0][2]
    wp = cq.Workplane(vlak).moveTo(*start)
    huidig = start
    for k in range(1, len(hoeken) + 1):
        t1, m, t2, _, _ = hoeken[k % len(hoeken)]
        if math.hypot(t1[0] - huidig[0], t1[1] - huidig[1]) > 1e-7:
            wp = wp.lineTo(*t1)
            huidig = t1
        if m is not None:
            wp = wp.threePointArc(m, t2)
            huidig = t2
    return wp.close()


def extrudeer(schets, hoogte, bovenrand=None, maat=0.0):
    """Extrusie van een gesloten contour tot `hoogte`; `schets` is een functie die de contour (een
    CadQuery-workplane) geeft. Bovenrand "afschuining": rondom een afschuining onder 45° met benen `maat`;
    "afronding": een afronding met straal `maat`."""
    if bovenrand == "afschuining":
        onder = schets().extrude(hoogte - maat)
        return onder.union(schets().extrude(maat, taper=45).translate((0, 0, hoogte - maat)))
    model = schets().extrude(hoogte)
    if bovenrand == "afronding":
        model = model.faces(">Z").fillet(maat)
    return model


def verzinking(cq, x, y, d, dk, boven, hoek=90.0):
    """Kegel voor een verzinking rond het gat (x, y) met diameter `d`: diameter `dk` aan het bovenvlak (hoogte
    `boven`), onder een tophoek van `hoek` graden; 1 mm boven het bovenvlak doorgetrokken (geen samenvallende
    vlakken)."""
    t = math.tan(math.radians(hoek / 2))
    diepte = (dk - d) / 2 / t
    return (cq.Workplane("XY").workplane(offset=boven - diepte).center(x, y).circle(d / 2)
            .workplane(offset=diepte + 1).circle(dk / 2 + t).loft())


def trede(cq, model, x, y, hoek, hoogte, totaal):
    """Trede: voorbij de lijn door (x, y), in de richting `hoek` (graden, 0 = +X), is het deel maar `hoogte`
    hoog; `totaal` is de hoogte van het deel."""
    blok = cq.Workplane("XY").box(2000, 2000, totaal - hoogte + 1, centered=(False, True, False))
    return model.cut(blok.rotate((0, 0, 0), (0, 0, 1), hoek).translate((x, y, hoogte)))
