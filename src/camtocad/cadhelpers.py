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
        if abs(phi) < 1e-9:
            hoeken.append(((x1, y1), None, (x1, y1), None, 0.0))
            continue
        t = r * math.tan(abs(phi) / 2)
        t1 = (x1 - t * d1x, y1 - t * d1y)
        t2 = (x1 + t * d2x, y1 + t * d2y)
        s = 1.0 if phi > 0 else -1.0
        c = (t1[0] - s * r * d1y, t1[1] + s * r * d1x)
        vx, vy = x1 - c[0], y1 - c[1]
        lv = math.hypot(vx, vy)
        m = (c[0] + r * vx / lv, c[1] + r * vy / lv)
        hoeken.append((t1, m, t2, c, r))
    return hoeken


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
