"""QR-code in de terminal: de telefoon opent de uploadpagina door hem te scannen, zonder adres en toegangscode over te
typen. Zwart op wit via ANSI-kleuren, dus ook goed te scannen in een terminal met een donker kleurschema; met halve
blokken (twee modulerijen per regel) waar de console die kan tonen, anders met spaties."""

from __future__ import annotations

import os
import sys

QUIET = 4  # witrand in modules (ISO/IEC 18004)
HALF = {(False, False): " ", (True, True): "█", (True, False): "▀", (False, True): "▄"}
DARK_ON_LIGHT, RESET = "\x1b[30;107m", "\x1b[0m"


def available() -> bool:
    try:
        import segno  # noqa: F401
    except ImportError:
        return False
    return True


def matrix(text: str) -> list[list[bool]]:
    """De modules van de QR-code (True = donker), met de witrand."""
    import segno

    rows = [[bool(v) for v in row] for row in segno.make(text, error="m", micro=False).matrix]
    width = len(rows[0]) + 2 * QUIET
    blank = [[False] * width for _ in range(QUIET)]
    return blank + [[False] * QUIET + r + [False] * QUIET for r in rows] + [row[:] for row in blank]


def lines(text: str, unicode: bool = True) -> list[str]:
    """De QR-code als regels tekst. Met `unicode` halve blokken (twee modulerijen per regel), anders twee spaties
    per module met een donkere of lichte achtergrond (ook in een console die alleen cp1252 kent)."""
    m = matrix(text)
    if unicode:
        if len(m) % 2:
            m.append([False] * len(m[0]))
        return [DARK_ON_LIGHT + "".join(HALF[(t, b)] for t, b in zip(top, bottom)) + RESET
                for top, bottom in zip(m[0::2], m[1::2])]
    return ["".join("\x1b[40m  " if v else "\x1b[107m  " for v in row) + RESET for row in m]


def print_qr(text: str, stream=None, indent: str = "  ") -> bool:
    """Drukt de QR-code af als de uitvoer een terminal is en segno geïnstalleerd is; anders niets (False)."""
    stream = stream or sys.stdout
    if not getattr(stream, "isatty", lambda: False)() or not available():
        return False
    if os.name == "nt":
        os.system("")  # zet de ANSI-kleuren aan in de Windows-console
    try:
        "▀▄█".encode(getattr(stream, "encoding", None) or "ascii")
        unicode = True
    except (UnicodeEncodeError, LookupError):
        unicode = False
    for line in lines(text, unicode):
        print(indent + line, file=stream)
    return True
