import io

import cv2
import numpy as np
import pytest

pytest.importorskip("segno")
from camtocad.server import qr  # noqa: E402

URL = "http://192.168.1.20:8000/?token=Ab3_x9Qk"


def decode(rows: list[list[bool]], px: int = 8) -> str:
    img = np.where(np.array(rows), 0, 255).astype(np.uint8)
    img = cv2.resize(img, None, fx=px, fy=px, interpolation=cv2.INTER_NEAREST)
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    return text


def unpack_half_blocks(lines: list[str]) -> list[list[bool]]:
    top = {" ": False, "█": True, "▀": True, "▄": False}
    bottom = {" ": False, "█": True, "▀": False, "▄": True}
    rows = []
    for line in lines:
        body = line.replace(qr.DARK_ON_LIGHT, "").replace(qr.RESET, "")
        rows += [[top[c] for c in body], [bottom[c] for c in body]]
    return rows


def unpack_spaces(lines: list[str]) -> list[list[bool]]:
    rows = []
    for line in lines:
        cells = line.replace(qr.RESET, "").split("\x1b[")[1:]
        rows.append([c.startswith("40m") for c in cells])
    return rows


def test_qr_code_in_the_terminal_scans_back_to_the_url():
    """Zwart op wit (ANSI), met witrand: zowel met halve blokken als met spaties leest een QR-lezer het adres."""
    m = qr.matrix(URL)
    assert not any(m[0]) and not any(r[0] for r in m) and decode(m) == URL
    half = qr.lines(URL, unicode=True)
    assert len(half) == (len(m) + 1) // 2 and all(line.startswith(qr.DARK_ON_LIGHT) for line in half)
    assert decode(unpack_half_blocks(half)[:len(m)]) == URL
    assert decode(unpack_spaces(qr.lines(URL, unicode=False))) == URL


class FakeTerminal(io.StringIO):
    def __init__(self, encoding: str, tty: bool = True):
        super().__init__()
        self._enc, self._tty = encoding, tty

    @property
    def encoding(self):
        return self._enc

    def isatty(self):
        return self._tty


def test_qr_code_only_on_a_terminal_and_without_blocks_in_cp1252():
    out = FakeTerminal("utf-8")
    assert qr.print_qr(URL, out) and any(c in out.getvalue() for c in "▀▄█")
    legacy = FakeTerminal("cp1252")
    assert qr.print_qr(URL, legacy) and not any(c in legacy.getvalue() for c in "▀▄█")
    piped = FakeTerminal("utf-8", tty=False)
    assert not qr.print_qr(URL, piped) and piped.getvalue() == ""
