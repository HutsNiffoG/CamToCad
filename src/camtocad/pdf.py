"""Minimale vector-PDF-writer (één pagina, gevulde rechthoeken en tekst).

Bewust zonder externe afhankelijkheden: de kalibratiemat moet exact op schaal printen,
dus alles is vectorgeometrie in millimeters. Oorsprong linksonder, y omhoog (PDF-conventie).
"""

from __future__ import annotations

PT_PER_MM = 72.0 / 25.4


def _num(v: float) -> str:
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def _escape(text: str) -> str:
    if not text.isascii():
        raise ValueError("PDF-tekst moet ASCII zijn (standaardfont zonder embedding)")
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PdfCanvas:
    """Verzamelt tekenoperaties in mm en schrijft een PDF 1.4-bestand."""

    def __init__(self, width_mm: float, height_mm: float):
        self.width_mm = width_mm
        self.height_mm = height_mm
        self._ops: list[str] = []
        self._fill_gray: float | None = None
        self._path_open = False

    def _set_fill(self, gray: float) -> None:
        if self._fill_gray != gray:
            self._flush_path()
            self._ops.append(f"{_num(gray)} g")
            self._fill_gray = gray

    def _flush_path(self) -> None:
        if self._path_open:
            self._ops.append("f")
            self._path_open = False

    def rect(self, x: float, y: float, w: float, h: float, gray: float = 0.0) -> None:
        """Gevulde rechthoek; (x, y) is de linkeronderhoek in mm."""
        if w <= 0 or h <= 0:
            return
        self._set_fill(gray)
        k = PT_PER_MM
        self._ops.append(f"{_num(x * k)} {_num(y * k)} {_num(w * k)} {_num(h * k)} re")
        self._path_open = True

    def text(self, x: float, y: float, text: str, size_pt: float = 9.0, gray: float = 0.0) -> None:
        """Tekst in Helvetica; (x, y) is het beginpunt van de basislijn in mm."""
        self._flush_path()
        k = PT_PER_MM
        self._ops.append(
            f"BT /F1 {_num(size_pt)} Tf {_num(gray)} g {_num(x * k)} {_num(y * k)} Td ({_escape(text)}) Tj ET"
        )
        self._fill_gray = gray

    def to_bytes(self) -> bytes:
        self._flush_path()
        content = ("\n".join(self._ops) + "\n").encode("ascii")
        w, h = self.width_mm * PT_PER_MM, self.height_mm * PT_PER_MM
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_num(w)} {_num(h)}] "
                f"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
            ).encode("ascii"),
            b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"endstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        ]
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for i, body in enumerate(objects, start=1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
        xref_at = len(out)
        out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
        out += b"0000000000 65535 f \n"
        for off in offsets:
            out += f"{off:010d} 00000 n \n".encode("ascii")
        out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode("ascii")
        return bytes(out)

    def save(self, path) -> None:
        with open(path, "wb") as f:
            f.write(self.to_bytes())
