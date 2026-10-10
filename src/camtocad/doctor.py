"""`camtocad doctor`: controleert of de installatie klopt (V23, v0.15).

Een installatie kan op een paar manieren stil verkeerd gaan:

* twee OpenCV-pakketten naast elkaar (bijv. `opencv-python` en `opencv-python-headless`): ze schrijven allebei
  de module `cv2`, en welke je krijgt hangt af van de volgorde van installeren;
* een OpenCV zonder de ChArUco-detector, of een die de mathoeken anders legt (OpenCV 4.x legt ze ~0,5 px
  verschoven; de verwerking meet en corrigeert dat, `calib.detector_bias`);
* CadQuery met een OpenCascade-versie die er niet bij past (cadquery-ocp 8 breekt CadQuery 2.8);
* geen HEIC-decoder (iPhone-foto's), of de extra's voor de webserver ontbreken;
* te weinig geheugen of schijfruimte, of de poorten van de webserver zijn al in gebruik.

Elke controle geeft "ok", "let op" of "fout", met wat je eraan kunt doen. Bij een fout geeft de opdracht een
foutcode, zodat ook een installatiescript of CI erop kan testen.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

OK, LET_OP, FOUT = "ok", "let op", "fout"

# De OpenCV-pakketten op PyPI schrijven allemaal de module cv2; er mag er maar één geïnstalleerd zijn
OPENCV_DISTS = ("opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless")
MIN_PYTHON = (3, 11)  # CadQuery 2.8 vraagt Python 3.11
MIN_MEMORY_GB = 8.0  # een scan van 46 foto's gebruikt ~5 GB
MIN_DISK_GB = 2.0  # een scan: ~0,2 GB foto's en ~0,05 GB resultaten
SERVER_PORTS = (8000, 8443)


@dataclass
class Check:
    name: str
    status: str  # OK, LET_OP of FOUT
    detail: str
    remedy: str = ""

    def to_dict(self) -> dict:
        return {"naam": self.name, "status": self.status, "detail": self.detail, "oplossing": self.remedy}


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _major_minor(version: str) -> tuple[int, int]:
    parts = (version.split(".") + ["0", "0"])[:2]
    try:
        return int(parts[0]), int("".join(c for c in parts[1] if c.isdigit()) or 0)
    except ValueError:
        return 0, 0


def check_python() -> Check:
    v = sys.version_info
    detail = f"Python {platform.python_version()} ({platform.system()} {platform.machine()})"
    if (v.major, v.minor) < MIN_PYTHON:
        return Check("Python", FOUT, detail, "installeer Python 3.11 of nieuwer")
    return Check("Python", OK, detail)


def check_opencv(installed: dict[str, str | None] | None = None) -> Check:
    """Precies één OpenCV-pakket, met de ChArUco-detector. `installed`: naam -> versie (voor de tests)."""
    if installed is None:
        installed = {d: _version(d) for d in OPENCV_DISTS}
    present = {d: v for d, v in installed.items() if v}
    if len(present) > 1:
        names = ", ".join(f"{d} {v}" for d, v in sorted(present.items()))
        return Check("OpenCV", FOUT, f"meer dan één OpenCV-pakket geïnstalleerd: {names}",
                     "ze schrijven allemaal de module cv2: verwijder ze (pip uninstall " + " ".join(sorted(present))
                     + ") en installeer alleen opencv-python-headless")
    try:
        import cv2
    except ImportError as e:
        return Check("OpenCV", FOUT, f"cv2 niet te laden: {e}", "pip install opencv-python-headless")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
        return Check("OpenCV", FOUT, f"OpenCV {cv2.__version__} zonder ChArUco-detector",
                     "installeer opencv-python-headless 4.10 of nieuwer")
    if _major_minor(cv2.__version__) < (4, 10):
        return Check("OpenCV", LET_OP, f"OpenCV {cv2.__version__}: ouder dan getest (4.10)",
                     "pip install -U opencv-python-headless")
    dist = next(iter(present), None)
    return Check("OpenCV", OK, f"OpenCV {cv2.__version__}" + (f" ({dist})" if dist else ""))


def check_detector() -> Check:
    """Vindt de ChArUco-detector de mat, en waar legt hij de hoeken? Gemeten op gerenderde beelden van de mat."""
    try:
        import cv2
        from .calib import detector_bias
        from .mat import PRESETS
        t0 = time.perf_counter()
        bias = detector_bias(PRESETS["A4"])
        ms = 1000 * (time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001 - elke fout hier is een kapotte installatie
        return Check("Matdetectie", FOUT, f"de mat is in een gerenderd beeld niet te vinden: {e}",
                     "installeer OpenCV opnieuw (pip install --force-reinstall opencv-python-headless)")
    size = float(abs(bias).max())
    detail = (f"ChArUco-hoeken {bias[0]:+.2f} / {bias[1]:+.2f} px verschoven (OpenCV {cv2.__version__}; wordt "
              f"gecorrigeerd; {ms:.0f} ms)")
    if size > 1.0:
        return Check("Matdetectie", LET_OP, detail, "onverwacht grote verschuiving: meld deze OpenCV-versie")
    return Check("Matdetectie", OK, detail)


def check_cadquery(tmp_dir: Path | None = None) -> Check:
    """CadQuery en OpenCascade: een blokje met een gat maken en als STEP schrijven."""
    try:
        import cadquery as cq
    except Exception as e:  # noqa: BLE001
        return Check("CadQuery", FOUT, f"CadQuery niet te laden: {e}", "pip install 'cadquery>=2.8,<3'")
    ocp = _version("cadquery-ocp") or _version("cadquery-ocp-novtk")
    detail = f"CadQuery {_version('cadquery') or '?'}, OpenCascade (OCP) {ocp or '?'}"
    try:
        with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
            path = Path(tmp) / "proef.step"
            part = cq.Workplane("XY").box(20, 10, 5).faces(">Z").workplane().hole(3)
            cq.exporters.export(part, str(path))
            ok = path.exists() and path.stat().st_size > 1000
    except Exception as e:  # noqa: BLE001
        return Check("CadQuery", FOUT, f"{detail}: een STEP-bestand schrijven mislukt: {e}",
                     "installeer CadQuery opnieuw: pip install --force-reinstall 'cadquery>=2.8,<3'")
    if not ok:
        return Check("CadQuery", FOUT, f"{detail}: het STEP-bestand is leeg", "installeer CadQuery opnieuw")
    if ocp and _major_minor(ocp) >= (8, 0) and _major_minor(_version("cadquery") or "0") < (2, 9):
        return Check("CadQuery", LET_OP, f"{detail}: deze combinatie is niet getest",
                     "pip install 'cadquery-ocp>=7.9.3,<8'")
    return Check("CadQuery", OK, detail + ", STEP schrijven werkt")


def check_heic() -> Check:
    from .imgio import heif_supported
    if heif_supported():
        dist = next((d for d in ("pi-heif", "pillow-heif") if _version(d)), "pi-heif")
        return Check("HEIC (iPhone)", OK, f"{dist} {_version(dist) or ''}".strip())
    return Check("HEIC (iPhone)", LET_OP, "geen HEIC-decoder: foto's van een iPhone (.heic) zijn niet te lezen",
                 "pip install 'camtocad[heic]' (of zet de iPhone op 'Meest compatibel': JPG)")


def check_server() -> Check:
    """De extra's voor `camtocad server`: webserver, uploads, QR-code en https (de camera op de telefoon)."""
    need = {"fastapi": "fastapi", "uvicorn": "uvicorn", "python-multipart": "python-multipart"}
    missing = [n for n, d in need.items() if not _version(d)]
    if missing:
        return Check("Webserver", LET_OP, "niet geïnstalleerd: " + ", ".join(missing),
                     "pip install 'camtocad[server]' (alleen nodig voor camtocad server)")
    notes = []
    if not _version("cryptography"):
        notes.append("geen https (cryptography): de camera op de telefoon werkt dan niet")
    if not _version("segno"):
        notes.append("geen QR-code (segno)")
    if notes:
        return Check("Webserver", LET_OP, "; ".join(notes), "pip install 'camtocad[server]'")
    return Check("Webserver", OK, f"FastAPI {_version('fastapi')}, uvicorn {_version('uvicorn')}, https en QR-code")


def total_memory_gb() -> float | None:
    """Het werkgeheugen van deze pc (GB), of None als dat niet te bepalen is."""
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = Status()
            st.dwLength = ctypes.sizeof(Status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return None
            return st.ullTotalPhys / 1024 ** 3
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (AttributeError, ValueError, OSError):
        return None


def check_memory(total_gb: float | None = None) -> Check:
    total_gb = total_memory_gb() if total_gb is None else total_gb
    cpus = os.cpu_count() or 1
    if total_gb is None:
        return Check("Geheugen", LET_OP, f"onbekend; {cpus} rekenkernen", "een scan gebruikt ~5 GB werkgeheugen")
    detail = f"{total_gb:.1f} GB werkgeheugen, {cpus} rekenkernen"
    if total_gb < MIN_MEMORY_GB:
        return Check("Geheugen", LET_OP, detail, "een scan van 46 foto's gebruikt ~5 GB: sluit andere programma's, "
                                                 "of verwerk met minder foto's of een lagere --max-zijde")
    return Check("Geheugen", OK, detail)


def check_data_dir(data_dir: Path) -> Check:
    """De datamap van de webserver (of een andere map om in te werken): schrijfbaar, en genoeg vrije ruimte."""
    data_dir = Path(data_dir).expanduser()
    probe = data_dir
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free_gb = shutil.disk_usage(probe).free / 1024 ** 3
    except OSError as e:
        return Check("Datamap", FOUT, f"{data_dir}: {e}", "kies een andere map met --data")
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=data_dir):
            pass
    except OSError as e:
        return Check("Datamap", FOUT, f"{data_dir} is niet schrijfbaar: {e}", "kies een andere map met --data")
    detail = f"{data_dir}: schrijfbaar, {free_gb:.1f} GB vrij"
    if free_gb < MIN_DISK_GB:
        return Check("Datamap", LET_OP, detail, "een scan kost ~0,25 GB: maak ruimte vrij of kies een andere map")
    return Check("Datamap", OK, detail)


def check_ports(ports=SERVER_PORTS) -> Check:
    """Zijn de poorten van de webserver vrij (http en https)?"""
    busy = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
            except OSError:
                busy.append(port)
    if busy:
        return Check("Poorten", LET_OP, "in gebruik: " + ", ".join(map(str, busy)),
                     "draait camtocad server al? Anders: camtocad server --poort 8001 --https-poort 8444")
    return Check("Poorten", OK, "vrij: " + ", ".join(map(str, ports)))


def run_checks(data_dir: Path, quick: bool = False) -> list[Check]:
    """Alle controles; `quick` slaat de trage over (matdetectie en CadQuery, samen een paar seconden)."""
    from . import __version__
    out = [Check("Cam-to-CAD", OK, f"versie {__version__} in {Path(__file__).parent}"), check_python()]
    for dist, mod in (("numpy", "numpy"), ("scipy", "scipy"), ("pillow", "PIL")):
        v = _version(dist)
        try:
            __import__(mod)
            out.append(Check(dist, OK, v or "?"))
        except Exception as e:  # noqa: BLE001
            out.append(Check(dist, FOUT, f"niet te laden: {e}", f"pip install --force-reinstall {dist}"))
    out.append(check_opencv())
    if not quick:
        out.append(check_detector())
        out.append(check_cadquery())
    out += [check_heic(), check_server(), check_memory(), check_data_dir(data_dir), check_ports()]
    return out


def report_lines(checks: list[Check]) -> list[str]:
    mark = {OK: "ok     ", LET_OP: "LET OP ", FOUT: "FOUT   "}
    lines = ["Cam-to-CAD: controle van de installatie", ""]
    for c in checks:
        lines.append(f"  {mark[c.status]} {c.name}: {c.detail}")
        if c.remedy and c.status != OK:
            lines.append(f"          -> {c.remedy}")
    errors = sum(c.status == FOUT for c in checks)
    warnings = sum(c.status == LET_OP for c in checks)
    lines.append("")
    if errors:
        lines.append(f"{errors} fout(en): los die eerst op. " + (f"{warnings} punt(en) om op te letten." if warnings
                                                                    else ""))
    elif warnings:
        lines.append(f"Alles werkt; {warnings} punt(en) om op te letten.")
    else:
        lines.append("Alles in orde.")
    return lines
