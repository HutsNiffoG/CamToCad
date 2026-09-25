"""Lokale webserver (route A): uploaden vanaf de telefoon via wifi, verwerken op deze pc.

Geen cloud en geen app-installatie: open de getoonde URL in de browser van je telefoon, maak of
kies de foto's en upload ze. Elke foto wordt direct gecontroleerd (mat, scherpte, belichting,
kijkhoek) en de pagina laat zien welke foto's nog ontbreken; daarna start je de verwerking. Alle
data blijft in de datamap op deze pc. Toegang vereist een toegangscode (token), omdat de server
op het lokale netwerk luistert.
"""

from __future__ import annotations

import json
import queue
import re
import secrets
import shutil
import socket
import sys
import threading
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .. import preflight
from ..mat import PRESETS, get_spec, write_mat
from ..pipeline import IMAGE_EXT, ScanOptions, run_scan

STATIC = Path(__file__).parent / "static"
OUTPUTS = {"model.step", "model.stl", "model.py", "report.html", "report.json"}
DEBUG_FILE = re.compile(r"^(masker_[\w.-]{1,80}\.jpg|lokalisatie\.png|bovenaanzicht\.png|diagnose\.json)$")
PHOTO_FILE = re.compile(r"^foto_\d{4}\.[a-z]{3,4}$")
MAX_FILES = 300
MAX_BYTES = 40 * 1024 * 1024
MAX_REQUEST = 2 * 1024 ** 3  # hele upload; losse bestanden blijven onder MAX_BYTES
JOB_ID = re.compile(r"^[0-9a-f]{12}$")
BUSY = ("wachtrij", "bezig")


def _mat_choice(mat: str) -> str:
    if mat.lower() == "auto":
        return "auto"
    if mat.upper() not in PRESETS:
        raise HTTPException(400, f"Onbekende mat: {mat}")
    return PRESETS[mat.upper()].name


def _rulers(meetlijn: float, meetlijn_y: float | None) -> list[float]:
    y = meetlijn if meetlijn_y is None else meetlijn_y
    for v in (meetlijn, y):
        if not 95.0 <= v <= 105.0:
            raise HTTPException(400, "Meetlijn buiten 95-105 mm: print de mat opnieuw op 100%")
    return [float(meetlijn), float(y)]


class JobStore:
    """Scans op schijf: <data>/<id>/fotos, <data>/<id>/resultaat, status.json en controle.json."""

    def __init__(self, root: Path, runner=run_scan):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue: queue.Queue[str] = queue.Queue()
        self.runner = runner
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def lock(self, job_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(job_id, threading.Lock())

    def path(self, job_id: str) -> Path:
        if not JOB_ID.match(job_id):
            raise HTTPException(404, "Onbekende scan")
        p = self.root / job_id
        if not p.is_dir():
            raise HTTPException(404, "Onbekende scan")
        return p

    def create(self, mat: str = "auto", meetlijn=(100.0, 100.0)) -> str:
        job_id = uuid.uuid4().hex[:12]
        (self.root / job_id / "fotos").mkdir(parents=True)
        self.write(job_id, {"id": job_id, "state": "upload", "mat": mat, "meetlijn": list(meetlijn),
                            "created": time.time(), "photos": 0, "log": []})
        return job_id

    def read(self, job_id: str) -> dict:
        return json.loads((self.path(job_id) / "status.json").read_text(encoding="utf-8"))

    def write(self, job_id: str, data: dict, name: str = "status.json") -> None:
        target = self.root / job_id / name
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(target)

    def update(self, job_id: str, **changes) -> dict:
        data = self.read(job_id)
        data.update(changes)
        self.write(job_id, data)
        return data

    def list(self) -> list[dict]:
        out = []
        for p in self.root.iterdir():
            if p.is_dir() and JOB_ID.match(p.name) and (p / "status.json").exists():
                out.append(self.read(p.name))
        return sorted(out, key=lambda d: d.get("created", 0), reverse=True)

    def public(self, job_id: str, data: dict | None = None) -> dict:
        data = {k: v for k, v in (data or self.read(job_id)).items() if k != "trace"}
        dbg = self.root / job_id / "resultaat" / "debug"
        data["debug"] = sorted(p.name for p in dbg.iterdir() if DEBUG_FILE.match(p.name)) if dbg.is_dir() else []
        return data

    # --- foto's en controle ---------------------------------------------------------------

    def save_photo(self, job_id: str, filename: str, stream) -> Path | None:
        """Slaat één upload op als foto_NNNN.ext; None als het geen foto is (op extensie)."""
        ext = Path(filename or "").suffix.lower()
        if ext not in IMAGE_EXT:
            return None
        folder = self.path(job_id) / "fotos"
        used = {int(p.stem[5:]) for p in folder.iterdir() if PHOTO_FILE.match(p.name)}
        target = folder / f"foto_{max(used, default=-1) + 1:04d}{ext}"
        size = 0
        try:
            with open(target, "wb") as out:
                while chunk := stream.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, f"{filename}: bestand te groot")
                    out.write(chunk)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return target

    def checks(self, job_id: str) -> dict[str, preflight.PhotoCheck]:
        f = self.path(job_id) / "controle.json"
        if not f.exists():
            return {}
        raw = json.loads(f.read_text(encoding="utf-8"))
        return {n: preflight.PhotoCheck.from_dict(d) for n, d in raw.get("fotos", {}).items()}

    def _scan_spec(self, job_id: str, checks: dict):
        status = self.read(job_id)
        seen = Counter(c.mat for c in checks.values() if c.mat)
        if seen:
            return get_spec(seen.most_common(1)[0][0])
        return None if status.get("mat", "auto") == "auto" else get_spec(status["mat"])

    def check_photo(self, job_id: str, path: Path) -> preflight.PhotoCheck:
        """Controleert een opgeslagen foto en bewaart het oordeel in controle.json."""
        with self.lock(job_id):
            spec = self._scan_spec(job_id, self.checks(job_id))
        chk = preflight.check_file(path, spec)  # buiten het slot: ~0,1-0,3 s
        with self.lock(job_id):
            checks = self.checks(job_id)
            checks[path.name] = chk
            self.write(job_id, {"fotos": {n: c.to_dict() for n, c in checks.items()}}, "controle.json")
            self._count(job_id)
        return chk

    def remove_photo(self, job_id: str, name: str) -> None:
        if not PHOTO_FILE.match(name):
            raise HTTPException(404, "Onbekende foto")
        with self.lock(job_id):
            (self.path(job_id) / "fotos" / name).unlink(missing_ok=True)
            checks = self.checks(job_id)
            checks.pop(name, None)
            self.write(job_id, {"fotos": {n: c.to_dict() for n, c in checks.items()}}, "controle.json")
            self._count(job_id)

    def _count(self, job_id: str) -> None:
        n = sum(1 for p in (self.path(job_id) / "fotos").iterdir() if p.suffix.lower() in IMAGE_EXT)
        self.update(job_id, photos=n)

    def overview(self, job_id: str) -> dict:
        checks = self.checks(job_id)
        folder = self.path(job_id) / "fotos"
        present = [c for n, c in sorted(checks.items()) if (folder / n).exists()]
        return {"fotos": [c.public() for c in present],
                "overzicht": preflight.summarize(present, self._scan_spec(job_id, checks))}

    # --- verwerken ------------------------------------------------------------------------

    def process(self, job_id: str) -> None:
        base = self.path(job_id)
        status = self.update(job_id, state="bezig", started=time.time(), error=None, log=[])
        shutil.rmtree(base / "resultaat", ignore_errors=True)  # geen resten van een vorige run
        lines: list[str] = []

        def log(msg: str) -> None:
            lines.append(msg)
            self.update(job_id, log=lines[-50:])

        try:
            rulers = status.get("meetlijn", 100.0)
            rulers = [rulers, rulers] if isinstance(rulers, (int, float)) else rulers
            opts = ScanOptions(mat=status.get("mat", "auto"), mat_scale=(rulers[0] / 100.0, rulers[1] / 100.0))
            result = self.runner(base / "fotos", base / "resultaat", opts, log=log, scan_name=job_id)
            self.update(job_id, state="klaar", finished=time.time(), summary=result.get("summary", {}),
                        warnings=result.get("warnings", [])[:20])
        except Exception as e:  # noqa: BLE001 - de fout moet in de UI zichtbaar worden
            self.update(job_id, state="fout", finished=time.time(), error=str(e) or e.__class__.__name__,
                        trace=traceback.format_exc()[-2000:])

    def start(self, job_id: str, run_inline: bool = False) -> None:
        with self.lock(job_id):
            status = self.read(job_id)
            if status["state"] in BUSY:
                raise HTTPException(409, "Deze scan wordt al verwerkt")
            if status.get("photos", 0) < 6:
                raise HTTPException(400, "Minimaal 6 foto's nodig (beter 30-60)")
            self.update(job_id, state="wachtrij")
        if run_inline:
            self.process(job_id)
        else:
            self.queue.put(job_id)

    def remove(self, job_id: str) -> None:
        shutil.rmtree(self.root / job_id, ignore_errors=True)
        with self._guard:
            self._locks.pop(job_id, None)

    def worker(self) -> None:
        while True:
            job_id = self.queue.get()
            try:
                self.process(job_id)
            except Exception:  # noqa: BLE001 - de enige worker mag nooit stoppen
                traceback.print_exc(file=sys.stderr)


def create_app(data_dir: Path, token: str | None, run_inline: bool = False, runner=run_scan) -> FastAPI:
    app = FastAPI(title="Cam-to-CAD (lokaal)", docs_url=None, redoc_url=None)
    store = JobStore(Path(data_dir), runner)
    if not run_inline:
        threading.Thread(target=store.worker, daemon=True).start()

    def authorized(request: Request) -> bool:
        if token is None:
            return True
        given = (request.query_params.get("token") or request.cookies.get("ctc_token")
                 or request.headers.get("x-token") or "")
        return secrets.compare_digest(given, token)

    def check(request: Request) -> None:
        if not authorized(request):
            raise HTTPException(401, "Ongeldige of ontbrekende toegangscode (token)")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # vóórdat de body gelezen wordt: anders spoolt een upload zonder token eerst gigabytes naar schijf
        if not authorized(request):
            return JSONResponse({"detail": "Ongeldige of ontbrekende toegangscode (token)"}, status_code=401)
        length = request.headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > MAX_REQUEST:
            return JSONResponse({"detail": "Upload te groot"}, status_code=413)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        check(request)
        resp = HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))
        if token is not None:
            resp.set_cookie("ctc_token", token, httponly=True, samesite="strict")
        return resp

    @app.get("/api/scans")
    def list_scans(request: Request):
        check(request)
        return [store.public(s["id"], s) for s in store.list()]

    @app.get("/api/scans/{job_id}")
    def get_scan(job_id: str, request: Request):
        check(request)
        return store.public(job_id)

    @app.post("/api/scans")
    def create_scan(request: Request, fotos: list[UploadFile] | None = File(None), mat: str = Form("auto"),
                    meetlijn: float = Form(100.0), meetlijn_y: float | None = Form(None)):
        """Nieuwe scan. Met foto's erbij wordt hij meteen verwerkt (alles in één keer); zonder foto's
        volgen die één voor één via /api/scans/{id}/fotos, met directe controle, en daarna /start."""
        check(request)
        mat, rulers = _mat_choice(mat), _rulers(meetlijn, meetlijn_y)
        if fotos is not None and len(fotos) > MAX_FILES:
            raise HTTPException(400, f"Upload hooguit {MAX_FILES} foto's")
        job_id = store.create(mat, rulers)
        if not fotos:
            return {"id": job_id, "photos": 0}
        try:
            saved = sum(store.save_photo(job_id, f.filename, f.file) is not None for f in fotos)
            if saved == 0:
                raise HTTPException(400, "Geen bruikbare foto's (JPG/PNG) ontvangen")
        except BaseException:
            store.remove(job_id)  # geen halve scans laten staan
            raise
        store.update(job_id, photos=saved, state="wachtrij")
        if run_inline:
            store.process(job_id)
        else:
            store.queue.put(job_id)
        return {"id": job_id, "photos": saved}

    @app.post("/api/scans/{job_id}/fotos")
    def add_photos(job_id: str, request: Request, fotos: list[UploadFile] = File(...)):
        """Foto's toevoegen (ook aan een scan die al verwerkt is); elke foto wordt direct gecontroleerd."""
        check(request)
        status = store.read(job_id)
        if status["state"] in BUSY:
            raise HTTPException(409, "Deze scan wordt verwerkt: wacht tot hij klaar is")
        if status.get("photos", 0) + len(fotos) > MAX_FILES:
            raise HTTPException(400, f"Hooguit {MAX_FILES} foto's per scan")
        results = []
        for f in fotos:
            path = store.save_photo(job_id, f.filename, f.file)
            if path is None:
                results.append({"name": f.filename, "verdict": "onbruikbaar", "notes": ["geen JPG of PNG"]})
                continue
            results.append({**store.check_photo(job_id, path).public(), "upload": f.filename})
        if status["state"] != "upload":
            store.update(job_id, state="upload")  # opnieuw verwerken met de extra foto's
        return {"nieuw": results, **store.overview(job_id)}

    @app.get("/api/scans/{job_id}/controle")
    def get_checks(job_id: str, request: Request):
        check(request)
        return store.overview(job_id)

    @app.delete("/api/scans/{job_id}/fotos/{name}")
    def delete_photo(job_id: str, name: str, request: Request):
        check(request)
        if store.read(job_id)["state"] in BUSY:
            raise HTTPException(409, "Deze scan wordt verwerkt")
        store.remove_photo(job_id, name)
        return store.overview(job_id)

    @app.post("/api/scans/{job_id}/start")
    def start_scan(job_id: str, request: Request, mat: str | None = Form(None), meetlijn: float | None = Form(None),
                   meetlijn_y: float | None = Form(None)):
        """Verwerking starten; mat en meetlijnen mogen hier nog aangepast worden."""
        check(request)
        changes = {}
        if mat is not None:
            changes["mat"] = _mat_choice(mat)
        if meetlijn is not None:
            changes["meetlijn"] = _rulers(meetlijn, meetlijn_y)
        if changes:
            if store.read(job_id)["state"] in BUSY:
                raise HTTPException(409, "Deze scan wordt al verwerkt")
            store.update(job_id, **changes)
        store.start(job_id, run_inline)
        return store.public(job_id)

    @app.delete("/api/scans/{job_id}")
    def delete_scan(job_id: str, request: Request):
        check(request)
        if store.read(job_id)["state"] in BUSY:
            raise HTTPException(409, "Deze scan wordt verwerkt")
        store.remove(job_id)
        return {"id": job_id, "verwijderd": True}

    @app.get("/scans/{job_id}/debug/{name}")
    def debug_file(job_id: str, name: str, request: Request):
        check(request)
        if not DEBUG_FILE.match(name):
            raise HTTPException(404, "Onbekend bestand")
        path = store.path(job_id) / "resultaat" / "debug" / name
        if not path.exists():
            raise HTTPException(404, "Niet beschikbaar")
        return FileResponse(path)

    @app.get("/scans/{job_id}/{name}")
    def result_file(job_id: str, name: str, request: Request):
        check(request)
        if name not in OUTPUTS:
            raise HTTPException(404, "Onbekend bestand")
        path = store.path(job_id) / "resultaat" / name
        if not path.exists():
            raise HTTPException(404, "Nog niet beschikbaar")
        return FileResponse(path, filename=None if name == "report.html" else f"{job_id}_{name}")

    @app.get("/mat/{formaat}.pdf")
    def mat_pdf(formaat: str, request: Request):
        check(request)
        if formaat.upper() not in PRESETS:
            raise HTTPException(404, "Onbekend formaat")
        spec = PRESETS[formaat.upper()]
        paths = write_mat(spec, Path(data_dir) / "_mat")
        return FileResponse(paths["pdf"], media_type="application/pdf", filename=f"kalibratiemat_{spec.name}.pdf")

    return app


def lan_addresses() -> list[str]:
    ips = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 80))  # geen verkeer; alleen om het uitgaande adres te bepalen
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        ips.update(a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def serve(host: str, port: int, data_dir: Path, token: str | None) -> None:
    import uvicorn

    token = token or secrets.token_urlsafe(6)
    app = create_app(data_dir, token)
    print("Cam-to-CAD lokale server")
    print(f"  datamap: {data_dir}")
    for ip in lan_addresses() or ["<ip-adres-van-deze-pc>"]:
        print(f"  open op je telefoon (zelfde wifi): http://{ip}:{port}/?token={token}")
    print(f"  op deze pc: http://127.0.0.1:{port}/?token={token}")
    print("  Stoppen: Ctrl+C")
    uvicorn.run(app, host=host, port=port, log_level="warning")
