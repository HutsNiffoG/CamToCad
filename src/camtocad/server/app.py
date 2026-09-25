"""Lokale webserver (route A): uploaden vanaf de telefoon via wifi, verwerken op deze pc.

Geen cloud en geen app-installatie: open de getoonde URL in de browser van je telefoon, maak of
kies de foto's en upload ze. Alle data blijft in de datamap op deze pc. Toegang vereist een
toegangscode (token), omdat de server op het lokale netwerk luistert.
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
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..mat import PRESETS, write_mat
from ..pipeline import IMAGE_EXT, ScanOptions, run_scan

STATIC = Path(__file__).parent / "static"
OUTPUTS = {"model.step", "model.stl", "model.py", "report.html", "report.json"}
MAX_FILES = 300
MAX_BYTES = 40 * 1024 * 1024
MAX_REQUEST = 2 * 1024 ** 3  # hele upload; losse bestanden blijven onder MAX_BYTES
JOB_ID = re.compile(r"^[0-9a-f]{12}$")


class JobStore:
    """Scans op schijf: <data>/<id>/fotos, <data>/<id>/resultaat en status.json."""

    def __init__(self, root: Path, runner=run_scan):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue: queue.Queue[str] = queue.Queue()
        self.runner = runner

    def path(self, job_id: str) -> Path:
        if not JOB_ID.match(job_id):
            raise HTTPException(404, "Onbekende scan")
        p = self.root / job_id
        if not p.is_dir():
            raise HTTPException(404, "Onbekende scan")
        return p

    def create(self, mat: str, meetlijn: float = 100.0) -> str:
        job_id = uuid.uuid4().hex[:12]
        (self.root / job_id / "fotos").mkdir(parents=True)
        self.write(job_id, {"id": job_id, "state": "upload", "mat": mat, "meetlijn": meetlijn, "created": time.time(),
                            "log": []})
        return job_id

    def read(self, job_id: str) -> dict:
        return json.loads((self.path(job_id) / "status.json").read_text(encoding="utf-8"))

    def write(self, job_id: str, data: dict) -> None:
        target = self.root / job_id / "status.json"
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

    def process(self, job_id: str) -> None:
        base = self.path(job_id)
        status = self.update(job_id, state="bezig", started=time.time())
        lines: list[str] = []

        def log(msg: str) -> None:
            lines.append(msg)
            self.update(job_id, log=lines[-50:])

        try:
            opts = ScanOptions(mat=status["mat"], mat_scale=float(status.get("meetlijn", 100.0)) / 100.0)
            result = self.runner(base / "fotos", base / "resultaat", opts, log=log, scan_name=job_id)
            self.update(job_id, state="klaar", finished=time.time(), summary=result.get("summary", {}),
                        warnings=result.get("warnings", [])[:20])
        except Exception as e:  # noqa: BLE001 - de fout moet in de UI zichtbaar worden
            self.update(job_id, state="fout", finished=time.time(), error=str(e) or e.__class__.__name__,
                        trace=traceback.format_exc()[-2000:])

    def remove(self, job_id: str) -> None:
        shutil.rmtree(self.root / job_id, ignore_errors=True)

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
        return [{k: v for k, v in s.items() if k != "trace"} for s in store.list()]

    @app.get("/api/scans/{job_id}")
    def get_scan(job_id: str, request: Request):
        check(request)
        return {k: v for k, v in store.read(job_id).items() if k != "trace"}

    @app.post("/api/scans")
    async def upload(request: Request, fotos: list[UploadFile] = File(...), mat: str = Form("A4"),
                     meetlijn: float = Form(100.0)):
        check(request)
        if mat not in PRESETS:
            raise HTTPException(400, f"Onbekende mat: {mat}")
        if not 95.0 <= meetlijn <= 105.0:
            raise HTTPException(400, "Meetlijn buiten 95-105 mm: print de mat opnieuw op 100%")
        if not fotos or len(fotos) > MAX_FILES:
            raise HTTPException(400, f"Upload tussen 1 en {MAX_FILES} foto's")
        job_id = store.create(mat, meetlijn)
        target = store.path(job_id) / "fotos"
        saved = 0
        try:
            for i, f in enumerate(fotos):
                ext = Path(f.filename or "").suffix.lower()
                if ext not in IMAGE_EXT:
                    continue
                size = 0
                with open(target / f"foto_{i:04d}{ext}", "wb") as out:
                    while chunk := await f.read(1 << 20):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise HTTPException(413, f"{f.filename}: bestand te groot")
                        out.write(chunk)
                saved += 1
            if saved == 0:
                raise HTTPException(400, "Geen bruikbare foto's (JPG/PNG) ontvangen")
        except BaseException:
            store.remove(job_id)  # geen halve scans laten staan
            raise
        store.update(job_id, state="wachtrij", photos=saved)
        if run_inline:
            store.process(job_id)
        else:
            store.queue.put(job_id)
        return {"id": job_id, "photos": saved}

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
        paths = write_mat(formaat.upper(), Path(data_dir) / "_mat")
        return FileResponse(paths["pdf"], media_type="application/pdf",
                            filename=f"kalibratiemat_{formaat.upper()}.pdf")

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
