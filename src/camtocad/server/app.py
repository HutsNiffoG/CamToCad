"""Lokale webserver (route A): uploaden vanaf de telefoon via wifi, verwerken op deze pc.

Geen cloud en geen app-installatie: open de getoonde URL in de browser van je telefoon, maak of
kies de foto's en upload ze. Elke foto wordt direct gecontroleerd (mat, scherpte, belichting,
kijkhoek) en de pagina laat zien welke foto's nog ontbreken; daarna start je de verwerking. Alle
data blijft in de datamap op deze pc. Toegang vereist een toegangscode (token), omdat de server
op het lokale netwerk luistert.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import re
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from .. import __version__, preflight, validate
from . import qr
from ..imgio import sniff_image
from ..mat import PRESETS, get_spec, write_mat
from ..pipeline import IMAGE_EXT, ScanOptions, run_scan

STATIC = Path(__file__).parent / "static"
OUTPUTS = {"model.step", "model.stl", "model.py", "report.html", "report.json"}
DEBUG_FILE = re.compile(r"^(masker_[\w.-]{1,80}\.jpg|lokalisatie\.png|bovenaanzicht\.png|diagnose\.json)$")
PHOTO_FILE = re.compile(r"^foto_\d{4}\.[a-z]{3,4}$")
MAX_FILES = 300
MAX_BYTES = 40 * 1024 * 1024
MAX_REQUEST = 2 * 1024 ** 3  # hele upload; losse bestanden blijven onder MAX_BYTES
MAX_LIVE_BYTES = 4 * 1024 * 1024  # één beeld van de livecamera (de pagina stuurt ~960 px, ~100 kB)
JOB_ID = re.compile(r"^[0-9a-f]{12}$")
UPLOAD_ID = re.compile(r"^[\w.-]{8,64}$")  # door de pagina gekozen id per foto: opnieuw sturen geeft geen dubbele
BUSY = ("wachtrij", "bezig")
MIN_FREE_BYTES = 1024 ** 3  # zoveel schijfruimte blijft altijd vrij (V24, v0.15)
USAGE_TTL = 60.0  # s: zo lang geldt een telling van de datamap


class StorageFull(Exception):
    """De datamap is vol (--max-gb), of de schijf bijna."""


def _mat_choice(mat: str) -> str:
    if mat.lower() == "auto":
        return "auto"
    if mat.upper() not in PRESETS:
        raise HTTPException(400, f"Onbekende mat: {mat}")
    return PRESETS[mat.upper()].name


def _rulers(meetlijn: float | None, meetlijn_y: float | None) -> list[float] | None:
    """Gemeten meetlijnen [X, Y], of None als ze niet zijn opgegeven (één waarde geldt voor beide)."""
    if meetlijn is None and meetlijn_y is None:
        return None
    x = meetlijn if meetlijn is not None else meetlijn_y
    y = meetlijn_y if meetlijn_y is not None else meetlijn
    for v in (x, y):
        if not 95.0 <= v <= 105.0:
            raise HTTPException(400, "Meetlijn buiten 95-105 mm: print de mat opnieuw op 100%")
    return [float(x), float(y)]


class JobStore:
    """Scans op schijf: <data>/<id>/fotos, <data>/<id>/resultaat, status.json en controle.json."""

    def __init__(self, root: Path, runner=run_scan, max_bytes: int | None = None):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes  # grens voor de hele datamap (--max-gb), of None
        self._usage: tuple[float, int] | None = None  # (tijd, bytes) van de laatste telling
        self._pending: dict[tuple[str, str], str] = {}  # (scan, upload-id) -> foto die nu binnenkomt
        self.queue: queue.Queue[str] = queue.Queue()
        self.runner = runner
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        # overzicht per scan, met de inhoud van controle.json en de matkeuze als sleutel: summarize kalibreert een
        # snelle camera (tot ~1 s), en de live begeleiding vraagt het overzicht bij elk beeld (V25)
        self._overviews: dict[str, tuple] = {}
        self._live: dict[str, dict] = {}  # per scan: brandpuntsafstand en mat van de vorige livebeelden

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

    def create(self, mat: str = "auto", meetlijn=None) -> str:
        job_id = uuid.uuid4().hex[:12]
        (self.root / job_id / "fotos").mkdir(parents=True)
        self.write(job_id, {"id": job_id, "state": "upload", "mat": mat,
                            "meetlijn": list(meetlijn) if meetlijn is not None else None,
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
        data["maten"] = (self.root / job_id / validate.REFERENCE).exists()
        return data

    # --- Fase 0: schuifmaatmetingen per scan (maten.json, zoals bij camtocad valideer) ----------------------

    def reference(self, job_id: str) -> dict | None:
        f = self.path(job_id) / validate.REFERENCE
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def set_reference(self, job_id: str, naam: str, maten: dict) -> None:
        """Schrijft maten.json met de meetlijnen en de mat van de scan; een fout in de maten geeft een 400."""
        status = self.read(job_id)
        data = {"naam": (str(naam).strip() or job_id)[:100], "maten": maten}
        if status.get("meetlijn"):
            data["meetlijn"] = status["meetlijn"]
        if status.get("mat", "auto") != "auto":
            data["mat"] = status["mat"]
        with self.lock(job_id):
            self.write(job_id, data, validate.REFERENCE + ".nieuw")
            new = self.path(job_id) / (validate.REFERENCE + ".nieuw")
            try:
                validate.read_reference(new)
            except (ValueError, TypeError) as e:
                new.unlink(missing_ok=True)
                raise HTTPException(400, str(e).replace(str(new) + ": ", "")) from None
            new.replace(self.path(job_id) / validate.REFERENCE)

    def sync_reference(self, job_id: str) -> None:
        """De meetlijnen en de mat in maten.json gelijk houden met die van de scan (ze zijn bij /start te wijzigen)."""
        ref = self.reference(job_id)
        if ref is not None:
            self.set_reference(job_id, ref.get("naam", ""), ref.get("maten", {}))

    def validation(self, job_id: str) -> validate.ScanValidation | None:
        """De vergelijking van het model met maten.json, als er een actueel resultaat is; er wordt niets verwerkt."""
        if self.reference(job_id) is None:
            return None
        res = validate.compare_existing(self.path(job_id))
        status = self.read(job_id)
        if status.get("state") in BUSY:
            res.status, res.fout, res.vergelijkingen = "bezig", "", []
        elif status.get("state") == "fout" and res.status == "niet verwerkt":
            res.status, res.fout = "fout", status.get("error") or "verwerking mislukt"
        return res

    def measured(self) -> list[tuple[str, validate.ScanValidation]]:
        """Alle scans met schuifmaatmetingen, met hun vergelijking."""
        return [(s["id"], self.validation(s["id"])) for s in self.list()
                if (self.root / s["id"] / validate.REFERENCE).exists()]

    # --- foto's en controle ---------------------------------------------------------------

    # --- opslag (V24, v0.15) -----------------------------------------------------------------

    def usage(self, refresh: bool = False) -> int:
        """Bytes in de datamap; een telling geldt USAGE_TTL seconden (bij elke upload tellen is te traag)."""
        now = time.monotonic()
        if refresh or self._usage is None or now - self._usage[0] > USAGE_TTL:
            total = 0
            for dirpath, _, files in os.walk(self.root):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(dirpath, f))
                    except OSError:
                        pass
            self._usage = (now, total)
        return self._usage[1]

    def storage(self) -> dict:
        """Gebruik van de datamap en vrije ruimte op de schijf, voor /api/info."""
        free = shutil.disk_usage(self.root).free
        return {"gebruikt_gb": round(self.usage() / 1024 ** 3, 2), "vrij_gb": round(free / 1024 ** 3, 1),
                "max_gb": None if self.max_bytes is None else round(self.max_bytes / 1024 ** 3, 1)}

    def ensure_space(self, incoming: int = MAX_BYTES) -> None:
        """StorageFull als er geen `incoming` bytes meer bij kunnen: de grens van de datamap, of de schijf bijna vol."""
        if self.max_bytes is not None and self.usage() + incoming > self.max_bytes:
            raise StorageFull(f"De datamap is vol ({self.usage() / 1024 ** 3:.1f} van {self.max_bytes / 1024 ** 3:.1f} "
                              "GB): verwijder oude scans, of start de server met een grotere --max-gb")
        if shutil.disk_usage(self.root).free - incoming < MIN_FREE_BYTES:
            raise StorageFull("De schijf is bijna vol: maak ruimte vrij, of verwijder oude scans")

    def _added(self, n: int) -> None:
        if self._usage is not None:
            self._usage = (self._usage[0], self._usage[1] + n)

    def cleanup(self, days: float, dry_run: bool = False) -> list[tuple[str, int]]:
        """Haalt de foto's weg van scans die al `days` dagen klaar (of mislukt) zijn; het model, het rapport, de
        controle van de foto's en de schuifmaatmetingen blijven. Geeft per scan (id, vrijgekomen bytes)."""
        cutoff = time.time() - days * 86400
        out = []
        for st in self.list():
            if st.get("state") not in ("klaar", "fout") or (st.get("finished") or st.get("created") or 0) > cutoff:
                continue
            folder = self.root / st["id"] / "fotos"
            photos = [f for f in folder.iterdir() if f.is_file()] if folder.is_dir() else []
            size = sum(f.stat().st_size for f in photos)
            if not photos:
                continue
            out.append((st["id"], size))
            if dry_run:
                continue
            with self.lock(st["id"]):
                for f in photos:
                    f.unlink(missing_ok=True)
                self.update(st["id"], photos=0, fotos_opgeruimd=time.time())
            self._added(-size)
            with self._guard:
                self._overviews.pop(st["id"], None)
        return out

    def save_photo(self, job_id: str, filename: str, stream, upload_id: str | None = None) -> tuple[Path | None, bool]:
        """Slaat één upload op als foto_NNNN.ext, met de extensie van wat het werkelijk is (magic bytes). Geeft
        (pad, nieuw): pad None als het geen foto is; nieuw False als deze `upload_id` al eerder binnenkwam (de pagina
        stuurt een foto na een netwerkfout opnieuw, met hetzelfde id: dan geen tweede kopie)."""
        head = stream.read(1 << 20)
        ext = sniff_image(head[:32])
        if ext is None:
            return None, False
        if ext == ".jpg" and Path(filename or "").suffix.lower() in (".jpg", ".jpeg"):
            ext = Path(filename).suffix.lower()
        uid = upload_id if upload_id and UPLOAD_ID.match(upload_id) else None
        folder = self.path(job_id) / "fotos"
        with self.lock(job_id):
            if uid is not None:
                known = self.read(job_id).get("uploads", {}).get(uid)
                if known and (folder / known).exists():
                    return folder / known, False
                if (job_id, uid) in self._pending:
                    raise HTTPException(409, "Deze foto komt al binnen: probeer het zo nog eens")
            self.ensure_space()
            used = {int(p.stem[5:]) for p in folder.iterdir() if PHOTO_FILE.match(p.name)}
            used |= {int(n[5:9]) for (j, _), n in self._pending.items() if j == job_id}
            target = folder / f"foto_{max(used, default=-1) + 1:04d}{ext}"
            target.touch(exist_ok=False)  # de naam is nu van deze upload
            if uid is not None:
                self._pending[(job_id, uid)] = target.name
        size = 0
        try:
            with open(target, "wb") as out:
                chunk = head
                while chunk:
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, f"{filename}: bestand te groot")
                    out.write(chunk)
                    chunk = stream.read(1 << 20)
        except BaseException:
            target.unlink(missing_ok=True)
            with self.lock(job_id):
                self._pending.pop((job_id, uid), None)
            raise
        self._added(size)
        if uid is not None:
            with self.lock(job_id):
                status = self.read(job_id)
                self.update(job_id, uploads={**status.get("uploads", {}), uid: target.name})
                self._pending.pop((job_id, uid), None)
        return target, True

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
        return self._overview(job_id)[0]

    def _overview(self, job_id: str) -> tuple[dict, object]:
        """(overzicht, mat van de scan); hergebruikt zolang controle.json en de matkeuze niet veranderen."""
        base = self.path(job_id)
        f = base / "controle.json"
        raw = f.read_bytes() if f.exists() else b""
        key = (hashlib.sha1(raw).hexdigest(), self.read(job_id).get("mat"))
        with self._guard:
            hit = self._overviews.get(job_id)
        if hit is not None and hit[0] == key:
            return hit[1], hit[2]
        checks = self.checks(job_id)
        present = [c for n, c in sorted(checks.items()) if (base / "fotos" / n).exists()]
        spec = self._scan_spec(job_id, checks)
        ov = {"fotos": [c.public() for c in present], "overzicht": preflight.summarize(present, spec)}
        with self._guard:
            self._overviews[job_id] = (key, ov, spec)
        return ov, spec

    def live(self, job_id: str, img: np.ndarray) -> dict:
        """Beoordeelt één beeld van de livecamera tegen de foto's van de scan tot nu toe (V25)."""
        ov, spec = self._overview(job_id)
        with self._guard:
            state = self._live.setdefault(job_id, {"f": [], "mat": None})
            f_rel = float(np.median(state["f"])) if state["f"] else None
            hint = state["mat"]
        t0 = time.perf_counter()
        res = preflight.live_check(img, spec, ov["overzicht"], f_rel, hint)
        res["ms"] = round(1000 * (time.perf_counter() - t0))
        with self._guard:
            if res["mat"]:
                state["mat"] = get_spec(res["mat"])
            if res["f_rel"]:
                state["f"] = (state["f"] + [res["f_rel"]])[-25:]
        res["fotos"] = len(ov["fotos"])
        return res

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
            rulers = status.get("meetlijn")  # None: niet gemeten
            rulers = [rulers, rulers] if isinstance(rulers, (int, float)) else rulers
            scale = (rulers[0] / 100.0, rulers[1] / 100.0) if rulers else None
            opts = ScanOptions(mat=status.get("mat", "auto"), mat_scale=scale)
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
            self._overviews.pop(job_id, None)
            self._live.pop(job_id, None)

    def worker(self) -> None:
        while True:
            job_id = self.queue.get()
            try:
                self.process(job_id)
            except Exception:  # noqa: BLE001 - de enige worker mag nooit stoppen
                traceback.print_exc(file=sys.stderr)


def create_app(data_dir: Path, token: str | None, run_inline: bool = False, runner=run_scan,
               max_gb: float | None = None) -> FastAPI:
    app = FastAPI(title="Cam-to-CAD (lokaal)", docs_url=None, redoc_url=None)
    app.state.https_port = None  # zet serve() als de https-server draait
    store = JobStore(Path(data_dir), runner, None if max_gb is None else int(max_gb * 1024 ** 3))
    app.state.store = store

    @app.exception_handler(StorageFull)
    async def storage_full(request: Request, exc: StorageFull):
        return JSONResponse({"detail": str(exc)}, status_code=507)
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

    @app.get("/api/info")
    def info(request: Request):
        """Versie, en de https-poort: de camera van de telefoon werkt alleen op een beveiligde pagina (V24). De
        toegangscode staat erbij voor de link naar https (de cookie gaat niet mee van http naar https)."""
        check(request)
        return {"versie": __version__, "https_poort": app.state.https_port, "token": token, "opslag": store.storage()}

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
                    meetlijn: float | None = Form(None), meetlijn_y: float | None = Form(None)):
        """Nieuwe scan. Met foto's erbij wordt hij meteen verwerkt (alles in één keer); zonder foto's
        volgen die één voor één via /api/scans/{id}/fotos, met directe controle, en daarna /start."""
        check(request)
        mat, rulers = _mat_choice(mat), _rulers(meetlijn, meetlijn_y)
        if fotos is not None and len(fotos) > MAX_FILES:
            raise HTTPException(400, f"Upload hooguit {MAX_FILES} foto's")
        store.ensure_space()
        job_id = store.create(mat, rulers)
        if not fotos:
            return {"id": job_id, "photos": 0}
        try:
            saved = sum(store.save_photo(job_id, f.filename, f.file)[0] is not None for f in fotos)
            if saved == 0:
                raise HTTPException(400, "Geen bruikbare foto's (JPG, PNG of HEIC) ontvangen")
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
    def add_photos(job_id: str, request: Request, fotos: list[UploadFile] = File(...), uid: str | None = Form(None)):
        """Foto's toevoegen (ook aan een scan die al verwerkt is); elke foto wordt direct gecontroleerd. `uid`: een id
        dat de pagina per foto kiest (bij één foto per verzoek). Stuurt ze dezelfde foto na een netwerkfout opnieuw,
        dan komt er geen tweede kopie bij (V24, v0.15)."""
        check(request)
        status = store.read(job_id)
        if status["state"] in BUSY:
            raise HTTPException(409, "Deze scan wordt verwerkt: wacht tot hij klaar is")
        if status.get("photos", 0) + len(fotos) > MAX_FILES:
            raise HTTPException(400, f"Hooguit {MAX_FILES} foto's per scan")
        results = []
        for f in fotos:
            path, new = store.save_photo(job_id, f.filename, f.file, uid if len(fotos) == 1 else None)
            if path is None:
                results.append({"name": f.filename, "verdict": "onbruikbaar", "notes": ["geen foto (JPG, PNG of HEIC)"]})
                continue
            known = store.checks(job_id).get(path.name) if not new else None
            chk = known if known is not None else store.check_photo(job_id, path)
            results.append({**chk.public(), "upload": f.filename, **({} if new else {"al_ontvangen": True})})
        if status["state"] != "upload":
            store.update(job_id, state="upload")  # opnieuw verwerken met de extra foto's
        return {"nieuw": results, **store.overview(job_id)}

    @app.get("/api/scans/{job_id}/controle")
    def get_checks(job_id: str, request: Request):
        check(request)
        return store.overview(job_id)

    @app.post("/api/scans/{job_id}/live")
    def live_frame(job_id: str, request: Request, beeld: UploadFile = File(...)):
        """Eén beeld van de livecamera (V25): waar staat de camera ten opzichte van het onderdeel, welke richting
        ontbreekt nog, en is dit een goed moment voor een foto? Het beeld wordt niet bewaard."""
        check(request)
        data = beeld.file.read(MAX_LIVE_BYTES + 1)
        if len(data) > MAX_LIVE_BYTES:
            raise HTTPException(413, "Livebeeld te groot")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE) if data else None
        if img is None:
            raise HTTPException(400, "Geen leesbaar beeld")
        return store.live(job_id, img)

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
        if meetlijn is not None or meetlijn_y is not None:
            changes["meetlijn"] = _rulers(meetlijn, meetlijn_y)
        if changes:
            if store.read(job_id)["state"] in BUSY:
                raise HTTPException(409, "Deze scan wordt al verwerkt")
            store.update(job_id, **changes)
            store.sync_reference(job_id)
        store.start(job_id, run_inline)
        return store.public(job_id)

    # --- Fase 0 (v0.13): schuifmaatmetingen invoeren, vergelijken en delen ------------------------------------

    def public_validation(job_id: str, res: validate.ScanValidation | None) -> dict | None:
        return None if res is None else {**{k: v for k, v in res.to_dict().items() if k != "map"}, "id": job_id}

    @app.get("/api/scans/{job_id}/maten")
    def get_reference(job_id: str, request: Request):
        """maten.json van de scan en, als hij verwerkt is, de vergelijking met het model."""
        check(request)
        return {"referentie": store.reference(job_id),
                "validatie": public_validation(job_id, store.validation(job_id))}

    @app.put("/api/scans/{job_id}/maten")
    def put_reference(job_id: str, request: Request, body: dict = Body(...)):
        """Schuifmaatmetingen opslaan: {"naam": ..., "maten": {"lengte": 80.02, "gaten": [6.62, 6.6], ...}}."""
        check(request)
        if not isinstance(body.get("maten"), dict):
            raise HTTPException(400, "'maten' ontbreekt")
        store.set_reference(job_id, str(body.get("naam") or ""), body["maten"])
        return {"referentie": store.reference(job_id),
                "validatie": public_validation(job_id, store.validation(job_id))}

    @app.delete("/api/scans/{job_id}/maten")
    def delete_reference(job_id: str, request: Request):
        check(request)
        (store.path(job_id) / validate.REFERENCE).unlink(missing_ok=True)
        return {"referentie": None, "validatie": None}

    def meetset() -> tuple[list[validate.ScanValidation], list[dict], dict]:
        """De scans met metingen: hun vergelijking, die voor de pagina, en de samenvatting van de verwerkte."""
        measured = store.measured()
        return ([r for _, r in measured], [public_validation(i, r) for i, r in measured],
                validate.summarize([r for _, r in measured if r.status == "ok"]))

    @app.get("/api/meetset")
    def get_meetset(request: Request):
        """Fase 0 over alle scans met schuifmaatmetingen: per soort maat bias, spreiding, aandeel binnen U95 en
        hoeveel U95 tekortkomt."""
        check(request)
        _, rows, summary = meetset()
        return {"scans": rows, "samenvatting": summary}

    @app.get("/api/meetset.zip")
    def meetset_zip(request: Request):
        """Alles om te delen, zonder foto's: per scan maten.json, report.json, diagnose.json en de status, plus
        validatie.json en validatie.html van de hele set (FASE-0.md, 'Resultaten delen')."""
        check(request)
        results, rows, summary = meetset()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("validatie.json", json.dumps({"versie": __version__, "samenvatting": summary, "scans": rows},
                                                    indent=2, ensure_ascii=False))
            with tempfile.TemporaryDirectory() as tmp:  # dezelfde pagina als camtocad valideer
                z.write(validate.write_report(results, summary, tmp)["html"], "validatie.html")
            for r in rows:
                base = store.path(r["id"])
                status = {k: v for k, v in store.read(r["id"]).items()
                          if k in ("id", "state", "mat", "meetlijn", "photos", "created", "finished", "summary",
                                   "warnings", "error")}
                z.writestr(f"{r['id']}/status.json", json.dumps(status, indent=2, ensure_ascii=False))
                for rel in (validate.REFERENCE, "resultaat/report.json", "resultaat/debug/diagnose.json"):
                    if (base / rel).exists():
                        z.write(base / rel, f"{r['id']}/{rel}")
            z.writestr("LEESMIJ.txt", "Cam-to-CAD Fase 0: schuifmaatmetingen en scanresultaten, zonder foto's.\n"
                                      "validatie.json: de vergelijking per maat en de samenvatting per soort.\n"
                                      "Per scan: maten.json (de metingen), resultaat/report.json (het model en de "
                                      "onzekerheid) en resultaat/debug/diagnose.json.\n")
        stamp = time.strftime("%Y%m%d")
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="camtocad_meetset_{stamp}.zip"'})

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
        pdf = Path(data_dir) / "_mat" / __version__ / f"kalibratiemat_{spec.name}.pdf"
        if not pdf.exists():  # één keer per versie; atomisch geschreven, dus een tweede verzoek ziet geen half bestand
            pdf = write_mat(spec, pdf.parent)["pdf"]
        return FileResponse(pdf, media_type="application/pdf", filename=f"kalibratiemat_{spec.name}.pdf")

    return app


def lan_addresses() -> list[str]:
    """IPv4-adressen van deze pc op het netwerk; het adres waarlangs verkeer naar buiten gaat eerst."""
    primary, ips = None, set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 80))  # geen verkeer; alleen om het uitgaande adres te bepalen
            primary = s.getsockname()[0]
    except OSError:
        pass
    try:
        ips.update(a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    rest = sorted(ip for ip in ips if not ip.startswith("127.") and ip != primary)
    return ([primary] if primary and not primary.startswith("127.") else []) + rest


def cert_hosts(ips: list[str]) -> list[str]:
    """Adressen en namen waarvoor het https-certificaat geldt: de LAN-adressen, localhost en de naam van de pc."""
    name = socket.gethostname()
    return [*ips, "127.0.0.1", "localhost"] + ([name, f"{name}.local"] if name and "." not in name else
                                              [name] if name else [])


CLEANUP_EVERY_S = 6 * 3600  # zo vaak ruimt de server oude foto's op (--bewaar-fotos)


def _cleanup_loop(store: JobStore, days: float) -> None:
    while True:
        try:
            done = store.cleanup(days)
            if done:
                print(f"  opgeruimd: de foto's van {len(done)} scan(s) ouder dan {days:g} dagen "
                      f"({sum(b for _, b in done) / 1024 ** 3:.2f} GB)")
        except Exception:  # noqa: BLE001 - opruimen mag de server nooit stoppen
            traceback.print_exc(file=sys.stderr)
        time.sleep(CLEANUP_EVERY_S)


def serve(host: str, port: int, data_dir: Path, token: str | None, https_port: int | None = 8443,
          max_gb: float | None = None, keep_days: float | None = None) -> None:
    """http op `port` en (V24, v0.13) https op `https_port` met een eigen certificaat: de camera op de telefoon (live
    begeleiding) werkt alleen via https. `https_port` None: alleen http. `max_gb`: grens voor de datamap;
    `keep_days`: de foto's van scans die zo lang klaar zijn, worden opgeruimd (v0.15)."""
    import uvicorn

    token = token or secrets.token_urlsafe(6)
    app = create_app(data_dir, token, max_gb=max_gb)
    if keep_days is not None:
        threading.Thread(target=_cleanup_loop, args=(app.state.store, keep_days), daemon=True).start()
    local = host in ("127.0.0.1", "localhost")
    ips = lan_addresses() if not local else []
    cert = None
    if https_port:
        try:
            from . import tls
            cert, key = tls.ensure_certificate(Path(data_dir) / "_tls", cert_hosts(ips))
        except ImportError:
            print("  (https niet beschikbaar: pip install cryptography, of installeer camtocad met de extra 'server')")
    if cert is not None:
        secure = uvicorn.Server(uvicorn.Config(app, host=host, port=https_port, ssl_certfile=str(cert),
                                               ssl_keyfile=str(key), log_level="warning"))
        thread = threading.Thread(target=secure.run, daemon=True)
        thread.start()
        deadline = time.time() + 10.0
        while not secure.started and thread.is_alive() and time.time() < deadline:
            time.sleep(0.05)
        if secure.started:
            app.state.https_port = https_port
        else:  # poort bezet of certificaat onleesbaar: uvicorn heeft de fout al gemeld
            print(f"  (https op poort {https_port} start niet: kies een andere poort met --https-poort)")
            cert = None
    print("Cam-to-CAD lokale server")
    st = app.state.store.storage()
    print(f"  datamap: {data_dir} ({st['gebruikt_gb']:.1f} GB in gebruik, {st['vrij_gb']:.0f} GB vrij"
          + (f", grens {st['max_gb']:g} GB" if st["max_gb"] is not None else "") + ")")
    if local:
        print("  alleen bereikbaar op deze pc. Foto's uploaden met je telefoon: start de server met --lan")
    shown = ips or (["<ip-adres-van-deze-pc>"] if not local else [])
    for ip in shown:
        if cert is not None:
            print(f"  open op je telefoon (zelfde wifi): https://{ip}:{https_port}/?token={token}")
        print(f"  {'zonder camera en zonder certificaatmelding' if cert else 'open op je telefoon (zelfde wifi)'}: "
              f"http://{ip}:{port}/?token={token}")
    print(f"  op deze pc: http://127.0.0.1:{port}/?token={token}")
    if cert is not None:
        print("  De eerste keer waarschuwt de telefoon voor het certificaat (het is van deze pc zelf): kies "
              "'Geavanceerd' en 'doorgaan' (Chrome) of 'Toon details' en 'bezoek deze website' (Safari).")
        print(f"  Vingerafdruk van het certificaat (SHA-256): {tls.fingerprint(cert)}")
    if ips:
        url = f"https://{ips[0]}:{https_port}/?token={token}" if cert else f"http://{ips[0]}:{port}/?token={token}"
        if qr.print_qr(url):
            print(f"  scan de QR-code met de camera van je telefoon (zelfde wifi): {ips[0]}")
        elif not qr.available():
            print("  (met pip install segno staat hier een QR-code om te scannen met je telefoon)")
    print("  Stoppen: Ctrl+C")
    uvicorn.run(app, host=host, port=port, log_level="warning")
