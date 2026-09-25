from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from camtocad.server.app import create_app  # noqa: E402


def fake_runner(photos: Path, out: Path, opts, log, scan_name=""):
    out.mkdir(parents=True, exist_ok=True)
    log(f"{len(list(photos.iterdir()))} foto's verwerkt")
    for name in ("model.step", "model.stl", "model.py", "report.json"):
        (out / name).write_text("x")
    (out / "report.html").write_text("<h1>rapport</h1>")
    return {"summary": {"gaten": 2}, "warnings": []}


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path, token="geheim", run_inline=True, runner=fake_runner)
    return TestClient(app)


def jpg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((20, 30), 128, np.uint8))
    return buf.tobytes()


def test_token_required(client):
    assert client.get("/api/scans").status_code == 401
    assert client.get("/?token=fout").status_code == 401


def test_upload_process_and_download(client):
    page = client.get("/?token=geheim")  # zet de cookie
    assert page.status_code == 200 and "Cam-to-CAD" in page.text
    files = [("fotos", (f"IMG_{i}.jpg", jpg(), "image/jpeg")) for i in range(3)]
    files.append(("fotos", ("../../etc/passwd", b"nee", "text/plain")))  # wordt genegeerd
    r = client.post("/api/scans", files=files, data={"mat": "A4"})
    assert r.status_code == 200 and r.json()["photos"] == 3
    job = r.json()["id"]
    status = client.get(f"/api/scans/{job}").json()
    assert status["state"] == "klaar" and status["summary"]["gaten"] == 2
    assert client.get(f"/scans/{job}/report.html").text == "<h1>rapport</h1>"
    assert client.get(f"/scans/{job}/status.json").status_code == 404
    assert client.get("/api/scans/..%2F..").status_code == 404


def test_mat_pdf(client):
    client.get("/?token=geheim")
    r = client.get("/mat/A4.pdf")
    assert r.status_code == 200 and r.content.startswith(b"%PDF")


def test_upload_without_token_is_refused_before_reading(tmp_path):
    app = create_app(tmp_path, token="geheim", run_inline=True, runner=fake_runner)
    c = TestClient(app)
    r = c.post("/api/scans", files=[("fotos", ("a.jpg", jpg(), "image/jpeg"))], data={"mat": "A4"})
    assert r.status_code == 401
    assert not [p for p in tmp_path.iterdir() if p.is_dir()]  # niets aangemaakt of weggeschreven


def test_too_large_upload_leaves_no_half_scan(tmp_path, monkeypatch):
    import camtocad.server.app as server

    monkeypatch.setattr(server, "MAX_BYTES", 100)
    c = TestClient(create_app(tmp_path, token="geheim", run_inline=True, runner=fake_runner))
    c.get("/?token=geheim")
    r = c.post("/api/scans", files=[("fotos", ("a.jpg", jpg() + b"x" * 200, "image/jpeg"))], data={"mat": "A4"})
    assert r.status_code == 413
    assert c.get("/api/scans").json() == []


def test_worker_survives_a_crashing_job(tmp_path):
    import threading
    import time

    from camtocad.server.app import JobStore

    store = JobStore(tmp_path, fake_runner)
    done = []

    def process(job_id):
        if job_id == "kapot":
            raise PermissionError("status.json is bezet")
        done.append(job_id)

    store.process = process
    threading.Thread(target=store.worker, daemon=True).start()
    store.queue.put("kapot")
    store.queue.put("goed")
    for _ in range(100):
        if done:
            break
        time.sleep(0.02)
    assert done == ["goed"]


def test_measured_ruler_is_passed_as_mat_scale(tmp_path):
    seen = {}

    def runner(photos, out, opts, log, scan_name=""):
        seen["scale"] = opts.mat_scale
        return fake_runner(photos, out, opts, log, scan_name)

    c = TestClient(create_app(tmp_path, token="geheim", run_inline=True, runner=runner))
    c.get("/?token=geheim")
    r = c.post("/api/scans", files=[("fotos", ("a.jpg", jpg(), "image/jpeg"))], data={"mat": "A4", "meetlijn": "99.5"})
    assert r.status_code == 200 and seen["scale"] == pytest.approx(0.995)
    bad = c.post("/api/scans", files=[("fotos", ("a.jpg", jpg(), "image/jpeg"))], data={"mat": "A4", "meetlijn": "90"})
    assert bad.status_code == 400
