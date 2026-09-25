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
