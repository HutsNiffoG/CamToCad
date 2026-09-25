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


def test_measured_rulers_are_passed_as_mat_scale(tmp_path):
    seen = {}

    def runner(photos, out, opts, log, scan_name=""):
        seen["scale"] = opts.scale_xy
        return fake_runner(photos, out, opts, log, scan_name)

    c = TestClient(create_app(tmp_path, token="geheim", run_inline=True, runner=runner))
    c.get("/?token=geheim")
    r = c.post("/api/scans", files=[("fotos", ("a.jpg", jpg(), "image/jpeg"))], data={"mat": "A4", "meetlijn": "99.5"})
    assert r.status_code == 200 and seen["scale"] == pytest.approx((0.995, 0.995))
    r = c.post("/api/scans", files=[("fotos", ("a.jpg", jpg(), "image/jpeg"))],
               data={"meetlijn": "100.2", "meetlijn_y": "99.6"})
    assert r.status_code == 200 and seen["scale"] == pytest.approx((1.002, 0.996))
    bad = c.post("/api/scans", files=[("fotos", ("a.jpg", jpg(), "image/jpeg"))], data={"mat": "A4", "meetlijn": "90"})
    assert bad.status_code == 400
    assert c.post("/api/scans", data={"mat": "B5"}).status_code == 400


@pytest.fixture(scope="module")
def mat_photos():
    from camtocad import mat, render

    views = render.render_scan(None, mat.PRESETS["A4"], render.default_camera(), rings=((45.0, 5),), top_views=2,
                               seed=8)
    return [cv2.imencode(".jpg", v.image, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes() for v in views]


def test_photos_are_checked_one_by_one_and_processed_on_request(tmp_path, mat_photos):
    c = TestClient(create_app(tmp_path, token="geheim", run_inline=True, runner=fake_runner))
    c.get("/?token=geheim")
    job = c.post("/api/scans", data={"mat": "auto"}).json()["id"]
    r = c.post(f"/api/scans/{job}/fotos", files=[("fotos", ("IMG_1.jpg", mat_photos[0], "image/jpeg"))])
    assert r.status_code == 200
    first = r.json()["nieuw"][0]
    assert first["verdict"] == "goed" and first["mat"] == "A4" and first["blur_px"] < 1.5
    assert "points" not in first  # detectiegegevens blijven op de server
    blank = c.post(f"/api/scans/{job}/fotos", files=[("fotos", ("leeg.jpg", jpg(), "image/jpeg"))]).json()
    assert blank["nieuw"][0]["verdict"] == "onbruikbaar"
    assert any("niet gevonden" in a for a in blank["overzicht"]["advies"])
    assert c.post(f"/api/scans/{job}/start").status_code == 400  # nog geen 6 foto's

    removed = c.delete(f"/api/scans/{job}/fotos/{blank['nieuw'][0]['name']}").json()
    assert [p["verdict"] for p in removed["fotos"]] == ["goed"]
    for i, data in enumerate(mat_photos[1:], start=2):
        r = c.post(f"/api/scans/{job}/fotos", files=[("fotos", (f"IMG_{i}.jpg", data, "image/jpeg"))])
    overview = r.json()["overzicht"]
    assert overview["bruikbaar"] == len(mat_photos) and overview["mat"] == "A4"
    assert any("recht boven" in a or "rondom" in a for a in overview["advies"])  # 7 foto's is geen complete scan

    r = c.post(f"/api/scans/{job}/start", data={"meetlijn": "100.4", "meetlijn_y": "99.8"})
    assert r.status_code == 200
    status = c.get(f"/api/scans/{job}").json()
    assert status["state"] == "klaar" and status["photos"] == len(mat_photos) and status["meetlijn"] == [100.4, 99.8]
    # nog een foto toevoegen aan een verwerkte scan: terug naar 'foto's verzamelen'
    c.post(f"/api/scans/{job}/fotos", files=[("fotos", ("IMG_9.jpg", mat_photos[0], "image/jpeg"))])
    assert c.get(f"/api/scans/{job}").json()["state"] == "upload"


def test_debug_images_are_served_but_nothing_else(tmp_path):
    c = TestClient(create_app(tmp_path, token="geheim", run_inline=True, runner=fake_runner))
    c.get("/?token=geheim")
    job = c.post("/api/scans", files=[("fotos", ("a.jpg", jpg(), "image/jpeg"))]).json()["id"]
    dbg = tmp_path / job / "resultaat" / "debug"
    dbg.mkdir(parents=True)
    (dbg / "masker_foto_0000.jpg").write_bytes(jpg())
    (dbg / "geheim.txt").write_text("nee")
    assert c.get(f"/api/scans/{job}").json()["debug"] == ["masker_foto_0000.jpg"]
    assert c.get(f"/scans/{job}/debug/masker_foto_0000.jpg").status_code == 200
    assert c.get(f"/scans/{job}/debug/geheim.txt").status_code == 404
    assert c.get(f"/scans/{job}/debug/..%2Fstatus.json").status_code == 404
    assert c.delete(f"/api/scans/{job}").status_code == 200
    assert c.get("/api/scans").json() == []
