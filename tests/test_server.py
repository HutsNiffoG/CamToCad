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
        seen["scale"], seen["measured"] = opts.scale_xy, opts.scale_measured
        return fake_runner(photos, out, opts, log, scan_name)

    c = TestClient(create_app(tmp_path, token="geheim", run_inline=True, runner=runner))
    c.get("/?token=geheim")
    r = c.post("/api/scans", files=[("fotos", ("a.jpg", jpg(), "image/jpeg"))], data={"mat": "A4", "meetlijn": "99.5"})
    assert r.status_code == 200 and seen["scale"] == pytest.approx((0.995, 0.995)) and seen["measured"]
    # niet opgegeven is iets anders dan precies 100,0 gemeten: dat bepaalt de printschaalterm in U95
    r = c.post("/api/scans", files=[("fotos", ("a.jpg", jpg(), "image/jpeg"))], data={"mat": "A4"})
    assert r.status_code == 200 and seen["scale"] == (1.0, 1.0) and not seen["measured"]
    r = c.post("/api/scans", files=[("fotos", ("a.jpg", jpg(), "image/jpeg"))], data={"meetlijn": "100.0"})
    assert r.status_code == 200 and seen["scale"] == (1.0, 1.0) and seen["measured"]
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


def test_https_certificate_covers_the_lan_addresses_and_is_reused(tmp_path):
    """V24 (v0.13): een eigen certificaat voor de adressen van de pc; hergebruikt zolang het geldig is en alle
    adressen dekt, anders een nieuw."""
    import datetime as dt
    import os

    from cryptography import x509

    from camtocad.server import tls

    hosts = ["192.168.1.20", "127.0.0.1", "localhost", "werkbank.local"]
    cert, key = tls.ensure_certificate(tmp_path / "_tls", hosts)
    c = x509.load_pem_x509_certificate(cert.read_bytes())
    san = c.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert {str(v) for v in san.get_values_for_type(x509.IPAddress)} == {"192.168.1.20", "127.0.0.1"}
    assert set(san.get_values_for_type(x509.DNSName)) == {"localhost", "werkbank.local"}
    assert (c.not_valid_after_utc - c.not_valid_before_utc).days <= 398
    if os.name == "posix":
        assert key.stat().st_mode & 0o077 == 0  # de sleutel is alleen voor de eigenaar leesbaar
    first = tls.fingerprint(cert)
    assert tls.ensure_certificate(tmp_path / "_tls", hosts[:2]) == (cert, key) and tls.fingerprint(cert) == first
    tls.ensure_certificate(tmp_path / "_tls", hosts + ["10.0.0.5"])  # een nieuw adres: een nieuw certificaat
    second = tls.fingerprint(cert)
    assert second != first
    soon = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=380)  # bijna verlopen: vernieuwen
    tls.ensure_certificate(tmp_path / "_tls", hosts, now=soon)
    assert tls.fingerprint(cert) != second


def test_live_frames_are_judged_against_the_photos_so_far(tmp_path, mat_photos, monkeypatch):
    """V25 (v0.13): een beeld van de livecamera krijgt een oordeel en één aanwijzing, en wordt niet bewaard. Het
    overzicht van de foto's wordt hergebruikt zolang er niets verandert: summarize kalibreert, te traag per beeld."""
    from camtocad import mat, preflight, render

    calls = []
    summarize = preflight.summarize
    monkeypatch.setattr(preflight, "summarize", lambda *a, **k: calls.append(1) or summarize(*a, **k))
    c = TestClient(create_app(tmp_path, token="geheim", run_inline=True, runner=fake_runner))
    assert c.get("/api/info").status_code == 401
    c.get("/?token=geheim")
    info = c.get("/api/info").json()
    assert info["https_poort"] is None and info["versie"] and info["token"] == "geheim"
    job = c.post("/api/scans", data={"mat": "auto"}).json()["id"]
    cam = render.default_camera(960, 540, 700.0, dist=(0, 0, 0, 0, 0))
    R, t = render.look_at([150.0, 80.0, 330.0], [150.0, 80.0, 0.0])
    img, _ = render.render_view(mat.rasterize_board(mat.PRESETS["A4"], 4.0, 3.0), cam, R, t,
                                rng=np.random.default_rng(1))
    frame = [("beeld", ("live.jpg", cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(),
                        "image/jpeg"))]
    res = c.post(f"/api/scans/{job}/live", files=frame).json()
    assert res["mat"] == "A4" and res["vak"] == "boven" and res["opnemen"] and res["nodig"]["boven"] == 5
    assert res["fotos"] == 0 and len(res["omtrek"]) == 4 and res["ms"] < 5000
    assert not list((tmp_path / job / "fotos").iterdir())  # een livebeeld wordt niet bewaard
    for i, data in enumerate(mat_photos):
        c.post(f"/api/scans/{job}/fotos", files=[("fotos", (f"IMG_{i}.jpg", data, "image/jpeg"))])
    n = len(calls)
    for _ in range(3):
        res = c.post(f"/api/scans/{job}/live", files=frame).json()
    overview = c.get(f"/api/scans/{job}/controle").json()["overzicht"]
    assert len(calls) == n  # hergebruikt
    assert res["fotos"] == len(mat_photos) and res["nodig"]["boven"] == 5 - overview["recht_van_boven"]
    c.delete(f"/api/scans/{job}/fotos/foto_0000.jpg")
    assert len(calls) == n + 1  # een foto weg: opnieuw
    bad = [("beeld", ("x.jpg", b"geen beeld", "image/jpeg"))]
    assert c.post(f"/api/scans/{job}/live", files=bad).status_code == 400
    assert c.post("/api/scans/000000000000/live", files=frame).status_code == 404


def test_measurements_are_compared_and_shared_without_photos(tmp_path):
    """Fase 0 in de webpagina (v0.13): schuifmaatmetingen per scan (maten.json, zoals bij camtocad valideer), de
    vergelijking zodra de scan verwerkt is (zonder opnieuw te verwerken), de meetset over alle scans en een zip om
    te delen, zonder foto's."""
    import io
    import json
    import zipfile

    from test_validate import UNC, bracket

    def runner(photos, out, opts, log, scan_name=""):
        (out / "debug").mkdir(parents=True, exist_ok=True)
        report = {"summary": {"betrouwbaarheid": "normaal"}, "uncertainty_model": UNC,
                  "geometry": {"gefit": bracket().to_dict(), "gesnapt": bracket().to_dict()}}
        (out / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (out / "debug" / "diagnose.json").write_text("{}", encoding="utf-8")
        return {"summary": {}, "warnings": []}

    c = TestClient(create_app(tmp_path, token="geheim", run_inline=True, runner=runner))
    c.get("/?token=geheim")
    files = [("fotos", (f"IMG_{i}.jpg", jpg(), "image/jpeg")) for i in range(6)]
    job = c.post("/api/scans", files=files, data={"mat": "A4", "meetlijn": "100.1"}).json()["id"]
    assert c.get(f"/api/scans/{job}/maten").json() == {"referentie": None, "validatie": None}
    assert c.get("/api/meetset").json()["scans"] == []

    bad = c.put(f"/api/scans/{job}/maten", json={"maten": {"dikte": 3}})
    assert bad.status_code == 400 and "dikte" in bad.json()["detail"] and str(tmp_path) not in bad.json()["detail"]
    assert c.put(f"/api/scans/{job}/maten", json={"maten": {"lengte": -1}}).status_code == 400
    assert not (tmp_path / job / "maten.json").exists()
    r = c.put(f"/api/scans/{job}/maten", json={"naam": "beugel", "maten": {"lengte": 80.1, "breedte": 40.0,
                                                                          "gaten": [6.6, 6.62]}}).json()
    assert r["referentie"]["meetlijn"] == [100.1, 100.1] and r["referentie"]["mat"] == "A4"
    v = r["validatie"]
    assert v["status"] == "ok" and v["id"] == job and "map" not in v and len(v["vergelijkingen"]) == 4
    assert all(row["binnen_u95"] for row in v["vergelijkingen"])
    assert next(s for s in c.get("/api/scans").json() if s["id"] == job)["maten"] is True
    meetset = c.get("/api/meetset").json()
    assert len(meetset["scans"]) == 1 and meetset["samenvatting"]["maten"] == 4
    assert meetset["samenvatting"]["per_soort"]["gat"]["u95_extra"] == 0.0  # de U95 klopt

    z = zipfile.ZipFile(io.BytesIO(c.get("/api/meetset.zip").content))
    names = set(z.namelist())
    assert {"validatie.json", "validatie.html", "LEESMIJ.txt", f"{job}/maten.json", f"{job}/status.json",
            f"{job}/resultaat/report.json", f"{job}/resultaat/debug/diagnose.json"} <= names
    assert not [n for n in names if n.endswith((".jpg", ".png", ".heic"))]  # geen foto's
    assert str(tmp_path) not in z.read("validatie.json").decode()  # geen paden van deze pc

    c.post(f"/api/scans/{job}/start", data={"meetlijn": "99.9"})  # andere meetlijn: maten.json gaat mee
    assert c.get(f"/api/scans/{job}/maten").json()["referentie"]["meetlijn"] == [99.9, 99.9]
    c.post(f"/api/scans/{job}/fotos", files=[("fotos", ("IMG_9.jpg", jpg(), "image/jpeg"))])
    assert c.get(f"/api/scans/{job}/maten").json()["validatie"]["status"] == "verouderd"  # niet opnieuw verwerkt
    assert c.get("/api/meetset").json()["samenvatting"]["maten"] == 0
    c.delete(f"/api/scans/{job}/maten")
    assert c.get(f"/api/scans/{job}/maten").json()["referentie"] is None
