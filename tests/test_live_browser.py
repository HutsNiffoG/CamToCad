"""De livecamera in een echte browser (V25, v0.13): Chromium met een nepcamera die een gerenderd bovenaanzicht van
de mat laat zien. De pagina moet de aanwijzing tonen en zelf de foto maken en uploaden. Draait alleen waar Playwright
en Chromium zijn (lokaal); anders overgeslagen. Een eigen Chromium: CTC_CHROMIUM=/pad/naar/chrome."""

import json
import os
import socket
import threading
import time

import numpy as np
import pytest

sync_api = pytest.importorskip("playwright.sync_api")
pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from camtocad import mat, render  # noqa: E402
from camtocad.server.app import create_app  # noqa: E402


def write_y4m(path, frames) -> None:
    """Grijze beelden als y4m-video (YUV 4:2:0) voor --use-file-for-fake-video-capture."""
    h, w = frames[0].shape
    chroma = np.full((h // 2) * (w // 2), 128, np.uint8).tobytes()
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F10:1 Ip A1:1 C420jpeg\n".encode())
        for img in frames:
            f.write(b"FRAME\n" + img.tobytes() + chroma + chroma)


def test_live_camera_guides_and_takes_the_photo(tmp_path):
    spec = mat.PRESETS["A4"]
    cam = render.default_camera(1280, 720, 940.0, dist=(0, 0, 0, 0, 0))
    R, t = render.look_at([125.0, 80.0, 330.0], [125.0, 80.0, 0.0])
    img, _ = render.render_view(mat.rasterize_board(spec, 5.0, 3.0), cam, R, t, rng=np.random.default_rng(2))
    video = tmp_path / "mat.y4m"
    write_y4m(video, [(img * 0.8).astype(np.uint8)] * 3)  # Chromium rekt de grijswaarden soms op naar 'full range'
    data = tmp_path / "data"
    app = create_app(data, token="t")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started and thread.is_alive():
        time.sleep(0.05)
    try:
        with sync_api.sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(executable_path=os.environ.get("CTC_CHROMIUM") or None,
                                             args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                                                   f"--use-file-for-fake-video-capture={video}"])
            except Exception as e:  # noqa: BLE001 - geen (passende) Chromium op deze machine
                pytest.skip(f"Chromium niet beschikbaar: {str(e).splitlines()[0]}")
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{port}/?token=t")  # localhost telt als beveiligde pagina
            page.click("#live-start")
            page.wait_for_function("document.querySelector('#live-tip').textContent.includes('recht van boven')",
                                   timeout=30000)
            assert page.locator("#live-overlay polygon").count() == 1  # de omtrek van de mat over het beeld
            # 'automatisch' staat aan: na twee keer 'opnemen' maakt de pagina de foto en uploadt hem
            page.wait_for_function("/^[1-9]/.test(document.querySelector('#live-count').textContent)", timeout=30000)
            page.wait_for_function("document.querySelector('#live-tip').textContent.includes('al een foto')",
                                   timeout=30000)  # vanaf dezelfde plek geen tweede foto
            page.click("#live-stop")
            assert page.locator("#live").is_hidden()
            browser.close()
    finally:
        server.should_exit = True
        thread.join(10)
    job = next(p for p in data.iterdir() if p.is_dir() and (p / "status.json").exists())
    photos = sorted(p.name for p in (job / "fotos").iterdir())
    assert len(photos) == 1
    check = json.loads((job / "controle.json").read_text())["fotos"][photos[0]]
    assert check["mat"] == "A4" and check["verdict"] == "goed", check["notes"]


def test_an_interrupted_upload_is_sent_again_without_a_double(tmp_path):
    """V24 (v0.15): de eerste poging om een foto te uploaden breekt af (de server heeft hem wel), de tweede komt
    niet aan (netwerkfout); de pagina probeert het opnieuw met hetzelfde id, en de foto staat er precies één keer."""
    import cv2

    data = tmp_path / "data"
    app = create_app(data, token="t")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started and thread.is_alive():
        time.sleep(0.05)
    photo = tmp_path / "IMG_1.jpg"
    photo.write_bytes(cv2.imencode(".jpg", np.full((60, 80), 128, np.uint8))[1].tobytes())
    tries = []
    try:
        with sync_api.sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(executable_path=os.environ.get("CTC_CHROMIUM") or None)
            except Exception as e:  # noqa: BLE001
                pytest.skip(f"Chromium niet beschikbaar: {str(e).splitlines()[0]}")
            page = browser.new_page()

            def flaky(route):
                tries.append(route.request.post_data_buffer)
                if len(tries) == 1:  # de server krijgt de foto, maar het antwoord gaat verloren
                    route.fetch()
                    route.abort("connectionreset")
                elif len(tries) == 2:  # verbinding weg
                    route.abort("internetdisconnected")
                else:
                    route.continue_()

            page.route("**/api/scans/*/fotos", flaky)
            page.goto(f"http://127.0.0.1:{port}/?token=t")
            page.uncheck("#verklein")
            page.set_input_files("#kies", str(photo))
            page.wait_for_function("document.querySelectorAll('#lijst .photo').length === 1", timeout=60000)
            browser.close()
    finally:
        server.should_exit = True
        thread.join(10)
    import re

    uids = {re.search(rb'name="uid"\r\n\r\n([^\r]+)', bytes(t)).group(1) for t in tries}
    assert len(tries) == 3 and len(uids) == 1  # drie keer dezelfde foto met hetzelfde id
    job = next(p for p in data.iterdir() if p.is_dir() and (p / "status.json").exists())
    assert [p.name for p in (job / "fotos").iterdir()] == ["foto_0000.jpg"]
