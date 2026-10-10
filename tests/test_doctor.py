"""`camtocad doctor` (V23, v0.15): de installatie controleren."""

from camtocad import cli, doctor


def test_two_opencv_packages_are_an_error():
    """opencv-python en opencv-python-headless schrijven allebei cv2: welke je krijgt, hangt af van de volgorde."""
    c = doctor.check_opencv({"opencv-python": "4.10.0.84", "opencv-python-headless": "5.0.0.93",
                             "opencv-contrib-python": None, "opencv-contrib-python-headless": None})
    assert c.status == doctor.FOUT and "opencv-python 4.10" in c.detail and "pip uninstall" in c.remedy
    assert doctor.check_opencv({"opencv-python-headless": "5.0.0.93"}).status == doctor.OK


def test_little_memory_and_a_full_disk_are_warnings(tmp_path, monkeypatch):
    assert doctor.check_memory(4.0).status == doctor.LET_OP
    assert doctor.check_memory(16.0).status == doctor.OK
    assert doctor.check_data_dir(tmp_path / "nieuw" / "data").status == doctor.OK  # wordt aangemaakt
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: type("U", (), {"free": 1024 ** 3})())
    c = doctor.check_data_dir(tmp_path)
    assert c.status == doctor.LET_OP and "1.0 GB vrij" in c.detail


def test_a_busy_port_is_reported():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        s.listen()
        port = s.getsockname()[1]
        c = doctor.check_ports((port,))
    assert c.status == doctor.LET_OP and str(port) in c.detail


def test_the_installation_of_the_tests_is_complete(tmp_path, capsys):
    """In de testomgeving is alles geïnstalleerd: geen fouten, en CadQuery schrijft een STEP-bestand."""
    checks = doctor.run_checks(tmp_path)
    by = {c.name: c for c in checks}
    assert not [c for c in checks if c.status == doctor.FOUT], [c.to_dict() for c in checks]
    assert by["CadQuery"].status == doctor.OK and "STEP" in by["CadQuery"].detail
    assert by["Matdetectie"].status == doctor.OK
    assert cli.main(["doctor", "--data", str(tmp_path), "--snel"]) == 0
    out = capsys.readouterr().out
    assert "controle van de installatie" in out and "Matdetectie" not in out  # --snel: zonder de trage controles


def test_an_error_gives_an_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(doctor, "check_opencv", lambda: doctor.Check("OpenCV", doctor.FOUT, "kapot", "herinstalleren"))
    assert cli.main(["doctor", "--data", str(tmp_path), "--snel"]) == 1
    out = capsys.readouterr().out
    assert "FOUT    OpenCV: kapot" in out and "-> herinstalleren" in out and "1 fout(en)" in out
