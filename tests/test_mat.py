import json

import cv2
import numpy as np
import pytest

from camtocad import mat


def test_raster_matches_opencv_board():
    spec = mat.PRESETS["A4"]
    board = mat.make_board(spec)
    ours = mat.rasterize_board(spec, px_per_mm=8, margin_mm=0, supersample=1, dots=False).image
    ref = board.generateImage((int(spec.board_w_mm * 8), int(spec.board_h_mm * 8)), marginSize=0, borderBits=1)
    assert ours.shape == ref.shape
    assert (np.abs(ours.astype(int) - ref.astype(int)) < 128).mean() > 0.99


@pytest.mark.parametrize("name", ["A4", "A3"])
def test_all_corners_detected_with_dots(name):
    spec = mat.PRESETS[name]
    board = mat.make_board(spec)
    raster = mat.rasterize_board(spec, px_per_mm=5, margin_mm=5)
    _, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(raster.image)
    assert len(ids) == (spec.squares_x - 1) * (spec.squares_y - 1)


def test_pdf_structure_and_descriptor(tmp_path):
    paths = mat.write_mat("A4", tmp_path)
    data = paths["pdf"].read_bytes()
    assert data.startswith(b"%PDF-1.4") and data.rstrip().endswith(b"%%EOF")
    # elke xref-offset moet naar het juiste object wijzen
    xref_at = int(data.rsplit(b"startxref", 1)[1].split()[0])
    entries = data[xref_at:].split(b"\n")[3:8]
    for i, entry in enumerate(entries, start=1):
        offset = int(entry.split()[0])
        assert data[offset:].startswith(f"{i} 0 obj".encode())
    desc = json.loads(paths["json"].read_text())
    assert desc["board_size_mm"] == [240.0, 160.0]
    assert paths["png"].stat().st_size > 10_000


def test_pdf_prints_at_exact_scale(tmp_path):
    pdfium = pytest.importorskip("pypdfium2")
    spec = mat.PRESETS["A4"]
    paths = mat.write_mat(spec, tmp_path)
    page = pdfium.PdfDocument(str(paths["pdf"]))[0]
    w_pt, h_pt = page.get_size()
    assert abs(w_pt - 297 * 72 / 25.4) < 0.01 and abs(h_pt - 210 * 72 / 25.4) < 0.01
    dpi = 150
    img = np.array(page.render(scale=dpi / 72).to_pil().convert("L"))
    board = mat.make_board(spec)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(img)
    assert len(ids) == 77
    obj, pts = board.matchImagePoints(corners, ids)
    H, _ = cv2.findHomography(obj.reshape(-1, 3)[:, :2], pts.reshape(-1, 2))
    px_per_mm = dpi / 25.4
    for col in (0, 1):
        scale = np.linalg.norm(H[:2, col]) / H[2, 2] / px_per_mm
        assert abs(scale - 1.0) < 0.001  # binnen 0,1%


def test_board_to_mat_is_proper_rotation():
    spec = mat.PRESETS["A3"]
    A, a = mat.board_to_mat_transform(spec)
    assert np.isclose(np.linalg.det(A), 1.0)
    p = mat.board_to_mat(np.array([[0.0, 0.0, 0.0], [spec.board_w_mm, spec.board_h_mm, 0.0]]), spec)
    assert np.allclose(p, [[0, spec.board_h_mm, 0], [spec.board_w_mm, 0, 0]])
