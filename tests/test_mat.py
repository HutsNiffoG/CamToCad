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


@pytest.mark.parametrize("name", list(mat.PRESETS))
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


def test_raster_follows_opencv_pixel_convention():
    """mat_to_pixel_matrix: pixelmiddens op gehele coördinaten, dus een vakrand ligt op het 50%-punt."""
    spec = mat.PRESETS["A4"]
    raster = mat.rasterize_board(spec, px_per_mm=10, margin_mm=3, dots=False)
    M = raster.mat_to_pixel_matrix()
    s = spec.square_mm
    for black_row in range(spec.squares_y):
        y_mm = spec.board_h_mm - (black_row + 0.5) * s  # midden van een rij vakken (mat-Y omhoog)
        x_edge = s  # grens tussen de eerste twee vakken van die rij
        col, row, _ = M @ [x_edge, y_mm, 1.0]
        line = raster.image[int(round(row))].astype(float)
        c0 = int(np.floor(col))
        left, right = line[c0 - 3], line[c0 + 4]
        if abs(left - right) < 200:  # rand tussen een marker en een zwart vak: niet egaal
            continue
        mid = np.interp(col, np.arange(len(line)), line)
        assert abs(mid - (left + right) / 2) < 3.0
        return
    pytest.fail("geen geschikte vakrand gevonden")


def test_v2_marker_ranges_identify_the_format():
    """Elke v2-mat heeft een eigen ID-bereik, buiten de ID's van v1 (0-249 uit DICT_5X5_250)."""
    v2 = [s for s in mat.PRESETS.values() if s.version == 2]
    ranges = [set(s.marker_ids.tolist()) for s in v2]
    for i, a in enumerate(ranges):
        assert min(a) >= 250
        for b in ranges[i + 1:]:
            assert not a & b
    for s in v2:  # het bord gebruikt die ID's echt
        assert np.array_equal(mat.make_board(s).getIds().ravel(), s.marker_ids)


def test_v2_texture_keeps_corners_and_edges_clear():
    """Stippen blijven weg van de schaakbordhoeken (daar verfijnt OpenCV) en van de vakranden."""
    spec = mat.PRESETS["A4"]
    s = spec.square_mm
    dots = [r for r in mat.board_rects(spec) if r[4] == 1.0]
    assert len(dots) > 30 * spec.n_markers  # dicht raster in elk zwart vak
    for x, y, w, h, _ in dots:
        i, j = int((x + w / 2) // s), int((y + h / 2) // s)
        lx, ly = x - i * s, y - j * s  # binnen het vak
        assert min(lx, ly, s - lx - w, s - ly - h) >= 1.5 - 1e-9
        xs, ys = np.clip([0.0, s], lx, lx + w), np.clip([0.0, s], ly, ly + h)  # dichtstbijzijnde stippunten
        assert min(np.hypot(abs(cx - px), abs(cy - py)) for cx, px in zip((0.0, s), xs)
                   for cy, py in zip((0.0, s), ys)) >= 4.0 - 1e-9


def test_print_scale_maps_true_mm_onto_the_nominal_drawing():
    spec = mat.PRESETS["A4"]
    scaled = spec.with_scale(1.012, 0.991)
    M0 = mat.rasterize_board(spec, 4, 3, supersample=1, dots=False).mat_to_pixel_matrix()
    M1 = mat.rasterize_board(scaled, 4, 3, supersample=1, dots=False).mat_to_pixel_matrix()
    for X, Y in ((0.0, 0.0), (37.0, 11.0), (240.0, 160.0)):
        assert np.allclose(M1 @ [1.012 * X, 0.991 * Y, 1.0], M0 @ [X, Y, 1.0])
    assert scaled.size_mm == pytest.approx((240 * 1.012, 160 * 0.991))
    p = mat.board_to_mat(np.array([[0.0, 0.0, 0.0], [240.0, 160.0, 0.0]]), scaled)
    assert np.allclose(p, [[0.0, 160 * 0.991, 0.0], [240 * 1.012, 0.0, 0.0]])
    assert scaled.nominal() == spec and mat.get_spec("letter").name == "Letter"
