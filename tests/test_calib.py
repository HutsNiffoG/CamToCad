import numpy as np
import pytest

from camtocad import calib, mat, render


@pytest.fixture(scope="module")
def mat_scan():
    spec = mat.PRESETS["A4"]
    cam = render.default_camera()
    views = render.render_scan(None, spec, cam, rings=((35.0, 7), (55.0, 7), (72.0, 5)), top_views=3, seed=3)
    return spec, cam, views


def test_self_calibration_and_poses(mat_scan):
    spec, cam, views = mat_scan
    board = mat.make_board(spec)
    detector = calib.make_detector(board)
    dets = [calib.detect(v.image, board, v.name, detector) for v in views]
    assert all(d is not None for d in dets)
    res = calib.calibrate(dets, spec)
    assert res.camera.rms_px < 0.3
    assert abs(res.camera.K[0, 0] / cam.K[0, 0] - 1) < 0.003
    assert np.allclose(res.camera.K[:2, 2], cam.K[:2, 2], atol=3.0)
    for v in views:
        p = res.poses[v.name]
        assert np.linalg.norm(p.center - v.pose.center) < 0.6  # mm
        dR = p.R @ v.pose.R.T
        angle = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        assert angle < 0.1


def test_known_camera_gives_same_poses(mat_scan):
    spec, cam, views = mat_scan
    board = mat.make_board(spec)
    dets = [calib.detect(v.image, board, v.name) for v in views[:5]]
    res = calib.calibrate(dets, spec, camera=cam)
    for v in views[:5]:
        assert np.linalg.norm(res.poses[v.name].center - v.pose.center) < 0.6


def test_too_few_views_raises(mat_scan):
    spec, _, views = mat_scan
    board = mat.make_board(spec)
    dets = [calib.detect(v.image, board, v.name) for v in views[:3]]
    with pytest.raises(ValueError, match="Te weinig"):
        calib.calibrate(dets, spec)


def test_detector_bias_is_measured_and_removed(mat_scan):
    """OpenCV 4.x legt ChArUco-hoeken ~0,5 px verschoven; na correctie liggen ze zuiver (elke versie)."""
    import cv2

    spec, cam, views = mat_scan
    bias = calib.detector_bias(spec)
    assert np.all(np.abs(bias) < 1.0)
    board = mat.make_board(spec)
    A, a = mat.board_to_mat_transform(spec)
    chess = np.asarray(board.getChessboardCorners(), float).reshape(-1, 3) @ A.T + a
    offsets = []
    for v in views[::3]:
        d = calib.detect(v.image, board, v.name, bias=bias)
        uv, _ = cv2.projectPoints(chess[d.ids], cv2.Rodrigues(v.pose.R)[0], v.pose.t, cam.K, cam.dist)
        offsets.append(d.corners - uv.reshape(-1, 2))
    assert np.all(np.abs(np.vstack(offsets).mean(axis=0)) < 0.1)


def test_outlier_rounds_keep_poses_with_their_photos(mat_scan):
    """Regressie: werd in een tweede ronde een foto afgekeurd, dan kregen de volgende foto's de pose
    van hun buurman (rvecs/tvecs hoorden nog bij de oude lijst)."""
    import copy

    spec, _, views = mat_scan
    board = mat.make_board(spec)
    dets = [copy.deepcopy(calib.detect(v.image, board, v.name)) for v in views]
    rng = np.random.default_rng(0)
    a = dets[2]  # grove uitschieter: hoek-ID's door elkaar
    a.corners = a.corners[rng.permutation(len(a.ids))]
    b = dets[9]  # matige ruis: pas in de tweede ronde afgekeurd
    b.corners = b.corners + rng.normal(0, 2.5 / np.sqrt(2), b.corners.shape)
    res = calib.calibrate(dets, spec)
    assert views[2].name in res.rejected
    truth = {v.name: v.pose for v in views}
    for name, p in res.poses.items():
        assert np.linalg.norm(p.center - truth[name].center) < 1.0, name
