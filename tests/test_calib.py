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
