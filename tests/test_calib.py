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
    # V10: de onzekerheid van de brandpuntsafstand uit de kalibratie; ruim onder de waarschuwingsgrens
    assert 0 < res.camera.f_std_rel < 0.001
    assert calib.CameraModel.from_dict(res.camera.to_dict()).f_std_rel == pytest.approx(res.camera.f_std_rel)
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


def test_mat_format_and_version_are_recognized(mat_scan):
    spec, cam, views = mat_scan
    found, _ = calib.identify_mat([(v.name, v.image) for v in views])
    assert found.name == spec.name
    for name in ("A3", "Letter", "A4-v1", "A3-v1"):
        other = mat.PRESETS[name.upper()]
        vs = render.render_scan(None, other, cam, rings=((50.0, 3),), top_views=1,
                                distance=450.0 if "A3" in name else 330.0, seed=4)
        found, _ = calib.identify_mat([v.image for v in vs])
        assert found is not None and found.name == other.name, name


def test_print_scale_is_part_of_the_calibration(mat_scan):
    """Mat geprint op 100,6% x 99,4%: met de gemeten meetlijnen kloppen poses in werkelijke mm."""
    spec, cam, _ = mat_scan
    printed = spec.with_scale(1.006, 0.994)
    views = render.render_scan(None, printed, cam, rings=((35.0, 5), (55.0, 5), (72.0, 4)), top_views=2, seed=6)
    board = mat.make_board(spec)
    dets = [calib.detect(v.image, board, v.name) for v in views]
    good = calib.calibrate(dets, printed)
    naive = calib.calibrate(dets, spec)
    err_good = max(np.linalg.norm(good.poses[v.name].center - v.pose.center) for v in views)
    err_naive = max(np.linalg.norm(naive.poses[v.name].center - v.pose.center) for v in views)
    assert err_good < 0.6 and err_naive > 1.0
    assert good.camera.rms_px < naive.camera.rms_px


def test_blurry_photos_stay_out_of_the_calibration_but_get_a_pose(mat_scan):
    """V28 (v0.12): een bewogen of niet scherpgestelde foto (σ > 3 px) gaat niet de kalibratie in; zijn pose komt
    uit de camera van de scherpe foto's. Met te weinig scherpe foto's toch allemaal."""
    import cv2

    from camtocad import pipeline, preflight

    spec, cam, views = mat_scan
    board = mat.make_board(spec)
    detector = calib.make_detector(board)
    images = {v.name: v.image for v in views}
    soft = [v.name for v in views[::5]]
    for n in soft:
        images[n] = cv2.GaussianBlur(images[n], (0, 0), 4.5)
    dets = [d for d in (calib.detect(images[v.name], board, v.name, detector) for v in views) if d is not None]
    blur = {d.name: preflight.measure_blur(images[d.name], d, spec) for d in dets}
    found = [d.name for d in dets if blur[d.name] is not None and blur[d.name] > pipeline.CALIB_BLUR_MAX]
    assert found and set(found) <= set(soft)
    lines = []
    res = pipeline.calibrate_sharp(dets, blur, spec, log=lines.append)
    assert any("niet in de kalibratie" in line for line in lines)
    assert res.camera.rms_px < 0.3 and abs(res.camera.K[0, 0] / cam.K[0, 0] - 1) < 0.003
    by_name = {v.name: v for v in views}
    posed = [n for n in found if n in res.poses]
    assert len(posed) >= 2 and all(n in res.rejected for n in found if n not in res.poses)  # te weinig hoeken
    for n in posed:
        assert np.linalg.norm(res.poses[n].center - by_name[n].pose.center) < 2.0  # mm, uit een onscherpe foto
    lines.clear()
    few = [d for d in dets if d.name in found] + [d for d in dets if d.name not in found][:5]
    pipeline.calibrate_sharp(few, blur, spec, log=lines.append)
    # te weinig scherpe foto's: gewoon allemaal, maar sinds v0.14 gewogen naar hun onscherpte (V28)
    assert not any("niet in de kalibratie" in line for line in lines) and any("gewogen" in line for line in lines)


def _synthetic_detections(spec, cam, noise: dict, seed: int = 0):
    """Mathoeken zoals de detector ze zou geven: geprojecteerd met lensvervorming, plus ruis per foto (px)."""
    import cv2

    board = mat.make_board(spec)
    A, a = mat.board_to_mat_transform(spec)
    chess = np.asarray(board.getChessboardCorners(), float)
    rng = np.random.default_rng(seed)
    target = (spec.board_w_mm / 2, spec.board_h_mm / 2, 0.0)
    dets = []
    for name, R, t in render.scan_poses(target, 330.0, ((35.0, 8), (55.0, 8), (72.0, 6)), 4, rng):
        rv, _ = cv2.Rodrigues(R @ A)
        uv, _ = cv2.projectPoints(chess, rv, R @ a + t, cam.K, cam.dist)
        uv = uv.reshape(-1, 2)
        inside = (uv[:, 0] > 5) & (uv[:, 0] < cam.width - 5) & (uv[:, 1] > 5) & (uv[:, 1] < cam.height - 5)
        ids = np.flatnonzero(inside).astype(np.int32)
        corners = uv[ids] + rng.normal(0, noise.get(name, 0.05), (len(ids), 2))
        dets.append(calib.BoardDetection(name, cam.width, cam.height, corners, ids, 0))
    return dets


def test_corners_are_weighted_by_the_blur_of_their_photo():
    """V28 (v0.14): de hoeken van een onscherpe foto liggen minder precies. Zonder gewichten trekken ze de camera mee;
    gewogen naar hun onscherpte (pipeline.calibrate_sharp) blijft de camera bij die van de scherpe foto's."""
    from camtocad import pipeline

    spec, cam = mat.PRESETS["A4"], render.default_camera()
    names = [n for n, _, _ in render.scan_poses((0, 0, 0), 330.0, ((35.0, 8), (55.0, 8), (72.0, 6)), 4,
                                                np.random.default_rng(0))]
    soft = set(names[1::3])  # een derde van de foto's bewogen: σ 2,5 px (net binnen de grens van de kalibratie)
    errs = {"gelijk": [], "gewogen": []}
    for seed in range(3):
        dets = _synthetic_detections(spec, cam, {n: 1.0 for n in soft}, seed=seed)
        blur = {d.name: 2.5 if d.name in soft else 0.6 for d in dets}
        plain = calib.calibrate(dets, spec)
        lines = []
        weighted = pipeline.calibrate_sharp(dets, blur, spec, log=lines.append)
        assert any("gewogen" in line for line in lines) and set(weighted.poses) == set(plain.poses)
        errs["gelijk"].append(abs(plain.camera.K[0, 0] / cam.K[0, 0] - 1))
        errs["gewogen"].append(abs(weighted.camera.K[0, 0] / cam.K[0, 0] - 1))
        assert np.isfinite(weighted.camera.f_std_rel) and weighted.camera.f_std_rel < plain.camera.f_std_rel
    assert np.mean(errs["gewogen"]) < 0.5 * np.mean(errs["gelijk"]) and max(errs["gewogen"]) < 1.5e-3
    # zonder verschil in onscherpte: precies de kalibratie van OpenCV
    dets = _synthetic_detections(spec, cam, {}, seed=5)
    same = pipeline.calibrate_sharp(dets, {d.name: 0.6 for d in dets}, spec, log=lambda m: None)
    assert np.array_equal(same.camera.K, calib.calibrate(dets, spec).camera.K)
