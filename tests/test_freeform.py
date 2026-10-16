"""Vrije vorm (v0.17): de vorm van een onderdeel dat niet plat is, als mesh uit silhouetten en stereo."""

import zipfile

import cv2
import numpy as np
import pytest

from camtocad import freeform, mvs
from camtocad.calib import Pose
from camtocad.masks import ViewMasks
from camtocad.render import look_at, scan_poses

K = np.array([[650.0, 0.0, 399.5], [0.0, 650.0, 299.5], [0.0, 0.0, 1.0]])
SIZE = (800, 600)


def sphere_views(c0, r, rings=((20.0, 10), (40.0, 10), (60.0, 8)), top=4):
    """Analytische maskers van een bol (middelpunt c0, straal r) vanuit camera's rondom en van boven."""
    w, h = SIZE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    views = []
    for name, R, t in scan_poses(np.array([c0[0], c0[1], c0[2] / 2]), 330.0, rings, top):
        pose = Pose(name, R, t)
        d = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)], -1) @ R
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        oc = pose.center - c0
        b = d @ oc
        fg = b * b - (oc @ oc - r * r) > 0
        views.append((pose, ViewMasks(fg=fg, bg=~fg, valid=np.ones_like(fg), edge_bg=~fg)))
    return views


def closed(mesh: freeform.Mesh) -> bool:
    """Waterdicht en overal even georiënteerd: bij elke ribbe ligt er ook een in de andere richting. Surface nets
    geeft af en toe een ribbe met vier driehoeken (twee cellen die elkaar alleen schuin raken); dat mag, zolang het
    zeldzaam is (slicers repareren het)."""
    E = np.concatenate([mesh.F[:, [0, 1]], mesh.F[:, [1, 2]], mesh.F[:, [2, 0]]])
    key, rev = E[:, 0] * len(mesh.V) + E[:, 1], E[:, 1] * len(mesh.V) + E[:, 0]
    return np.setdiff1d(key, rev).size == 0 and len(key) - np.unique(key).size < 1e-3 * len(key)


def test_a_sphere_from_its_silhouettes():
    """Een bol boven de mat: de mesh is gesloten, ligt (behalve aan de onderkant, die geen foto ziet) binnen een
    tiende mm op de bol, en valt in elke foto op het silhouet."""
    c0, r = np.array([140.0, 100.0, 25.0]), 20.0
    views = sphere_views(c0, r)
    mesh = freeform.reconstruct(views, K, c0 - 30, c0 + 30, h=0.5)
    assert closed(mesh)
    dist = np.linalg.norm(mesh.V - c0, axis=1) - r
    upper = mesh.V[:, 2] > c0[2]
    assert abs(np.median(dist[upper])) < 0.1 and np.percentile(np.abs(dist[upper]), 95) < 0.6
    assert mesh.volume() == pytest.approx(4 / 3 * np.pi * r ** 3, rel=0.06)  # een kegeltje eronder
    assert np.median(freeform.view_iou(mesh, views, K)) > 0.98  # de bol is maar ~80 px breed in deze foto's


def test_one_bad_mask_does_not_cut_into_the_part():
    """Een foto waarin het masker het onderdeel half mist (alles 'zekere mat'), snijdt niets weg: daarvoor zijn
    minstens drie foto's nodig."""
    c0, r = np.array([140.0, 100.0, 25.0]), 20.0
    views = sphere_views(c0, r)
    pose, m = views[3]
    half = m.fg.copy()
    _, xs = np.nonzero(half)
    half[:, : int(np.median(xs))] = False  # de linkerhelft van het onderdeel 'kwijt'
    views[3] = (pose, ViewMasks(fg=half, bg=~half, valid=m.valid, edge_bg=~half))
    mesh = freeform.reconstruct(views, K, c0 - 30, c0 + 30, h=0.5)
    dist = np.linalg.norm(mesh.V - c0, axis=1) - r
    assert dist[mesh.V[:, 2] > c0[2]].min() > -0.3


def test_specks_of_mat_on_a_textured_part_are_ignored():
    """Losse vlekjes 'zekere mat' midden op het onderdeel (textuur die op het matpatroon lijkt) tellen niet."""
    fg = np.zeros((200, 200), bool)
    fg[50:150, 50:150] = True
    mat = ~fg
    mat[90:94, 90:94] = True  # vlekje van 16 px op het onderdeel
    sd = freeform.signed_distance(ViewMasks(fg=fg, bg=mat, valid=np.ones_like(fg), edge_bg=mat))
    assert sd[92, 92] > 30 and sd[100, 49] < 0 < sd[100, 50]


def test_part_frame_and_robust_extent():
    """Het assenstelsel van het onderdeel: lange zijde langs X, oorsprong linksonder; een los piekje telt niet mee
    in de buitenmaten."""
    import cadquery as cq

    from camtocad.render import tessellate
    box = cq.Workplane("XY").box(60, 25, 10, centered=(True, True, False)).rotate((0, 0, 0), (0, 0, 1), 30)
    V, F = tessellate(box.translate((100, 80, 0)), 0.05, 0.2)
    mesh = freeform.Mesh(V, F)
    R, t = freeform.part_frame(mesh)
    pm = mesh.transformed(R, t)
    lo, hi = freeform.robust_extent(pm)
    assert hi - lo == pytest.approx([60, 25, 10], abs=0.2) and lo[:2] == pytest.approx([0, 0], abs=0.1)
    # een los piekje van 15 mm op een fijne mesh (een bol uit silhouetten) telt niet mee
    c0, r = np.array([140.0, 100.0, 25.0]), 20.0
    ball = freeform.reconstruct(sphere_views(c0, r, rings=((30.0, 8), (60.0, 6)), top=2), K, c0 - 30, c0 + 30, h=1.0)
    top = int(np.argmax(ball.V[:, 2]))
    nb = ball.F[np.any(ball.F == top, axis=1)][0]
    spike = freeform.Mesh(np.vstack([ball.V, ball.V[top] + [0.0, 0.0, 15.0]]),
                          np.vstack([ball.F, [[nb[0], nb[1], len(ball.V)]]]))
    assert freeform.robust_extent(spike)[1][2] == pytest.approx(ball.V[:, 2].max(), abs=0.6)


def test_mesh_files(tmp_path):
    """Binaire STL (50 bytes per driehoek) en 3MF (zip met het model in mm)."""
    c0, r = np.array([140.0, 100.0, 25.0]), 20.0
    mesh = freeform.reconstruct(sphere_views(c0, r, rings=((30.0, 8), (60.0, 6)), top=2), K, c0 - 30, c0 + 30, h=1.0)
    freeform.write_stl(mesh, tmp_path / "m.stl", "bol")
    raw = (tmp_path / "m.stl").read_bytes()
    assert len(raw) == 84 + 50 * len(mesh.F) and int.from_bytes(raw[80:84], "little") == len(mesh.F)
    freeform.write_3mf(mesh, tmp_path / "m.3mf", "bol")
    with zipfile.ZipFile(tmp_path / "m.3mf") as z:
        model = z.read("3D/3dmodel.model").decode()
        assert "[Content_Types].xml" in z.namelist()
    assert 'unit="millimeter"' in model and model.count("<vertex ") == len(mesh.V)
    assert model.count("<triangle ") == len(mesh.F)
    img = freeform.render_preview(mesh)
    assert img.shape == (360, 480, 3) and (img < 250).any()


def textured_plane_views(depth_plane: float = 2.0):
    """Drie foto's van een vlak met textuur (z = depth_plane boven de mat), van 60° hoog en 20° uit elkaar."""
    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur(rng.uniform(0, 255, (400, 400)).astype(np.float32), (0, 0), 1.5)
    tex = cv2.normalize(tex, None, 30, 220, cv2.NORM_MINMAX)
    target = np.array([100.0, 100.0, depth_plane])
    poses = []
    for k, az in enumerate((-90.0, -70.0, -110.0)):
        a, e = np.radians(az), np.radians(60.0)
        c = target + 300.0 * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        R, t = look_at(c, target)
        poses.append(Pose(f"v{k}", R, t))
    imgs = []
    for pose in poses:  # het vlak (mat-mm, 0,25 mm per textuurpixel vanaf (50, 50)) in beeld: een homografie
        A = np.array([[0.25, 0, 50.0], [0, 0.25, 50.0], [0, 0, 1.0]])
        M = K @ np.column_stack([pose.R[:, 0], pose.R[:, 1], pose.R[:, 2] * depth_plane + pose.t]) @ A
        imgs.append(cv2.warpPerspective(tex, M, SIZE, flags=cv2.INTER_LINEAR))
    return poses, imgs


def test_stereo_finds_the_depth_of_a_textured_surface():
    """Een vlak met textuur, 2 mm boven waar de hull begint: de dieptekaart vindt het binnen een paar tiende mm."""
    poses, imgs = textured_plane_views(2.0)
    ref = poses[0]
    w, h = SIZE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ray = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)], -1)
    # ware diepte: het snijpunt van de straal met z = 2, met X_mat = d · R^T straal + cameracentrum
    d_true = (2.0 - ref.center[2]) / (ray @ ref.R)[..., 2]
    front = (d_true - 1.5).astype(np.float32)  # de hull begint 1,5 mm ervoor
    back = (d_true + 20.0).astype(np.float32)
    mask = np.zeros((h, w), bool)
    mask[250:350, 340:460] = True
    depth, _ = mvs.depth_map(imgs[0], ref, [(imgs[1], poses[1]), (imgs[2], poses[2])], K, mask, front, back)
    ok = np.isfinite(depth)
    assert ok[mask].mean() > 0.8
    err = depth[ok] - d_true[ok]
    assert abs(np.median(err)) < 0.2 and np.percentile(np.abs(err), 95) < 0.6


def test_fused_depths_put_the_surface_at_the_measured_depth():
    """TSDF: het nulniveau ligt op de gemeten diepte; één foute diepte tussen goede trekt het nauwelijks weg, en
    met minder dan drie foto's is er geen bewijs."""
    R, t = look_at((0.0, 0.0, 300.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    maps = []
    for k, d in enumerate((290.0, 290.0, 290.0, 290.0, 291.5)):  # het oppervlak op z = 10; één kaart 1,5 mm te diep
        depth = np.full((SIZE[1], SIZE[0]), d, np.float32)
        maps.append(mvs.DepthView(Pose(f"v{k}", R, t), depth, np.ones_like(depth)))
    pts = np.array([[0.0, 0.0, z] for z in (12.0, 10.5, 10.0, 9.5, 8.0)])
    f = mvs.fused(pts, maps, K, 1e4)
    assert f[0] < 0 and f[1] < 0 and f[3] > 0 and f[4] > 0 and abs(f[2]) < 0.3
    assert mvs.fused(pts, maps[:2], K, 1e4)[0] == 1e4


@pytest.mark.slow
@pytest.mark.stress
def test_a_free_form_scan_of_a_textured_bracket(tmp_path):
    """Van begin tot eind: een L-beugel met een opstaande flens (geen plat onderdeel) met textuur, als vrije vorm.
    Met 32 foto's in plaats van 46 (sneller) zijn de buitenmaten ~1 mm te ruim (met 46: ~0,4 mm); de eis is 1,5 mm.
    STL, 3MF en het rapport, geen model.step; sinds v0.18 de vlakken in referentie.step, en als maten de dikte van
    de plaat en de flens (6 mm) en de afstanden tussen de zijkanten (60 en 40 mm)."""
    from camtocad import stresstest
    cfg = dict(stresstest.SCENARIOS["vrij_beugel"], rings=((35.0, 12), (55.0, 10), (72.0, 6)), top_views=4)
    summary = stresstest.run("vrij_beugel", tmp_path / "run", cfg=cfg, log=lambda m: None)
    assert summary["status"] == "OK", summary["log"][-3:]
    for name, truth in (("lengte (X)", 60.0), ("breedte (Y)", 40.0), ("hoogte (Z)", 40.0)):
        assert abs(summary["dims"][name] - truth) < 1.5, (name, summary["dims"])
    res = tmp_path / "run" / "vrij_beugel" / "resultaat"
    assert (res / "model.stl").exists() and (res / "model.3mf").exists() and not (res / "model.step").exists()
    assert "vrije vorm" in (res / "report.html").read_text(encoding="utf-8")
    assert (res / "referentie.step").exists()
    shapes = {k: v for k, v in summary["dims"].items() if not k.endswith((" (X)", " (Y)", " (Z)"))}
    thick = [v for k, v in shapes.items() if k.startswith("dikte")]
    dist = [v for k, v in shapes.items() if k.startswith("afstand")]
    assert thick and all(abs(v - 6.0) < 0.5 for v in thick), shapes
    assert any(abs(v - 60.0) < 0.5 for v in dist) and any(abs(v - 40.0) < 0.5 for v in dist), shapes
