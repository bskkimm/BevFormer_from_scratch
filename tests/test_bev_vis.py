import numpy as np
import torch
from PIL import Image

from bevformer.engine.bev_vis import (
    BevVisSample,
    box_corners_3d,
    box_corners_xy,
    camera_rgb,
    distinctiveness,
    pca_rgb,
    project_box_edges,
    render_bev_figure,
    visualize_model_bev,
)
from tests.test_bevformer_model import PC_RANGE, _build_model, _make_img_metas_queue


def test_pca_rgb_is_in_unit_range_and_sign_deterministic():
    torch.manual_seed(0)
    features = torch.randn(6, 7, 16)
    rgb = pca_rgb(features)
    assert rgb.shape == (6, 7, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    np.testing.assert_allclose(pca_rgb(-features), rgb, atol=1e-5)  # sign flips don't change colors


def test_box_corners_follow_yaw():
    corners = box_corners_xy(torch.tensor([10.0, 0.0, 0.0, 2.0, 4.0, 1.5, np.pi / 2, 0.0, 0.0]))
    # Length (4 m) now points along +y: corners span x in [9, 11] and y in [-2, 2].
    np.testing.assert_allclose(sorted(set(np.round(corners[:, 0], 5))), [9.0, 11.0], atol=1e-5)
    np.testing.assert_allclose(sorted(set(np.round(corners[:, 1], 5))), [-2.0, 2.0], atol=1e-5)


def test_box_corners_3d_span_the_box_height_around_its_center():
    corners = box_corners_3d(torch.tensor([10.0, 0.0, 1.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0]))
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners[:4, 2], 0.25)
    np.testing.assert_allclose(corners[4:, 2], 1.75)
    np.testing.assert_allclose(corners[4:, :2], corners[:4, :2])


def test_project_box_edges_lands_on_the_pinhole_pixels_and_skips_boxes_behind():
    # Camera looking along +y (LIDAR forward): u = f x / y + cx, v = -f z / y + cy.
    focal, cx, cy = 100.0, 50.0, 40.0
    lidar2cam = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])
    intrinsic = np.array([[focal, 0, cx, 0], [0, focal, cy, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    lidar2img = intrinsic @ lidar2cam
    ahead = torch.tensor([0.0, 10.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0])  # 2 m cube centered 10 m ahead
    segments = project_box_edges(ahead, lidar2img)
    assert len(segments) == 12
    pixels = np.concatenate(segments)
    # Near face (y = 9) spans x, z in [-1, 1]: u, v in cx/cy +- 100 / 9.
    np.testing.assert_allclose(pixels[:, 0].min(), cx - focal / 9.0, rtol=1e-6)
    np.testing.assert_allclose(pixels[:, 1].max(), cy + focal / 9.0, rtol=1e-6)
    behind = torch.tensor([0.0, -10.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0])
    assert project_box_edges(behind, lidar2img) == []


def test_render_bev_figure_writes_one_row_per_sample(tmp_path):
    gt = torch.tensor([[5.0, 3.0, 0.0, 2.0, 4.5, 1.6, 0.3, 0.0, 0.0]])
    camera = camera_rgb(torch.randint(0, 256, (3, 9, 16), dtype=torch.uint8))
    rows = [
        BevVisSample(torch.randn(20 * 20, 16), gt, gt, camera, np.eye(4), name=f"#{i}") for i in range(3)
    ]
    paths = []
    for count in (1, 3):
        path = tmp_path / f"bev{count}.png"
        render_bev_figure(rows[:count], 20, 20, (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0), path, title="t")
        paths.append(path)
    with Image.open(paths[0]) as one, Image.open(paths[1]) as three:
        assert one.size[0] > 1500  # three panels side by side
        assert three.size[1] > 2.5 * one.size[1]  # three rows stacked


def test_distinctiveness_is_informative_after_layernorm():
    torch.manual_seed(0)
    background = torch.randn(32)  # a near-uniform scene: every cell ~ the same feature
    features = torch.nn.functional.layer_norm(background + 0.05 * torch.randn(10, 10, 32), (32,))
    features[3, 4] = torch.nn.functional.layer_norm(torch.randn(32), (32,))  # one different cell
    assert np.ptp(features.norm(dim=-1).numpy()) < 1e-3          # norms are flat after LayerNorm
    score = distinctiveness(features)
    assert np.unravel_index(score.argmax(), score.shape) == (3, 4)  # the odd cell stands out


def test_visualize_model_bev_restores_train_mode(tmp_path):
    model = _build_model(embed_dims=8, num_cams=2)
    model.train()
    batch = {
        "imgs": torch.randn(1, 2, 2, 3, 64, 64),
        "img_metas": _make_img_metas_queue(1, 2, 2),
        "can_bus": torch.zeros(1, 2, 18),
        "gt_boxes_3d": [torch.tensor([[1.0, 2.0, 0.0, 2.0, 4.0, 1.5, 0.1, 0.0, 0.0]])],
        "gt_labels_3d": [torch.tensor([1])],
    }
    path = visualize_model_bev(model, [batch, batch], torch.device("cpu"), None, PC_RANGE, tmp_path / "e.png", title="epoch 0")
    assert path.exists()
    assert model.training
