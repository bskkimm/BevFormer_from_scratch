import numpy as np
import torch
from PIL import Image

from bevformer.engine.bev_vis import (
    box_corners_xy,
    camera_rgb,
    distinctiveness,
    pca_rgb,
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


def test_render_bev_figure_writes_a_png(tmp_path):
    bev = torch.randn(20 * 20, 16)
    gt = torch.tensor([[5.0, 3.0, 0.0, 2.0, 4.5, 1.6, 0.3, 0.0, 0.0]])
    path = tmp_path / "bev.png"
    camera = camera_rgb(torch.randint(0, 256, (3, 9, 16), dtype=torch.uint8))
    render_bev_figure(
        bev, 20, 20, (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0), path, gt_boxes=gt, pred_boxes=gt, title="t", front_camera=camera
    )
    with Image.open(path) as image:
        assert image.size[0] > 1000 and image.size[1] > 200  # three panels side by side


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
    path = visualize_model_bev(model, batch, torch.device("cpu"), None, PC_RANGE, tmp_path / "e.png", title="epoch 0")
    assert path.exists()
    assert model.training
