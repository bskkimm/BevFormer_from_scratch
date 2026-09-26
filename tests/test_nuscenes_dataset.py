import numpy as np
import pytest
import torch

from bevformer.data.nuscenes_categories import CLASS_TO_ID
from bevformer.data.nuscenes_dataset import BevFormerNuScenesDataset
from bevformer.data.transforms import normalize_images
from tests.fixtures.build_synthetic_nuscenes import build_synthetic_nuscenes


def _build_dataset(tmp_path, queue_length=4):
    info = build_synthetic_nuscenes(tmp_path)
    dataset = BevFormerNuScenesDataset(
        dataroot=info["dataroot"],
        version=info["version"],
        queue_length=queue_length,
        image_size=(8, 16),
    )
    return dataset, info


def test_dataset_length_matches_total_samples(tmp_path):
    dataset, _ = _build_dataset(tmp_path)
    assert len(dataset) == 5 + 2


def test_full_queue_has_no_padding_in_long_scene(tmp_path):
    dataset, info = _build_dataset(tmp_path, queue_length=4)
    last_token = info["scene_a_sample_tokens"][-1]
    idx = dataset.sample_tokens.index(last_token)
    sample = dataset[idx]

    assert sample["imgs"].shape == (4, 6, 3, 8, 16)
    assert sample["imgs"].dtype == torch.float32
    assert len(sample["img_metas"]) == 4

    prev_bev_exists = [meta["prev_bev_exists"] for meta in sample["img_metas"]]
    assert prev_bev_exists == [False, True, True, True]

    tokens_in_queue = [meta["sample_token"] for meta in sample["img_metas"]]
    assert tokens_in_queue == info["scene_a_sample_tokens"][1:5]


def test_short_scene_pads_with_earliest_frame(tmp_path):
    dataset, info = _build_dataset(tmp_path, queue_length=4)
    last_token = info["scene_b_sample_tokens"][-1]
    idx = dataset.sample_tokens.index(last_token)
    sample = dataset[idx]

    tokens_in_queue = [meta["sample_token"] for meta in sample["img_metas"]]
    # Only 2 real samples exist in scene_b; earliest one repeats to fill the queue.
    assert tokens_in_queue == [
        info["scene_b_sample_tokens"][0],
        info["scene_b_sample_tokens"][0],
        info["scene_b_sample_tokens"][0],
        info["scene_b_sample_tokens"][1],
    ]

    prev_bev_exists = [meta["prev_bev_exists"] for meta in sample["img_metas"]]
    # Padded repeats carry no new temporal info; only the final real transition does.
    assert prev_bev_exists == [False, False, False, True]


def test_can_bus_deltas_zero_when_no_prev_bev_and_nonzero_otherwise(tmp_path):
    dataset, info = _build_dataset(tmp_path, queue_length=4)
    last_token = info["scene_a_sample_tokens"][-1]
    idx = dataset.sample_tokens.index(last_token)
    sample = dataset[idx]

    can_bus = sample["can_bus"]
    assert can_bus.shape == (4, 18)

    delta_x = can_bus[:, 16]
    delta_y = can_bus[:, 17]

    assert delta_x[0].item() == 0.0
    assert delta_y[0].item() == 0.0
    # Ego moves +1.0m in x per sample, so real consecutive frames have a
    # positive x delta and zero y delta (no lateral motion in fixture).
    for i in range(1, 4):
        assert delta_x[i].item() == pytest.approx(1.0)
        assert delta_y[i].item() == 0.0


def test_gt_boxes_populated_only_for_current_frame(tmp_path):
    dataset, info = _build_dataset(tmp_path, queue_length=4)
    last_token = info["scene_a_sample_tokens"][-1]
    idx = dataset.sample_tokens.index(last_token)
    sample = dataset[idx]

    assert sample["gt_boxes_3d"].shape == (1, 9)
    assert sample["gt_labels_3d"].shape == (1,)
    assert sample["gt_labels_3d"][0].item() == CLASS_TO_ID["car"]


def test_lidar2img_has_one_matrix_per_camera(tmp_path):
    dataset, info = _build_dataset(tmp_path, queue_length=4)
    idx = 0
    sample = dataset[idx]
    lidar2img = sample["img_metas"][-1]["lidar2img"]
    assert len(lidar2img) == 6
    for matrix in lidar2img:
        assert matrix.shape == (4, 4)
        assert np.isfinite(matrix).all()


def test_lidar2img_is_scaled_with_image_resize(tmp_path):
    # Fixture images are natively 8x16 (HxW); resizing to 4x8 halves both axes,
    # so the projection's pixel rows (x and y) must halve while depth is unchanged.
    info = build_synthetic_nuscenes(tmp_path)
    native = BevFormerNuScenesDataset(info["dataroot"], info["version"], queue_length=1, image_size=(8, 16))
    resized = BevFormerNuScenesDataset(info["dataroot"], info["version"], queue_length=1, image_size=(4, 8))
    for m_native, m_resized in zip(native[0]["img_metas"][0]["lidar2img"], resized[0]["img_metas"][0]["lidar2img"]):
        np.testing.assert_allclose(m_resized[0], 0.5 * m_native[0], rtol=1e-6)
        np.testing.assert_allclose(m_resized[1], 0.5 * m_native[1], rtol=1e-6)
        np.testing.assert_allclose(m_resized[2:], m_native[2:], rtol=1e-6)


def test_img_metas_include_image_size(tmp_path):
    dataset, info = _build_dataset(tmp_path, queue_length=4)
    sample = dataset[0]
    for meta in sample["img_metas"]:
        assert meta["image_size"] == (8, 16)


def test_dataroot_expands_user_home(tmp_path, monkeypatch):
    info = build_synthetic_nuscenes(tmp_path / "data")
    monkeypatch.setenv("HOME", str(tmp_path))
    dataset = BevFormerNuScenesDataset("~/data", info["version"], queue_length=1, image_size=(8, 16))
    assert len(dataset) == 7


def test_uint8_images_normalize_to_float32_images(tmp_path):
    info = build_synthetic_nuscenes(tmp_path)
    as_float = BevFormerNuScenesDataset(info["dataroot"], info["version"], queue_length=2, image_size=(8, 16))
    as_uint8 = BevFormerNuScenesDataset(
        info["dataroot"], info["version"], queue_length=2, image_size=(8, 16), image_dtype="uint8"
    )
    raw = as_uint8[0]["imgs"]
    assert raw.dtype == torch.uint8
    torch.testing.assert_close(normalize_images(raw), as_float[0]["imgs"])
