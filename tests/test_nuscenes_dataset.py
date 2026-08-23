import numpy as np
import torch

from bevformer.data.nuscenes_categories import CLASS_TO_ID
from bevformer.data.nuscenes_dataset import BevFormerNuScenesDataset
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

    delta_translation = can_bus[:, 16]
    delta_yaw = can_bus[:, 17]

    assert delta_translation[0].item() == 0.0
    assert delta_yaw[0].item() == 0.0
    # Ego moves +1.0m in x per sample, so real consecutive frames have a
    # nonzero translation delta and zero yaw delta (no rotation in fixture).
    for i in range(1, 4):
        assert delta_translation[i].item() > 0.0


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
