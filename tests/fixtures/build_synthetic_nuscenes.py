"""Builds a tiny synthetic nuScenes-style metadata + image tree for tests.

Mirrors the real nuScenes v1.0 JSON table schema closely enough for
BevFormerNuScenesDataset to load it, without requiring the real dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

IDENTITY_ROTATION = [1.0, 0.0, 0.0, 0.0]
IMAGE_SIZE = (8, 16)  # (height, width), kept tiny for fast tests


def _token(prefix: str, index: int) -> str:
    return f"{prefix}_{index:03d}"


def build_synthetic_nuscenes(root: Path, version: str = "v1.0-trainval") -> dict:
    """Writes a synthetic dataset tree under `root` and returns bookkeeping info.

    Layout: two scenes.
      - "scene_a" has 5 samples (enough for a full queue_length=4 with no padding
        on the last sample).
      - "scene_b" has 2 samples (forces left-padding for queue_length=4).

    Each sample's ego vehicle moves +1.0m in x per step, so consecutive real
    frames have a nonzero translation/yaw delta.
    """
    meta_root = root / version
    meta_root.mkdir(parents=True, exist_ok=True)
    samples_root = root / "samples"
    for cam in CAMERA_NAMES:
        (samples_root / cam).mkdir(parents=True, exist_ok=True)

    category_rows = [
        {"token": "cat_car", "name": "vehicle.car", "description": ""},
        {"token": "cat_ped", "name": "human.pedestrian.adult", "description": ""},
    ]
    instance_rows = [
        {
            "token": "instance_car_0",
            "category_token": "cat_car",
            "nbr_annotations": 0,
            "first_annotation_token": "",
            "last_annotation_token": "",
        }
    ]
    sensor_rows = [{"token": f"sensor_{cam}", "channel": cam, "modality": "camera"} for cam in CAMERA_NAMES]

    calibrated_sensor_rows = []
    for cam_index, cam in enumerate(CAMERA_NAMES):
        calibrated_sensor_rows.append(
            {
                "token": f"calib_{cam}",
                "sensor_token": f"sensor_{cam}",
                # small distinct translation per camera so lidar2img differs per camera
                "translation": [0.1 * cam_index, 0.0, 1.5],
                "rotation": IDENTITY_ROTATION,
                "camera_intrinsic": [
                    [100.0, 0.0, 8.0],
                    [0.0, 100.0, 4.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        )
    # LIDAR_TOP calibration used as the reference sensor frame.
    calibrated_sensor_rows.append(
        {
            "token": "calib_LIDAR_TOP",
            "sensor_token": "sensor_LIDAR_TOP",
            "translation": [0.0, 0.0, 1.8],
            "rotation": IDENTITY_ROTATION,
            "camera_intrinsic": [],
        }
    )
    sensor_rows.append({"token": "sensor_LIDAR_TOP", "channel": "LIDAR_TOP", "modality": "lidar"})

    scenes = {}
    samples = []
    sample_data_rows = []
    ego_pose_rows = []
    sample_annotation_rows = []

    global_sample_index = 0
    for scene_name, num_samples in (("scene_a", 5), ("scene_b", 2)):
        sample_tokens = [_token(f"sample_{scene_name}", i) for i in range(num_samples)]
        for local_index, sample_token in enumerate(sample_tokens):
            timestamp = global_sample_index * 500_000  # microseconds, 0.5s apart
            ego_translation = [float(global_sample_index), 0.0, 0.0]
            ego_pose_token = f"egopose_{sample_token}"
            ego_pose_rows.append(
                {
                    "token": ego_pose_token,
                    "timestamp": timestamp,
                    "rotation": IDENTITY_ROTATION,
                    "translation": ego_translation,
                }
            )

            prev_token = sample_tokens[local_index - 1] if local_index > 0 else ""
            next_token = sample_tokens[local_index + 1] if local_index < num_samples - 1 else ""
            samples.append(
                {
                    "token": sample_token,
                    "timestamp": timestamp,
                    "scene_token": scene_name,
                    "prev": prev_token,
                    "next": next_token,
                }
            )

            for cam in CAMERA_NAMES:
                filename = f"samples/{cam}/{sample_token}.jpg"
                image = Image.fromarray(
                    np.full((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), 128, dtype=np.uint8)
                )
                image.save(root / filename)
                sample_data_rows.append(
                    {
                        "token": f"sd_{cam}_{sample_token}",
                        "sample_token": sample_token,
                        "ego_pose_token": ego_pose_token,
                        "calibrated_sensor_token": f"calib_{cam}",
                        "filename": filename,
                        "fileformat": "jpg",
                        "is_key_frame": True,
                        "height": IMAGE_SIZE[0],
                        "width": IMAGE_SIZE[1],
                        "timestamp": timestamp,
                    }
                )
            sample_data_rows.append(
                {
                    "token": f"sd_LIDAR_TOP_{sample_token}",
                    "sample_token": sample_token,
                    "ego_pose_token": ego_pose_token,
                    "calibrated_sensor_token": "calib_LIDAR_TOP",
                    "filename": "",
                    "fileformat": "pcd",
                    "is_key_frame": True,
                    "height": 0,
                    "width": 0,
                    "timestamp": timestamp,
                }
            )

            # One car annotation per sample, placed 10m ahead of the ego vehicle.
            sample_annotation_rows.append(
                {
                    "token": f"ann_{sample_token}",
                    "sample_token": sample_token,
                    "instance_token": "instance_car_0",
                    "translation": [ego_translation[0] + 10.0, 5.0, 1.0],
                    "size": [2.0, 4.5, 1.6],
                    "rotation": IDENTITY_ROTATION,
                    "num_lidar_pts": 10,
                    "num_radar_pts": 0,
                    "prev": "",
                    "next": "",
                }
            )

            global_sample_index += 1

        scenes[scene_name] = {
            "token": scene_name,
            "name": scene_name,
            "log_token": "log_0",
            "nbr_samples": num_samples,
            "first_sample_token": sample_tokens[0],
            "last_sample_token": sample_tokens[-1],
            "description": "",
        }

    tables = {
        "category": category_rows,
        "instance": instance_rows,
        "sensor": sensor_rows,
        "calibrated_sensor": calibrated_sensor_rows,
        "scene": list(scenes.values()),
        "sample": samples,
        "sample_data": sample_data_rows,
        "ego_pose": ego_pose_rows,
        "sample_annotation": sample_annotation_rows,
    }
    for name, rows in tables.items():
        with (meta_root / f"{name}.json").open("w", encoding="utf-8") as handle:
            json.dump(rows, handle)

    return {
        "dataroot": root,
        "version": version,
        "scene_a_sample_tokens": [_token("sample_scene_a", i) for i in range(5)],
        "scene_b_sample_tokens": [_token("sample_scene_b", i) for i in range(2)],
    }
