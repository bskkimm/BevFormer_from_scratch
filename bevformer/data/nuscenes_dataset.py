"""Pure-PyTorch nuScenes dataset returning temporal queues of frames.

Matches official BEVFormer's data contract: each item is a queue of
`queue_length` consecutive frames (multi-camera images + ego pose), with
ground-truth 3D boxes attached only to the current (last) frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from bevformer.data.nuscenes_categories import CLASS_TO_ID, category_to_detection_class
from bevformer.data.nuscenes_splits import SPLIT_VERSION, SPLITS, in_split
from bevformer.data.nuscenes_geometry import (
    invert_se3,
    pose_to_matrix,
    yaw_from_rotation_matrix,
)
from bevformer.data.transforms import resize_and_normalize_image, resize_image_uint8

CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

DEFAULT_QUEUE_LENGTH = 4
DEFAULT_BEV_H = 200
DEFAULT_BEV_W = 200
DEFAULT_PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
DEFAULT_IMAGE_SIZE = (900, 1600)
CAN_BUS_DIM = 18


def _load_table(meta_root: Path, name: str) -> list[dict]:
    with (meta_root / f"{name}.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _index_by(rows: list[dict], key: str = "token") -> dict[str, dict]:
    return {row[key]: row for row in rows}


def _viewpad(intrinsic: np.ndarray) -> np.ndarray:
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = np.asarray(intrinsic, dtype=np.float32)
    return view


class BevFormerNuScenesDataset(Dataset):
    def __init__(
        self,
        dataroot: str | Path,
        version: str = "v1.0-trainval",
        queue_length: int = DEFAULT_QUEUE_LENGTH,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        pc_range: tuple[float, float, float, float, float, float] = DEFAULT_PC_RANGE,
        image_dtype: str = "float32",
        split: str = "all",
    ) -> None:
        """`image_dtype="uint8"` returns raw pixels, to be normalized on the GPU with
        `bevformer.data.transforms.normalize_images` (`move_batch_to_device` does this
        automatically); "float32" returns ImageNet-normalized images.

        `split` selects the official v1.0-trainval scenes: "train" (700 scenes),
        "val" (150), or "all"."""
        if image_dtype not in ("float32", "uint8"):
            raise ValueError(f"Unsupported image_dtype: {image_dtype}")
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split} (expected one of {SPLITS})")
        if split != "all" and version != SPLIT_VERSION:
            raise ValueError(f"split={split!r} is defined for {SPLIT_VERSION} only; use split='all' for {version}")
        self.dataroot = Path(dataroot).expanduser()
        self.meta_root = self.dataroot / version
        self.queue_length = queue_length
        self.image_size = image_size
        self.pc_range = pc_range
        self.image_dtype = image_dtype
        self.split = split

        self.samples = _index_by(_load_table(self.meta_root, "sample"))
        self.scenes = _index_by(_load_table(self.meta_root, "scene"))
        self.sample_data = _load_table(self.meta_root, "sample_data")
        self.ego_poses = _index_by(_load_table(self.meta_root, "ego_pose"))
        self.calibrated_sensors = _index_by(_load_table(self.meta_root, "calibrated_sensor"))
        self.sensors = _index_by(_load_table(self.meta_root, "sensor"))
        self.instances = _index_by(_load_table(self.meta_root, "instance"))
        self.categories = _index_by(_load_table(self.meta_root, "category"))

        sample_annotations = _load_table(self.meta_root, "sample_annotation")
        self.annotations = _index_by(sample_annotations)
        self.annotations_by_sample: dict[str, list[dict]] = {}
        for annotation in sample_annotations:
            self.annotations_by_sample.setdefault(annotation["sample_token"], []).append(annotation)

        self.sample_data_by_sample: dict[str, dict[str, dict]] = {}
        for sd in self.sample_data:
            if not sd["is_key_frame"]:
                continue
            calib = self.calibrated_sensors[sd["calibrated_sensor_token"]]
            channel = self.sensors[calib["sensor_token"]]["channel"]
            self.sample_data_by_sample.setdefault(sd["sample_token"], {})[channel] = sd

        self.sample_tokens = self._collect_sample_tokens_in_scene_order()

    def _collect_sample_tokens_in_scene_order(self) -> list[str]:
        tokens: list[str] = []
        for scene in self.scenes.values():
            if not in_split(scene["name"], self.split):
                continue
            token = scene["first_sample_token"]
            while token:
                tokens.append(token)
                token = self.samples[token]["next"]
        return tokens

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def _build_queue_tokens(self, current_token: str) -> list[str]:
        chain = [current_token]
        token = current_token
        while len(chain) < self.queue_length:
            prev_token = self.samples[token]["prev"]
            if not prev_token:
                break
            chain.append(prev_token)
            token = prev_token
        chain.reverse()  # oldest -> newest so far
        if len(chain) < self.queue_length:
            pad_count = self.queue_length - len(chain)
            chain = [chain[0]] * pad_count + chain
        return chain

    def _reference_pose(self, sample_token: str) -> tuple[np.ndarray, dict, dict]:
        lidar_sd = self.sample_data_by_sample[sample_token]["LIDAR_TOP"]
        lidar_calib = self.calibrated_sensors[lidar_sd["calibrated_sensor_token"]]
        lidar_ego_pose = self.ego_poses[lidar_sd["ego_pose_token"]]

        lidar2ego = pose_to_matrix(lidar_calib["rotation"], lidar_calib["translation"])
        ego2global = pose_to_matrix(lidar_ego_pose["rotation"], lidar_ego_pose["translation"])
        ref_lidar2global = ego2global @ lidar2ego
        return ref_lidar2global, lidar_ego_pose, lidar_calib

    def _load_frame(self, sample_token: str) -> dict:
        ref_lidar2global, ego_pose, _lidar_calib = self._reference_pose(sample_token)

        camera_records = self.sample_data_by_sample[sample_token]
        imgs = []
        lidar2img = []
        for cam in CAMERA_NAMES:
            sd = camera_records[cam]
            image = Image.open(self.dataroot / sd["filename"]).convert("RGB")
            native_w, native_h = image.size
            load_image = resize_image_uint8 if self.image_dtype == "uint8" else resize_and_normalize_image
            imgs.append(load_image(image, image_size=self.image_size))
            # Intrinsics project into native pixels; rescale them to the resized image.
            img_scale = np.diag([self.image_size[1] / native_w, self.image_size[0] / native_h, 1.0, 1.0])

            calib = self.calibrated_sensors[sd["calibrated_sensor_token"]]
            cam_ego_pose = self.ego_poses[sd["ego_pose_token"]]

            cam2ego = pose_to_matrix(calib["rotation"], calib["translation"])
            ego2global_cam = pose_to_matrix(cam_ego_pose["rotation"], cam_ego_pose["translation"])
            cam2global = ego2global_cam @ cam2ego
            global2cam = invert_se3(cam2global)
            lidar2cam = global2cam @ ref_lidar2global
            lidar2img.append((img_scale @ _viewpad(calib["camera_intrinsic"]) @ lidar2cam).astype(np.float32))

        boxes, labels = self._load_boxes(sample_token, ref_lidar2global)

        return {
            "imgs": torch.stack(imgs, dim=0),
            "sample_token": sample_token,
            "scene_token": self.samples[sample_token]["scene_token"],
            "lidar2img": lidar2img,
            "ego2global_translation": np.asarray(ego_pose["translation"], dtype=np.float32),
            "lidar2global": ref_lidar2global,
            "ego2global_rotation": ego_pose["rotation"],
            "gt_boxes_3d": boxes,
            "gt_labels_3d": labels,
        }

    def _global_velocity(self, annotation: dict, max_time_diff: float = 1.5) -> np.ndarray | None:
        """nuscenes-devkit's box_velocity: central difference over the instance's
        previous/next annotations (one-sided at track ends), None if the track has
        no neighbor or the neighbors are too far apart in time."""
        has_prev, has_next = bool(annotation["prev"]), bool(annotation["next"])
        if not has_prev and not has_next:
            return None
        first = self.annotations[annotation["prev"]] if has_prev else annotation
        last = self.annotations[annotation["next"]] if has_next else annotation
        time_diff = 1e-6 * (
            self.samples[last["sample_token"]]["timestamp"] - self.samples[first["sample_token"]]["timestamp"]
        )
        if time_diff <= 0 or time_diff > (2 * max_time_diff if has_prev and has_next else max_time_diff):
            return None
        return (np.asarray(last["translation"]) - np.asarray(first["translation"])) / time_diff

    def _load_boxes(self, sample_token: str, ref_lidar2global: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        global2ref_lidar = invert_se3(ref_lidar2global)
        boxes = []
        labels = []
        for annotation in self.annotations_by_sample.get(sample_token, []):
            instance = self.instances[annotation["instance_token"]]
            category_name = self.categories[instance["category_token"]]["name"]
            detection_class = category_to_detection_class(category_name)
            if detection_class is None:
                continue
            # Official use_valid_flag: drop objects no lidar or radar return hit.
            if annotation["num_lidar_pts"] + annotation["num_radar_pts"] <= 0:
                continue

            center_global = np.array([*annotation["translation"], 1.0], dtype=np.float32)
            center_ref = global2ref_lidar @ center_global
            # Official ObjectRangeFilter: BEV center strictly inside pc_range x/y.
            x_min, y_min, _, x_max, y_max, _ = self.pc_range
            if not (x_min < center_ref[0] < x_max and y_min < center_ref[1] < y_max):
                continue

            box_rotation_global = pose_to_matrix(annotation["rotation"], (0.0, 0.0, 0.0))[:3, :3]
            box_rotation_ref = global2ref_lidar[:3, :3] @ box_rotation_global
            yaw_ref = yaw_from_rotation_matrix(box_rotation_ref)

            width, length, height = annotation["size"]
            # Official (mmdet3d converter): the global xy velocity rotated into the
            # LIDAR_TOP axes; undefined velocities become 0.
            velocity_ref = np.zeros(2)
            velocity_global = self._global_velocity(annotation)
            if velocity_global is not None:
                velocity_ref = (global2ref_lidar[:3, :3] @ np.array([velocity_global[0], velocity_global[1], 0.0]))[:2]
            boxes.append(
                [
                    center_ref[0],
                    center_ref[1],
                    center_ref[2],
                    width,
                    length,
                    height,
                    yaw_ref,
                    velocity_ref[0],
                    velocity_ref[1],
                ]
            )
            labels.append(CLASS_TO_ID[detection_class])

        if not boxes:
            return torch.zeros((0, 9), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def _build_can_bus(self, frames: list[dict], queue_tokens: list[str]) -> torch.Tensor:
        can_bus = torch.zeros((self.queue_length, CAN_BUS_DIM), dtype=torch.float32)
        for i, frame in enumerate(frames):
            can_bus[i, 0:3] = torch.from_numpy(frame["ego2global_translation"])
            can_bus[i, 3:7] = torch.tensor(frame["ego2global_rotation"], dtype=torch.float32)
            # Indices 7:16 (accel, rotation_rate, velocity) are zero-filled:
            # the CAN bus expansion tables are not part of the standard
            # v1.0-trainval metadata this dataset reads. Indices 16:18 hold
            # this frame's LIDAR_TOP origin expressed in the previous queue
            # frame's LIDAR_TOP (= BEV) coordinates, in meters: the translation
            # the encoder's temporal warp applies. (A global-frame delta would
            # point the wrong way whenever the car is not heading along global
            # x.) The yaw delta is not stored: it is recomputed from the
            # absolute rotation quaternions at indices 3:7 of consecutive frames.
            if i > 0 and queue_tokens[i] != queue_tokens[i - 1]:
                current_in_prev = invert_se3(frames[i - 1]["lidar2global"]) @ frame["lidar2global"]
                can_bus[i, 16:18] = torch.from_numpy(current_in_prev[:2, 3].astype(np.float32))
        return can_bus

    def __getitem__(self, idx: int) -> dict:
        current_token = self.sample_tokens[idx]
        queue_tokens = self._build_queue_tokens(current_token)
        frames = [self._load_frame(token) for token in queue_tokens]

        imgs = torch.stack([frame["imgs"] for frame in frames], dim=0)
        img_metas = []
        for i, frame in enumerate(frames):
            prev_bev_exists = i > 0 and queue_tokens[i] != queue_tokens[i - 1]
            img_metas.append(
                {
                    "sample_token": frame["sample_token"],
                    "scene_token": frame["scene_token"],
                    "lidar2img": frame["lidar2img"],
                    "image_size": self.image_size,
                    "prev_bev_exists": prev_bev_exists,
                }
            )
        can_bus = self._build_can_bus(frames, queue_tokens)

        current = frames[-1]
        return {
            "imgs": imgs,
            "img_metas": img_metas,
            "can_bus": can_bus,
            "gt_boxes_3d": current["gt_boxes_3d"],
            "gt_labels_3d": current["gt_labels_3d"],
        }
