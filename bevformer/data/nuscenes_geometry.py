"""SE3 pose and quaternion math shared by the nuScenes dataset."""

from __future__ import annotations

import math

import numpy as np


def quaternion_to_rotation_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def pose_to_matrix(rotation, translation) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = quaternion_to_rotation_matrix(rotation)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
    return matrix


def invert_se3(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float32)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def yaw_from_rotation_matrix(rotation: np.ndarray) -> float:
    return math.atan2(rotation[1, 0], rotation[0, 0])
