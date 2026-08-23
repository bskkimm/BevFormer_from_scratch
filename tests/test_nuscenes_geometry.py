import numpy as np

from bevformer.data.nuscenes_geometry import (
    invert_se3,
    pose_to_matrix,
    quaternion_to_rotation_matrix,
    yaw_from_rotation_matrix,
)


def test_identity_quaternion_gives_identity_rotation():
    rotation = quaternion_to_rotation_matrix((1.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(rotation, np.eye(3, dtype=np.float32), atol=1e-6)


def test_pose_to_matrix_places_rotation_and_translation():
    matrix = pose_to_matrix((1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    np.testing.assert_allclose(matrix[:3, :3], np.eye(3, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_invert_se3_round_trip_composes_to_identity():
    # 90 degree rotation about Z: quaternion (w, x, y, z) = (cos45, 0, 0, sin45)
    c = float(np.cos(np.pi / 4))
    s = float(np.sin(np.pi / 4))
    matrix = pose_to_matrix((c, 0.0, 0.0, s), (5.0, -2.0, 0.5))
    inverse = invert_se3(matrix)
    composed = matrix @ inverse
    np.testing.assert_allclose(composed, np.eye(4, dtype=np.float32), atol=1e-5)


def test_yaw_from_rotation_matrix_recovers_known_angle():
    angle = np.pi / 6
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    recovered = yaw_from_rotation_matrix(rotation)
    assert abs(recovered - angle) < 1e-5
