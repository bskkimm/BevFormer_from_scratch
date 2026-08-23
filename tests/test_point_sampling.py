import numpy as np
import torch

from bevformer.models.transformer.point_sampling import project_pillar_points_to_cameras
from bevformer.models.transformer.reference_points import get_pillar_reference_points_3d

PC_RANGE = (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0)


def _identity_lidar2img():
    # Identity extrinsic (lidar frame == camera frame) with a simple pinhole
    # intrinsic (fx=fy=1, cx=cy=0), so image coords before normalization are
    # just (x, y) and depth is z.
    return np.eye(4, dtype=np.float32)


def test_projection_output_shapes():
    ref_points = get_pillar_reference_points_3d(bev_h=2, bev_w=2, pc_range=PC_RANGE, num_points_in_pillar=3)
    img_metas = [
        {"lidar2img": [_identity_lidar2img(), _identity_lidar2img()], "image_size": (20, 20)}
    ]
    ref_cam, mask = project_pillar_points_to_cameras(ref_points, PC_RANGE, img_metas)

    num_cam, batch, num_query, num_points = 2, 1, 4, 3
    assert ref_cam.shape == (num_cam, batch, num_query, num_points, 2)
    assert mask.shape == (num_cam, batch, num_query, num_points)
    assert mask.dtype == torch.bool


def test_point_behind_camera_is_invalid():
    # A single BEV cell (bev_h=bev_w=1) with pillar heights spanning
    # pc_range z in [-2, 2]; since our identity extrinsic maps lidar z
    # directly to camera depth, negative-z pillar points are behind the
    # camera and must be masked invalid.
    ref_points = get_pillar_reference_points_3d(bev_h=1, bev_w=1, pc_range=PC_RANGE, num_points_in_pillar=5)
    img_metas = [{"lidar2img": [_identity_lidar2img()], "image_size": (100, 100)}]
    _, mask = project_pillar_points_to_cameras(ref_points, PC_RANGE, img_metas)

    heights = torch.linspace(PC_RANGE[2], PC_RANGE[5], 5)
    for d, height in enumerate(heights):
        expected_valid = height.item() > 0
        assert mask[0, 0, 0, d].item() == expected_valid


def test_batch_dimension_matches_number_of_img_metas():
    ref_points = get_pillar_reference_points_3d(bev_h=2, bev_w=2, pc_range=PC_RANGE, num_points_in_pillar=2)
    img_metas = [
        {"lidar2img": [_identity_lidar2img()], "image_size": (20, 20)},
        {"lidar2img": [_identity_lidar2img()], "image_size": (20, 20)},
        {"lidar2img": [_identity_lidar2img()], "image_size": (20, 20)},
    ]
    ref_cam, mask = project_pillar_points_to_cameras(ref_points, PC_RANGE, img_metas)
    assert ref_cam.shape[1] == 3
    assert mask.shape[1] == 3
