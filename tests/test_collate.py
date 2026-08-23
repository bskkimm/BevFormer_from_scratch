import torch

from bevformer.data.collate import collate_fn


def _make_sample(num_boxes: int, queue_length: int = 4, num_cams: int = 6, h: int = 8, w: int = 16):
    return {
        "imgs": torch.randn(queue_length, num_cams, 3, h, w),
        "img_metas": [{"sample_token": f"tok_{i}"} for i in range(queue_length)],
        "can_bus": torch.randn(queue_length, 18),
        "gt_boxes_3d": torch.randn(num_boxes, 9),
        "gt_labels_3d": torch.randint(0, 10, (num_boxes,)),
    }


def test_collate_stacks_fixed_shape_fields():
    batch = [_make_sample(num_boxes=2), _make_sample(num_boxes=3)]
    collated = collate_fn(batch)

    assert collated["imgs"].shape == (2, 4, 6, 3, 8, 16)
    assert collated["can_bus"].shape == (2, 4, 18)


def test_collate_keeps_img_metas_and_boxes_ragged_per_sample():
    batch = [_make_sample(num_boxes=2), _make_sample(num_boxes=3)]
    collated = collate_fn(batch)

    assert isinstance(collated["img_metas"], list)
    assert len(collated["img_metas"]) == 2
    assert len(collated["img_metas"][0]) == 4

    assert isinstance(collated["gt_boxes_3d"], list)
    assert collated["gt_boxes_3d"][0].shape == (2, 9)
    assert collated["gt_boxes_3d"][1].shape == (3, 9)

    assert isinstance(collated["gt_labels_3d"], list)
    assert collated["gt_labels_3d"][0].shape == (2,)
    assert collated["gt_labels_3d"][1].shape == (3,)
