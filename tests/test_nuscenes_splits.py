import re

import pytest

from bevformer.data.nuscenes_splits import VAL_SCENES, in_split


def test_val_split_has_the_150_official_scenes():
    assert len(VAL_SCENES) == 150
    assert all(re.fullmatch(r"scene-\d{4}", name) for name in VAL_SCENES)


def test_in_split_partitions_scenes():
    assert in_split("scene-0003", "val") and not in_split("scene-0003", "train")
    assert in_split("scene-0001", "train") and not in_split("scene-0001", "val")
    assert in_split("scene-0003", "all") and in_split("scene-0001", "all")


def test_unknown_split_raises():
    with pytest.raises(ValueError):
        in_split("scene-0001", "trainval")
