import json
import shutil

import numpy as np
import pytest
import torch

from bevformer.data.nuscenes_dataset import BevFormerNuScenesDataset
from bevformer.scripts.make_demo_videos import (
    HEIGHT,
    WIDTH,
    open_encoder,
    pick_scenes,
    render_frame,
    scenes_by_map,
    write_index,
)
from tests.fixtures.build_synthetic_nuscenes import build_synthetic_nuscenes


def _scenes(count, frames=40):
    return [(f"scene-{i}: d", list(range(i * frames, (i + 1) * frames))) for i in range(count)]


def _scene_numbers(chosen):
    return [int(label.split("-")[1].split(":")[0]) for label, _ in chosen]


def test_pick_scenes_spreads_over_the_map_and_stops_at_the_frame_budget():
    chosen = pick_scenes(_scenes(20), frames=120)
    assert _scene_numbers(chosen) == [0, 5, 10]  # evenly spread, in order, 3 x 40 = 120 frames


def test_pick_scenes_skips_truncated_scenes_and_never_repeats_a_scene():
    scenes = _scenes(2) + [("scene-9: short", list(range(10)))]
    chosen = pick_scenes(scenes, frames=240)  # more frames wanted than the map has
    assert _scene_numbers(chosen) == [0, 1]


def test_scenes_by_map_groups_scenes_by_log_location(tmp_path):
    info = build_synthetic_nuscenes(tmp_path)
    (tmp_path / info["version"] / "log.json").write_text(json.dumps([{"token": "log_0", "location": "test-map"}]))
    dataset = BevFormerNuScenesDataset(info["dataroot"], info["version"], queue_length=2, image_size=(8, 16))

    plan = scenes_by_map(dataset, tmp_path, info["version"])

    assert list(plan) == ["test-map"]
    labels = [label for label, _ in plan["test-map"]]
    assert labels == ["scene_a: ", "scene_b: "]
    assert [len(indices) for _, indices in plan["test-map"]] == [5, 2]
    assert sorted(i for _, indices in plan["test-map"] for i in indices) == list(range(7))


def _front_camera_lidar2img():
    # Camera looking along +y (lidar forward): u = f x / y + cx, v = -f z / y + cy.
    lidar2cam = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])
    intrinsic = np.array([[100.0, 0, 80.0, 0], [0, 100.0, 45.0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    return intrinsic @ lidar2cam


def test_render_frame_draws_projected_predictions_on_the_camera_only():
    box = torch.tensor([[0.0, 10.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]])
    camera = np.zeros((90, 160, 3), dtype=np.float32)
    features = torch.zeros(10, 10, 4)

    with_boxes = render_frame(camera, box, box, _front_camera_lidar2img(), features, "map  scene", "t = 0.0 s")
    without = render_frame(camera, box[:0], box[:0], _front_camera_lidar2img(), features, "map  scene", "t = 0.0 s")

    assert with_boxes.shape == (HEIGHT, WIDTH, 3) and with_boxes.dtype == np.uint8
    camera_panel = slice(0, 1280)  # the left 1280 px: camera only, so any orange there is a projected box
    orange = lambda image: ((image[:, camera_panel, 0] > 200) & (image[:, camera_panel, 2] < 80)).sum()  # noqa: E731
    assert orange(with_boxes) > 100
    assert orange(without) == 0


def test_write_index_lists_every_video_and_escapes_text(tmp_path):
    manifest = [{"map": "boston-seaport", "file": "boston-seaport.mp4", "seconds": 120.0, "scenes": ["scene-1: <rain> & fog"]}]
    page = write_index(tmp_path, manifest, "Demo (epoch 24)").read_text()
    assert '<video src="boston-seaport.mp4"' in page
    assert "scene-1: &lt;rain&gt; &amp; fog" in page
    assert "<title>Demo (epoch 24)</title>" in page


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_encoder_turns_raw_frames_into_an_mp4(tmp_path):
    path = tmp_path / "clip.mp4"
    encoder = open_encoder(path)
    for _ in range(2):
        encoder.stdin.write(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8).tobytes())
    encoder.stdin.close()
    assert encoder.wait() == 0
    assert path.stat().st_size > 0
