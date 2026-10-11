from types import SimpleNamespace

import pytest
import torch

from bevformer.engine.validation import evaluate_loss, log_result, select_indices
from bevformer.scripts.watch_validation import pending_checkpoints


def test_selection_spreads_across_scenes_and_is_repeatable():
    tokens = [str(i) for i in range(30)]
    dataset = SimpleNamespace(
        sample_tokens=tokens,
        samples={t: {"scene_token": str(int(t) // 10)} for t in tokens},
    )
    chosen = select_indices(dataset, 3)
    assert len(set(chosen)) == 3
    assert {i // 10 for i in chosen} == {0, 1, 2}
    assert chosen == select_indices(dataset, 3)
    assert len(select_indices(dataset, 100)) == 30
    with pytest.raises(ValueError):
        select_indices(dataset, 0)


class Model(torch.nn.Module):
    def forward(self, imgs, metas, bus):
        assert not self.training and not torch.is_grad_enabled()
        return {"cls_scores": imgs, "bbox_preds": imgs}


class Loss:
    def loss_by_feat(self, cls, bbox, boxes, labels):
        return {"loss_cls": cls.mean(), "loss_bbox": cls.mean() * 2}


def batch(values):
    return dict(
        imgs=torch.tensor(values, dtype=torch.float32),
        img_metas=[],
        can_bus=torch.zeros(len(values), 18),
        gt_boxes_3d=[],
        gt_labels_3d=[],
    )


def test_loss_is_sample_weighted_and_model_mode_restored():
    model = Model().train()
    metrics = evaluate_loss(
        model, Loss(), [batch([1.0, 3.0]), batch([8.0])], torch.device("cpu")
    )
    assert metrics["loss"] == pytest.approx(12.0)
    assert metrics["loss_cls"] == pytest.approx(4.0)
    assert metrics["samples"] == 3
    assert model.training


def test_nonfinite_loss_is_rejected():
    with pytest.raises(RuntimeError, match="non-finite"):
        evaluate_loss(Model(), Loss(), [batch([float("nan")])], torch.device("cpu"))


def test_completed_epochs_and_partial_files_are_skipped(tmp_path):
    for name in [
        "epoch_01.pth",
        "epoch_02.pth",
        "epoch_10.pth",
        "epoch_03.pth.tmp",
        "latest.pth",
    ]:
        (tmp_path / name).touch()
    assert [p.name for p in pending_checkpoints(tmp_path, {1})] == [
        "epoch_10.pth",
        "epoch_02.pth",
    ]


def test_logging_preserves_run_status_and_supports_restart(tmp_path):
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/tracking.db")
    experiment = client.create_experiment("validation-test")
    run_id = client.create_run(experiment).info.run_id
    log_result(client, run_id, 2, {"loss": 12.0, "loss_cls": 4.0, "samples": 3}, 1.5)
    assert client.get_run(run_id).info.status == "RUNNING"
    history = client.get_metric_history(run_id, "val/subset_loss")
    assert [(m.step, m.value) for m in history] == [(2, 12.0)]


def test_validation_cache_roundtrip_preserves_temporal_samples(tmp_path, monkeypatch):
    import numpy as np

    from bevformer.data.nuscenes_dataset import BevFormerNuScenesDataset
    from tests.fixtures.build_synthetic_nuscenes import build_synthetic_nuscenes
    from bevformer.scripts.watch_validation import load_cache, prepare_cache

    info = build_synthetic_nuscenes(
        tmp_path / "data", scene_names={"scene_b": "scene-0003"}
    )
    config = dict(
        dataroot=str(info["dataroot"]),
        version="v1.0-trainval",
        queue_length=4,
        image_height=8,
        image_width=16,
    )
    args = SimpleNamespace(run_dir=tmp_path, samples=2, seed=0)
    original = BevFormerNuScenesDataset(
        info["dataroot"],
        split="val",
        queue_length=4,
        image_size=(8, 16),
        image_dtype="uint8",
    )
    expected = [original[i] for i in select_indices(original, 2)]
    prepare_cache(args, config)

    def forbidden(*a, **kw):
        raise AssertionError("Cached validation must not load the full metadata tables")

    monkeypatch.setattr(BevFormerNuScenesDataset, "__init__", forbidden)
    dataset, indices = load_cache(args, config)
    for old, index in zip(expected, indices):
        new = dataset[index]
        for key in ["imgs", "can_bus", "gt_boxes_3d", "gt_labels_3d"]:
            torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
        for old_meta, new_meta in zip(old["img_metas"], new["img_metas"]):
            assert old_meta["sample_token"] == new_meta["sample_token"]
            np.testing.assert_array_equal(old_meta["lidar2img"], new_meta["lidar2img"])
    with pytest.raises(ValueError, match="settings"):
        load_cache(SimpleNamespace(run_dir=tmp_path, samples=1, seed=0), config)


def test_ram_guard_parses_available_memory(tmp_path):
    from bevformer.scripts.watch_validation import available_memory_gib

    path = tmp_path / "meminfo"
    path.write_text("MemFree: 100 kB\nMemAvailable: 8388608 kB\n")
    assert available_memory_gib(path) == 8.0
