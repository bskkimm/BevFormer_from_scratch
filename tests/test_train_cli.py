"""End-to-end check of train.py: training, MLflow logging, checkpoints, and resume."""

import mlflow
import pytest
from mlflow.tracking import MlflowClient

import train
from tests.fixtures.build_synthetic_nuscenes import build_synthetic_nuscenes


def _argv(dataroot, epochs, extra=()):
    return [
        "--dataroot", str(dataroot), "--queue-length", "2", "--image-height", "64", "--image-width", "64",
        "--num-workers", "0", "--no-pin-memory", "--device", "cpu", "--amp", "none",
        "--no-compile-backbone", "--no-fused-adamw",
        "--backbone-variant", "resnet50", "--no-pretrained-backbone", "--embed-dims", "16",
        "--bev-h", "4", "--bev-w", "4", "--num-queries", "6", "--num-encoder-layers", "1",
        "--num-decoder-layers", "2", "--num-heads", "2", "--num-points-in-pillar", "2",
        "--epochs", str(epochs), "--accumulation-steps", "2", "--warmup-iters", "2", "--log-every-updates", "1",
        "--vis-every-epochs", "1", "--vis-split", "val", "--vis-sample-index", "0",
        "--output-dir", "run", "--mlflow", "--mlflow-tracking-uri", "sqlite:///mlflow.db",
        *extra,
    ]


@pytest.mark.filterwarnings("ignore")
def test_train_logs_progress_visualizes_checkpoints_and_resumes(tmp_path, monkeypatch):
    info = build_synthetic_nuscenes(tmp_path / "data", scene_names={"scene_b": "scene-0003"})  # scene_b is val
    monkeypatch.chdir(tmp_path)  # keeps mlflow.db / mlruns / run inside tmp_path

    train.main(_argv(info["dataroot"], epochs=2))
    run_dir = tmp_path / "run"
    for name in ("checkpoints/latest.pth", "checkpoints/epoch_01.pth", "checkpoints/epoch_02.pth", "final.pth",
                 "bev_features/epoch_00.png", "bev_features/epoch_02.png", "config.json"):
        assert (run_dir / name).exists(), name

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    (run,) = client.search_runs(client.get_experiment_by_name("bevformer-training").experiment_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["accumulation_steps"] == "2" and run.data.params["effective_batch_size"] == "2"
    for metric in ("train/loss", "train/lr", "progress/eta_hours", "progress/avg_epoch_hours", "epoch/loss"):
        assert metric in run.data.metrics, metric
    description = run.data.tags["mlflow.note.content"]
    assert "Progress" in description and "ETA" in description and "Elapsed" in description
    artifacts = {a.path for a in client.list_artifacts(run.info.run_id, "bev_features")}
    assert artifacts == {f"bev_features/epoch_{e:02d}.png" for e in (0, 1, 2)}
    # train split = scene_a only (5 samples) -> 3 updates/epoch with accumulation 2.
    assert client.get_metric_history(run.info.run_id, "epoch/global_update")[-1].value == 6

    train.main(_argv(info["dataroot"], epochs=3, extra=["--resume", "run/checkpoints/latest.pth"]))
    (resumed,) = client.search_runs(client.get_experiment_by_name("bevformer-training").experiment_id)
    assert resumed.info.run_id == run.info.run_id  # same MLflow run continued
    assert (run_dir / "checkpoints/epoch_03.pth").exists()
    epochs_logged = [m.value for m in client.get_metric_history(run.info.run_id, "epoch/epoch")]
    assert epochs_logged == [1.0, 2.0, 3.0]
    assert client.get_metric_history(run.info.run_id, "epoch/global_update")[-1].value == 9
