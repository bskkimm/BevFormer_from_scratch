from types import SimpleNamespace

import numpy as np
from PIL import Image

from bevformer.scripts.plot_training_curves import latest_per_step, load_series, moving_average, render


def _metric(step, timestamp, value):
    return SimpleNamespace(step=step, timestamp=timestamp, value=value)


def test_latest_per_step_keeps_the_newest_entry_of_a_re_logged_step():
    history = [_metric(1, 10, 5.0), _metric(2, 11, 4.0), _metric(2, 99, 3.0), _metric(1, 5, 9.0)]
    steps, values = latest_per_step(history)
    assert steps.tolist() == [1, 2]
    assert values.tolist() == [5.0, 3.0]


def test_moving_average_is_valid_mode_and_tolerates_short_series():
    np.testing.assert_allclose(moving_average(np.array([1.0, 3.0, 5.0, 7.0]), 2), [2.0, 4.0, 6.0])
    assert len(moving_average(np.array([1.0, 2.0]), 50)) == 1


def _series(updates=200, epochs=4):
    steps = np.arange(10, 10 * updates + 1, 10)
    return {
        "train/loss": (steps, 10.0 / (1.0 + steps / 300.0)),
        "progress/epoch": (steps, steps / (10.0 * updates / epochs)),
        "epoch/loss": (np.arange(1, epochs + 1), np.array([6.0, 4.0, 3.0, 2.5])),
        "val/subset_loss": (np.arange(1, epochs + 1), np.array([7.0, 6.0, 5.5, 5.8])),
    }


def test_render_writes_a_two_panel_png_even_without_validation_points(tmp_path):
    path = render(_series(), tmp_path / "curves.png", title="run")
    with Image.open(path) as image:
        assert image.size[0] > 1500 and image.size[1] > 500
    series = _series()
    series["val/subset_loss"] = (np.array([], dtype=np.int64), np.array([]))
    assert render(series, tmp_path / "no_val.png").exists()


def test_load_series_reads_mlflow_history_and_drops_superseded_entries(tmp_path):
    from mlflow.entities import Metric
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/tracking.db")
    run_id = client.create_run(client.create_experiment("curves-test")).info.run_id
    client.log_batch(
        run_id,
        metrics=[
            Metric("train/loss", 9.0, 1_000, 10),  # logged by the interrupted attempt
            Metric("train/loss", 5.0, 2_000, 10),  # re-logged after the resume
            Metric("train/loss", 4.0, 2_001, 20),
        ],
    )
    steps, values = load_series(client, run_id)["train/loss"]
    assert steps.tolist() == [10, 20]
    assert values.tolist() == [5.0, 4.0]
    assert len(load_series(client, run_id)["val/subset_loss"][0]) == 0
