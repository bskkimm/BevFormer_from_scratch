import pytest

from bevformer.engine.progress import TrainingProgress, format_duration


class _Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def test_format_duration():
    assert format_duration(59) == "1m"
    assert format_duration(3 * 3600 + 5 * 60) == "3h 05m"
    assert format_duration(2 * 86400 + 4 * 3600) == "2d 4h 00m"


def test_snapshot_estimates_epoch_time_and_eta_from_measured_rate():
    clock = _Clock()
    progress = TrainingProgress(total_epochs=4, micro_batches_per_epoch=100, clock=clock)
    clock.now += 1800  # 50 micro-batches in 30 min -> 1 h per epoch
    snap = progress.snapshot(epoch=0, micro_batches_done=50)
    assert snap["epoch"] == pytest.approx(0.5)
    assert snap["percent"] == pytest.approx(12.5)
    assert snap["elapsed_hours"] == pytest.approx(0.5)
    assert snap["avg_epoch_hours"] == pytest.approx(1.0)
    assert snap["eta_hours"] == pytest.approx(3.5)


def test_resumed_run_counts_prior_time_and_only_measures_its_own_rate():
    clock = _Clock()
    progress = TrainingProgress(
        total_epochs=4, micro_batches_per_epoch=100, start_epoch=2, elapsed_offset_s=7200, clock=clock
    )
    clock.now += 900  # 25 micro-batches of epoch 2 in 15 min -> 1 h per epoch
    snap = progress.snapshot(epoch=2, micro_batches_done=25)
    assert snap["percent"] == pytest.approx(56.25)
    assert snap["elapsed_hours"] == pytest.approx(2.25)
    assert snap["eta_hours"] == pytest.approx(1.75)


def test_description_mentions_every_progress_field():
    clock = _Clock()
    progress = TrainingProgress(total_epochs=24, micro_batches_per_epoch=100, clock=clock)
    clock.now += 600
    text = progress.description(progress.snapshot(0, 10), extra_lines=["**Loss:** 12.3"])
    for fragment in ("epoch 0.10 / 24", "Elapsed", "Avg epoch time", "ETA", "finishes", "Loss"):
        assert fragment in text
