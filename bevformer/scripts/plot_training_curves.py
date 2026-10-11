"""Plot the training and validation curves of an MLflow run as one PNG (README / run reports).

    python bevformer/scripts/plot_training_curves.py --run-id <mlflow run id> --out docs/assets/training_curves.png

When a resumed run re-logged a step (an interrupted epoch is trained again), the newest
entry per step wins, so the plot is the same whether or not the duplicates were cleaned up.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

KEYS = ("train/loss", "progress/epoch", "epoch/loss", "val/subset_loss")
TRAIN_COLOR, VAL_COLOR = "#1f6fb4", "#d9621a"


def latest_per_step(history) -> tuple[np.ndarray, np.ndarray]:
    """(steps, values) sorted by step; the entry with the newest timestamp wins when a step repeats."""
    newest = {}
    for metric in history:
        if metric.step not in newest or metric.timestamp >= newest[metric.step].timestamp:
            newest[metric.step] = metric
    steps = np.array(sorted(newest), dtype=np.int64)
    return steps, np.array([newest[step].value for step in steps], dtype=np.float64)


def load_series(client, run_id: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {key: latest_per_step(client.get_metric_history(run_id, key)) for key in KEYS}


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(window, len(values)))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def render(series: dict[str, tuple[np.ndarray, np.ndarray]], out_path: str | Path, title: str = "") -> Path:
    """Left: loss per logged update against fractional epoch. Right: epoch-mean train loss vs validation loss."""
    update_steps, update_loss = series["train/loss"]
    progress_steps, progress_epoch = series["progress/epoch"]
    epoch_at = dict(zip(progress_steps.tolist(), progress_epoch.tolist()))
    keep = [i for i, step in enumerate(update_steps.tolist()) if step in epoch_at]
    x = np.array([epoch_at[update_steps[i]] for i in keep])
    y = update_loss[keep]

    fig = Figure(figsize=(13, 4.6), dpi=150, layout="constrained")
    left, right = fig.subplots(1, 2)

    window = 50
    left.plot(x, y, color=TRAIN_COLOR, alpha=0.18, linewidth=0.6, label="each logged update")
    if len(y) > window:
        left.plot(x[window - 1 :], moving_average(y, window), color=TRAIN_COLOR, linewidth=1.8, label=f"moving average ({window} logs)")
    left.set_ylim(0, float(np.percentile(y, 99.5)) * 1.05 if len(y) else 1)
    left.set_xlabel("epoch")
    left.set_ylabel("training loss")
    left.set_title("Training loss")
    left.legend(frameon=False)
    left.grid(alpha=0.25)

    epochs, epoch_loss = series["epoch/loss"]
    val_epochs, val_loss = series["val/subset_loss"]
    right.plot(epochs, epoch_loss, "o-", color=TRAIN_COLOR, markersize=3.5, linewidth=1.6, label="train (epoch mean)")
    if len(val_loss):
        right.plot(val_epochs, val_loss, "o-", color=VAL_COLOR, markersize=3.5, linewidth=1.6, label="validation (128 fixed val samples)")
        best = int(np.argmin(val_loss))
        right.annotate(
            f"lowest {val_loss[best]:.2f} (epoch {val_epochs[best]})",
            (val_epochs[best], val_loss[best]),
            xytext=(0, -26),
            textcoords="offset points",
            ha="center",
            color=VAL_COLOR,
            arrowprops={"arrowstyle": "-", "color": VAL_COLOR},
        )
        right.annotate(f"{val_loss[-1]:.2f}", (val_epochs[-1], val_loss[-1]), xytext=(0, 8), textcoords="offset points", ha="center", color=VAL_COLOR)
    if len(epoch_loss):
        right.annotate(f"{epoch_loss[-1]:.2f}", (epochs[-1], epoch_loss[-1]), xytext=(0, 8), textcoords="offset points", ha="center", color=TRAIN_COLOR)
    right.set_xlabel("epoch")
    right.set_ylabel("loss")
    right.set_title("Train vs. validation loss per epoch")
    right.legend(frameon=False)
    right.grid(alpha=0.25)

    if title:
        fig.suptitle(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--out", type=Path, default=Path("training_curves.png"))
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=args.tracking_uri)
    print(f"wrote {render(load_series(client, args.run_id), args.out, args.title)}")


if __name__ == "__main__":
    main()
