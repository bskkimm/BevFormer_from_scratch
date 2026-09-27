"""Training progress and ETA, as MLflow metrics and a human-readable run description."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Callable


def format_duration(seconds: float) -> str:
    minutes = max(int(round(seconds / 60)), 0)
    days, minutes = divmod(minutes, 24 * 60)
    hours, minutes = divmod(minutes, 60)
    if days:
        return f"{days}d {hours}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


class TrainingProgress:
    """Tracks progress over `total_epochs` epochs of `micro_batches_per_epoch` each.

    Rates come from this process's own measurements; `elapsed_offset_s` carries
    time spent before a resume so "elapsed" covers the whole training run.
    """

    def __init__(
        self,
        total_epochs: int,
        micro_batches_per_epoch: int,
        start_epoch: int = 0,
        elapsed_offset_s: float = 0.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.total_epochs = total_epochs
        self.per_epoch = micro_batches_per_epoch
        self.start_epoch = start_epoch
        self.elapsed_offset_s = elapsed_offset_s
        self.clock = clock
        self.start_time = clock()

    def snapshot(self, epoch: int, micro_batches_done: int) -> dict[str, float]:
        """`epoch` is 0-based; `micro_batches_done` counts within that epoch."""
        now = self.clock()
        run_elapsed = max(now - self.start_time, 1e-9)
        done_this_run = (epoch - self.start_epoch) * self.per_epoch + micro_batches_done
        done_total = epoch * self.per_epoch + micro_batches_done
        remaining = self.total_epochs * self.per_epoch - done_total
        rate = done_this_run / run_elapsed  # micro-batches per second
        result = {
            "epoch": epoch + micro_batches_done / self.per_epoch,
            "percent": 100.0 * done_total / (self.total_epochs * self.per_epoch),
            "elapsed_hours": (self.elapsed_offset_s + run_elapsed) / 3600.0,
        }
        if rate > 0:
            result.update(
                {
                    "avg_epoch_hours": self.per_epoch / rate / 3600.0,
                    "eta_hours": remaining / rate / 3600.0,
                    "micro_batches_per_s": rate,
                    "finish_unix_s": now + remaining / rate,
                }
            )
        return result

    def elapsed_s(self) -> float:
        return self.elapsed_offset_s + (self.clock() - self.start_time)

    def description(self, snapshot: dict[str, float], extra_lines: list[str] | None = None) -> str:
        """Markdown shown as the MLflow run description."""
        lines = [
            f"**Progress:** epoch {snapshot['epoch']:.2f} / {self.total_epochs} "
            f"({snapshot['percent']:.1f}%)",
            f"**Elapsed:** {format_duration(snapshot['elapsed_hours'] * 3600)}",
        ]
        if "eta_hours" in snapshot:
            finish = datetime.fromtimestamp(snapshot["finish_unix_s"]).astimezone()
            lines += [
                f"**Avg epoch time:** {format_duration(snapshot['avg_epoch_hours'] * 3600)}",
                f"**ETA:** {format_duration(snapshot['eta_hours'] * 3600)} "
                f"(finishes ~{finish:%Y-%m-%d %H:%M %Z})",
            ]
        lines += extra_lines or []
        return "  \n".join(lines)
