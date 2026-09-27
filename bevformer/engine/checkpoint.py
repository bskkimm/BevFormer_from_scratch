"""Training checkpoints: save every epoch, resume after interruption."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model, optimizer=None, **state: Any) -> Path:
    """Atomically writes model (+ optimizer) weights and extra `state` to `path`.

    Written to a temporary file first, then renamed, so an interrupted save can
    never corrupt the previous checkpoint at the same path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), **state}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, model, optimizer=None, map_location="cpu") -> dict[str, Any]:
    """Loads weights into `model` (and `optimizer` if given); returns the remaining saved state.

    Also accepts a bare model state_dict (the format of older train.py outputs)."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "model" not in payload:
        payload = {"model": payload}
    model.load_state_dict(payload.pop("model"))
    optimizer_state = payload.pop("optimizer", None)
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return payload
