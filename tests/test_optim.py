import math

import pytest
import torch
import torch.nn as nn

from bevformer.engine.optim import OfficialLrSchedule, build_optimizer


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)
        self.frozen = nn.Linear(4, 4)
        for param in self.frozen.parameters():
            param.requires_grad = False


def test_build_optimizer_scales_backbone_lr_and_skips_frozen_params():
    model = _TinyModel()
    optimizer = build_optimizer(model, lr=2e-4, weight_decay=0.01, backbone_lr_mult=0.1)
    lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert lrs == {"backbone": pytest.approx(2e-5), "other": pytest.approx(2e-4)}
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) not in optimized for p in model.frozen.parameters())
    assert all(group["weight_decay"] == 0.01 for group in optimizer.param_groups)


def _factor(schedule, optimizer, epoch, update):
    schedule.apply(epoch, update)
    return optimizer.param_groups[1]["lr"] / optimizer.param_groups[1]["initial_lr"]


def test_official_schedule_warmup_matches_mmcv_linear_warmup():
    optimizer = build_optimizer(_TinyModel(), lr=2e-4, weight_decay=0.01)
    schedule = OfficialLrSchedule(optimizer, max_epochs=24)
    assert _factor(schedule, optimizer, 0, 0) == pytest.approx(1 / 3)
    assert _factor(schedule, optimizer, 0, 250) == pytest.approx(2 / 3)
    assert _factor(schedule, optimizer, 0, 500) == pytest.approx(1.0)


def test_official_schedule_cosine_steps_per_epoch_to_min_lr_ratio():
    optimizer = build_optimizer(_TinyModel(), lr=2e-4, weight_decay=0.01)
    schedule = OfficialLrSchedule(optimizer, max_epochs=24)
    min_ratio = 1e-3
    for epoch in (0, 6, 12, 23):
        expected = min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * epoch / 24))
        assert _factor(schedule, optimizer, epoch, 10_000) == pytest.approx(expected)
    # The backbone group keeps its 0.1x ratio throughout.
    schedule.apply(12, 10_000)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * optimizer.param_groups[1]["lr"])
