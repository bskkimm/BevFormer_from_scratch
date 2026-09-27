import numpy as np
import pytest
import torch

from bevformer.data.transforms import normalize_images
from bevformer.engine.trainer import fit, move_batch_to_device, train_one_epoch
from bevformer.models.backbone.image_backbone import MultiViewImageBackbone
from bevformer.models.bevformer import BEVFormerModel
from bevformer.models.heads.bevformer_head import BEVFormerHead
from bevformer.models.losses.bevformer_loss import BEVFormerLoss
from bevformer.models.neck.fpn import ImageFPN
from bevformer.models.transformer.decoder import BEVFormerDecoder
from bevformer.models.transformer.encoder import BEVFormerEncoder

PC_RANGE = (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0)


def _identity_lidar2img():
    return np.eye(4, dtype=np.float32)


def _build_model(bev_h=4, bev_w=4, embed_dims=8, num_cams=2, num_classes=3, num_queries=6):
    backbone = MultiViewImageBackbone(variant="resnet50", pretrained=False, frozen_stages=-1)
    neck = ImageFPN(in_channels=(512, 1024, 2048), out_channels=embed_dims)
    encoder = BEVFormerEncoder(
        num_layers=1,
        bev_h=bev_h,
        bev_w=bev_w,
        embed_dims=embed_dims,
        pc_range=PC_RANGE,
        num_cams=num_cams,
        num_heads=2,
        num_levels=4,
        num_points_in_pillar=2,
        num_points_temporal=2,
        feedforward_dims=16,
    )
    decoder = BEVFormerDecoder(embed_dims=embed_dims, num_queries=num_queries, num_layers=2, num_heads=2, num_points=2, ffn_channels=16)
    head = BEVFormerHead(embed_dims=embed_dims, num_classes=num_classes, box_dim=10, num_decoder_layers=2, pc_range=PC_RANGE)
    return BEVFormerModel(backbone, neck, encoder, decoder, head)


def _make_batch(batch, queue_length, num_cams):
    imgs = torch.randn(batch, queue_length, num_cams, 3, 64, 64)
    img_metas = [
        [
            {"lidar2img": [_identity_lidar2img() for _ in range(num_cams)], "image_size": (64, 64)}
            for _ in range(queue_length)
        ]
        for _ in range(batch)
    ]
    can_bus = torch.zeros(batch, queue_length, 18)
    gt_boxes_3d = [torch.tensor([[1.0, 2.0, 0.0, 2.0, 4.0, 1.5, 0.1, 0.0, 0.0]]) for _ in range(batch)]
    gt_labels_3d = [torch.tensor([1]) for _ in range(batch)]
    return {
        "imgs": imgs,
        "img_metas": img_metas,
        "can_bus": can_bus,
        "gt_boxes_3d": gt_boxes_3d,
        "gt_labels_3d": gt_labels_3d,
    }


def test_move_batch_to_device_is_noop_on_cpu():
    batch = _make_batch(1, 2, 2)
    moved = move_batch_to_device(batch, torch.device("cpu"))
    assert moved["imgs"].device.type == "cpu"
    assert moved["can_bus"].device.type == "cpu"


def test_train_one_epoch_returns_finite_loss_and_updates_parameters():
    model = _build_model()
    criterion = BEVFormerLoss(num_classes=3, pc_range=PC_RANGE, use_auxiliary_losses=False)
    dataloader = [_make_batch(1, 2, 2), _make_batch(1, 2, 2)]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    # use_auxiliary_losses=False means only the final decoder layer's
    # branch receives gradient.
    param_before = next(model.head.reg_branches[-1].parameters()).clone()
    metrics = train_one_epoch(model, criterion, dataloader, optimizer, device)

    assert "loss" in metrics
    assert np.isfinite(metrics["loss"])
    param_after = next(model.head.reg_branches[-1].parameters())
    assert not torch.equal(param_before, param_after)


def test_fit_runs_multiple_epochs():
    model = _build_model()
    criterion = BEVFormerLoss(num_classes=3, pc_range=PC_RANGE, use_auxiliary_losses=False)
    dataloader = [_make_batch(1, 2, 2)]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    history = fit(model, criterion, dataloader, optimizer, device, epochs=2, log_every_epoch=False)
    assert len(history) == 2
    for entry in history:
        assert np.isfinite(entry["loss"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast path is exercised on CUDA")
@pytest.mark.parametrize("amp_dtype", [torch.bfloat16, torch.float16])
def test_train_one_epoch_mixed_precision_on_cuda(amp_dtype):
    device = torch.device("cuda")
    model = _build_model().to(device)
    criterion = BEVFormerLoss(num_classes=3, pc_range=PC_RANGE, use_auxiliary_losses=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = move_batch_to_device(_make_batch(1, 2, 2), device)

    param_before = next(model.head.reg_branches[-1].parameters()).detach().clone()
    # fp16's GradScaler starts at scale 2**16 and skips the first few overflowing
    # steps while it backs off, so give it several steps before checking updates.
    history = fit(model, criterion, [batch], optimizer, device, epochs=6, amp_dtype=amp_dtype, log_every_epoch=False)

    assert np.isfinite(history[0]["loss"])
    assert not torch.equal(param_before, next(model.head.reg_branches[-1].parameters()))


def test_move_batch_to_device_normalizes_uint8_images():
    batch = _make_batch(1, 2, 2)
    batch["imgs"] = torch.randint(0, 256, batch["imgs"].shape, dtype=torch.uint8)
    moved = move_batch_to_device(batch, torch.device("cpu"))
    assert moved["imgs"].dtype == torch.float32
    torch.testing.assert_close(moved["imgs"], normalize_images(batch["imgs"]))


class _LinearModel(torch.nn.Module):
    """Stand-in exposing the trainer's model/criterion interface with a plain mean loss."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1, bias=False)

    def forward(self, imgs, img_metas, can_bus):
        pred = self.linear(imgs.flatten(1))
        return {"cls_scores": pred, "bbox_preds": pred}


class _MeanSquaredCriterion:
    def loss_by_feat(self, cls_scores, bbox_preds, gt_boxes, gt_labels):
        target = torch.stack([boxes.sum() for boxes in gt_boxes]).view(-1, 1)
        return {"loss_mse": ((cls_scores - target) ** 2).mean()}


def _linear_batch(values, targets):
    return {
        "imgs": torch.tensor(values, dtype=torch.float32),
        "img_metas": [[{}] for _ in values],
        "can_bus": torch.zeros(len(values), 1, 18),
        "gt_boxes_3d": [torch.tensor([t]) for t in targets],
        "gt_labels_3d": [torch.tensor([0]) for _ in values],
    }


def test_accumulating_two_micro_batches_equals_one_batch_of_two():
    samples = [([1.0, 2.0, 3.0], 1.0), ([-1.0, 0.5, 2.0], -2.0)]
    torch.manual_seed(0)
    accumulated = _LinearModel()
    single = _LinearModel()
    single.load_state_dict(accumulated.state_dict())
    opt_a = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    opt_s = torch.optim.SGD(single.parameters(), lr=0.1)

    micro = [_linear_batch([v], [t]) for v, t in samples]
    whole = [_linear_batch([v for v, _ in samples], [t for _, t in samples])]
    metrics_a = train_one_epoch(accumulated, _MeanSquaredCriterion(), micro, opt_a, torch.device("cpu"), accumulation_steps=2)
    metrics_s = train_one_epoch(single, _MeanSquaredCriterion(), whole, opt_s, torch.device("cpu"))

    assert metrics_a["updates"] == metrics_s["updates"] == 1
    torch.testing.assert_close(accumulated.linear.weight, single.linear.weight)


def test_partial_last_window_still_steps_and_hooks_see_global_updates():
    batches = [_linear_batch([[1.0, 0.0, 0.0]], [1.0]) for _ in range(5)]
    model = _LinearModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lr_calls, updates = [], []
    metrics = train_one_epoch(
        model, _MeanSquaredCriterion(), batches, optimizer, torch.device("cpu"),
        accumulation_steps=2, epoch=3, start_update=10,
        lr_schedule=lambda epoch, update: lr_calls.append((epoch, update)),
        update_callback=lambda state: updates.append((state["update"], state["micro_batches_done"])),
    )
    assert metrics["updates"] == 3  # windows of 2, 2, and a final partial 1
    assert lr_calls == [(3, 10), (3, 11), (3, 12)]
    assert updates == [(11, 2), (12, 4), (13, 5)]
