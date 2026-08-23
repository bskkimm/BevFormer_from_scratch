import numpy as np
import torch

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
