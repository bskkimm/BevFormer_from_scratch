import numpy as np
import torch

from bevformer.models.backbone.image_backbone import MultiViewImageBackbone
from bevformer.models.bevformer import BEVFormerModel
from bevformer.models.heads.bevformer_head import BEVFormerHead
from bevformer.models.neck.fpn import ImageFPN
from bevformer.models.transformer.decoder import BEVFormerDecoder
from bevformer.models.transformer.encoder import BEVFormerEncoder

PC_RANGE = (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0)


def _identity_lidar2img():
    return np.eye(4, dtype=np.float32)


def _build_model(bev_h=4, bev_w=4, embed_dims=8, num_cams=2, num_classes=3, num_queries=6):
    backbone = MultiViewImageBackbone(variant="resnet50", pretrained=False, frozen_stages=-1)
    neck = ImageFPN(in_channels=(512, 1024, 2048), out_channels=embed_dims, out_names=("p3", "p4", "p5", "p6"))
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
    decoder = BEVFormerDecoder(
        embed_dims=embed_dims, num_queries=num_queries, num_layers=2, num_heads=2, num_points=2, ffn_channels=16
    )
    head = BEVFormerHead(embed_dims=embed_dims, num_classes=num_classes, box_dim=10, num_decoder_layers=2, pc_range=PC_RANGE)
    return BEVFormerModel(backbone, neck, encoder, decoder, head)


def _make_img_metas_queue(batch, queue_length, num_cams):
    return [
        [
            {"lidar2img": [_identity_lidar2img() for _ in range(num_cams)], "image_size": (64, 64)}
            for _ in range(queue_length)
        ]
        for _ in range(batch)
    ]


def test_forward_output_shapes():
    batch, queue_length, num_cams, embed_dims, num_classes, num_queries = 1, 3, 2, 8, 3, 6
    model = _build_model(embed_dims=embed_dims, num_cams=num_cams, num_classes=num_classes, num_queries=num_queries)

    imgs_queue = torch.randn(batch, queue_length, num_cams, 3, 64, 64)
    img_metas_queue = _make_img_metas_queue(batch, queue_length, num_cams)
    can_bus_queue = torch.zeros(batch, queue_length, 18)

    outputs = model(imgs_queue, img_metas_queue, can_bus_queue)

    assert outputs["cls_scores"].shape == (2, batch, num_queries, num_classes)
    assert outputs["bbox_preds"].shape == (2, batch, num_queries, 10)
    assert outputs["bev_embed"].shape == (batch, 16, embed_dims)
    assert torch.isfinite(outputs["cls_scores"]).all()
    assert torch.isfinite(outputs["bbox_preds"]).all()


def test_gradient_flows_only_through_last_frame_backbone_and_is_finite():
    batch, queue_length, num_cams, embed_dims = 1, 2, 2, 8
    model = _build_model(embed_dims=embed_dims, num_cams=num_cams)

    imgs_queue = torch.randn(batch, queue_length, num_cams, 3, 64, 64)
    img_metas_queue = _make_img_metas_queue(batch, queue_length, num_cams)
    can_bus_queue = torch.zeros(batch, queue_length, 18)

    outputs = model(imgs_queue, img_metas_queue, can_bus_queue)
    (outputs["cls_scores"].sum() + outputs["bbox_preds"].sum()).backward()

    for param in model.backbone.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
