import torch

from bevformer.models.transformer.encoder import BEVFormerEncoder

PC_RANGE = (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0)


def _make_mlvl_feats(batch, num_cams, embed_dims, shapes):
    return [torch.randn(batch, num_cams, embed_dims, h, w) for h, w in shapes]


def _identity_lidar2img():
    import numpy as np

    return np.eye(4, dtype=np.float32)


def _make_img_metas(batch, num_cams):
    return [
        {"lidar2img": [_identity_lidar2img() for _ in range(num_cams)], "image_size": (100, 100)}
        for _ in range(batch)
    ]


def _build_encoder(bev_h=4, bev_w=4, embed_dims=8, num_cams=2, num_layers=2):
    return BEVFormerEncoder(
        num_layers=num_layers,
        bev_h=bev_h,
        bev_w=bev_w,
        embed_dims=embed_dims,
        pc_range=PC_RANGE,
        num_cams=num_cams,
        num_heads=2,
        num_levels=1,
        num_points_in_pillar=3,
        num_points_temporal=2,
        feedforward_dims=16,
    )


def test_output_shape_without_prev_bev():
    batch, num_cams, embed_dims = 1, 2, 8
    encoder = _build_encoder(embed_dims=embed_dims, num_cams=num_cams)
    mlvl_feats = _make_mlvl_feats(batch, num_cams, embed_dims, shapes=[(4, 4)])
    img_metas = _make_img_metas(batch, num_cams)

    bev_embed = encoder(mlvl_feats, img_metas)
    assert bev_embed.shape == (batch, 16, embed_dims)
    assert torch.isfinite(bev_embed).all()


def test_output_shape_with_prev_bev():
    batch, num_cams, embed_dims = 1, 2, 8
    encoder = _build_encoder(embed_dims=embed_dims, num_cams=num_cams)
    mlvl_feats = _make_mlvl_feats(batch, num_cams, embed_dims, shapes=[(4, 4)])
    img_metas = _make_img_metas(batch, num_cams)
    prev_bev = torch.randn(batch, 16, embed_dims)
    delta_translation = torch.zeros(batch, 2)
    delta_yaw = torch.zeros(batch)

    bev_embed = encoder(
        mlvl_feats,
        img_metas,
        prev_bev=prev_bev,
        delta_translation_bev=delta_translation,
        delta_yaw=delta_yaw,
    )
    assert bev_embed.shape == (batch, 16, embed_dims)
    assert torch.isfinite(bev_embed).all()


def test_gradient_flows_end_to_end():
    batch, num_cams, embed_dims = 1, 2, 8
    encoder = _build_encoder(embed_dims=embed_dims, num_cams=num_cams)
    mlvl_feats = [
        torch.randn(batch, num_cams, embed_dims, 4, 4, requires_grad=True)
    ]
    img_metas = _make_img_metas(batch, num_cams)

    bev_embed = encoder(mlvl_feats, img_metas)
    bev_embed.sum().backward()

    assert mlvl_feats[0].grad is not None
    for param in encoder.parameters():
        assert param.grad is not None
