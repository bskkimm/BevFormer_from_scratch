import torch

from bevformer.models.heads.bevformer_head import BEVFormerHead
from bevformer.models.transformer.decoder import BEVFormerDecoder

PC_RANGE = (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0)


def _build_decoder_and_head(num_layers=2, embed_dims=8, num_queries=6, num_classes=3):
    decoder = BEVFormerDecoder(
        embed_dims=embed_dims,
        num_queries=num_queries,
        num_layers=num_layers,
        num_heads=2,
        num_points=2,
        ffn_channels=16,
    )
    head = BEVFormerHead(
        embed_dims=embed_dims,
        num_classes=num_classes,
        box_dim=10,
        num_decoder_layers=num_layers,
        pc_range=PC_RANGE,
    )
    return decoder, head


def test_output_shapes_across_layers():
    num_layers, embed_dims, num_queries, bev_h, bev_w = 2, 8, 6, 4, 4
    decoder, head = _build_decoder_and_head(num_layers, embed_dims, num_queries)
    bev_embed = torch.randn(1, bev_h * bev_w, embed_dims)

    hidden_states, init_reference, inter_references = decoder(
        bev_embed, bev_h, bev_w, reference_point_predictor=head.predict_reference_points
    )

    assert hidden_states.shape == (num_layers, 1, num_queries, embed_dims)
    assert init_reference.shape == (1, num_queries, 3)
    assert inter_references.shape == (num_layers, 1, num_queries, 3)
    assert torch.isfinite(hidden_states).all()


def test_gradient_flows_end_to_end():
    num_layers, embed_dims, num_queries, bev_h, bev_w = 2, 8, 6, 4, 4
    decoder, head = _build_decoder_and_head(num_layers, embed_dims, num_queries)
    bev_embed = torch.randn(1, bev_h * bev_w, embed_dims, requires_grad=True)

    hidden_states, _, inter_references = decoder(
        bev_embed, bev_h, bev_w, reference_point_predictor=head.predict_reference_points
    )
    cls_scores, bbox_preds = head.forward(hidden_states, inter_references)
    (cls_scores.sum() + bbox_preds.sum()).backward()

    assert bev_embed.grad is not None
    for param in decoder.parameters():
        assert param.grad is not None
