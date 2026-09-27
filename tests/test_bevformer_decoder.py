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
        bev_embed, bev_h, bev_w, init_reference_fn=head.init_reference_points, refine_reference_fn=head.refine_reference_points
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
        bev_embed, bev_h, bev_w, init_reference_fn=head.init_reference_points, refine_reference_fn=head.refine_reference_points
    )
    cls_scores, bbox_preds = head.forward(hidden_states, inter_references)
    (cls_scores.sum() + bbox_preds.sum()).backward()

    assert bev_embed.grad is not None
    for param in decoder.parameters():
        assert param.grad is not None


def test_reference_points_follow_official_box_refinement():
    torch.manual_seed(0)
    num_layers, embed_dims, bev_h, bev_w = 3, 8, 4, 4
    decoder, head = _build_decoder_and_head(num_layers, embed_dims, num_queries=6)
    bev_embed = torch.randn(1, bev_h * bev_w, embed_dims)
    hs, init_ref, refs = decoder(
        bev_embed, bev_h, bev_w, init_reference_fn=head.init_reference_points, refine_reference_fn=head.refine_reference_points
    )
    # Layer 0 attends at the reference predicted from the positional query embedding.
    _, query_pos = decoder.init_decoder_state(1, bev_embed.device)
    torch.testing.assert_close(refs[0], head.init_reference_points(query_pos))
    torch.testing.assert_close(init_ref, refs[0])
    # Each later layer attends where the previous layer's box is: decoded centers of
    # layer l equal the (metric) reference point of layer l + 1.
    _, boxes = head(hs, refs)
    for layer in range(num_layers - 1):
        next_ref = refs[layer + 1]
        expected = torch.stack([
            next_ref[..., 0] * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0],
            next_ref[..., 1] * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1],
            next_ref[..., 2] * (PC_RANGE[5] - PC_RANGE[2]) + PC_RANGE[2],
        ], dim=-1)
        torch.testing.assert_close(boxes[layer][..., [0, 1, 4]], expected, atol=1e-4, rtol=1e-4)
    # Refined references are detached, as in the official decoder: no gradient flows
    # from later layers' references back into the decoder layers that produced them.
    refs[1:].sum().backward()
    assert all(p.grad is None for p in decoder.layers.parameters())
