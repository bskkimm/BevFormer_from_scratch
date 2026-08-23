import torch

from bevformer.models.transformer.temporal_self_attention import TemporalSelfAttention


def test_output_shape_without_prev_bev():
    batch, bev_h, bev_w, embed_dims = 2, 4, 4, 8
    num_query = bev_h * bev_w
    tsa = TemporalSelfAttention(embed_dims, num_heads=2, num_points=2)

    query = torch.randn(batch, num_query, embed_dims)
    output = tsa(query, prev_bev=None, bev_h=bev_h, bev_w=bev_w)

    assert output.shape == (batch, num_query, embed_dims)
    assert torch.isfinite(output).all()


def test_output_shape_with_prev_bev():
    batch, bev_h, bev_w, embed_dims = 2, 4, 4, 8
    num_query = bev_h * bev_w
    tsa = TemporalSelfAttention(embed_dims, num_heads=2, num_points=2)

    query = torch.randn(batch, num_query, embed_dims)
    prev_bev = torch.randn(batch, num_query, embed_dims)
    output = tsa(query, prev_bev=prev_bev, bev_h=bev_h, bev_w=bev_w)

    assert output.shape == (batch, num_query, embed_dims)
    assert torch.isfinite(output).all()


def test_gradient_flows_to_parameters():
    batch, bev_h, bev_w, embed_dims = 1, 3, 3, 8
    num_query = bev_h * bev_w
    tsa = TemporalSelfAttention(embed_dims, num_heads=2, num_points=2)

    query = torch.randn(batch, num_query, embed_dims, requires_grad=True)
    prev_bev = torch.randn(batch, num_query, embed_dims)
    output = tsa(query, prev_bev=prev_bev, bev_h=bev_h, bev_w=bev_w)
    output.sum().backward()

    assert query.grad is not None
    for param in tsa.parameters():
        assert param.grad is not None
