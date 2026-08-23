import torch

from bevformer.models.transformer.deformable_attention import MultiScaleDeformableAttention


def _make_value(batch, spatial_shapes, embed_dims):
    total = sum(h * w for h, w in spatial_shapes)
    return torch.randn(batch, total, embed_dims)


def test_output_shape_matches_query():
    batch, num_query, embed_dims = 2, 5, 16
    num_heads, num_levels, num_points = 4, 2, 3
    spatial_shapes = [(4, 4), (2, 2)]

    attn = MultiScaleDeformableAttention(embed_dims, num_heads, num_levels, num_points)
    query = torch.randn(batch, num_query, embed_dims)
    reference_points = torch.rand(batch, num_query, num_levels, num_points, 2)
    value = _make_value(batch, spatial_shapes, embed_dims)

    output = attn(query, reference_points, value, spatial_shapes)
    assert output.shape == (batch, num_query, embed_dims)
    assert torch.isfinite(output).all()


def test_gradient_flows_to_parameters():
    batch, num_query, embed_dims = 1, 3, 8
    num_heads, num_levels, num_points = 2, 2, 2
    spatial_shapes = [(3, 3), (2, 2)]

    attn = MultiScaleDeformableAttention(embed_dims, num_heads, num_levels, num_points)
    query = torch.randn(batch, num_query, embed_dims, requires_grad=True)
    reference_points = torch.rand(batch, num_query, num_levels, num_points, 2)
    value = _make_value(batch, spatial_shapes, embed_dims)

    output = attn(query, reference_points, value, spatial_shapes)
    output.sum().backward()

    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    for param in attn.parameters():
        assert param.grad is not None


def test_fully_masked_point_gives_finite_output():
    batch, num_query, embed_dims = 1, 2, 8
    num_heads, num_levels, num_points = 2, 1, 2
    spatial_shapes = [(3, 3)]

    attn = MultiScaleDeformableAttention(embed_dims, num_heads, num_levels, num_points)
    query = torch.randn(batch, num_query, embed_dims)
    reference_points = torch.rand(batch, num_query, num_levels, num_points, 2)
    value = _make_value(batch, spatial_shapes, embed_dims)

    point_mask = torch.zeros(batch, num_query, num_levels, num_points, dtype=torch.bool)
    point_mask[:, 0] = True  # query 0 fully masked out (all points invalid)

    output = attn(query, reference_points, value, spatial_shapes, point_mask=point_mask)
    assert torch.isfinite(output).all()
