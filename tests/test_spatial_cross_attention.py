import torch

from bevformer.models.transformer.spatial_cross_attention import SpatialCrossAttention


def _make_mlvl_feats(batch, num_cams, embed_dims, shapes):
    return [torch.randn(batch, num_cams, embed_dims, h, w) for h, w in shapes]


def test_output_shape_matches_query():
    batch, num_query, embed_dims = 1, 6, 8
    num_cams, num_levels, num_points_in_pillar = 3, 2, 4
    shapes = [(4, 4), (2, 2)]

    sca = SpatialCrossAttention(embed_dims, num_cams, num_levels, num_points_in_pillar, num_heads=2)
    query = torch.randn(batch, num_query, embed_dims)
    mlvl_feats = _make_mlvl_feats(batch, num_cams, embed_dims, shapes)
    reference_points_cam = torch.rand(num_cams, batch, num_query, num_points_in_pillar, 2)
    bev_mask = torch.ones(num_cams, batch, num_query, num_points_in_pillar, dtype=torch.bool)

    output = sca(query, mlvl_feats, reference_points_cam, bev_mask)
    assert output.shape == (batch, num_query, embed_dims)
    assert torch.isfinite(output).all()


def test_query_with_no_valid_camera_gives_finite_zero_output():
    batch, num_query, embed_dims = 1, 2, 8
    num_cams, num_levels, num_points_in_pillar = 2, 1, 3
    shapes = [(4, 4)]

    sca = SpatialCrossAttention(embed_dims, num_cams, num_levels, num_points_in_pillar, num_heads=2)
    query = torch.randn(batch, num_query, embed_dims)
    mlvl_feats = _make_mlvl_feats(batch, num_cams, embed_dims, shapes)
    reference_points_cam = torch.rand(num_cams, batch, num_query, num_points_in_pillar, 2)
    bev_mask = torch.ones(num_cams, batch, num_query, num_points_in_pillar, dtype=torch.bool)
    bev_mask[:, :, 0, :] = False  # query 0 invisible to every camera

    output = sca(query, mlvl_feats, reference_points_cam, bev_mask)
    assert torch.isfinite(output).all()
    assert torch.all(output[:, 0] == 0.0)


def test_gradient_flows_to_parameters():
    batch, num_query, embed_dims = 1, 4, 8
    num_cams, num_levels, num_points_in_pillar = 2, 1, 2
    shapes = [(3, 3)]

    sca = SpatialCrossAttention(embed_dims, num_cams, num_levels, num_points_in_pillar, num_heads=2)
    query = torch.randn(batch, num_query, embed_dims, requires_grad=True)
    mlvl_feats = _make_mlvl_feats(batch, num_cams, embed_dims, shapes)
    reference_points_cam = torch.rand(num_cams, batch, num_query, num_points_in_pillar, 2)
    bev_mask = torch.ones(num_cams, batch, num_query, num_points_in_pillar, dtype=torch.bool)

    output = sca(query, mlvl_feats, reference_points_cam, bev_mask)
    output.sum().backward()
    for param in sca.parameters():
        assert param.grad is not None


def test_multiple_learned_points_per_pillar_anchor_follow_their_anchor():
    # Official MSDeformableAttention3D: num_points = points_per_anchor * pillar anchors,
    # point k attached to anchor k % D, starting at that anchor's projection and
    # masked by that anchor's validity.
    batch, num_query, embed_dims, num_cams, anchors, per_anchor = 1, 5, 8, 2, 3, 2
    sca = SpatialCrossAttention(embed_dims, num_cams, 1, anchors, num_heads=2, num_points_per_anchor=per_anchor)
    assert sca.deform_attn.num_points == anchors * per_anchor

    ref = torch.rand(num_cams, batch, num_query, anchors, 2)
    mask = torch.rand(num_cams, batch, num_query, anchors) > 0.3
    tiled_ref, tiled_mask = sca.expand_anchor_points(ref[0], mask[0])
    assert tiled_ref.shape == (batch, num_query, 1, anchors * per_anchor, 2)
    for k in range(anchors * per_anchor):
        torch.testing.assert_close(tiled_ref[:, :, 0, k], ref[0][:, :, k % anchors])
        assert torch.equal(tiled_mask[:, :, 0, k], ~mask[0][:, :, k % anchors])

    out = sca(torch.randn(batch, num_query, embed_dims), [torch.randn(batch, num_cams, embed_dims, 4, 4)], ref, mask)
    assert out.shape == (batch, num_query, embed_dims) and torch.isfinite(out).all()
