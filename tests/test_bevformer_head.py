import torch

from bevformer.models.heads.bevformer_head import BEVFormerHead

PC_RANGE = (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0)


def _build_head(embed_dims=8, num_classes=3, num_decoder_layers=2):
    return BEVFormerHead(
        embed_dims=embed_dims,
        num_classes=num_classes,
        box_dim=10,
        num_decoder_layers=num_decoder_layers,
        pc_range=PC_RANGE,
    )


def test_init_reference_points_shape_and_range():
    head = _build_head()
    query_pos = torch.randn(1, 6, 8)
    ref_points = head.init_reference_points(query_pos)
    assert ref_points.shape == (1, 6, 3)
    assert torch.all(ref_points >= 0.0) and torch.all(ref_points <= 1.0)


def test_forward_single_shapes():
    head = _build_head(embed_dims=8, num_classes=3)
    layer_hs = torch.randn(1, 6, 8)
    reference_points = torch.rand(1, 6, 3)
    cls_score, bbox_pred = head.forward_single(0, layer_hs, reference_points)
    assert cls_score.shape == (1, 6, 3)
    assert bbox_pred.shape == (1, 6, 10)


def test_forward_stacks_all_decoder_layers():
    num_layers, embed_dims, num_classes = 2, 8, 3
    head = _build_head(embed_dims=embed_dims, num_classes=num_classes, num_decoder_layers=num_layers)
    hs = torch.randn(num_layers, 1, 6, embed_dims)
    inter_references = torch.rand(num_layers, 1, 6, 3)

    cls_scores, bbox_preds = head.forward(hs, inter_references)
    assert cls_scores.shape == (num_layers, 1, 6, num_classes)
    assert bbox_preds.shape == (num_layers, 1, 6, 10)


def test_gradient_flows_to_parameters():
    head = _build_head()
    hs = torch.randn(2, 1, 6, 8, requires_grad=True)
    inter_references = torch.rand(2, 1, 6, 3)

    cls_scores, bbox_preds = head.forward(hs, inter_references)
    (cls_scores.sum() + bbox_preds.sum()).backward()

    assert hs.grad is not None
    for branch in list(head.cls_branches) + list(head.reg_branches):
        for param in branch.parameters():
            assert param.grad is not None
