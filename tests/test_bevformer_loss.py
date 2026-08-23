import torch

from bevformer.models.losses.bevformer_loss import BEVFormerLoss

PC_RANGE = (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0)


def _build_loss(num_classes=3):
    return BEVFormerLoss(num_classes=num_classes, pc_range=PC_RANGE)


def test_loss_by_feat_returns_finite_scalars_with_gradient():
    num_layers, batch, num_queries, num_classes = 2, 1, 5, 3
    loss_fn = _build_loss(num_classes)

    cls_scores = torch.randn(num_layers, batch, num_queries, num_classes, requires_grad=True)
    bbox_preds = torch.randn(num_layers, batch, num_queries, 10, requires_grad=True)
    gt_boxes = [torch.tensor([[1.0, 2.0, 0.0, 2.0, 4.0, 1.5, 0.1, 0.0, 0.0]])]
    gt_labels = [torch.tensor([1])]

    losses = loss_fn.loss_by_feat(cls_scores, bbox_preds, gt_boxes, gt_labels)

    assert "loss_cls" in losses and "loss_bbox" in losses
    assert "d0.loss_cls" in losses and "d0.loss_bbox" in losses
    for value in losses.values():
        assert torch.isfinite(value).all()

    total = losses["loss_cls"] + losses["loss_bbox"]
    total.backward()
    assert cls_scores.grad is not None
    assert bbox_preds.grad is not None


def test_zero_gt_batch_gives_finite_all_background_loss():
    num_layers, batch, num_queries, num_classes = 1, 2, 4, 3
    loss_fn = _build_loss(num_classes)

    cls_scores = torch.randn(num_layers, batch, num_queries, num_classes)
    bbox_preds = torch.randn(num_layers, batch, num_queries, 10)
    gt_boxes = [torch.zeros((0, 9)), torch.zeros((0, 9))]
    gt_labels = [torch.zeros((0,), dtype=torch.long), torch.zeros((0,), dtype=torch.long)]

    losses = loss_fn.loss_by_feat(cls_scores, bbox_preds, gt_boxes, gt_labels)

    assert torch.isfinite(losses["loss_cls"])
    assert torch.isfinite(losses["loss_bbox"])
    assert losses["loss_bbox"].item() == 0.0
