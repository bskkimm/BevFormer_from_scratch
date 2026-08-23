import torch

from bevformer.models.losses.matcher import HungarianMatcher3D

PC_RANGE = (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0)


def test_near_perfect_prediction_matches_correct_pair():
    num_classes = 3
    matcher = HungarianMatcher3D(num_classes=num_classes, pc_range=PC_RANGE)

    gt_box = torch.tensor([[1.0, 2.0, 0.0, 2.0, 4.0, 1.5, 0.1, 0.0, 0.0]])
    gt_label = torch.tensor([1])

    # Two queries: query 0 predicts near-perfectly, query 1 is way off.
    cls_logits = torch.tensor([[[-5.0, 5.0, -5.0], [5.0, -5.0, -5.0]]])
    from bevformer.models.losses.loss_utils import encode_bbox_targets

    good_box = encode_bbox_targets(gt_box)
    bad_box = torch.zeros_like(good_box) + 100.0
    box_preds = torch.stack([good_box[0], bad_box[0]]).unsqueeze(0)

    assignments = matcher(cls_logits, box_preds, [gt_box], [gt_label])
    pred_ids, gt_ids = assignments[0]

    assert pred_ids.tolist() == [0]
    assert gt_ids.tolist() == [0]


def test_empty_gt_gives_empty_assignment():
    num_classes = 3
    matcher = HungarianMatcher3D(num_classes=num_classes, pc_range=PC_RANGE)

    cls_logits = torch.randn(1, 4, num_classes)
    box_preds = torch.randn(1, 4, 10)
    gt_boxes = [torch.zeros((0, 9))]
    gt_labels = [torch.zeros((0,), dtype=torch.long)]

    assignments = matcher(cls_logits, box_preds, gt_boxes, gt_labels)
    pred_ids, gt_ids = assignments[0]
    assert pred_ids.numel() == 0
    assert gt_ids.numel() == 0
