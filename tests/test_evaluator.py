import torch

from bevformer.engine.evaluator import decode_predictions, evaluate_predictions

PC_RANGE = (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0)


def test_decode_predictions_respects_max_num():
    num_queries, num_classes = 10, 3
    cls_logits = torch.randn(num_queries, num_classes)
    bbox_preds = torch.randn(num_queries, 10)

    boxes, scores, labels = decode_predictions(cls_logits, bbox_preds, max_num=5, post_center_range=(-1e6,) * 3 + (1e6,) * 3)
    assert boxes.shape[0] <= 5
    assert scores.shape[0] == boxes.shape[0]
    assert labels.shape[0] == boxes.shape[0]


def test_decode_predictions_filters_outside_post_center_range():
    cls_logits = torch.full((1, 1), 10.0)  # very confident single query/class
    bbox_preds = torch.zeros(1, 10)
    bbox_preds[0, 0] = 1000.0  # x way outside any reasonable range

    boxes, scores, labels = decode_predictions(
        cls_logits, bbox_preds, max_num=10, post_center_range=(-10, -10, -10, 10, 10, 10)
    )
    assert boxes.shape[0] == 0


def test_evaluate_predictions_perfect_match():
    pred_boxes = torch.tensor([[1.0, 2.0, 0.0, 2.0, 4.0, 1.5, 0.1, 0.0, 0.0]])
    pred_labels = torch.tensor([1])
    pred_scores = torch.tensor([0.9])
    gt_boxes = [torch.tensor([[1.0, 2.0, 0.0, 2.0, 4.0, 1.5, 0.1, 0.0, 0.0]])]
    gt_labels = [torch.tensor([1])]

    metrics = evaluate_predictions([pred_boxes], [pred_labels], [pred_scores], gt_boxes, gt_labels)
    assert metrics["match_rate"] == 1.0
    assert metrics["mean_center_error"] < 1e-4


def test_evaluate_predictions_no_matches():
    pred_boxes = torch.tensor([[100.0, 100.0, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0]])
    pred_labels = torch.tensor([1])
    pred_scores = torch.tensor([0.9])
    gt_boxes = [torch.tensor([[1.0, 2.0, 0.0, 2.0, 4.0, 1.5, 0.1, 0.0, 0.0]])]
    gt_labels = [torch.tensor([1])]

    metrics = evaluate_predictions([pred_boxes], [pred_labels], [pred_scores], gt_boxes, gt_labels)
    assert metrics["match_rate"] == 0.0
