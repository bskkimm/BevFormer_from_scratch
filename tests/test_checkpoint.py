import torch
import torch.nn as nn

from bevformer.engine.checkpoint import load_checkpoint, save_checkpoint


def _model_and_optimizer():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(3, 4)).sum().backward()
    optimizer.step()
    return model, optimizer


def test_checkpoint_round_trip_restores_weights_optimizer_and_state(tmp_path):
    torch.manual_seed(0)
    model, optimizer = _model_and_optimizer()
    path = save_checkpoint(tmp_path / "ckpt" / "latest.pth", model, optimizer, epoch=3, global_update=1200, mlflow_run_id="abc")

    torch.manual_seed(1)
    restored, restored_opt = _model_and_optimizer()
    state = load_checkpoint(path, restored, restored_opt)

    assert state == {"epoch": 3, "global_update": 1200, "mlflow_run_id": "abc"}
    for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
        torch.testing.assert_close(a, b)
    assert restored_opt.state_dict()["state"][0]["step"] == optimizer.state_dict()["state"][0]["step"]
    assert not list(tmp_path.glob("ckpt/*.tmp"))  # atomic write left no temp file


def test_bare_state_dict_checkpoint_still_loads(tmp_path):
    model, _ = _model_and_optimizer()
    torch.save(model.state_dict(), tmp_path / "old.pth")
    fresh, _ = _model_and_optimizer()
    assert load_checkpoint(tmp_path / "old.pth", fresh) == {}
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values()):
        torch.testing.assert_close(a, b)


def test_weights_only_checkpoint_loads_without_optimizer(tmp_path):
    model, _ = _model_and_optimizer()
    path = save_checkpoint(tmp_path / "w.pth", model, epoch=1)
    fresh, fresh_opt = _model_and_optimizer()
    assert load_checkpoint(path, fresh, fresh_opt) == {"epoch": 1}
