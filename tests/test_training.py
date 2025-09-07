import torch
from torch.utils.data import DataLoader, TensorDataset

from tvs.training import Trainer, get_loss, mae, pearsonr, rmse, spearmanr


def test_loss_factory():
    assert get_loss("mse").__class__.__name__ == "MSELoss"
    assert get_loss("huber").__class__.__name__ == "SmoothL1Loss"


def test_metrics():
    preds = torch.tensor([0.0, 1.0, 2.0])
    targets = torch.tensor([0.0, 2.0, 1.0])
    assert mae(preds, targets) == 2 / 3
    assert rmse(preds, targets) == torch.sqrt(torch.tensor(2 / 3)).item()
    p = pearsonr(preds, targets)
    s = spearmanr(preds, targets)
    assert abs(p) <= 1 and abs(s) <= 1


def test_trainer_early_stopping_and_checkpoint(tmp_path):
    x = torch.zeros(10, 3, 4, 4)
    y = torch.zeros(10)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=2)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 4 * 4, 1, bias=True),
        torch.nn.Tanh(),
    )
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    trainer = Trainer(
        model,
        optimizer,
        torch.nn.MSELoss(),
        torch.device("cpu"),
        scheduler=scheduler,
        early_stopping_patience=1,
        output_dir=tmp_path,
    )
    trainer.fit(loader, loader, epochs=5)
    assert trainer.epochs_trained == 2
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "best.pt").exists()
    assert optimizer.param_groups[0]["lr"] < 0.1


def test_trainer_overfits_small_batch():
    """Trainer should be able to overfit a tiny constant batch."""
    x = torch.zeros(4, 3, 4, 4)
    y = torch.full((4, 1), 0.8)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 4 * 4, 1),
        torch.nn.Tanh(),
    )
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    trainer = Trainer(
        model,
        optimizer,
        torch.nn.MSELoss(),
        torch.device("cpu"),
    )
    trainer.fit(loader, loader, epochs=25)
    with torch.no_grad():
        preds = model(x)
        mse = torch.nn.functional.mse_loss(preds, y)
    assert mse.item() < 1e-4


def _make_dummy_model():
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(3 * 4 * 4, 8),
                torch.nn.ReLU(),
            )
            self.backbone = torch.nn.Linear(8, 8)
            self.head = torch.nn.Linear(8, 1)
            self.act = torch.nn.Tanh()
            for p in self.backbone.parameters():
                p.requires_grad = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.stem(x)
            x = self.backbone(x)
            x = self.head(x)
            x = self.act(x)
            return x

    return DummyModel()


def test_trainer_unfreezes_backbone(tmp_path):
    x = torch.randn(16, 3, 4, 4)
    y = torch.randn(16, 1)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4)
    model = _make_dummy_model()
    init_params = [p.clone() for p in model.backbone.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model,
        optimizer,
        torch.nn.MSELoss(),
        torch.device("cpu"),
        unfreeze_at_epoch=2,
        unfreeze_params=model.backbone.parameters(),
        output_dir=tmp_path,
    )
    trainer.fit(loader, loader, epochs=1)
    for p, init in zip(model.backbone.parameters(), init_params):
        assert torch.allclose(p, init)
    trainer.fit(loader, loader, epochs=2)
    changed = any(
        not torch.allclose(p, init)
        for p, init in zip(model.backbone.parameters(), init_params)
    )
    assert changed
