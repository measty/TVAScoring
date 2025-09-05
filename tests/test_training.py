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
