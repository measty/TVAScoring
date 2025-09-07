"""Training and validation loops with logging and checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import mae, rmse

__all__ = ["Trainer"]


class Trainer:
    """Utility class encapsulating a model training loop.

    Parameters
    ----------
    model:
        ``torch.nn.Module`` to optimise.
    optimizer:
        Optimizer instance.
    loss_fn:
        Loss function callable.
    device:
        Device on which to run training.
    scheduler:
        Optional learning rate scheduler stepping each epoch.
    max_grad_norm:
        If provided, gradients are clipped to this maximum norm.
    precision:
        ``"fp16"`` enables mixed precision via ``torch.cuda.amp`` when running
        on CUDA devices.
    early_stopping_patience:
        Number of epochs with no improvement in validation loss before training
        stops early.
    output_dir:
        Directory in which to save ``last.pt`` and ``best.pt`` checkpoints.
    unfreeze_at_epoch:
        If provided, backbone parameters in ``unfreeze_params`` remain frozen
        until this epoch and are unfrozen thereafter.
    unfreeze_params:
        Iterable of parameters to unfreeze at ``unfreeze_at_epoch``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        device: torch.device,
        *,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: Optional[float] = None,
        precision: str = "fp32",
        early_stopping_patience: Optional[int] = None,
        output_dir: Optional[Path] = None,
        unfreeze_at_epoch: Optional[int] = None,
        unfreeze_params: Optional[Iterable[torch.nn.Parameter]] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.precision = precision
        self.early_stopping_patience = early_stopping_patience
        self.output_dir = Path(output_dir) if output_dir else None
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.unfreeze_params = (
            list(unfreeze_params) if unfreeze_params is not None else None
        )
        self._backbone_unfrozen = False

        self.console = Console()
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if precision == "fp16" and device.type == "cuda"
            else None
        )
        self.best_val_loss: float | None = None
        self.epochs_trained = 0
        self._no_improve_epochs = 0

    # ------------------------------------------------------------------
    def _step_batch(self, batch, train: bool = True):
        imgs, targets = batch
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        autocast_ctx = (
            torch.cuda.amp.autocast()
            if self.scaler is not None
            else torch.enable_grad()
        )
        with autocast_ctx:
            preds = self.model(imgs)
            loss = self.loss_fn(preds, targets)
        if train:
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
        return loss.detach(), preds.detach(), targets.detach()

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        mode = "train" if train else "val"
        self.model.train(mode == "train")
        losses = []
        all_preds, all_targets = [], []
        pbar = tqdm(loader, desc=f"{mode}", leave=False)
        for batch in pbar:
            loss, preds, targets = self._step_batch(batch, train=train)
            losses.append(loss.item())
            all_preds.append(preds)
            all_targets.append(targets)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        preds_t = torch.cat(all_preds)
        targets_t = torch.cat(all_targets)
        return {
            "loss": float(torch.mean(torch.tensor(losses))),
            "mae": mae(preds_t, targets_t),
            "rmse": rmse(preds_t, targets_t),
        }

    # ------------------------------------------------------------------
    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int
    ) -> None:
        """Run the training loop."""

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, epochs + 1):
            if (
                self.unfreeze_at_epoch is not None
                and not self._backbone_unfrozen
                and epoch == self.unfreeze_at_epoch
            ):
                if self.unfreeze_params is not None:
                    for p in self.unfreeze_params:
                        p.requires_grad = True
                self._backbone_unfrozen = True
            self.console.rule(f"Epoch {epoch}")
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)
            if self.scheduler is not None:
                self.scheduler.step()
            self.console.log(
                f"train loss={train_metrics['loss']:.4f} val loss={val_metrics['loss']:.4f}"
            )
            stop = self._check_early_stopping(val_metrics["loss"])
            self._save_checkpoint(epoch, val_metrics["loss"])
            self.epochs_trained = epoch
            if stop:
                break

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        if self.output_dir is None:
            return
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }
        if self.scheduler is not None:
            ckpt["scheduler"] = self.scheduler.state_dict()
        if self.scaler is not None:
            ckpt["scaler"] = self.scaler.state_dict()
        torch.save(ckpt, self.output_dir / "last.pt")
        if self.best_val_loss is None or val_loss <= self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(ckpt, self.output_dir / "best.pt")

    # ------------------------------------------------------------------
    def _check_early_stopping(self, val_loss: float) -> bool:
        if self.early_stopping_patience is None:
            return False
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._no_improve_epochs = 0
            return False
        self._no_improve_epochs += 1
        return self._no_improve_epochs >= self.early_stopping_patience
