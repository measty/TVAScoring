"""Conch backbone with regression heads for TubuloVillous scoring.

This module provides :class:`ConchHead` which wraps the CONCH vision
transformer backbone and attaches either a small MLP or a lightweight
transformer encoder head to produce a scalar score in ``[-1, 1]``.

The CONCH weights can optionally be loaded from a checkpoint and the
backbone can be frozen or fine-tuned during training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import conch.open_clip_custom.factory as conch_factory
import torch
from conch.open_clip_custom.coca_model import CoCa
from conch.open_clip_custom.factory import create_model as conch_create_model
from torch import nn

__all__ = ["ConchHead"]


class _TransformerRegressionHead(nn.Module):
    """Single-layer Transformer encoder followed by ``tanh`` regressor."""

    def __init__(self, embed_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.out = nn.Linear(embed_dim, 1)
        self.activation = nn.Tanh()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Encode token sequence and regress a score."""

        x = self.encoder(tokens)
        cls = x[:, 0]  # take CLS token
        x = self.out(cls)
        x = self.activation(x)
        return x.squeeze(-1)


class _MLPRegressionHead(nn.Module):
    """Two-layer MLP with ``tanh`` output."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Regress score from feature vector."""

        x = self.net(x)
        x = self.activation(x)
        return x.squeeze(-1)


class ConchHead(nn.Module):
    """CONCH backbone with configurable regression head.

    Parameters
    ----------
    checkpoint_path:
        Optional path to pretrained CONCH weights.  If ``None`` the
        backbone is randomly initialised.
    finetune_backbone:
        If ``True`` the CONCH backbone is left trainable, otherwise it is
        frozen and only the regression head learns.
    head_type:
        Choice of regression head: ``"mlp"`` or ``"transformer"``.
    image_size:
        Input spatial size expected by the backbone.  Defaults to ``448``
        (the native CONCH resolution).  Smaller values can be specified
        for testing purposes.
    """

    def __init__(
        self,
        *,
        checkpoint_path: Optional[str] = None,
        finetune_backbone: bool = False,
        head_type: str = "mlp",
        image_size: int = 448,
    ) -> None:
        super().__init__()
        self.backbone = self._build_backbone(checkpoint_path, image_size)
        if not finetune_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head_type = head_type.lower()
        if self.head_type == "transformer":
            token_dim = self.backbone.visual.trunk.embed_dim
            self.head = _TransformerRegressionHead(token_dim)
        elif self.head_type == "mlp":
            embed_dim = self.backbone.embed_dim
            self.head = _MLPRegressionHead(embed_dim)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown head_type '{head_type}'")

    @staticmethod
    def _build_backbone(checkpoint_path: Optional[str], image_size: int) -> CoCa:
        """Construct CONCH backbone, optionally loading weights."""

        if checkpoint_path:
            return conch_create_model(
                "conch_ViT-B-16",
                checkpoint_path=checkpoint_path,
                device="cpu",
                force_image_size=image_size,
            )

        cfg_dir = Path(conch_factory.__file__).parent / "model_configs"
        with open(cfg_dir / "conch_ViT-B-16.json") as f:
            cfg = json.load(f)
        cfg["vision_cfg"]["image_size"] = image_size
        cfg.pop("custom_text", None)
        model = CoCa(**cfg)
        return model

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - short
        """Compute forward pass producing scores in ``[-1, 1]``."""

        image_latent, tokens = self.backbone.visual(x)
        if self.head_type == "transformer":
            return self.head(tokens)
        return self.head(image_latent)
