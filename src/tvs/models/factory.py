"""Factory for constructing model instances."""

from __future__ import annotations

from typing import Any, Dict, Type

from torch import nn

from .cnn_baseline import CNNBaseline
from .conch_head import ConchHead
from .resnet import ResNet18Baseline

__all__ = ["MODEL_REGISTRY", "make_model"]

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "cnn_baseline": CNNBaseline,
    "resnet18": ResNet18Baseline,
    "conch_head": ConchHead,
}


def make_model(model_cfg: dict) -> nn.Module:
    """Instantiate a model by name.

    Parameters
    ----------
    model_cfg:
        Key identifying the model architecture.  Options are
        ``"cnn_baseline"``, ``"resnet18"`` and ``"conch_head"``.
    **kwargs:
        Additional keyword arguments passed to the model constructor.

    Returns
    -------
    nn.Module
        Instantiated model.

    Raises
    ------
    ValueError
        If ``name`` is not a known model identifier.
    """
    name = model_cfg.pop("name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'")
    model_cls = MODEL_REGISTRY[name]
    return model_cls(**model_cfg)
