import random

import torch

from tvs.data.dataset import PatchDataset
from tvs.data.transforms import BasicAugment, build_transforms


def test_build_transforms_strong_contains_components():
    tfms = build_transforms(train=True, augmentation="strong")
    names = [type(t).__name__ for t in tfms.transforms]
    assert names[0] == "StrongAugment"
    ba = next(t for t in tfms.transforms if isinstance(t, BasicAugment))
    assert ba.jitter is False


def test_build_transforms_basic_enables_jitter():
    tfms = build_transforms(train=True, augmentation="basic")
    ba = next(t for t in tfms.transforms if isinstance(t, BasicAugment))
    assert ba.jitter is True


def test_build_transforms_validation_no_augment():
    tfms = build_transforms(train=False)
    names = [type(t).__name__ for t in tfms.transforms]
    assert "StrongAugment" not in names
    assert not any(isinstance(t, BasicAugment) for t in tfms.transforms)


def test_patch_dataset_with_strong_aug(extractors):
    tfms = build_transforms(train=True, augmentation="strong")
    dataset = PatchDataset(extractors, {"T": -1.0, "V": 1.0}, transform=tfms)
    img, target = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape[1] == img.shape[2] == 512


def test_basic_augment_jitter_changes_colors():
    img = torch.full((3, 4, 4), 0.5)
    random.seed(0)
    torch.manual_seed(0)
    with_jitter = BasicAugment(jitter=True)(img.clone())
    random.seed(0)
    torch.manual_seed(0)
    no_jitter = BasicAugment(jitter=False)(img.clone())
    assert not torch.allclose(with_jitter, no_jitter)
