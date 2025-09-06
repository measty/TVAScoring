import torch

from tvs.data.dataset import PatchDataset
from tvs.data.transforms import BasicAugment, build_transforms


def test_build_transforms_strong_contains_components():
    tfms = build_transforms(train=True, augmentation="strong")
    names = [type(t).__name__ for t in tfms.transforms]
    assert names[0] == "StrongAugment"
    assert any(isinstance(t, BasicAugment) for t in tfms.transforms)


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
