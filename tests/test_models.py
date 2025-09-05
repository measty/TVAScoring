import torch

from tvs.models import CNNBaseline, ConchHead, ResNet18Baseline, make_model


def test_cnn_baseline_forward_shape_and_range():
    model = CNNBaseline()
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    assert out.shape == (4,)
    assert torch.all(out <= 1.0) and torch.all(out >= -1.0)


def test_make_model_returns_cnn_baseline():
    model = make_model("cnn_baseline")
    assert isinstance(model, CNNBaseline)


def test_resnet18_forward_shape_and_range():
    model = ResNet18Baseline(pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2,)
    assert torch.all(out <= 1.0) and torch.all(out >= -1.0)


def test_make_model_returns_resnet18():
    model = make_model("resnet18", pretrained=False)
    assert isinstance(model, ResNet18Baseline)


def test_conch_head_forward_shape_and_range():
    model = ConchHead(image_size=64)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == (1,)
    assert torch.all(out <= 1.0) and torch.all(out >= -1.0)


def test_make_model_returns_conch_head():
    model = make_model("conch_head", image_size=64)
    assert isinstance(model, ConchHead)
