from __future__ import annotations

import torch

from pods.data.transforms import build_image_transforms
from pods.models.resnet import PodsResNet


def test_cifar_train_and_eval_transforms_differ() -> None:
    train_transform = build_image_transforms("cifar10", train=True)
    eval_transform = build_image_transforms("cifar10", train=False)
    assert train_transform != eval_transform
    assert len(train_transform.transforms) > len(eval_transform.transforms)


def test_resnet_extract_prefix_matches_forward_prefix_logits() -> None:
    torch.manual_seed(0)
    model = PodsResNet(variant="resnet56", num_classes=10, candidate_depths=(1, 2, 3), norm_type="groupnorm")
    inputs = torch.randn(2, 3, 32, 32)
    extracted = model.extract_prefix(2)
    assert torch.allclose(extracted(inputs), model.forward_prefix(inputs, 2), atol=1e-6, rtol=1e-5)


def test_resnet_rejects_unsupported_depth() -> None:
    model = PodsResNet(variant="resnet56", num_classes=10, candidate_depths=(1, 2, 3), norm_type="groupnorm")
    try:
        model.forward_prefix(torch.randn(1, 3, 32, 32), 4)
    except ValueError as exc:
        assert "Unsupported depth" in str(exc)
    else:
        raise AssertionError("Expected unsupported depth to raise ValueError")
