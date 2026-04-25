from __future__ import annotations

import pytest
import torch

from pods.models.cnn import PodsCNN
from pods.models.mlp import PodsMLP


def test_mlp_extract_prefix_matches_forward_prefix_shape() -> None:
    torch.manual_seed(0)
    model = PodsMLP(input_shape=(1, 28, 28), hidden_dims=(16, 12, 8), num_classes=10, candidate_depths=(1, 2, 3))
    inputs = torch.randn(4, 1, 28, 28)
    extracted = model.extract_prefix(2)
    assert extracted(inputs).shape == model.forward_prefix(inputs, 2).shape


def test_mlp_extract_prefix_matches_forward_prefix_logits() -> None:
    torch.manual_seed(0)
    model = PodsMLP(input_shape=(1, 28, 28), hidden_dims=(16, 12, 8), num_classes=10, candidate_depths=(1, 2, 3))
    inputs = torch.randn(4, 1, 28, 28)
    extracted = model.extract_prefix(2)
    assert torch.allclose(extracted(inputs), model.forward_prefix(inputs, 2), atol=1e-6, rtol=1e-6)


def test_cnn_extract_prefix_contains_only_requested_prefix_depth() -> None:
    model = PodsCNN(input_channels=3, channels=(8, 16, 32), num_classes=10, candidate_depths=(1, 2, 3), norm_type="groupnorm")
    extracted = model.extract_prefix(2)
    assert len(extracted.blocks) == 2


def test_cnn_extract_prefix_matches_forward_prefix_logits() -> None:
    torch.manual_seed(0)
    model = PodsCNN(input_channels=3, channels=(8, 16, 32), num_classes=10, candidate_depths=(1, 2, 3), norm_type="groupnorm")
    inputs = torch.randn(4, 3, 32, 32)
    extracted = model.extract_prefix(2)
    assert torch.allclose(extracted(inputs), model.forward_prefix(inputs, 2), atol=1e-6, rtol=1e-6)


def test_prefix_extraction_rejects_unsupported_depth() -> None:
    model = PodsMLP(input_shape=(1, 28, 28), hidden_dims=(16, 12, 8), num_classes=10, candidate_depths=(1, 2, 3))
    with pytest.raises(ValueError, match="Unsupported depth"):
        model.extract_prefix(4)
