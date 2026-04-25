from __future__ import annotations

import pytest
import torch

from pods.training.losses import aggregate_multidepth_loss


def test_aggregate_multidepth_loss_applies_depth_weights_and_normalises() -> None:
    losses = {1: torch.tensor(2.0), 4: torch.tensor(4.0)}
    total = aggregate_multidepth_loss(losses, max_depth=4, weight_power=1.0, normalise=True)
    expected = (2.0 * 0.25 + 4.0 * 1.0) / (0.25 + 1.0)
    assert total.item() == pytest.approx(expected)


def test_aggregate_multidepth_loss_fails_on_empty_loss_map() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        aggregate_multidepth_loss({}, max_depth=4, weight_power=1.0)
