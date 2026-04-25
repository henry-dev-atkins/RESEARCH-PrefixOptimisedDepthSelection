from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def aggregate_multidepth_loss(
    losses_by_depth: dict[int, torch.Tensor],
    max_depth: int,
    weight_power: float,
    normalise: bool = True,
) -> torch.Tensor:
    if not losses_by_depth:
        raise ValueError("losses_by_depth must be non-empty")
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")

    weighted_terms = []
    weights = []
    for depth, loss in sorted(losses_by_depth.items()):
        if depth <= 0 or depth > max_depth:
            raise ValueError(f"Depth {depth} is outside the valid range 1..{max_depth}")
        weight = (depth / max_depth) ** weight_power
        weighted_terms.append(loss * weight)
        weights.append(weight)

    total = torch.stack(weighted_terms).sum()
    if normalise:
        total = total / sum(weights)
    return total
