from __future__ import annotations

from torch import nn


def make_classification_head(in_features: int, num_classes: int) -> nn.Module:
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    return nn.Linear(in_features, num_classes)
