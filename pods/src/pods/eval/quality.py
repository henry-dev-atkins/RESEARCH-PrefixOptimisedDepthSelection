from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch import nn

from pods.models.interfaces import PrefixCapableModel


@dataclass(frozen=True)
class PrefixQualityResult:
    depth: int
    loss: float
    accuracy: float
    num_examples: int


@torch.no_grad()
def evaluate_all_prefixes(
    model: PrefixCapableModel,
    data_loader: DataLoader,
    candidate_depths: tuple[int, ...],
    device: torch.device,
) -> dict[int, PrefixQualityResult]:
    model = model.to(device)
    model.eval()

    loss_sums = {depth: 0.0 for depth in candidate_depths}
    correct = {depth: 0 for depth in candidate_depths}
    total = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits_by_depth = model.forward_depths(inputs, candidate_depths)
        for depth, logits in logits_by_depth.items():
            loss_sums[depth] += float(F.cross_entropy(logits, labels, reduction="sum").item())
            correct[depth] += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.numel()

    if total == 0:
        raise ValueError("data_loader produced no examples")

    return {
        depth: PrefixQualityResult(
            depth=depth,
            loss=loss_sums[depth] / total,
            accuracy=correct[depth] / total,
            num_examples=total,
        )
        for depth in candidate_depths
    }


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    depth: int,
) -> PrefixQualityResult:
    model = model.to(device)
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.numel()

    if total == 0:
        raise ValueError("data_loader produced no examples")

    return PrefixQualityResult(
        depth=depth,
        loss=loss_sum / total,
        accuracy=correct / total,
        num_examples=total,
    )
