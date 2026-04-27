from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch import nn

from pods.data.batches import move_batch_to_device
from pods.eval.metrics import bootstrap_confidence_interval, compute_metric
from pods.models.interfaces import PrefixCapableModel


@dataclass(frozen=True)
class PrefixQualityResult:
    depth: int
    loss: float
    accuracy: float
    metric_name: str
    metric_value: float
    metric_ci_low: float | None
    metric_ci_high: float | None
    num_examples: int


@torch.no_grad()
def evaluate_all_prefixes(
    model: PrefixCapableModel,
    data_loader: DataLoader,
    candidate_depths: tuple[int, ...],
    device: torch.device,
    metric_name: str = "accuracy",
    bootstrap_samples: int = 0,
    confidence_level: float = 0.95,
) -> dict[int, PrefixQualityResult]:
    model = model.to(device)
    model.eval()

    loss_sums = {depth: 0.0 for depth in candidate_depths}
    correct = {depth: 0 for depth in candidate_depths}
    labels_by_depth = {depth: [] for depth in candidate_depths}
    predictions_by_depth = {depth: [] for depth in candidate_depths}
    total = 0

    for batch in data_loader:
        inputs, labels = move_batch_to_device(batch, device)
        logits_by_depth = model.forward_depths(inputs, candidate_depths)
        for depth, logits in logits_by_depth.items():
            predictions = logits.argmax(dim=1)
            loss_sums[depth] += float(F.cross_entropy(logits, labels, reduction="sum").item())
            correct[depth] += int((predictions == labels).sum().item())
            predictions_by_depth[depth].extend(int(value) for value in predictions.cpu().tolist())
            labels_by_depth[depth].extend(int(value) for value in labels.cpu().tolist())
        total += labels.numel()

    if total == 0:
        raise ValueError("data_loader produced no examples")

    bootstrap_seed_base = len(candidate_depths) * 1009
    results: dict[int, PrefixQualityResult] = {}
    for depth in candidate_depths:
        ci_low, ci_high = bootstrap_confidence_interval(
            metric_name,
            labels_by_depth[depth],
            predictions_by_depth[depth],
            num_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=bootstrap_seed_base + depth,
        )
        results[depth] = PrefixQualityResult(
            depth=depth,
            loss=loss_sums[depth] / total,
            accuracy=correct[depth] / total,
            metric_name=metric_name,
            metric_value=compute_metric(metric_name, labels_by_depth[depth], predictions_by_depth[depth]),
            metric_ci_low=ci_low,
            metric_ci_high=ci_high,
            num_examples=total,
        )
    return results


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    depth: int,
    metric_name: str = "accuracy",
    bootstrap_samples: int = 0,
    confidence_level: float = 0.95,
) -> PrefixQualityResult:
    model = model.to(device)
    model.eval()

    loss_sum = 0.0
    correct = 0
    labels_seen: list[int] = []
    predictions_seen: list[int] = []
    total = 0

    for batch in data_loader:
        inputs, labels = move_batch_to_device(batch, device)
        logits = model(**inputs) if isinstance(inputs, dict) else model(inputs)
        predictions = logits.argmax(dim=1)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        correct += int((predictions == labels).sum().item())
        predictions_seen.extend(int(value) for value in predictions.cpu().tolist())
        labels_seen.extend(int(value) for value in labels.cpu().tolist())
        total += labels.numel()

    if total == 0:
        raise ValueError("data_loader produced no examples")

    ci_low, ci_high = bootstrap_confidence_interval(
        metric_name,
        labels_seen,
        predictions_seen,
        num_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=depth * 1013,
    )
    return PrefixQualityResult(
        depth=depth,
        loss=loss_sum / total,
        accuracy=correct / total,
        metric_name=metric_name,
        metric_value=compute_metric(metric_name, labels_seen, predictions_seen),
        metric_ci_low=ci_low,
        metric_ci_high=ci_high,
        num_examples=total,
    )
