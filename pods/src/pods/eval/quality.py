from __future__ import annotations

from dataclasses import dataclass, field

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
    accuracy: float | None
    metric_name: str
    metric_value: float
    metric_ci_low: float | None
    metric_ci_high: float | None
    num_examples: int
    aux_metrics: dict[str, float] = field(default_factory=dict)


def _loss_for_task(logits: torch.Tensor, labels: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type in {"image_classification", "text_classification"}:
        return F.cross_entropy(logits, labels.long(), reduction="sum")
    if task_type == "text_regression":
        return F.mse_loss(logits.reshape(-1), labels.reshape(-1).to(logits.dtype), reduction="sum")
    raise ValueError(f"Unsupported task_type: {task_type}")


def _prediction_payload(logits: torch.Tensor, labels: torch.Tensor, task_type: str) -> tuple[list[int | float], list[int | float], float | None]:
    if task_type in {"image_classification", "text_classification"}:
        predictions = logits.argmax(dim=1)
        labels_list = [int(value) for value in labels.cpu().tolist()]
        predictions_list = [int(value) for value in predictions.cpu().tolist()]
        accuracy = sum(1 for label, prediction in zip(labels_list, predictions_list) if label == prediction) / len(labels_list)
        return labels_list, predictions_list, accuracy
    if task_type == "text_regression":
        predictions = logits.reshape(-1)
        labels_list = [float(value) for value in labels.reshape(-1).cpu().tolist()]
        predictions_list = [float(value) for value in predictions.cpu().tolist()]
        return labels_list, predictions_list, None
    raise ValueError(f"Unsupported task_type: {task_type}")


def _aux_metrics(metric_name: str, task_type: str, labels: list[int | float], predictions: list[int | float], accuracy: float | None) -> dict[str, float]:
    metrics = {metric_name: compute_metric(metric_name, labels, predictions)}
    if task_type in {"image_classification", "text_classification"} and accuracy is not None:
        metrics["accuracy"] = accuracy
        if metric_name != "matthews_correlation" and set(labels).issubset({0, 1}) and set(predictions).issubset({0, 1}):
            metrics["matthews_correlation"] = compute_metric("matthews_correlation", labels, predictions)
    if task_type == "text_regression":
        metrics["pearson_correlation"] = compute_metric("pearson_correlation", labels, predictions)
        metrics["spearman_correlation"] = compute_metric("spearman_correlation", labels, predictions)
    return metrics


@torch.no_grad()
def evaluate_all_prefixes(
    model: PrefixCapableModel,
    data_loader: DataLoader,
    candidate_depths: tuple[int, ...],
    device: torch.device,
    metric_name: str = "accuracy",
    bootstrap_samples: int = 0,
    confidence_level: float = 0.95,
    task_type: str = "image_classification",
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
            loss_sums[depth] += float(_loss_for_task(logits, labels, task_type).item())
            labels_list, predictions_list, batch_accuracy = _prediction_payload(logits, labels, task_type)
            labels_by_depth[depth].extend(labels_list)
            predictions_by_depth[depth].extend(predictions_list)
            if batch_accuracy is not None:
                correct[depth] += int(batch_accuracy * len(labels_list))
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
        accuracy = correct[depth] / total if task_type in {"image_classification", "text_classification"} else None
        aux_metrics = _aux_metrics(metric_name, task_type, labels_by_depth[depth], predictions_by_depth[depth], accuracy)
        results[depth] = PrefixQualityResult(
            depth=depth,
            loss=loss_sums[depth] / total,
            accuracy=accuracy,
            metric_name=metric_name,
            metric_value=aux_metrics[metric_name],
            metric_ci_low=ci_low,
            metric_ci_high=ci_high,
            num_examples=total,
            aux_metrics=aux_metrics,
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
    task_type: str = "image_classification",
) -> PrefixQualityResult:
    model = model.to(device)
    model.eval()

    loss_sum = 0.0
    correct = 0
    labels_seen: list[int | float] = []
    predictions_seen: list[int | float] = []
    total = 0

    for batch in data_loader:
        inputs, labels = move_batch_to_device(batch, device)
        logits = model(**inputs) if isinstance(inputs, dict) else model(inputs)
        loss_sum += float(_loss_for_task(logits, labels, task_type).item())
        batch_labels, batch_predictions, batch_accuracy = _prediction_payload(logits, labels, task_type)
        labels_seen.extend(batch_labels)
        predictions_seen.extend(batch_predictions)
        if batch_accuracy is not None:
            correct += int(batch_accuracy * len(batch_labels))
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
    accuracy = correct / total if task_type in {"image_classification", "text_classification"} else None
    aux_metrics = _aux_metrics(metric_name, task_type, labels_seen, predictions_seen, accuracy)
    return PrefixQualityResult(
        depth=depth,
        loss=loss_sum / total,
        accuracy=accuracy,
        metric_name=metric_name,
        metric_value=aux_metrics[metric_name],
        metric_ci_low=ci_low,
        metric_ci_high=ci_high,
        num_examples=total,
        aux_metrics=aux_metrics,
    )
