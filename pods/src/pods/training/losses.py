from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def regression_loss(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predictions.reshape(-1), labels.reshape(-1).to(predictions.dtype))


def supervised_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: str,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if task_type in {"image_classification", "text_classification"}:
        return classification_loss(logits, labels.long(), label_smoothing=label_smoothing)
    if task_type == "text_regression":
        return regression_loss(logits, labels)
    raise ValueError(f"Unsupported task_type: {task_type}")


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    task_type: str,
    temperature: float,
) -> torch.Tensor:
    if task_type in {"image_classification", "text_classification"}:
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)
    if task_type == "text_regression":
        return regression_loss(student_logits, teacher_logits)
    raise ValueError(f"Unsupported task_type: {task_type}")


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
