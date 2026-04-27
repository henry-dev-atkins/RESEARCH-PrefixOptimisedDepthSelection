from __future__ import annotations

import math
import random
from typing import Callable


def accuracy_score(labels: list[int], predictions: list[int]) -> float:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    if not labels:
        raise ValueError("labels must be non-empty")
    correct = sum(1 for label, prediction in zip(labels, predictions) if label == prediction)
    return correct / len(labels)


def matthews_correlation(labels: list[int], predictions: list[int]) -> float:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    if not labels:
        raise ValueError("labels must be non-empty")

    true_positive = sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 1)
    true_negative = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 0)
    false_positive = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 1)
    false_negative = sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 0)

    denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    if denominator == 0.0:
        return 0.0
    numerator = (true_positive * true_negative) - (false_positive * false_negative)
    return numerator / denominator


def metric_fn(metric_name: str) -> Callable[[list[int], list[int]], float]:
    if metric_name == "accuracy":
        return accuracy_score
    if metric_name == "matthews_correlation":
        return matthews_correlation
    raise ValueError(f"Unsupported metric_name: {metric_name}")


def compute_metric(metric_name: str, labels: list[int], predictions: list[int]) -> float:
    return metric_fn(metric_name)(labels, predictions)


def bootstrap_confidence_interval(
    metric_name: str,
    labels: list[int],
    predictions: list[int],
    num_samples: int,
    confidence_level: float,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    if num_samples <= 0:
        return None, None
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    if not labels:
        raise ValueError("labels must be non-empty")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = random.Random(seed)
    scorer = metric_fn(metric_name)
    values: list[float] = []
    size = len(labels)
    for _ in range(num_samples):
        sample_indices = [rng.randrange(size) for _ in range(size)]
        sample_labels = [labels[index] for index in sample_indices]
        sample_predictions = [predictions[index] for index in sample_indices]
        values.append(scorer(sample_labels, sample_predictions))

    values.sort()
    alpha = (1.0 - confidence_level) / 2.0
    low_index = max(0, min(len(values) - 1, int(alpha * len(values))))
    high_index = max(0, min(len(values) - 1, int((1.0 - alpha) * len(values)) - 1))
    return values[low_index], values[high_index]
