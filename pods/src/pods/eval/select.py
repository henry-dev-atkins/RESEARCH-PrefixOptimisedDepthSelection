from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pods.eval.quality import PrefixQualityResult


@dataclass(frozen=True)
class PrefixCostResult:
    depth: int
    latency_ms_p50: float
    latency_ms_p95: float
    throughput_items_per_s: float
    peak_memory_mb: float
    model_bytes: int


@dataclass(frozen=True)
class SelectionSpec:
    quality_metric: Literal["accuracy"]
    cost_metric: Literal["latency_ms_p50", "latency_ms_p95", "peak_memory_mb", "model_bytes"]
    relative_to_full_accuracy_drop: float | None
    absolute_accuracy_target: float | None


def select_prefix_depth(
    quality_results: dict[int, PrefixQualityResult],
    cost_results: dict[int, PrefixCostResult],
    spec: SelectionSpec,
) -> int:
    if not quality_results:
        raise ValueError("quality_results must be non-empty")
    if quality_results.keys() != cost_results.keys():
        raise ValueError("quality_results and cost_results must cover the same depths")
    if (spec.relative_to_full_accuracy_drop is None) == (spec.absolute_accuracy_target is None):
        raise ValueError("Exactly one selection target must be set")

    full_depth = max(quality_results)
    full_accuracy = quality_results[full_depth].accuracy
    target_accuracy = (
        full_accuracy - spec.relative_to_full_accuracy_drop
        if spec.relative_to_full_accuracy_drop is not None
        else spec.absolute_accuracy_target
    )

    qualifying_depths = [depth for depth, result in quality_results.items() if result.accuracy >= target_accuracy]
    if not qualifying_depths:
        summary = {depth: result.accuracy for depth, result in sorted(quality_results.items())}
        raise ValueError(f"No depth satisfies the quality target {target_accuracy:.4f}. Accuracies: {summary}")

    return min(qualifying_depths, key=lambda depth: (getattr(cost_results[depth], spec.cost_metric), depth))
