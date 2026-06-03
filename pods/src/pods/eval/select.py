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
    latency_ms_p50_ci_low: float | None = None
    latency_ms_p50_ci_high: float | None = None
    latency_ms_p95_ci_low: float | None = None
    latency_ms_p95_ci_high: float | None = None
    benchmark_repeats: int = 1


@dataclass(frozen=True)
class SelectionSpec:
    quality_metric: str
    cost_metric: Literal["latency_ms_p50", "latency_ms_p95", "peak_memory_mb", "model_bytes"]
    target_kind: Literal["relative", "absolute", "confidence_bound_relative", "confidence_bound_absolute"]
    relative_to_full_accuracy_drop: float | None
    absolute_accuracy_target: float | None


def quality_metric_value(result: PrefixQualityResult, metric_name: str, use_lower_confidence_bound: bool = False) -> float:
    if result.metric_name != metric_name:
        raise ValueError(f"Expected metric {metric_name!r}, got {result.metric_name!r}")
    if use_lower_confidence_bound:
        if result.metric_ci_low is None:
            raise ValueError("Confidence-bound selection requires metric_ci_low")
        return result.metric_ci_low
    return result.metric_value


def select_prefix_depth(
    quality_results: dict[int, PrefixQualityResult],
    cost_results: dict[int, PrefixCostResult],
    spec: SelectionSpec,
) -> int:
    if not quality_results:
        raise ValueError("quality_results must be non-empty")
    if quality_results.keys() != cost_results.keys():
        raise ValueError("quality_results and cost_results must cover the same depths")
    if spec.target_kind in {"relative", "confidence_bound_relative"} and spec.relative_to_full_accuracy_drop is None:
        raise ValueError("Relative selection requires relative_to_full_accuracy_drop")
    if spec.target_kind in {"absolute", "confidence_bound_absolute"} and spec.absolute_accuracy_target is None:
        raise ValueError("Absolute selection requires absolute_accuracy_target")

    full_depth = max(quality_results)
    full_metric = quality_metric_value(quality_results[full_depth], spec.quality_metric)
    target_metric = (
        full_metric - spec.relative_to_full_accuracy_drop
        if spec.target_kind in {"relative", "confidence_bound_relative"}
        else spec.absolute_accuracy_target
    )
    use_lower_confidence_bound = spec.target_kind in {"confidence_bound_relative", "confidence_bound_absolute"}

    qualifying_depths = [
        depth
        for depth, result in quality_results.items()
        if quality_metric_value(result, spec.quality_metric, use_lower_confidence_bound=use_lower_confidence_bound) >= target_metric
    ]
    if not qualifying_depths:
        summary = {
            depth: quality_metric_value(result, spec.quality_metric, use_lower_confidence_bound=use_lower_confidence_bound)
            for depth, result in sorted(quality_results.items())
        }
        raise ValueError(f"No depth satisfies the quality target {target_metric:.4f}. Metrics: {summary}")

    return min(qualifying_depths, key=lambda depth: (getattr(cost_results[depth], spec.cost_metric), depth))
