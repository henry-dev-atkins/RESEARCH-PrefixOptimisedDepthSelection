from __future__ import annotations

import pytest

from pods.eval.quality import PrefixQualityResult
from pods.eval.select import PrefixCostResult, SelectionSpec, select_prefix_depth


def test_select_prefix_depth_uses_relative_quality_target_and_cost() -> None:
    quality = {
        1: PrefixQualityResult(depth=1, loss=1.0, accuracy=0.84, num_examples=10),
        2: PrefixQualityResult(depth=2, loss=0.8, accuracy=0.89, num_examples=10),
        3: PrefixQualityResult(depth=3, loss=0.7, accuracy=0.90, num_examples=10),
    }
    costs = {
        1: PrefixCostResult(depth=1, latency_ms_p50=1.0, latency_ms_p95=1.1, throughput_items_per_s=100.0, peak_memory_mb=10.0, model_bytes=100),
        2: PrefixCostResult(depth=2, latency_ms_p50=2.0, latency_ms_p95=2.1, throughput_items_per_s=90.0, peak_memory_mb=20.0, model_bytes=200),
        3: PrefixCostResult(depth=3, latency_ms_p50=3.0, latency_ms_p95=3.1, throughput_items_per_s=80.0, peak_memory_mb=30.0, model_bytes=300),
    }
    selected = select_prefix_depth(
        quality,
        costs,
        SelectionSpec(
            quality_metric="accuracy",
            cost_metric="latency_ms_p50",
            relative_to_full_accuracy_drop=0.01,
            absolute_accuracy_target=None,
        ),
    )
    assert selected == 2


def test_select_prefix_depth_uses_shallower_depth_as_tie_break() -> None:
    quality = {
        2: PrefixQualityResult(depth=2, loss=0.8, accuracy=0.90, num_examples=10),
        3: PrefixQualityResult(depth=3, loss=0.7, accuracy=0.91, num_examples=10),
    }
    costs = {
        2: PrefixCostResult(depth=2, latency_ms_p50=2.0, latency_ms_p95=2.0, throughput_items_per_s=90.0, peak_memory_mb=20.0, model_bytes=100),
        3: PrefixCostResult(depth=3, latency_ms_p50=2.0, latency_ms_p95=2.0, throughput_items_per_s=80.0, peak_memory_mb=30.0, model_bytes=100),
    }
    selected = select_prefix_depth(
        quality,
        costs,
        SelectionSpec(
            quality_metric="accuracy",
            cost_metric="model_bytes",
            relative_to_full_accuracy_drop=0.02,
            absolute_accuracy_target=None,
        ),
    )
    assert selected == 2


def test_select_prefix_depth_fails_loudly_when_no_depth_qualifies() -> None:
    quality = {
        1: PrefixQualityResult(depth=1, loss=1.0, accuracy=0.70, num_examples=10),
        2: PrefixQualityResult(depth=2, loss=0.9, accuracy=0.71, num_examples=10),
    }
    costs = {
        1: PrefixCostResult(depth=1, latency_ms_p50=1.0, latency_ms_p95=1.0, throughput_items_per_s=10.0, peak_memory_mb=10.0, model_bytes=100),
        2: PrefixCostResult(depth=2, latency_ms_p50=2.0, latency_ms_p95=2.0, throughput_items_per_s=10.0, peak_memory_mb=20.0, model_bytes=200),
    }
    with pytest.raises(ValueError, match="No depth satisfies"):
        select_prefix_depth(
            quality,
            costs,
            SelectionSpec(
                quality_metric="accuracy",
                cost_metric="latency_ms_p50",
                relative_to_full_accuracy_drop=None,
                absolute_accuracy_target=0.9,
            ),
        )
