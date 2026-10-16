from __future__ import annotations

import statistics
import time
from typing import Any

import torch
from torch import nn

from pods.eval.select import PrefixCostResult


def model_size_bytes(model: nn.Module) -> int:
    """Return the bytes occupied by the tensors in a model state."""

    return sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())


@torch.no_grad()
def benchmark_prefix(
    model: nn.Module,
    sample_batch: Any,
    warmup_steps: int,
    timed_steps: int,
    device: torch.device,
    depth: int = -1,
) -> PrefixCostResult:
    if warmup_steps < 0 or timed_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and timed_steps must be positive")

    model = model.to(device).eval()
    if isinstance(sample_batch, dict):
        sample_batch = {key: value.to(device) for key, value in sample_batch.items()}
        batch_size = next(iter(sample_batch.values())).shape[0]
    else:
        sample_batch = sample_batch.to(device)
        batch_size = sample_batch.shape[0]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup_steps):
        _ = model(**sample_batch) if isinstance(sample_batch, dict) else model(sample_batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings = []
    for _ in range(timed_steps):
        t0 = time.perf_counter()
        _ = model(**sample_batch) if isinstance(sample_batch, dict) else model(sample_batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - t0) * 1000.0)

    latency_p95 = timings[0] if len(timings) < 2 else statistics.quantiles(timings, n=100)[94]
    peak_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    throughput = batch_size / (sum(timings) / len(timings) / 1000.0)

    return PrefixCostResult(
        depth=depth,
        latency_ms_p50=statistics.median(timings),
        latency_ms_p95=latency_p95,
        latency_ms_p50_ci_low=None,
        latency_ms_p50_ci_high=None,
        latency_ms_p95_ci_low=None,
        latency_ms_p95_ci_high=None,
        throughput_items_per_s=throughput,
        peak_memory_mb=peak_memory_bytes / (1024 * 1024),
        model_bytes=model_size_bytes(model),
        benchmark_repeats=1,
    )
