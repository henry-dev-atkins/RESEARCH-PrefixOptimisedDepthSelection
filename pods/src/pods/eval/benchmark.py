from __future__ import annotations

import io
import statistics
import time

import torch
from torch import nn

from pods.eval.select import PrefixCostResult


def serialised_model_size_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


@torch.no_grad()
def benchmark_prefix(
    model: nn.Module,
    sample_batch: torch.Tensor,
    warmup_steps: int,
    timed_steps: int,
    device: torch.device,
    depth: int = -1,
) -> PrefixCostResult:
    if warmup_steps < 0 or timed_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and timed_steps must be positive")

    model = model.to(device).eval()
    sample_batch = sample_batch.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup_steps):
        _ = model(sample_batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings = []
    for _ in range(timed_steps):
        t0 = time.perf_counter()
        _ = model(sample_batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - t0) * 1000.0)

    latency_p95 = timings[0] if len(timings) < 2 else statistics.quantiles(timings, n=100)[94]
    peak_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    throughput = sample_batch.shape[0] / (sum(timings) / len(timings) / 1000.0)

    return PrefixCostResult(
        depth=depth,
        latency_ms_p50=statistics.median(timings),
        latency_ms_p95=latency_p95,
        throughput_items_per_s=throughput,
        peak_memory_mb=peak_memory_bytes / (1024 * 1024),
        model_bytes=serialised_model_size_bytes(model),
    )
