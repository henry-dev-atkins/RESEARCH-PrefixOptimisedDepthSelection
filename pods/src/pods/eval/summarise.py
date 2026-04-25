from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def _group_numeric_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(float(row["value"]))
    return {group_key: _mean_std(values) for group_key, values in sorted(grouped.items(), key=lambda item: int(item[0]))}


def _summarise_selection_results(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for depth, metrics in run.get("selection_results", {}).items():
            rows.append({"depth": int(depth), "metric": "accuracy", "value": metrics["accuracy"]})
            rows.append({"depth": int(depth), "metric": "loss", "value": metrics["loss"]})
    metrics_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metrics_by_name[row["metric"]].append(row)
    return {
        metric_name: _group_numeric_by_key(metric_rows, "depth")
        for metric_name, metric_rows in sorted(metrics_by_name.items())
    }


def _summarise_cost_results(runs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for depth, batch_results in run.get(key, {}).items():
            for batch_size, metrics in batch_results.items():
                for metric_name in ("latency_ms_p50", "latency_ms_p95", "throughput_items_per_s", "peak_memory_mb", "model_bytes"):
                    rows.append(
                        {
                            "depth": int(depth),
                            "batch_size": int(batch_size),
                            "metric": metric_name,
                            "value": metrics[metric_name],
                        }
                    )
    metrics_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metrics_by_name[row["metric"]].append(row)
    return {
        metric_name: {
            batch_size: _group_numeric_by_key(batch_rows, "depth")
            for batch_size, batch_rows in sorted(
                ((str(group_batch_size), [row for row in metric_rows if str(row["batch_size"]) == str(group_batch_size)]) for group_batch_size in {row["batch_size"] for row in metric_rows}),
                key=lambda item: int(item[0]),
            )
        }
        for metric_name, metric_rows in sorted(metrics_by_name.items())
    }


def _summarise_final_results(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for depth, metrics in run.get("final_results", {}).items():
            rows.append({"depth": int(depth), "metric": "accuracy", "value": metrics["accuracy"]})
            rows.append({"depth": int(depth), "metric": "loss", "value": metrics["loss"]})
    metrics_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metrics_by_name[row["metric"]].append(row)
    return {
        metric_name: _group_numeric_by_key(metric_rows, "depth")
        for metric_name, metric_rows in sorted(metrics_by_name.items())
    }


def summarise_results(raw_root: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result_path in raw_root.glob("*/results.json"):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if "task" not in payload or "model" not in payload or "baseline" not in payload:
            continue
        key = (
            payload["task"]["dataset"],
            payload["model"]["family"],
            payload["baseline"]["kind"],
        )
        grouped[key].append(payload)

    summary: dict[str, Any] = {}
    for key, runs in grouped.items():
        dataset, model_family, baseline_kind = key
        final_accuracies = [next(iter(run["final_results"].values()))["accuracy"] for run in runs]
        summary[f"{dataset}:{model_family}:{baseline_kind}"] = {
            "num_runs": len(runs),
            "seeds": [run["experiment"]["seed"] for run in runs if "experiment" in run and "seed" in run["experiment"]],
            "selected_depths": [run["selected_depth"] for run in runs],
            "selection_quality_metric": runs[0].get("selection_spec", {}).get("quality_metric"),
            "selection_cost_metric": runs[0].get("selection_spec", {}).get("cost_metric"),
            "final_accuracy_mean": mean(final_accuracies),
            "final_accuracy_std": pstdev(final_accuracies) if len(final_accuracies) > 1 else 0.0,
            "selection_results_by_depth": _summarise_selection_results(runs),
            "selection_costs_by_depth_and_batch_size": _summarise_cost_results(runs, "cost_results"),
            "final_results_by_depth": _summarise_final_results(runs),
            "final_costs_by_batch_size": _summarise_cost_results(
                [
                    {
                        "final_costs": {
                            str(run["selected_depth"]): {
                                str(batch_size): metrics for batch_size, metrics in run.get("final_costs", {}).items()
                            }
                        }
                    }
                    for run in runs
                ],
                "final_costs",
            ),
        }
    return summary


def summarise_by_dataset(run_index: dict[str, Any], dataset: str) -> dict[str, Any]:
    return {key: value for key, value in run_index.items() if key.startswith(f"{dataset}:")}
