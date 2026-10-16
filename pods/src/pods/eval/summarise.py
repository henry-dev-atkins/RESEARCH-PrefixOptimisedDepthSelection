from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "std": pstdev(values) if len(values) > 1 else 0.0}


def _group_numeric_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(float(row["value"]))
    return {group_key: _mean_std(values) for group_key, values in sorted(grouped.items(), key=lambda item: int(item[0]))}


def _quality_rows(run_section: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for depth, metrics in run_section.items():
        rows.append({"depth": int(depth), "metric": "loss", "value": metrics["loss"]})
        if metrics.get("accuracy") is not None:
            rows.append({"depth": int(depth), "metric": "accuracy", "value": metrics["accuracy"]})
        metric_name = metrics.get("metric_name")
        metric_value = metrics.get("metric_value")
        if metric_name is not None and metric_value is not None:
            rows.append({"depth": int(depth), "metric": metric_name, "value": metric_value})
        for aux_name, aux_value in metrics.get("aux_metrics", {}).items():
            rows.append({"depth": int(depth), "metric": aux_name, "value": aux_value})
    return rows


def _summarise_quality_results(runs: list[dict[str, Any]], section_name: str) -> dict[str, Any]:
    metrics_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for row in _quality_rows(run.get(section_name, {})):
            metrics_by_name[row["metric"]].append(row)
    return {metric_name: _group_numeric_by_key(metric_rows, "depth") for metric_name, metric_rows in sorted(metrics_by_name.items())}


def _summarise_cost_results(runs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for depth, batch_results in run.get(key, {}).items():
            for batch_size, metrics in batch_results.items():
                for metric_name in (
                    "latency_ms_p50",
                    "latency_ms_p95",
                    "latency_ms_p50_ci_low",
                    "latency_ms_p50_ci_high",
                    "latency_ms_p95_ci_low",
                    "latency_ms_p95_ci_high",
                    "throughput_items_per_s",
                    "peak_memory_mb",
                    "model_bytes",
                ):
                    if metrics.get(metric_name) is None:
                        continue
                    rows.append({"depth": int(depth), "batch_size": int(batch_size), "metric": metric_name, "value": metrics[metric_name]})
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


def _summary_variant_key(payload: dict[str, Any]) -> str:
    baseline_kind = payload["baseline"]["kind"]
    if baseline_kind == "pods":
        training_mode = payload.get("training", {}).get("mode")
        if training_mode == "all":
            return "pods_all"
        if training_mode == "sandwich":
            return "pods_sandwich"
    if baseline_kind == "layerdrop":
        return "layerdrop"
    if baseline_kind in {"shallow_independent", "distilled_shallow"}:
        target_depth = payload.get("baseline", {}).get("target_depth")
        if target_depth is not None:
            prefix = "distilled_shallow" if baseline_kind == "distilled_shallow" else "shallow_independent"
            return f"{prefix}_d{int(target_depth)}"
    return baseline_kind


def summarise_results(raw_root: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result_path in raw_root.glob("*/results.json"):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if "task" not in payload or "model" not in payload or "baseline" not in payload:
            continue
        key = (payload["task"]["dataset"], payload["model"]["family"], _summary_variant_key(payload))
        grouped[key].append(payload)

    summary: dict[str, Any] = {}
    for key, runs in grouped.items():
        dataset, model_family, baseline_kind = key
        first_final_result = next(iter(runs[0]["final_results"].values()))
        task_metric_name = first_final_result.get("metric_name", "accuracy")
        final_metric_values = [next(iter(run["final_results"].values())).get("metric_value", next(iter(run["final_results"].values())).get("accuracy", 0.0)) for run in runs]
        summary[f"{dataset}:{model_family}:{baseline_kind}"] = {
            "num_runs": len(runs),
            "seeds": [run["experiment"]["seed"] for run in runs if "experiment" in run and "seed" in run["experiment"]],
            "selected_depths": [run["selected_depth"] for run in runs],
            "selection_quality_metric": runs[0].get("selection_spec", {}).get("quality_metric"),
            "selection_cost_metric": runs[0].get("selection_spec", {}).get("cost_metric"),
            "evaluation_splits": runs[0].get("evaluation_splits", {}),
            "task_metric_name": task_metric_name,
            "final_metric_mean": mean(final_metric_values),
            "final_metric_std": pstdev(final_metric_values) if len(final_metric_values) > 1 else 0.0,
            "final_accuracy_mean": mean(final_metric_values),
            "final_accuracy_std": pstdev(final_metric_values) if len(final_metric_values) > 1 else 0.0,
            "selection_results_by_depth": _summarise_quality_results(runs, "selection_results"),
            "selection_costs_by_depth_and_batch_size": _summarise_cost_results(runs, "cost_results"),
            "final_results_by_depth": _summarise_quality_results(runs, "final_results"),
            "final_costs_by_batch_size": _summarise_cost_results(
                [{"final_costs": {str(run["selected_depth"]): {str(batch_size): metrics for batch_size, metrics in run.get("final_costs", {}).items()}}} for run in runs],
                "final_costs",
            ),
        }
        if any("final_results_secondary" in run for run in runs):
            summary[f"{dataset}:{model_family}:{baseline_kind}"]["final_results_secondary_by_depth"] = _summarise_quality_results(
                [{"final_results": run.get("final_results_secondary", {})} for run in runs],
                "final_results",
            )
    return summary


def summarise_by_dataset(run_index: dict[str, Any], dataset: str) -> dict[str, Any]:
    return {key: value for key, value in run_index.items() if key.startswith(f"{dataset}:")}
