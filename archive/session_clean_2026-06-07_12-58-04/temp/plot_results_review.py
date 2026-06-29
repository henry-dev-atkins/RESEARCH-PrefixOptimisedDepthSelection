from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


VARIANT_COLOURS = {
    "full": "#1b1b1b",
    "naive_truncation": "#b04a3a",
    "shallow_independent_d1": "#0f766e",
    "shallow_independent_d2": "#155e75",
    "shallow_independent_d4": "#0b7285",
    "layerdrop": "#c2410c",
    "pods_all": "#1d4ed8",
    "pods_sandwich": "#7c3aed",
}

VARIANT_MARKERS = {
    "full": "o",
    "naive_truncation": "s",
    "shallow_independent_d1": "^",
    "shallow_independent_d2": "v",
    "shallow_independent_d4": "X",
    "layerdrop": "*",
    "pods_all": "D",
    "pods_sandwich": "P",
}

DISPLAY_NAMES = {
    "full": "Full",
    "naive_truncation": "Naive truncation",
    "shallow_independent_d1": "Shallow d1",
    "shallow_independent_d2": "Shallow d2",
    "shallow_independent_d4": "Shallow d4",
    "layerdrop": "LayerDrop",
    "pods_all": "PODS all",
    "pods_sandwich": "PODS sandwich",
}

ALL_DATASETS = [
    "mnist",
    "cifar10",
    "cifar100",
    "sst2",
    "mrpc",
    "rte",
    "qqp",
    "qnli",
    "cola",
]

MILESTONE4_TASKS = ["sst2", "mrpc", "rte", "qqp", "qnli", "cola"]
NEW_TASKS = ["qqp", "qnli", "cola"]


@dataclass
class VariantSummary:
    dataset: str
    variant: str
    metric_name: str
    num_runs: int
    seeds: list[int]
    selected_depths: list[int]
    final_metric_mean: float
    final_metric_std: float
    selection_results_by_depth: dict[str, dict[str, dict[str, float]]]
    selection_costs_by_depth_and_batch_size: dict[str, dict[str, dict[str, dict[str, float]]]]
    final_results_by_depth: dict[str, dict[str, dict[str, float]]]
    final_costs_by_batch_size: dict[str, dict[str, dict[str, dict[str, float]]]]


def _load_summary(path: Path) -> dict[str, VariantSummary]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary: dict[str, VariantSummary] = {}
    for key, entry in payload.items():
        dataset, _, variant = key.split(":", 2)
        summary[variant] = VariantSummary(
            dataset=dataset,
            variant=variant,
            metric_name=str(entry["task_metric_name"]),
            num_runs=int(entry["num_runs"]),
            seeds=[int(seed) for seed in entry.get("seeds", [])],
            selected_depths=[int(depth) for depth in entry.get("selected_depths", [])],
            final_metric_mean=float(entry["final_metric_mean"]),
            final_metric_std=float(entry["final_metric_std"]),
            selection_results_by_depth=entry["selection_results_by_depth"],
            selection_costs_by_depth_and_batch_size=entry["selection_costs_by_depth_and_batch_size"],
            final_results_by_depth=entry["final_results_by_depth"],
            final_costs_by_batch_size=entry["final_costs_by_batch_size"],
        )
    return summary


def load_processed_summaries(processed_root: Path, datasets: list[str]) -> dict[str, dict[str, VariantSummary]]:
    summaries: dict[str, dict[str, VariantSummary]] = {}
    for dataset in datasets:
        summary_path = processed_root / f"{dataset}_frontier_summary.json"
        if not summary_path.exists():
            continue
        summaries[dataset] = _load_summary(summary_path)
    return summaries


def _ensure_metric_depths(variant: VariantSummary) -> list[int]:
    metric_rows = variant.final_results_by_depth.get(variant.metric_name)
    if not metric_rows:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing final metric rows for {variant.metric_name}")
    return sorted(int(depth) for depth in metric_rows.keys())


def _ensure_selection_metric_depths(variant: VariantSummary) -> list[int]:
    metric_rows = variant.selection_results_by_depth.get(variant.metric_name)
    if not metric_rows:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing selection metric rows for {variant.metric_name}")
    return sorted(int(depth) for depth in metric_rows.keys())


def _depth_metric(variant: VariantSummary, depth: int) -> float:
    metric_rows = variant.final_results_by_depth.get(variant.metric_name)
    if metric_rows is None or str(depth) not in metric_rows:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing metric depth {depth}")
    return float(metric_rows[str(depth)]["mean"])


def _depth_metric_std(variant: VariantSummary, depth: int) -> float:
    metric_rows = variant.final_results_by_depth.get(variant.metric_name)
    if metric_rows is None or str(depth) not in metric_rows:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing metric std depth {depth}")
    return float(metric_rows[str(depth)]["std"])


def _selection_depth_metric(variant: VariantSummary, depth: int) -> float:
    metric_rows = variant.selection_results_by_depth.get(variant.metric_name)
    if metric_rows is None or str(depth) not in metric_rows:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing selection metric depth {depth}")
    return float(metric_rows[str(depth)]["mean"])


def _selection_depth_metric_std(variant: VariantSummary, depth: int) -> float:
    metric_rows = variant.selection_results_by_depth.get(variant.metric_name)
    if metric_rows is None or str(depth) not in metric_rows:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing selection metric std depth {depth}")
    return float(metric_rows[str(depth)]["std"])


def _depth_cost(variant: VariantSummary, metric_name: str, batch_size: int, depth: int) -> float:
    metrics = variant.final_costs_by_batch_size.get(metric_name)
    if metrics is None:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing cost metric {metric_name}")
    batch_rows = metrics.get(str(batch_size))
    if batch_rows is None or str(depth) not in batch_rows:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing cost depth {depth} for batch {batch_size}")
    return float(batch_rows[str(depth)]["mean"])


def _depth_cost_std(variant: VariantSummary, metric_name: str, batch_size: int, depth: int) -> float:
    metrics = variant.final_costs_by_batch_size.get(metric_name)
    if metrics is None:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing cost std metric {metric_name}")
    batch_rows = metrics.get(str(batch_size))
    if batch_rows is None or str(depth) not in batch_rows:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing cost std depth {depth} for batch {batch_size}")
    return float(batch_rows[str(depth)]["std"])


def _selection_depth_cost(variant: VariantSummary, metric_name: str, batch_size: int, depth: int) -> float:
    metrics = variant.selection_costs_by_depth_and_batch_size.get(metric_name)
    if metrics is None:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing selection cost metric {metric_name}")
    batch_rows = metrics.get(str(batch_size))
    if batch_rows is None or str(depth) not in batch_rows:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing selection cost depth {depth} for batch {batch_size}")
    return float(batch_rows[str(depth)]["mean"])


def _selection_depth_cost_std(variant: VariantSummary, metric_name: str, batch_size: int, depth: int) -> float:
    metrics = variant.selection_costs_by_depth_and_batch_size.get(metric_name)
    if metrics is None:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing selection cost std metric {metric_name}")
    batch_rows = metrics.get(str(batch_size))
    if batch_rows is None or str(depth) not in batch_rows:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing selection cost std depth {depth} for batch {batch_size}")
    return float(batch_rows[str(depth)]["std"])


def _selected_depth_mean(variant: VariantSummary) -> float:
    if not variant.selected_depths:
        raise ValueError(f"{variant.dataset}:{variant.variant} has no selected depths")
    return sum(variant.selected_depths) / len(variant.selected_depths)


def _representative_selected_depth(variant: VariantSummary) -> int:
    if not variant.selected_depths:
        raise ValueError(f"{variant.dataset}:{variant.variant} has no selected depths")
    counts: dict[int, int] = {}
    for depth in variant.selected_depths:
        counts[depth] = counts.get(depth, 0) + 1
    best_count = max(counts.values())
    candidates = sorted(depth for depth, count in counts.items() if count == best_count)
    return candidates[0]


def _selected_depth_label(variant: VariantSummary) -> str:
    if not variant.selected_depths:
        return "n/a"
    unique = sorted(set(variant.selected_depths))
    if len(unique) == 1:
        return f"d{unique[0]}"
    return ",".join(f"d{depth}" for depth in unique)


def _best_pods_variant(dataset_summary: dict[str, VariantSummary]) -> str | None:
    candidates = [variant for variant in ("pods_all", "pods_sandwich") if variant in dataset_summary]
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda name: (
            dataset_summary[name].final_metric_mean,
            -_selected_depth_mean(dataset_summary[name]),
        ),
        reverse=True,
    )
    return ranked[0]


def _save_figure(fig: plt.Figure, output_base: Path, formats: list[str]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=200 if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)


def render_pods_flow_diagram(output_root: Path, formats: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.set_axis_off()

    boxes = [
        (0.03, 0.25, 0.2, 0.5, "#e5eefc", "Train one model\n\nShared deep backbone\nHeads at candidate depths"),
        (0.28, 0.25, 0.2, 0.5, "#e9f8f3", "Measure each prefix\n\nHeld-out metric\nLatency, memory,\nthroughput"),
        (0.53, 0.25, 0.2, 0.5, "#fef3c7", "Select depth\n\nShallowest prefix\nmeeting quality target"),
        (0.78, 0.25, 0.19, 0.5, "#f3e8ff", "Deploy fixed prefix\n\nDiscard deeper layers\nNo dynamic exit"),
    ]

    for x, y, w, h, colour, text in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=colour, edgecolor="#334155", linewidth=1.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12, transform=ax.transAxes)

    arrow_y = 0.5
    for x_start, x_end in ((0.23, 0.28), (0.48, 0.53), (0.73, 0.78)):
        ax.annotate("", xy=(x_end, arrow_y), xytext=(x_start, arrow_y), xycoords="axes fraction", textcoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 2, "color": "#475569"})

    ax.text(0.5, 0.95, "PODS training, selection, and deployment flow", ha="center", va="top", fontsize=16, weight="bold", transform=ax.transAxes)
    _save_figure(fig, output_root / "overview" / "pods_flow_diagram", formats)


def render_task_outcome_grid(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in ALL_DATASETS:
        dataset_summary = summaries.get(dataset, {})
        if not dataset_summary:
            continue
        best_pods = _best_pods_variant(dataset_summary)
        if best_pods is None:
            continue
        best_variant = dataset_summary[best_pods]
        full = dataset_summary.get("full")
        naive = dataset_summary.get("naive_truncation")
        delta_vs_full = best_variant.final_metric_mean - full.final_metric_mean if full else math.nan
        delta_vs_naive = best_variant.final_metric_mean - naive.final_metric_mean if naive else math.nan
        selected_depth_mean = _selected_depth_mean(best_variant)
        if selected_depth_mean < max(best_variant.selected_depths):
            selected_depth_outcome = "shallower"
        elif selected_depth_mean < 12:
            selected_depth_outcome = "shallower"
        else:
            selected_depth_outcome = "full-depth"

        if not math.isnan(delta_vs_full) and delta_vs_full > 0 and (math.isnan(delta_vs_naive) or delta_vs_naive > 0) and selected_depth_outcome == "shallower":
            verdict = "clear win"
        elif not math.isnan(delta_vs_full) and delta_vs_full >= 0 and selected_depth_outcome == "shallower":
            verdict = "partial win"
        elif not math.isnan(delta_vs_naive) and delta_vs_naive < 0:
            verdict = "mixed"
        else:
            verdict = "loss"

        rows.append(
            {
                "dataset": dataset,
                "best_pods": best_pods,
                "delta_vs_full": delta_vs_full,
                "delta_vs_naive": delta_vs_naive,
                "selected_depth": _selected_depth_label(best_variant),
                "verdict": verdict,
            }
        )

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.set_axis_off()
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["dataset"].upper(),
                DISPLAY_NAMES.get(row["best_pods"], row["best_pods"]),
                "n/a" if math.isnan(row["delta_vs_full"]) else f"{row['delta_vs_full']:+.4f}",
                "n/a" if math.isnan(row["delta_vs_naive"]) else f"{row['delta_vs_naive']:+.4f}",
                row["selected_depth"],
                row["verdict"],
            ]
        )
    table = ax.table(
        cellText=table_rows,
        colLabels=["Task", "Best PODS", "Delta vs full", "Delta vs truncation", "Selected depth", "Verdict"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e2e8f0")
            continue
        verdict = table_rows[row_idx - 1][-1]
        if col_idx == 5:
            colour = {"clear win": "#bbf7d0", "partial win": "#fef3c7", "mixed": "#fde68a", "loss": "#fecaca"}.get(verdict, "#ffffff")
            cell.set_facecolor(colour)
    ax.set_title("Outcome summary across PODS results so far", fontsize=15, weight="bold", pad=20)
    _save_figure(fig, output_root / "overview" / "task_outcome_grid", formats)
    return rows


def render_frontier_plots(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    for dataset, dataset_summary in summaries.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        seen_labels: set[str] = set()
        for variant_name, variant in sorted(dataset_summary.items()):
            for depth in _ensure_metric_depths(variant):
                latency = _depth_cost(variant, "latency_ms_p50", 1, depth)
                latency_std = _depth_cost_std(variant, "latency_ms_p50", 1, depth)
                metric = _depth_metric(variant, depth)
                metric_std = _depth_metric_std(variant, depth)
                ax.errorbar(
                    latency,
                    metric,
                    xerr=latency_std,
                    yerr=metric_std,
                    fmt=VARIANT_MARKERS.get(variant_name, "o"),
                    color=VARIANT_COLOURS.get(variant_name, "#475569"),
                    markersize=7,
                    capsize=3,
                    linestyle="none",
                    alpha=0.95,
                    label=DISPLAY_NAMES.get(variant_name, variant_name) if variant_name not in seen_labels else None,
                )
                ax.annotate(f"d{depth}", (latency, metric), xytext=(4, 4), textcoords="offset points", fontsize=8)
                seen_labels.add(variant_name)
        ax.set_title(f"{dataset.upper()} measured frontier", fontsize=13, weight="bold")
        ax.set_xlabel("Measured p50 latency (ms), batch 1")
        ax.set_ylabel(dataset_summary[next(iter(dataset_summary))].metric_name.replace("_", " ").title())
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        _save_figure(fig, output_root / "frontiers" / f"{dataset}_frontier", formats)


def render_m4_main_panel(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    variants = ["full", "pods_all", "pods_sandwich"]
    for ax, dataset in zip(axes, MILESTONE4_TASKS):
        dataset_summary = summaries[dataset]
        xs = range(len(variants))
        values = [dataset_summary[name].final_metric_mean for name in variants if name in dataset_summary]
        errors = [dataset_summary[name].final_metric_std for name in variants if name in dataset_summary]
        labels = [name for name in variants if name in dataset_summary]
        colours = [VARIANT_COLOURS[name] for name in labels]
        ax.bar(range(len(labels)), values, yerr=errors, color=colours, capsize=4)
        ax.set_xticks(range(len(labels)), [DISPLAY_NAMES[name] for name in labels], rotation=20, ha="right")
        ax.set_title(dataset.upper(), weight="bold")
        ax.set_ylabel(dataset_summary["full"].metric_name.replace("_", " ").title())
        ax.grid(True, axis="y", alpha=0.25)
        for idx, name in enumerate(labels):
            ax.text(idx, values[idx], _selected_depth_label(dataset_summary[name]), ha="center", va="bottom", fontsize=9)
    fig.suptitle("Milestone 4 main-variant outcomes across six GLUE tasks", fontsize=16, weight="bold")
    fig.tight_layout()
    _save_figure(fig, output_root / "milestone4" / "m4_main_variants_panel", formats)


def render_m4_baseline_panel(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, dataset in zip(axes, NEW_TASKS):
        dataset_summary = summaries[dataset]
        best_pods = _best_pods_variant(dataset_summary)
        ordered = ["full", best_pods, "naive_truncation", "layerdrop", "shallow_independent_d2", "shallow_independent_d4"]
        labels = [name for name in ordered if name and name in dataset_summary]
        values = [dataset_summary[name].final_metric_mean for name in labels]
        errors = [dataset_summary[name].final_metric_std for name in labels]
        colours = [VARIANT_COLOURS.get(name, "#475569") for name in labels]
        ax.bar(range(len(labels)), values, yerr=errors, color=colours, capsize=4)
        ax.set_xticks(range(len(labels)), [DISPLAY_NAMES.get(name, name) for name in labels], rotation=25, ha="right")
        ax.set_title(dataset.upper(), weight="bold")
        ax.set_ylabel(dataset_summary["full"].metric_name.replace("_", " ").title())
        ax.grid(True, axis="y", alpha=0.25)
        best_value = max(values)
        for idx, name in enumerate(labels):
            if values[idx] == best_value:
                ax.text(idx, values[idx], f"{_selected_depth_label(dataset_summary[name])}\nBEST", ha="center", va="bottom", fontsize=8, weight="bold")
            else:
                ax.text(idx, values[idx], _selected_depth_label(dataset_summary[name]), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Milestone 4 baseline comparisons on new GLUE tasks", fontsize=16, weight="bold")
    fig.tight_layout()
    _save_figure(fig, output_root / "milestone4" / "m4_baseline_comparison", formats)


def render_selected_depth_plot(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    rows: list[tuple[str, str, list[int]]] = []
    ordered_variants = ["full", "pods_all", "pods_sandwich", "naive_truncation", "layerdrop"]
    for dataset in MILESTONE4_TASKS:
        dataset_summary = summaries.get(dataset, {})
        for variant in ordered_variants:
            if variant in dataset_summary:
                rows.append((dataset, variant, dataset_summary[variant].selected_depths))

    fig_height = max(6, len(rows) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    yticks = []
    ylabels = []
    for idx, (dataset, variant, depths) in enumerate(rows):
        y = len(rows) - idx
        yticks.append(y)
        ylabels.append(f"{dataset.upper()} {DISPLAY_NAMES.get(variant, variant)}")
        x_values = depths
        y_values = [y] * len(depths)
        ax.scatter(x_values, y_values, color=VARIANT_COLOURS.get(variant, "#475569"), marker=VARIANT_MARKERS.get(variant, "o"), s=60)
        ax.plot([sum(depths) / len(depths)], [y], marker="|", markersize=18, color="#111827", linewidth=2)
    ax.set_yticks(yticks, ylabels)
    ax.set_xlabel("Selected deployment depth")
    ax.set_title("Selected deployment depths by task and variant", fontsize=15, weight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    _save_figure(fig, output_root / "milestone4" / "selected_depths", formats)


def render_quality_delta_plot(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    groups = [("CIFAR", ["cifar10", "cifar100"]), ("Transformer proof", ["sst2", "mrpc", "rte"]), ("Milestone 4 new tasks", ["qqp", "qnli", "cola"])]
    compare_variants = ["pods_all", "pods_sandwich", "naive_truncation", "layerdrop", "shallow_independent_d2", "shallow_independent_d4"]
    for ax, (title, datasets) in zip(axes, groups):
        y = 0
        yticks = []
        ylabels = []
        for dataset in datasets:
            dataset_summary = summaries.get(dataset, {})
            if "full" not in dataset_summary:
                continue
            full_value = dataset_summary["full"].final_metric_mean
            for variant in compare_variants:
                if variant not in dataset_summary:
                    continue
                delta = dataset_summary[variant].final_metric_mean - full_value
                ax.barh(y, delta, color=VARIANT_COLOURS.get(variant, "#475569"))
                yticks.append(y)
                ylabels.append(f"{dataset.upper()} {DISPLAY_NAMES.get(variant, variant)}")
                y += 1
        ax.axvline(0.0, color="#111827", linewidth=1)
        ax.set_yticks(yticks, ylabels)
        ax.set_title(title, weight="bold")
        ax.grid(True, axis="x", alpha=0.25)
    fig.suptitle("Quality delta versus full model", fontsize=16, weight="bold")
    fig.tight_layout()
    _save_figure(fig, output_root / "supporting" / "quality_delta_vs_full", formats)


def render_cost_quality_delta_plot(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    compare_variants = ["pods_all", "pods_sandwich", "naive_truncation", "layerdrop", "shallow_independent_d2", "shallow_independent_d4"]
    for dataset in MILESTONE4_TASKS:
        dataset_summary = summaries.get(dataset, {})
        full = dataset_summary.get("full")
        if full is None:
            continue
        full_depth = _representative_selected_depth(full)
        full_metric = full.final_metric_mean
        full_latency = _depth_cost(full, "latency_ms_p50", 1, full_depth)
        for variant_name in compare_variants:
            variant = dataset_summary.get(variant_name)
            if variant is None:
                continue
            depth = _representative_selected_depth(variant)
            delta_metric = variant.final_metric_mean - full_metric
            delta_latency = _depth_cost(variant, "latency_ms_p50", 1, depth) - full_latency
            ax.scatter(delta_latency, delta_metric, color=VARIANT_COLOURS.get(variant_name, "#475569"), marker=VARIANT_MARKERS.get(variant_name, "o"), s=80, alpha=0.9)
            ax.annotate(f"{dataset.upper()} {DISPLAY_NAMES.get(variant_name, variant_name)}", (delta_latency, delta_metric), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0.0, color="#111827", linewidth=1)
    ax.axvline(0.0, color="#111827", linewidth=1)
    ax.set_xlabel("Latency delta vs full (ms, p50, batch 1)")
    ax.set_ylabel("Quality delta vs full")
    ax.set_title("Cost-quality delta versus full model on GLUE tasks", fontsize=15, weight="bold")
    ax.grid(True, alpha=0.25)
    _save_figure(fig, output_root / "supporting" / "cost_quality_delta", formats)


def render_depth_metric_tradeoff_panels(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.flatten()
    ordered_variants = ["full", "pods_all", "pods_sandwich", "naive_truncation", "layerdrop", "shallow_independent_d2", "shallow_independent_d4"]
    for ax, dataset in zip(axes, MILESTONE4_TASKS):
        dataset_summary = summaries.get(dataset, {})
        for variant_name in ordered_variants:
            variant = dataset_summary.get(variant_name)
            if variant is None:
                continue
            depths = _ensure_selection_metric_depths(variant)
            xs = depths
            ys = [_selection_depth_metric(variant, depth) for depth in depths]
            yerrs = [_selection_depth_metric_std(variant, depth) for depth in depths]
            ax.errorbar(
                xs,
                ys,
                yerr=yerrs,
                color=VARIANT_COLOURS.get(variant_name, "#475569"),
                marker=VARIANT_MARKERS.get(variant_name, "o"),
                linewidth=1.8,
                capsize=3,
                label=DISPLAY_NAMES.get(variant_name, variant_name),
            )
        ax.set_title(dataset.upper(), weight="bold")
        ax.set_xlabel("Depth")
        ax.set_ylabel(dataset_summary["full"].metric_name.replace("_", " ").title())
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Depth versus task metric across Milestone 4 tasks", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, output_root / "tradeoffs" / "depth_vs_metric_glue", formats)


def render_depth_cost_tradeoff_panels(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.flatten()
    ordered_variants = ["full", "pods_all", "pods_sandwich", "naive_truncation", "layerdrop", "shallow_independent_d2", "shallow_independent_d4"]
    for ax, dataset in zip(axes, MILESTONE4_TASKS):
        dataset_summary = summaries.get(dataset, {})
        for variant_name in ordered_variants:
            variant = dataset_summary.get(variant_name)
            if variant is None:
                continue
            depths = _ensure_selection_metric_depths(variant)
            xs = depths
            ys = [_selection_depth_cost(variant, "latency_ms_p50", 1, depth) for depth in depths]
            yerrs = [_selection_depth_cost_std(variant, "latency_ms_p50", 1, depth) for depth in depths]
            ax.errorbar(
                xs,
                ys,
                yerr=yerrs,
                color=VARIANT_COLOURS.get(variant_name, "#475569"),
                marker=VARIANT_MARKERS.get(variant_name, "o"),
                linewidth=1.8,
                capsize=3,
                label=DISPLAY_NAMES.get(variant_name, variant_name),
            )
        ax.set_title(dataset.upper(), weight="bold")
        ax.set_xlabel("Depth")
        ax.set_ylabel("Measured p50 latency (ms), batch 1")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Depth versus measured latency across Milestone 4 tasks", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, output_root / "tradeoffs" / "depth_vs_latency_glue", formats)


def render_depth_coloured_pareto_panels(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    ordered_variants = ["full", "pods_all", "pods_sandwich", "naive_truncation", "layerdrop", "shallow_independent_d2", "shallow_independent_d4"]
    cmap = plt.get_cmap("viridis")
    for ax, dataset in zip(axes, NEW_TASKS):
        dataset_summary = summaries.get(dataset, {})
        max_depth = 1
        for variant in dataset_summary.values():
            max_depth = max(max_depth, max(_ensure_selection_metric_depths(variant)))
        for variant_name in ordered_variants:
            variant = dataset_summary.get(variant_name)
            if variant is None:
                continue
            depths = _ensure_selection_metric_depths(variant)
            costs = [_selection_depth_cost(variant, "latency_ms_p50", 1, depth) for depth in depths]
            metrics = [_selection_depth_metric(variant, depth) for depth in depths]
            colours = [cmap(depth / max_depth) for depth in depths]
            ax.plot(costs, metrics, color=VARIANT_COLOURS.get(variant_name, "#475569"), linewidth=1.2, alpha=0.7)
            ax.scatter(costs, metrics, c=depths, cmap=cmap, vmin=min(depths), vmax=max_depth, marker=VARIANT_MARKERS.get(variant_name, "o"), s=80, edgecolors=VARIANT_COLOURS.get(variant_name, "#475569"), linewidths=1.0)
            for depth, cost, metric in zip(depths, costs, metrics):
                ax.annotate(f"d{depth}", (cost, metric), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_title(dataset.upper(), weight="bold")
        ax.set_xlabel("Measured p50 latency (ms), batch 1")
        ax.set_ylabel(dataset_summary["full"].metric_name.replace("_", " ").title())
        ax.grid(True, alpha=0.25)
    norm = plt.Normalize(vmin=2, vmax=12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Depth", shrink=0.85)
    fig.suptitle("Pareto-style cost-quality curves coloured by depth", fontsize=16, weight="bold")
    _save_figure(fig, output_root / "tradeoffs" / "pareto_depth_cost_quality_new_tasks", formats)


def render_memory_quality_pareto_panels(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    ordered_variants = ["full", "pods_all", "pods_sandwich", "naive_truncation", "layerdrop", "shallow_independent_d2", "shallow_independent_d4"]
    cmap = plt.get_cmap("viridis")
    for ax, dataset in zip(axes, NEW_TASKS):
        dataset_summary = summaries.get(dataset, {})
        max_depth = 1
        for variant in dataset_summary.values():
            max_depth = max(max_depth, max(_ensure_selection_metric_depths(variant)))
        for variant_name in ordered_variants:
            variant = dataset_summary.get(variant_name)
            if variant is None:
                continue
            depths = _ensure_selection_metric_depths(variant)
            costs = [_selection_depth_cost(variant, "peak_memory_mb", 1, depth) for depth in depths]
            metrics = [_selection_depth_metric(variant, depth) for depth in depths]
            ax.plot(costs, metrics, color=VARIANT_COLOURS.get(variant_name, "#475569"), linewidth=1.2, alpha=0.7)
            ax.scatter(costs, metrics, c=depths, cmap=cmap, vmin=min(depths), vmax=max_depth, marker=VARIANT_MARKERS.get(variant_name, "o"), s=80, edgecolors=VARIANT_COLOURS.get(variant_name, "#475569"), linewidths=1.0)
            for depth, cost, metric in zip(depths, costs, metrics):
                ax.annotate(f"d{depth}", (cost, metric), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_title(dataset.upper(), weight="bold")
        ax.set_xlabel("Peak memory (MB), batch 1")
        ax.set_ylabel(dataset_summary["full"].metric_name.replace("_", " ").title())
        ax.grid(True, alpha=0.25)
    norm = plt.Normalize(vmin=2, vmax=12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Depth", shrink=0.85)
    fig.suptitle("Pareto-style memory-quality curves coloured by depth", fontsize=16, weight="bold")
    _save_figure(fig, output_root / "tradeoffs" / "pareto_depth_memory_quality_new_tasks", formats)


def render_reduced_clutter_pareto_panels(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    ordered_variants = ["full", "pods_all", "pods_sandwich", "naive_truncation"]
    for ax, dataset in zip(axes, NEW_TASKS):
        dataset_summary = summaries.get(dataset, {})
        for variant_name in ordered_variants:
            variant = dataset_summary.get(variant_name)
            if variant is None:
                continue
            depths = _ensure_selection_metric_depths(variant)
            costs = [_selection_depth_cost(variant, "latency_ms_p50", 1, depth) for depth in depths]
            metrics = [_selection_depth_metric(variant, depth) for depth in depths]
            ax.plot(
                costs,
                metrics,
                color=VARIANT_COLOURS.get(variant_name, "#475569"),
                marker=VARIANT_MARKERS.get(variant_name, "o"),
                linewidth=2.2,
                markersize=7,
                label=DISPLAY_NAMES.get(variant_name, variant_name),
            )
            for depth, cost, metric in zip(depths, costs, metrics):
                ax.annotate(f"d{depth}", (cost, metric), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title(dataset.upper(), weight="bold")
        ax.set_xlabel("Measured p50 latency (ms), batch 1")
        ax.set_ylabel(dataset_summary["full"].metric_name.replace("_", " ").title())
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Core-comparator Pareto curves for new GLUE tasks", fontsize=16, weight="bold")
    _save_figure(fig, output_root / "tradeoffs" / "pareto_core_comparators_latency_new_tasks", formats)


def render_reduced_clutter_memory_pareto_panels(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    ordered_variants = ["full", "pods_all", "pods_sandwich", "naive_truncation"]
    for ax, dataset in zip(axes, NEW_TASKS):
        dataset_summary = summaries.get(dataset, {})
        for variant_name in ordered_variants:
            variant = dataset_summary.get(variant_name)
            if variant is None:
                continue
            depths = _ensure_selection_metric_depths(variant)
            costs = [_selection_depth_cost(variant, "peak_memory_mb", 1, depth) for depth in depths]
            metrics = [_selection_depth_metric(variant, depth) for depth in depths]
            ax.plot(
                costs,
                metrics,
                color=VARIANT_COLOURS.get(variant_name, "#475569"),
                marker=VARIANT_MARKERS.get(variant_name, "o"),
                linewidth=2.2,
                markersize=7,
                label=DISPLAY_NAMES.get(variant_name, variant_name),
            )
            for depth, cost, metric in zip(depths, costs, metrics):
                ax.annotate(f"d{depth}", (cost, metric), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title(dataset.upper(), weight="bold")
        ax.set_xlabel("Peak memory (MB), batch 1")
        ax.set_ylabel(dataset_summary["full"].metric_name.replace("_", " ").title())
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Core-comparator memory-quality curves for new GLUE tasks", fontsize=16, weight="bold")
    _save_figure(fig, output_root / "tradeoffs" / "pareto_core_comparators_memory_new_tasks", formats)


def write_manifest(output_root: Path, rows: list[dict[str, Any]]) -> None:
    manifest = {
        "overview_rows": rows,
        "generated_at": "local_run",
        "output_root": str(output_root),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="pods/results/processed")
    parser.add_argument("--output-root", default="temp/analysis_figures")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--formats", default="png,svg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ALL_DATASETS if args.datasets == "all" else [item.strip() for item in args.datasets.split(",") if item.strip()]
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    processed_root = Path(args.processed_root)
    output_root = Path(args.output_root)
    summaries = load_processed_summaries(processed_root, datasets)
    if not summaries:
        raise ValueError(f"No processed summaries found under {processed_root}")

    render_pods_flow_diagram(output_root, formats)
    outcome_rows = render_task_outcome_grid(summaries, output_root, formats)
    render_frontier_plots(summaries, output_root, formats)
    render_m4_main_panel(summaries, output_root, formats)
    render_m4_baseline_panel(summaries, output_root, formats)
    render_selected_depth_plot(summaries, output_root, formats)
    render_quality_delta_plot(summaries, output_root, formats)
    render_cost_quality_delta_plot(summaries, output_root, formats)
    render_depth_metric_tradeoff_panels(summaries, output_root, formats)
    render_depth_cost_tradeoff_panels(summaries, output_root, formats)
    render_depth_coloured_pareto_panels(summaries, output_root, formats)
    render_memory_quality_pareto_panels(summaries, output_root, formats)
    render_reduced_clutter_pareto_panels(summaries, output_root, formats)
    render_reduced_clutter_memory_pareto_panels(summaries, output_root, formats)
    write_manifest(output_root, outcome_rows)


if __name__ == "__main__":
    main()
