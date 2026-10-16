from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pods.eval.paper_scope import PAPER_DATASETS, PAPER_MAIN_VARIANTS


VARIANT_COLOURS = {
    "full": "#1b1b1b",
    "naive_truncation": "#b04a3a",
    "shallow_independent_d2": "#155e75",
    "shallow_independent_d4": "#0b7285",
    "distilled_shallow_d2": "#0f766e",
    "distilled_shallow_d4": "#0e7490",
    "layerdrop": "#c2410c",
    "progressive_shrinking": "#a16207",
    "pods_all": "#1d4ed8",
    "pods_sandwich": "#7c3aed",
}

VARIANT_MARKERS = {
    "full": "o",
    "naive_truncation": "s",
    "shallow_independent_d2": "v",
    "shallow_independent_d4": "X",
    "distilled_shallow_d2": "^",
    "distilled_shallow_d4": "<",
    "layerdrop": "*",
    "progressive_shrinking": "h",
    "pods_all": "D",
    "pods_sandwich": "P",
}

DISPLAY_NAMES = {
    "full": "Full",
    "naive_truncation": "Naive truncation",
    "shallow_independent_d2": "Shallow d2",
    "shallow_independent_d4": "Shallow d4",
    "distilled_shallow_d2": "Distilled d2",
    "distilled_shallow_d4": "Distilled d4",
    "layerdrop": "LayerDrop",
    "progressive_shrinking": "Progressive shrinking",
    "pods_all": "PODS all",
    "pods_sandwich": "PODS sandwich",
}

ALL_DATASETS = list(PAPER_DATASETS)
MILESTONE4_TASKS = ["sst2", "mrpc", "rte", "qqp", "qnli", "cola"]
NEW_TASKS = ["qqp", "qnli", "cola"]
SUPPORTED_FORMATS = {"png", "svg", "pdf"}
PAPER_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 120,
}


@dataclass
class VariantSummary:
    dataset: str
    variant: str
    metric_name: str
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
        path = processed_root / f"{dataset}_frontier_summary.json"
        if path.exists():
            summaries[dataset] = _load_summary(path)
    return summaries


def _metric_depths(variant: VariantSummary, section: str) -> list[int]:
    rows = getattr(variant, section).get(variant.metric_name)
    if not rows:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing metric rows for {variant.metric_name}")
    return sorted(int(depth) for depth in rows.keys())


def _metric_value(variant: VariantSummary, depth: int, section: str) -> float:
    rows = getattr(variant, section).get(variant.metric_name)
    if rows is None or str(depth) not in rows:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing depth {depth}")
    return float(rows[str(depth)]["mean"])


def _metric_std(variant: VariantSummary, depth: int, section: str) -> float:
    rows = getattr(variant, section).get(variant.metric_name)
    if rows is None or str(depth) not in rows:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing std depth {depth}")
    return float(rows[str(depth)]["std"])


def _cost_value(variant: VariantSummary, metric_name: str, depth: int, batch_size: int, section: str) -> float:
    rows = getattr(variant, section).get(metric_name)
    if rows is None or str(batch_size) not in rows or str(depth) not in rows[str(batch_size)]:
        raise ValueError(f"{variant.dataset}:{variant.variant} missing {metric_name} depth {depth} batch {batch_size}")
    return float(rows[str(batch_size)][str(depth)]["mean"])


def _selected_depth_label(variant: VariantSummary) -> str:
    if not variant.selected_depths:
        return "n/a"
    unique = sorted(set(variant.selected_depths))
    if len(unique) == 1:
        return f"d{unique[0]}"
    return ",".join(f"d{depth}" for depth in unique)


def _representative_selected_depth(variant: VariantSummary) -> int:
    counts: dict[int, int] = {}
    for depth in variant.selected_depths:
        counts[depth] = counts.get(depth, 0) + 1
    if not counts:
        raise ValueError(f"{variant.dataset}:{variant.variant} has no selected depths")
    best_count = max(counts.values())
    return min(depth for depth, count in counts.items() if count == best_count)


def _best_pods_variant(dataset_summary: dict[str, VariantSummary]) -> str | None:
    candidates = [name for name in ("pods_all", "pods_sandwich") if name in dataset_summary]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda name: (
            dataset_summary[name].final_metric_mean,
            -sum(dataset_summary[name].selected_depths) / max(len(dataset_summary[name].selected_depths), 1),
        ),
    )


def _save_figure(fig: plt.Figure, output_base: Path, formats: list[str], dpi: int | None = None) -> None:
    """Save one figure in the requested vector and raster formats."""

    invalid_formats = sorted(set(formats) - SUPPORTED_FORMATS)
    if not formats:
        raise ValueError("At least one figure format is required")
    if invalid_formats:
        raise ValueError(f"Unsupported figure formats: {invalid_formats}")
    if dpi is not None and dpi <= 0:
        raise ValueError("Figure DPI must be positive")

    resolved_dpi = int(plt.rcParams["savefig.dpi"] if dpi is None else dpi)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=resolved_dpi if fmt == "png" else None, bbox_inches="tight")
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
    for x, y, width, height, colour, text in boxes:
        rect = plt.Rectangle((x, y), width, height, facecolor=colour, edgecolor="#334155", linewidth=1.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=12, transform=ax.transAxes)
    for x_start, x_end in ((0.23, 0.28), (0.48, 0.53), (0.73, 0.78)):
        ax.annotate("", xy=(x_end, 0.5), xytext=(x_start, 0.5), xycoords="axes fraction", textcoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 2, "color": "#475569"})
    ax.text(0.5, 0.95, "PODS training, selection, and deployment flow", ha="center", va="top", fontsize=16, weight="bold", transform=ax.transAxes)
    _save_figure(fig, output_root / "overview" / "pods_flow_diagram", formats)


def render_task_outcome_grid(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in PAPER_DATASETS:
        dataset_summary = summaries.get(dataset, {})
        best_pods = _best_pods_variant(dataset_summary)
        if best_pods is None:
            continue
        best_variant = dataset_summary[best_pods]
        full = dataset_summary.get("full")
        naive = dataset_summary.get("naive_truncation")
        delta_vs_full = best_variant.final_metric_mean - full.final_metric_mean if full else math.nan
        delta_vs_naive = best_variant.final_metric_mean - naive.final_metric_mean if naive else math.nan
        mean_depth = sum(best_variant.selected_depths) / max(len(best_variant.selected_depths), 1)
        full_depth = max(_metric_depths(full, "final_results_by_depth")) if full else math.inf
        shallower = mean_depth < full_depth
        if not math.isnan(delta_vs_full) and delta_vs_full > 0 and (math.isnan(delta_vs_naive) or delta_vs_naive > 0) and shallower:
            verdict = "clear win"
        elif not math.isnan(delta_vs_full) and delta_vs_full >= 0 and shallower:
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
    table_rows = [
        [
            row["dataset"].upper(),
            DISPLAY_NAMES.get(row["best_pods"], row["best_pods"]),
            "n/a" if math.isnan(row["delta_vs_full"]) else f"{row['delta_vs_full']:+.4f}",
            "n/a" if math.isnan(row["delta_vs_naive"]) else f"{row['delta_vs_naive']:+.4f}",
            row["selected_depth"],
            row["verdict"],
        ]
        for row in rows
    ]
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
        elif col_idx == 5:
            verdict = table_rows[row_idx - 1][-1]
            colour = {"clear win": "#bbf7d0", "partial win": "#fef3c7", "mixed": "#fde68a", "loss": "#fecaca"}[verdict]
            cell.set_facecolor(colour)
    ax.set_title("Outcome summary across PODS experiments", fontsize=15, weight="bold", pad=20)
    _save_figure(fig, output_root / "overview" / "task_outcome_grid", formats)
    return rows


def render_frontier_plots(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    for dataset, dataset_summary in summaries.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        seen_labels: set[str] = set()
        for variant_name, variant in sorted(dataset_summary.items()):
            for depth in _metric_depths(variant, "final_results_by_depth"):
                latency = _cost_value(variant, "latency_ms_p50", depth, 1, "final_costs_by_batch_size")
                metric = _metric_value(variant, depth, "final_results_by_depth")
                ax.errorbar(
                    latency,
                    metric,
                    xerr=_cost_value(variant, "latency_ms_p50", depth, 1, "final_costs_by_batch_size") * 0.0,
                    yerr=_metric_std(variant, depth, "final_results_by_depth"),
                    fmt=VARIANT_MARKERS.get(variant_name, "o"),
                    color=VARIANT_COLOURS.get(variant_name, "#475569"),
                    markersize=7,
                    capsize=3,
                    linestyle="none",
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
    available_datasets = [dataset for dataset in PAPER_DATASETS if dataset in summaries]
    if not available_datasets:
        return
    rows = max(1, math.ceil(len(available_datasets) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(14, 4.2 * rows), squeeze=False)
    flat_axes = axes.flatten()
    for ax, dataset in zip(flat_axes, available_datasets):
        dataset_summary = summaries[dataset]
        labels = [name for name in PAPER_MAIN_VARIANTS if name in dataset_summary]
        values = [dataset_summary[name].final_metric_mean for name in labels]
        errors = [dataset_summary[name].final_metric_std for name in labels]
        ax.bar(range(len(labels)), values, yerr=errors, color=[VARIANT_COLOURS[name] for name in labels], capsize=4)
        ax.set_xticks(range(len(labels)), [DISPLAY_NAMES[name] for name in labels], rotation=20, ha="right")
        ax.set_title(dataset.upper(), weight="bold")
        ax.set_ylabel(dataset_summary["full"].metric_name.replace("_", " ").title())
        ax.grid(True, axis="y", alpha=0.25)
        for idx, name in enumerate(labels):
            ax.text(idx, values[idx], _selected_depth_label(dataset_summary[name]), ha="center", va="bottom", fontsize=9)
    for ax in flat_axes[len(available_datasets):]:
        ax.set_axis_off()
    fig.suptitle("Main-variant outcomes across the publication scope", fontsize=16, weight="bold")
    fig.tight_layout()
    _save_figure(fig, output_root / "milestone4" / "m4_main_variants_panel", formats)


def render_m4_baseline_panel(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    available_datasets = [dataset for dataset in NEW_TASKS if dataset in summaries]
    if not available_datasets:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes_list = list(axes) if isinstance(axes, (list, tuple)) else list(axes.flat)
    for ax, dataset in zip(axes_list, available_datasets):
        dataset_summary = summaries[dataset]
        best_pods = _best_pods_variant(dataset_summary)
        ordered = [
            "full",
            best_pods,
            "naive_truncation",
            "layerdrop",
            "progressive_shrinking",
            "shallow_independent_d2",
            "shallow_independent_d4",
            "distilled_shallow_d2",
            "distilled_shallow_d4",
        ]
        labels = [name for name in ordered if name and name in dataset_summary]
        values = [dataset_summary[name].final_metric_mean for name in labels]
        errors = [dataset_summary[name].final_metric_std for name in labels]
        ax.bar(range(len(labels)), values, yerr=errors, color=[VARIANT_COLOURS.get(name, "#475569") for name in labels], capsize=4)
        ax.set_xticks(range(len(labels)), [DISPLAY_NAMES.get(name, name) for name in labels], rotation=25, ha="right")
        ax.set_title(dataset.upper(), weight="bold")
        ax.set_ylabel(dataset_summary["full"].metric_name.replace("_", " ").title())
        ax.grid(True, axis="y", alpha=0.25)
        for idx, name in enumerate(labels):
            ax.text(idx, values[idx], _selected_depth_label(dataset_summary[name]), ha="center", va="bottom", fontsize=8)
    for ax in axes_list[len(available_datasets):]:
        ax.set_axis_off()
    fig.suptitle("Baseline comparisons across GLUE tasks", fontsize=16, weight="bold")
    fig.tight_layout()
    _save_figure(fig, output_root / "milestone4" / "m4_baseline_comparison", formats)


def render_selected_depth_plot(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    rows: list[tuple[str, str, list[int]]] = []
    for dataset in PAPER_DATASETS:
        for variant in PAPER_MAIN_VARIANTS:
            dataset_summary = summaries.get(dataset, {})
            if variant in dataset_summary:
                rows.append((dataset, variant, dataset_summary[variant].selected_depths))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, max(6, len(rows) * 0.35)))
    yticks: list[int] = []
    ylabels: list[str] = []
    for idx, (dataset, variant, depths) in enumerate(rows):
        y = len(rows) - idx
        yticks.append(y)
        ylabels.append(f"{dataset.upper()} {DISPLAY_NAMES.get(variant, variant)}")
        ax.scatter(depths, [y] * len(depths), color=VARIANT_COLOURS.get(variant, "#475569"), marker=VARIANT_MARKERS.get(variant, "o"), s=60)
    ax.set_yticks(yticks, ylabels)
    ax.set_xlabel("Selected deployment depth")
    ax.set_title("Selected deployment depths by task and variant", fontsize=15, weight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    _save_figure(fig, output_root / "milestone4" / "selected_depths", formats)


def render_depth_metric_tradeoff_panels(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    available_datasets = [dataset for dataset in PAPER_DATASETS if dataset in summaries]
    if not available_datasets:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    flat_axes = axes.flatten()
    for ax, dataset in zip(flat_axes, available_datasets):
        dataset_summary = summaries[dataset]
        for variant_name in PAPER_MAIN_VARIANTS:
            variant = dataset_summary.get(variant_name)
            if variant is None:
                continue
            depths = _metric_depths(variant, "selection_results_by_depth")
            ax.errorbar(
                depths,
                [_metric_value(variant, depth, "selection_results_by_depth") for depth in depths],
                yerr=[_metric_std(variant, depth, "selection_results_by_depth") for depth in depths],
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
    for ax in flat_axes[len(available_datasets):]:
        ax.set_axis_off()
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Depth versus task metric across GLUE tasks", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, output_root / "tradeoffs" / "depth_vs_metric_glue", formats)


def render_core_comparator_pareto_panels(summaries: dict[str, dict[str, VariantSummary]], output_root: Path, formats: list[str]) -> None:
    available_datasets = [dataset for dataset in NEW_TASKS if dataset in summaries]
    if not available_datasets:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    axes_list = list(axes) if isinstance(axes, (list, tuple)) else list(axes.flat)
    for ax, dataset in zip(axes_list, available_datasets):
        dataset_summary = summaries[dataset]
        for variant_name in ["full", "pods_all", "pods_sandwich", "naive_truncation", "progressive_shrinking"]:
            variant = dataset_summary.get(variant_name)
            if variant is None:
                continue
            depths = _metric_depths(variant, "selection_results_by_depth")
            costs = [_cost_value(variant, "latency_ms_p50", depth, 1, "selection_costs_by_depth_and_batch_size") for depth in depths]
            metrics = [_metric_value(variant, depth, "selection_results_by_depth") for depth in depths]
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
    for ax in axes_list[len(available_datasets):]:
        ax.set_axis_off()
    fig.suptitle("Core-comparator Pareto curves across GLUE tasks", fontsize=16, weight="bold")
    _save_figure(fig, output_root / "tradeoffs" / "pareto_core_comparators_latency_new_tasks", formats)


def write_manifest(output_root: Path, rows: list[dict[str, Any]], formats: list[str]) -> None:
    """Write the figure inventory and requested output formats."""

    assets = sorted(str(path) for path in output_root.rglob("*") if path.is_file() and path.name != "manifest.json")
    manifest = {"overview_rows": rows, "formats": formats, "assets": assets, "output_root": str(output_root)}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def generate_review_figures(processed_root: Path, output_root: Path, datasets: list[str], formats: list[str], dpi: int = 300) -> None:
    """Generate review and paper figures from processed experiment summaries."""

    invalid_formats = sorted(set(formats) - SUPPORTED_FORMATS)
    if not formats:
        raise ValueError("At least one figure format is required")
    if invalid_formats:
        raise ValueError(f"Unsupported figure formats: {invalid_formats}")
    if dpi <= 0:
        raise ValueError("Figure DPI must be positive")

    summaries = load_processed_summaries(processed_root, datasets)
    if not summaries:
        raise ValueError(f"No processed summaries found under {processed_root}")
    with plt.rc_context({**PAPER_STYLE, "savefig.dpi": dpi}):
        render_pods_flow_diagram(output_root, formats)
        outcome_rows = render_task_outcome_grid(summaries, output_root, formats)
        render_frontier_plots(summaries, output_root, formats)
        render_m4_main_panel(summaries, output_root, formats)
        render_m4_baseline_panel(summaries, output_root, formats)
        render_selected_depth_plot(summaries, output_root, formats)
        render_depth_metric_tradeoff_panels(summaries, output_root, formats)
        render_core_comparator_pareto_panels(summaries, output_root, formats)
    write_manifest(output_root, outcome_rows, formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="pods/results/processed")
    parser.add_argument("--output-root", default="pods/results/processed/review_figures")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--formats", default="png,svg")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ALL_DATASETS if args.datasets == "all" else [item.strip() for item in args.datasets.split(",") if item.strip()]
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    generate_review_figures(Path(args.processed_root), Path(args.output_root), datasets, formats, dpi=args.dpi)


if __name__ == "__main__":
    main()
