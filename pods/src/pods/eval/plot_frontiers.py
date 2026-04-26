from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _mean_metric(entry: dict[str, Any], section: str, metric: str, batch_size: int, depth: int) -> float:
    return float(entry[section][metric][str(batch_size)][str(depth)]["mean"])


def _extract_frontier_points(summary: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for key, entry in sorted(summary.items()):
        label = key.split(":")[-1]
        for depth_key, accuracy_summary in sorted(entry["final_results_by_depth"]["accuracy"].items(), key=lambda item: int(item[0])):
            depth = int(depth_key)
            points.append(
                {
                    "label": label,
                    "depth": depth,
                    "accuracy_mean": float(accuracy_summary["mean"]),
                    "accuracy_std": float(accuracy_summary["std"]),
                    "latency_ms_p50_mean": _mean_metric(entry, "final_costs_by_batch_size", "latency_ms_p50", 1, depth),
                    "latency_ms_p50_std": float(entry["final_costs_by_batch_size"]["latency_ms_p50"]["1"][str(depth)]["std"]),
                    "peak_memory_mb_mean": _mean_metric(entry, "final_costs_by_batch_size", "peak_memory_mb", 1, depth),
                    "throughput_items_per_s_mean": _mean_metric(entry, "final_costs_by_batch_size", "throughput_items_per_s", 1, depth),
                }
            )
    return points


def _plot_dataset(dataset: str, summary: dict[str, Any], output_root: Path) -> None:
    points = _extract_frontier_points(summary)
    if not points:
        raise ValueError(f"No frontier points available for dataset {dataset}")

    colours = {
        "full": "#1b1b1b",
        "naive_truncation": "#b04a3a",
        "shallow_independent_d1": "#0f766e",
        "shallow_independent_d2": "#155e75",
        "shallow_independent_d4": "#0b7285",
        "layerdrop": "#c2410c",
        "pods_all": "#1d4ed8",
        "pods_sandwich": "#7c3aed",
    }
    markers = {
        "full": "o",
        "naive_truncation": "s",
        "shallow_independent_d1": "^",
        "shallow_independent_d2": "v",
        "shallow_independent_d4": "X",
        "layerdrop": "*",
        "pods_all": "D",
        "pods_sandwich": "P",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    seen_labels: set[str] = set()
    for point in points:
        label = point["label"]
        ax.errorbar(
            point["latency_ms_p50_mean"],
            point["accuracy_mean"],
            xerr=point["latency_ms_p50_std"],
            yerr=point["accuracy_std"],
            fmt=markers.get(label, "o"),
            color=colours.get(label, "#374151"),
            markersize=7,
            capsize=4,
            linestyle="none",
            label=label if label not in seen_labels else None,
        )
        ax.annotate(f"d{point['depth']}", (point["latency_ms_p50_mean"], point["accuracy_mean"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        seen_labels.add(label)

    ax.set_title(f"{dataset.upper()} Measured Frontier")
    ax.set_xlabel("Measured p50 latency (ms), batch 1")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_root / f"{dataset}_frontier.png", dpi=200)
    fig.savefig(output_root / f"{dataset}_frontier.svg")
    (output_root / f"{dataset}_frontier_points.json").write_text(json.dumps(points, indent=2), encoding="utf-8")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="pods/results/processed")
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    for summary_path in sorted(processed_root.glob("*_frontier_summary.json")):
        dataset = summary_path.name.removesuffix("_frontier_summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary:
            _plot_dataset(dataset, summary, processed_root)


if __name__ == "__main__":
    main()
