from __future__ import annotations

import json
from pathlib import Path

from pods.eval.review_figures import (
    _best_pods_variant,
    _representative_selected_depth,
    _selected_depth_label,
    generate_review_figures,
    load_processed_summaries,
)


def _make_summary_entry(dataset: str, variant: str, final_metric_mean: float, selected_depths: list[int]) -> tuple[str, dict[str, object]]:
    key = f"{dataset}:bert:{variant}"
    entry = {
        "task_metric_name": "accuracy",
        "selected_depths": selected_depths,
        "final_metric_mean": final_metric_mean,
        "final_metric_std": 0.01,
        "selection_results_by_depth": {
            "accuracy": {
                "2": {"mean": 0.7, "std": 0.0},
                "12": {"mean": 0.8, "std": 0.0},
            }
        },
        "selection_costs_by_depth_and_batch_size": {
            "latency_ms_p50": {
                "1": {
                    "2": {"mean": 1.0, "std": 0.0},
                    "12": {"mean": 3.0, "std": 0.0},
                }
            }
        },
        "final_results_by_depth": {
            "accuracy": {
                "2": {"mean": 0.69, "std": 0.0},
                "12": {"mean": final_metric_mean, "std": 0.01},
            }
        },
        "final_costs_by_batch_size": {
            "latency_ms_p50": {
                "1": {
                    "2": {"mean": 1.1, "std": 0.0},
                    "12": {"mean": 3.1, "std": 0.0},
                }
            }
        },
    }
    return key, entry


def _write_dataset_summary(processed_root: Path, dataset: str, variants: dict[str, tuple[float, list[int]]]) -> None:
    payload = dict(_make_summary_entry(dataset, variant, score, depths) for variant, (score, depths) in variants.items())
    (processed_root / f"{dataset}_frontier_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_processed_summaries_reads_variant_entries(tmp_path) -> None:
    _write_dataset_summary(tmp_path, "qqp", {"pods_all": (0.9, [8, 8, 8])})

    summaries = load_processed_summaries(tmp_path, ["qqp"])

    assert summaries["qqp"]["pods_all"].selected_depths == [8, 8, 8]
    assert summaries["qqp"]["pods_all"].metric_name == "accuracy"


def test_best_pods_variant_prefers_shallower_depth_on_metric_tie(tmp_path) -> None:
    _write_dataset_summary(
        tmp_path,
        "qqp",
        {
            "pods_all": (0.90, [8, 8, 8]),
            "pods_sandwich": (0.90, [12, 12, 12]),
        },
    )

    dataset_summary = load_processed_summaries(tmp_path, ["qqp"])["qqp"]

    assert _best_pods_variant(dataset_summary) == "pods_all"


def test_selected_depth_helpers_handle_modes_and_labels(tmp_path) -> None:
    _write_dataset_summary(tmp_path, "cola", {"pods_all": (0.55, [10, 12, 10])})

    variant = load_processed_summaries(tmp_path, ["cola"])["cola"]["pods_all"]

    assert _representative_selected_depth(variant) == 10
    assert _selected_depth_label(variant) == "d10,d12"


def test_generate_review_figures_writes_manifest_and_key_outputs(tmp_path) -> None:
    processed_root = tmp_path / "processed"
    output_root = tmp_path / "review_figures"
    processed_root.mkdir()
    for dataset in ["sst2", "mrpc", "rte", "qqp", "qnli", "cola"]:
        _write_dataset_summary(
            processed_root,
            dataset,
            {
                "full": (0.90, [12, 12, 12]),
                "pods_all": (0.89, [8, 8, 8]),
                "pods_sandwich": (0.91, [12, 12, 12]),
                "naive_truncation": (0.905, [10, 10, 10]),
                "layerdrop": (0.88, [8, 10, 12]),
                "shallow_independent_d2": (0.84, [2, 2, 2]),
                "shallow_independent_d4": (0.86, [4, 4, 4]),
            },
        )

    generate_review_figures(processed_root, output_root, ["sst2", "mrpc", "rte", "qqp", "qnli", "cola"], ["png"])

    assert (output_root / "manifest.json").exists()
    assert (output_root / "overview" / "pods_flow_diagram.png").exists()
    assert (output_root / "milestone4" / "m4_main_variants_panel.png").exists()
    assert (output_root / "tradeoffs" / "pareto_core_comparators_latency_new_tasks.png").exists()


def test_generate_review_figures_handles_partial_dataset_coverage_and_new_variants(tmp_path) -> None:
    processed_root = tmp_path / "processed"
    output_root = tmp_path / "review_figures"
    processed_root.mkdir()
    _write_dataset_summary(
        processed_root,
        "stsb",
        {
            "full": (0.89, [12, 12, 12]),
            "pods_all": (0.88, [8, 8, 8]),
            "progressive_shrinking": (0.87, [6, 8, 8]),
            "distilled_shallow_d2": (0.82, [2, 2, 2]),
            "distilled_shallow_d4": (0.85, [4, 4, 4]),
        },
    )

    generate_review_figures(processed_root, output_root, ["stsb"], ["png"])

    assert (output_root / "manifest.json").exists()
    assert (output_root / "frontiers" / "stsb_frontier.png").exists()
    assert (output_root / "overview" / "task_outcome_grid.png").exists()
