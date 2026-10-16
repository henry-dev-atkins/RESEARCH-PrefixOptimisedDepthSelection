from __future__ import annotations

import json

from pods.eval.export_paper_assets import export_paper_assets, validate_paper_summaries
from pods.eval.paper_scope import PAPER_DATASETS, PAPER_MAIN_VARIANTS


def test_default_paper_scope_defers_cifar100() -> None:
    assert "cifar10" in PAPER_DATASETS
    assert "cifar100" not in PAPER_DATASETS


def test_paper_scope_has_three_core_variants() -> None:
    assert PAPER_MAIN_VARIANTS == ("full", "pods_all", "pods_sandwich")


def test_export_paper_assets_writes_csv_and_tex(tmp_path) -> None:
    processed_root = tmp_path / "processed"
    output_root = tmp_path / "paper_generated"
    processed_root.mkdir()
    (processed_root / "run_index.json").write_text(
        json.dumps(
            {
                "sst2:bert:full": {
                    "selected_depths": [12, 12, 12],
                    "final_metric_mean": 0.91,
                }
            }
        ),
        encoding="utf-8",
    )

    export_paper_assets(processed_root, output_root)

    assert (output_root / "manifest.json").exists()
    assert (output_root / "tables" / "sst2_summary.csv").exists()
    assert (output_root / "tables" / "sst2_summary.tex").exists()
    csv_text = (output_root / "tables" / "sst2_summary.csv").read_text(encoding="utf-8")
    assert "final_metric_std" in csv_text
    assert "final_primary_split" in csv_text


def test_export_paper_assets_writes_secondary_tables_when_present(tmp_path) -> None:
    processed_root = tmp_path / "processed"
    output_root = tmp_path / "paper_generated"
    processed_root.mkdir()
    (processed_root / "run_index.json").write_text(
        json.dumps(
            {
                "mnli:bert:full": {
                    "selected_depths": [12],
                    "final_metric_mean": 0.91,
                    "task_metric_name": "accuracy",
                    "final_results_secondary_by_depth": {
                        "accuracy": {
                            "12": {"mean": 0.89, "std": 0.01}
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    export_paper_assets(processed_root, output_root)

    assert (output_root / "tables" / "mnli_secondary_summary.csv").exists()
    assert (output_root / "tables" / "mnli_secondary_summary.tex").exists()


def test_export_paper_assets_writes_core_tables_with_scope_manifest(tmp_path) -> None:
    processed_root = tmp_path / "processed"
    output_root = tmp_path / "paper_generated"
    processed_root.mkdir()
    entries = {}
    for dataset in PAPER_DATASETS:
        for variant in PAPER_MAIN_VARIANTS:
            entries[f"{dataset}:bert:{variant}"] = {
                "num_runs": 1,
                "seeds": [7],
                "selected_depths": [12],
                "final_metric_mean": 0.9123456789,
                "final_metric_std": 0.0,
                "task_metric_name": "accuracy",
                "evaluation_splits": {"final_primary": "validation"},
                "final_costs_by_batch_size": {
                    "latency_ms_p50": {"1": {"12": {"mean": 3.25}}},
                    "peak_memory_mb": {"1": {"12": {"mean": 12.5}}},
                    "model_bytes": {"1": {"12": {"mean": 1234}}},
                },
            }
    (processed_root / "run_index.json").write_text(json.dumps(entries), encoding="utf-8")
    figures_root = output_root / "figures"
    figures_root.mkdir(parents=True)
    (figures_root / "manifest.json").write_text("{}", encoding="utf-8")

    export_paper_assets(
        processed_root,
        output_root,
        datasets=PAPER_DATASETS,
        required_datasets=PAPER_DATASETS,
    )

    assert (output_root / "tables" / "core_results.csv").exists()
    assert (output_root / "tables" / "core_deployment.csv").exists()
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["datasets"] == list(PAPER_DATASETS)
    assert manifest["required_variants"] == list(PAPER_MAIN_VARIANTS)
    assert all(path != str(figures_root / "manifest.json") for path in manifest["figures"])
    assert "0.9123456789" in (output_root / "tables" / "core_results.csv").read_text(encoding="utf-8")


def test_validate_paper_summaries_rejects_missing_required_dataset() -> None:
    try:
        validate_paper_summaries(
            {"sst2:bert:full": {"selected_depths": [12]}},
            required_datasets=("sst2", "mnli"),
        )
    except ValueError as exc:
        assert "mnli" in str(exc)
    else:
        raise AssertionError("Missing paper dataset was accepted")
