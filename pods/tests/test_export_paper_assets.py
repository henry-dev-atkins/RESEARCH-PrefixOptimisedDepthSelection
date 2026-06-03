from __future__ import annotations

import json

from pods.eval.export_paper_assets import export_paper_assets


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
