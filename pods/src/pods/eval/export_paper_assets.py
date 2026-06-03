from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _latex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def _row_for_variant(key: str, entry: dict[str, Any]) -> tuple[str, str, str, str]:
    _, model_family, variant = key.split(":", 2)
    selected_depths = entry.get("selected_depths", [])
    depth_label = ",".join(str(depth) for depth in sorted(set(selected_depths))) if selected_depths else "n/a"
    return (
        model_family,
        variant,
        depth_label,
        f"{entry['final_metric_mean']:.4f}",
    )


def _write_secondary_tables(dataset: str, entries: dict[str, Any], output_root: Path) -> None:
    rows: list[tuple[str, str, str, str]] = []
    for key, entry in sorted(entries.items()):
        secondary = entry.get("final_results_secondary_by_depth")
        if not secondary:
            continue
        metric_name = entry.get("task_metric_name", "metric")
        metric_rows = secondary.get(metric_name)
        if not metric_rows:
            continue
        _, model_family, variant = key.split(":", 2)
        for depth, values in metric_rows.items():
            rows.append((model_family, variant, depth, f"{values['mean']:.4f}"))
    if not rows:
        return

    header = "model,variant,depth,secondary_metric_mean\n"
    csv_body = "\n".join(",".join(row) for row in rows)
    _write_text(output_root / "tables" / f"{dataset}_secondary_summary.csv", header + csv_body + ("\n" if csv_body else ""))

    latex_lines = [
        "\\begin{tabular}{llll}",
        "\\hline",
        "Model & Variant & Depth & Secondary metric \\\\",
        "\\hline",
    ]
    latex_lines.extend(
        f"{_latex_escape(model)} & {_latex_escape(variant)} & {_latex_escape(depth)} & {metric} \\\\"
        for model, variant, depth, metric in rows
    )
    latex_lines.extend(["\\hline", "\\end{tabular}", ""])
    _write_text(output_root / "tables" / f"{dataset}_secondary_summary.tex", "\n".join(latex_lines))


def export_paper_assets(processed_root: Path, output_root: Path) -> None:
    run_index_path = processed_root / "run_index.json"
    run_index = json.loads(run_index_path.read_text(encoding="utf-8"))
    datasets = sorted({key.split(":", 1)[0] for key in run_index})

    manifest = {"datasets": datasets, "num_variants": len(run_index)}
    _write_text(output_root / "manifest.json", json.dumps(manifest, indent=2))

    for dataset in datasets:
        entries = {key: value for key, value in run_index.items() if key.startswith(f"{dataset}:")}
        rows = [_row_for_variant(key, entry) for key, entry in sorted(entries.items())]
        header = "model,variant,selected_depths,final_metric_mean\n"
        csv_body = "\n".join(",".join(row) for row in rows)
        _write_text(output_root / "tables" / f"{dataset}_summary.csv", header + csv_body + ("\n" if csv_body else ""))

        latex_lines = [
            "\\begin{tabular}{llll}",
            "\\hline",
            "Model & Variant & Selected depths & Final metric \\\\",
            "\\hline",
        ]
        latex_lines.extend(
            f"{_latex_escape(model)} & {_latex_escape(variant)} & {_latex_escape(depths)} & {metric} \\\\"
            for model, variant, depths, metric in rows
        )
        latex_lines.extend(["\\hline", "\\end{tabular}", ""])
        _write_text(output_root / "tables" / f"{dataset}_summary.tex", "\n".join(latex_lines))
        _write_secondary_tables(dataset, entries, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="pods/results/processed")
    parser.add_argument("--output-root", default="paper/generated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_paper_assets(Path(args.processed_root), Path(args.output_root))


if __name__ == "__main__":
    main()
