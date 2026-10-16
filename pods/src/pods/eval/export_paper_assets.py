from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable

from pods.eval.paper_scope import PAPER_DATASETS, PAPER_MAIN_VARIANTS

REQUIRED_MAIN_VARIANTS = PAPER_MAIN_VARIANTS


def _write_text(path: Path, content: str) -> None:
    """Write UTF-8 text and create the destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _latex_escape(value: str) -> str:
    """Escape characters that have meaning in a LaTeX table."""

    return value.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")


def _write_csv(path: Path, headers: list[str], rows: Iterable[Iterable[str]]) -> None:
    """Write a standards-compliant UTF-8 CSV table."""

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    _write_text(path, buffer.getvalue())


def _write_latex_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    """Write a compact standalone LaTeX table."""

    columns = "l" * len(headers)
    lines = [f"\\begin{{tabular}}{{{columns}}}", "\\hline"]
    lines.append(" & ".join(_latex_escape(value) for value in headers) + " \\\\")
    lines.append("\\hline")
    lines.extend([" & ".join(_latex_escape(value) for value in row) + " \\\\" for row in rows])
    lines.extend(["\\hline", "\\end{tabular}", ""])
    _write_text(path, "\n".join(lines))


def _number(value: Any, decimals: int | None = None) -> str:
    """Format a numeric table value without inventing missing values."""

    if value == "n/a":
        return "n/a"
    if decimals is None:
        return str(value)
    return f"{float(value):.{decimals}f}"


def _dataset_entries(run_index: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Return all model-family variants for one dataset."""

    return {key: entry for key, entry in run_index.items() if key.startswith(f"{dataset}:")}


def validate_paper_summaries(
    run_index: dict[str, Any],
    required_datasets: tuple[str, ...],
    required_variants: tuple[str, ...] = REQUIRED_MAIN_VARIANTS,
) -> None:
    """Reject incomplete inputs before writing a paper package."""

    missing_datasets = [dataset for dataset in required_datasets if not _dataset_entries(run_index, dataset)]
    if missing_datasets:
        raise ValueError(f"Missing paper datasets: {missing_datasets}")

    missing_variants: list[str] = []
    for dataset in required_datasets:
        entries = _dataset_entries(run_index, dataset)
        variants = {key.split(":", 2)[2] for key in entries}
        for variant in required_variants:
            if variant not in variants:
                missing_variants.append(f"{dataset}:{variant}")
    if missing_variants:
        raise ValueError(f"Missing required paper variants: {missing_variants}")


def _split_values(entry: dict[str, Any]) -> tuple[str, str]:
    """Return primary and secondary final evaluation split labels."""

    splits = entry.get("evaluation_splits", {})
    return str(splits.get("final_primary", "unspecified")), str(splits.get("final_secondary", ""))


def _selected_depth_label(entry: dict[str, Any]) -> str:
    """Format all selected depths observed across runs."""

    depths = sorted({int(depth) for depth in entry.get("selected_depths", [])})
    return ";".join(str(depth) for depth in depths) if depths else "n/a"


def _summary_rows(dataset: str, entries: dict[str, Any], decimals: int | None = None) -> list[list[str]]:
    """Build the main paper table rows for one dataset."""

    rows: list[list[str]] = []
    for key, entry in sorted(entries.items()):
        _, model_family, variant = key.split(":", 2)
        primary_split, secondary_split = _split_values(entry)
        rows.append(
            [
                dataset,
                model_family,
                variant,
                str(entry.get("num_runs", 0)),
                ";".join(str(seed) for seed in entry.get("seeds", [])),
                _selected_depth_label(entry),
                _number(entry.get("final_metric_mean", 0.0), decimals),
                _number(entry.get("final_metric_std", 0.0), decimals),
                primary_split,
                secondary_split,
            ]
        )
    return rows


def _deployment_rows(dataset: str, entries: dict[str, Any], decimals: int | None = None) -> list[list[str]]:
    """Build selected-prefix deployment cost rows for one dataset."""

    rows: list[list[str]] = []
    for key, entry in sorted(entries.items()):
        _, model_family, variant = key.split(":", 2)
        depths = sorted({str(depth) for depth in entry.get("selected_depths", [])})
        costs = entry.get("final_costs_by_batch_size", {})
        for depth in depths:
            latency = costs.get("latency_ms_p50", {}).get("1", {}).get(depth, {}).get("mean", "n/a")
            memory = costs.get("peak_memory_mb", {}).get("1", {}).get(depth, {}).get("mean", "n/a")
            model_bytes = costs.get("model_bytes", {}).get("1", {}).get(depth, {}).get("mean", "n/a")
            rows.append(
                [
                    dataset,
                    model_family,
                    variant,
                    depth,
                    _number(latency, decimals),
                    _number(memory, decimals),
                    _number(model_bytes, 0 if model_bytes != "n/a" else None),
                ]
            )
    return rows


def _core_entries(entries: dict[str, Any], required_variants: tuple[str, ...]) -> dict[str, Any]:
    """Keep only the variants authorised for the core publication tables."""

    return {
        key: entry
        for key, entry in entries.items()
        if key.rsplit(":", 1)[-1] in required_variants
    }


def _core_summary_rows(
    dataset: str,
    entries: dict[str, Any],
    decimals: int | None = None,
) -> list[list[str]]:
    """Build the cross-dataset core results table."""

    rows: list[list[str]] = []
    for key, entry in sorted(entries.items()):
        _, model_family, variant = key.split(":", 2)
        primary_split, secondary_split = _split_values(entry)
        rows.append(
            [
                dataset,
                model_family,
                variant,
                str(entry.get("task_metric_name", "metric")),
                primary_split,
                secondary_split,
                str(entry.get("num_runs", 0)),
                ";".join(str(seed) for seed in entry.get("seeds", [])),
                _selected_depth_label(entry),
                _number(entry.get("final_metric_mean", 0.0), decimals),
                _number(entry.get("final_metric_std", 0.0), decimals),
            ]
        )
    return rows


def _write_secondary_tables(dataset: str, entries: dict[str, Any], output_root: Path) -> list[str]:
    """Write secondary evaluation tables, such as MNLI mismatched validation."""

    rows: list[list[str]] = []
    for key, entry in sorted(entries.items()):
        secondary = entry.get("final_results_secondary_by_depth")
        if not secondary:
            continue
        metric_name = entry.get("task_metric_name", "metric")
        for depth, values in secondary.get(metric_name, {}).items():
            _, model_family, variant = key.split(":", 2)
            rows.append([dataset, model_family, variant, depth, f"{float(values['mean']):.4f}", f"{float(values['std']):.4f}"])
    if not rows:
        return []

    headers = ["dataset", "model", "variant", "depth", "secondary_metric_mean", "secondary_metric_std"]
    csv_path = output_root / "tables" / f"{dataset}_secondary_summary.csv"
    tex_path = output_root / "tables" / f"{dataset}_secondary_summary.tex"
    _write_csv(csv_path, headers, rows)
    _write_latex_table(tex_path, headers, rows)
    return [str(csv_path), str(tex_path)]


def export_paper_assets(
    processed_root: Path,
    output_root: Path,
    datasets: tuple[str, ...] | None = None,
    required_datasets: tuple[str, ...] | None = None,
) -> None:
    """Export validated processed summaries as paper tables and a manifest."""

    run_index_path = processed_root / "run_index.json"
    if not run_index_path.exists():
        raise ValueError(f"Processed run index is missing: {run_index_path}")
    run_index = json.loads(run_index_path.read_text(encoding="utf-8"))
    available_datasets = sorted({key.split(":", 1)[0] for key in run_index})
    selected_datasets = tuple(datasets or available_datasets)
    missing_requested = [dataset for dataset in selected_datasets if dataset not in available_datasets]
    if missing_requested:
        raise ValueError(f"Requested datasets are missing: {missing_requested}")
    if required_datasets is not None:
        validate_paper_summaries(run_index, required_datasets)

    asset_paths: list[str] = []
    for dataset in selected_datasets:
        entries = _dataset_entries(run_index, dataset)
        summary_rows = _summary_rows(dataset, entries)
        summary_headers = [
            "dataset",
            "model",
            "variant",
            "num_runs",
            "seeds",
            "selected_depths",
            "final_metric_mean",
            "final_metric_std",
            "final_primary_split",
            "final_secondary_split",
        ]
        summary_csv = output_root / "tables" / f"{dataset}_summary.csv"
        summary_tex = output_root / "tables" / f"{dataset}_summary.tex"
        _write_csv(summary_csv, summary_headers, summary_rows)
        _write_latex_table(summary_tex, summary_headers, _summary_rows(dataset, entries, decimals=4))
        asset_paths.extend([str(summary_csv), str(summary_tex)])

        deployment_rows = _deployment_rows(dataset, entries)
        deployment_headers = ["dataset", "model", "variant", "depth", "latency_ms_p50", "peak_memory_mb", "model_bytes"]
        deployment_csv = output_root / "tables" / f"{dataset}_deployment.csv"
        deployment_tex = output_root / "tables" / f"{dataset}_deployment.tex"
        _write_csv(deployment_csv, deployment_headers, deployment_rows)
        _write_latex_table(deployment_tex, deployment_headers, _deployment_rows(dataset, entries, decimals=4))
        asset_paths.extend([str(deployment_csv), str(deployment_tex)])
        asset_paths.extend(_write_secondary_tables(dataset, entries, output_root))

    core_datasets = tuple(required_datasets or selected_datasets)
    core_entries = {
        dataset: _core_entries(_dataset_entries(run_index, dataset), REQUIRED_MAIN_VARIANTS)
        for dataset in core_datasets
    }
    core_headers = [
        "dataset",
        "model",
        "variant",
        "metric",
        "final_primary_split",
        "final_secondary_split",
        "num_runs",
        "seeds",
        "selected_depths",
        "final_metric_mean",
        "final_metric_std",
    ]
    core_rows = [
        row
        for dataset in core_datasets
        for row in _core_summary_rows(dataset, core_entries[dataset])
    ]
    core_csv = output_root / "tables" / "core_results.csv"
    core_tex = output_root / "tables" / "core_results.tex"
    _write_csv(core_csv, core_headers, core_rows)
    _write_latex_table(
        core_tex,
        core_headers,
        [
            row
            for dataset in core_datasets
            for row in _core_summary_rows(dataset, core_entries[dataset], decimals=4)
        ],
    )
    asset_paths.extend([str(core_csv), str(core_tex)])

    core_deployment_headers = [
        "dataset",
        "model",
        "variant",
        "depth",
        "latency_ms_p50",
        "peak_memory_mb",
        "model_bytes",
    ]
    core_deployment_rows = [
        row
        for dataset in core_datasets
        for row in _deployment_rows(dataset, core_entries[dataset])
    ]
    core_deployment_csv = output_root / "tables" / "core_deployment.csv"
    core_deployment_tex = output_root / "tables" / "core_deployment.tex"
    _write_csv(core_deployment_csv, core_deployment_headers, core_deployment_rows)
    _write_latex_table(
        core_deployment_tex,
        core_deployment_headers,
        [
            row
            for dataset in core_datasets
            for row in _deployment_rows(dataset, core_entries[dataset], decimals=4)
        ],
    )
    asset_paths.extend([str(core_deployment_csv), str(core_deployment_tex)])

    manifest = {
        "datasets": list(selected_datasets),
        "required_variants": list(REQUIRED_MAIN_VARIANTS),
        "num_variants": sum(len(_dataset_entries(run_index, dataset)) for dataset in selected_datasets),
        "tables": asset_paths,
        "figures": sorted(
            str(path)
            for path in (output_root / "figures").rglob("*")
            if path.is_file() and path.name != "manifest.json"
        ) if (output_root / "figures").exists() else [],
    }
    _write_text(output_root / "manifest.json", json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    """Parse the paper export command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="pods/results/processed")
    parser.add_argument("--output-root", default="paper")
    parser.add_argument("--datasets", default=",".join(PAPER_DATASETS))
    return parser.parse_args()


def main() -> None:
    """Export the configured paper package."""

    args = parse_args()
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    export_paper_assets(
        Path(args.processed_root),
        Path(args.output_root),
        datasets=datasets,
        required_datasets=datasets,
    )


if __name__ == "__main__":
    main()
