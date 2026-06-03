from __future__ import annotations

import argparse
import json
from pathlib import Path


def _validate_raw_run(run_dir: Path) -> str | None:
    results_path = run_dir / "results.json"
    config_path = run_dir / "config.snapshot.yaml"
    if not results_path.exists():
        return "missing results.json"
    if not config_path.exists():
        return "missing config.snapshot.yaml"
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"invalid results.json: {exc.msg}"
    for required_key in ("task", "model", "baseline", "selected_depth", "final_results"):
        if required_key not in payload:
            return f"results.json missing key '{required_key}'"
    return None


def collect_matrix_status(config_root: Path, raw_root: Path) -> dict[str, object]:
    config_stems = sorted(path.stem for path in config_root.glob("*.yaml"))
    raw_dirs = sorted(path for path in raw_root.iterdir() if path.is_dir()) if raw_root.exists() else []
    raw_names = [path.name for path in raw_dirs]
    invalid_raw: dict[str, str] = {}
    valid_raw: set[str] = set()
    for run_dir in raw_dirs:
        reason = _validate_raw_run(run_dir)
        if reason is None:
            valid_raw.add(run_dir.name)
        else:
            invalid_raw[run_dir.name] = reason

    completed = [stem for stem in config_stems if stem in valid_raw]
    missing = [stem for stem in config_stems if stem not in valid_raw]
    extra_raw = [name for name in raw_names if name not in set(config_stems)]

    return {
        "config_root": str(config_root),
        "raw_root": str(raw_root),
        "num_configs": len(config_stems),
        "num_completed": len(completed),
        "num_missing": len(missing),
        "num_invalid_raw": len(invalid_raw),
        "completed": completed,
        "missing": missing,
        "invalid_raw": invalid_raw,
        "extra_raw": extra_raw,
    }


def write_matrix_status(output_path: Path, status: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-root", default="pods/configs/glue")
    parser.add_argument("--raw-root", default="pods/results/raw")
    parser.add_argument("--output-path", default="pods/results/processed/planning/glue_matrix_status.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = collect_matrix_status(Path(args.config_root), Path(args.raw_root))
    write_matrix_status(Path(args.output_path), status)


if __name__ == "__main__":
    main()
