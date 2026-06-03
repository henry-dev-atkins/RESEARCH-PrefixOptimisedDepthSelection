from __future__ import annotations

import json

from pods.experiments.matrix_status import collect_matrix_status, write_matrix_status


def test_collect_matrix_status_reports_missing_and_completed(tmp_path) -> None:
    config_root = tmp_path / "configs"
    raw_root = tmp_path / "raw"
    config_root.mkdir()
    raw_root.mkdir()

    (config_root / "a.yaml").write_text("x: 1", encoding="utf-8")
    (config_root / "b.yaml").write_text("x: 1", encoding="utf-8")
    (raw_root / "a").mkdir()
    (raw_root / "a" / "results.json").write_text(
        '{"task": {}, "model": {}, "baseline": {}, "selected_depth": 12, "final_results": {"12": {"metric_value": 0.9}}}',
        encoding="utf-8",
    )
    (raw_root / "a" / "config.snapshot.yaml").write_text("experiment:\n  name: a\n", encoding="utf-8")
    (raw_root / "orphan").mkdir()

    status = collect_matrix_status(config_root, raw_root)

    assert status["num_configs"] == 2
    assert status["num_completed"] == 1
    assert status["num_missing"] == 1
    assert status["num_invalid_raw"] == 1
    assert status["completed"] == ["a"]
    assert status["missing"] == ["b"]
    assert status["invalid_raw"] == {"orphan": "missing results.json"}
    assert status["extra_raw"] == ["orphan"]


def test_collect_matrix_status_treats_invalid_matching_run_as_missing(tmp_path) -> None:
    config_root = tmp_path / "configs"
    raw_root = tmp_path / "raw"
    config_root.mkdir()
    raw_root.mkdir()

    (config_root / "bad.yaml").write_text("x: 1", encoding="utf-8")
    (raw_root / "bad").mkdir()
    (raw_root / "bad" / "results.json").write_text("{}", encoding="utf-8")
    (raw_root / "bad" / "config.snapshot.yaml").write_text("experiment:\n  name: bad\n", encoding="utf-8")

    status = collect_matrix_status(config_root, raw_root)

    assert status["completed"] == []
    assert status["missing"] == ["bad"]
    assert status["invalid_raw"] == {"bad": "results.json missing key 'task'"}


def test_write_matrix_status_writes_json(tmp_path) -> None:
    output_path = tmp_path / "status.json"
    write_matrix_status(output_path, {"ok": True})
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True}
