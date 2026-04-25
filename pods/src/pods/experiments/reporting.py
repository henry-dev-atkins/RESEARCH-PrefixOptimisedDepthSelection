from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from pods.config.schema import PodsConfig, config_to_dict


def build_runtime_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)) if hasattr(torch.backends, "cudnn") else False,
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)) if hasattr(torch.backends, "cudnn") else False,
    }


def save_raw_run(output_dir: Path, config: PodsConfig, results: dict[str, Any], model_state: dict[str, Any] | None, selected_prefix_state: dict[str, Any] | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False), encoding="utf-8")
    if model_state is not None and config.experiment.save_model:
        torch.save(model_state, output_dir / "model.pt")
    if selected_prefix_state is not None and config.experiment.save_selected_prefix:
        torch.save(selected_prefix_state, output_dir / "selected_prefix.pt")


def write_processed_summary(output_path: Path, summary: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
