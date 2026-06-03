from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from pods.config.schema import load_config
from pods.experiments.runner import require_supported_runtime


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _base_config() -> dict:
    return {
        "experiment": {"name": "test", "seed": 1, "output_dir": "temp/test"},
        "task": {"type": "image_classification", "dataset": "mnist", "num_classes": 10},
        "model": {
            "family": "mlp",
            "input_shape": [1, 28, 28],
            "hidden_dims": [16, 8],
            "candidate_depths": [1, 2],
            "head_type": "linear",
            "norm_type": "none",
        },
        "training": {
            "mode": "sandwich",
            "epochs": 1,
            "batch_size": 8,
            "optimiser": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "mixed_precision": "none",
            "weight_power": 1.0,
            "num_middle_depths": 0,
            "scheduler": "none",
        },
        "selection": {
            "quality_metric": "accuracy",
            "cost_metric": "latency_ms_p50",
            "relative_accuracy_drop": 0.01,
            "absolute_accuracy_target": None,
        },
        "benchmark": {
            "batch_sizes": [1, 8],
            "selection_batch_size": 8,
            "warmup_steps": 1,
            "timed_steps": 2,
        },
        "baseline": {
            "kind": "pods",
            "target_depth": None,
            "head_fit_epochs": 1,
            "head_fit_learning_rate": 0.001,
        },
    }


def test_config_rejects_unsupported_head_type(tmp_path: Path) -> None:
    data = _base_config()
    data["model"]["head_type"] = "mlp"
    with pytest.raises(ValueError, match="head_type"):
        load_config(_write_config(tmp_path, data))


def test_config_rejects_invalid_selection_batch_size(tmp_path: Path) -> None:
    data = _base_config()
    data["benchmark"]["selection_batch_size"] = 4
    with pytest.raises(ValueError, match="selection_batch_size"):
        load_config(_write_config(tmp_path, data))


def test_require_supported_runtime_rejects_cpu_mixed_precision(tmp_path: Path) -> None:
    data = _base_config()
    data["training"]["mixed_precision"] = "bf16"
    config = load_config(_write_config(tmp_path, data))
    with pytest.raises(ValueError, match="CPU execution is unsupported"):
        require_supported_runtime(config, torch.device("cpu"))


def test_require_supported_runtime_rejects_cpu_peak_memory_selection(tmp_path: Path) -> None:
    data = _base_config()
    data["selection"]["cost_metric"] = "peak_memory_mb"
    config = load_config(_write_config(tmp_path, data))
    with pytest.raises(ValueError, match="CPU execution is unsupported"):
        require_supported_runtime(config, torch.device("cpu"))


def test_require_supported_runtime_rejects_cpu_even_without_special_cuda_features(tmp_path: Path) -> None:
    data = _base_config()
    config = load_config(_write_config(tmp_path, data))
    with pytest.raises(ValueError, match="CPU execution is unsupported"):
        require_supported_runtime(config, torch.device("cpu"))


def test_config_accepts_bert_text_classification(tmp_path: Path) -> None:
    data = _base_config()
    data["task"] = {
        "type": "text_classification",
        "dataset": "sst2",
        "num_classes": 2,
        "selection_fraction": 0.1,
        "text_fields": ["sentence"],
        "max_length": 128,
        "metric_name": "accuracy",
    }
    data["model"] = {
        "family": "bert",
        "pretrained_model_name": "bert-base-uncased",
        "candidate_depths": [2, 4, 6, 8, 10, 12],
        "head_type": "bert_cls",
        "norm_type": "none",
        "dropout": 0.1,
    }
    config = load_config(_write_config(tmp_path, data))
    assert config.model.family == "bert"
    assert config.task.text_fields == ("sentence",)


def test_config_rejects_text_task_with_resnet(tmp_path: Path) -> None:
    data = _base_config()
    data["task"] = {
        "type": "text_classification",
        "dataset": "sst2",
        "num_classes": 2,
        "selection_fraction": 0.1,
        "text_fields": ["sentence"],
        "max_length": 128,
        "metric_name": "accuracy",
    }
    data["model"] = {
        "family": "resnet",
        "variant": "resnet56",
        "input_shape": [3, 32, 32],
        "candidate_depths": [1, 2, 3],
        "head_type": "linear",
        "norm_type": "groupnorm",
        "groupnorm_groups": 8,
    }
    with pytest.raises(ValueError, match="ResNet models require image_classification tasks"):
        load_config(_write_config(tmp_path, data))


def test_config_accepts_roberta_text_classification(tmp_path: Path) -> None:
    data = _base_config()
    data["task"] = {
        "type": "text_classification",
        "dataset": "mrpc",
        "num_classes": 2,
        "selection_fraction": 0.1,
        "text_fields": ["sentence1", "sentence2"],
        "max_length": 128,
        "metric_name": "accuracy",
    }
    data["model"] = {
        "family": "roberta",
        "pretrained_model_name": "roberta-base",
        "candidate_depths": [2, 4, 6, 8, 10, 12],
        "head_type": "roberta_cls",
        "norm_type": "none",
        "dropout": 0.1,
    }
    data["baseline"]["kind"] = "layerdrop"
    data["training"]["mode"] = "layerdrop"
    data["training"]["layerdrop_prob"] = 0.2
    config = load_config(_write_config(tmp_path, data))
    assert config.model.family == "roberta"
    assert config.training.mode == "layerdrop"


def test_config_accepts_extended_glue_tasks_and_transformer_variants(tmp_path: Path) -> None:
    data = _base_config()
    data["task"] = {
        "type": "text_classification",
        "dataset": "qqp",
        "num_classes": 2,
        "selection_fraction": 0.1,
        "text_fields": ["question1", "question2"],
        "max_length": 128,
        "metric_name": "accuracy",
    }
    data["model"] = {
        "family": "bert",
        "pretrained_model_name": "bert-base-uncased",
        "candidate_depths": [3, 6, 9, 12],
        "head_type": "bert_cls_mlp",
        "norm_type": "none",
        "dropout": 0.1,
    }
    data["training"]["mode"] = "progressive_sandwich"
    data["training"]["progressive_middle_depths"] = [0, 1, 2]
    data["baseline"]["kind"] = "progressive_shrinking"
    data["selection"]["target_kind"] = "confidence_bound_relative"
    config = load_config(_write_config(tmp_path, data))
    assert config.task.dataset == "qqp"
    assert config.model.candidate_depths == (3, 6, 9, 12)
    assert config.training.mode == "progressive_sandwich"
    assert config.baseline.kind == "progressive_shrinking"


def test_config_accepts_cola_metric_and_selection_metric(tmp_path: Path) -> None:
    data = _base_config()
    data["task"] = {
        "type": "text_classification",
        "dataset": "cola",
        "num_classes": 2,
        "selection_fraction": 0.1,
        "text_fields": ["sentence"],
        "max_length": 128,
        "metric_name": "matthews_correlation",
    }
    data["model"] = {
        "family": "bert",
        "pretrained_model_name": "bert-base-uncased",
        "candidate_depths": [2, 6, 12],
        "head_type": "bert_cls_linear",
        "norm_type": "none",
        "dropout": 0.1,
    }
    data["selection"]["quality_metric"] = "matthews_correlation"
    config = load_config(_write_config(tmp_path, data))
    assert config.task.metric_name == "matthews_correlation"
    assert config.selection.quality_metric == "matthews_correlation"


def test_config_accepts_stsb_text_regression(tmp_path: Path) -> None:
    data = _base_config()
    data["task"] = {
        "type": "text_regression",
        "dataset": "stsb",
        "num_classes": 1,
        "selection_fraction": 0.1,
        "text_fields": ["sentence1", "sentence2"],
        "max_length": 128,
        "metric_name": "spearman_correlation",
    }
    data["model"] = {
        "family": "bert",
        "pretrained_model_name": "bert-base-uncased",
        "candidate_depths": [2, 4, 6, 8, 10, 12],
        "head_type": "bert_cls",
        "norm_type": "none",
        "dropout": 0.1,
    }
    data["selection"]["quality_metric"] = "spearman_correlation"
    config = load_config(_write_config(tmp_path, data))
    assert config.task.type == "text_regression"
    assert config.task.dataset == "stsb"


def test_config_accepts_mnli_text_classification(tmp_path: Path) -> None:
    data = _base_config()
    data["task"] = {
        "type": "text_classification",
        "dataset": "mnli",
        "num_classes": 3,
        "selection_fraction": 0.1,
        "text_fields": ["premise", "hypothesis"],
        "max_length": 128,
        "metric_name": "accuracy",
    }
    data["model"] = {
        "family": "roberta",
        "pretrained_model_name": "roberta-base",
        "candidate_depths": [2, 4, 6, 8, 10, 12],
        "head_type": "roberta_cls_linear",
        "norm_type": "none",
        "dropout": 0.1,
    }
    config = load_config(_write_config(tmp_path, data))
    assert config.task.dataset == "mnli"
    assert config.task.num_classes == 3


def test_config_accepts_distilled_shallow_baseline(tmp_path: Path) -> None:
    data = _base_config()
    data["task"] = {
        "type": "text_classification",
        "dataset": "sst2",
        "num_classes": 2,
        "selection_fraction": 0.1,
        "text_fields": ["sentence"],
        "max_length": 128,
        "metric_name": "accuracy",
    }
    data["model"] = {
        "family": "bert",
        "pretrained_model_name": "bert-base-uncased",
        "candidate_depths": [2, 4, 6, 8, 10, 12],
        "head_type": "bert_cls",
        "norm_type": "none",
        "dropout": 0.1,
    }
    data["training"]["mode"] = "full_only"
    data["baseline"] = {
        "kind": "distilled_shallow",
        "target_depth": 2,
        "teacher_output_dir": "pods/results/raw/sst2_bert_base_full",
        "distillation_alpha": 0.5,
        "distillation_temperature": 2.0,
    }
    config = load_config(_write_config(tmp_path, data))
    assert config.baseline.kind == "distilled_shallow"
    assert config.baseline.teacher_output_dir is not None


def test_config_accepts_snapshot_style_null_input_shape_for_transformers(tmp_path: Path) -> None:
    data = _base_config()
    data["task"] = {
        "type": "text_classification",
        "dataset": "sst2",
        "num_classes": 2,
        "selection_fraction": 0.1,
        "text_fields": ["sentence"],
        "max_length": 128,
        "metric_name": "accuracy",
    }
    data["model"] = {
        "family": "bert",
        "pretrained_model_name": "bert-base-uncased",
        "candidate_depths": [2, 4, 6, 8, 10, 12],
        "head_type": "bert_cls",
        "norm_type": "none",
        "input_shape": None,
        "hidden_dims": [],
        "channels": [],
        "variant": None,
        "dropout": 0.1,
    }
    config = load_config(_write_config(tmp_path, data))
    assert config.model.input_shape is None
