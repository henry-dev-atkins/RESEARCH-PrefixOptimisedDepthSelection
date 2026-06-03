from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset
import yaml

from pods.config.schema import (
    BaselineConfig,
    BenchmarkConfig,
    ExperimentConfig,
    ModelConfig,
    PodsConfig,
    SelectionConfig,
    TaskConfig,
    TrainingConfig,
)
from pods.models.mlp import PodsMLP
from pods.experiments.runner import (
    _load_checkpoint_with_allowed_missing,
    build_benchmark_batches,
    load_teacher_model,
    make_selection_cost_results,
    run_workflow,
)
from pods.eval.summarise import summarise_results


def _make_config(output_dir: str) -> PodsConfig:
    return PodsConfig(
        experiment=ExperimentConfig(name="test", seed=1, output_dir=output_dir),
        task=TaskConfig(type="image_classification", dataset="mnist", num_classes=3),
        model=ModelConfig(
            family="mlp",
            candidate_depths=(1, 2),
            head_type="linear",
            norm_type="none",
            input_shape=(1, 4, 4),
            hidden_dims=(8, 8),
        ),
        training=TrainingConfig(
            mode="sandwich",
            epochs=1,
            batch_size=8,
            optimiser="adamw",
            learning_rate=0.001,
            weight_decay=0.0,
            grad_clip=1.0,
            mixed_precision="none",
            weight_power=1.0,
            num_middle_depths=0,
            scheduler="none",
        ),
        selection=SelectionConfig(
            quality_metric="accuracy",
            cost_metric="latency_ms_p50",
            target_kind="relative",
            relative_accuracy_drop=0.1,
            absolute_accuracy_target=None,
        ),
        benchmark=BenchmarkConfig(
            batch_sizes=(1, 4),
            selection_batch_size=4,
            warmup_steps=1,
            timed_steps=2,
        ),
        baseline=BaselineConfig(kind="pods"),
    )


def test_build_benchmark_batches_covers_every_configured_batch_size() -> None:
    dataset = TensorDataset(torch.randn(10, 1, 4, 4), torch.randint(0, 3, (10,)))
    batches = build_benchmark_batches(dataset, (1, 4))
    assert set(batches) == {1, 4}
    assert batches[1].shape[0] == 1
    assert batches[4].shape[0] == 4


def test_make_selection_cost_results_uses_configured_batch_size() -> None:
    from pods.eval.select import PrefixCostResult

    results = {
        1: {
            1: PrefixCostResult(depth=1, latency_ms_p50=1.0, latency_ms_p95=1.0, throughput_items_per_s=1.0, peak_memory_mb=1.0, model_bytes=1),
            4: PrefixCostResult(depth=1, latency_ms_p50=4.0, latency_ms_p95=4.0, throughput_items_per_s=1.0, peak_memory_mb=1.0, model_bytes=1),
        }
    }
    selected = make_selection_cost_results(results, 4)
    assert selected[1].latency_ms_p50 == 4.0


def test_run_workflow_persists_cost_results_and_final_result(tmp_path) -> None:
    config = _make_config(str(tmp_path))
    model = PodsMLP(input_shape=(1, 4, 4), hidden_dims=(8, 8), num_classes=3, candidate_depths=(1, 2))

    train_dataset = TensorDataset(torch.randn(16, 1, 4, 4), torch.randint(0, 3, (16,)))
    val_dataset = TensorDataset(torch.randn(8, 1, 4, 4), torch.randint(0, 3, (8,)))
    test_dataset = TensorDataset(torch.randn(8, 1, 4, 4), torch.randint(0, 3, (8,)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    output, _, _ = run_workflow(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        selection_val_dataset=val_dataset,
        device=torch.device("cpu"),
    )

    assert "cost_results" in output
    assert "selection_spec" in output
    assert "final_results" in output
    assert set(output["cost_results"][1].keys()) == {1, 4}
    assert output["selected_depth"] in {1, 2}


def test_run_workflow_layerdrop_evaluates_and_selects_across_all_candidate_depths(tmp_path) -> None:
    config = _make_config(str(tmp_path))
    config = PodsConfig(
        experiment=config.experiment,
        task=config.task,
        model=config.model,
        training=TrainingConfig(
            mode="layerdrop",
            epochs=1,
            batch_size=8,
            optimiser="adamw",
            learning_rate=0.001,
            weight_decay=0.0,
            grad_clip=1.0,
            mixed_precision="none",
            weight_power=1.0,
            num_middle_depths=0,
            scheduler="none",
            layerdrop_prob=0.2,
        ),
        selection=config.selection,
        benchmark=config.benchmark,
        baseline=BaselineConfig(kind="layerdrop"),
    )
    model = PodsMLP(input_shape=(1, 4, 4), hidden_dims=(8, 8), num_classes=3, candidate_depths=(1, 2))

    train_dataset = TensorDataset(torch.randn(16, 1, 4, 4), torch.randint(0, 3, (16,)))
    val_dataset = TensorDataset(torch.randn(8, 1, 4, 4), torch.randint(0, 3, (8,)))
    test_dataset = TensorDataset(torch.randn(8, 1, 4, 4), torch.randint(0, 3, (8,)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    output, _, _ = run_workflow(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        selection_val_dataset=val_dataset,
        device=torch.device("cpu"),
    )

    assert set(output["selection_results"]) == {1, 2}
    assert set(output["cost_results"]) == {1, 2}
    assert output["resolved_candidate_depths"] == [1, 2]
    assert output["selected_depth"] in {1, 2}


def test_summarise_results_groups_runs_by_dataset_model_and_baseline(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    run_dir = raw_root / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(
        """
        {
          "task": {"dataset": "cifar10"},
          "model": {"family": "resnet"},
          "baseline": {"kind": "pods"},
          "training": {"mode": "all"},
          "experiment": {"seed": 7},
          "selected_depth": 2,
          "selection_spec": {"quality_metric": "accuracy", "cost_metric": "latency_ms_p50"},
          "selection_results": {
            "1": {"accuracy": 0.6, "loss": 1.1},
            "2": {"accuracy": 0.8, "loss": 0.7}
          },
          "cost_results": {
            "1": {
              "1": {"latency_ms_p50": 1.0, "latency_ms_p95": 1.2, "throughput_items_per_s": 100.0, "peak_memory_mb": 10.0, "model_bytes": 1000},
              "4": {"latency_ms_p50": 1.5, "latency_ms_p95": 1.8, "throughput_items_per_s": 300.0, "peak_memory_mb": 12.0, "model_bytes": 1000}
            },
            "2": {
              "1": {"latency_ms_p50": 2.0, "latency_ms_p95": 2.2, "throughput_items_per_s": 80.0, "peak_memory_mb": 20.0, "model_bytes": 2000},
              "4": {"latency_ms_p50": 2.5, "latency_ms_p95": 2.8, "throughput_items_per_s": 260.0, "peak_memory_mb": 22.0, "model_bytes": 2000}
            }
          },
          "final_results": {"2": {"accuracy": 0.8, "loss": 0.65}},
          "final_costs": {
            "1": {"latency_ms_p50": 2.1, "latency_ms_p95": 2.3, "throughput_items_per_s": 79.0, "peak_memory_mb": 21.0, "model_bytes": 2000},
            "4": {"latency_ms_p50": 2.6, "latency_ms_p95": 2.9, "throughput_items_per_s": 255.0, "peak_memory_mb": 23.0, "model_bytes": 2000}
          }
        }
        """,
        encoding="utf-8",
    )
    summary = summarise_results(raw_root)
    key = "cifar10:resnet:pods_all"
    assert key in summary
    assert summary[key]["seeds"] == [7]
    assert summary[key]["selection_quality_metric"] == "accuracy"
    assert summary[key]["selection_cost_metric"] == "latency_ms_p50"
    assert summary[key]["selection_results_by_depth"]["accuracy"]["2"]["mean"] == 0.8
    assert summary[key]["selection_costs_by_depth_and_batch_size"]["latency_ms_p50"]["4"]["2"]["mean"] == 2.5
    assert summary[key]["final_results_by_depth"]["loss"]["2"]["mean"] == 0.65
    assert summary[key]["final_costs_by_batch_size"]["model_bytes"]["1"]["2"]["mean"] == 2000.0


def test_summarise_results_separates_pods_training_modes(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    for run_name, training_mode, accuracy in (
        ("run_all", "all", 0.81),
        ("run_sandwich", "sandwich", 0.77),
    ):
        run_dir = raw_root / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "results.json").write_text(
            f"""
            {{
              "task": {{"dataset": "cifar10"}},
              "model": {{"family": "resnet"}},
              "baseline": {{"kind": "pods"}},
              "training": {{"mode": "{training_mode}"}},
              "experiment": {{"seed": 7}},
              "selected_depth": 3,
              "selection_spec": {{"quality_metric": "accuracy", "cost_metric": "latency_ms_p50"}},
              "selection_results": {{
                "3": {{"accuracy": {accuracy}, "loss": 0.7}}
              }},
              "cost_results": {{
                "3": {{
                  "1": {{"latency_ms_p50": 2.0, "latency_ms_p95": 2.2, "throughput_items_per_s": 80.0, "peak_memory_mb": 20.0, "model_bytes": 2000}}
                }}
              }},
              "final_results": {{"3": {{"accuracy": {accuracy}, "loss": 0.65}}}},
              "final_costs": {{
                "1": {{"latency_ms_p50": 2.1, "latency_ms_p95": 2.3, "throughput_items_per_s": 79.0, "peak_memory_mb": 21.0, "model_bytes": 2000}}
              }}
            }}
            """,
            encoding="utf-8",
        )

    summary = summarise_results(raw_root)
    assert "cifar10:resnet:pods_all" in summary
    assert "cifar10:resnet:pods_sandwich" in summary
    assert summary["cifar10:resnet:pods_all"]["final_accuracy_mean"] == 0.81
    assert summary["cifar10:resnet:pods_sandwich"]["final_accuracy_mean"] == 0.77


def test_summarise_results_separates_shallow_targets(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    for run_name, target_depth, accuracy in (
        ("run_d1", 1, 0.61),
        ("run_d2", 2, 0.74),
    ):
        run_dir = raw_root / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "results.json").write_text(
            f"""
            {{
              "task": {{"dataset": "cifar10"}},
              "model": {{"family": "resnet"}},
              "baseline": {{"kind": "shallow_independent", "target_depth": {target_depth}}},
              "training": {{"mode": "full_only"}},
              "experiment": {{"seed": 7}},
              "selected_depth": {target_depth},
              "selection_spec": {{"quality_metric": "accuracy", "cost_metric": "latency_ms_p50"}},
              "selection_results": {{
                "{target_depth}": {{"accuracy": {accuracy}, "loss": 0.7}}
              }},
              "cost_results": {{
                "{target_depth}": {{
                  "1": {{"latency_ms_p50": 2.0, "latency_ms_p95": 2.2, "throughput_items_per_s": 80.0, "peak_memory_mb": 20.0, "model_bytes": 2000}}
                }}
              }},
              "final_results": {{"{target_depth}": {{"accuracy": {accuracy}, "loss": 0.65}}}},
              "final_costs": {{
                "1": {{"latency_ms_p50": 2.1, "latency_ms_p95": 2.3, "throughput_items_per_s": 79.0, "peak_memory_mb": 21.0, "model_bytes": 2000}}
              }}
            }}
            """,
            encoding="utf-8",
        )

    summary = summarise_results(raw_root)
    assert "cifar10:resnet:shallow_independent_d1" in summary
    assert "cifar10:resnet:shallow_independent_d2" in summary
    assert summary["cifar10:resnet:shallow_independent_d1"]["final_accuracy_mean"] == 0.61
    assert summary["cifar10:resnet:shallow_independent_d2"]["final_accuracy_mean"] == 0.74


def test_load_teacher_model_rejects_mismatched_teacher_metadata(tmp_path) -> None:
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    torch.save({}, teacher_dir / "model.pt")
    (teacher_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "sst2_roberta_base_full", "seed": 7, "output_dir": str(teacher_dir)},
                "task": {
                    "type": "text_classification",
                    "dataset": "sst2",
                    "num_classes": 2,
                    "selection_fraction": 0.1,
                    "text_fields": ["sentence"],
                    "max_length": 128,
                    "metric_name": "accuracy",
                },
                "model": {
                    "family": "roberta",
                    "pretrained_model_name": "roberta-base",
                    "candidate_depths": [2, 4, 6, 8, 10, 12],
                    "head_type": "roberta_cls",
                    "norm_type": "none",
                    "dropout": 0.1,
                },
                "training": {
                    "mode": "full_only",
                    "epochs": 1,
                    "batch_size": 8,
                    "optimiser": "adamw",
                    "learning_rate": 0.00002,
                    "weight_decay": 0.01,
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
                    "selection_batch_size": 1,
                    "warmup_steps": 1,
                    "timed_steps": 1,
                },
                "baseline": {"kind": "full", "target_depth": None},
            }
        ),
        encoding="utf-8",
    )

    config = PodsConfig(
        experiment=ExperimentConfig(name="sst2_bert_base_distilled_shallow_d2", seed=7, output_dir=str(tmp_path / "student")),
        task=TaskConfig(
            type="text_classification",
            dataset="sst2",
            num_classes=2,
            selection_fraction=0.1,
            text_fields=("sentence",),
            max_length=128,
            metric_name="accuracy",
        ),
        model=ModelConfig(
            family="bert",
            candidate_depths=(2, 4, 6, 8, 10, 12),
            head_type="bert_cls",
            norm_type="none",
            pretrained_model_name="bert-base-uncased",
            dropout=0.1,
        ),
        training=TrainingConfig(
            mode="full_only",
            epochs=1,
            batch_size=8,
            optimiser="adamw",
            learning_rate=0.001,
            weight_decay=0.0,
            grad_clip=1.0,
            mixed_precision="none",
            weight_power=1.0,
            num_middle_depths=0,
            scheduler="none",
        ),
        selection=SelectionConfig(
            quality_metric="accuracy",
            cost_metric="latency_ms_p50",
            target_kind="relative",
            relative_accuracy_drop=0.01,
            absolute_accuracy_target=None,
        ),
        benchmark=BenchmarkConfig(batch_sizes=(1, 8), selection_batch_size=1, warmup_steps=1, timed_steps=1),
        baseline=BaselineConfig(kind="distilled_shallow", target_depth=2, teacher_output_dir=str(teacher_dir)),
    )

    with pytest.raises(ValueError, match="Teacher backbone family does not match"):
        load_teacher_model(config, torch.device("cpu"))


def test_load_checkpoint_with_allowed_missing_accepts_only_whitelisted_gap() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    state_dict = model.state_dict()
    partial = {key: value for key, value in state_dict.items() if key != "1.bias"}

    _load_checkpoint_with_allowed_missing(model, partial, allowed_missing_keys=("1.bias",))


def test_load_checkpoint_with_allowed_missing_rejects_other_gaps() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    state_dict = model.state_dict()
    partial = {key: value for key, value in state_dict.items() if key != "1.bias"}

    with pytest.raises(RuntimeError, match="Checkpoint state_dict is incompatible"):
        _load_checkpoint_with_allowed_missing(model, partial, allowed_missing_keys=())
