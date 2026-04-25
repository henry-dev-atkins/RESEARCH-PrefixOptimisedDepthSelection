from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

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
from pods.experiments.runner import build_benchmark_batches, make_selection_cost_results, run_workflow
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
