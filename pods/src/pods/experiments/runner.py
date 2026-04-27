from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from pods.config.schema import PodsConfig
from pods.data.batches import move_batch_to_device
from pods.data.datasets import build_image_dataset, build_text_datasets
from pods.data.splits import DatasetSplit, make_split_indices
from pods.eval.benchmark import benchmark_prefix
from pods.eval.quality import evaluate_all_prefixes, evaluate_classifier
from pods.eval.select import PrefixCostResult, SelectionSpec, select_prefix_depth
from pods.eval.summarise import summarise_by_dataset, summarise_results
from pods.experiments.reporting import build_runtime_metadata, save_raw_run, write_processed_summary
from pods.models.interfaces import PrefixCapableModel
from pods.models.registry import build_model
from pods.training.loops import train_multidepth_model
from pods.utils.seeds import seed_everything


def build_optimizer(config: PodsConfig, model: torch.nn.Module):
    if config.training.optimiser == "adamw":
        return optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    if config.training.optimiser == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            momentum=config.training.momentum,
        )
    raise ValueError(f"Unsupported optimiser: {config.training.optimiser}")


def require_supported_runtime(config: PodsConfig, device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("PODS ML experiments require CUDA. CPU execution is unsupported for training and benchmarking.")
    if device.type != "cuda" and config.training.mixed_precision != "none":
        raise ValueError(f"training.mixed_precision={config.training.mixed_precision!r} requires CUDA")
    if device.type != "cuda" and config.selection.cost_metric == "peak_memory_mb":
        raise ValueError("selection.cost_metric='peak_memory_mb' requires CUDA for meaningful measurement")


def make_data_loaders(config: PodsConfig):
    collate_fn = None
    if config.task.type == "image_classification":
        dataset_train_full = build_image_dataset(config.task.dataset, train=True)
        dataset_selection_full = build_image_dataset(config.task.dataset, train=True, use_eval_transform=True)
        dataset_test = build_image_dataset(config.task.dataset, train=False)
        indices = make_split_indices(len(dataset_train_full), selection_fraction=config.task.selection_fraction, seed=config.experiment.seed)
        split = DatasetSplit(
            train=torch.utils.data.Subset(dataset_train_full, indices.train_indices),
            selection_val=torch.utils.data.Subset(dataset_selection_full, indices.selection_indices),
        )
    elif config.task.type == "text_classification":
        try:
            from transformers import AutoTokenizer, DataCollatorWithPadding
        except ImportError as exc:
            raise ImportError("Text tasks require the 'transformers' package to be installed") from exc
        datasets = build_text_datasets(config)
        indices = make_split_indices(len(datasets["train"]), selection_fraction=config.task.selection_fraction, seed=config.experiment.seed)
        split = DatasetSplit(
            train=torch.utils.data.Subset(datasets["train"], indices.train_indices),
            selection_val=torch.utils.data.Subset(datasets["train"], indices.selection_indices),
        )
        dataset_test = datasets["validation"]
        tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name)
        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        raise ValueError(f"Unsupported task.type: {config.task.type}")

    train_loader = DataLoader(
        split.train,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers,
        drop_last=config.training.drop_last,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        split.selection_val,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers,
        collate_fn=collate_fn,
    )
    return split, train_loader, val_loader, test_loader


def build_benchmark_batches(dataset, batch_sizes: tuple[int, ...], collate_fn=None) -> dict[int, Any]:
    max_batch_size = max(batch_sizes)
    loader = DataLoader(dataset, batch_size=max_batch_size, shuffle=False, collate_fn=collate_fn)
    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise ValueError("Selection-validation dataset is empty; cannot build benchmark batches") from exc
    inputs, _ = move_batch_to_device(batch, torch.device("cpu"))
    if isinstance(inputs, dict):
        batch_size_seen = next(iter(inputs.values())).shape[0]
        if batch_size_seen < max_batch_size:
            raise ValueError(
                f"Selection-validation loader produced batch size {batch_size_seen}, smaller than requested benchmark batch size {max_batch_size}"
            )
        return {
            batch_size: {key: value[:batch_size].clone() for key, value in inputs.items()}
            for batch_size in batch_sizes
        }
    if inputs.shape[0] < max_batch_size:
        raise ValueError(
            f"Selection-validation loader produced batch size {inputs.shape[0]}, smaller than requested benchmark batch size {max_batch_size}"
        )
    return {batch_size: inputs[:batch_size].clone() for batch_size in batch_sizes}


def benchmark_prefixes(
    model: PrefixCapableModel,
    benchmark_batches: dict[int, torch.Tensor],
    candidate_depths: tuple[int, ...],
    warmup_steps: int,
    timed_steps: int,
    device: torch.device,
    repeat_runs: int,
    confidence_level: float,
) -> dict[int, dict[int, PrefixCostResult]]:
    alpha = (1.0 - confidence_level) / 2.0

    def _ci(values: list[float]) -> tuple[float | None, float | None]:
        if len(values) <= 1:
            return None, None
        sorted_values = sorted(values)
        low_index = max(0, min(len(sorted_values) - 1, int(alpha * len(sorted_values))))
        high_index = max(0, min(len(sorted_values) - 1, int((1.0 - alpha) * len(sorted_values)) - 1))
        return sorted_values[low_index], sorted_values[high_index]

    def _aggregate_runs(depth: int, sample_batch) -> PrefixCostResult:
        runs = [
            benchmark_prefix(
                model=model.extract_prefix(depth),
                sample_batch=sample_batch,
                warmup_steps=warmup_steps,
                timed_steps=timed_steps,
                device=device,
                depth=depth,
            )
            for _ in range(repeat_runs)
        ]
        p50_values = [result.latency_ms_p50 for result in runs]
        p95_values = [result.latency_ms_p95 for result in runs]
        p50_ci_low, p50_ci_high = _ci(p50_values)
        p95_ci_low, p95_ci_high = _ci(p95_values)
        return PrefixCostResult(
            depth=depth,
            latency_ms_p50=sum(p50_values) / len(p50_values),
            latency_ms_p95=sum(p95_values) / len(p95_values),
            throughput_items_per_s=sum(result.throughput_items_per_s for result in runs) / len(runs),
            peak_memory_mb=max(result.peak_memory_mb for result in runs),
            model_bytes=runs[0].model_bytes,
            latency_ms_p50_ci_low=p50_ci_low,
            latency_ms_p50_ci_high=p50_ci_high,
            latency_ms_p95_ci_low=p95_ci_low,
            latency_ms_p95_ci_high=p95_ci_high,
            benchmark_repeats=repeat_runs,
        )

    return {
        depth: {
            batch_size: _aggregate_runs(depth, sample_batch)
            for batch_size, sample_batch in benchmark_batches.items()
        }
        for depth in candidate_depths
    }


def make_selection_cost_results(
    cost_results_by_batch: dict[int, dict[int, PrefixCostResult]],
    selection_batch_size: int,
) -> dict[int, PrefixCostResult]:
    return {depth: batch_results[selection_batch_size] for depth, batch_results in cost_results_by_batch.items()}


def compute_selection_target(selection_results, spec: SelectionSpec) -> float:
    full_depth = max(selection_results)
    full_metric = selection_results[full_depth].metric_value
    if spec.target_kind in {"relative", "confidence_bound_relative"}:
        assert spec.relative_to_full_accuracy_drop is not None
        return full_metric - spec.relative_to_full_accuracy_drop
    assert spec.absolute_accuracy_target is not None
    return spec.absolute_accuracy_target


def fit_posthoc_head(
    model: PrefixCapableModel,
    depth: int,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = model.heads[str(depth)]
    for parameter in head.parameters():
        parameter.requires_grad = True

    optimiser = optim.AdamW(head.parameters(), lr=learning_rate)
    model = model.to(device)
    head.train()
    for _ in range(epochs):
        for batch in train_loader:
            inputs, labels = move_batch_to_device(batch, device)
            with torch.no_grad():
                features = model.forward_features(inputs, depth)
            logits = head(features)
            loss = F.cross_entropy(logits, labels)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

    for parameter in model.parameters():
        parameter.requires_grad = True


def run_workflow(
    config: PodsConfig,
    model: PrefixCapableModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    selection_val_dataset,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    benchmark_batches = build_benchmark_batches(selection_val_dataset, config.benchmark.batch_sizes, collate_fn=val_loader.collate_fn)
    optimizer = build_optimizer(config, model)
    rng = random.Random(config.experiment.seed)
    model, history = train_multidepth_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=config.training.epochs,
        mode=config.training.mode,
        num_middle_depths=config.training.num_middle_depths,
        weight_power=config.training.weight_power,
        device=device,
        mixed_precision=config.training.mixed_precision,
        label_smoothing=config.training.label_smoothing,
        scheduler_name=config.training.scheduler,
        scheduler_t_max=config.training.scheduler_t_max,
        scheduler_milestones=config.training.scheduler_milestones,
        scheduler_gamma=config.training.scheduler_gamma,
        grad_clip=config.training.grad_clip,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        rng=rng,
        progressive_middle_depths=config.training.progressive_middle_depths,
    )

    selection_spec = SelectionSpec(
        quality_metric=config.selection.quality_metric,
        cost_metric=config.selection.cost_metric,
        target_kind=config.selection.target_kind,
        relative_to_full_accuracy_drop=config.selection.relative_accuracy_drop,
        absolute_accuracy_target=config.selection.absolute_accuracy_target,
    )

    if config.baseline.kind == "naive_truncation":
        for depth in model.candidate_depths[:-1]:
            fit_posthoc_head(
                model=model,
                depth=depth,
                train_loader=train_loader,
                device=device,
                epochs=config.baseline.head_fit_epochs,
                learning_rate=config.baseline.head_fit_learning_rate,
            )

    candidate_depths = (
        model.candidate_depths
        if config.baseline.kind in {"pods", "naive_truncation", "layerdrop", "progressive_shrinking"}
        else (max(model.candidate_depths),)
    )
    selection_results = evaluate_all_prefixes(
        model,
        val_loader,
        candidate_depths,
        device,
        metric_name=config.task.metric_name,
        bootstrap_samples=config.experiment.quality_bootstrap_samples,
        confidence_level=config.experiment.confidence_level,
    )
    cost_results_by_batch = benchmark_prefixes(
        model=model,
        benchmark_batches=benchmark_batches,
        candidate_depths=candidate_depths,
        warmup_steps=config.benchmark.warmup_steps,
        timed_steps=config.benchmark.timed_steps,
        device=device,
        repeat_runs=config.benchmark.repeat_runs,
        confidence_level=config.experiment.confidence_level,
    )
    selection_cost_results = make_selection_cost_results(cost_results_by_batch, config.benchmark.selection_batch_size)

    if config.baseline.kind in {"pods", "naive_truncation", "layerdrop", "progressive_shrinking"}:
        selected_depth = select_prefix_depth(selection_results, selection_cost_results, selection_spec)
        selection_target_value = compute_selection_target(selection_results, selection_spec)
    else:
        selected_depth = candidate_depths[0]
        selection_target_value = selection_results[selected_depth].metric_value

    selected_prefix = model.extract_prefix(selected_depth)
    final_result = evaluate_classifier(
        selected_prefix,
        test_loader,
        device,
        depth=selected_depth,
        metric_name=config.task.metric_name,
        bootstrap_samples=config.experiment.quality_bootstrap_samples,
        confidence_level=config.experiment.confidence_level,
    )
    final_cost_by_batch = {
        batch_size: benchmark_prefixes(
            model=model,
            benchmark_batches={batch_size: sample_batch},
            candidate_depths=(selected_depth,),
            warmup_steps=config.benchmark.warmup_steps,
            timed_steps=config.benchmark.timed_steps,
            device=device,
            repeat_runs=config.benchmark.repeat_runs,
            confidence_level=config.experiment.confidence_level,
        )
        [selected_depth][batch_size]
        for batch_size, sample_batch in benchmark_batches.items()
    }

    results = {
        "history": [
            {"train_loss": epoch.train_loss, "val_metrics_by_depth": epoch.val_metrics_by_depth}
            for epoch in history
        ],
        "selection_results": {depth: asdict(result) for depth, result in selection_results.items()},
        "cost_results": {
            depth: {batch_size: asdict(result) for batch_size, result in batch_results.items()}
            for depth, batch_results in cost_results_by_batch.items()
        },
        "selection_spec": asdict(selection_spec),
        "selection_target_value": selection_target_value,
        "selected_depth": selected_depth,
        "final_results": {selected_depth: asdict(final_result)},
        "final_costs": {batch_size: asdict(result) for batch_size, result in final_cost_by_batch.items()},
        "baseline": asdict(config.baseline),
        "experiment": asdict(config.experiment),
        "task": asdict(config.task),
        "model": asdict(config.model),
        "training": asdict(config.training),
        "benchmark": asdict(config.benchmark),
        "runtime": build_runtime_metadata(device),
        "resolved_candidate_depths": list(candidate_depths),
    }
    return results, model.state_dict(), selected_prefix.state_dict()


def run_experiment(config: PodsConfig) -> dict[str, Any]:
    seed_everything(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require_supported_runtime(config, device)

    split, train_loader, val_loader, test_loader = make_data_loaders(config)
    target_depth = config.baseline.target_depth if config.baseline.kind == "shallow_independent" else None
    model = build_model(config, target_depth=target_depth)

    results, model_state, selected_prefix_state = run_workflow(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        selection_val_dataset=split.selection_val,
        device=device,
    )

    output_dir = Path(config.experiment.output_dir)
    save_raw_run(output_dir, config, results, model_state, selected_prefix_state)
    processed = summarise_results(output_dir.parents[1] / "raw")
    processed_root = output_dir.parents[1] / "processed"
    write_processed_summary(processed_root / "run_index.json", processed)
    dataset_names = sorted({key.split(":", 1)[0] for key in processed})
    for dataset_name in dataset_names:
        write_processed_summary(processed_root / f"{dataset_name}_frontier_summary.json", summarise_by_dataset(processed, dataset_name))
    return results
