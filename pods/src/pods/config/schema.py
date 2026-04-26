from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    output_dir: str
    tags: tuple[str, ...] = ()
    save_model: bool = True
    save_selected_prefix: bool = True


@dataclass(frozen=True)
class TaskConfig:
    type: str
    dataset: str
    num_classes: int
    selection_fraction: float = 0.1
    text_fields: tuple[str, ...] = ()
    max_length: int = 128
    metric_name: str = "accuracy"


@dataclass(frozen=True)
class ModelConfig:
    family: str
    candidate_depths: tuple[int, ...]
    head_type: str
    norm_type: str
    input_shape: tuple[int, ...] | None = None
    hidden_dims: tuple[int, ...] = ()
    channels: tuple[int, ...] = ()
    groupnorm_groups: int = 8
    variant: str | None = None
    pretrained_model_name: str | None = None
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    mode: str
    epochs: int
    batch_size: int
    optimiser: str
    learning_rate: float
    weight_decay: float
    grad_clip: float | None
    mixed_precision: str
    weight_power: float
    num_middle_depths: int
    scheduler: str = "none"
    scheduler_t_max: int | None = None
    scheduler_milestones: tuple[int, ...] = ()
    scheduler_gamma: float | None = None
    label_smoothing: float = 0.0
    momentum: float = 0.9
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    drop_last: bool = False
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.0
    max_train_steps: int | None = None
    layerdrop_prob: float = 0.0


@dataclass(frozen=True)
class SelectionConfig:
    quality_metric: str
    cost_metric: str
    relative_accuracy_drop: float | None
    absolute_accuracy_target: float | None


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_sizes: tuple[int, ...]
    selection_batch_size: int
    warmup_steps: int
    timed_steps: int


@dataclass(frozen=True)
class BaselineConfig:
    kind: str
    target_depth: int | None = None
    head_fit_epochs: int = 3
    head_fit_learning_rate: float = 0.001


@dataclass(frozen=True)
class PodsConfig:
    experiment: ExperimentConfig
    task: TaskConfig
    model: ModelConfig
    training: TrainingConfig
    selection: SelectionConfig
    benchmark: BenchmarkConfig
    baseline: BaselineConfig


def _as_tuple_ints(values: list[Any], field_name: str) -> tuple[int, ...]:
    _ensure(isinstance(values, list) and values, f"{field_name} must be a non-empty list")
    result = tuple(int(value) for value in values)
    _ensure(all(value > 0 for value in result), f"{field_name} must contain positive integers")
    return result


def load_config(path: str | Path) -> PodsConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")

    experiment_data = data["experiment"]
    task_data = data["task"]
    model_data = data["model"]
    training_data = data["training"]
    selection_data = data["selection"]
    benchmark_data = data["benchmark"]
    baseline_data = data.get("baseline", {"kind": "pods", "target_depth": None, "head_fit_epochs": 3, "head_fit_learning_rate": 0.001})

    candidate_depths = _as_tuple_ints(model_data["candidate_depths"], "model.candidate_depths")
    _ensure(tuple(sorted(set(candidate_depths))) == candidate_depths, "model.candidate_depths must be sorted and unique")

    config = PodsConfig(
        experiment=ExperimentConfig(
            name=experiment_data["name"],
            seed=int(experiment_data["seed"]),
            output_dir=experiment_data["output_dir"],
            tags=tuple(experiment_data.get("tags", [])),
            save_model=bool(experiment_data.get("save_model", True)),
            save_selected_prefix=bool(experiment_data.get("save_selected_prefix", True)),
        ),
        task=TaskConfig(
            type=task_data["type"],
            dataset=task_data["dataset"],
            num_classes=int(task_data["num_classes"]),
            selection_fraction=float(task_data.get("selection_fraction", 0.1)),
            text_fields=tuple(task_data.get("text_fields", [])),
            max_length=int(task_data.get("max_length", 128)),
            metric_name=task_data.get("metric_name", "accuracy"),
        ),
        model=ModelConfig(
            family=model_data["family"],
            candidate_depths=candidate_depths,
            head_type=model_data["head_type"],
            norm_type=model_data["norm_type"],
            input_shape=tuple(model_data["input_shape"]) if "input_shape" in model_data else None,
            hidden_dims=tuple(model_data.get("hidden_dims", [])),
            channels=tuple(model_data.get("channels", [])),
            groupnorm_groups=int(model_data.get("groupnorm_groups", 8)),
            variant=model_data.get("variant"),
            pretrained_model_name=model_data.get("pretrained_model_name"),
            dropout=float(model_data.get("dropout", 0.1)),
        ),
        training=TrainingConfig(
            mode=training_data["mode"],
            epochs=int(training_data["epochs"]),
            batch_size=int(training_data["batch_size"]),
            optimiser=training_data["optimiser"],
            learning_rate=float(training_data["learning_rate"]),
            weight_decay=float(training_data["weight_decay"]),
            grad_clip=None if training_data["grad_clip"] is None else float(training_data["grad_clip"]),
            mixed_precision=training_data["mixed_precision"],
            weight_power=float(training_data["weight_power"]),
            num_middle_depths=int(training_data["num_middle_depths"]),
            scheduler=training_data.get("scheduler", "none"),
            scheduler_t_max=None if training_data.get("scheduler_t_max") is None else int(training_data["scheduler_t_max"]),
            scheduler_milestones=tuple(int(value) for value in training_data.get("scheduler_milestones", [])),
            scheduler_gamma=None if training_data.get("scheduler_gamma") is None else float(training_data["scheduler_gamma"]),
            label_smoothing=float(training_data.get("label_smoothing", 0.0)),
            momentum=float(training_data.get("momentum", 0.9)),
            num_workers=int(training_data.get("num_workers", 0)),
            pin_memory=bool(training_data.get("pin_memory", False)),
            persistent_workers=bool(training_data.get("persistent_workers", False)),
            drop_last=bool(training_data.get("drop_last", False)),
            gradient_accumulation_steps=int(training_data.get("gradient_accumulation_steps", 1)),
            warmup_ratio=float(training_data.get("warmup_ratio", 0.0)),
            max_train_steps=None if training_data.get("max_train_steps") is None else int(training_data["max_train_steps"]),
            layerdrop_prob=float(training_data.get("layerdrop_prob", 0.0)),
        ),
        selection=SelectionConfig(
            quality_metric=selection_data["quality_metric"],
            cost_metric=selection_data["cost_metric"],
            relative_accuracy_drop=selection_data["relative_accuracy_drop"],
            absolute_accuracy_target=selection_data["absolute_accuracy_target"],
        ),
        benchmark=BenchmarkConfig(
            batch_sizes=_as_tuple_ints(benchmark_data["batch_sizes"], "benchmark.batch_sizes"),
            selection_batch_size=int(benchmark_data["selection_batch_size"]),
            warmup_steps=int(benchmark_data["warmup_steps"]),
            timed_steps=int(benchmark_data["timed_steps"]),
        ),
        baseline=BaselineConfig(
            kind=baseline_data["kind"],
            target_depth=baseline_data.get("target_depth"),
            head_fit_epochs=int(baseline_data.get("head_fit_epochs", 3)),
            head_fit_learning_rate=float(baseline_data.get("head_fit_learning_rate", 0.001)),
        ),
    )
    _validate_config(config)
    return config


def config_to_dict(config: PodsConfig) -> dict[str, Any]:
    return asdict(config)


def _validate_config(config: PodsConfig) -> None:
    _ensure(config.task.type in {"image_classification", "text_classification"}, "task.type is unsupported")
    _ensure(config.task.selection_fraction > 0.0 and config.task.selection_fraction < 1.0, "task.selection_fraction must be between 0 and 1")
    _ensure(config.model.head_type in {"linear", "bert_cls", "roberta_cls"}, "model.head_type is unsupported")
    _ensure(config.model.norm_type in {"none", "groupnorm", "batchnorm"}, "model.norm_type is unsupported")
    _ensure(config.training.mode in {"all", "sandwich", "full_only", "layerdrop"}, "training.mode must be 'all', 'sandwich', 'full_only', or 'layerdrop'")
    _ensure(config.training.mixed_precision in {"none", "fp16", "bf16"}, "training.mixed_precision must be 'none', 'fp16', or 'bf16'")
    _ensure(config.training.scheduler in {"none", "cosine", "multistep"}, "training.scheduler is unsupported")
    _ensure(config.training.label_smoothing >= 0.0, "training.label_smoothing must be non-negative")
    _ensure(config.training.epochs > 0, "training.epochs must be positive")
    _ensure(config.training.batch_size > 0, "training.batch_size must be positive")
    _ensure(config.training.num_middle_depths >= 0, "training.num_middle_depths must be non-negative")
    _ensure(config.training.num_workers >= 0, "training.num_workers must be non-negative")
    _ensure(config.training.gradient_accumulation_steps > 0, "training.gradient_accumulation_steps must be positive")
    _ensure(config.training.warmup_ratio >= 0.0 and config.training.warmup_ratio <= 1.0, "training.warmup_ratio must be between 0 and 1")
    _ensure(config.training.layerdrop_prob >= 0.0 and config.training.layerdrop_prob < 1.0, "training.layerdrop_prob must be between 0 and 1")
    _ensure(not config.training.persistent_workers or config.training.num_workers > 0, "training.persistent_workers requires training.num_workers > 0")
    if config.training.scheduler == "cosine":
        _ensure(config.training.scheduler_t_max is not None, "training.scheduler_t_max is required for cosine scheduler")
    if config.training.scheduler == "multistep":
        _ensure(config.training.scheduler_milestones, "training.scheduler_milestones are required for multistep scheduler")
        _ensure(config.training.scheduler_gamma is not None, "training.scheduler_gamma is required for multistep scheduler")

    _ensure(config.selection.quality_metric == "accuracy", "selection.quality_metric must be 'accuracy'")
    _ensure(
        config.selection.cost_metric in {"latency_ms_p50", "latency_ms_p95", "peak_memory_mb", "model_bytes"},
        "selection.cost_metric is unsupported",
    )
    _ensure(
        (config.selection.relative_accuracy_drop is None) != (config.selection.absolute_accuracy_target is None),
        "Exactly one of selection.relative_accuracy_drop and selection.absolute_accuracy_target must be set",
    )
    _ensure(config.benchmark.selection_batch_size in config.benchmark.batch_sizes, "benchmark.selection_batch_size must be present in benchmark.batch_sizes")
    _ensure(config.baseline.kind in {"pods", "full", "shallow_independent", "naive_truncation", "layerdrop"}, "baseline.kind is unsupported")
    _ensure(config.baseline.head_fit_epochs > 0, "baseline.head_fit_epochs must be positive")
    _ensure(config.baseline.head_fit_learning_rate > 0.0, "baseline.head_fit_learning_rate must be positive")

    if config.training.mode == "sandwich":
        _ensure(len(config.model.candidate_depths) >= 2, "sandwich mode requires at least two candidate depths")
    if config.training.mode == "layerdrop":
        _ensure(config.baseline.kind == "layerdrop", "training.mode='layerdrop' requires baseline.kind='layerdrop'")
        _ensure(config.task.type == "text_classification", "training.mode='layerdrop' currently supports only text_classification")
    if config.training.mode == "full_only":
        _ensure(config.baseline.kind in {"full", "shallow_independent", "naive_truncation"}, "training.mode='full_only' requires a baseline workflow")
    if config.baseline.kind == "shallow_independent":
        _ensure(config.baseline.target_depth is not None, "baseline.target_depth is required for shallow_independent")
        _ensure(config.baseline.target_depth in config.model.candidate_depths, "baseline.target_depth must be one of model.candidate_depths")
    if config.baseline.kind in {"pods", "full", "naive_truncation", "layerdrop"}:
        _ensure(config.baseline.target_depth is None, "baseline.target_depth is only valid for shallow_independent")

    if config.task.type == "image_classification":
        _ensure(config.task.metric_name == "accuracy", "image classification currently supports only metric_name='accuracy'")
    elif config.task.type == "text_classification":
        _ensure(config.task.dataset in {"sst2", "mrpc", "rte"}, "text_classification currently supports only sst2, mrpc, and rte")
        _ensure(config.task.text_fields, "text classification tasks require task.text_fields")
        _ensure(config.task.max_length > 0, "task.max_length must be positive")

    if config.model.family == "mlp":
        _ensure(config.task.type == "image_classification", "MLP models require image_classification tasks")
        _ensure(config.model.input_shape is not None, "MLP models require model.input_shape")
        _ensure(config.model.hidden_dims, "MLP models require model.hidden_dims")
        _ensure(max(config.model.candidate_depths) == len(config.model.hidden_dims), "MLP max depth must match hidden_dims")
    elif config.model.family == "cnn":
        _ensure(config.task.type == "image_classification", "CNN models require image_classification tasks")
        _ensure(config.model.input_shape is not None, "CNN models require model.input_shape")
        _ensure(config.model.channels, "CNN models require model.channels")
        _ensure(max(config.model.candidate_depths) == len(config.model.channels), "CNN max depth must match channels")
    elif config.model.family == "resnet":
        _ensure(config.task.type == "image_classification", "ResNet models require image_classification tasks")
        _ensure(config.model.variant in {"resnet56", "resnet110"}, "resnet variant is unsupported")
        _ensure(config.model.input_shape == (3, 32, 32), "CIFAR ResNet requires model.input_shape=[3, 32, 32]")
        _ensure(config.model.candidate_depths == (1, 2, 3), "ResNet candidate_depths must be stage-aligned as [1, 2, 3]")
    elif config.model.family in {"bert", "roberta"}:
        _ensure(config.task.type == "text_classification", "Transformer models require text_classification tasks")
        _ensure(config.model.pretrained_model_name is not None, "Transformer models require model.pretrained_model_name")
        _ensure(config.model.head_type in {"bert_cls", "roberta_cls"}, "Transformer models require a transformer classification head type")
        _ensure(config.model.norm_type == "none", "Transformer models require model.norm_type='none'")
        _ensure(config.model.candidate_depths == (2, 4, 6, 8, 10, 12), "Transformer candidate_depths must be [2, 4, 6, 8, 10, 12]")
        _ensure(config.model.dropout >= 0.0, "model.dropout must be non-negative")
    else:
        raise ValueError(f"Unsupported model family: {config.model.family}")
