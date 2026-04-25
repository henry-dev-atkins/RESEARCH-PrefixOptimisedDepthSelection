from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from torchvision import datasets

from pods.config.schema import PodsConfig
from pods.data.transforms import build_image_transforms


def build_image_dataset(name: str, train: bool, root: str | Path = "temp/data", use_eval_transform: bool = False) -> Dataset:
    root = Path(root)
    transform = build_image_transforms(name, train=(False if use_eval_transform else train))

    if name == "mnist":
        return datasets.MNIST(root=root, train=train, download=True, transform=transform)
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    if name == "cifar100":
        return datasets.CIFAR100(root=root, train=train, download=True, transform=transform)

    raise ValueError(f"Unsupported image dataset: {name}")


def build_text_datasets(config: PodsConfig, root: str | Path = "temp/data") -> dict[str, Dataset]:
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Text tasks require the 'datasets' and 'transformers' packages to be installed") from exc

    root = Path(root)
    dataset_name = config.task.dataset
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name)
    text_fields = config.task.text_fields

    if dataset_name not in {"sst2", "mrpc", "rte"}:
        raise ValueError(f"Unsupported text dataset: {dataset_name}")

    raw = load_dataset("glue", dataset_name, cache_dir=str(root / "huggingface"))

    def _tokenise(batch: dict[str, Any]) -> dict[str, Any]:
        texts = [batch[text_fields[0]]]
        if len(text_fields) > 1:
            texts.append(batch[text_fields[1]])
        tokenised = tokenizer(
            *texts,
            padding=False,
            truncation=True,
            max_length=config.task.max_length,
        )
        tokenised["labels"] = batch["label"]
        return tokenised

    keep_columns = set(raw["train"].column_names) | {"labels"}
    encoded = raw.map(_tokenise, batched=True)
    for split_name, split_dataset in encoded.items():
        remove = [column for column in split_dataset.column_names if column not in {"input_ids", "attention_mask", "token_type_ids", "labels"}]
        encoded[split_name] = split_dataset.remove_columns(remove)
        encoded[split_name].set_format(type="torch")
    _ = keep_columns
    return {"train": encoded["train"], "validation": encoded["validation"], "test": encoded["validation"]}
