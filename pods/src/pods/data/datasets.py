from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets

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

    raise ValueError(f"Unsupported dataset: {name}")
