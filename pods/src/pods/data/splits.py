from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class DatasetSplit:
    train: Dataset
    selection_val: Dataset


@dataclass(frozen=True)
class SplitIndices:
    train_indices: tuple[int, ...]
    selection_indices: tuple[int, ...]


def make_split_indices(dataset_size: int, selection_fraction: float, seed: int) -> SplitIndices:
    if not 0.0 < selection_fraction < 1.0:
        raise ValueError("selection_fraction must be between 0 and 1")
    selection_size = max(1, int(round(dataset_size * selection_fraction)))
    if selection_size >= dataset_size:
        raise ValueError("selection split must leave at least one training example")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_size, generator=generator).tolist()
    selection_indices = tuple(permutation[:selection_size])
    train_indices = tuple(permutation[selection_size:])
    return SplitIndices(train_indices=train_indices, selection_indices=selection_indices)


def split_train_selection(dataset: Dataset, selection_fraction: float, seed: int) -> DatasetSplit:
    indices = make_split_indices(len(dataset), selection_fraction, seed)
    return DatasetSplit(
        train=Subset(dataset, indices.train_indices),
        selection_val=Subset(dataset, indices.selection_indices),
    )
