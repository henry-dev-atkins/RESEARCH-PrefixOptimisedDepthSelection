from __future__ import annotations

from typing import Iterable

import copy
import torch
from torch import nn

from pods.models.heads import make_classification_head
from pods.models.interfaces import PrefixCapableModel


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, start_dim=1)


class ExtractedPrefixMLP(nn.Module):
    def __init__(self, flatten: nn.Module, layers: nn.ModuleList, head: nn.Module) -> None:
        super().__init__()
        self.flatten = flatten
        self.layers = layers
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


class PodsMLP(PrefixCapableModel):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        hidden_dims: Iterable[int],
        num_classes: int,
        candidate_depths: tuple[int, ...],
    ) -> None:
        super().__init__()
        hidden_dims = tuple(hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        if max(candidate_depths) != len(hidden_dims):
            raise ValueError("max candidate depth must match number of hidden layers")

        self.input_shape = input_shape
        self.candidate_depths = candidate_depths
        self.flatten = Flatten()

        in_features = 1
        for dim in input_shape:
            in_features *= dim

        layers: list[nn.Module] = []
        feature_dims: dict[int, int] = {}
        current = in_features
        for depth, hidden in enumerate(hidden_dims, start=1):
            layers.append(nn.Sequential(nn.Linear(current, hidden), nn.ReLU()))
            feature_dims[depth] = hidden
            current = hidden

        self.layers = nn.ModuleList(layers)
        self.heads = nn.ModuleDict(
            {str(depth): make_classification_head(feature_dims[depth], num_classes) for depth in candidate_depths}
        )

    def max_depth(self) -> int:
        return len(self.layers)

    def _validate_depth(self, depth: int) -> None:
        if depth not in self.candidate_depths:
            raise ValueError(f"Unsupported depth {depth}; expected one of {self.candidate_depths}")

    def _validate_depths(self, depths: tuple[int, ...]) -> None:
        if not depths:
            raise ValueError("depths must be non-empty")
        for depth in depths:
            self._validate_depth(depth)

    def forward_depths(self, x: torch.Tensor, depths: tuple[int, ...]) -> dict[int, torch.Tensor]:
        self._validate_depths(depths)
        requested = set(depths)
        outputs: dict[int, torch.Tensor] = {}
        hidden = self.flatten(x)
        for depth, layer in enumerate(self.layers, start=1):
            hidden = layer(hidden)
            if depth in requested:
                outputs[depth] = self.heads[str(depth)](hidden)
        return outputs

    def forward_features(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        self._validate_depth(depth)
        hidden = self.flatten(x)
        for layer in self.layers[:depth]:
            hidden = layer(hidden)
        return hidden

    def forward_prefix(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        hidden = self.forward_features(x, depth)
        return self.heads[str(depth)](hidden)

    def extract_prefix(self, depth: int) -> nn.Module:
        self._validate_depth(depth)
        return ExtractedPrefixMLP(
            flatten=copy.deepcopy(self.flatten),
            layers=nn.ModuleList(copy.deepcopy(list(self.layers[:depth]))),
            head=copy.deepcopy(self.heads[str(depth)]),
        )
