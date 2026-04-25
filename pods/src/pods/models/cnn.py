from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from pods.models.heads import make_classification_head
from pods.models.interfaces import PrefixCapableModel
from pods.models.prefix_extract import clone_prefix_module


def _make_norm(norm_type: str, channels: int, groupnorm_groups: int) -> nn.Module:
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "groupnorm":
        groups = min(groupnorm_groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def _make_conv_block(in_channels: int, out_channels: int, norm_type: str, groupnorm_groups: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        _make_norm(norm_type, out_channels, groupnorm_groups),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )


class GlobalAveragePool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(-1, -2))


class PodsCNN(PrefixCapableModel):
    def __init__(
        self,
        input_channels: int,
        channels: Iterable[int],
        num_classes: int,
        candidate_depths: tuple[int, ...],
        norm_type: str = "groupnorm",
        groupnorm_groups: int = 8,
    ) -> None:
        super().__init__()
        channels = tuple(channels)
        if not channels:
            raise ValueError("channels must be non-empty")
        if max(candidate_depths) != len(channels):
            raise ValueError("max candidate depth must match number of CNN blocks")

        self.candidate_depths = candidate_depths
        self.stem = nn.Identity()
        self.blocks = nn.ModuleList()
        in_channels = input_channels
        for out_channels in channels:
            self.blocks.append(_make_conv_block(in_channels, out_channels, norm_type, groupnorm_groups))
            in_channels = out_channels
        self.pool = GlobalAveragePool()
        self.heads = nn.ModuleDict(
            {str(depth): make_classification_head(channels[depth - 1], num_classes) for depth in candidate_depths}
        )

    def max_depth(self) -> int:
        return len(self.blocks)

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
        hidden = self.stem(x)
        for depth, block in enumerate(self.blocks, start=1):
            hidden = block(hidden)
            if depth in requested:
                outputs[depth] = self.heads[str(depth)](self.pool(hidden))
        return outputs

    def forward_features(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        self._validate_depth(depth)
        hidden = self.stem(x)
        for block in self.blocks[:depth]:
            hidden = block(hidden)
        return self.pool(hidden)

    def forward_prefix(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        hidden = self.forward_features(x, depth)
        return self.heads[str(depth)](hidden)

    def extract_prefix(self, depth: int) -> nn.Module:
        self._validate_depth(depth)
        return clone_prefix_module(self.stem, self.blocks, self.pool, self.heads[str(depth)], depth)
