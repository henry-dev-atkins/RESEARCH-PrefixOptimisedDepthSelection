from __future__ import annotations

import copy

import torch
from torch import nn

from pods.models.heads import make_classification_head
from pods.models.interfaces import PrefixCapableModel


def _make_norm(norm_type: str, channels: int, groupnorm_groups: int) -> nn.Module:
    if norm_type == "groupnorm":
        groups = min(groupnorm_groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm_type == "batchnorm":
        return nn.BatchNorm2d(channels)
    raise ValueError(f"Unsupported norm_type for ResNet: {norm_type}")


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, norm_type: str, groupnorm_groups: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = _make_norm(norm_type, out_channels, groupnorm_groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = _make_norm(norm_type, out_channels, groupnorm_groups)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _make_norm(norm_type, out_channels, groupnorm_groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class GlobalAveragePool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(-1, -2))


class ExtractedPrefixResNet(nn.Module):
    def __init__(self, stem: nn.Module, stages: nn.ModuleList, pool: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.stem = stem
        self.stages = stages
        self.pool = pool
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x)
        return self.head(x)


class PodsResNet(PrefixCapableModel):
    def __init__(
        self,
        variant: str,
        num_classes: int,
        candidate_depths: tuple[int, ...],
        norm_type: str,
        groupnorm_groups: int = 8,
    ) -> None:
        super().__init__()
        blocks_per_stage = {"resnet56": 9, "resnet110": 18}[variant]
        max_stage = max(candidate_depths)
        if candidate_depths != tuple(range(1, max_stage + 1)) and candidate_depths != (max_stage,):
            raise ValueError("ResNet candidate depths must be full stage-aligned prefixes or a single shallow target depth")
        self.variant = variant
        self.candidate_depths = candidate_depths
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm(norm_type, 16, groupnorm_groups),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList(
            [
                self._make_stage(16, 16, blocks_per_stage, stride=1, norm_type=norm_type, groupnorm_groups=groupnorm_groups),
                self._make_stage(16, 32, blocks_per_stage, stride=2, norm_type=norm_type, groupnorm_groups=groupnorm_groups),
                self._make_stage(32, 64, blocks_per_stage, stride=2, norm_type=norm_type, groupnorm_groups=groupnorm_groups),
            ][: max_stage]
        )
        self.pool = GlobalAveragePool()
        stage_channels = {1: 16, 2: 32, 3: 64}
        self.heads = nn.ModuleDict({str(depth): make_classification_head(stage_channels[depth], num_classes) for depth in candidate_depths})

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        norm_type: str,
        groupnorm_groups: int,
    ) -> nn.Sequential:
        layers = [ResNetBasicBlock(in_channels, out_channels, stride, norm_type, groupnorm_groups)]
        for _ in range(1, blocks):
            layers.append(ResNetBasicBlock(out_channels, out_channels, 1, norm_type, groupnorm_groups))
        return nn.Sequential(*layers)

    def max_depth(self) -> int:
        return len(self.stages)

    def _validate_depth(self, depth: int) -> None:
        if depth not in self.candidate_depths:
            raise ValueError(f"Unsupported depth {depth}; expected one of {self.candidate_depths}")

    def _validate_depths(self, depths: tuple[int, ...]) -> None:
        if not depths:
            raise ValueError("depths must be non-empty")
        for depth in depths:
            self._validate_depth(depth)

    def _run_to_depth(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        hidden = self.stem(x)
        for stage in self.stages[:depth]:
            hidden = stage(hidden)
        return hidden

    def forward_features(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        self._validate_depth(depth)
        return self.pool(self._run_to_depth(x, depth))

    def forward_depths(self, x: torch.Tensor, depths: tuple[int, ...]) -> dict[int, torch.Tensor]:
        self._validate_depths(depths)
        requested = set(depths)
        outputs: dict[int, torch.Tensor] = {}
        hidden = self.stem(x)
        for depth, stage in enumerate(self.stages, start=1):
            hidden = stage(hidden)
            if depth in requested:
                outputs[depth] = self.heads[str(depth)](self.pool(hidden))
        return outputs

    def forward_prefix(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        hidden = self.forward_features(x, depth)
        return self.heads[str(depth)](hidden)

    def extract_prefix(self, depth: int) -> nn.Module:
        self._validate_depth(depth)
        return ExtractedPrefixResNet(
            stem=copy.deepcopy(self.stem),
            stages=nn.ModuleList(copy.deepcopy(list(self.stages[:depth]))),
            pool=copy.deepcopy(self.pool),
            head=copy.deepcopy(self.heads[str(depth)]),
        )
