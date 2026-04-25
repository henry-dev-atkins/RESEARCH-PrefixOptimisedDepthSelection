from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class PrefixCapableModel(nn.Module, ABC):
    candidate_depths: tuple[int, ...]

    @abstractmethod
    def forward_features(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_depths(self, x: torch.Tensor, depths: tuple[int, ...]) -> dict[int, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def forward_prefix(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def extract_prefix(self, depth: int) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def max_depth(self) -> int:
        raise NotImplementedError
