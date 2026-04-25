from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import nn


class PrefixCapableModel(nn.Module, ABC):
    candidate_depths: tuple[int, ...]

    @abstractmethod
    def forward_features(self, x: Any, depth: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def forward_depths(self, x: Any, depths: tuple[int, ...]) -> dict[int, Any]:
        raise NotImplementedError

    @abstractmethod
    def forward_prefix(self, x: Any, depth: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def extract_prefix(self, depth: int) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def max_depth(self) -> int:
        raise NotImplementedError
