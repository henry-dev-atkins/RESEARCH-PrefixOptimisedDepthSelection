from __future__ import annotations

import copy

import torch
from torch import nn


class ExtractedPrefixClassifier(nn.Module):
    def __init__(self, stem: nn.Module, blocks: nn.ModuleList, pool: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.stem = stem
        self.blocks = blocks
        self.pool = pool
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        return self.head(x)


def clone_prefix_module(stem: nn.Module, blocks: nn.ModuleList, pool: nn.Module, head: nn.Module, depth: int) -> nn.Module:
    if depth <= 0 or depth > len(blocks):
        raise ValueError(f"Unsupported prefix depth: {depth}")
    return ExtractedPrefixClassifier(
        stem=copy.deepcopy(stem),
        blocks=nn.ModuleList(copy.deepcopy(list(blocks[:depth]))),
        pool=copy.deepcopy(pool),
        head=copy.deepcopy(head),
    )
