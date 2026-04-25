from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def move_batch_to_device(batch: Any, device: torch.device) -> tuple[Any, torch.Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        inputs, labels = batch
        return inputs.to(device), labels.to(device)

    if isinstance(batch, Mapping):
        if "labels" not in batch:
            raise ValueError("Text batches must include a 'labels' field")
        labels = batch["labels"].to(device)
        inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
        return inputs, labels

    raise TypeError(f"Unsupported batch type: {type(batch)!r}")
