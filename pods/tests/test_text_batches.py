from __future__ import annotations

import torch

from pods.data.batches import move_batch_to_device


def test_move_batch_to_device_handles_text_dict_batches() -> None:
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "labels": torch.tensor([0, 1]),
    }

    inputs, labels = move_batch_to_device(batch, torch.device("cpu"))

    assert set(inputs) == {"input_ids", "attention_mask"}
    assert labels.tolist() == [0, 1]
