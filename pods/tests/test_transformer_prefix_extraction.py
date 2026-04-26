from __future__ import annotations

import torch

from pods.models.bert import PodsBert
from pods.models.roberta import PodsRoberta


def _sample_bert_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[101, 2023, 2003, 1037, 3231, 102], [101, 2054, 2003, 2009, 102, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]]),
        "token_type_ids": torch.zeros((2, 6), dtype=torch.long),
    }


def _sample_roberta_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[0, 713, 16, 10, 1296, 2], [0, 134, 16, 24, 2, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]]),
    }


def test_bert_extract_prefix_matches_forward_prefix_logits() -> None:
    torch.manual_seed(0)
    model = PodsBert(
        pretrained_model_name="bert-base-uncased",
        num_classes=2,
        candidate_depths=(2, 4, 6, 8, 10, 12),
        dropout=0.1,
    )
    model.eval()
    batch = _sample_bert_batch()
    extracted = model.extract_prefix(4)
    extracted.eval()
    expected = model.forward_prefix(batch, 4)
    actual = extracted(**batch)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_roberta_extract_prefix_matches_forward_prefix_logits() -> None:
    torch.manual_seed(0)
    model = PodsRoberta(
        pretrained_model_name="roberta-base",
        num_classes=2,
        candidate_depths=(2, 4, 6, 8, 10, 12),
        dropout=0.1,
    )
    model.eval()
    batch = _sample_roberta_batch()
    extracted = model.extract_prefix(4)
    extracted.eval()
    expected = model.forward_prefix(batch, 4)
    actual = extracted(**batch)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
