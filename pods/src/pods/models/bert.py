from __future__ import annotations

import copy

import torch
from torch import nn

from pods.models.interfaces import PrefixCapableModel


class BertPrefixClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        cls = hidden[:, 0]
        return self.out(self.dropout(self.norm(cls)))


class ExtractedPrefixBert(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return self.head(outputs.last_hidden_state)


class PodsBert(PrefixCapableModel):
    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int,
        candidate_depths: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoConfig, BertModel
        except ImportError as exc:
            raise ImportError("BERT models require the 'transformers' package to be installed") from exc

        config = AutoConfig.from_pretrained(pretrained_model_name)
        if getattr(config, "model_type", None) != "bert":
            raise ValueError(f"Unsupported transformer model for Session 2: {config.model_type!r}")
        self.backbone = BertModel.from_pretrained(pretrained_model_name, add_pooling_layer=False)
        self.backbone_config = config
        self.pretrained_model_name = pretrained_model_name
        self.candidate_depths = candidate_depths
        self.hidden_size = config.hidden_size
        self.dropout = dropout
        self.heads = nn.ModuleDict(
            {
                str(depth): BertPrefixClassificationHead(self.hidden_size, num_classes, dropout)
                for depth in candidate_depths
            }
        )

    def max_depth(self) -> int:
        return self.backbone.config.num_hidden_layers

    def _validate_depth(self, depth: int) -> None:
        if depth not in self.candidate_depths:
            raise ValueError(f"Unsupported depth {depth}; expected one of {self.candidate_depths}")

    def _validate_depths(self, depths: tuple[int, ...]) -> None:
        if not depths:
            raise ValueError("depths must be non-empty")
        for depth in depths:
            self._validate_depth(depth)

    def _forward_backbone(self, x: dict[str, torch.Tensor]):
        return self.backbone(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
            token_type_ids=x.get("token_type_ids"),
            output_hidden_states=True,
            return_dict=True,
        )

    def forward_features(self, x: dict[str, torch.Tensor], depth: int) -> torch.Tensor:
        self._validate_depth(depth)
        outputs = self._forward_backbone(x)
        return outputs.hidden_states[depth]

    def forward_depths(self, x: dict[str, torch.Tensor], depths: tuple[int, ...]) -> dict[int, torch.Tensor]:
        self._validate_depths(depths)
        outputs = self._forward_backbone(x)
        return {depth: self.heads[str(depth)](outputs.hidden_states[depth]) for depth in depths}

    def forward_prefix(self, x: dict[str, torch.Tensor], depth: int) -> torch.Tensor:
        hidden = self.forward_features(x, depth)
        return self.heads[str(depth)](hidden)

    def extract_prefix(self, depth: int) -> nn.Module:
        self._validate_depth(depth)
        try:
            from transformers import BertConfig, BertModel
        except ImportError as exc:
            raise ImportError("BERT models require the 'transformers' package to be installed") from exc

        prefix_config = BertConfig.from_dict(self.backbone_config.to_dict())
        prefix_config.num_hidden_layers = depth
        prefix_backbone = BertModel(prefix_config, add_pooling_layer=False)
        prefix_backbone.embeddings.load_state_dict(copy.deepcopy(self.backbone.embeddings.state_dict()))
        for layer_index in range(depth):
            prefix_backbone.encoder.layer[layer_index].load_state_dict(copy.deepcopy(self.backbone.encoder.layer[layer_index].state_dict()))
        return ExtractedPrefixBert(
            backbone=prefix_backbone,
            head=copy.deepcopy(self.heads[str(depth)]),
        )
