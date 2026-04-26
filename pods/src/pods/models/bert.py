from __future__ import annotations

from pods.models.transformer_prefix import PodsTransformerEncoder


class PodsBert(PodsTransformerEncoder):
    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int,
        candidate_depths: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__(
            pretrained_model_name=pretrained_model_name,
            num_classes=num_classes,
            candidate_depths=candidate_depths,
            dropout=dropout,
            expected_model_type="bert",
        )
