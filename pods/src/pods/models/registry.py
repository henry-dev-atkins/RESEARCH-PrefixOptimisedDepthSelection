from __future__ import annotations

from pods.config.schema import PodsConfig
from pods.models.bert import PodsBert
from pods.models.cnn import PodsCNN
from pods.models.interfaces import PrefixCapableModel
from pods.models.mlp import PodsMLP
from pods.models.roberta import PodsRoberta
from pods.models.resnet import PodsResNet


def build_model(config: PodsConfig, target_depth: int | None = None) -> PrefixCapableModel:
    candidate_depths = config.model.candidate_depths if target_depth is None else (target_depth,)

    if config.model.family == "mlp":
        assert config.model.input_shape is not None
        hidden_dims = config.model.hidden_dims if target_depth is None else config.model.hidden_dims[:target_depth]
        return PodsMLP(
            input_shape=config.model.input_shape,
            hidden_dims=hidden_dims,
            num_classes=config.task.num_classes,
            candidate_depths=candidate_depths,
        )
    if config.model.family == "cnn":
        assert config.model.input_shape is not None
        channels = config.model.channels if target_depth is None else config.model.channels[:target_depth]
        return PodsCNN(
            input_channels=config.model.input_shape[0],
            channels=channels,
            num_classes=config.task.num_classes,
            candidate_depths=candidate_depths,
            norm_type=config.model.norm_type,
            groupnorm_groups=config.model.groupnorm_groups,
        )
    if config.model.family == "resnet":
        assert config.model.variant is not None
        return PodsResNet(
            variant=config.model.variant,
            num_classes=config.task.num_classes,
            candidate_depths=candidate_depths,
            norm_type=config.model.norm_type,
            groupnorm_groups=config.model.groupnorm_groups,
        )
    if config.model.family == "bert":
        assert config.model.pretrained_model_name is not None
        return PodsBert(
            pretrained_model_name=config.model.pretrained_model_name,
            num_classes=config.task.num_classes,
            candidate_depths=candidate_depths,
            dropout=config.model.dropout,
        )
    if config.model.family == "roberta":
        assert config.model.pretrained_model_name is not None
        return PodsRoberta(
            pretrained_model_name=config.model.pretrained_model_name,
            num_classes=config.task.num_classes,
            candidate_depths=candidate_depths,
            dropout=config.model.dropout,
        )
    raise ValueError(f"Unsupported model family: {config.model.family}")
