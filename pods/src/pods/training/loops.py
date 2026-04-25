from __future__ import annotations

from contextlib import nullcontext
import copy
import random
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.optim import lr_scheduler, Optimizer
from torch.utils.data import DataLoader

from pods.eval.quality import PrefixQualityResult, evaluate_all_prefixes
from pods.models.interfaces import PrefixCapableModel
from pods.training.depth_sampling import sample_training_depths
from pods.training.losses import aggregate_multidepth_loss, classification_loss


@dataclass
class EpochSummary:
    train_loss: float
    val_metrics_by_depth: dict[int, dict[str, float]]


def _move_batch(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, labels = batch
    return inputs.to(device), labels.to(device)


def _build_scheduler(optimizer: Optimizer, scheduler_name: str, epochs: int, scheduler_t_max: int | None, scheduler_milestones: tuple[int, ...], scheduler_gamma: float | None):
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_t_max or epochs)
    if scheduler_name == "multistep":
        return lr_scheduler.MultiStepLR(optimizer, milestones=list(scheduler_milestones), gamma=scheduler_gamma or 0.1)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def train_multidepth_model(
    model: PrefixCapableModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    epochs: int,
    mode: str,
    num_middle_depths: int,
    weight_power: float,
    device: torch.device,
    mixed_precision: str,
    label_smoothing: float = 0.0,
    scheduler_name: str = "none",
    scheduler_t_max: int | None = None,
    scheduler_milestones: tuple[int, ...] = (),
    scheduler_gamma: float | None = None,
    grad_clip: float | None = None,
    rng: random.Random | None = None,
    evaluation_fn: Callable[[PrefixCapableModel, DataLoader, tuple[int, ...], torch.device], dict[int, PrefixQualityResult]] = evaluate_all_prefixes,
) -> tuple[PrefixCapableModel, list[EpochSummary]]:
    if rng is None:
        rng = random.Random()

    model = model.to(device)
    history: list[EpochSummary] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("-inf")
    use_fp16_scaler = device.type == "cuda" and mixed_precision == "fp16"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_fp16_scaler)
    scheduler = _build_scheduler(optimizer, scheduler_name, epochs, scheduler_t_max, scheduler_milestones, scheduler_gamma)

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        seen_batches = 0

        for batch in train_loader:
            inputs, labels = _move_batch(batch, device)
            depths = sample_training_depths(model.candidate_depths, mode, num_middle_depths, rng)
            if device.type == "cuda" and mixed_precision in {"fp16", "bf16"}:
                autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16 if mixed_precision == "fp16" else torch.bfloat16)
            else:
                autocast_context = nullcontext()

            with autocast_context:
                logits_by_depth = model.forward_depths(inputs, depths)
                losses_by_depth = {
                    depth: classification_loss(logits, labels, label_smoothing=label_smoothing)
                    for depth, logits in logits_by_depth.items()
                }
                total_loss = aggregate_multidepth_loss(losses_by_depth, model.max_depth(), weight_power, normalise=True)

            optimizer.zero_grad(set_to_none=True)
            if use_fp16_scaler:
                scaler.scale(total_loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            running_loss += float(total_loss.detach().item())
            seen_batches += 1

        if scheduler is not None:
            scheduler.step()

        val_results = evaluation_fn(model, val_loader, model.candidate_depths, device)
        full_depth = max(model.candidate_depths)
        score = val_results[full_depth].accuracy
        if score > best_score:
            best_state = copy.deepcopy(model.state_dict())
            best_score = score

        history.append(
            EpochSummary(
                train_loss=running_loss / max(seen_batches, 1),
                val_metrics_by_depth={
                    depth: {"accuracy": result.accuracy, "loss": result.loss}
                    for depth, result in val_results.items()
                },
            )
        )

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state")

    model.load_state_dict(best_state)
    return model, history
