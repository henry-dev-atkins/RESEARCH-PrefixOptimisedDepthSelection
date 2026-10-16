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

from pods.data.batches import move_batch_to_device
from pods.eval.quality import PrefixQualityResult, evaluate_all_prefixes
from pods.models.interfaces import PrefixCapableModel
from pods.training.depth_sampling import sample_training_depths
from pods.training.losses import aggregate_multidepth_loss, supervised_loss


@dataclass
class EpochSummary:
    train_loss: float
    val_metrics_by_depth: dict[int, dict[str, float]]


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
    gradient_accumulation_steps: int = 1,
    rng: random.Random | None = None,
    progressive_middle_depths: tuple[int, ...] = (),
    evaluation_fn: Callable[[PrefixCapableModel, DataLoader, tuple[int, ...], torch.device], dict[int, PrefixQualityResult]] = evaluate_all_prefixes,
    task_type: str = "image_classification",
    metric_name: str = "accuracy",
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

    for epoch_index in range(epochs):
        model.train()
        running_loss = 0.0
        seen_batches = 0
        optimizer.zero_grad(set_to_none=True)

        current_num_middle = num_middle_depths
        if mode == "progressive_sandwich":
            if not progressive_middle_depths:
                raise ValueError("progressive_sandwich requires progressive_middle_depths")
            stage_index = min((epoch_index * len(progressive_middle_depths)) // max(epochs, 1), len(progressive_middle_depths) - 1)
            current_num_middle = progressive_middle_depths[stage_index]

        for batch_index, batch in enumerate(train_loader, start=1):
            inputs, labels = move_batch_to_device(batch, device)
            depths = sample_training_depths(model.candidate_depths, mode, current_num_middle, rng)
            if device.type == "cuda" and mixed_precision in {"fp16", "bf16"}:
                autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16 if mixed_precision == "fp16" else torch.bfloat16)
            else:
                autocast_context = nullcontext()

            with autocast_context:
                logits_by_depth = model.forward_depths(inputs, depths)
                losses_by_depth = {
                    depth: supervised_loss(logits, labels, task_type=task_type, label_smoothing=label_smoothing)
                    for depth, logits in logits_by_depth.items()
                }
                total_loss = aggregate_multidepth_loss(losses_by_depth, model.max_depth(), weight_power, normalise=True)
                step_loss = total_loss / gradient_accumulation_steps

            if use_fp16_scaler:
                scaler.scale(step_loss).backward()
            else:
                step_loss.backward()

            should_step = (batch_index % gradient_accumulation_steps == 0) or (batch_index == len(train_loader))
            if should_step:
                if grad_clip is not None:
                    if use_fp16_scaler:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if use_fp16_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(total_loss.detach().item())
            seen_batches += 1

        if scheduler is not None:
            scheduler.step()

        val_results = evaluation_fn(model, val_loader, model.candidate_depths, device, metric_name=metric_name, task_type=task_type)
        full_depth = max(model.candidate_depths)
        score = val_results[full_depth].metric_value
        if score > best_score:
            best_state = copy.deepcopy(model.state_dict())
            best_score = score

        history.append(
            EpochSummary(
                train_loss=running_loss / max(seen_batches, 1),
                val_metrics_by_depth={
                    depth: {
                        "accuracy": result.accuracy,
                        "loss": result.loss,
                        result.metric_name: result.metric_value,
                    }
                    for depth, result in val_results.items()
                },
            )
        )

    if best_state is None:
        raise RuntimeError("Training produced no best model state")

    model.load_state_dict(best_state)
    return model, history
