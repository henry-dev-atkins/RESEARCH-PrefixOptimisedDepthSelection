from __future__ import annotations

import random
from typing import Literal


def sample_training_depths(
    candidate_depths: tuple[int, ...],
    mode: Literal["all", "sandwich", "full_only", "layerdrop"],
    num_middle: int,
    rng: random.Random,
) -> tuple[int, ...]:
    if not candidate_depths:
        raise ValueError("candidate_depths must be non-empty")
    if tuple(sorted(set(candidate_depths))) != candidate_depths:
        raise ValueError("candidate_depths must be sorted and unique")

    if mode == "all":
        return candidate_depths
    if mode == "full_only":
        return (candidate_depths[-1],)
    if mode == "layerdrop":
        return (rng.choice(candidate_depths),)
    if mode != "sandwich":
        raise ValueError(f"Unsupported training mode: {mode}")
    if len(candidate_depths) < 2:
        raise ValueError("sandwich mode requires at least two candidate depths")
    if num_middle < 0:
        raise ValueError("num_middle must be non-negative")

    middle_candidates = list(candidate_depths[1:-1])
    if num_middle > len(middle_candidates):
        raise ValueError("num_middle exceeds available middle depths")

    sampled = rng.sample(middle_candidates, k=num_middle) if num_middle else []
    return tuple((candidate_depths[0], *sorted(sampled), candidate_depths[-1]))
