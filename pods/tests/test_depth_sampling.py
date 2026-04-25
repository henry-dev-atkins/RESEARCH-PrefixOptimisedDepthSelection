from __future__ import annotations

import random

import pytest

from pods.training.depth_sampling import sample_training_depths


def test_all_mode_returns_all_candidate_depths() -> None:
    depths = sample_training_depths((1, 2, 3, 4), "all", 1, random.Random(3))
    assert depths == (1, 2, 3, 4)


def test_full_only_mode_returns_only_deepest_depth() -> None:
    depths = sample_training_depths((1, 2, 3, 4), "full_only", 0, random.Random(3))
    assert depths == (4,)


def test_sandwich_mode_includes_anchors_and_sorted_unique_middle_depths() -> None:
    depths = sample_training_depths((1, 2, 3, 4, 5), "sandwich", 2, random.Random(0))
    assert depths[0] == 1
    assert depths[-1] == 5
    assert depths == tuple(sorted(set(depths)))


def test_sandwich_mode_rejects_impossible_middle_count() -> None:
    with pytest.raises(ValueError, match="num_middle exceeds"):
        sample_training_depths((1, 2, 3), "sandwich", 5, random.Random(0))
