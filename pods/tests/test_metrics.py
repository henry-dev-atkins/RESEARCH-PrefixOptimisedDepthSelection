from __future__ import annotations

from pods.eval.metrics import accuracy_score, bootstrap_confidence_interval, matthews_correlation, pearson_correlation, spearman_correlation


def test_matthews_correlation_matches_perfect_binary_prediction() -> None:
    labels = [1, 1, 0, 0]
    predictions = [1, 1, 0, 0]
    assert matthews_correlation(labels, predictions) == 1.0


def test_accuracy_score_computes_fraction_correct() -> None:
    labels = [1, 0, 1, 0]
    predictions = [1, 0, 0, 0]
    assert accuracy_score(labels, predictions) == 0.75


def test_bootstrap_confidence_interval_returns_bounds() -> None:
    labels = [1, 0, 1, 0, 1, 0]
    predictions = [1, 0, 1, 0, 0, 0]
    low, high = bootstrap_confidence_interval("accuracy", labels, predictions, num_samples=32, confidence_level=0.95, seed=7)
    assert low is not None
    assert high is not None
    assert low <= high


def test_pearson_correlation_matches_positive_linear_relationship() -> None:
    labels = [0.0, 1.0, 2.0, 3.0]
    predictions = [0.0, 2.0, 4.0, 6.0]
    assert pearson_correlation(labels, predictions) == 1.0


def test_spearman_correlation_matches_rank_order_relationship() -> None:
    labels = [10.0, 20.0, 30.0, 40.0]
    predictions = [1.0, 2.0, 3.0, 4.0]
    assert spearman_correlation(labels, predictions) == 1.0
