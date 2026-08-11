"""Deterministic paired-bootstrap comparisons for per-query retrieval metrics."""

from __future__ import annotations

import numpy as np


def paired_bootstrap(
    baseline: list[float],
    candidate: list[float],
    *,
    samples: int = 20_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Estimate a paired mean-difference interval by resampling queries.

    The two-sided tail probability is the doubled smaller bootstrap mass on
    either side of zero with an add-one correction. It is an exploratory
    bootstrap significance estimate, not a multiplicity-corrected test.
    """
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must have equal length")
    if not baseline:
        raise ValueError("at least one paired observation is required")
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must satisfy 0 < confidence < 1")

    baseline_array = np.asarray(baseline, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if not np.isfinite(baseline_array).all() or not np.isfinite(candidate_array).all():
        raise ValueError("observations must be finite")

    differences = candidate_array - baseline_array
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    alpha = 1 - confidence
    lower, upper = np.quantile(
        bootstrap_means, [alpha / 2, 1 - alpha / 2]
    )
    non_positive = (np.count_nonzero(bootstrap_means <= 0) + 1) / (samples + 1)
    non_negative = (np.count_nonzero(bootstrap_means >= 0) + 1) / (samples + 1)
    two_sided_probability = min(1.0, 2 * min(non_positive, non_negative))
    tolerance = 1e-12

    return {
        "queries": len(differences),
        "baseline_mean": round(float(baseline_array.mean()), 6),
        "candidate_mean": round(float(candidate_array.mean()), 6),
        "mean_difference": round(float(differences.mean()), 6),
        "ci_lower": round(float(lower), 6),
        "ci_upper": round(float(upper), 6),
        "two_sided_bootstrap_p": round(float(two_sided_probability), 6),
        "probability_candidate_better": round(
            float(np.mean(bootstrap_means > 0)), 6
        ),
        "query_wins": int(np.count_nonzero(differences > tolerance)),
        "query_ties": int(np.count_nonzero(np.abs(differences) <= tolerance)),
        "query_losses": int(np.count_nonzero(differences < -tolerance)),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Apply Holm's step-down family-wise error correction."""
    if not p_values:
        return {}
    for name, value in p_values.items():
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"p-value for {name} must be finite and within [0, 1]")

    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted = {}
    running_max = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (count - rank) * value))
        adjusted[name] = round(running_max, 6)
    return adjusted
