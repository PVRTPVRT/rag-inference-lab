"""Pure helpers shared by inference backends and benchmark aggregation."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Iterable


def nanoseconds_to_milliseconds(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / 1_000_000, 3)


def tokens_per_second(token_count: int | None, duration_ns: int | None) -> float | None:
    """Return token throughput from an authoritative count and duration."""
    if token_count is None or duration_ns is None or token_count < 0 or duration_ns <= 0:
        return None
    return round(token_count / (duration_ns / 1_000_000_000), 3)


def client_tokens_per_second(
    token_count: int | None, e2e_ms: float, ttft_ms: float | None
) -> float | None:
    """Compute client-observed decode throughput after the first token."""
    if token_count is None or token_count < 0 or ttft_ms is None:
        return None
    decode_seconds = (e2e_ms - ttft_ms) / 1_000
    if decode_seconds <= 0:
        return None
    return round(token_count / decode_seconds, 3)


def percentile(values: Iterable[float], probability: float) -> float | None:
    """Linearly interpolated percentile without a NumPy dependency."""
    ordered = sorted(float(value) for value in values if value is not None)
    if not ordered:
        return None
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def summarize(values: Iterable[float | None]) -> dict[str, float | int | None]:
    cleaned = [float(value) for value in values if value is not None]
    return {
        "n": len(cleaned),
        "mean": round(fmean(cleaned), 3) if cleaned else None,
        "p50": percentile(cleaned, 0.50),
        "p95": percentile(cleaned, 0.95),
        "min": round(min(cleaned), 3) if cleaned else None,
        "max": round(max(cleaned), 3) if cleaned else None,
    }
