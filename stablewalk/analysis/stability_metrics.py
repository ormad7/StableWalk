"""
Continuous stability metric transforms — avoid threshold saturation.

Movement magnitude is never mapped directly to instability; only asymmetry,
variability, smoothness, and repeatability penalties apply.
"""

from __future__ import annotations

import math


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def continuous_low_better_score(
    value: float,
    *,
    steepness: float = 6.0,
    reference: float = 0.25,
) -> tuple[float, float]:
    """
    Smooth decay from 100 at zero penalty — no flat 100 plateau except near-perfect.

    Returns (normalized_penalty in [0,1], score in [0,100]).
    Score reaches 100 only when value ≈ 0.
    """
    v = max(0.0, float(value))
    penalty = min(1.0, v / max(reference, 1e-9))
    score = clamp(100.0 * math.exp(-steepness * v))
    return penalty, score


def continuous_high_better_score(
    value: float,
    *,
    good: float,
    poor: float,
) -> float:
    """Smooth ramp for higher-is-better metrics (repeatability, symmetry)."""
    v = float(value)
    if v >= good:
        return clamp(85.0 + 15.0 * min(1.0, (v - good) / max(good * 0.05, 1e-6)))
    if v <= poor:
        return clamp(100.0 * v / max(poor, 1e-9))
    frac = (v - poor) / max(good - poor, 1e-9)
    return clamp(100.0 * (0.35 + 0.65 * frac ** 0.85))


def combine_control_scores(parts: list[float | None], *, weights: list[float] | None = None) -> float | None:
    valid = [(p, w) for p, w in zip(parts, weights or [1.0] * len(parts)) if p is not None]
    if not valid:
        return None
    denom = sum(w for _, w in valid)
    return sum(p * w for p, w in valid) / denom
