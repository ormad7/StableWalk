"""Tests for control-oriented stability metrics."""

from __future__ import annotations

import numpy as np

from stablewalk.analysis.controlled_motion import (
    _lr_symmetry_score,
    _repeatability_score,
    _smoothness_score,
)
from stablewalk.analysis.stability_metrics import continuous_low_better_score


def test_continuous_temporal_never_saturates_at_small_asymmetry():
    _, score_perfect = continuous_low_better_score(0.0)
    _, score_small = continuous_low_better_score(0.02)
    _, score_mid = continuous_low_better_score(0.15)
    assert score_perfect == 100.0
    assert score_small < 100.0
    assert score_mid < score_small


def test_high_rom_low_jerk_scores_well():
    # Large smooth sinusoid cycles
    t = np.linspace(0, 1, 101)
    cycles = [80 + 35 * np.sin(2 * np.pi * t + phase) for phase in (0, 0.05, -0.03)]
    rep = _repeatability_score(cycles)
    smooth = _smoothness_score(np.mean(np.vstack(cycles), axis=0))
    assert rep is not None and rep > 70
    assert smooth is not None and smooth > 70


def test_lr_shape_symmetry_independent_of_rom_magnitude():
    t = np.linspace(0, 1, 101)
    left = 40 + 20 * np.sin(2 * np.pi * t)
    right = 70 + 40 * np.sin(2 * np.pi * t)  # larger ROM, same shape
    score = _lr_symmetry_score(left, right)
    assert score is not None and score > 75
