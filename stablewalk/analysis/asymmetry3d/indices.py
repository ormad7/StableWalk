"""Documented left–right symmetry / asymmetry indices.

These four families are **not interchangeable**:

1. Signed Robinson SI (percent of the bilateral mean)::

       SI = 200 * (L − R) / (L + R)

   Positive SI means the **left** value is larger. Zero is perfect symmetry.
   This is a *relative percent*, not a difference in the native unit and not
   a difference in percentage points.

2. Absolute symmetry index (unsigned percent)::

       ASI = 200 * |L − R| / (|L| + |R|)

   Zero is perfect symmetry. ASI is always ≥ 0.

3. Normalized similarity (unitless, [0, 1])::

       SI_norm = 2 * min(L, R) / (L + R)

   Defined only for non-negative quantities. **1.0 is perfect symmetry.**
   This is the helper already used by ``gait_feature_analysis.symmetry_index``.

4. Composite scores (e.g. overall symmetry 0–100, per-frame moment score)
   mix several features with weights. They are **not** a knee-only SI_norm.

Peak timing
-----------
Left and right peak locations are reported in **% gait cycle** (0–100).
Their difference is **percentage points**, using the shortest circular path
across the 0% / 100% boundary::

    Δτ = ((τ_L − τ_R + 50) mod 100) − 50    ∈ [−50, +50] percentage points

Robinson SI is **not** applied to peak phase: 40% vs 50% would yield
SI = −22% (a relative percent of the mean phase), which is a different
quantity from the 10 percentage-point lag.

Alignment
---------
Left and right gait cycles are ipsilateral heel-strike → heel-strike,
resampled onto a common 0–100% grid **before** any pointwise comparison.
Unaligned frames are never subtracted.

Insufficient data
-----------------
When a side is missing, cycles are unaligned, or L + R ≈ 0, every index
is ``None``. Missing comparisons are not scored as 0% / 1.0.

None of these indices is a clinical diagnosis.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from stablewalk.analysis.gait_feature_analysis import (
    SYMMETRY_EPS,
    symmetry_index as normalized_symmetry_index,
)

INDEX_EPS = SYMMETRY_EPS

ALIGNMENT_EVENT = "ipsilateral_heel_strike"
ALIGNMENT_EVENT_LABEL = (
    "Ipsilateral heel-strike to heel-strike, resampled to 0–100% gait cycle"
)

ROBINSON_SI_DEFINITION = (
    "Asymmetry index AI / SI = 200 * (L - R) / (L + R)  "
    "(signed relative percent of the bilateral mean; 0 = symmetry; "
    "not percentage points)"
)
ASI_DEFINITION = (
    "Absolute symmetry index ASI = 200 * |L - R| / (|L| + |R|)  "
    "(unsigned relative percent; 0 = symmetry; not SI_norm)"
)
SI_NORM_DEFINITION = (
    "Normalized similarity SI_norm = 2 * min(L, R) / (L + R)  "
    "([0, 1]; 1 = symmetry; non-negative only; not a percent)"
)
TIMING_DEFINITION = (
    "Peak times τ_L, τ_R in % gait cycle. "
    "Δτ = wrap(τ_L − τ_R) ∈ [−50, +50] percentage points "
    "(shortest path across 0%/100%). Not Robinson SI."
)
OVERALL_VS_KNEE_DEFINITION = (
    "Overall symmetry (Results / Biomechanics) is 100 × a weighted mean of "
    "step, stride, stance, swing, cadence consistency, and knee/hip/ankle "
    "ROM SI_norm values. Knee-specific SI_norm uses only that joint's "
    "phase-aligned ROM. A symmetric knee can still sit inside an asymmetric "
    "composite, and the reverse is also true."
)
COMPOSITE_DEFINITION = (
    "Composite scores mix several features and/or a 0–100 moment scale. "
    "They are not SI, ASI, or SI_norm."
)


def asymmetry_index(
    left: float | None, right: float | None, *, eps: float = INDEX_EPS
) -> float | None:
    """Signed asymmetry index (AI) — same formula as Robinson SI."""
    return robinson_si(left, right, eps=eps)


def robinson_si(left: float | None, right: float | None, *, eps: float = INDEX_EPS) -> float | None:
    """Signed Robinson SI in relative percent. Positive ⇒ left larger."""
    if left is None or right is None:
        return None
    try:
        l_v = float(left)
        r_v = float(right)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(l_v) or not math.isfinite(r_v):
        return None
    denom = l_v + r_v
    if abs(denom) < eps:
        return None
    return float(200.0 * (l_v - r_v) / denom)


def absolute_symmetry_index(
    left: float | None,
    right: float | None,
    *,
    eps: float = INDEX_EPS,
) -> float | None:
    """Unsigned ASI in relative percent. Zero ⇒ perfect symmetry."""
    if left is None or right is None:
        return None
    try:
        l_v = abs(float(left))
        r_v = abs(float(right))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(l_v) or not math.isfinite(r_v):
        return None
    denom = l_v + r_v
    if denom < eps:
        return None
    return float(200.0 * abs(l_v - r_v) / denom)


def circular_phase_difference_pp(
    left_pct: float | None,
    right_pct: float | None,
) -> float | None:
    """Signed circular peak-timing difference in percentage points.

    ``Δτ = ((τ_L − τ_R + 50) mod 100) − 50`` so 98% vs 2% is −4 pp, not 96.
    Positive means the left peak is later in the gait cycle (shortest path).
    """
    if left_pct is None or right_pct is None:
        return None
    try:
        left = float(left_pct)
        right = float(right_pct)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(left) or not math.isfinite(right):
        return None
    delta = (left - right + 50.0) % 100.0 - 50.0
    if abs(delta) < 1e-12:
        return 0.0
    return float(delta)


def symmetry_bundle(
    left: float | None,
    right: float | None,
    *,
    allow_signed_norm: bool = False,
    publish: bool = True,
) -> dict[str, float | None]:
    """Return SI, ASI, and SI_norm for one paired scalar.

    When ``publish`` is false (insufficient / unaligned comparison) every
    index is ``None`` so the UI cannot show a reassuring 0 or 1.0.
    """
    left_v = None if left is None else float(left)
    right_v = None if right is None else float(right)
    if not publish:
        return {
            "left": left_v,
            "right": right_v,
            "si_percent": None,
            "asi_percent": None,
            "si_norm": None,
        }
    si_norm = None
    if allow_signed_norm:
        si_norm = normalized_symmetry_index(
            None if left_v is None else abs(left_v),
            None if right_v is None else abs(right_v),
        )
    else:
        si_norm = normalized_symmetry_index(left, right)
    return {
        "left": left_v,
        "right": right_v,
        "si_percent": robinson_si(left, right),
        "asi_percent": absolute_symmetry_index(left, right),
        "si_norm": None if si_norm is None else float(si_norm),
    }


def aligned_rmse(left: np.ndarray, right: np.ndarray) -> float | None:
    """RMSE of two already-aligned 1-D series. NaNs are skipped pairwise."""
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    if a.size != b.size or a.size == 0:
        return None
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 8:
        return None
    diff = a[ok] - b[ok]
    return float(np.sqrt(np.mean(diff * diff)))


def aligned_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    if a.size != b.size or a.size == 0:
        return None
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 8:
        return None
    aa = a[ok]
    bb = b[ok]
    if float(np.std(aa)) < INDEX_EPS or float(np.std(bb)) < INDEX_EPS:
        return None
    corr = np.corrcoef(aa, bb)
    val = float(corr[0, 1])
    return val if math.isfinite(val) else None


def index_documentation() -> dict[str, Any]:
    return {
        "robinson_si": ROBINSON_SI_DEFINITION,
        "asi": ASI_DEFINITION,
        "si_norm": SI_NORM_DEFINITION,
        "peak_timing": TIMING_DEFINITION,
        "overall_vs_knee": OVERALL_VS_KNEE_DEFINITION,
        "composite": COMPOSITE_DEFINITION,
        "alignment_event": ALIGNMENT_EVENT,
        "alignment": ALIGNMENT_EVENT_LABEL,
        "trajectory_rule": (
            "Left and right cycles are resampled to a common 0–100% gait-phase "
            "grid before comparison. Unaligned frames are not used."
        ),
        "insufficient": (
            "Indices are omitted (None) when a side is missing, cycles are "
            "unaligned, or L+R ≈ 0. Missing data is not scored as symmetry."
        ),
        "clinical": False,
    }
