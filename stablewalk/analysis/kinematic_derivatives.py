"""Kinematic derivatives with gap-aware differences and one justified smooth.

Winter-style marker processing: despike isolated samples, Savitzky–Golay
smooth the *base* trajectory (angle or XYZ), then differentiate. Do not
smooth after each derivative — that would over-attenuate genuine gait.

Third-order unsmoothed finite differences of 25 Hz pose noise produce
O(10⁴) deg/s³ peaks that are not physiological angular jerk.

Polynomial order 3 reproduces cubics in the interior of the window, so
analytic constant-jerk tests remain exact away from the filter transients.
"""

from __future__ import annotations

import math

import numpy as np

from stablewalk.analysis.joint_path_trace import max_connected_dt

# Landmark visibility below this is treated as missing, not as a real sample.
MIN_DERIVATIVE_VISIBILITY = 0.35
# SG polynomial degree: cubics (constant jerk) are reproduced exactly.
SAVGOL_POLYORDER = 3
# Target cutoff near typical marker-filter practice (~6 Hz), bounded by Nyquist.
SAVGOL_CUTOFF_HZ = 6.0


def kinematic_sg_window(fps: float, n_samples: int) -> int:
    """Odd SG window: ~0.20–0.40 s, at least 5, not longer than the series."""
    rate = float(fps) if fps and fps > 1e-6 else 30.0
    nyquist = 0.5 * rate
    cutoff = min(SAVGOL_CUTOFF_HZ, 0.45 * nyquist) if nyquist > 1.0 else SAVGOL_CUTOFF_HZ
    # Longer than a 6 Hz period at 25 Hz so a cubic cannot track pose flicker,
    # still short versus a ~1 s gait cycle so ROM steps remain.
    duration_s = min(0.64, max(0.52, 3.0 / max(cutoff, 1.0)))
    w = int(round(rate * duration_s))
    if w % 2 == 0:
        w += 1
    w = max(13, w)
    max_odd = n_samples if n_samples % 2 == 1 else n_samples - 1
    if max_odd < 5:
        return 0
    return min(w, max_odd)


def savgol_smooth(values: np.ndarray, *, window_samples: int, polyorder: int) -> np.ndarray:
    """Centered Savitzky–Golay on fully finite windows; NaN samples stay NaN."""
    y = np.asarray(values, dtype=float).copy()
    n = y.size
    w = int(window_samples)
    if w < 5 or n < 5:
        return y
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 5:
        return y
    order = max(1, min(int(polyorder), w - 1))
    half = w // 2
    x = np.arange(-half, half + 1, dtype=float)
    vandermonde = np.vander(x, N=order + 1, increasing=True)
    coeffs = np.linalg.solve(vandermonde.T @ vandermonde, vandermonde.T)[0]
    out = y.copy()
    for i in range(half, n - half):
        sl = y[i - half : i + half + 1]
        if not np.all(np.isfinite(sl)):
            continue
        out[i] = float(np.dot(coeffs, sl))
    return out


def _savgol_derivative_rows(window_samples: int, polyorder: int, h: float, max_deriv: int) -> np.ndarray:
    """Rows 0..max_deriv: SG coefficients for the value and time derivatives."""
    w = int(window_samples)
    half = w // 2
    x = np.arange(-half, half + 1, dtype=float)
    order = max(1, min(int(polyorder), w - 1))
    vandermonde = np.vander(x, N=order + 1, increasing=True)
    proj = np.linalg.solve(vandermonde.T @ vandermonde, vandermonde.T)
    rows = []
    for deriv in range(max_deriv + 1):
        if deriv > order:
            rows.append(np.zeros(w, dtype=float))
            continue
        scale = math.factorial(deriv) / (float(h) ** deriv)
        rows.append(scale * proj[deriv])
    return np.vstack(rows)


def savgol_derivatives(
    values: np.ndarray,
    times: np.ndarray,
    *,
    fps: float,
    max_deriv: int,
    window_samples: int,
    polyorder: int = SAVGOL_POLYORDER,
) -> list[np.ndarray]:
    """Local polynomial value and derivatives; NaN where the window is incomplete."""
    y = np.asarray(values, dtype=float)
    n = y.size
    w = int(window_samples)
    out = [np.full(n, np.nan, dtype=float) for _ in range(max_deriv + 1)]
    if w < 5 or n < w or fps is None or fps <= 1e-6:
        return out
    if w % 2 == 0:
        w += 1
    half = w // 2
    rows = _savgol_derivative_rows(w, polyorder, 1.0 / float(fps), max_deriv)
    for i in range(half, n - half):
        sl = y[i - half : i + half + 1]
        if not np.all(np.isfinite(sl)):
            continue
        for d in range(max_deriv + 1):
            out[d][i] = float(np.dot(rows[d], sl))
    return out


def mask_low_confidence(
    values: np.ndarray,
    confidence: np.ndarray | None,
    *,
    min_visibility: float = MIN_DERIVATIVE_VISIBILITY,
) -> np.ndarray:
    """NaN samples whose landmark visibility is below the derivative gate."""
    out = np.asarray(values, dtype=float).copy()
    if confidence is None:
        return out
    conf = np.asarray(confidence, dtype=float)
    if conf.shape != out.shape:
        return out
    poor = np.isfinite(conf) & (conf < float(min_visibility))
    out[poor] = np.nan
    return out


def finite_diff(
    values: np.ndarray,
    times: np.ndarray,
    *,
    fps: float | None = None,
) -> np.ndarray:
    """Central / one-sided difference; NaN across tracking gaps.

    A pair is a gap when Δt exceeds ``max_connected_dt(fps)`` (1.5/fps),
    matching joint-path length. Differencing across a hole is not a derivative.
    """
    y = np.asarray(values, dtype=float)
    t = np.asarray(times, dtype=float)
    n = y.size
    out = np.full(n, np.nan, dtype=float)
    if n < 2:
        return out
    if fps is None:
        dts = np.diff(t)
        dts = dts[np.isfinite(dts) & (dts > 1e-9)]
        fps = float(1.0 / np.median(dts)) if dts.size else 30.0
    max_dt = max_connected_dt(fps) + 1e-12

    def _next_finite(i: int, step: int) -> int | None:
        j = i + step
        while 0 <= j < n:
            if np.isfinite(y[j]) and np.isfinite(t[j]):
                return j
            j += step
        return None

    for i in range(n):
        if not np.isfinite(y[i]) or not np.isfinite(t[i]):
            continue
        if i == 0:
            j = _next_finite(i, 1)
            if j is None:
                continue
            dt = t[j] - t[i]
            if 1e-9 < dt <= max_dt:
                out[i] = (y[j] - y[i]) / dt
            continue
        if i == n - 1:
            j = _next_finite(i, -1)
            if j is None:
                continue
            dt = t[i] - t[j]
            if 1e-9 < dt <= max_dt:
                out[i] = (y[i] - y[j]) / dt
            continue
        j0 = _next_finite(i, -1)
        j1 = _next_finite(i, 1)
        if j0 is None or j1 is None:
            continue
        dt = t[j1] - t[j0]
        left_ok = (t[i] - t[j0]) <= max_dt
        right_ok = (t[j1] - t[i]) <= max_dt
        if left_ok and right_ok and dt > 1e-9:
            out[i] = (y[j1] - y[j0]) / dt
        elif left_ok and (t[i] - t[j0]) > 1e-9:
            out[i] = (y[i] - y[j0]) / (t[i] - t[j0])
        elif right_ok and (t[j1] - t[i]) > 1e-9:
            out[i] = (y[j1] - y[i]) / (t[j1] - t[i])
    return out


def smooth_base_series(
    values: np.ndarray,
    times: np.ndarray,
    *,
    fps: float | None = None,
    confidence: np.ndarray | None = None,
    despike: bool = False,
    max_jump: float | None = None,
) -> np.ndarray:
    """Confidence-mask, optional despike, then one SG pass. Genuine multi-frame motion remains."""
    y = mask_low_confidence(values, confidence)
    if despike:
        from stablewalk.analysis.angle_filter import despike_angle_series

        kwargs = {}
        if max_jump is not None:
            kwargs["max_jump_deg"] = float(max_jump)
        y = despike_angle_series(y, **kwargs)
    t = np.asarray(times, dtype=float)
    if fps is None:
        dts = np.diff(t)
        dts = dts[np.isfinite(dts) & (dts > 1e-9)]
        fps = float(1.0 / np.median(dts)) if dts.size else 30.0
    window = kinematic_sg_window(fps, y.size)
    if window >= 5:
        y = savgol_smooth(y, window_samples=window, polyorder=SAVGOL_POLYORDER)
    return y


def differentiate_chain(
    values: np.ndarray,
    times: np.ndarray,
    *,
    order: int,
    fps: float | None = None,
    confidence: np.ndarray | None = None,
    despike: bool = False,
    max_jump: float | None = None,
) -> list[np.ndarray]:
    """Smooth the base series once, then take ``order`` gap-aware derivatives.

    Returns ``[base, d1, d2, ...]`` of length ``order + 1``.
    """
    base = mask_low_confidence(values, confidence)
    if despike:
        from stablewalk.analysis.angle_filter import despike_angle_series

        kwargs = {}
        if max_jump is not None:
            kwargs["max_jump_deg"] = float(max_jump)
        base = despike_angle_series(base, **kwargs)
    t = np.asarray(times, dtype=float)
    if fps is None:
        dts = np.diff(t)
        dts = dts[np.isfinite(dts) & (dts > 1e-9)]
        fps = float(1.0 / np.median(dts)) if dts.size else 30.0
    window = kinematic_sg_window(fps, base.size)
    if window >= 9 and int(order) >= 1:
        chain = savgol_derivatives(
            base,
            t,
            fps=fps,
            max_deriv=int(order),
            window_samples=window,
            polyorder=SAVGOL_POLYORDER,
        )
        # Edges have no full SG window. Keep the despiked value; first
        # derivative may use a one-sided difference. Do not fill 2nd/3rd
        # derivatives with raw finite differences — that reintroduces the
        # 10⁴ deg/s³ pose-noise artifact the filter is meant to avoid.
        missing_val = ~np.isfinite(chain[0]) & np.isfinite(base)
        chain[0][missing_val] = base[missing_val]
        if int(order) >= 1:
            omega_fd = finite_diff(base, times, fps=fps)
            missing_w = ~np.isfinite(chain[1]) & np.isfinite(omega_fd)
            chain[1][missing_w] = omega_fd[missing_w]
        return chain
    chain = [base]
    current = base
    for _ in range(max(0, int(order))):
        current = finite_diff(current, times, fps=fps)
        chain.append(current)
    return chain


__all__ = [
    "MIN_DERIVATIVE_VISIBILITY",
    "SAVGOL_POLYORDER",
    "differentiate_chain",
    "finite_diff",
    "kinematic_sg_window",
    "mask_low_confidence",
    "savgol_derivatives",
    "savgol_smooth",
    "smooth_base_series",
]
