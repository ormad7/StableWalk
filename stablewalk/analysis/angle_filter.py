"""Isolated-spike cleanup for geometric / flexion angle series."""

from __future__ import annotations

import numpy as np


def despike_angle_series(
    values: np.ndarray,
    *,
    max_jump_deg: float = 40.0,
    return_n: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Replace isolated one-frame spikes; keep sustained ROM changes.

    A sample is a spike when both adjacent deltas exceed ``max_jump_deg``.
    Genuine abnormalities last more than one frame and are left unchanged.
    """
    out = np.array(values, dtype=float, copy=True)
    n = out.size
    n_replaced = 0
    if n < 3:
        return (out, n_replaced) if return_n else out
    for i in range(1, n - 1):
        cur = out[i]
        prev, nxt = out[i - 1], out[i + 1]
        if not (np.isfinite(cur) and np.isfinite(prev) and np.isfinite(nxt)):
            continue
        if abs(cur - prev) > max_jump_deg and abs(cur - nxt) > max_jump_deg:
            out[i] = 0.5 * (prev + nxt)
            n_replaced += 1
    return (out, n_replaced) if return_n else out


__all__ = ["despike_angle_series"]
