"""Publication-quality reference overlays for biomechanics charts (display only)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib.axes import Axes

from stablewalk.ui.colors import BORDER, MUTED, PANEL, SIDE_LEFT, SIDE_RIGHT, TEXT

# Semantic reference colors (not L/R limb colors).
REF_NORMAL = "#2ec99a"
REF_ABNORMAL = "#ff6b81"
REF_CONFIDENCE = "#74c0fc"
EVENT_HS = "#f0f4f8"
EVENT_TO = "#ffc857"

# Display-only normative Y bands for common dashboard metrics.
# These do not alter analysis — they annotate charts for lab readability.
KNEE_FLEXION_NORMAL_DEG = (0.0, 70.0)
KNEE_FLEXION_ABNORMAL_ABOVE_DEG = 90.0
# Hip-centered COM height typically oscillates near 0 (mid-hip origin).
# Floor-absolute stature bands (~0.55 BH) do not apply in this frame.
COM_HEIGHT_NORMAL_BH = (-0.03, 0.03)
COM_HEIGHT_ABNORMAL_BELOW_BH = -0.06
COM_HEIGHT_ABNORMAL_ABOVE_BH = 0.06
# Stability margin chart/storage use metres after stature scaling.
STABILITY_MARGIN_NORMAL_M = (0.04, 0.25)
STABILITY_MARGIN_NORMAL_BH = STABILITY_MARGIN_NORMAL_M  # back-compat alias
CONTACT_ASYMMETRY_NORMAL = (0.0, 0.35)
VGRF_BW_NORMAL = (0.7, 1.35)


def draw_reference_y_bands(
    ax: Axes,
    *,
    normal: tuple[float, float] | None = None,
    abnormal_below: float | None = None,
    abnormal_above: float | None = None,
    label_normal: bool = True,
) -> None:
    """Green normal band and red abnormal zones (Visual3D / Qualisys style)."""
    y0, y1 = ax.get_ylim()
    if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
        return

    if abnormal_below is not None:
        lo = float(y0)
        hi = min(float(abnormal_below), float(y1))
        if hi > lo:
            ax.axhspan(
                lo,
                hi,
                facecolor=REF_ABNORMAL,
                edgecolor="none",
                alpha=0.10,
                zorder=0.2,
                label="_nolegend_",
            )
    if abnormal_above is not None:
        lo = max(float(abnormal_above), float(y0))
        hi = float(y1)
        if hi > lo:
            ax.axhspan(
                lo,
                hi,
                facecolor=REF_ABNORMAL,
                edgecolor="none",
                alpha=0.10,
                zorder=0.2,
                label="_nolegend_",
            )
    if normal is not None:
        n0, n1 = float(normal[0]), float(normal[1])
        lo = max(n0, float(y0))
        hi = min(n1, float(y1))
        if hi > lo:
            ax.axhspan(
                lo,
                hi,
                facecolor=REF_NORMAL,
                edgecolor="none",
                alpha=0.16,
                zorder=0.3,
                label="Normal range" if label_normal else "_nolegend_",
            )
            ax.axhline(n0, color=REF_NORMAL, linewidth=0.7, alpha=0.35, linestyle=":", zorder=0.4)
            ax.axhline(n1, color=REF_NORMAL, linewidth=0.7, alpha=0.35, linestyle=":", zorder=0.4)


def draw_confidence_overlay(
    ax: Axes,
    times: Sequence[float] | np.ndarray,
    confidence: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.55,
) -> None:
    """Shade low-confidence intervals (cyan wash) without changing series data."""
    t = np.asarray(times, dtype=float)
    c = np.asarray(confidence, dtype=float)
    if t.size < 2 or c.size != t.size:
        return
    low = c < float(threshold)
    if not np.any(low):
        return
    # Expand contiguous low-confidence runs into axvspan bands.
    start = None
    for i, flag in enumerate(low):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            t0 = float(t[start])
            t1 = float(t[i - 1])
            if t1 < t0:
                t0, t1 = t1, t0
            pad = max((float(t[-1]) - float(t[0])) * 0.002, 1e-4)
            ax.axvspan(
                t0 - pad,
                t1 + pad,
                facecolor=REF_CONFIDENCE,
                edgecolor="none",
                alpha=0.12,
                zorder=0.5,
                label="_nolegend_",
            )
            start = None
    if start is not None:
        ax.axvspan(
            float(t[start]),
            float(t[-1]),
            facecolor=REF_CONFIDENCE,
            edgecolor="none",
            alpha=0.12,
            zorder=0.5,
            label="_nolegend_",
        )


def draw_gait_event_markers(
    ax: Axes,
    *,
    heel_strike_times: Sequence[float] | None = None,
    toe_off_times: Sequence[float] | None = None,
    left_hs: Sequence[float] | None = None,
    right_hs: Sequence[float] | None = None,
    left_to: Sequence[float] | None = None,
    right_to: Sequence[float] | None = None,
    show_legend: bool = True,
) -> None:
    """OpenSim / Vicon-style heel-strike and toe-off markers."""
    y0, y1 = ax.get_ylim()
    if not np.isfinite(y0) or not np.isfinite(y1):
        return
    yr = y1 - y0 if y1 > y0 else 1.0
    hs_y = y1 - yr * 0.04
    to_y = y0 + yr * 0.04

    def _marks(
        times: Sequence[float] | None,
        *,
        color: str,
        marker: str,
        y: float,
        label: str | None,
    ) -> None:
        if not times:
            return
        xs = [float(t) for t in times if np.isfinite(t)]
        if not xs:
            return
        for x in xs:
            ax.axvline(
                x,
                color=color,
                linewidth=0.85,
                alpha=0.35,
                linestyle=(0, (3, 2)),
                zorder=2.5,
            )
        ax.scatter(
            xs,
            [y] * len(xs),
            marker=marker,
            s=28,
            color=color,
            edgecolors=PANEL,
            linewidths=0.45,
            zorder=6,
            label=label if (show_legend and label) else "_nolegend_",
            clip_on=False,
        )

    # Bilateral / unlabeled
    _marks(heel_strike_times, color=EVENT_HS, marker="v", y=hs_y, label="Heel strike")
    _marks(toe_off_times, color=EVENT_TO, marker="^", y=to_y, label="Toe off")
    # Side-specific (L green / R red — laboratory convention)
    _marks(left_hs, color=SIDE_LEFT, marker="v", y=hs_y, label="L HS")
    _marks(right_hs, color=SIDE_RIGHT, marker="v", y=hs_y, label="R HS")
    _marks(left_to, color=SIDE_LEFT, marker="^", y=to_y, label="L TO")
    _marks(right_to, color=SIDE_RIGHT, marker="^", y=to_y, label="R TO")


def annotate_playhead_value(
    ax: Axes,
    x: float,
    value: float | None,
    *,
    unit: str = "",
    color: str = TEXT,
    zorder: int = 22,
) -> None:
    """Current-value callout next to the playhead (publication readout)."""
    if value is None or not np.isfinite(value):
        return
    y0, y1 = ax.get_ylim()
    yr = y1 - y0 if y1 > y0 else 1.0
    x0, x1 = ax.get_xlim()
    xr = x1 - x0 if x1 > x0 else 1.0
    text = f"{value:.2f}{unit}"
    ax.text(
        x + xr * 0.008,
        min(max(value, y0 + yr * 0.06), y1 - yr * 0.08),
        text,
        color=color,
        fontsize=8.0,
        fontweight="semibold",
        va="center",
        ha="left",
        zorder=zorder,
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": PANEL,
            "edgecolor": BORDER,
            "alpha": 0.92,
            "linewidth": 0.6,
        },
    )


__all__ = [
    "COM_HEIGHT_ABNORMAL_ABOVE_BH",
    "COM_HEIGHT_ABNORMAL_BELOW_BH",
    "COM_HEIGHT_NORMAL_BH",
    "CONTACT_ASYMMETRY_NORMAL",
    "EVENT_HS",
    "EVENT_TO",
    "KNEE_FLEXION_ABNORMAL_ABOVE_DEG",
    "KNEE_FLEXION_NORMAL_DEG",
    "REF_ABNORMAL",
    "REF_CONFIDENCE",
    "REF_NORMAL",
    "STABILITY_MARGIN_NORMAL_BH",
    "STABILITY_MARGIN_NORMAL_M",
    "VGRF_BW_NORMAL",
    "annotate_playhead_value",
    "draw_confidence_overlay",
    "draw_gait_event_markers",
    "draw_reference_y_bands",
]
