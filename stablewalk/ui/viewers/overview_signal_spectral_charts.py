"""Overview Signal & Spectral Analysis drawers.

Reuses Spectral Analysis chart style. Display only — no new analysis pipeline.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from stablewalk.ui.colors import COM, MUTED, SIDE_LEFT, SIDE_RIGHT
from stablewalk.ui.theme import DANGER as THEME_DANGER
from stablewalk.ui.viewers.chart_playhead import PlayheadState, apply_playhead_to_axes
from stablewalk.ui.viewers.chart_style import (
    LINESTYLE_LEFT,
    LINESTYLE_RIGHT,
    SERIES_LINE_WIDTH,
    apply_chart_panel_style,
    configure_time_axis,
    style_chart_title,
)
from stablewalk.ui.viewers.spectral_charts import (
    draw_autocorr,
    draw_processed_signal,
    draw_psd,
)

_MOMENT_COLOR = THEME_DANGER


def _empty(ax: Axes, message: str) -> None:
    ax.clear()
    apply_chart_panel_style(ax)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        color=MUTED,
        transform=ax.transAxes,
        wrap=True,
    )


def draw_left_right_time(
    ax: Axes,
    *,
    times_s: np.ndarray | None,
    left: np.ndarray | None,
    right: np.ndarray | None,
    ylabel: str,
    left_label: str,
    right_label: str,
) -> None:
    ax.clear()
    apply_chart_panel_style(ax)
    style_chart_title(ax, "Left vs right")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    has_l = left is not None and times_s is not None and np.isfinite(left).any()
    has_r = right is not None and times_s is not None and np.isfinite(right).any()
    if times_s is None or (not has_l and not has_r):
        ax.text(
            0.5, 0.5, "No left/right pair for this selection",
            ha="center", va="center", color=MUTED, transform=ax.transAxes,
        )
        return
    if has_l:
        ax.plot(
            times_s, left, color=SIDE_LEFT, lw=SERIES_LINE_WIDTH,
            ls=LINESTYLE_LEFT, label=left_label,
        )
    if has_r:
        ax.plot(
            times_s, right, color=SIDE_RIGHT, lw=SERIES_LINE_WIDTH,
            ls=LINESTYLE_RIGHT, label=right_label,
        )
    ax.set_xlim(float(times_s[0]), float(times_s[-1]))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc="upper right",
            fontsize=7,
            frameon=True,
            fancybox=False,
            framealpha=0.88,
            edgecolor=MUTED,
            labelspacing=0.22,
            handlelength=1.35,
        )
    ax._sw_time_seek = True  # type: ignore[attr-defined]


def draw_asymmetry_score(
    ax: Axes,
    *,
    times_s: Sequence[float] | np.ndarray,
    scores: Sequence[float | None] | np.ndarray,
    moments: Sequence[Any] = (),
) -> None:
    ax.clear()
    apply_chart_panel_style(ax)
    style_chart_title(ax, "Asymmetry score")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score (0–100)")
    t = np.asarray(list(times_s), dtype=float)
    y = np.asarray(
        [float(v) if v is not None and np.isfinite(float(v)) else np.nan for v in scores],
        dtype=float,
    )
    if t.size == 0 or not np.isfinite(y).any():
        ax.text(
            0.5, 0.5, "Asymmetry score unavailable",
            ha="center", va="center", color=MUTED, transform=ax.transAxes,
        )
        ax.set_ylim(0, 100)
        return
    ax.plot(t, y, color=COM, lw=SERIES_LINE_WIDTH, label="Score")
    ax.set_ylim(0, 100)
    ax.set_xlim(float(t[0]), float(t[-1]))
    labeled = False
    for moment in moments:
        ts = getattr(moment, "time_s", None)
        if ts is None or not np.isfinite(float(ts)):
            continue
        ax.axvline(
            float(ts),
            color=_MOMENT_COLOR,
            ls="--",
            lw=1.2,
            label="Detected moment" if not labeled else None,
        )
        labeled = True
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc="upper right",
            fontsize=7,
            frameon=True,
            fancybox=False,
            framealpha=0.88,
            edgecolor=MUTED,
            labelspacing=0.22,
            handlelength=1.35,
        )
    ax._sw_time_seek = True  # type: ignore[attr-defined]


def layout_signal_spectral_figure(fig: Figure) -> None:
    """Keep titles/labels inside the axes on short Overview strips."""
    try:
        try:
            fig.set_layout_engine("none")
        except Exception:
            fig.set_constrained_layout(False)
        h_px = float(fig.get_size_inches()[1]) * float(fig.get_dpi() or 100.0)
        fig.subplots_adjust(
            left=0.07,
            right=0.985,
            top=0.88 if h_px < 280 else 0.90,
            bottom=0.16 if h_px < 280 else 0.12,
            hspace=0.92 if h_px < 280 else 0.62,
            wspace=0.36 if h_px < 280 else 0.30,
        )
    except Exception:
        pass


def build_signal_spectral_axes(fig: Figure) -> tuple[Axes, Axes, Axes, Axes, Axes]:
    """2×3 grid: signal | ACF | PSD on top; L/R | asymmetry (span 2) below."""
    fig.clear()
    gs = GridSpec(2, 3, figure=fig)
    ax_sig = fig.add_subplot(gs[0, 0])
    ax_acf = fig.add_subplot(gs[0, 1])
    ax_psd = fig.add_subplot(gs[0, 2])
    ax_lr = fig.add_subplot(gs[1, 0])
    ax_asym = fig.add_subplot(gs[1, 1:])
    for ax in (ax_sig, ax_acf, ax_psd, ax_lr, ax_asym):
        apply_chart_panel_style(ax)
        ax.set_xticks([])
        ax.set_yticks([])
    return ax_sig, ax_acf, ax_psd, ax_lr, ax_asym


def render_overview_signal_spectral(
    fig: Figure,
    axes: Sequence[Axes],
    *,
    report: Any | None,
    channel_id: str,
    times_lr: np.ndarray | None = None,
    left_vals: np.ndarray | None = None,
    right_vals: np.ndarray | None = None,
    lr_ylabel: str = "",
    left_label: str = "Left",
    right_label: str = "Right",
    asym_times: Sequence[float] | np.ndarray | None = None,
    asym_scores: Sequence[float | None] | np.ndarray | None = None,
    moments: Sequence[Any] = (),
    playhead: PlayheadState | None = None,
    empty_message: str | None = None,
) -> None:
    ax_sig, ax_acf, ax_psd, ax_lr, ax_asym = axes

    def _mark_seek() -> None:
        ax_sig._sw_time_seek = True  # type: ignore[attr-defined]
        ax_lr._sw_time_seek = True  # type: ignore[attr-defined]
        ax_asym._sw_time_seek = True  # type: ignore[attr-defined]
        ax_acf._sw_time_seek = False  # type: ignore[attr-defined]
        ax_psd._sw_time_seek = False  # type: ignore[attr-defined]

    if report is None or not getattr(report, "available", False):
        msg = empty_message or "Load a walking video to compute signal and spectral analysis."
        for ax in axes:
            _empty(ax, msg)
        _mark_seek()
        layout_signal_spectral_figure(fig)
        return
    draw_processed_signal(ax_sig, report, channel_id)
    style_chart_title(ax_sig, "Joint signal")
    draw_autocorr(ax_acf, report, channel_id)
    style_chart_title(ax_acf, "Autocorrelation")
    draw_psd(ax_psd, report, channel_id)
    style_chart_title(ax_psd, "Power spectrum")
    draw_left_right_time(
        ax_lr,
        times_s=times_lr,
        left=left_vals,
        right=right_vals,
        ylabel=lr_ylabel,
        left_label=left_label,
        right_label=right_label,
    )
    draw_asymmetry_score(
        ax_asym,
        times_s=() if asym_times is None else asym_times,
        scores=() if asym_scores is None else asym_scores,
        moments=moments,
    )
    _mark_seek()
    for ax in (ax_sig, ax_lr, ax_asym):
        configure_time_axis(ax, nbins=4)
    time_axes = [ax_sig, ax_lr, ax_asym]
    if playhead is not None:
        apply_playhead_to_axes(time_axes, playhead)
    layout_signal_spectral_figure(fig)
