"""Drawers for the Four Gait Signals workspace. Display only."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from stablewalk.analysis.four_gait_signals import (
    COMPARE_PAIRS,
    FOCUS_PAIRS,
    FOUR_GAIT_CHANNEL_IDS,
    FOUR_GAIT_SHORT_LABELS,
    LR_XCORR_PAIRS,
    FourGaitSignalsReport,
    SignalMode,
    correlation_at_lag,
    lag_seconds,
)
from stablewalk.analysis.spectral.channels import (
    FOUR_GAIT_ANGLE_YLABEL,
    FOUR_GAIT_PANEL_TITLES,
)
from stablewalk.analysis.spectral.config import DEFAULT_GAIT_BAND_HZ, DEFAULT_STRIDE_LAG_S
from stablewalk.ui.colors import ACCENT, COM, MUTED, SIDE_LEFT, SIDE_RIGHT, TEXT, TEXT_SECONDARY
from stablewalk.ui.viewers.chart_playhead import PlayheadState, apply_playhead_to_axes
from stablewalk.ui.viewers.chart_style import (
    LINESTYLE_LEFT,
    LINESTYLE_RIGHT,
    SERIES_LINE_WIDTH,
    TIMELINE_X_LABEL,
    Y_AUTOSCALE_PAD,
    apply_chart_grid,
    apply_chart_panel_style,
    configure_time_axis,
    style_chart_legend,
    style_chart_title,
    unclip_axis_chrome,
    unclip_figure_chrome,
)

_PANEL_ORDER: tuple[tuple[str, str], ...] = tuple(
    zip(FOUR_GAIT_CHANNEL_IDS, FOUR_GAIT_PANEL_TITLES)
)

_SERIES_STYLE: dict[str, tuple[str, str | tuple, float]] = {
    "seg:left_thigh": (SIDE_LEFT, LINESTYLE_LEFT, SERIES_LINE_WIDTH),
    "seg:right_thigh": (SIDE_RIGHT, LINESTYLE_RIGHT, SERIES_LINE_WIDTH),
    "seg:left_shank": (SIDE_LEFT, (0, (2.0, 1.8)), SERIES_LINE_WIDTH * 0.92),
    "seg:right_shank": (SIDE_RIGHT, (0, (2.0, 1.8)), SERIES_LINE_WIDTH * 0.92),
}

_FOCUS_CELLS = {(i, j) for _name, i, j, _lab in FOCUS_PAIRS} | {
    (j, i) for _name, i, j, _lab in FOCUS_PAIRS
}

_THIGH_IDS: tuple[str, ...] = ("seg:left_thigh", "seg:right_thigh")
_SHANK_IDS: tuple[str, ...] = ("seg:left_shank", "seg:right_shank")
_LR_OVERLAY_PAIRS: dict[str, tuple[tuple[str, str], tuple[str, str], str]] = {
    "thigh": (
        ("seg:left_thigh", "Left Thigh"),
        ("seg:right_thigh", "Right Thigh"),
        "Thigh — left vs right (hip → knee)",
    ),
    "shank": (
        ("seg:left_shank", "Left Shank"),
        ("seg:right_shank", "Right Shank"),
        "Shank — left vs right (knee → ankle)",
    ),
}


def _clear(ax: Axes) -> None:
    ax.clear()
    apply_chart_panel_style(ax)


def _finish(ax: Axes) -> None:
    unclip_axis_chrome(ax)


def _empty(ax: Axes, text: str) -> None:
    _clear(ax)
    ax.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, transform=ax.transAxes)
    ax.set_axis_off()


def _time_xlim(report: FourGaitSignalsReport) -> tuple[float, float]:
    times = np.asarray(report.times_s, dtype=float)
    if times.size:
        x0, x1 = float(times[0]), float(times[-1])
        if x1 <= x0:
            x1 = x0 + 1.0
        return x0, x1
    return 0.0, 1.0


def _ylim_for(
    series: dict[str, np.ndarray],
    channel_ids: Sequence[str],
) -> tuple[float, float]:
    finite: list[float] = []
    for cid in channel_ids:
        col = np.asarray(series.get(cid), dtype=float)
        ok = col[np.isfinite(col)]
        if ok.size:
            finite.extend(float(v) for v in ok)
    if not finite:
        return (-10.0, 80.0)
    lo = min(finite)
    hi = max(finite)
    span = hi - lo
    pad = 5.0 if span < 1.0 else max(2.0, span * Y_AUTOSCALE_PAD)
    return (lo - pad, hi + pad)


def shared_limits(
    report: FourGaitSignalsReport,
    mode: SignalMode,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Union limits for the combined overlay (all four traces)."""
    series = report.series_for(mode)
    return _time_xlim(report), _ylim_for(series, FOUR_GAIT_CHANNEL_IDS)


def pair_limits(
    report: FourGaitSignalsReport,
    mode: SignalMode,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Shared time axis; one Y range per L/R equivalent segment pair."""
    series = report.series_for(mode)
    return (
        _time_xlim(report),
        _ylim_for(series, _THIGH_IDS),
        _ylim_for(series, _SHANK_IDS),
    )


def layout_four_gait_figure(
    fig: Figure,
    *,
    combined: bool = False,
    matrix: bool = False,
    overlay: bool = False,
    spectral: bool = False,
) -> None:
    try:
        fig.set_constrained_layout(False)
    except Exception:
        pass
    if matrix:
        fig.subplots_adjust(left=0.16, right=0.88, top=0.88, bottom=0.18)
        return
    height_px = 0.0
    try:
        height_px = float(fig.get_figheight() or 0.0) * float(fig.dpi or 100.0)
    except Exception:
        height_px = 0.0
    short = bool(height_px and height_px < 560)
    top = 0.82 if short else 0.88
    bottom = 0.14 if short else 0.12
    hspace = 0.68 if short else 0.58
    if overlay:
        fig.subplots_adjust(
            left=0.10,
            right=0.985,
            top=top,
            bottom=bottom,
            hspace=hspace,
            wspace=0.18,
        )
        return
    if spectral:
        fig.subplots_adjust(
            left=0.10,
            right=0.96,
            top=top,
            bottom=max(0.12, bottom - 0.02),
            hspace=max(0.50, hspace - 0.04),
            wspace=0.26,
        )
        return
    if combined:
        fig.subplots_adjust(
            left=0.10,
            right=0.985,
            top=top,
            bottom=bottom,
            hspace=hspace,
            wspace=0.20,
        )
        return
    # Full-width 2×2: titles, y-labels, and Time (s) stay inside the figure.
    fig.subplots_adjust(
        left=0.11,
        right=0.985,
        top=top,
        bottom=bottom,
        hspace=hspace,
        wspace=0.22,
    )


def _draw_one_signal(
    ax: Axes,
    times: np.ndarray,
    values: np.ndarray,
    *,
    channel_id: str,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    ylabel: bool,
    show_xlabel: bool = True,
    show_xticklabels: bool = True,
) -> None:
    _clear(ax)
    color, ls, lw = _SERIES_STYLE[channel_id]
    if values.size and times.size:
        ax.plot(times, values, color=color, linestyle=ls, lw=lw, label=title)
    style_chart_title(ax, title, pad=8.0)
    ax.set_ylabel(FOUR_GAIT_ANGLE_YLABEL, color=TEXT_SECONDARY)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    configure_time_axis(ax, show_xlabel=show_xlabel, nbins=5)
    if not show_xlabel:
        ax.set_xlabel("")
    if not show_xticklabels:
        ax.tick_params(axis="x", labelbottom=False)
    apply_chart_grid(ax)
    ax._sw_time_seek = True  # type: ignore[attr-defined]
    ax._sw_nav_xlim = True  # type: ignore[attr-defined]
    ax._stablewalk_playhead_value_source = (  # type: ignore[attr-defined]
        times,
        values,
        title,
        "deg",
    )
    _finish(ax)


def draw_lr_overlay(
    ax: Axes,
    report: FourGaitSignalsReport,
    mode: SignalMode,
    *,
    pair: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    show_xlabel: bool = True,
) -> None:
    spec = _LR_OVERLAY_PAIRS.get(pair)
    if spec is None:
        _empty(ax, "Unknown left–right pair.")
        return
    left, right, title = spec
    _clear(ax)
    times = report.times_s
    series = report.series_for(mode)
    drawn = 0
    for cid, lab in (left, right):
        color, ls, lw = _SERIES_STYLE[cid]
        values = np.asarray(series.get(cid), dtype=float)
        if values.size and times.size:
            ax.plot(times, values, color=color, linestyle=ls, lw=lw, label=lab)
            drawn += 1
    style_chart_title(ax, title, pad=6.0)
    ax.set_ylabel(FOUR_GAIT_ANGLE_YLABEL, color=TEXT_SECONDARY)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    configure_time_axis(ax, show_xlabel=show_xlabel, nbins=6)
    if not show_xlabel:
        ax.set_xlabel("")
    apply_chart_grid(ax)
    if drawn:
        style_chart_legend(ax, loc="upper right", fontsize=8, ncol=2, outside=False)
    else:
        ax.text(
            0.5,
            0.5,
            "Segment angle unavailable",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
        )
    ax._sw_time_seek = True  # type: ignore[attr-defined]
    ax._sw_nav_xlim = True  # type: ignore[attr-defined]
    _finish(ax)


def draw_combined(
    ax: Axes,
    report: FourGaitSignalsReport,
    mode: SignalMode,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    _clear(ax)
    times = report.times_s
    series = report.series_for(mode)
    for cid, title in _PANEL_ORDER:
        color, ls, lw = _SERIES_STYLE[cid]
        values = np.asarray(series.get(cid), dtype=float)
        if values.size and times.size:
            ax.plot(times, values, color=color, linestyle=ls, lw=lw, label=title)
    style_chart_title(ax, "Combined — all four segment angles", pad=6.0)
    ax.set_xlabel(TIMELINE_X_LABEL, color=TEXT_SECONDARY)
    ax.set_ylabel(FOUR_GAIT_ANGLE_YLABEL, color=TEXT_SECONDARY)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    configure_time_axis(ax, show_xlabel=True, nbins=6)
    apply_chart_grid(ax)
    style_chart_legend(ax, loc="upper right", fontsize=8, ncol=2, outside=False)
    ax._sw_time_seek = True  # type: ignore[attr-defined]
    ax._sw_nav_xlim = True  # type: ignore[attr-defined]
    _finish(ax)


def draw_correlation_heatmap(
    ax: Axes,
    report: FourGaitSignalsReport,
    mode: SignalMode,
    *,
    lag_samples: int = 0,
    zero_lag: bool = False,
) -> None:
    cb = getattr(ax, "_sw_colorbar", None)
    if cb is not None:
        try:
            cb.remove()
        except Exception:
            pass
        ax._sw_colorbar = None
    _clear(ax)
    fps = max(float(report.fps), 1e-6)
    if zero_lag:
        lag_samples = 0
        style_chart_title(ax, "Zero-lag Pearson r  ·  C(0)", pad=6.0)
        ax.set_xlabel("Segment j at the same time", color=TEXT_SECONDARY)
        ax.set_ylabel("Segment i at the same time", color=TEXT_SECONDARY)
        bar_label = "C(0) Pearson r"
    else:
        lag_s = lag_seconds(report, int(lag_samples))
        style_chart_title(
            ax,
            f"C(τ) Pearson r  ·  τ = {lag_s:+.3f} s  "
            f"({int(lag_samples)} samples @ {fps:.2f} Hz)",
            pad=6.0,
        )
        ax.set_xlabel("Segment j at t+τ", color=TEXT_SECONDARY)
        ax.set_ylabel("Segment i at t", color=TEXT_SECONDARY)
        bar_label = "C(τ) Pearson r"
    corr = np.asarray(correlation_at_lag(report, mode, int(lag_samples)), dtype=float)
    labels = list(FOUR_GAIT_SHORT_LABELS)
    cmap = None
    try:
        from matplotlib import colormaps

        cmap = colormaps["coolwarm"].copy()
        cmap.set_bad("#c5c9d0")
    except Exception:
        cmap = "coolwarm"
    im = ax.imshow(
        corr,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        origin="upper",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8, color=TEXT)
    ax.set_yticklabels(labels, fontsize=8, color=TEXT)
    n = corr.shape[0]
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            txt = "—" if not np.isfinite(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color=TEXT, fontsize=8)
            if (i, j) in _FOCUS_CELLS and i != j:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor=TEXT,
                        linewidth=1.6,
                    )
                )
    fig = ax.figure
    if fig is not None:
        try:
            bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            bar.set_label(bar_label + " (−1 to +1)", color=TEXT, fontsize=8)
            bar.ax.tick_params(colors=TEXT, labelsize=8)
            bar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax._sw_colorbar = bar
        except Exception:
            pass
    _finish(ax)


def draw_four_gait_acf(
    ax: Axes,
    report: FourGaitSignalsReport,
    mode: SignalMode,
    channel_id: str,
    title: str,
) -> None:
    _clear(ax)
    color, ls, lw = _SERIES_STYLE[channel_id]
    ac = report.autocorr_for(mode).get(channel_id)
    style_chart_title(ax, f"ACF · {title}", pad=6.0)
    ax.set_xlabel("Lag τ (s)")
    ax.set_ylabel("ρ(τ)")
    apply_chart_grid(ax)
    ax._sw_time_seek = False  # type: ignore[attr-defined]
    if ac is None or getattr(ac, "lags_s", None) is None or np.size(ac.lags_s) == 0:
        ax.text(0.5, 0.5, "ACF unavailable", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    lo, hi = DEFAULT_STRIDE_LAG_S
    ax.axvspan(lo, hi, color=ACCENT, alpha=0.07, label="Stride search")
    ax.plot(ac.lags_s, ac.rho, color=color, linestyle=ls, lw=lw, label=title)
    ax.axhline(0.0, color=MUTED, lw=0.8, alpha=0.6)
    peaks = list(getattr(ac, "peaks", None) or [])
    if peaks:
        ax.scatter(
            [float(p.lag_s) for p in peaks[:4]],
            [float(p.value) for p in peaks[:4]],
            s=26,
            c=ACCENT,
            zorder=5,
            label="Local peaks",
        )
    period = getattr(ac, "gait_period_s", None)
    withheld = bool(getattr(ac, "insufficient_reason", None)) or (
        getattr(ac, "reliability_label", "") == "insufficient"
    )
    if period and not withheld:
        ax.axvline(
            float(period),
            color=ACCENT,
            ls="--",
            lw=1.15,
            label=f"ACF period {float(period):.2f}s ({ac.reliability_label})",
        )
    elif withheld:
        reason = str(getattr(ac, "insufficient_reason", "") or "ACF period withheld")
        ax.text(
            0.02,
            0.08,
            reason,
            transform=ax.transAxes,
            color=MUTED,
            fontsize=7,
            va="bottom",
        )
    ax.set_ylim(-1.05, 1.05)
    style_chart_legend(ax, loc="upper right", fontsize=7, ncol=1, outside=False)


def draw_four_gait_xcorr(
    ax: Axes,
    report: FourGaitSignalsReport,
    mode: SignalMode,
    pair_name: str,
    title: str,
) -> None:
    _clear(ax)
    xc = report.xcorr_for(mode).get(pair_name)
    style_chart_title(ax, f"XCorr · {title}", pad=6.0)
    ax.set_xlabel("Lag τ (s)")
    ax.set_ylabel("ρ(τ)")
    apply_chart_grid(ax)
    ax._sw_time_seek = False  # type: ignore[attr-defined]
    if xc is None or getattr(xc, "lags_s", None) is None or np.size(xc.lags_s) == 0:
        ax.text(0.5, 0.5, "XCorr unavailable", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    ax.plot(xc.lags_s, xc.rho, color=COM, lw=SERIES_LINE_WIDTH, label=title)
    ax.axhline(0.0, color=MUTED, lw=0.8, alpha=0.6)
    ax.axvline(0.0, color=MUTED, lw=0.8, alpha=0.45)
    if xc.peak_lag_s is not None and xc.peak_value is not None:
        ax.axvline(float(xc.peak_lag_s), color=ACCENT, ls="--", lw=1.15)
        ax.scatter(
            [float(xc.peak_lag_s)],
            [float(xc.peak_value)],
            s=32,
            c=ACCENT,
            zorder=5,
            label=(
                f"best phase r={float(xc.peak_value):.2f} "
                f"@ {float(xc.peak_lag_s):+.2f}s"
            ),
        )
    ax.set_ylim(-1.05, 1.05)
    style_chart_legend(ax, loc="upper right", fontsize=7, ncol=1, outside=False)


def build_matrix_axes(fig: Figure) -> list[Axes]:
    fig.clear()
    ax = fig.add_subplot(111)
    layout_four_gait_figure(fig, matrix=True)
    return [ax]


def render_four_gait_matrix_figure(
    fig: Figure,
    axes: Sequence[Axes],
    report: FourGaitSignalsReport | None,
    mode: SignalMode,
    *,
    lag_samples: int = 0,
) -> None:
    ax = axes[0] if axes else None
    if ax is None:
        return
    if report is None or not report.available:
        _empty(ax, "Load a walking video to compute the correlation matrix.")
        return
    # Main figure is always C(0). Selectable C(τ) is the side heatmap.
    draw_correlation_heatmap(ax, report, mode, lag_samples=0, zero_lag=True)
    layout_four_gait_figure(fig, matrix=True)
    unclip_figure_chrome(fig)


def build_corr_axes(fig: Figure) -> list[Axes]:
    fig.clear()
    gs = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 1.08))
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]
    layout_four_gait_figure(fig, combined=True)
    try:
        fig.subplots_adjust(
            left=0.10, right=0.985, top=0.94, bottom=0.10, hspace=0.50, wspace=0.24
        )
    except Exception:
        pass
    return axes


def render_four_gait_corr_figure(
    fig: Figure,
    axes: Sequence[Axes],
    report: FourGaitSignalsReport | None,
    mode: SignalMode,
) -> None:
    if report is None or not report.available:
        for ax in axes:
            _empty(ax, "Load a walking video to compute ACF and cross-correlation.")
        return
    for ax, (cid, title) in zip(axes[:4], _PANEL_ORDER):
        draw_four_gait_acf(ax, report, mode, cid, title)
    for ax, (_name, _l, _r, title) in zip(axes[4:6], LR_XCORR_PAIRS):
        draw_four_gait_xcorr(ax, report, mode, _name, title)
    unclip_figure_chrome(fig)


def build_signal_axes(
    fig: Figure, *, combined: bool, lr_overlay: bool = False
) -> list[Axes]:
    fig.clear()
    axes: list[Axes] = []
    if lr_overlay:
        rows = 3 if combined else 2
        ratios = (1.0, 1.0, 1.12)[:rows]
        gs = fig.add_gridspec(rows, 1, height_ratios=ratios)
        first = fig.add_subplot(gs[0, 0])
        axes.append(first)
        axes.append(fig.add_subplot(gs[1, 0], sharex=first))
        if combined:
            axes.append(fig.add_subplot(gs[2, 0], sharex=first))
    elif combined:
        gs = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 1.15))
        thigh_l = fig.add_subplot(gs[0, 0])
        axes.append(thigh_l)
        axes.append(fig.add_subplot(gs[0, 1], sharex=thigh_l, sharey=thigh_l))
        shank_l = fig.add_subplot(gs[1, 0], sharex=thigh_l)
        axes.append(shank_l)
        axes.append(fig.add_subplot(gs[1, 1], sharex=thigh_l, sharey=shank_l))
        axes.append(fig.add_subplot(gs[2, :], sharex=thigh_l))
    else:
        gs = fig.add_gridspec(2, 2)
        thigh_l = fig.add_subplot(gs[0, 0])
        axes.append(thigh_l)
        axes.append(fig.add_subplot(gs[0, 1], sharex=thigh_l, sharey=thigh_l))
        shank_l = fig.add_subplot(gs[1, 0], sharex=thigh_l)
        axes.append(shank_l)
        axes.append(fig.add_subplot(gs[1, 1], sharex=thigh_l, sharey=shank_l))
    layout_four_gait_figure(fig, combined=combined, overlay=lr_overlay)
    return axes


def shared_spectrum_limits(
    report: FourGaitSignalsReport,
    mode: SignalMode,
) -> tuple[tuple[float, float], tuple[float, float]]:
    spectra = report.spectrum_for(mode)
    xmax = 8.0
    ymax = 0.0
    last_f: list[float] = []
    for spec in spectra.values():
        freq = spec.freq_hz if spec.freq_hz.size else spec.fft_freq_hz
        psd = spec.psd if spec.freq_hz.size else spec.fft_power
        if freq.size:
            last_f.append(float(freq[-1]))
        ok = np.asarray(psd, dtype=float)
        ok = ok[np.isfinite(ok)]
        if ok.size:
            ymax = max(ymax, float(np.max(ok)))
    if last_f:
        xmax = min(8.0, max(last_f))
    if xmax <= 0.0:
        xmax = 8.0
    if ymax <= 0.0:
        ymax = 1.0
    return (0.0, xmax), (0.0, ymax * 1.12)


_DELTA_PAIR = {
    "seg:left_thigh": "thigh_lr",
    "seg:right_thigh": "thigh_lr",
    "seg:left_shank": "shank_lr",
    "seg:right_shank": "shank_lr",
}


def _metric_box(ax: Axes, lines: Sequence[str], *, loc: str = "upper left") -> None:
    text = "\n".join(str(line) for line in lines if line)
    if not text:
        return
    ha, va, x, y = "left", "top", 0.02, 0.97
    if loc == "upper right":
        ha, x = "right", 0.98
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=6.5,
        color=TEXT,
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
        },
    )


def draw_four_gait_spectrum(
    ax: Axes,
    report: FourGaitSignalsReport,
    mode: SignalMode,
    channel_id: str,
    title: str,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    _clear(ax)
    color, ls, lw = _SERIES_STYLE[channel_id]
    spec = report.spectrum_for(mode).get(channel_id)
    quality = str(getattr(spec, "quality_status", "") or "") if spec is not None else ""
    suffix = f" · {quality}" if quality and quality != "computed" else ""
    style_chart_title(ax, f"PSD · {title}{suffix}", pad=6.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (z²/Hz)")
    apply_chart_grid(ax)
    ax._sw_time_seek = False  # type: ignore[attr-defined]
    ax._sw_nav_xlim = True  # type: ignore[attr-defined]
    if spec is None:
        ax.text(
            0.5,
            0.5,
            "Spectrum unavailable",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return
    freq = spec.freq_hz if spec.freq_hz.size else spec.fft_freq_hz
    psd = spec.psd if spec.freq_hz.size else spec.fft_power
    if freq.size == 0 or psd.size == 0:
        ax.text(
            0.5,
            0.5,
            "Too few samples for FFT",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return
    ax.plot(freq, psd, color=color, linestyle=ls, lw=lw, label=title)
    lo, hi = DEFAULT_GAIT_BAND_HZ
    ax.axvspan(lo, hi, color=ACCENT, alpha=0.08, label="Walking band")
    if spec.dominant_hz is not None:
        ax.axvline(
            float(spec.dominant_hz),
            color=ACCENT,
            ls="--",
            lw=1.15,
            label=f"f₀ {float(spec.dominant_hz):.2f} Hz",
        )
    peaks = report.spectrum_peaks_for(mode).get(channel_id) or []
    if peaks:
        ax.scatter(
            [float(f) for f, _p in peaks],
            [float(p) for _f, p in peaks],
            s=26,
            c=ACCENT,
            zorder=5,
            label="Peaks",
        )
    metrics: list[str] = []
    if spec.dominant_hz is not None:
        metrics.append(f"f₀ {float(spec.dominant_hz):.2f} Hz")
    else:
        metrics.append("f₀ unavailable")
    if spec.spectral_entropy is not None:
        metrics.append(f"H {float(spec.spectral_entropy):.2f}")
    energy = getattr(spec, "band_energy", None)
    if energy is not None and np.isfinite(float(energy)):
        metrics.append(f"E {float(energy):.3g}")
    if peaks:
        metrics.append(f"{len(peaks)} peak{'s' if len(peaks) != 1 else ''}")
    pair_name = _DELTA_PAIR.get(channel_id)
    if pair_name:
        cmp = report.freq_compare_for(mode).get(pair_name)
        if cmp is not None and cmp.delta_hz is not None:
            metrics.append(f"L–R Δf₀ {float(cmp.delta_hz):+.2f} Hz")
    if quality and quality != "computed":
        metrics.append(f"{quality} — not a classification")
    _metric_box(ax, metrics)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    style_chart_legend(ax, loc="upper right", fontsize=7, ncol=1, outside=False)


def render_four_gait_spectrum_figure(
    fig: Figure,
    axes: Sequence[Axes],
    report: FourGaitSignalsReport | None,
    mode: SignalMode,
) -> None:
    if report is None or not report.available:
        for ax in axes:
            _empty(ax, "Load a walking video to compute thigh and shank spectra.")
        return
    xlim, ylim = shared_spectrum_limits(report, mode)
    for ax, (cid, title) in zip(axes[:4], _PANEL_ORDER):
        draw_four_gait_spectrum(
            ax, report, mode, cid, title, xlim=xlim, ylim=ylim
        )
    unclip_figure_chrome(fig)


def draw_spectral_correlation_heatmap(
    ax: Axes,
    report: FourGaitSignalsReport,
    mode: SignalMode,
) -> None:
    cb = getattr(ax, "_sw_colorbar", None)
    if cb is not None:
        try:
            cb.remove()
        except Exception:
            pass
        ax._sw_colorbar = None
    _clear(ax)
    style_chart_title(ax, "4×4 spectral shape r — Pearson of unit-energy PSD", pad=6.0)
    corr = np.asarray(report.spectral_corr_for(mode), dtype=float)
    labels = list(FOUR_GAIT_SHORT_LABELS)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, origin="upper")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8, color=TEXT)
    ax.set_yticklabels(labels, fontsize=8, color=TEXT)
    n = corr.shape[0]
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            txt = "—" if not np.isfinite(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color=TEXT, fontsize=8)
            if (i, j) in _FOCUS_CELLS and i != j:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor=TEXT,
                        linewidth=1.6,
                    )
                )
    ax.set_xlabel("PSD shape j")
    ax.set_ylabel("PSD shape i")
    fig = ax.figure
    if fig is not None:
        try:
            bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            bar.set_label("shape r", color=TEXT, fontsize=8)
            bar.ax.tick_params(colors=TEXT, labelsize=8)
            ax._sw_colorbar = bar
        except Exception:
            pass


def _psd_xy(spec) -> tuple[np.ndarray, np.ndarray]:
    if spec is None:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if spec.freq_hz.size:
        return np.asarray(spec.freq_hz, dtype=float), np.asarray(spec.psd, dtype=float)
    return np.asarray(spec.fft_freq_hz, dtype=float), np.asarray(spec.fft_power, dtype=float)


def draw_psd_pair_overlay(
    ax: Axes,
    report: FourGaitSignalsReport,
    mode: SignalMode,
    pair_name: str,
    title: str,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    shape_r: float | None = None,
    delta_hz: float | None = None,
    msc_note: str | None = None,
) -> None:
    _clear(ax)
    style_chart_title(ax, f"PSD · {title}", pad=6.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (z²/Hz)")
    apply_chart_grid(ax)
    ax._sw_time_seek = False  # type: ignore[attr-defined]
    pair = next((p for p in COMPARE_PAIRS if p[0] == pair_name), None)
    if pair is None:
        ax.text(0.5, 0.5, "Unknown pair", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    _name, id_a, id_b, _lab = pair
    spectra = report.spectrum_for(mode)
    lo, hi = DEFAULT_GAIT_BAND_HZ
    ax.axvspan(lo, hi, color=ACCENT, alpha=0.08, label="Walking band")
    drawn = 0
    for cid in (id_a, id_b):
        spec = spectra.get(cid)
        freq, psd = _psd_xy(spec)
        label = FOUR_GAIT_SHORT_LABELS[FOUR_GAIT_CHANNEL_IDS.index(cid)]
        color, ls, lw = _SERIES_STYLE[cid]
        if freq.size and psd.size:
            ax.plot(freq, psd, color=color, linestyle=ls, lw=lw, label=label)
            drawn += 1
        if spec is not None and spec.dominant_hz is not None:
            ax.axvline(
                float(spec.dominant_hz),
                color=color,
                ls=":",
                lw=1.05,
                alpha=0.85,
            )
    if drawn == 0:
        ax.text(
            0.5,
            0.5,
            "Spectrum unavailable",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
        )
    metrics: list[str] = []
    if delta_hz is not None and np.isfinite(float(delta_hz)):
        metrics.append(f"Δf₀ {float(delta_hz):+.2f} Hz")
    if shape_r is not None and np.isfinite(float(shape_r)):
        metrics.append(f"shape r {float(shape_r):.2f}")
    if msc_note:
        metrics.append(msc_note)
    _metric_box(ax, metrics)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    style_chart_legend(ax, loc="upper right", fontsize=7, ncol=1, outside=False)


def draw_coherence(
    ax: Axes,
    report: FourGaitSignalsReport,
    mode: SignalMode,
    pair_name: str,
    title: str,
    *,
    xlim: tuple[float, float],
) -> None:
    _clear(ax)
    style_chart_title(ax, f"MSC · {title}", pad=6.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("γ²(f)")
    apply_chart_grid(ax)
    ax._sw_time_seek = False  # type: ignore[attr-defined]
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.05)
    lo, hi = DEFAULT_GAIT_BAND_HZ
    ax.axvspan(lo, hi, color=ACCENT, alpha=0.08)
    coh = report.coherence_for(mode).get(pair_name)
    if coh is None or not coh.sufficient or coh.freq_hz.size == 0:
        reason = "Coherence withheld — not enough Welch segments."
        if coh is not None:
            extra = [n for n in coh.notes if "segment" in n.lower() or "few" in n.lower()]
            if extra:
                reason = extra[0]
        ax.text(0.5, 0.5, reason, ha="center", va="center", color=MUTED, transform=ax.transAxes, wrap=True)
        return
    ax.plot(coh.freq_hz, coh.coherence, color=COM, lw=SERIES_LINE_WIDTH, label=title)
    if coh.peak_hz is not None and coh.peak_band is not None:
        ax.scatter(
            [float(coh.peak_hz)],
            [float(coh.peak_band)],
            s=28,
            c=ACCENT,
            zorder=5,
            label=f"peak {float(coh.peak_band):.2f} @ {float(coh.peak_hz):.2f} Hz",
        )
    style_chart_legend(ax, loc="upper right", fontsize=7, ncol=1, outside=False)


def build_compare_axes(fig: Figure) -> list[Axes]:
    fig.clear()
    gs = fig.add_gridspec(3, 2, height_ratios=(1.28, 1.0, 1.0))
    axes = [
        fig.add_subplot(gs[0, :]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]
    layout_four_gait_figure(fig, spectral=True)
    return axes


def _msc_overlay_note(report: FourGaitSignalsReport, mode: SignalMode, pair_name: str) -> str:
    coh = report.coherence_for(mode).get(pair_name)
    if coh is None or not coh.sufficient:
        return "MSC withheld — not enough Welch segments"
    if coh.mean_band is not None and np.isfinite(float(coh.mean_band)):
        return f"MSC mean γ² {float(coh.mean_band):.2f}"
    return "MSC available"


def render_four_gait_compare_figure(
    fig: Figure,
    axes: Sequence[Axes],
    report: FourGaitSignalsReport | None,
    mode: SignalMode,
) -> None:
    if report is None or not report.available:
        for ax in axes:
            _empty(ax, "Load a walking video to compare thigh and shank spectra.")
        return
    if not axes:
        return
    draw_spectral_correlation_heatmap(axes[0], report, mode)
    xlim, ylim = shared_spectrum_limits(report, mode)
    spec_corr = np.asarray(report.spectral_corr_for(mode), dtype=float)
    idx = {name: (i, j) for name, i, j, _lab in FOCUS_PAIRS}
    overlays = list(axes[1:5])
    for overlay, (name, _a, _b, title) in zip(overlays, COMPARE_PAIRS):
        i, j = idx.get(name, (-1, -1))
        shape_r = None
        if 0 <= i < spec_corr.shape[0] and 0 <= j < spec_corr.shape[1]:
            val = float(spec_corr[i, j])
            shape_r = val if np.isfinite(val) else None
        cmp = report.freq_compare_for(mode).get(name)
        draw_psd_pair_overlay(
            overlay,
            report,
            mode,
            name,
            title,
            xlim=xlim,
            ylim=ylim,
            shape_r=shape_r,
            delta_hz=None if cmp is None else cmp.delta_hz,
            msc_note=_msc_overlay_note(report, mode, name),
        )
    unclip_figure_chrome(fig)


def render_four_gait_figure(
    fig: Figure,
    axes: Sequence[Axes],
    report: FourGaitSignalsReport | None,
    mode: SignalMode,
    *,
    playhead: PlayheadState | None = None,
    combined: bool = False,
    lr_overlay: bool = False,
) -> None:
    if report is None or not report.available:
        for ax in axes:
            _empty(ax, "Load a walking video to compute thigh and shank angles.")
        return
    xlim, thigh_ylim, shank_ylim = pair_limits(report, mode)
    union_ylim = shared_limits(report, mode)[1]
    if lr_overlay:
        if len(axes) >= 1:
            draw_lr_overlay(
                axes[0],
                report,
                mode,
                pair="thigh",
                xlim=xlim,
                ylim=thigh_ylim,
                show_xlabel=False,
            )
        if len(axes) >= 2:
            draw_lr_overlay(
                axes[1],
                report,
                mode,
                pair="shank",
                xlim=xlim,
                ylim=shank_ylim,
                show_xlabel=not combined,
            )
        if combined and len(axes) >= 3:
            draw_combined(axes[2], report, mode, xlim=xlim, ylim=union_ylim)
    else:
        times = report.times_s
        series = report.series_for(mode)
        for idx, (ax, (cid, title)) in enumerate(zip(axes[:4], _PANEL_ORDER)):
            bottom = idx >= 2
            _draw_one_signal(
                ax,
                times,
                np.asarray(series.get(cid), dtype=float),
                channel_id=cid,
                title=title,
                xlim=xlim,
                ylim=thigh_ylim if idx < 2 else shank_ylim,
                ylabel=True,
                show_xlabel=bottom and not combined,
                show_xticklabels=bottom or combined,
            )
        if combined and len(axes) >= 5:
            draw_combined(axes[4], report, mode, xlim=xlim, ylim=union_ylim)
    if playhead is not None:
        apply_playhead_to_axes(list(axes), playhead)
    layout_four_gait_figure(fig, combined=combined, overlay=lr_overlay)
    unclip_figure_chrome(fig)
