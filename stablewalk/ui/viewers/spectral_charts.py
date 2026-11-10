"""Matplotlib drawers for the Spectral Analysis workspace (display only)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from stablewalk.analysis.spectral.report import SpectralAnalysisReport
from stablewalk.ui.colors import ACCENT, COM, MUTED, PANEL, SIDE_LEFT, SIDE_RIGHT, TEXT, TEXT_SECONDARY
from stablewalk.ui.viewers.chart_playhead import PlayheadState, apply_playhead_to_axes
from stablewalk.ui.viewers.chart_style import (
    LINESTYLE_LEFT,
    LINESTYLE_RIGHT,
    SERIES_LINE_WIDTH,
    apply_chart_panel_style,
    style_chart_title,
    unclip_axis_chrome,
    unclip_figure_chrome,
)


def _legend_inside(ax: Axes, *, compact: bool = True) -> None:
    """Keep series keys readable without covering the axes title."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if not compact:
        from stablewalk.ui.viewers.chart_style import style_chart_legend

        style_chart_legend(ax, outside=True, fontsize=10, ncol=1)
        return
    ax.legend(
        loc="upper right",
        fontsize=7,
        frameon=True,
        fancybox=False,
        framealpha=0.88,
        edgecolor=MUTED,
        facecolor=PANEL,
        labelcolor=TEXT,
        borderaxespad=0.35,
        labelspacing=0.22,
        handlelength=1.35,
    )


def _short_label(channel_id: str) -> str:
    return (
        channel_id.replace("dof:", "")
        .replace("pos:", "")
        .replace("pelvis:", "")
        .replace("_", " ")
    )


def _matrix_tick_label(channel_id: str) -> str:
    """Compact heatmap ticks: 'left knee flexion' → 'L knee'."""
    label = _short_label(channel_id)
    label = label.replace("left ", "L ").replace("right ", "R ")
    for tail in (" flexion", "flexion", " :y", ":y", " :x", ":x", " :z", ":z"):
        label = label.replace(tail, "")
    return label.strip()


def _clear(ax: Axes) -> None:
    ax.clear()
    apply_chart_panel_style(ax)


def _note(ax: Axes, text: str) -> None:
    ax.text(
        0.99,
        0.02,
        text,
        transform=ax.transAxes,
        color=MUTED,
        fontsize=7,
        ha="right",
        va="bottom",
    )


def apply_spectral_axis_chrome(ax: Axes, *, compact: bool) -> None:
    """Shrink title/ticks on the 3×2 grid so labels do not collide."""
    title = ax.get_title()
    if title:
        artist = ax.set_title(
            title,
            color=TEXT,
            fontsize=11 if compact else 14,
            pad=6 if compact else 12,
        )
        try:
            artist.set_clip_on(False)
        except Exception:
            pass
    xlab = ax.xaxis.label
    ylab = ax.yaxis.label
    if xlab is not None:
        xlab.set_fontsize(9 if compact else 12)
        try:
            xlab.set_clip_on(False)
        except Exception:
            pass
    if ylab is not None:
        ylab.set_fontsize(9 if compact else 12)
        try:
            ylab.set_clip_on(False)
        except Exception:
            pass
    ax.tick_params(
        labelsize=7 if compact else 10,
        pad=1 if compact else 3,
        colors=TEXT_SECONDARY,
        labelcolor=TEXT_SECONDARY,
    )
    if xlab is not None:
        xlab.set_color(TEXT_SECONDARY)
    if ylab is not None:
        ylab.set_color(TEXT_SECONDARY)
    unclip_axis_chrome(ax)


SPECTRAL_SINGLE_MARGINS = dict(
    left=0.12,
    right=0.72,
    top=0.86,
    bottom=0.14,
    hspace=0.28,
    wspace=0.28,
)


def layout_spectral_figure(fig: Figure, *, single: bool = False) -> None:
    """Fractional margins that survive host resize. Avoid tight_layout."""
    try:
        fig.set_constrained_layout(False)
    except Exception:
        pass
    try:
        h_px = float(fig.get_size_inches()[1]) * float(fig.get_dpi() or 100.0)
        if single:
            # Room for a full title, axis labels, units, and an outside legend.
            fig.subplots_adjust(**SPECTRAL_SINGLE_MARGINS)
            return
        fig.subplots_adjust(
            left=0.10 if h_px >= 700 else 0.12,
            right=0.98,
            top=0.93 if h_px >= 700 else 0.90,
            bottom=0.11 if h_px >= 700 else 0.14,
            hspace=0.56 if h_px >= 700 else 0.68,
            wspace=0.30 if h_px >= 700 else 0.38,
        )
    except Exception:
        pass


def draw_spectral_plot(
    ax: Axes,
    plot_id: str,
    report: SpectralAnalysisReport,
    channel_id: str,
    *,
    compact: bool = False,
) -> None:
    """Draw one Spectral Analysis panel. ``plot_id`` matches SPECTRAL_PLOT_SPECS."""
    drawers = {
        "signal": lambda: draw_processed_signal(ax, report, channel_id, compact=compact),
        "acf": lambda: draw_autocorr(ax, report, channel_id, compact=compact),
        "psd": lambda: draw_psd(ax, report, channel_id, compact=compact),
        "lr_psd": lambda: draw_left_right_psd(ax, report, channel_id, compact=compact),
        "r0": lambda: draw_correlation_heatmap(ax, report, compact=compact),
        "eigen": lambda: draw_eigenvalues(ax, report, compact=compact),
    }
    drawer = drawers.get(str(plot_id))
    if drawer is None:
        _clear(ax)
        ax.text(0.5, 0.5, f"Unknown plot: {plot_id}", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    drawer()
    apply_spectral_axis_chrome(ax, compact=compact)
    # Time-seek only on the processed-signal axis. Shared-xlim zoom stays on
    # that axis in the 3×2 grid; expanded ACF/PSD windows may zoom themselves.
    is_signal = str(plot_id) == "signal"
    ax._sw_time_seek = is_signal  # type: ignore[attr-defined]
    if compact:
        ax._sw_nav_xlim = is_signal  # type: ignore[attr-defined]
    else:
        ax._sw_nav_xlim = str(plot_id) not in {"r0", "eigen"}  # type: ignore[attr-defined]


def draw_processed_signal(
    ax: Axes,
    report: SpectralAnalysisReport,
    channel_id: str,
    *,
    compact: bool = False,
) -> None:
    _clear(ax)
    series = report.bundle.get(channel_id)
    style_chart_title(ax, "Processed signal (time domain)")
    ax.set_xlabel("Time (s)")
    if series is None:
        ax.text(0.5, 0.5, "Channel unavailable", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    from stablewalk.pose.knee_convention import (
        KNEE_DETRENDED_AXIS,
        KNEE_FLEXION_AXIS,
        is_knee_flexion_dof,
    )

    knee = is_knee_flexion_dof(getattr(series.spec, "dof_id", None))
    units = str(getattr(series.spec, "units", "") or "signal")
    ax.plot(
        series.times_s,
        series.raw,
        color=MUTED,
        lw=1.0,
        alpha=0.55,
        label="Raw" if compact else ("Raw anatomical flexion" if knee else f"Raw ({units})"),
    )
    ax.plot(
        series.times_s,
        series.processed,
        color=ACCENT,
        lw=SERIES_LINE_WIDTH,
        label="Detrended" if compact else (
            "Linearly detrended" if knee else f"Linearly detrended ({units})"
        ),
    )
    # Processed here is still in native units. ACF / PSD use the separate z-score.
    if compact:
        ax.set_ylabel("Detrended flexion (deg)" if knee else f"{units} (detrended)")
    else:
        ax.set_ylabel(KNEE_DETRENDED_AXIS if knee else f"{units} (raw and detrended)")
    style_chart_title(
        ax,
        "Knee flexion — raw vs detrended (deg)"
        if knee
        else f"Processed signal — raw vs detrended ({units})",
    )
    if not compact:
        ax.text(
            0.01,
            0.02,
            (
                f"{KNEE_FLEXION_AXIS}. Temporal ACF and PSD use the z-scored copy, not this axis."
                if knee
                else f"Units: {units}. Temporal ACF and PSD use the z-scored copy, not this axis."
            ),
            transform=ax.transAxes,
            color=MUTED,
            fontsize=7,
            va="bottom",
        )
    _legend_inside(ax, compact=compact)
    ax.set_xlim(float(series.times_s[0]) if series.times_s.size else 0.0,
                float(series.times_s[-1]) if series.times_s.size else 1.0)


def draw_autocorr(
    ax: Axes,
    report: SpectralAnalysisReport,
    channel_id: str,
    *,
    compact: bool = False,
) -> None:
    _clear(ax)
    ac = report.autocorr.get(channel_id)
    style_chart_title(ax, "Temporal ACF ρ(τ)" if compact else "Temporal ACF ρ(τ) — not R(0)")
    ax.set_xlabel("Lag τ (s)")
    ax.set_ylabel("ρ(τ)" if compact else "ρ(τ) (z-scored series, dimensionless)")
    if not compact:
        _note(ax, "Time-lag of one channel — not the R(0) matrix")
    if ac is None or ac.lags_s.size == 0:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    ax.plot(ac.lags_s, ac.rho, color=COM, lw=SERIES_LINE_WIDTH)
    ax.axhline(0.0, color=MUTED, lw=0.8, alpha=0.6)
    peaks = []
    rho = getattr(ac, "rho", None)
    for peak in list(getattr(ac, "peaks", None) or []):
        idx = int(getattr(peak, "lag_samples", -1))
        if rho is None:
            continue
        from stablewalk.analysis.spectral.autocorrelation import is_qualifying_local_maximum

        if not is_qualifying_local_maximum(rho, idx):
            continue
        if abs(float(rho[idx]) - float(peak.value)) > 1e-9:
            continue
        peaks.append(peak)
    if peaks:
        ax.scatter(
            [float(p.lag_s) for p in peaks],
            [float(p.value) for p in peaks],
            s=28,
            c=ACCENT,
            zorder=5,
            label="ACF local maxima",
        )
    if ac.stride_period_s:
        ax.axvline(ac.stride_period_s, color=ACCENT, ls="--", lw=1.2, label="Stride lag")
    if ac.step_period_s:
        ax.axvline(ac.step_period_s, color=SIDE_RIGHT, ls=":", lw=1.2, label="Step lag")
    if peaks or ac.stride_period_s or ac.step_period_s:
        _legend_inside(ax, compact=compact)
    ax.set_ylim(-1.05, 1.05)


def draw_psd(
    ax: Axes,
    report: SpectralAnalysisReport,
    channel_id: str,
    *,
    compact: bool = False,
) -> None:
    _clear(ax)
    sp = report.spectra.get(channel_id)
    style_chart_title(ax, "Welch PSD" if compact else "Welch PSD — not R(0) eigenvalues")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (z²/Hz)" if compact else "PSD (z²/Hz of the z-scored series)")
    if not compact:
        _note(ax, "Frequency density of the z-scored series — not R(0) eigenvalues")
    if sp is None or sp.freq_hz.size == 0:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    ax.plot(sp.freq_hz, sp.psd, color=COM, lw=SERIES_LINE_WIDTH)
    lo, hi = report.config.gait_band_hz
    ax.axvspan(lo, hi, color=ACCENT, alpha=0.08, label="Walking band")
    if sp.dominant_hz:
        ax.axvline(sp.dominant_hz, color=ACCENT, ls="--", lw=1.2, label=f"f₀ {sp.dominant_hz:.2f} Hz")
    for idx, harm in enumerate(getattr(sp, "harmonic_hz", None) or []):
        ax.axvline(
            float(harm),
            color=MUTED,
            ls=":",
            lw=1.1,
            label=f"{idx + 2}f₀ {float(harm):.2f} Hz",
        )
    ax.set_xlim(0, min(8.0, float(sp.freq_hz[-1]) if sp.freq_hz.size else 8.0))
    _legend_inside(ax, compact=compact)


def draw_left_right_psd(
    ax: Axes,
    report: SpectralAnalysisReport,
    channel_id: str,
    *,
    compact: bool = False,
) -> None:
    _clear(ax)
    style_chart_title(ax, "Left vs right Welch PSD")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (z²/Hz)" if compact else "PSD (z²/Hz of the z-scored series)")
    if not compact:
        _note(ax, "Bilateral frequency density — not R(0)")
    pair = None
    for item in report.lr_compare:
        if channel_id in (item.left_id, item.right_id, item.pair_id):
            pair = item
            break
    if pair is None and report.lr_compare:
        pair = report.lr_compare[0]
    if pair is None:
        ax.text(0.5, 0.5, "No left/right pair", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    sl = report.spectra.get(pair.left_id)
    sr = report.spectra.get(pair.right_id)
    if sl is not None and sl.freq_hz.size:
        ax.plot(sl.freq_hz, sl.psd, color=SIDE_LEFT, lw=SERIES_LINE_WIDTH, ls=LINESTYLE_LEFT, label=_short_label(pair.left_id))
    if sr is not None and sr.freq_hz.size:
        ax.plot(sr.freq_hz, sr.psd, color=SIDE_RIGHT, lw=SERIES_LINE_WIDTH, ls=LINESTYLE_RIGHT, label=_short_label(pair.right_id))
    lo, hi = report.config.gait_band_hz
    ax.axvspan(lo, hi, color=ACCENT, alpha=0.08)
    _legend_inside(ax, compact=compact)
    xmax = 8.0
    for spec in (sl, sr):
        if spec is not None and spec.freq_hz.size:
            xmax = min(8.0, float(spec.freq_hz[-1]))
    ax.set_xlim(0, xmax)


def draw_correlation_heatmap(ax: Axes, report: SpectralAnalysisReport, *, compact: bool = False) -> None:
    cb = getattr(ax, "_sw_colorbar", None)
    if cb is not None:
        try:
            cb.remove()
        except Exception:
            pass
        ax._sw_colorbar = None
    _clear(ax)
    style_chart_title(
        ax,
        "R(0) inter-joint correlation" if compact else "R(0) inter-joint correlation (same-time, not ACF)",
    )
    mats = report.matrices
    if mats is None or mats.r0 is None or not mats.channel_ids:
        ax.text(0.5, 0.5, "Matrix unavailable", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    corr = np.asarray(mats.r0.correlation, dtype=float)
    labels = [_matrix_tick_label(c) for c in mats.channel_ids]
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, origin="upper")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    tick_fs = 6.5 if compact else 9
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=tick_fs, color=TEXT)
    ax.set_yticklabels(labels, fontsize=tick_fs, color=TEXT)
    if not compact:
        _note(ax, "Same-time r — not temporal ACF")
    ax.set_xlabel("Joint / DOF")
    ax.set_ylabel("Joint / DOF")
    fig = ax.figure
    if fig is not None and getattr(ax, "_sw_colorbar", None) is None:
        try:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("r (dimensionless)", color=TEXT, fontsize=8)
            cb.ax.tick_params(colors=TEXT, labelsize=8)
            ax._sw_colorbar = cb
        except Exception:
            pass


def draw_eigenvalues(ax: Axes, report: SpectralAnalysisReport, *, compact: bool = False) -> None:
    _clear(ax)
    style_chart_title(
        ax,
        "Eigenvalues of R(0)" if compact else "Eigenvalues of R(0) (spatial modes, not PSD)",
    )
    eig = report.eigen.get("correlation") or report.eigen.get("covariance")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("λ of R(0)" if compact else "Eigenvalue λ of R(0) (dimensionless)")
    if not compact:
        _note(ax, "Spatial modes of R(0) — not PSD")
    if eig is None or eig.eigenvalues.size == 0:
        ax.text(0.5, 0.5, "Eigenvalues unavailable", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    idx = np.arange(1, eig.eigenvalues.size + 1)
    ax.bar(idx, eig.eigenvalues, color=COM, width=0.72)
    ax.set_xticks(idx)


def render_spectral_figure(
    fig: Figure,
    axes: Sequence[Axes],
    report: SpectralAnalysisReport | None,
    channel_id: str,
    *,
    playhead: PlayheadState | None = None,
) -> None:
    ax_sig, ax_acf, ax_psd, ax_lr, ax_heat, ax_eig = axes
    if report is None or not report.available:
        for ax in axes:
            _clear(ax)
            ax.text(
                0.5,
                0.5,
                "Load a walking video to compute spectral analysis.",
                ha="center",
                va="center",
                color=MUTED,
                transform=ax.transAxes,
                wrap=True,
            )
        return
    plot_ids = ("signal", "acf", "psd", "lr_psd", "r0", "eigen")
    for ax, plot_id in zip(axes, plot_ids, strict=False):
        draw_spectral_plot(ax, plot_id, report, channel_id, compact=True)
    if playhead is not None:
        apply_playhead_to_axes([ax_sig], playhead)
    layout_spectral_figure(fig, single=False)
    unclip_figure_chrome(fig)
