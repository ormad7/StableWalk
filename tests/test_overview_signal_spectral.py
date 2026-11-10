"""Overview Signal & Spectral Analysis panel — mapping, charts, layout."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from stablewalk.analysis.spectral.autocorrelation import AutocorrPeak
from stablewalk.ui.tk.overview_signal_spectral import (
    axis_enabled,
    resolve_signal_channels,
)
from stablewalk.ui.viewers.overview_signal_spectral_charts import (
    build_signal_spectral_axes,
    layout_signal_spectral_figure,
    render_overview_signal_spectral,
)
from stablewalk.ui.viewers.spectral_charts import draw_autocorr, draw_psd


ROOT = Path(__file__).resolve().parents[1]


def _fake_report(channel_id: str = "dof:left_knee_flexion"):
    t = np.linspace(0.0, 4.0, 201)
    y = np.sin(2.0 * np.pi * 1.0 * t)
    series = SimpleNamespace(
        times_s=t,
        raw=y,
        processed=y,
        spec=SimpleNamespace(units="deg"),
    )
    lags = np.linspace(0.0, 2.0, 41)
    rho = np.linspace(1.0, 0.0, 41)
    rho[19] = 0.40
    rho[20] = 0.82
    rho[21] = 0.38
    ac = SimpleNamespace(
        lags_s=lags,
        rho=rho,
        peaks=[AutocorrPeak(lag_samples=20, lag_s=1.0, value=0.82, prominence=0.4)],
        stride_period_s=1.0,
        step_period_s=0.5,
    )
    sp = SimpleNamespace(
        freq_hz=np.linspace(0.0, 8.0, 65),
        psd=np.ones(65),
        dominant_hz=1.0,
        harmonic_hz=[2.0],
    )
    return SimpleNamespace(
        available=True,
        bundle=SimpleNamespace(get=lambda cid, _s=series: _s if cid else None),
        autocorr={channel_id: ac},
        spectra={channel_id: sp},
        config=SimpleNamespace(gait_band_hz=(0.4, 4.0)),
    )


def test_resolve_knee_angle_maps_to_flexion_dofs() -> None:
    mapping = resolve_signal_channels("knee", "angle", "Z")
    assert mapping["primary_id"] == "dof:left_knee_flexion"
    assert mapping["left_id"] == "dof:left_knee_flexion"
    assert mapping["right_id"] == "dof:right_knee_flexion"
    assert mapping["units"] == "deg"
    assert mapping["note"] is None
    assert not axis_enabled("angle")


def test_resolve_height_forces_vertical_axis() -> None:
    mapping = resolve_signal_channels("hip", "height", "X")
    assert mapping["left_id"] == "pos:left_hip:y"
    assert mapping["right_id"] == "pos:right_hip:y"
    assert mapping["axis"] == "y"


def test_resolve_position_keeps_selected_axis() -> None:
    mapping = resolve_signal_channels("ankle", "position", "Z")
    assert mapping["left_id"] == "pos:left_ankle:z"
    assert mapping["right_id"] == "pos:right_ankle:z"
    assert axis_enabled("position")


def test_resolve_heel_angle_is_unavailable() -> None:
    mapping = resolve_signal_channels("heel", "angle", "Y")
    assert mapping["primary_id"] is None
    assert mapping["note"]


def test_resolve_foot_ground_uses_clearance_not_pelvis() -> None:
    mapping = resolve_signal_channels("knee", "foot_ground", "Y")
    assert mapping["kind"] == "clearance"
    assert mapping["left_id"] == "clearance:left"
    pelvis = resolve_signal_channels("pelvis", "foot_ground", "Y")
    assert pelvis["note"]
    assert pelvis["kind"] == "spectral"


def test_resolve_pelvis_angle_and_root() -> None:
    yaw = resolve_signal_channels("pelvis", "angle", "Y")
    assert yaw["left_id"] == "pelvis:yaw_deg"
    assert yaw["right_id"] is None
    root = resolve_signal_channels("pelvis", "position", "X")
    assert root["left_id"] == "pelvis:root:x"


def test_gridspec_places_asymmetry_across_two_columns() -> None:
    fig = Figure(figsize=(10.0, 4.0), dpi=100)
    axes = build_signal_spectral_axes(fig)
    assert len(axes) == 5
    spec = axes[-1].get_subplotspec()
    assert spec.colspan.start == 1
    assert spec.colspan.stop == 3


def test_time_axes_opt_in_to_seek_acf_and_psd_do_not() -> None:
    fig = Figure(figsize=(10.0, 4.0), dpi=100)
    axes = build_signal_spectral_axes(fig)
    cid = "dof:left_knee_flexion"
    times = np.linspace(0.0, 4.0, 50)
    left = np.sin(times)
    right = np.cos(times)
    moments = [SimpleNamespace(time_s=1.25)]
    render_overview_signal_spectral(
        fig,
        axes,
        report=_fake_report(cid),
        channel_id=cid,
        times_lr=times,
        left_vals=left,
        right_vals=right,
        lr_ylabel="deg",
        asym_times=times,
        asym_scores=list(np.linspace(10.0, 40.0, times.size)),
        moments=moments,
    )
    ax_sig, ax_acf, ax_psd, ax_lr, ax_asym = axes
    assert ax_sig._sw_time_seek is True
    assert ax_lr._sw_time_seek is True
    assert ax_asym._sw_time_seek is True
    assert ax_acf._sw_time_seek is False
    assert ax_psd._sw_time_seek is False
    assert ax_sig.get_title() == "Joint signal"
    assert ax_acf.get_title() == "Autocorrelation"
    assert ax_psd.get_title() == "Power spectrum"


def test_layout_uses_subplots_adjust_not_tight_layout() -> None:
    src = (
        ROOT
        / "stablewalk"
        / "ui"
        / "viewers"
        / "overview_signal_spectral_charts.py"
    ).read_text(encoding="utf-8")
    assert "fig.subplots_adjust(" in src
    assert "tight_layout" not in src
    fig = Figure(figsize=(10.0, 3.0), dpi=100)
    build_signal_spectral_axes(fig)
    layout_signal_spectral_figure(fig)
    assert abs(float(fig.subplotpars.left) - 0.07) < 1e-6


def test_autocorr_marks_gait_cycle_peaks() -> None:
    fig = Figure(figsize=(4.0, 2.5), dpi=100)
    ax = fig.add_subplot(111)
    draw_autocorr(ax, _fake_report(), "dof:left_knee_flexion")
    _handles, labels = ax.get_legend_handles_labels()
    assert "ACF local maxima" in labels
    assert any("Stride" in lab for lab in labels)


def test_autocorr_does_not_mark_sampled_non_maxima() -> None:
    fig = Figure(figsize=(4.0, 2.5), dpi=100)
    ax = fig.add_subplot(111)
    report = _fake_report()
    ac = report.autocorr["dof:left_knee_flexion"]
    ac.peaks = [AutocorrPeak(lag_samples=10, lag_s=0.5, value=float(ac.rho[10]), prominence=0.1)]
    draw_autocorr(ax, report, "dof:left_knee_flexion")
    _handles, labels = ax.get_legend_handles_labels()
    assert "ACF local maxima" not in labels


def test_psd_marks_dominant_frequency() -> None:
    fig = Figure(figsize=(4.0, 2.5), dpi=100)
    ax = fig.add_subplot(111)
    draw_psd(ax, _fake_report(), "dof:left_knee_flexion")
    _handles, labels = ax.get_legend_handles_labels()
    assert any("f₀" in lab for lab in labels)


def test_overview_hosts_panel_not_a_new_tab() -> None:
    layout = (ROOT / "stablewalk" / "ui" / "tk" / "dashboard_layout.py").read_text(
        encoding="utf-8"
    )
    view = (
        ROOT / "stablewalk" / "ui" / "tk" / "dashboard_overview_view_mode.py"
    ).read_text(encoding="utf-8")
    notebook = (ROOT / "stablewalk" / "ui" / "tk" / "dashboard_notebook.py").read_text(
        encoding="utf-8"
    )
    assert "build_overview_signal_spectral_panel" in layout
    assert "Signal & Spectral" in view
    assert "Hide Signal & Spectral" in view
    assert "notebook.add" in notebook
    assert "Signal & Spectral" not in notebook
    assert "TAB_SIGNAL" not in notebook
