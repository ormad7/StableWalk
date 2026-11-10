"""Export a research-validation pack from computed session outputs only."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from stablewalk.analysis.research_validation import ResearchValidationSnapshot
from stablewalk.analysis.spectral.features import FEATURE_DEFINITIONS

REPORT_SCHEMA_VERSION = "1.0"


def _status_line(status: str) -> str:
    labels = {
        "computed": "computed",
        "insufficient": "insufficient data",
        "unavailable": "not computed",
        "failed": "analysis failed",
    }
    return labels.get(status, status)


def format_research_report_markdown(snapshot: ResearchValidationSnapshot) -> str:
    lines = [
        "# StableWalk Research Validation Report",
        "",
        "This pack lists outputs that were **actually computed** for the current recording.",
        "Status is not an acceptance of a research requirement and is not a diagnosis.",
        "",
        f"- Schema: `{REPORT_SCHEMA_VERSION}`",
        f"- Source: `{snapshot.source or '—'}`",
        "",
        "## Input-video metadata",
        "",
    ]
    if snapshot.video_metadata:
        for key, value in snapshot.video_metadata.items():
            lines.append(f"- **{key}:** {value}")
    else:
        lines.append("- No video metadata was available.")
    lines += ["", "## Results", ""]
    for item in snapshot.items:
        lines.append(f"### {item.title}")
        lines.append("")
        lines.append(f"- **Status:** {_status_line(item.status)}")
        lines.append(f"- **Summary:** {item.summary}")
        if item.reason:
            lines.append(f"- **Failure / withhold reason:** {item.reason}")
        if item.units:
            unit_txt = ", ".join(f"{k} ({v})" for k, v in item.units.items())
            lines.append(f"- **Units:** {unit_txt}")
        details = dict(item.details or {})
        features = details.pop("features", None)
        moments = details.pop("moments", None)
        joints = details.pop("joints", None)
        scalars = details.pop("scalars", None)
        limitations = details.pop("limitations", None)
        fields = details.pop("fields", None)
        if details:
            lines.append("- **Computed fields:**")
            for key, value in details.items():
                if value in (None, "", [], {}):
                    continue
                lines.append(f"  - `{key}`: {value}")
        if joints:
            lines.append("- **Phase-normalized joints:**")
            for row in joints:
                lines.append(
                    "  - "
                    f"{row.get('name')}: ROM L {row.get('rom_left')} / R {row.get('rom_right')}; "
                    f"peak Δ {row.get('peak_difference')}; "
                    f"timing Δ {row.get('timing_difference_pct')} percentage points "
                    f"(circular; not Robinson SI); "
                    f"SI {row.get('si_percent')}; ASI {row.get('asi_percent')}; "
                    f"mean-curve ROM SI_norm {row.get('si_norm')} "
                    f"(max−min of HS–HS mean; N {row.get('n_left_cycles')}/{row.get('n_right_cycles')})"
                )
        if scalars:
            lines.append("- **Bilateral scalars:**")
            for row in scalars:
                lines.append(
                    "  - "
                    f"{row.get('display') or row.get('name')}: "
                    f"L {row.get('left')} / R {row.get('right')}; "
                    f"SI {row.get('si_percent')} (signed %); "
                    f"ASI {row.get('asi_percent')} (abs %); "
                    f"SI_norm {row.get('si_norm')} (0–1"
                    f"{'; P95−P5 ROM' if str(row.get('name') or '').endswith('_flexion_rom') else ''}); "
                    f"Δ {row.get('difference')} {row.get('difference_units')}; "
                    f"N {row.get('n_left_cycles')}/{row.get('n_right_cycles')}; "
                    f"quality {row.get('quality')}"
                )
        if features:
            lines.append("- **Spectral features (session):**")
            for row in features:
                mark = "computed" if row.get("available") else "not computed"
                lines.append(
                    f"  - {row.get('label') or row.get('name')}: {row.get('display')} [{mark}]"
                )
        if moments:
            lines.append("- **Detected moments:**")
            for row in moments:
                lines.append(
                    "  - "
                    f"{row.get('time_s')} s · frame {row.get('frame_index')} · "
                    f"score {row.get('score')} · conf {row.get('confidence')} · "
                    f"joints {', '.join(row.get('involved_joints') or [])}"
                )
        if fields:
            lines.append("- **Confidence fields:**")
            for key, row in fields.items():
                avail = "computed" if row.get("available") else "not computed"
                lines.append(f"  - {row.get('label') or key}: {row.get('value') or '—'} [{avail}]")
        if limitations:
            lines.append("- **Limitations:**")
            for line in limitations:
                lines.append(f"  - {line}")
        if item.definitions:
            lines.append("- **Definitions:**")
            for key, text in item.definitions.items():
                lines.append(f"  - `{key}`: {text}")
        lines.append("")
    lines += ["## Parameter definitions (spectral features)", ""]
    for key, text in FEATURE_DEFINITIONS.items():
        lines.append(f"- `{key}`: {text}")
    lines += ["", "## Limitations", ""]
    if snapshot.limitations:
        for line in snapshot.limitations:
            lines.append(f"- {line}")
    else:
        lines.append("- No additional limitation strings were attached.")
    lines += [
        "",
        "## Plots included",
        "",
    ]
    plots = []
    if "acf" in snapshot.plot_series:
        plots.append("`plots/acf.png` — normalized autocorrelation vs lag")
    if "spectrum" in snapshot.plot_series:
        plots.append("`plots/fft_psd.png` — FFT periodogram and Welch PSD")
    if "moment3d" in snapshot.plot_series:
        plots.append("`plots/wire3d_moment.png` — 16-joint wire at the strongest moment")
    if plots:
        lines.extend(f"- {p}" for p in plots)
    else:
        lines.append("- No plots: the underlying series were not successfully computed.")
    lines.append("")
    return "\n".join(lines)


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())


def render_research_plots(
    snapshot: ResearchValidationSnapshot,
    out_dir: Path,
    *,
    session: Any | None = None,
) -> dict[str, str]:
    """Write computed plot files. Skip any panel whose series were not computed."""
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.chart_style import apply_chart_panel_style

    written: dict[str, str] = {}
    acf = snapshot.plot_series.get("acf") or {}
    if acf.get("lags_s") and acf.get("rho"):
        fig = Figure(figsize=(6.4, 3.2), dpi=120)
        ax = fig.add_subplot(111)
        apply_chart_panel_style(ax)
        lags = acf["lags_s"]
        rho = [float("nan") if v is None else float(v) for v in acf["rho"]]
        ax.plot(lags, rho, color="#38bdf8", lw=1.6)
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("ρ")
        ax.set_title("Normalized autocorrelation")
        path = out_dir / "acf.png"
        _savefig(fig, path)
        written["acf"] = str(path)
    spec = snapshot.plot_series.get("spectrum") or {}
    if spec.get("fft_freq_hz") or spec.get("psd_freq_hz"):
        fig = Figure(figsize=(6.4, 3.2), dpi=120)
        ax = fig.add_subplot(111)
        apply_chart_panel_style(ax)
        if spec.get("fft_freq_hz") and spec.get("fft_power"):
            ax.plot(spec["fft_freq_hz"], spec["fft_power"], color="#94a3b8", lw=1.1, label="FFT")
        if spec.get("psd_freq_hz") and spec.get("psd"):
            ax.plot(spec["psd_freq_hz"], spec["psd"], color="#38bdf8", lw=1.6, label="Welch PSD")
        ax.set_xlim(0, 4.0)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (z²/Hz)")
        ax.set_title("FFT / PSD")
        ax.legend(frameon=False, fontsize=8)
        path = out_dir / "fft_psd.png"
        _savefig(fig, path)
        written["spectrum"] = str(path)
    moment = snapshot.plot_series.get("moment3d")
    if moment and session is not None:
        try:
            path = out_dir / "wire3d_moment.png"
            if _render_moment_wire(session, moment, path):
                written["moment3d"] = str(path)
        except Exception:
            pass
    return written


def _render_moment_wire(session: Any, moment: dict[str, Any], path: Path) -> bool:
    from matplotlib.figure import Figure

    from stablewalk.models.gait_wireframe import (
        SKELETON_DISPLAY_MODE_GAIT16,
        project_snapshot_to_gait_wire,
        session_body_scale,
    )
    from stablewalk.ui.asymmetric_moments_view import moment_highlight_joints
    from stablewalk.ui.viewers.gait_skeleton_renderer import (
        draw_gait_skeleton,
        setup_skeleton_axes,
    )

    recording = getattr(session, "gait_motion", None)
    snaps = list(getattr(recording, "snapshots", None) or [])
    if not snaps:
        return False
    frame_index = int(moment.get("frame_index") or 0)
    snap = next((s for s in snaps if int(s.frame_index) == frame_index), None)
    if snap is None:
        t = moment.get("time_s")
        if t is None:
            return False
        snap = min(snaps, key=lambda s: abs(float(s.time_s) - float(t)))
    scale = session_body_scale(recording)
    display = project_snapshot_to_gait_wire(snap, body_scale=scale)
    highlight = set()
    strongest = getattr(session, "_research_strongest_moment_obj", None)
    if strongest is None:
        report = getattr(session, "_gait_asymmetry_moments", None)
        moments = list(getattr(report, "moments", None) or [])
        strongest = moments[0] if moments else None
    if strongest is not None:
        highlight = moment_highlight_joints(strongest, SKELETON_DISPLAY_MODE_GAIT16)
    fig = Figure(figsize=(4.6, 4.6), dpi=120)
    ax = fig.add_subplot(111)
    setup_skeleton_axes(ax, display_mode=SKELETON_DISPLAY_MODE_GAIT16)
    draw_gait_skeleton(
        ax,
        display,
        clear=True,
        paused=True,
        lite_draw=False,
        highlight_joints=highlight or None,
        dim_unhighlighted=False,
        display_mode=SKELETON_DISPLAY_MODE_GAIT16,
        show_labels=False,
        show_legend=False,
        title="",
        show_ground_plane=True,
        show_com=False,
        show_support_polygon=False,
        show_contact_points=False,
        show_gait_direction=False,
    )
    ax.set_title(
        f"16-joint wire · t={float(getattr(snap, 'time_s', 0.0)):.2f}s · frame {int(snap.frame_index)}"
    )
    _savefig(fig, path)
    return path.is_file()


def export_research_report_zip(
    snapshot: ResearchValidationSnapshot,
    path: Path,
    *,
    session: Any | None = None,
) -> Path:
    """Write JSON, markdown, definitions, metadata, and computed plots to a zip."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot.to_dict()
    payload["schema_version"] = REPORT_SCHEMA_VERSION
    payload["parameter_definitions"] = dict(FEATURE_DEFINITIONS)
    markdown = format_research_report_markdown(snapshot)

    plot_files: dict[str, bytes] = {}
    tmp_parent = path.parent
    plot_dir = tmp_parent / f".{path.stem}_plots"
    try:
        written = render_research_plots(snapshot, plot_dir, session=session)
        for key, file_path in written.items():
            data = Path(file_path).read_bytes()
            name = {
                "acf": "plots/acf.png",
                "spectrum": "plots/fft_psd.png",
                "moment3d": "plots/wire3d_moment.png",
            }.get(key, f"plots/{key}.png")
            plot_files[name] = data
    finally:
        if plot_dir.exists():
            for child in plot_dir.glob("*"):
                try:
                    child.unlink()
                except OSError:
                    pass
            try:
                plot_dir.rmdir()
            except OSError:
                pass

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "research_validation.json",
            json.dumps(payload, indent=2),
        )
        zf.writestr("research_report.md", markdown)
        zf.writestr(
            "parameter_definitions.json",
            json.dumps(
                {
                    "spectral_features": dict(FEATURE_DEFINITIONS),
                    "units_by_item": {
                        item.key: dict(item.units) for item in snapshot.items if item.units
                    },
                },
                indent=2,
            ),
        )
        zf.writestr(
            "video_metadata.json",
            json.dumps(snapshot.video_metadata, indent=2),
        )
        zf.writestr(
            "failure_reasons.json",
            json.dumps(
                {
                    item.key: {
                        "status": item.status,
                        "reason": item.reason,
                        "summary": item.summary,
                    }
                    for item in snapshot.items
                    if item.status != "computed"
                },
                indent=2,
            ),
        )
        for name, data in plot_files.items():
            zf.writestr(name, data)
        if not plot_files:
            zf.writestr(
                "plots/README.txt",
                "No plots were written because the underlying series were not successfully computed.\n",
            )
    return path


__all__ = [
    "export_research_report_zip",
    "format_research_report_markdown",
    "render_research_plots",
]
