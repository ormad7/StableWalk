"""
StableWalk — Streamlit gait analysis UI.

Run:  streamlit run app.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk.advanced.comparison import compare_gait_sessions
from stablewalk.advanced.pipeline import AdvancedGaitReport
from stablewalk.force_analysis import GRFAnalyzer
from stablewalk.gait_events import analyze_gait_sequence
from stablewalk.gait_metrics import GaitMetricsResult
from stablewalk.models.pose_data import PoseSequence
from stablewalk.contact_detection import ContactDetector
from stablewalk.services.video_analysis import analyze_video_file
from stablewalk.stability_analysis import ScorePenalty, StabilityReport

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="StableWalk",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded",
)

DASHBOARD_CSS = """
<style>
.block-container { padding-top: 1.25rem; max-width: 1280px; }
h1 { font-weight: 600; letter-spacing: -0.02em; margin-bottom: 0.15rem; }
.subtitle { color: #64748b; font-size: 1rem; margin-bottom: 1rem; }
.section-title {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #64748b; margin: 0 0 0.5rem 0;
}
.dash-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.score-hero {
    text-align: center;
    padding: 1.35rem 1rem 1.1rem;
    border-radius: 14px;
    border: 2px solid var(--score-border, #e2e8f0);
    background: linear-gradient(180deg, var(--score-bg-top, #f8fafc) 0%, #fff 100%);
}
.score-hero .score-value {
    font-size: 3.4rem; font-weight: 800; line-height: 1;
    color: var(--score-color, #0f172a); letter-spacing: -0.03em;
}
.score-hero .score-denom {
    font-size: 1.1rem; color: #64748b; font-weight: 500;
}
.score-hero .score-status {
    display: inline-block;
    margin-top: 0.65rem;
    padding: 0.28rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    background: var(--badge-bg, #f1f5f9);
    color: var(--badge-fg, #334155);
}
.penalty-pill {
    font-size: 0.78rem; color: #64748b; margin-top: 0.35rem;
}
.issue-card {
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.45rem;
    font-size: 0.88rem;
    line-height: 1.4;
    border-left: 4px solid var(--issue-accent, #f59e0b);
    background: var(--issue-bg, #fffbeb);
}
.issue-card strong { color: #0f172a; }
.issue-ok {
    border-left-color: #22c55e;
    background: #f0fdf4;
    color: #166534;
}
.explain-box {
    font-size: 0.88rem;
    line-height: 1.55;
    color: #334155;
    white-space: pre-wrap;
    max-height: 420px;
    overflow-y: auto;
}
div[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
}
</style>
"""

st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

ISSUE_LABELS: dict[str, str] = {
    "asymmetric_legs": "Asymmetric leg loading or timing",
    "asymmetric_stance_duration": "Unequal stance duration between legs",
    "large_joint_angle_asymmetry": "Large knee/hip angle difference left vs right",
    "irregular_step_timing": "Irregular step timing (high variability)",
    "irregular_stride_length": "Inconsistent stride length",
    "excessive_com_lateral_sway": "Excessive side-to-side body sway",
    "abnormal_com_vertical_motion": "Abnormal vertical center-of-mass motion",
    "asymmetric_ground_forces": "Asymmetric estimated ground reaction forces",
    "atypical_force_pattern": "Atypical vertical force pattern (double-peak)",
    "slow_cadence": "Cadence slower than typical walking",
    "fast_cadence": "Cadence faster than typical walking",
    "insufficient_frames": "Too few analyzed frames for reliable scoring",
}


@dataclass
class AnalysisBundle:
    sequence: PoseSequence
    overlay_path: Path
    advanced: AdvancedGaitReport
    stability: StabilityReport
    label: str
    score: float
    status: str
    explanation: str
    abnormal_patterns: list[str]
    penalties: list[ScorePenalty]
    warnings: list[str]
    cadence_spm: float | None
    symmetry_pct: float
    stride_m: float | None
    step_cv: float | None
    stride_cv: float | None
    grf: object
    metrics: object
    advanced_metrics: GaitMetricsResult | None


def _frame_time(sequence: PoseSequence, frame_index: int) -> float:
    for f in sequence.frames:
        if f.frame_index == frame_index:
            return f.timestamp_s
    return frame_index / max(sequence.fps, 1e-6)


def build_grf_figure(grf) -> plt.Figure:
    """Left vs right vertical GRF with labeled axes."""
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    GRFAnalyzer(body_mass_kg=grf.body_mass_kg).plot_grf(
        grf,
        ax=ax,
        show_bw=True,
        title="Estimated vertical ground reaction force",
    )
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Vertical force (body weights, BW)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


def build_step_timing_figure(sequence: PoseSequence) -> plt.Figure:
    events, _ = analyze_gait_sequence(sequence.frames)
    duration = max(
        (f.timestamp_s for f in sequence.frames if f.detected),
        default=1.0,
    )
    duration = max(duration, 0.1)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 3.2), sharex=True)
    colors = {"left": "#3b82f6", "right": "#ef4444"}
    markers = {"heel_strike": "|", "toe_off": "v"}
    names = {"heel_strike": "Heel strike", "toe_off": "Toe-off"}

    for ax, side in zip(axes, ("left", "right")):
        side_events = [e for e in events if e.side == side]
        for etype in ("heel_strike", "toe_off"):
            times = [
                _frame_time(sequence, e.frame_index)
                for e in side_events
                if e.event_type == etype
            ]
            if times:
                ax.scatter(
                    times,
                    [1.0] * len(times),
                    marker=markers[etype],
                    s=100,
                    color=colors[side],
                    label=names[etype],
                    linewidths=2,
                )
        for f in sequence.frames:
            if not f.detected or not f.gait_phase:
                continue
            phase = f.gait_phase.get(side, "unknown")
            y = 0.35 if phase == "stance" else 0.15
            ax.scatter(f.timestamp_s, y, s=6, alpha=0.3, color=colors[side])

        ax.set_ylim(0, 1.3)
        ax.set_yticks([])
        ax.set_ylabel(side.capitalize(), fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_xlim(0, duration)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Step timing", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def run_analysis(
    video_path: Path,
    *,
    max_frames: int | None,
    body_mass_kg: float,
    work_dir: Path,
) -> AnalysisBundle:
    result = analyze_video_file(
        video_path,
        work_dir=work_dir,
        body_mass_kg=body_mass_kg,
        max_frames=max_frames,
        source_id=video_path.name,
    )
    advanced = result.advanced
    gait = advanced.gait
    stab = gait.stability
    report = gait
    overlay = result.overlay_path or work_dir / "overlay.mp4"
    adv_m = report.advanced_metrics

    step_cv = None
    stride_cv = None
    if stab.metrics:
        step_cv = stab.metrics.step_time_variability
        stride_cv = stab.metrics.stride_length_variability
    if adv_m and adv_m.step_timing_cv is not None:
        step_cv = adv_m.step_timing_cv

    return AnalysisBundle(
        sequence=result.sequence,
        overlay_path=overlay,
        advanced=advanced,
        stability=stab,
        label=stab.label,
        score=stab.score,
        status=stab.status or ("Stable" if stab.score >= 60 else "Unstable"),
        explanation=stab.explanation,
        abnormal_patterns=list(stab.abnormal_patterns),
        penalties=list(stab.penalties),
        warnings=list(report.warnings),
        cadence_spm=report.metrics.cadence_steps_per_min,
        symmetry_pct=stab.symmetry_score * 100.0,
        stride_m=report.metrics.stride_length_m,
        step_cv=step_cv,
        stride_cv=stride_cv,
        grf=report.grf,
        metrics=report.metrics,
        advanced_metrics=adv_m,
    )


def _score_theme(score: float, label: str) -> dict[str, str]:
    """Green / yellow / red palette from score and label."""
    if label in ("—", "-", ""):
        return {
            "color": "#64748b",
            "border": "#e2e8f0",
            "bg_top": "#f8fafc",
            "badge_bg": "#f1f5f9",
            "badge_fg": "#475569",
        }
    if label == "Borderline" or (45 <= score < 60):
        return {
            "color": "#b45309",
            "border": "#fcd34d",
            "bg_top": "#fffbeb",
            "badge_bg": "#fef3c7",
            "badge_fg": "#92400e",
        }
    if score >= 60 or label == "Stable":
        return {
            "color": "#15803d",
            "border": "#86efac",
            "bg_top": "#f0fdf4",
            "badge_bg": "#dcfce7",
            "badge_fg": "#166534",
        }
    return {
        "color": "#b91c1c",
        "border": "#fca5a5",
        "bg_top": "#fef2f2",
        "badge_bg": "#fee2e2",
        "badge_fg": "#991b1b",
    }


def render_stability_score(score: float, label: str, status: str, penalties: list[ScorePenalty]) -> None:
    theme = _score_theme(score, label)
    total_pen = sum(p.points for p in penalties)
    penalty_line = (
        f'<div class="penalty-pill">−{total_pen:.0f} from base 100</div>'
        if penalties
        else '<div class="penalty-pill">No penalties applied</div>'
    )
    score_display = f"{score:.0f}" if label not in ("—", "-") else "—"
    st.markdown('<p class="section-title">Stability score</p>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="score-hero" style="
            --score-color: {theme['color']};
            --score-border: {theme['border']};
            --score-bg-top: {theme['bg_top']};
            --badge-bg: {theme['badge_bg']};
            --badge-fg: {theme['badge_fg']};
        ">
            <div class="score-value">{score_display}<span class="score-denom"> / 100</span></div>
            <span class="score-status">{status}</span>
            {penalty_line}
            <div style="margin-top:0.5rem;font-size:0.85rem;color:#64748b;">
                Classification: <strong style="color:{theme['color']};">{label}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_status(value: float | None, *, good_max: float | None = None, good_min: float | None = None) -> str:
    if value is None:
        return "—"
    if good_max is not None and value <= good_max:
        return "✓ Good"
    if good_min is not None and value >= good_min:
        return "✓ Good"
    if good_max is not None and value > good_max * 1.35:
        return "⚠ High"
    if good_min is not None and value < good_min:
        return "⚠ Low"
    return "○ Monitor"


def build_gait_metrics_rows(bundle: AnalysisBundle) -> list[dict[str, str]]:
    sym = bundle.symmetry_pct
    cad = bundle.cadence_spm
    stride = bundle.stride_m
    stride_unit = " m"
    if stride is None and bundle.metrics.stride_length_normalized is not None:
        stride = bundle.metrics.stride_length_normalized
        stride_unit = " (norm.)"

    step_cv = bundle.step_cv
    stride_cv = bundle.stride_cv

    rows = [
        {
            "Metric": "Symmetry",
            "Value": f"{sym:.0f}%",
            "Assessment": _metric_status(sym / 100.0, good_min=0.70),
            "Notes": "Left/right balance (timing, stance, forces)",
        },
        {
            "Metric": "Cadence",
            "Value": f"{cad:.0f} spm" if cad else "n/a",
            "Assessment": _metric_status(cad, good_min=90, good_max=130) if cad else "—",
            "Notes": "Steps per minute",
        },
        {
            "Metric": "Stride length",
            "Value": (f"{stride:.2f}{stride_unit}" if stride is not None else "n/a"),
            "Assessment": "—" if stride is None else "○ Monitor",
            "Notes": "Mean step length (estimated)",
        },
        {
            "Metric": "Step variability (CV)",
            "Value": f"{step_cv:.2f}" if step_cv is not None else "n/a",
            "Assessment": _metric_status(step_cv, good_max=0.18),
            "Notes": "Lower = more regular footfalls",
        },
    ]
    if stride_cv is not None:
        rows.append({
            "Metric": "Stride variability (CV)",
            "Value": f"{stride_cv:.2f}",
            "Assessment": _metric_status(stride_cv, good_max=0.20),
            "Notes": "Consistency of stride length",
        })
    sm = bundle.stability.metrics
    if sm and sm.grf_symmetry is not None:
        rows.append({
            "Metric": "GRF symmetry",
            "Value": f"{sm.grf_symmetry:.0%}",
            "Assessment": _metric_status(sm.grf_symmetry, good_min=0.65),
            "Notes": "Peak force balance L vs R",
        })
    return rows


def render_gait_metrics_table(bundle: AnalysisBundle) -> None:
    st.markdown('<p class="section-title">Gait metrics</p>', unsafe_allow_html=True)
    rows = build_gait_metrics_rows(bundle)
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="small"),
            "Assessment": st.column_config.TextColumn("Status", width="small"),
            "Notes": st.column_config.TextColumn("Description", width="large"),
        },
    )


def render_detected_issues(bundle: AnalysisBundle) -> None:
    st.markdown('<p class="section-title">Detected issues</p>', unsafe_allow_html=True)

    items: list[tuple[str, str, str]] = []  # title, detail, severity
    seen: set[str] = set()

    for p in bundle.penalties:
        key = p.rule_id
        if key in seen:
            continue
        seen.add(key)
        items.append(
            (p.reason, f"Penalty −{p.points:.0f} points", "warning"),
        )

    for pattern in bundle.abnormal_patterns:
        title = ISSUE_LABELS.get(pattern, pattern.replace("_", " ").capitalize())
        if title not in {i[0] for i in items}:
            items.append((title, "Flagged by stability rules", "warning"))

    for w in bundle.warnings:
        items.append((w, "Analysis warning", "alert"))

    adv = bundle.advanced
    if adv.anomaly.triggered_rules:
        for rule in adv.anomaly.triggered_rules[:4]:
            items.append((rule.replace("_", " "), f"Anomaly ({adv.anomaly.severity})", "alert"))

    if not items:
        st.markdown(
            '<div class="issue-card issue-ok"><strong>No significant issues detected</strong>'
            "<br>Metrics and penalties are within expected ranges for this clip.</div>",
            unsafe_allow_html=True,
        )
        return

    styles = {
        "warning": ("#fff7ed", "#fed7aa", "#ea580c"),
        "alert": ("#fef2f2", "#fecaca", "#dc2626"),
    }
    for title, detail, kind in items:
        bg, border, accent = styles.get(kind, styles["warning"])
        st.markdown(
            f'<div class="issue-card" style="background:{bg};border:1px solid {border};'
            f'--issue-accent:{accent};"><strong>{title}</strong><br>'
            f'<span style="color:#64748b;font-size:0.82rem;">{detail}</span></div>',
            unsafe_allow_html=True,
        )


def render_explanation(bundle: AnalysisBundle) -> None:
    st.markdown('<p class="section-title">Explanation</p>', unsafe_allow_html=True)
    text = bundle.explanation.strip() if bundle.explanation else "No explanation available."
    st.markdown(f'<div class="explain-box dash-panel">{_escape_html(text)}</div>', unsafe_allow_html=True)
    with st.expander("Scoring methodology"):
        st.markdown(bundle.stability.scoring_notes)
        st.markdown("---")
        from stablewalk.stability_analysis import THRESHOLD_RATIONALE

        st.markdown(THRESHOLD_RATIONALE)


def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def _class_badge(label: str) -> str:
    colors = {"normal": "#16a34a", "borderline": "#d97706", "abnormal": "#dc2626"}
    c = colors.get(label, "#64748b")
    return (
        f'<span style="background:{c}22;color:{c};padding:0.2rem 0.6rem;'
        f'border-radius:6px;font-weight:600;">{label.upper()}</span>'
    )


def render_analysis_tab(bundle: AnalysisBundle) -> None:
    adv = bundle.advanced

    # --- Row 1: Video + hero score ---
    vid_col, dash_col = st.columns([1.65, 1], gap="large")
    with vid_col:
        st.markdown('<p class="section-title">Processed walk</p>', unsafe_allow_html=True)
        if bundle.overlay_path.is_file():
            st.video(str(bundle.overlay_path))
        n_det = sum(1 for f in bundle.sequence.frames if f.detected)
        st.caption(f"{n_det} / {len(bundle.sequence.frames)} frames with pose detected")
    with dash_col:
        render_stability_score(
            bundle.score, bundle.label, bundle.status, bundle.penalties
        )
        st.markdown(
            f'<div style="margin-top:0.75rem;">Gait class {_class_badge(adv.classification.label)} '
            f'<span style="color:#64748b;">({adv.classification.confidence:.0%})</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Anomaly screen: **{adv.anomaly.severity}** "
            f"(score {adv.anomaly.anomaly_score:.2f})"
        )

    st.divider()

    # --- Row 2: GRF + metrics table ---
    grf_col, metrics_col = st.columns([1.45, 1], gap="large")
    with grf_col:
        st.markdown('<p class="section-title">Ground reaction force</p>', unsafe_allow_html=True)
        if bundle.grf:
            fig = build_grf_figure(bundle.grf)
            st.pyplot(fig, use_container_width=True, clear_figure=True)
            plt.close(fig)
            st.caption(
                "Left (blue) vs right (red) vertical force in body weights (BW). "
                "Estimated from pose acceleration — not measured on a force plate."
            )
        else:
            st.info("GRF data unavailable for this session.")
    with metrics_col:
        render_gait_metrics_table(bundle)

    st.divider()

    # --- Row 3: Issues + explanation ---
    issues_col, explain_col = st.columns([1, 1.25], gap="large")
    with issues_col:
        render_detected_issues(bundle)
    with explain_col:
        render_explanation(bundle)

    # --- Row 4: Supplementary charts ---
    with st.expander("Contact timeline & step timing", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            contact = ContactDetector().detect(bundle.sequence)
            fig_c, _ = ContactDetector().plot_contact_timeline(contact)
            st.pyplot(fig_c, use_container_width=True, clear_figure=True)
            plt.close(fig_c)
            st.caption(
                f"Touch-downs L={len(contact.left_foot_contacts)} "
                f"R={len(contact.right_foot_contacts)}"
            )
        with c2:
            fig = build_step_timing_figure(bundle.sequence)
            st.pyplot(fig, use_container_width=True, clear_figure=True)
            plt.close(fig)


def render_compare_tab(body_mass: float, max_frames: int | None) -> None:
    st.subheader("Compare two walks")
    st.caption("Baseline vs current session (e.g. pre/post rehab).")
    c1, c2 = st.columns(2)
    with c1:
        ref_up = st.file_uploader("Reference video", type=["mp4", "avi", "mov", "mkv"], key="ref_vid")
    with c2:
        samp_up = st.file_uploader("Sample video", type=["mp4", "avi", "mov", "mkv"], key="samp_vid")
    if st.button("Compare videos", type="primary"):
        if not ref_up or not samp_up:
            st.warning("Upload both videos.")
        else:
            with st.spinner("Processing both videos…"):
                try:
                    wdir = Path(tempfile.mkdtemp(prefix="stablewalk_cmp_"))
                    ref_path = wdir / ref_up.name
                    samp_path = wdir / samp_up.name
                    ref_path.write_bytes(ref_up.getvalue())
                    samp_path.write_bytes(samp_up.getvalue())
                    ref_w, samp_w = wdir / "ref", wdir / "samp"
                    ref_w.mkdir(parents=True, exist_ok=True)
                    samp_w.mkdir(parents=True, exist_ok=True)
                    ref_bundle = run_analysis(
                        ref_path, max_frames=max_frames, body_mass_kg=body_mass, work_dir=ref_w
                    )
                    samp_bundle = run_analysis(
                        samp_path, max_frames=max_frames, body_mass_kg=body_mass, work_dir=samp_w
                    )
                    cmp = compare_gait_sessions(
                        ref_bundle.sequence,
                        samp_bundle.sequence,
                        reference_name=ref_up.name,
                        sample_name=samp_up.name,
                        body_mass_kg=body_mass,
                        reference_report=ref_bundle.advanced.gait,
                        sample_report=samp_bundle.advanced.gait,
                    )
                    st.session_state.comparison = cmp
                except Exception as exc:
                    st.error(str(exc))

    cmp = st.session_state.get("comparison")
    if cmp is None:
        return

    a, b = st.columns(2)
    with a:
        st.markdown(f"**{cmp.reference_name}** — {_class_badge(cmp.reference_class.label)}", unsafe_allow_html=True)
        st.metric("Stability", f"{cmp.reference_features.values.get('stability_score', 0):.0f}")
    with b:
        st.markdown(f"**{cmp.sample_name}** — {_class_badge(cmp.sample_class.label)}", unsafe_allow_html=True)
        st.metric("Stability", f"{cmp.sample_features.values.get('stability_score', 0):.0f}")
    st.info(f"More stable session: **{cmp.more_stable}**")

    rows = []
    for d in cmp.metric_deltas:
        if d.delta is None:
            continue
        rows.append({
            "Metric": d.name.replace("_", " "),
            "Reference": f"{d.reference:.3f}" if d.reference is not None else "n/a",
            "Sample": f"{d.sample:.3f}" if d.sample is not None else "n/a",
            "Delta": f"{d.delta:+.3f}",
            "Note": d.interpretation,
        })
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    st.text_area("Comparison report", cmp.summary, height=220, label_visibility="collapsed")


def render_research_tab() -> None:
    doc = ROOT / "stablewalk" / "advanced" / "RESEARCH.md"
    if doc.is_file():
        st.markdown(doc.read_text(encoding="utf-8"))
    else:
        st.caption("See stablewalk/advanced/RESEARCH.md in the repository.")


def main() -> None:
    st.title("StableWalk")
    st.markdown(
        '<p class="subtitle">Gait stability dashboard — pose-based metrics, estimated forces, and issue detection.</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Input")
        upload = st.file_uploader("Walking video", type=["mp4", "avi", "mov", "mkv", "webm"])
        body_mass = st.number_input("Body mass (kg)", 40.0, 150.0, 70.0, 1.0)
        limit = st.checkbox("Limit frames (faster)", value=False)
        max_frames = st.slider("Max frames", 30, 500, 120, 10) if limit else None
        run = st.button("Analyze gait", type="primary", use_container_width=True)
        st.divider()
        st.caption("GRF and stability scores are estimated from video pose, not clinical instruments.")

    if "bundle" not in st.session_state:
        st.session_state.bundle = None
    if "comparison" not in st.session_state:
        st.session_state.comparison = None

    if upload is not None:
        _save_upload(upload)

    if run:
        if not st.session_state.get("video_path"):
            st.warning("Upload a video first.")
        else:
            with st.spinner("Processing pose, overlay, and metrics…"):
                try:
                    st.session_state.bundle = run_analysis(
                        Path(st.session_state.video_path),
                        max_frames=max_frames,
                        body_mass_kg=body_mass,
                        work_dir=Path(st.session_state.work_dir),
                    )
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

    tab1, tab2, tab3 = st.tabs(["Analysis dashboard", "Compare videos", "Research"])

    with tab1:
        bundle: AnalysisBundle | None = st.session_state.bundle
        if bundle is None:
            hero, hint = st.columns([1, 2])
            with hero:
                render_stability_score(0, "—", "—", [])
            with hint:
                st.info("Upload a walking video and click **Analyze gait** to open the dashboard.")
        else:
            render_analysis_tab(bundle)

    with tab2:
        render_compare_tab(body_mass, max_frames)

    with tab3:
        render_research_tab()


def _save_upload(uploaded) -> Path:
    if st.session_state.get("upload_name") != uploaded.name:
        old = st.session_state.get("work_dir")
        if old:
            shutil.rmtree(old, ignore_errors=True)
        work = Path(tempfile.mkdtemp(prefix="stablewalk_"))
        st.session_state.work_dir = str(work)
        st.session_state.upload_name = uploaded.name
        path = work / uploaded.name
        path.write_bytes(uploaded.getvalue())
        st.session_state.video_path = str(path)
        st.session_state.bundle = None
    return Path(st.session_state.video_path)


if __name__ == "__main__":
    main()
