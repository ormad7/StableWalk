"""
Compare two analyzed gait sessions (metrics, diffs, and interpretation).

Numeric values are derived from live analysis results — nothing is hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from stablewalk.models.pose_data import PoseSequence

if TYPE_CHECKING:
    from stablewalk.analysis.analysis_summary import AnalysisSummary
    from stablewalk.analysis.biomechanical.orchestrator import BiomechanicalAnalysisResult
    from stablewalk.analysis.estimated_vgrf_analysis import EstimatedVGRFResult
    from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult

Tone = Literal["better", "worse", "neutral", "na"]
Directionality = Literal["higher_better", "lower_better", "none"]


@dataclass
class CompareMetrics:
    """Scalar metrics used by Comparison Mode cards and diffs."""

    label: str = ""
    session_key: str = ""
    gait_quality: float | None = None
    cadence_spm: float | None = None
    walking_speed_m_s: float | None = None
    symmetry_pct: float | None = None
    stability_label: str = "—"
    stability_margin_m: float | None = None
    stable_pct: float | None = None
    com_excursion_m: float | None = None
    knee_rom_deg: float | None = None
    hip_rom_deg: float | None = None
    ankle_rom_deg: float | None = None
    step_length_m: float | None = None
    stride_length_m: float | None = None
    duration_s: float | None = None
    virtual_grf_peak_bw: float | None = None
    usable_gait_cycles: int | None = None
    detected_gait_cycles: int | None = None
    foot_clearance_m: float | None = None

    def display_map(self) -> dict[str, str]:
        return {
            "Gait Quality": (
                f"{self.gait_quality:.0f}/100"
                if self.gait_quality is not None
                else "N/A"
            ),
            "Cadence": (
                f"{self.cadence_spm:.0f} steps/min"
                if self.cadence_spm is not None
                else "N/A"
            ),
            "Speed": (
                f"{self.walking_speed_m_s:.2f} m/s"
                if self.walking_speed_m_s is not None
                else "N/A"
            ),
            "Step Length": (
                f"{self.step_length_m * 100:.0f} cm"
                if self.step_length_m is not None
                else "N/A"
            ),
            "Knee ROM": (
                f"{self.knee_rom_deg:.0f}°" if self.knee_rom_deg is not None else "N/A"
            ),
            "Symmetry": (
                f"{self.symmetry_pct:.0f}%" if self.symmetry_pct is not None else "N/A"
            ),
            "COM Excursion": (
                f"{self.com_excursion_m:.3f} m"
                if self.com_excursion_m is not None
                else "N/A"
            ),
            "Mean COM–BoS Margin": (
                f"{self.stability_margin_m:.3f} m"
                if self.stability_margin_m is not None
                else "N/A"
            ),
        }


@dataclass
class MetricDiff:
    name: str
    left_value: float | None
    right_value: float | None
    delta: float | None
    unit: str
    higher_is_better: bool
    tone_right: Tone
    display: str
    directionality: Directionality = "none"
    unavailable_reason: str = ""


@dataclass
class SessionComparisonResult:
    left: CompareMetrics
    right: CompareMetrics
    diffs: list[MetricDiff] = field(default_factory=list)
    interpretation: str = ""
    clinical_summary: str = ""
    research_summary: str = ""
    overall_recommendation: str = ""
    lab_report_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left.__dict__,
            "right": self.right.__dict__,
            "diffs": [d.__dict__ for d in self.diffs],
            "interpretation": self.interpretation,
            "clinical_summary": self.clinical_summary,
            "research_summary": self.research_summary,
            "overall_recommendation": self.overall_recommendation,
            "lab_report_summary": self.lab_report_summary,
        }


def _stability_label(ba: BiomechanicalAnalysisResult | None) -> str:
    """Shared vocabulary: Acceptable / Borderline / Reduced (pose-derived heuristic)."""
    from stablewalk.metrics.status import DisplayStatus, build_metric_status

    if ba is None or ba.stability_margin is None:
        return "—"
    pct = ba.stability_margin.stable_pct
    if pct is not None:
        status = build_metric_status("stable_pct", value=float(pct), available=True)
        return status.display_status.value
    # Fall back to dominant per-frame engine state when session % unavailable.
    if not ba.stability_margin.per_frame:
        return "—"
    counts: dict[str, int] = {}
    for f in ba.stability_margin.per_frame:
        if f.stability_state == "Unavailable":
            continue
        counts[f.stability_state] = counts.get(f.stability_state, 0) + 1
    if not counts:
        return "—"
    dominant = max(counts, key=counts.get)
    key = str(dominant).lower()
    if key == "stable":
        return DisplayStatus.ACCEPTABLE.value
    if "reduced" in key:
        return DisplayStatus.BORDERLINE.value
    if key == "unstable":
        return DisplayStatus.REDUCED.value
    return DisplayStatus.UNAVAILABLE.value


def _com_excursion_m(ba: BiomechanicalAnalysisResult | None) -> float | None:
    """Vertical COM range in metres (stature-scaled from body-normalized pose)."""
    from stablewalk.config import DEFAULT_SUBJECT_HEIGHT_M

    if ba is None or ba.center_of_mass is None or ba.center_of_mass.positions is None:
        return None
    pos = np.asarray(ba.center_of_mass.positions, dtype=float)
    if pos.ndim != 2 or pos.shape[0] < 3 or pos.shape[1] < 2:
        return None
    y = pos[:, 1]
    rom_norm = float(np.nanmax(y) - np.nanmin(y))
    if not np.isfinite(rom_norm):
        return None
    return rom_norm * float(DEFAULT_SUBJECT_HEIGHT_M)


def _rom_from_series(series: list[float | None]) -> float | None:
    vals = [float(v) for v in series if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None
    rom = float(max(vals) - min(vals))
    return rom if np.isfinite(rom) else None


def _bilateral_angle_series(
    sequence: PoseSequence | None,
    left_attr: str,
    right_attr: str,
) -> list[float | None]:
    if sequence is None or not sequence.frames:
        return []
    out: list[float | None] = []
    for fr in sequence.frames:
        if not fr.detected or fr.joint_angles is None:
            continue
        la = getattr(fr.joint_angles, left_attr, None)
        ra = getattr(fr.joint_angles, right_attr, None)
        vals = [float(v) for v in (la, ra) if v is not None and np.isfinite(v)]
        out.append(float(np.mean(vals)) if vals else None)
    return out


def extract_compare_metrics(
    *,
    label: str,
    session_key: str = "",
    biomechanical: BiomechanicalAnalysisResult | None = None,
    estimated_vgrf: EstimatedVGRFResult | None = None,
    cycles: GaitCycleAnalysisResult | None = None,
    usable_gait_cycles: int | None = None,
    detected_gait_cycles: int | None = None,
    summary: AnalysisSummary | None = None,
    sequence: PoseSequence | None = None,
    knee_rom_deg: float | None = None,
    duration_s: float | None = None,
) -> CompareMetrics:
    """Build compare-card metrics from an analyzed session bundle."""
    from stablewalk.analysis.biomechanical.walking_speed import (
        is_reportable_walking_speed,
    )

    m = CompareMetrics(label=label, session_key=session_key)
    ba = biomechanical

    if (
        ba is not None
        and ba.gait_quality is not None
        and getattr(ba.gait_quality, "available", True)
        and ba.gait_quality.score is not None
    ):
        m.gait_quality = float(ba.gait_quality.score)

    if ba is not None and ba.gait_metrics is not None:
        gm = ba.gait_metrics
        if gm.cadence is not None and gm.cadence.value is not None:
            m.cadence_spm = float(gm.cadence.value)
        if is_reportable_walking_speed(gm.walking_speed):
            m.walking_speed_m_s = float(gm.walking_speed.value)  # type: ignore[union-attr]
        if gm.step_length is not None and gm.step_length.value is not None:
            sl = float(gm.step_length.value)
            if np.isfinite(sl) and sl > 0:
                m.step_length_m = sl
        if gm.stride_length is not None and gm.stride_length.value is not None:
            st = float(gm.stride_length.value)
            if np.isfinite(st) and st > 0:
                m.stride_length_m = st

    # Use the same analyzed gait-cycle cadence source as Overview/Biomechanics
    # when the biomechanical aggregate did not expose cadence.
    if m.cadence_spm is None:
        from stablewalk.metrics.session_view import _extract_cadence

        cadence, _confidence = _extract_cadence(ba, cycles)
        if cadence is not None and np.isfinite(cadence) and cadence > 0:
            m.cadence_spm = float(cadence)

    if ba is not None and ba.symmetry is not None:
        symmetry = ba.symmetry.overall_symmetry_pct
        v = getattr(symmetry, "value", None) if symmetry is not None else None
        if v is not None and np.isfinite(float(v)):
            m.symmetry_pct = float(v)

    m.stability_label = _stability_label(ba)
    if ba is not None and ba.stability_margin is not None:
        margin = ba.stability_margin.mean_margin_m
        if margin is not None and np.isfinite(float(margin)):
            m.stability_margin_m = float(margin)
        if ba.stability_margin.stable_pct is not None:
            m.stable_pct = float(ba.stability_margin.stable_pct)

    m.com_excursion_m = _com_excursion_m(ba)
    m.knee_rom_deg = knee_rom_deg
    if m.knee_rom_deg is None and sequence is not None:
        try:
            _t, knee_y = knee_angle_series(sequence)
            m.knee_rom_deg = _rom_from_series(knee_y)
        except Exception:
            m.knee_rom_deg = None

    if sequence is not None:
        m.hip_rom_deg = _rom_from_series(
            _bilateral_angle_series(sequence, "left_hip", "right_hip")
        )
        ankle_series = _bilateral_angle_series(sequence, "left_ankle", "right_ankle")
        if not any(v is not None for v in ankle_series):
            ankle_series = _bilateral_angle_series(
                sequence, "left_ankle_flexion", "right_ankle_flexion"
            )
        m.ankle_rom_deg = _rom_from_series(ankle_series)

    m.duration_s = duration_s
    if m.duration_s is None and sequence is not None:
        try:
            n = len(getattr(sequence, "frames", []) or [])
            fps = float(getattr(sequence, "fps", 0.0) or 0.0)
            if n > 0 and fps > 1e-6:
                m.duration_s = (n - 1) / fps
        except Exception:
            m.duration_s = None

    if estimated_vgrf is not None and estimated_vgrf.available:
        peak = estimated_vgrf.metrics.peak_force_bw
        if peak is not None and np.isfinite(peak):
            m.virtual_grf_peak_bw = float(peak)

    if usable_gait_cycles is not None:
        m.usable_gait_cycles = int(usable_gait_cycles)
    elif cycles is not None:
        m.usable_gait_cycles = int(cycles.metrics.gait_cycle_count)

    if detected_gait_cycles is not None:
        m.detected_gait_cycles = int(detected_gait_cycles)
    elif cycles is not None:
        m.detected_gait_cycles = int(getattr(cycles.metrics, "detected_cycle_count", 0) or 0)

    clearance_vals: list[float] = []
    if cycles is not None:
        for fr in getattr(cycles, "per_frame", []) or []:
            for side_name in ("left", "right"):
                side = getattr(fr, side_name, None)
                raw = getattr(side, "foot_clearance_m", None) if side is not None else None
                if raw is None:
                    raw = getattr(fr, f"{side_name}_foot_clearance_m", None)
                if raw is None:
                    continue
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(val):
                    clearance_vals.append(val)
    if clearance_vals:
        m.foot_clearance_m = float(sum(clearance_vals) / len(clearance_vals))

    # Fallback parse from summary strings when biomech bundle is incomplete.
    if summary is not None:
        if m.gait_quality is None and summary.overall_gait_quality and summary.overall_gait_quality.available:
            raw = summary.overall_gait_quality.value.split("/")[0].strip()
            try:
                m.gait_quality = float(raw)
            except ValueError:
                pass
        cadence_field = getattr(summary, "cadence", None)
        if (
            m.cadence_spm is None
            and cadence_field is not None
            and cadence_field.available
        ):
            import re

            match = re.search(
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
                str(cadence_field.value),
            )
            if match is not None:
                try:
                    cadence = float(match.group(0))
                    if np.isfinite(cadence) and cadence > 0:
                        m.cadence_spm = cadence
                except ValueError:
                    pass
        if m.symmetry_pct is None and summary.symmetry and summary.symmetry.available:
            raw = summary.symmetry.value.replace("%", "").strip()
            try:
                m.symmetry_pct = float(raw)
            except ValueError:
                pass
        if m.stability_label == "—" and summary.stability_margin and summary.stability_margin.available:
            from stablewalk.metrics.status import DisplayStatus

            val = summary.stability_margin.value
            low = val.lower()
            if "acceptable" in low or (
                "stable" in low and "unstable" not in low and "reduced" not in low
            ):
                m.stability_label = DisplayStatus.ACCEPTABLE.value
            elif "borderline" in low or "reduced stability" in low or "moderate" in low:
                m.stability_label = DisplayStatus.BORDERLINE.value
            elif "reduced" in low or "unstable" in low or "low" in low:
                m.stability_label = DisplayStatus.REDUCED.value
            elif "high" in low:
                # Legacy summary wording → shared vocabulary
                m.stability_label = DisplayStatus.ACCEPTABLE.value
            else:
                m.stability_label = val.split()[0] if val else "—"

    return m


def _tone(
    delta: float | None,
    *,
    directionality: Directionality,
) -> Tone:
    """Label better/worse only when directionality is explicitly defined."""
    if delta is None or not np.isfinite(delta) or abs(delta) < 1e-9:
        return "neutral"
    if directionality == "none":
        return "neutral"
    if directionality == "higher_better":
        return "better" if delta > 0 else "worse"
    if directionality == "lower_better":
        return "better" if delta < 0 else "worse"
    return "neutral"


def _diff(
    name: str,
    left: float | None,
    right: float | None,
    *,
    unit: str,
    directionality: Directionality = "none",
    fmt: str,
    unavailable_reason: str = "",
    # Backward-compatible alias — prefer directionality=.
    higher_is_better: bool | None = None,
) -> MetricDiff:
    if higher_is_better is not None and directionality == "none":
        directionality = "higher_better" if higher_is_better else "lower_better"
    # Δ = B − A
    delta = None if left is None or right is None else float(right) - float(left)
    tone = _tone(delta, directionality=directionality)
    reason = unavailable_reason or f"{name} unavailable in one or both sessions."
    if left is None and right is None:
        display = f"{name}: unavailable — {reason}"
    elif delta is None:
        missing = "Session A" if left is None else "Session B"
        display = f"{name}: unavailable in {missing} — {reason}"
    else:
        sign = "+" if delta >= 0 else ""
        display = f"{name}: {sign}{fmt.format(delta)}{unit}"
    return MetricDiff(
        name=name,
        left_value=left,
        right_value=right,
        delta=delta,
        unit=unit,
        higher_is_better=directionality == "higher_better",
        tone_right=tone,
        display=display,
        directionality=directionality,
        unavailable_reason=reason if delta is None else "",
    )


def _stability_rank(label: str) -> float | None:
    key = (label or "").strip().lower()
    # Prefer longer phrases before the bare "reduced" token.
    if key in ("high", "stable", "acceptable"):
        return 3.0
    if key in ("moderate", "mod", "borderline") or key == "reduced stability":
        return 2.0
    if key in ("low", "unstable", "reduced"):
        return 1.0
    return None


def _mean_joint_rom(m: CompareMetrics) -> float | None:
    vals = [m.hip_rom_deg, m.knee_rom_deg, m.ankle_rom_deg]
    ok = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(ok)) if ok else None


def compare_session_metrics(
    left: CompareMetrics,
    right: CompareMetrics,
) -> SessionComparisonResult:
    """Compute highlighted diffs and an automatic comparison summary."""
    # Priority order for the Difference Panel (user-facing names).
    diffs = [
        _diff(
            "Cadence difference",
            left.cadence_spm,
            right.cadence_spm,
            unit=" steps/min",
            directionality="higher_better",
            fmt="{:.0f}",
            unavailable_reason="Insufficient heel-strike cycles for cadence.",
        ),
        _diff(
            "ROM difference",
            left.knee_rom_deg,
            right.knee_rom_deg,
            unit="°",
            directionality="higher_better",
            fmt="{:.1f}",
            unavailable_reason="Knee ROM not estimated.",
        ),
        _diff(
            "Hip ROM difference",
            left.hip_rom_deg,
            right.hip_rom_deg,
            unit="°",
            directionality="none",
            fmt="{:.1f}",
            unavailable_reason="Hip ROM not estimated.",
        ),
        _diff(
            "Ankle ROM difference",
            left.ankle_rom_deg,
            right.ankle_rom_deg,
            unit="°",
            directionality="none",
            fmt="{:.1f}",
            unavailable_reason="Ankle ROM not estimated.",
        ),
        _diff(
            "Step length difference",
            left.step_length_m,
            right.step_length_m,
            unit=" m",
            directionality="none",
            fmt="{:.3f}",
            unavailable_reason="Step length not computed.",
        ),
        _diff(
            "Joint angle difference",
            _mean_joint_rom(left),
            _mean_joint_rom(right),
            unit="° mean ROM",
            directionality="none",
            fmt="{:.1f}",
            unavailable_reason="Joint ROM not estimated.",
        ),
        _diff(
            "Stability difference",
            left.stability_margin_m,
            right.stability_margin_m,
            unit=" m",
            directionality="higher_better",
            fmt="{:.3f}",
            unavailable_reason="Stability margin unavailable (insufficient COM–BoS frames).",
        ),
        _diff(
            "Virtual GRF difference",
            left.virtual_grf_peak_bw,
            right.virtual_grf_peak_bw,
            unit=" BW",
            directionality="none",
            fmt="{:.2f}",
            unavailable_reason="Estimated GRF not available.",
        ),
        _diff(
            "Timeline difference",
            left.duration_s,
            right.duration_s,
            unit=" s",
            directionality="none",
            fmt="{:.2f}",
            unavailable_reason="Session duration unavailable.",
        ),
        _diff(
            "Walking speed",
            left.walking_speed_m_s,
            right.walking_speed_m_s,
            unit=" m/s",
            directionality="higher_better",
            fmt="{:.2f}",
            unavailable_reason="Walking speed unavailable — scale/view not reportable.",
        ),
        _diff(
            "Symmetry",
            left.symmetry_pct,
            right.symmetry_pct,
            unit="%",
            directionality="higher_better",
            fmt="{:.0f}",
            unavailable_reason="Symmetry not computed.",
        ),
        _diff(
            "COM excursion",
            left.com_excursion_m,
            right.com_excursion_m,
            unit=" m",
            directionality="lower_better",
            fmt="{:.3f}",
            unavailable_reason="COM path missing.",
        ),
        _diff(
            "Foot clearance",
            left.foot_clearance_m,
            right.foot_clearance_m,
            unit=" m",
            directionality="none",
            fmt="{:.3f}",
            unavailable_reason="Foot clearance was not computed.",
        ),
        _diff(
            "Usable gait cycles",
            None if left.usable_gait_cycles is None else float(left.usable_gait_cycles),
            None if right.usable_gait_cycles is None else float(right.usable_gait_cycles),
            unit="",
            directionality="higher_better",
            fmt="{:.0f}",
            unavailable_reason="No usable gait cycles detected.",
        ),
        _diff(
            "Gait quality",
            left.gait_quality,
            right.gait_quality,
            unit="/100",
            directionality="higher_better",
            fmt="{:.0f}",
            unavailable_reason="Gait quality not scored.",
        ),
    ]

    # Stability category as ordinal (Acceptable/Borderline/Reduced or legacy High/Low).
    lr = _stability_rank(left.stability_label)
    rr = _stability_rank(right.stability_label)
    stab = _diff(
        "Stability",
        lr,
        rr,
        unit="",
        directionality="higher_better",
        fmt="{:.0f}",
        unavailable_reason="Stability label unavailable.",
    )
    if lr is not None and rr is not None and stab.delta is not None:
        stab.display = (
            f"Stability: {left.stability_label} → {right.stability_label}"
        )
    diffs.insert(2, stab)

    interpretation = build_comparison_interpretation(left, right, diffs)
    clinical = build_clinical_summary(left, right, diffs)
    research = build_research_summary(left, right, diffs)
    recommendation = build_overall_recommendation(left, right, diffs)
    lab = build_lab_report_summary(
        left,
        right,
        diffs,
        interpretation=interpretation,
        clinical=clinical,
        research=research,
        recommendation=recommendation,
    )
    return SessionComparisonResult(
        left=left,
        right=right,
        diffs=diffs,
        interpretation=interpretation,
        clinical_summary=clinical,
        research_summary=research,
        overall_recommendation=recommendation,
        lab_report_summary=lab,
    )


def build_comparison_interpretation(
    left: CompareMetrics,
    right: CompareMetrics,
    diffs: list[MetricDiff],
) -> str:
    """Natural-language comparison relative to the left (reference) session."""
    ref = left.label or "the reference gait"
    samp = right.label or "the compared gait"
    bullets: list[str] = []

    if left.walking_speed_m_s and right.walking_speed_m_s and left.walking_speed_m_s > 1e-6:
        pct = (right.walking_speed_m_s / left.walking_speed_m_s - 1.0) * 100.0
        if abs(pct) >= 3:
            direction = "lower" if pct < 0 else "higher"
            bullets.append(f"{abs(pct):.0f}% {direction} walking speed")

    if left.symmetry_pct is not None and right.symmetry_pct is not None:
        d = right.symmetry_pct - left.symmetry_pct
        if abs(d) >= 3:
            direction = "reduced" if d < 0 else "improved"
            bullets.append(f"{abs(d):.0f}% {direction} symmetry")

    cad = next((d for d in diffs if d.name == "Cadence difference"), None)
    if cad and cad.delta is not None and abs(cad.delta) >= 3:
        bullets.append(
            f"{'lower' if cad.delta < 0 else 'higher'} cadence "
            f"({abs(cad.delta):.0f} steps/min)"
        )

    rom = next((d for d in diffs if d.name == "ROM difference"), None)
    if rom and rom.delta is not None and abs(rom.delta) >= 3:
        bullets.append(
            f"{'reduced' if rom.delta < 0 else 'increased'} knee ROM "
            f"({abs(rom.delta):.0f}°)"
        )

    step = next((d for d in diffs if d.name == "Step length difference"), None)
    if step and step.delta is not None and abs(step.delta) >= 0.02:
        bullets.append(
            f"{'shorter' if step.delta < 0 else 'longer'} step length "
            f"({abs(step.delta) * 100:.0f} cm)"
        )

    com = next((d for d in diffs if d.name == "COM excursion"), None)
    if com and com.delta is not None and abs(com.delta) > 0.005:
        bullets.append(
            "higher COM oscillation" if com.delta > 0 else "lower COM oscillation"
        )

    cycles = next((d for d in diffs if d.name == "Usable gait cycles"), None)
    if cycles and cycles.delta is not None and abs(cycles.delta) >= 1:
        bullets.append(
            "fewer usable gait cycles"
            if cycles.delta < 0
            else "more usable gait cycles"
        )

    margin = next((d for d in diffs if d.name == "Stability difference"), None)
    if margin and margin.delta is not None and abs(margin.delta) > 0.002:
        bullets.append(
            "reduced stability margin"
            if margin.delta < 0
            else "increased stability margin"
        )

    gq = next((d for d in diffs if d.name == "Gait quality"), None)
    if gq and gq.delta is not None and abs(gq.delta) >= 3:
        bullets.append(
            "lower gait quality score"
            if gq.delta < 0
            else "higher gait quality score"
        )

    if not bullets:
        return (
            f"Compared with {ref}, {samp} shows similar overall gait metrics "
            "within the resolution of this session."
        )

    body = "\n".join(f"• {b}" for b in bullets)
    return f"Compared with {ref}, {samp} shows\n{body}"


def build_clinical_summary(
    left: CompareMetrics,
    right: CompareMetrics,
    diffs: list[MetricDiff],
) -> str:
    """Clinician-oriented narrative (non-diagnostic) for thesis / case notes."""
    ref = left.label or "Session A"
    samp = right.label or "Session B"
    parts: list[str] = [
        f"This comparative gait review summarises pose-estimated biomechanical "
        f"parameters for {ref} (reference) and {samp} (comparator). "
        "Values are derived from monocular video and must not be interpreted as "
        "a clinical diagnosis or outcome measure for treatment decisions."
    ]

    speed = next((d for d in diffs if d.name == "Walking speed"), None)
    if speed and speed.delta is not None and abs(speed.delta) >= 0.05:
        parts.append(
            f"Estimated walking speed differs by {speed.delta:+.2f} m/s "
            f"({samp} relative to {ref}), which may be relevant when discussing "
            "functional walking capacity in observational contexts."
        )

    sym = next((d for d in diffs if d.name == "Symmetry"), None)
    if sym and sym.delta is not None and abs(sym.delta) >= 5:
        parts.append(
            f"Left–right symmetry differs by {sym.delta:+.0f} percentage points. "
            "Reduced symmetry on video-based estimates can flag movement asymmetry "
            "worth confirming with instrumented assessment where available."
        )

    margin = next((d for d in diffs if d.name == "Stability difference"), None)
    stab = next((d for d in diffs if d.name == "Stability"), None)
    if (margin and margin.delta is not None) or (stab and stab.delta is not None):
        parts.append(
            f"Stability descriptors differ between sessions "
            f"({ref}: {left.stability_label}; {samp}: {right.stability_label}). "
            "Stability margin here is a pose-derived COM–base-of-support proxy."
        )

    step = next((d for d in diffs if d.name == "Step length difference"), None)
    if step and step.delta is not None and abs(step.delta) >= 0.02:
        parts.append(
            f"Estimated step length differs by {step.delta * 100:+.0f} cm "
            f"({samp} relative to {ref})."
        )

    rom = next((d for d in diffs if d.name == "ROM difference"), None)
    if rom and rom.delta is not None and abs(rom.delta) >= 5:
        parts.append(
            f"Knee range of motion differs by {rom.delta:+.0f}°, which may reflect "
            "altered swing or stance kinematics on video-based estimates."
        )

    com = next((d for d in diffs if d.name == "COM excursion"), None)
    if com and com.delta is not None and abs(com.delta) > 0.005:
        parts.append(
            "Vertical centre-of-mass excursion differs between sessions; greater "
            "oscillation can accompany compensatory trunk or limb strategies, but "
            "requires corroboration beyond monocular reconstruction."
        )

    parts.append(
        "Recommendation for clinical use: treat this report as a visual and "
        "quantitative screening companion to history and standardised testing, "
        "not as a substitute for clinical examination."
    )
    return " ".join(parts)


def build_research_summary(
    left: CompareMetrics,
    right: CompareMetrics,
    diffs: list[MetricDiff],
) -> str:
    """Methods-focused summary suitable for MSc thesis documentation."""
    ref = left.label or "reference session"
    samp = right.label or "comparator session"
    available = [d.name for d in diffs if d.delta is not None]
    miss = [d.name for d in diffs if d.delta is None]

    lines = [
        "StableWalk Comparison Mode exports paired session metrics computed from "
        "the same monocular pose-estimation pipeline. Primary outcomes include gait "
        "quality score, cadence, estimated walking speed, lateral symmetry, "
        "stability margin, COM excursion, estimated virtual GRF peak, and usable "
        f"gait-cycle count. Sessions compared: {ref} vs {samp}.",
        f"Difference metrics available for this pair: {', '.join(available) or 'none'}.",
    ]
    if miss:
        lines.append(
            "Unavailable comparisons (insufficient signals): " + ", ".join(miss) + "."
        )

    if left.usable_gait_cycles is not None or right.usable_gait_cycles is not None:
        lines.append(
            f"Usable gait cycles — {ref}: {left.usable_gait_cycles if left.usable_gait_cycles is not None else 'n/a'}; "
            f"{samp}: {right.usable_gait_cycles if right.usable_gait_cycles is not None else 'n/a'}."
        )

    lines.append(
        "Limitations: absolute scale of speed and GRF depends on anthropometric "
        "assumptions; camera viewpoint and occlusions affect landmark visibility; "
        "synchronisation in Comparison Mode uses normalised clip progress when "
        "session lengths differ. Findings are therefore appropriate for within-study "
        "relative comparison rather than absolute clinical thresholds."
    )
    return " ".join(lines)


def build_overall_recommendation(
    left: CompareMetrics,
    right: CompareMetrics,
    diffs: list[MetricDiff],
) -> str:
    """Balanced recommendation highlighting relative strengths without diagnosis."""
    ref = left.label or "Session A"
    samp = right.label or "Session B"

    left_wins = 0
    right_wins = 0
    for d in diffs:
        if d.tone_right == "better":
            right_wins += 1
        elif d.tone_right == "worse":
            left_wins += 1

    if left_wins == 0 and right_wins == 0:
        verdict = (
            f"Overall, {ref} and {samp} present similar reported metrics within "
            "the resolution of this recording pair."
        )
    elif left_wins > right_wins:
        verdict = (
            f"On the majority of reported outcomes, {ref} shows more favourable "
            f"values than {samp} under StableWalk's scoring conventions "
            f"(higher speed/symmetry/margin/quality; lower COM excursion)."
        )
    elif right_wins > left_wins:
        verdict = (
            f"On the majority of reported outcomes, {samp} shows more favourable "
            f"values than {ref} under StableWalk's scoring conventions."
        )
    else:
        verdict = (
            f"{ref} and {samp} each lead on a similar number of metrics; "
            "interpret domain-specific outcomes (speed vs symmetry vs stability) "
            "according to the research or clinical question."
        )

    detail_bits: list[str] = []
    for name in ("Walking speed", "Symmetry", "Gait quality", "COM excursion"):
        d = next((x for x in diffs if x.name == name), None)
        if d is None or d.delta is None:
            continue
        leader = samp if d.tone_right == "better" else ref if d.tone_right == "worse" else "neither"
        if d.tone_right != "neutral":
            detail_bits.append(f"{name}: advantage {leader}")

    closing = (
        "Overall recommendation: use the overlaid kinematics and difference table "
        "to document relative gait presentation for thesis appendices or lab notes, "
        "and state monocular estimation limitations alongside any tabulated claim. "
        "Do not use this export alone for clinical certification or disability decisions."
    )
    if detail_bits:
        return f"{verdict} Key contrasts — {'; '.join(detail_bits)}. {closing}"
    return f"{verdict} {closing}"


def build_lab_report_summary(
    left: CompareMetrics,
    right: CompareMetrics,
    diffs: list[MetricDiff],
    *,
    interpretation: str,
    clinical: str,
    research: str,
    recommendation: str,
) -> str:
    """Laboratory-style comparison report text for the Compare workstation."""
    ref = left.label or "Session A (Video A)"
    samp = right.label or "Session B (Video B)"
    lines = [
        "BIOMECHANICS LABORATORY — GAIT COMPARISON REPORT",
        f"Reference (Video A): {ref}",
        f"Comparator (Video B): {samp}",
        "",
        "1. AUTOMATIC FINDINGS",
        interpretation,
        "",
        "2. SPATIOTEMPORAL & KINEMATIC DIFFERENCES",
    ]
    priority = (
        "Cadence difference",
        "Step length difference",
        "ROM difference",
        "Hip ROM difference",
        "Ankle ROM difference",
        "Joint angle difference",
        "COM excursion",
        "Stability difference",
        "Walking speed",
        "Symmetry",
        "Gait quality",
    )
    by_name = {d.name: d for d in diffs}
    for name in priority:
        d = by_name.get(name)
        if d is None or d.delta is None:
            continue
        lines.append(f"  • {d.display}")
    lines.extend(
        [
            "",
            "3. CLINICAL CONTEXT (NON-DIAGNOSTIC)",
            clinical,
            "",
            "4. METHODS / RESEARCH NOTES",
            research,
            "",
            "5. OVERALL RECOMMENDATION",
            recommendation,
        ]
    )
    return "\n".join(lines)


HEATMAP_JOINTS: tuple[tuple[str, str, str], ...] = (
    ("L Hip", "left_hip", "left_hip"),
    ("R Hip", "right_hip", "right_hip"),
    ("L Knee", "left_knee", "left_knee"),
    ("R Knee", "right_knee", "right_knee"),
    ("L Ankle", "left_ankle", "left_ankle_flexion"),
    ("R Ankle", "right_ankle", "right_ankle_flexion"),
)


def _angle_series_for_attr(
    sequence: PoseSequence | None,
    primary: str,
    fallback: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (normalized_time, angles) for one joint attribute."""
    if sequence is None or not sequence.frames:
        return np.asarray([]), np.asarray([])
    t: list[float] = []
    y: list[float] = []
    for fr in sequence.frames:
        if not fr.detected or fr.joint_angles is None:
            continue
        val = getattr(fr.joint_angles, primary, None)
        if val is None:
            val = getattr(fr.joint_angles, fallback, None)
        if val is None or not np.isfinite(val):
            continue
        t.append(float(fr.timestamp_s or 0.0))
        y.append(float(val))
    if len(t) < 2:
        return np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    tt = np.asarray(t, dtype=float)
    yy = np.asarray(y, dtype=float)
    tn = (tt - tt[0]) / max(float(tt[-1] - tt[0]), 1e-6)
    return tn, yy


def joint_angle_difference_heatmap(
    left_sequence: PoseSequence | None,
    right_sequence: PoseSequence | None,
    *,
    n_bins: int = 40,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a joints×time absolute angle-difference matrix (degrees).

    Both sessions are resampled onto a shared normalised timeline [0, 1].
    """
    labels = [name for name, _p, _f in HEATMAP_JOINTS]
    grid = np.full((len(HEATMAP_JOINTS), n_bins), np.nan, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins)

    for i, (_name, primary, fallback) in enumerate(HEATMAP_JOINTS):
        t_a, y_a = _angle_series_for_attr(left_sequence, primary, fallback)
        t_b, y_b = _angle_series_for_attr(right_sequence, primary, fallback)
        if len(t_a) < 2 or len(t_b) < 2:
            continue
        ya = np.interp(bins, t_a, y_a, left=np.nan, right=np.nan)
        yb = np.interp(bins, t_b, y_b, left=np.nan, right=np.nan)
        grid[i, :] = np.abs(ya - yb)

    return grid, labels


def knee_angle_series(sequence: PoseSequence | None) -> tuple[list[float], list[float | None]]:
    """Return (time_s, mean L/R knee flexion) for overlay charts."""
    if sequence is None or not sequence.frames:
        return [], []
    t: list[float] = []
    y: list[float | None] = []
    for fr in sequence.frames:
        if not fr.detected or fr.joint_angles is None:
            continue
        la = fr.joint_angles.left_knee
        ra = fr.joint_angles.right_knee
        if la is None and ra is None:
            continue
        vals = [v for v in (la, ra) if v is not None]
        t.append(float(fr.timestamp_s or 0.0))
        y.append(float(np.mean(vals)) if vals else None)
    return t, y


def com_path_xyz(biomechanical: BiomechanicalAnalysisResult | None) -> np.ndarray | None:
    """Nx3 COM trajectory for 3D path overlay, or None."""
    if biomechanical is None or biomechanical.center_of_mass is None:
        return None
    pos = biomechanical.center_of_mass.positions
    if pos is None:
        return None
    arr = np.asarray(pos, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
        return None
    return arr


__all__ = [
    "CompareMetrics",
    "HEATMAP_JOINTS",
    "MetricDiff",
    "SessionComparisonResult",
    "build_clinical_summary",
    "build_comparison_interpretation",
    "build_lab_report_summary",
    "build_overall_recommendation",
    "build_research_summary",
    "com_path_xyz",
    "compare_session_metrics",
    "extract_compare_metrics",
    "joint_angle_difference_heatmap",
    "knee_angle_series",
]
