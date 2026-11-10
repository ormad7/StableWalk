"""
Overview clinical summary — display-only helpers from existing session analysis.

Never invents metric values. Unavailable fields become \"Not available\" with a reason.
Bands and vocabulary come from ``stablewalk.metrics`` (shared across tabs).
"""

from __future__ import annotations

from typing import Any, Literal

from stablewalk.metrics.definitions import (
    CADENCE_BORDER_HI,
    CADENCE_BORDER_LO,
    CADENCE_NORMAL_HI,
    CADENCE_NORMAL_LO,
    CONFIDENCE_BORDERLINE_MIN,
    CONFIDENCE_NORMAL_MIN,
    GQ_BORDERLINE_MIN,
    GQ_NORMAL_MIN,
    MARGIN_STABLE_M,
    STABLE_PCT_BORDERLINE_MIN,
    STABLE_PCT_NORMAL_MIN,
    SYMMETRY_BORDERLINE_MIN,
    SYMMETRY_NORMAL_MIN,
    get_metric_definition,
)
from stablewalk.metrics.session_view import _gq_available
from stablewalk.metrics.status import DisplayStatus

StatusLevel = Literal["normal", "borderline", "abnormal", "neutral", "unavailable"]

NOT_AVAILABLE = "Not available"
INSUFFICIENT_EVIDENCE = "Insufficient evidence"

# Literature-cited adult walking cadence band (tooltip / finding prose only —
# not a software configuration gate and not used to invent missing cadence).
_CADENCE_LITERATURE_LO_SPM = 100.0
_CADENCE_LITERATURE_HI_SPM = 120.0

_STATUS_LABEL = {
    "normal": DisplayStatus.ACCEPTABLE.value,
    "borderline": DisplayStatus.BORDERLINE.value,
    "abnormal": DisplayStatus.REDUCED.value,
    "neutral": "",
    # Overview keeps the longer empty-state phrase for KPI cards.
    "unavailable": "Not available",
}

# Short on-card provenance tags (full technical copy lives in tooltips).
_CARD_PROVENANCE = {
    "gait_quality": "Derived",
    "stability_state": "Derived",
    "symmetry": "Derived",
    "cadence": "Estimated",
    "analysis_confidence": "Data quality",
    "gait_cycles": "Measured",
}

_CARD_DESCRIPTIONS = {
    "gait_quality": "Heuristic composite descriptor (not a validated gait-quality instrument)",
    "stability_state": "Physical COM–BoS stability is unreportable from monocular video",
    "symmetry": "Left–right gait similarity (derived kinematic indices)",
    "cadence": "Estimated walking step rate from heel-strike cycles",
    "analysis_confidence": "Interpretation confidence (cycles/events), distinct from pose tracking",
    "gait_cycles": "Detected gait cycles in the current session",
}


def na_display(reason: str) -> tuple[str, str]:
    """Return (\"Not available\", tooltip reason)."""
    tip = (reason or "value not available from the current analyzed session").strip()
    return NOT_AVAILABLE, tip


def card_description(key: str) -> str:
    """Full technical blurb (prefer tooltips; use provenance for on-card text)."""
    alias = {
        "gait_quality": "gait_quality_composite",
        "stability_state": "stable_pct",
    }.get(key, key)
    definition = get_metric_definition(alias)
    if definition is not None:
        return definition.user_explanation
    return _CARD_DESCRIPTIONS.get(key, "")


def card_provenance(key: str) -> str:
    """Short measured / estimated / derived / data-quality tag for the card."""
    alias = {
        "gait_quality": "gait_quality_composite",
        "stability_state": "stable_pct",
    }.get(key, key)
    definition = get_metric_definition(alias)
    if definition is not None:
        return {
            "measured": "Measured",
            "estimated": "Estimated",
            "derived": "Derived",
            "opensim": "OpenSim",
        }.get(definition.source, _CARD_PROVENANCE.get(key, ""))
    return _CARD_PROVENANCE.get(key, "")


def status_label(level: StatusLevel) -> str:
    """Map internal bands to Overview status vocabulary."""
    return _STATUS_LABEL.get(level, "")


def _session_validity_report(gui: Any):
    """Reuse the cached Results validity report, or rebuild from session objects."""
    cache = getattr(gui, "_analysis_summary_cache", None)
    report = getattr(cache, "result_validity", None) if cache is not None else None
    if report is not None:
        return report
    ba = getattr(gui, "_biomech_analysis", None)
    contact = getattr(gui, "_foot_contact", None)
    cycles = getattr(gui, "_gait_cycle", None)
    spectral = getattr(gui, "_spectral_report", None)
    if ba is None and contact is None and cycles is None and spectral is None:
        return None
    from stablewalk.analysis.result_validity import build_result_validity

    n_frames = 0
    if ba is not None and getattr(ba, "video_quality", None) is not None:
        n_frames = int((ba.video_quality.checks or {}).get("n_frames") or 0)
    return build_result_validity(
        biomechanical=ba,
        contact=contact,
        cycles=cycles,
        spectral=spectral,
        n_pose_frames=n_frames,
        coordinate_system=str(
            getattr(getattr(ba, "center_of_mass", None), "coordinate_frame", "") or ""
        )
        if ba is not None
        else "",
    )


def status_label_for_overview(
    key: str,
    level: StatusLevel,
    *,
    display: str = "",
    cadence_spm: float | None = None,
) -> str:
    """Status chip text for Overview cards (Acceptable / Borderline / …)."""
    if level == "unavailable":
        return "Not available"
    if level == "neutral":
        return ""

    if key in {"gait_quality", "symmetry"}:
        return DisplayStatus.UNVALIDATED_HEURISTIC.value

    if key == "analysis_confidence":
        if level == "abnormal":
            return "Low confidence"
        if level == "borderline":
            return "Borderline"
        return "Heuristic"

    # Pose-derived score bands (gait quality, symmetry, stable %) have no
    # documented clinical cut-off. Do not label the upper band Acceptable.
    if key in {
        "stability_state",
        "movement_stability",
        "video_quality",
        "tracking_confidence",
    } and level == "normal":
        if key in {"video_quality", "tracking_confidence"}:
            return "Not accuracy"
        return DisplayStatus.HEURISTIC.value

    if key == "cadence":
        spm = cadence_spm
        if spm is None:
            try:
                spm = float(
                    "".join(ch for ch in (display or "").split()[0] if ch.isdigit() or ch == ".")
                )
            except (ValueError, IndexError):
                spm = None
        if spm is not None:
            if spm > 130:
                return "Elevated" if level == "abnormal" else "Borderline"
            if spm < 60:
                return "Reduced" if level == "abnormal" else "Borderline"

    if key == "stability_state" and level == "abnormal":
        # Never present as an absolute clinical diagnosis.
        return "Reduced"

    return status_label(level)


def classify_overview_metric(
    key: str,
    *,
    display: str,
    gait_quality_score: float | None = None,
    symmetry_pct: float | None = None,
    cadence_spm: float | None = None,
    stable_pct: float | None = None,
    stability_margin_m: float | None = None,
    detected_cycles: int | None = None,
    confidence_pct: float | None = None,
    confidence_level: str | None = None,
) -> StatusLevel:
    """Map an Overview metric to internal bands (display labels via status_label_for_overview)."""
    text = (display or "").strip()
    lower = text.lower()
    if not text or text == NOT_AVAILABLE or lower in ("—", "-", "n/a"):
        return "unavailable"

    if key == "gait_quality":
        score = gait_quality_score
        if score is None:
            try:
                score = float(text.split("/", 1)[0])
            except (TypeError, ValueError, IndexError):
                return "neutral"
        if score >= GQ_NORMAL_MIN:
            return "normal"
        if score >= GQ_BORDERLINE_MIN:
            return "borderline"
        return "abnormal"

    if key == "symmetry":
        pct = symmetry_pct
        if pct is None:
            try:
                pct = float("".join(ch for ch in text if ch.isdigit() or ch == "."))
            except ValueError:
                return "neutral"
        if pct >= SYMMETRY_NORMAL_MIN:
            return "normal"
        if pct >= SYMMETRY_BORDERLINE_MIN:
            return "borderline"
        return "abnormal"

    if key == "cadence":
        spm = cadence_spm
        if spm is None:
            try:
                spm = float("".join(ch for ch in text.split()[0] if ch.isdigit() or ch == "."))
            except (ValueError, IndexError):
                return "neutral"
        if CADENCE_NORMAL_LO <= spm <= CADENCE_NORMAL_HI:
            return "normal"
        if CADENCE_BORDER_LO <= spm < CADENCE_NORMAL_LO or CADENCE_NORMAL_HI < spm <= CADENCE_BORDER_HI:
            return "borderline"
        return "abnormal"

    if key == "stability_state":
        if stable_pct is not None:
            if stable_pct >= STABLE_PCT_NORMAL_MIN:
                return "normal"
            if stable_pct >= STABLE_PCT_BORDERLINE_MIN:
                return "borderline"
            return "abnormal"
        # Prefer numeric session % embedded in display (e.g. "42% stable frames").
        if "%" in lower and "stable" in lower:
            try:
                pct = float("".join(ch for ch in text.split("%", 1)[0] if ch.isdigit() or ch == "."))
                if pct >= STABLE_PCT_NORMAL_MIN:
                    return "normal"
                if pct >= STABLE_PCT_BORDERLINE_MIN:
                    return "borderline"
                return "abnormal"
            except ValueError:
                pass
        if "unstable" in lower or lower == "reduced":
            return "abnormal"
        if "reduced" in lower or "borderline" in lower:
            return "borderline"
        if stability_margin_m is not None:
            if stability_margin_m >= MARGIN_STABLE_M:
                return "normal"
            if stability_margin_m >= 0.02:
                return "borderline"
            return "abnormal"
        if "stable" in lower or lower == "acceptable":
            return "normal"
        return "neutral"

    if key == "gait_cycles":
        n = detected_cycles
        if n is None:
            return "neutral"
        if n >= 2:
            return "normal"
        if n == 1:
            return "borderline"
        return "abnormal"

    if key == "analysis_confidence":
        if confidence_pct is not None:
            if confidence_pct >= CONFIDENCE_NORMAL_MIN:
                return "normal"
            if confidence_pct >= CONFIDENCE_BORDERLINE_MIN:
                return "borderline"
            return "abnormal"
        level = (confidence_level or text).strip().upper()
        if level.startswith("INSUFFICIENT"):
            return "unavailable"
        if level.startswith("HIGH"):
            return "normal"
        if level.startswith("MODERATE") or "PARTIAL" in level:
            return "borderline"
        if level.startswith("LOW"):
            return "abnormal"
        return "neutral"

    return "neutral"


def build_overview_primary_finding(
    *,
    usable_cycles: int | None,
    detected_cycles: int | None,
    cadence_spm: float | None,
    symmetry_pct: float | None,
    stable_pct: float | None,
    stability_margin_m: float | None,
    analysis_confidence_level: str | None,
    gait_quality_score: float | None,
) -> str:
    """Compatibility string: joined Clinical Summary bullets (no long prose)."""
    bullets = build_clinical_summary_bullets(
        usable_cycles=usable_cycles,
        detected_cycles=detected_cycles,
        cadence_spm=cadence_spm,
        symmetry_pct=symmetry_pct,
        stable_pct=stable_pct,
        stability_margin_m=stability_margin_m,
        analysis_confidence_level=analysis_confidence_level,
        gait_quality_score=gait_quality_score,
    )
    return " ".join(bullets)


def build_overview_interpretation(
    *,
    usable_cycles: int | None,
    detected_cycles: int | None,
    cadence_spm: float | None,
    symmetry_pct: float | None,
    stable_pct: float | None,
    stability_margin_m: float | None,
    analysis_confidence_level: str | None,
    gait_quality_score: float | None,
    has_session: bool = True,
    analyzing: bool = False,
) -> str:
    """One clearly worded overall interpretation for the Overview top summary."""
    if analyzing:
        return "Analysis in progress — clinical indicators will appear when processing finishes."
    if not has_session:
        return "Load and analyze a video to see the clinical gait overview."

    bullets = build_clinical_summary_bullets(
        usable_cycles=usable_cycles,
        detected_cycles=detected_cycles,
        cadence_spm=cadence_spm,
        symmetry_pct=symmetry_pct,
        stable_pct=stable_pct,
        stability_margin_m=stability_margin_m,
        analysis_confidence_level=analysis_confidence_level,
        gait_quality_score=gait_quality_score,
    )
    if not bullets:
        return "Session loaded — waiting for analyzable gait indicators."
    cycle_line = next((b for b in bullets if "gait cycle" in b.lower()), None)
    gq_line = next((b for b in bullets if b.lower().startswith("heuristic gait-quality")), None)
    if cycle_line and gq_line:
        return f"{cycle_line} {gq_line}"
    if gq_line:
        return gq_line
    return bullets[0]


def build_overview_key_findings_texts(
    *,
    usable_cycles: int | None,
    detected_cycles: int | None,
    cadence_spm: float | None,
    symmetry_pct: float | None,
    stable_pct: float | None,
    stability_margin_m: float | None,
    analysis_confidence_level: str | None,
    gait_quality_score: float | None,
    tracking_score: float | None = None,
    max_findings: int = 3,
) -> list[str]:
    """Up to three concise Key Findings for the Overview (no method boilerplate)."""
    findings = build_key_findings(
        usable_cycles=usable_cycles,
        detected_cycles=detected_cycles,
        cadence_spm=cadence_spm,
        symmetry_pct=symmetry_pct,
        stable_pct=stable_pct,
        stability_margin_m=stability_margin_m,
        analysis_confidence_level=analysis_confidence_level,
        gait_quality_score=gait_quality_score,
        tracking_score=tracking_score,
    )
    texts: list[str] = []
    for _icon, text, _level in findings:
        if "Pose-derived estimation" in text:
            continue
        texts.append(text)
        if len(texts) >= max_findings:
            break
    return texts


def build_data_quality_warning(
    *,
    analysis_confidence_level: str | None,
    confidence_pct: float | None,
    usable_cycles: int | None,
    detected_cycles: int | None,
    tracking_score: float | None,
) -> str | None:
    """Warning copy when confidence or completeness is insufficient; else None."""
    level = (analysis_confidence_level or "").strip().upper()
    if level.startswith("INSUFFICIENT"):
        return (
            "Data quality is limited — interpretation confidence is insufficient. "
            "Pose tracking quality is a separate score and is not gait-event reliability."
        )
    if level.startswith("LOW"):
        return (
            "Data quality is limited — analysis confidence is low. "
            "Interpret primary metrics cautiously."
        )
    if confidence_pct is not None and float(confidence_pct) < 45:
        return (
            "Data quality is limited — overall confidence is below 45%. "
            "Some indicators have limited evidence."
        )
    if tracking_score is not None and float(tracking_score) < 50:
        return (
            "Data quality is limited — landmark tracking confidence is low "
            "for parts of this session."
        )
    if usable_cycles is not None and int(usable_cycles) <= 0:
        if detected_cycles is not None and int(detected_cycles) > 0:
            return (
                "Data quality is limited — gait cycle candidates were detected "
                "but none met usable-cycle criteria."
            )
        return (
            "Data quality is limited — no usable gait cycles were detected "
            "in this session."
        )
    if usable_cycles is not None and int(usable_cycles) == 1:
        return (
            "Data completeness is limited — only one usable gait cycle was "
            "detected. Findings have limited cycle evidence."
        )
    return None


def build_clinical_summary_bullets(
    *,
    usable_cycles: int | None,
    detected_cycles: int | None,
    cadence_spm: float | None,
    symmetry_pct: float | None,
    stable_pct: float | None,
    stability_margin_m: float | None,
    analysis_confidence_level: str | None,
    gait_quality_score: float | None,
) -> list[str]:
    """3–5 concise Summary statements from session fields only (no diagnosis)."""
    del analysis_confidence_level  # confidence lives on KPI / limitations
    bullets: list[str] = []

    cycle_n = usable_cycles if usable_cycles is not None else detected_cycles
    if cycle_n is not None:
        n = int(cycle_n)
        if usable_cycles is not None:
            if n <= 0:
                if detected_cycles is not None and int(detected_cycles) > 0:
                    bullets.append(
                        f"{int(detected_cycles)} gait cycle candidate(s) detected; "
                        "none met usable-cycle criteria."
                    )
                else:
                    bullets.append("No usable gait cycles were detected.")
            elif n == 1:
                bullets.append("One usable gait cycle was detected.")
            elif n == 2:
                bullets.append("Two usable gait cycles were detected.")
            else:
                bullets.append(f"{n} usable gait cycles were detected.")
        else:
            if n == 1:
                bullets.append("One gait cycle was detected.")
            elif n == 2:
                bullets.append("Two gait cycles were detected.")
            else:
                bullets.append(f"{n} gait cycles were detected.")

    # Explicit gait-quality vs derived-stability reconciliation when they diverge.
    gq_ok = (
        gait_quality_score is not None and float(gait_quality_score) >= GQ_NORMAL_MIN
    )
    stab_reduced = False
    if stable_pct is not None and float(stable_pct) < STABLE_PCT_BORDERLINE_MIN:
        stab_reduced = True
    elif stability_margin_m is not None and float(stability_margin_m) < 0.02:
        stab_reduced = True

    if gq_ok and stab_reduced:
        bullets.append(
            f"Heuristic gait-quality descriptor: {float(gait_quality_score):.0f}/100."
        )
        bullets.append("Pose-derived stability proxy is in the lower band.")
        bullets.append(
            "The heuristic gait score and stable-frame share use different inputs; "
            "a high descriptor does not validate physical stability."
        )
    else:
        if gait_quality_score is not None:
            score = float(gait_quality_score)
            if score >= GQ_NORMAL_MIN:
                band = "upper band"
            elif score >= GQ_BORDERLINE_MIN:
                band = "middle band"
            else:
                band = "lower band"
            bullets.append(
                f"Heuristic gait-quality descriptor: {score:.0f}/100 ({band}; not validated)."
            )

        if stable_pct is not None:
            pct = float(stable_pct)
            if pct < STABLE_PCT_BORDERLINE_MIN:
                bullets.append(f"Pose-derived stability proxy: {pct:.0f}% of frames in the lower band.")
            elif pct < STABLE_PCT_NORMAL_MIN:
                bullets.append(f"Pose-derived stability proxy: {pct:.0f}% of frames in the middle band.")
            else:
                bullets.append(f"Pose-derived stability proxy: {pct:.0f}% of frames in the upper band.")
        elif stability_margin_m is not None:
            m = float(stability_margin_m)
            bullets.append(
                f"Pose-derived COM–BoS proxy margin {m:+.3f} m (not physical stability)."
            )

    if symmetry_pct is not None and len(bullets) < 5:
        pct = float(symmetry_pct)
        bullets.append(
            f"Left–right composite similarity {pct:.0f}% (not knee-only SI_norm)."
        )

    if cadence_spm is not None and len(bullets) < 5:
        bullets.append(
            f"Cadence was estimated at {float(cadence_spm):.0f} steps/min."
        )

    if not bullets:
        bullets.append("Load and analyze a video to populate the clinical summary.")
    return bullets[:5]


def build_key_limitations(
    *,
    tracking_score: float | None = None,
    pose_confidence_level: str | None = None,
    walking_speed_available: bool | None = None,
    opensim_enabled: bool | None = None,
    opensim_ik_completed: bool | None = None,
    monocular: bool = True,
) -> list[str]:
    """Meaningful Key Limitations for the current session only."""
    notes: list[str] = []
    if monocular:
        notes.append("Monocular video — absolute scale and depth are approximate.")
    notes.append("Measurements are pose-derived estimates, not instrumented mocap.")
    notes.append("Absolute distances are approximate (stature-scaled / root-relative).")

    level = (pose_confidence_level or "").strip()
    if tracking_score is not None and float(tracking_score) < 50:
        notes.append("Low-confidence joint tracking for parts of this session.")
    elif level and level.lower() == "low":
        notes.append("Pose estimation confidence is low for this session.")

    if walking_speed_available is False:
        notes.append("Walking speed is not available for this session.")

    if opensim_enabled is False:
        notes.append("OpenSim SDK is not available in this environment.")
    elif opensim_enabled is True and opensim_ik_completed is False:
        notes.append(
            "OpenSim IK is not completed (TRC/MOT export is separate from IK)."
        )

    return notes


def extract_overview_session_badge(gui: Any) -> dict[str, str]:
    """Session badge fields from the current analyzed session only."""
    from pathlib import Path
    from datetime import datetime

    src = (getattr(gui, "_session_display_src", None) or "").strip()
    if not src:
        src = (getattr(gui, "_current_source", None) or "").strip()
    video_name = "No video loaded"
    if src:
        short_fn = getattr(gui, "_short_source_name", None)
        if callable(short_fn):
            try:
                video_name = str(short_fn(src) or src)
            except Exception:
                video_name = src
        else:
            video_name = Path(src).name or src

    backend = "—"
    meta = getattr(gui, "_run_metadata", None)
    used = getattr(meta, "pose_backend_used", None) if meta is not None else None
    if used:
        backend = str(used)
    else:
        selector = getattr(gui, "_selected_pose_backend", None)
        if callable(selector):
            try:
                backend = str(selector() or "—")
            except Exception:
                backend = "—"

    n_frames = 0
    player = getattr(gui, "skeleton_player", None)
    if player is not None and getattr(player, "frame_count", 0) > 0:
        n_frames = int(player.frame_count)
    else:
        indices = getattr(gui, "pose_indices", None)
        if indices:
            n_frames = len(indices)

    from stablewalk.ui.tracking_coverage import measure_tracking_coverage

    coverage = measure_tracking_coverage(
        getattr(gui, "sequence", None),
        getattr(gui, "pose_indices", None),
    )
    if coverage is not None:
        confidence = coverage.compact_text()
    else:
        confidence = NOT_AVAILABLE

    timestamp = NOT_AVAILABLE
    stored = getattr(gui, "_overview_analysis_timestamp", None)
    if stored:
        timestamp = str(stored)
    elif meta is not None:
        poses_path = getattr(meta, "poses_path", None)
        if poses_path:
            try:
                mtime = Path(poses_path).stat().st_mtime
                timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            except (OSError, TypeError, ValueError):
                pass

    return {
        "video_name": video_name,
        "pose_backend": backend,
        "analyzed_frames": str(n_frames) if n_frames > 0 else NOT_AVAILABLE,
        "tracking_confidence": confidence,
        "analysis_timestamp": timestamp,
    }


def walking_speed_available(gui: Any) -> bool | None:
    """True/False when gait metrics exist; None if analysis not loaded."""
    ba = getattr(gui, "_biomech_analysis", None)
    if ba is None or getattr(ba, "gait_metrics", None) is None:
        return None
    ws = getattr(ba.gait_metrics, "walking_speed", None)
    if ws is None:
        return False
    value = getattr(ws, "value", None)
    return value is not None


def build_system_notes(
    *,
    pose_confidence_level: str | None = None,
    opensim_enabled: bool | None = None,
    monocular: bool = True,
) -> list[str]:
    """Back-compat alias → Key Limitations subset."""
    return build_key_limitations(
        pose_confidence_level=pose_confidence_level,
        opensim_enabled=opensim_enabled,
        monocular=monocular,
        walking_speed_available=None,
        opensim_ik_completed=None,
        tracking_score=None,
    )


def build_main_findings(
    *,
    usable_cycles: int | None,
    detected_cycles: int | None,
    cadence_spm: float | None,
    symmetry_pct: float | None,
    stable_pct: float | None,
    stability_margin_m: float | None,
    analysis_confidence_level: str | None,
    gait_quality_score: float | None,
    tracking_score: float | None = None,
) -> list[tuple[str, str, StatusLevel]]:
    """Return (icon, text, level) rows for the Main Findings section."""
    from stablewalk.ui.theme import GLYPH_OK, GLYPH_WARN

    del cadence_spm  # cadence belongs in Clinical Summary when shown
    findings: list[tuple[str, str, StatusLevel]] = []
    ok, warn = GLYPH_OK, GLYPH_WARN

    if tracking_score is not None:
        if tracking_score >= 72:
            findings.append((
                ok,
                "Detector landmark visibility high (not tracking accuracy)",
                "normal",
            ))
        elif tracking_score >= 50:
            findings.append((
                warn,
                "Detector landmark visibility moderate (not tracking accuracy)",
                "borderline",
            ))
        else:
            findings.append((
                warn,
                "Detector landmark visibility low (not tracking accuracy)",
                "abnormal",
            ))

    if symmetry_pct is not None:
        if float(symmetry_pct) >= 70:
            findings.append((ok, f"Left–right composite similarity {float(symmetry_pct):.0f}%", "normal"))
        elif float(symmetry_pct) >= 55:
            findings.append((warn, f"Left–right composite similarity {float(symmetry_pct):.0f}%", "borderline"))
        else:
            findings.append((warn, f"Left–right composite similarity {float(symmetry_pct):.0f}%", "abnormal"))

    if stable_pct is not None:
        if float(stable_pct) < 55:
            findings.append((warn, "Pose-derived stability proxy in the lower band", "abnormal"))
        elif float(stable_pct) < 70:
            findings.append((warn, "Pose-derived stability proxy in the middle band", "borderline"))
        else:
            findings.append((ok, "Pose-derived stability proxy in the upper band", "normal"))
    elif stability_margin_m is not None:
        if float(stability_margin_m) < 0.02:
            findings.append((warn, "Pose-derived COM–BoS proxy in the lower band", "abnormal"))
        elif float(stability_margin_m) < 0.04:
            findings.append((warn, "Pose-derived COM–BoS proxy in the middle band", "borderline"))
        else:
            findings.append((ok, "Pose-derived COM–BoS proxy in the upper band", "normal"))

    if gait_quality_score is not None:
        score = float(gait_quality_score)
        if score >= 60:
            findings.append((ok, f"Heuristic gait-quality descriptor {score:.0f}/100", "normal"))
        elif score >= 45:
            findings.append((warn, f"Heuristic gait-quality descriptor {score:.0f}/100", "borderline"))
        else:
            findings.append((warn, f"Heuristic gait-quality descriptor {score:.0f}/100", "abnormal"))

    if usable_cycles is not None and int(usable_cycles) >= 2:
        findings.append((ok, "Multiple usable gait cycles", "normal"))
    elif detected_cycles is not None and int(detected_cycles) < 2:
        findings.append((warn, "Limited gait-cycle evidence", "borderline"))

    level = (analysis_confidence_level or "").strip().upper()
    if level.startswith("LOW"):
        findings.append((warn, "Low analysis confidence", "abnormal"))

    # Method note — always true for monocular pose pipeline (not a measured score).
    method_note = (warn, "Pose-derived estimation (current session)", "borderline")

    seen: set[str] = set()
    out: list[tuple[str, str, StatusLevel]] = []
    for icon, text, st in findings:
        if text in seen:
            continue
        seen.add(text)
        out.append((icon, text, st))
        if len(out) >= 5:
            break
    if method_note[1] not in seen:
        out.append(method_note)
    return out


def build_key_findings(
    *,
    usable_cycles: int | None,
    detected_cycles: int | None,
    cadence_spm: float | None,
    symmetry_pct: float | None,
    stable_pct: float | None,
    stability_margin_m: float | None,
    analysis_confidence_level: str | None,
    gait_quality_score: float | None,
    tracking_score: float | None = None,
) -> list[tuple[str, str, StatusLevel]]:
    """Alias for Main Findings (back-compat)."""
    return build_main_findings(
        usable_cycles=usable_cycles,
        detected_cycles=detected_cycles,
        cadence_spm=cadence_spm,
        symmetry_pct=symmetry_pct,
        stable_pct=stable_pct,
        stability_margin_m=stability_margin_m,
        analysis_confidence_level=analysis_confidence_level,
        gait_quality_score=gait_quality_score,
        tracking_score=tracking_score,
    )


def _pct_0_100(value: Any) -> float | None:
    """Normalize a 0–1 or 0–100 quality signal to 0–100; None if missing."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 1.0:
        v *= 100.0
    return max(0.0, min(100.0, v))


def _usable_cycle_quality_pct(usable: Any, detected: Any) -> float | None:
    """Data-quality contribution from usable gait cycles (not a clinical score)."""
    if usable is None and detected is None:
        return None
    try:
        u = int(usable) if usable is not None else None
    except (TypeError, ValueError):
        u = None
    if u is not None:
        # Meeting the cadence minimum (2 cycles) is not 100% measurement evidence.
        if u >= 5:
            return 90.0
        if u >= 3:
            return 75.0
        if u == 2:
            return 55.0
        if u == 1:
            return 40.0
        return 0.0
    try:
        d = int(detected) if detected is not None else None
    except (TypeError, ValueError):
        return None
    if d is None:
        return None
    if d >= 2:
        return 70.0
    if d == 1:
        return 45.0
    return 20.0


def extract_confidence_visualization(gui: Any) -> dict[str, Any]:
    """
    Overview Analysis Confidence from real data-quality signals only.

    Separate from tracking confidence (HUD) and from metric status chips.
    Does not invent percentages from HIGH/MODERATE/LOW enums.
    OpenSim incompleteness is reported in Key Limitations and does not
    zero out pose-only confidence here.

    Returns keys: pct, display, pose_level, biomech_level, tip, available,
    tracking_score, signal_names
    """
    ba = getattr(gui, "_biomech_analysis", None)
    result = getattr(gui, "_stability_result", None) or getattr(
        gui, "_last_stability_result", None
    )
    summary = getattr(result, "gait_summary", None) if result is not None else None
    ac = getattr(summary, "analysis_confidence", None) if summary is not None else None

    signals: list[tuple[str, float]] = []
    tracking_score: float | None = None

    if ba is not None and getattr(ba, "video_quality", None) is not None:
        checks = getattr(ba.video_quality, "checks", None) or {}
        tracking_score = _pct_0_100(checks.get("pose_confidence_score"))
        if tracking_score is None:
            tracking_score = _pct_0_100(
                getattr(ba.video_quality, "overall_quality_score", None)
            )
        # Tracking is reported as pose_level only — it is not analysis confidence.

    pose_valid = _pct_0_100(
        getattr(ac, "pose_valid_frame_pct", None) if ac is not None else None
    )
    if pose_valid is not None:
        signals.append(("valid pose-frame ratio", pose_valid))

    completeness = _pct_0_100(
        getattr(ac, "completeness_pct", None) if ac is not None else None
    )
    if completeness is not None:
        signals.append(("analysis completeness", completeness))
    else:
        evidence = getattr(ac, "evidence_index", None) if ac is not None else None
        evidence_pct = _pct_0_100(evidence)
        if evidence_pct is not None:
            signals.append(("evidence index", evidence_pct))

    if ba is not None and getattr(ba, "stability_margin", None) is not None:
        sm = ba.stability_margin
        if getattr(sm, "reportable", True):
            valid_raw = getattr(sm, "valid_frame_ratio", None)
            if valid_raw is not None and float(valid_raw) > 0:
                valid_ratio = _pct_0_100(valid_raw)
                if valid_ratio is not None:
                    signals.append(("stability valid-frame ratio", valid_ratio))

    usable = detected = None
    resolver = getattr(gui, "_resolved_gait_cycle_count", None)
    if callable(resolver):
        try:
            usable, detected = resolver()
        except Exception:
            usable = detected = None
    if usable is None and result is not None:
        usable = getattr(result, "usable_gait_cycles", None)
    cycle_q = _usable_cycle_quality_pct(usable, detected)
    if cycle_q is not None:
        signals.append(("usable gait-cycle evidence", cycle_q))

    # Do not fold gait-quality.confidence into data-quality — that mixes
    # a performance composite with measurement reliability.

    # Mean of available signals only — never invent a decorative %.
    pct: float | None = None
    tip = (
        "Data-quality confidence from available session signals "
        "(not clinical certainty)."
    )
    signal_names = [name for name, _ in signals]
    if signals:
        pct = sum(v for _, v in signals) / float(len(signals))
        tip = (
            "Mean of data-quality signals: "
            + ", ".join(signal_names)
            + ". Separate from metric status and joint tracking confidence."
        )

    pose_level = "Not available"
    ts_for_level = tracking_score
    if ts_for_level is None and pose_valid is not None:
        ts_for_level = pose_valid
    if ts_for_level is not None:
        if ts_for_level >= 72:
            pose_level = "Landmark visibility high"
        elif ts_for_level >= 50:
            pose_level = "Landmark visibility moderate"
        else:
            pose_level = "Landmark visibility low"

    biomech_level = "Not available"
    level = getattr(ac, "level", None) if ac is not None else None
    if level:
        biomech_level = str(level).strip().title()

    usable_n = None
    try:
        if usable is not None:
            usable_n = int(usable)
    except (TypeError, ValueError):
        usable_n = None
    if usable_n == 0:
        if pct is not None:
            pct = min(float(pct), 35.0)
        level_text = str(level).strip().upper() if level else ""
        if level_text.startswith("INSUFFICIENT") or pct is not None:
            display = "Insufficient"
            available = True
            tip = (
                "No usable gait cycles — analysis confidence is insufficient. "
                "Pose/landmark quality is reported separately and is not a "
                "substitute for gait-event reliability."
            )
        elif level:
            display = str(level).strip().title()
            available = True
        else:
            display = NOT_AVAILABLE
            available = False
    elif pct is not None:
        display = f"{pct:.0f}%"
        available = True
    elif level:
        # Categorical session level only — do not map to an invented percentage.
        display = str(level).strip().title()
        available = True
        tip = (
            f"Session analysis confidence level ({display}). "
            "No numeric data-quality aggregate was available."
        )
    else:
        display = NOT_AVAILABLE
        available = False

    level_u = str(level).strip().upper() if level else ""
    if level_u.startswith("INSUFFICIENT"):
        display = "Insufficient"
        available = True
    elif pct is not None and float(pct) < 45:
        display = f"{float(pct):.0f}%"
        available = True
    elif usable_n is not None and usable_n < 3 and (
        level_u.startswith("HIGH") or (pct is not None and float(pct) >= 72)
    ):
        display = "Moderate"
        available = True
        tip = (
            f"{tip} High requires >=3 complete gait cycles; this session has "
            f"{usable_n}."
        )
    elif level_u.startswith("HIGH") and (
        (pct is not None and float(pct) < 68) or usable_n == 0
    ):
        display = "Insufficient" if usable_n == 0 else (f"{float(pct):.0f}%" if pct is not None else "Low")
        available = True
    tip = (
        f"{tip} Pose/landmark visibility: {pose_level} "
        "(detector score, not tracking accuracy). "
        f"Interpretation confidence: {biomech_level}. These are different quantities."
    )

    return {
        "pct": pct,
        "display": display,
        "pose_level": pose_level,
        "biomech_level": biomech_level,
        "tip": tip,
        "available": available,
        "tracking_score": tracking_score,
        "signal_names": signal_names,
    }


def extract_overview_summary_fields(gui: Any) -> dict[str, tuple[str, str]]:
    """
    Map Overview clinical summary cells to (display, tooltip).

    Sources match Biomechanics / Motion Analysis session fields — never guessed.
    """
    out: dict[str, tuple[str, str]] = {}
    ba = getattr(gui, "_biomech_analysis", None)
    gait = getattr(gui, "_gait_cycle", None)
    summary = None
    result = getattr(gui, "_stability_result", None) or getattr(gui, "_last_stability_result", None)
    if result is not None:
        summary = getattr(result, "gait_summary", None)
        if summary is None:
            try:
                from stablewalk.analysis.gait_analysis_summary import (
                    build_gait_analysis_summary,
                )

                summary = build_gait_analysis_summary(result)
            except Exception:
                summary = None

    validity = _session_validity_report(gui)
    from stablewalk.analysis.result_validity import (
        gait_quality_withhold_reason,
        laterality_withhold_reason,
    )

    quality_hold = gait_quality_withhold_reason(validity)
    laterality_hold = laterality_withhold_reason(validity)

    # Gait Quality — only compute_gait_quality_score. Do not substitute
    # gait_analysis_summary.gait_quality (different domains / algorithm).
    # Filename / demo key must never raise this score.
    gq = None
    if (
        not quality_hold
        and ba is not None
        and _gq_available(getattr(ba, "gait_quality", None))
    ):
        gq = getattr(ba.gait_quality, "score", None)
    if quality_hold:
        out["gait_quality"] = na_display(quality_hold)
    elif gq is not None:
        out["gait_quality"] = (
            f"{float(gq):.0f}/100",
            "Unvalidated heuristic session descriptor (same source as "
            "Biomechanics). Not pose coverage, landmark visibility, spectral "
            "reliability, or a clinical grade. Independent of source video labels.",
        )
    else:
        missing = []
        gq_obj = getattr(ba, "gait_quality", None) if ba is not None else None
        if gq_obj is not None:
            components = getattr(gq_obj, "components", None) or {}
            missing = [name for name in _GAIT_QUALITY_WEIGHTS if name not in components]
        reason = "N/A — Composite Gait Quality was not scored for this session (not a 0/100)."
        if missing:
            reason += " Missing inputs: " + ", ".join(missing) + "."
        else:
            reason += " Required pose-derived components were unavailable."
        out["gait_quality"] = na_display(reason)

    # Symmetry — overall % only when computed; never substitute another score.
    sym = None
    tip_sym = (
        "Overall symmetry is a weighted composite of step, stride, stance, "
        "swing, cadence, and joint-ROM similarity (same source as "
        "Biomechanics). It is not knee-only SI_norm, signed SI, or ASI."
    )
    symmetry_obj = getattr(ba, "symmetry", None) if ba is not None else None
    if symmetry_obj is not None:
        field = getattr(symmetry_obj, "overall_symmetry_pct", None)
        if field is not None:
            sym = getattr(field, "value", None)
            conf = getattr(field, "confidence", None)
            if conf is not None:
                tip_sym = f"{tip_sym} Confidence {float(conf):.0%}."
        if sym is None:
            warns = [
                str(item).strip()
                for item in (getattr(symmetry_obj, "warnings", None) or ())
                if str(item).strip()
            ]
            tip_sym = warns[0] if warns else (
                "Overall symmetry unavailable: fewer than two valid bilateral "
                "components for this session."
            )
    if laterality_hold:
        out["symmetry"] = na_display(laterality_hold)
    elif ba is None:
        out["symmetry"] = na_display("symmetry not computed for this session")
    elif sym is not None:
        out["symmetry"] = (
            f"{float(sym):.0f}%",
            f"{tip_sym} Unvalidated heuristic — not a gait-laterality finding "
            "from missing-data coverage. Independent of source video labels.",
        )
    elif symmetry_obj is None:
        out["symmetry"] = na_display("symmetry analysis did not run for this session")
    else:
        out["symmetry"] = na_display(tip_sym)

    # Cadence — withhold low-confidence / single-channel exploratory values.
    from stablewalk.analysis.gait_cycle_analysis import (
        cadence_is_exploratory,
        summary_cadence_spm,
    )

    cadence = None
    tip_cad = "Session cadence (same source as Biomechanics / Motion Analysis)."
    gait_metrics_obj = getattr(gait, "metrics", None) if gait is not None else None
    exploratory = cadence_is_exploratory(gait_metrics_obj, gait)
    if exploratory:
        out["cadence"] = na_display(
            "exploratory cadence withheld from summary "
            "(low-confidence or single-channel HS–HS timing)"
        )
    if "cadence" not in out and ba is not None and getattr(ba, "gait_metrics", None) is not None:
        cfield = getattr(ba.gait_metrics, "cadence", None)
        if cfield is not None:
            cadence = getattr(cfield, "value", None)
            conf = getattr(cfield, "confidence", None)
            note = str(getattr(cfield, "note", "") or "").lower()
            if cadence is None and ("exploratory" in note or conf is not None):
                out["cadence"] = na_display(
                    note or f"cadence unavailable (confidence {float(conf or 0):.0%})"
                )
            elif cadence is not None:
                if conf is not None:
                    tip_cad = f"{tip_cad} Confidence {float(conf):.0%}."
                out["cadence"] = (f"{float(cadence):.0f}", tip_cad)
    if "cadence" not in out:
        withheld = summary_cadence_spm(gait_metrics_obj, gait)
        if withheld is not None:
            out["cadence"] = (f"{float(withheld):.0f}", tip_cad)
        else:
            out["cadence"] = na_display("cadence not computed for this session")

    # Detected gait cycles — emphasize detected count on the Overview card
    usable = detected = None
    resolver = getattr(gui, "_resolved_gait_cycle_count", None)
    if callable(resolver):
        try:
            usable, detected = resolver()
        except Exception:
            usable = detected = None
    if usable is None and result is not None:
        usable = getattr(result, "usable_gait_cycles", None)
    if detected is None and gait is not None:
        detected = getattr(gait, "detected_cycle_count", None)
        if detected is None and getattr(gait, "cycles", None) is not None:
            try:
                detected = len(gait.cycles)
            except Exception:
                detected = None
    extras = {}
    if validity is not None:
        extras = getattr(validity.cycle_detection, "extras", None) or {}
    n_left = extras.get("n_left")
    n_right = extras.get("n_right")
    if n_left is not None or n_right is not None:
        n_cand = extras.get("n_candidates")
        cycle_text = f"{int(n_left or 0)} L / {int(n_right or 0)} R accepted"
        if n_cand is not None:
            cycle_text = f"{cycle_text} · {int(n_cand)} candidates"
        tip = (
            "Accepted per-foot event cycles vs raw HS–HS candidates. "
            "Unequal cycle counts are not a gait-laterality finding."
        )
        if usable is not None:
            tip = f"{tip} Usable cycles: {int(usable)}."
        if detected is not None:
            tip = f"{tip} Detected cycles: {int(detected)}."
        out["gait_cycles"] = (cycle_text, tip)
        out["cycle_evidence"] = (cycle_text, tip)
    elif detected is not None or usable is not None:
        if detected is not None:
            tip = "Detected gait cycles from the current analyzed session."
            if usable is not None:
                tip = f"{tip} Usable cycles: {int(usable)}."
            out["gait_cycles"] = (str(int(detected)), tip)
        else:
            out["gait_cycles"] = (
                str(int(usable)),
                "Usable gait cycles from the current analyzed session.",
            )
        out["cycle_evidence"] = out["gait_cycles"]
    else:
        out["gait_cycles"] = na_display("gait cycle counts not available")
        out["cycle_evidence"] = out["gait_cycles"]

    if validity is not None:
        contact_extras = getattr(validity.contact_coverage, "extras", None) or {}
        left_pct = contact_extras.get("left_pct")
        right_pct = contact_extras.get("right_pct")
        status = str(getattr(validity.contact_coverage, "status", "") or "")
        if left_pct is not None:
            out["contact_coverage_left"] = (
                f"{float(left_pct):.0f}% known",
                f"Left foot valid contact coverage is {float(left_pct):.0f}% "
                f"({status or 'unknown'}). Missing contact is not gait asymmetry.",
            )
        else:
            out["contact_coverage_left"] = na_display(
                "Left-foot contact coverage was not recorded."
            )
        if right_pct is not None:
            out["contact_coverage_right"] = (
                f"{float(right_pct):.0f}% known",
                f"Right foot valid contact coverage is {float(right_pct):.0f}% "
                f"({status or 'unknown'}). Missing contact is not gait asymmetry.",
            )
        else:
            out["contact_coverage_right"] = na_display(
                "Right-foot contact coverage was not recorded."
            )
    else:
        out["contact_coverage_left"] = na_display(
            "Per-foot contact coverage is not available."
        )
        out["contact_coverage_right"] = na_display(
            "Per-foot contact coverage is not available."
        )

    # Stability — monocular COM–BoS is unvalidated. Cards stay N/A with the
    # shared reason; the Balance plot is an unvalidated signed-distance preview.
    tip_stab = (
        "Pose-derived stability estimate from estimated COM vs base of support. "
        "Research/education only — not a clinical diagnosis."
    )
    if ba is not None and getattr(ba, "stability_margin", None) is not None:
        sm = ba.stability_margin
        reportable_stab = getattr(sm, "reportable", True)
        try:
            from stablewalk.analysis.biomechanical.stability_margin import (
                balance_display_contract,
            )

            stab_contract = balance_display_contract(sm)
        except Exception:
            stab_contract = None
        if reportable_stab is False or (
            stab_contract is not None
            and (
                stab_contract.validity != "valid"
                or not stab_contract.classification_supported
            )
        ):
            reason = (
                (stab_contract.reason if stab_contract is not None else None)
                or getattr(sm, "experimental_note", None)
                or "Monocular COM–BoS is experimental and is not a physical stability score."
            )
            out["stability_state"] = na_display(str(reason))
        diag = ""
        try:
            from stablewalk.analysis.biomechanical.stability_margin import (
                format_stability_diagnostics,
            )

            diag = format_stability_diagnostics(sm)
        except Exception:
            diag = ""
        stable_pct = getattr(sm, "stable_pct", None) if reportable_stab else None
        if reportable_stab is False or (
            stab_contract is not None and not stab_contract.classification_supported
        ):
            pass
        elif stable_pct is not None:
            # Short visible value; full “derived stable-frame” wording stays in tip.
            tip = (
                tip_stab
                + " Derived stable-frame estimate: share of valid frames with "
                "COM inside the estimated support polygon (pose-derived from "
                "monocular video; not a clinical diagnosis)."
            )
            if diag:
                tip = f"{tip} {diag}"
            out["stability_state"] = (
                f"{float(stable_pct):.0f}%",
                tip,
            )
        else:
            # Fallback: map engine state labels to presentation vocabulary.
            frame_index = None
            try:
                from stablewalk.ui.tk.dashboard_status_bar import playhead_frame_0based

                frame_index = playhead_frame_0based(gui)
            except Exception:
                frame_index = None
            stab_text = None
            per_frame = getattr(sm, "per_frame", None) or []
            if frame_index is not None and per_frame:
                try:
                    for entry in per_frame:
                        if getattr(entry, "frame_index", None) == int(frame_index):
                            state = getattr(entry, "stability_state", None)
                            if state:
                                stab_text = str(state)
                            break
                except Exception:
                    pass
            if stab_text:
                lower = stab_text.strip().lower()
                if lower == "unstable":
                    label = "Reduced"
                elif lower == "reduced stability":
                    label = "Borderline"
                elif lower == "stable":
                    label = "Stable"
                else:
                    label = stab_text.strip()
                out["stability_state"] = (
                    label,
                    tip_stab
                    + f" Engine frame state: {stab_text}."
                    + (f" {diag}" if diag else ""),
                )
            else:
                reason = "stability state not computed for this session"
                if diag:
                    reason = f"{reason}. {diag}"
                out["stability_state"] = na_display(reason)
    else:
        out["stability_state"] = na_display(
            "stability state not computed for this session"
        )

    # Confidence — prefer numeric % for the Overview card
    conf_viz = extract_confidence_visualization(gui)
    if conf_viz.get("available"):
        out["analysis_confidence"] = (
            conf_viz["display"],
            conf_viz.get("tip") or "Analysis confidence for the current session.",
        )
    else:
        level = None
        if summary is not None and getattr(summary, "analysis_confidence", None) is not None:
            level = getattr(summary.analysis_confidence, "level", None)
        if level:
            out["analysis_confidence"] = (str(level).title(), "Analysis confidence level.")
        else:
            out["analysis_confidence"] = na_display(
                "analysis confidence not computed for this session"
            )

    return out


def primary_finding_inputs(gui: Any) -> dict[str, Any]:
    """Collect raw inputs for clinical summary helpers (no invention)."""
    ba = getattr(gui, "_biomech_analysis", None)
    result = getattr(gui, "_stability_result", None) or getattr(
        gui, "_last_stability_result", None
    )
    summary = None
    if result is not None:
        summary = getattr(result, "gait_summary", None)

    usable = detected = None
    resolver = getattr(gui, "_resolved_gait_cycle_count", None)
    if callable(resolver):
        try:
            usable, detected = resolver()
        except Exception:
            usable = detected = None
    if usable is None and result is not None:
        usable = getattr(result, "usable_gait_cycles", None)

    cadence = None
    gait = getattr(gui, "_gait_cycle", None)
    from stablewalk.analysis.gait_cycle_analysis import (
        cadence_is_exploratory,
        summary_cadence_spm,
    )

    gait_metrics_obj = getattr(gait, "metrics", None) if gait is not None else None
    if not cadence_is_exploratory(gait_metrics_obj, gait):
        if ba is not None and getattr(ba, "gait_metrics", None) is not None:
            cfield = getattr(ba.gait_metrics, "cadence", None)
            if cfield is not None:
                cadence = getattr(cfield, "value", None)
        if cadence is None:
            cadence = summary_cadence_spm(gait_metrics_obj, gait)

    validity = _session_validity_report(gui)
    from stablewalk.analysis.result_validity import (
        gait_quality_withhold_reason,
        laterality_withhold_reason,
    )

    laterality_hold = laterality_withhold_reason(validity)
    quality_hold = gait_quality_withhold_reason(validity)

    symmetry = None
    if (
        not laterality_hold
        and ba is not None
        and getattr(ba, "symmetry", None) is not None
    ):
        field = getattr(ba.symmetry, "overall_symmetry_pct", None)
        if field is not None:
            symmetry = getattr(field, "value", None)

    stable_pct = None
    margin_m = None
    if ba is not None and getattr(ba, "stability_margin", None) is not None:
        sm = ba.stability_margin
        classification_ok = bool(getattr(sm, "classification_supported", False))
        if getattr(sm, "reportable", True) and classification_ok:
            stable_pct = getattr(sm, "stable_pct", None)
            margin_m = getattr(sm, "mean_margin_m", None)
            if margin_m is None:
                margin_m = getattr(sm, "mean_stability_margin_m", None)

    gq = None
    if (
        not quality_hold
        and ba is not None
        and _gq_available(getattr(ba, "gait_quality", None))
    ):
        gq = getattr(ba.gait_quality, "score", None)

    level = None
    if summary is not None and getattr(summary, "analysis_confidence", None) is not None:
        level = getattr(summary.analysis_confidence, "level", None)

    tracking_score = None
    if ba is not None and getattr(ba, "video_quality", None) is not None:
        checks = getattr(ba.video_quality, "checks", None) or {}
        if checks.get("pose_confidence_score") is not None:
            tracking_score = float(checks["pose_confidence_score"])

    return {
        "usable_cycles": usable,
        "detected_cycles": detected,
        "cadence_spm": cadence,
        "symmetry_pct": symmetry,
        "stable_pct": stable_pct,
        "stability_margin_m": margin_m,
        "analysis_confidence_level": level,
        "gait_quality_score": gq,
        "tracking_score": tracking_score,
    }


# Nominal weights from compute_gait_quality_score (renormalized over active).
_GAIT_QUALITY_WEIGHTS: dict[str, float] = {
    "stability": 0.22,
    "stability_margin": 0.15,
    "symmetry": 0.20,
    "cadence_consistency": 0.12,
    "cycle_consistency": 0.12,
    "contact_reliability": 0.10,
    "motion_quality": 0.09,
}


def format_gait_quality_session_details(gui: Any) -> str | None:
    """
    Live session block for Metric Details: inputs, weights, confidence,
    missing components, limiting factor. None when gait quality is absent.
    """
    ba = getattr(gui, "_biomech_analysis", None)
    gq = getattr(ba, "gait_quality", None) if ba is not None else None
    if gq is None:
        return None

    score = getattr(gq, "score", None)
    confidence = getattr(gq, "confidence", None)
    components = getattr(gq, "components", None) or {}
    factors = getattr(gq, "dominant_factors", None) or []
    explanation = getattr(gq, "explanation", None) or ""

    lines: list[str] = [
        "Session instance (from ba.gait_quality — same source as Overview / Biomechanics):",
    ]
    if score is not None:
        lines.append(f"  Score: {float(score):.0f}/100")
    if confidence is not None:
        conf = float(confidence)
        conf_txt = f"{conf:.0%}" if conf <= 1.0 else f"{conf:.0f}%"
        lines.append(f"  Confidence: {conf_txt}")

    lines.append("  Active components (value · nominal weight):")
    if components:
        for name, value in sorted(components.items()):
            w = _GAIT_QUALITY_WEIGHTS.get(str(name))
            w_txt = f"{w:.2f}" if w is not None else "—"
            lines.append(f"    • {name}: {float(value):.1f} (w={w_txt})")
    else:
        lines.append("    • none")

    missing = [
        name for name in _GAIT_QUALITY_WEIGHTS if name not in components
    ]
    lines.append("  Missing components (omitted; weights renormalized):")
    if missing:
        for name in missing:
            lines.append(
                f"    • {name} (nominal w={_GAIT_QUALITY_WEIGHTS[name]:.2f})"
            )
    else:
        lines.append("    • none")

    if factors:
        lines.append("  Limiting / dominant factors:")
        for f in factors:
            lines.append(f"    • {f}")
    if explanation:
        lines.append(f"  Explanation: {explanation}")

    formula = str(getattr(gq, "formula", None) or "").strip()
    if formula:
        lines.append(f"  Formula: {formula}")
    else:
        from stablewalk.analysis.quality_layers import GAIT_QUALITY_FORMULA

        lines.append(f"  Formula: {GAIT_QUALITY_FORMULA}")
    from stablewalk.analysis.quality_layers import GAIT_QUALITY_THRESHOLD_SOURCE

    lines.append(f"  Threshold source: {GAIT_QUALITY_THRESHOLD_SOURCE}")
    lines.append(
        "  Note: Heuristic descriptor — not a validated gait-quality instrument. "
        "Pose coverage, landmark visibility, cycle evidence, spectral reliability, "
        "and physical stability are separate layers."
    )
    return "\n".join(lines)


def format_symmetry_session_details(gui: Any) -> str | None:
    """Live session block for Gait Symmetry: formula, inputs, missing, threshold."""
    from stablewalk.analysis.biomechanical.symmetry_metrics import (
        SYMMETRY_COMPONENT_WEIGHTS,
        SYMMETRY_FORMULA,
        SYMMETRY_THRESHOLD_SOURCE,
    )

    ba = getattr(gui, "_biomech_analysis", None)
    sym = getattr(ba, "symmetry", None) if ba is not None else None
    if sym is None:
        return None
    overall = getattr(sym, "overall_symmetry_pct", None)
    score = getattr(overall, "value", None) if overall is not None else None
    lines: list[str] = [
        "Session instance (from ba.symmetry — same source as Overview / Biomechanics):",
    ]
    if score is not None:
        lines.append(f"  Overall: {float(score):.0f}%")
    conf_txt = _fmt_confidence(getattr(overall, "confidence", None) if overall else None)
    if conf_txt:
        lines.append(f"  Confidence: {conf_txt}")
    lines.append(f"  Formula: {SYMMETRY_FORMULA}")
    lines.append("  Contributing components (index · weight):")
    present: list[str] = []
    for attr, weight in SYMMETRY_COMPONENT_WEIGHTS.items():
        metric = getattr(sym, attr, None)
        value = getattr(metric, "value", None) if metric is not None else None
        if value is None:
            continue
        present.append(attr)
        note = str(getattr(metric, "note", "") or "")
        extra = f" — {note}" if note else ""
        lines.append(f"    • {attr}: {float(value):.3f} (w={weight:.1f}){extra}")
    if not present:
        lines.append("    • none")
    missing = [name for name in SYMMETRY_COMPONENT_WEIGHTS if name not in present]
    lines.append("  Missing inputs (omitted; weights renormalized over active):")
    if missing:
        for name in missing:
            lines.append(
                f"    • {name} (nominal w={SYMMETRY_COMPONENT_WEIGHTS[name]:.1f})"
            )
    else:
        lines.append("    • none")
    warnings = list(getattr(sym, "warnings", None) or [])
    if warnings:
        lines.append("  Notes:")
        for warn in warnings[:4]:
            lines.append(f"    • {warn}")
    lines.append(f"  Threshold source: {SYMMETRY_THRESHOLD_SOURCE}")
    lines.append(
        "  Note: Derived kinematic composite — not Robinson SI/ASI, not knee-only "
        "SI_norm, and not a clinical symmetry grade. Pose coverage, landmark "
        "visibility, spectral reliability, and biomechanical validity are separate."
    )
    return "\n".join(lines)


def format_pelvis_stability_session_details(gui: Any) -> str | None:
    """Live pelvis-stability domain inputs. Display only — no rescoring."""
    result = getattr(gui, "_biomech", None)
    metric = None
    if result is not None and hasattr(result, "metric"):
        try:
            metric = result.metric("pelvis_stability")
        except Exception:
            metric = None
    if metric is None or getattr(metric, "score", None) is None:
        return None
    values = getattr(metric, "values", None) or {}
    lines = [
        "Session instance (pelvis_stability domain):",
        f"  Score: {float(metric.score):.1f}/100",
        "  Formula: score = mean of scored gait-frame submetrics "
        "(mediolateral sway, vertical oscillation, ML acceleration CV, lateral jerk). "
        "Each submetric uses a lower-is-better heuristic band. Not a clinical grade.",
        "  Inputs:",
    ]
    labels = (
        ("lateral_sway_ratio", "mediolateral sway / hip width"),
        ("normalized_pelvis_sway", "mediolateral sway / hip width"),
        ("pelvis_vertical_range", "vertical range"),
        ("pelvis_acceleration_variability", "ML acceleration CV"),
        ("pelvis_velocity_variability", "ML velocity CV"),
        ("pelvis_jerk_metric", "lateral jerk"),
        ("tracked_frame_ratio", "tracked-frame ratio"),
    )
    seen: set[str] = set()
    for key, label in labels:
        if key in seen or key not in values or values[key] is None:
            continue
        seen.add(key)
        try:
            lines.append(f"    • {label}: {float(values[key]):.4g}")
        except (TypeError, ValueError):
            lines.append(f"    • {label}: {values[key]}")
    if len(lines) == 4:
        lines.append("    • raw submetrics were not stored on this metric")
    lines.append(
        "  Note: Pose-derived heuristic. The displayed 0–100 band has no "
        "documented clinical Acceptable threshold."
    )
    return "\n".join(lines)


# Overview KPI Details popup — short copy + already-computed session values only.
_OVERVIEW_KPI_ALIASES: dict[str, str] = {
    "gait_quality": "gait_quality",
    "gait_quality_composite": "gait_quality",
    "stability_state": "stability_state",
    "stability_margin": "stability_state",
    "stability": "stability_state",
    "symmetry": "symmetry",
    "symmetry_pct": "symmetry",
    "cadence": "cadence",
    "cadence_spm": "cadence",
}

_OVERVIEW_KPI_TITLES: dict[str, str] = {
    "gait_quality": "Gait quality",
    "stability_state": "Stability",
    "symmetry": "Symmetry",
    "cadence": "Cadence",
}

_OVERVIEW_KPI_WHAT: dict[str, str] = {
    "gait_quality": (
        "Heuristic pose-derived session descriptor (0–100) from the gait-quality "
        "components that were available. Missing inputs are omitted and weights "
        "renormalized. Not a validated instrument. Not a clinical diagnosis. "
        "Independent of source video labels (Normal / Abnormal / Fast Walking)."
    ),
    "stability_state": (
        "Physical COM–BoS stability is unreportable from monocular video. "
        "Cards show N/A with the analysis reason. The Balance plot, when shown, "
        "is an unvalidated signed-distance preview without Normal/Stable/Unstable "
        "classification. Not Hof margin of stability."
    ),
    "symmetry": (
        "Left–right similarity of pose-derived gait timing and motion "
        "(100% = equal sides). Weighted composite of available bilateral indices. "
        "Not a clinical symmetry index. Independent of source video labels."
    ),
    "cadence": (
        "Walking step rate calculated from detected step times "
        "(60 / mean step time). Not a metronome or IMU count."
    ),
}


def resolve_overview_kpi_key(key: str | None) -> str | None:
    """Map a Details control key to an Overview KPI id, or None."""
    if not key:
        return None
    return _OVERVIEW_KPI_ALIASES.get(str(key).strip())


def _fmt_confidence(raw: Any) -> str | None:
    if raw is None:
        return None
    try:
        conf = float(raw)
    except (TypeError, ValueError):
        return None
    if conf <= 1.0:
        return f"{conf:.0%}"
    return f"{conf:.0f}%"


def _metric_value(obj: Any) -> float | None:
    if obj is None:
        return None
    value = getattr(obj, "value", obj)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_overview_kpi_detail(gui: Any, key: str | None) -> dict[str, str]:
    """
    Lightweight Details popup payload for Overview Quality / Stability /
    Symmetry / Cadence. Uses already-computed session fields only.
    """
    kpi = resolve_overview_kpi_key(key)
    title = _OVERVIEW_KPI_TITLES.get(kpi or "", "Metric")
    what = _OVERVIEW_KPI_WHAT.get(kpi or "", "Pose-derived session metric.")
    if kpi is None:
        return {
            "title": title,
            "body": "No short explanation is registered for this control.",
        }

    fields = extract_overview_summary_fields(gui)
    inputs = primary_finding_inputs(gui)
    display, _tip = fields.get(kpi, (NOT_AVAILABLE, ""))
    cadence_spm = inputs.get("cadence_spm")
    try:
        cadence_f = float(cadence_spm) if cadence_spm is not None else None
    except (TypeError, ValueError):
        cadence_f = None
    status = status_label_for_overview(
        kpi,
        classify_overview_metric(
            kpi,
            display=display,
            gait_quality_score=inputs.get("gait_quality_score"),
            symmetry_pct=inputs.get("symmetry_pct"),
            cadence_spm=cadence_f,
            stable_pct=inputs.get("stable_pct"),
            stability_margin_m=inputs.get("stability_margin_m"),
        ),
        display=display,
        cadence_spm=cadence_f,
    )
    provenance = card_provenance(kpi)

    lines: list[str] = [what, "", f"This session: {display}"]
    if status:
        lines.append(f"Status: {status}")
    if provenance:
        lines.append(f"Source: {provenance}")

    ba = getattr(gui, "_biomech_analysis", None)

    if kpi == "gait_quality":
        gq = getattr(ba, "gait_quality", None) if ba is not None else None
        conf_txt = _fmt_confidence(getattr(gq, "confidence", None) if gq else None)
        if conf_txt:
            lines.append(f"Confidence: {conf_txt}")
        factors = list(getattr(gq, "dominant_factors", None) or [])[:2]
        if factors:
            lines.append("Main drivers: " + "; ".join(str(f) for f in factors))
        components = getattr(gq, "components", None) or {}
        if components:
            bits = [
                f"{name} {float(value):.1f} (w={_GAIT_QUALITY_WEIGHTS.get(str(name), 0):.2f})"
                for name, value in sorted(components.items())
            ]
            lines.append("Inputs: " + "; ".join(bits))
        formula = str(getattr(gq, "formula", None) or "").strip()
        if formula:
            lines.append(f"Formula: {formula}")
        else:
            lines.append(
                "Formula: score = Σ(component·weight) / Σ(weight) over active components; "
                "missing terms omitted and weights renormalized."
            )
        explanation = str(getattr(gq, "explanation", None) or "").strip()
        missing = [
            name for name in _GAIT_QUALITY_WEIGHTS if name not in components
        ]
        if missing:
            lines.append("Missing inputs: " + ", ".join(missing))
        from stablewalk.analysis.quality_layers import GAIT_QUALITY_THRESHOLD_SOURCE

        lines.append(f"Threshold source: {GAIT_QUALITY_THRESHOLD_SOURCE}")
        if explanation:
            first = explanation.split(". ", 1)[0].strip()
            if first and len(first) <= 160:
                if not first.endswith("."):
                    first += "."
                lines.append(first)

    elif kpi == "stability_state":
        sm = getattr(ba, "stability_margin", None) if ba is not None else None
        contract = None
        if sm is not None:
            try:
                from stablewalk.analysis.biomechanical.stability_margin import (
                    balance_display_contract,
                )

                contract = balance_display_contract(sm)
            except Exception:
                contract = None
        if contract is not None:
            lines.append(f"Unit: {contract.unit}")
            lines.append(f"Coordinate frame: {contract.coordinate_frame}")
            lines.append(f"Method: {contract.method}")
            lines.append(f"Validity: {contract.validity}")
            if contract.reason:
                lines.append(contract.reason)
            if contract.display_mode == "unvalidated_preview":
                lines.append(
                    "Balance plot: unvalidated signed-distance preview. "
                    "Summary cards remain N/A."
                )
        elif inputs.get("stability_margin_m") is not None:
            lines.append(
                f"Mean margin: {float(inputs['stability_margin_m']):+.3f} m"
            )

    elif kpi == "symmetry":
        from stablewalk.analysis.biomechanical.symmetry_metrics import (
            SYMMETRY_COMPONENT_WEIGHTS,
            SYMMETRY_FORMULA,
            SYMMETRY_THRESHOLD_SOURCE,
        )

        sym = getattr(ba, "symmetry", None) if ba is not None else None
        overall = getattr(sym, "overall_symmetry_pct", None) if sym is not None else None
        conf_txt = _fmt_confidence(getattr(overall, "confidence", None) if overall else None)
        if conf_txt:
            lines.append(f"Confidence: {conf_txt}")
        present: list[str] = []
        contrib: list[str] = []
        if sym is not None:
            for attr, weight in SYMMETRY_COMPONENT_WEIGHTS.items():
                val = _metric_value(getattr(sym, attr, None))
                if val is None:
                    continue
                present.append(attr)
                shown = f"{val:.0%}" if 0.0 <= val <= 1.5 else f"{val:.1f}"
                contrib.append(f"{attr} {shown} (w={weight:.1f})")
        if contrib:
            lines.append("Contributing components: " + "; ".join(contrib))
        missing = [name for name in SYMMETRY_COMPONENT_WEIGHTS if name not in present]
        lines.append(f"Formula: {SYMMETRY_FORMULA}")
        if missing:
            lines.append("Missing inputs: " + ", ".join(missing))
        else:
            lines.append("Missing inputs: none")
        lines.append(f"Threshold source: {SYMMETRY_THRESHOLD_SOURCE}")
        lines.append(
            "Overall score is a weighted composite — not knee SI_norm, "
            "signed SI, or ASI. Display bands are heuristic, not clinical "
            "normality, and independent of source video labels."
        )

    elif kpi == "cadence":
        gm = getattr(ba, "gait_metrics", None) if ba is not None else None
        cfield = getattr(gm, "cadence", None) if gm is not None else None
        conf_txt = _fmt_confidence(getattr(cfield, "confidence", None) if cfield else None)
        if conf_txt:
            lines.append(f"Confidence: {conf_txt}")
        gait = getattr(gui, "_gait_cycle", None)
        metrics = getattr(gait, "metrics", None) if gait is not None else None
        step_t = getattr(metrics, "step_time_s", None) if metrics is not None else None
        if step_t is not None:
            lines.append(f"Mean step time: {float(step_t):.2f} s")
        usable = inputs.get("usable_cycles")
        detected = inputs.get("detected_cycles")
        if usable is not None or detected is not None:
            u = "—" if usable is None else str(int(usable))
            d = "—" if detected is None else str(int(detected))
            lines.append(f"Gait cycles (usable / detected): {u} / {d}")

    lines.append("")
    lines.append("Research/education estimate from this recording — not a clinical result.")
    return {"title": title, "body": "\n".join(lines)}
