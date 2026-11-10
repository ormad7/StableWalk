"""
Pipeline validation quality from artifacts and analysis results actually present.

Never invents ground-truth comparisons. OpenSim IK marker residuals and pose-MOT
vs IK-MOT knee agreement are reported only when those files exist on disk.
Pose knee columns are converted with the project’s documented interior→flexion
map before comparison; results are coordinate agreement, not clinical validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from stablewalk import config
from stablewalk.analysis.gait_feature_analysis import read_opensim_mot_timeseries

ValidationQuality = Literal[
    "not_assessed",
    "insufficient_data",
    "poor",
    "limited",
    "acceptable",
]

QUALITY_LABEL: dict[ValidationQuality, str] = {
    "not_assessed": "Not assessed",
    "insufficient_data": "Insufficient data",
    "poor": "Poor",
    "limited": "Limited",
    "acceptable": "Acceptable",
}

# Engineering residual bands for video-based IK (meters). Optical mocap is typically
# millimetre-scale; MediaPipe-driven IK is expected to be centimetre-scale or worse.
_IK_RMS_ACCEPTABLE_M = 0.05
_IK_RMS_LIMITED_M = 0.15

# Knee flexion coordinate agreement after interior→flexion conversion (degrees).
_KNEE_RMSE_ACCEPTABLE_DEG = 15.0
_KNEE_RMSE_LIMITED_DEG = 35.0

_KNEE_COLS_L = ("knee_angle_l", "knee_l", "knee_flexion_l")
_KNEE_COLS_R = ("knee_angle_r", "knee_r", "knee_flexion_r")

_COORD_AGREEMENT_DISCLAIMER = (
    "Coordinate agreement only (pose MOT knee interior→flexion vs OpenSim IK). "
    "Not clinical validation; large residuals may reflect convention mismatch "
    "and/or weak marker constraints."
)


@dataclass(frozen=True)
class IkMarkerErrorSummary:
    """Parsed OpenSim IK marker-error ``.sto`` summary."""

    path: str
    n_frames: int
    mean_rms_m: float
    max_rms_m: float
    mean_max_error_m: float
    peak_max_error_m: float
    quality: ValidationQuality
    warnings: tuple[str, ...] = ()
    fail_threshold_m: float = _IK_RMS_LIMITED_M
    n_failed_frames: int = 0
    failed_frames: tuple[dict[str, Any], ...] = ()
    highest_error_frames: tuple[dict[str, Any], ...] = ()

    def summary_line(self) -> str:
        failed = (
            f"; {self.n_failed_frames} frame(s) above "
            f"{self.fail_threshold_m * 100:.0f} cm RMS"
            if self.n_failed_frames
            else ""
        )
        return (
            f"IK marker RMS mean {self.mean_rms_m * 100:.1f} cm "
            f"(max {self.max_rms_m * 100:.1f} cm) over {self.n_frames} frames"
            f"{failed}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "quality_label": QUALITY_LABEL[self.quality],
            "summary": self.summary_line(),
            "note": "OpenSim IK marker residuals from on-disk .sto — not force-plate GT.",
        }


@dataclass(frozen=True)
class PoseIkKneeAgreement:
    """Pose-export MOT vs OpenSim IK MOT knee coordinate agreement."""

    pose_mot_path: str
    ik_mot_path: str
    n_compared: int
    rmse_left_deg: float | None
    rmse_right_deg: float | None
    rmse_mean_deg: float | None
    rom_pose_left_deg: float | None
    rom_ik_left_deg: float | None
    rom_pose_right_deg: float | None
    rom_ik_right_deg: float | None
    quality: ValidationQuality
    warnings: tuple[str, ...] = ()
    note: str = _COORD_AGREEMENT_DISCLAIMER

    def summary_line(self) -> str:
        if self.rmse_mean_deg is None:
            return "Pose vs IK knee agreement unavailable"
        return (
            f"Pose vs IK knee RMSE {self.rmse_mean_deg:.1f} deg "
            f"(L {self.rmse_left_deg:.1f} / R {self.rmse_right_deg:.1f}, "
            f"n={self.n_compared})"
            if self.rmse_left_deg is not None and self.rmse_right_deg is not None
            else f"Pose vs IK knee RMSE {self.rmse_mean_deg:.1f} deg (n={self.n_compared})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "quality_label": QUALITY_LABEL[self.quality],
            "summary": self.summary_line(),
        }


@dataclass(frozen=True)
class StageValidation:
    """Per monitored pipeline sub-step validation snapshot."""

    stage_key: str
    quality: ValidationQuality = "not_assessed"
    confidence: float | None = None
    warnings: tuple[str, ...] = ()
    summary: str = "No independent validation artifact for this stage."

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_key": self.stage_key,
            "quality": self.quality,
            "quality_label": QUALITY_LABEL[self.quality],
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "summary": self.summary,
        }


@dataclass
class PipelineValidationReport:
    """Session validation derived only from available inputs/artifacts."""

    stages: dict[str, StageValidation] = field(default_factory=dict)
    ik_marker_errors: IkMarkerErrorSummary | None = None
    pose_ik_agreement: PoseIkKneeAgreement | None = None
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "pipeline_validation",
            "ik_marker_errors": None
            if self.ik_marker_errors is None
            else self.ik_marker_errors.to_dict(),
            "pose_ik_agreement": None
            if self.pose_ik_agreement is None
            else self.pose_ik_agreement.to_dict(),
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }


def _quality_from_rms_m(mean_rms_m: float) -> ValidationQuality:
    if not np.isfinite(mean_rms_m) or mean_rms_m < 0:
        return "insufficient_data"
    if mean_rms_m <= _IK_RMS_ACCEPTABLE_M:
        return "acceptable"
    if mean_rms_m <= _IK_RMS_LIMITED_M:
        return "limited"
    return "poor"


def _quality_from_knee_rmse(rmse_deg: float) -> ValidationQuality:
    if not np.isfinite(rmse_deg) or rmse_deg < 0:
        return "insufficient_data"
    if rmse_deg <= _KNEE_RMSE_ACCEPTABLE_DEG:
        return "acceptable"
    if rmse_deg <= _KNEE_RMSE_LIMITED_DEG:
        return "limited"
    return "poor"


def _worst_quality(*qualities: ValidationQuality) -> ValidationQuality:
    order = ("not_assessed", "insufficient_data", "poor", "limited", "acceptable")
    # Prefer the most severe assessed quality (exclude not_assessed when others exist).
    assessed = [q for q in qualities if q != "not_assessed"]
    if not assessed:
        return "not_assessed"
    return min(assessed, key=lambda q: order.index(q) if q in order else 0)


def _interior_to_flexion_deg(interior: float) -> float:
    return float(180.0 - interior)


def _pick_column(mot: dict[str, np.ndarray], candidates: tuple[str, ...]) -> str | None:
    return next((c for c in candidates if c in mot), None)


def _series_rom(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        return None
    return float(np.nanmax(finite) - np.nanmin(finite))


def _rmse(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(mask)) < 4:
        return None
    diff = a[mask] - b[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def opensim_session_dir(run_name: str | None) -> Path | None:
    if not run_name:
        return None
    path = config.OPENSIM_DIR / str(run_name)
    return path if path.is_dir() else None


def find_ik_marker_errors_sto(session_dir: Path | None) -> Path | None:
    if session_dir is None or not session_dir.is_dir():
        return None
    matches = sorted(session_dir.glob("*_ik_marker_errors.sto"))
    return matches[0] if matches else None


def find_pose_mot(session_dir: Path | None, run_name: str | None) -> Path | None:
    if session_dir is None or not run_name:
        return None
    for candidate in (
        session_dir / f"{run_name}.mot",
        session_dir / f"{run_name}.csv",
    ):
        if candidate.is_file():
            return candidate
    return None


def find_ik_mot(session_dir: Path | None, run_name: str | None) -> Path | None:
    if session_dir is None or not run_name:
        return None
    path = session_dir / f"{run_name}_ik.mot"
    return path if path.is_file() else None


def parse_ik_marker_errors(sto_path: Path | str) -> IkMarkerErrorSummary | None:
    """Parse OpenSim IK marker-error Storage file when present."""
    path = Path(sto_path)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_end = 0
    for i, ln in enumerate(lines):
        if ln.lower() == "endheader":
            header_end = i + 1
            break
    if header_end >= len(lines):
        return None
    header = lines[header_end].split()
    needed = {"time", "marker_error_RMS", "marker_error_max"}
    if not needed.issubset(set(header)):
        return None
    cols: dict[str, list[float]] = {name: [] for name in header}
    for ln in lines[header_end + 1 :]:
        parts = ln.split()
        if len(parts) != len(header):
            continue
        try:
            for name, val in zip(header, parts):
                cols[name].append(float(val))
        except ValueError:
            continue
    rms = np.asarray(cols.get("marker_error_RMS", []), dtype=float)
    mx = np.asarray(cols.get("marker_error_max", []), dtype=float)
    times = np.asarray(cols.get("time", []), dtype=float)
    if rms.size < 2:
        return None
    mean_rms = float(np.nanmean(rms))
    max_rms = float(np.nanmax(rms))
    mean_max = float(np.nanmean(mx)) if mx.size else float("nan")
    peak_max = float(np.nanmax(mx)) if mx.size else float("nan")
    quality = _quality_from_rms_m(mean_rms)
    warnings: list[str] = []
    if quality == "poor":
        warnings.append(
            f"IK marker RMS mean {mean_rms * 100:.1f} cm exceeds {_IK_RMS_LIMITED_M * 100:.0f} cm "
            "engineering band for video-based IK."
        )
    elif quality == "limited":
        warnings.append(
            f"IK marker residuals are elevated (mean RMS {mean_rms * 100:.1f} cm); "
            "treat IK kinematics as experimental."
        )
    ranked: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for i in range(int(rms.size)):
        rms_i = float(rms[i]) if np.isfinite(rms[i]) else None
        max_i = float(mx[i]) if i < mx.size and np.isfinite(mx[i]) else None
        time_i = float(times[i]) if i < times.size and np.isfinite(times[i]) else None
        row = {
            "frame_index": i,
            "time_s": time_i,
            "marker_error_rms_m": rms_i,
            "marker_error_max_m": max_i,
        }
        ranked.append(row)
        if rms_i is None or rms_i > _IK_RMS_LIMITED_M:
            failed.append(
                {
                    **row,
                    "reason": (
                        "non_finite_rms"
                        if rms_i is None
                        else f"rms_exceeds_{_IK_RMS_LIMITED_M}m"
                    ),
                }
            )
    ranked.sort(
        key=lambda item: (
            item["marker_error_rms_m"] is None,
            -(item["marker_error_rms_m"] or 0.0),
        )
    )
    return IkMarkerErrorSummary(
        path=str(path),
        n_frames=int(rms.size),
        mean_rms_m=mean_rms,
        max_rms_m=max_rms,
        mean_max_error_m=mean_max,
        peak_max_error_m=peak_max,
        quality=quality,
        warnings=tuple(warnings),
        fail_threshold_m=_IK_RMS_LIMITED_M,
        n_failed_frames=len(failed),
        failed_frames=tuple(failed[:200]),
        highest_error_frames=tuple(ranked[:10]),
    )


def compare_pose_mot_to_ik_mot(
    pose_mot_path: Path | str,
    ik_mot_path: Path | str,
) -> PoseIkKneeAgreement | None:
    """
    Compare pose-export MOT knees to OpenSim IK MOT after interior→flexion.

    Returns None when either file is missing/unreadable. Does not invent values.
    """
    pose_path = Path(pose_mot_path)
    ik_path = Path(ik_mot_path)
    if not pose_path.is_file() or not ik_path.is_file():
        return None
    pose = read_opensim_mot_timeseries(pose_path)
    ik = read_opensim_mot_timeseries(ik_path)
    if pose is None or ik is None:
        return None
    t_pose = pose.get("time")
    t_ik = ik.get("time")
    if t_pose is None or t_ik is None or len(t_pose) < 4 or len(t_ik) < 4:
        return None

    warnings: list[str] = []
    rmse_parts: list[float] = []
    n_compared = 0
    rmse_l: float | None = None
    rmse_r: float | None = None

    t0, t1 = float(t_ik[0]), float(t_ik[-1])
    in_span = (t_pose >= t0) & (t_pose <= t1)

    def _side_rmse(cols: tuple[str, ...]) -> tuple[float | None, int, np.ndarray | None, np.ndarray | None]:
        c_pose = _pick_column(pose, cols)
        c_ik = _pick_column(ik, cols)
        if c_pose is None or c_ik is None:
            return None, 0, None, None
        pose_flex = np.asarray(
            [_interior_to_flexion_deg(float(v)) for v in pose[c_pose]],
            dtype=float,
        )
        ik_vals = np.asarray(ik[c_ik], dtype=float)
        ik_on_pose = np.interp(t_pose, t_ik, ik_vals, left=np.nan, right=np.nan)
        a = pose_flex.copy()
        b = ik_on_pose.copy()
        a[~in_span] = np.nan
        b[~in_span] = np.nan
        return _rmse(a, b), int(np.count_nonzero(np.isfinite(a) & np.isfinite(b))), pose_flex, ik_vals

    left_rmse, left_n, pose_l, ik_l = _side_rmse(_KNEE_COLS_L)
    right_rmse, right_n, pose_r, ik_r = _side_rmse(_KNEE_COLS_R)
    if pose_l is None and pose_r is None:
        warnings.append("Missing knee columns in pose or IK MOT.")
    if left_rmse is not None:
        rmse_l = left_rmse
        rmse_parts.append(left_rmse)
    elif _pick_column(pose, _KNEE_COLS_L) is None or _pick_column(ik, _KNEE_COLS_L) is None:
        warnings.append("Missing knee column for left in pose or IK MOT.")
    if right_rmse is not None:
        rmse_r = right_rmse
        rmse_parts.append(right_rmse)
    elif _pick_column(pose, _KNEE_COLS_R) is None or _pick_column(ik, _KNEE_COLS_R) is None:
        warnings.append("Missing knee column for right in pose or IK MOT.")
    n_compared = max(left_n, right_n)

    rom_pose_l = _series_rom(pose_l) if pose_l is not None else None
    rom_ik_l = _series_rom(ik_l) if ik_l is not None else None
    rom_pose_r = _series_rom(pose_r) if pose_r is not None else None
    rom_ik_r = _series_rom(ik_r) if ik_r is not None else None

    if not rmse_parts:
        return PoseIkKneeAgreement(
            pose_mot_path=str(pose_path),
            ik_mot_path=str(ik_path),
            n_compared=0,
            rmse_left_deg=None,
            rmse_right_deg=None,
            rmse_mean_deg=None,
            rom_pose_left_deg=rom_pose_l,
            rom_ik_left_deg=rom_ik_l,
            rom_pose_right_deg=rom_pose_r,
            rom_ik_right_deg=rom_ik_r,
            quality="insufficient_data",
            warnings=tuple(warnings) or ("Insufficient overlapping knee samples for RMSE.",),
            note=_COORD_AGREEMENT_DISCLAIMER,
        )

    rmse_mean = float(np.mean(rmse_parts))
    quality = _quality_from_knee_rmse(rmse_mean)
    if quality == "poor":
        warnings.append(
            f"Pose vs IK knee RMSE {rmse_mean:.1f} deg exceeds {_KNEE_RMSE_LIMITED_DEG:.0f} deg "
            "coordinate-agreement band."
        )
    elif quality == "limited":
        warnings.append(
            f"Pose vs IK knee RMSE {rmse_mean:.1f} deg — limited coordinate agreement."
        )
    warnings.append(_COORD_AGREEMENT_DISCLAIMER)
    return PoseIkKneeAgreement(
        pose_mot_path=str(pose_path),
        ik_mot_path=str(ik_path),
        n_compared=n_compared,
        rmse_left_deg=rmse_l,
        rmse_right_deg=rmse_r,
        rmse_mean_deg=rmse_mean,
        rom_pose_left_deg=rom_pose_l,
        rom_ik_left_deg=rom_ik_l,
        rom_pose_right_deg=rom_pose_r,
        rom_ik_right_deg=rom_ik_r,
        quality=quality,
        warnings=tuple(dict.fromkeys(warnings)),
        note=_COORD_AGREEMENT_DISCLAIMER,
    )


def _stage(
    key: str,
    *,
    quality: ValidationQuality = "not_assessed",
    confidence: float | None = None,
    warnings: tuple[str, ...] | list[str] = (),
    summary: str = "No independent validation artifact for this stage.",
) -> StageValidation:
    return StageValidation(
        stage_key=key,
        quality=quality,
        confidence=confidence,
        warnings=tuple(warnings),
        summary=summary,
    )


def _quality_from_confidence(confidence: float) -> ValidationQuality:
    if confidence >= 0.7:
        return "acceptable"
    if confidence >= 0.45:
        return "limited"
    if confidence > 0.0:
        return "poor"
    return "insufficient_data"


def assess_pipeline_validation(ctx: Any) -> PipelineValidationReport:
    """
    Build per-stage validation from ``PipelineStatusContext`` and on-disk OpenSim files.

    ``ctx`` is typed loosely to avoid a hard import cycle with monitoring.
    """
    report = PipelineValidationReport()
    stages: dict[str, StageValidation] = {}

    run_name = getattr(ctx, "run_name", None)
    session_dir = opensim_session_dir(run_name)

    # --- Pose detection coverage (session facts only) ---
    detected = int(getattr(ctx, "detected_frames", 0) or 0)
    total = int(getattr(ctx, "total_frames", 0) or 0)
    if total > 0:
        ratio = detected / total
        stages["pose_mediapipe"] = _stage(
            "pose_mediapipe",
            quality=_quality_from_confidence(ratio),
            confidence=ratio,
            warnings=()
            if ratio >= 0.7
            else (f"Pose detection coverage {detected}/{total} ({ratio:.0%}).",),
            summary=f"Pose detection coverage {detected}/{total} frames ({ratio:.0%})",
        )
    else:
        stages["pose_mediapipe"] = _stage("pose_mediapipe")

    # --- Biomechanics from executed analysis ---
    contact = getattr(ctx, "contact", None)
    cycles = getattr(ctx, "cycles", None)
    biomech = getattr(ctx, "biomech", None)
    vgrf = getattr(ctx, "estimated_vgrf", None)

    contact_conf = None
    if contact is not None and getattr(contact, "metrics", None) is not None:
        contact_conf = float(contact.metrics.contact_confidence)
    elif cycles is not None and getattr(cycles, "metrics", None) is not None:
        contact_conf = float(cycles.metrics.contact_confidence)

    if contact_conf is not None:
        cq = _quality_from_confidence(contact_conf)
        cwarn: list[str] = []
        if cq in ("poor", "limited", "insufficient_data"):
            cwarn.append(f"Foot contact confidence {contact_conf:.0%}.")
        stages["bio_contact_mask"] = _stage(
            "bio_contact_mask",
            quality=cq,
            confidence=contact_conf,
            warnings=cwarn,
            summary=f"Foot contact confidence {contact_conf:.0%}",
        )
        for key, label in (
            ("bio_heel_strike", "Heel-strike"),
            ("bio_toe_off", "Toe-off"),
            ("bio_cadence", "Cadence"),
        ):
            stages[key] = _stage(
                key,
                quality=cq,
                confidence=contact_conf,
                warnings=tuple(cwarn),
                summary=f"{label} inherits contact confidence {contact_conf:.0%}",
            )
    else:
        for key in ("bio_contact_mask", "bio_heel_strike", "bio_toe_off", "bio_cadence"):
            stages[key] = _stage(key)

    vq = getattr(biomech, "video_quality", None) if biomech is not None else None
    if vq is not None:
        score = float(vq.overall_quality_score) / 100.0
        prev = stages.get("pose_mediapipe", _stage("pose_mediapipe"))
        prev_conf = prev.confidence
        conf = min(prev_conf, score) if prev_conf is not None else score
        if prev.summary == "No independent validation artifact for this stage.":
            summary = f"Video/pose quality score {vq.overall_quality_score:.0f}/100"
        else:
            summary = (
                f"{prev.summary}; video/pose quality score "
                f"{vq.overall_quality_score:.0f}/100"
            )
        stages["pose_mediapipe"] = _stage(
            "pose_mediapipe",
            quality=_worst_quality(prev.quality, _quality_from_confidence(score)),
            confidence=conf,
            warnings=tuple(dict.fromkeys([*prev.warnings, *list(vq.warnings[:4])])),
            summary=summary,
        )

    if biomech is not None and biomech.center_of_mass and biomech.center_of_mass.per_frame:
        bw = tuple(
            w
            for w in (biomech.warnings or [])
            if "COM" in w.upper() or "mass" in w.lower()
        )[:3]
        stages["bio_com"] = _stage(
            "bio_com",
            quality="limited",
            confidence=contact_conf,
            warnings=bw
            or (
                "COM is segment-weighted estimate from pose — not force-plate measured.",
            ),
            summary="COM estimated from pose segments (no force-plate validation in project).",
        )
    else:
        stages["bio_com"] = _stage("bio_com")

    if biomech is not None and biomech.base_of_support and biomech.base_of_support.per_frame:
        stages["bio_bos"] = _stage(
            "bio_bos",
            quality="limited",
            confidence=contact_conf,
            warnings=(
                "BoS derived from stance-foot landmarks — not pressure-mat validated.",
            ),
            summary="BoS from pose foot landmarks (no pressure-mat GT in project).",
        )
    else:
        stages["bio_bos"] = _stage("bio_bos")

    if vgrf is not None and getattr(vgrf, "confidence", None) is not None and len(vgrf.confidence) > 0:
        mean_v = float(np.nanmean(np.asarray(vgrf.confidence, dtype=float)))
        stages["bio_vgrf"] = _stage(
            "bio_vgrf",
            quality=_quality_from_confidence(mean_v),
            confidence=mean_v,
            warnings=(
                "Virtual GRF is model-estimated — not instrumented force-plate data.",
            ),
            summary=f"Virtual GRF estimator confidence {mean_v:.0%}",
        )
    else:
        stages["bio_vgrf"] = _stage(
            "bio_vgrf",
            summary="No virtual GRF confidence series available.",
        )

    # --- OpenSim mapping / coverage from session comparison ---
    map_pct = getattr(ctx, "opensim_mapping_percent", None)
    cov_pct = getattr(ctx, "opensim_coverage_percent", None)
    map_status = (getattr(ctx, "opensim_mapping_status", None) or "").lower()

    if map_pct is not None:
        conf = float(map_pct) / 100.0
        stages["osim_mapping"] = _stage(
            "osim_mapping",
            quality=_quality_from_confidence(conf),
            confidence=conf,
            warnings=()
            if map_status == "improved" and conf >= 0.85
            else (f"Marker mapping {map_pct:.0f}% ({map_status or 'unknown'}).",),
            summary=f"Marker mapping {map_pct:.0f}% ({map_status or 'status unknown'})",
        )
    else:
        stages["osim_mapping"] = _stage("osim_mapping")

    if cov_pct is not None:
        conf = float(cov_pct) / 100.0
        stages["osim_coverage"] = _stage(
            "osim_coverage",
            quality=_quality_from_confidence(conf),
            confidence=conf,
            warnings=()
            if conf >= 0.85
            else (f"OpenSim marker coverage {cov_pct:.0f}%.",),
            summary=f"OpenSim marker coverage {cov_pct:.0f}%",
        )
    else:
        stages["osim_coverage"] = _stage("osim_coverage")

    # --- On-disk IK residuals ---
    sto = find_ik_marker_errors_sto(session_dir)
    ik_errors = parse_ik_marker_errors(sto) if sto else None
    report.ik_marker_errors = ik_errors
    if ik_errors is not None:
        # Confidence proxy: invert residual band (1 at 0 m, 0 at / beyond limited band).
        conf = max(0.0, min(1.0, 1.0 - (ik_errors.mean_rms_m / _IK_RMS_LIMITED_M)))
        stages["osim_ik"] = _stage(
            "osim_ik",
            quality=ik_errors.quality,
            confidence=conf,
            warnings=ik_errors.warnings,
            summary=ik_errors.summary_line(),
        )
    else:
        stages["osim_ik"] = _stage(
            "osim_ik",
            summary="No IK marker-error .sto found for this session.",
        )

    # --- Pose MOT vs IK MOT (knees only, documented conversion) ---
    pose_mot = find_pose_mot(session_dir, run_name)
    ik_mot = find_ik_mot(session_dir, run_name)
    agreement = None
    if pose_mot is not None and ik_mot is not None:
        agreement = compare_pose_mot_to_ik_mot(pose_mot, ik_mot)
    report.pose_ik_agreement = agreement
    if agreement is not None and agreement.rmse_mean_deg is not None:
        conf = max(
            0.0,
            min(1.0, 1.0 - (agreement.rmse_mean_deg / _KNEE_RMSE_LIMITED_DEG)),
        )
        stages["osim_mot"] = _stage(
            "osim_mot",
            quality=agreement.quality,
            confidence=conf,
            warnings=agreement.warnings,
            summary=agreement.summary_line(),
        )
        # Also attach agreement note onto IK stage when both exist.
        prev = stages.get("osim_ik")
        if prev is not None and prev.quality != "not_assessed":
            stages["osim_ik"] = _stage(
                "osim_ik",
                quality=_worst_quality(prev.quality, agreement.quality),
                confidence=min(x for x in (prev.confidence, conf) if x is not None)
                if prev.confidence is not None
                else conf,
                warnings=tuple(dict.fromkeys([*prev.warnings, *agreement.warnings[:2]])),
                summary=f"{prev.summary}; {agreement.summary_line()}",
            )
    elif pose_mot is not None and ik_mot is None:
        stages["osim_mot"] = _stage(
            "osim_mot",
            quality="insufficient_data",
            summary="Pose MOT present; IK MOT missing — cannot compare.",
            warnings=("OpenSim IK MOT not found for pose vs IK comparison.",),
        )
    else:
        stages["osim_mot"] = _stage(
            "osim_mot",
            summary="Pose MOT and/or IK MOT not available for comparison.",
        )

    # Default not_assessed for remaining known keys when never set.
    for key in (
        "pose_smpl",
        "pose_romp",
        "motion_angles",
        "motion_velocities",
        "osim_sdk",
        "osim_model",
        "osim_trc",
    ):
        stages.setdefault(key, _stage(key))

    report.stages = stages
    return report


def aggregate_stage_validation(
    item_keys: tuple[str, ...] | list[str],
    report: PipelineValidationReport | None,
) -> StageValidation:
    """Combine sub-step validations for a diagram stage."""
    if report is None or not item_keys:
        return _stage("aggregate")
    matched = [report.stages[k] for k in item_keys if k in report.stages]
    if not matched:
        return _stage("aggregate")
    qualities = [m.quality for m in matched]
    quality = _worst_quality(*qualities)
    confs = [m.confidence for m in matched if m.confidence is not None]
    confidence = float(np.mean(confs)) if confs else None
    warnings: list[str] = []
    for m in matched:
        for w in m.warnings:
            if w not in warnings:
                warnings.append(w)
    summaries = [
        m.summary
        for m in matched
        if m.summary and m.summary != "No independent validation artifact for this stage."
    ]
    summary = "; ".join(summaries[:3]) if summaries else "No independent validation artifact for this stage."
    return _stage(
        "aggregate",
        quality=quality,
        confidence=confidence,
        warnings=warnings[:8],
        summary=summary,
    )


__all__ = [
    "QUALITY_LABEL",
    "IkMarkerErrorSummary",
    "PipelineValidationReport",
    "PoseIkKneeAgreement",
    "StageValidation",
    "ValidationQuality",
    "aggregate_stage_validation",
    "assess_pipeline_validation",
    "compare_pose_mot_to_ik_mot",
    "find_ik_marker_errors_sto",
    "find_ik_mot",
    "find_pose_mot",
    "opensim_session_dir",
    "parse_ik_marker_errors",
]
