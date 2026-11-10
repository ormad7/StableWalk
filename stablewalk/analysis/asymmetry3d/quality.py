"""Reconstruction quality checks and unusual L/R time sections.

These are kinematic sanity checks on the existing 3D recording. They are
not a clinical diagnosis and do not claim calibrated physical units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stablewalk.analysis.asymmetry3d.events import GaitEventSummary
from stablewalk.analysis.asymmetry3d.kinematics import (
    BONE_LENGTH_REL_TOL,
    KinematicBundle,
    PROVENANCE_INFERRED,
)
from stablewalk.analysis.asymmetry3d.laterality import assess_laterality
from stablewalk.analysis.asymmetry3d.trajectories import AlignedTrajectoryPair
from stablewalk.analysis.ground_reference import estimate_ground_plane
from stablewalk.models.gait_motion import GaitMotionRecording

_CV_FAIL = 0.25
_CV_WARN = 0.12
_HOMOLOGY_FAIL = 0.30
_HOMOLOGY_WARN = 0.15

_HKA = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
_LOWER_BONES = {
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
}
_RESIDUAL_MIN_DEG = 10.0
_MIN_INTERVAL_S = 0.08
_MAX_INTERVALS = 12


@dataclass
class QualityCheck:
    name: str
    label: str
    status: str  # pass | warn | fail | unavailable
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class UnusualInterval:
    start_s: float
    end_s: float
    phase_start_pct: float
    phase_end_pct: float
    channel: str
    peak_delta: float
    side: str
    start_frame: int | None = None
    end_frame: int | None = None
    note: str = (
        "Phase-matched left–right difference on ipsilateral HS–HS cycles. "
        "Not a clinical finding."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "phase_start_pct": self.phase_start_pct,
            "phase_end_pct": self.phase_end_pct,
            "channel": self.channel,
            "peak_delta": self.peak_delta,
            "side": self.side,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "note": self.note,
        }


@dataclass
class ReconstructionQuality:
    checks: list[QualityCheck] = field(default_factory=list)
    ground_reliable: bool = False
    floor_y: float | None = None
    unusual_intervals: list[UnusualInterval] = field(default_factory=list)
    unusual_phase_bands: list[tuple[float, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        parts = []
        for check in self.checks:
            mark = {"pass": "ok", "warn": "warn", "fail": "fail"}.get(check.status, "n/a")
            parts.append(f"{check.label}: {mark}")
        return " · ".join(parts) if parts else "Quality checks unavailable."

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [c.to_dict() for c in self.checks],
            "ground_reliable": self.ground_reliable,
            "floor_y": self.floor_y,
            "unusual_intervals": [u.to_dict() for u in self.unusual_intervals],
            "unusual_phase_bands": [list(b) for b in self.unusual_phase_bands],
            "warnings": list(self.warnings),
            "clinical": False,
        }


def _status(ok: bool, *, warn: bool = False, available: bool = True) -> str:
    if not available:
        return "unavailable"
    if ok and not warn:
        return "pass"
    if ok or warn:
        return "warn"
    return "fail"


def assess_reconstruction_quality(
    recording: GaitMotionRecording,
    kin: KinematicBundle,
    events: GaitEventSummary,
    aligned: list[AlignedTrajectoryPair],
) -> ReconstructionQuality:
    checks: list[QualityCheck] = []
    warnings: list[str] = []

    checks.append(_segment_length_check(kin))
    checks.append(_lr_identity_check(kin))
    checks.append(_temporal_check(kin))
    ground_check, ground_reliable, floor_y = _ground_check(recording, kin)
    checks.append(ground_check)
    checks.append(_hka_tracking_check(kin))

    intervals, bands = detect_unusual_lr_intervals(kin, events, aligned)
    if intervals:
        warnings.append(
            f"{len(intervals)} time section(s) with large phase-matched L/R "
            "differences — kinematic observation only."
        )
    for check in checks:
        if check.status in {"fail", "warn"}:
            warnings.append(f"{check.label}: {check.detail}")

    return ReconstructionQuality(
        checks=checks,
        ground_reliable=ground_reliable,
        floor_y=floor_y,
        unusual_intervals=intervals,
        unusual_phase_bands=bands,
        warnings=warnings,
    )


def _bone_cv(col: np.ndarray) -> float | None:
    finite = col[np.isfinite(col) & (col > 1e-6)]
    if finite.size < 4:
        return None
    med = float(np.median(finite))
    if med < 1e-6:
        return None
    return float(np.std(finite, ddof=0) / med)


def _segment_length_check(kin: KinematicBundle) -> QualityCheck:
    """Temporal consistency and L/R homology; table ratios are advisory.

    FAIL is reserved for bones that change length in time or disagree
    left vs right. Foreshortened monocular lengths vs Winter ratios are a
    warning, not a fail.
    """
    cvs: list[tuple[str, float]] = []
    medians: dict[tuple[str, str], float] = {}
    anthro: list[tuple[str, float]] = []
    for b, pair in enumerate(kin.bone_pairs):
        if pair not in _LOWER_BONES:
            continue
        col = np.asarray(kin.bone_length[:, b], dtype=float)
        med = float(np.nanmedian(col)) if np.isfinite(col).any() else float("nan")
        if np.isfinite(med) and med > 1e-6:
            medians[pair] = med
        cv = _bone_cv(col)
        if cv is not None:
            cvs.append((f"{pair[0]}–{pair[1]}", cv))
        tgt = (
            float(kin.bone_length_expected[b])
            if b < kin.bone_length_expected.size
            else float("nan")
        )
        if np.isfinite(med) and np.isfinite(tgt) and tgt > 1e-6:
            anthro.append((f"{pair[0]}–{pair[1]}", abs(med - tgt) / tgt))
    if not cvs and not medians:
        return QualityCheck(
            "segment_lengths",
            "Segment lengths",
            "unavailable",
            "Lower-limb bone lengths could not be measured.",
        )

    worst_cv = max((cv for _name, cv in cvs), default=0.0)
    homology: list[tuple[str, float]] = []
    for left_pair, right_pair, label in (
        (("left_hip", "left_knee"), ("right_hip", "right_knee"), "thigh"),
        (("left_knee", "left_ankle"), ("right_knee", "right_ankle"), "shank"),
    ):
        left_m = medians.get(left_pair)
        right_m = medians.get(right_pair)
        if left_m is None or right_m is None:
            continue
        denom = 0.5 * (left_m + right_m)
        if denom < 1e-6:
            continue
        homology.append((label, abs(left_m - right_m) / denom))
    worst_h = max((rel for _name, rel in homology), default=0.0)
    worst_anthro = max((rel for _name, rel in anthro), default=0.0)
    inferred = kin.provenance == PROVENANCE_INFERRED

    fail = worst_cv > _CV_FAIL or worst_h > _HOMOLOGY_FAIL
    if not inferred and worst_anthro > BONE_LENGTH_REL_TOL:
        fail = True
    warn = (not fail) and (
        worst_cv > _CV_WARN
        or worst_h > _HOMOLOGY_WARN
        or worst_anthro > 0.15
    )
    parts = []
    if cvs:
        parts.append(
            "length CV "
            + ", ".join(f"{name} {cv:.0%}" for name, cv in cvs)
        )
    if homology:
        parts.append(
            "L/R "
            + ", ".join(f"{name} {rel:.0%}" for name, rel in homology)
        )
    if anthro:
        kind = "advisory" if inferred else "calibrated"
        parts.append(
            f"{kind} vs stature-scaled ratios "
            + ", ".join(f"{name} {rel:.0%}" for name, rel in anthro)
        )
    if inferred and worst_anthro > 0.15:
        parts.append(
            "Monocular foreshortening can disagree with Winter ratios; "
            "that is not scored as a tracking failure."
        )
    stature = kin.stature
    if stature is not None:
        parts.append(f"observed stature ≈ {stature:.2f} ({kin.units})")
    return QualityCheck(
        "segment_lengths",
        "Segment lengths",
        _status(not fail, warn=warn),
        (
            "; ".join(parts) + ". Body-normalized, not tape-measure metres."
            if parts
            else "Lower-limb bone lengths could not be summarised."
        ),
    )


def _lr_identity_check(kin: KinematicBundle) -> QualityCheck:
    def _xyz(joint_id: str) -> np.ndarray | None:
        idx = kin.joint_index(joint_id)
        if idx is None:
            return None
        return np.asarray(kin.positions[:, idx, :], dtype=float)

    assessment = assess_laterality(
        _xyz("left_hip"),
        _xyz("right_hip"),
        left_knee=_xyz("left_knee"),
        right_knee=_xyz("right_knee"),
        left_ankle=_xyz("left_ankle"),
        right_ankle=_xyz("right_ankle"),
        head=_xyz("head") if kin.joint_index("head") is not None else _xyz("nose"),
        pelvis=_xyz("pelvis"),
    )
    return QualityCheck(
        "lr_identity",
        "Left/right joints",
        assessment.status,
        assessment.detail,
    )


def _temporal_check(kin: KinematicBundle) -> QualityCheck:
    times = np.asarray(kin.times_s, dtype=float)
    if times.size < 3:
        return QualityCheck(
            "temporal_continuity",
            "Temporal continuity",
            "unavailable",
            "Too few frames to assess continuity.",
        )
    dt = np.diff(times)
    expected = 1.0 / max(float(kin.fps), 1e-6)
    gaps = int(np.sum(dt > 1.75 * expected)) if dt.size else 0
    jumps = 0
    n = kin.n_frames
    for jid in _HKA:
        for axis in range(3):
            col = kin.series_for_joint(jid, axis)
            d = np.abs(np.diff(col))
            jumps += int(np.nansum(d > 0.20))
    jump_frac = jumps / max(n * 6, 1)
    ok = gaps == 0 and jump_frac < 0.08
    warn = gaps <= 2 and jump_frac < 0.15 and not ok
    return QualityCheck(
        "temporal_continuity",
        "Temporal continuity",
        _status(ok, warn=warn),
        (
            f"{gaps} snapshot gap(s) larger than one frame; "
            f"large joint jumps on {jump_frac:.0%} of lower-limb samples. "
            "Short gaps are interpolated; long gaps stay missing."
        ),
    )


def _ground_check(
    recording: GaitMotionRecording,
    kin: KinematicBundle,
) -> tuple[QualityCheck, bool, float | None]:
    try:
        plane = estimate_ground_plane(recording, float(max(kin.n_frames - 1, 0)))
    except Exception:
        plane = None
    if plane is None:
        return (
            QualityCheck(
                "ground_reference",
                "Ground reference",
                "unavailable",
                "No stable floor plane from heel/toe heights. "
                "Foot-clearance-to-ground is withheld.",
            ),
            False,
            None,
        )
    std = plane.floor_candidate_std
    span = plane.body_vertical_span
    noisy = std is not None and float(std) > 0.08
    odd_span = span is not None and (float(span) < 0.30 or float(span) > 2.50)
    reliable = (not noisy) and (not odd_span)
    status = "pass" if reliable else "warn"
    reasons: list[str] = []
    if noisy:
        reasons.append(
            f"heel/toe height scatter σ={float(std):.3f} > 0.08 "
            f"({kin.units})"
        )
    if odd_span:
        reasons.append(
            f"body vertical span {float(span):.2f} is outside 0.30–2.50 "
            f"({kin.units})"
        )
    if reliable:
        usability = (
            "Stance-weighted heel/toe plane is usable for estimated clearance."
        )
    else:
        usability = (
            "Floor estimate is noisy or implausible"
            + (f" ({'; '.join(reasons)})" if reasons else "")
            + " — ground clearance withheld. No physical calibration was invented."
        )
    return (
        QualityCheck(
            "ground_reference",
            "Ground reference",
            status,
            (
                f"Floor Y ≈ {plane.floor_y:.3f} ({kin.units}). "
                + usability
                + " Estimated from stance-weighted heel/toe heights — "
                "not a laboratory force-plate origin."
            ),
        ),
        reliable,
        float(plane.floor_y),
    )


def _hka_tracking_check(kin: KinematicBundle) -> QualityCheck:
    n = max(kin.n_frames, 1)
    vis_ok = []
    for jid in _HKA:
        idx = kin.joint_index(jid)
        if idx is None:
            vis_ok.append(0.0)
            continue
        vis_ok.append(float(np.mean(kin.visibility[:, idx] >= 0.50)))
    mean_vis = float(np.mean(vis_ok)) if vis_ok else 0.0
    missing = [jid for jid, frac in zip(_HKA, vis_ok) if frac < 0.40]
    ok = mean_vis >= 0.70 and not missing
    warn = mean_vis >= 0.50 and not ok
    detail = (
        f"Hip/knee/ankle visibility ≥0.50 on {mean_vis:.0%} of frames."
    )
    if missing:
        detail += " Weak: " + ", ".join(missing) + "."
    return QualityCheck(
        "hka_tracking",
        "Hip/knee/ankle tracking",
        _status(ok, warn=warn, available=bool(vis_ok)),
        detail,
    )


def detect_unusual_lr_intervals(
    kin: KinematicBundle,
    events: GaitEventSummary,
    aligned: list[AlignedTrajectoryPair],
) -> tuple[list[UnusualInterval], list[tuple[float, float]]]:
    """Time windows where phase-matched L/R residual is large.

    Residual is the ipsilateral series minus the contralateral mean curve
    at the same gait-cycle percent — not L(t)−R(t) at the same clock time.
    """
    pair = next((p for p in aligned if p.name == "knee_flexion" and p.aligned), None)
    if pair is None or kin.times_s.size < 8:
        return [], []
    phase = np.asarray(pair.phase_pct, dtype=float)
    left_mean = np.asarray(pair.left_mean, dtype=float)
    right_mean = np.asarray(pair.right_mean, dtype=float)
    if phase.size < 8:
        return [], []
    curve_delta = np.abs(left_mean - right_mean)
    curve_delta = np.where(np.isfinite(curve_delta), curve_delta, 0.0)
    band_mask = curve_delta >= _RESIDUAL_MIN_DEG
    bands = _merge_phase_bands(phase, band_mask)

    left_vals = kin.series_for_dof("left_knee_flexion")
    right_vals = kin.series_for_dof("right_knee_flexion")
    times = np.asarray(kin.times_s, dtype=float)
    residuals: list[float] = []
    samples: list[tuple[float, float, str, float]] = []
    for side, cycles, values, contra_mean in (
        ("left", events.left_cycles, left_vals, right_mean),
        ("right", events.right_cycles, right_vals, left_mean),
    ):
        for cycle in cycles:
            dur = float(cycle.duration_s)
            if dur < 0.25:
                continue
            inside = (times >= cycle.start_time_s) & (times <= cycle.end_time_s)
            for t, v in zip(times[inside], values[inside]):
                if not np.isfinite(v):
                    continue
                pct = 100.0 * (float(t) - float(cycle.start_time_s)) / dur
                ref = float(np.interp(pct, phase, contra_mean))
                if not np.isfinite(ref):
                    continue
                res = abs(float(v) - ref)
                residuals.append(res)
                samples.append((float(t), res, side, pct))
    if not samples:
        return [], bands
    arr = np.asarray(residuals, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    thresh = max(_RESIDUAL_MIN_DEG, med + 1.5 * (1.4826 * mad if mad > 1e-9 else 8.0))
    flagged = [(t, res, side, pct) for t, res, side, pct in samples if res >= thresh]
    if not flagged:
        return [], bands
    flagged.sort(key=lambda item: item[0])
    intervals: list[UnusualInterval] = []
    run_t0, run_peak, run_side, run_p0 = flagged[0][0], flagged[0][1], flagged[0][2], flagged[0][3]
    run_t1, run_p1 = flagged[0][0], flagged[0][3]
    gap = 2.0 / max(float(kin.fps), 1e-6)
    for t, res, side, pct in flagged[1:]:
        if side == run_side and (t - run_t1) <= max(gap, 0.06):
            run_t1 = t
            run_p1 = pct
            run_peak = max(run_peak, res)
            continue
        if (run_t1 - run_t0) >= _MIN_INTERVAL_S:
            intervals.append(
                _interval_from_run(
                    times, run_t0, run_t1, run_p0, run_p1, run_peak, run_side
                )
            )
        run_t0, run_t1, run_peak, run_side, run_p0, run_p1 = t, t, res, side, pct, pct
    if (run_t1 - run_t0) >= _MIN_INTERVAL_S:
        intervals.append(
            _interval_from_run(
                times, run_t0, run_t1, run_p0, run_p1, run_peak, run_side
            )
        )
    intervals.sort(key=lambda item: item.peak_delta, reverse=True)
    return intervals[:_MAX_INTERVALS], bands


def _nearest_frame(times: np.ndarray, t: float) -> int | None:
    if times.size == 0:
        return None
    return int(np.argmin(np.abs(np.asarray(times, dtype=float) - float(t))))


def _interval_from_run(
    times: np.ndarray,
    t0: float,
    t1: float,
    p0: float,
    p1: float,
    peak: float,
    side: str,
) -> UnusualInterval:
    return UnusualInterval(
        start_s=float(t0),
        end_s=float(t1),
        phase_start_pct=min(float(p0), float(p1)),
        phase_end_pct=max(float(p0), float(p1)),
        channel="knee_flexion",
        peak_delta=float(peak),
        side=side,
        start_frame=_nearest_frame(times, t0),
        end_frame=_nearest_frame(times, t1),
    )


def resolve_unusual_interval_target(
    interval: UnusualInterval,
    recording: GaitMotionRecording | None,
) -> tuple[int | None, float]:
    """Map an unusual interval to ``(frame_index, time_s)`` on the recording.

    Prefers the stored start frame when that snapshot's clock is still
    within one-and-a-half frames of ``start_s``. Otherwise uses the
    nearest snapshot time. Does not invent a different video.
    """
    target_t = float(interval.start_s)
    stored = interval.start_frame
    if recording is None or int(getattr(recording, "frame_count", 0) or 0) <= 0:
        return (None if stored is None else int(stored), target_t)
    fps = float(getattr(recording, "fps", 0.0) or 0.0)
    tol = 1.5 / max(fps, 1.0)
    if stored is not None:
        snap = recording.snapshot_at(int(stored))
        if snap is not None and abs(float(snap.time_s) - target_t) <= tol:
            return int(stored), float(snap.time_s)
    best_i = 0
    best_dt = 1e9
    for i in range(int(recording.frame_count)):
        snap = recording.snapshot_at(i)
        if snap is None:
            continue
        dt = abs(float(snap.time_s) - target_t)
        if dt < best_dt:
            best_dt = dt
            best_i = i
    snap = recording.snapshot_at(best_i)
    return best_i, float(snap.time_s) if snap is not None else target_t


def _merge_phase_bands(
    phase: np.ndarray,
    mask: np.ndarray,
) -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    i = 0
    n = int(mask.size)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        bands.append((float(phase[i]), float(phase[j - 1])))
        i = j
    return bands
