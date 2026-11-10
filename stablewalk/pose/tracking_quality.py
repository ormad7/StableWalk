"""Pose-tracking quality: observed vs unreliable vs missing.

Does not interpolate missing frames and does not treat “a pose exists” as
usable gait tracking. Visibility is detector confidence, not accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# Display/audit cutoffs only — not gait-score thresholds.
VIS_OK = 0.45
VIS_DEGRADED = 0.70
_JUMP_FRAC_OF_BODY = 0.28
_JUMP_FRAC_OF_IMAGE = 0.10
_IDENTITY_JUMP_IMAGE = 0.12
_IDENTITY_BBOX_RATIO = 0.35
_ANATOMY_MARGIN = 0.015
_SWAP_SEP = 0.02

GAIT_JOINTS: tuple[str, ...] = (
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
)
FOOT_JOINTS: tuple[str, ...] = (
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
LOWER_LIMB_JOINTS: tuple[str, ...] = GAIT_JOINTS + FOOT_JOINTS

# Frame status — pose JSON itself is observed-or-missing; interpolation is
# a later analysis/display step and must be labeled separately.
STATUS_MISSING = "missing"
STATUS_OBSERVED = "observed"
STATUS_UNRELIABLE = "unreliable"
STATUS_INTERPOLATED = "interpolated"


@dataclass(frozen=True)
class JointVisStats:
    name: str
    n_observed: int
    n_low_vis: int
    n_degraded: int
    mean_visibility: float | None
    min_visibility: float | None
    p10_visibility: float | None
    median_visibility: float | None
    pct_ok: float


@dataclass
class FrameTrackQuality:
    frame_index: int
    timestamp_s: float
    status: str
    reasons: list[str] = field(default_factory=list)
    pelvis_xy: tuple[float, float] | None = None
    bbox_height: float | None = None
    laterality_sign: float | None = None
    interpolated: bool = False


@dataclass
class TrackingAudit:
    total_frames: int
    missing_frames: int
    observed_frames: int
    unreliable_frames: int
    interpolated_frames: int
    usable_gait_frames: int
    identity_switches: list[int]
    laterality_flips: list[int]
    jump_frames: list[int]
    anatomy_fail_frames: list[int]
    unreliable_intervals: list[tuple[int, int, str]]
    missing_intervals: list[tuple[int, int]]
    joint_stats: dict[str, JointVisStats]
    samples: dict[str, dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "missing_frames": self.missing_frames,
            "observed_frames": self.observed_frames,
            "unreliable_frames": self.unreliable_frames,
            "interpolated_frames": self.interpolated_frames,
            "usable_gait_frames": self.usable_gait_frames,
            "identity_switches": list(self.identity_switches),
            "laterality_flips": list(self.laterality_flips),
            "jump_frames": list(self.jump_frames),
            "anatomy_fail_frames": list(self.anatomy_fail_frames),
            "unreliable_intervals": [
                {"start": a, "end": b, "reason": reason}
                for a, b, reason in self.unreliable_intervals
            ],
            "missing_intervals": [
                {"start": a, "end": b} for a, b in self.missing_intervals
            ],
            "joint_stats": {
                name: {
                    "n_observed": st.n_observed,
                    "n_low_vis": st.n_low_vis,
                    "n_degraded": st.n_degraded,
                    "mean_visibility": st.mean_visibility,
                    "min_visibility": st.min_visibility,
                    "p10_visibility": st.p10_visibility,
                    "median_visibility": st.median_visibility,
                    "pct_ok": st.pct_ok,
                }
                for name, st in self.joint_stats.items()
            },
            "samples": self.samples,
            "notes": list(self.notes),
        }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _skeleton_visibility(frame: Any, name: str) -> float | None:
    sk = getattr(frame, "skeleton_3d", None) or {}
    joints = sk.get("joints") if isinstance(sk, dict) else None
    raw = joints.get(name) if isinstance(joints, dict) else None
    if not isinstance(raw, dict):
        return None
    return _finite(raw.get("visibility"))


def landmark_record(frame: Any, name: str) -> dict[str, float] | None:
    """Image-plane x/y from 2D keypoints only.

    skeleton_3d is pelvis-centered reconstructed 3D (mid_hip at the origin).
    Its x/y must not be used for jumps, identity, or image anatomy. Visibility
    may be copied from skeleton_3d when the compact 2D keypoint omitted it.
    """
    for kp in getattr(frame, "keypoints", None) or []:
        if getattr(kp, "name", "") != name:
            continue
        x = _finite(getattr(kp, "x", None))
        y = _finite(getattr(kp, "y", None))
        if x is None or y is None:
            break
        vis = _finite(getattr(kp, "visibility", None))
        if vis is None:
            vis = _skeleton_visibility(frame, name)
        return {"x": x, "y": y, "visibility": vis if vis is not None else float("nan")}
    return None


def landmark_visibility(frame: Any, name: str) -> float | None:
    rec = landmark_record(frame, name)
    if rec is None:
        return None
    return _finite(rec.get("visibility"))


def is_interpolated_frame(frame: Any) -> bool:
    """True only when the record itself is marked interpolated — never inferred."""
    if bool(getattr(frame, "interpolated", False)):
        return True
    meta = getattr(frame, "metadata", None)
    if isinstance(meta, dict) and meta.get("interpolated"):
        return True
    return False


def _body_height(frame: Any) -> float | None:
    top = None
    for name in ("nose", "left_shoulder", "right_shoulder"):
        rec = landmark_record(frame, name)
        if rec is None:
            continue
        top = rec["y"] if top is None else min(top, rec["y"])
    bottom = None
    for name in ("left_ankle", "right_ankle", "left_heel", "right_heel"):
        rec = landmark_record(frame, name)
        if rec is None:
            continue
        bottom = rec["y"] if bottom is None else max(bottom, rec["y"])
    if top is None or bottom is None or bottom <= top:
        return None
    return float(bottom - top)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _pelvis_xy(frame: Any) -> tuple[float, float] | None:
    left = landmark_record(frame, "left_hip")
    right = landmark_record(frame, "right_hip")
    if left is not None and right is not None:
        return ((left["x"] + right["x"]) * 0.5, (left["y"] + right["y"]) * 0.5)
    mid = landmark_record(frame, "mid_hip")
    if mid is not None:
        return (mid["x"], mid["y"])
    return None


def _laterality_sign(frame: Any) -> float | None:
    left = landmark_record(frame, "left_hip")
    right = landmark_record(frame, "right_hip")
    if left is None or right is None:
        return None
    delta = right["x"] - left["x"]
    if abs(delta) <= _SWAP_SEP:
        return 0.0
    return 1.0 if delta > 0 else -1.0


def _image_chain_ok(proximal: dict[str, float], distal: dict[str, float]) -> bool:
    """Image +Y is down: distal (ankle) must sit below proximal (hip/knee)."""
    return distal["y"] >= proximal["y"] + _ANATOMY_MARGIN


def anatomy_failures(frame: Any) -> list[str]:
    fails: list[str] = []
    for side in ("left", "right"):
        hip = landmark_record(frame, f"{side}_hip")
        knee = landmark_record(frame, f"{side}_knee")
        ankle = landmark_record(frame, f"{side}_ankle")
        if hip is None or knee is None or ankle is None:
            fails.append(f"{side} limb incomplete")
            continue
        if not _image_chain_ok(hip, knee):
            fails.append(f"{side} knee not below hip")
        if not _image_chain_ok(knee, ankle):
            fails.append(f"{side} ankle not below knee")
    return fails


def gait_joints_visible(frame: Any, *, min_vis: float = VIS_OK) -> bool:
    for name in GAIT_JOINTS:
        vis = landmark_visibility(frame, name)
        if vis is None or vis < min_vis:
            return False
    return True


def _intervals(indices: Iterable[int]) -> list[tuple[int, int]]:
    ordered = sorted(int(i) for i in indices)
    if not ordered:
        return []
    out: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for idx in ordered[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        out.append((start, prev))
        start = prev = idx
    out.append((start, prev))
    return out


def _reason_intervals(
    frames: list[FrameTrackQuality], status: str
) -> list[tuple[int, int, str]]:
    chosen = [fq for fq in frames if fq.status == status]
    if not chosen:
        return []
    grouped: list[tuple[int, int, str]] = []
    start = prev = chosen[0].frame_index
    reason = ", ".join(chosen[0].reasons) or status
    for fq in chosen[1:]:
        if fq.frame_index == prev + 1:
            prev = fq.frame_index
            continue
        grouped.append((start, prev, reason))
        start = prev = fq.frame_index
        reason = ", ".join(fq.reasons) or status
    grouped.append((start, prev, reason))
    return grouped


def _sample_frame(fq: FrameTrackQuality | None, frame: Any | None) -> dict[str, Any]:
    if fq is None or frame is None:
        return {}
    joints = {}
    for name in LOWER_LIMB_JOINTS:
        rec = landmark_record(frame, name)
        if rec is None:
            joints[name] = None
            continue
        joints[name] = {
            "x": round(rec["x"], 4),
            "y": round(rec["y"], 4),
            "visibility": None if _finite(rec["visibility"]) is None else round(float(rec["visibility"]), 3),
        }
    return {
        "frame_index": fq.frame_index,
        "timestamp_s": round(fq.timestamp_s, 3),
        "status": fq.status,
        "reasons": list(fq.reasons),
        "interpolated": fq.interpolated,
        "pelvis_xy": None if fq.pelvis_xy is None else [round(v, 4) for v in fq.pelvis_xy],
        "laterality_sign": fq.laterality_sign,
        "joints": joints,
    }


def audit_pose_sequence(sequence: Any) -> TrackingAudit:
    """Classify every frame. Missing stays missing; interpolation is never invented."""
    frames = list(getattr(sequence, "frames", None) or [])
    notes: list[str] = [
        "Pose records are observed detections or missing frames.",
        "Usable gait frames require hips, knees, and ankles with visibility "
        f">={VIS_OK:.2f}, anatomical image-plane order, and no sudden jump.",
        "A detected pose is not automatically usable.",
        "Joint visibility is detector confidence, not tracking accuracy.",
    ]
    qualities: list[FrameTrackQuality] = []
    prev_pelvis: tuple[float, float] | None = None
    prev_height: float | None = None
    prev_sign: float | None = None
    prev_detected_index: int | None = None
    identity_switches: list[int] = []
    laterality_flips: list[int] = []
    jump_frames: list[int] = []
    anatomy_fail_frames: list[int] = []

    vis_values: dict[str, list[float]] = {name: [] for name in LOWER_LIMB_JOINTS}

    for frame in frames:
        idx = int(getattr(frame, "frame_index", 0) or 0)
        ts = float(getattr(frame, "timestamp_s", 0.0) or 0.0)
        interpolated = is_interpolated_frame(frame)
        detected = bool(getattr(frame, "detected", False)) and bool(
            getattr(frame, "keypoints", None) or (getattr(frame, "skeleton_3d", None) or {}).get("joints")
        )
        if interpolated:
            qualities.append(
                FrameTrackQuality(
                    frame_index=idx,
                    timestamp_s=ts,
                    status=STATUS_INTERPOLATED,
                    reasons=["record marked interpolated"],
                    interpolated=True,
                )
            )
            continue
        if not detected:
            qualities.append(
                FrameTrackQuality(
                    frame_index=idx,
                    timestamp_s=ts,
                    status=STATUS_MISSING,
                    reasons=["no detected pose"],
                )
            )
            prev_pelvis = None
            continue

        reasons: list[str] = []
        for name in LOWER_LIMB_JOINTS:
            vis = landmark_visibility(frame, name)
            if vis is None:
                continue
            vis_values[name].append(vis)

        if not gait_joints_visible(frame):
            reasons.append("low hip/knee/ankle visibility")

        anatomy = anatomy_failures(frame)
        if anatomy:
            reasons.append("anatomical order")
            anatomy_fail_frames.append(idx)

        pelvis = _pelvis_xy(frame)
        height = _body_height(frame)
        sign = _laterality_sign(frame)
        jumped = False
        if pelvis is not None and prev_pelvis is not None and prev_detected_index is not None:
            gap = max(1, idx - prev_detected_index)
            dx = pelvis[0] - prev_pelvis[0]
            dy = pelvis[1] - prev_pelvis[1]
            dist = (dx * dx + dy * dy) ** 0.5
            per_frame = dist / gap
            limit = _JUMP_FRAC_OF_IMAGE
            if height is not None:
                limit = max(limit, height * _JUMP_FRAC_OF_BODY)
            if per_frame > limit:
                jumped = True
                reasons.append("sudden landmark jump")
                jump_frames.append(idx)
            identity = per_frame > _IDENTITY_JUMP_IMAGE
            if (
                identity
                and height is not None
                and prev_height is not None
                and prev_height > 1e-6
                and abs(height - prev_height) / prev_height > _IDENTITY_BBOX_RATIO
            ):
                identity_switches.append(idx)
                reasons.append("possible subject switch")
        if (
            sign is not None
            and prev_sign is not None
            and sign != 0.0
            and prev_sign != 0.0
            and sign != prev_sign
        ):
            laterality_flips.append(idx)
            reasons.append("left/right hip sign flip")

        status = STATUS_UNRELIABLE if reasons else STATUS_OBSERVED
        qualities.append(
            FrameTrackQuality(
                frame_index=idx,
                timestamp_s=ts,
                status=status,
                reasons=reasons,
                pelvis_xy=pelvis,
                bbox_height=height,
                laterality_sign=sign,
                interpolated=False,
            )
        )
        prev_pelvis = pelvis
        prev_height = height
        if sign:
            prev_sign = sign
        prev_detected_index = idx

    by_index = {fq.frame_index: fq for fq in qualities}
    frame_by_index = {
        int(getattr(fr, "frame_index", 0) or 0): fr for fr in frames
    }
    n = len(frames)
    detected_idxs = [
        fq.frame_index
        for fq in qualities
        if fq.status in (STATUS_OBSERVED, STATUS_UNRELIABLE)
    ]
    sample_idx = {
        "begin": 0 if n == 0 else int(getattr(frames[0], "frame_index", 0) or 0),
        "middle": 0 if n == 0 else int(getattr(frames[n // 2], "frame_index", n // 2) or 0),
        "end": 0 if n == 0 else int(getattr(frames[-1], "frame_index", n - 1) or 0),
    }
    if detected_idxs:
        sample_idx["first_detected"] = detected_idxs[0]
        sample_idx["last_detected"] = detected_idxs[-1]
    samples = {
        label: _sample_frame(by_index.get(idx), frame_by_index.get(idx))
        for label, idx in sample_idx.items()
    }
    failure_samples = []
    seen_fail: set[int] = set()
    for idx in identity_switches + jump_frames + laterality_flips + anatomy_fail_frames:
        if idx in seen_fail:
            continue
        seen_fail.add(idx)
        failure_samples.append(_sample_frame(by_index.get(idx), frame_by_index.get(idx)))
        if len(failure_samples) >= 10:
            break
    samples["failures"] = failure_samples
    neighborhood = []
    seen_nb: set[int] = set()
    for start, end in _intervals(
        fq.frame_index for fq in qualities if fq.status == STATUS_MISSING
    )[:6]:
        for idx in (start - 1, start, end, end + 1):
            if idx in seen_nb or idx not in by_index:
                continue
            seen_nb.add(idx)
            neighborhood.append(_sample_frame(by_index.get(idx), frame_by_index.get(idx)))
    samples["missing_neighborhoods"] = neighborhood

    joint_stats = {}
    for name in LOWER_LIMB_JOINTS:
        values = vis_values[name]
        count = len(values)
        n_low = sum(1 for v in values if v < VIS_OK)
        n_degraded = sum(1 for v in values if v < VIS_DEGRADED)
        mean = (sum(values) / count) if count else None
        joint_stats[name] = JointVisStats(
            name=name,
            n_observed=count,
            n_low_vis=n_low,
            n_degraded=n_degraded,
            mean_visibility=mean,
            min_visibility=min(values) if values else None,
            p10_visibility=_percentile(values, 0.10),
            median_visibility=_percentile(values, 0.50),
            pct_ok=(100.0 * (count - n_low) / count) if count else 0.0,
        )

    observed = sum(1 for fq in qualities if fq.status == STATUS_OBSERVED)
    unreliable = sum(1 for fq in qualities if fq.status == STATUS_UNRELIABLE)
    missing = sum(1 for fq in qualities if fq.status == STATUS_MISSING)
    interpolated = sum(1 for fq in qualities if fq.status == STATUS_INTERPOLATED)
    if identity_switches:
        notes.append(
            f"Possible subject switches at frames {identity_switches[:12]} "
            "(pelvis jump plus bbox-height change)."
        )
    else:
        notes.append("No pelvis+bbox identity switch detected in this recording.")
    if laterality_flips:
        notes.append(
            f"Left/right hip image-x sign flipped at {len(laterality_flips)} frames "
            f"{laterality_flips[:16]}. In a near-sagittal view this can be hip overlap "
            "noise; a flip that then holds is a sustained label swap."
        )
    if interpolated:
        notes.append(
            f"{interpolated} frames are explicitly marked interpolated; "
            "they are not counted as observed."
        )
    else:
        notes.append("No interpolated pose records in this sequence.")

    return TrackingAudit(
        total_frames=n,
        missing_frames=missing,
        observed_frames=observed,
        unreliable_frames=unreliable,
        interpolated_frames=interpolated,
        usable_gait_frames=observed,
        identity_switches=identity_switches,
        laterality_flips=laterality_flips,
        jump_frames=jump_frames,
        anatomy_fail_frames=anatomy_fail_frames,
        unreliable_intervals=_reason_intervals(qualities, STATUS_UNRELIABLE),
        missing_intervals=_intervals(
            fq.frame_index for fq in qualities if fq.status == STATUS_MISSING
        ),
        joint_stats=joint_stats,
        samples=samples,
        notes=notes,
    )
