"""
Contact-mask synchronization reward (AMP gait reward spec).

Compares reference-video contact timing with retargeted/simulated contact timing
or estimated vertical force proxies.

From the research design:
  reward[t] = video_contact_mask[t] * (simulated_force_z[t] > threshold)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONTACT_SYNC_SCHEMA_VERSION = "1.1"


@dataclass(frozen=True)
class ContactSyncSummary:
    """Summary of contact–force or contact–contact synchronization quality."""

    mean_reward: float
    left_mean_reward: float
    right_mean_reward: float
    mean_sync_score: float
    left_contact_frames: int
    right_contact_frames: int
    left_force_during_contact_pct: float
    right_force_during_contact_pct: float
    mismatch_frame_count: int
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_reward": self.mean_reward,
            "left_mean_reward": self.left_mean_reward,
            "right_mean_reward": self.right_mean_reward,
            "mean_sync_score": self.mean_sync_score,
            "left_contact_frames": self.left_contact_frames,
            "right_contact_frames": self.right_contact_frames,
            "left_force_during_contact_pct": self.left_force_during_contact_pct,
            "right_force_during_contact_pct": self.right_force_during_contact_pct,
            "mismatch_frame_count": self.mismatch_frame_count,
            "interpretation": self.interpretation,
        }


def _align_arrays(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    n = min(len(a) for a in arrays)
    return tuple(a[:n].astype(np.float64) for a in arrays)


def contact_force_sync_reward(
    video_contact_mask: np.ndarray,
    simulated_force_z: np.ndarray,
    *,
    force_threshold_n: float = 10.0,
) -> np.ndarray:
    """
    Per-frame reward when simulated vertical force aligns with video contact.

    Returns values in {0.0, 1.0} per frame.
    """
    mask, force = _align_arrays(video_contact_mask, simulated_force_z)
    force_active = (force > force_threshold_n).astype(np.float64)
    return mask * force_active


def contact_timing_sync_reward(
    reference_mask: np.ndarray,
    compared_mask: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
) -> np.ndarray:
    """
    Per-frame reward when reference and compared contact masks agree.

    Confidence weights each frame (0..1); defaults to uniform weighting.
    """
    ref, cmp = _align_arrays(reference_mask, compared_mask)
    agree = (ref > 0.5) == (cmp > 0.5)
    reward = agree.astype(np.float64)
    if confidence is not None and len(confidence) >= len(reward):
        w = np.clip(confidence[: len(reward)], 0.0, 1.0)
        reward = reward * w
    return reward


def mismatch_frames(
    reference_mask: np.ndarray,
    compared_mask: np.ndarray,
) -> np.ndarray:
    """Boolean array — True where reference and compared contact disagree."""
    ref, cmp = _align_arrays(reference_mask, compared_mask)
    return (ref > 0.5) != (cmp > 0.5)


def summarize_contact_sync(
    left_contact_mask: np.ndarray,
    right_contact_mask: np.ndarray,
    left_force_n: np.ndarray,
    right_force_n: np.ndarray,
    *,
    force_threshold_n: float = 10.0,
    reference_left_mask: np.ndarray | None = None,
    reference_right_mask: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
) -> ContactSyncSummary:
    """Summarize bilateral contact–force or contact–timing synchronization."""
    n = min(
        len(left_contact_mask),
        len(right_contact_mask),
        len(left_force_n),
        len(right_force_n),
    )
    left_r = contact_force_sync_reward(
        left_contact_mask[:n], left_force_n[:n], force_threshold_n=force_threshold_n
    )
    right_r = contact_force_sync_reward(
        right_contact_mask[:n], right_force_n[:n], force_threshold_n=force_threshold_n
    )

    if reference_left_mask is not None and reference_right_mask is not None:
        ref_l = contact_timing_sync_reward(
            reference_left_mask[:n],
            left_contact_mask[:n],
            confidence=confidence,
        )
        ref_r = contact_timing_sync_reward(
            reference_right_mask[:n],
            right_contact_mask[:n],
            confidence=confidence,
        )
        left_r = (left_r + ref_l) * 0.5
        right_r = (right_r + ref_r) * 0.5

    combined = (left_r + right_r) * 0.5

    left_contact_frames = int(left_contact_mask[:n].sum())
    right_contact_frames = int(right_contact_mask[:n].sum())

    def _pct_during_contact(mask: np.ndarray, force: np.ndarray) -> float:
        contact_idx = mask.astype(bool)
        if not contact_idx.any():
            return 0.0
        active = (force[contact_idx] > force_threshold_n).sum()
        return float(active / contact_idx.sum() * 100.0)

    left_pct = _pct_during_contact(left_contact_mask[:n], left_force_n[:n])
    right_pct = _pct_during_contact(right_contact_mask[:n], right_force_n[:n])
    mean_r = float(combined.mean()) if len(combined) else 0.0

    mismatch_count = 0
    if reference_left_mask is not None and reference_right_mask is not None:
        mm_l = mismatch_frames(reference_left_mask[:n], left_contact_mask[:n])
        mm_r = mismatch_frames(reference_right_mask[:n], right_contact_mask[:n])
        mismatch_count = int((mm_l | mm_r).sum())

    if mean_r >= 0.65:
        interp = (
            "Good contact alignment — estimated forces or retargeted timing "
            "match reference-video foot contact."
        )
    elif mean_r >= 0.35:
        interp = (
            "Moderate alignment — timing partly matches; noisy pose depth "
            "may reduce force estimates."
        )
    else:
        interp = (
            "Weak alignment — check contact detection or force estimation; "
            "short clips often score low."
        )

    return ContactSyncSummary(
        mean_reward=mean_r,
        left_mean_reward=float(left_r.mean()) if len(left_r) else 0.0,
        right_mean_reward=float(right_r.mean()) if len(right_r) else 0.0,
        mean_sync_score=mean_r,
        left_contact_frames=left_contact_frames,
        right_contact_frames=right_contact_frames,
        left_force_during_contact_pct=left_pct,
        right_force_during_contact_pct=right_pct,
        mismatch_frame_count=mismatch_count,
        interpretation=interp,
    )


def export_contact_sync_reward_npz(
    left_contact_mask: np.ndarray,
    right_contact_mask: np.ndarray,
    left_force_n: np.ndarray,
    right_force_n: np.ndarray,
    output_path: Path,
    *,
    force_threshold_n: float = 10.0,
    timestamps: np.ndarray | None = None,
    reference_left_mask: np.ndarray | None = None,
    reference_right_mask: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    run_name: str = "",
) -> Path:
    """Export per-frame contact sync rewards for AMP / validation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(
        len(left_contact_mask),
        len(right_contact_mask),
        len(left_force_n),
        len(right_force_n),
    )
    left_r = contact_force_sync_reward(
        left_contact_mask[:n], left_force_n[:n], force_threshold_n=force_threshold_n
    )
    right_r = contact_force_sync_reward(
        right_contact_mask[:n], right_force_n[:n], force_threshold_n=force_threshold_n
    )

    ref_left = reference_left_mask[:n] if reference_left_mask is not None else None
    ref_right = reference_right_mask[:n] if reference_right_mask is not None else None
    conf = confidence[:n] if confidence is not None else None

    if ref_left is not None and ref_right is not None:
        timing_l = contact_timing_sync_reward(ref_left, left_contact_mask[:n], confidence=conf)
        timing_r = contact_timing_sync_reward(ref_right, right_contact_mask[:n], confidence=conf)
        left_r = (left_r + timing_l) * 0.5
        right_r = (right_r + timing_r) * 0.5

    combined = (left_r + right_r) * 0.5
    mean_sync = float(combined.mean()) if len(combined) else 0.0

    mm = np.zeros(n, dtype=np.int8)
    if ref_left is not None and ref_right is not None:
        mm = (
            mismatch_frames(ref_left, left_contact_mask[:n])
            | mismatch_frames(ref_right, right_contact_mask[:n])
        ).astype(np.int8)

    payload: dict[str, Any] = {
        "schema_version": CONTACT_SYNC_SCHEMA_VERSION,
        "run_name": run_name or output_path.parent.name,
        "frame_count": np.int32(n),
        "left_reward": left_r.astype(np.float64),
        "right_reward": right_r.astype(np.float64),
        "combined_reward": combined.astype(np.float64),
        "mean_sync_score": np.float64(mean_sync),
        "mismatch_frames": mm,
        "left_contact_mask": left_contact_mask[:n].astype(np.int8),
        "right_contact_mask": right_contact_mask[:n].astype(np.int8),
        "left_force_n": left_force_n[:n].astype(np.float64),
        "right_force_n": right_force_n[:n].astype(np.float64),
        "force_threshold_n": np.float64(force_threshold_n),
    }
    if timestamps is not None and len(timestamps) >= n:
        payload["timestamps"] = timestamps[:n].astype(np.float64)
    if ref_left is not None:
        payload["reference_left_contact_mask"] = ref_left.astype(np.int8)
    if ref_right is not None:
        payload["reference_right_contact_mask"] = ref_right.astype(np.int8)
    if conf is not None:
        payload["confidence"] = conf.astype(np.float64)

    np.savez_compressed(output_path, **payload)
    return output_path


__all__ = [
    "CONTACT_SYNC_SCHEMA_VERSION",
    "ContactSyncSummary",
    "contact_force_sync_reward",
    "contact_timing_sync_reward",
    "export_contact_sync_reward_npz",
    "mismatch_frames",
    "summarize_contact_sync",
]
