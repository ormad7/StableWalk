"""
Contact-mask × simulated-force synchronization reward (AMP gait reward spec).

From the research design:
  reward[t] = video_contact_mask[t] * (simulated_force_z[t] > threshold)

Used offline to evaluate how well estimated or simulated forces align with
video-derived foot contact timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ContactSyncSummary:
    """Summary of contact–force synchronization quality."""

    mean_reward: float
    left_mean_reward: float
    right_mean_reward: float
    left_contact_frames: int
    right_contact_frames: int
    left_force_during_contact_pct: float
    right_force_during_contact_pct: float
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_reward": self.mean_reward,
            "left_mean_reward": self.left_mean_reward,
            "right_mean_reward": self.right_mean_reward,
            "left_contact_frames": self.left_contact_frames,
            "right_contact_frames": self.right_contact_frames,
            "left_force_during_contact_pct": self.left_force_during_contact_pct,
            "right_force_during_contact_pct": self.right_force_during_contact_pct,
            "interpretation": self.interpretation,
        }


def _align_length(
    mask: np.ndarray,
    force: np.ndarray,
    target_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(mask), len(force), target_len)
    return mask[:n].astype(np.float64), force[:n].astype(np.float64)


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
    mask, force = _align_length(
        video_contact_mask,
        simulated_force_z,
        min(len(video_contact_mask), len(simulated_force_z)),
    )
    force_active = (force > force_threshold_n).astype(np.float64)
    return mask * force_active


def summarize_contact_sync(
    left_contact_mask: np.ndarray,
    right_contact_mask: np.ndarray,
    left_force_n: np.ndarray,
    right_force_n: np.ndarray,
    *,
    force_threshold_n: float = 10.0,
) -> ContactSyncSummary:
    """Summarize bilateral contact–force synchronization."""
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

    if mean_r >= 0.65:
        interp = (
            "Good contact–force alignment — estimated forces rise when "
            "the video shows foot contact."
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
        left_contact_frames=left_contact_frames,
        right_contact_frames=right_contact_frames,
        left_force_during_contact_pct=left_pct,
        right_force_during_contact_pct=right_pct,
        interpretation=interp,
    )


__all__ = [
    "ContactSyncSummary",
    "contact_force_sync_reward",
    "summarize_contact_sync",
]
