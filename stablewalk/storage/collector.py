"""In-memory collector for playback kinematic samples (no UI row cap)."""

from __future__ import annotations

from dataclasses import dataclass, field

from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot
from stablewalk.storage.models import KinematicSample
from stablewalk.ui.dof_position_table import (
    GUI_DOF_ITEM_IDS,
    kinematic_sample_for_item,
)
from stablewalk.ui.dof_selection import label_for_item



@dataclass
class SessionKinematicCollector:
    """
    Accumulates structured kinematic samples during playback.

    Mirrors ``DofPositionTableHistory`` tick semantics but stores numeric
    ``KinematicSample`` records without truncating older frames.
    """

    samples: list[KinematicSample] = field(default_factory=list)
    _last_frame_by_item: dict[str, int] = field(default_factory=dict, repr=False)

    def clear(self) -> None:
        self.samples.clear()
        self._last_frame_by_item.clear()

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def append_tick(
        self,
        snapshot: SkeletonSnapshot,
        selected_item_ids: set[str],
        *,
        next_snapshot: SkeletonSnapshot | None = None,
    ) -> bool:
        """Append one sample per selected DOF for the current frame."""
        frame_index = snapshot.frame_index
        added = False
        ordered = [
            item_id for item_id in GUI_DOF_ITEM_IDS if item_id in selected_item_ids
        ]
        for item_id in ordered:
            if self._last_frame_by_item.get(item_id) == frame_index:
                continue
            self._last_frame_by_item[item_id] = frame_index
            self.samples.append(
                kinematic_sample_for_item(
                    item_id,
                    snapshot,
                    next_snapshot=next_snapshot,
                )
            )
            added = True
        return added

    def samples_from_recording(
        self,
        recording: GaitMotionRecording,
        selected_item_ids: set[str],
    ) -> list[KinematicSample]:
        """Build samples for every frame in a recording (fallback when no playback history)."""
        if not selected_item_ids:
            return []
        ordered = [
            item_id for item_id in GUI_DOF_ITEM_IDS if item_id in selected_item_ids
        ]
        out: list[KinematicSample] = []
        for index in range(recording.frame_count):
            snap = recording.snapshot_at(index)
            if snap is None:
                continue
            next_snap = recording.snapshot_at(index + 1)
            for item_id in ordered:
                out.append(
                    kinematic_sample_for_item(
                        item_id,
                        snap,
                        next_snapshot=next_snap,
                    )
                )
        return out
