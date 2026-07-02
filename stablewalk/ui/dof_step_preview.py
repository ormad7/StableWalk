"""
Current vs next-step position preview for selected GUI degrees of freedom.

Computes joint positions at the current playback time and at the next frame
or next data-refresh interval (0.25 s / 0.5 s), plus the positional delta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot, Vec3
from stablewalk.models.joint_registry import JOINT_DISPLAY_NAMES
from stablewalk.ui.dof_position_table import angle_value_for_item
from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS, anchor_joint_for_item, label_for_item

StepKind = Literal["frame", "time"]


@dataclass(frozen=True)
class StepConfig:
    """How far ahead to look for the 'next' position."""

    kind: StepKind
    interval_s: float = 0.5

    @classmethod
    def from_refresh_label(cls, label: str) -> StepConfig:
        if label == "0.25 s":
            return cls(kind="time", interval_s=0.25)
        return cls(kind="time", interval_s=0.5)

    def describe(self) -> str:
        return f"+{self.interval_s:.2f} s"


@dataclass(frozen=True)
class DofStepPreview:
    """Current and next joint positions for one selected GUI DOF."""

    item_id: str
    dof_label: str
    joint_id: str
    joint_name: str
    current: Vec3
    next: Vec3
    current_time_s: float
    next_time_s: float
    current_frame: int
    next_frame: int
    step_label: str
    current_angle_deg: float | None = None
    next_angle_deg: float | None = None

    @property
    def delta_angle_deg(self) -> float | None:
        if self.current_angle_deg is None or self.next_angle_deg is None:
            return None
        return self.next_angle_deg - self.current_angle_deg

    @property
    def delta(self) -> Vec3:
        return Vec3(
            self.next.x - self.current.x,
            self.next.y - self.current.y,
            self.next.z - self.current.z,
        )

    @property
    def distance_m(self) -> float:
        d = self.delta
        return (d.x * d.x + d.y * d.y + d.z * d.z) ** 0.5

    def direction_label(self) -> str:
        parts: list[str] = []
        for value, axis in (
            (self.delta.x, "X"),
            (self.delta.y, "Y"),
            (self.delta.z, "Z"),
        ):
            if abs(value) < 1e-6:
                continue
            parts.append(f"{'+' if value > 0 else '-'}{axis}")
        return ", ".join(parts) if parts else "—"


def _wrap_frame(frame_float: float, frame_count: int) -> float:
    if frame_count <= 0:
        return 0.0
    if frame_count == 1:
        return 0.0
    wrapped = frame_float % frame_count
    return wrapped


def resolve_next_frame_float(
    frame_float: float,
    frame_count: int,
    *,
    config: StepConfig,
    fps: float,
) -> float:
    """Playback frame index (float) for the next inspection step."""
    if frame_count <= 1:
        return 0.0

    max_index = float(frame_count - 1)
    if config.kind == "frame":
        if frame_float >= max_index - 1e-6:
            return 0.0
        return min(frame_float + 1.0, max_index)

    advance_frames = config.interval_s * max(fps, 1e-6)
    return _wrap_frame(frame_float + advance_frames, frame_count)


def snapshot_at_frame(
    recording: GaitMotionRecording,
    frame_float: float,
    *,
    smooth: bool = True,
) -> SkeletonSnapshot | None:
    """Interpolate a snapshot at a fractional frame index."""
    frame_count = recording.frame_count
    if frame_count == 0:
        return None

    from stablewalk.ui.viewers.gait_skeleton_renderer import interpolate_snapshots

    frame_float = max(0.0, min(frame_float, frame_count - 1))
    i0 = int(frame_float)
    i1 = min(i0 + 1, frame_count - 1)
    alpha = frame_float - i0
    snap_a = recording.snapshot_at(i0)
    snap_b = recording.snapshot_at(i1)
    if not snap_a:
        return snap_b
    if not snap_b or alpha <= 0.0 or not smooth:
        return snap_a
    return interpolate_snapshots(snap_a, snap_b, alpha)


def joint_position(
    snapshot: SkeletonSnapshot | None,
    joint_id: str,
) -> Vec3 | None:
    if not snapshot:
        return None
    sample = snapshot.joints.get(joint_id)
    return sample.position if sample else None


def compute_dof_step_previews(
    recording: GaitMotionRecording,
    selected_item_ids: set[str],
    frame_float: float,
    *,
    config: StepConfig,
    smooth: bool = True,
) -> list[DofStepPreview]:
    """Build current/next previews for each selected DOF."""
    if not recording.snapshots or not selected_item_ids:
        return []

    fps = max(recording.fps, 1e-6)
    frame_count = recording.frame_count
    current_snap = snapshot_at_frame(recording, frame_float, smooth=smooth)
    next_frame = resolve_next_frame_float(
        frame_float, frame_count, config=config, fps=fps
    )
    next_snap = snapshot_at_frame(recording, next_frame, smooth=smooth)
    if not current_snap or not next_snap:
        return []

    step_label = config.describe()
    ordered = [item_id for item_id in GUI_DOF_ITEM_IDS if item_id in selected_item_ids]
    previews: list[DofStepPreview] = []

    for item_id in ordered:
        joint_id = anchor_joint_for_item(item_id)
        if not joint_id:
            continue
        current = joint_position(current_snap, joint_id)
        nxt = joint_position(next_snap, joint_id)
        if current is None or nxt is None:
            continue
        previews.append(
            DofStepPreview(
                item_id=item_id,
                dof_label=label_for_item(item_id),
                joint_id=joint_id,
                joint_name=JOINT_DISPLAY_NAMES.get(
                    joint_id, joint_id.replace("_", " ").title()
                ),
                current=current,
                next=nxt,
                current_time_s=current_snap.time_s,
                next_time_s=next_snap.time_s,
                current_frame=current_snap.frame_index,
                next_frame=next_snap.frame_index,
                step_label=step_label,
                current_angle_deg=angle_value_for_item(item_id, current_snap),
                next_angle_deg=angle_value_for_item(item_id, next_snap),
            )
        )
    return previews


def format_vec3(vec: Vec3, *, decimals: int = 3) -> str:
    return f"({vec.x:.{decimals}f}, {vec.y:.{decimals}f}, {vec.z:.{decimals}f})"


def preview_table_rows(previews: list[DofStepPreview]) -> list[tuple[str, ...]]:
    """Rows for the movement-step Treeview."""
    rows: list[tuple[str, ...]] = []
    for preview in previews:
        delta = preview.delta
        rows.append(
            (
                preview.dof_label,
                format_vec3(preview.current),
                format_vec3(preview.next),
                f"Δx={delta.x:+.3f}  Δy={delta.y:+.3f}  Δz={delta.z:+.3f}",
                f"{preview.distance_m:.4f} m",
                preview.direction_label(),
            )
        )
    return rows


STEP_PREVIEW_COLUMNS: tuple[str, ...] = (
    "dof",
    "current",
    "next",
    "delta",
    "distance",
    "direction",
)

STEP_PREVIEW_HEADINGS: dict[str, str] = {
    "dof": "DOF",
    "current": "Current (x, y, z)",
    "next": "Next (x, y, z)",
    "delta": "Difference",
    "distance": "|Δ|",
    "direction": "Direction",
}

STEP_PREVIEW_WIDTHS: dict[str, int] = {
    "dof": 88,
    "current": 130,
    "next": 130,
    "delta": 150,
    "distance": 64,
    "direction": 72,
}
