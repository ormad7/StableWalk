"""Simple cross-demo gait comparison summaries for academic presentations.

Values are computed from the real ``GaitMotionRecording`` time series — nothing
is hardcoded. Documented source labels stay separate from those computed
results. StableWalk displays biomechanical movement data only; it does not
diagnose medical conditions or rank gaits as better/worse.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.ui.dof_selection import anchor_joint_for_item, label_for_item
from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES, DemoGaitExample

DOCUMENTED_UNAVAILABLE = "Not documented"
COMPUTED_PENDING = "—"


@dataclass(frozen=True)
class DemoGaitComparisonRow:
    demo_key: str
    demo_type: str
    source_label: str
    camera_view: str
    locomotion_setting: str
    duration_display: str
    usable_cycles_display: str
    tracking_quality: str
    joint_label: str
    max_angle_deg: float | None
    avg_velocity: float | None
    velocity_unit: str = "m/s"
    analyzed_duration_s: float | None = None
    recording_duration_s: float | None = None
    usable_cycles: int | None = None


def documented_source_label(example: DemoGaitExample) -> str:
    """Dataset/protocol label from the example metadata — never a computed score."""
    return (example.source_name or "").strip() or DOCUMENTED_UNAVAILABLE


def documented_camera_view(example: DemoGaitExample) -> str:
    return (example.camera_view or "").strip() or DOCUMENTED_UNAVAILABLE


def documented_locomotion(example: DemoGaitExample) -> str:
    return (example.locomotion_setting or "").strip() or DOCUMENTED_UNAVAILABLE


def analyzed_duration_s(recording: GaitMotionRecording | None) -> float | None:
    if recording is None:
        return None
    snaps = list(getattr(recording, "snapshots", None) or [])
    if not snaps:
        return None
    times = [float(getattr(snap, "time_s", 0.0) or 0.0) for snap in snaps]
    return max(times) - min(times)


def format_duration_display(
    *,
    analyzed_s: float | None,
    recording_s: float | None,
    usable_s: float | None = None,
    coverage: float | None = None,
    longest_valid_segment_s: float | None = None,
) -> str:
    """Source, analyzed span, and cumulative valid duration. Empty until analysis exists."""
    analyzed = float(analyzed_s) if analyzed_s is not None else None
    recording = float(recording_s) if recording_s is not None else None
    valid = float(usable_s) if usable_s is not None else None
    if analyzed is not None and analyzed < 0:
        analyzed = None
    if recording is not None and recording <= 0:
        recording = None
    if valid is not None and valid < 0:
        valid = None
    if analyzed is None and recording is None and valid is None:
        return COMPUTED_PENDING
    if valid is None:
        if recording is not None and analyzed is not None and recording > analyzed + 0.5:
            return f"{analyzed:.2f} s analyzed span of {recording:.2f} s recording"
        if analyzed is not None:
            return f"{analyzed:.2f} s analyzed span"
        return f"{recording:.2f} s"
    parts: list[str] = []
    if analyzed is None or abs(valid - analyzed) > 0.05:
        parts.append(f"{valid:.2f} s valid")
    if analyzed is not None and recording is not None:
        parts.append(f"{analyzed:.2f} s analyzed span of {recording:.2f} s source")
    elif analyzed is not None:
        parts.append(f"{analyzed:.2f} s analyzed span")
    elif recording is not None:
        parts.append(f"{recording:.2f} s source")
    if coverage is not None:
        parts.append(f"coverage {float(coverage) * 100.0:.0f}%")
    if longest_valid_segment_s is not None:
        parts.append(f"longest {float(longest_valid_segment_s):.2f} s")
    return " · ".join(parts) if parts else f"{valid:.2f} s valid"


def format_usable_cycles_display(usable_cycles: int | None) -> str:
    if usable_cycles is None:
        return COMPUTED_PENDING
    return str(int(usable_cycles))


def compute_joint_summary(
    recording: GaitMotionRecording,
    item_id: str,
) -> tuple[float | None, float | None, str]:
    """Max flexion angle and mean scalar speed for a selected joint over the clip."""
    joint_id = anchor_joint_for_item(item_id) or item_id
    ts = recording.build_time_series()
    angle_vals = [float(a) for a in ts.angles.get(joint_id, []) if a is not None]
    vel_vals = [float(v) for v in ts.velocities.get(joint_id, []) if v is not None]
    max_angle = max(angle_vals) if angle_vals else None
    avg_vel = mean(vel_vals) if vel_vals else None
    return max_angle, avg_vel, "m/s"


def documented_placeholder_row(example: DemoGaitExample) -> DemoGaitComparisonRow:
    """Documented source facts only — computed cells stay pending."""
    return DemoGaitComparisonRow(
        demo_key=example.key,
        demo_type=example.display_name,
        source_label=documented_source_label(example),
        camera_view=documented_camera_view(example),
        locomotion_setting=documented_locomotion(example),
        duration_display=COMPUTED_PENDING,
        usable_cycles_display=COMPUTED_PENDING,
        tracking_quality=COMPUTED_PENDING,
        joint_label=COMPUTED_PENDING,
        max_angle_deg=None,
        avg_velocity=None,
    )


def build_comparison_row(
    recording: GaitMotionRecording,
    example: DemoGaitExample,
    item_id: str,
    *,
    usable_cycles: int | None = None,
    tracking_quality: str | None = None,
    recording_duration_s: float | None = None,
) -> DemoGaitComparisonRow:
    max_angle, avg_vel, unit = compute_joint_summary(recording, item_id)
    analyzed = analyzed_duration_s(recording)
    return DemoGaitComparisonRow(
        demo_key=example.key,
        demo_type=example.display_name,
        source_label=documented_source_label(example),
        camera_view=documented_camera_view(example),
        locomotion_setting=documented_locomotion(example),
        duration_display=format_duration_display(
            analyzed_s=analyzed,
            recording_s=recording_duration_s,
            usable_s=(
                None
                if recording.metadata.get("usable_duration_s") is None
                else float(recording.metadata.get("usable_duration_s"))
            )
            if getattr(recording, "metadata", None)
            else None,
        ),
        usable_cycles_display=format_usable_cycles_display(usable_cycles),
        tracking_quality=(tracking_quality or "").strip() or COMPUTED_PENDING,
        joint_label=label_for_item(item_id),
        max_angle_deg=max_angle,
        avg_velocity=avg_vel,
        velocity_unit=unit,
        analyzed_duration_s=analyzed,
        recording_duration_s=recording_duration_s,
        usable_cycles=usable_cycles,
    )


COMPARISON_TREE_COLUMNS: tuple[str, ...] = (
    "demo_type",
    "source_label",
    "camera",
    "condition",
    "duration",
    "cycles",
    "tracking",
)

COMPARISON_TREE_HEADINGS: dict[str, str] = {
    "demo_type": "Demo",
    "source_label": "Source (documented)",
    "camera": "Camera",
    "condition": "Treadmill / overground",
    "duration": "Duration",
    "cycles": "Usable cycles",
    "tracking": "Tracking",
}

COMPARISON_TREE_WIDTHS: dict[str, int] = {
    "demo_type": 88,
    "source_label": 160,
    "camera": 92,
    "condition": 110,
    "duration": 150,
    "cycles": 78,
    "tracking": 170,
}

COMPARISON_TREE_TEXT_COLS = frozenset(
    {"demo_type", "source_label", "camera", "condition", "duration", "tracking"}
)


def comparison_tree_values(row: DemoGaitComparisonRow) -> tuple[str, ...]:
    return (
        row.demo_type,
        row.source_label,
        row.camera_view,
        row.locomotion_setting,
        row.duration_display,
        row.usable_cycles_display,
        row.tracking_quality,
    )


def example_comparison_order() -> tuple[DemoGaitExample, ...]:
    return tuple(ex for ex in DEMO_GAIT_EXAMPLES if ex.key in ("abnormal", "normal", "athletic"))
