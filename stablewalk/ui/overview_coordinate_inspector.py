"""Overview 3D Coordinate Inspector — display-only fields from cached joint data.

Formats live playhead kinematics for the Overview path panel. Does not rebuild
trajectories; callers supply tip XYZ / snapshots already computed for the frame.

Four laboratory sections: Identity · Position · Motion · Quality.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Mapping

from stablewalk.models.joint_registry import JOINT_DISPLAY_NAMES
from stablewalk.ui.dof_position_table import angle_value_for_item
from stablewalk.ui.dof_selection import anchor_joint_for_item, label_for_item
from stablewalk.ui.joint_data_table import (
    LOW_CONFIDENCE_THRESHOLD,
    confidence_for_joint_at_frame,
)
from stablewalk.ui.value_provenance import NA_DISPLAY, format_na

if TYPE_CHECKING:
    from stablewalk.models.gait_motion import (
        GaitMotionRecording,
        JointSample,
        SkeletonSnapshot,
    )
    from stablewalk.models.pose_data import PoseSequence
    from stablewalk.ui.playback_state import PlaybackState

# Keys owned by the visible inspector card.
COORDINATE_INSPECTOR_KEYS: tuple[str, ...] = (
    "joint_name",
    "frame",
    "time",
    "x",
    "y",
    "z",
    "velocity",
    "acceleration",
    "angle",
    "rom",
    "tracking_confidence",
    "pose_confidence",
    "visibility",
)

COORDINATE_INSPECTOR_TITLES: dict[str, str] = {
    "joint_name": "Joint",
    "frame": "Frame",
    "time": "Time",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "velocity": "Velocity",
    "acceleration": "Acceleration",
    "angle": "Angle",
    "rom": "ROM",
    "tracking_confidence": "Tracking",
    "pose_confidence": "Pose",
    "visibility": "Visibility",
}

# Glanceable unavailable glyph shown in the inspector UI (data layer still uses N/A).
INSPECTOR_UNAVAILABLE_DISPLAY = "—"

# Four clean laboratory sections.
COORDINATE_INSPECTOR_SECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("identity", "Identity", ("joint_name", "frame", "time")),
    ("position", "Position", ("x", "y", "z")),
    ("motion", "Motion", ("velocity", "acceleration", "angle", "rom")),
    ("quality", "Quality", ("tracking_confidence", "pose_confidence", "visibility")),
)

# Supervisor fields stay visible: joint, frame, time, XYZ, velocity,
# acceleration, angle, confidence. Remaining quality keys sit in Details.
COORDINATE_INSPECTOR_PRIMARY_KEYS: tuple[str, ...] = (
    "joint_name",
    "frame",
    "time",
    "x",
    "y",
    "z",
    "velocity",
    "acceleration",
    "angle",
    "tracking_confidence",
)
COORDINATE_INSPECTOR_DETAILS_KEYS: tuple[str, ...] = (
    "rom",
    "pose_confidence",
    "visibility",
)

COORDINATE_INSPECTOR_FIELD_FAMILY: dict[str, str] = {
    "joint_name": "identity",
    "frame": "identity",
    "time": "identity",
    "x": "position",
    "y": "position",
    "z": "position",
    "velocity": "motion",
    "acceleration": "motion",
    "angle": "motion",
    "rom": "motion",
    "tracking_confidence": "quality",
    "pose_confidence": "quality",
    "visibility": "quality",
}

# Compatibility alias.
COORDINATE_INSPECTOR_COLUMNS: tuple[tuple[str, ...], ...] = tuple(
    keys for _sid, _title, keys in COORDINATE_INSPECTOR_SECTIONS
)

# Numeric keys that should visually emphasize live updates during playback.
COORDINATE_INSPECTOR_LIVE_KEYS: frozenset[str] = frozenset(
    {
        "frame",
        "time",
        "x",
        "y",
        "z",
        "velocity",
        "acceleration",
        "angle",
        "pose_confidence",
        "visibility",
        "tracking_confidence",
    }
)

# Values tied to the selected joint — painted with SELECTED_JOINT_ACCENT.
COORDINATE_INSPECTOR_SELECTED_KEYS: frozenset[str] = frozenset(
    {
        "joint_name",
        "x",
        "y",
        "z",
        "velocity",
        "acceleration",
        "angle",
        "rom",
    }
)

_MAX_PLAUSIBLE_LINEAR_SPEED_MPS = 20.0
_MAX_PLAUSIBLE_LINEAR_ACCEL_MPS2 = 200.0


def empty_coordinate_inspector_fields() -> dict[str, str]:
    """All inspector slots as unavailable (never invent numbers)."""
    return {key: NA_DISPLAY for key in COORDINATE_INSPECTOR_KEYS}


def _vec3_speed_mps(sample: JointSample | None) -> float | None:
    """Linear speed (m/s) from velocity_vector, else scalar velocity when plausible."""
    if sample is None:
        return None
    vv = sample.velocity_vector
    if vv is not None:
        try:
            mag = math.sqrt(
                float(vv.x) ** 2 + float(vv.y) ** 2 + float(vv.z) ** 2
            )
        except (TypeError, ValueError):
            mag = None
        if mag is not None and math.isfinite(mag):
            if mag > _MAX_PLAUSIBLE_LINEAR_SPEED_MPS:
                return None
            return float(mag)
    if sample.velocity is not None:
        try:
            speed = abs(float(sample.velocity))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(speed) or speed > _MAX_PLAUSIBLE_LINEAR_SPEED_MPS:
            return None
        return speed
    return None


def _vec3_accel_mps2(
    sample: JointSample | None,
    prev: JointSample | None,
    dt: float,
) -> float | None:
    """Instantaneous linear acceleration magnitude (m/s²) from Δv/Δt."""
    if sample is None or prev is None or dt <= 1e-9:
        return None
    vv = sample.velocity_vector
    pv = prev.velocity_vector
    if vv is not None and pv is not None:
        try:
            ax = (float(vv.x) - float(pv.x)) / dt
            ay = (float(vv.y) - float(pv.y)) / dt
            az = (float(vv.z) - float(pv.z)) / dt
            mag = math.sqrt(ax * ax + ay * ay + az * az)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(mag) or mag > _MAX_PLAUSIBLE_LINEAR_ACCEL_MPS2:
            return None
        return float(mag)
    v_now = _vec3_speed_mps(sample)
    v_prev = _vec3_speed_mps(prev)
    if v_now is None or v_prev is None:
        return None
    mag = abs(v_now - v_prev) / dt
    if not math.isfinite(mag) or mag > _MAX_PLAUSIBLE_LINEAR_ACCEL_MPS2:
        return None
    return float(mag)


def _fmt_cm(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return format_na("position not available")
    # Compact; column alignment is handled by the inspector grid, not spaces.
    return f"{float(value):.1f} cm"


def _fmt_mps(value: float | None) -> str:
    if value is None:
        return format_na("linear velocity not available")
    if not math.isfinite(float(value)):
        return format_na("non-finite velocity")
    return f"{float(value):.2f} m/s"


def _fmt_mps2(value: float | None) -> str:
    if value is None:
        return format_na("linear acceleration not available")
    if not math.isfinite(float(value)):
        return format_na("non-finite acceleration")
    return f"{float(value):.2f} m/s²"


def _fmt_angle_deg(value: float | None) -> str:
    if value is None:
        return format_na("joint/DOF angle not available")
    if not math.isfinite(float(value)):
        return format_na("non-finite joint angle")
    return f"{float(value):.1f}°"


def _fmt_percent(value: float | None, *, unavailable: str) -> str:
    if value is None:
        return format_na(unavailable)
    conf = float(value)
    if not math.isfinite(conf):
        return format_na("non-finite confidence")
    frac = conf if conf <= 1.0 + 1e-9 else conf / 100.0
    pct = frac * 100.0 if conf <= 1.0 + 1e-9 else conf
    return f"{pct:.0f}%"


def _fmt_visibility(value: float | None) -> str:
    if value is None:
        return format_na("keypoint visibility unavailable")
    conf = float(value)
    if not math.isfinite(conf):
        return format_na("non-finite visibility")
    frac = conf if conf <= 1.0 + 1e-9 else conf / 100.0
    if frac < LOW_CONFIDENCE_THRESHOLD:
        return f"Low ({frac * 100.0:.0f}%)"
    return f"Visible ({frac * 100.0:.0f}%)"


def _joint_display_name(item_id: str | None) -> str:
    if not item_id:
        return format_na("no joint selected")
    anchor = anchor_joint_for_item(item_id)
    if anchor:
        return JOINT_DISPLAY_NAMES.get(anchor, label_for_item(item_id))
    return label_for_item(item_id)


def _playback_frame_time(
    state: PlaybackState | None,
    *,
    recording: GaitMotionRecording | None,
    list_pos: int | None,
) -> tuple[str, str]:
    if state is not None and state.total_frames > 0:
        frame = f"{state.display_frame_number} / {state.total_frames}"
        t = float(state.current_time_seconds)
        dur = float(state.duration_seconds)
        time_s = f"{t:.2f} s / {dur:.2f} s"
        return frame, time_s
    total = int(getattr(recording, "frame_count", 0) or 0) if recording else 0
    if list_pos is None or total <= 0:
        return (
            format_na("playback frame unavailable"),
            format_na("playback time unavailable"),
        )
    fps = float(getattr(recording, "fps", 0.0) or 0.0)
    cur = int(list_pos) + 1
    time_now = (float(list_pos) / fps) if fps > 1e-9 else 0.0
    duration = ((total - 1) / fps) if fps > 1e-9 and total > 1 else 0.0
    return (
        f"{cur} / {total}",
        f"{time_now:.2f} s / {duration:.2f} s",
    )


def tip_xyz_cm_from_cache(ax: Any | None) -> tuple[float, float, float] | None:
    """Read stature-scaled tip XYZ (cm) published by the path tip updater."""
    if ax is None:
        return None
    tip_cm = getattr(ax, "_stablewalk_tip_xyz_cm", None)
    if (
        isinstance(tip_cm, (tuple, list))
        and len(tip_cm) == 3
        and all(isinstance(v, (int, float)) for v in tip_cm)
        and all(math.isfinite(float(v)) for v in tip_cm)
    ):
        return (float(tip_cm[0]), float(tip_cm[1]), float(tip_cm[2]))
    return None


# Path-panel extent keys (AABB of the selected-joint trajectory).
PATH_EXTENT_KEYS: tuple[str, ...] = (
    "bounding_box",
    "x_range",
    "y_range",
    "z_range",
)

PATH_EXTENT_TITLES: dict[str, str] = {
    "bounding_box": "Bounding Box",
    "x_range": "X range",
    "y_range": "Y range",
    "z_range": "Z range",
}


def empty_path_extent_fields() -> dict[str, str]:
    return {key: NA_DISPLAY for key in PATH_EXTENT_KEYS}


def _fmt_range_cm(lo: float, hi: float) -> str:
    return f"{float(lo):6.1f} … {float(hi):6.1f} cm"


def path_bounds_m_from_cache(
    ax: Any | None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    """Axis-aligned path bounds in meters from the cached full trajectory."""
    if ax is None:
        return None
    cached = getattr(ax, "_stablewalk_path_bounds_m", None)
    if (
        isinstance(cached, tuple)
        and len(cached) == 3
        and all(
            isinstance(pair, (tuple, list)) and len(pair) == 2 for pair in cached
        )
    ):
        try:
            xr = (float(cached[0][0]), float(cached[0][1]))
            yr = (float(cached[1][0]), float(cached[1][1]))
            zr = (float(cached[2][0]), float(cached[2][1]))
        except (TypeError, ValueError):
            return None
        if all(math.isfinite(v) for v in (*xr, *yr, *zr)):
            return (xr, yr, zr)

    # Fallback: derive from full path XYZ cache (same units as drawn path).
    full = getattr(ax, "_stablewalk_full_path_xyz", None)
    if not (
        isinstance(full, tuple)
        and len(full) >= 4
        and isinstance(full[1], list)
        and len(full[1]) >= 2
    ):
        return None
    xs, ys, zs = full[1], full[2], full[3]
    try:
        xr = (float(min(xs)), float(max(xs)))
        yr = (float(min(ys)), float(max(ys)))
        zr = (float(min(zs)), float(max(zs)))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (*xr, *yr, *zr)):
        return None
    return (xr, yr, zr)


def _m_to_cm(meters: float) -> float:
    return float(meters) * 100.0


def build_path_extent_fields(ax: Any | None) -> dict[str, str]:
    """Format Bounding Box + X/Y/Z ranges for the Overview Path panel."""
    fields = empty_path_extent_fields()
    bounds = path_bounds_m_from_cache(ax)
    if bounds is None:
        return fields
    (x0, x1), (y0, y1), (z0, z1) = bounds
    sx = abs(_m_to_cm(x1) - _m_to_cm(x0))
    sy = abs(_m_to_cm(y1) - _m_to_cm(y0))
    sz = abs(_m_to_cm(z1) - _m_to_cm(z0))
    fields["bounding_box"] = f"{sx:5.1f} × {sy:5.1f} × {sz:5.1f} cm"
    fields["x_range"] = _fmt_range_cm(_m_to_cm(x0), _m_to_cm(x1))
    fields["y_range"] = _fmt_range_cm(_m_to_cm(y0), _m_to_cm(y1))
    fields["z_range"] = _fmt_range_cm(_m_to_cm(z0), _m_to_cm(z1))
    return fields


def _frame_mean_visibility(
    sequence: PoseSequence | None,
    frame_index: int | None,
    pose_indices: list[int] | tuple[int, ...] | None = None,
) -> float | None:
    if sequence is None or frame_index is None:
        return None
    if pose_indices is not None and len(pose_indices) > 0:
        if frame_index < 0 or frame_index >= len(pose_indices):
            return None
        frame_index = int(pose_indices[int(frame_index)])
    if frame_index < 0 or frame_index >= len(sequence.frames):
        return None
    frame = sequence.frames[int(frame_index)]
    vals = [
        float(kp.visibility)
        for kp in (frame.keypoints or ())
        if getattr(kp, "visibility", None) is not None
    ]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _session_tracking_confidence(gui: Any | None) -> float | None:
    """Session tracking quality as 0–1 when available (no invented values)."""
    if gui is None:
        return None
    ba = getattr(gui, "_biomech_analysis", None)
    if ba is not None and getattr(ba, "video_quality", None) is not None:
        checks = getattr(ba.video_quality, "checks", None) or {}
        score = checks.get("pose_confidence_score")
        if score is not None:
            try:
                v = float(score)
            except (TypeError, ValueError):
                v = None
            if v is not None and math.isfinite(v):
                return v / 100.0 if v > 1.0 + 1e-9 else v
        overall = getattr(ba.video_quality, "overall_quality_score", None)
        if overall is not None:
            try:
                v = float(overall)
            except (TypeError, ValueError):
                return None
            if math.isfinite(v):
                return v / 100.0 if v > 1.0 + 1e-9 else v
    return None


def build_coordinate_inspector_fields(
    item_id: str | None,
    *,
    snapshot: SkeletonSnapshot | None = None,
    recording: GaitMotionRecording | None = None,
    sequence: PoseSequence | None = None,
    playback_state: PlaybackState | None = None,
    tip_xyz_cm: tuple[float, float, float] | None = None,
    session_rom_display: str | None = None,
    list_pos: int | None = None,
    gui: Any | None = None,
    gait_phase_display: str | None = None,
) -> dict[str, str]:
    """Build inspector display strings from cached frame / tip data.

    Never invents biomechanics values. Missing or unreliable readings become
    ``N/A`` / ``Low``.
    """
    del gait_phase_display  # retained for call-site compatibility; phase strip owns it
    fields = empty_coordinate_inspector_fields()

    frame_i = list_pos
    if frame_i is None and playback_state is not None:
        frame_i = int(playback_state.current_frame_index)
    elif frame_i is None and snapshot is not None:
        frame_i = int(snapshot.frame_index)

    frame, time_s = _playback_frame_time(
        playback_state, recording=recording, list_pos=frame_i
    )
    fields["frame"] = frame
    fields["time"] = time_s

    track = _session_tracking_confidence(gui)
    if track is None:
        pose_indices = getattr(gui, "pose_indices", None) if gui is not None else None
        track = _frame_mean_visibility(sequence, frame_i, pose_indices)
    fields["tracking_confidence"] = _fmt_percent(
        track, unavailable="tracking confidence unavailable"
    )

    if not item_id:
        fields["joint_name"] = format_na("no joint selected")
        return fields

    fields["joint_name"] = _joint_display_name(item_id)

    if tip_xyz_cm is not None:
        fields["x"] = _fmt_cm(tip_xyz_cm[0])
        fields["y"] = _fmt_cm(tip_xyz_cm[1])
        fields["z"] = _fmt_cm(tip_xyz_cm[2])
    else:
        fields["x"] = format_na("path tip not ready")
        fields["y"] = format_na("path tip not ready")
        fields["z"] = format_na("path tip not ready")

    anchor = anchor_joint_for_item(item_id)
    joint = snapshot.joints.get(anchor) if snapshot is not None and anchor else None
    prev_joint: JointSample | None = None
    dt = 0.0
    if recording is not None and frame_i is not None and frame_i > 0:
        prev_snap = recording.snapshot_at(int(frame_i) - 1)
        if prev_snap is not None and anchor:
            prev_joint = prev_snap.joints.get(anchor)
            if snapshot is not None:
                dt = float(snapshot.time_s) - float(prev_snap.time_s)
            if dt <= 1e-9:
                fps = float(getattr(recording, "fps", 0.0) or 0.0)
                dt = (1.0 / fps) if fps > 1e-9 else 0.0

    fields["velocity"] = _fmt_mps(_vec3_speed_mps(joint))
    fields["acceleration"] = _fmt_mps2(_vec3_accel_mps2(joint, prev_joint, dt))

    angle = angle_value_for_item(item_id, snapshot) if snapshot is not None else None
    fields["angle"] = _fmt_angle_deg(angle)

    rom = (session_rom_display or "").strip()
    if not rom or rom in ("—", "-", "Not available"):
        fields["rom"] = format_na("session ROM not available")
    elif rom.startswith(NA_DISPLAY):
        fields["rom"] = rom
    else:
        fields["rom"] = rom if "°" in rom else f"{rom}°"

    pose_conf = confidence_for_joint_at_frame(
        sequence,
        anchor,
        int(snapshot.frame_index) if snapshot is not None else int(frame_i or 0),
    )
    fields["pose_confidence"] = _fmt_percent(
        pose_conf, unavailable="pose confidence unavailable"
    )
    fields["visibility"] = _fmt_visibility(pose_conf)
    return fields


def _is_unavailable_inspector_text(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    return (
        cleaned == NA_DISPLAY
        or cleaned.startswith(f"{NA_DISPLAY} ")
        or cleaned.startswith(f"{NA_DISPLAY}—")
        or cleaned.startswith(f"{NA_DISPLAY} —")
        or cleaned.startswith("Low ")
        or cleaned == INSPECTOR_UNAVAILABLE_DISPLAY
        or cleaned == "-"
    )


def normalize_inspector_display(text: str | None) -> str:
    """Map empty / N/A provenance strings to a single glanceable em dash."""
    raw = "" if text is None else str(text)
    if _is_unavailable_inspector_text(raw):
        # Keep explicit Low-confidence visibility (still a real reading).
        cleaned = raw.strip()
        if cleaned.startswith("Low "):
            return cleaned
        return INSPECTOR_UNAVAILABLE_DISPLAY
    return raw.strip()


def inspector_value_color(
    key: str,
    *,
    text: str,
    family_colors: Mapping[str, str],
    muted_fg: str,
    text_fg: str,
    changed: bool = False,
    highlight_fg: str | None = None,
    joint_selected: bool = False,
    selected_accent: str | None = None,
) -> str:
    """Pick value color from Position / Motion / Quality family (or muted for N/A)."""
    if _is_unavailable_inspector_text(text):
        return muted_fg
    # Playback pulse wins so changing numerals are readable during scrub/play.
    if changed and key in COORDINATE_INSPECTOR_LIVE_KEYS and highlight_fg:
        return highlight_fg
    # Persistent accent for selected-joint values (video/skeleton/path sync).
    if (
        joint_selected
        and selected_accent
        and key in COORDINATE_INSPECTOR_SELECTED_KEYS
    ):
        return selected_accent
    family = COORDINATE_INSPECTOR_FIELD_FAMILY.get(key, "identity")
    return family_colors.get(family, text_fg)


def apply_coordinate_inspector_labels(
    value_labels: Mapping[str, Any],
    fields: Mapping[str, str],
    *,
    text_fg: str,
    muted_fg: str,
    last_fields: dict[str, str] | None,
    family_colors: Mapping[str, str] | None = None,
    highlight_fg: str | None = None,
    joint_selected: bool = False,
    selected_accent: str | None = None,
    force_recolor: bool = False,
) -> dict[str, str]:
    """Push fields onto Tk labels; accent selected-joint values consistently.

    Unavailable readings always paint as ``—`` (never blank).
    """
    colors = family_colors or {}
    display = {
        key: normalize_inspector_display(fields.get(key, NA_DISPLAY))
        for key in value_labels
    }
    if last_fields == display and not force_recolor:
        # Still refresh accent colors when selection state changed.
        if not joint_selected or not selected_accent:
            return display
    for key, lbl in value_labels.items():
        text = display.get(key, INSPECTOR_UNAVAILABLE_DISPLAY)
        prev = None if last_fields is None else last_fields.get(key)
        fg = inspector_value_color(
            key,
            text=text,
            family_colors=colors,
            muted_fg=muted_fg,
            text_fg=text_fg,
            changed=prev is not None and prev != text,
            highlight_fg=highlight_fg,
            joint_selected=joint_selected,
            selected_accent=selected_accent,
        )
        if prev == text and not force_recolor:
            # Text-only update for selected-joint accent persistence.
            if joint_selected and selected_accent and key in COORDINATE_INSPECTOR_SELECTED_KEYS:
                try:
                    lbl.configure(fg=fg)
                except Exception:
                    pass
            continue
        try:
            lbl.configure(text=text, fg=fg)
        except Exception:
            pass
    return dict(display)


def apply_path_extent_labels(
    value_labels: Mapping[str, Any],
    fields: Mapping[str, str],
    *,
    text_fg: str,
    muted_fg: str,
    last_fields: dict[str, str] | None,
) -> dict[str, str]:
    """Push path extent strings onto Tk labels; skip unchanged text."""
    display = {key: fields.get(key, NA_DISPLAY) for key in value_labels}
    if last_fields == display:
        return display
    for key, lbl in value_labels.items():
        text = display.get(key, NA_DISPLAY)
        prev = None if last_fields is None else last_fields.get(key)
        if prev == text:
            continue
        fg = muted_fg if _is_unavailable_inspector_text(text) else text_fg
        try:
            lbl.configure(text=text, fg=fg)
        except Exception:
            pass
    return dict(display)
