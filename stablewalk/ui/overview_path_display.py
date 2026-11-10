"""Display-only helpers for Overview 3D path cameras, coordinates, and lengths.

Path length
    Sum of Euclidean 3D segments between valid consecutive samples. A pair is
    connected only when both points are finite and, if timestamps are given,
    dt <= 1.5/fps. Long tracking gaps are not bridged.

Endpoint displacement
    Euclidean distance from the first valid sample to the current/last valid
    sample in the same coordinate frame. This is not world travel when the
    frame is pelvis-centered (root-relative), and it is not treadmill belt
    distance. Unavailable values are N/A, never a misleading 0.0.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

# Named camera views shown in the Overview / Motion toolbars.
# Values are (elev, azim) in the Y-up matplotlib convention.
CAMERA_VIEW_PERSPECTIVE = "Perspective"
CAMERA_VIEW_SAGITTAL = "Sagittal (side)"
CAMERA_VIEW_FRONTAL = "Frontal"
CAMERA_VIEW_TRANSVERSE = "Transverse (top)"
CAMERA_VIEW_SOURCE = "Source camera"

NAMED_CAMERA_VIEWS: tuple[str, ...] = (
    CAMERA_VIEW_PERSPECTIVE,
    CAMERA_VIEW_SAGITTAL,
    CAMERA_VIEW_FRONTAL,
    CAMERA_VIEW_TRANSVERSE,
)

CAMERA_VIEW_SHORT_LABELS: dict[str, str] = {
    CAMERA_VIEW_PERSPECTIVE: "Persp",
    CAMERA_VIEW_SAGITTAL: "Sagittal",
    CAMERA_VIEW_FRONTAL: "Frontal",
    CAMERA_VIEW_TRANSVERSE: "Transverse",
    CAMERA_VIEW_SOURCE: "Source",
}

# Legacy toolbar names still accepted by set_trajectory_camera_preset.
CAMERA_VIEW_ALIASES: dict[str, str] = {
    "Side": CAMERA_VIEW_SAGITTAL,
    "Front": CAMERA_VIEW_FRONTAL,
    "Top": CAMERA_VIEW_TRANSVERSE,
}

_CAMERA_ALIGNED_TOKENS: tuple[str, ...] = (
    "camera",
    "hip_centered",
    "pelvis_centered",
    "sw_canonical",
    "y_up_z_forward",
    "canonical",
)


def resolve_camera_view_name(name: str | None) -> str:
    """Map a toolbar or alias name to the canonical camera-view id."""
    raw = str(name or "").strip()
    if raw in CAMERA_VIEW_ALIASES:
        return CAMERA_VIEW_ALIASES[raw]
    return raw or CAMERA_VIEW_PERSPECTIVE


def named_camera_views(*, include_source: bool = False) -> tuple[str, ...]:
    """Ordered camera views for menus. Source camera is opt-in."""
    if include_source:
        return (*NAMED_CAMERA_VIEWS, CAMERA_VIEW_SOURCE)
    return NAMED_CAMERA_VIEWS


def source_camera_view_available(
    recording: Any | None = None,
    *,
    coord_mode: str | None = None,
) -> bool:
    """True when reconstructed coordinates are already source-camera aligned.

    Monocular reconstruction stores X lateral, Y up, Z depth in the camera
    frame (identity transform). Global/gait-frame series without that frame
    do not get a Source-camera button.
    """
    if recording is None:
        return False
    system = str(getattr(recording, "coordinate_system", "") or "").lower()
    if any(token in system for token in _CAMERA_ALIGNED_TOKENS):
        return True
    root = getattr(recording, "root_trajectory", None)
    try:
        if root is not None and len(root) > 0:
            return True
    except TypeError:
        pass
    mode = str(coord_mode or "").lower()
    return mode.startswith("root")


def distances_are_calibrated(recording: Any | None = None) -> bool:
    """True only when the recording advertises calibrated world units."""
    if recording is None:
        return False
    system = str(getattr(recording, "coordinate_system", "") or "").lower()
    return "calibrat" in system


def coordinate_frame_caption(
    *,
    coord_mode: str = "Root-relative",
    recording: Any | None = None,
) -> str:
    """One-line origin / axis / estimated-vs-calibrated explanation."""
    root = str(coord_mode or "").lower().startswith("root")
    origin = (
        "pelvis (root-relative; not world)"
        if root
        else "global pelvis path (camera-aligned before hip centering)"
    )
    axes = "X lateral · Y up · Z depth (source-camera frame)"
    if distances_are_calibrated(recording):
        dist = "calibrated world units"
    else:
        dist = "stature-scaled estimates — not mocap-calibrated"
    extra = (
        " Treadmill belt travel is not the same measurement as overground path."
        if not root
        else " Root-relative length is limb motion about the pelvis, not overground travel."
    )
    return f"Origin: {origin}. Axes: {axes}. Distances: {dist}.{extra}"


def _finite_xyz(point: Any) -> tuple[float, float, float] | None:
    if point is None:
        return None
    if isinstance(point, (tuple, list)) and len(point) >= 3:
        coords = (float(point[0]), float(point[1]), float(point[2]))
    else:
        try:
            coords = (float(point.x), float(point.y), float(point.z))
        except (AttributeError, TypeError, ValueError):
            return None
    if not all(math.isfinite(v) for v in coords):
        return None
    return coords


def cumulative_path_length_m(
    path: Sequence[Any],
    *,
    end_index: int | None = None,
    times_s: Sequence[float] | None = None,
    fps: float | None = None,
) -> float | None:
    """Sum of finite connected segment lengths up to ``end_index`` (inclusive).

    When ``times_s`` is provided, pairs separated by more than ``1.5/fps`` are
    tracking gaps and are not added. Does not invent interpolated samples.
    """
    if not path:
        return None
    last = len(path) - 1 if end_index is None else int(end_index)
    last = max(0, min(last, len(path) - 1))
    max_dt = None
    if times_s is not None:
        from stablewalk.analysis.joint_path_trace import max_connected_dt

        max_dt = max_connected_dt(float(fps or 30.0))
    total = 0.0
    added = False
    prev = _finite_xyz(path[0])
    prev_t = float(times_s[0]) if times_s is not None and times_s else None
    for i, item in enumerate(path[1 : last + 1], start=1):
        cur = _finite_xyz(item)
        cur_t = float(times_s[i]) if times_s is not None and i < len(times_s) else None
        if prev is None or cur is None:
            prev = cur
            prev_t = cur_t
            continue
        if max_dt is not None and prev_t is not None and cur_t is not None:
            dt = cur_t - prev_t
            if dt <= 1e-12 or dt > max_dt + 1e-12:
                prev = cur
                prev_t = cur_t
                continue
        total += math.dist(prev, cur)
        added = True
        prev = cur
        prev_t = cur_t
    if last == 0 and not added:
        return 0.0
    return total


def displacement_m(
    path: Sequence[Any],
    *,
    end_index: int | None = None,
    origin_locked: bool = False,
) -> float | None:
    """Euclidean first-valid → current/last-valid distance in this frame.

    Returns None (display N/A) when the value is unavailable: empty path,
    current still on the first sample, non-finite endpoints, or a
    pelvis-centered origin that cannot translate (``origin_locked``).
    """
    if not path or origin_locked:
        return None
    start = None
    start_i = 0
    for i, item in enumerate(path):
        start = _finite_xyz(item)
        if start is not None:
            start_i = i
            break
    if start is None:
        return None
    last = len(path) - 1 if end_index is None else int(end_index)
    last = max(start_i, min(last, len(path) - 1))
    if last == start_i:
        return None
    end = _finite_xyz(path[last])
    if end is None:
        return None
    return math.dist(start, end)


def format_est_cm(meters: float | None) -> str:
    """Display a stature-scaled length in centimetres. Unavailable is N/A."""
    if meters is None or not math.isfinite(float(meters)):
        return "N/A"
    return f"{float(meters) * 100.0:.1f} cm est."


def path_and_displacement_labels(
    path: Sequence[Any],
    *,
    current_index: int | None = None,
) -> tuple[str, str]:
    """Return (path-length label, displacement label) for HUD / KPI cells."""
    return (
        format_est_cm(cumulative_path_length_m(path, end_index=current_index)),
        format_est_cm(displacement_m(path, end_index=current_index)),
    )


__all__ = [
    "CAMERA_VIEW_ALIASES",
    "CAMERA_VIEW_FRONTAL",
    "CAMERA_VIEW_PERSPECTIVE",
    "CAMERA_VIEW_SAGITTAL",
    "CAMERA_VIEW_SHORT_LABELS",
    "CAMERA_VIEW_SOURCE",
    "CAMERA_VIEW_TRANSVERSE",
    "NAMED_CAMERA_VIEWS",
    "coordinate_frame_caption",
    "cumulative_path_length_m",
    "displacement_m",
    "distances_are_calibrated",
    "format_est_cm",
    "named_camera_views",
    "path_and_displacement_labels",
    "resolve_camera_view_name",
    "source_camera_view_available",
]
