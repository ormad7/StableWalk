"""
Human skeleton renderer for ``GaitMotionRecording`` / ``SkeletonSnapshot``.

Front-view stick figure aligned with MediaPipe pose topology: pelvis → chest →
neck → head, bilateral arms and legs, ankle → heel → toe. Designed for academic
gait analysis dashboards (clean lines, joint dots, selection highlights).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from stablewalk.models.gait_motion import SkeletonSnapshot, Vec3
from stablewalk.coordinates.coordinate_map import canonical_to_visualization_oblique
from stablewalk.models.joint_registry import JOINT_DISPLAY_NAMES, ROOT_JOINT_ID
from stablewalk.ui.dof_selection import PICKABLE_JOINTS
from stablewalk.ui.colors import ACCENT, ACCENT_ALT, BORDER, MUTED, PANEL, TEXT, WARNING

# Side colour coding
_COLOR_LEFT = ACCENT
_COLOR_RIGHT = WARNING
_COLOR_CENTER = ACCENT_ALT
_COLOR_JOINT_FILL = "#d8dee9"
_COLOR_JOINT_EDGE = "#ffffff"
_COLOR_SELECT_BONE = "#ffb020"
_COLOR_SELECT_JOINT = "#ffe566"
_COLOR_SELECT_RING = "#ff6b6b"
_COLOR_MOTION_ARROW = "#a78bfa"
_COLOR_GROUND = BORDER

# MediaPipe-aligned topology (parent → child, side for colour)
# Trunk follows: pelvis → chest (spine) → shoulders / neck → head
_GAIT_EDGES: tuple[tuple[str, str, str], ...] = (
    (ROOT_JOINT_ID, "left_hip", "left"),
    (ROOT_JOINT_ID, "right_hip", "right"),
    (ROOT_JOINT_ID, "spine", "center"),
    ("spine", "neck", "center"),
    ("neck", "head", "center"),
    ("neck", "left_shoulder", "left"),
    ("neck", "right_shoulder", "right"),
    ("left_shoulder", "left_elbow", "left"),
    ("left_elbow", "left_wrist", "left"),
    ("right_shoulder", "right_elbow", "right"),
    ("right_elbow", "right_wrist", "right"),
    ("left_hip", "left_knee", "left"),
    ("left_knee", "left_ankle", "left"),
    ("right_hip", "right_knee", "right"),
    ("right_knee", "right_ankle", "right"),
)

_LIMB_JOINTS: frozenset[str] = frozenset(
    {
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_toe",
        "right_toe",
    }
)

_TRUNK_DERIVED: frozenset[str] = frozenset({ROOT_JOINT_ID, "spine", "neck"})

_MARKER_JOINTS: tuple[str, ...] = (
    "head",
    "neck",
    "spine",
    ROOT_JOINT_ID,
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_toe",
    "right_toe",
)

_MAX_CALLOUT_LABELS = 3

# Axes fill the canvas evenly — symmetric inset, no side bias.
_SKEL_INSET = 0.005
_SKEL_AX_LEFT = _SKEL_INSET
_SKEL_AX_BOTTOM = _SKEL_INSET
_SKEL_AX_WIDTH = 1.0 - 2.0 * _SKEL_INSET
_SKEL_AX_HEIGHT = 1.0 - 2.0 * _SKEL_INSET
SKELETON_PANEL_MARGINS = dict(
    left=_SKEL_AX_LEFT,
    bottom=_SKEL_AX_BOTTOM,
    right=_SKEL_AX_LEFT + _SKEL_AX_WIDTH,
    top=_SKEL_AX_BOTTOM + _SKEL_AX_HEIGHT,
)
_VIEW_PAD_FRAC = 0.018
# Slightly larger margin for the sequence-global view box so the body always has
# comfortable headroom and is never cropped at the extremes of the walk cycle.
_GLOBAL_VIEW_PAD_FRAC = 0.06
_MIN_WIDTH_FRAC_OF_HEIGHT = 0.14
_VIEW_MIN_HEIGHT_FRAC = 0.72
_SKELETON_VIEW_FILL = 0.94
# Headroom around the selected-point callout so its text box is never clipped by
# the panel edge (the label uses a padded bbox drawn with clip_on=False).
_LABEL_VIEW_PAD_FRAC = 0.11

# Landmarks used for body span / view limits (include feet so nothing is cropped).
_VIEW_SPAN_JOINTS: tuple[str, ...] = (
    "head",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    ROOT_JOINT_ID,
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_toe",
    "right_toe",
)

# Landmarks used for height estimates (ignore outlier toes/heels when measuring).
_CORE_SPAN_JOINTS: tuple[str, ...] = (
    "head",
    "neck",
    "left_shoulder",
    "right_shoulder",
    ROOT_JOINT_ID,
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Skip segments longer than this × body height (guards bad pose data).
_MAX_SEGMENT_FRAC = 0.50
_MIN_SEGMENT_FRAC = 0.012

# Display modes (internal key → GUI label)
SKELETON_DISPLAY_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("gait", "Walking Skeleton"),
    ("2d_pose", "2D Pose Skeleton"),
    ("3d_normalized", "3D Normalized Skeleton"),
    ("biomechanical", "Biomechanical Skeleton"),
)
DEFAULT_SKELETON_DISPLAY_MODE = "gait"
SKELETON_MODE_LABELS: tuple[str, ...] = tuple(l for _, l in SKELETON_DISPLAY_MODE_OPTIONS)
LABEL_TO_SKELETON_MODE: dict[str, str] = {l: k for k, l in SKELETON_DISPLAY_MODE_OPTIONS}
MODE_TO_SKELETON_LABEL: dict[str, str] = {k: l for k, l in SKELETON_DISPLAY_MODE_OPTIONS}

# Biomechanical / OpenSim-style segment chain (torso · pelvis · limbs · feet)
_BIOMECH_EDGES: tuple[tuple[str, str, str], ...] = (
    (ROOT_JOINT_ID, "spine", "center"),
    ("spine", "neck", "center"),
    ("neck", "head", "center"),
    ("left_hip", "right_hip", "center"),
    ("left_shoulder", "right_shoulder", "center"),
    (ROOT_JOINT_ID, "left_hip", "left"),
    (ROOT_JOINT_ID, "right_hip", "right"),
    ("spine", "left_shoulder", "left"),
    ("spine", "right_shoulder", "right"),
    ("left_shoulder", "left_elbow", "left"),
    ("left_elbow", "left_wrist", "left"),
    ("right_shoulder", "right_elbow", "right"),
    ("right_elbow", "right_wrist", "right"),
    ("left_hip", "left_knee", "left"),
    ("left_knee", "left_ankle", "left"),
    ("right_hip", "right_knee", "right"),
    ("right_knee", "right_ankle", "right"),
)

_BIOMECH_JOINT_DOTS: frozenset[str] = frozenset(
    {
        "head",
        "neck",
        "spine",
        ROOT_JOINT_ID,
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    }
)


def setup_skeleton_axes(ax: Axes, *, display_mode: str = DEFAULT_SKELETON_DISPLAY_MODE) -> None:
    """Front-view 2D axes tuned for the dashboard skeleton panel."""
    del display_mode  # axis styling is mode-neutral in the dashboard
    ax.set_facecolor(PANEL)
    ax.figure.patch.set_facecolor(PANEL)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(colors=MUTED, labelsize=7, length=0, pad=0)
    ax.margins(0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def _layout_skeleton_figure(ax: Axes) -> None:
    """Center the skeleton with inset so it stays inside the panel bounds."""
    fig = ax.figure
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    ax.set_position(
        [_SKEL_AX_LEFT, _SKEL_AX_BOTTOM, _SKEL_AX_WIDTH, _SKEL_AX_HEIGHT]
    )
    fig.patch.set_facecolor(PANEL)


def relayout_skeleton_viewport(ax: Axes) -> None:
    """Re-fit the skeleton viewport after a panel resize (no pose recompute)."""
    _layout_skeleton_figure(ax)
    snapshot = getattr(ax, "_sw_view_snapshot", None)
    if snapshot is None:
        return
    anchors = getattr(ax, "_sw_label_anchors", None)
    _apply_view_limits(ax, snapshot, extra_points=anchors)


def _drawable_aspect(ax: Axes) -> float:
    """Width / height of the skeleton drawable area (widget × axes box)."""
    canvas = ax.figure.canvas
    w_px, h_px = 0.0, 0.0
    widget_fn = getattr(canvas, "get_tk_widget", None)
    if callable(widget_fn):
        tk_widget = widget_fn()
        if tk_widget is not None:
            tk_widget.update_idletasks()
            w_px = float(tk_widget.winfo_width())
            h_px = float(tk_widget.winfo_height())
    if w_px < 60.0 or h_px < 60.0:
        fig = ax.figure
        w_in, h_in = fig.get_size_inches()
        dpi = fig.get_dpi()
        w_px = w_in * dpi
        h_px = h_in * dpi
    box = ax.get_position()
    draw_w = w_px * max(float(box.width), 1e-6)
    draw_h = h_px * max(float(box.height), 1e-6)
    if draw_h < 1e-6:
        return 1.0
    return draw_w / draw_h


def _drawable_pixels(ax: Axes) -> tuple[float, float]:
    """Rounded (width, height) in pixels of the panel widget/figure.

    Used as a stable cache key so the fixed-view limits only recompute when the
    panel is actually resized (not on every frame).
    """
    canvas = ax.figure.canvas
    w_px, h_px = 0.0, 0.0
    widget_fn = getattr(canvas, "get_tk_widget", None)
    if callable(widget_fn):
        tk_widget = widget_fn()
        if tk_widget is not None:
            tk_widget.update_idletasks()
            w_px = float(tk_widget.winfo_width())
            h_px = float(tk_widget.winfo_height())
    if w_px < 60.0 or h_px < 60.0:
        fig = ax.figure
        w_in, h_in = fig.get_size_inches()
        dpi = fig.get_dpi()
        w_px = w_in * dpi
        h_px = h_in * dpi
    return (round(w_px), round(h_px))


def _panel_aspect(ax: Axes) -> float:
    """Backward-compatible alias for viewport fitting."""
    return _drawable_aspect(ax)


def _shoulder_and_hip_spans(snapshot: SkeletonSnapshot) -> tuple[float, float]:
    ls = _xy(snapshot, "left_shoulder")
    rs = _xy(snapshot, "right_shoulder")
    lh = _xy(snapshot, "left_hip")
    rh = _xy(snapshot, "right_hip")
    shoulder_span = _segment_len(ls, rs) if ls and rs else 0.0
    hip_span = _segment_len(lh, rh) if lh and rh else 0.0
    return shoulder_span, hip_span


def _vertical_body_center(snapshot: SkeletonSnapshot, body_h: float) -> float:
    """Center the viewport on the full body height (head to feet)."""
    head = _xy(snapshot, "head")
    foot_pts = [
        _xy(snapshot, j)
        for j in ("left_toe", "right_toe", "left_heel", "right_heel", "left_ankle", "right_ankle")
    ]
    foot_pts = [p for p in foot_pts if p]
    if head and foot_pts:
        top_y = head[1] + max(body_h * 0.05, 0.02)
        bottom_y = min(p[1] for p in foot_pts) - body_h * 0.03
        return (top_y + bottom_y) / 2.0
    _min_x, _max_x, min_y, max_y, _cx, cy = _body_bounds(snapshot)
    return cy if max_y > min_y else cy


def _xy(snapshot: SkeletonSnapshot, joint_id: str) -> tuple[float, float] | None:
    display = getattr(snapshot, "_sw_display_xy", None)
    if isinstance(display, dict) and joint_id in display:
        return display[joint_id]
    sample = snapshot.joints.get(joint_id)
    if not sample:
        return None
    return (sample.position.x, sample.position.y)


def _build_display_xy(snapshot: SkeletonSnapshot) -> dict[str, tuple[float, float]]:
    """
    Build front-view coordinates: preserve MediaPipe limb landmarks, derive trunk.

    Limb joints (shoulders, elbows, wrists, hips, knees, ankles, heels, toes)
    keep their measured positions so the figure tracks the video pose.
    """
    measured: dict[str, tuple[float, float]] = {}
    for jid, sample in snapshot.joints.items():
        if jid in _LIMB_JOINTS or jid == "head":
            measured[jid] = (sample.position.x, sample.position.y)

    # Also keep any explicitly stored trunk points
    for jid in (ROOT_JOINT_ID, "spine", "neck"):
        sample = snapshot.joints.get(jid)
        if sample:
            measured.setdefault(jid, (sample.position.x, sample.position.y))

    out = dict(measured)

    _refine_trunk(out)

    # Fill any remaining limb joints from snapshot (fallback)
    for jid in _MARKER_JOINTS:
        if jid not in out:
            pt = _xy(snapshot, jid)
            if pt:
                out[jid] = pt

    return out


# Trunk interpolation fractions (pelvis → shoulder mid → head)
_SPINE_FROM_PELVIS = 0.68
_NECK_FROM_SPINE = 0.42


def _refine_trunk(out: dict[str, tuple[float, float]]) -> None:
    """Derive pelvis / spine / neck with stable human proportions."""
    lh = out.get("left_hip")
    rh = out.get("right_hip")
    pelvis = out.get(ROOT_JOINT_ID)
    if lh and rh:
        pelvis = pelvis or ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
        out[ROOT_JOINT_ID] = pelvis

    ls = out.get("left_shoulder")
    rs = out.get("right_shoulder")
    shoulder_mid = (
        ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0) if ls and rs else None
    )
    head = out.get("head")

    if shoulder_mid and pelvis:
        out["spine"] = (
            pelvis[0] + _SPINE_FROM_PELVIS * (shoulder_mid[0] - pelvis[0]),
            pelvis[1] + _SPINE_FROM_PELVIS * (shoulder_mid[1] - pelvis[1]),
        )
    spine = out.get("spine")
    if spine and head:
        out["neck"] = (
            spine[0] + _NECK_FROM_SPINE * (head[0] - spine[0]),
            spine[1] + _NECK_FROM_SPINE * (head[1] - spine[1]),
        )
    elif spine and shoulder_mid:
        out["neck"] = (
            spine[0] + 0.55 * (shoulder_mid[0] - spine[0]),
            spine[1] + 0.55 * (shoulder_mid[1] - spine[1]),
        )


def _build_display_oblique(snapshot: SkeletonSnapshot) -> dict[str, tuple[float, float]]:
    """Oblique front view from hip-centered 3D coords (mild depth cue, no stretch)."""
    out: dict[str, tuple[float, float]] = {}
    for jid, sample in snapshot.joints.items():
        out[jid] = canonical_to_visualization_oblique(
            sample.position.x,
            sample.position.y,
            sample.position.z,
        )
    _refine_trunk(out)
    return out


def _build_display_2d_pose(snapshot: SkeletonSnapshot) -> dict[str, tuple[float, float]]:
    """
    Video-frame 2D coords (matches the pose overlay).

    Prefer ``image_xy`` metadata from MediaPipe; fall back to oblique 3D projection.
    """
    raw_meta = snapshot.metadata.get("image_xy")
    if isinstance(raw_meta, dict) and raw_meta:
        out: dict[str, tuple[float, float]] = {}
        for jid, val in raw_meta.items():
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                out[str(jid)] = (float(val[0]), float(val[1]))
        if len(out) >= 6:
            _refine_trunk(out)
            return out
    return _build_display_oblique(snapshot)


def _build_display_3d_normalized(snapshot: SkeletonSnapshot) -> dict[str, tuple[float, float]]:
    """Oblique projection of hip-centered 3D coordinates (uniform scale, no shear)."""
    out: dict[str, tuple[float, float]] = {}
    for jid, sample in snapshot.joints.items():
        out[jid] = canonical_to_visualization_oblique(
            sample.position.x,
            sample.position.y,
            sample.position.z,
        )

    _refine_trunk(out)
    return out


def _build_display_coords(
    snapshot: SkeletonSnapshot, display_mode: str
) -> dict[str, tuple[float, float]]:
    if display_mode == "2d_pose":
        return _build_display_2d_pose(snapshot)
    if display_mode == "3d_normalized":
        return _build_display_3d_normalized(snapshot)
    # Walking / biomechanical views: true front projection preserves limb proportions.
    return _build_display_xy(snapshot)


def _joint_within_body(
    jid: str,
    pt: tuple[float, float],
    snapshot: SkeletonSnapshot,
    height: float,
) -> bool:
    """Hide stray markers far from the body (bad landmark outliers)."""
    min_x, max_x, min_y, max_y, cx, cy = _body_bounds(snapshot)
    pad_x = height * 0.28
    pad_y = height * 0.18
    if not (min_x - pad_x <= pt[0] <= max_x + pad_x):
        return False
    if not (min_y - pad_y <= pt[1] <= max_y + pad_y):
        return False
    return True


def _segment_len(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])


def _plausible_segment(
    parent: str,
    child: str,
    p0: tuple[float, float],
    p1: tuple[float, float],
    height: float,
) -> bool:
    """Reject impossibly long/short bones from bad landmark data."""
    length = _segment_len(p0, p1)
    if length < height * _MIN_SEGMENT_FRAC:
        return False
    limit = height * _MAX_SEGMENT_FRAC
    if child == "head" or parent in (ROOT_JOINT_ID, "spine", "neck", "head"):
        limit = height * 0.55
    return length <= limit


def _robust_body_height(snapshot: SkeletonSnapshot) -> float:
    """Body height from core landmarks (robust to foot/toe outliers)."""
    ys: list[float] = []
    for jid in _CORE_SPAN_JOINTS:
        pt = _xy(snapshot, jid)
        if pt:
            ys.append(pt[1])
    if len(ys) < 2:
        return _body_height(snapshot)
    return max(ys) - min(ys)


def _body_height(snapshot: SkeletonSnapshot) -> float:
    ys: list[float] = []
    for jid in _MARKER_JOINTS:
        pt = _xy(snapshot, jid)
        if pt:
            ys.append(pt[1])
    if len(ys) < 2:
        return 1.0
    return max(ys) - min(ys)


def _side_color(side: str) -> str:
    if side == "left":
        return _COLOR_LEFT
    if side == "right":
        return _COLOR_RIGHT
    return _COLOR_CENTER


def _bone_width(
    parent: str,
    child: str,
    height: float,
    *,
    display_mode: str = DEFAULT_SKELETON_DISPLAY_MODE,
) -> float:
    pair = f"{parent}_{child}"
    # Uniform academic stick figure — trunk slightly thicker than limbs.
    scale = 1.48
    if ROOT_JOINT_ID in pair or "spine" in pair or "neck" in pair or "head" in pair:
        w = max(height * 0.034, 2.2)
    elif "shoulder" in pair and "elbow" in pair:
        w = max(height * 0.029, 2.0)
    elif "elbow" in pair and "wrist" in pair:
        w = max(height * 0.027, 1.9)
    elif "hip" in pair and "knee" in pair:
        w = max(height * 0.030, 2.1)
    elif "knee" in pair and "ankle" in pair:
        w = max(height * 0.029, 2.0)
    elif any(x in pair for x in ("heel", "toe", "foot", "ankle")):
        w = max(height * 0.023, 1.7)
    elif "spine" in pair and "shoulder" in pair:
        w = max(height * 0.027, 1.9)
    elif "hip" in pair and ROOT_JOINT_ID in pair:
        w = max(height * 0.027, 1.9)
    else:
        w = max(height * 0.028, 2.0)

    w *= scale

    if display_mode == "biomechanical":
        if ROOT_JOINT_ID in pair or "spine" in pair or "neck" in pair:
            return w * 1.25
        if "hip" in pair and "knee" in pair:
            return w * 1.20
        if "knee" in pair and "ankle" in pair:
            return w * 1.15
    return w


def _foot_edges(snapshot: SkeletonSnapshot, side: str) -> list[tuple[str, str, str]]:
    ankle = f"{side}_ankle"
    heel = f"{side}_heel"
    toe = f"{side}_toe"
    edges: list[tuple[str, str, str]] = []
    if _xy(snapshot, heel):
        edges.append((ankle, heel, side))
        if _xy(snapshot, toe):
            edges.append((heel, toe, side))
    elif _xy(snapshot, toe):
        edges.append((ankle, toe, side))
    return edges


def _all_edges(
    snapshot: SkeletonSnapshot,
    display_mode: str = DEFAULT_SKELETON_DISPLAY_MODE,
) -> list[tuple[str, str, str]]:
    if display_mode == "biomechanical":
        return list(_BIOMECH_EDGES)
    edges = list(_GAIT_EDGES)
    edges.extend(_foot_edges(snapshot, "left"))
    edges.extend(_foot_edges(snapshot, "right"))
    return edges


def _draw_bone(
    ax: Axes,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    color: str,
    width: float,
    zorder: int = 3,
    alpha: float = 0.92,
) -> None:
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color=color,
        linewidth=width,
        solid_capstyle="round",
        zorder=zorder,
        alpha=alpha,
    )


def _draw_ground(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    height: float,
    *,
    floor_y: float | None = None,
    display_mode: str = "biomechanical",
) -> None:
    foot_pts = [
        _xy(snapshot, j)
        for j in ("left_toe", "right_toe", "left_heel", "right_heel", "left_ankle", "right_ankle")
    ]
    foot_pts = [p for p in foot_pts if p]
    if not foot_pts and floor_y is None:
        return
    pelvis = _xy(snapshot, ROOT_JOINT_ID) or _xy(snapshot, "spine")
    cx = pelvis[0] if pelvis else 0.0
    half = height * 0.32

    if floor_y is not None and display_mode != "2d_pose":
        y_floor = floor_y
    elif foot_pts:
        y_floor = min(p[1] for p in foot_pts) - height * 0.025
    else:
        y_floor = floor_y if floor_y is not None else 0.0

    ax.plot(
        [cx - half, cx + half],
        [y_floor, y_floor],
        color=_COLOR_GROUND,
        linewidth=0.9,
        linestyle=(0, (4, 3)),
        alpha=0.55,
        zorder=0,
    )

    if floor_y is not None and display_mode != "2d_pose":
        for side, joints in (
            ("left", ("left_toe", "left_heel", "left_ankle")),
            ("right", ("right_toe", "right_heel", "right_ankle")),
        ):
            lowest: tuple[float, float] | None = None
            for jid in joints:
                pt = _xy(snapshot, jid)
                if pt and (lowest is None or pt[1] < lowest[1]):
                    lowest = pt
            if lowest is None:
                continue
            ax.plot(
                [lowest[0], lowest[0]],
                [lowest[1], y_floor],
                color=_COLOR_GROUND,
                linewidth=0.6,
                linestyle=":",
                alpha=0.35,
                zorder=0,
            )


def _draw_head(ax: Axes, snapshot: SkeletonSnapshot, height: float) -> None:
    """Small head dot at the nose landmark."""
    from matplotlib.patches import Circle

    head = _xy(snapshot, "head")
    if not head:
        return
    r = max(min(height * 0.048, 0.070), 0.018)
    ax.add_patch(
        Circle(
            head,
            r,
            facecolor=_COLOR_JOINT_FILL,
            edgecolor=_COLOR_CENTER,
            linewidth=0.9,
            zorder=6,
            alpha=0.95,
        )
    )


def _joint_marker_radius(jid: str, height: float) -> float:
    """Joint dot size by anatomical role."""
    base = max(height * 0.017, 0.014)
    if jid in ("left_knee", "right_knee", "left_elbow", "right_elbow"):
        return base * 1.18
    if jid in ("left_hip", "right_hip", "left_shoulder", "right_shoulder"):
        return base * 1.12
    if jid in (ROOT_JOINT_ID, "spine", "neck"):
        return base * 0.78
    if "heel" in jid or "toe" in jid:
        return base * 0.9
    return base


def _draw_joint_markers(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    height: float,
    *,
    highlight_joints: set[str] | None,
    display_mode: str = DEFAULT_SKELETON_DISPLAY_MODE,
) -> None:
    from matplotlib.patches import Circle

    r_base = max(height * 0.011, 0.009)
    highlight = highlight_joints or set()
    joint_ids = _MARKER_JOINTS
    if display_mode == "biomechanical":
        joint_ids = tuple(j for j in _MARKER_JOINTS if j in _BIOMECH_JOINT_DOTS)
    elif display_mode == "gait":
        # Walking view: limb + head dots only (trunk shown as bones, not dots).
        joint_ids = tuple(
            j for j in _MARKER_JOINTS
            if j not in (ROOT_JOINT_ID, "spine", "neck")
        )

    for jid in joint_ids:
        if jid == "head":
            continue  # drawn separately
        pt = _xy(snapshot, jid)
        if not pt or jid in highlight:
            continue
        if not _joint_within_body(jid, pt, snapshot, height):
            continue
        if jid.startswith("left"):
            edge = _COLOR_LEFT
        elif jid.startswith("right"):
            edge = _COLOR_RIGHT
        else:
            edge = _COLOR_CENTER
        r = _joint_marker_radius(jid, height)
        ax.add_patch(
            Circle(
                pt,
                r,
                facecolor=_COLOR_JOINT_FILL,
                edgecolor=edge,
                linewidth=0.9,
                zorder=7,
                alpha=0.95,
            )
        )


def _draw_selection_highlights(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    highlight_joints: set[str],
    height: float,
    *,
    display_mode: str = DEFAULT_SKELETON_DISPLAY_MODE,
    primary_joint: str | None = None,
) -> None:
    from matplotlib.patches import Circle

    from stablewalk.ui.theme import DOF_TRAJ_DOT_COLOR, DOF_TRAJ_PATH_COLOR

    if not highlight_joints:
        return

    drawn: set[tuple[str, str]] = set()
    for parent, child, _side in _all_edges(snapshot, display_mode):
        if parent not in highlight_joints or child not in highlight_joints:
            continue
        key = tuple(sorted((parent, child)))
        if key in drawn:
            continue
        p0, p1 = _xy(snapshot, parent), _xy(snapshot, child)
        if not p0 or not p1:
            continue
        if not _plausible_segment(parent, child, p0, p1, height):
            continue
        drawn.add(key)
        _draw_bone(
            ax,
            p0,
            p1,
            color=DOF_TRAJ_PATH_COLOR,
            width=_bone_width(parent, child, height, display_mode=display_mode) + 1.4,
            zorder=12,
            alpha=0.92,
        )

    for jid in highlight_joints:
        pt = _xy(snapshot, jid)
        if not pt:
            continue
        is_primary = primary_joint is not None and jid == primary_joint
        r = max(height * (0.021 if is_primary else 0.014), 0.010 if is_primary else 0.008)
        ring_color = DOF_TRAJ_DOT_COLOR if is_primary else DOF_TRAJ_PATH_COLOR
        fill_color = DOF_TRAJ_DOT_COLOR if is_primary else _COLOR_JOINT_FILL
        outer_scale = 1.45 if is_primary else 1.22
        inner_scale = 0.98 if is_primary else 0.88
        ax.add_patch(
            Circle(
                pt,
                r * outer_scale,
                facecolor="none",
                edgecolor=ring_color,
                linewidth=2.2 if is_primary else 1.2,
                zorder=14,
            )
        )
        ax.add_patch(
            Circle(
                pt,
                r * inner_scale,
                facecolor=fill_color,
                edgecolor=_COLOR_JOINT_EDGE if is_primary else ring_color,
                linewidth=1.3 if is_primary else 0.85,
                zorder=15,
            )
        )


def _body_center_x(snapshot: SkeletonSnapshot) -> float:
    pelvis = _xy(snapshot, ROOT_JOINT_ID)
    if pelvis:
        return pelvis[0]
    xs = [p[0] for jid in _MARKER_JOINTS if (p := _xy(snapshot, jid))]
    return sum(xs) / len(xs) if xs else 0.0


def _body_bounds(snapshot: SkeletonSnapshot) -> tuple[float, float, float, float, float, float]:
    """Return (min_x, max_x, min_y, max_y, cx, cy) from core landmarks."""
    xs: list[float] = []
    ys: list[float] = []
    for jid in _CORE_SPAN_JOINTS:
        pt = _xy(snapshot, jid)
        if pt:
            xs.append(pt[0])
            ys.append(pt[1])
    if not xs:
        return -0.3, 0.3, 0.0, 1.0, 0.0, 0.5
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return min_x, max_x, min_y, max_y, (min_x + max_x) / 2.0, (min_y + max_y) / 2.0


def _label_side(jid: str, pt: tuple[float, float], cx: float, body_h: float) -> str:
    """Place labels on the anatomical side (hips/knees never go to center)."""
    if jid.startswith("left_"):
        return "left"
    if jid.startswith("right_"):
        return "right"
    if jid in ("head", "neck"):
        return "top"
    margin = body_h * 0.04
    if pt[0] < cx - margin:
        return "left"
    if pt[0] > cx + margin:
        return "right"
    return "right" if pt[0] >= cx else "left"


def _nudge_y(y: float, placed: list[float], min_sep: float) -> float:
    """Shift label y until it clears previously placed labels on the same side."""
    if not placed:
        return y
    out = y
    for _ in range(12):
        if all(abs(out - py) >= min_sep for py in placed):
            return out
        # Prefer shifting away from cluster centroid.
        avg = sum(placed) / len(placed)
        out += min_sep if out <= avg else -min_sep
    return out


def _layout_callout_positions(
    items: list[tuple[str, str]],
    snapshot: SkeletonSnapshot,
    body_h: float,
) -> list[tuple[str, str, tuple[float, float], tuple[float, float], str, str]]:
    """
    Compute label anchors outside the skeleton with side-aware stacking.

    Returns (joint_id, label, joint_xy, text_xy, ha, va) per callout.
    """
    min_x, max_x, min_y, max_y, cx, _cy = _body_bounds(snapshot)
    pad_x = max(body_h * 0.42, 0.26)
    pad_y = max(body_h * 0.16, 0.09)
    min_sep = max(body_h * 0.13, 0.075)

    buckets: dict[str, list[tuple[str, str, tuple[float, float]]]] = {
        "left": [],
        "right": [],
        "top": [],
    }
    for jid, label in items:
        pt = _xy(snapshot, jid)
        if not pt:
            continue
        side = _label_side(jid, pt, cx, body_h)
        buckets[side if side in buckets else "right"].append((jid, label, pt))

    layouts: list[tuple[str, str, tuple[float, float], tuple[float, float], str, str]] = []

    placed_left: list[float] = []
    for jid, label, pt in sorted(buckets["left"], key=lambda t: -t[2][1]):
        tx = min_x - pad_x
        ty = _nudge_y(pt[1], placed_left, min_sep)
        placed_left.append(ty)
        layouts.append((jid, label, pt, (tx, ty), "right", "center"))

    placed_right: list[float] = []
    for jid, label, pt in sorted(buckets["right"], key=lambda t: -t[2][1]):
        tx = max_x + pad_x
        ty = _nudge_y(pt[1], placed_right, min_sep)
        placed_right.append(ty)
        layouts.append((jid, label, pt, (tx, ty), "left", "center"))

    placed_top: list[float] = []
    for jid, label, pt in sorted(buckets["top"], key=lambda t: t[2][0]):
        ty = max_y + pad_y
        tx = _nudge_y(pt[0], placed_top, min_sep * 0.85)
        placed_top.append(tx)
        layouts.append((jid, label, pt, (tx, ty), "center", "bottom"))

    return layouts


def _labels_overlap(a: tuple[float, float], b: tuple[float, float], min_dist: float) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) < min_dist


def _body_bbox(
    snapshot: SkeletonSnapshot,
    *,
    height: float,
) -> tuple[float, float, float, float]:
    """Axis-aligned bounds of visible body landmarks."""
    points: list[tuple[float, float]] = []
    for jid in _VIEW_SPAN_JOINTS:
        pt = _xy(snapshot, jid)
        if pt:
            points.append(pt)
    if not points:
        return (-0.2, 0.2, 0.0, height)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad = height * 0.02
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def _draw_inline_joint_labels(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    labeled_joints: dict[str, str],
    height: float,
) -> list[tuple[float, float]]:
    """Place the selected-point label outside the body with a subtle leader."""
    from stablewalk.ui.theme import DOF_TRAJ_DOT_COLOR

    anchors: list[tuple[float, float]] = []
    if not labeled_joints:
        return anchors

    min_x, max_x, min_y, max_y = _body_bbox(snapshot, height=height)
    cx = _body_center_x(snapshot)
    offset = max(height * 0.044, 0.030)
    items = list(labeled_joints.items())[:1]

    for jid, label in items:
        pt = _xy(snapshot, jid)
        if not pt:
            continue

        # Any explicitly left/right joint (hip, shoulder, knee, ankle, …) is
        # labeled on its own anatomical side at the joint's height. This holds
        # even when the subject walks away from camera and the body projects
        # very narrow — previously such poses pushed the label above the head
        # where the panel clipped it. Only truly central joints (head, neck,
        # spine, pelvis) get the top callout.
        if jid.startswith("left_"):
            tx = min_x - offset * 0.34
            ty = pt[1]
            ha, va = "right", "center"
        elif jid.startswith("right_"):
            tx = max_x + offset * 0.34
            ty = pt[1]
            ha, va = "left", "center"
        else:
            tx = pt[0]
            ty = max(max_y + offset * 0.38, pt[1] + offset * 0.48)
            ha, va = "center", "bottom"

        anchors.append((tx, ty))
        ax.plot(
            [pt[0], tx],
            [pt[1], ty],
            color=DOF_TRAJ_DOT_COLOR,
            linewidth=1.0,
            alpha=0.62,
            solid_capstyle="round",
            zorder=19,
        )
        ax.text(
            tx,
            ty,
            label,
            ha=ha,
            va=va,
            fontsize=9.5,
            fontweight="bold",
            color=TEXT,
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor=PANEL,
                edgecolor=DOF_TRAJ_DOT_COLOR,
                linewidth=1.2,
                alpha=0.98,
            ),
            zorder=20,
            clip_on=False,
        )
    return anchors


def _draw_foot_skeleton_labels(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    foot_labels: tuple["FootSkeletonLabel", "FootSkeletonLabel"],
    height: float,
    *,
    avoid_anchors: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Compact L/R foot-to-floor distance + contact labels near each foot."""
    from stablewalk.ui.foot_clearance_display import FootSkeletonLabel
    from stablewalk.ui.theme import ACCENT, MUTED, ORANGE, PANEL

    anchors: list[tuple[float, float]] = []
    offset = max(height * 0.035, 0.022)
    min_sep = max(height * 0.07, 0.035)
    existing = list(avoid_anchors or [])

    foot_targets = (
        ("left", foot_labels[0], ("left_ankle", "left_heel", "left_toe")),
        ("right", foot_labels[1], ("right_ankle", "right_heel", "right_toe")),
    )
    min_x, max_x, _min_y, _max_y = _body_bbox(snapshot, height=height)

    clearance_values: list[float | None] = [None, None]

    for idx, (side, label, joints) in enumerate(foot_targets):
        if not isinstance(label, FootSkeletonLabel):
            continue
        pt = None
        for jid in joints:
            pt = _xy(snapshot, jid)
            if pt:
                break
        if not pt:
            continue

        clearance_values[idx] = label.clearance_cm
        text = label.compact_text()
        if side == "left":
            tx = min(min_x - offset * 0.18, pt[0] - offset * 0.42)
            ha = "right"
        else:
            tx = max(max_x + offset * 0.18, pt[0] + offset * 0.42)
            ha = "left"
        ty = pt[1] - offset * 0.85

        for _ in range(6):
            too_close = False
            for ax_pt, ay_pt in existing:
                if math.hypot(tx - ax_pt, ty - ay_pt) < min_sep:
                    too_close = True
                    break
            if not too_close:
                break
            ty -= min_sep * 0.55
            if side == "left":
                tx -= min_sep * 0.25
            else:
                tx += min_sep * 0.25

        edge_color = ACCENT if label.contact else MUTED
        text_color = ORANGE if label.clearance_cm is not None else MUTED
        anchors.append((tx, ty))
        existing.append((tx, ty))
        ax.text(
            tx,
            ty,
            text,
            ha=ha,
            va="center",
            fontsize=7.0,
            fontweight="medium",
            color=text_color,
            linespacing=1.15,
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor=PANEL,
                edgecolor=edge_color,
                linewidth=0.8,
                alpha=0.94,
            ),
            zorder=18,
            clip_on=False,
        )

    ax._sw_foot_skeleton_clearance_cm = (  # type: ignore[attr-defined]
        clearance_values[0],
        clearance_values[1],
    )
    return anchors


def _draw_foot_contact_labels(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    foot_contact: tuple[int, int],
    height: float,
) -> list[tuple[float, float]]:
    """Deprecated — use ``_draw_foot_skeleton_labels`` (kept for imports)."""
    from stablewalk.ui.foot_clearance_display import FootSkeletonLabel

    left_on, right_on = foot_contact
    labels = (
        FootSkeletonLabel("left", None, bool(left_on)),
        FootSkeletonLabel("right", None, bool(right_on)),
    )
    return _draw_foot_skeleton_labels(ax, snapshot, labels, height)


_COLOR_OSIM_DIRECT = "#4ade80"
_COLOR_OSIM_DERIVED = "#fbbf24"
_COLOR_OSIM_LOW_CONF = "#f87171"


def _draw_opensim_marker_debug(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    height: float,
    *,
    display_mode: str,
) -> list[tuple[float, float]]:
    """
    Overlay reconstructed OpenSim markers (disabled by default).

    Styles: DIRECT = green diamond, DERIVED_ANATOMICAL = amber triangle,
    LOW_CONFIDENCE = red hollow circle.
    """
    from matplotlib.lines import Line2D

    from stablewalk.biomechanics.marker_reconstruction import (
        DEFAULT_RECONSTRUCTION_CONFIG,
        landmarks_from_skeleton_joints,
        project_marker_to_display_xy,
        reconstruct_markers_single_frame,
    )

    joints = {
        jid: (sample.position.x, sample.position.y, sample.position.z)
        for jid, sample in snapshot.joints.items()
    }
    landmarks = landmarks_from_skeleton_joints(joints)
    if len(landmarks) < 6:
        return []

    markers = reconstruct_markers_single_frame(landmarks)
    anchors: list[tuple[float, float]] = []
    thr = DEFAULT_RECONSTRUCTION_CONFIG.high_confidence_threshold
    size = max(height * 0.012, 0.018)

    for name, result in markers.items():
        if result.position is None:
            continue
        pt = project_marker_to_display_xy(result.position, display_mode=display_mode)
        anchors.append(pt)
        low_conf = result.confidence < thr
        if low_conf:
            color = _COLOR_OSIM_LOW_CONF
            marker_style = "o"
            mfc = "none"
            mew = 1.4
        elif result.mapping_type == "DIRECT":
            color = _COLOR_OSIM_DIRECT
            marker_style = "D"
            mfc = color
            mew = 0.8
        else:
            color = _COLOR_OSIM_DERIVED
            marker_style = "^"
            mfc = color
            mew = 0.8

        ax.plot(
            pt[0],
            pt[1],
            marker=marker_style,
            markersize=size * 42,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=mew,
            linestyle="None",
            zorder=22,
            clip_on=False,
        )

    legend_handles = [
        Line2D(
            [0], [0], marker="D", color="w", markerfacecolor=_COLOR_OSIM_DIRECT,
            markersize=6, label="OpenSim DIRECT",
        ),
        Line2D(
            [0], [0], marker="^", color="w", markerfacecolor=_COLOR_OSIM_DERIVED,
            markersize=6, label="OpenSim DERIVED",
        ),
        Line2D(
            [0], [0], marker="o", color=_COLOR_OSIM_LOW_CONF, markerfacecolor="none",
            markersize=6, markeredgewidth=1.2, label="OpenSim LOW_CONF",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=6.5,
        framealpha=0.85,
        facecolor=PANEL,
        edgecolor=BORDER,
    )
    return anchors


def _draw_joint_callouts(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    labeled_joints: dict[str, str],
    height: float,
) -> list[tuple[float, float]]:
    """
    Place selected joint names outside the body with leader arrows.

    Returns text anchor positions so view limits can include the labels.
    """
    items = list(labeled_joints.items())[:_MAX_CALLOUT_LABELS]
    if not items:
        return []

    layouts = _layout_callout_positions(items, snapshot, height)
    text_anchors: list[tuple[float, float]] = []
    min_sep = height * 0.10
    placed: list[tuple[float, float]] = []

    for jid, label, pt, (tx, ty), ha, va in layouts:
        # Final pass: resolve any cross-side overlap by nudging along the leader axis.
        for _ in range(8):
            if not any(_labels_overlap((tx, ty), p, min_sep) for p in placed):
                break
            if ha == "right":
                tx -= height * 0.05
            elif ha == "left":
                tx += height * 0.05
            else:
                ty += height * 0.05
        placed.append((tx, ty))
        text_anchors.append((tx, ty))

        ax.annotate(
            label,
            xy=pt,
            xytext=(tx, ty),
            ha=ha,
            va=va,
            fontsize=8,
            fontweight="medium",
            color=TEXT,
            annotation_clip=False,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor=PANEL,
                edgecolor=_COLOR_SELECT_RING,
                linewidth=0.85,
                alpha=0.96,
            ),
            arrowprops=dict(
                arrowstyle="-|>",
                color=_COLOR_SELECT_RING,
                linewidth=0.9,
                mutation_scale=8,
                shrinkA=3,
                shrinkB=5,
                connectionstyle="arc3,rad=0.12",
            ),
            zorder=20,
        )

    return text_anchors


def _draw_legend(ax: Axes) -> None:
    """Compact L/R color key — lower-left, out of the way of the figure."""
    from matplotlib.lines import Line2D

    x0, y0 = 0.025, 0.035
    fs = 7.0
    line_w = 0.07

    ax.text(
        x0,
        y0,
        "L",
        transform=ax.transAxes,
        color=_COLOR_LEFT,
        fontsize=fs,
        fontweight="bold",
        va="center",
        ha="left",
        zorder=30,
    )
    ax.add_line(
        Line2D(
            [x0 + 0.028, x0 + 0.028 + line_w],
            [y0, y0],
            transform=ax.transAxes,
            color=_COLOR_LEFT,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=30,
        )
    )
    ax.text(
        x0 + 0.11,
        y0,
        "R",
        transform=ax.transAxes,
        color=_COLOR_RIGHT,
        fontsize=fs,
        fontweight="bold",
        va="center",
        ha="left",
        zorder=30,
    )
    ax.add_line(
        Line2D(
            [x0 + 0.138, x0 + 0.138 + line_w],
            [y0, y0],
            transform=ax.transAxes,
            color=_COLOR_RIGHT,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=30,
        )
    )


def _attach_joint_pickers(ax: Axes, snapshot: SkeletonSnapshot, height: float) -> None:
    xs, ys, ids = [], [], []
    for jid in PICKABLE_JOINTS:
        pt = _xy(snapshot, jid)
        if pt:
            xs.append(pt[0])
            ys.append(pt[1])
            ids.append(jid)
    if not xs:
        ax._sw_joint_picker = None  # type: ignore[attr-defined]
        return
    size = max(80.0, (height * 420) ** 2)
    scatter = ax.scatter(
        xs, ys, s=size, c="#ffffff", alpha=0.01, edgecolors="none", zorder=18, picker=True
    )
    scatter._sw_joint_ids = ids  # type: ignore[attr-defined]
    ax._sw_joint_picker = scatter  # type: ignore[attr-defined]


def _draw_motion_arrows(
    ax: Axes,
    arrows: dict[str, tuple[Vec3, Vec3]],
    height: float,
) -> None:
    from matplotlib.patches import FancyArrowPatch

    min_len = height * 0.006
    for _joint_id, (current, nxt) in arrows.items():
        dx = nxt.x - current.x
        dy = nxt.y - current.y
        if math.hypot(dx, dy) < min_len:
            continue
        ax.add_patch(
            FancyArrowPatch(
                (current.x, current.y),
                (nxt.x, nxt.y),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.4,
                color=_COLOR_MOTION_ARROW,
                zorder=11,
                alpha=0.7,
            )
        )


def joint_id_from_pick(event) -> str | None:
    artist = getattr(event, "artist", None)
    if artist is None or not hasattr(artist, "_sw_joint_ids"):
        return None
    ind = getattr(event, "ind", None)
    if ind is None or len(ind) == 0:
        return None
    ids: list[str] = artist._sw_joint_ids  # type: ignore[attr-defined]
    idx = int(ind[0])
    if 0 <= idx < len(ids):
        return ids[idx]
    return None


def _single_frame_view_box(
    snapshot: SkeletonSnapshot,
) -> tuple[float, float, float, float] | None:
    """Per-frame content box (cx, cy, half_x, half_y) from the current pose.

    Used only when no stable sequence-global box has been supplied.
    """
    points: list[tuple[float, float]] = []
    for jid in _VIEW_SPAN_JOINTS:
        pt = _xy(snapshot, jid)
        if pt:
            points.append(pt)

    if not points:
        return None

    body_h = max(_robust_body_height(snapshot), 1e-3)
    head_pt = _xy(snapshot, "head")
    if head_pt:
        points.append((head_pt[0], head_pt[1] + max(body_h * 0.035, 0.018)))

    foot_pts = [
        _xy(snapshot, j)
        for j in ("left_toe", "right_toe", "left_heel", "right_heel", "left_ankle", "right_ankle")
    ]
    foot_pts = [p for p in foot_pts if p]
    if foot_pts:
        points.append((foot_pts[0][0], min(p[1] for p in foot_pts) - body_h * 0.025))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    cx = (min_x + max_x) / 2.0
    cy = _vertical_body_center(snapshot, body_h)

    shoulder_span, hip_span = _shoulder_and_hip_spans(snapshot)
    span_x = max(
        max_x - min_x,
        shoulder_span * 1.06,
        hip_span * 1.08,
        body_h * _MIN_WIDTH_FRAC_OF_HEIGHT,
    )
    span_y = max(max_y - min_y, body_h * _VIEW_MIN_HEIGHT_FRAC)

    pad = body_h * _VIEW_PAD_FRAC
    return (cx, cy, span_x / 2.0 + pad, span_y / 2.0 + pad)


def compute_skeleton_view_box(
    recording,
    display_mode: str = DEFAULT_SKELETON_DISPLAY_MODE,
) -> tuple[float, float, float, float] | None:
    """Stable content view box (cx, cy, half_x, half_y) over the WHOLE recording.

    Scans every snapshot's display coordinates for ``display_mode`` and returns a
    single centered box that encloses every pose in the sequence plus a
    comfortable margin. Feeding one box to every frame keeps the scale constant
    during playback (no zoom "breathing") and guarantees the full body — head to
    both feet — is always visible regardless of the walk-cycle phase.
    """
    snaps = getattr(recording, "snapshots", None)
    if not snaps:
        return None

    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    core_min_y = float("inf")
    core_max_y = float("-inf")
    found = False

    for snap in snaps:
        coords = _build_display_coords(snap, display_mode)
        if not coords:
            continue
        for jid in _VIEW_SPAN_JOINTS:
            pt = coords.get(jid)
            if not pt:
                continue
            found = True
            x, y = float(pt[0]), float(pt[1])
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        for jid in _CORE_SPAN_JOINTS:
            pt = coords.get(jid)
            if pt:
                y = float(pt[1])
                core_min_y = min(core_min_y, y)
                core_max_y = max(core_max_y, y)

    if not found:
        return None

    body_h = max(core_max_y - core_min_y, max_y - min_y, 1e-3)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    span_x = max(max_x - min_x, body_h * _MIN_WIDTH_FRAC_OF_HEIGHT)
    span_y = max(max_y - min_y, body_h * _VIEW_MIN_HEIGHT_FRAC)
    pad = body_h * _GLOBAL_VIEW_PAD_FRAC
    return (cx, cy, span_x / 2.0 + pad, span_y / 2.0 + pad)


def _apply_view_limits(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    *,
    extra_points: list[tuple[float, float]] | None = None,
) -> None:
    """
    Fit and center the body in the panel with equal X/Y scaling (no stretch).

    Prefers a stable sequence-global content box (``ax._sw_fixed_view_box``) so
    the scale stays constant across frames and the whole body is always visible;
    falls back to a per-frame fit when no global box has been computed.
    """
    fixed = getattr(ax, "_sw_fixed_view_box", None)
    use_fixed = isinstance(fixed, (tuple, list)) and len(fixed) == 4
    if use_fixed:
        cx, cy, content_half_x, content_half_y = (float(v) for v in fixed)
        # A stable sequence-global box means the limits are frame-independent.
        # Cache them so the tiny equal-aspect box feedback loop cannot drift the
        # scale between frames; reuse verbatim while the box and panel size match.
        _layout_skeleton_figure(ax)
        panel_px = _drawable_pixels(ax)
        cache = getattr(ax, "_sw_view_cache", None)
        key = (round(cx, 6), round(cy, 6), round(content_half_x, 6),
               round(content_half_y, 6), panel_px)
        if isinstance(cache, tuple) and len(cache) == 3 and cache[0] == key:
            ax.set_xlim(*cache[1])
            ax.set_ylim(*cache[2])
            _layout_skeleton_figure(ax)
            return
    else:
        box = _single_frame_view_box(snapshot)
        if box is None:
            ax.set_xlim(-0.35, 0.35)
            ax.set_ylim(0.0, 1.0)
            _layout_skeleton_figure(ax)
            return
        cx, cy, content_half_x, content_half_y = box

    _layout_skeleton_figure(ax)
    panel_aspect = _drawable_aspect(ax)
    fill = _SKELETON_VIEW_FILL
    content_aspect = content_half_x / max(content_half_y, 1e-6)

    if panel_aspect >= content_aspect:
        half_y = content_half_y / fill
        half_x = half_y * panel_aspect
    else:
        half_x = content_half_x / fill
        half_y = half_x / panel_aspect

    half_x = max(half_x, content_half_x / fill)
    half_y = max(half_y, content_half_y / fill)
    if half_x / max(half_y, 1e-6) > panel_aspect:
        half_y = half_x / panel_aspect
    else:
        half_x = half_y * panel_aspect

    # In fixed mode we deliberately skip per-frame label expansion: it would move
    # the limits every frame (labels track the moving joint) and reintroduce zoom
    # drift. Callout labels use clip_on=False and the global box carries a
    # comfortable margin, so they remain readable without resizing the view.
    if extra_points and not use_fixed:
        # Approximate body height from the content box for label headroom.
        label_pad = 2.0 * content_half_y * _LABEL_VIEW_PAD_FRAC
        for tx, ty in extra_points:
            half_x = max(half_x, abs(tx - cx) + label_pad)
            half_y = max(half_y, abs(ty - cy) + label_pad)
        if half_x / max(half_y, 1e-6) > panel_aspect:
            half_y = half_x / panel_aspect
        else:
            half_x = half_y * panel_aspect

    xlim = (cx - half_x, cx + half_x)
    ylim = (cy - half_y, cy + half_y)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _layout_skeleton_figure(ax)
    if use_fixed:
        ax._sw_view_cache = (key, xlim, ylim)  # type: ignore[attr-defined]


def interpolate_snapshots(
    snap_a: SkeletonSnapshot,
    snap_b: SkeletonSnapshot,
    alpha: float,
) -> SkeletonSnapshot:
    alpha = max(0.0, min(1.0, alpha))
    if alpha <= 0.0:
        return snap_a
    if alpha >= 1.0:
        return snap_b

    from stablewalk.models.gait_motion import JointSample

    joint_ids = set(snap_a.joints) | set(snap_b.joints)
    joints: dict[str, JointSample] = {}
    for jid in joint_ids:
        sa = snap_a.joints.get(jid)
        sb = snap_b.joints.get(jid)
        if not sa and not sb:
            continue
        if not sa:
            joints[jid] = sb
            continue
        if not sb:
            joints[jid] = sa
            continue
        pa, pb = sa.position, sb.position
        pos = Vec3(
            x=pa.x + (pb.x - pa.x) * alpha,
            y=pa.y + (pb.y - pa.y) * alpha,
            z=pa.z + (pb.z - pa.z) * alpha,
        )
        angle = None
        if sa.angle_deg is not None and sb.angle_deg is not None:
            angle = sa.angle_deg + (sb.angle_deg - sa.angle_deg) * alpha
        velocity = None
        if sa.velocity is not None and sb.velocity is not None:
            velocity = sa.velocity + (sb.velocity - sa.velocity) * alpha
        elif sa.velocity is not None:
            velocity = sa.velocity
        elif sb.velocity is not None:
            velocity = sb.velocity
        velocity_vector = None
        if sa.velocity_vector is not None and sb.velocity_vector is not None:
            va, vb = sa.velocity_vector, sb.velocity_vector
            velocity_vector = Vec3(
                x=va.x + (vb.x - va.x) * alpha,
                y=va.y + (vb.y - va.y) * alpha,
                z=va.z + (vb.z - va.z) * alpha,
            )
        elif sa.velocity_vector is not None:
            velocity_vector = sa.velocity_vector
        elif sb.velocity_vector is not None:
            velocity_vector = sb.velocity_vector
        joints[jid] = JointSample(
            joint_id=jid,
            position=pos,
            parent_id=sa.parent_id,
            angle_deg=angle,
            velocity=velocity,
            velocity_vector=velocity_vector,
        )

    from stablewalk.models.gait_motion import DofSample

    dof_ids = set(snap_a.dofs) | set(snap_b.dofs)
    dofs: dict[str, DofSample] = {}
    for dof_id in dof_ids:
        da = snap_a.dofs.get(dof_id)
        db = snap_b.dofs.get(dof_id)
        if not da and not db:
            continue
        if not da:
            dofs[dof_id] = db
            continue
        if not db:
            dofs[dof_id] = da
            continue
        angle = None
        if da.angle_deg is not None and db.angle_deg is not None:
            angle = da.angle_deg + (db.angle_deg - da.angle_deg) * alpha
        velocity_deg_s = None
        if da.velocity_deg_s is not None and db.velocity_deg_s is not None:
            velocity_deg_s = da.velocity_deg_s + (db.velocity_deg_s - da.velocity_deg_s) * alpha
        elif da.velocity_deg_s is not None:
            velocity_deg_s = da.velocity_deg_s
        elif db.velocity_deg_s is not None:
            velocity_deg_s = db.velocity_deg_s
        dofs[dof_id] = DofSample(
            dof_id=dof_id,
            angle_deg=angle,
            velocity_deg_s=velocity_deg_s,
            joint_id=da.joint_id or db.joint_id,
        )

    meta = dict(snap_a.metadata)
    img_a = meta.get("image_xy")
    img_b = snap_b.metadata.get("image_xy")
    if isinstance(img_a, dict) or isinstance(img_b, dict):
        img_a = img_a if isinstance(img_a, dict) else {}
        img_b = img_b if isinstance(img_b, dict) else {}
        merged_xy: dict[str, list[float]] = {}
        for jid in set(img_a) | set(img_b):
            va = img_a.get(jid)
            vb = img_b.get(jid)
            if (
                isinstance(va, (list, tuple))
                and isinstance(vb, (list, tuple))
                and len(va) >= 2
                and len(vb) >= 2
            ):
                merged_xy[jid] = [
                    float(va[0]) + (float(vb[0]) - float(va[0])) * alpha,
                    float(va[1]) + (float(vb[1]) - float(va[1])) * alpha,
                ]
            elif isinstance(va, (list, tuple)) and len(va) >= 2:
                merged_xy[jid] = [float(va[0]), float(va[1])]
            elif isinstance(vb, (list, tuple)) and len(vb) >= 2:
                merged_xy[jid] = [float(vb[0]), float(vb[1])]
        meta["image_xy"] = merged_xy

    return SkeletonSnapshot(
        frame_index=snap_a.frame_index,
        time_s=snap_a.time_s + (snap_b.time_s - snap_a.time_s) * alpha,
        joints=joints,
        dofs=dofs,
        metadata=meta,
    )


def nearest_joint_from_event(
    event,
    snapshot: SkeletonSnapshot | None,
    *,
    display_mode: str = DEFAULT_SKELETON_DISPLAY_MODE,
    max_distance: float | None = None,
) -> str | None:
    """Return the pickable joint closest to a mouse event (for hover labels)."""
    if snapshot is None or event.inaxes is None:
        return None
    if event.xdata is None or event.ydata is None:
        return None

    display = getattr(snapshot, "_sw_display_xy", None)
    if not isinstance(display, dict):
        snapshot._sw_display_xy = _build_display_coords(snapshot, display_mode)  # type: ignore[attr-defined]
        display = snapshot._sw_display_xy  # type: ignore[attr-defined]

    height = _body_height(snapshot)
    limit = max_distance if max_distance is not None else height * 0.08

    best_id: str | None = None
    best_dist = limit
    for jid in PICKABLE_JOINTS:
        pt = display.get(jid) if isinstance(display, dict) else None
        if not pt:
            continue
        d = math.hypot(pt[0] - float(event.xdata), pt[1] - float(event.ydata))
        if d < best_dist:
            best_dist = d
            best_id = jid
    return best_id


def draw_gait_skeleton(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    *,
    clear: bool = True,
    title: str | None = None,
    show_labels: bool = False,
    show_legend: bool = False,
    paused: bool = False,
    highlight_joints: set[str] | None = None,
    labeled_joints: dict[str, str] | None = None,
    motion_arrows: dict[str, tuple[Vec3, Vec3]] | None = None,
    display_mode: str = DEFAULT_SKELETON_DISPLAY_MODE,
    ground_floor_y: float | None = None,
    foot_skeleton_labels: tuple | None = None,
    foot_contact: tuple[int, int] | None = None,
    show_opensim_markers: bool = False,
) -> None:
    """
    Render a stick-figure skeleton from a ``SkeletonSnapshot``.

    *display_mode*:
      - ``2d_pose`` — video-frame 2D landmarks (compare with overlay)
      - ``3d_normalized`` — oblique 3D normalized body
      - ``biomechanical`` — OpenSim-style segment model (default)
    """
    if clear:
        ax.cla()
        setup_skeleton_axes(ax, display_mode=display_mode)

    if not snapshot.joints:
        ax.text(
            0.5, 0.5, "No skeleton data",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=10,
        )
        return

    snapshot._sw_display_xy = _build_display_coords(snapshot, display_mode)  # type: ignore[attr-defined]
    height = _robust_body_height(snapshot)
    highlight = highlight_joints or set()
    labels = labeled_joints or {}

    _draw_ground(
        ax,
        snapshot,
        height,
        floor_y=ground_floor_y,
        display_mode=display_mode,
    )

    for parent, child, side in _all_edges(snapshot, display_mode):
        p0, p1 = _xy(snapshot, parent), _xy(snapshot, child)
        if not p0 or not p1:
            continue
        if not _plausible_segment(parent, child, p0, p1, height):
            continue
        _draw_bone(
            ax, p0, p1,
            color=_side_color(side),
            width=_bone_width(parent, child, height, display_mode=display_mode),
            zorder=3,
        )

    _draw_head(ax, snapshot, height)
    _draw_joint_markers(
        ax, snapshot, height, highlight_joints=highlight, display_mode=display_mode
    )

    if highlight:
        primary = next(iter(labels.keys()), None) if labels else None
        _draw_selection_highlights(
            ax,
            snapshot,
            highlight,
            height,
            display_mode=display_mode,
            primary_joint=primary,
        )

    if motion_arrows and paused:
        _draw_motion_arrows(ax, motion_arrows, height)

    _attach_joint_pickers(ax, snapshot, height)

    label_anchors: list[tuple[float, float]] = []
    if labels:
        label_anchors = _draw_inline_joint_labels(ax, snapshot, labels, height)

    if foot_skeleton_labels is not None:
        label_anchors.extend(
            _draw_foot_skeleton_labels(
                ax,
                snapshot,
                foot_skeleton_labels,
                height,
                avoid_anchors=label_anchors,
            )
        )
    elif foot_contact is not None:
        label_anchors.extend(
            _draw_foot_contact_labels(ax, snapshot, foot_contact, height)
        )

    if show_opensim_markers:
        label_anchors.extend(
            _draw_opensim_marker_debug(ax, snapshot, height, display_mode=display_mode)
        )

    ax._sw_label_anchors = label_anchors  # type: ignore[attr-defined]
    ax._sw_view_snapshot = snapshot  # type: ignore[attr-defined]
    if not labels:
        _draw_legend(ax)
    _layout_skeleton_figure(ax)
    _apply_view_limits(ax, snapshot, extra_points=label_anchors)

    if title is not None and title != "":
        mode_label = MODE_TO_SKELETON_LABEL.get(display_mode, "")
        if mode_label:
            title = f"{title}  ·  {mode_label}"
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="medium", pad=4)
