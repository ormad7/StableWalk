"""
Human manikin renderer for ``GaitMotionRecording`` / ``SkeletonSnapshot``.

OpenSim / Visual3D–style articulated body: layered torso, tapered limbs,
rounded joints, feet (heel→toe), and display-only hand stubs.

Tracked joints are projected without repositioning. Human proportions come
from render-only torso envelopes and tapered segment thicknesses; biomechanics
coordinates and animation samples are never modified. Motion follows the
video frame-by-frame with no artificial temporal smoothing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from stablewalk.models.gait_motion import SkeletonSnapshot, Vec3
from stablewalk.coordinates.coordinate_map import (
    OBLIQUE_DISPLAY_SHEAR,
    canonical_to_visualization_oblique,
)
from stablewalk.models.joint_registry import JOINT_DISPLAY_NAMES, ROOT_JOINT_ID
from stablewalk.ui.dof_selection import PICKABLE_JOINTS
from stablewalk.ui.colors import (
    ACCENT,
    ACCENT_ALT,
    BORDER,
    BOS_EDGE_REDUCED,
    BOS_EDGE_STABLE,
    BOS_EDGE_UNSTABLE,
    BOS_FILL_REDUCED,
    BOS_FILL_STABLE,
    BOS_FILL_UNSTABLE,
    COM_EDGE_REDUCED,
    COM_EDGE_STABLE,
    COM_EDGE_UNSTABLE,
    COM_FILL_REDUCED,
    COM_FILL_STABLE,
    COM_FILL_UNSTABLE,
    MUTED,
    PANEL,
    SIDE_LEFT,
    SIDE_RIGHT,
    STABILITY_REDUCED,
    STABILITY_STABLE,
    STABILITY_UNSTABLE,
    TEXT,
    WARNING,
)

# Side colour coding — canonical laboratory L/R convention (green / red).
_COLOR_LEFT = SIDE_LEFT
_COLOR_RIGHT = SIDE_RIGHT
_COLOR_CENTER = ACCENT_ALT
_COLOR_JOINT_FILL = "#e8d5c4"
_COLOR_JOINT_EDGE = "#c4a484"
_COLOR_SELECT_BONE = "#ffb020"
_COLOR_SELECT_JOINT = "#ffe566"
_COLOR_SELECT_RING = "#ff3d5a"
# When a joint is selected, non-selected joints/bones stay readable (not ghosted).
_DIMMED_JOINT_ALPHA = 0.42
_DIMMED_BONE_ALPHA = 0.55
# Soft de-emphasis for non-selected limbs (still readable — not ghost sticks).
_UNSELECTED_LIMB_ALPHA = 0.78
_COLOR_MOTION_ARROW = "#a78bfa"
_COLOR_GROUND = BORDER
# Contrast under-stroke so bones stay visible over BoS / ground fills.
_COLOR_BONE_OUTLINE = "#1a1520"

# OpenSim / Visual3D–inspired anatomical paint (high contrast on dark panels).
_SKIN = "#e8c4a0"
_SKIN_SHADOW = "#c9a078"
_SKIN_HIGHLIGHT = "#f6e2c8"
_SKIN_ARM = "#f0d4b8"
_CLOTH_SHIRT = "#9ec8e4"
_CLOTH_SHIRT_EDGE = "#6a9bb8"
_CLOTH_SHIRT_HIGHLIGHT = "#c5e0f2"
_CLOTH_PANTS = "#6b849c"
_CLOTH_PANTS_EDGE = "#8aa0b4"
_CLOTH_PANTS_SHADOW = "#556a7e"
_CLOTH_SHOES = "#3a4550"
_NECK_SKIN = "#deb892"
_HAIR = "#4a3828"
_MANIKIN_EDGE = "#5a4a38"


def _blend_hex(color: str, toward: str, amount: float) -> str:
    """Blend ``color`` toward ``toward`` by ``amount`` in [0, 1] (display paint)."""
    amount = max(0.0, min(1.0, float(amount)))

    def _rgb(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        if len(h) != 6:
            return (128, 128, 128)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    r0, g0, b0 = _rgb(color)
    r1, g1, b1 = _rgb(toward)
    r = int(round(r0 + (r1 - r0) * amount))
    g = int(round(g0 + (g1 - g0) * amount))
    b = int(round(b0 + (b1 - b0) * amount))
    return f"#{r:02x}{g:02x}{b:02x}"


# Base-of-support support-state labels (GUI overlay)
_SUPPORT_STATE_LABELS: dict[str, str] = {
    "left_stance": "Left Support",
    "right_stance": "Right Support",
    "double_support": "Double Support",
    "swing": "Swing",
}

# COM trail length (frames) for skeleton overlay
COM_TRAIL_FRAME_COUNT = 14

# MediaPipe-aligned topology (parent → child, side for colour)
# Trunk follows: pelvis → chest (spine) → shoulders / neck → head
# Plus hip→shoulder sides for a readable torso silhouette (not a single stick).
_GAIT_EDGES: tuple[tuple[str, str, str], ...] = (
    (ROOT_JOINT_ID, "left_hip", "left"),
    (ROOT_JOINT_ID, "right_hip", "right"),
    ("left_hip", "right_hip", "center"),
    (ROOT_JOINT_ID, "spine", "center"),
    ("spine", "neck", "center"),
    ("neck", "head", "center"),
    ("neck", "left_shoulder", "left"),
    ("neck", "right_shoulder", "right"),
    ("left_shoulder", "right_shoulder", "center"),
    ("spine", "left_shoulder", "left"),
    ("spine", "right_shoulder", "right"),
    # Torso outline — gives chest/waist volume like MediaPipe / OpenSim figures.
    ("left_hip", "left_shoulder", "left"),
    ("right_hip", "right_shoulder", "right"),
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
_VIEW_PAD_FRAC = 0.010
# Margin for the sequence-global view box: enough headroom that the body is
# never cropped at the extremes of the walk cycle.
_GLOBAL_VIEW_PAD_FRAC = 0.045
_MIN_WIDTH_FRAC_OF_HEIGHT = 0.18
_VIEW_MIN_HEIGHT_FRAC = 0.72
# Target body fill of the skeleton panel: large enough to read joint motion at
# dashboard scale while retaining head / foot clearance for the whole sequence.
_SKELETON_VIEW_FILL = 0.76
# A modest zoom is still clamped by ``_SKELETON_MAX_FILL`` so proportions and
# sequence-wide framing remain stable rather than pumping between frames.
SKELETON_SIZE_BOOST = 1.06
# Hard ceiling: body span may fill at most this fraction of the view.
_SKELETON_MAX_FILL = 0.78
# Invisible joint pick/hover hit targets (visible markers stay compact).
_PICK_HITBOX_SCALE = 2.9

# Headroom around the selected-point callout so its text box is never clipped by
# the panel edge (the label uses a padded bbox drawn with clip_on=False).
_LABEL_VIEW_PAD_FRAC = 0.04
# Selected / hovered marker scale vs normal joint radius.
_HOVER_JOINT_SCALE = 1.55
_SELECTED_JOINT_SCALE = 1.70
# Small perspective camera offset for the default walking view. Orthographic
# rotation preserves measured scale and joint positions. Matches the laboratory
# 3D camera azimuth (~35°) used by the true 3D skeleton viewer.
_WALKING_VIEW_AZIMUTH_DEG = 35.0

# Display-only anthropometric targets (fractions of standing body height).
# Soft-applied so MediaPipe pose direction is preserved.
_DISP_SHOULDER_WIDTH_FRAC = 0.205
_DISP_PELVIS_WIDTH_FRAC = 0.165
_DISP_UPPER_ARM_FRAC = 0.186
_DISP_FOREARM_FRAC = 0.146
_DISP_THIGH_FRAC = 0.245
_DISP_SHANK_FRAC = 0.246
_DISP_FOOT_FRAC = 0.130
# How strongly to pull segment lengths toward targets (0 = keep measured).
_DISP_LENGTH_BLEND = 0.55
# Cap how far shoulders/hips may be widened vs measured span.
_DISP_WIDTH_MAX_SCALE = 1.85
_DISP_WIDTH_MIN_SCALE = 0.95

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
    ("side", "Side Skeleton"),
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
    ("left_hip", "left_shoulder", "left"),
    ("right_hip", "right_shoulder", "right"),
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
    """Width / height of the skeleton panel widget (pre–equal-aspect shrink).

    Must NOT use ``ax.get_position()``: with ``set_aspect('equal', adjustable='box')``
    Matplotlib shrinks the axes box to a square, which feeds a circular aspect of
    1.0 and leaves large unused side bands that make the body look tiny.
    """
    w_px, h_px = _drawable_pixels(ax)
    if h_px < 1e-6:
        return 1.0
    return float(w_px) / float(h_px)


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


def _walking_camera_project(x: float, y: float, z: float) -> tuple[float, float]:
    """Orthographic X/Z camera rotation; never modifies the tracked 3D point."""
    angle = math.radians(_WALKING_VIEW_AZIMUTH_DEG)
    return (math.cos(angle) * x + math.sin(angle) * z, y)


def _build_display_xy(snapshot: SkeletonSnapshot) -> dict[str, tuple[float, float]]:
    """
    Build a mildly oblique walking view from measured 3D joints.

    The small camera shear separates near/far limbs without moving any tracked
    point or changing its proportions. Only missing trunk landmarks are derived.
    """
    measured: dict[str, tuple[float, float]] = {}
    for jid, sample in snapshot.joints.items():
        if jid in _LIMB_JOINTS or jid == "head":
            measured[jid] = _walking_camera_project(
                sample.position.x, sample.position.y, sample.position.z
            )

    # Also keep any explicitly stored trunk points
    for jid in (ROOT_JOINT_ID, "spine", "neck"):
        sample = snapshot.joints.get(jid)
        if sample:
            measured.setdefault(
                jid,
                _walking_camera_project(
                    sample.position.x, sample.position.y, sample.position.z
                ),
            )

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
_SPINE_FROM_PELVIS = 0.58
_NECK_FROM_SPINE = 0.48


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

    if shoulder_mid and pelvis and "spine" not in out:
        out["spine"] = (
            pelvis[0] + _SPINE_FROM_PELVIS * (shoulder_mid[0] - pelvis[0]),
            pelvis[1] + _SPINE_FROM_PELVIS * (shoulder_mid[1] - pelvis[1]),
        )
    spine = out.get("spine")
    if spine and head and "neck" not in out:
        out["neck"] = (
            spine[0] + _NECK_FROM_SPINE * (head[0] - spine[0]),
            spine[1] + _NECK_FROM_SPINE * (head[1] - spine[1]),
        )
    elif spine and shoulder_mid and "neck" not in out:
        out["neck"] = (
            spine[0] + 0.62 * (shoulder_mid[0] - spine[0]),
            spine[1] + 0.62 * (shoulder_mid[1] - spine[1]),
        )


def _display_body_height(out: dict[str, tuple[float, float]]) -> float:
    """
    Robust standing height for display proportion targets.

    Prefer head→ankle span. When vertical span is inflated relative to the
    measured shoulder girdle (common with poorly scaled pose exports), fall
    back to shoulder-implied stature so pelvis/shoulder targets stay realistic.
    """
    head = out.get("head")
    ankle_ys = [
        out[j][1]
        for j in ("left_ankle", "right_ankle", "left_heel", "right_heel")
        if j in out
    ]
    core = 0.0
    if head and ankle_ys:
        core = max(head[1] - min(ankle_ys), 0.0)

    ls = out.get("left_shoulder")
    rs = out.get("right_shoulder")
    shoulder_span = (
        math.hypot(rs[0] - ls[0], rs[1] - ls[1]) if ls and rs else 0.0
    )
    shoulder_stature = (
        shoulder_span / _DISP_SHOULDER_WIDTH_FRAC if shoulder_span > 1e-6 else 0.0
    )

    if core >= 0.35 and shoulder_stature >= 0.35:
        # Inflated Y range vs a plausible shoulder width → trust the girdle.
        if core > shoulder_stature * 1.25:
            return shoulder_stature
        return core
    if core >= 0.35:
        return core
    if shoulder_stature >= 0.35:
        return shoulder_stature

    ys: list[float] = []
    for jid in (
        "head",
        "neck",
        "left_shoulder",
        "right_shoulder",
        ROOT_JOINT_ID,
        "left_hip",
        "right_hip",
        "left_ankle",
        "right_ankle",
    ):
        pt = out.get(jid)
        if pt:
            ys.append(pt[1])
    if len(ys) < 2:
        return 1.0
    return max(max(ys) - min(ys), 0.35)


def _widen_bilateral_pair(
    left: tuple[float, float],
    right: tuple[float, float],
    target_span: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Widen a L/R landmark pair about its midpoint to ``target_span``.

    Returns (new_left, new_right, delta_left, delta_right). Only expands (or
    mildly shrinks) within ``_DISP_WIDTH_*_SCALE`` so pose direction is kept.
    """
    mx = 0.5 * (left[0] + right[0])
    my = 0.5 * (left[1] + right[1])
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    current = math.hypot(dx, dy)
    if current < 1e-9:
        half = target_span * 0.5
        new_l = (mx - half, my)
        new_r = (mx + half, my)
    else:
        scale = target_span / current
        scale = max(_DISP_WIDTH_MIN_SCALE, min(_DISP_WIDTH_MAX_SCALE, scale))
        # Prefer widening a collapsed MediaPipe girdle; avoid shrinking a good pose.
        if scale < 1.0:
            scale = max(scale, 0.97)
        new_l = (mx - 0.5 * dx * scale, my - 0.5 * dy * scale)
        new_r = (mx + 0.5 * dx * scale, my + 0.5 * dy * scale)
    dl = (new_l[0] - left[0], new_l[1] - left[1])
    dr = (new_r[0] - right[0], new_r[1] - right[1])
    return new_l, new_r, dl, dr


def _offset_point(
    pt: tuple[float, float] | None, delta: tuple[float, float]
) -> tuple[float, float] | None:
    if pt is None:
        return None
    return (pt[0] + delta[0], pt[1] + delta[1])


def _soft_set_child_length(
    parent: tuple[float, float],
    child: tuple[float, float],
    target_len: float,
    *,
    blend: float = _DISP_LENGTH_BLEND,
) -> tuple[float, float]:
    """Move ``child`` along the parent→child ray toward ``target_len`` (soft blend)."""
    dx = child[0] - parent[0]
    dy = child[1] - parent[1]
    current = math.hypot(dx, dy)
    if current < 1e-9:
        return child
    # Keep correction gentle so extreme poses stay readable.
    desired = max(current * 0.72, min(current * 1.38, target_len))
    scale = 1.0 + blend * (desired / current - 1.0)
    return (parent[0] + dx * scale, parent[1] + dy * scale)


def _apply_display_human_proportions(out: dict[str, tuple[float, float]]) -> None:
    """
    Display-only: realistic shoulder/pelvis width and soft limb proportions.

    1. Normalize stature so head→ankle matches a robust height estimate (fixes
       pose exports where Y is inflated vs X — otherwise the figure looks like
       a tall stick even with correct width fractions).
    2. Widen shoulder / pelvis girdles to anthropometric targets.
    3. Soft-correct limb segment lengths along existing bone directions.
    """
    stature = _display_body_height(out)
    _rescale_display_to_stature(out, stature)

    ls = out.get("left_shoulder")
    rs = out.get("right_shoulder")
    if ls and rs:
        new_ls, new_rs, dl, dr = _widen_bilateral_pair(
            ls, rs, stature * _DISP_SHOULDER_WIDTH_FRAC
        )
        out["left_shoulder"] = new_ls
        out["right_shoulder"] = new_rs
        for jid, delta in (
            ("left_elbow", dl),
            ("left_wrist", dl),
            ("right_elbow", dr),
            ("right_wrist", dr),
        ):
            moved = _offset_point(out.get(jid), delta)
            if moved:
                out[jid] = moved

    lh = out.get("left_hip")
    rh = out.get("right_hip")
    if lh and rh:
        new_lh, new_rh, dl, dr = _widen_bilateral_pair(
            lh, rh, stature * _DISP_PELVIS_WIDTH_FRAC
        )
        out["left_hip"] = new_lh
        out["right_hip"] = new_rh
        for jid, delta in (
            ("left_knee", dl),
            ("left_ankle", dl),
            ("left_heel", dl),
            ("left_toe", dl),
            ("right_knee", dr),
            ("right_ankle", dr),
            ("right_heel", dr),
            ("right_toe", dr),
        ):
            moved = _offset_point(out.get(jid), delta)
            if moved:
                out[jid] = moved
        out[ROOT_JOINT_ID] = (
            0.5 * (new_lh[0] + new_rh[0]),
            0.5 * (new_lh[1] + new_rh[1]),
        )

    for side in ("left", "right"):
        sh = out.get(f"{side}_shoulder")
        el = out.get(f"{side}_elbow")
        wr = out.get(f"{side}_wrist")
        if sh and el:
            out[f"{side}_elbow"] = _soft_set_child_length(
                sh, el, stature * _DISP_UPPER_ARM_FRAC
            )
            el = out[f"{side}_elbow"]
        if el and wr:
            out[f"{side}_wrist"] = _soft_set_child_length(
                el, wr, stature * _DISP_FOREARM_FRAC
            )

        hp = out.get(f"{side}_hip")
        kn = out.get(f"{side}_knee")
        an = out.get(f"{side}_ankle")
        if hp and kn:
            out[f"{side}_knee"] = _soft_set_child_length(
                hp, kn, stature * _DISP_THIGH_FRAC
            )
            kn = out[f"{side}_knee"]
        if kn and an:
            out[f"{side}_ankle"] = _soft_set_child_length(
                kn, an, stature * _DISP_SHANK_FRAC
            )
            an = out[f"{side}_ankle"]
        toe = out.get(f"{side}_toe")
        if an and toe:
            out[f"{side}_toe"] = _soft_set_child_length(
                an, toe, stature * _DISP_FOOT_FRAC
            )
        heel = out.get(f"{side}_heel")
        if an and heel:
            out[f"{side}_heel"] = _soft_set_child_length(
                an, heel, stature * (_DISP_FOOT_FRAC * 0.55)
            )

    _refine_trunk(out)


def _rescale_display_to_stature(
    out: dict[str, tuple[float, float]], stature: float
) -> None:
    """Uniformly scale display points so head→ankle equals ``stature``."""
    if stature < 0.2:
        return
    head = out.get("head")
    foot_pts = [
        out[j]
        for j in (
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_toe",
            "right_toe",
        )
        if j in out
    ]
    if not head or not foot_pts:
        return
    y_foot = min(p[1] for p in foot_pts)
    current = head[1] - y_foot
    if current < 1e-6:
        return
    scale = stature / current
    if abs(scale - 1.0) < 0.02:
        return
    # Anchor at the foot line and body mid-X so the figure stays centered.
    pelvis = out.get(ROOT_JOINT_ID)
    if pelvis:
        cx = pelvis[0]
    else:
        xs = [p[0] for p in out.values()]
        cx = 0.5 * (min(xs) + max(xs)) if xs else head[0]
    cy = y_foot
    for jid, (x, y) in list(out.items()):
        out[jid] = (cx + (x - cx) * scale, cy + (y - cy) * scale)


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


def _build_display_side(snapshot: SkeletonSnapshot) -> dict[str, tuple[float, float]]:
    """True sagittal projection (forward Z × vertical Y) of measured joints."""
    out = {
        jid: (sample.position.z, sample.position.y)
        for jid, sample in snapshot.joints.items()
    }
    _refine_trunk(out)
    return out


def _build_display_coords(
    snapshot: SkeletonSnapshot, display_mode: str
) -> dict[str, tuple[float, float]]:
    if display_mode == "2d_pose":
        return _build_display_2d_pose(snapshot)
    if display_mode == "3d_normalized":
        return _build_display_3d_normalized(snapshot)
    if display_mode == "side":
        return _build_display_side(snapshot)
    # Walking / biomechanical views use a small real-depth camera offset.
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
    # Hip→shoulder torso rails are naturally long (~0.3–0.4 body height).
    if ("hip" in parent and "shoulder" in child) or (
        "shoulder" in parent and "hip" in child
    ):
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


def _is_torso_structure_edge(parent: str, child: str) -> bool:
    """Edges covered by the filled torso — skip as stick overlays."""
    pair = frozenset((parent, child))
    torso_pairs = {
        frozenset((ROOT_JOINT_ID, "spine")),
        frozenset(("spine", "neck")),
        frozenset(("neck", "head")),
        frozenset(("neck", "left_shoulder")),
        frozenset(("neck", "right_shoulder")),
        frozenset(("spine", "left_shoulder")),
        frozenset(("spine", "right_shoulder")),
        frozenset(("left_shoulder", "right_shoulder")),
        frozenset(("left_hip", "right_hip")),
        frozenset((ROOT_JOINT_ID, "left_hip")),
        frozenset((ROOT_JOINT_ID, "right_hip")),
        frozenset(("left_hip", "left_shoulder")),
        frozenset(("right_hip", "right_shoulder")),
    }
    return pair in torso_pairs


def _limb_paint_color(parent: str, child: str, side: str) -> str:
    """Anatomical limb fill with subtle L/R biomechanics tint on the rim only."""
    pair = f"{parent}_{child}".lower()
    if "hip" in pair and "knee" in pair:
        return _CLOTH_PANTS
    if "knee" in pair and "ankle" in pair:
        return _CLOTH_PANTS
    if any(t in pair for t in ("heel", "toe", "foot")):
        return _CLOTH_SHOES
    if "shoulder" in pair and "elbow" in pair:
        return _SKIN_ARM
    if "elbow" in pair and "wrist" in pair:
        return _SKIN
    return _side_color(side)


def _bone_width(
    parent: str,
    child: str,
    height: float,
    *,
    display_mode: str = DEFAULT_SKELETON_DISPLAY_MODE,
) -> float:
    pair = f"{parent}_{child}"
    # Laboratory stick-figure weights (OpenSim / Vicon-like; not game-thick).
    scale = 1.08
    if ROOT_JOINT_ID in pair or "spine" in pair or "neck" in pair or "head" in pair:
        w = max(height * 0.030, 2.8)
    elif "shoulder" in pair and "elbow" in pair:
        w = max(height * 0.028, 2.6)
    elif "elbow" in pair and "wrist" in pair:
        w = max(height * 0.024, 2.3)
    elif "hip" in pair and "knee" in pair:
        w = max(height * 0.030, 2.8)
    elif "knee" in pair and "ankle" in pair:
        w = max(height * 0.026, 2.5)
    elif any(x in pair for x in ("heel", "toe", "foot", "ankle")):
        w = max(height * 0.022, 2.1)
    elif "shoulder" in parent and "shoulder" in child:
        # Cross-shoulder bar — chest width cue without a solid slab.
        w = max(height * 0.028, 2.6)
    elif "hip" in parent and "hip" in child:
        # Visible pelvis width (not a single vertical line).
        w = max(height * 0.030, 2.8)
    elif ("hip" in parent and "shoulder" in child) or (
        "shoulder" in parent and "hip" in child
    ):
        # Torso side rails (hip→shoulder) — keep thin so the trunk is not a box.
        w = max(height * 0.022, 2.1)
    elif "spine" in pair and "shoulder" in pair:
        w = max(height * 0.024, 2.3)
    elif "hip" in pair and ROOT_JOINT_ID in pair:
        w = max(height * 0.026, 2.4)
    else:
        w = max(height * 0.024, 2.3)

    w *= scale

    if display_mode == "biomechanical":
        if ROOT_JOINT_ID in pair or "spine" in pair or "neck" in pair:
            return w * 1.12
        if "hip" in pair and "knee" in pair:
            return w * 1.08
        if "knee" in pair and "ankle" in pair:
            return w * 1.05
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
        edges = list(_BIOMECH_EDGES)
        edges.extend(_foot_edges(snapshot, "left"))
        edges.extend(_foot_edges(snapshot, "right"))
        return edges
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
    # Dark under-stroke + round caps → soft anti-aliased limb edges.
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color=_COLOR_BONE_OUTLINE,
        linewidth=width + 0.55,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=max(zorder - 1, 0),
        alpha=min(1.0, alpha + 0.06),
        antialiased=True,
    )
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color=color,
        linewidth=width,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder,
        alpha=alpha,
        antialiased=True,
    )


def _anatomical_segment_profile(
    parent: str,
    child: str,
    height: float,
) -> tuple[float, float, float] | None:
    """Data-space half-widths (proximal, mid-bulge, distal) for limb segments.

    Fractions of standing height follow Visual3D / OpenSim manikin
    anthropometry. Display-only — joint coordinates are never moved.
    """
    pair = frozenset((parent, child))
    # (proximal, mid muscle bulge, distal) — reads as thigh/arm, not cylinder.
    profiles: dict[frozenset[str], tuple[float, float, float]] = {
        # Legs — thicker thigh / calf so the manikin reads less like sticks
        frozenset(("left_hip", "left_knee")): (0.108, 0.122, 0.078),
        frozenset(("right_hip", "right_knee")): (0.108, 0.122, 0.078),
        frozenset(("left_knee", "left_ankle")): (0.072, 0.068, 0.046),
        frozenset(("right_knee", "right_ankle")): (0.072, 0.068, 0.046),
        # Arms
        frozenset(("left_shoulder", "left_elbow")): (0.070, 0.078, 0.054),
        frozenset(("right_shoulder", "right_elbow")): (0.070, 0.078, 0.054),
        frozenset(("left_elbow", "left_wrist")): (0.050, 0.048, 0.038),
        frozenset(("right_elbow", "right_wrist")): (0.050, 0.048, 0.038),
        # Feet
        frozenset(("left_ankle", "left_heel")): (0.036, 0.038, 0.034),
        frozenset(("right_ankle", "right_heel")): (0.036, 0.038, 0.034),
        frozenset(("left_heel", "left_toe")): (0.036, 0.044, 0.048),
        frozenset(("right_heel", "right_toe")): (0.036, 0.044, 0.048),
        frozenset(("left_ankle", "left_toe")): (0.034, 0.042, 0.046),
        frozenset(("right_ankle", "right_toe")): (0.034, 0.042, 0.046),
        # Neck / clavicles / torso rails
        frozenset(("neck", "head")): (0.038, 0.042, 0.044),
        frozenset(("spine", "neck")): (0.048, 0.046, 0.036),
        frozenset((ROOT_JOINT_ID, "spine")): (0.055, 0.050, 0.045),
        frozenset(("neck", "left_shoulder")): (0.030, 0.038, 0.044),
        frozenset(("neck", "right_shoulder")): (0.030, 0.038, 0.044),
        frozenset(("spine", "left_shoulder")): (0.034, 0.042, 0.046),
        frozenset(("spine", "right_shoulder")): (0.034, 0.042, 0.046),
        frozenset(("left_hip", "left_shoulder")): (0.042, 0.048, 0.048),
        frozenset(("right_hip", "right_shoulder")): (0.042, 0.048, 0.048),
        frozenset(("left_shoulder", "right_shoulder")): (0.032, 0.034, 0.032),
        frozenset(("left_hip", "right_hip")): (0.048, 0.052, 0.048),
        frozenset((ROOT_JOINT_ID, "left_hip")): (0.044, 0.050, 0.048),
        frozenset((ROOT_JOINT_ID, "right_hip")): (0.044, 0.050, 0.048),
    }
    profile = profiles.get(pair)
    if profile is None:
        return None
    start, mid, end = profile
    # Profiles are authored proximal→distal. Reverse when an edge is supplied
    # in the opposite direction.
    proximal_names = ("shoulder", "elbow", "hip", "knee", "ankle", "heel", "spine", ROOT_JOINT_ID, "neck")
    parent_is_proximal = any(name in parent for name in proximal_names)
    if not parent_is_proximal:
        start, end = end, start
    return height * start, height * mid, height * end


def _limb_stroke_points(parent: str, child: str) -> float:
    """Screen-space stroke width so limbs stay visible at any data scale."""
    pair = f"{parent}_{child}".lower()
    if "hip" in pair and "knee" in pair:
        return 16.0
    if "knee" in pair and "ankle" in pair:
        return 13.5
    if "shoulder" in pair and "elbow" in pair:
        return 12.5
    if "elbow" in pair and "wrist" in pair:
        return 10.5
    if any(t in pair for t in ("heel", "toe", "foot", "ankle")):
        return 9.5
    return 11.0


def _draw_limb_segment(
    ax: Axes,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    parent: str,
    child: str,
    color: str,
    height: float,
    zorder: float,
    alpha: float,
) -> None:
    """Human limb: thick rounded stroke (always visible) + soft volume fill."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return
    fill_a = min(1.0, max(0.88, alpha))
    lw = _limb_stroke_points(parent, child)
    # Guaranteed silhouette in screen points (not data units).
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color=_MANIKIN_EDGE,
        linewidth=lw + 2.4,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder,
        alpha=fill_a,
        antialiased=True,
    )
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color=color,
        linewidth=lw,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder + 0.02,
        alpha=fill_a,
        antialiased=True,
    )
    profile = _anatomical_segment_profile(parent, child, height)
    if profile is None:
        return
    # Extra soft volume when the segment is long enough in the view.
    if length >= height * 0.045:
        _draw_tapered_capsule(
            ax,
            p0,
            p1,
            start_half_width=profile[0],
            end_half_width=profile[2],
            mid_half_width=profile[1],
            color=color,
            zorder=zorder + 0.05,
            alpha=fill_a * 0.92,
        )


def _draw_tapered_capsule(
    ax: Axes,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    start_half_width: float,
    end_half_width: float,
    color: str,
    zorder: float,
    alpha: float,
    mid_half_width: float | None = None,
) -> None:
    """Solid opaque limb bar with round caps — readable on dark lab panels."""
    from matplotlib.patches import Circle, Polygon

    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    # Shrink thickness when foreshortened so limbs do not become plate stubs.
    fores = min(1.0, length / max(start_half_width * 7.0, end_half_width * 7.0, 1e-6))
    fores = max(0.40, fores)
    w0 = min(max(start_half_width * fores, length * 0.07), length * 0.40)
    w1 = min(max(end_half_width * fores, length * 0.06), length * 0.36)
    wm_raw = mid_half_width if mid_half_width is not None else max(w0, w1) * 1.10
    wm = min(max(wm_raw * fores, length * 0.08), length * 0.44)
    mx, my = p0[0] + ux * length * 0.48, p0[1] + uy * length * 0.48
    # Opaque silhouette (no fragile Bezier paths — always fills solidly).
    verts = [
        (p0[0] + nx * w0, p0[1] + ny * w0),
        (mx + nx * wm, my + ny * wm),
        (p1[0] + nx * w1, p1[1] + ny * w1),
        (p1[0] - nx * w1, p1[1] - ny * w1),
        (mx - nx * wm, my - ny * wm),
        (p0[0] - nx * w0, p0[1] - ny * w0),
    ]
    fill_a = min(1.0, max(0.92, alpha))
    ax.add_patch(
        Polygon(
            verts,
            closed=True,
            facecolor=color,
            edgecolor=_MANIKIN_EDGE,
            linewidth=0.55,
            alpha=fill_a,
            joinstyle="round",
            zorder=zorder,
            antialiased=True,
        )
    )
    # Soft volume: thin highlight strip on the light side of the limb.
    hi_w0, hi_w1, hi_wm = w0 * 0.42, w1 * 0.42, wm * 0.42
    hi_verts = [
        (p0[0] + nx * hi_w0 * 0.15, p0[1] + ny * hi_w0 * 0.15),
        (mx + nx * hi_wm * 0.20, my + ny * hi_wm * 0.20),
        (p1[0] + nx * hi_w1 * 0.15, p1[1] + ny * hi_w1 * 0.15),
        (p1[0] + nx * hi_w1, p1[1] + ny * hi_w1),
        (mx + nx * hi_wm, my + ny * hi_wm),
        (p0[0] + nx * hi_w0, p0[1] + ny * hi_w0),
    ]
    ax.add_patch(
        Polygon(
            hi_verts,
            closed=True,
            facecolor=_blend_hex(color, "#ffffff", 0.28),
            edgecolor="none",
            alpha=fill_a * 0.38,
            joinstyle="round",
            zorder=zorder + 0.015,
            antialiased=True,
        )
    )
    # Small soft end rounding only (not full joint balls — those looked like a
    # wooden mannequin and hid the real limb silhouette).
    tip = max(min(w0, w1) * 0.55, length * 0.012)
    ax.add_patch(
        Circle(
            p0,
            tip,
            facecolor=color,
            edgecolor="none",
            alpha=fill_a,
            zorder=zorder + 0.01,
            antialiased=True,
        )
    )
    ax.add_patch(
        Circle(
            p1,
            tip,
            facecolor=color,
            edgecolor="none",
            alpha=fill_a,
            zorder=zorder + 0.01,
            antialiased=True,
        )
    )


def _joint_camera_depth(
    snapshot: SkeletonSnapshot,
    joint_id: str,
    *,
    display_mode: str,
) -> float:
    """Measured camera-depth coordinate used only for painter's ordering."""
    sample = snapshot.joints.get(joint_id)
    if sample is not None:
        return (
            float(sample.position.x)
            if display_mode == "side"
            else float(sample.position.z)
        )
    if joint_id in (ROOT_JOINT_ID, "spine", "neck"):
        values = [
            _joint_camera_depth(snapshot, jid, display_mode=display_mode)
            for jid in (
                "left_hip",
                "right_hip",
                "left_shoulder",
                "right_shoulder",
            )
            if jid in snapshot.joints
        ]
        if values:
            return sum(values) / len(values)
    return 0.0


def _depth_alpha(
    depth: float,
    depth_limits: tuple[float, float],
) -> float:
    """Subtle far→near shading without changing left/right identity."""
    lo, hi = depth_limits
    if hi - lo < 1e-9:
        return 0.96
    t = max(0.0, min(1.0, (depth - lo) / (hi - lo)))
    return 0.76 + 0.22 * t


def _draw_articulated_body_structure(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    height: float,
) -> None:
    """Layered ribcage / waist / pelvis envelopes from tracked landmarks only.

    Display-only soft-body fill (Visual3D / OpenSim manikin style). Joint
    positions are never moved — envelopes widen around measured shoulders/hips.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    ls = _xy(snapshot, "left_shoulder")
    rs = _xy(snapshot, "right_shoulder")
    lh = _xy(snapshot, "left_hip")
    rh = _xy(snapshot, "right_hip")
    neck = _xy(snapshot, "neck")
    spine = _xy(snapshot, "spine")

    def _body_pair(
        left: tuple[float, float],
        right: tuple[float, float],
        min_half_width: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        cx = 0.5 * (left[0] + right[0])
        cy = 0.5 * (left[1] + right[1])
        dx, dy = right[0] - left[0], right[1] - left[1]
        span = math.hypot(dx, dy)
        if span > 1e-9:
            ux, uy = dx / span, dy / span
        else:
            ux, uy = 1.0, 0.0
        half = max(span * 0.5, min_half_width)
        return (cx - ux * half, cy - uy * half), (cx + ux * half, cy + uy * half)

    def _lerp(
        a: tuple[float, float], b: tuple[float, float], t: float
    ) -> tuple[float, float]:
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    def _add_torso_band(
        left_a: tuple[float, float],
        right_a: tuple[float, float],
        left_b: tuple[float, float],
        right_b: tuple[float, float],
        *,
        face: str,
        edge: str,
        alpha: float,
        zorder: float,
        linewidth: float = 0.9,
    ) -> None:
        # Soft curved band (Bezier sides) — less “blocky rectangle”.
        mid_top = (
            0.5 * (left_a[0] + right_a[0]),
            0.5 * (left_a[1] + right_a[1]),
        )
        mid_bot = (
            0.5 * (left_b[0] + right_b[0]),
            0.5 * (left_b[1] + right_b[1]),
        )
        # Outward bulge for a cylindrical torso silhouette.
        bulge = height * 0.012
        dx = right_a[0] - left_a[0]
        dy = right_a[1] - left_a[1]
        span = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / span, dx / span
        c_left = (
            0.5 * (left_a[0] + left_b[0]) - nx * bulge,
            0.5 * (left_a[1] + left_b[1]) - ny * bulge,
        )
        c_right = (
            0.5 * (right_a[0] + right_b[0]) + nx * bulge,
            0.5 * (right_a[1] + right_b[1]) + ny * bulge,
        )
        verts = [
            left_a,
            mid_top,
            right_a,
            c_right,
            right_b,
            mid_bot,
            left_b,
            c_left,
            left_a,
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
        ]
        ax.add_patch(
            PathPatch(
                Path(verts, codes),
                facecolor=face,
                edgecolor=edge,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
                antialiased=True,
                joinstyle="round",
            )
        )

    if ls and rs and lh and rh:
        from matplotlib.patches import Ellipse, Polygon as MplPolygon

        # Solid vest torso anchored to measured shoulders/hips (opaque).
        # Prefer measured girdle span so frontal-narrow tracks do not balloon.
        sh_span = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
        hp_span = math.hypot(rh[0] - lh[0], rh[1] - lh[1])
        chest_l, chest_r = _body_pair(ls, rs, max(sh_span * 0.55, height * 0.10))
        pelvis_l, pelvis_r = _body_pair(lh, rh, max(hp_span * 0.55, height * 0.08))
        shoulder_c = (
            0.5 * (chest_l[0] + chest_r[0]),
            0.5 * (chest_l[1] + chest_r[1]),
        )
        pelvis_c = (
            0.5 * (pelvis_l[0] + pelvis_r[0]),
            0.5 * (pelvis_l[1] + pelvis_r[1]),
        )
        waist_c = _lerp(shoulder_c, pelvis_c, 0.62)
        axis_dx = chest_r[0] - chest_l[0]
        axis_dy = chest_r[1] - chest_l[1]
        axis_len = max(math.hypot(axis_dx, axis_dy), 1e-9)
        ax_u = (axis_dx / axis_len, axis_dy / axis_len)
        waist_half = max(axis_len * 0.38, height * 0.07)
        chest_half = max(axis_len * 0.52, height * 0.10)
        hip_half = max(axis_len * 0.45, height * 0.085)
        # Rebuild vest from measured centers with anthropometric widths.
        chest_l = (shoulder_c[0] - ax_u[0] * chest_half, shoulder_c[1] - ax_u[1] * chest_half)
        chest_r = (shoulder_c[0] + ax_u[0] * chest_half, shoulder_c[1] + ax_u[1] * chest_half)
        waist_l = (waist_c[0] - ax_u[0] * waist_half, waist_c[1] - ax_u[1] * waist_half)
        waist_r = (waist_c[0] + ax_u[0] * waist_half, waist_c[1] + ax_u[1] * waist_half)
        pelvis_l = (pelvis_c[0] - ax_u[0] * hip_half, pelvis_c[1] - ax_u[1] * hip_half)
        pelvis_r = (pelvis_c[0] + ax_u[0] * hip_half, pelvis_c[1] + ax_u[1] * hip_half)
        # Soft side bulge for a cylindrical torso (not a flat rectangle).
        bulge = height * 0.018
        nx, ny = -ax_u[1], ax_u[0]
        mid_shirt_l = (
            0.5 * (chest_l[0] + waist_l[0]) - nx * bulge,
            0.5 * (chest_l[1] + waist_l[1]) - ny * bulge,
        )
        mid_shirt_r = (
            0.5 * (chest_r[0] + waist_r[0]) + nx * bulge,
            0.5 * (chest_r[1] + waist_r[1]) + ny * bulge,
        )

        ax.add_patch(
            MplPolygon(
                [chest_l, chest_r, mid_shirt_r, waist_r, waist_l, mid_shirt_l],
                closed=True,
                facecolor=_CLOTH_SHIRT,
                edgecolor=_MANIKIN_EDGE,
                linewidth=0.65,
                alpha=1.0,
                joinstyle="round",
                zorder=2.6,
                antialiased=True,
            )
        )
        # Soft chest highlight — reads as volume, not a flat plate.
        hi_l = _lerp(chest_l, waist_l, 0.18)
        hi_r = _lerp(chest_r, waist_r, 0.18)
        hi_lb = _lerp(chest_l, waist_l, 0.48)
        hi_rb = _lerp(chest_r, waist_r, 0.48)
        mid_hi_l = (
            0.5 * (hi_l[0] + hi_lb[0]) - nx * bulge * 0.45,
            0.5 * (hi_l[1] + hi_lb[1]) - ny * bulge * 0.45,
        )
        mid_hi_r = (
            0.5 * (hi_r[0] + hi_rb[0]) + nx * bulge * 0.20,
            0.5 * (hi_r[1] + hi_rb[1]) + ny * bulge * 0.20,
        )
        ax.add_patch(
            MplPolygon(
                [hi_l, hi_r, mid_hi_r, hi_rb, hi_lb, mid_hi_l],
                closed=True,
                facecolor=_CLOTH_SHIRT_HIGHLIGHT,
                edgecolor="none",
                alpha=0.28,
                joinstyle="round",
                zorder=2.62,
                antialiased=True,
            )
        )
        ax.add_patch(
            MplPolygon(
                [waist_l, waist_r, pelvis_r, pelvis_l],
                closed=True,
                facecolor=_CLOTH_PANTS,
                edgecolor=_MANIKIN_EDGE,
                linewidth=0.65,
                alpha=1.0,
                joinstyle="round",
                zorder=2.7,
                antialiased=True,
            )
        )
        # Soft waist shadow so shirt/pants read as separate layers.
        shade_t = _lerp(waist_l, waist_r, 0.0)
        shade_b = _lerp(pelvis_l, pelvis_r, 0.0)
        ax.add_patch(
            MplPolygon(
                [
                    waist_l,
                    waist_r,
                    _lerp(waist_r, pelvis_r, 0.35),
                    _lerp(waist_l, pelvis_l, 0.35),
                ],
                closed=True,
                facecolor=_CLOTH_PANTS_SHADOW,
                edgecolor="none",
                alpha=0.22,
                joinstyle="round",
                zorder=2.72,
                antialiased=True,
            )
        )
        del shade_t, shade_b
        ax.plot(
            [shoulder_c[0], waist_c[0]],
            [shoulder_c[1], waist_c[1]],
            color="#ffffff",
            linewidth=2.2,
            alpha=0.20,
            solid_capstyle="round",
            zorder=2.85,
            antialiased=True,
        )
        deltoid_w = min(height * 0.072, max(sh_span * 0.28, height * 0.035))
        deltoid_h = deltoid_w * 0.75
        for sh in (ls, rs):
            ax.add_patch(
                Ellipse(
                    sh,
                    width=deltoid_w,
                    height=deltoid_h,
                    facecolor=_SKIN_ARM,
                    edgecolor=_MANIKIN_EDGE,
                    linewidth=0.40,
                    alpha=1.0,
                    zorder=3.05,
                    antialiased=True,
                )
            )
        if neck is not None:
            collar = _lerp(shoulder_c, neck, 0.45)
            _draw_tapered_capsule(
                ax,
                collar,
                neck,
                start_half_width=height * 0.045,
                end_half_width=height * 0.038,
                mid_half_width=height * 0.042,
                color=_NECK_SKIN,
                zorder=3.0,
                alpha=1.0,
            )

    if lh and rh:
        from matplotlib.patches import Ellipse

        pelvis_l, pelvis_r = _body_pair(lh, rh, height * 0.15)
        dx, dy = pelvis_r[0] - pelvis_l[0], pelvis_r[1] - pelvis_l[1]
        span = math.hypot(dx, dy)
        if span > height * 0.012:
            # Rounded pelvic bowl (ellipse) instead of a grey box.
            mid = (0.5 * (lh[0] + rh[0]), 0.5 * (lh[1] + rh[1]))
            angle = math.degrees(math.atan2(dy, dx))
            pw = max(span * 1.22, height * 0.22)
            ph = max(height * 0.095, span * 0.42)
            ax.add_patch(
                Ellipse(
                    mid,
                    width=pw,
                    height=ph,
                    angle=angle,
                    facecolor=_CLOTH_PANTS,
                    edgecolor=_MANIKIN_EDGE,
                    linewidth=0.70,
                    alpha=1.0,
                    zorder=3.3,
                    antialiased=True,
                )
            )
            # Soft highlight on the upper pelvis rim.
            ax.add_patch(
                Ellipse(
                    (mid[0], mid[1] + ph * 0.08),
                    width=pw * 0.72,
                    height=ph * 0.42,
                    angle=angle,
                    facecolor=_blend_hex(_CLOTH_PANTS, "#ffffff", 0.22),
                    edgecolor="none",
                    alpha=0.30,
                    zorder=3.32,
                    antialiased=True,
                )
            )
            for hip_pt in (lh, rh):
                hw = min(height * 0.058, max(span * 0.26, height * 0.028))
                ax.add_patch(
                    Ellipse(
                        hip_pt,
                        width=hw,
                        height=hw * 0.88,
                        facecolor=_CLOTH_PANTS,
                        edgecolor=_MANIKIN_EDGE,
                        linewidth=0.35,
                        alpha=1.0,
                        zorder=3.35,
                        antialiased=True,
                    )
                )


def _stability_overlay_color(stability_state: str | None) -> str:
    if stability_state == "Stable":
        return COM_FILL_STABLE
    if stability_state == "Reduced Stability":
        return COM_FILL_REDUCED
    if stability_state == "Unstable":
        return COM_FILL_UNSTABLE
    return MUTED


def _stability_overlay_edge_color(stability_state: str | None) -> str:
    if stability_state == "Stable":
        return COM_EDGE_STABLE
    if stability_state == "Reduced Stability":
        return COM_EDGE_REDUCED
    if stability_state == "Unstable":
        return COM_EDGE_UNSTABLE
    return BORDER


def _com_floor_projection_display(
    snapshot: SkeletonSnapshot,
    com_xyz: tuple[float, float, float],
    height: float,
    *,
    com_projection_xz: tuple[float, float] | None,
    floor_y: float | None,
    display_mode: str,
) -> tuple[float, float]:
    """Horizontal COM projection on the skeleton floor plane."""
    cx, _cy, cz = com_xyz
    y_floor = _resolve_floor_y(snapshot, height, floor_y=floor_y, display_mode=display_mode)
    if com_projection_xz is not None:
        px, pz = com_projection_xz
        return _canonical_point_to_display(snapshot, px, y_floor, pz, display_mode=display_mode)
    return _canonical_point_to_display(snapshot, cx, y_floor, cz, display_mode=display_mode)


def _stability_polygon_style(stability_state: str | None) -> tuple[str, str, float, float]:
    """Face color, edge color, face alpha, edge alpha for BoS floor polygon."""
    if stability_state == "Stable":
        return BOS_FILL_STABLE, BOS_EDGE_STABLE, 0.12, 0.58
    if stability_state == "Reduced Stability":
        return BOS_FILL_REDUCED, BOS_EDGE_REDUCED, 0.14, 0.62
    if stability_state == "Unstable":
        return BOS_FILL_UNSTABLE, BOS_EDGE_UNSTABLE, 0.16, 0.66
    return MUTED, BORDER, 0.10, 0.48


def _support_edge_accent(support_type: str | None) -> str | None:
    """Foot-side accent for single-stance floor patches."""
    if support_type == "left_stance":
        return _COLOR_LEFT
    if support_type == "right_stance":
        return _COLOR_RIGHT
    if support_type == "double_support":
        return ACCENT_ALT
    return None


def _line_to_floor_strip(
    p0: tuple[float, float],
    p1: tuple[float, float],
    height: float,
) -> list[tuple[float, float]]:
    """Render-only: widen a 2-point support segment into a visible floor quadrilateral."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-9:
        pad = max(height * 0.020, 0.008)
        return [
            (p0[0] - pad, p0[1] - pad),
            (p0[0] + pad, p0[1] - pad),
            (p0[0] + pad, p0[1] + pad),
            (p0[0] - pad, p0[1] + pad),
        ]
    nx, ny = -dy / length, dx / length
    half_w = max(height * 0.016, 0.007)
    return [
        (p0[0] + nx * half_w, p0[1] + ny * half_w),
        (p1[0] + nx * half_w, p1[1] + ny * half_w),
        (p1[0] - nx * half_w, p1[1] - ny * half_w),
        (p0[0] - nx * half_w, p0[1] - ny * half_w),
    ]


def _floor_polygon_verts(
    snapshot: SkeletonSnapshot,
    polygon_xz: list[tuple[float, float]],
    height: float,
    *,
    floor_y: float | None,
    display_mode: str,
) -> list[tuple[float, float]]:
    """Map horizontal support polygon vertices onto the skeleton floor plane."""
    if not polygon_xz:
        return []
    y_floor = _resolve_floor_y(snapshot, height, floor_y=floor_y, display_mode=display_mode)
    verts = [
        _canonical_point_to_display(snapshot, x, y_floor, z, display_mode=display_mode)
        for x, z in polygon_xz
    ]
    if len(verts) == 2:
        return _line_to_floor_strip(verts[0], verts[1], height)
    return verts


def _support_state_display(support_type: str | None) -> str | None:
    if not support_type:
        return None
    return _SUPPORT_STATE_LABELS.get(support_type, support_type.replace("_", " ").title())


def _com_support_status_label(stability_state: str | None) -> str | None:
    """Human-readable COM vs BoS status when support overlay is active."""
    if stability_state == "Stable":
        return "Inside support"
    if stability_state == "Reduced Stability":
        return "Near boundary"
    if stability_state == "Unstable":
        return "Outside support"
    return None


def _draw_com_sphere(
    ax: Axes,
    center: tuple[float, float],
    radius: float,
    color: str,
    *,
    edge_color: str | None = None,
) -> None:
    """Render COM as a stability-colored shaded sphere in the skeleton projection."""
    from matplotlib.patches import Circle

    rim = edge_color or color
    glow = Circle(
        center,
        radius * 1.20,
        facecolor=color,
        edgecolor="none",
        alpha=0.09,
        zorder=8,
    )
    ax.add_patch(glow)
    body = Circle(
        center,
        radius,
        facecolor=color,
        edgecolor=rim,
        linewidth=0.8,
        alpha=0.48,
        zorder=9,
    )
    ax.add_patch(body)
    highlight = Circle(
        (center[0] - radius * 0.28, center[1] + radius * 0.30),
        radius * 0.30,
        facecolor="#ffffff",
        edgecolor="none",
        alpha=0.22,
        zorder=10,
    )
    ax.add_patch(highlight)


def _draw_com_trail(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    trail_xyz: list[tuple[float, float, float]],
    height: float,
    *,
    display_mode: str,
    color: str,
    edge_color: str,
) -> None:
    """Fading COM trajectory over recent frames, brightening toward the current frame."""
    if len(trail_xyz) < 2:
        return
    points = [
        _canonical_point_to_display(snapshot, x, y, z, display_mode=display_mode)
        for x, y, z in trail_xyz
    ]
    n = len(points)
    for i in range(n - 1):
        t = i / max(n - 2, 1)
        alpha = 0.07 + 0.21 * t
        width = 0.65 + 0.45 * t
        ax.plot(
            [points[i][0], points[i + 1][0]],
            [points[i][1], points[i + 1][1]],
            color=color,
            linewidth=width,
            alpha=alpha,
            zorder=8,
            solid_capstyle="round",
        )
    # Trail dots stay tiny — markersize is in screen points, not data units.
    for i, pt in enumerate(points[:-1]):
        t = i / max(n - 2, 1)
        alpha = 0.10 + 0.28 * t
        ax.plot(
            [pt[0]],
            [pt[1]],
            marker="o",
            color=color,
            markeredgecolor=edge_color,
            markeredgewidth=0.3,
            markersize=2.2 + 1.4 * t,
            alpha=alpha,
            zorder=8,
        )
    if n >= 2:
        ax.plot(
            [points[-2][0]],
            [points[-2][1]],
            marker="o",
            color=color,
            markeredgecolor=edge_color,
            markeredgewidth=0.4,
            markersize=3.5,
            alpha=0.45,
            zorder=9,
        )


def _draw_com_velocity_arrow(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    com_xyz: tuple[float, float, float],
    velocity: tuple[float, float, float],
    height: float,
    *,
    display_mode: str,
    color: str,
    edge_color: str,
) -> None:
    """Optional horizontal COM velocity arrow from the current COM position."""
    from matplotlib.patches import FancyArrow

    vx, _vy, vz = velocity
    speed = (vx * vx + vz * vz) ** 0.5
    if speed < 1e-4:
        return
    cx, cy, cz = com_xyz
    origin = _canonical_point_to_display(snapshot, cx, cy, cz, display_mode=display_mode)
    tip = _canonical_point_to_display(
        snapshot,
        cx + vx * 0.14,
        cy,
        cz + vz * 0.14,
        display_mode=display_mode,
    )
    dx = tip[0] - origin[0]
    dy = tip[1] - origin[1]
    mag = (dx * dx + dy * dy) ** 0.5
    if mag < 1e-5:
        return
    scale = min(max(height * 0.26, 0.10), mag * 2.8) / mag
    ax.add_patch(
        FancyArrow(
            origin[0],
            origin[1],
            dx * scale,
            dy * scale,
            width=height * 0.022,
            head_width=height * 0.050,
            head_length=height * 0.036,
            length_includes_head=True,
            facecolor=color,
            edgecolor=edge_color,
            linewidth=0.8,
            alpha=0.92,
            zorder=14,
        )
    )
    ax.text(
        origin[0] + dx * scale * 1.05,
        origin[1] + dy * scale * 1.05,
        "v",
        color=edge_color,
        fontsize=6,
        ha="left",
        va="center",
        fontweight="bold",
        zorder=15,
    )


def _canonical_point_to_display(
    snapshot: SkeletonSnapshot,
    x: float,
    y: float,
    z: float,
    *,
    display_mode: str,
) -> tuple[float, float]:
    """Map a canonical 3D point into the active skeleton display frame."""
    if display_mode == "3d_normalized":
        return canonical_to_visualization_oblique(x, y, z)
    if display_mode == "side":
        return (z, y)
    if display_mode == "2d_pose":
        return canonical_to_visualization_oblique(x, y, z)

    pelvis = snapshot.joints.get("left_hip") or snapshot.joints.get(ROOT_JOINT_ID)
    pelvis_xy = _xy(snapshot, "left_hip") or _xy(snapshot, ROOT_JOINT_ID)
    if pelvis is None or not pelvis_xy:
        return (x + OBLIQUE_DISPLAY_SHEAR * z, y)
    dx = x - pelvis.position.x
    dy = y - pelvis.position.y
    dz = z - pelvis.position.z
    angle = math.radians(_WALKING_VIEW_AZIMUTH_DEG)
    return (
        pelvis_xy[0] + math.cos(angle) * dx + math.sin(angle) * dz,
        pelvis_xy[1] + dy,
    )


def _resolve_floor_y(
    snapshot: SkeletonSnapshot,
    height: float,
    *,
    floor_y: float | None,
    display_mode: str,
) -> float:
    foot_pts = [
        _xy(snapshot, j)
        for j in ("left_toe", "right_toe", "left_heel", "right_heel", "left_ankle", "right_ankle")
    ]
    foot_pts = [p for p in foot_pts if p]
    if floor_y is not None and display_mode != "2d_pose":
        return floor_y
    if foot_pts:
        return min(p[1] for p in foot_pts) - height * 0.025
    return floor_y if floor_y is not None else 0.0


def _draw_ground(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    height: float,
    *,
    floor_y: float | None = None,
    display_mode: str = "biomechanical",
    show_ground_plane: bool = True,
) -> float:
    foot_pts = [
        _xy(snapshot, j)
        for j in ("left_toe", "right_toe", "left_heel", "right_heel", "left_ankle", "right_ankle")
    ]
    foot_pts = [p for p in foot_pts if p]
    y_floor = _resolve_floor_y(snapshot, height, floor_y=floor_y, display_mode=display_mode)
    if not foot_pts and floor_y is None:
        return y_floor

    pelvis = _xy(snapshot, ROOT_JOINT_ID) or _xy(snapshot, "spine")
    cx = pelvis[0] if pelvis else 0.0
    half = height * 0.32

    if show_ground_plane:
        ax.plot(
            [cx - half, cx + half],
            [y_floor, y_floor],
            color=_COLOR_GROUND,
            linewidth=1.05,
            linestyle=(0, (4, 3)),
            alpha=0.62,
            zorder=0,
        )

    if floor_y is not None and display_mode != "2d_pose" and show_ground_plane:
        for _side, joints in (
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
    return y_floor


def _draw_head(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    height: float,
    *,
    dimmed: bool = False,
    selected: bool = False,
) -> None:
    """Oval head + hair cap + jaw + neck (OpenSim / Vicon manikin style)."""
    from matplotlib.patches import Circle, Ellipse

    head = _xy(snapshot, "head")
    neck = _xy(snapshot, "neck")
    if not head:
        return
    if selected:
        return  # drawn by selection highlights

    alpha = _DIMMED_JOINT_ALPHA if dimmed else 0.97
    # Slightly elongated cranial oval (not a flat sphere).
    rx = max(height * 0.070, 0.034)
    ry = max(height * 0.085, 0.040)
    # Lift center slightly above the nose landmark toward the cranium.
    cy = head[1] + ry * 0.18
    # Neck connection from neck landmark up toward the head.
    if neck is not None:
        _draw_tapered_capsule(
            ax,
            neck,
            (head[0], cy - ry * 0.62),
            start_half_width=height * 0.028,
            end_half_width=height * 0.034,
            mid_half_width=height * 0.030,
            color=_NECK_SKIN,
            zorder=5.8,
            alpha=alpha * 0.95,
        )
    ax.add_patch(
        Ellipse(
            (head[0], cy),
            width=rx * 2.0,
            height=ry * 2.0,
            facecolor=_SKIN,
            edgecolor=_SKIN_SHADOW,
            linewidth=0.85,
            zorder=6.2,
            alpha=alpha,
            antialiased=True,
        )
    )
    # Soft jaw / chin disc for a more human silhouette.
    jaw_r = rx * 0.58
    ax.add_patch(
        Circle(
            (head[0], cy - ry * 0.42),
            jaw_r,
            facecolor=_SKIN,
            edgecolor=_SKIN_SHADOW,
            linewidth=0.55,
            zorder=6.15,
            alpha=alpha * 0.94,
            antialiased=True,
        )
    )
    # Hair cap on the upper skull (wedge — not a flat brown bar).
    from matplotlib.patches import Wedge

    ax.add_patch(
        Wedge(
            (head[0], cy + ry * 0.12),
            ry * 1.05,
            15,
            165,
            width=ry * 0.58,
            facecolor=_HAIR,
            edgecolor="none",
            zorder=6.22,
            alpha=0.92 * alpha,
        )
    )
    # Ambient highlight on the upper skull.
    ax.add_patch(
        Ellipse(
            (head[0] - rx * 0.22, cy + ry * 0.14),
            width=rx * 0.95,
            height=ry * 0.52,
            facecolor="#ffffff",
            edgecolor="none",
            zorder=6.28,
            alpha=0.16 * alpha,
            antialiased=True,
        )
    )
    # Soft cheek rim for a less flat face silhouette.
    ax.add_patch(
        Ellipse(
            (head[0] + rx * 0.28, cy - ry * 0.05),
            width=rx * 0.42,
            height=ry * 0.55,
            facecolor=_SKIN_SHADOW,
            edgecolor="none",
            zorder=6.24,
            alpha=0.22 * alpha,
            antialiased=True,
        )
    )


def _display_hand_tip(
    snapshot: SkeletonSnapshot,
    side: str,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Wrist → synthetic hand tip along the forearm axis (display-only)."""
    elbow = _xy(snapshot, f"{side}_elbow")
    wrist = _xy(snapshot, f"{side}_wrist")
    if not elbow or not wrist:
        return None
    dx, dy = wrist[0] - elbow[0], wrist[1] - elbow[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return None
    # Hand length ≈ 35% of forearm (anthropometric display stub).
    tip = (wrist[0] + dx / length * length * 0.35, wrist[1] + dy / length * length * 0.35)
    return wrist, tip


def _draw_foot_sole(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    side: str,
    height: float,
    *,
    alpha: float = 0.88,
) -> None:
    """Filled foot wedge (heel → toe) so foot orientation reads clearly."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    heel = _xy(snapshot, f"{side}_heel")
    toe = _xy(snapshot, f"{side}_toe")
    ankle = _xy(snapshot, f"{side}_ankle")
    if heel is None or toe is None:
        return
    dx, dy = toe[0] - heel[0], toe[1] - heel[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return
    nx, ny = -dy / length, dx / length
    half = max(height * 0.024, length * 0.22)
    # Slightly wider at the toe for a natural foot silhouette.
    heel_w, toe_w = half * 0.88, half * 1.22
    verts = [
        (heel[0] + nx * heel_w, heel[1] + ny * heel_w),
        (toe[0] + nx * toe_w, toe[1] + ny * toe_w),
        (toe[0] - nx * toe_w, toe[1] - ny * toe_w),
        (heel[0] - nx * heel_w, heel[1] - ny * heel_w),
        (heel[0] + nx * heel_w, heel[1] + ny * heel_w),
    ]
    color = _CLOTH_SHOES
    ax.add_patch(
        PathPatch(
            Path(
                verts,
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
            ),
            facecolor=color,
            edgecolor=_COLOR_BONE_OUTLINE,
            linewidth=0.65,
            alpha=alpha * 0.92,
            zorder=4.6,
            antialiased=True,
            joinstyle="round",
        )
    )
    # Soft sole shadow under the foot wedge (reads as ground contact volume).
    shadow_offset = height * 0.006
    shadow_verts = [
        (v[0], v[1] - shadow_offset) for v in verts[:-1]
    ] + [(verts[0][0], verts[0][1] - shadow_offset)]
    ax.add_patch(
        PathPatch(
            Path(
                shadow_verts,
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
            ),
            facecolor="#000000",
            edgecolor="none",
            alpha=0.14 * alpha,
            zorder=4.55,
            antialiased=True,
        )
    )
    rim = _COLOR_LEFT if side == "left" else _COLOR_RIGHT
    ax.plot(
        [heel[0], toe[0]],
        [heel[1], toe[1]],
        color=rim,
        linewidth=1.4,
        alpha=0.45 * alpha,
        solid_capstyle="round",
        zorder=4.7,
        antialiased=True,
    )
    if ankle is not None:
        # Soft ankle→foot bridge.
        _draw_tapered_capsule(
            ax,
            ankle,
            ((heel[0] + toe[0]) * 0.5, (heel[1] + toe[1]) * 0.5),
            start_half_width=height * 0.022,
            end_half_width=height * 0.028,
            mid_half_width=height * 0.026,
            color=color,
            zorder=4.55,
            alpha=alpha * 0.80,
        )


def _draw_hands(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    height: float,
    *,
    dimmed: bool = False,
) -> None:
    """Palm stubs past the wrists so limbs do not end abruptly."""
    from matplotlib.patches import Circle, Ellipse

    alpha = (_DIMMED_BONE_ALPHA if dimmed else 0.94)
    for side in ("left", "right"):
        ends = _display_hand_tip(snapshot, side)
        if ends is None:
            continue
        wrist, tip = ends
        rim = _COLOR_LEFT if side == "left" else _COLOR_RIGHT
        _draw_tapered_capsule(
            ax,
            wrist,
            tip,
            start_half_width=height * 0.022,
            end_half_width=height * 0.030,
            mid_half_width=height * 0.028,
            color=_SKIN,
            zorder=5.2,
            alpha=alpha,
        )
        # Soft palm disc at the tip.
        ax.add_patch(
            Ellipse(
                tip,
                width=max(height * 0.042, 0.016),
                height=max(height * 0.034, 0.013),
                facecolor=_SKIN,
                edgecolor=_SKIN_SHADOW,
                linewidth=0.55,
                zorder=5.4,
                alpha=alpha,
                antialiased=True,
            )
        )
        ax.add_patch(
            Circle(
                tip,
                max(height * 0.010, 0.005),
                facecolor=rim,
                edgecolor="none",
                zorder=5.45,
                alpha=0.35 * alpha,
                antialiased=True,
            )
        )


def _joint_marker_radius(jid: str, height: float) -> float:
    """Tiny articulation dots — limbs carry the volume, not spheres."""
    base = max(height * 0.007, 0.004)
    if jid in ("left_knee", "right_knee", "left_elbow", "right_elbow"):
        return base * 1.35
    if jid in ("left_hip", "right_hip", "left_shoulder", "right_shoulder"):
        return base * 1.20
    if jid in (ROOT_JOINT_ID, "spine", "neck"):
        return base * 0.85
    if "heel" in jid or "toe" in jid:
        return base * 0.90
    if "wrist" in jid or "ankle" in jid:
        return base * 1.10
    return base


def _joint_anatomical_fill(jid: str) -> str:
    """Soft cloth/skin fill so joints blend into the manikin (not stick dots)."""
    if any(tok in jid for tok in ("hip", "knee", "ankle", "heel", "toe")):
        return _CLOTH_PANTS
    if any(tok in jid for tok in ("shoulder", "elbow", "wrist")):
        return _SKIN_ARM
    return _SKIN


def _draw_joint_markers(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    height: float,
    *,
    highlight_joints: set[str] | None,
    display_mode: str = DEFAULT_SKELETON_DISPLAY_MODE,
) -> None:
    from matplotlib.patches import Circle

    highlight = highlight_joints or set()
    dim_others = bool(highlight)
    joint_ids = _MARKER_JOINTS
    if display_mode == "biomechanical":
        joint_ids = tuple(j for j in _MARKER_JOINTS if j in _BIOMECH_JOINT_DOTS)
    elif display_mode == "gait":
        # Visible pick targets so users can click a joint to select its DOF.
        joint_ids = (
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
        fill = (
            _joint_anatomical_fill(jid)
            if display_mode in ("gait", "3d_normalized", "side", "biomechanical")
            else edge
        )
        r = _joint_marker_radius(jid, height)
        # Tiny pick dots — anatomical fill, thin L/R rim (not glowing spheres).
        ax.add_patch(
            Circle(
                pt,
                r,
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.70,
                zorder=7,
                alpha=_DIMMED_JOINT_ALPHA if dim_others else 0.85,
                antialiased=True,
            )
        )


def _draw_joint_glow(
    ax: Axes,
    center: tuple[float, float],
    radius: float,
    color: str,
    *,
    zorder: int = 13,
    max_alpha: float = 0.18,
    layers: int = 2,
) -> None:
    """Subtle radial glow (few translucent rings) behind a selected/hovered joint."""
    from matplotlib.patches import Circle

    layers = max(1, min(int(layers), 3))
    for i in range(layers, 0, -1):
        frac = i / layers
        ax.add_patch(
            Circle(
                center,
                radius * frac,
                facecolor=color,
                edgecolor="none",
                alpha=max_alpha * (1.0 - frac) * 0.85 + max_alpha * 0.08,
                zorder=zorder,
            )
        )


def _draw_hover_highlight(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    hover_joint: str,
    height: float,
) -> None:
    """Highlight the joint currently under the cursor (~1.3× normal size)."""
    from matplotlib.patches import Circle

    pt = _xy(snapshot, hover_joint)
    if not pt:
        return
    r = _joint_marker_radius(hover_joint, height) * _HOVER_JOINT_SCALE
    # Keep hover compact so it does not turn into a giant sphere.
    r = max(r, height * 0.012)
    _draw_joint_glow(ax, pt, r * 1.45, _COLOR_SELECT_JOINT, zorder=14, max_alpha=0.12, layers=2)
    ax.add_patch(
        Circle(
            pt,
            r * 1.15,
            facecolor="none",
            edgecolor=_COLOR_SELECT_JOINT,
            linewidth=1.2,
            alpha=0.90,
            zorder=15,
        )
    )
    ax.add_patch(
        Circle(
            pt,
            r,
            facecolor=_COLOR_SELECT_JOINT,
            edgecolor=_COLOR_JOINT_EDGE,
            linewidth=0.8,
            alpha=0.92,
            zorder=16,
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

    # Soften adjacent bones linked to the selection without repainting the torso.
    drawn: set[tuple[str, str]] = set()
    for parent, child, _side in _all_edges(snapshot, display_mode):
        # Brighten any segment that touches a selected joint (chain glow).
        if parent not in highlight_joints and child not in highlight_joints:
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
        both = parent in highlight_joints and child in highlight_joints
        _draw_bone(
            ax,
            p0,
            p1,
            color=DOF_TRAJ_PATH_COLOR if both else _COLOR_SELECT_BONE,
            width=_bone_width(parent, child, height, display_mode=display_mode)
            + (0.75 if both else 0.45),
            zorder=12,
            alpha=0.92 if both else 0.72,
        )

    for jid in highlight_joints:
        pt = _xy(snapshot, jid)
        if not pt:
            continue
        is_primary = primary_joint is None or jid == primary_joint
        base_r = _joint_marker_radius(jid, height)
        scale = _SELECTED_JOINT_SCALE if is_primary else max(1.35, _SELECTED_JOINT_SCALE * 0.88)
        r = max(base_r * scale, height * 0.014)
        ring_color = _COLOR_SELECT_RING if is_primary else DOF_TRAJ_PATH_COLOR
        fill_color = _COLOR_SELECT_JOINT if is_primary else DOF_TRAJ_DOT_COLOR
        _draw_joint_glow(
            ax,
            pt,
            r * (1.35 if is_primary else 1.20),
            ring_color if is_primary else DOF_TRAJ_PATH_COLOR,
            zorder=13,
            max_alpha=0.11 if is_primary else 0.07,
            layers=1,
        )
        ax.add_patch(
            Circle(
                pt,
                r * 1.12,
                facecolor="none",
                edgecolor=ring_color,
                linewidth=1.3 if is_primary else 1.0,
                zorder=14,
            )
        )
        ax.add_patch(
            Circle(
                pt,
                r,
                facecolor=fill_color,
                edgecolor=_COLOR_JOINT_EDGE,
                linewidth=1.0 if is_primary else 0.8,
                zorder=15,
                alpha=1.0,
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
    pad_x = max(body_h * 0.48, 0.30)
    pad_y = max(body_h * 0.18, 0.10)
    min_sep = max(body_h * 0.15, 0.085)

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


def _collect_body_view_points(
    snapshot: SkeletonSnapshot,
) -> list[tuple[float, float]]:
    """Visible body landmark points used to measure body span / fit the view."""
    points: list[tuple[float, float]] = []
    for jid in _VIEW_SPAN_JOINTS:
        pt = _xy(snapshot, jid)
        if pt:
            points.append((float(pt[0]), float(pt[1])))
    return points


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
    # Keep the callout clear of the body and the central COM annotation. The
    # Overview panel has ample lateral room once the body is fitted vertically.
    offset = max(height * 0.12, 0.065)
    items = list(labeled_joints.items())[:1]

    for jid, label in items:
        pt = _xy(snapshot, jid)
        if not pt:
            continue

        # Offset from the joint itself (not body bbox) so the callout never
        # sits on the spine / opposite limb.
        if jid.startswith("left_"):
            tx = pt[0] - offset
            ty = pt[1] + (height * 0.025 if "hip" in jid else 0.0)
            ha, va = "right", "center"
        elif jid.startswith("right_"):
            tx = pt[0] + offset
            ty = pt[1] + (height * 0.025 if "hip" in jid else 0.0)
            ha, va = "left", "center"
        else:
            tx = pt[0] + offset * 0.70
            ty = max(max_y + offset * 0.25, pt[1] + offset * 0.55)
            ha, va = "left", "bottom"
        _ = (min_x, max_x, min_y)

        anchors.append((tx, ty))
        ax.plot(
            [pt[0], tx],
            [pt[1], ty],
            color=DOF_TRAJ_DOT_COLOR,
            linewidth=0.9,
            alpha=0.70,
            solid_capstyle="round",
            zorder=19,
        )
        ax.text(
            tx,
            ty,
            label,
            ha=ha,
            va=va,
            fontsize=8.0,
            fontweight="bold",
            color=TEXT,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor=PANEL,
                edgecolor=DOF_TRAJ_DOT_COLOR,
                linewidth=0.9,
                alpha=0.94,
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


def _draw_com_vertical_projection(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    com_xyz: tuple[float, float, float],
    height: float,
    *,
    com_projection_xz: tuple[float, float] | None,
    floor_y: float | None,
    display_mode: str,
    stability_state: str | None,
    show_floor_marker: bool,
) -> None:
    """Vertical line from 3D COM to its floor projection."""
    cx, cy, cz = com_xyz
    com_display = _canonical_point_to_display(snapshot, cx, cy, cz, display_mode=display_mode)
    floor_display = _com_floor_projection_display(
        snapshot,
        com_xyz,
        height,
        com_projection_xz=com_projection_xz,
        floor_y=floor_y,
        display_mode=display_mode,
    )
    fill = _stability_overlay_color(stability_state)
    edge = _stability_overlay_edge_color(stability_state)
    ax.plot(
        [com_display[0], floor_display[0]],
        [com_display[1], floor_display[1]],
        color=edge,
        linewidth=1.0,
        linestyle=(0, (5, 4)),
        alpha=0.48,
        zorder=8,
        solid_capstyle="round",
    )
    if show_floor_marker:
        r_proj = max(height * 0.010, 0.006)
        _draw_com_sphere(ax, floor_display, r_proj, fill, edge_color=edge)


def _draw_com_overlay(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    com_xyz: tuple[float, float, float],
    height: float,
    *,
    floor_y: float | None = None,
    display_mode: str = "biomechanical",
    stability_state: str | None = None,
    com_projection_xz: tuple[float, float] | None = None,
    com_trail: list[tuple[float, float, float]] | None = None,
    com_velocity: tuple[float, float, float] | None = None,
    show_velocity_arrow: bool = False,
    show_bos_status: bool = False,
) -> None:
    cx, cy, cz = com_xyz
    com_display = _canonical_point_to_display(snapshot, cx, cy, cz, display_mode=display_mode)
    fill = _stability_overlay_color(stability_state)
    edge = _stability_overlay_edge_color(stability_state)

    if com_trail:
        _draw_com_trail(
            ax,
            snapshot,
            com_trail,
            height,
            display_mode=display_mode,
            color=fill,
            edge_color=edge,
        )

    _draw_com_vertical_projection(
        ax,
        snapshot,
        com_xyz,
        height,
        com_projection_xz=com_projection_xz,
        floor_y=floor_y,
        display_mode=display_mode,
        stability_state=stability_state,
        show_floor_marker=not show_bos_status,
    )

    # COM remains legible but does not compete with the articulated body.
    r = max(height * 0.008, 0.005)
    _draw_com_sphere(ax, com_display, r, fill, edge_color=edge)

    if show_velocity_arrow and com_velocity is not None:
        _draw_com_velocity_arrow(
            ax,
            snapshot,
            com_xyz,
            com_velocity,
            height,
            display_mode=display_mode,
            color=fill,
            edge_color=edge,
        )

    # Tiny COM glyph only; BoS/support status lives in the overlay toolbar / metrics.
    _ = show_bos_status
    ax.text(
        com_display[0],
        com_display[1] + r * 1.6,
        "COM",
        color="#ffffff",
        fontsize=4.5,
        ha="center",
        va="bottom",
        zorder=10,
        fontweight="bold",
        alpha=0.50,
        clip_on=True,
    )


def _draw_support_polygon_overlay(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    polygon_xz: list[tuple[float, float]],
    height: float,
    *,
    floor_y: float | None = None,
    display_mode: str = "biomechanical",
    support_type: str | None = None,
    stability_state: str | None = None,
) -> None:
    from matplotlib.patches import Polygon

    face, edge, face_alpha, edge_alpha = _stability_polygon_style(stability_state)
    accent = _support_edge_accent(support_type)
    if accent is not None and support_type in ("left_stance", "right_stance", "double_support"):
        edge = accent

    verts = _floor_polygon_verts(
        snapshot,
        polygon_xz,
        height,
        floor_y=floor_y,
        display_mode=display_mode,
    )
    if len(verts) >= 3:
        ax.add_patch(
            Polygon(
                verts,
                closed=True,
                facecolor=face,
                edgecolor=edge,
                alpha=face_alpha,
                linewidth=2.2,
                zorder=1,
            )
        )
        for vx, vy in verts:
            ax.plot(
                [vx],
                [vy],
                marker="s",
                color=edge,
                markersize=3.5,
                markeredgecolor="#ffffff",
                markeredgewidth=0.3,
                alpha=0.55,
                zorder=2,
            )
    elif len(verts) == 2:
        ax.plot(
            [verts[0][0], verts[1][0]],
            [verts[0][1], verts[1][1]],
            color=edge,
            linewidth=2.8,
            alpha=edge_alpha,
            zorder=2,
        )

    # No on-canvas "Left/Right Support" badge — it covered the feet and made
    # the skeleton look truncated. Support state is available via overlays/metrics.


def _draw_com_ground_projection_marker(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    com_projection_xz: tuple[float, float],
    height: float,
    *,
    floor_y: float | None = None,
    display_mode: str = "biomechanical",
    stability_state: str | None = None,
) -> None:
    from matplotlib.patches import Circle

    px, pz = com_projection_xz
    y_floor = _resolve_floor_y(snapshot, height, floor_y=floor_y, display_mode=display_mode)
    ground_display = _canonical_point_to_display(snapshot, px, y_floor, pz, display_mode=display_mode)
    color = _stability_overlay_color(stability_state)
    outer_r = max(height * 0.018, 0.008)
    inner_r = max(height * 0.008, 0.004)
    cross = outer_r * 1.15

    ax.add_patch(
        Circle(
            ground_display,
            outer_r,
            facecolor="none",
            edgecolor=color,
            linewidth=2.0,
            alpha=0.95,
            zorder=4,
        )
    )
    ax.add_patch(
        Circle(
            ground_display,
            inner_r,
            facecolor=color,
            edgecolor="#ffffff",
            linewidth=0.9,
            alpha=0.96,
            zorder=5,
        )
    )
    gx, gy = ground_display
    ax.plot(
        [gx - cross, gx + cross],
        [gy, gy],
        color="#ffffff",
        linewidth=1.0,
        alpha=0.9,
        zorder=5,
        solid_capstyle="round",
    )
    ax.plot(
        [gx, gx],
        [gy - cross, gy + cross],
        color="#ffffff",
        linewidth=1.0,
        alpha=0.9,
        zorder=5,
        solid_capstyle="round",
    )


def _draw_com_floor_connector(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    com_xyz: tuple[float, float, float],
    com_projection_xz: tuple[float, float],
    height: float,
    *,
    floor_y: float | None = None,
    display_mode: str = "biomechanical",
    stability_state: str | None = None,
) -> None:
    """Vertical guide from 3D COM to its floor projection."""
    cx, cy, cz = com_xyz
    px, pz = com_projection_xz
    y_floor = _resolve_floor_y(snapshot, height, floor_y=floor_y, display_mode=display_mode)
    com_display = _canonical_point_to_display(snapshot, cx, cy, cz, display_mode=display_mode)
    floor_display = _canonical_point_to_display(snapshot, px, y_floor, pz, display_mode=display_mode)
    color = _stability_overlay_color(stability_state)
    ax.plot(
        [com_display[0], floor_display[0]],
        [com_display[1], floor_display[1]],
        color=color,
        linewidth=1.3,
        linestyle=(0, (4, 3)),
        alpha=0.72,
        zorder=3,
    )


def _draw_base_of_support_overlay(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    polygon_xz: list[tuple[float, float]] | None,
    height: float,
    *,
    floor_y: float | None = None,
    display_mode: str = "biomechanical",
    support_type: str | None = None,
    stability_state: str | None = None,
    com_projection_xz: tuple[float, float] | None = None,
    com_xyz: tuple[float, float, float] | None = None,
    draw_com_projection: bool = True,
    draw_com_connector: bool = False,
) -> None:
    """Draw BoS floor polygon, projected COM, and optional vertical connector."""
    poly = list(polygon_xz or [])
    if support_type != "swing" and len(poly) >= 2:
        _draw_support_polygon_overlay(
            ax,
            snapshot,
            poly,
            height,
            floor_y=floor_y,
            display_mode=display_mode,
            support_type=support_type,
            stability_state=stability_state,
        )
    elif support_type == "swing":
        label = _support_state_display(support_type)
        if label:
            anchor = _xy(snapshot, "left_hip") or _xy(snapshot, ROOT_JOINT_ID)
            y_floor = _resolve_floor_y(snapshot, height, floor_y=floor_y, display_mode=display_mode)
            if anchor is not None:
                ax.text(
                    anchor[0],
                    y_floor - height * 0.04,
                    label,
                    color=MUTED,
                    fontsize=6,
                    ha="center",
                    va="top",
                    alpha=0.85,
                    zorder=2,
                )

    if draw_com_projection and com_projection_xz is not None:
        _draw_com_ground_projection_marker(
            ax,
            snapshot,
            com_projection_xz,
            height,
            floor_y=floor_y,
            display_mode=display_mode,
            stability_state=stability_state,
        )
    if draw_com_connector and com_xyz is not None and com_projection_xz is not None:
        _draw_com_floor_connector(
            ax,
            snapshot,
            com_xyz,
            com_projection_xz,
            height,
            floor_y=floor_y,
            display_mode=display_mode,
            stability_state=stability_state,
        )


def _draw_contact_points_overlay(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    foot_contact: tuple[int, int],
    height: float,
    *,
    floor_y: float | None = None,
    display_mode: str = "biomechanical",
) -> None:
    """Ground-projected foot contact landmarks for the stance foot/feet."""
    left_on, right_on = foot_contact
    y_floor = _resolve_floor_y(snapshot, height, floor_y=floor_y, display_mode=display_mode)
    marker_size = 9.0

    for side, active, color in (
        ("left", left_on, _COLOR_LEFT),
        ("right", right_on, _COLOR_RIGHT),
    ):
        if not active:
            continue
        for jid in (f"{side}_heel", f"{side}_toe", f"{side}_ankle"):
            sample = snapshot.joints.get(jid)
            if sample is None:
                continue
            pt = _canonical_point_to_display(
                snapshot,
                sample.position.x,
                y_floor,
                sample.position.z,
                display_mode=display_mode,
            )
            ax.scatter(
                [pt[0]],
                [pt[1]],
                s=marker_size,
                color=color,
                edgecolors="#ffffff",
                linewidths=0.3,
                alpha=0.52,
                zorder=5,
            )


def _draw_gait_direction_overlay(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    direction_xz: tuple[float, float],
    height: float,
) -> None:
    from matplotlib.patches import FancyArrow

    pelvis_xy = _xy(snapshot, "left_hip")
    if not pelvis_xy:
        return
    dx, dz = direction_xz
    mag = (dx * dx + dz * dz) ** 0.5
    if mag < 1e-6:
        return
    scale = height * 0.24
    ax.add_patch(
        FancyArrow(
            pelvis_xy[0],
            pelvis_xy[1],
            (dx / mag) * scale,
            (dz / mag) * scale * 0.4,
            width=height * 0.006,
            head_width=height * 0.022,
            head_length=height * 0.018,
            color="#94d82d",
            alpha=0.45,
            zorder=8,
        )
    )


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
    min_sep = height * 0.14
    placed: list[tuple[float, float]] = []

    for jid, label, pt, (tx, ty), ha, va in layouts:
        # Keep labels further from the limb so they do not sit on the thigh.
        dx = tx - pt[0]
        dy = ty - pt[1]
        dist = math.hypot(dx, dy)
        min_leader = height * 0.16
        if dist < min_leader and dist > 1e-9:
            scale = min_leader / dist
            tx = pt[0] + dx * scale
            ty = pt[1] + dy * scale
        # Final pass: resolve any cross-side overlap by nudging along the leader axis.
        for _ in range(12):
            if not any(_labels_overlap((tx, ty), p, min_sep) for p in placed):
                break
            if ha == "right":
                tx -= height * 0.06
                ty += height * 0.02
            elif ha == "left":
                tx += height * 0.06
                ty += height * 0.02
            else:
                ty += height * 0.06
        placed.append((tx, ty))
        text_anchors.append((tx, ty))

        ax.annotate(
            label,
            xy=pt,
            xytext=(tx, ty),
            ha=ha,
            va=va,
            fontsize=8.5,
            fontweight="medium",
            color=TEXT,
            annotation_clip=False,
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor=PANEL,
                edgecolor=_COLOR_SELECT_RING,
                linewidth=0.9,
                alpha=0.97,
            ),
            arrowprops=dict(
                arrowstyle="-|>",
                color=_COLOR_SELECT_RING,
                linewidth=0.95,
                mutation_scale=9,
                shrinkA=4,
                shrinkB=7,
                connectionstyle="arc3,rad=0.14",
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
    # Invisible hit targets: keep the drawn mark tiny (huge markers still tint
    # the canvas at low alpha) and rely on pickradius for click/hover tolerance.
    scatter = ax.scatter(
        xs,
        ys,
        s=12.0,
        c="#ffffff",
        alpha=0.001,
        edgecolors="none",
        zorder=18,
        picker=True,
        pickradius=max(18, int(12 * _PICK_HITBOX_SCALE)),
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
) -> tuple[float, float, float, float, float, float] | None:
    """Per-frame view box from the current pose.

    Returns ``(cx, cy, half_x, half_y, tight_half_x, tight_half_y)`` where the
    ``tight_*`` halves exclude padding (used to clamp the size boost so the full
    body stays visible). Used only when no stable sequence-global box exists.
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
    return (
        cx,
        cy,
        span_x / 2.0 + pad,
        span_y / 2.0 + pad,
        span_x / 2.0,
        span_y / 2.0,
    )


def compute_skeleton_view_box(
    recording,
    display_mode: str = DEFAULT_SKELETON_DISPLAY_MODE,
) -> tuple[float, float, float, float, float, float] | None:
    """Stable view box ``(cx, cy, half_x, half_y, tight_half_x, tight_half_y)``.

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
    return (
        cx,
        cy,
        span_x / 2.0 + pad,
        span_y / 2.0 + pad,
        span_x / 2.0,
        span_y / 2.0,
    )


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
    def _unpack(box):
        """Return (cx, cy, content_half_x, content_half_y, tight_half_x, tight_half_y)."""
        vals = [float(v) for v in box]
        if len(vals) >= 6:
            return tuple(vals[:6])
        cx, cy, hx, hy = vals[:4]
        # Legacy 4-tuple: treat content halves as the tight body extent.
        return (cx, cy, hx, hy, hx, hy)

    fixed = getattr(ax, "_sw_fixed_view_box", None)
    use_fixed = isinstance(fixed, (tuple, list)) and len(fixed) in (4, 6)
    if use_fixed:
        cx, cy, content_half_x, content_half_y, tight_half_x, tight_half_y = _unpack(fixed)
        # A stable sequence-global box means the limits are frame-independent.
        # Cache them so the tiny equal-aspect box feedback loop cannot drift the
        # scale between frames; reuse verbatim while the box and panel size match.
        _layout_skeleton_figure(ax)
        panel_px = _drawable_pixels(ax)
        cache = getattr(ax, "_sw_view_cache", None)
        key = (round(cx, 6), round(cy, 6), round(content_half_x, 6),
               round(content_half_y, 6), round(tight_half_x, 6),
               round(tight_half_y, 6), round(SKELETON_SIZE_BOOST, 4), panel_px)
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
        cx, cy, content_half_x, content_half_y, tight_half_x, tight_half_y = _unpack(box)

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

    # Enlarge the skeleton (~30% target) by zooming into the fitted view, then
    # clamp so the full head-to-toe body of the widest frame still fits with a
    # sliver of margin. This keeps the model centered and always fully visible
    # while filling the panel far better than the previous conservative fit.
    if SKELETON_SIZE_BOOST > 1.0:
        half_x /= SKELETON_SIZE_BOOST
        half_y /= SKELETON_SIZE_BOOST
        floor_x = tight_half_x / _SKELETON_MAX_FILL
        floor_y = tight_half_y / _SKELETON_MAX_FILL
        grow = max(
            floor_x / max(half_x, 1e-6),
            floor_y / max(half_y, 1e-6),
            1.0,
        )
        half_x *= grow
        half_y *= grow
        # Re-lock the equal-aspect coupling after clamping.
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
    # Wider hover catch radius so joints are easy to target without pixel-perfect aim.
    limit = max_distance if max_distance is not None else height * 0.12

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


# Neutral steel tone for the fading pose echo — reads clearly behind the
# colour-coded live skeleton without competing with it.
_COLOR_GHOST = "#8093a8"


def _draw_ghost_skeletons(
    ax: Axes,
    ghosts: list[SkeletonSnapshot],
    *,
    display_mode: str,
    height: float,
) -> None:
    """Render fading "echo" copies of the previous poses behind the live one.

    ``ghosts`` is ordered oldest → newest (newest is the pose just before the
    current live pose). Older copies are fainter and thinner, so the sequence
    reads as a motion echo flowing toward the current pose. All ghosts share a
    single ``LineCollection`` so the extra poses cost one artist, not hundreds.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgb

    n = len(ghosts)
    if n == 0:
        return
    r, g, b = to_rgb(_COLOR_GHOST)
    segments: list[list[tuple[float, float]]] = []
    colors: list[tuple[float, float, float, float]] = []
    widths: list[float] = []
    for idx, ghost in enumerate(ghosts):
        if not getattr(ghost, "joints", None):
            continue
        frac = (idx + 1) / (n + 1)  # newer → closer to 1.0
        # Smooth ease-in so older poses fade gently; newest ghosts stay readable.
        ease = frac * frac * (3.0 - 2.0 * frac)
        alpha = 0.018 + 0.14 * ease
        width_scale = 0.22 + 0.38 * ease
        ghost._sw_display_xy = _build_display_coords(ghost, display_mode)  # type: ignore[attr-defined]
        for parent, child, side in _all_edges(ghost, display_mode):
            p0, p1 = _xy(ghost, parent), _xy(ghost, child)
            if not p0 or not p1:
                continue
            if not _plausible_segment(parent, child, p0, p1, height):
                continue
            segments.append([p0, p1])
            colors.append((r, g, b, alpha))
            widths.append(
                max(_bone_width(parent, child, height, display_mode=display_mode) * width_scale, 0.55)
            )
    if not segments:
        return
    lc = LineCollection(
        segments,
        colors=colors,
        linewidths=widths,
        capstyle="round",
        joinstyle="round",
        zorder=2,
        antialiased=True,
    )
    ax.add_collection(lc)


def _draw_joint_motion_trail(
    ax: Axes,
    snapshots: list[SkeletonSnapshot],
    joint_ids: set[str],
    *,
    display_mode: str,
    color: str = _COLOR_SELECT_JOINT,
) -> None:
    """Subtle fading trail through a joint's recent positions (oldest → newest)."""
    if not joint_ids or len(snapshots) < 2:
        return
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    for joint_id in joint_ids:
        pts: list[tuple[float, float]] = []
        for s in snapshots:
            if getattr(s, "_sw_display_xy", None) is None:
                s._sw_display_xy = _build_display_coords(s, display_mode)  # type: ignore[attr-defined]
            xy = _xy(s, joint_id)
            if xy:
                pts.append(xy)
        if len(pts) < 2:
            continue
        m = len(pts)
        segments = [[pts[i - 1], pts[i]] for i in range(1, m)]
        colors = [(r, g, b, 0.10 + 0.55 * (i / (m - 1))) for i in range(1, m)]
        widths = [1.0 + 1.8 * (i / (m - 1)) for i in range(1, m)]
        lc = LineCollection(
            segments,
            colors=colors,
            linewidths=widths,
            capstyle="round",
            zorder=3,
            antialiased=True,
        )
        ax.add_collection(lc)


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
    hover_joint: str | None = None,
    labeled_joints: dict[str, str] | None = None,
    motion_arrows: dict[str, tuple[Vec3, Vec3]] | None = None,
    display_mode: str = DEFAULT_SKELETON_DISPLAY_MODE,
    ground_floor_y: float | None = None,
    foot_skeleton_labels: tuple | None = None,
    foot_contact: tuple[int, int] | None = None,
    show_opensim_markers: bool = False,
    com_overlay: tuple[float, float, float] | None = None,
    com_projection_xz: tuple[float, float] | None = None,
    com_trail: list[tuple[float, float, float]] | None = None,
    com_velocity: tuple[float, float, float] | None = None,
    support_polygon: list[tuple[float, float]] | None = None,
    support_type: str | None = None,
    gait_direction: tuple[float, float] | None = None,
    stability_state: str | None = None,
    show_ground_plane: bool = True,
    show_com: bool = True,
    show_com_velocity: bool = False,
    show_support_polygon: bool = True,
    show_contact_points: bool = True,
    show_gait_direction: bool = True,
    ghost_snapshots: list[SkeletonSnapshot] | None = None,
    trail_joints: set[str] | None = None,
) -> None:
    """
    Render an anatomical walking manikin from a ``SkeletonSnapshot``.

    Visualization only — tracked MediaPipe / OpenSim joint *positions* are
    projected and never rewritten. Segment orientation uses the measured
    joint-to-joint axis (OpenSim-style when rotations are unavailable).

    *display_mode*:
      - ``gait`` — articulated walking body with a mild perspective camera
      - ``2d_pose`` — video-frame 2D landmarks (compare with overlay)
      - ``3d_normalized`` — oblique 3D normalized body
      - ``side`` — sagittal side view from measured forward/vertical coordinates
      - ``biomechanical`` — OpenSim-style segment model
    """
    if clear:
        ax.cla()
        setup_skeleton_axes(ax, display_mode=display_mode)

    if not snapshot.joints:
        ax.text(
            0.5, 0.5, "Load a walking video to begin",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=9,
        )
        return

    snapshot._sw_display_xy = _build_display_coords(snapshot, display_mode)  # type: ignore[attr-defined]
    height = _robust_body_height(snapshot)
    highlight = highlight_joints or set()
    labels = labeled_joints or {}

    y_floor = _draw_ground(
        ax,
        snapshot,
        height,
        floor_y=ground_floor_y,
        display_mode=display_mode,
        show_ground_plane=show_ground_plane,
    )

    if show_support_polygon and (support_polygon or support_type):
        _draw_base_of_support_overlay(
            ax,
            snapshot,
            support_polygon,
            height,
            floor_y=y_floor,
            display_mode=display_mode,
            support_type=support_type,
            stability_state=stability_state,
            com_projection_xz=com_projection_xz,
            com_xyz=com_overlay,
            draw_com_projection=com_projection_xz is not None,
            draw_com_connector=False,
        )

    # Motion echo: fading copies of the previous poses drawn behind the live
    # skeleton (proportions untouched — same display transform per pose).
    if ghost_snapshots:
        _draw_ghost_skeletons(
            ax, ghost_snapshots, display_mode=display_mode, height=height
        )

    # Subtle fading trail behind the selected joint(s) to read movement path.
    if trail_joints:
        _draw_joint_motion_trail(
            ax,
            [*(ghost_snapshots or []), snapshot],
            trail_joints,
            display_mode=display_mode,
        )

    # Draw order: ground/BoS (above) → bones → joints → COM/direction →
    # selection → label. Bones must stay visible over overlays.
    _draw_articulated_body_structure(ax, snapshot, height)
    dim_others = bool(highlight)
    # Soft limb de-emphasis when a DOF is selected — torso stays opaque;
    # selected chain stays bright so the active joint reads clearly.
    body_dimmed = False
    edges_with_depth = [
        (
            0.5
            * (
                _joint_camera_depth(snapshot, parent, display_mode=display_mode)
                + _joint_camera_depth(snapshot, child, display_mode=display_mode)
            ),
            parent,
            child,
            side,
        )
        for parent, child, side in _all_edges(snapshot, display_mode)
    ]
    edges_with_depth.sort(key=lambda item: item[0])
    depths = [item[0] for item in edges_with_depth] or [0.0]
    depth_limits = (min(depths), max(depths))
    for depth, parent, child, side in edges_with_depth:
        p0, p1 = _xy(snapshot, parent), _xy(snapshot, child)
        if not p0 or not p1:
            continue
        if not _plausible_segment(parent, child, p0, p1, height):
            continue
        # Anatomical torso fill already draws the trunk — skip stick rails.
        if display_mode in ("gait", "3d_normalized", "side", "biomechanical") and _is_torso_structure_edge(
            parent, child
        ):
            continue
        depth_alpha = _depth_alpha(depth, depth_limits)
        seg_hot = parent in highlight or child in highlight
        if display_mode in ("gait", "3d_normalized", "side", "biomechanical"):
            if dim_others and not seg_hot:
                bone_alpha = _UNSELECTED_LIMB_ALPHA
            else:
                bone_alpha = 1.0
        else:
            bone_alpha = (
                _DIMMED_BONE_ALPHA * depth_alpha if dim_others else depth_alpha
            )
        paint = (
            _limb_paint_color(parent, child, side)
            if display_mode in ("gait", "3d_normalized", "side", "biomechanical")
            else _side_color(side)
        )
        if display_mode in ("gait", "3d_normalized", "side", "biomechanical"):
            _draw_limb_segment(
                ax,
                p0,
                p1,
                parent=parent,
                child=child,
                color=paint,
                height=height,
                zorder=4.0 + depth_alpha * 0.4 + (0.15 if seg_hot else 0.0),
                alpha=bone_alpha,
            )
        else:
            _draw_bone(
                ax,
                p0,
                p1,
                color=paint,
                width=_bone_width(parent, child, height, display_mode=display_mode),
                zorder=4.0 + depth_alpha * 0.4,
                alpha=bone_alpha,
            )

    if show_contact_points and foot_contact is not None:
        _draw_contact_points_overlay(
            ax,
            snapshot,
            foot_contact,
            height,
            floor_y=y_floor,
            display_mode=display_mode,
        )

    _draw_head(
        ax,
        snapshot,
        height,
        dimmed=body_dimmed,
        selected="head" in highlight,
    )
    if display_mode in ("gait", "3d_normalized", "side", "biomechanical"):
        for side in ("left", "right"):
            _draw_foot_sole(
                ax,
                snapshot,
                side,
                height,
                alpha=1.0,
            )
    _draw_hands(ax, snapshot, height, dimmed=body_dimmed)
    _draw_joint_markers(
        ax, snapshot, height, highlight_joints=highlight, display_mode=display_mode
    )

    if show_com and com_overlay is not None:
        _draw_com_overlay(
            ax,
            snapshot,
            com_overlay,
            height,
            floor_y=y_floor,
            display_mode=display_mode,
            stability_state=stability_state,
            com_projection_xz=com_projection_xz,
            com_trail=com_trail,
            com_velocity=com_velocity,
            show_velocity_arrow=show_com_velocity,
            show_bos_status=show_support_polygon,
        )
    if show_gait_direction and gait_direction is not None:
        _draw_gait_direction_overlay(ax, snapshot, gait_direction, height)

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

    # Hovered joint highlight (skip when it's already a selected/highlighted joint
    # so the selection glow stays authoritative).
    if hover_joint and hover_joint not in highlight:
        _draw_hover_highlight(ax, snapshot, hover_joint, height)

    if motion_arrows and paused:
        _draw_motion_arrows(ax, motion_arrows, height)

    _attach_joint_pickers(ax, snapshot, height)

    label_anchors: list[tuple[float, float]] = []
    if labels:
        label_anchors = _draw_inline_joint_labels(ax, snapshot, labels, height)

    # Foot clearance L/R text lives in the strip under the viewport — never on
    # the body canvas (those badges looked like "Support" cards covering the feet).
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
