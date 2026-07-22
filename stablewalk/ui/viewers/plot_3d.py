"""
Matplotlib 3D skeleton drawing and view setup.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from stablewalk.models.pose_data import PoseFrame
    from stablewalk.pose.skeleton_3d import Skeleton3D

from stablewalk.ui.colors import (
    BORDER,
    COM,
    INFO,
    METRIC_GLOBAL,
    MUTED,
    PANEL,
    SIDE_LEFT,
    SIDE_RIGHT,
    TEXT,
    WARNING,
)

BG = PANEL
# Blender / OpenSim laboratory default: slight elevation, front-right azimuth.
DEFAULT_SKELETON_CAMERA_ELEV = 25.0
DEFAULT_SKELETON_CAMERA_AZIM = 35.0
# Perspective camera distance — slightly pulled back so head/feet clear the box.
DEFAULT_SKELETON_CAMERA_DIST = 9.2
# Target body fill of the cubic view (~70%) with extra margin against clipping.
_SKELETON_FRAME_FILL = 0.70
_SKELETON_CLIP_MARGIN = 1.08
_SKELETON_BOX_ZOOM = 0.88
_SKELETON_CAMERA_RESET_MS = 280
_SKELETON_CAMERA_RESET_STEPS = 14

# Accent hints on painted body (left / right / center)
COLOR_LEFT = SIDE_LEFT
COLOR_RIGHT = SIDE_RIGHT
COLOR_TORSO = COM
COLOR_HEAD = WARNING
COLOR_VEL = INFO

# Human figure palette (person in clothes — not colored robot segments)
PAINT_SKIN = "#e8c4a0"
PAINT_SKIN_SHADOW = "#c9a078"
PAINT_HAIR = "#4a3828"
PAINT_NECK = "#deb892"
CLOTH_SHIRT = "#9ec8e4"      # shirt / jacket (readable when small)
CLOTH_PANTS = "#5a7088"      # trousers
CLOTH_SHOES = "#3a4550"
ARM_SKIN = "#f0d4b8"         # forearms & hands
DISPLAY_BODY_SCALE = 1.26    # fill the panel; limbs stay proportional
PAINT_SKIN_EDGE = "#6a5540"
BODY_VIEW_PADDING = 1.06     # tight zoom for compact panel
# Subtle gait side hint (thin accent only — not full limb color)
HINT_LEFT = "#7ab89a"
HINT_RIGHT = "#c88898"
HIGHLIGHT_JOINT = "#ffd700"
HIGHLIGHT_BONE = "#ffb020"

# Which bone connects to which color group
_TORSO_BONES = {
    ("mid_hip", "mid_shoulder"),
    ("mid_shoulder", "nose"),
    ("mid_hip", "left_hip"),
    ("mid_hip", "right_hip"),
}
_LEFT_BONES = {
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("mid_shoulder", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
}
_RIGHT_BONES = {
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("mid_shoulder", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
}

# DoF angle → joint used for 3D label placement
_ANGLE_LABEL_JOINT: dict[str, str] = {
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_ankle_flexion": "left_ankle",
    "right_ankle_flexion": "right_ankle",
    "neck": "mid_shoulder",
    "head_neck": "nose",
    "torso_tilt": "mid_shoulder",
    "pelvis_rotation": "mid_hip",
    "spine": "mid_hip",
}


def _bone_color(a: str, b: str) -> str:
    edge = (a, b)
    rev = (b, a)
    if edge in _TORSO_BONES or rev in _TORSO_BONES:
        return COLOR_TORSO
    if edge in _LEFT_BONES or rev in _LEFT_BONES:
        return COLOR_LEFT
    if edge in _RIGHT_BONES or rev in _RIGHT_BONES:
        return COLOR_RIGHT
    return COLOR_TORSO


def _view_skeleton_camera(ax: Axes, *, elev: float, azim: float) -> None:
    """Apply the Y-up laboratory camera across supported Matplotlib versions."""
    try:
        ax.view_init(elev=elev, azim=azim, vertical_axis="y")
    except TypeError:
        ax.view_init(elev=elev, azim=azim)
    try:
        ax.dist = float(DEFAULT_SKELETON_CAMERA_DIST)
    except AttributeError:
        pass


def remember_skeleton_camera(ax: Axes) -> None:
    """Persist the user orbit so playback redraws never reset the camera."""
    try:
        camera = (float(ax.elev), float(ax.azim))
    except (AttributeError, TypeError, ValueError):
        return
    ax._stablewalk_skeleton_camera = camera  # type: ignore[attr-defined]


def _cancel_skeleton_camera_reset(ax: Axes, scheduler=None) -> None:
    """Stop an in-flight smooth reset so a new gesture owns the camera."""
    job = getattr(ax, "_stablewalk_skeleton_camera_reset_job", None)
    if job is None or scheduler is None:
        ax._stablewalk_skeleton_camera_reset_job = None  # type: ignore[attr-defined]
        return
    cancel = getattr(scheduler, "after_cancel", None)
    if callable(cancel):
        try:
            cancel(job)
        except Exception:
            pass
    ax._stablewalk_skeleton_camera_reset_job = None  # type: ignore[attr-defined]


def setup_3d_axes(
    ax: Axes,
    *,
    elev: float = DEFAULT_SKELETON_CAMERA_ELEV,
    azim: float = DEFAULT_SKELETON_CAMERA_AZIM,
) -> None:
    """Professional Y-up perspective while preserving a remembered user orbit."""
    ax.set_facecolor(BG)
    ax.figure.patch.set_facecolor(BG)
    remembered = getattr(ax, "_stablewalk_skeleton_camera", None)
    if isinstance(remembered, (tuple, list)) and len(remembered) == 2:
        elev, azim = float(remembered[0]), float(remembered[1])
    _view_skeleton_camera(ax, elev=elev, azim=azim)
    ax._stablewalk_skeleton_camera = (float(elev), float(azim))  # type: ignore[attr-defined]
    try:
        ax.set_proj_type("persp", focal_length=1.12)
    except (AttributeError, TypeError, ValueError):
        try:
            ax.set_proj_type("persp")
        except (AttributeError, ValueError):
            pass
    ax.set_xlabel("← left · right →", color=MUTED, fontsize=8)
    ax.set_ylabel("height", color=MUTED, fontsize=8)
    ax.set_zlabel("", color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, alpha=0.16, linewidth=0.4)


def smooth_reset_skeleton_camera(
    ax: Axes,
    *,
    canvas=None,
    scheduler=None,
    duration_ms: int = _SKELETON_CAMERA_RESET_MS,
    steps: int = _SKELETON_CAMERA_RESET_STEPS,
) -> None:
    """Ease the current orbit back to the default Blender/OpenSim-like view."""
    _cancel_skeleton_camera_reset(ax, scheduler)
    try:
        start_elev, start_azim = float(ax.elev), float(ax.azim)
    except (AttributeError, TypeError, ValueError):
        start_elev, start_azim = (
            DEFAULT_SKELETON_CAMERA_ELEV,
            DEFAULT_SKELETON_CAMERA_AZIM,
        )
    target_elev = DEFAULT_SKELETON_CAMERA_ELEV
    target_azim = DEFAULT_SKELETON_CAMERA_AZIM
    # Follow the shortest azimuth arc instead of spinning through ±180°.
    azim_delta = (target_azim - start_azim + 180.0) % 360.0 - 180.0
    steps = max(1, int(steps))
    interval = max(1, int(duration_ms) // steps)

    def _frame(index: int) -> None:
        t = min(max(index / steps, 0.0), 1.0)
        # Smoothstep ease-in-out for a laboratory camera reset.
        eased = t * t * (3.0 - 2.0 * t)
        elev = start_elev + (target_elev - start_elev) * eased
        azim = start_azim + azim_delta * eased
        _view_skeleton_camera(ax, elev=elev, azim=azim)
        ax._stablewalk_skeleton_camera = (elev, azim)  # type: ignore[attr-defined]
        if canvas is not None:
            canvas.draw_idle()
        if index < steps and scheduler is not None:
            job = scheduler.after(interval, lambda: _frame(index + 1))
            ax._stablewalk_skeleton_camera_reset_job = job  # type: ignore[attr-defined]
        else:
            ax._stablewalk_skeleton_camera_reset_job = None  # type: ignore[attr-defined]
            ax._stablewalk_skeleton_camera = (
                target_elev,
                target_azim,
            )  # type: ignore[attr-defined]

    if scheduler is None:
        _frame(steps)
    else:
        _frame(0)


def _rotation_matrix(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix mapping unit vector from_vec → to_vec."""
    a = from_vec.astype(float)
    b = to_vec.astype(float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.eye(3)
    a /= na
    b /= nb
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-8:
        if c < 0:
            # 180° flip around X
            return np.diag([1.0, -1.0, 1.0])
        return np.eye(3)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def orient_skeleton_for_display(skeleton: Skeleton3D) -> Skeleton3D:
    """Rotate skeleton upright (+Y); delegates to ``align_skeleton_upright``."""
    from stablewalk.pose.skeleton_3d import align_skeleton_upright

    return align_skeleton_upright(skeleton)


def _skeleton_axis_limit(skeleton: Skeleton3D, *, padding: float = 1.12) -> float:
    """Half-axis range that tightly fits the current pose."""
    if not skeleton.joints:
        return 0.55
    xs = [j.x for j in skeleton.joints.values()]
    ys = [j.y for j in skeleton.joints.values()]
    zs = [j.z for j in skeleton.joints.values()]
    height = _body_height(skeleton)
    max_r = max(
        max(abs(v) for v in xs),
        max(abs(v) for v in ys),
        max(abs(v) for v in zs),
        height * 0.28,
        0.2,
    )
    return max(0.42, min(max_r * padding, height * 0.72))


def apply_skeleton_limits(ax: Axes, skeleton: Skeleton3D, *, padding: float = 1.12) -> None:
    """Fit axes to the current skeleton (fixes tiny blob in huge grid)."""
    lim = _skeleton_axis_limit(skeleton, padding=padding)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)


def apply_display_limits(
    ax: Axes,
    skeleton: Skeleton3D,
    sequence_limit: float | None = None,
    *,
    padding: float | None = None,
    flat_frontal: bool = False,
) -> None:
    """Center and frame the complete skeleton with equal 3D unit scaling.

    The body is auto-framed to about 70% of the cubic viewport, then padded a
    little more so perspective foreshortening cannot clip the head or feet.
    """
    if flat_frontal and skeleton.joints:
        xs = [j.x for j in skeleton.joints.values()]
        ys = [j.y for j in skeleton.joints.values()]
        pad = float(padding) if padding is not None else (1.0 / _SKELETON_FRAME_FILL)
        lim_xy = max(
            max(abs(v) for v in xs),
            max(abs(v) for v in ys),
            _body_height(skeleton) * 0.42,
            0.25,
        ) * pad
        lim_xy = max(0.45, lim_xy)
        if sequence_limit is not None:
            lim_xy = min(lim_xy, max(0.45, sequence_limit))
        ax.set_xlim(-lim_xy, lim_xy)
        ax.set_ylim(-lim_xy, lim_xy)
        ax.set_zlim(-0.06, 0.06)
        return
    if not skeleton.joints:
        return
    xs = [joint.x for joint in skeleton.joints.values()]
    ys = [joint.y for joint in skeleton.joints.values()]
    zs = [joint.z for joint in skeleton.joints.values()]
    centers = (
        0.5 * (min(xs) + max(xs)),
        0.5 * (min(ys) + max(ys)),
        0.5 * (min(zs) + max(zs)),
    )
    max_span = max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
        _body_height(skeleton) * 0.55,
        0.40,
    )
    # Expand so the body occupies ~70% of the cube, then add a clip margin for
    # perspective projection (corners otherwise crop extremities).
    fill_pad = float(padding) if padding is not None else (1.0 / _SKELETON_FRAME_FILL)
    half = max_span * max(fill_pad, 1.08) * 0.5 * _SKELETON_CLIP_MARGIN
    if sequence_limit is not None:
        half = max(half, float(sequence_limit))
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=_SKELETON_BOX_ZOOM)
    except TypeError:
        try:
            ax.set_box_aspect((1, 1, 1))
        except (AttributeError, ValueError):
            pass
    except (AttributeError, ValueError):
        pass


def scale_skeleton_uniform(skeleton: Skeleton3D, factor: float) -> Skeleton3D:
    out = skeleton.copy()
    for j in out.joints.values():
        j.x *= factor
        j.y *= factor
        j.z *= factor
    return out


def _body_height(skeleton: Skeleton3D) -> float:
    j = skeleton.joints
    if "nose" in j:
        ys_top = j["nose"].y
    else:
        ys_top = max(p.y for p in j.values())
    ankles = [j[n].y for n in ("left_ankle", "right_ankle") if n in j]
    if ankles:
        return max(0.35, ys_top - min(ankles))
    ys = [p.y for p in j.values()]
    return max(0.35, max(ys) - min(ys)) if ys else 0.9


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = direction.astype(float)
    n = np.linalg.norm(d)
    if n < 1e-12:
        d = np.array([0.0, 1.0, 0.0])
    else:
        d = d / n
    helper = np.array([0.0, 0.0, 1.0]) if abs(float(d[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, helper)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(d, u)
    return d, u, v


def _sphere_mesh(
    center: tuple[float, float, float],
    radius: float,
    *,
    nu: int = 16,
    nv: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy, cz = center
    u = np.linspace(0.0, 2.0 * np.pi, nu)
    v = np.linspace(0.0, np.pi, nv)
    x = cx + radius * np.outer(np.cos(u), np.sin(v))
    y = cy + radius * np.outer(np.sin(u), np.sin(v))
    z = cz + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _segment_profile(t_norm: float, kind: str) -> float:
    """Radius multiplier along limb (t in [0,1]): organic taper, thin at joints."""
    t = float(np.clip(t_norm, 0.0, 1.0))
    bulge = np.sin(np.pi * t) ** 0.85
    if kind == "thigh":
        return 0.72 + 0.38 * bulge
    if kind == "calf":
        return 0.68 + 0.28 * bulge
    if kind == "upper_arm":
        return 0.70 + 0.32 * bulge
    if kind == "forearm":
        return 0.65 + 0.22 * bulge
    if kind == "neck":
        return 0.75 + 0.20 * bulge
    return 0.70 + 0.25 * bulge


def _segment_kind(a: str, b: str) -> str:
    pair = {a, b}
    if "knee" in pair and ("hip" in pair or "left_hip" in pair or "right_hip" in pair):
        return "thigh"
    if "ankle" in pair:
        return "calf"
    if "elbow" in pair and "shoulder" in pair:
        return "upper_arm"
    if "wrist" in pair:
        return "forearm"
    if "nose" in pair or "mid_shoulder" in pair:
        return "neck"
    return "limb"


def _organic_limb_mesh(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    radius: float,
    kind: str,
    *,
    n_ring: int = 16,
    n_len: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Smooth limb surface with muscle-like taper (no robot joint balls)."""
    p0a = np.array(p0, dtype=float)
    p1a = np.array(p1, dtype=float)
    seg = p1a - p0a
    length = float(np.linalg.norm(seg))
    if length < 1e-8:
        return None

    d, u, v = _orthonormal_basis(seg)
    t_vals = np.linspace(0.0, length, n_len)
    theta = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    X = np.zeros((n_len, n_ring))
    Y = np.zeros((n_len, n_ring))
    Z = np.zeros((n_len, n_ring))
    for i, t in enumerate(t_vals):
        center = p0a + d * t
        r = radius * _segment_profile(t / length, kind)
        for j, th in enumerate(theta):
            offset = r * (np.cos(th) * u + np.sin(th) * v)
            pt = center + offset
            X[i, j], Y[i, j], Z[i, j] = pt
    return X, Y, Z


def _paint_surface(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    color: str,
    alpha: float = 0.94,
) -> None:
    ax.plot_surface(
        x,
        y,
        z,
        color=color,
        alpha=alpha,
        shade=True,
        linewidth=0,
        antialiased=True,
        edgecolor="none",
        rcount=max(2, x.shape[0] - 1),
        ccount=max(2, x.shape[1] - 1),
    )


def _limb_cloth_color(a: str, b: str) -> str:
    pair = {a, b}
    if "wrist" in pair or ("elbow" in pair and "wrist" in pair):
        return ARM_SKIN
    if "nose" in pair:
        return PAINT_NECK
    if "shoulder" in pair and "elbow" in pair:
        return CLOTH_SHIRT
    if "hip" in pair or "knee" in pair or "ankle" in pair:
        return CLOTH_PANTS
    return CLOTH_SHIRT


def _limb_radius(a: str, b: str, height: float) -> float:
    """Anthropometric radii as fraction of body height."""
    key = (a, b)
    rev = (b, a)
    ratios: dict[tuple[str, str], float] = {
        ("left_hip", "left_knee"): 0.068,
        ("right_hip", "right_knee"): 0.068,
        ("left_knee", "left_ankle"): 0.050,
        ("right_knee", "right_ankle"): 0.050,
        ("left_shoulder", "left_elbow"): 0.044,
        ("right_shoulder", "right_elbow"): 0.044,
        ("left_elbow", "left_wrist"): 0.034,
        ("right_elbow", "right_wrist"): 0.034,
        ("mid_shoulder", "left_shoulder"): 0.040,
        ("mid_shoulder", "right_shoulder"): 0.040,
        ("mid_hip", "left_hip"): 0.048,
        ("mid_hip", "right_hip"): 0.048,
        ("mid_hip", "mid_shoulder"): 0.088,
        ("mid_shoulder", "nose"): 0.032,
    }
    r = ratios.get(key) or ratios.get(rev)
    return height * (r if r is not None else 0.05)


def _ellipse_ring(
    center: tuple[float, float, float],
    rx: float,
    ry: float,
    rz: float,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    n: int = 14,
) -> list[tuple[float, float, float]]:
    cx, cy, cz = center
    pts = []
    for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
        offset = rx * np.cos(th) * u_vec + ry * np.sin(th) * v_vec
        pts.append((cx + offset[0], cy + offset[1], cz + offset[2]))
    return pts


def _draw_loft_torso(ax: Axes, skeleton: Skeleton3D, height: float) -> None:
    """Smooth jacket torso (lofted slices) — reads as a person, not a ball robot."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    j = skeleton.joints
    if not all(n in j for n in ("mid_hip", "mid_shoulder")):
        return
    mh, ms = j["mid_hip"], j["mid_shoulder"]
    spine = np.array([mh.x, mh.y, mh.z]), np.array([ms.x, ms.y, ms.z])
    d, u, v = _orthonormal_basis(spine[1] - spine[0])

    hip_w = height * 0.16
    chest_w = height * 0.20
    if "left_shoulder" in j and "right_shoulder" in j:
        chest_w = max(chest_w, abs(j["left_shoulder"].x - j["right_shoulder"].x) * 0.42)
    if "left_hip" in j and "right_hip" in j:
        hip_w = max(hip_w, abs(j["left_hip"].x - j["right_hip"].x) * 0.38)

    depth = height * 0.09
    slices = [
        (0.0, hip_w, hip_w * 0.85, depth),
        (0.22, hip_w * 1.02, hip_w * 0.95, depth * 1.05),
        (0.45, chest_w * 0.92, chest_w * 0.78, depth * 1.1),
        (0.72, chest_w, chest_w * 0.72, depth * 1.15),
        (1.0, chest_w * 0.88, chest_w * 0.65, depth),
    ]
    rings: list[list[tuple[float, float, float]]] = []
    span = spine[1] - spine[0]
    for t, rx, ry, rz in slices:
        c = spine[0] + span * t
        center = (float(c[0]), float(c[1]), float(c[2]))
        rings.append(_ellipse_ring(center, rx, ry, rz, u, v, n=16))

    quads: list[list[tuple[float, float, float]]] = []
    for i in range(len(rings) - 1):
        r0, r1 = rings[i], rings[i + 1]
        n = min(len(r0), len(r1))
        for k in range(n):
            k2 = (k + 1) % n
            quads.append([r0[k], r0[k2], r1[k2], r1[k]])
    if quads:
        poly = Poly3DCollection(
            quads, alpha=0.93, facecolor=CLOTH_SHIRT, edgecolor="none", linewidths=0
        )
        ax.add_collection3d(poly)

    if all(n in j for n in ("left_shoulder", "right_shoulder")):
        ls, rs = j["left_shoulder"], j["right_shoulder"]
        mesh = _organic_limb_mesh(
            (ls.x, ls.y, ls.z), (rs.x, rs.y, rs.z),
            height * 0.038, "limb", n_ring=12, n_len=4,
        )
        if mesh:
            _paint_surface(ax, *mesh, color=CLOTH_SHIRT, alpha=0.9)


def _draw_head_body(ax: Axes, skeleton: Skeleton3D, height: float) -> None:
    j = skeleton.joints
    if "nose" not in j:
        return
    n = j["nose"]
    rx, ry, rz = height * 0.075, height * 0.095, height * 0.08
    cx, cy, cz = n.x, n.y + ry * 0.35, n.z
    x, y, z = _sphere_mesh((cx, cy, cz), 1.0, nu=20, nv=14)
    x = cx + (x - cx) * rx
    y = cy + (y - cy) * ry
    z = cz + (z - cz) * rz
    _paint_surface(ax, x, y, z, color=PAINT_SKIN, alpha=0.96)
    hx, hy, hz = cx, cy + ry * 0.95, cz
    xh, yh, zh = _sphere_mesh((hx, hy, hz), 1.0, nu=14, nv=8)
    xh = hx + (xh - hx) * (rx * 1.05)
    yh = hy + (yh - hy) * (ry * 0.55)
    zh = hz + (zh - hz) * (rz * 1.02)
    _paint_surface(ax, xh, yh, zh, color=PAINT_HAIR, alpha=0.92)

    if "mid_shoulder" in j:
        ms = j["mid_shoulder"]
        mesh = _organic_limb_mesh(
            (ms.x, ms.y, ms.z), (n.x, n.y, n.z),
            height * 0.030, "neck", n_ring=12, n_len=6,
        )
        if mesh:
            _paint_surface(ax, *mesh, color=PAINT_NECK, alpha=0.95)


def _draw_hands_feet(ax: Axes, skeleton: Skeleton3D, height: float) -> None:
    j = skeleton.joints
    for side in ("left", "right"):
        ankle = j.get(f"{side}_ankle")
        wrist = j.get(f"{side}_wrist")
        if ankle:
            ax_f, ay_f, az_f = ankle.x, ankle.y - height * 0.02, ankle.z
            x, y, z = _sphere_mesh((ax_f, ay_f, az_f), 1.0, nu=12, nv=8)
            x = ax_f + (x - ax_f) * (height * 0.055)
            y = ay_f + (y - ay_f) * (height * 0.028)
            z = az_f + (z - az_f) * (height * 0.07)
            _paint_surface(ax, x, y, z, color=CLOTH_SHOES, alpha=0.95)
        if wrist:
            x, y, z = _sphere_mesh((wrist.x, wrist.y, wrist.z), height * 0.028, nu=10, nv=8)
            _paint_surface(ax, x, y, z, color=PAINT_SKIN, alpha=0.92)


def _pt(joints: dict, name: str, _z_base: float = 0.0) -> tuple[float, float, float] | None:
    """All body paint on the frontal plane (z=0) for a clear 2D person view."""
    p = joints.get(name)
    if not p:
        return None
    return (p.x, p.y, 0.0)


def _ribbon_half_width(a: str, b: str, height: float) -> float:
    pair = {a, b}
    if "hip" in pair and "knee" in pair:
        return height * 0.105
    if "knee" in pair and "ankle" in pair:
        return height * 0.082
    if "shoulder" in pair and "elbow" in pair:
        return height * 0.072
    if "elbow" in pair and "wrist" in pair:
        return height * 0.052
    if "nose" in pair:
        return height * 0.045
    return height * 0.06


def _ribbon_color(a: str, b: str) -> str:
    pair = {a, b}
    if "wrist" in pair or ("elbow" in pair and "wrist" in pair):
        return ARM_SKIN
    if "nose" in pair:
        return PAINT_NECK
    if "hip" in pair or "knee" in pair or "ankle" in pair:
        return CLOTH_PANTS
    return CLOTH_SHIRT


def _add_poly(ax: Axes, verts: list[tuple[float, float, float]], color: str, *, alpha: float = 0.96) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if len(verts) < 3:
        return
    ax.add_collection3d(
        Poly3DCollection(
            [verts],
            facecolor=color,
            edgecolor=PAINT_SKIN_EDGE,
            alpha=alpha,
            linewidths=0.35,
        )
    )


def _draw_ribbon(
    ax: Axes,
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    half_w: float,
    color: str,
) -> None:
    """Filled limb strip (visible width in data units — reads as a body part)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-8:
        return
    nx, ny = -dy / length * half_w, dx / length * half_w
    z = (p0[2] + p1[2]) * 0.5
    _add_poly(
        ax,
        [
            (p0[0] + nx, p0[1] + ny, z),
            (p0[0] - nx, p0[1] - ny, z),
            (p1[0] - nx, p1[1] - ny, z),
            (p1[0] + nx, p1[1] + ny, z),
        ],
        color,
    )
    # Round caps
    _add_poly(ax, _circle_verts(p0[0], p0[1], z, half_w * 0.92), color)
    _add_poly(ax, _circle_verts(p1[0], p1[1], z, half_w * 0.92), color)


def _circle_verts(cx: float, cy: float, cz: float, r: float, n: int = 14) -> list[tuple[float, float, float]]:
    th = np.linspace(0, 2 * np.pi, n, endpoint=True)
    return [(cx + r * math.cos(t), cy + r * math.sin(t), cz) for t in th]


def _draw_body_glow_outline(ax: Axes, skeleton: Skeleton3D, height: float) -> None:
    """Soft outer contour so the shape reads as one person."""
    from stablewalk.pose.skeleton_3d import SKELETON_3D_JOINTS

    j = skeleton.joints
    pts: list[tuple[float, float]] = []
    for name in SKELETON_3D_JOINTS:
        p = j.get(name)
        if p:
            pts.append((p.x, p.y))
    if len(pts) < 4:
        return
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    pad = height * 0.14
    ring = [
        (x + pad * (x - cx) / (math.hypot(x - cx, y - cy) + 1e-8), y + pad * (y - cy) / (math.hypot(x - cx, y - cy) + 1e-8))
        for x, y in pts
    ]
    _add_poly(ax, [(x, y, 0.0) for x, y in ring], "#3d4f5e", alpha=0.35)


def _draw_human_manikin(ax: Axes, skeleton: Skeleton3D) -> None:
    """Frontal painted person (head, shirt, arms, legs) — fills the view."""
    j = skeleton.joints
    if "mid_hip" not in j:
        return
    height = _body_height(skeleton)

    def P(name: str) -> tuple[float, float, float] | None:
        return _pt(j, name)

    _draw_body_glow_outline(ax, skeleton, height)

    # --- Legs ---
    for a, b in (
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("mid_hip", "left_hip"),
        ("mid_hip", "right_hip"),
    ):
        p0, p1 = P(a), P(b)
        if p0 and p1:
            _draw_ribbon(ax, p0, p1, _ribbon_half_width(a, b, height), CLOTH_PANTS)

    # --- Torso (shirt) ---
    ls, rs = P("left_shoulder"), P("right_shoulder")
    lh, rh, mh, ms = P("left_hip"), P("right_hip"), P("mid_hip"), P("mid_shoulder")
    if ls and rs and lh and rh and ms:
        chest = (ms[0], ms[1] + height * 0.04, 0.0)
        _add_poly(ax, [ls, rs, chest, mh, lh], CLOTH_SHIRT)
    elif ms and mh:
        w = height * 0.18
        _add_poly(
            ax,
            [
                (ms[0] - w, ms[1], 0.0),
                (ms[0] + w, ms[1], 0.0),
                (mh[0] + w * 0.75, mh[1], 0.0),
                (mh[0] - w * 0.75, mh[1], 0.0),
            ],
            CLOTH_SHIRT,
        )

    # --- Arms ---
    for a, b in (
        ("mid_shoulder", "left_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("mid_shoulder", "right_shoulder"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    ):
        p0, p1 = P(a), P(b)
        if p0 and p1:
            _draw_ribbon(ax, p0, p1, _ribbon_half_width(a, b, height), _ribbon_color(a, b))

    # --- Neck ---
    nose, ms = P("nose"), P("mid_shoulder")
    if nose and ms:
        _draw_ribbon(ax, ms, nose, height * 0.048, PAINT_NECK)

    # --- Head, hair, face ---
    if nose:
        hr = height * 0.12
        head_cy = nose[1] + hr * 0.32
        _add_poly(ax, _circle_verts(nose[0], head_cy, 0.0, hr), PAINT_SKIN)
        _add_poly(ax, _circle_verts(nose[0], head_cy + hr * 0.5, 0.0, hr * 0.9), PAINT_HAIR)
        for ex in (-hr * 0.32, hr * 0.32):
            _add_poly(ax, _circle_verts(nose[0] + ex, head_cy + hr * 0.08, 0.0, hr * 0.11), "#2e2418")
        _add_poly(ax, _circle_verts(nose[0], nose[1] + hr * 0.05, 0.0, hr * 0.07), "#5a4030")

    # --- Feet & hands ---
    for side in ("left", "right"):
        ankle, wrist = P(f"{side}_ankle"), P(f"{side}_wrist")
        if ankle:
            _add_poly(ax, _circle_verts(ankle[0], ankle[1] - height * 0.04, 0.0, height * 0.06), CLOTH_SHOES)
        if wrist:
            _add_poly(ax, _circle_verts(wrist[0], wrist[1], 0.0, height * 0.045), ARM_SKIN)


def _draw_ghost_manikin(ax: Axes, skeleton: Skeleton3D) -> None:
    """Faint previous pose for motion feedback."""
    j = skeleton.joints
    height = _body_height(skeleton)
    for a, b in (
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    ):
        p0, p1 = _pt(j, a), _pt(j, b)
        if p0 and p1:
            _draw_ribbon(ax, p0, p1, _ribbon_half_width(a, b, height) * 0.7, COLOR_VEL)


def _draw_motion_trail(ax: Axes, trail: list[Skeleton3D], *, view_limit: float) -> None:
    """Hip path on the frontal plane."""
    if len(trail) < 2:
        return
    xs, ys, zs = [], [], []
    for sk in trail:
        h = sk.joints.get("mid_hip")
        if h:
            xs.append(h.x)
            ys.append(h.y)
            zs.append(0.0)
    if len(xs) >= 2:
        ax.plot(xs, ys, zs, color=COLOR_VEL, alpha=0.55, linewidth=2.5, linestyle="--")


def setup_body_axes(ax: Axes) -> None:
    """2D body view — equal aspect, no 3D projection artifacts."""
    ax.set_facecolor(BG)
    ax.figure.patch.set_facecolor(BG)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _xy(joints: dict, name: str) -> tuple[float, float] | None:
    p = joints.get(name)
    if not p:
        return None
    return (p.x, p.y)


def _limb_bar_2d(
    ax: Axes,
    p0: tuple[float, float],
    p1: tuple[float, float],
    half_w: float,
    color: str,
    *,
    alpha: float = 1.0,
    zorder: int = 2,
    edgecolor: str | None = None,
) -> None:
    """Flat limb bar (no round caps — avoids tube blobs when small or ghosted)."""
    from matplotlib.patches import Polygon as MplPolygon

    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-8:
        return
    nx, ny = -dy / length * half_w, dx / length * half_w
    ec = edgecolor if edgecolor is not None else PAINT_SKIN_EDGE
    lw = 0.0 if edgecolor == "none" else 0.55
    ax.add_patch(
        MplPolygon(
            [
                (p0[0] + nx, p0[1] + ny),
                (p0[0] - nx, p0[1] - ny),
                (p1[0] - nx, p1[1] - ny),
                (p1[0] + nx, p1[1] + ny),
            ],
            closed=True,
            facecolor=color,
            edgecolor=ec,
            linewidth=lw,
            alpha=alpha,
            joinstyle="round",
            zorder=zorder,
        )
    )


def _torso_vest_2d(
    ax: Axes,
    ls: tuple[float, float],
    rs: tuple[float, float],
    lh: tuple[float, float],
    rh: tuple[float, float],
    ms: tuple[float, float],
    mh: tuple[float, float],
    height: float,
) -> None:
    """Rounded vest torso (not a flat trapezoid)."""
    from matplotlib.patches import Polygon as MplPolygon

    top_y = max(ls[1], rs[1], ms[1]) + height * 0.04
    bot_y = min(lh[1], rh[1], mh[1]) - height * 0.01
    mid_y = (top_y + bot_y) * 0.5
    cx = (ls[0] + rs[0] + lh[0] + rh[0]) * 0.25
    w_top = abs(rs[0] - ls[0]) * 0.52 + height * 0.10
    w_mid = height * 0.14
    w_bot = abs(rh[0] - lh[0]) * 0.48 + height * 0.08
    verts = [
        (cx - w_top, top_y),
        (cx + w_top, top_y),
        (cx + w_mid, mid_y),
        (cx + w_bot, bot_y),
        (cx - w_bot, bot_y),
        (cx - w_mid, mid_y),
    ]
    ax.add_patch(
        MplPolygon(
            verts,
            closed=True,
            facecolor=CLOTH_SHIRT,
            edgecolor=PAINT_SKIN_EDGE,
            linewidth=0.55,
            joinstyle="round",
            zorder=3,
        )
    )


def _limb_half_width_2d(a: str, b: str, height: float) -> float:
    """Slightly slimmer bars — cleaner at small panel size."""
    return _ribbon_half_width(a, b, height) * 0.88


def _draw_painted_human_2d_body(ax: Axes, skeleton: Skeleton3D) -> None:
    """Compact illustrated person (icon-style, readable when small)."""
    from matplotlib.patches import Circle, Ellipse, Wedge

    j = skeleton.joints
    if "mid_hip" not in j:
        return
    height = _body_height(skeleton)

    def P(name: str) -> tuple[float, float] | None:
        return _xy(j, name)

    # Legs (no hip-to-mid_hip bars — torso covers hips)
    for a, b in (
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ):
        p0, p1 = P(a), P(b)
        if p0 and p1:
            _limb_bar_2d(
                ax, p0, p1, _limb_half_width_2d(a, b, height), CLOTH_PANTS, zorder=2
            )

    # Torso vest (under arms)
    ls, rs = P("left_shoulder"), P("right_shoulder")
    lh, rh, mh, ms = P("left_hip"), P("right_hip"), P("mid_hip"), P("mid_shoulder")
    if ls and rs and lh and rh and ms and mh:
        _torso_vest_2d(ax, ls, rs, lh, rh, ms, mh, height)

    # Arms (shoulder → elbow → wrist only)
    for a, b in (
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    ):
        p0, p1 = P(a), P(b)
        if p0 and p1:
            col = CLOTH_SHIRT if "shoulder" in a and "elbow" in b else ARM_SKIN
            _limb_bar_2d(ax, p0, p1, _limb_half_width_2d(a, b, height), col, zorder=4)

    # Neck
    nose, ms = P("nose"), P("mid_shoulder")
    if nose and ms:
        _limb_bar_2d(ax, ms, nose, height * 0.042, PAINT_NECK, zorder=5)

    # Head (simple — reads at small size)
    if nose:
        hr = height * 0.115
        head_cx, head_cy = nose[0], nose[1] + hr * 0.42
        ax.add_patch(
            Circle(
                (head_cx, head_cy),
                hr,
                facecolor=PAINT_SKIN,
                edgecolor=PAINT_SKIN_EDGE,
                linewidth=0.55,
                zorder=6,
            )
        )
        ax.add_patch(
            Wedge(
                (head_cx, head_cy + hr * 0.35),
                hr * 1.05,
                20,
                160,
                width=hr * 0.55,
                facecolor=PAINT_HAIR,
                edgecolor="none",
                zorder=7,
            )
        )
        eye_r = hr * 0.09
        for ex in (-hr * 0.32, hr * 0.32):
            ax.add_patch(
                Circle(
                    (head_cx + ex, head_cy + hr * 0.08),
                    eye_r,
                    facecolor="#2a2218",
                    edgecolor="none",
                    zorder=8,
                )
            )

    # Shoes & hands
    for side in ("left", "right"):
        ankle, wrist = P(f"{side}_ankle"), P(f"{side}_wrist")
        if ankle:
            ax.add_patch(
                Ellipse(
                    (ankle[0], ankle[1] - height * 0.028),
                    width=height * 0.11,
                    height=height * 0.05,
                    facecolor=CLOTH_SHOES,
                    edgecolor="#252c34",
                    linewidth=0.35,
                    zorder=9,
                )
            )
        if wrist:
            ax.add_patch(
                Circle(
                    (wrist[0], wrist[1]),
                    height * 0.04,
                    facecolor=ARM_SKIN,
                    edgecolor=PAINT_SKIN_EDGE,
                    linewidth=0.3,
                    zorder=9,
                )
            )


def _draw_ghost_human_2d(ax: Axes, skeleton: Skeleton3D) -> None:
    """Tiny joint dots only — no overlapping ghost tubes."""
    from matplotlib.patches import Circle

    j = skeleton.joints
    h = _body_height(skeleton)
    r = max(0.018, h * 0.022)
    for name in (
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_wrist",
        "right_wrist",
    ):
        p = _xy(j, name)
        if p:
            ax.add_patch(
                Circle(
                    p,
                    r,
                    facecolor=COLOR_VEL,
                    edgecolor="none",
                    alpha=0.28,
                    zorder=1,
                )
            )


def _draw_motion_trail_2d(ax: Axes, trail: list[Skeleton3D]) -> None:
    xs, ys = [], []
    for sk in trail:
        h = sk.joints.get("mid_hip")
        if h:
            xs.append(h.x)
            ys.append(h.y)
    if len(xs) >= 2:
        ax.plot(xs, ys, color=COLOR_VEL, alpha=0.45, linewidth=1.4, linestyle="--", zorder=1)


def _draw_pelvis_velocity_arrow_2d(
    ax: Axes,
    skeleton: Skeleton3D,
    lin_vel: dict[str, dict[str, float]],
) -> None:
    from matplotlib.patches import FancyArrow

    joints = skeleton.joints
    anchor = joints.get("mid_hip") or joints.get("left_hip")
    vel = lin_vel.get("mid_hip") or lin_vel.get("left_hip")
    if not anchor or not vel:
        return
    vx = float(vel.get("vx", 0.0))
    vy = float(vel.get("vy", 0.0))
    speed = float(vel.get("speed", 0.0))
    if speed < 1e-5:
        return
    bh = _body_height(skeleton) or 0.5
    scale = bh * 0.32
    ax.add_patch(
        FancyArrow(
            anchor.x,
            anchor.y,
            vx * scale,
            -vy * scale,
            width=bh * 0.022,
            head_width=bh * 0.065,
            head_length=bh * 0.05,
            facecolor=COLOR_VEL,
            edgecolor="none",
            alpha=0.88,
            zorder=10,
        )
    )


def apply_body_2d_limits(
    ax: Axes,
    skeleton: Skeleton3D,
    sequence_limit: float | None = None,
    *,
    padding: float = BODY_VIEW_PADDING,
) -> None:
    """Zoom 2D axes to the painted figure (tight for small panel)."""
    if not skeleton.joints:
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        return
    xs = [j.x for j in skeleton.joints.values()]
    ys = [j.y for j in skeleton.joints.values()]
    cx = (min(xs) + max(xs)) * 0.5
    cy = (min(ys) + max(ys)) * 0.5
    span = max(max(xs) - min(xs), max(ys) - min(ys), _body_height(skeleton) * 0.55)
    r = max(0.42, span * 0.55 * padding)
    if sequence_limit is not None:
        r = min(r, max(0.42, sequence_limit))
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r            )


def _draw_selection_highlights_2d(
    ax: Axes,
    skeleton: Skeleton3D,
    highlight_joints: set[str],
) -> None:
    """Accent selected joints and their connecting bones on the 2D body view."""
    from matplotlib.patches import Circle

    from stablewalk.pose.skeleton_3d import SKELETON_3D_CONNECTIONS

    if not highlight_joints:
        return

    j = skeleton.joints
    height = _body_height(skeleton) or 0.5
    drawn_edges: set[tuple[str, str]] = set()

    for a, b in SKELETON_3D_CONNECTIONS:
        if a not in highlight_joints and b not in highlight_joints:
            continue
        edge = tuple(sorted((a, b)))
        if edge in drawn_edges:
            continue
        p0, p1 = _xy(j, a), _xy(j, b)
        if not p0 or not p1:
            continue
        drawn_edges.add(edge)
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=HIGHLIGHT_BONE,
            linewidth=2.65,
            solid_capstyle="round",
            zorder=12,
            alpha=0.95,
        )

    for name in highlight_joints:
        p = _xy(j, name)
        if not p:
            continue
        ax.add_patch(
            Circle(
                p,
                max(0.025, height * 0.035),
                facecolor=HIGHLIGHT_JOINT,
                edgecolor="#ffffff",
                linewidth=1.2,
                zorder=13,
            )
        )


def _draw_joint_trajectory_2d(
    ax: Axes,
    trajectory: list[tuple[float, float, float]],
    *,
    current_index: int | None = None,
) -> None:
    """Plot a joint path in the 2D body panel (X vs height Y)."""
    if len(trajectory) < 2:
        return
    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    ax.plot(xs, ys, color=HIGHLIGHT_BONE, linewidth=2.2, alpha=0.75, zorder=11, linestyle="--")
    ax.scatter(xs, ys, s=8, color=HIGHLIGHT_BONE, alpha=0.45, zorder=11)
    if current_index is not None and 0 <= current_index < len(trajectory):
        cx, cy, _ = trajectory[current_index]
        ax.scatter(
            [cx],
            [cy],
            s=55,
            color=HIGHLIGHT_JOINT,
            edgecolors="#ffffff",
            linewidths=1.0,
            zorder=14,
        )


def draw_painted_human_2d(
    ax: Axes,
    skeleton: Skeleton3D,
    *,
    clear: bool = True,
    frame_label: str | None = None,
    show_velocity_hint: bool = False,
    lin_vel: dict[str, dict[str, float]] | None = None,
    view_limit: float | None = None,
    motion_trail: list[Skeleton3D] | None = None,
    orient_upright: bool = True,
    highlight_joints: set[str] | None = None,
    joint_trajectory: list[tuple[float, float, float]] | None = None,
    trajectory_index: int | None = None,
) -> Skeleton3D:
    """Draw an illustrated human figure on 2D axes; returns display-oriented skeleton."""
    if clear:
        ax.cla()
        setup_body_axes(ax)

    joints = skeleton.joints
    if not joints:
        ax.text(0.5, 0.5, "No body pose", transform=ax.transAxes, ha="center", color=MUTED, fontsize=9)
        return skeleton

    display = orient_skeleton_for_display(skeleton) if orient_upright else skeleton
    display = scale_skeleton_uniform(display, DISPLAY_BODY_SCALE)
    height = _body_height(display)

    if motion_trail and len(motion_trail) >= 2:
        prev = scale_skeleton_uniform(motion_trail[-2], DISPLAY_BODY_SCALE)
        if orient_upright:
            prev = orient_skeleton_for_display(prev)
        _draw_ghost_human_2d(ax, prev)

    if motion_trail:
        oriented_trail = [
            scale_skeleton_uniform(
                orient_skeleton_for_display(s) if orient_upright else s,
                DISPLAY_BODY_SCALE,
            )
            for s in motion_trail
        ]
        _draw_motion_trail_2d(ax, oriented_trail)

    ankles = [display.joints.get(n) for n in ("left_ankle", "right_ankle") if n in display.joints]
    if ankles:
        y_floor = min(a.y for a in ankles if a) - height * 0.04
        ext = height * 0.55
        cx = display.joints["mid_hip"].x if "mid_hip" in display.joints else 0.0
        ax.plot(
            [cx - ext, cx + ext],
            [y_floor, y_floor],
            color=BORDER,
            alpha=0.5,
            linewidth=1.0,
            linestyle="--",
            zorder=0,
        )

    _draw_painted_human_2d_body(ax, display)

    if joint_trajectory:
        _draw_joint_trajectory_2d(ax, joint_trajectory, current_index=trajectory_index)
    if highlight_joints:
        _draw_selection_highlights_2d(ax, display, highlight_joints)

    if show_velocity_hint and lin_vel:
        _draw_pelvis_velocity_arrow_2d(ax, display, lin_vel)

    apply_body_2d_limits(ax, display, view_limit)

    title = frame_label or "Body (front view)"
    ax.set_title(title, color=TEXT, fontsize=9, fontweight="medium", pad=6)
    return display


def _draw_side_hints(ax: Axes, skeleton: Skeleton3D) -> None:
    """Thin L/R markers on wrists/ankles (data hint, not body paint)."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    j = skeleton.joints
    segs = []
    colors = []
    for side, col in (("left", HINT_LEFT), ("right", HINT_RIGHT)):
        for joint in (f"{side}_wrist", f"{side}_ankle"):
            p = j.get(joint)
            if not p:
                continue
            off = 0.04 * (_body_height(skeleton))
            segs.append([(p.x, p.y, p.z), (p.x + off, p.y, p.z)])
            colors.append(col)
    if segs:
        ax.add_collection3d(
            Line3DCollection(segs, colors=colors, linewidths=2.0, alpha=0.7)
        )


def draw_skeleton_3d(
    ax: Axes,
    skeleton: Skeleton3D,
    *,
    clear: bool = True,
    frame_label: str | None = None,
    show_velocity_hint: bool = False,
    lin_vel: dict[str, dict[str, float]] | None = None,
    view_limit: float | None = None,
    motion_trail: list[Skeleton3D] | None = None,
    orient_upright: bool = True,
) -> Skeleton3D:
    """Draw a painted 3D human figure; returns display-oriented skeleton."""
    if clear:
        if hasattr(ax, "_stablewalk_skeleton_camera"):
            remember_skeleton_camera(ax)
        ax.cla()
        setup_3d_axes(ax)

    joints = skeleton.joints
    if not joints:
        ax.text2D(
            0.5, 0.5, "No 3D skeleton", transform=ax.transAxes, ha="center", color=MUTED, fontsize=9
        )
        return skeleton

    display = orient_skeleton_for_display(skeleton) if orient_upright else skeleton
    display = scale_skeleton_uniform(display, DISPLAY_BODY_SCALE)
    height = _body_height(display)

    if motion_trail and len(motion_trail) >= 2:
        _draw_ghost_manikin(ax, scale_skeleton_uniform(motion_trail[-2], DISPLAY_BODY_SCALE))

    if motion_trail:
        oriented_trail = [
            scale_skeleton_uniform(
                orient_skeleton_for_display(s) if orient_upright else s,
                DISPLAY_BODY_SCALE,
            )
            for s in motion_trail
        ]
        _draw_motion_trail(ax, oriented_trail, view_limit=_skeleton_axis_limit(display))

    ankles = [display.joints.get(n) for n in ("left_ankle", "right_ankle") if n in display.joints]
    if ankles:
        y_floor = min(a.y for a in ankles if a) - 0.03
        ext = _body_height(display) * 0.55
        ax.plot(
            [-ext, ext],
            [y_floor, y_floor],
            [0.0, 0.0],
            color=BORDER,
            alpha=0.45,
            linewidth=1.0,
            linestyle="--",
        )

    _draw_human_manikin(ax, display)

    if show_velocity_hint and lin_vel:
        _draw_pelvis_velocity_arrow(ax, display, lin_vel)

    apply_display_limits(ax, display, view_limit, flat_frontal=False)

    title = frame_label or "3D gait reconstruction"
    ax.set_title(title, color=TEXT, fontsize=9.5, fontweight="medium", pad=5)
    return display


def _draw_pelvis_velocity_arrow(
    ax: Axes,
    skeleton: Skeleton3D,
    lin_vel: dict[str, dict[str, float]],
) -> None:
    """Single movement-direction arrow at pelvis (no text)."""
    joints = skeleton.joints
    anchor = joints.get("mid_hip") or joints.get("left_hip")
    vel = lin_vel.get("mid_hip") or lin_vel.get("left_hip")
    if not anchor or not vel:
        return
    vx = float(vel.get("vx", 0.0))
    vy = float(vel.get("vy", 0.0))
    speed = float(vel.get("speed", 0.0))
    if speed < 1e-5:
        return
    scale = (_body_height(skeleton) or 0.5) * 0.45
    ax.quiver(
        anchor.x, anchor.y, 0.0,
        vx * scale, -vy * scale, 0.0,
        color=COLOR_VEL,
        alpha=0.9,
        linewidth=2.0,
        arrow_length_ratio=0.25,
        length=1.0,
    )


def draw_joint_positions_chart(
    ax: Axes,
    skeleton: Skeleton3D,
    *,
    lin_vel: dict[str, dict[str, float]] | None = None,
    clear: bool = True,
) -> None:
    """Per-joint X, Y, Z (hip-centered) and linear speed for the current frame."""
    from stablewalk.pose.skeleton_3d import SKELETON_3D_JOINTS

    if clear:
        ax.cla()
        ax.set_facecolor(PANEL)

    joints = skeleton.joints
    if not joints:
        ax.text(0.5, 0.5, "No joint positions", transform=ax.transAxes, ha="center", color=MUTED)
        return

    lin_vel = lin_vel or {}
    names: list[str] = []
    xs, ys, zs, speeds = [], [], [], []
    for name in SKELETON_3D_JOINTS:
        j = joints.get(name)
        if not j:
            continue
        short = name.replace("_", " ")[:14]
        names.append(short)
        xs.append(j.x)
        ys.append(j.y)
        zs.append(j.z)
        speeds.append(float(lin_vel.get(name, {}).get("speed", 0.0)))

    if not names:
        return

    y = np.arange(len(names))
    h = 0.18
    ax.barh(y - 1.5 * h, xs, height=h, color=SIDE_LEFT, alpha=0.9, label="X")
    ax.barh(y - 0.5 * h, ys, height=h, color=COM, alpha=0.9, label="Y")
    ax.barh(y + 0.5 * h, zs, height=h, color=COLOR_TORSO, alpha=0.9, label="Z")
    ax.barh(y + 1.5 * h, speeds, height=h, color=COLOR_VEL, alpha=0.9, label="|v|")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6, color=TEXT)
    ax.set_xlabel("Hip-centered coords · speed", color=MUTED, fontsize=8)
    ax.axvline(0, color=BORDER, linewidth=0.8)
    ax.grid(True, axis="x", color=BORDER, alpha=0.35, linestyle="--")
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=6, loc="lower right", ncol=2)
    ax.set_title("Joint positions & linear velocity", color=TEXT, fontsize=9, pad=4)
    for spine in ax.spines.values():
        spine.set_color(BORDER)


def draw_dof_motion_chart(
    ax: Axes,
    frame: PoseFrame,
    *,
    ang_vel: dict[str, float] | None = None,
    clear: bool = True,
) -> None:
    """Bar chart of all DoF angles (°) and angular velocities (°/s) for this frame."""
    from stablewalk.pose.dof import DOF_LABELS, GAIT_ANGLE_FIELDS

    if clear:
        ax.cla()
        ax.set_facecolor(PANEL)

    angles = frame.joint_angles
    if not angles:
        ax.text(0.5, 0.5, "No DoF data", transform=ax.transAxes, ha="center", color=MUTED)
        return

    ang_vel = ang_vel or {}
    labels: list[str] = []
    angle_vals: list[float] = []
    omega_vals: list[float] = []

    for name in GAIT_ANGLE_FIELDS:
        val = getattr(angles, name, None)
        if val is None:
            continue
        labels.append(DOF_LABELS.get(name, name))
        angle_vals.append(float(val))
        omega_vals.append(float(ang_vel.get(name, 0.0)))

    if not labels:
        return

    y = np.arange(len(labels))
    h = 0.35
    ax.barh(y - h / 4, angle_vals, height=h / 2, color=METRIC_GLOBAL, alpha=0.85, label="Angle °")
    ax.barh(y + h / 4, omega_vals, height=h / 2, color=COLOR_VEL, alpha=0.85, label="ω °/s")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7, color=TEXT)
    ax.set_xlabel("Degrees / ° per second", color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.axvline(0, color=BORDER, linewidth=0.8)
    ax.grid(True, axis="x", color=BORDER, alpha=0.35, linestyle="--")
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=7, loc="lower right")
    ax.set_title("All DoF · angles & angular velocity", color=TEXT, fontsize=9, pad=4)
    for spine in ax.spines.values():
        spine.set_color(BORDER)


def compute_view_limit(skeletons: list[Skeleton3D], padding: float = 1.12) -> float:
    """Stable zoom cap from body height (ignores outlier joints)."""
    caps: list[float] = []
    for skel in skeletons:
        if skel.joints:
            caps.append(_skeleton_axis_limit(skel, padding=padding))
    if not caps:
        return 0.55
    return min(max(caps), 1.15)


def apply_view_limit(ax: Axes, limit: float) -> None:
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)


def draw_robot_geometry(
    ax: Axes,
    geometry,
    *,
    clear: bool = True,
    title: str = "Robot simulation",
) -> None:
    """Draw robotic walk model from RobotGeometry."""
    from stablewalk.simulation.robotic_model import ROBOT_SEGMENT_NAMES

    if clear:
        ax.cla()

    ax.set_facecolor(PANEL)
    ax.figure.patch.set_facecolor(BG)
    pts = geometry.points

    for a, b in ROBOT_SEGMENT_NAMES:
        pa, pb = pts.get(a), pts.get(b)
        if not pa or not pb:
            continue
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            [pa[2], pb[2]],
            color=COM,
            linewidth=2.5,
        )

    if pts:
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        zs = [p[2] for p in pts.values()]
        ax.scatter(xs, ys, zs, c=WARNING, s=35, edgecolors="white", linewidths=0.6)

    if "pelvis" in pts:
        p = pts["pelvis"]
        ax.scatter([p[0]], [p[1]], [p[2]], c=COM, s=55, depthshade=True)

    ax.set_xlabel("X", color=MUTED, fontsize=8)
    ax.set_ylabel("Y (forward)", color=MUTED, fontsize=8)
    ax.set_zlabel("Z (up)", color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.view_init(elev=18, azim=-72)
    ax.grid(True, color=BORDER, alpha=0.45)
    margin = 0.5
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_zlim(-0.05, 0.55)
    ax.set_title(title, color=TEXT, fontsize=9, pad=8)
