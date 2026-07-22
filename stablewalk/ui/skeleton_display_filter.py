"""
Display-only kinematic filter for the Overview walking skeleton.

Produces a motion-capture–like figure for rendering without mutating the
analysis ``GaitMotionRecording``. Raw snapshots keep driving metrics, contact,
clearance, and charts.

Features (visualization only):
  - Adaptive low-lag temporal smoothing (One-Euro style)
  - Confidence-weighted hold when landmark visibility drops
  - Soft limb-length preservation from a running median
  - Pelvis stability and shoulder leveling
  - Knee / ankle orientation cleanup (reduce impossible lateral flips)
  - Soft joint limits: knee/elbow hyperextension, shoulder flip, hip twist
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from stablewalk.models.gait_motion import JointSample, SkeletonSnapshot, Vec3
from stablewalk.models.joint_registry import ROOT_JOINT_ID

# Kinematic chains used for length locking (parent, child).
_LENGTH_EDGES: tuple[tuple[str, str], ...] = (
    (ROOT_JOINT_ID, "spine"),
    ("spine", "neck"),
    ("neck", "head"),
    ("spine", "left_shoulder"),
    ("spine", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    (ROOT_JOINT_ID, "left_hip"),
    (ROOT_JOINT_ID, "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_toe"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_toe"),
)

# Process children after parents so length constraints propagate outward.
# (Length edges already encode parent→child order.)

_MIN_VIS = 0.18
_LENGTH_HISTORY = 48
_HISTORY_MAX = 24
_LENGTH_LOCK_AFTER = 18


def _v_sub(a: Vec3, b: Vec3) -> tuple[float, float, float]:
    return (a.x - b.x, a.y - b.y, a.z - b.z)


def _v_add(a: Vec3, d: tuple[float, float, float]) -> Vec3:
    return Vec3(a.x + d[0], a.y + d[1], a.z + d[2])


def _v_scale(d: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (d[0] * s, d[1] * s, d[2] * s)


def _v_len(d: tuple[float, float, float]) -> float:
    return math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])


def _v_lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    u = max(0.0, min(1.0, t))
    return Vec3(
        a.x + (b.x - a.x) * u,
        a.y + (b.y - a.y) * u,
        a.z + (b.z - a.z) * u,
    )


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _visibility_map(snap: SkeletonSnapshot) -> dict[str, float]:
    raw = snap.metadata.get("landmark_visibility") if snap.metadata else None
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


@dataclass
class SkeletonDisplayFilter:
    """Stateful display filter — one instance per GUI session / recording."""

    # One-Euro parameters (tuned for ~25–30 fps gait video).
    # Lower min_cutoff + beta = less jitter while still tracking swing.
    min_cutoff: float = 0.78
    beta: float = 0.38
    d_cutoff: float = 1.05
    # Soft length lock blend (1 = hard FK length, 0 = free).
    length_blend: float = 0.94
    # Structural cleanup strengths.
    pelvis_smooth: float = 0.55
    shoulder_level: float = 0.68
    knee_plane: float = 0.58
    ankle_plane: float = 0.52
    # Soft joint-limit blends (display only; never invents motion).
    hyperext_blend: float = 0.72
    shoulder_flip_blend: float = 0.65
    hip_twist_blend: float = 0.48
    # Count frames used to build length medians; freeze after lock.
    _length_lock_frames: int = 0

    _pos: dict[str, Vec3] = field(default_factory=dict)
    _dx: dict[str, Vec3] = field(default_factory=dict)
    _lengths: dict[tuple[str, str], float] = field(default_factory=dict)
    _length_hist: dict[tuple[str, str], deque[float]] = field(default_factory=dict)
    _last_t: float | None = None
    _last_frame: int | None = None
    _history: deque[SkeletonSnapshot] = field(
        default_factory=lambda: deque(maxlen=_HISTORY_MAX)
    )

    def reset(self) -> None:
        self._pos.clear()
        self._dx.clear()
        self._lengths.clear()
        self._length_hist.clear()
        self._last_t = None
        self._last_frame = None
        self._history.clear()
        self._length_lock_frames = 0

    def filter_snapshot(self, snap: SkeletonSnapshot) -> SkeletonSnapshot:
        """Return a filtered *copy* suitable for ``draw_gait_skeleton``."""
        if not snap.joints:
            return snap

        # Scrub / seek: reseed so we do not smear across distant frames.
        if self._last_frame is not None:
            jump = int(snap.frame_index) - int(self._last_frame)
            if jump < 0 or jump > 3:
                self._reseed(snap)
                filtered = self._clone(snap, self._pos)
                self._history.append(filtered)
                self._last_frame = snap.frame_index
                self._last_t = snap.time_s
                return filtered

        dt = self._delta_t(snap.time_s)
        vis = _visibility_map(snap)
        self._update_length_priors(snap)

        smoothed: dict[str, Vec3] = {}
        for jid, sample in snap.joints.items():
            conf = vis.get(jid, 1.0)
            if conf < _MIN_VIS and jid in self._pos:
                # Hold previous filtered pose instead of jumping.
                smoothed[jid] = self._pos[jid]
                continue
            prev = self._pos.get(jid)
            if prev is None:
                smoothed[jid] = sample.position
                self._dx[jid] = Vec3(0.0, 0.0, 0.0)
                continue
            # Stronger smoothing when confidence is weak; still tracks motion.
            conf_w = max(0.25, min(1.0, conf))
            filtered = self._one_euro(jid, prev, sample.position, dt, conf_w)
            # Blend toward hold when confidence is mid-low.
            if conf_w < 0.85:
                hold = 1.0 - conf_w
                filtered = _v_lerp(filtered, prev, hold * 0.55)
            smoothed[jid] = filtered

        smoothed = self._stabilize_pelvis(smoothed)
        smoothed = self._level_shoulders(smoothed)
        smoothed = self._constrain_shoulder_flip(smoothed)
        smoothed = self._constrain_hip_twist(smoothed)
        smoothed = self._constrain_lengths(smoothed)
        smoothed = self._stabilize_knees(smoothed)
        smoothed = self._constrain_hinge_hyperextension(smoothed)
        smoothed = self._stabilize_ankles(smoothed)
        # Second length pass after orientation cleanup.
        smoothed = self._constrain_lengths(smoothed)
        # Re-level after length locks so shoulders stay even.
        smoothed = self._level_shoulders(smoothed)
        smoothed = self._constrain_shoulder_flip(smoothed)

        self._pos = dict(smoothed)
        self._last_t = snap.time_s
        self._last_frame = snap.frame_index
        filtered_snap = self._clone(snap, smoothed)
        self._history.append(filtered_snap)
        return filtered_snap

    def ghost_snapshots(self, count: int) -> list[SkeletonSnapshot]:
        """Previous filtered poses for motion echo (oldest → newest)."""
        if count <= 0 or len(self._history) < 2:
            return []
        prior = list(self._history)[:-1]
        return prior[-count:]

    # ------------------------------------------------------------------ internals

    def _reseed(self, snap: SkeletonSnapshot) -> None:
        self._pos = {jid: sample.position for jid, sample in snap.joints.items()}
        self._dx = {jid: Vec3(0.0, 0.0, 0.0) for jid in snap.joints}
        self._update_length_priors(snap)
        # Keep length priors; drop temporal history so ghosts do not flash.
        self._history.clear()

    def _delta_t(self, time_s: float) -> float:
        if self._last_t is None:
            return 1.0 / 30.0
        dt = float(time_s) - float(self._last_t)
        if dt <= 1e-4:
            return 1.0 / 30.0
        return max(1.0 / 120.0, min(dt, 0.12))

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / max(dt, 1e-4))

    def _one_euro(
        self,
        jid: str,
        prev: Vec3,
        raw: Vec3,
        dt: float,
        conf: float,
    ) -> Vec3:
        dx_prev = self._dx.get(jid, Vec3(0.0, 0.0, 0.0))
        # Derivative estimate.
        inv_dt = 1.0 / max(dt, 1e-4)
        dx_raw = Vec3(
            (raw.x - prev.x) * inv_dt,
            (raw.y - prev.y) * inv_dt,
            (raw.z - prev.z) * inv_dt,
        )
        a_d = self._alpha(self.d_cutoff, dt)
        dx = Vec3(
            dx_prev.x + a_d * (dx_raw.x - dx_prev.x),
            dx_prev.y + a_d * (dx_raw.y - dx_prev.y),
            dx_prev.z + a_d * (dx_raw.z - dx_prev.z),
        )
        self._dx[jid] = dx
        speed = math.sqrt(dx.x * dx.x + dx.y * dx.y + dx.z * dx.z)
        # Low cutoff when slow (kill jitter); rises with speed (no lag).
        cutoff = self.min_cutoff + self.beta * speed
        # Confidence scales effective cutoff downward (more smooth when unsure).
        cutoff *= 0.55 + 0.45 * conf
        a = self._alpha(cutoff, dt)
        return Vec3(
            prev.x + a * (raw.x - prev.x),
            prev.y + a * (raw.y - prev.y),
            prev.z + a * (raw.z - prev.z),
        )

    def _update_length_priors(self, snap: SkeletonSnapshot) -> None:
        # After lock-in, keep median limb lengths fixed so jitter cannot stretch bones.
        if self._length_lock_frames >= _LENGTH_LOCK_AFTER and self._lengths:
            return
        vis = _visibility_map(snap)
        updated = 0
        for parent, child in _LENGTH_EDGES:
            pa = snap.joints.get(parent)
            ch = snap.joints.get(child)
            if pa is None or ch is None:
                continue
            if min(vis.get(parent, 1.0), vis.get(child, 1.0)) < 0.45:
                continue
            length = _v_len(_v_sub(ch.position, pa.position))
            if length < 1e-5:
                continue
            hist = self._length_hist.get((parent, child))
            if hist is None:
                hist = deque(maxlen=_LENGTH_HISTORY)
                self._length_hist[(parent, child)] = hist
            hist.append(length)
            self._lengths[(parent, child)] = _median(list(hist))
            updated += 1
        if updated:
            self._length_lock_frames += 1

    def _stabilize_pelvis(
        self, positions: dict[str, Vec3]
    ) -> dict[str, Vec3]:
        lh = positions.get("left_hip")
        rh = positions.get("right_hip")
        root = positions.get(ROOT_JOINT_ID)
        if lh is None or rh is None:
            return positions
        mid = Vec3(0.5 * (lh.x + rh.x), 0.5 * (lh.y + rh.y), 0.5 * (lh.z + rh.z))
        if root is None:
            root = mid
        # Softly pull root toward hip midpoint; damp high-frequency bounce.
        blend = self.pelvis_smooth
        new_root = _v_lerp(root, mid, blend)
        # Preserve measured hip width while centering on the filtered root.
        half = _v_scale(_v_sub(rh, lh), 0.5)
        width = _v_len(half)
        if width > 1e-6:
            # Keep Y of each hip mostly as measured (stance), stabilize XZ toward midline.
            target_lh = Vec3(new_root.x - half[0], lh.y * (1.0 - 0.25 * blend) + new_root.y * (0.25 * blend), new_root.z - half[2])
            target_rh = Vec3(new_root.x + half[0], rh.y * (1.0 - 0.25 * blend) + new_root.y * (0.25 * blend), new_root.z + half[2])
            positions["left_hip"] = _v_lerp(lh, target_lh, blend * 0.85)
            positions["right_hip"] = _v_lerp(rh, target_rh, blend * 0.85)
        positions[ROOT_JOINT_ID] = new_root
        return positions

    def _level_shoulders(
        self, positions: dict[str, Vec3]
    ) -> dict[str, Vec3]:
        ls = positions.get("left_shoulder")
        rs = positions.get("right_shoulder")
        if ls is None or rs is None:
            return positions
        mid_y = 0.5 * (ls.y + rs.y)
        mid_z = 0.5 * (ls.z + rs.z)
        b = self.shoulder_level
        positions["left_shoulder"] = Vec3(
            ls.x,
            ls.y + (mid_y - ls.y) * b,
            ls.z + (mid_z - ls.z) * (b * 0.35),
        )
        positions["right_shoulder"] = Vec3(
            rs.x,
            rs.y + (mid_y - rs.y) * b,
            rs.z + (mid_z - rs.z) * (b * 0.35),
        )
        return positions

    def _constrain_lengths(
        self, positions: dict[str, Vec3]
    ) -> dict[str, Vec3]:
        """Lock child distance from parent to the running median length."""
        blend = self.length_blend
        hard = self._length_lock_frames >= _LENGTH_LOCK_AFTER
        for parent, child in _LENGTH_EDGES:
            if parent not in positions or child not in positions:
                continue
            target = self._lengths.get((parent, child))
            if target is None or target < 1e-5:
                continue
            p = positions[parent]
            c = positions[child]
            delta = _v_sub(c, p)
            current = _v_len(delta)
            if current < 1e-6:
                continue
            if hard:
                scale = target / current
                positions[child] = _v_add(p, _v_scale(delta, scale))
                continue
            desired = max(target * 0.88, min(target * 1.12, current))
            desired = target * blend + desired * (1.0 - blend)
            scale = desired / current
            scale = 1.0 + blend * (scale - 1.0)
            positions[child] = _v_add(p, _v_scale(delta, scale))
        return positions

    def _stabilize_knees(
        self, positions: dict[str, Vec3]
    ) -> dict[str, Vec3]:
        """Pull each knee toward the hip–ankle axis (reduces lateral flip)."""
        b = self.knee_plane
        for side in ("left", "right"):
            hip = positions.get(f"{side}_hip")
            knee = positions.get(f"{side}_knee")
            ankle = positions.get(f"{side}_ankle")
            if hip is None or knee is None or ankle is None:
                continue
            axis = _v_sub(ankle, hip)
            axis_len = _v_len(axis)
            if axis_len < 1e-6:
                continue
            ux, uy, uz = _v_scale(axis, 1.0 / axis_len)
            hk = _v_sub(knee, hip)
            t = max(0.0, min(1.0, hk[0] * ux + hk[1] * uy + hk[2] * uz))
            on_axis = Vec3(hip.x + ux * t, hip.y + uy * t, hip.z + uz * t)
            # Keep a fraction of the measured offset so knees stay in front of the leg.
            offset = _v_sub(knee, on_axis)
            reduced = _v_add(on_axis, _v_scale(offset, 1.0 - b))
            # Restore thigh length toward the hip.
            thigh = self._lengths.get((f"{side}_hip", f"{side}_knee"))
            if thigh and thigh > 1e-5:
                d = _v_sub(reduced, hip)
                L = _v_len(d)
                if L > 1e-6:
                    reduced = _v_add(hip, _v_scale(d, thigh / L))
            positions[f"{side}_knee"] = _v_lerp(knee, reduced, b)
        return positions

    def _stabilize_ankles(
        self, positions: dict[str, Vec3]
    ) -> dict[str, Vec3]:
        """Keep heel→toe roughly orthogonal to the shank and length-stable."""
        b = self.ankle_plane
        for side in ("left", "right"):
            knee = positions.get(f"{side}_knee")
            ankle = positions.get(f"{side}_ankle")
            heel = positions.get(f"{side}_heel")
            toe = positions.get(f"{side}_toe")
            if knee is None or ankle is None:
                continue
            shank = _v_sub(ankle, knee)
            shank_len = _v_len(shank)
            if shank_len < 1e-6:
                continue
            sx, sy, sz = _v_scale(shank, 1.0 / shank_len)

            if heel is not None and toe is not None:
                foot = _v_sub(toe, heel)
                foot_len = _v_len(foot)
                if foot_len > 1e-6:
                    # Remove shank-parallel component from foot direction.
                    parallel = foot[0] * sx + foot[1] * sy + foot[2] * sz
                    flat = (
                        foot[0] - sx * parallel,
                        foot[1] - sy * parallel,
                        foot[2] - sz * parallel,
                    )
                    flat_len = _v_len(flat)
                    if flat_len > 1e-6:
                        target_len = self._lengths.get(
                            (f"{side}_heel", f"{side}_toe"), foot_len
                        )
                        flat_u = _v_scale(flat, target_len / flat_len)
                        mid = Vec3(
                            0.5 * (heel.x + toe.x),
                            0.5 * (heel.y + toe.y),
                            0.5 * (heel.z + toe.z),
                        )
                        # Keep ankle near the foot midpoint laterally.
                        new_heel = Vec3(
                            mid.x - flat_u[0] * 0.5,
                            heel.y,
                            mid.z - flat_u[2] * 0.5,
                        )
                        new_toe = Vec3(
                            mid.x + flat_u[0] * 0.5,
                            toe.y,
                            mid.z + flat_u[2] * 0.5,
                        )
                        # Prefer measured vertical so soles stay near the ground.
                        new_heel = Vec3(new_heel.x, heel.y, new_heel.z)
                        new_toe = Vec3(new_toe.x, toe.y, new_toe.z)
                        positions[f"{side}_heel"] = _v_lerp(heel, new_heel, b)
                        positions[f"{side}_toe"] = _v_lerp(toe, new_toe, b)

            # Softly keep ankle under the knee along the shank length.
            shank_target = self._lengths.get((f"{side}_knee", f"{side}_ankle"))
            if shank_target and shank_target > 1e-5:
                desired = _v_add(knee, _v_scale((sx, sy, sz), shank_target))
                positions[f"{side}_ankle"] = _v_lerp(ankle, desired, b * 0.65)
        return positions

    def _constrain_hinge_hyperextension(
        self, positions: dict[str, Vec3]
    ) -> dict[str, Vec3]:
        """Clamp knee/elbow past straight (display) without inventing swing."""
        b = self.hyperext_blend
        chains = (
            ("left_hip", "left_knee", "left_ankle", ("left_hip", "left_knee")),
            ("right_hip", "right_knee", "right_ankle", ("right_hip", "right_knee")),
            ("left_shoulder", "left_elbow", "left_wrist", ("left_shoulder", "left_elbow")),
            ("right_shoulder", "right_elbow", "right_wrist", ("right_shoulder", "right_elbow")),
        )
        for proximal, mid, distal, length_key in chains:
            p = positions.get(proximal)
            m = positions.get(mid)
            d = positions.get(distal)
            if p is None or m is None or d is None:
                continue
            v1 = _v_sub(p, m)
            v2 = _v_sub(d, m)
            n1 = _v_len(v1)
            n2 = _v_len(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos_a = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
            cos_a = max(-1.0, min(1.0, cos_a))
            # Interior angle at the hinge; near 180° = straight, > ~178° with
            # the mid behind the proximal–distal axis = hyperextension.
            interior = math.degrees(math.acos(cos_a))
            axis = _v_sub(d, p)
            axis_len = _v_len(axis)
            if axis_len < 1e-6:
                continue
            ux, uy, uz = _v_scale(axis, 1.0 / axis_len)
            pm = _v_sub(m, p)
            t = max(0.0, min(1.0, pm[0] * ux + pm[1] * uy + pm[2] * uz))
            on_axis = Vec3(p.x + ux * t, p.y + uy * t, p.z + uz * t)
            offset = _v_sub(m, on_axis)
            off_len = _v_len(offset)
            # Hyperextension / locked-straight: nearly collinear hinge.
            if interior < 172.0 and off_len > 1e-5:
                continue
            # Prefer existing off-axis direction; if collinear, nudge anterior.
            if off_len < 1e-5:
                lh = positions.get("left_hip")
                rh = positions.get("right_hip")
                if lh is not None and rh is not None:
                    lat = _v_sub(rh, lh)
                else:
                    lat = (1.0, 0.0, 0.0)
                # Forward ≈ up(Y) × lateral in the lab frame.
                fwd = (-lat[2], 0.0, lat[0])
                fl = _v_len(fwd)
                if fl < 1e-6:
                    continue
                offset = _v_scale(fwd, axis_len * 0.04 / fl)
            elif interior >= 176.0:
                # Softly restore a few degrees of flexion off the axis.
                scale = max(axis_len * 0.03 / off_len, 1.0)
                offset = _v_scale(offset, scale)
            target = _v_add(on_axis, offset)
            # Restore segment length to the proximal joint.
            target_len = self._lengths.get(length_key)
            if target_len and target_len > 1e-5:
                delta = _v_sub(target, p)
                L = _v_len(delta)
                if L > 1e-6:
                    target = _v_add(p, _v_scale(delta, target_len / L))
            positions[mid] = _v_lerp(m, target, b)
        return positions

    def _constrain_shoulder_flip(
        self, positions: dict[str, Vec3]
    ) -> dict[str, Vec3]:
        """Keep left/right shoulders on their tracked sides of the midline."""
        ls = positions.get("left_shoulder")
        rs = positions.get("right_shoulder")
        lh = positions.get("left_hip")
        rh = positions.get("right_hip")
        if ls is None or rs is None:
            return positions
        b = self.shoulder_flip_blend
        mid = Vec3(0.5 * (ls.x + rs.x), 0.5 * (ls.y + rs.y), 0.5 * (ls.z + rs.z))
        width = _v_len(_v_sub(rs, ls))
        if width < 1e-6:
            return positions
        half_w = width * 0.5
        # Body-left direction from the hip girdle (stable even when shoulders flip).
        if lh is not None and rh is not None:
            hip_mid = Vec3(
                0.5 * (lh.x + rh.x),
                0.5 * (lh.y + rh.y),
                0.5 * (lh.z + rh.z),
            )
            left_dir = _v_sub(lh, hip_mid)
        else:
            left_dir = _v_sub(ls, mid)
        # Prefer horizontal (XZ) left; fall back to full 3D.
        left_hz = (left_dir[0], 0.0, left_dir[2])
        left_len = _v_len(left_hz)
        if left_len < 1e-6:
            left_hz = left_dir
            left_len = _v_len(left_hz)
        if left_len < 1e-6:
            return positions
        left_u = _v_scale(left_hz, 1.0 / left_len)
        # Left shoulder should lie on the +left_u half-space.
        to_ls = _v_sub(ls, mid)
        proj = to_ls[0] * left_u[0] + to_ls[1] * left_u[1] + to_ls[2] * left_u[2]
        if proj >= 0.0:
            return positions
        target_ls = Vec3(
            mid.x + left_u[0] * half_w,
            ls.y,
            mid.z + left_u[2] * half_w,
        )
        target_rs = Vec3(
            mid.x - left_u[0] * half_w,
            rs.y,
            mid.z - left_u[2] * half_w,
        )
        positions["left_shoulder"] = _v_lerp(ls, target_ls, b)
        positions["right_shoulder"] = _v_lerp(rs, target_rs, b)
        return positions

    def _constrain_hip_twist(
        self, positions: dict[str, Vec3]
    ) -> dict[str, Vec3]:
        """Limit impossible pelvis yaw relative to the shoulder girdle."""
        lh = positions.get("left_hip")
        rh = positions.get("right_hip")
        ls = positions.get("left_shoulder")
        rs = positions.get("right_shoulder")
        if lh is None or rh is None or ls is None or rs is None:
            return positions
        b = self.hip_twist_blend
        hip = _v_sub(rh, lh)
        sh = _v_sub(rs, ls)
        hip_xz = (hip[0], hip[2])
        sh_xz = (sh[0], sh[2])
        hn = math.hypot(*hip_xz)
        sn = math.hypot(*sh_xz)
        if hn < 1e-6 or sn < 1e-6:
            return positions
        # Signed yaw difference in the horizontal plane.
        cross = hip_xz[0] * sh_xz[1] - hip_xz[1] * sh_xz[0]
        dot = hip_xz[0] * sh_xz[0] + hip_xz[1] * sh_xz[1]
        yaw = abs(math.degrees(math.atan2(cross, dot)))
        # Soft clamp beyond ~55° relative twist (display stability).
        if yaw < 55.0:
            return positions
        # Blend hip orientation toward shoulder orientation, keep hip width.
        target_dir = (sh_xz[0] / sn, sh_xz[1] / sn)
        half_w = hn * 0.5
        mid = Vec3(0.5 * (lh.x + rh.x), 0.5 * (lh.y + rh.y), 0.5 * (lh.z + rh.z))
        target_lh = Vec3(
            mid.x - target_dir[0] * half_w,
            lh.y,
            mid.z - target_dir[1] * half_w,
        )
        target_rh = Vec3(
            mid.x + target_dir[0] * half_w,
            rh.y,
            mid.z + target_dir[1] * half_w,
        )
        # Strength scales with how far past the limit.
        strength = min(1.0, (yaw - 55.0) / 40.0) * b
        positions["left_hip"] = _v_lerp(lh, target_lh, strength)
        positions["right_hip"] = _v_lerp(rh, target_rh, strength)
        return positions

    @staticmethod
    def _clone(snap: SkeletonSnapshot, positions: dict[str, Vec3]) -> SkeletonSnapshot:
        joints: dict[str, JointSample] = {}
        for jid, sample in snap.joints.items():
            pos = positions.get(jid, sample.position)
            joints[jid] = JointSample(
                joint_id=sample.joint_id,
                position=pos,
                parent_id=sample.parent_id,
                angle_deg=sample.angle_deg,
                velocity=sample.velocity,
                velocity_vector=sample.velocity_vector,
            )
        meta = dict(snap.metadata or {})
        meta["display_filtered"] = True
        return SkeletonSnapshot(
            frame_index=snap.frame_index,
            time_s=snap.time_s,
            joints=joints,
            dofs=dict(snap.dofs),
            metadata=meta,
        )


def ensure_display_filter(gui: Any) -> SkeletonDisplayFilter:
    """Get or create the session display filter on the GUI object."""
    filt = getattr(gui, "_skeleton_display_filter", None)
    if not isinstance(filt, SkeletonDisplayFilter):
        filt = SkeletonDisplayFilter()
        gui._skeleton_display_filter = filt
    return filt


__all__ = [
    "SkeletonDisplayFilter",
    "ensure_display_filter",
]
