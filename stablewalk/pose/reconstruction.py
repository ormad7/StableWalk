"""
Simplified 3D skeleton reconstruction from 2D pose keypoints.

This is **not** SMPL or multi-view triangulation. See module docstring and
``ASSUMPTIONS`` for limitations.

Quick start::

    from stablewalk.pose.reconstruction import build_skeleton_sequence_3d, plot_skeleton_3d
    from stablewalk.io.pose_loader import load_pose_sequence

    seq = load_pose_sequence("data/output/poses/walk_poses.json")
    skel3d = build_skeleton_sequence_3d(seq)
    print(skel3d.positions.shape)  # (frames, joints, 3)
    plot_skeleton_3d(skel3d.skeletons[0])
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stablewalk.pose.coordinates import hip_center
from stablewalk.io.pose_loader import detected_frame_indices
from stablewalk.models.pose_data import Keypoint, PoseFrame, PoseSequence
from stablewalk.pose.skeleton_3d import (
    SKELETON_3D_CONNECTIONS,
    SKELETON_3D_JOINTS,
    Skeleton3D,
    align_skeleton_upright,
    reconstruct_skeleton_3d,
    sequence_skeleton_scale,
)

# Coordinate system: hip-centered, Y up, X left-right, Z toward camera (approximate)
COORDINATE_SYSTEM = "hip_centered_y_up_z_forward"

ASSUMPTIONS = """
Assumptions (simplified 3D from monocular 2D)
---------------------------------------------
1. Single camera, person mostly upright in the frame; walking is roughly sagittal.
2. Pelvis (mid_hip) is the body root; every frame is translated so root = (0,0,0).
3. Image x maps to skeleton X (left-right); image y (down) maps to -Y so Y points up.
4. Depth Z combines:
   - fixed anatomical layers (hips near 0, ankles back, head forward), and
   - MediaPipe relative z scaled (~0.2) — not metric depth.
5. Body height is normalized to ~1.0 per frame (nose-to-ankle span); limb lengths
   are pulled toward population-average segment ratios.
6. Optional upright alignment rotates torso to +Y for consistent matplotlib view.
7. Root trajectory is pelvis position *before* per-frame centering, in a weak
   "camera" frame (lateral offset from image center, vertical bob, heuristic Z).

Limitations vs true SMPL / mocap
---------------------------------
- No learned shape (beta) or pose (theta) parameters; no mesh surface.
- No multi-view geometry or scale from known height; depths are heuristic.
- Self-occlusion and foreshortening break joint angles and limb lengths.
- Left/right and forward/back are ambiguous from one 2D view.
- Not suitable for clinical joint moments or ground reaction force inference alone.
"""


@dataclass
class SkeletonSequence3D:
    """
    3D skeleton time series derived from 2D pose.

    Attributes:
        positions: (T, J, 3) float32 — hip-centered joint coordinates per frame.
        visibility: (T, J) float32 — joint confidence in [0, 1].
        root_trajectory: (T, 3) float32 — pelvis path before hip centering.
        joint_names: length-J tuple, fixed order matching axis 1 of ``positions``.
        frame_indices: original index in ``PoseSequence.frames`` per row.
        detected: (T,) bool — whether pose was detected that frame.
        fps: video frame rate.
        coordinate_system: semantic label for axes.
    """

    positions: np.ndarray
    visibility: np.ndarray
    root_trajectory: np.ndarray
    joint_names: tuple[str, ...] = SKELETON_3D_JOINTS
    frame_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    detected: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    fps: float = 30.0
    coordinate_system: str = COORDINATE_SYSTEM
    skeletons: list[Skeleton3D] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.positions.shape[1])

    def save_npz(self, path: str) -> None:
        """Persist arrays to compressed ``.npz``."""
        np.savez_compressed(
            path,
            positions=self.positions,
            visibility=self.visibility,
            root_trajectory=self.root_trajectory,
            joint_names=np.array(self.joint_names, dtype=object),
            frame_indices=self.frame_indices,
            detected=self.detected,
            fps=np.float64(self.fps),
            coordinate_system=self.coordinate_system,
        )


def compute_root_trajectory_point(keypoints: list[Keypoint]) -> tuple[float, float, float] | None:
    """
    Pelvis position in camera-aligned coordinates *before* hip centering.

    - X: lateral offset from image center (0.5).
    - Y: vertical offset from image midline (up positive).
    - Z: heuristic depth from hips.
    """
    center = hip_center(keypoints)
    if center is None:
        return None
    cx, cy = center
    kp_map = {kp.name: kp for kp in keypoints}
    hip_kp = kp_map.get("mid_hip") or kp_map.get("left_hip")
    z = 0.0
    if hip_kp:
        from stablewalk.pose.skeleton_3d import estimate_joint_depth

        z = estimate_joint_depth(hip_kp.name, hip_kp)
    return (float(cx - 0.5), float(-(cy - 0.5)), float(z))


def reconstruct_frame_3d(
    keypoints: list[Keypoint],
    *,
    uniform_scale: float | None = None,
    align_upright: bool = True,
) -> tuple[Skeleton3D, tuple[float, float, float] | None]:
    """
    One frame: 2D keypoints -> hip-centered ``Skeleton3D`` + root camera position.

    Returns:
        (skeleton, root_xyz) where root is before centering (for trajectory).
    """
    root = compute_root_trajectory_point(keypoints)
    skeleton = reconstruct_skeleton_3d(keypoints, scale=uniform_scale)
    if align_upright:
        skeleton = align_skeleton_upright(skeleton)
    return skeleton, root


def skeleton_to_array(
    skeleton: Skeleton3D,
    joint_names: tuple[str, ...] = SKELETON_3D_JOINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack one ``Skeleton3D`` into (J, 3) positions and (J,) visibility."""
    pos = np.full((len(joint_names), 3), np.nan, dtype=np.float32)
    vis = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        j = skeleton.joints.get(name)
        if j is None:
            continue
        pos[i] = (j.x, j.y, j.z)
        vis[i] = j.visibility
    return pos, vis


def build_skeleton_sequence_3d(
    sequence: PoseSequence,
    *,
    align_upright: bool = True,
    detected_only: bool = False,
) -> SkeletonSequence3D:
    """
    Convert a full ``PoseSequence`` to 3D arrays.

    Args:
        sequence: Output from pose estimation (2D keypoints per frame).
        align_upright: Apply torso upright rotation per frame.
        detected_only: If True, output rows only for detected frames; otherwise
            all frames (missing pose -> NaN).

    Returns:
        ``SkeletonSequence3D`` with ``positions`` shape (T, J, 3).
    """
    if detected_only:
        indices = detected_frame_indices(sequence)
        frames = [sequence.frames[i] for i in indices]
    else:
        indices = list(range(len(sequence.frames)))
        frames = sequence.frames

    detected_kps = [f.keypoints for f in frames if f.detected and f.keypoints]
    uniform_scale = sequence_skeleton_scale(detected_kps) if detected_kps else 1.0

    joint_names = SKELETON_3D_JOINTS
    t = len(frames)
    j = len(joint_names)
    positions = np.full((t, j, 3), np.nan, dtype=np.float32)
    visibility = np.zeros((t, j), dtype=np.float32)
    root_traj = np.full((t, 3), np.nan, dtype=np.float32)
    detected_mask = np.zeros(t, dtype=bool)
    skeletons: list[Skeleton3D] = []

    for ti, frame in enumerate(frames):
        detected_mask[ti] = frame.detected and bool(frame.keypoints)
        if not frame.keypoints:
            skeletons.append(Skeleton3D())
            continue

        skel, root = reconstruct_frame_3d(
            frame.keypoints,
            uniform_scale=uniform_scale,
            align_upright=align_upright,
        )
        skeletons.append(skel)
        pos, vis = skeleton_to_array(skel, joint_names)
        positions[ti] = pos
        visibility[ti] = vis
        if root is not None:
            root_traj[ti] = root

    return SkeletonSequence3D(
        positions=positions,
        visibility=visibility,
        root_trajectory=root_traj,
        joint_names=joint_names,
        frame_indices=np.array(indices, dtype=np.int32),
        detected=detected_mask,
        fps=sequence.fps,
        skeletons=skeletons,
    )


def sequence_to_numpy(
    sequence: PoseSequence,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """
    Convenience: return (positions, root_trajectory, visibility, joint_names).

    ``positions`` shape is (frames, joints, 3).
    """
    sk = build_skeleton_sequence_3d(sequence, **kwargs)  # type: ignore[arg-type]
    return sk.positions, sk.root_trajectory, sk.visibility, sk.joint_names


def plot_skeleton_3d(
    skeleton: Skeleton3D,
    *,
    ax=None,
    title: str | None = None,
    show_root: bool = True,
    color: str = "#00d4aa",
    alpha: float = 0.9,
):
    """
    Matplotlib 3D stick figure for one hip-centered skeleton.

    Returns:
        (fig, ax) — call ``plt.show()`` to display.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ax is None:
        fig = plt.figure(figsize=(6, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
        ax.cla()

    joints = skeleton.joints
    if not joints:
        ax.set_title(title or "No skeleton")
        return fig, ax

    for a, b in SKELETON_3D_CONNECTIONS:
        ja, jb = joints.get(a), joints.get(b)
        if not ja or not jb:
            continue
        ax.plot(
            [ja.x, jb.x],
            [ja.y, jb.y],
            [ja.z, jb.z],
            color=color,
            linewidth=2.0,
            alpha=alpha,
        )

    xs = [j.x for j in joints.values()]
    ys = [j.y for j in joints.values()]
    zs = [j.z for j in joints.values()]
    ax.scatter(xs, ys, zs, c=color, s=28, depthshade=True, alpha=alpha)

    if show_root and "mid_hip" in joints:
        r = joints["mid_hip"]
        ax.scatter([r.x], [r.y], [r.z], c="#ff6b6b", s=80, marker="s", label="root")

    pad = max(skeleton.max_extent(), 0.25) * 1.15
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_zlim(-pad * 0.6, pad * 0.6)
    ax.set_xlabel("X (left-right)")
    ax.set_ylabel("Y (up)")
    ax.set_zlabel("Z (depth)")
    ax.view_init(elev=15, azim=-70)
    ax.set_title(title or "3D skeleton (hip-centered)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_skeleton_sequence(
    skel_seq: SkeletonSequence3D,
    *,
    frame_step: int = 1,
    color_by_time: bool = True,
    show_root_path: bool = True,
):
    """
    Plot multiple frames as colored stick figures + optional root trajectory.

    Returns:
        (fig, ax)
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    n = skel_seq.num_frames
    if n == 0:
        return fig, ax

    cmap = plt.cm.viridis
    indices = range(0, n, max(1, frame_step))

    for k, ti in enumerate(indices):
        frac = k / max(len(indices) - 1, 1)
        color = cmap(frac) if color_by_time else "#00d4aa"
        sk = (
            skel_seq.skeletons[ti]
            if ti < len(skel_seq.skeletons)
            else _array_row_to_skeleton(skel_seq.positions[ti], skel_seq.visibility[ti])
        )
        for a, b in SKELETON_3D_CONNECTIONS:
            ja, jb = sk.joints.get(a), sk.joints.get(b)
            if not ja or not jb:
                continue
            ax.plot(
                [ja.x, jb.x],
                [ja.y, jb.y],
                [ja.z, jb.z],
                color=color,
                linewidth=1.5,
                alpha=0.65,
            )

    if show_root_path:
        rt = skel_seq.root_trajectory
        valid = ~np.isnan(rt).any(axis=1)
        if valid.any():
            ax.plot(
                rt[valid, 0],
                rt[valid, 1],
                rt[valid, 2],
                color="#ff6b6b",
                linewidth=2.0,
                linestyle="--",
                label="root trajectory",
            )

    pad = float(np.nanmax(np.abs(skel_seq.positions))) * 1.2 if skel_seq.positions.size else 0.5
    pad = max(pad, 0.35)
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_zlim(-pad * 0.6, pad * 0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y (up)")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-75)
    ax.set_title(f"3D sequence ({n} frames)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig, ax


def _array_row_to_skeleton(
    row: np.ndarray,
    vis: np.ndarray,
    joint_names: tuple[str, ...] = SKELETON_3D_JOINTS,
) -> Skeleton3D:
    from stablewalk.pose.skeleton_3d import Joint3D

    joints: dict[str, Joint3D] = {}
    for i, name in enumerate(joint_names):
        if np.isnan(row[i, 0]):
            continue
        joints[name] = Joint3D(
            name=name,
            x=float(row[i, 0]),
            y=float(row[i, 1]),
            z=float(row[i, 2]),
            visibility=float(vis[i]),
        )
    return Skeleton3D(joints=joints)


def enrich_sequence_with_3d_numpy(sequence: PoseSequence) -> SkeletonSequence3D:
    """
    Attach 3D data to each ``PoseFrame`` and return the numpy bundle.

    Updates ``frame.skeleton_3d`` and ``positions_normalized`` in place.
    """
    sk_seq = build_skeleton_sequence_3d(sequence, align_upright=True, detected_only=False)
    name_to_i = {n: i for i, n in enumerate(sk_seq.joint_names)}

    for ti, frame_idx in enumerate(sk_seq.frame_indices):
        if ti >= sk_seq.num_frames:
            break
        frame = sequence.frames[int(frame_idx)]
        sk = sk_seq.skeletons[ti] if ti < len(sk_seq.skeletons) else None
        if sk and sk.joints:
            frame.skeleton_3d = sk.to_export_dict()
            frame.positions_normalized = {
                name: {
                    "x": float(sk_seq.positions[ti, name_to_i[name], 0]),
                    "y": float(sk_seq.positions[ti, name_to_i[name], 1]),
                    "z": float(sk_seq.positions[ti, name_to_i[name], 2]),
                }
                for name in sk_seq.joint_names
                if name in name_to_i and not np.isnan(sk_seq.positions[ti, name_to_i[name], 0])
            }

    return sk_seq
