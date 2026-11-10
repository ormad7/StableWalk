"""Four Motion Signals: joint mapping, 3D tilt, timestamps, and 2×2 layout."""

from __future__ import annotations

import math

import numpy as np

from stablewalk.analysis.four_gait_signals import (
    FOUR_GAIT_CHANNEL_IDS,
    FOUR_GAIT_SHORT_LABELS,
    analyze_four_gait_signals,
)
from stablewalk.analysis.spectral.channels import (
    FOUR_GAIT_ANGLE_UNITS,
    FOUR_GAIT_ANGLE_YLABEL,
    FOUR_GAIT_JOINTS,
    FOUR_GAIT_PANEL_TITLES,
    SEGMENT_SPECS,
    four_gait_joints,
)
from stablewalk.analysis.spectral.series import segment_sagittal_tilt_deg
from stablewalk.models.gait_motion import (
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.models.joint_registry import JOINT_IDS, JOINT_PARENTS, ROOT_JOINT_ID


class _Pos:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


def _pose(
    *,
    left_hip: tuple[float, float, float] = (-0.10, 0.00, 0.00),
    right_hip: tuple[float, float, float] = (0.10, 0.00, 0.00),
    left_knee: tuple[float, float, float] = (-0.10, -0.40, 0.00),
    right_knee: tuple[float, float, float] = (0.10, -0.40, 0.00),
    left_ankle: tuple[float, float, float] = (-0.10, -0.82, 0.00),
    right_ankle: tuple[float, float, float] = (0.10, -0.82, 0.00),
    time_s: float = 0.0,
    frame_index: int = 0,
) -> SkeletonSnapshot:
    coords = {
        "left_hip": left_hip,
        "right_hip": right_hip,
        "left_knee": left_knee,
        "right_knee": right_knee,
        "left_ankle": left_ankle,
        "right_ankle": right_ankle,
    }
    joints = {}
    for jid in (*JOINT_IDS, ROOT_JOINT_ID):
        xyz = coords.get(jid, (0.0, 0.2, 0.0))
        joints[jid] = JointSample(
            joint_id=jid,
            position=Vec3(*xyz),
            parent_id=JOINT_PARENTS.get(jid),
        )
    return SkeletonSnapshot(
        frame_index=frame_index,
        time_s=time_s,
        joints=joints,
        dofs={},
        metadata={
            "detected": True,
            "landmark_visibility": {jid: 0.95 for jid in (*JOINT_IDS, ROOT_JOINT_ID)},
        },
    )


def _recording(snaps: list[SkeletonSnapshot], *, fps: float = 50.0) -> GaitMotionRecording:
    return GaitMotionRecording(
        source="four-motion-test",
        fps=fps,
        snapshots=snaps,
        source_kind="mock",
        coordinate_system="sw_canonical_y_up",
    )


def test_four_gait_channels_use_the_correct_joints() -> None:
    expected = {
        "seg:left_thigh": ("left_hip", "left_knee"),
        "seg:right_thigh": ("right_hip", "right_knee"),
        "seg:left_shank": ("left_knee", "left_ankle"),
        "seg:right_shank": ("right_knee", "right_ankle"),
    }
    assert FOUR_GAIT_CHANNEL_IDS == (
        "seg:left_thigh",
        "seg:right_thigh",
        "seg:left_shank",
        "seg:right_shank",
    )
    assert FOUR_GAIT_JOINTS == expected
    for spec in SEGMENT_SPECS:
        assert four_gait_joints(spec.channel_id) == (
            spec.proximal_joint_id,
            spec.distal_joint_id,
        )
        assert spec.units == FOUR_GAIT_ANGLE_UNITS
        assert spec.source == "segment_sagittal"
        assert spec.kind == "angle_deg"


def test_each_signal_changes_only_when_its_own_joints_move() -> None:
    hanging = _pose()
    # Move the unique endpoint: hip for thigh (knee is shared with shank),
    # ankle for shank (knee is shared with thigh).
    moved = {
        "seg:left_thigh": _pose(left_hip=(-0.10, 0.00, -0.12), time_s=0.02, frame_index=1),
        "seg:right_thigh": _pose(right_hip=(0.10, 0.00, -0.12), time_s=0.02, frame_index=1),
        "seg:left_shank": _pose(left_ankle=(-0.10, -0.82, 0.12), time_s=0.02, frame_index=1),
        "seg:right_shank": _pose(right_ankle=(0.10, -0.82, 0.12), time_s=0.02, frame_index=1),
    }
    hanging.time_s = 0.0
    base = analyze_four_gait_signals(_recording([hanging]))
    for cid, snap in moved.items():
        report = analyze_four_gait_signals(_recording([hanging, snap], fps=50.0))
        for other in FOUR_GAIT_CHANNEL_IDS:
            series = report.raw[other]
            assert series.size == 2
            if other == cid:
                assert abs(float(series[1]) - float(series[0])) > 5.0
            else:
                assert abs(float(series[1]) - float(series[0])) < 1e-6
        assert abs(float(base.raw[cid][0])) < 1e-6


def test_right_joints_do_not_affect_left_signals() -> None:
    base = _pose()
    right_only = _pose(
        right_knee=(0.10, -0.40, 0.18),
        right_ankle=(0.10, -0.82, 0.30),
        time_s=0.02,
        frame_index=1,
    )
    report = analyze_four_gait_signals(_recording([base, right_only]))
    assert abs(float(report.raw["seg:left_thigh"][1] - report.raw["seg:left_thigh"][0])) < 1e-6
    assert abs(float(report.raw["seg:left_shank"][1] - report.raw["seg:left_shank"][0])) < 1e-6
    assert float(report.raw["seg:right_thigh"][1]) > 10.0
    assert float(report.raw["seg:right_shank"][1]) > 5.0


def test_all_four_share_the_same_angle_definition() -> None:
    assert all("hip → knee" in title or "knee → ankle" in title for title in FOUR_GAIT_PANEL_TITLES)
    assert FOUR_GAIT_ANGLE_YLABEL == "Sagittal tilt (deg)"
    hanging = _Pos(0.0, 0.0, 0.0)
    down = _Pos(0.0, -0.4, 0.0)
    anterior = _Pos(0.0, -0.4, 0.2)
    assert abs(segment_sagittal_tilt_deg(hanging, down)) < 1e-9
    assert segment_sagittal_tilt_deg(hanging, anterior) > 0.0
    snaps = [
        _pose(
            left_knee=(-0.10, -0.40, 0.10),
            right_knee=(0.10, -0.40, 0.10),
            left_ankle=(-0.10, -0.82, 0.22),
            right_ankle=(0.10, -0.82, 0.22),
        )
    ]
    report = analyze_four_gait_signals(_recording(snaps))
    assert report.angle_units == "deg"
    assert report.angle_ylabel == FOUR_GAIT_ANGLE_YLABEL
    values = [float(report.raw[cid][0]) for cid in FOUR_GAIT_CHANNEL_IDS]
    assert all(v > 0.0 for v in values)
    assert "deg" in report.angle_definition.lower()
    assert "hip→knee" in report.angle_definition.replace(" ", "")
    assert FOUR_GAIT_SHORT_LABELS == (
        "Left Thigh",
        "Right Thigh",
        "Left Shank",
        "Right Shank",
    )


def test_degenerate_and_mediolateral_segments_are_nan() -> None:
    assert math.isnan(segment_sagittal_tilt_deg(_Pos(0.0, 0.0, 0.0), _Pos(0.0, 0.0, 0.0)))
    assert math.isnan(segment_sagittal_tilt_deg(_Pos(0.0, 0.0, 0.0), _Pos(0.4, 0.0, 0.0)))
    collapsed = _pose(left_knee=(-0.10, 0.00, 0.00))
    report = analyze_four_gait_signals(_recording([collapsed]))
    assert not np.isfinite(report.raw["seg:left_thigh"][0])
    assert np.isfinite(report.raw["seg:right_thigh"][0])


def test_signals_use_full_frame_timestamps() -> None:
    fps = 40.0
    snaps = [_pose(time_s=i / fps, frame_index=i) for i in range(20)]
    report = analyze_four_gait_signals(_recording(snaps, fps=fps))
    assert abs(report.fps - fps) < 1e-12
    assert report.times_s.size == 20
    dt = float(np.median(np.diff(report.times_s)))
    assert abs(dt - 1.0 / fps) < 1e-12
    for cid in FOUR_GAIT_CHANNEL_IDS:
        assert report.raw[cid].size == report.times_s.size
        assert report.filtered[cid].size == report.times_s.size


def test_swapped_hip_laterality_is_warned() -> None:
    snaps = [
        _pose(
            left_hip=(0.12, 0.0, 0.0),
            right_hip=(-0.12, 0.0, 0.0),
            time_s=i / 50.0,
            frame_index=i,
        )
        for i in range(8)
    ]
    report = analyze_four_gait_signals(_recording(snaps))
    assert any("left/right" in w.lower() for w in report.warnings)


def test_facing_camera_laterality_is_not_warned() -> None:
    snaps = [
        _pose(
            left_hip=(0.12, 0.0, 0.0),
            right_hip=(-0.12, 0.0, 0.0),
            left_knee=(0.12, -0.40, 0.0),
            right_knee=(-0.12, -0.40, 0.0),
            left_ankle=(0.12, -0.82, 0.0),
            right_ankle=(-0.12, -0.82, 0.0),
            time_s=i / 50.0,
            frame_index=i,
        )
        for i in range(8)
    ]
    for snap in snaps:
        head = snap.joints.get("head")
        if head is not None:
            snap.joints["head"] = JointSample(
                joint_id="head",
                position=Vec3(0.0, 0.43, 0.15),
                parent_id=head.parent_id,
            )
    report = analyze_four_gait_signals(_recording(snaps))
    assert not any("disagree" in w.lower() for w in report.warnings)


def test_four_motion_layout_leaves_room_for_titles() -> None:
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.four_gait_charts import layout_four_gait_figure

    fig = Figure(figsize=(7.2, 5.0), dpi=100)
    axes = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
    layout_four_gait_figure(fig)
    for ax in axes:
        pos = ax.get_position()
        assert pos.y1 <= 0.92
        assert pos.y0 >= 0.12
        assert pos.x0 >= 0.08


def test_time_figure_is_a_2x2_with_shared_units() -> None:
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.four_gait_charts import (
        build_signal_axes,
        render_four_gait_figure,
    )

    snaps = [
        _pose(
            left_knee=(-0.10, -0.40, 0.08 * math.sin(2 * math.pi * i / 20)),
            right_knee=(0.10, -0.40, 0.08 * math.sin(2 * math.pi * i / 20 + math.pi)),
            time_s=i / 50.0,
            frame_index=i,
        )
        for i in range(20)
    ]
    report = analyze_four_gait_signals(_recording(snaps))
    fig = Figure(figsize=(7.2, 6.0))
    axes = build_signal_axes(fig, combined=False)
    assert len(axes) == 4
    render_four_gait_figure(fig, axes, report, "filtered", combined=False)
    for ax, title in zip(axes, FOUR_GAIT_PANEL_TITLES):
        assert ax.get_ylabel() == FOUR_GAIT_ANGLE_YLABEL
        assert title.split()[0] in (ax.get_title() or "")
    assert axes[0].get_xlabel() in {"", None}
    assert "Time" in (axes[2].get_xlabel() or "")
    assert "Time" in (axes[3].get_xlabel() or "")


def test_equivalent_segments_share_y_limits() -> None:
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.four_gait_charts import (
        build_signal_axes,
        pair_limits,
        render_four_gait_figure,
    )

    snaps = [
        _pose(
            left_knee=(-0.10, -0.40, 0.02 * math.sin(2 * math.pi * i / 12)),
            right_knee=(0.10, -0.40, 0.02 * math.sin(2 * math.pi * i / 12 + math.pi)),
            left_ankle=(-0.10, -0.82, 0.28 * math.sin(2 * math.pi * i / 12)),
            right_ankle=(0.10, -0.82, 0.28 * math.sin(2 * math.pi * i / 12 + math.pi)),
            time_s=i / 50.0,
            frame_index=i,
        )
        for i in range(24)
    ]
    report = analyze_four_gait_signals(_recording(snaps))
    xlim, thigh_ylim, shank_ylim = pair_limits(report, "filtered")
    assert xlim[1] > xlim[0]
    assert thigh_ylim == pair_limits(report, "filtered")[1]
    fig = Figure(figsize=(8.0, 6.4))
    axes = build_signal_axes(fig, combined=False)
    render_four_gait_figure(fig, axes, report, "filtered")
    assert axes[0].get_ylim() == axes[1].get_ylim()
    assert axes[2].get_ylim() == axes[3].get_ylim()
    thigh_span = axes[1].get_ylim()[1] - axes[1].get_ylim()[0]
    shank_span = axes[3].get_ylim()[1] - axes[3].get_ylim()[0]
    assert shank_span > thigh_span * 1.3
    assert shank_ylim[1] - shank_ylim[0] > thigh_ylim[1] - thigh_ylim[0]


def test_lr_overlay_compares_matching_segments() -> None:
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.four_gait_charts import (
        build_signal_axes,
        render_four_gait_figure,
    )

    snaps = [
        _pose(
            left_knee=(-0.10, -0.40, 0.10 * math.sin(2 * math.pi * i / 16)),
            right_knee=(0.10, -0.40, 0.10 * math.sin(2 * math.pi * i / 16 + math.pi)),
            left_ankle=(-0.10, -0.82, 0.16 * math.sin(2 * math.pi * i / 16)),
            right_ankle=(0.10, -0.82, 0.16 * math.sin(2 * math.pi * i / 16 + math.pi)),
            time_s=i / 50.0,
            frame_index=i,
        )
        for i in range(16)
    ]
    report = analyze_four_gait_signals(_recording(snaps))
    fig = Figure(figsize=(8.0, 6.0))
    axes = build_signal_axes(fig, combined=False, lr_overlay=True)
    assert len(axes) == 2
    render_four_gait_figure(fig, axes, report, "filtered", lr_overlay=True)
    assert "Thigh" in (axes[0].get_title() or "")
    assert "Shank" in (axes[1].get_title() or "")
    assert axes[0].get_ylabel() == FOUR_GAIT_ANGLE_YLABEL
    assert "Time" in (axes[1].get_xlabel() or "")
    assert len(axes[0].lines) == 2
    assert len(axes[1].lines) == 2
    both = build_signal_axes(fig, combined=True, lr_overlay=True)
    render_four_gait_figure(
        fig, both, report, "filtered", combined=True, lr_overlay=True
    )
    assert len(both) == 3
    assert len(both[2].lines) == 4
