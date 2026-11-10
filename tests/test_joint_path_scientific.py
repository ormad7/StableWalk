"""Scientific validation of the 3D Joint Path (reconstructed samples, not splines)."""

from __future__ import annotations

import math

from stablewalk.analysis.joint_path_trace import (
    SOURCE_ARRAY,
    gap_aware_prefix_lengths_m,
    index_nearest_time,
    polyline_length_m,
    segment_is_connected,
    trace_joint_path,
)
from stablewalk.models.gait_motion import (
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.ui.viewers.dof_trajectory_3d import (
    _joint_path_with_times,
    _prepare_display_path,
    _resolve_trajectory_points,
    stature_display_scale,
)


def _recording(*, gap: bool = False) -> GaitMotionRecording:
    snaps: list[SkeletonSnapshot] = []
    for i in range(10):
        if gap and i in (4, 5):
            joints = {
                "pelvis": JointSample("pelvis", Vec3(0.0, 0.0, 0.0)),
            }
        else:
            joints = {
                "pelvis": JointSample("pelvis", Vec3(0.0, 0.0, 0.0)),
                "right_knee": JointSample(
                    "right_knee",
                    Vec3(0.02 * i, -0.40, 0.01 * i),
                ),
            }
        snaps.append(
            SkeletonSnapshot(
                frame_index=i,
                time_s=i / 25.0,
                joints=joints,
                metadata={"landmark_visibility": {"right_knee": 0.9}},
            )
        )
    return GaitMotionRecording(source="test", fps=25.0, snapshots=snaps, source_kind="mock")


def test_source_array_is_snapshot_joint_positions() -> None:
    rec = _recording()
    trace = trace_joint_path(rec, "right_knee", rec.frame_count - 1)
    assert trace.source_array == SOURCE_ARRAY
    assert trace.samples[0].frame_index == 0
    assert len(trace.samples) == rec.frame_count
    plotted = _joint_path_with_times(rec, "right_knee", float(rec.frame_count - 1))
    assert len(plotted) == len(trace.samples)
    for (pos, t), sample in zip(plotted, trace.samples, strict=True):
        assert pos == sample.position
        assert abs(t - sample.time_s) < 1e-12


def test_display_path_is_one_vertex_per_reconstructed_sample() -> None:
    rec = _recording()
    raw = [p for p, _t in _joint_path_with_times(rec, "right_knee", 9.0)]
    display = _prepare_display_path(raw, overview=True, motion_dock=True, joint_id="right_knee")
    assert len(display) == len(raw)
    assert display[0] == raw[0]
    assert display[-1] == raw[-1]


def test_independent_path_length_matches_viewer_prefix() -> None:
    rec = _recording()
    pts = [p for p, _t in _joint_path_with_times(rec, "right_knee", 9.0)]
    independent = polyline_length_m(pts)
    timed = _joint_path_with_times(rec, "right_knee", 9.0)
    prefixes = gap_aware_prefix_lengths_m(timed, fps=25.0)
    assert abs(independent - prefixes[-1]) < 1e-12
    expected = 0.0
    for a, b in zip(pts, pts[1:], strict=False):
        expected += math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)
    assert abs(independent - expected) < 1e-12


def test_gap_is_not_counted_as_traveled_distance() -> None:
    rec = _recording(gap=True)
    timed = _joint_path_with_times(rec, "right_knee", 9.0)
    assert len(timed) == 8
    naive = polyline_length_m([p for p, _t in timed])
    prefixes = gap_aware_prefix_lengths_m(timed, fps=25.0)
    assert prefixes[-1] < naive - 1e-6
    t_before = timed[3][1]
    t_after = timed[4][1]
    assert not segment_is_connected(t_before, t_after, fps=25.0)


def test_path_cache_extend_does_not_mutate_prefix() -> None:
    rec = _recording()
    prefix = _joint_path_with_times(rec, "right_knee", 4.0)
    n_prefix = len(prefix)
    full = _joint_path_with_times(rec, "right_knee", 9.0)
    assert len(prefix) == n_prefix == 5
    assert len(full) == 10


def test_current_point_is_reconstructed_sample_not_appended() -> None:
    rec = _recording()
    tip = rec.snapshot_at(4)
    path, current, _end = _resolve_trajectory_points(
        rec,
        "right_knee",
        playback_frame_float=4.0,
        tip_snapshot=tip,
        display_mode="Current progress",
        coord_mode="Root-relative",
        motion_series=None,
        position_scale=1.0,
    )
    assert len(path) == 5
    assert current is not None
    assert abs(current.x - path[-1][0].x) < 1e-12
    assert abs(current.y - path[-1][0].y) < 1e-12
    assert abs(current.z - path[-1][0].z) < 1e-12
    assert abs(path[-1][1] - tip.time_s) < 1e-12


def test_playhead_index_matches_snapshot_time() -> None:
    rec = _recording()
    timed = _joint_path_with_times(rec, "right_knee", 9.0)
    tip = rec.snapshot_at(7)
    idx = index_nearest_time(timed, tip.time_s, frame_index=7)
    assert idx == 7
    assert abs(timed[idx][1] - tip.time_s) < 1e-12


def test_units_are_estimated_not_mocap() -> None:
    rec = _recording()
    trace = trace_joint_path(rec, "right_knee", 9)
    assert "not optically calibrated" in trace.units_note.lower() or "estimated" in trace.units_note.lower()
    scale = stature_display_scale(rec)
    assert scale == 1.0 or scale > 0.0


def test_draw_uses_reconstructed_vertices() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.dof_trajectory_3d import draw_single_dof_trajectory_3d

    rec = _recording()
    fig = Figure()
    ax = fig.add_subplot(111, projection="3d")
    ax._stablewalk_motion_dock = True
    ok, _status = draw_single_dof_trajectory_3d(
        ax,
        rec,
        "right_knee",
        end_frame_float=9.0,
        tip_snapshot=rec.snapshot_at(9),
        clear=True,
        display_mode="Current progress",
        coord_mode="Root-relative",
    )
    assert ok
    assert getattr(ax, "_stablewalk_path_source", "").endswith("position")
    times = getattr(ax, "_stablewalk_path_times", None)
    assert isinstance(times, list) and len(times) == 10
    cum = getattr(ax, "_stablewalk_progress_cumdist_m", None)
    scale = stature_display_scale(rec)
    independent = polyline_length_m(
        [
            p
            for p, _t in _joint_path_with_times(
                rec, "right_knee", 9.0, position_scale=scale
            )
        ]
    )
    assert cum and abs(float(cum[-1]) - independent) < 1e-9
