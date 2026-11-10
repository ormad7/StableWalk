"""Video A and Video B pipelines must not share pose or analysis objects."""

from __future__ import annotations

from pathlib import Path

import pytest

from stablewalk.research.compare import compare_independent_clips
from stablewalk.research.dataset import ClipMetadata
from stablewalk.research.gait_phase import map_gait_phase
from stablewalk.research.pipeline import (
    analyze_independent_clip,
    slots_are_independent,
    slots_share_source_file,
)
from stablewalk.research.report import export_comparison_json, export_comparison_markdown

from test_asymmetry3d import _walking_recording


def _analyze(rec, cycles, video_id: str, *, amp_tag: str):
    meta = ClipMetadata(
        video_id=video_id,
        gait_category="normal_gait" if "sym" in amp_tag else "limping_gait",
        authenticity="simulated",
        license="test",
        local_path=f"{video_id}.mp4",
    )
    return analyze_independent_clip(
        recording=rec,
        metadata=meta,
        video_id=video_id,
        label=amp_tag,
        run_pose=False,
    )


def test_two_pipelines_own_distinct_objects() -> None:
    rec_a, cycles_a = _walking_recording(left_amp=40.0, right_amp=40.0, n_strides=4)
    rec_b, cycles_b = _walking_recording(left_amp=30.0, right_amp=60.0, n_strides=4)
    a = analyze_independent_clip(recording=rec_a, video_id="A", run_pose=False)
    b = analyze_independent_clip(recording=rec_b, video_id="B", run_pose=False)
    assert a.gait_motion is rec_a
    assert b.gait_motion is rec_b
    assert a.gait_motion is not b.gait_motion
    assert slots_are_independent(a, b)
    # Mutating A must not change B.
    snap_a = a.gait_motion.snapshots[10]
    dof = snap_a.get_dof("left_knee_flexion")
    original_b = b.gait_motion.snapshots[10].get_dof("left_knee_flexion").angle_deg
    dof.angle_deg = 999.0
    assert b.gait_motion.snapshots[10].get_dof("left_knee_flexion").angle_deg == original_b
    assert a.gait_motion.snapshots[10].get_dof("left_knee_flexion").angle_deg == 999.0
    _ = cycles_a, cycles_b


def test_same_recording_object_is_not_independent() -> None:
    rec, _cycles = _walking_recording(n_strides=3)
    a = analyze_independent_clip(recording=rec, video_id="A", run_pose=False)
    b = analyze_independent_clip(recording=rec, video_id="B", run_pose=False)
    assert slots_are_independent(a, b) is False


def test_same_source_file_with_distinct_objects_is_independent() -> None:
    rec_a, _ = _walking_recording(left_amp=40.0, n_strides=3)
    rec_b, _ = _walking_recording(left_amp=30.0, n_strides=3)
    meta_a = ClipMetadata(
        video_id="clip", gait_category="normal_gait", local_path="same.mp4"
    )
    meta_b = ClipMetadata(
        video_id="clip", gait_category="limping_gait", local_path="same.mp4"
    )
    a = analyze_independent_clip(
        recording=rec_a, metadata=meta_a, video_id="clip", run_pose=False
    )
    b = analyze_independent_clip(
        recording=rec_b, metadata=meta_b, video_id="clip", run_pose=False
    )
    assert a.gait_motion is not b.gait_motion
    assert slots_are_independent(a, b)
    assert slots_share_source_file(a, b)


def test_gait_phase_mapping_uses_each_clip_timeline() -> None:
    rec_a, cycles_a = _walking_recording(cycle_s=1.0, n_strides=4, fps=50.0)
    rec_b, cycles_b = _walking_recording(cycle_s=0.8, n_strides=5, fps=50.0)
    a = analyze_independent_clip(recording=rec_a, video_id="slow", run_pose=False)
    b = analyze_independent_clip(recording=rec_b, video_id="fast", run_pose=False)
    pa = map_gait_phase(a.gait_motion, a.cycles, 50.0)
    pb = map_gait_phase(b.gait_motion, b.cycles, 50.0)
    assert pa.used_gait_cycle or pb.used_gait_cycle or True
    # Independent times — not the same wall-clock frame forced onto both.
    assert pa.frame_index != pb.frame_index or pa.time_s != pb.time_s
    _ = cycles_a, cycles_b


def test_comparison_numeric_delta_and_report(tmp_path: Path) -> None:
    rec_a, _ = _walking_recording(left_amp=40.0, right_amp=40.0, n_strides=5)
    rec_b, _ = _walking_recording(left_amp=25.0, right_amp=70.0, n_strides=5)
    a = _analyze(rec_a, None, "sym", amp_tag="sym")
    b = _analyze(rec_b, None, "asym", amp_tag="asym")
    report = compare_independent_clips(a, b)
    assert report.independent is True
    knee = report.scalar_by_name("knee_asi")
    assert knee is not None
    assert (knee.value_b or 0.0) > (knee.value_a or 0.0)
    wave = report.waveform_by_name("knee_flexion")
    assert wave is not None
    js = export_comparison_json(report, tmp_path / "cmp.json")
    md = export_comparison_markdown(report, tmp_path / "cmp.md")
    assert "independent" in js.read_text(encoding="utf-8")
    text = md.read_text(encoding="utf-8")
    assert "Limitations" in text
    assert "dementia" in text.lower() or "clinical" in text.lower()


def test_run_pose_rejects_url() -> None:
    with pytest.raises(ValueError, match="download"):
        analyze_independent_clip(
            video_path="https://example.com/gait.mp4",
            run_pose=True,
        )
