"""Tests for Overview frame-index consistency across dashboard readouts."""

from __future__ import annotations

import unittest

from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.ui.foot_clearance_display import foot_clearance_dashboard_for_panel
from stablewalk.ui.overview_frame_consistency import (
    OverviewFrameIndices,
    assert_overview_frames_consistent,
    audit_recording_frames,
    collect_overview_frame_indices,
)


def _demo_recording():
    from stablewalk import config

    path = config.POSES_DIR / "normal_gait_poses.json"
    if not path.is_file():
        return None, None
    sequence = load_pose_sequence(path)
    enrich_pose_sequence(sequence)
    recording = pose_sequence_to_gait_motion(sequence)
    gait = analyze_gait_cycles(recording)
    return recording, gait


class OverviewFrameConsistencyTests(unittest.TestCase):
    def test_collect_indices_match_at_frame(self) -> None:
        recording, gait = _demo_recording()
        if recording is None:
            self.skipTest("normal_gait_poses.json not available")
        snap = recording.snapshot_at(10)
        assert snap is not None
        indices = collect_overview_frame_indices(
            snapshot=snap,
            gait_result=gait,
            video_frame_index=snap.frame_index,
            clearance_frame_index=snap.frame_index,
        )
        self.assertTrue(indices.consistent())
        self.assertEqual(indices.current_skeleton_frame, snap.frame_index)
        self.assertEqual(indices.current_video_frame, snap.frame_index)
        self.assertEqual(indices.current_clearance_frame, snap.frame_index)
        self.assertEqual(indices.current_contact_frame, snap.frame_index)
        self.assertEqual(indices.current_phase_frame, snap.frame_index)

    def test_assert_raises_on_mismatch(self) -> None:
        bad = OverviewFrameIndices(0, 1, 2, 1, 1)
        with self.assertRaises(AssertionError):
            assert_overview_frames_consistent(bad)

    def test_audit_recording_sampled_frames(self) -> None:
        recording, gait = _demo_recording()
        if recording is None:
            self.skipTest("normal_gait_poses.json not available")
        rows = audit_recording_frames(recording, gait, sample_every=8)
        self.assertGreater(len(rows), 0)

    def test_foot_clearance_uses_snapshot_frame(self) -> None:
        recording, _gait = _demo_recording()
        if recording is None:
            self.skipTest("normal_gait_poses.json not available")
        for index in (0, 12, 24):
            snap = recording.snapshot_at(index)
            if snap is None:
                continue
            panel = foot_clearance_dashboard_for_panel(snap, recording, float(index))
            self.assertIsNotNone(panel)
            indices = collect_overview_frame_indices(
                snapshot=snap,
                gait_result=_gait,
                clearance_frame_index=snap.frame_index,
            )
            assert_overview_frames_consistent(indices)

    def test_debug_lines_format(self) -> None:
        indices = OverviewFrameIndices(5, 5, 5, 5, 5)
        text = "\n".join(indices.as_debug_lines())
        self.assertIn("current_video_frame=5", text)
        self.assertIn("current_clearance_frame=5", text)


if __name__ == "__main__":
    unittest.main()
