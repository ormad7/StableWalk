"""Tests for session bundle export and import."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from stablewalk.io.session_bundle import (
    FILE_ANALYSIS_SUMMARY,
    FILE_GAIT_MOTION,
    FILE_SELECTED_POINTS,
    FILE_SESSION_METADATA,
    FILE_TRACKING_HISTORY,
    FILE_WORKSPACE_STATE,
    SessionBundleError,
    SessionBundleSnapshot,
    export_session_bundle,
    load_session_bundle,
    selected_ids_from_payload,
    tracking_rows_to_kinematic_samples,
)
from stablewalk.models.gait_motion import (
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.storage.collector import SessionKinematicCollector


def _sample_recording(frame_count: int = 4, *, include_hip: bool = False) -> GaitMotionRecording:
    snapshots: list[SkeletonSnapshot] = []
    for index in range(frame_count):
        y = 0.05 + index * 0.02
        joints = {
            "left_toe": JointSample(
                "left_toe",
                Vec3(0.1, y, 0.2),
                angle_deg=10.0 + index,
            ),
            "right_toe": JointSample(
                "right_toe",
                Vec3(-0.1, 0.04, 0.2),
            ),
        }
        if include_hip:
            joints["right_hip"] = JointSample(
                "right_hip",
                Vec3(0.0, 0.9 + index * 0.01, 0.0),
            )
        snapshots.append(
            SkeletonSnapshot(
                frame_index=index,
                time_s=index / 30.0,
                joints=joints,
            )
        )
    return GaitMotionRecording(source="test.mp4", fps=30.0, snapshots=snapshots)


class SessionBundleTests(unittest.TestCase):
    def test_export_and_load_round_trip(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            recording = _sample_recording()
            collector = SessionKinematicCollector()
            snap = recording.snapshot_at(0)
            assert snap is not None
            collector.append_tick(snap, {"left_toe"}, next_snapshot=recording.snapshot_at(1))

            snapshot = SessionBundleSnapshot(
                video_source="test.mp4",
                poses_json_path="/tmp/poses.json",
                fps=30.0,
                frame_count=recording.frame_count,
                selected_item_ids={"left_toe", "right_toe"},
                last_selected="left_toe",
                charted_item_id="left_toe",
                active_item_id="left_toe",
                analysis_mode="foot",
                frame_index=1,
                frame_float=1.0,
                time_s=1 / 30.0,
                dof_table_display_mode="Tracking History",
                tracking_samples=collector.samples,
                recording=recording,
            )

            bundle_dir = export_session_bundle(snapshot, tmp_path)
            self.assertTrue(bundle_dir.is_dir())
            for name in (
                FILE_SESSION_METADATA,
                FILE_SELECTED_POINTS,
                FILE_TRACKING_HISTORY,
                FILE_ANALYSIS_SUMMARY,
                FILE_GAIT_MOTION,
            ):
                self.assertTrue((bundle_dir / name).is_file())

            loaded = load_session_bundle(bundle_dir)
            self.assertIsNotNone(loaded.recording)
            assert loaded.recording is not None
            self.assertEqual(loaded.recording.frame_count, recording.frame_count)
            self.assertEqual(
                selected_ids_from_payload(loaded.selected_points),
                {"left_toe", "right_toe"},
            )
            self.assertGreaterEqual(len(loaded.tracking_rows), 1)
            samples = tracking_rows_to_kinematic_samples(loaded.tracking_rows)
            self.assertTrue(samples)
            self.assertEqual(loaded.metadata["video_source"], "test.mp4")
            self.assertEqual(loaded.metadata.get("active_item_id"), "left_toe")

    def test_workspace_and_overwrite_save(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            recording = _sample_recording()
            snapshot = SessionBundleSnapshot(
                video_source="cam.mp4",
                selected_item_ids={"left_toe"},
                last_selected="left_toe",
                active_item_id="left_toe",
                charted_item_id="left_toe",
                frame_index=2,
                frame_float=2.0,
                fps=30.0,
                frame_count=recording.frame_count,
                recording=recording,
                workspace={
                    "graphs": {"dof_projection": "Sagittal Plane"},
                    "cameras": {"trajectory": {"elev": 25.0, "azim": -70.0, "zoom": 1.2}},
                    "overview_view_mode": "side_by_side",
                },
                results={"note": "digest"},
            )
            first = export_session_bundle(snapshot, tmp_path, copy_poses=False)
            self.assertTrue((first / FILE_WORKSPACE_STATE).is_file())
            loaded = load_session_bundle(first)
            self.assertEqual(
                loaded.workspace.get("graphs", {}).get("dof_projection"),
                "Sagittal Plane",
            )
            self.assertEqual(
                loaded.workspace.get("cameras", {})
                .get("trajectory", {})
                .get("elev"),
                25.0,
            )

            snapshot.frame_index = 3
            snapshot.workspace["graphs"]["dof_projection"] = "3D"
            second = export_session_bundle(
                snapshot,
                tmp_path,
                target_bundle_dir=first,
                copy_poses=False,
            )
            self.assertEqual(second, first)
            reloaded = load_session_bundle(first)
            self.assertEqual(reloaded.metadata["playback"]["frame_index"], 3)
            self.assertEqual(
                reloaded.workspace.get("graphs", {}).get("dof_projection"),
                "3D",
            )

    def test_load_missing_metadata_raises(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SessionBundleError):
                load_session_bundle(Path(tmp))

    def test_load_corrupt_metadata_raises(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bad_session"
            bundle.mkdir()
            (bundle / FILE_SESSION_METADATA).write_text("{not json", encoding="utf-8")
            with self.assertRaises(SessionBundleError):
                load_session_bundle(bundle)

    def test_selected_points_payload_schema(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            recording = _sample_recording(frame_count=2)
            snapshot = SessionBundleSnapshot(
                video_source="demo.mp4",
                selected_item_ids={"left_heel"},
                last_selected="left_heel",
                recording=recording,
                frame_count=2,
                fps=30.0,
            )
            bundle_dir = export_session_bundle(snapshot, Path(tmp))
            payload = json.loads(
                (bundle_dir / FILE_SELECTED_POINTS).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["selected_item_ids"], ["left_heel"])
            self.assertEqual(payload["labels"]["left_heel"], "Left Heel")

    def test_tracking_history_includes_foot_clearance_fields(self) -> None:
        import csv
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            recording = _sample_recording(frame_count=3)
            snapshot = SessionBundleSnapshot(
                video_source="test.mp4",
                selected_item_ids={"left_toe"},
                last_selected="left_toe",
                active_item_id="left_toe",
                charted_item_id="left_toe",
                analysis_mode="foot",
                frame_index=1,
                recording=recording,
                frame_count=recording.frame_count,
                fps=30.0,
            )
            bundle_dir = export_session_bundle(snapshot, Path(tmp))
            csv_path = bundle_dir / FILE_TRACKING_HISTORY
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertTrue(rows)
            row = rows[0]
            for heading in (
                "Time (s)",
                "Frame",
                "Selected Point",
                "Vertical Position (m)",
                "Foot Clearance (m)",
                "Foot Clearance (cm)",
                "Contact State",
                "Min Clearance (m)",
                "Max Clearance (m)",
                "Average Clearance (m)",
            ):
                self.assertIn(heading, row)
            foot_rows = [r for r in rows if r.get("Selected Point") == "Left Toe"]
            self.assertEqual(len(foot_rows), recording.frame_count)
            sample = foot_rows[0]
            self.assertTrue(sample.get("Foot Clearance (m)", "").strip())
            self.assertIn(
                sample.get("Contact State", ""),
                {"On Ground", "Near Ground", "In Air", "Check calibration", ""},
            )

            summary = json.loads(
                (bundle_dir / FILE_ANALYSIS_SUMMARY).read_text(encoding="utf-8")
            )
            self.assertIn("exported_at", summary)
            self.assertEqual(summary.get("active_item_id"), "left_toe")
            self.assertEqual(summary.get("analysis_mode"), "foot")
            self.assertEqual(summary.get("points_analyzed"), ["left_toe"])
            current = summary.get("current_frame_analysis") or {}
            self.assertEqual(current.get("selected_point"), "Left Toe")
            self.assertEqual(current.get("analysis_mode"), "foot")
            self.assertIn("current_foot_clearance", current)
            self.assertIn("foot_clearance_summary", current)
            self.assertIn("delta_from_start", current)

    def test_general_point_export_omits_foot_columns(self) -> None:
        import csv
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            recording = _sample_recording(frame_count=3, include_hip=True)
            snapshot = SessionBundleSnapshot(
                video_source="test.mp4",
                selected_item_ids={"right_hip", "left_toe"},
                active_item_id="right_hip",
                charted_item_id="right_hip",
                analysis_mode="general",
                frame_index=1,
                recording=recording,
                frame_count=recording.frame_count,
                fps=30.0,
            )
            bundle_dir = export_session_bundle(snapshot, Path(tmp))
            csv_path = bundle_dir / FILE_TRACKING_HISTORY
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                rows = list(reader)
            self.assertNotIn("Foot Clearance (m)", fieldnames)
            self.assertNotIn("Contact State", fieldnames)
            self.assertIn("Speed (m/s)", fieldnames)
            self.assertTrue(rows)
            self.assertTrue(
                all(r.get("Selected Point") == "Right Hip" for r in rows)
            )

            summary = json.loads(
                (bundle_dir / FILE_ANALYSIS_SUMMARY).read_text(encoding="utf-8")
            )
            self.assertEqual(summary.get("analysis_mode"), "general")
            self.assertEqual(summary.get("active_item_id"), "right_hip")
            current = summary.get("current_frame_analysis") or {}
            self.assertNotIn("foot_clearance_summary", current)


if __name__ == "__main__":
    unittest.main()
