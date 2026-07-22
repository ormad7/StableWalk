"""Tests for Generated Files scanning and formatting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from stablewalk.ui.tk.generated_files_panel import (
    format_file_size,
    scan_generated_files,
)


class GeneratedFilesPanelTests(unittest.TestCase):
    def test_format_file_size(self) -> None:
        self.assertEqual(format_file_size(512), "512 B")
        self.assertEqual(format_file_size(2048), "2.0 KB")

    def test_scan_classifies_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "reports"
            tracking = root / "tracking"
            opensim = root / "opensim" / "run1"
            motion = root / "motion_reference" / "run1"
            sessions = root / "sessions"
            analysis = root / "analysis"
            poses = root / "poses"
            for path in (reports, tracking, opensim, motion, sessions, analysis, poses):
                path.mkdir(parents=True)

            (reports / "analysis_report_20260101_120000.txt").write_text("ok", encoding="utf-8")
            (tracking / "gait_metrics_20260101_120000.json").write_text("{}", encoding="utf-8")
            (opensim / "run1.trc").write_text("trc", encoding="utf-8")
            (motion / "stablewalk_motion.npz").write_bytes(b"npz")
            bundle = sessions / "stablewalk_session_demo"
            bundle.mkdir()
            (bundle / "session_metadata.json").write_text("{}", encoding="utf-8")

            with mock.patch("stablewalk.ui.tk.generated_files_panel.config") as cfg:
                cfg.REPORTS_DIR = reports
                cfg.TRACKING_EXPORT_DIR = tracking
                cfg.ANALYSIS_EXPORT_DIR = analysis
                cfg.SESSION_EXPORT_DIR = sessions
                cfg.POSES_DIR = poses
                cfg.OPENSIM_DIR = opensim.parent
                cfg.MOTION_REFERENCE_EXPORT_DIR = motion.parent
                cfg.ensure_output_dirs = lambda: None

                entries = scan_generated_files(run_name="run1")

            types = {e.file_type for e in entries}
            names = {e.filename for e in entries}
            self.assertIn("Analysis Report", types)
            self.assertIn("Gait Metrics", types)
            self.assertIn("OpenSim", types)
            self.assertIn("Motion Reference", types)
            self.assertIn("Session Bundle", types)
            self.assertIn("analysis_report_20260101_120000.txt", names)
            self.assertTrue(all(e.status == "Ready" for e in entries if e.filename.endswith(".txt")))


if __name__ == "__main__":
    unittest.main()
