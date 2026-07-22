"""Tests for Real-to-Sim pipeline card metrics and formatting."""

from __future__ import annotations

import unittest

from stablewalk.monitoring.pipeline_status import (
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    STATUS_UNAVAILABLE,
)
from stablewalk.ui.theme import MUTED, ORANGE, SUCCESS
from stablewalk.ui.tk.dashboard_pipeline_visual import (
    PIPELINE_STATUS_FG,
    compact_generated_files,
    format_duration_seconds,
    metrics_summary_line,
    parse_confidence_percent,
    PipelineStageMetrics,
)


class DashboardPipelineVisualTests(unittest.TestCase):
    def test_parse_confidence_percent(self) -> None:
        self.assertEqual(parse_confidence_percent("Foot contact confidence: 75%"), 75)
        self.assertEqual(parse_confidence_percent("8/10 frames (80%)"), 80)
        self.assertIsNone(parse_confidence_percent("—"))

    def test_format_duration_seconds(self) -> None:
        self.assertEqual(format_duration_seconds(None), "—")
        self.assertEqual(format_duration_seconds(0.02), "<0.1s")
        self.assertEqual(format_duration_seconds(2.34), "2.3s")

    def test_compact_generated_files(self) -> None:
        text, count = compact_generated_files(
            "• exports/run/stablewalk_motion.npz\n• exports/run/retargeted_motion.npz\n"
            "• exports/run/amp_reference_motion.npz"
        )
        self.assertEqual(count, 3)
        self.assertIn("+1", text)

    def test_metrics_summary_line(self) -> None:
        line = metrics_summary_line(
            PipelineStageMetrics(
                duration_text="1.2s",
                confidence_pct=88,
                confidence_text="88%",
                files_text="motion.npz",
                files_count=1,
            )
        )
        self.assertIn("1.2s", line)
        self.assertIn("88% conf", line)

    def test_shared_status_colors(self) -> None:
        self.assertEqual(PIPELINE_STATUS_FG[STATUS_COMPLETED], SUCCESS)
        self.assertEqual(PIPELINE_STATUS_FG[STATUS_PARTIAL], ORANGE)
        self.assertEqual(PIPELINE_STATUS_FG[STATUS_UNAVAILABLE], MUTED)


if __name__ == "__main__":
    unittest.main()
