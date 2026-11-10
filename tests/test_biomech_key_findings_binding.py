"""Key Findings cards bind to computed SessionMetricView values — not false N/A."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from stablewalk.analysis.biomechanical.types import MetricWithConfidence
from stablewalk.metrics.session_view import build_session_metric_snapshots
from stablewalk.ui.tk.dashboard_biomechanics import (
    _KEY_FINDINGS_SNAPSHOT_KEYS,
    bind_biomech_key_findings_from_session_view,
)
from stablewalk.ui.value_provenance import NA_DISPLAY


class BiomechKeyFindingsBindingTests(unittest.TestCase):
    def test_snapshot_keys_cover_sidebar_findings(self) -> None:
        self.assertIn("gait_quality", _KEY_FINDINGS_SNAPSHOT_KEYS)
        self.assertIn("cadence", _KEY_FINDINGS_SNAPSHOT_KEYS)
        self.assertIn("stability_state", _KEY_FINDINGS_SNAPSHOT_KEYS)

    def test_bind_shows_computed_values_not_na(self) -> None:
        ba = SimpleNamespace(
            gait_quality=SimpleNamespace(
                score=72.0,
                available=True,
                confidence=0.8,
                components={"symmetry": 70.0},
                dominant_factors=(),
            ),
            stability_margin=SimpleNamespace(
                stable_pct=68.0,
                mean_margin_m=0.05,
                valid_frame_ratio=0.9,
                confidence=0.7,
                warnings=(),
                per_frame=(),
                reportable=False,
                experimental_note=(
                    "Monocular COM–BoS is experimental and is not a physical stability score."
                ),
            ),
            symmetry=SimpleNamespace(
                overall_symmetry_pct=MetricWithConfidence(value=81.0, confidence=0.75)
            ),
            gait_metrics=SimpleNamespace(
                cadence=MetricWithConfidence(value=112.0, confidence=0.8),
                walking_speed=MetricWithConfidence(value=1.15, confidence=0.2),
            ),
        )
        cycles = SimpleNamespace(
            metrics=SimpleNamespace(cadence_steps_per_min=118.0),
        )
        view = build_session_metric_snapshots(biomechanical=ba, cycles=cycles)
        painted: dict[str, dict] = {}

        def _capture(card, *, value, unit="", available=True, numeric=None, **_kw):
            painted[card] = {
                "value": value,
                "unit": unit,
                "available": available,
                "numeric": numeric,
            }

        cards = {
            key: key
            for key in (
                "gait_quality",
                "stability_state",
                "symmetry",
                "stability_margin",
                "cadence",
                "walking_speed",
            )
        }
        gui = SimpleNamespace(_biomech_kpi_cards=cards)

        with patch(
            "stablewalk.ui.tk.dashboard_biomechanics.update_kpi_card",
            side_effect=_capture,
        ):
            bind_biomech_key_findings_from_session_view(gui, view, ba=ba)

        self.assertTrue(painted["gait_quality"]["available"])
        self.assertNotEqual(painted["gait_quality"]["value"], NA_DISPLAY)
        self.assertAlmostEqual(float(painted["gait_quality"]["numeric"]), 72.0)

        self.assertTrue(painted["symmetry"]["available"])
        self.assertNotEqual(painted["symmetry"]["value"], NA_DISPLAY)

        self.assertFalse(painted["stability_margin"]["available"])
        self.assertEqual(painted["stability_margin"]["value"], NA_DISPLAY)

        self.assertTrue(painted["cadence"]["available"])
        self.assertAlmostEqual(float(painted["cadence"]["numeric"]), 112.0)

        self.assertFalse(painted["stability_state"]["available"])
        self.assertEqual(painted["stability_state"]["value"], NA_DISPLAY)

        self.assertTrue(painted["walking_speed"]["available"])
        self.assertNotEqual(painted["walking_speed"]["value"], NA_DISPLAY)

    def test_bind_na_only_when_truly_missing(self) -> None:
        view = build_session_metric_snapshots(biomechanical=None, cycles=None)
        painted: dict[str, dict] = {}

        def _capture(card, *, value, unit="", available=True, numeric=None, **_kw):
            painted[card] = {"value": value, "available": available}

        cards = {
            key: key
            for key in ("gait_quality", "cadence", "walking_speed", "symmetry")
        }
        gui = SimpleNamespace(_biomech_kpi_cards=cards)

        with patch(
            "stablewalk.ui.tk.dashboard_biomechanics.update_kpi_card",
            side_effect=_capture,
        ):
            bind_biomech_key_findings_from_session_view(gui, view, ba=None)

        for key in cards:
            self.assertFalse(painted[key]["available"])
            self.assertEqual(painted[key]["value"], NA_DISPLAY)


if __name__ == "__main__":
    unittest.main()
