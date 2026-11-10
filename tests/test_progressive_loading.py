"""Progressive loading — media first, panel spinners, staged paints."""

from __future__ import annotations

import tkinter as tk
import unittest
from types import SimpleNamespace
from tkinter import ttk

from stablewalk.ui.theme import apply_theme
from stablewalk.ui.tk.panel_loading import (
    clear_all_panel_loading,
    set_panel_loading,
    set_panels_loading,
)


class PanelLoadingSpinnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = tk.Tk()
        cls.root.withdraw()
        apply_theme(cls.root, ttk.Style(cls.root))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.root.destroy()

    def setUp(self) -> None:
        self.host = tk.Frame(self.root, width=200, height=80)
        self.host.pack()
        self.gui = SimpleNamespace(
            _overview_kpi_metrics_host=self.host,
            knee_chart_host=self.host,
            biomech_chart_host=self.host,
            _sidebar_opensim_panel=self.host,
        )

    def tearDown(self) -> None:
        clear_all_panel_loading(self.gui)
        self.host.destroy()

    def test_spinner_place_and_clear(self) -> None:
        set_panel_loading(self.gui, "metrics", True, message="Loading metrics…")
        self.root.update_idletasks()
        store = self.gui._panel_loading
        self.assertIn("metrics", store)
        overlay = store["metrics"]["overlay"]
        self.assertEqual(str(overlay.winfo_manager()), "place")
        set_panel_loading(self.gui, "metrics", False)
        self.assertNotIn("metrics", self.gui._panel_loading)

    def test_set_panels_loading_arms_all_keys(self) -> None:
        set_panels_loading(self.gui, ("metrics", "graphs", "biomech", "opensim"), True)
        self.assertEqual(
            set(self.gui._panel_loading.keys()),
            {"metrics", "graphs", "biomech", "opensim"},
        )
        clear_all_panel_loading(self.gui)
        self.assertEqual(self.gui._panel_loading, {})


class ProgressivePaintStagingTests(unittest.TestCase):
    def test_paint_analysis_panels_delegates_to_staged_painters(self) -> None:
        from stablewalk.ui.tk.app import StableWalkGUI

        calls: list[str] = []

        class _Gui:
            def _paint_metrics_panels(self) -> None:
                calls.append("metrics")

            def _paint_graphs_panels(self) -> None:
                calls.append("graphs")

            def _paint_biomech_panels(self) -> None:
                calls.append("biomech")

        StableWalkGUI._paint_analysis_panels_from_cache(_Gui())  # type: ignore[arg-type]
        self.assertEqual(calls, ["metrics", "graphs", "biomech"])

    def test_schedule_progressive_panel_paints_registers_timers(self) -> None:
        from stablewalk.ui.tk.app import StableWalkGUI

        scheduled: list[tuple[str, int]] = []

        class _Gui:
            def _paint_metrics_panels(self) -> None:
                return None

            def _paint_graphs_panels(self) -> None:
                return None

            def _paint_biomech_panels(self) -> None:
                return None

            def _refresh_opensim_status(self) -> None:
                return None

            def _maybe_auto_opensim_pipeline(self) -> None:
                return None

        gui = _Gui()
        # Patch schedule_unique to record delays without needing a Tk root.
        import stablewalk.ui.tk.timer_registry as tr

        original = tr.schedule_unique

        def _fake(owner, attr, delay_ms, callback, **_kw):
            scheduled.append((attr, int(delay_ms)))
            return None

        tr.schedule_unique = _fake  # type: ignore[assignment]
        try:
            StableWalkGUI._schedule_progressive_panel_paints(gui)  # type: ignore[arg-type]
        finally:
            tr.schedule_unique = original  # type: ignore[assignment]

        attrs = {a for a, _d in scheduled}
        self.assertIn("_progressive_metrics_after", attrs)
        self.assertIn("_progressive_graphs_after", attrs)
        self.assertIn("_progressive_biomech_after", attrs)
        self.assertIn("_progressive_opensim_after", attrs)
        delays = {a: d for a, d in scheduled}
        self.assertLess(delays["_progressive_metrics_after"], delays["_progressive_graphs_after"])
        self.assertLess(delays["_progressive_graphs_after"], delays["_progressive_biomech_after"])
        self.assertLess(delays["_progressive_biomech_after"], delays["_progressive_opensim_after"])


if __name__ == "__main__":
    unittest.main()
