"""Tests for per-panel maximize / restore."""

from __future__ import annotations

import tkinter as tk
import unittest
from tkinter import ttk
from types import SimpleNamespace
from unittest import mock

from stablewalk.ui.theme import apply_theme
from stablewalk.ui.tk.dashboard_panel_maximize import (
    PANEL_TITLES,
    current_maximized_panel_id,
    install_maximize_button,
    is_panel_maximized,
    maximize_panel,
    restore_panel,
)


class PanelMaximizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = tk.Tk()
        cls.root.withdraw()
        apply_theme(cls.root, ttk.Style(cls.root))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.root.destroy()

    def setUp(self) -> None:
        self.main = ttk.Frame(self.root)
        self.main.pack(fill=tk.BOTH, expand=True)
        self.main.configure(width=640, height=480)
        self.panel = ttk.LabelFrame(self.main, text="Video")
        self.panel.grid(row=0, column=0, sticky="nsew")
        self.main.columnconfigure(0, weight=1)
        self.main.rowconfigure(0, weight=1)

        self.gui = SimpleNamespace(
            root=self.root,
            _dashboard_main=self.main,
            video_frame=self.panel,
            playing=False,
            _playback_pos=12.0,
            skeleton_player=None,
            _panel_maximize_buttons={},
            status=SimpleNamespace(configure=lambda **_k: None),
        )
        self.root.update_idletasks()

    def tearDown(self) -> None:
        if is_panel_maximized(self.gui):
            restore_panel(self.gui)
        try:
            self.main.destroy()
        except tk.TclError:
            pass

    def test_panel_titles_cover_required_set(self) -> None:
        self.assertEqual(
            set(PANEL_TITLES),
            {
                "video",
                "skeleton",
                "path_3d",
                "motion",
                "biomechanics",
                "results_summary",
                "pipeline",
            },
        )
        self.assertEqual(PANEL_TITLES["path_3d"], "3D Path")
        self.assertEqual(PANEL_TITLES["pipeline"], "Advanced Pipeline")

    def test_maximize_and_restore_preserves_grid(self) -> None:
        with mock.patch(
            "stablewalk.ui.tk.dashboard_panel_maximize._capture_playback",
            return_value=(False, 7.5),
        ), mock.patch(
            "stablewalk.ui.tk.dashboard_panel_maximize._restore_playback"
        ) as restore_play:
            ok = maximize_panel(self.gui, "video")
            self.assertTrue(ok)
            self.assertTrue(is_panel_maximized(self.gui))
            self.assertEqual(current_maximized_panel_id(self.gui), "video")
            self.root.update_idletasks()

            # Placed over host
            place = self.panel.place_info()
            self.assertTrue(place)

            ok = restore_panel(self.gui)
            self.assertTrue(ok)
            self.assertFalse(is_panel_maximized(self.gui))
            self.root.update_idletasks()

            info = self.panel.grid_info()
            self.assertEqual(int(info["row"]), 0)
            self.assertEqual(int(info["column"]), 0)
            self.assertGreaterEqual(restore_play.call_count, 2)

    def test_maximize_button_installs(self) -> None:
        bar = tk.Frame(self.main)
        bar.grid(row=1, column=0)
        btn = install_maximize_button(self.gui, bar, "video")
        self.assertIn("video", self.gui._panel_maximize_buttons)
        self.assertEqual(str(btn.cget("text")), "⛶")

    def test_path_3d_maximize_restores_parent(self) -> None:
        dock = ttk.Frame(self.main)
        dock.grid(row=0, column=1, sticky="nsew")
        path = ttk.LabelFrame(dock, text="3D Path")
        path.grid(row=0, column=0, sticky="nsew")
        self.gui.overview_traj_panel = path
        self.gui._overview_traj_dock_visible = True
        self.gui.traj_panel = None

        with mock.patch(
            "stablewalk.ui.tk.dashboard_panel_maximize._capture_playback",
            return_value=(True, 3.0),
        ), mock.patch(
            "stablewalk.ui.tk.dashboard_panel_maximize._restore_playback"
        ) as restore_play, mock.patch(
            "stablewalk.ui.tk.dashboard_panel_maximize._ensure_panel_tab"
        ), mock.patch(
            "stablewalk.ui.tk.dashboard_overview_view_mode.apply_overview_view_mode"
        ):
            self.assertTrue(maximize_panel(self.gui, "path_3d"))
            self.assertEqual(current_maximized_panel_id(self.gui), "path_3d")
            self.assertTrue(restore_panel(self.gui))
            self.root.update_idletasks()
            info = path.grid_info()
            self.assertEqual(str(info.get("in", dock)), str(dock))
            self.assertGreaterEqual(restore_play.call_count, 2)


if __name__ == "__main__":
    unittest.main()
