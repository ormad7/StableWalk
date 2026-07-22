"""Tests for the 3D reconstruction overlay toolbar (above-viz ribbon)."""

from __future__ import annotations

import tkinter as tk
import unittest
from types import SimpleNamespace

from tkinter import ttk

from stablewalk.ui.tk.skeleton_overlay_toolbar import (
    OVERLAY_GROUPS,
    OVERLAY_TOOL_SPECS,
    build_overlay_control_bar,
)
from stablewalk.ui.theme import apply_theme


class SkeletonOverlayToolbarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = tk.Tk()
        cls.root.withdraw()
        apply_theme(cls.root, ttk.Style(cls.root))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.root.destroy()

    def setUp(self) -> None:
        self.host = tk.Frame(self.root, width=640, height=480)
        self.host.pack()
        self.gui = SimpleNamespace(
            _on_biomech_overlay_toggle=lambda: None,
            skel_canvas_host=self.host,
        )

    def tearDown(self) -> None:
        self.host.destroy()

    def test_overlay_specs_cover_expected_controls(self) -> None:
        attrs = {spec[2] for spec in OVERLAY_TOOL_SPECS}
        self.assertEqual(
            attrs,
            {
                "var_overlay_com",
                "var_overlay_bos",
                "var_overlay_com_velocity",
                "var_overlay_contact",
                "var_overlay_direction",
                "var_overlay_ground",
            },
        )
        for icon, label, _attr, _default, _tip, group in OVERLAY_TOOL_SPECS:
            self.assertTrue(icon)
            self.assertTrue(label)
            self.assertIn(group, {g[0] for g in OVERLAY_GROUPS})

    def test_groups_are_motion_biomechanics_environment(self) -> None:
        keys = [key for key, _title, _icon in OVERLAY_GROUPS]
        self.assertEqual(keys, ["motion", "biomechanics", "environment"])
        for _key, title, icon in OVERLAY_GROUPS:
            self.assertTrue(icon)
            self.assertTrue(title)

    def test_builds_bar_above_host_without_side_rail(self) -> None:
        bar, right_slot = build_overlay_control_bar(self.gui, self.host)
        bar.grid(row=0, column=0, sticky="ew")
        plot = tk.Frame(self.host, bg="#000000")
        plot.grid(row=1, column=0, sticky="nsew")
        self.host.rowconfigure(0, weight=0)
        self.host.rowconfigure(1, weight=1)
        self.root.update_idletasks()

        self.assertIs(bar, self.gui._skel_overlay_bar)
        self.assertIs(right_slot, self.gui._skel_overlay_right_slot)
        self.assertEqual(int(bar.grid_info()["row"]), 0)
        self.assertEqual(int(plot.grid_info()["row"]), 1)
        # No left-rail widgets — overlays must never steal skeleton columns.
        self.assertFalse(hasattr(self.gui, "_skel_overlay_rail"))
        self.assertFalse(hasattr(self.gui, "_skel_plot_slot"))

        for _icon, _label, attr, default_on, _tip, _group in OVERLAY_TOOL_SPECS:
            var = getattr(self.gui, attr)
            self.assertIsInstance(var, tk.BooleanVar)
            self.assertEqual(bool(var.get()), default_on)

    def test_icons_appear_on_toggles(self) -> None:
        bar, _right = build_overlay_control_bar(self.gui, self.host)
        bar.pack(fill=tk.X)
        self.root.update_idletasks()

        texts: list[str] = []

        def _walk(widget: tk.Misc) -> None:
            for child in widget.winfo_children():
                try:
                    texts.append(str(child.cget("text")))
                except tk.TclError:
                    pass
                _walk(child)

        _walk(bar)
        joined = " ".join(texts)
        self.assertIn("Motion", joined)
        self.assertIn("Biomechanics", joined)
        self.assertIn("Environment", joined)
        for icon, label, _attr, _default, _tip, _group in OVERLAY_TOOL_SPECS:
            self.assertTrue(
                any(icon in t and label in t for t in texts),
                f"missing toggle for {icon} {label}",
            )


if __name__ == "__main__":
    unittest.main()
