"""Tests for the simplified 3D Skeleton Viewer toolbar."""

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
            _toggle_play=lambda: None,
            skel_canvas_host=self.host,
        )

    def tearDown(self) -> None:
        close = getattr(self.gui, "_skel_advanced_close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
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
        for icon, label, _attr, default_on, _tip, group in OVERLAY_TOOL_SPECS:
            self.assertTrue(icon)
            self.assertTrue(label)
            self.assertFalse(default_on, f"{label} must default OFF")
            self.assertIn(group, {g[0] for g in OVERLAY_GROUPS})

    def test_groups_are_motion_biomechanics_environment(self) -> None:
        keys = [key for key, _title, _icon in OVERLAY_GROUPS]
        self.assertEqual(keys, ["motion", "biomechanics", "environment"])

    def test_primary_bar_shows_only_joint_and_advanced(self) -> None:
        bar, display_slot = build_overlay_control_bar(
            self.gui, self.host, primary_only=True
        )
        bar.grid(row=0, column=0, sticky="ew")
        self.root.update_idletasks()

        self.assertIs(bar, self.gui._skel_overlay_bar)
        self.assertIs(display_slot, self.gui._skel_advanced_display_slot)
        self.assertIsNone(getattr(self.gui, "btn_play_skeleton", "missing"))
        self.assertIsNotNone(getattr(self.gui, "cmb_overview_joint_pick", None))
        # Advanced panel exists but is not mapped until opened.
        self.assertIsNotNone(getattr(self.gui, "_skel_advanced_body", None))
        self.assertFalse(bool(self.gui._skel_advanced_body.winfo_ismapped()))

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
        self.assertNotIn("Play", joined)
        self.assertIn("Advanced", joined)
        self.assertIn("Joint", joined)
        self.assertIn("Fit", joined)
        self.assertIsNotNone(getattr(self.gui, "_skel_fit_full_body_btn", None))
        # Select DOF and overlay labels stay out of the primary bar.
        self.assertNotIn("Select DOF", joined)
        self.assertNotIn("Direction", joined)
        self.assertNotIn("Ground", joined)

        for _icon, _label, attr, default_on, _tip, _group in OVERLAY_TOOL_SPECS:
            var = getattr(self.gui, attr)
            self.assertIsInstance(var, tk.BooleanVar)
            self.assertEqual(bool(var.get()), default_on)

    def test_advanced_dropdown_reveals_dof_and_overlays(self) -> None:
        bar, _slot = build_overlay_control_bar(self.gui, self.host)
        bar.pack(fill=tk.X)
        self.root.update()

        toggle = self.gui._skel_advanced_toggle
        toggle.invoke()
        self.root.update()

        body = self.gui._skel_advanced_body
        self.assertTrue(bool(body.winfo_ismapped()) or body.winfo_manager() == "place")

        texts: list[str] = []

        def _walk(widget: tk.Misc) -> None:
            for child in widget.winfo_children():
                try:
                    texts.append(str(child.cget("text")))
                except tk.TclError:
                    pass
                _walk(child)

        _walk(body)
        self.assertTrue(any("Select DOF" in t for t in texts))
        for _icon, label, _attr, _default, _tip, _group in OVERLAY_TOOL_SPECS:
            self.assertTrue(
                any(label in t for t in texts),
                f"missing Advanced toggle for {label!r}",
            )


if __name__ == "__main__":
    unittest.main()
