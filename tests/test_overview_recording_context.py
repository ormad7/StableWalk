"""Overview recording context keeps documented labels separate from computed results."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

tk = pytest.importorskip("tkinter")


def test_recording_context_shows_documented_and_pending_computed() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    try:
        from stablewalk.ui.media.demo_comparison import COMPUTED_PENDING, DOCUMENTED_UNAVAILABLE
        from stablewalk.ui.media.demo_gait import example_by_key
        from stablewalk.ui.tk.dashboard_sections import build_overview_clinical_summary

        from stablewalk.ui.tk.app import StableWalkGUI

        gui = SimpleNamespace(
            _active_demo_gait=example_by_key("athletic"),
            gait_motion=None,
            sequence=None,
            pose_indices=[],
            _demo_file_duration_s=None,
            _demo_comparison_rows={},
            selection=SimpleNamespace(active_item_id="right_knee"),
            _biomech=None,
            _gait_cycle=None,
        )
        parent = tk.Frame(root)
        build_overview_clinical_summary(gui, parent)
        StableWalkGUI._refresh_overview_recording_context(gui)
        labels = gui._overview_context_labels
        assert "walk_54" in labels["source_label"].cget("text")
        assert labels["camera"].cget("text") == "Left side (fixed)"
        assert labels["condition"].cget("text") == "Treadmill"
        assert labels["duration"].cget("text") == COMPUTED_PENDING
        assert labels["cycles"].cget("text") == COMPUTED_PENDING
        assert labels["tracking"].cget("text") == COMPUTED_PENDING
        assert labels["source_label"].cget("text") != DOCUMENTED_UNAVAILABLE
        assert labels["source_label"].cget("text") != labels["duration"].cget("text")
        assert gui.overview_comparison_tree is not None
    finally:
        root.destroy()


def test_clear_overview_surfaces_blanks_computed_kpis() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    try:
        from stablewalk.ui.media.demo_comparison import COMPUTED_PENDING
        from stablewalk.ui.media.demo_gait import example_by_key
        from stablewalk.ui.tk.app import StableWalkGUI
        from stablewalk.ui.tk.dashboard_sections import (
            build_overview_clinical_summary,
            build_overview_playback_hud,
        )

        gui = SimpleNamespace(
            _active_demo_gait=example_by_key("normal"),
            gait_motion=None,
            sequence=None,
            pose_indices=[],
            _demo_file_duration_s=None,
            _demo_comparison_rows={},
            selection=SimpleNamespace(active_item_id="right_knee"),
            _biomech=None,
            _gait_cycle=None,
        )
        parent = tk.Frame(root)
        build_overview_playback_hud(gui, parent)
        build_overview_clinical_summary(gui, parent)
        gui._refresh_overview_recording_context = (
            lambda: StableWalkGUI._refresh_overview_recording_context(gui)
        )
        gui._refresh_demo_comparison_panel = (
            lambda: StableWalkGUI._refresh_demo_comparison_panel(gui)
        )
        gui._sync_gait_comparison_visibility = lambda **_k: None
        gui._overview_hud_value_labels["phase"].configure(text="Stance")
        gui._overview_clinical_value_labels["gait_quality"].configure(text="72")
        gui._overview_context_labels["duration"].configure(text="9.98 s")
        gui._overview_context_labels["cycles"].configure(text="6")

        StableWalkGUI._clear_overview_analysis_surfaces(gui)

        assert gui._overview_hud_value_labels["phase"].cget("text") == "—"
        assert gui._overview_clinical_value_labels["gait_quality"].cget("text") == "—"
        assert gui._overview_context_labels["duration"].cget("text") == COMPUTED_PENDING
        assert gui._overview_context_labels["cycles"].cget("text") == COMPUTED_PENDING
        assert "gait_pat=normal" in gui._overview_context_labels["source_label"].cget("text")
        assert gui._overview_context_labels["camera"].cget("text") == "Right side"
    finally:
        root.destroy()
