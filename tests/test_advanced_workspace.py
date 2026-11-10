"""Advanced tab categorized workspace structure."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.ui.tk.dashboard_advanced_workspace import (
    ADVANCED_CATEGORIES,
    _CATEGORY_ACTIONS,
)


def test_advanced_category_order() -> None:
    keys = [k for k, _t, _e in ADVANCED_CATEGORIES]
    assert keys == ["session", "opensim", "real_to_sim", "exports", "reports"]


def test_primary_categories_expanded_by_default() -> None:
    by_key = {k: (title, expanded) for k, title, expanded in ADVANCED_CATEGORIES}
    assert by_key["session"] == ("Session", True)
    assert by_key["opensim"] == ("OpenSim", True)
    assert by_key["real_to_sim"] == ("Real-to-Sim", False)
    assert by_key["exports"] == ("Exports", False)
    assert by_key["reports"] == ("Reports", False)


def test_high_level_actions_present() -> None:
    assert "btn_save_session" in {a[0] for a in _CATEGORY_ACTIONS["session"]}
    assert "btn_view_detailed_data" in {a[0] for a in _CATEGORY_ACTIONS["session"]}
    assert "btn_export_pdf_report" in {a[0] for a in _CATEGORY_ACTIONS["reports"]}
    assert "btn_export_analysis_report" in {a[0] for a in _CATEGORY_ACTIONS["reports"]}
    assert "btn_export_gait_metrics" not in {a[0] for a in _CATEGORY_ACTIONS["reports"]}
    assert "btn_export_gait_metrics" in {a[0] for a in _CATEGORY_ACTIONS["exports"]}
    assert "btn_opensim_export_data" in {a[0] for a in _CATEGORY_ACTIONS["opensim"]}
    assert "btn_real_to_sim_pipeline" in {
        a[0] for a in _CATEGORY_ACTIONS["real_to_sim"]
    }
    assert "btn_export_motion_reference" in {
        a[0] for a in _CATEGORY_ACTIONS["real_to_sim"]
    }
    assert "btn_export_amp_reference" in {
        a[0] for a in _CATEGORY_ACTIONS["real_to_sim"]
    }
    assert "btn_real_to_sim_pipeline" not in {
        a[0] for a in _CATEGORY_ACTIONS["exports"]
    }
    assert "btn_export_motion_reference" not in {
        a[0] for a in _CATEGORY_ACTIONS["exports"]
    }


def test_session_and_reports_have_informative_captions() -> None:
    from stablewalk.ui.tk.dashboard_advanced_workspace import _CATEGORY_CAPTIONS

    assert "Save" in _CATEGORY_CAPTIONS["session"]
    assert "joint data" in _CATEGORY_CAPTIONS["session"]
    assert "Generated" in _CATEGORY_CAPTIONS["session"] or "generated" in _CATEGORY_CAPTIONS["session"]
    assert _CATEGORY_CAPTIONS["session"].startswith("Start here")
    assert "IK" in _CATEGORY_CAPTIONS["opensim"]
    assert _CATEGORY_CAPTIONS["opensim"].startswith("Model workspace")
    assert "Retargeting" in _CATEGORY_CAPTIONS["real_to_sim"]
    assert _CATEGORY_CAPTIONS["real_to_sim"].startswith("Simulation bridge")
    assert "gait metrics" in _CATEGORY_CAPTIONS["exports"].lower()
    assert _CATEGORY_CAPTIONS["exports"].startswith("Research data")
    assert "Results" in _CATEGORY_CAPTIONS["reports"]
    assert _CATEGORY_CAPTIONS["reports"].startswith("Technical archives")


def test_session_and_reports_share_top_row() -> None:
    from types import SimpleNamespace
    import tkinter as tk

    from stablewalk.ui.tk.dashboard_advanced_workspace import (
        ADVANCED_TWO_COL_MIN,
        _CATEGORY_ACTIONS,
        build_advanced_workspace,
        reflow_advanced_workspace,
    )

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        import pytest

        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    try:
        gui = SimpleNamespace(_sidebar_hidden=tk.Frame(root))

        def _noop(*_args, **_kwargs):
            return None

        for actions in _CATEGORY_ACTIONS.values():
            for _attr, _label, cmd, _primary, _tip in actions:
                setattr(gui, cmd, _noop)
        gui._refresh_pose_backend_status_label = _noop
        gui._add_point_from_combo = _noop

        host = build_advanced_workspace(gui, root)
        assert host is gui._advanced_workspace
        cards = gui._advanced_category_hosts
        session = cards["session"].grid_info()
        opensim = cards["opensim"].grid_info()
        assert int(session["row"]) == 0
        assert int(session["column"]) == 0
        assert int(opensim["row"]) == 0
        assert int(opensim["column"]) == 1
        real_to_sim = cards["real_to_sim"].grid_info()
        assert int(real_to_sim["row"]) == 1
        assert int(real_to_sim["column"]) == 0
        assert int(real_to_sim["columnspan"]) == 2
        exports = cards["exports"].grid_info()
        reports = cards["reports"].grid_info()
        assert int(exports["row"]) == 2
        assert int(exports["column"]) == 0
        assert int(reports["row"]) == 2
        assert int(reports["column"]) == 1
        assert host.grid_columnconfigure(0)["uniform"] == "advanced-column"
        assert host.grid_columnconfigure(1)["uniform"] == "advanced-column"
        assert int(host.grid_columnconfigure(0)["minsize"]) >= 240
        assert int(host.grid_columnconfigure(1)["minsize"]) >= 240
        assert all(
            int(host.grid_rowconfigure(row)["weight"]) == 0
            for row in range(len(ADVANCED_CATEGORIES))
        )
        assert (
            gui._advanced_category_bodies["real_to_sim_slot"]
            is not gui._advanced_category_bodies["exports_slot"]
        )

        reflow_advanced_workspace(gui, width=ADVANCED_TWO_COL_MIN - 1)
        assert int(cards["opensim"].grid_info()["row"]) == 1
        assert int(cards["opensim"].grid_info()["column"]) == 0
        assert int(cards["real_to_sim"].grid_info()["row"]) == 2
        assert int(cards["exports"].grid_info()["row"]) == 3
        assert int(cards["reports"].grid_info()["row"]) == 4
        assert int(cards["real_to_sim"].grid_info()["columnspan"]) == 1
        assert not host.grid_columnconfigure(0)["uniform"]
        assert not host.grid_columnconfigure(1)["uniform"]
        reports_btn = gui._advanced_action_buttons["reports"][0]
        assert int(reports_btn.grid_info()["columnspan"]) == 2

        reflow_advanced_workspace(gui, width=ADVANCED_TWO_COL_MIN)
        assert int(cards["opensim"].grid_info()["row"]) == 0
        assert int(cards["opensim"].grid_info()["column"]) == 1
        assert int(cards["real_to_sim"].grid_info()["columnspan"]) == 2
        assert int(cards["reports"].grid_info()["row"]) == 2
        assert int(cards["reports"].grid_info()["column"]) == 1
        assert host.grid_columnconfigure(0)["uniform"] == "advanced-column"
        assert host.grid_columnconfigure(1)["uniform"] == "advanced-column"
        assert int(gui._advanced_action_buttons["reports"][0].grid_info()["columnspan"]) == 1
        caption = gui._advanced_caption_labels["session"]
        wrap = int(str(caption.cget("wraplength") or 0).split(".")[0] or 0)
        assert wrap >= 200
        assert wrap <= 560
        for key, expected in (
            ("session", "Session"),
            ("opensim", "OpenSim"),
            ("real_to_sim", "Real-to-Sim"),
            ("exports", "Exports"),
            ("reports", "Reports"),
        ):
            title_lbl = gui._advanced_title_labels[key]
            assert str(title_lbl.cget("text") or "") == expected
            assert int(title_lbl.winfo_reqheight() or 0) >= 16
            title_wrap = int(str(title_lbl.cget("wraplength") or 0).split(".")[0] or 0)
            assert title_wrap >= 120
    finally:
        root.destroy()


def test_advanced_gait_metrics_section_has_intro_copy() -> None:
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "stablewalk"
        / "ui"
        / "tk"
        / "dashboard_sections.py"
    )
    text = src.read_text(encoding="utf-8")
    assert "Contact state and cycle completeness" in text
    assert 'SECTION_GAIT_METRICS_TITLE' in text
    assert "tk_labelframe_title" in text
    assert "labelwidget=title" in text


def test_advanced_gait_metrics_use_clear_unavailable_wording() -> None:
    from unittest.mock import MagicMock

    from stablewalk.ui.tk.app import StableWalkGUI

    gui = StableWalkGUI.__new__(StableWalkGUI)
    gui._biomech = None
    gui._gait_cycle = None
    gui._sync_demo_category_compare_strip = MagicMock()
    gui.lbl_gait_card_cycles_value = MagicMock()
    gui.lbl_gait_card_completeness_value = MagicMock()
    gui.lbl_overview_gait_cycles_usable = MagicMock()
    gui.lbl_overview_gait_cycles_completeness = MagicMock()

    StableWalkGUI._sync_gait_cycles_labels(
        gui,
        usable=None,
        detected=None,
        completeness_pct=None,
    )

    assert (
        gui.lbl_gait_card_cycles_value.configure.call_args.kwargs["text"]
        == "Not computed yet"
    )
    assert (
        gui.lbl_gait_card_completeness_value.configure.call_args.kwargs["text"]
        == "Completeness: Available after full analysis"
    )
    # This pass is Advanced-only; Overview keeps its compact placeholder.
    assert (
        gui.lbl_overview_gait_cycles_usable.configure.call_args.kwargs["text"]
        == "Usable: —"
    )


def test_advanced_gait_cycles_bind_after_full_analysis() -> None:
    from unittest.mock import MagicMock

    from stablewalk.ui.tk.app import StableWalkGUI

    gui = StableWalkGUI.__new__(StableWalkGUI)
    gui._biomech = SimpleNamespace(completeness_pct=100, usable_gait_cycles=0)
    gui._gait_cycle = SimpleNamespace(cycles=[])
    gui._sync_demo_category_compare_strip = MagicMock()
    gui.lbl_gait_card_cycles_value = MagicMock()
    gui.lbl_gait_card_completeness_value = MagicMock()
    gui.lbl_overview_gait_cycles_usable = MagicMock()
    gui.lbl_overview_gait_cycles_completeness = MagicMock()

    StableWalkGUI._sync_gait_cycles_labels(
        gui,
        usable=0,
        detected=0,
        completeness_pct=100,
    )

    cycles = gui.lbl_gait_card_cycles_value.configure.call_args.kwargs["text"]
    assert cycles != "Not computed yet"
    assert "0 usable" in cycles.lower()
    assert (
        gui.lbl_gait_card_completeness_value.configure.call_args.kwargs["text"]
        == "Completeness: 100%"
    )


def test_advanced_contact_placeholder_uses_unavailable_after_analysis() -> None:
    from unittest.mock import MagicMock

    from stablewalk.ui.tk.app import StableWalkGUI

    gui = StableWalkGUI.__new__(StableWalkGUI)
    gui.lbl_gait_card_contact_left = MagicMock()
    gui.lbl_gait_card_contact_right = MagicMock()
    gui.lbl_overview_contact_left = MagicMock()
    gui.lbl_overview_contact_right = MagicMock()
    gui.lbl_foot_left_state = MagicMock()
    gui.lbl_foot_right_state = MagicMock()

    StableWalkGUI._bind_gait_contact_placeholders(gui, computed=True)

    assert (
        gui.lbl_gait_card_contact_left.configure.call_args.kwargs["text"]
        == "Unavailable"
    )
    assert (
        gui.lbl_foot_left_state.configure.call_args.kwargs["text"] == "Unavailable"
    )


def test_advanced_contact_binds_without_sidebar_labels() -> None:
    from unittest.mock import MagicMock

    from stablewalk.ui.tk.app import StableWalkGUI

    state = SimpleNamespace(
        frame_index=3,
        left_contact=1,
        right_contact=0,
        left=SimpleNamespace(foot_clearance_m=0.01),
        right=SimpleNamespace(foot_clearance_m=0.04),
    )
    result = SimpleNamespace(
        per_frame=[state],
        cycles=[object()],
        metrics=SimpleNamespace(
            contact_confidence=0.8,
            confidence_tier="HIGH",
            cadence_steps_per_min=110.0,
            left_right_stance_symmetry=0.9,
            left_right_swing_symmetry=0.9,
            double_support_pct=20.0,
        ),
        frame_at=lambda _idx: state,
    )
    gui = StableWalkGUI.__new__(StableWalkGUI)
    gui._gait_cycle = result
    gui._biomech = SimpleNamespace(completeness_pct=100, usable_gait_cycles=2)
    gui._session_analysis = SimpleNamespace(
        ready=True,
        frame_snapshot=lambda _i: SimpleNamespace(
            phase_raw="stance",
            phase_dashboard="Stance",
        ),
    )
    gui.skeleton_player = None
    gui.playing = False
    gui._gait_cycle_metric_slots = None
    gui.lbl_gait_cycle_phase = None
    gui.lbl_gait_cycle_left_contact = None
    gui.lbl_gait_cycle_right_contact = None
    gui.lbl_gait_card_phase_value = None
    gui.lbl_gait_cycle_confidence = None
    gui.lbl_gait_cycle_interp_compact = None
    gui.lbl_gait_card_contact_left = MagicMock()
    gui.lbl_gait_card_contact_right = MagicMock()
    gui.lbl_overview_contact_left = MagicMock()
    gui.lbl_overview_contact_right = MagicMock()
    gui.lbl_foot_left_state = MagicMock()
    gui.lbl_foot_right_state = MagicMock()
    gui.lbl_gait_card_cycles_value = MagicMock()
    gui.lbl_gait_card_completeness_value = MagicMock()
    gui.lbl_overview_gait_cycles_usable = MagicMock()
    gui.lbl_overview_gait_cycles_completeness = MagicMock()
    gui._sync_demo_category_compare_strip = MagicMock()
    gui._refresh_real_to_sim_advanced_panel = MagicMock()
    gui._format_dashboard_gait_phase = lambda phase: phase
    gui._format_contact_state = StableWalkGUI._format_contact_state
    gui._resolved_gait_cycle_count = lambda: (2, 2)

    StableWalkGUI._update_gait_cycle_panel(gui, result)

    left = gui.lbl_gait_card_contact_left.configure.call_args.kwargs["text"]
    right = gui.lbl_gait_card_contact_right.configure.call_args.kwargs["text"]
    assert left != "Not computed yet"
    assert right != "Not computed yet"
    assert left.strip()
    assert right.strip()
