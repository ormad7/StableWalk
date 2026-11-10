"""Active-tab playback: hidden tabs stay dirty, visible tab owns rendering."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from stablewalk.ui.tk.dashboard_notebook import (
    TAB_BIOMECHANICS,
    TAB_MOTION,
    TAB_OVERVIEW,
)
from stablewalk.ui.tk.playback_active_tab import (
    TAB_PLAYBACK_GROUPS,
    mark_hidden_tab_panels_dirty,
    playback_groups_for_active_tab,
)


def test_overview_play_excludes_motion_and_biomech_groups() -> None:
    gui = SimpleNamespace(
        playing=True,
        _overview_traj_dock_visible=False,
        _overview_joint_motion_expanded=True,
    )
    with patch(
        "stablewalk.ui.tk.dashboard_tab_lifecycle.active_tab_id",
        return_value=TAB_OVERVIEW,
    ):
        groups = playback_groups_for_active_tab(gui)
    assert "video" in groups
    assert "skeleton" in groups
    assert "knee" not in groups
    assert "biomech" not in groups
    assert "joint_motion" not in groups
    assert groups == TAB_PLAYBACK_GROUPS[TAB_OVERVIEW]


def test_motion_play_excludes_overview_video() -> None:
    gui = SimpleNamespace(playing=True)
    with patch(
        "stablewalk.ui.tk.dashboard_tab_lifecycle.active_tab_id",
        return_value=TAB_MOTION,
    ):
        groups = playback_groups_for_active_tab(gui)
    assert "motion" in groups
    assert "knee" in groups
    assert "video" not in groups
    assert "biomech" not in groups


def test_playing_without_tab_paints_nothing() -> None:
    gui = SimpleNamespace(playing=True)
    with patch(
        "stablewalk.ui.tk.dashboard_tab_lifecycle.active_tab_id",
        return_value=None,
    ):
        assert playback_groups_for_active_tab(gui) == frozenset()


def test_playback_reuses_tab_id_cached_by_lifecycle() -> None:
    gui = SimpleNamespace(
        playing=True,
        _active_dashboard_tab_id=TAB_MOTION,
    )
    with patch(
        "stablewalk.ui.tk.dashboard_tab_lifecycle.active_tab_id"
    ) as resolve:
        groups = playback_groups_for_active_tab(gui)
    resolve.assert_not_called()
    assert groups == TAB_PLAYBACK_GROUPS[TAB_MOTION]


def test_hidden_tab_panels_marked_dirty_keep_shared_analysis() -> None:
    gui = SimpleNamespace(playing=True, _analysis_panel_dirty=set())
    with patch(
        "stablewalk.ui.tk.dashboard_tab_lifecycle.active_tab_id",
        return_value=TAB_OVERVIEW,
    ):
        mark_hidden_tab_panels_dirty(gui)
    assert "knee" in gui._analysis_panel_dirty
    assert "biomech" in gui._analysis_panel_dirty
    assert "contact" in gui._analysis_panel_dirty
    assert "compare" in gui._analysis_panel_dirty


def test_biomech_tab_does_not_dirty_its_own_charts() -> None:
    gui = SimpleNamespace(playing=True, _analysis_panel_dirty=set())
    with patch(
        "stablewalk.ui.tk.dashboard_tab_lifecycle.active_tab_id",
        return_value=TAB_BIOMECHANICS,
    ):
        mark_hidden_tab_panels_dirty(gui)
    assert "biomech" not in gui._analysis_panel_dirty
    assert "contact" not in gui._analysis_panel_dirty
    assert "knee" in gui._analysis_panel_dirty


def test_tab_deactivation_clears_real_contact_chart_cache() -> None:
    import inspect

    from stablewalk.ui.tk.dashboard_tab_lifecycle import deactivate_tab

    source = inspect.getsource(deactivate_tab)
    assert '"fig_contact_gait"' in source
    assert '"fig_contact"' not in source
