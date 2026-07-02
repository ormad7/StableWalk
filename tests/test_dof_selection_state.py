"""Tests for central DOF selection / active-point state."""

from __future__ import annotations

from stablewalk.ui.dof_selection import (
    DofSelectionState,
    GUI_DOF_ITEM_IDS,
    label_for_item,
)


def test_activate_item_adds_and_sets_active() -> None:
    state = DofSelectionState()
    assert state.activate_item("left_heel") is True
    assert "left_heel" in state.selected
    assert state.active_item_id == "left_heel"
    assert state.active_label() == "Left Heel"


def test_set_active_only_when_already_selected() -> None:
    state = DofSelectionState()
    state.set_selection({"right_hip", "left_heel"}, last_selected="right_hip")
    assert state.set_active("left_heel") is True
    assert state.active_item_id == "left_heel"
    assert state.set_active("right_knee") is False
    assert state.active_item_id == "left_heel"


def test_ensure_last_selected_picks_stable_item() -> None:
    state = DofSelectionState()
    state.selected = {"left_toe", "right_hip"}
    state.last_selected = "right_knee"
    state.ensure_last_selected()
    assert state.active_item_id in state.selected
    assert state.active_item_id == "right_hip"


def test_count_label_names_active_in_multi_select() -> None:
    state = DofSelectionState()
    state.set_selection({"right_hip", "left_heel"}, last_selected="left_heel")
    label = state.count_label()
    assert "2 selected" in label
    assert "Left Heel" in label


def test_count_label_single_shows_active_name() -> None:
    state = DofSelectionState()
    state.activate_item("right_toe")
    assert state.count_label() == f"Active: {label_for_item('right_toe')}"


def test_toggle_off_clears_active_when_needed() -> None:
    state = DofSelectionState()
    state.set_selection({"right_hip", "left_heel"}, last_selected="left_heel")
    state.toggle("left_heel")
    assert "left_heel" not in state.selected
    assert state.active_item_id == "right_hip"


def test_gui_item_order_for_restore() -> None:
    state = DofSelectionState()
    state.selected = {"left_heel", "right_hip"}
    state.last_selected = None
    state.ensure_last_selected()
    first_in_list = next(i for i in GUI_DOF_ITEM_IDS if i in state.selected)
    assert state.active_item_id == first_in_list
