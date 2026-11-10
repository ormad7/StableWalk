"""GUI wiring helpers for the Spectral Analysis page."""

from __future__ import annotations

from stablewalk.ui.tk.dashboard_spectral import _id_for_label, _JOINT_TO_CHANNEL, _label_for_id
from stablewalk.ui.tk.playback_active_tab import TAB_PLAYBACK_GROUPS
from stablewalk.ui.tk.dashboard_notebook import TAB_MOTION


def test_joint_selector_maps_to_flexion_channels() -> None:
    assert _JOINT_TO_CHANNEL["left_knee"] == "dof:left_knee_flexion"
    assert _JOINT_TO_CHANNEL["right_hip"] == "dof:right_hip_flexion"
    assert _JOINT_TO_CHANNEL["left_heel"] == "pos:left_heel:y"


def test_channel_label_roundtrip() -> None:
    cid = "dof:left_knee_flexion"
    assert _id_for_label(_label_for_id(cid)) == cid


def test_motion_tab_includes_spectral_playback_group() -> None:
    assert "spectral" in TAB_PLAYBACK_GROUPS[TAB_MOTION]


def test_insufficient_spectral_copy_uses_report_warnings() -> None:
    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_spectral import insufficient_spectral_copy
    from stablewalk.ui.tk.dashboard_spectral_features import insufficient_features_copy

    none_copy = insufficient_spectral_copy(None)
    assert "walking video" in none_copy.lower()
    report = SimpleNamespace(warnings=["Clip duration 0.50 s is shorter than 2.0 s; periodic estimates are unreliable."])
    text = insufficient_spectral_copy(report)
    assert "Insufficient data" in text
    assert "0.50 s" in text
    feat = insufficient_features_copy(report)
    assert "Insufficient data" in feat
    assert "0.50 s" in feat
    cycles = SimpleNamespace(warnings=["insufficient cycles: 1 detected gait cycle(s); need ≥ 2."])
    cycle_copy = insufficient_spectral_copy(cycles)
    assert cycle_copy.lower().startswith("insufficient cycles")
    feat_cycle = insufficient_features_copy(cycles)
    assert feat_cycle.lower().startswith("insufficient cycles")
