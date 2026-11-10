"""Advanced Analysis dialog must format without the leftover import crash."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.ui.tk.app import StableWalkGUI


def test_format_advanced_analysis_details_uses_real_modules() -> None:
    result = SimpleNamespace(
        validity=SimpleNamespace(comparable_score="YES"),
        metrics=[],
        score=64.2,
        completeness_pct=80.0,
        domain_evidence={"pelvis_stability": "pelvis sway within session range"},
        data_limitations=["MediaPipe landmarks are not clinical mocap."],
    )
    text = StableWalkGUI._format_advanced_analysis_details(object(), result)
    assert "Advanced Gait Analysis" in text
    assert "Legacy composite" in text
    assert "64" in text
    assert "YES" in text
    assert "MediaPipe landmarks are not clinical mocap." in text


def test_format_advanced_analysis_details_imports_are_live() -> None:
    import inspect

    source = inspect.getsource(StableWalkGUI._format_advanced_analysis_details)
    assert "stablewalk.analysis.stability_scoring import build_gait_analysis_summary" not in source
    assert "stablewalk.analysis.validity import" not in source
    assert "stablewalk.analysis.stability_validity" in source
