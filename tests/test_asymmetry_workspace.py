"""GUI wiring helpers for the Motion Asymmetry page."""

from __future__ import annotations

from stablewalk.ui.tk.dashboard_notebook import TAB_MOTION
from stablewalk.ui.tk.playback_active_tab import TAB_PLAYBACK_GROUPS


def test_motion_tab_includes_asymmetry_playback_group() -> None:
    assert "asymmetry" in TAB_PLAYBACK_GROUPS[TAB_MOTION]


def test_no_duplicate_skeleton_on_asymmetry_module() -> None:
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "stablewalk/ui/tk/dashboard_asymmetry.py").read_text(encoding="utf-8")
    assert "setup_skeleton_axes" not in text
    assert "FigureCanvasTkAgg" in text
    assert "second 3D" in text or "does not create a second 3D" in text


def test_insufficient_asymmetry_copy_uses_report_warnings() -> None:
    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_asymmetry import insufficient_asymmetry_copy

    none_copy = insufficient_asymmetry_copy(None)
    assert "walking video" in none_copy.lower()
    report = SimpleNamespace(warnings=["Need left and right gait cycles."])
    text = insufficient_asymmetry_copy(report)
    assert "Insufficient data" in text
    assert "gait cycles" in text

    failed = insufficient_asymmetry_copy(
        SimpleNamespace(failed=True, warnings=["Analysis failed: boom"])
    )
    assert "Analysis failed" in failed
    assert "Insufficient data" not in failed


def test_seek_unusual_interval_uses_recording_time() -> None:
    from types import SimpleNamespace

    from stablewalk.analysis.asymmetry3d.quality import UnusualInterval
    from stablewalk.ui.tk.dashboard_asymmetry import _seek_unusual_interval

    sought: list[float] = []
    interval = UnusualInterval(
        start_s=1.24,
        end_s=1.48,
        phase_start_pct=18.0,
        phase_end_pct=40.0,
        channel="knee_flexion",
        peak_delta=16.0,
        side="right",
        start_frame=31,
    )
    box = SimpleNamespace(curselection=lambda: (0,))
    gui = SimpleNamespace(
        lst_asymmetry_intervals=box,
        _asymmetry_interval_items=[interval],
        gait_motion=None,
        _seek_motion_workspace_time=sought.append,
    )
    _seek_unusual_interval(gui)
    assert len(sought) == 1
    assert abs(float(sought[0]) - 1.24) < 1e-9
    assert gui._asymmetry_last_seek[2] == "right"
    assert gui._asymmetry_seek_side == "right"


def test_quality_explanation_covers_ground_and_cycles() -> None:
    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_asymmetry import _quality_explanation

    ground = SimpleNamespace(
        name="foot_clearance_ground",
        quality="unavailable",
        reliable=False,
        left=None,
        right=None,
        n_left_cycles=2,
        n_right_cycles=2,
        note="",
    )
    text = _quality_explanation(ground).lower()
    assert "withheld" in text
    assert "calibration" in text or "force-plate" in text or "floor" in text

    missing = SimpleNamespace(
        name="knee_flexion_rom",
        quality="insufficient",
        reliable=False,
        left=None,
        right=12.0,
        n_left_cycles=0,
        n_right_cycles=2,
        note="Need usable flexion series.",
    )
    expl = _quality_explanation(missing)
    assert "insufficient" in expl.lower()
    assert "cycle" in expl.lower()
