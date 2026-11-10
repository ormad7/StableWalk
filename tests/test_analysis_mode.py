"""Examples is labeled preview; Analyze is full source (no DEMO_MAX_FRAMES)."""

from __future__ import annotations

import inspect

from stablewalk import config
from stablewalk.ui.analysis_mode import (
    MODE_FULL,
    MODE_PREVIEW,
    analysis_modes_comparable,
    format_analysis_mode_label,
    resolve_compare_analysis_mode,
    resolve_gui_max_frames,
)


def test_full_analysis_never_caps_demo_or_generic_video() -> None:
    demo_n, demo_mode = resolve_gui_max_frames(mode=MODE_FULL, is_demo=True)
    other_n, other_mode = resolve_gui_max_frames(mode=MODE_FULL, is_demo=False)
    assert demo_n is None and demo_mode == MODE_FULL
    assert other_n is None and other_mode == MODE_FULL


def test_preview_uses_demo_cap_and_is_labeled() -> None:
    n, mode = resolve_gui_max_frames(mode=MODE_PREVIEW, is_demo=True)
    assert mode == MODE_PREVIEW
    assert n == int(config.DEMO_MAX_FRAMES)
    label = format_analysis_mode_label(mode, n)
    assert "Preview" in label
    assert str(n) in label


def test_analyze_and_examples_call_sites() -> None:
    from stablewalk.ui.tk.app import StableWalkGUI

    full = inspect.getsource(StableWalkGUI._run_full_analysis)
    demo = inspect.getsource(StableWalkGUI._load_demo_gait)
    load = inspect.getsource(StableWalkGUI._load_video)
    assert 'analysis_mode="full"' in full
    assert 'analysis_mode="preview"' in demo
    assert "resolve_gui_max_frames" in load
    assert "DEMO_MAX_FRAMES" in load or "MODE_PREVIEW" in load
    # Full analysis must not inherit the silent demo cap.
    assert "if detected_demo is not None:\n                max_frames = DEMO_MAX_FRAMES" not in load


def test_compare_rejects_full_versus_preview() -> None:
    ok, reason = analysis_modes_comparable(MODE_FULL, MODE_PREVIEW, max_frames_b=900)
    assert ok is False
    assert "matching analysis modes" in reason.lower() or "full-versus-preview" in reason.lower()
    ok_same, _ = analysis_modes_comparable(MODE_FULL, MODE_FULL)
    assert ok_same is True
    ok_preview, _ = analysis_modes_comparable(
        MODE_PREVIEW, MODE_PREVIEW, max_frames_a=900, max_frames_b=900
    )
    assert ok_preview is True


def test_compare_session_b_defaults_to_full_not_silent_preview() -> None:
    assert resolve_compare_analysis_mode() == MODE_FULL
    assert resolve_compare_analysis_mode(session_a_mode=MODE_PREVIEW) == MODE_PREVIEW
    from stablewalk.ui.tk import session_cache
    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController

    src = inspect.getsource(session_cache.run_independent_session_b_analysis)
    assert "DEMO_MAX_FRAMES" not in src
    prepare_src = inspect.getsource(
        ComparisonModeController._prepare_demo_analysis_options
    )
    window_src = inspect.getsource(
        ComparisonModeController._apply_session_b_analysis_window
    )
    assert "DEMO_MAX_FRAMES" not in prepare_src
    assert "DEMO_MAX_FRAMES" not in window_src
    assert "resolve_gui_max_frames" in window_src
    assert "resolve_compare_analysis_mode" in window_src
