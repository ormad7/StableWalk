"""Window-resize coalescing — ignore jitter, defer work until drag settles."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.ui.tk.layout_playback import should_defer_layout
from stablewalk.ui.tk.resize_coalesce import (
    FIGURE_RESIZE_SETTLE_MS,
    RESIZE_IGNORE_PX,
    RESIZE_SETTLE_MS,
    resize_drag_active,
    size_change_is_noise,
)


def test_resize_settle_is_trailing_and_ignores_jitter() -> None:
    assert RESIZE_SETTLE_MS >= 120
    assert FIGURE_RESIZE_SETTLE_MS >= 120
    assert RESIZE_IGNORE_PX >= 4
    assert size_change_is_noise(800, 600, 803, 602)
    assert not size_change_is_noise(800, 600, 820, 600)


def test_layout_deferred_during_live_resize_not_only_playback() -> None:
    gui = SimpleNamespace(playing=False, _force_layout_once=False, _ui_resize_active=True)
    assert resize_drag_active(gui) is True
    assert should_defer_layout(gui) is True
    gui._ui_resize_active = False
    assert should_defer_layout(gui) is False


def test_responsive_window_debounce_matches_settle() -> None:
    from stablewalk.ui.tk.dashboard_responsive import _RESIZE_DEBOUNCE_MS

    assert _RESIZE_DEBOUNCE_MS == RESIZE_SETTLE_MS


def test_figure_resize_debounce_matches_settle() -> None:
    from stablewalk.ui.tk.dashboard_layout import _FIGURE_RESIZE_DEBOUNCE_MS

    assert _FIGURE_RESIZE_DEBOUNCE_MS == FIGURE_RESIZE_SETTLE_MS
