"""Overview session badge labels must survive clinical summary construction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

tk = pytest.importorskip("tkinter")


def test_clinical_summary_does_not_clear_session_badge_labels() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    try:
        from stablewalk.ui.tk.dashboard_sections import (
            build_overview_clinical_summary,
            build_overview_session_badge,
        )

        gui = SimpleNamespace()
        badge_host = tk.Frame(root)
        build_overview_session_badge(gui, badge_host)
        assert "video_name" in gui._overview_session_badge_labels
        gui._overview_session_badge_labels["video_name"].configure(text="Normal Gait")

        parent = tk.Frame(root)
        build_overview_clinical_summary(gui, parent)

        assert "video_name" in gui._overview_session_badge_labels
        assert gui._overview_session_badge_labels["video_name"].cget("text") == "Normal Gait"
    finally:
        root.destroy()
