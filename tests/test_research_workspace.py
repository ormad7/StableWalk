"""GUI wiring for the research comparison workspace."""

from __future__ import annotations

from pathlib import Path

from stablewalk.ui.tk.dashboard_notebook import ENABLE_COMPARE_TAB, TAB_COMPARE
from stablewalk.ui.tk.playback_active_tab import TAB_PLAYBACK_GROUPS


def test_compare_tab_enabled_for_research_workspace() -> None:
    assert ENABLE_COMPARE_TAB is True
    assert "compare" in TAB_PLAYBACK_GROUPS[TAB_COMPARE]


def test_research_gui_does_not_create_second_main_skeleton() -> None:
    text = (
        Path(__file__).resolve().parents[1] / "stablewalk/ui/tk/dashboard_research.py"
    ).read_text(encoding="utf-8")
    assert "setup_skeleton_axes" not in text
    assert "gait_motion" not in text or "slot.gait_motion" in text
    assert "run_pose=True" in text
    assert "slots_are_independent" in text
    assert "btn_research_compare" in text
    assert "reset_research_compare" in text
    assert "_research_token_a" in text
    assert "Enabled only when Video A and Video B have both finished" in text


def test_research_compare_button_stays_disabled_until_both_ready() -> None:
    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_research import _sync_research_actions

    compare = SimpleNamespace(configure=lambda **kw: states.append(kw.get("state")))
    states: list = []
    gui = SimpleNamespace(
        _research_slot_a=None,
        _research_slot_b=None,
        _research_busy_a=False,
        _research_busy_b=False,
        _research_comparison=None,
        btn_research_analyze_a=SimpleNamespace(configure=lambda **_k: None),
        btn_research_analyze_b=SimpleNamespace(configure=lambda **_k: None),
        btn_research_compare=compare,
        btn_research_export=SimpleNamespace(configure=lambda **_k: None),
        lbl_research_slot_a=None,
        lbl_research_slot_b=None,
    )
    _sync_research_actions(gui)
    assert states[-1] == "disabled"

    ready = SimpleNamespace(ready=True, sequence=object(), gait_motion=object(), cycles=object(), spectral=object(), asymmetry=object())
    other = SimpleNamespace(ready=True, sequence=object(), gait_motion=object(), cycles=object(), spectral=object(), asymmetry=object())
    gui._research_slot_a = ready
    gui._research_slot_b = other
    _sync_research_actions(gui)
    assert states[-1] == "normal"


def test_opening_compare_does_not_build_laboratory_controller() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from stablewalk.ui.tk.dashboard_notebook import TAB_COMPARE
    from stablewalk.ui.tk.dashboard_tab_lifecycle import ensure_tab_content

    ensure = MagicMock()
    gui = SimpleNamespace(
        _ensure_comparison_mode_loaded=ensure,
        _compare_page_notebook=SimpleNamespace(
            select=lambda: "research",
            tab=lambda _id, _key: "Research dataset",
        ),
        _comparison_mode=None,
    )
    ensure_tab_content(gui, TAB_COMPARE)
    ensure.assert_not_called()


def test_research_module_never_downloads() -> None:
    root = Path(__file__).resolve().parents[1] / "stablewalk" / "research"
    blob = "\n".join(p.read_text(encoding="utf-8") for p in root.glob("*.py"))
    assert "requests.get" not in blob
    assert "urllib.request.urlretrieve" not in blob
    assert "youtube" not in blob.lower()
