"""Display-step control must not change pose extraction or gait timing."""

from __future__ import annotations

from stablewalk.ui.dof_step_preview import StepConfig
from stablewalk.ui.theme import REFRESH_INTERVAL_DEFAULT


def test_sampling_choices_only_change_display_lookahead() -> None:
    assert REFRESH_INTERVAL_DEFAULT == "0.5 s"
    slow = StepConfig.from_refresh_label("0.5 s")
    fast = StepConfig.from_refresh_label("0.25 s")
    assert slow.kind == "time" and fast.kind == "time"
    assert slow.interval_s == 0.5
    assert fast.interval_s == 0.25


def test_apply_refresh_interval_does_not_throttle_analysis(tmp_path=None) -> None:
    import inspect

    from stablewalk.ui.tk.app import StableWalkGUI

    src = inspect.getsource(StableWalkGUI._apply_refresh_interval)
    assert "_traj_redraw_interval_s = 0.0" in src
    assert "_ui_sync_interval_s = 0.0" in src
    src_step = inspect.getsource(StableWalkGUI._step_config)
    assert "from_refresh_label" in src_step
    pipeline = inspect.getsource(StableWalkGUI._apply_refresh_interval)
    assert "max_frames" not in pipeline
    assert "pose" not in pipeline.lower() or "Never changes pose" in pipeline
