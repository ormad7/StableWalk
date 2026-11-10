"""Tk lifecycle guards — no second root / no pyplot ghost windows."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from stablewalk.ui.tk.gui_memory import dispose_orphan_matplotlib_figures
from stablewalk.ui.tk.tk_lifecycle import (
    assert_single_tk_root,
    canvas_widget_alive,
    close_pyplot_figures_quietly,
    safe_canvas_draw_idle,
    safe_messagebox,
)


def test_close_pyplot_figures_quietly_never_calls_figure_activator() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    num = fig.number
    with patch("matplotlib.pyplot.figure") as fig_fn:
        closed = close_pyplot_figures_quietly()
    fig_fn.assert_not_called()
    assert closed >= 1
    assert num not in plt.get_fignums()


def test_dispose_orphan_does_not_activate_figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    gui = SimpleNamespace(
        fig=None,
        fig_3d=None,
        fig_dof_traj=None,
        fig_dof_traj_overview=None,
        fig_contact_gait=None,
        fig_biomech=None,
        fig_joint_motion=None,
        fig_robot=None,
        _comparison_mode=None,
    )
    with patch("matplotlib.pyplot.figure") as fig_fn:
        dispose_orphan_matplotlib_figures(gui)
    fig_fn.assert_not_called()
    assert plt.get_fignums() == []


def test_safe_canvas_draw_skips_dead_widget() -> None:
    canvas = MagicMock()
    canvas.get_tk_widget.side_effect = Exception("destroyed")
    safe_canvas_draw_idle(canvas)
    canvas.draw_idle.assert_not_called()
    assert canvas_widget_alive(canvas) is False


def test_safe_messagebox_refuses_without_parent(capsys) -> None:
    with patch("tkinter._default_root", None):
        with patch("tkinter.messagebox.showinfo") as show:
            safe_messagebox("info", "T", "M", parent=None)
            show.assert_not_called()


def test_assert_single_tk_root_with_mock() -> None:
    root = MagicMock()
    root.winfo_toplevel.return_value = root
    with patch("tkinter._default_root", root):
        assert assert_single_tk_root(root) is True
