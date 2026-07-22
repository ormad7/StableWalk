"""Motion Analysis contact/vGRF layout minsizes and figure fill helpers."""

from __future__ import annotations

from types import SimpleNamespace


def test_contact_gait_min_height_keeps_three_plots_readable() -> None:
    from stablewalk.ui.tk.dashboard_layout import (
        CONTACT_GAIT_CHART_MIN_H,
        MOTION_TOP_COLUMN_WEIGHTS,
        _MOTION_GRAPH_ROW_MINSIZE,
        _MOTION_KNEE_PANEL_MIN_W,
        _MOTION_METRICS_PANEL_MIN_W,
        _MOTION_TOP_ROW_MIN_H,
        _MOTION_TRAJ_PANEL_MIN_W,
    )

    assert CONTACT_GAIT_CHART_MIN_H >= 400
    assert _MOTION_TOP_ROW_MIN_H >= 450
    assert _MOTION_GRAPH_ROW_MINSIZE >= 240
    # Contact strip must be tall enough for contact + phase + Estimated Virtual GRF.
    assert CONTACT_GAIT_CHART_MIN_H >= 280
    assert MOTION_TOP_COLUMN_WEIGHTS == (3, 5, 2)
    assert sum(MOTION_TOP_COLUMN_WEIGHTS) == 10
    assert _MOTION_KNEE_PANEL_MIN_W >= 280
    assert _MOTION_TRAJ_PANEL_MIN_W >= 420
    assert _MOTION_METRICS_PANEL_MIN_W >= 180


def test_fit_figure_to_host_expands_width() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from stablewalk.ui.tk.dashboard_layout import _fit_figure_to_host

    fig = Figure(figsize=(4.0, 2.0), dpi=100)
    drawn: list[bool] = []

    class _Canvas:
        def get_tk_widget(self):
            return SimpleNamespace(
                winfo_width=lambda: 100,
                winfo_height=lambda: 80,
                configure=lambda **_kwargs: None,
            )

        def resize(self, event) -> None:
            dpi = float(fig.get_dpi() or 100.0)
            fig.set_size_inches(event.width / dpi, event.height / dpi, forward=True)

        def draw_idle(self) -> None:
            drawn.append(True)

    host = SimpleNamespace(
        update_idletasks=lambda: None,
        winfo_width=lambda: 900,
        winfo_height=lambda: 400,
    )
    canvas = _Canvas()
    ok = _fit_figure_to_host(canvas, fig, host=host, min_px=80)  # type: ignore[arg-type]
    assert ok
    w_in, h_in = fig.get_size_inches()
    dpi = float(fig.get_dpi() or 100.0)
    assert w_in * dpi >= 880
    assert h_in * dpi >= 380
    assert drawn
