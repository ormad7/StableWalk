"""Interactive zoom/pan navigation for dashboard time-series charts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from stablewalk.ui.colors import COM

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

FIG_NAV_STATE_ATTR = "_chart_nav_state"
FIG_NAV_CIDS_ATTR = "_chart_nav_cids"
FIG_NAV_BLOCK_HOVER_ATTR = "_chart_nav_block_hover"

_WHEEL_ZOOM_BASE = 1.15
_MIN_X_SPAN_FRAC = 0.01
_SELECT_MIN_FRAC = 0.008


@dataclass
class ChartNavState:
    """Per-figure pan/zoom view state (survives axis redraw via fig.clear)."""

    home_xlim: tuple[float, float] | None = None
    view_xlim: tuple[float, float] | None = None
    panning: bool = False
    selecting: bool = False
    pan_ax: object | None = None
    pan_x0: float = 0.0
    pan_start_xlim: tuple[float, float] = (0.0, 1.0)
    select_ax: object | None = None
    select_x0: float = 0.0
    select_patch: object | None = None


def _get_state(fig: Figure) -> ChartNavState:
    state = getattr(fig, FIG_NAV_STATE_ATTR, None)
    if state is None:
        state = ChartNavState()
        setattr(fig, FIG_NAV_STATE_ATTR, state)
    return state


def chart_nav_is_active(fig: Figure) -> bool:
    """True while the user is panning or rubber-band selecting."""
    state = getattr(fig, FIG_NAV_STATE_ATTR, None)
    if state is None:
        return False
    return bool(state.panning or state.selecting)


def reset_chart_navigation(fig: Figure) -> None:
    """Clear saved zoom/pan (e.g. on new session load)."""
    setattr(fig, FIG_NAV_STATE_ATTR, ChartNavState())


def _time_axes(fig: Figure) -> list[Axes]:
    axes = [ax for ax in fig.axes if getattr(ax, "get_xlim", None) is not None]
    axes.sort(key=lambda a: a.get_position().y0, reverse=True)
    return axes


def _merged_xlim(axes: list[Axes]) -> tuple[float, float] | None:
    if not axes:
        return None
    x0 = min(ax.get_xlim()[0] for ax in axes)
    x1 = max(ax.get_xlim()[1] for ax in axes)
    if x1 <= x0:
        return None
    return float(x0), float(x1)


def _clamp_xlim(
    xlim: tuple[float, float],
    home: tuple[float, float] | None,
) -> tuple[float, float]:
    lo, hi = float(xlim[0]), float(xlim[1])
    if hi <= lo:
        hi = lo + 1e-6
    if home is not None:
        home_lo, home_hi = home
        span = hi - lo
        min_span = max((home_hi - home_lo) * _MIN_X_SPAN_FRAC, 1e-6)
        span = max(span, min_span)
        if span >= home_hi - home_lo:
            return float(home_lo), float(home_hi)
        lo = max(home_lo, min(lo, home_hi - span))
        hi = lo + span
    return lo, hi


def zoom_xlim_at(
    xlim: tuple[float, float],
    *,
    center: float,
    scale: float,
    home: tuple[float, float] | None,
) -> tuple[float, float]:
    """Zoom x-axis about ``center``; ``scale`` > 1 zooms in."""
    lo, hi = xlim
    width = hi - lo
    if width <= 0:
        return _clamp_xlim(xlim, home)
    rel = (center - lo) / width
    rel = max(0.0, min(1.0, rel))
    new_width = width / scale
    new_lo = center - rel * new_width
    new_hi = new_lo + new_width
    return _clamp_xlim((new_lo, new_hi), home)


def apply_chart_xlim(
    fig: Figure,
    xlim: tuple[float, float],
    *,
    home: tuple[float, float] | None = None,
    persist: bool = True,
) -> tuple[float, float]:
    """Apply x limits to every axes in ``fig`` and optionally persist as user view."""
    state = _get_state(fig)
    home_lim = home if home is not None else state.home_xlim
    clamped = _clamp_xlim(xlim, home_lim)
    for ax in _time_axes(fig):
        ax.set_xlim(clamped)
    if persist:
        state.view_xlim = clamped
    setattr(fig, FIG_NAV_BLOCK_HOVER_ATTR, False)
    return clamped


def capture_chart_home_xlim(fig: Figure) -> tuple[float, float] | None:
    """Record the full-data x span as the home view (after a chart redraw)."""
    state = _get_state(fig)
    merged = _merged_xlim(_time_axes(fig))
    if merged is not None:
        state.home_xlim = merged
    return state.home_xlim


def finalize_chart_view(fig: Figure, canvas: FigureCanvasTkAgg | None = None) -> None:
    """After chart content is drawn: capture home limits and restore user zoom."""
    state = _get_state(fig)
    capture_chart_home_xlim(fig)
    target = state.view_xlim or state.home_xlim
    if target is not None:
        apply_chart_xlim(fig, target, persist=bool(state.view_xlim))
    if canvas is not None:
        canvas.draw_idle()


def reset_chart_view(fig: Figure, canvas: FigureCanvasTkAgg | None = None) -> None:
    """Double-click reset — return to full timeline."""
    state = _get_state(fig)
    state.view_xlim = None
    _clear_select_patch(state)
    if state.home_xlim is not None:
        apply_chart_xlim(fig, state.home_xlim, persist=False)
    if canvas is not None:
        canvas.draw_idle()


def _clear_select_patch(state: ChartNavState) -> None:
    patch = state.select_patch
    if patch is not None:
        try:
            patch.remove()
        except Exception:
            pass
    state.select_patch = None
    state.selecting = False
    state.select_ax = None


def _set_select_patch(state: ChartNavState, ax: Axes, x0: float, x1: float) -> None:
    from matplotlib.patches import Rectangle

    lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
    y0, y1 = ax.get_ylim()
    if state.select_patch is None or state.select_ax is not ax:
        _clear_select_patch(state)
        state.select_patch = Rectangle(
            (lo, y0),
            max(hi - lo, 1e-9),
            y1 - y0,
            facecolor=COM,
            edgecolor=COM,
            alpha=0.14,
            linewidth=1.0,
            zorder=40,
        )
        ax.add_patch(state.select_patch)
        state.select_ax = ax
    else:
        patch = state.select_patch
        patch.set_x(lo)
        patch.set_width(max(hi - lo, 1e-9))
        patch.set_y(y0)
        patch.set_height(y1 - y0)


def attach_chart_navigation(
    fig: Figure,
    canvas: FigureCanvasTkAgg,
) -> None:
    """Wire wheel zoom, drag pan, shift-drag selection zoom, and double-click reset."""
    existing = getattr(fig, FIG_NAV_CIDS_ATTR, None)
    if existing:
        for cid in existing:
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass

    cids: list[int] = []

    def _draw() -> None:
        canvas.draw_idle()

    def _on_scroll(event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        state = _get_state(fig)
        axes = _time_axes(fig)
        if not axes:
            return
        cur = axes[0].get_xlim()
        home = state.home_xlim
        scale = _WHEEL_ZOOM_BASE if event.step > 0 else 1.0 / _WHEEL_ZOOM_BASE
        new_lim = zoom_xlim_at(cur, center=float(event.xdata), scale=scale, home=home)
        apply_chart_xlim(fig, new_lim)
        _draw()

    def _on_press(event) -> None:
        if getattr(event, "dblclick", False):
            reset_chart_view(fig, canvas)
            return
        if event.inaxes is None or event.xdata is None:
            return
        state = _get_state(fig)
        shift = bool(getattr(event, "key", None) == "shift")
        if event.button == 1 and shift:
            state.selecting = True
            state.select_x0 = float(event.xdata)
            state.select_ax = event.inaxes
            setattr(fig, FIG_NAV_BLOCK_HOVER_ATTR, True)
            _set_select_patch(state, event.inaxes, state.select_x0, state.select_x0)
            _draw()
            return
        if event.button in (1, 3):
            state.panning = True
            state.pan_ax = event.inaxes
            state.pan_x0 = float(event.xdata)
            state.pan_start_xlim = event.inaxes.get_xlim()
            setattr(fig, FIG_NAV_BLOCK_HOVER_ATTR, True)

    def _on_motion(event) -> None:
        state = _get_state(fig)
        if state.selecting and state.select_ax is not None and event.xdata is not None:
            if event.inaxes is state.select_ax:
                _set_select_patch(state, state.select_ax, state.select_x0, float(event.xdata))
                _draw()
            return
        if not state.panning or state.pan_ax is None:
            return
        if event.xdata is None:
            return
        if event.inaxes is not state.pan_ax:
            return
        dx = float(event.xdata) - state.pan_x0
        lo, hi = state.pan_start_xlim
        apply_chart_xlim(fig, (lo - dx, hi - dx))
        _draw()

    def _on_release(event) -> None:
        state = _get_state(fig)
        if state.selecting:
            if state.select_ax is not None and event.xdata is not None:
                x0, x1 = state.select_x0, float(event.xdata)
                _clear_select_patch(state)
                home = state.home_xlim
                if home is not None:
                    span = abs(home[1] - home[0])
                    if abs(x1 - x0) >= span * _SELECT_MIN_FRAC:
                        apply_chart_xlim(fig, (x0, x1))
            else:
                _clear_select_patch(state)
            setattr(fig, FIG_NAV_BLOCK_HOVER_ATTR, False)
            _draw()
            return
        if state.panning:
            state.panning = False
            state.pan_ax = None
            setattr(fig, FIG_NAV_BLOCK_HOVER_ATTR, False)

    cids.append(canvas.mpl_connect("scroll_event", _on_scroll))
    cids.append(canvas.mpl_connect("button_press_event", _on_press))
    cids.append(canvas.mpl_connect("motion_notify_event", _on_motion))
    cids.append(canvas.mpl_connect("button_release_event", _on_release))
    setattr(fig, FIG_NAV_CIDS_ATTR, cids)


__all__ = [
    "ChartNavState",
    "apply_chart_xlim",
    "attach_chart_navigation",
    "capture_chart_home_xlim",
    "chart_nav_is_active",
    "finalize_chart_view",
    "reset_chart_navigation",
    "reset_chart_view",
    "zoom_xlim_at",
]
