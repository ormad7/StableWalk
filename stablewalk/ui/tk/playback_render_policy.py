"""Playback render policy — which surfaces may update each play frame.

StableWalk uses Tkinter + Matplotlib (not Qt). Expensive artists are reused
via pooled limb meshes, cached trajectory paths, and blit playheads.

While ``playing`` is True, only these surfaces may repaint:

- video frame
- 3D skeleton pose
- current trajectory tip (cached path mesh)
- timeline / transport cursor
- Overview 3D Coordinate Inspector
- live gait phase indicator (precomputed cache lookup)

Everything else (metric cards, graph geometries, legends, statistics, tabs,
layouts, Joint Information, compact HUD KPIs) stays static until pause,
joint change, or an explicit dirty flush.

While scrubbing, graph *geometry* stays frozen (playheads move only). Live
HUD widgets (tip / inspector / phase) are capped at ``MAX_LIVE_WIDGET_HZ``
even when the play clock targets ``TARGET_PLAY_HZ``.
"""

from __future__ import annotations

from typing import Any, Container

# Authoritative live set for Overview play ticks.
LIVE_PLAY_SURFACES: frozenset[str] = frozenset(
    {
        "video",
        "skeleton",
        "trajectory_tip",
        "timeline_cursor",
        "coordinate_inspector",
        "gait_phase_indicator",
    }
)

# Surfaces that must never rebuild geometry / Tk chrome during play.
STATIC_WHILE_PLAYING: frozenset[str] = frozenset(
    {
        "metric_cards",
        "graphs",
        "tabs",
        "labels",
        "legends",
        "statistics",
        "layouts",
        "joint_information",
        "clinical_hud",
        "status_bar",
        "panels_trees",
        "motion_stats",
    }
)

# Target UI cadence (spend-compensated timer aims near this until source FPS is known).
TARGET_PLAY_HZ = 60
# Live HUD / tip / inspector / phase widgets — never above this while playing.
MAX_LIVE_WIDGET_HZ = 30
# Chart playheads / blit cursors while playing (geometry stays frozen).
PLAY_CHART_HZ = 30
# Soft RSS target for a typical analyzed session (trim caches when exceeded).
TYPICAL_SESSION_RSS_BUDGET_MB = 500.0


def is_live_surface(name: str) -> bool:
    return name in LIVE_PLAY_SURFACES


def allow_play_surface(name: str, *, playing: bool) -> bool:
    """Return True when ``name`` may update for the current play state."""
    if not playing:
        return True
    return name in LIVE_PLAY_SURFACES


def playing_skips_static(gui: Any, surface: str) -> bool:
    """True when ``gui`` is playing and ``surface`` must stay frozen."""
    if not bool(getattr(gui, "playing", False)):
        return False
    return surface in STATIC_WHILE_PLAYING or surface not in LIVE_PLAY_SURFACES


def is_scrubbing(gui: Any) -> bool:
    """True while the timeline slider / step seek is driving a frame sync."""
    return bool(getattr(gui, "_scrubbing", False))


def allow_graph_geometry_rebuild(gui: Any, *, force: bool = False) -> bool:
    """Graph series geometry may rebuild only when idle (not play / scrub).

    Play and scrub move cached playheads only — never recompute plot artists.
    """
    if force:
        return True
    if bool(getattr(gui, "playing", False)):
        return False
    if is_scrubbing(gui):
        return False
    return True


def rate_limited_due(
    gui: Any,
    key: str,
    *,
    max_hz: float,
    now: float | None = None,
) -> bool:
    """Return True at most ``max_hz`` times per second for ``key`` on ``gui``."""
    import time

    store = getattr(gui, "_rate_limit_ts", None)
    if not isinstance(store, dict):
        store = {}
        try:
            gui._rate_limit_ts = store
        except Exception:
            return True
    t = time.monotonic() if now is None else float(now)
    min_dt = 1.0 / max(float(max_hz), 1e-3)
    last = float(store.get(key, 0.0) or 0.0)
    if t - last < min_dt - 1e-9:
        return False
    store[key] = t
    return True


def live_widgets_due(gui: Any, *, now: float | None = None) -> bool:
    """Gate tip / inspector / phase updates to ``MAX_LIVE_WIDGET_HZ``."""
    return rate_limited_due(
        gui, "live_widgets", max_hz=float(MAX_LIVE_WIDGET_HZ), now=now
    )


def mark_analysis_dirty(gui: Any, *panel_ids: str) -> None:
    """Defer panel rebuilds until pause / tab activate."""
    dirty = getattr(gui, "_analysis_panel_dirty", None)
    if dirty is None:
        gui._analysis_panel_dirty = set()
        dirty = gui._analysis_panel_dirty
    for panel_id in panel_ids:
        dirty.add(panel_id)


def configure_text_if_changed(lbl: Any, text: str) -> bool:
    """Tk label text update only when the string actually changed."""
    if lbl is None:
        return False
    try:
        if str(lbl.cget("text")) == str(text):
            return False
        lbl.configure(text=text)
        return True
    except Exception:
        return False


def paint_signature_changed(gui: Any, key: str, signature: object) -> bool:
    """Return True when ``signature`` differs from the last paint for ``key``.

    Stores the new signature on ``gui._paint_signatures``. Use to skip
    full widget rebuilds when visible data is unchanged.
    """
    store = getattr(gui, "_paint_signatures", None)
    if store is None:
        store = {}
        try:
            gui._paint_signatures = store
        except Exception:
            return True
    if store.get(key) == signature:
        return False
    store[key] = signature
    return True


def clear_paint_signatures(gui: Any, *keys: str) -> None:
    """Drop cached paint signatures (all keys, or only ``keys``)."""
    store = getattr(gui, "_paint_signatures", None)
    if not isinstance(store, dict):
        return
    if not keys:
        store.clear()
        return
    for key in keys:
        store.pop(key, None)


def groups_allow_any(visible: Container[str] | None, *names: str) -> bool:
    if visible is None:
        return True
    return any(name in visible for name in names)
