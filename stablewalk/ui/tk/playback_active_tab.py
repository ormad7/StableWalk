"""Playback paints only the active dashboard tab.

Analysis caches stay shared. Hidden tabs are marked dirty and catch up on
activate / pause — they are not rasterized while off-screen.
"""

from __future__ import annotations

from typing import Any

from stablewalk.ui.tk.dashboard_notebook import (
    TAB_ADVANCED,
    TAB_BIOMECHANICS,
    TAB_COMPARE,
    TAB_MOTION,
    TAB_OVERVIEW,
    TAB_RESULTS_SUMMARY,
)

# Render groups owned by each notebook page (see ``_group_visible``).
TAB_PLAYBACK_GROUPS: dict[str, frozenset[str]] = {
    TAB_OVERVIEW: frozenset({"overview", "video", "skeleton"}),
    TAB_MOTION: frozenset(
        {"motion", "knee", "trajectory", "joint_motion", "spectral", "asymmetry", "four_gait"}
    ),
    TAB_BIOMECHANICS: frozenset({"biomech", "contact"}),
    TAB_RESULTS_SUMMARY: frozenset({"summary"}),
    TAB_COMPARE: frozenset({"compare"}),
    TAB_ADVANCED: frozenset(),
}

ALL_PLAYBACK_GROUPS: frozenset[str] = frozenset(
    {
        "overview",
        "video",
        "skeleton",
        "joint_motion",
        "trajectory",
        "knee",
        "biomech",
        "contact",
        "summary",
        "compare",
        "motion",
        "spectral",
        "asymmetry",
        "four_gait",
    }
)

# Analysis panel ids to defer when the owning tab is hidden.
_GROUP_DIRTY_PANELS: dict[str, tuple[str, ...]] = {
    "knee": ("knee",),
    "motion": ("motion",),
    "joint_motion": ("joint_motion",),
    "biomech": ("biomech",),
    "contact": ("contact",),
    "compare": ("compare",),
    "summary": ("summary",),
}


def playback_groups_for_active_tab(gui: Any) -> frozenset[str]:
    """Panel groups that may paint for the selected notebook tab.

    While playing, an unresolved tab returns an empty set so we never rasterize
    every chart as a fallback. When paused and the notebook is not built yet,
    all groups are allowed so the first session paint can populate panels.
    """
    from stablewalk.ui.tk.dashboard_tab_lifecycle import active_tab_id, resolve_tab_id

    playing = bool(getattr(gui, "playing", False))
    # The lifecycle hook updates this value on every notebook selection change.
    # Reuse it from the playback clock instead of asking Tk to resolve the
    # selected widget on every frame.
    tab_id = getattr(gui, "_active_dashboard_tab_id", None)
    if tab_id is None:
        tab_id = active_tab_id(gui)
    if tab_id is None:
        getter = getattr(gui, "_active_tab_widget", None)
        widget = None
        if callable(getter):
            try:
                widget = getter()
            except Exception:
                widget = None
        tab_id = resolve_tab_id(gui, widget)
    if tab_id is None:
        return frozenset() if playing else ALL_PLAYBACK_GROUPS
    groups = set(TAB_PLAYBACK_GROUPS.get(tab_id, ()))
    if tab_id == TAB_OVERVIEW:
        if getattr(gui, "_overview_traj_dock_visible", False):
            groups.add("trajectory")
        if not playing and getattr(gui, "_overview_joint_motion_expanded", False):
            groups.add("joint_motion")
    return frozenset(groups)


def mark_hidden_tab_panels_dirty(gui: Any) -> None:
    """Keep shared analysis, but defer rendering for off-screen tab panels."""
    visible = playback_groups_for_active_tab(gui)
    dirty = getattr(gui, "_analysis_panel_dirty", None)
    if dirty is None:
        gui._analysis_panel_dirty = set()
        dirty = gui._analysis_panel_dirty
    pending: set[str] = set()
    for group, panels in _GROUP_DIRTY_PANELS.items():
        if group in visible:
            continue
        pending.update(panels)
    # This runs from the playback clock. Avoid mutating the same set on every
    # frame once all hidden owners are already marked for a deferred refresh.
    missing = pending.difference(dirty)
    if missing:
        dirty.update(missing)
