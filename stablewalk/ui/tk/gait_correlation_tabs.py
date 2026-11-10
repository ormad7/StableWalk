"""Tab titles and cache key for Gait Correlation & Spectral Analysis."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SECTION_TITLE = "Gait Correlation & Spectral Analysis"

# Inner tab title → four-gait view id. Asymmetry is handled separately.
FOUR_GAIT_TAB_VIEWS: dict[str, str] = {
    "Four Motion Signals": "time",
    "Correlation Matrix": "matrix",
    "Autocorrelation & Cross-Correlation": "corr",
    "Frequency Analysis": "freq",
    "Spectral Correlation": "compare",
}

ASYMMETRY_TAB = "Asymmetry Summary"

FOUR_GAIT_TAB_ORDER: tuple[str, ...] = tuple(FOUR_GAIT_TAB_VIEWS.keys())
SECTION_TAB_ORDER: tuple[str, ...] = FOUR_GAIT_TAB_ORDER + (ASYMMETRY_TAB,)


def recording_cache_key(recording: Any) -> tuple[Any, ...]:
    """Stable per-video key so tab switches do not recompute analysis."""
    if recording is None:
        return ("empty",)
    snaps = getattr(recording, "snapshots", None) or []
    n = int(getattr(recording, "frame_count", 0) or len(snaps))
    return (
        id(recording),
        n,
        str(getattr(recording, "source", "") or ""),
        float(getattr(recording, "fps", 0.0) or 0.0),
    )


def motion_inner_title(gui: Any) -> str:
    nb = getattr(gui, "_motion_page_notebook", None)
    if nb is None:
        return ""
    try:
        return str(nb.tab(nb.select(), "text")).strip()
    except Exception:
        return ""


def gait_corr_inner_title(gui: Any) -> str:
    nb = getattr(gui, "_gait_corr_notebook", None)
    if nb is None:
        return ""
    try:
        return str(nb.tab(nb.select(), "text")).strip()
    except Exception:
        return ""


def section_selected(gui: Any) -> bool:
    return motion_inner_title(gui) == SECTION_TITLE


def four_gait_tab_selected(gui: Any) -> bool:
    if not section_selected(gui):
        return False
    return gait_corr_inner_title(gui) in FOUR_GAIT_TAB_VIEWS


def asymmetry_tab_selected(gui: Any) -> bool:
    if not section_selected(gui):
        return False
    return gait_corr_inner_title(gui) == ASYMMETRY_TAB


def select_notebook_tab(notebook: Any, title: str) -> bool:
    """Select a ttk.Notebook tab by its visible title."""
    if notebook is None:
        return False
    wanted = str(title).strip()
    try:
        for tab_id in notebook.tabs():
            if str(notebook.tab(tab_id, "text")).strip() == wanted:
                notebook.select(tab_id)
                return True
    except Exception:
        return False
    return False


def select_gait_corr_tab(gui: Any, title: str) -> bool:
    """Select a Gait Correlation sub-tab, activating the Motion section first."""
    if not section_selected(gui):
        if not select_notebook_tab(getattr(gui, "_motion_page_notebook", None), SECTION_TITLE):
            return False
    if not select_notebook_tab(getattr(gui, "_gait_corr_notebook", None), title):
        return False
    sync = getattr(gui, "_sync_gait_corr_tab", None)
    if callable(sync):
        try:
            sync()
        except Exception:
            logger.exception("gait correlation tab sync failed")
            return False
    return True
