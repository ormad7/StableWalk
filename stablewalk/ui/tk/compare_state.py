"""Strict Compare tab load-state machine (stability-critical).

States
------
IDLE       — no active Compare load; may show empty / prior chrome
LOADING    — exactly one background worker may run
READY      — both Session A/B snapshots are cached; reopen is paint-only
FAILED     — last load failed or timed out; one clean retry allowed
CANCELLED  — user or leave-tab cancelled an in-flight load

Rules
-----
- Never start a second worker while LOADING.
- READY with cache → show immediately (no MediaPipe / OpenSim / decode).
- Worker results carry a generation id; stale gens are ignored.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Soft deadline for one Compare load attempt (seconds).
COMPARE_LOAD_TIMEOUT_S = 15.0

COMPARE_TIMEOUT_MESSAGE = (
    "Comparison could not be loaded. The rest of the application is still available."
)
COMPARE_ALREADY_LOADING_MESSAGE = "Comparison is already loading"
COMPARE_SESSION_B_NOT_ANALYZED = "Session B has not been analyzed yet."
COMPARE_SESSION_A_NOT_ANALYZED = "Session A has not been analyzed yet."

# Workflow copy when Session B is missing — Compare must never show blank panels.
SESSION_B_WAITING_TITLE = "Waiting for Session B"
SESSION_B_NEXT_ACTION = "Next: choose “Analyze Session B” or “Load recording…”."
SESSION_B_WORKFLOW_TITLE = "How to create Session B"
SESSION_B_WORKFLOW_STEPS: tuple[str, ...] = (
    "1. Choose Analyze Session B or Load recording…",
    "2. Select a recording different from Session A",
    "3. Wait — Video B, Skeleton B, differences, and metrics appear automatically",
)
SESSION_B_GUIDANCE_TEXT = "\n".join(SESSION_B_WORKFLOW_STEPS)
PLACEHOLDER_VIDEO_TITLE = SESSION_B_WAITING_TITLE
PLACEHOLDER_SKELETON_TITLE = SESSION_B_WAITING_TITLE
PLACEHOLDER_VIDEO_DETAIL = (
    f"{SESSION_B_NEXT_ACTION}\nVideo B appears automatically after analysis."
)
PLACEHOLDER_SKELETON_DETAIL = (
    f"{SESSION_B_NEXT_ACTION}\n"
    "Skeleton B appears here automatically when analysis finishes."
)
PLACEHOLDER_DIFF_GRAPH_DETAIL = (
    "Both analyzed sessions are required. No difference curve is shown until Session B is analyzed."
)
PLACEHOLDER_DIFF_PATH_DETAIL = (
    "Trajectory and heatmap differences appear here after Session B is analyzed."
)

# Metrics / Difference Summary when Session B is not loaded — never show blank Δ cells.
SESSION_B_METRICS_PLACEHOLDER_TITLE = SESSION_B_WAITING_TITLE
SESSION_B_METRICS_PLACEHOLDER_BODY = (
    f"{SESSION_B_GUIDANCE_TEXT}\n\n"
    "Only values available in both analyzed sessions are compared."
)

# Side-by-side dual video + skeleton + synchronized playback.
# Set False only for constrained environments that need static metrics-only Compare.
ENABLE_DETAILED_COMPARE = True


class CompareState(str, Enum):
    IDLE = "IDLE"
    LOADING = "LOADING"
    READY = "READY"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


def log_compare_event(gui: Any, event: str, **fields: Any) -> None:
    """Structured Compare lifecycle log with system counters when available."""
    parts = [f"Compare {event}"]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    try:
        from stablewalk.ui.tk.gui_memory import (
            approximate_rss_mb,
            collect_active_timer_ids,
            collect_owned_figures,
        )

        timers = collect_active_timer_ids(gui) if gui is not None else []
        figs = collect_owned_figures(gui) if gui is not None else []
        parts.append(f"threads={threading.active_count()}")
        parts.append(f"after_callbacks={len(timers)}")
        parts.append(f"figures={len(figs)}")
        rss = approximate_rss_mb()
        if rss is not None:
            parts.append(f"rss_mb={rss:.1f}")
    except Exception:
        parts.append(f"threads={threading.active_count()}")
    logger.info(" | ".join(str(p) for p in parts))


def compare_error_log_path() -> Any:
    """``logs/compare_error.log`` under the project root (created on demand)."""
    from pathlib import Path

    try:
        from stablewalk.config import PROJECT_ROOT

        base = Path(PROJECT_ROOT)
    except Exception:
        base = Path(__file__).resolve().parents[3]
    return base / "logs" / "compare_error.log"


def log_compare_exception(gui: Any, where: str, exc: BaseException) -> str:
    """Write full traceback to logs/compare_error.log; return short status text."""
    import traceback
    from datetime import datetime, timezone

    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    path = compare_error_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"\n--- {datetime.now(timezone.utc).isoformat()} | {where} ---\n"
            )
            fh.write(tb)
    except Exception:
        logger.exception("Failed to write compare_error.log")
    logger.error("Compare error in %s: %s\n%s", where, exc, tb)
    log_compare_event(gui, "exception", where=where, error=type(exc).__name__)
    return f"Compare error: {exc}. See logs/compare_error.log"


__all__ = [
    "COMPARE_ALREADY_LOADING_MESSAGE",
    "COMPARE_LOAD_TIMEOUT_S",
    "COMPARE_SESSION_A_NOT_ANALYZED",
    "COMPARE_SESSION_B_NOT_ANALYZED",
    "COMPARE_TIMEOUT_MESSAGE",
    "CompareState",
    "ENABLE_DETAILED_COMPARE",
    "PLACEHOLDER_SKELETON_DETAIL",
    "PLACEHOLDER_SKELETON_TITLE",
    "PLACEHOLDER_VIDEO_DETAIL",
    "PLACEHOLDER_VIDEO_TITLE",
    "PLACEHOLDER_DIFF_GRAPH_DETAIL",
    "PLACEHOLDER_DIFF_PATH_DETAIL",
    "SESSION_B_METRICS_PLACEHOLDER_BODY",
    "SESSION_B_METRICS_PLACEHOLDER_TITLE",
    "SESSION_B_NEXT_ACTION",
    "SESSION_B_GUIDANCE_TEXT",
    "SESSION_B_WAITING_TITLE",
    "SESSION_B_WORKFLOW_STEPS",
    "SESSION_B_WORKFLOW_TITLE",
    "compare_error_log_path",
    "log_compare_event",
    "log_compare_exception",
]
