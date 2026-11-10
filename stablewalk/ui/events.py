"""Lightweight UI event bus — separates analysis completion from display updates.

Events
------
``analysis_ready``
    Session analysis finished once. Payload: ``SessionAnalysisCache``.
``frame_changed``
    Playback / scrub committed a new frame. Payload: ``FrameChangedEvent``.
``session_cleared``
    Analysis results were discarded (welcome / new load).

Subscribers must be display-only: no ``analyze_*`` / pipeline calls.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

ANALYSIS_READY = "analysis_ready"
FRAME_CHANGED = "frame_changed"
SESSION_CLEARED = "session_cleared"

EventHandler = Callable[..., None]


@dataclass(frozen=True, slots=True)
class FrameChangedEvent:
    """Authoritative playback position for display subscribers.

    ``list_pos`` / ``frame_index`` are 0-based playback list indices.
    ``source_frame_index`` is the original video / PoseFrame id when known.
    ``time_s`` and ``total_frames`` mirror :class:`PlaybackState`.
    """

    frame_float: float
    frame_index: int
    playing: bool
    list_pos: int | None = None
    source_frame_index: int | None = None
    time_s: float = 0.0
    total_frames: int = 0
    fps: float = 0.0


class UIEventBus:
    """Simple synchronous pub/sub used by the Tk dashboard."""

    def __init__(self) -> None:
        self._subs: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event: str, handler: EventHandler) -> None:
        if handler not in self._subs[event]:
            self._subs[event].append(handler)

    def unsubscribe(self, event: str, handler: EventHandler) -> None:
        handlers = self._subs.get(event)
        if not handlers:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            pass

    def publish(self, event: str, *args: Any, **kwargs: Any) -> None:
        for handler in list(self._subs.get(event, ())):
            try:
                handler(*args, **kwargs)
            except Exception:
                logger.exception("UI event handler failed for %s", event)

    def clear(self) -> None:
        self._subs.clear()
