"""High-level service for saving analysis playback sessions."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from stablewalk import config
from stablewalk.storage.models import AnalysisSession, KinematicSample, utc_now_iso
from stablewalk.storage.repository import SessionRepository

logger = logging.getLogger(__name__)


class SessionStorageService:
    """Application service for persisting playback kinematic data."""

    def __init__(self, db_path: Path | None = None) -> None:
        path = db_path or config.SESSIONS_DB_PATH
        self._repository = SessionRepository(path)

    @property
    def db_path(self) -> Path:
        return self._repository.db_path

    def save_session(
        self,
        *,
        video_source: str,
        samples: list[KinematicSample],
        fps: float | None = None,
        selected_dofs: set[str] | list[str] | None = None,
        notes: str | None = None,
        session_id: str | None = None,
    ) -> AnalysisSession:
        """
        Persist one analysis session and all kinematic samples.

        Raises:
            ValueError: When ``video_source`` or ``samples`` is empty.
        """
        source = (video_source or "").strip()
        if not source:
            raise ValueError("video_source is required to save a session")
        if not samples:
            raise ValueError("No kinematic samples to save")

        session = AnalysisSession(
            session_id=session_id or str(uuid.uuid4()),
            video_source=source,
            created_at=utc_now_iso(),
            fps=fps,
            sample_count=len(samples),
            selected_dofs=_format_selected_dofs(selected_dofs),
            notes=notes,
        )
        try:
            return self._repository.save_session(session, samples)
        except Exception:
            logger.exception("Failed to save session %s", session.session_id)
            raise

    def list_recent_sessions(self, *, limit: int = 50) -> list[AnalysisSession]:
        return self._repository.list_sessions(limit=limit)


def _format_selected_dofs(selected: set[str] | list[str] | None) -> str | None:
    if not selected:
        return None
    ordered = sorted(selected)
    return ",".join(ordered)
