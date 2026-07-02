"""SQLite repository for analysis sessions and kinematic samples."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from stablewalk.storage.models import AnalysisSession, KinematicSample
from stablewalk.storage.schema import initialize_database

logger = logging.getLogger(__name__)

_SAMPLE_INSERT = """
INSERT INTO kinematic_samples (
    session_id,
    frame_number,
    time_s,
    dof_name,
    joint_name,
    x,
    y,
    z,
    angle_deg,
    velocity,
    velocity_deg_s,
    next_angle_deg,
    delta_angle_deg
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


class SessionRepository:
    """Low-level persistence for sessions and kinematic samples."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        initialize_database(db_path)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def save_session(
        self,
        session: AnalysisSession,
        samples: list[KinematicSample],
    ) -> AnalysisSession:
        """Persist one session and all associated samples in a single transaction."""
        if not samples:
            raise ValueError("Cannot save a session with no kinematic samples")

        logger.info(
            "Saving session %s (%d samples) to %s",
            session.session_id,
            len(samples),
            self._db_path,
        )
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id,
                    video_source,
                    created_at,
                    fps,
                    sample_count,
                    selected_dofs,
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    session.session_id,
                    session.video_source,
                    session.created_at,
                    session.fps,
                    len(samples),
                    session.selected_dofs,
                    session.notes,
                ),
            )
            conn.executemany(
                _SAMPLE_INSERT,
                [
                    (
                        session.session_id,
                        sample.frame_number,
                        sample.time_s,
                        sample.dof_name,
                        sample.joint_name,
                        sample.x,
                        sample.y,
                        sample.z,
                        sample.angle_deg,
                        sample.velocity,
                        sample.velocity_deg_s,
                        sample.next_angle_deg,
                        sample.delta_angle_deg,
                    )
                    for sample in samples
                ],
            )
            conn.commit()

        saved = AnalysisSession(
            session_id=session.session_id,
            video_source=session.video_source,
            created_at=session.created_at,
            fps=session.fps,
            sample_count=len(samples),
            selected_dofs=session.selected_dofs,
            notes=session.notes,
        )
        logger.info(
            "Saved session %s with %d samples",
            saved.session_id,
            saved.sample_count,
        )
        return saved

    def list_sessions(self, *, limit: int = 50) -> list[AnalysisSession]:
        """Return recent sessions ordered by creation time descending."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT session_id, video_source, created_at, fps, sample_count,
                       selected_dofs, notes
                FROM sessions
                ORDER BY created_at DESC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()
        return [
            AnalysisSession(
                session_id=str(row["session_id"]),
                video_source=str(row["video_source"]),
                created_at=str(row["created_at"]),
                fps=row["fps"],
                sample_count=int(row["sample_count"]),
                selected_dofs=row["selected_dofs"],
                notes=row["notes"],
            )
            for row in rows
        ]

    def count_samples(self, session_id: str) -> int:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM kinematic_samples WHERE session_id = ?;",
                (session_id,),
            ).fetchone()
        return int(row[0]) if row else 0
