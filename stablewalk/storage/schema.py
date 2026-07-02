"""SQLite schema initialization and lightweight migrations."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    video_source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    fps REAL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    selected_dofs TEXT,
    notes TEXT
);
"""

_CREATE_SAMPLES = """
CREATE TABLE IF NOT EXISTS kinematic_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    frame_number INTEGER NOT NULL,
    time_s REAL NOT NULL,
    dof_name TEXT NOT NULL,
    joint_name TEXT NOT NULL,
    x REAL,
    y REAL,
    z REAL,
    angle_deg REAL,
    velocity REAL,
    velocity_deg_s REAL,
    next_angle_deg REAL,
    delta_angle_deg REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
"""

_CREATE_INDEX_SESSION = """
CREATE INDEX IF NOT EXISTS idx_kinematic_samples_session
ON kinematic_samples (session_id);
"""

_CREATE_INDEX_SESSION_FRAME = """
CREATE INDEX IF NOT EXISTS idx_kinematic_samples_session_frame
ON kinematic_samples (session_id, frame_number);
"""

_CREATE_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
"""


def initialize_database(db_path: Path) -> None:
    """
    Create the database file and apply the current schema if needed.

    Safe to call repeatedly; uses ``CREATE IF NOT EXISTS`` and a version table.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing session database at %s", db_path.resolve())
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(
            "\n".join(
                (
                    _CREATE_SESSIONS,
                    _CREATE_SAMPLES,
                    _CREATE_INDEX_SESSION,
                    _CREATE_INDEX_SESSION_FRAME,
                    _CREATE_SCHEMA_VERSION,
                )
            )
        )
        row = conn.execute("SELECT version FROM schema_version LIMIT 1;").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?);",
                (SCHEMA_VERSION,),
            )
            logger.info("Created session database schema v%d", SCHEMA_VERSION)
        elif int(row[0]) < SCHEMA_VERSION:
            _apply_migrations(conn, int(row[0]))
            conn.execute(
                "UPDATE schema_version SET version = ?;",
                (SCHEMA_VERSION,),
            )
            logger.info("Migrated session database to schema v%d", SCHEMA_VERSION)
        conn.commit()


def _apply_migrations(conn: sqlite3.Connection, current_version: int) -> None:
    """Apply incremental migrations from ``current_version`` to ``SCHEMA_VERSION``."""
    del conn, current_version
    # v1 is the initial release; future migrations go here.
