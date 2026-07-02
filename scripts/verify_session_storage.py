"""Verify session storage initialization and round-trip save."""

from __future__ import annotations

import tempfile
from pathlib import Path

from stablewalk.ui.presentation_demo import generate_presentation_recording
from stablewalk.storage.collector import SessionKinematicCollector
from stablewalk.storage.service import SessionStorageService
from stablewalk.ui.dof_position_table import snapshot_for_next_frame


def main() -> None:
    recording = generate_presentation_recording()
    collector = SessionKinematicCollector()
    selected = {"right_knee", "left_knee"}

    for index in range(min(10, recording.frame_count)):
        snap = recording.snapshot_at(index)
        if snap is None:
            continue
        next_snap = snapshot_for_next_frame(recording, snap)
        collector.append_tick(snap, selected, next_snapshot=next_snap)

    assert collector.sample_count == 20

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_sessions.db"
        service = SessionStorageService(db_path=db_path)
        saved = service.save_session(
            video_source="test_walk.mp4",
            samples=list(collector.samples),
            fps=recording.fps,
            selected_dofs=selected,
        )
        assert saved.sample_count == 20
        assert service._repository.count_samples(saved.session_id) == 20
        recent = service.list_recent_sessions()
        assert len(recent) == 1
        print(f"OK: saved session {saved.session_id} with {saved.sample_count} samples")


if __name__ == "__main__":
    main()
