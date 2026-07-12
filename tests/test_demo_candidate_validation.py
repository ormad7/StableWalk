"""Tests for demo candidate validation and selection helpers."""

from __future__ import annotations

from pathlib import Path

from stablewalk.demo.candidate_validation import validate_demo_candidate
from stablewalk.demo.healthgait_selection import rank_healthgait_trials


def test_validate_missing_video_rejects():
    report = validate_demo_candidate("/nonexistent/demo_video.mp4")
    assert report.verdict == "REJECT"
    assert report.issues


def test_rank_healthgait_ugs_prefers_moderate_speed(tmp_path: Path):
    csv_path = tmp_path / "gait_parameters.csv"
    csv_path.write_text(
        "ID,Velocity_UGS,Cadence_UGS,Step_UGS,Stride_UGS\n"
        "PA001,1.10,110,50,100\n"
        "PA002,1.40,120,55,110\n"
        "PA003,1.25,115,52,105\n",
        encoding="utf-8",
    )
    ranked = rank_healthgait_trials(csv_path, gait_speed="UGS")
    assert len(ranked) == 3
    assert ranked[0].participant_id in {"PA002", "PA003"}
    assert any("median" in " ".join(c.notes).lower() for c in ranked)
