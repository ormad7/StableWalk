"""Integration checks on demo videos that exist on disk.

Does not invent pose. When a pose JSON for the clip is found, gait metrics
are computed. Otherwise only container metadata is reported and pose-level
asserts are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.asymmetry3d.report import analyze_asymmetry3d
from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.analysis.spectral.report import analyze_spectral
from stablewalk.io.pose_loader import load_pose_sequence

_VIDEOS = {
    "A_unobstructed": config.DEMO_VIDEOS_DIR / "normal_gait.mp4",
    "B_asymmetric": config.DEMO_VIDEOS_DIR / "candidates" / "weightgait_limping_gait.mp4",
    "C_short_aid": config.DEMO_VIDEOS_DIR / "abnormal_gait.mp4",
}


def _video_meta(path: Path) -> dict[str, float | int | str]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "cannot_open", "path": str(path)}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = n / fps if fps > 1e-6 else 0.0
    return {"path": str(path), "fps": fps, "frames": n, "duration_s": duration}


def _pose_path_for_video(video: Path) -> Path | None:
    stem = video.stem
    named = config.POSES_DIR / f"{stem}_poses.json"
    if named.is_file():
        return named
    autosave = config.SESSION_EXPORT_DIR / "autosave" / "stablewalk_autosave" / "poses.json"
    if autosave.is_file():
        try:
            data = json.loads(autosave.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        src = str(data.get("source_video") or "")
        if video.name in src.replace("\\", "/"):
            return autosave
    if not config.POSES_DIR.is_dir():
        return None
    matches: list[Path] = []
    for path in config.POSES_DIR.glob("*_poses.json"):
        try:
            head = path.read_text(encoding="utf-8")[:800]
        except Exception:
            continue
        if video.name in head.replace("\\", "/"):
            matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


@pytest.mark.parametrize("label,path", list(_VIDEOS.items()))
def test_demo_video_files_exist_and_have_fps(label: str, path: Path) -> None:
    assert path.is_file(), f"{label} missing: {path}"
    meta = _video_meta(path)
    assert "error" not in meta
    assert float(meta["fps"]) > 1.0
    assert int(meta["frames"]) >= 2


def test_short_aid_clip_metadata_matches_observation() -> None:
    path = _VIDEOS["C_short_aid"]
    if not path.is_file():
        pytest.skip("abnormal_gait.mp4 not present")
    meta = _video_meta(path)
    # User observation ~120 frames / 4 s / 30 FPS; container may report a nearby count.
    assert 90 <= int(meta["frames"]) <= 180
    assert 20.0 <= float(meta["fps"]) <= 35.0
    assert 3.0 <= float(meta["duration_s"]) <= 6.5


def test_abnormal_gait_pose_does_not_fabricate_cycles() -> None:
    video = _VIDEOS["C_short_aid"]
    pose_path = _pose_path_for_video(video)
    if pose_path is None or not pose_path.is_file():
        pytest.skip("No pose JSON for abnormal_gait.mp4")
    seq = load_pose_sequence(pose_path)
    rec = pose_sequence_to_gait_motion(seq)
    cycles = analyze_gait_cycles(rec)
    if cycles.metrics.gait_cycle_count < 2 or not cycles.metrics.metrics_reliable:
        assert cycles.event_sufficiency in ("input_insufficient", "processing_failure")
        assert cycles.metrics.cadence_steps_per_min is None
        assert cycles.metrics.metrics_reliable is False
        assert cycles.event_sufficiency_reason
    spec = analyze_spectral(rec)
    assert spec.bundle.times_s.size == rec.frame_count or spec.bundle.times_s.size > 0
    if spec.features is not None:
        # Features may be unavailable; values must not be invented when flagged.
        for rec_feat in spec.features.records:
            if not rec_feat.available:
                assert rec_feat.value is None
    asym = analyze_asymmetry3d(rec, cycles=cycles)
    assert asym is not None


def test_normal_gait_pose_reports_cycles_if_pose_exists() -> None:
    video = _VIDEOS["A_unobstructed"]
    pose_path = _pose_path_for_video(video)
    if pose_path is None or not pose_path.is_file():
        pytest.skip("No pose JSON for normal_gait.mp4")
    rec = pose_sequence_to_gait_motion(load_pose_sequence(pose_path))
    result = analyze_gait_cycles(rec)
    assert result.fps > 1.0
    assert result.event_sufficiency in ("none", "input_insufficient", "processing_failure")
    if result.event_sufficiency == "none":
        assert result.metrics.gait_cycle_count >= 2
