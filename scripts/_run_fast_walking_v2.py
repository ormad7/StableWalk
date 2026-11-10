#!/usr/bin/env python3
"""Invalidate old Fast Walking caches, run pose pipeline, inspect tracking."""

from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.core.pipeline import run_gait_pipeline
from stablewalk.pose.video_source import content_cache_key, derive_run_name
from stablewalk.ui.media.demo_gait import demo_path, example_by_key, format_about_active_example

OUT = config.OUTPUT_DIR / "example_v2_verification"
NEED = ("left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle")
FEET = ("left_ankle", "right_ankle", "left_heel", "right_heel")
CACHE_STEMS = (
    "athletic_walking",
    "fast_walking",
    "fast_walking_v2",
    "validation_athletic",
)


def _clear_caches() -> list[str]:
    removed: list[str] = []
    for stem in CACHE_STEMS:
        for path in (
            config.POSES_DIR / f"{stem}_poses.json",
            config.FRAMES_DIR / stem,
            config.OPENSIM_DIR / stem,
        ):
            if path.is_file():
                path.unlink()
                removed.append(str(path))
            elif path.is_dir():
                shutil.rmtree(path)
                removed.append(str(path))
        if config.POSES_DIR.is_dir():
            for extra in config.POSES_DIR.glob(f"*{stem}*"):
                if extra.is_file():
                    extra.unlink()
                    removed.append(str(extra))
        if config.FRAMES_DIR.is_dir():
            for extra in config.FRAMES_DIR.glob(f"*{stem}*"):
                if extra.is_dir():
                    shutil.rmtree(extra)
                    removed.append(str(extra))
    leftover = config.OUTPUT_DIR / "example_ui_flow" / "athletic_analysis.json"
    if leftover.is_file():
        leftover.unlink()
        removed.append(str(leftover))
    return removed


def _inspect(path: Path) -> dict:
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles_from_pose_sequence
    from stablewalk.io.pose_loader import load_pose_sequence

    raw = json.loads(path.read_text(encoding="utf-8"))
    seq = load_pose_sequence(path)
    usable = fail = left = right = feet = vis_fail = 0
    for fr in seq.frames:
        kps = {kp.name: kp for kp in fr.keypoints}
        if not fr.detected or not kps:
            fail += 1
            continue
        usable += 1
        if all(n in kps for n in ("left_hip", "left_knee", "left_ankle")):
            left += 1
        if all(n in kps for n in ("right_hip", "right_knee", "right_ankle")):
            right += 1
        if all(n in kps for n in FEET):
            feet += 1
        if any(kps[n].visibility < 0.5 for n in NEED if n in kps):
            vis_fail += 1
    cycles = analyze_gait_cycles_from_pose_sequence(seq)
    metrics = cycles.metrics.to_dict() if cycles.metrics else {}
    return {
        "poses_path": str(path),
        "source_video": raw.get("source_video"),
        "json_frame_count": raw.get("frame_count"),
        "json_detected_count": raw.get("detected_count"),
        "json_fps": raw.get("fps"),
        "loaded_frames": len(seq.frames),
        "usable_detected_frames": usable,
        "tracking_failures": fail,
        "left_limb_complete": left,
        "right_limb_complete": right,
        "feet_present": feet,
        "low_visibility_on_core_joints": vis_fail,
        "detected_cycles": len(cycles.cycles),
        "event_count": len(cycles.events),
        "event_sufficiency": cycles.event_sufficiency,
        "event_sufficiency_reason": cycles.event_sufficiency_reason,
        "warnings": list(cycles.warnings or []),
        "metrics": metrics,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ex = example_by_key("athletic")
    official = demo_path(ex)
    digest = sha256(official.read_bytes()).hexdigest()
    removed = _clear_caches()
    print(f"cleared {len(removed)} cache paths", flush=True)
    print(f"about url={ex.source_url}", flush=True)
    print(format_about_active_example(ex), flush=True)
    run_name = derive_run_name(str(official), unique_session=True)
    print(f"pipeline {official} run={run_name}", flush=True)
    result = run_gait_pipeline(
        str(official),
        run_name=run_name,
        max_frames=900,
        force_reprocess=True,
        pose_backend="mediapipe",
        cache_frames=False,
    )
    inspect = _inspect(result.poses_path)
    report = {
        "cleared_caches": removed,
        "official_path": str(official.resolve()),
        "sha256": digest,
        "source_url": ex.source_url,
        "trial_id": "Subj04/walk_54_side_anonymized.avi",
        "documented_speed": "treadmill walking 5.4 km/h (walk_54)",
        "pipeline": {
            "run_name": result.run_name,
            "source": result.source,
            "content_key": result.content_key,
            "expected_content_key": content_cache_key(str(official)),
            "frame_count": result.frame_count,
            "fps": result.fps,
            "poses_path": str(result.poses_path),
            "same_video": Path(result.source).resolve() == official.resolve()
            and result.content_key == content_cache_key(str(official)),
        },
        "pose_inspect": inspect,
    }
    dest = OUT / "fast_walking_v2_pipeline.json"
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["pose_inspect"], indent=2, default=str), flush=True)
    print(f"Wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
