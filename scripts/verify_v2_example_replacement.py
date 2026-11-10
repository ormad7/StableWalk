#!/usr/bin/env python3
"""End-to-end replacement checks for Examples *_v2 files (no score retuning)."""

from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.core.pipeline import run_gait_pipeline
from stablewalk.pose.video_source import content_cache_key, derive_run_name
from stablewalk.ui.media.demo_gait import (
    DEMO_GAIT_EXAMPLES,
    demo_path,
    format_about_active_example,
    is_legacy_demo_filename,
)
from stablewalk.ui.tk.session_cache import resolve_demo_video_path

OUT = config.OUTPUT_DIR / "example_v2_verification"
PREVIOUS_HASHES = {
    "abnormal": "0c5a13524373051c9ee7be25e457cb6add6b1c5a0043f6a93c58c959060a0cc3",
    "normal": "9c29da2523ed2b4fbe2cb4b41eb570af26993a1ef0a0e8b66756e257a91e712c",
    "athletic": "326ca207d9594dc43af7d7c0295c6e4da1d2d95069489ecaba136485b60689d4",
}
LEGACY_STEMS = ("abnormal_gait", "normal_gait", "athletic_walking")


def _hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256(path.read_bytes()).hexdigest()


def _decode(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = {
        "opened": bool(cap.isOpened()),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "decoded_ok": 0,
        "decoded_fail": 0,
    }
    n = info["frame_count"]
    samples = sorted({0, n // 4, n // 2, (3 * n) // 4, max(0, n - 1)}) if n else [0]
    frames = []
    for idx in samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            info["decoded_ok"] += 1
            frames.append((idx, frame))
        else:
            info["decoded_fail"] += 1
    cap.release()
    fps = info["fps"] if info["fps"] > 1e-6 else 0.0
    info["duration_s"] = (info["frame_count"] / fps) if fps else 0.0
    info["sample_indices"] = [i for i, _ in frames]
    return info, frames


def _contact_sheet(frames: list[tuple[int, np.ndarray]], dest: Path) -> None:
    if not frames:
        return
    tiles = []
    for idx, frame in frames:
        h, w = frame.shape[:2]
        scale = 240 / max(h, 1)
        tile = cv2.resize(frame, (int(w * scale), int(h * scale)))
        cv2.putText(
            tile,
            f"f{idx}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        tiles.append(tile)
    height = max(t.shape[0] for t in tiles)
    width = sum(t.shape[1] for t in tiles)
    sheet = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for tile in tiles:
        sheet[: tile.shape[0], x : x + tile.shape[1]] = tile
        x += tile.shape[1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), sheet)


def _clear_legacy_caches() -> list[str]:
    removed: list[str] = []
    for stem in LEGACY_STEMS:
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
            for extra in config.POSES_DIR.glob(f"{stem}_*_poses.json"):
                extra.unlink()
                removed.append(str(extra))
        if config.FRAMES_DIR.is_dir():
            for extra in config.FRAMES_DIR.glob(f"{stem}_*"):
                if extra.is_dir() and "v2" not in extra.name:
                    shutil.rmtree(extra)
                    removed.append(str(extra))
    return removed


def _inspect_poses(poses_path: Path) -> dict:
    from stablewalk.io.pose_loader import load_pose_sequence

    seq = load_pose_sequence(poses_path)
    frames = list(getattr(seq, "frames", None) or [])
    usable = 0
    fail = 0
    left_ok = 0
    right_ok = 0
    for fr in frames:
        kps = {kp.name: kp for kp in getattr(fr, "keypoints", None) or []}
        if not getattr(fr, "detected", False) or not kps:
            fail += 1
            continue
        usable += 1
        if all(name in kps for name in ("left_hip", "left_knee", "left_ankle")):
            left_ok += 1
        if all(name in kps for name in ("right_hip", "right_knee", "right_ankle")):
            right_ok += 1
    cycles = None
    try:
        from stablewalk.analysis.gait_cycle_analysis import (
            analyze_gait_cycles_from_pose_sequence,
        )

        cycles = analyze_gait_cycles_from_pose_sequence(seq)
    except Exception as exc:
        cycles = exc
    detected = 0
    if cycles is not None and not isinstance(cycles, Exception):
        detected = len(getattr(cycles, "cycles", None) or [])
    return {
        "pose_frames": len(frames),
        "usable_pose_frames": usable,
        "tracking_failures": fail,
        "left_limb_tracked": left_ok,
        "right_limb_tracked": right_ok,
        "detected_cycles": detected,
        "poses_path": str(poses_path),
        "source_in_sequence": getattr(seq, "source", None),
        "cycle_error": str(cycles) if isinstance(cycles, Exception) else None,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    removed = _clear_legacy_caches()
    report: dict = {"cleared_legacy_caches": removed, "examples": []}

    for ex in DEMO_GAIT_EXAMPLES:
        official = demo_path(ex)
        resolved = resolve_demo_video_path(ex.key)
        about = format_about_active_example(ex)
        rec: dict = {
            "key": ex.key,
            "display_name": ex.display_name,
            "official_path": str(official),
            "resolved_path": resolved,
            "points_to_v2": official.name.endswith("_v2.mp4")
            and (resolved is None or Path(resolved).name == official.name),
            "legacy_name": is_legacy_demo_filename(official.name),
            "exists": official.is_file(),
            "source_url": ex.source_url,
            "about_has_source_url": "Source URL:" in about,
            "about_has_scores": "/100" in about or "usable cycle" in about.lower(),
            "about_excerpt": about.splitlines()[:8],
        }
        if not official.is_file():
            rec["replacement_verified"] = False
            rec["blocker"] = f"{official.name} is not on disk"
            report["examples"].append(rec)
            print(f"{ex.display_name}: MISSING {official.name}")
            continue

        digest = _hash(official)
        rec["sha256"] = digest
        rec["hash_differs_from_previous"] = digest != PREVIOUS_HASHES.get(ex.key)
        decode, frames = _decode(official)
        rec.update(decode)
        sheet = OUT / f"{ex.key}_contact_sheet.png"
        _contact_sheet(frames, sheet)
        rec["contact_sheet"] = str(sheet)

        print(f"Pipeline {ex.key} {official.name} ...", flush=True)
        run_name = derive_run_name(str(official), unique_session=True)
        result = run_gait_pipeline(
            str(official),
            run_name=run_name,
            max_frames=900,
            force_reprocess=True,
            pose_backend="mediapipe",
        )
        rec["pipeline"] = {
            "run_name": result.run_name,
            "source": result.source,
            "content_key": result.content_key,
            "expected_content_key": content_cache_key(str(official)),
            "frame_count": result.frame_count,
            "fps": result.fps,
            "first_frame_hash": result.first_frame_hash,
            "poses_path": str(result.poses_path),
            "frames_dir": str(result.frames_dir) if result.frames_dir else None,
            "same_video": Path(result.source).resolve() == official.resolve()
            and result.content_key == content_cache_key(str(official)),
        }
        rec["pose_inspect"] = _inspect_poses(result.poses_path)
        rec["replacement_verified"] = bool(
            rec["points_to_v2"]
            and rec["hash_differs_from_previous"]
            and decode["opened"]
            and rec["pipeline"]["same_video"]
            and not rec["about_has_scores"]
        )
        print(
            f"  {ex.display_name}: {official.name} "
            f"{decode['duration_s']:.2f}s {decode['fps']:.3f}fps "
            f"frames={decode['frame_count']} verified={rec['replacement_verified']}"
        )
        report["examples"].append(rec)

    dest = OUT / "replacement_report.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
