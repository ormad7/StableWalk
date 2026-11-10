"""Export identity overlays from the latest Normal walker-seed dump."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.skeleton import draw_identity_on_bgr, draw_skeleton_on_bgr
from stablewalk.pose.video import VideoReader

VIDEO = ROOT / "data" / "demo_videos" / "normal_gait_v2.mp4"
OUT = ROOT / "validation_outputs" / "normal" / "identity_probe"
SUMMARY = OUT / "identity_summary.json"
STILL = (80, 148, 191, 220)
CLIP_START = 130
CLIP_END = 200


def _poses_path() -> Path:
    if SUMMARY.is_file():
        payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
        path = Path(str(payload.get("poses_path") or ""))
        if path.is_file():
            return path
    raise FileNotFoundError("identity_summary.json does not point to a pose dump")


def _overlay(bgr, pf):
    out = bgr.copy()
    if pf is not None and pf.keypoints and (pf.detected or pf.identity_locked):
        out = draw_skeleton_on_bgr(out, pf.keypoints)
    if pf is not None:
        out = draw_identity_on_bgr(out, pf)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    poses = _poses_path()
    seq = load_pose_sequence(poses)
    by = {int(f.frame_index): f for f in seq.frames}
    wanted = set(STILL)
    clip_path = OUT / "overlap_clip_0130_0200.mp4"
    writer = None
    with VideoReader(str(VIDEO)) as reader:
        meta = reader.metadata
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for vf in reader.iter_frames():
            pf = by.get(vf.index)
            if vf.index in wanted:
                cv2.imwrite(str(OUT / f"src_{vf.index:04d}.png"), vf.bgr)
                cv2.imwrite(str(OUT / f"overlay_{vf.index:04d}.png"), _overlay(vf.bgr, pf))
                wanted.discard(vf.index)
            if CLIP_START <= vf.index <= CLIP_END:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(clip_path), fourcc, meta.fps, (meta.width, meta.height)
                    )
                writer.write(_overlay(vf.bgr, pf))
            if vf.index > CLIP_END and not wanted:
                break
    if writer is not None:
        writer.release()
    print("wrote", OUT, "from", poses.name)


if __name__ == "__main__":
    main()
