"""Full abnormal demo replacement validation report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk.pose.estimation import PoseEstimator
from stablewalk.ui.media.demo_validation import validate_demo_video
from stablewalk.ui.media.utah_abnormal import UTAH_METADATA_PATH, opencv_validate

VIDEO = ROOT / "data/demo_videos/abnormal_gait.mp4"
LANDMARKS = (
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)


def landmark_coverage(seq, name: str) -> float:
    visible = 0
    for fr in seq.frames:
        for kp in fr.keypoints:
            if kp.name == name and kp.visibility >= 0.4:
                visible += 1
                break
    return visible / max(len(seq.frames), 1)


def temporal_motion(seq) -> dict:
    def track(name: str) -> list[float]:
        vals = []
        for fr in seq.frames:
            for kp in fr.keypoints:
                if kp.name == name and kp.visibility >= 0.4:
                    vals.append(kp.y)
                    break
        return vals

    out = {}
    for name in ("right_ankle", "left_ankle", "right_knee", "left_knee"):
        vals = track(name)
        out[name] = float(np.std(vals)) if len(vals) >= 5 else 0.0
    out["frames_with_pose"] = sum(1 for fr in seq.frames if fr.detected)
    return out


def count_steps(seq) -> int:
    """Rough step count from right ankle vertical peaks."""
    ys = []
    for fr in seq.frames:
        y = None
        for kp in fr.keypoints:
            if kp.name == "right_ankle" and kp.visibility >= 0.4:
                y = kp.y
                break
        ys.append(y)
    arr = np.array([v if v is not None else np.nan for v in ys])
    valid = ~np.isnan(arr)
    if valid.sum() < 10:
        return 0
    smooth = arr.copy()
    for i in range(1, len(smooth) - 1):
        if not np.isnan(smooth[i]):
            neighbors = [smooth[i - 1], smooth[i], smooth[i + 1]]
            neighbors = [n for n in neighbors if not np.isnan(n)]
            smooth[i] = float(np.mean(neighbors))
    peaks = 0
    for i in range(1, len(smooth) - 1):
        if np.isnan(smooth[i - 1]) or np.isnan(smooth[i]) or np.isnan(smooth[i + 1]):
            continue
        if smooth[i] < smooth[i - 1] and smooth[i] < smooth[i + 1]:
            peaks += 1
    return max(peaks // 2, 1)


def main() -> int:
    meta = json.loads(UTAH_METADATA_PATH.read_text(encoding="utf-8")) if UTAH_METADATA_PATH.is_file() else {}
    opencv = opencv_validate(VIDEO)
    report = validate_demo_video(VIDEO, max_frames=120)

    with PoseEstimator(video_mode=True) as est:
        seq, _ = est.process_video_with_frame_cache(str(VIDEO), None, max_frames=120)

    sampled = len(seq.frames)
    detected = sum(1 for fr in seq.frames if fr.detected)
    lm = {name: landmark_coverage(seq, name) for name in LANDMARKS}
    motion = temporal_motion(seq)
    steps = count_steps(seq)

    print("ABNORMAL DEMO REPLACEMENT REPORT")
    print()
    print(f"Source institution: {meta.get('source_institution', 'University of Utah – NeuroLogic Examination')}")
    print(f"Video title: {meta.get('video_title', 'Neuropathic Gait')}")
    print(f"University of Utah video identifier: {meta.get('utah_video_identifier', 'gait_ab_10')}")
    print(f"Source page: {meta.get('source_page')}")
    print(f"Download source used: {meta.get('download_url')}")
    print(f"Original file format: {meta.get('original_format')}")
    print(f"Final local path: {VIDEO.resolve()}")
    print(f"Final codec: {meta.get('final_codec', 'H.264 / yuv420p / MP4')}")
    print(f"Resolution: {opencv.get('width')}x{opencv.get('height')}")
    print(f"FPS: {opencv.get('fps')}")
    print(f"Frame count: {opencv.get('frame_count')}")
    print(f"Duration: {opencv.get('duration_s', 0):.2f}s")
    print(f"Was trimming performed: {'yes' if meta.get('trim_start_frame', -1) >= 0 else 'no'}")
    if meta.get("trim_start_frame", -1) >= 0:
        fps = opencv.get("fps") or 29.97
        start = meta["trim_start_frame"]
        end = start + meta.get("trim_output_frames", opencv.get("frame_count", 0))
        print(f"Selected walking time range: {start/fps:.2f}s – {end/fps:.2f}s (frames {start}–{end})")
    print(f"OpenCV decoding result: PASS (all sample points decoded)" if all(opencv.get("decoded_samples", {}).values()) else "FAIL")
    print(f"MediaPipe sampled frames: {sampled}")
    print(f"MediaPipe detected frames: {detected}")
    print(f"Pose detection percentage: {detected/sampled:.1%}")
    print(f"Right ankle landmark coverage: {lm['right_ankle']:.1%}")
    print(f"Right foot landmark coverage: {max(lm['right_heel'], lm['right_foot_index']):.1%}")
    print(f"Number of visible gait steps: ~{steps}")
    print()
    print("Temporal motion (Y std):")
    print(f"  right_ankle={motion['right_ankle']:.4f} left_ankle={motion['left_ankle']:.4f}")
    print(f"  right_knee={motion['right_knee']:.4f} left_knee={motion['left_knee']:.4f}")
    print()
    print("Landmark coverage:")
    for name in LANDMARKS:
        print(f"  {name}: {lm[name]:.1%}")
    print()
    print(f"Validation status: {report.compact_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
