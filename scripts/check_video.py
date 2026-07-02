"""Quick check: does a video produce valid poses? Usage: python scripts/check_video.py [video]"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.pose_estimation import PoseEstimator
from stablewalk.pose_validation import is_plausible_human_pose
from stablewalk.video_processing import VideoProcessor


def main() -> int:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else config.INPUT_DIR / "my_walk.mp4"
    if not video.is_file():
        print(f"Video not found: {video}")
        return 1

    frames_dir = config.FRAMES_DIR / "_check_temp"
    processor = VideoProcessor()
    result = processor.extract_frames(video, frames_dir, max_frames=15)

    valid = 0
    with PoseEstimator() as est:
        for i, fp in enumerate(result.frame_paths[:15]):
            frame = est.process_image(fp, frame_index=i)
            if frame.detected and frame.keypoints and is_plausible_human_pose(frame.keypoints):
                valid += 1

    print(f"Video: {video.name}")
    print(f"Valid poses in first 15 frames: {valid}/15")
    if valid == 0:
        print("This video is not suitable. Use a clear full-body walking shot.")
        return 1
    print("OK — run: python main.py --view")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
