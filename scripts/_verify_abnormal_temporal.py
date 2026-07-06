"""Verify temporal gait data changes for abnormal demo."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk.pose.estimation import PoseEstimator
from stablewalk.ui.media.demo_gait import demo_path, example_by_key

VIDEO = demo_path(example_by_key("abnormal"))


def main() -> int:
    with PoseEstimator(video_mode=True) as est:
        seq, _ = est.process_video_with_frame_cache(str(VIDEO), None, max_frames=150)
    frames = seq.frames
    if len(frames) < 10:
        print("FAIL: too few frames")
        return 1

    def series(name: str, axis: str) -> np.ndarray:
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        vals = []
        for fr in frames:
            for kp in fr.keypoints:
                if kp.name == name and kp.visibility >= 0.4:
                    vals.append((kp.x, kp.y, kp.z)[axis_idx])
                    break
            else:
                vals.append(float("nan"))
        return np.array(vals)

    checks = {}
    for joint in ("right_ankle", "right_knee", "left_ankle"):
        for axis in ("x", "y", "z"):
            arr = series(joint, axis)
            valid = arr[~np.isnan(arr)]
            checks[f"{joint}_{axis}"] = float(np.std(valid)) if len(valid) > 5 else 0.0

    # Approximate speed from frame-to-frame right ankle displacement
    ra = series("right_ankle", "y")
    diffs = np.abs(np.diff(ra[~np.isnan(ra)]))
    checks["joint_speed"] = float(np.std(diffs)) if len(diffs) > 5 else 0.0

    print(f"Pipeline frames: {len(frames)}")
    for key, val in checks.items():
        status = "OK" if val > 1e-4 else "STATIC"
        print(f"  {key}: std={val:.6f} [{status}]")

    moving = sum(1 for v in checks.values() if v > 1e-4)
    print(f"Moving channels: {moving}/{len(checks)}")
    return 0 if moving >= 8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
