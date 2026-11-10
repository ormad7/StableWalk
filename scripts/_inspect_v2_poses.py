from pathlib import Path
import json

from stablewalk.io.pose_loader import load_pose_sequence

FILES = [
    Path(r"data/output/poses/run_8a254c44e19fc08b_1793524712974_poses.json"),
    Path(r"data/output/poses/run_90d21a9367838db7_1793524779679_poses.json"),
]
NEED = ("left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle")
FEET = ("left_ankle", "right_ankle", "left_heel", "right_heel")


def inspect(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    seq = load_pose_sequence(path)
    usable = 0
    fail = 0
    left = 0
    right = 0
    feet = 0
    vis_fail = 0
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
    return {
        "file": path.name,
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
    }


if __name__ == "__main__":
    for f in FILES:
        print(inspect(f))
