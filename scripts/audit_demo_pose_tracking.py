#!/usr/bin/env python3
"""Audit pose tracking on the three current demo videos.

Uses saved pose JSON (visibility recovered from skeleton_3d when the compact
2D keypoints omitted it). Probes Normal Gait for a second person. Does not
invent interpolated frames.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/output/example_v2_verification/pose_tracking_audit.json"
DEMOS = (
    ("abnormal", "abnormal_gait_v2.mp4"),
    ("normal", "normal_gait_v2.mp4"),
    ("athletic", "fast_walking_v2.mp4"),
)


def _latest_poses(filename: str) -> Path | None:
    from stablewalk import config

    matches = []
    for path in config.POSES_DIR.glob("*_poses.json"):
        try:
            with path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            continue
        src = str(raw.get("source_video") or "")
        if Path(src).name == filename:
            matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _probe_multiperson(video_path: Path, frame_indices: list[int]) -> list[dict]:
    """IMAGE-mode multi-pose probe — does not change saved analysis."""
    import cv2

    from stablewalk.pose.estimation import PoseEstimator
    from stablewalk.pose.subject_lock import pose_bbox_height, pose_centroid
    from stablewalk.pose.validation import has_full_body_pose

    if not video_path.is_file() or not frame_indices:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [{"error": f"could not open {video_path}"}]
    rows = []
    try:
        with PoseEstimator(
            video_mode=False,
            require_full_body=False,
            num_poses=4,
            lock_subject=False,
        ) as estimator:
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    rows.append({"frame": int(idx), "read": False})
                    continue
                image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                from mediapipe import Image, ImageFormat

                mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)
                results = estimator._landmarker.detect(mp_image)
                people = []
                for landmarks in results.pose_landmarks or []:
                    kps = PoseEstimator._add_mid_hip(
                        PoseEstimator._clamp_keypoints(
                            PoseEstimator._landmarks_to_keypoints(landmarks)
                        )
                    )
                    centroid = pose_centroid(kps)
                    people.append(
                        {
                            "bbox_h": pose_bbox_height(kps),
                            "centroid": None if centroid is None else [round(centroid[0], 3), round(centroid[1], 3)],
                            "full_body": bool(has_full_body_pose(kps)),
                            "n_kps": len(kps),
                        }
                    )
                rows.append(
                    {
                        "frame": int(idx),
                        "people": people,
                        "n_people": len(people),
                    }
                )
    finally:
        cap.release()
    return rows


def main() -> int:
    from stablewalk.io.pose_loader import load_pose_sequence
    from stablewalk.pose.tracking_quality import LOWER_LIMB_JOINTS, audit_pose_sequence
    from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES, demo_path
    from stablewalk.ui.tracking_coverage import measure_tracking_coverage

    report: dict = {"videos": []}
    demo_by_key = {ex.key: ex for ex in DEMO_GAIT_EXAMPLES}
    for key, filename in DEMOS:
        poses_path = _latest_poses(filename)
        entry: dict = {
            "key": key,
            "filename": filename,
            "poses_path": None if poses_path is None else str(poses_path),
        }
        if poses_path is None:
            entry["error"] = "no pose JSON for this demo"
            report["videos"].append(entry)
            continue
        seq = load_pose_sequence(poses_path)
        audit = audit_pose_sequence(seq)
        cov = measure_tracking_coverage(seq)
        payload = audit.to_dict()
        payload["tracking_hud"] = None if cov is None else {
            "detected": cov.detected_display(),
            "joints": cov.joint_confidence_display(),
            "usable": cov.usable_display(),
        }
        payload["lower_limb_mean_visibility"] = {
            name: payload["joint_stats"][name]["mean_visibility"]
            for name in LOWER_LIMB_JOINTS
            if name in payload["joint_stats"]
        }
        entry.update(payload)
        if key == "normal":
            ex = demo_by_key.get(key)
            video = demo_path(ex) if ex is not None else None
            first_det = next(
                (fr.frame_index for fr in seq.frames if getattr(fr, "detected", False)),
                9,
            )
            sample_frames = sorted(
                {
                    0,
                    first_det,
                    max(0, first_det + 1),
                    150,
                    161,
                    167,
                    228,
                    230,
                    max(0, len(seq.frames) - 1),
                    *audit.identity_switches[:4],
                    *audit.jump_frames[:4],
                }
            )
            probe = _probe_multiperson(video, sample_frames) if video is not None else []
            from stablewalk.pose.tracking_quality import _pelvis_xy

            for row in probe:
                idx = int(row.get("frame", -1))
                frame = next(
                    (fr for fr in seq.frames if int(fr.frame_index) == idx),
                    None,
                )
                saved = None if frame is None else _pelvis_xy(frame)
                row["saved_pelvis"] = None if saved is None else [round(saved[0], 3), round(saved[1], 3)]
                row["saved_detected"] = bool(getattr(frame, "detected", False)) if frame else False
            entry["multiperson_probe"] = probe
        report["videos"].append(entry)
        print(
            f"{filename}: total={audit.total_frames} missing={audit.missing_frames} "
            f"observed={audit.observed_frames} unreliable={audit.unreliable_frames} "
            f"interp={audit.interpolated_frames} jumps={len(audit.jump_frames)} "
            f"swaps={len(audit.laterality_flips)} identity={audit.identity_switches} "
            f"usable={audit.usable_gait_frames}",
            flush=True,
        )
        for name in ("left_hip", "left_knee", "left_ankle", "left_heel", "left_foot_index",
                     "right_hip", "right_knee", "right_ankle", "right_heel", "right_foot_index"):
            st = audit.joint_stats[name]
            mean = "n/a" if st.mean_visibility is None else f"{st.mean_visibility:.2f}"
            vmin = "n/a" if st.min_visibility is None else f"{st.min_visibility:.2f}"
            p10 = "n/a" if st.p10_visibility is None else f"{st.p10_visibility:.2f}"
            print(
                f"  {name:18s} mean={mean} min={vmin} p10={p10} "
                f"pct_ok={st.pct_ok:.0f}% low(<0.45)={st.n_low_vis} "
                f"degraded(<0.70)={st.n_degraded}/{st.n_observed}",
                flush=True,
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
