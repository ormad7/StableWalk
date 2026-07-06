"""Download, prepare, and install best athletic walking demo (Pexels 27727783)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.analysis.biomech_stability import analyze_biomech_stability
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.pose.estimation import PoseEstimator
from stablewalk.ui.media.demo_download import _download_pexels_video
from stablewalk.ui.media.demo_gait import demo_path, example_by_key
from stablewalk.ui.media.demo_prepare import find_best_walking_segment, trim_to_gait_segment
from scripts.validate_athletic_video import validate_video

PEXELS_ID = 27727783
TITLE = "Tennis court walk"
SOURCE_URL = "https://www.pexels.com/video/tennis-court-walk-27727783/"
CREATOR = "Pexels contributor"  # updated after page verify


def _reencode_h264(src: Path, dest: Path) -> bool:
    """Re-encode to H.264 yuv420p with constant frame rate if ffmpeg available."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copy2(src, dest)
        return dest.is_file()
    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", "25", "-vsync", "cfr",
        "-movflags", "+faststart",
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return dest.is_file() and dest.stat().st_size > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        shutil.copy2(src, dest)
        return dest.is_file()


def _stability_scores() -> tuple[dict, dict]:
    results = {}
    for key in ("normal", "athletic"):
        ex = example_by_key(key)
        path = demo_path(ex)
        with PoseEstimator(video_mode=True) as est:
            seq = est.process_video(path, enrich_gait=False, max_frames=120)
        enrich_pose_sequence(seq)
        r = analyze_biomech_stability(seq)
        mm = {m.key: m for m in r.metrics}
        results[key] = {
            "stability": r.score,
            "symmetry": mm["symmetry"].score,
            "step": mm["step_consistency"].score,
            "body": mm["body_stability"].score,
        }
    return results["normal"], results["athletic"]


def main() -> int:
    config.ensure_output_dirs()
    dest = demo_path(example_by_key("athletic"))
    backup = dest.with_name(
        f"athletic_walking_pexels5823532_backup_{datetime.now():%Y%m%d}.mp4"
    )

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        trimmed = Path(tmp) / "trimmed.mp4"
        encoded = Path(tmp) / "encoded.mp4"

        print(f"Downloading Pexels {PEXELS_ID} ...")
        if not _download_pexels_video(PEXELS_ID, raw):
            print("Download failed")
            return 1

        start = find_best_walking_segment(raw, output_frames=120)
        ok, detail = trim_to_gait_segment(raw, trimmed, start_frame=start or 0, output_frames=120)
        print(f"Trim: ok={ok} start={start} detail={detail}")
        src = trimmed if ok and trimmed.is_file() else raw

        print("Validating candidate ...")
        candidate = validate_video(src, max_frames=120, label="Candidate")
        current = validate_video(dest, max_frames=120, label="Current")

        print(f"Current heel={current['heel_visibility']:.3f} conf={current['step_confidence']}")
        print(f"Candidate heel={candidate['heel_visibility']:.3f} conf={candidate['step_confidence']}")

        if candidate["heel_visibility"] <= current["heel_visibility"] + 0.05:
            print("Candidate not clearly better — aborting replace")
            return 1

        if not _reencode_h264(src, encoded):
            print("Encode failed")
            return 1

        if dest.is_file() and not backup.is_file():
            shutil.copy2(dest, backup)
            print(f"Backed up current video to {backup.name}")

        shutil.copy2(encoded, dest)
        print(f"Installed {dest}")

    normal, athletic = _stability_scores()
    print("\nPost-replace scores:")
    print(f"  Normal:   Stability={normal['stability']:.1f}  Symmetry={normal['symmetry']:.1f}  Step={normal['step']:.1f}  Body={normal['body']:.1f}")
    print(f"  Athletic: Stability={athletic['stability']:.1f}  Symmetry={athletic['symmetry']:.1f}  Step={athletic['step']:.1f}  Body={athletic['body']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
