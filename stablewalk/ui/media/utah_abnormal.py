"""Download and prepare University of Utah Neuropathic Gait (gait_ab_10) demo."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

from stablewalk import config

logger = logging.getLogger(__name__)

UTAH_GAIT_ABNORMAL_PAGE = (
    "https://neurologicexam.med.utah.edu/adult/html/gait_abnormal.html"
)
UTAH_DOWNLOAD_PAGE = (
    "https://neurologicexam.med.utah.edu/adult/html/download_by_exam.html"
)
UTAH_VIDEO_ID = "gait_ab_10"
UTAH_VIDEO_TITLE = "Neuropathic Gait"
UTAH_KALTURA_ENTRY = "0_z1p8nsbi"
UTAH_KALTURA_PARTNER = "816122"
UTAH_MP4_ZIP_URL = (
    "https://neurologicexam.med.utah.edu/adult/zips/MP4_MOBILE/"
    f"{UTAH_VIDEO_ID}_MP4_MOBILE.zip"
)
UTAH_METADATA_PATH = config.DEMO_VIDEOS_DIR / "utah_abnormal_source.json"
GAIT_DESCRIPTION = (
    "Abnormal gait pattern: right foot dorsiflexion weakness with compensatory "
    "high stepping for foot clearance."
)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def kaltura_download_url(entry_id: str) -> str:
    return (
        f"https://cdnapisec.kaltura.com/p/{UTAH_KALTURA_PARTNER}/sp/"
        f"{UTAH_KALTURA_PARTNER}00/playManifest/entryId/{entry_id}/"
        f"format/download/protocol/https"
    )


def _http_get(url: str, *, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_official_mp4_zip(dest_mp4: Path, *, password: str | None = None) -> bool:
    """Download Utah MP4 mobile zip and extract ``gait_ab_10.mp4``."""
    import os

    dest_mp4.parent.mkdir(parents=True, exist_ok=True)
    password = password or os.environ.get("UTAH_ZIP_PASSWORD")
    try:
        payload = _http_get(UTAH_MP4_ZIP_URL)
    except urllib.error.URLError as exc:
        logger.warning("Official Utah zip download failed: %s", exc)
        return False

    with zipfile.ZipFile(BytesIO(payload)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".mp4")]
        if not names:
            logger.warning("No MP4 found inside Utah zip")
            return False
        member = names[0]
        if not password:
            logger.info(
                "Utah MP4 zip is password-protected; set UTAH_ZIP_PASSWORD or use Kaltura CDN"
            )
            return False
        tmp = dest_mp4.with_suffix(".zip_extract.part")
        with zf.open(member, pwd=password.encode("utf-8")) as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out)
        tmp.replace(dest_mp4)
    return dest_mp4.is_file() and dest_mp4.stat().st_size > 10_000


def download_kaltura_entry(entry_id: str, dest: Path) -> bool:
    """Fallback: Kaltura stream for the same Utah video."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = kaltura_download_url(entry_id)
    tmp = dest.with_suffix(".part")
    try:
        payload = _http_get(url)
        tmp.write_bytes(payload)
        tmp.replace(dest)
        return dest.is_file() and dest.stat().st_size > 10_000
    except urllib.error.URLError as exc:
        logger.warning("Kaltura download failed for %s: %s", entry_id, exc)
        tmp.unlink(missing_ok=True)
        return False


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
            timeout=10,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def transcode_h264(source: Path, dest: Path) -> bool:
    """Re-encode to OpenCV-friendly H.264 MP4, preserving aspect ratio."""
    if not _ffmpeg_available():
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
        return dest.is_file()

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".transcode.part.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("FFmpeg transcode failed: %s", exc)
        tmp.unlink(missing_ok=True)
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
        return dest.is_file()

    tmp.replace(dest)
    return dest.is_file()


def opencv_validate(video_path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    result = {
        "opened": cap.isOpened(),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "decoded_samples": {},
    }
    if not cap.isOpened():
        cap.release()
        return result

    total = result["frame_count"]
    sample_indices = sorted(
        {0, max(0, total // 4), max(0, total // 2), max(0, 3 * total // 4), max(0, total - 1)}
    )
    for index in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, _ = cap.read()
        result["decoded_samples"][str(index)] = bool(ok)
    cap.release()
    result["duration_s"] = (
        result["frame_count"] / result["fps"] if result["fps"] > 0 else 0.0
    )
    return result


def opencv_can_decode(video_path: Path) -> bool:
    info = opencv_validate(video_path)
    if not (
        info["opened"]
        and info["frame_count"] > 0
        and info["fps"] > 0
        and info["width"] > 0
        and info["height"] > 0
    ):
        return False
    samples = info.get("decoded_samples") or {}
    return samples and all(samples.values())


def prepare_abnormal_demo(
    dest: Path | None = None,
    *,
    trim: bool = True,
    output_frames: int = 150,
) -> tuple[bool, dict]:
    from stablewalk.ui.media.demo_prepare import trim_to_gait_segment
    from stablewalk.ui.media.demo_validation import report_to_dict, validate_demo_video

    config.ensure_output_dirs()
    dest = dest or (config.DEMO_VIDEOS_DIR / "abnormal_gait.mp4")

    download_source = "unknown"
    original_format = "unknown"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        raw = tmp / f"{UTAH_VIDEO_ID}_raw.mp4"
        if download_kaltura_entry(UTAH_KALTURA_ENTRY, raw):
            download_source = kaltura_download_url(UTAH_KALTURA_ENTRY)
            original_format = "MP4/H.264 (Kaltura CDN — official Utah site embed)"
        elif download_official_mp4_zip(raw):
            download_source = UTAH_MP4_ZIP_URL
            original_format = "MP4 mobile (zipped, Utah official download page)"
        else:
            return False, {}

        encoded = tmp / f"{UTAH_VIDEO_ID}_h264.mp4"
        transcode_h264(raw, encoded)
        if not opencv_can_decode(encoded):
            return False, {}

        working = encoded
        trim_note = "none"
        trim_start = -1
        if trim:
            trimmed = tmp / f"{UTAH_VIDEO_ID}_trim.mp4"
            ok, start = trim_to_gait_segment(
                encoded,
                trimmed,
                output_frames=output_frames,
                scan_frames=1400,
                prefer_best_motion=True,
            )
            if ok:
                working = trimmed
                trim_start = start
                trim_note = f"trimmed from frame {start}, {output_frames} frames"
                transcode_h264(trimmed, dest)
            else:
                transcode_h264(encoded, dest)
        else:
            transcode_h264(encoded, dest)

        final_path = dest if dest.is_file() else working
        report = validate_demo_video(final_path, max_frames=80)

        meta = {
            "source_institution": "University of Utah – NeuroLogic Examination",
            "source_page": UTAH_GAIT_ABNORMAL_PAGE,
            "download_page": UTAH_DOWNLOAD_PAGE,
            "video_title": UTAH_VIDEO_TITLE,
            "utah_video_identifier": UTAH_VIDEO_ID,
            "kaltura_entry_id": UTAH_KALTURA_ENTRY,
            "download_url": download_source,
            "gait_type": "Neuropathic gait",
            "gait_description": GAIT_DESCRIPTION,
            "original_format": original_format,
            "final_codec": "H.264 / yuv420p / MP4",
            "trim": trim_note,
            "trim_start_frame": trim_start,
            "trim_output_frames": output_frames if trim and trim_start >= 0 else None,
            "opencv": opencv_validate(final_path),
            "validation": report_to_dict(report),
            "local_path": str(dest.resolve()),
        }
        UTAH_METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return True, meta


def ensure_utah_abnormal_demo(*, force: bool = False) -> bool:
    from stablewalk.ui.media.demo_gait import demo_cached_file_ready, demo_path, example_by_key

    ex = example_by_key("abnormal")
    if ex is None:
        return False
    dest = demo_path(ex)
    if dest.is_file() and not force and demo_cached_file_ready(ex):
        return True
    ok, _ = prepare_abnormal_demo(dest)
    return ok and dest.is_file()
