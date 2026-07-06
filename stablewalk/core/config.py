"""Project-wide paths and default settings."""

from pathlib import Path

# Repository root (parent of the stablewalk package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
FRAMES_DIR = OUTPUT_DIR / "frames"
POSES_DIR = OUTPUT_DIR / "poses"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Preferred default when multiple videos exist (first match wins)
DEFAULT_VIDEO_NAMES = (
    "my_walk.mp4",
    "sample_walk.mp4",
    "walking.mp4",
)

# JPEG quality when saving extracted frames (0–100)
DEFAULT_JPEG_QUALITY = 95

# MediaPipe pose model: "lite" (fast) or "full" (accurate)
DEFAULT_POSE_MODEL_VARIANT = "full"

# Legacy two-pass pipeline: IMAGE mode on extracted stills
DEFAULT_POSE_IMAGE_MODE = True

# Main pipeline: VIDEO mode temporal tracking on single decode
DEFAULT_PIPELINE_VIDEO_MODE = True

# Save JPG frames during single-pass processing (needed for GUI video panel)
DEFAULT_CACHE_FRAMES = True

# Minimum detection confidence to accept a pose
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.5
DEFAULT_MIN_TRACKING_CONFIDENCE = 0.5

# Default gait demo: pose-verified man's walk (full body, ~100% detection).
# Pexels direct-download — OpenCV streams without downloading the whole file.
# Page: https://www.pexels.com/video/man-walking-on-sidewalk-5319095/
DEFAULT_WALKING_VIDEO_URL = "https://www.pexels.com/download/video/5319095/"

# GUI pipeline: "quick" = open+read frames only; True = full pose sampling; False = skip
GUI_VIDEO_VALIDATE_MODE = "quick"

# Fraction of sampled frames that must pass full-body validation
DEFAULT_MIN_VALID_FRAME_RATIO = 0.25
DEFAULT_VIDEO_VALIDATION_SAMPLES = 20

# Presentation demo (see stablewalk.ui.demo)
DEMO_MAX_FRAMES = 120

# Pose/frames are always regenerated on load (keyed by content hash in video_source)
DISABLE_POSE_CACHE_REUSE = True

# GUI: limit frames per load for responsive switching (set None for full video)
GUI_MAX_FRAMES_PER_LOAD = 80


def ensure_output_dirs() -> None:
    """Create standard output directories if they do not exist."""
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    POSES_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)


def list_input_videos() -> list[Path]:
    """Return video files in data/input, sorted by name."""
    if not INPUT_DIR.is_dir():
        return []
    videos = [
        p
        for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos, key=lambda p: p.name.lower())


def resolve_video_path(video_arg: str | None) -> Path:
    """
    Resolve a user-supplied path or pick a default video from data/input.

    Raises:
        FileNotFoundError: No matching file, with a message listing available videos.
    """
    if video_arg:
        candidate = Path(video_arg)
        if candidate.is_file():
            return candidate.resolve()
        alt = INPUT_DIR / candidate.name
        if alt.is_file():
            return alt.resolve()
        available = list_input_videos()
        msg = f"Video not found: {video_arg}"
        if available:
            names = "\n  ".join(str(v.name) for v in available)
            msg += f"\n\nVideos in {INPUT_DIR}:\n  {names}"
        else:
            msg += (
                f"\n\nPut a walking video in {INPUT_DIR} "
                f"(e.g. my_walk.mp4) and run again."
            )
        raise FileNotFoundError(msg)

    available = list_input_videos()
    if not available:
        raise FileNotFoundError(
            f"No videos in {INPUT_DIR}.\n"
            f"Add a file such as: {INPUT_DIR / 'my_walk.mp4'}"
        )

    for preferred in DEFAULT_VIDEO_NAMES:
        for video in available:
            if video.name.lower() == preferred.lower():
                return video.resolve()

    return available[0].resolve()
