"""Project-wide paths and default settings."""

from __future__ import annotations

import os
from pathlib import Path

# Repository root (parent of the stablewalk package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data layout (research pipeline artifacts) ---
DATA_DIR = PROJECT_ROOT / "data"
CATALOG_DIR = DATA_DIR / "catalog"

# Uploaded / local walking videos
VIDEOS_DIR = DATA_DIR / "videos"
LEGACY_VIDEOS_DIR = DATA_DIR / "input"  # pre-restructure location
INPUT_DIR = VIDEOS_DIR  # backward-compatible alias

# Predefined demo/comparison gait videos for presentations (user places files here)
DEMO_VIDEOS_DIR = DATA_DIR / "demo_videos"

OUTPUT_DIR = DATA_DIR / "output"
FRAMES_DIR = OUTPUT_DIR / "frames"
POSES_DIR = OUTPUT_DIR / "poses"
METRICS_DIR = OUTPUT_DIR / "metrics"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
OVERLAYS_DIR = VISUALIZATIONS_DIR / "overlays"
REPORTS_DIR = OUTPUT_DIR / "reports"
OPENSIM_DIR = OUTPUT_DIR / "opensim"  # OpenSim-compatible exports (.trc/.mot/.json)
TRACKING_EXPORT_DIR = OUTPUT_DIR / "tracking"  # Playback DOF tracking CSV/JSON exports
ANALYSIS_EXPORT_DIR = OUTPUT_DIR / "analysis"  # Selected-point analysis CSV/JSON exports
SESSION_EXPORT_DIR = OUTPUT_DIR / "sessions"  # Full session bundle folders (CSV + JSON)
MOTION_REFERENCE_EXPORT_DIR = OUTPUT_DIR / "motion_reference"  # Real-to-Sim retargeting .npz
OPENSIM_MODELS_DIR = PROJECT_ROOT / "models" / "opensim"  # user-provided .osim models
SESSIONS_DB_PATH = OUTPUT_DIR / "sessions" / "stablewalk_sessions.db"

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

# Pose / HMR backend selection.
# Primary: mediapipe | smpl | auto
# Legacy aliases romp/hybrik/wham resolve to the SMPL stack.
POSE_BACKEND = os.environ.get("POSE_BACKEND", "mediapipe").strip().lower()
POSE_BACKEND_ALLOW_FALLBACK = os.environ.get(
    "POSE_BACKEND_ALLOW_FALLBACK", "true"
).strip().lower() in ("1", "true", "yes", "on")
SUPPORTED_POSE_BACKENDS = ("mediapipe", "smpl", "auto")

# Licensed SMPL / SMPL-X model directories (user-provided — never auto-downloaded).
# Set SMPL_MODEL_DIR env var to folder containing SMPL_NEUTRAL.pkl.
_smpl_dir = os.environ.get("SMPL_MODEL_DIR", "").strip()
SMPL_MODEL_DIR: Path | None = Path(_smpl_dir).expanduser() if _smpl_dir else None
_smplx_dir = os.environ.get("SMPLX_MODEL_DIR", "").strip()
SMPLX_MODEL_DIR: Path | None = Path(_smplx_dir).expanduser() if _smplx_dir else None

# Per-frame image mode: reliable detection on each frame (legacy two-pass pipeline)
DEFAULT_POSE_IMAGE_MODE = True

# Main pipeline: VIDEO mode temporal tracking on single decode
DEFAULT_PIPELINE_VIDEO_MODE = True

# Save JPG frames during single-pass processing (needed for GUI video panel)
DEFAULT_CACHE_FRAMES = True

# Minimum detection confidence to accept a pose
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.5
DEFAULT_MIN_TRACKING_CONFIDENCE = 0.5

# Subject stature (m) for monocular pixel-to-meter scaling (walking speed, clearance cm).
# Override via STABLEWALK_SUBJECT_HEIGHT_M environment variable.
_subject_h = os.environ.get("STABLEWALK_SUBJECT_HEIGHT_M", "").strip()
DEFAULT_SUBJECT_HEIGHT_M: float = (
    float(_subject_h) if _subject_h else 1.70
)

# Default gait demo: pose-verified man's walk (full body, ~100% detection).
DEFAULT_WALKING_VIDEO_URL = "https://www.pexels.com/download/video/5319095/"

# GUI pipeline: "quick" = open+read frames only; True = full pose sampling; False = skip
GUI_VIDEO_VALIDATE_MODE = "quick"

# Fraction of sampled frames that must pass full-body validation
DEFAULT_MIN_VALID_FRAME_RATIO = 0.25
DEFAULT_VIDEO_VALIDATION_SAMPLES = 20

# Presentation demo: analyze full demo walking segment (~4.8 s @ 25 fps)
DEMO_MAX_FRAMES = 120

# Pose/frames are always regenerated on load (keyed by content hash in video_source)
DISABLE_POSE_CACHE_REUSE = True

# GUI: limit frames per load for responsive switching (set None for full video)
GUI_MAX_FRAMES_PER_LOAD = 80

# --- GUI playback performance (Tk dashboard) ---
# Lower values = smoother UI but higher CPU; raise strides on slower machines.
GUI_PLAYBACK_HZ = 24
GUI_PANEL_UPDATE_STRIDE_PLAYING = 3
GUI_CHART_UPDATE_STRIDE_PLAYING = 6
GUI_3D_UPDATE_STRIDE_PLAYING = 2
GUI_RGB_CACHE_ENTRIES = 128

# Frame stride for very long local videos (1 = every frame)
DEFAULT_FRAME_STRIDE = 1
LARGE_VIDEO_FRAME_THRESHOLD = 1800
LARGE_VIDEO_FRAME_STRIDE = 2


def video_search_dirs() -> list[Path]:
    """Directories scanned for local walking videos (new + legacy layout)."""
    dirs: list[Path] = []
    for candidate in (VIDEOS_DIR, LEGACY_VIDEOS_DIR):
        if candidate not in dirs:
            dirs.append(candidate)
    return dirs


def resolve_catalog_path(filename: str) -> Path:
    """Resolve a catalog file, preferring data/catalog/ with legacy fallback."""
    primary = CATALOG_DIR / filename
    if primary.is_file():
        return primary
    legacy = DATA_DIR / filename
    if legacy.is_file():
        return legacy
    return primary


def ensure_output_dirs() -> None:
    """Create standard data and output directories if they do not exist."""
    for path in (
        VIDEOS_DIR,
        LEGACY_VIDEOS_DIR,
        DEMO_VIDEOS_DIR,
        CATALOG_DIR,
        FRAMES_DIR,
        POSES_DIR,
        METRICS_DIR,
        VISUALIZATIONS_DIR,
        OVERLAYS_DIR,
        REPORTS_DIR,
        OPENSIM_DIR,
        TRACKING_EXPORT_DIR,
        ANALYSIS_EXPORT_DIR,
        SESSION_EXPORT_DIR,
        MOTION_REFERENCE_EXPORT_DIR,
        OPENSIM_MODELS_DIR,
        SESSIONS_DB_PATH.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def find_default_opensim_model() -> Path | None:
    """No automatic model selection — callers must pass an explicit ``.osim`` path."""
    return None


def list_input_videos() -> list[Path]:
    """Return video files in data/videos (and legacy data/input), sorted by name."""
    seen: set[str] = set()
    videos: list[Path] = []
    for folder in video_search_dirs():
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            key = path.name.lower()
            if key in seen:
                continue
            seen.add(key)
            videos.append(path)
    return sorted(videos, key=lambda p: p.name.lower())


def resolve_video_path(video_arg: str | None) -> Path:
    """
    Resolve a user-supplied path or pick a default video from data/videos.

    Raises:
        FileNotFoundError: No matching file, with a message listing available videos.
    """
    if video_arg:
        candidate = Path(video_arg)
        if candidate.is_file():
            return candidate.resolve()
        for folder in video_search_dirs():
            alt = folder / candidate.name
            if alt.is_file():
                return alt.resolve()
        available = list_input_videos()
        msg = f"Video not found: {video_arg}"
        if available:
            names = "\n  ".join(str(v.name) for v in available)
            msg += f"\n\nVideos in {VIDEOS_DIR}:\n  {names}"
        else:
            msg += (
                f"\n\nPut a walking video in {VIDEOS_DIR} "
                f"(e.g. my_walk.mp4) and run again."
            )
        raise FileNotFoundError(msg)

    available = list_input_videos()
    if not available:
        raise FileNotFoundError(
            f"No videos in {VIDEOS_DIR}.\n"
            f"Add a file such as: {VIDEOS_DIR / 'my_walk.mp4'}"
        )

    for preferred in DEFAULT_VIDEO_NAMES:
        for video in available:
            if video.name.lower() == preferred.lower():
                return video.resolve()

    return available[0].resolve()
