"""Download and cache MediaPipe task models."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

from stablewalk.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"

# Official MediaPipe pose landmarker bundles (lite = fast, full = accurate)
POSE_MODELS = {
    "lite": {
        "filename": "pose_landmarker_lite.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/1/"
            "pose_landmarker_lite.task"
        ),
    },
    "full": {
        "filename": "pose_landmarker_full.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_full/float16/1/"
            "pose_landmarker_full.task"
        ),
    },
}


def get_pose_model_path(variant: str = "lite", download: bool = True) -> Path:
    """
    Return path to pose landmarker `.task` file, downloading on first use.

    Args:
        variant: ``lite`` or ``full``.
        download: If True, fetch the model when missing.
    """
    if variant not in POSE_MODELS:
        raise ValueError(f"Unknown model variant: {variant}. Use: lite, full")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    info = POSE_MODELS[variant]
    path = MODELS_DIR / info["filename"]

    if path.is_file():
        return path

    if not download:
        raise FileNotFoundError(f"Model not found: {path}")

    logger.info("Downloading pose model (%s)...", variant)
    urllib.request.urlretrieve(info["url"], path)
    logger.info("Saved model → %s", path)
    return path
