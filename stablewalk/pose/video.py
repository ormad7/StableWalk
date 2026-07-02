"""
Video processing: read walking videos and optionally extract frames to disk.

This module handles **only** video I/O (OpenCV). Pose estimation lives in
``stablewalk.pose.estimation`` and must not be imported here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import cv2
import numpy as np

from stablewalk.models.pose_data import FrameExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class VideoFrame:
    """One decoded BGR frame with timing metadata."""

    index: int
    timestamp_s: float
    timestamp_ms: int
    bgr: np.ndarray


@dataclass
class VideoMetadata:
    """Properties of an opened video source."""

    source: str
    fps: float
    width: int
    height: int
    frame_count: int | None = None


class VideoReader:
    """
    Stream frames from a local file or URL without pose logic.

    Example::

        with VideoReader("walk.mp4") as reader:
            for frame in reader.iter_frames():
                # frame.bgr, frame.timestamp_s
                ...
    """

    def __init__(self, source: str | Path) -> None:
        self._source = str(source).strip()
        self._cap: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None

    def open(self) -> VideoMetadata:
        from stablewalk.core.pipeline_reset import register_capture, release_all_captures

        release_all_captures()
        self._cap = cv2.VideoCapture(self._source)
        register_capture(self._cap)
        if not self._cap.isOpened():
            release_all_captures()
            raise RuntimeError(f"Could not open video: {self._source}")

        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count_val = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = int(count_val) if count_val and count_val > 0 else None

        self._metadata = VideoMetadata(
            source=self._source,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
        )
        return self._metadata

    def close(self) -> None:
        from stablewalk.core.pipeline_reset import release_all_captures

        release_all_captures()
        self._cap = None

    def __enter__(self) -> VideoReader:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            raise RuntimeError("VideoReader not opened; use 'with VideoReader(...) as r:'")
        return self._metadata

    def iter_frames(
        self,
        *,
        max_frames: int | None = None,
    ) -> Iterator[VideoFrame]:
        """Yield BGR frames with monotonic timestamps derived from FPS."""
        if self._cap is None:
            raise RuntimeError("VideoReader not opened")
        meta = self.metadata
        index = 0
        while True:
            if max_frames is not None and index >= max_frames:
                break
            ret, bgr = self._cap.read()
            if not ret:
                break
            ts_s = index / max(meta.fps, 1e-6)
            yield VideoFrame(
                index=index,
                timestamp_s=ts_s,
                timestamp_ms=int(round(ts_s * 1000)),
                bgr=bgr,
            )
            index += 1


def is_video_url(source: str) -> bool:
    """True when source looks like an HTTP(S) video stream."""
    parsed = urlparse(source.strip())
    return parsed.scheme in ("http", "https")


class VideoProcessor:
    """
    Extract frames from a video file and write them as JPG images.

    Use this when you want a frame cache on disk. For in-memory processing,
    use ``VideoReader`` and pass frames to ``PoseEstimator.process_video``.
    """

    def __init__(self, jpeg_quality: int = 95) -> None:
        if not 0 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 0 and 100")
        self.jpeg_quality = jpeg_quality

    def extract_frames(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        *,
        prefix: str = "frame",
        max_frames: int | None = None,
        skip_existing: bool = False,
    ) -> FrameExtractionResult:
        """Read a local video file and save frames as JPG."""
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        return self._extract_from_capture(
            str(video_path.resolve()),
            str(video_path.resolve()),
            output_dir,
            prefix=prefix,
            max_frames=max_frames,
            skip_existing=skip_existing,
        )

    def extract_frames_from_url(
        self,
        url: str,
        output_dir: str | Path,
        *,
        prefix: str = "frame",
        max_frames: int | None = None,
        skip_existing: bool = False,
    ) -> FrameExtractionResult:
        """
        Stream frames from a video URL (no full-file download).

        Only extracted JPG frames are written to disk.
        """
        url = url.strip()
        if not is_video_url(url):
            raise ValueError(f"Not a valid video URL: {url}")
        return self._extract_from_capture(
            url,
            url,
            output_dir,
            prefix=prefix,
            max_frames=max_frames,
            skip_existing=skip_existing,
        )

    def _extract_from_capture(
        self,
        source: str,
        source_label: str,
        output_dir: str | Path,
        *,
        prefix: str,
        max_frames: int | None,
        skip_existing: bool,
    ) -> FrameExtractionResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for old in output_dir.glob(f"{prefix}_*.jpg"):
            old.unlink()

        from stablewalk.core.pipeline_reset import register_capture, release_all_captures

        release_all_captures()
        cap = cv2.VideoCapture(source)
        register_capture(cap)
        if not cap.isOpened():
            release_all_captures()
            raise RuntimeError(f"Could not open video source: {source_label}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_paths: list[str] = []
        index = 0
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

        try:
            while True:
                if max_frames is not None and index >= max_frames:
                    break

                out_path = output_dir / f"{prefix}_{index:06d}.jpg"
                if skip_existing and out_path.exists():
                    frame_paths.append(str(out_path))
                    index += 1
                    cap.grab()
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                if not cv2.imwrite(str(out_path), frame, encode_params):
                    raise RuntimeError(f"Failed to write frame: {out_path}")

                frame_paths.append(str(out_path))
                index += 1

                if index % 100 == 0:
                    logger.info("Extracted %d frames...", index)
        finally:
            release_all_captures()

        display_name = source_label if len(source_label) < 80 else source_label[:77] + "..."
        logger.info(
            "Extracted %d frames from %s → %s (%.2f fps, %dx%d)",
            len(frame_paths),
            display_name,
            output_dir,
            fps,
            width,
            height,
        )

        return FrameExtractionResult(
            video_path=source_label,
            output_dir=str(output_dir.resolve()),
            frame_count=len(frame_paths),
            fps=float(fps),
            width=width,
            height=height,
            frame_paths=frame_paths,
        )

    @staticmethod
    def list_frames(frames_dir: str | Path, pattern: str = "frame_*.jpg") -> list[Path]:
        """Return sorted frame paths from a directory."""
        frames_dir = Path(frames_dir)
        return sorted(frames_dir.glob(pattern))
