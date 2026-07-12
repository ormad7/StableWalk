"""
Abstract interface for pose / human mesh recovery backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from stablewalk.pose.backends.types import CoordinateSystemMetadata, HumanMotionFrame, HumanMotionSequence


class BackendUnavailableError(RuntimeError):
    """Raised when a configured backend cannot run (missing dependencies)."""

    def __init__(self, backend_name: str, reason: str) -> None:
        self.backend_name = backend_name
        self.reason = reason
        super().__init__(f"Backend unavailable: {backend_name} — {reason}")


class HumanMotionBackend(ABC):
    """
    Pluggable backend for monocular video → 3D human motion.

    Implementations must populate canonical joint ids in ``HumanMotionFrame``.
    """

    name: str = "base"
    display_name: str = "Base Backend"
    description: str = ""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """True when runtime dependencies for this backend are importable."""

    @classmethod
    def availability_reason(cls) -> str:
        """Human-readable reason when ``is_available()`` is False."""
        return "Dependencies not installed"

    @classmethod
    def dependency_summary(cls) -> list[str]:
        """Packages required by this backend (for documentation / diagnostics)."""
        return []

    @property
    @abstractmethod
    def coordinate_system(self) -> CoordinateSystemMetadata:
        """Coordinate frame metadata for outputs from this backend."""

    @abstractmethod
    def reset(self) -> None:
        """Reset temporal state before a new video."""

    @abstractmethod
    def process_frame(
        self,
        bgr: np.ndarray,
        *,
        frame_index: int,
        timestamp_s: float,
    ) -> HumanMotionFrame:
        """Process one BGR video frame."""

    def process_video(
        self,
        video_path: str,
        *,
        max_frames: int | None = None,
    ) -> HumanMotionSequence:
        """Default video loop using ``VideoReader``."""
        from stablewalk.pose.video import VideoReader

        self.reset()
        frames: list[HumanMotionFrame] = []
        with VideoReader(video_path) as reader:
            meta = reader.metadata
            for vf in reader.iter_frames(max_frames=max_frames):
                frames.append(
                    self.process_frame(
                        vf.bgr,
                        frame_index=vf.index,
                        timestamp_s=vf.timestamp_s,
                    )
                )
        return HumanMotionSequence(
            frames=frames,
            fps=meta.fps,
            source_video=str(video_path),
            backend_name=self.name,
            coordinate_system=self.coordinate_system,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "available": self.is_available(),
            "availability_reason": self.availability_reason(),
            "dependencies": self.dependency_summary(),
            "coordinate_system": self.coordinate_system.to_dict(),
        }
