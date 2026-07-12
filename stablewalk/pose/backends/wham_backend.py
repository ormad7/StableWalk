"""
Experimental WHAM backend adapter (placeholder).

WHAM (World-grounded Humans with Accurate Motion) requires PyTorch, SMPL, and
typically GPU acceleration for video sequences.
"""

from __future__ import annotations

from stablewalk.pose.backends.base import BackendUnavailableError, HumanMotionBackend
from stablewalk.pose.backends.environment import HMR_COMPATIBILITY_NOTES
from stablewalk.pose.backends.types import CoordinateSystemMetadata, HumanMotionFrame

WHAM_COORDINATE_SYSTEM = CoordinateSystemMetadata(
    name="wham_world",
    units="meters",
    origin_description="WHAM world / root trajectory frame",
    x_axis="+x forward (walking)",
    y_axis="+y up",
    z_axis="+z lateral",
    notes="Placeholder — WHAM provides world-grounded root + SMPL when integrated.",
)


class WHAMBackend(HumanMotionBackend):
    name = "wham"
    display_name = "WHAM"
    description = "World-grounded video HMR (experimental, optional)."

    def __init__(self, **kwargs: object) -> None:
        _ = kwargs
        raise BackendUnavailableError(
            self.name,
            "WHAM dependencies are not installed. "
            f"{HMR_COMPATIBILITY_NOTES['wham']}",
        )

    @classmethod
    def is_available(cls) -> bool:
        try:
            import wham  # type: ignore  # noqa: F401

            return False
        except ImportError:
            return False

    @classmethod
    def availability_reason(cls) -> str:
        return "WHAM dependencies are not installed"

    @classmethod
    def dependency_summary(cls) -> list[str]:
        return ["torch", "wham", "SMPL", "CUDA (strongly recommended)"]

    @property
    def coordinate_system(self) -> CoordinateSystemMetadata:
        return WHAM_COORDINATE_SYSTEM

    def reset(self) -> None:
        pass

    def process_frame(self, bgr, *, frame_index: int, timestamp_s: float) -> HumanMotionFrame:
        raise BackendUnavailableError(self.name, self.availability_reason())
