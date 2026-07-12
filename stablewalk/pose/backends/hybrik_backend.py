"""
Experimental HybrIK backend adapter (placeholder).
"""

from __future__ import annotations

from stablewalk.pose.backends.base import BackendUnavailableError, HumanMotionBackend
from stablewalk.pose.backends.environment import HMR_COMPATIBILITY_NOTES
from stablewalk.pose.backends.types import CoordinateSystemMetadata, HumanMotionFrame

HYBRIK_COORDINATE_SYSTEM = CoordinateSystemMetadata(
    name="hybrik_smpl",
    units="meters",
    origin_description="HybrIK SMPL root / camera frame",
    x_axis="+x right",
    y_axis="+y up",
    z_axis="+z forward",
    notes="Placeholder — HybrIK outputs 3D joints + SMPL pose when integrated.",
)


class HybrIKBackend(HumanMotionBackend):
    name = "hybrik"
    display_name = "HybrIK"
    description = "Hybrid analytical-neural IK for SMPL bodies (experimental, optional)."

    def __init__(self, **kwargs: object) -> None:
        _ = kwargs
        raise BackendUnavailableError(
            self.name,
            "HybrIK dependencies are not installed. "
            f"{HMR_COMPATIBILITY_NOTES['hybrik']}",
        )

    @classmethod
    def is_available(cls) -> bool:
        try:
            import hybrik  # type: ignore  # noqa: F401

            return False  # import alone is insufficient until adapter is wired
        except ImportError:
            return False

    @classmethod
    def availability_reason(cls) -> str:
        return "HybrIK dependencies are not installed"

    @classmethod
    def dependency_summary(cls) -> list[str]:
        return ["torch", "hybrik", "SMPL", "CUDA (recommended)"]

    @property
    def coordinate_system(self) -> CoordinateSystemMetadata:
        return HYBRIK_COORDINATE_SYSTEM

    def reset(self) -> None:
        pass

    def process_frame(self, bgr, *, frame_index: int, timestamp_s: float) -> HumanMotionFrame:
        raise BackendUnavailableError(self.name, self.availability_reason())
