"""
Experimental ROMP / SMPL backend adapter (placeholder).

ROMP is not bundled with StableWalk. Install in a separate ``stablewalk-hmr``
research environment if you want to experiment with this backend.
"""

from __future__ import annotations

import numpy as np

from stablewalk.pose.backends.base import BackendUnavailableError, HumanMotionBackend
from stablewalk.pose.backends.environment import HMR_COMPATIBILITY_NOTES
from stablewalk.pose.backends.types import CoordinateSystemMetadata, HumanMotionFrame

ROMP_COORDINATE_SYSTEM = CoordinateSystemMetadata(
    name="romp_smpl_camera",
    units="meters",
    origin_description="Camera-centered SMPL root (ROMP convention)",
    x_axis="+x right",
    y_axis="+y up",
    z_axis="+z forward",
    notes="Placeholder adapter — map ROMP joints to canonical ids when integrated.",
)


class ROMPBackend(HumanMotionBackend):
    """Optional ROMP / SMPL mesh recovery backend (not installed by default)."""

    name = "romp"
    display_name = "ROMP (SMPL)"
    description = "Monocular SMPL mesh recovery via ROMP (experimental, optional)."

    def __init__(self, **kwargs: object) -> None:
        _ = kwargs
        if not self.is_available():
            raise BackendUnavailableError(
                self.name,
                "ROMP dependencies are not installed. "
                f"{HMR_COMPATIBILITY_NOTES['romp']}",
            )
        raise BackendUnavailableError(
            self.name,
            "ROMP adapter is a placeholder — install ROMP in stablewalk-hmr and "
            "implement inference wiring before use.",
        )

    @classmethod
    def is_available(cls) -> bool:
        ok, _ = cls._import_romp()
        return ok

    @classmethod
    def _import_romp(cls) -> tuple[bool, object | None]:
        try:
            import romp  # type: ignore

            return True, romp
        except ImportError:
            return False, None

    @classmethod
    def availability_reason(cls) -> str:
        if cls.is_available():
            return "ROMP importable but StableWalk adapter not fully wired"
        return "ROMP dependencies are not installed"

    @classmethod
    def dependency_summary(cls) -> list[str]:
        return ["torch", "romp", "smpl/smplx models", "CUDA (recommended)"]

    @property
    def coordinate_system(self) -> CoordinateSystemMetadata:
        return ROMP_COORDINATE_SYSTEM

    def reset(self) -> None:
        pass

    def process_frame(
        self,
        bgr: np.ndarray,
        *,
        frame_index: int,
        timestamp_s: float,
    ) -> HumanMotionFrame:
        raise BackendUnavailableError(self.name, self.availability_reason())
