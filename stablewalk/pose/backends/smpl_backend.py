"""
SMPL / SMPL-X human mesh recovery backend (optional).

Uses ROMP when PyTorch, ROMP, and licensed SMPL model files are configured.
Does not synthesize fake SMPL parameters — returns empty detections when inference fails.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from stablewalk.pose.backends.base import BackendUnavailableError, HumanMotionBackend
from stablewalk.pose.backends.smpl_providers.romp_provider import RompSmplProvider
from stablewalk.pose.backends.smpl_validation import validate_smpl_assets
from stablewalk.pose.backends.types import CoordinateSystemMetadata, HumanMotionFrame

logger = logging.getLogger(__name__)

SMPL_COORDINATE_SYSTEM = CoordinateSystemMetadata(
    name="smpl_camera_meters",
    units="meters",
    origin_description="Camera-centered SMPL root (ROMP convention)",
    x_axis="+x right",
    y_axis="+y up",
    z_axis="+z forward (into scene)",
    notes=(
        "Mesh-based monocular recovery — differs from MediaPipe landmark heuristics. "
        "Requires licensed SMPL model files and ROMP in stablewalk-hmr env."
    ),
)


class SMPLPoseBackend(HumanMotionBackend):
    """SMPL body model extraction via ROMP (when fully configured)."""

    name = "smpl"
    display_name = "SMPL (ROMP)"
    description = (
        "Monocular SMPL mesh recovery via ROMP — optional; requires PyTorch, "
        "licensed SMPL models, and GPU for practical use."
    )

    def __init__(self, **kwargs: object) -> None:
        _ = kwargs
        validation = validate_smpl_assets()
        if not validation.ready:
            raise BackendUnavailableError("smpl", validation.summary())
        self._provider = RompSmplProvider()
        self._provider_name = self._provider.name

    @classmethod
    def is_available(cls) -> bool:
        return RompSmplProvider.is_available()

    @classmethod
    def availability_reason(cls) -> str:
        return validate_smpl_assets().summary()

    @classmethod
    def dependency_summary(cls) -> list[str]:
        return [
            "torch",
            "romp",
            "SMPL_NEUTRAL.pkl (licensed — set SMPL_MODEL_DIR)",
            "CUDA (recommended)",
        ]

    @property
    def coordinate_system(self) -> CoordinateSystemMetadata:
        return SMPL_COORDINATE_SYSTEM

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def close(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def process_frame(
        self,
        bgr: np.ndarray,
        *,
        frame_index: int,
        timestamp_s: float,
    ) -> HumanMotionFrame:
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            parsed = self._provider.process_image_rgb(image_rgb)
        except Exception as exc:
            logger.warning("SMPL inference failed frame %d: %s", frame_index, exc)
            parsed = None

        if not parsed:
            return HumanMotionFrame(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                joint_positions_3d={},
                landmark_confidence={},
                backend_name=self.name,
                coordinate_system=self.coordinate_system,
                detected=False,
                metadata={"provider": self._provider_name, "error": "inference_failed"},
            )

        fields = self._provider.to_canonical_frame(
            parsed,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
        )
        conf = float(fields.get("confidence", 0.85))
        positions = fields["joint_positions_3d"]
        confidence = {jid: conf for jid in positions}

        return HumanMotionFrame(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            joint_positions_3d=positions,
            landmark_confidence=confidence,
            backend_name=self.name,
            coordinate_system=self.coordinate_system,
            detected=bool(fields.get("detected", False)),
            joint_rotations=fields.get("joint_rotations") or None,
            root_position=fields.get("root_position"),
            root_orientation=fields.get("root_orientation"),
            body_shape=fields.get("body_shape") or None,
            metadata=dict(fields.get("metadata", {})),
        )

    def diagnostics(self) -> dict[str, Any]:
        base = super().diagnostics()
        base["validation"] = validate_smpl_assets().to_dict()
        base["provider"] = self._provider_name
        return base


__all__ = ["SMPLPoseBackend", "SMPL_COORDINATE_SYSTEM"]
