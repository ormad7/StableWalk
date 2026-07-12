"""
ROMP-based SMPL video extraction provider.

Requires PyTorch, the ``romp`` package, and licensed SMPL model files
(see ``smpl_validation`` and ``docs/SMPL_BACKEND_SETUP.md``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from stablewalk.pose.backends.smpl_joint_map import (
    smpl_joints_to_canonical,
    smpl_thetas_to_joint_rotations,
)
from stablewalk.pose.backends.smpl_validation import (
    resolve_smpl_model_dir,
    validate_smpl_assets,
)

logger = logging.getLogger(__name__)


class RompSmplProvider:
    """Run monocular SMPL mesh recovery through ROMP when fully configured."""

    name = "romp"

    def __init__(self) -> None:
        validation = validate_smpl_assets()
        if not validation.ready:
            raise RuntimeError(validation.summary())

        self._smpl_dir = resolve_smpl_model_dir()
        self._model = self._create_romp_model()
        self._device = self._detect_device()

    @staticmethod
    def is_available() -> bool:
        return validate_smpl_assets().ready

    @staticmethod
    def availability_reason() -> str:
        return validate_smpl_assets().summary()

    def _detect_device(self) -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _create_romp_model(self) -> Any:
        from romp import ROMP, romp_settings

        settings = romp_settings()
        settings.show = False
        settings.render = False
        settings.save_obj = False
        settings.save_mesh = False
        if hasattr(settings, "mode"):
            settings.mode = "image"
        if self._smpl_dir is not None:
            for attr in ("smpl_model_path", "smpl_path", "model_path"):
                if hasattr(settings, attr):
                    setattr(settings, attr, str(self._smpl_dir))
                    break

        model = ROMP(settings)
        if self._device == "cuda":
            try:
                model = model.cuda()
            except Exception as exc:
                logger.warning("ROMP CUDA init failed (%s) — using CPU", exc)
                self._device = "cpu"
        return model

    def process_image_rgb(self, image_rgb: np.ndarray) -> dict[str, Any] | None:
        """Run ROMP on one RGB uint8 image. Returns parsed SMPL outputs or None."""
        outputs = self._model(image_rgb)
        return self._parse_romp_output(outputs)

    def _parse_romp_output(self, outputs: Any) -> dict[str, Any] | None:
        if outputs is None:
            return None

        data: dict[str, Any] = {}
        if isinstance(outputs, dict):
            data = outputs
        elif isinstance(outputs, (list, tuple)) and outputs:
            data = outputs[0] if isinstance(outputs[0], dict) else {}
        else:
            return None

        parsed: dict[str, Any] = {"provider": self.name}

        for key in ("joints", "joints3d", "j3d", "kp3d"):
            if key in data:
                parsed["joints3d"] = np.asarray(data[key], dtype=np.float64)
                break

        for key in ("smpl_thetas", "thetas", "poses", "pose"):
            if key in data:
                parsed["smpl_thetas"] = np.asarray(data[key], dtype=np.float64)
                break

        for key in ("smpl_betas", "betas", "shape"):
            if key in data:
                parsed["smpl_betas"] = np.asarray(data[key], dtype=np.float64)
                break

        for key in ("cam", "camera", "cam_trans"):
            if key in data:
                parsed["cam"] = np.asarray(data[key], dtype=np.float64)
                break

        for key in ("trans", "transl", "translation"):
            if key in data:
                parsed["trans"] = np.asarray(data[key], dtype=np.float64)
                break

        if "joints3d" not in parsed and "smpl_thetas" not in parsed:
            return None

        conf = data.get("confidence")
        if conf is not None:
            parsed["confidence"] = float(np.asarray(conf).reshape(-1)[0])
        else:
            parsed["confidence"] = 0.85

        return parsed

    def to_canonical_frame(
        self,
        parsed: dict[str, Any],
        *,
        frame_index: int,
        timestamp_s: float,
    ) -> dict[str, Any]:
        """Convert parsed ROMP output to unified frame fields."""
        joints3d = parsed.get("joints3d")
        positions: dict[str, tuple[float, float, float]] = {}
        if joints3d is not None:
            j = np.asarray(joints3d, dtype=np.float64)
            if j.ndim == 3:
                j = j[0]
            positions = smpl_joints_to_canonical(j)

        thetas = parsed.get("smpl_thetas")
        rotations: dict[str, tuple[float, float, float, float]] = {}
        if thetas is not None:
            t = np.asarray(thetas, dtype=np.float64)
            if t.ndim == 3:
                t = t[0]
            rotations = smpl_thetas_to_joint_rotations(t)

        betas = parsed.get("smpl_betas")
        shape: dict[str, float] = {}
        if betas is not None:
            b = np.asarray(betas, dtype=np.float64).reshape(-1)
            for i, val in enumerate(b[:10]):
                shape[f"beta_{i}"] = float(val)

        root_pos = positions.get("pelvis")
        trans = parsed.get("trans") or parsed.get("cam")
        if trans is not None and root_pos is not None:
            t = np.asarray(trans, dtype=np.float64).reshape(-1)
            if t.size >= 3:
                root_pos = (
                    root_pos[0] + float(t[0]),
                    root_pos[1] + float(t[1]),
                    root_pos[2] + float(t[2]),
                )

        return {
            "frame_index": frame_index,
            "timestamp_s": timestamp_s,
            "joint_positions_3d": positions,
            "joint_rotations": rotations,
            "body_shape": shape,
            "root_position": root_pos,
            "root_orientation": (1.0, 0.0, 0.0, 0.0),
            "confidence": float(parsed.get("confidence", 0.85)),
            "detected": bool(positions),
            "metadata": {
                "provider": self.name,
                "smpl_thetas_present": thetas is not None,
                "joints3d_present": joints3d is not None,
            },
        }


__all__ = ["RompSmplProvider"]
