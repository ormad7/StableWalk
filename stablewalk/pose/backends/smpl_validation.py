"""
Dependency and model-file validation for the SMPL / SMPL-X backend.

StableWalk never downloads restricted SMPL assets automatically. Users must
obtain model files legally and point ``SMPL_MODEL_DIR`` (or ``SMPLX_MODEL_DIR``)
at a local directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stablewalk import config


@dataclass
class SmplValidationResult:
    """Outcome of SMPL backend readiness checks."""

    ready: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    smpl_model_dir: Path | None = None
    smplx_model_dir: Path | None = None
    torch_available: bool = False
    cuda_available: bool = False
    romp_importable: bool = False
    provider_name: str = ""

    def summary(self) -> str:
        if self.ready:
            return f"SMPL backend ready via {self.provider_name}"
        return "; ".join(self.issues) if self.issues else "SMPL backend not ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "smpl_model_dir": str(self.smpl_model_dir) if self.smpl_model_dir else None,
            "smplx_model_dir": str(self.smplx_model_dir) if self.smplx_model_dir else None,
            "torch_available": self.torch_available,
            "cuda_available": self.cuda_available,
            "romp_importable": self.romp_importable,
            "provider_name": self.provider_name,
            "summary": self.summary(),
        }


def resolve_smpl_model_dir() -> Path | None:
    """Resolve SMPL model directory from env or config (never hard-coded)."""
    raw = os.environ.get("SMPL_MODEL_DIR") or os.environ.get("SMPL_MODEL_PATH") or ""
    if not raw and config.SMPL_MODEL_DIR:
        raw = str(config.SMPL_MODEL_DIR)
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def resolve_smplx_model_dir() -> Path | None:
    """Optional SMPL-X model directory."""
    raw = os.environ.get("SMPLX_MODEL_DIR") or os.environ.get("SMPLX_MODEL_PATH") or ""
    if not raw and config.SMPLX_MODEL_DIR:
        raw = str(config.SMPLX_MODEL_DIR)
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def _find_smpl_pkl(model_dir: Path) -> Path | None:
    candidates = [
        model_dir / "SMPL_NEUTRAL.pkl",
        model_dir / "smpl" / "SMPL_NEUTRAL.pkl",
        model_dir / "basicModel_neutral_lbs_10_207_0_v1.1.0.pkl",
        model_dir / "models" / "SMPL_NEUTRAL.pkl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for pkl in model_dir.rglob("SMPL_NEUTRAL.pkl"):
        if pkl.is_file():
            return pkl
    return None


def _find_smplx_npz(model_dir: Path) -> Path | None:
    candidates = [
        model_dir / "SMPLX_NEUTRAL.npz",
        model_dir / "smplx" / "SMPLX_NEUTRAL.npz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for npz in model_dir.rglob("SMPLX_NEUTRAL.npz"):
        if npz.is_file():
            return npz
    return None


def validate_smpl_assets() -> SmplValidationResult:
    """
    Check whether SMPL model files and runtime dependencies are present.

    Does not import heavy packages unless basic checks pass.
    """
    result = SmplValidationResult(ready=False)

    smpl_dir = resolve_smpl_model_dir()
    smplx_dir = resolve_smplx_model_dir()
    result.smpl_model_dir = smpl_dir
    result.smplx_model_dir = smplx_dir

    if smpl_dir is None:
        result.issues.append(
            "SMPL_MODEL_DIR is not set or is not a directory. "
            "Obtain SMPL/SMPL-X from https://smpl.is.tue.mpg.de (or SMPL-X project page) "
            "and set SMPL_MODEL_DIR to the folder containing SMPL_NEUTRAL.pkl. "
            "See docs/SMPL_BACKEND_SETUP.md."
        )
    else:
        pkl = _find_smpl_pkl(smpl_dir)
        if pkl is None:
            result.issues.append(
                f"No SMPL_NEUTRAL.pkl found under {smpl_dir}. "
                "Place the licensed SMPL neutral model file in that directory."
            )

    if smplx_dir is not None and _find_smplx_npz(smplx_dir) is None:
        result.warnings.append(
            f"SMPLX_MODEL_DIR set ({smplx_dir}) but SMPLX_NEUTRAL.npz not found — "
            "SMPL-X mode unavailable; SMPL via ROMP may still work."
        )

    try:
        import torch  # noqa: F401

        result.torch_available = True
        import torch as th

        result.cuda_available = bool(th.cuda.is_available())
        if not result.cuda_available:
            result.warnings.append(
                "PyTorch CUDA is not available — SMPL inference may run on CPU (slow)."
            )
    except ImportError:
        result.issues.append(
            "PyTorch is not installed. Install torch in a separate stablewalk-hmr conda env."
        )

    try:
        import romp  # noqa: F401

        result.romp_importable = True
        result.provider_name = "romp"
    except ImportError:
        result.issues.append(
            "ROMP is not installed. Install ROMP in stablewalk-hmr to enable SMPL video "
            "extraction: https://github.com/Arthur151/ROMP"
        )

    result.ready = (
        len(result.issues) == 0
        and smpl_dir is not None
        and _find_smpl_pkl(smpl_dir) is not None
        and result.torch_available
        and result.romp_importable
    )
    return result


__all__ = [
    "SmplValidationResult",
    "resolve_smpl_model_dir",
    "resolve_smplx_model_dir",
    "validate_smpl_assets",
]
