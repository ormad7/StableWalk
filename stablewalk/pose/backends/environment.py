"""
Runtime environment inspection for pose / HMR backend compatibility.

Inspects Python, OS, CUDA, and optional research dependencies **without**
installing anything. Used by backend registry and comparison scripts.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DependencyStatus:
    name: str
    installed: bool
    version: str | None = None
    notes: str = ""


@dataclass
class RuntimeEnvironmentReport:
    python_version: str
    platform: str
    architecture: str
    cuda_available: bool | None = None
    cuda_version: str | None = None
    torch_version: str | None = None
    mediapipe_version: str | None = None
    dependencies: list[DependencyStatus] = field(default_factory=list)
    recommended_environments: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "architecture": self.architecture,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
            "mediapipe_version": self.mediapipe_version,
            "dependencies": [
                {
                    "name": d.name,
                    "installed": d.installed,
                    "version": d.version,
                    "notes": d.notes,
                }
                for d in self.dependencies
            ],
            "recommended_environments": self.recommended_environments,
            "warnings": self.warnings,
        }


def _try_import(module: str) -> tuple[bool, str | None]:
    try:
        mod = __import__(module)
        return True, getattr(mod, "__version__", None)
    except ImportError:
        return False, None


def inspect_runtime_environment() -> RuntimeEnvironmentReport:
    """
    Collect environment facts for backend selection and research setup guidance.

    Does not modify the environment or install packages.
    """
    report = RuntimeEnvironmentReport(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        architecture=platform.machine(),
        recommended_environments=[
            "stablewalk-opensim — main StableWalk app (MediaPipe + OpenSim IK)",
            "stablewalk-hmr — optional research env for ROMP / HybrIK / WHAM experiments",
        ],
    )

    mp_ok, mp_ver = _try_import("mediapipe")
    report.mediapipe_version = mp_ver
    report.dependencies.append(
        DependencyStatus("mediapipe", mp_ok, mp_ver, "Default pose backend")
    )

    torch_ok, torch_ver = _try_import("torch")
    report.torch_version = torch_ver
    report.dependencies.append(
        DependencyStatus(
            "torch",
            torch_ok,
            torch_ver,
            "Required for ROMP, HybrIK, WHAM (not required for main app)",
        )
    )

    if torch_ok:
        import torch

        report.cuda_available = torch.cuda.is_available()
        if report.cuda_available:
            report.cuda_version = torch.version.cuda
        else:
            report.warnings.append(
                "PyTorch is installed but CUDA is not available — "
                "HMR backends may run on CPU only (slow) or fail."
            )
    else:
        report.cuda_available = None
        report.warnings.append(
            "PyTorch is not installed — ROMP, HybrIK, and WHAM backends "
            "cannot run in this environment."
        )

    for pkg, notes in (
        ("romp", "ROMP / SMPL mesh recovery — use POSE_BACKEND=smpl in stablewalk-hmr env"),
        ("hybrik", "HybrIK — PyTorch + SMPL; see official HybrIK repo for CUDA/PyTorch pins"),
        ("wham", "WHAM — video HMR; requires PyTorch, SMPL, CUDA for practical use"),
        ("smplx", "SMPL-X body model (set SMPLX_MODEL_DIR when using SMPL-X)"),
    ):
        ok, ver = _try_import(pkg)
        report.dependencies.append(DependencyStatus(pkg, ok, ver, notes))

    try:
        from stablewalk.pose.backends.smpl_validation import validate_smpl_assets

        smpl_val = validate_smpl_assets()
        report.dependencies.append(
            DependencyStatus(
                "smpl_backend",
                smpl_val.ready,
                None,
                smpl_val.summary(),
            )
        )
        if not smpl_val.ready:
            report.warnings.append(
                f"SMPL backend not ready: {smpl_val.summary()[:120]}"
            )
    except Exception as exc:
        report.warnings.append(f"SMPL validation check failed: {exc}")

    if sys.version_info < (3, 10):
        report.warnings.append("Python 3.10+ is recommended for MediaPipe Tasks API.")
    if sys.version_info >= (3, 12):
        report.warnings.append(
            "Python 3.12+ may lack wheels for some research HMR stacks — "
            "use a dedicated stablewalk-hmr conda env on 3.10/3.11 if needed."
        )

    return report


# Compatibility notes sourced from official project requirements (informational only).
HMR_COMPATIBILITY_NOTES: dict[str, str] = {
    "romp": (
        "ROMP (https://github.com/Arthur151/ROMP) uses PyTorch and SMPL. "
        "Install in an isolated env; do not merge into the OpenSim conda stack."
    ),
    "hybrik": (
        "HybrIK (https://github.com/Jeff-sjtu/HybrIK) pins specific PyTorch/CUDA "
        "versions. Verify against their README before installing."
    ),
    "wham": (
        "WHAM (https://github.com/yohanshin/WHAM) requires PyTorch, SMPL, and "
        "GPU for practical video processing. Experimental adapter only in StableWalk."
    ),
    "mediapipe": (
        "MediaPipe Pose Landmarker (Tasks API) is the StableWalk default — "
        "CPU-friendly, no PyTorch required."
    ),
}
