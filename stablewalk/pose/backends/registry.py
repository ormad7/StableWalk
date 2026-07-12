"""
Backend registry and factory.

Select backend via ``POSE_BACKEND`` environment variable, GUI, or ``stablewalk.config``.

Primary modes: ``mediapipe``, ``smpl``, ``auto``.
Legacy aliases ``romp``, ``hybrik``, ``wham`` map to ``smpl``.
"""

from __future__ import annotations

import logging
from typing import Any, Type

from stablewalk import config
from stablewalk.pose.backends.base import BackendUnavailableError, HumanMotionBackend
from stablewalk.pose.backends.hybrik_backend import HybrIKBackend
from stablewalk.pose.backends.mediapipe_backend import MediaPipePoseBackend
from stablewalk.pose.backends.pipeline_runner import normalize_backend_mode, resolve_pose_backend
from stablewalk.pose.backends.romp_backend import ROMPBackend
from stablewalk.pose.backends.smpl_backend import SMPLPoseBackend
from stablewalk.pose.backends.wham_backend import WHAMBackend

logger = logging.getLogger(__name__)

BACKEND_REGISTRY: dict[str, Type[HumanMotionBackend]] = {
    "mediapipe": MediaPipePoseBackend,
    "smpl": SMPLPoseBackend,
    # Legacy research aliases (resolve to SMPL stack)
    "romp": ROMPBackend,
    "hybrik": HybrIKBackend,
    "wham": WHAMBackend,
}

PRIMARY_BACKENDS: tuple[str, ...] = ("mediapipe", "smpl", "auto")
SUPPORTED_BACKENDS: tuple[str, ...] = PRIMARY_BACKENDS + tuple(
    k for k in BACKEND_REGISTRY if k not in PRIMARY_BACKENDS
)


def list_backend_diagnostics() -> list[dict[str, Any]]:
    """Return availability diagnostics for all known backends."""
    from stablewalk.pose.backends.pipeline_runner import get_backend_diagnostics

    return get_backend_diagnostics()


def create_pose_backend(
    backend_name: str | None = None,
    *,
    allow_fallback: bool | None = None,
    **kwargs: Any,
) -> HumanMotionBackend:
    """
    Instantiate the configured pose / HMR backend.

    Parameters
    ----------
    backend_name:
        One of ``mediapipe``, ``smpl``, ``auto`` (or legacy ``romp``/``hybrik``/``wham``).
    allow_fallback:
        When True and SMPL is unavailable, fall back to MediaPipe with a warning.
    """
    requested = normalize_backend_mode(backend_name or config.POSE_BACKEND)
    fallback_ok = (
        config.POSE_BACKEND_ALLOW_FALLBACK
        if allow_fallback is None
        else allow_fallback
    )

    if requested in ("auto", "smpl"):
        backend, resolution = resolve_pose_backend(requested, allow_fallback=fallback_ok)
        if resolution.warnings:
            for w in resolution.warnings:
                logger.warning(w)
        return backend

    if requested == "mediapipe":
        if not MediaPipePoseBackend.is_available():
            raise BackendUnavailableError("mediapipe", MediaPipePoseBackend.availability_reason())
        return MediaPipePoseBackend(**kwargs)

    # Legacy direct backend names
    if requested not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown POSE_BACKEND '{requested}'. "
            f"Supported: {', '.join(PRIMARY_BACKENDS)}"
        )

    cls = BACKEND_REGISTRY[requested]
    if requested == "mediapipe":
        return cls(**kwargs)

    if not cls.is_available():
        msg = f"{cls.display_name} — {cls.availability_reason()}"
        if fallback_ok:
            logger.warning("Backend unavailable: %s — falling back to MediaPipe", msg)
            return MediaPipePoseBackend(**kwargs)
        raise BackendUnavailableError(requested, cls.availability_reason())

    if requested in ("romp", "hybrik", "wham"):
        msg = cls.availability_reason()
        if fallback_ok:
            logger.warning(
                "Legacy backend %s not wired — use POSE_BACKEND=smpl. Falling back to MediaPipe.",
                requested,
            )
            return MediaPipePoseBackend(**kwargs)
        raise BackendUnavailableError(requested, msg)

    return cls(**kwargs)


def unavailable_message(backend_name: str) -> str:
    """Formatted message for UI / logs when a backend cannot run."""
    key = normalize_backend_mode(backend_name)
    if key == "smpl":
        return f"Backend unavailable: SMPL — {SMPLPoseBackend.availability_reason()}"
    if key == "auto":
        from stablewalk.pose.backends.smpl_validation import validate_smpl_assets

        val = validate_smpl_assets()
        if val.ready:
            return "Auto mode: SMPL is available"
        return (
            f"Auto mode will use MediaPipe until SMPL is configured: {val.summary()}"
        )
    cls = BACKEND_REGISTRY.get(key)
    if cls is None:
        return f"Unknown backend: {backend_name}"
    return f"Backend unavailable: {cls.display_name} — {cls.availability_reason()}"


__all__ = [
    "BACKEND_REGISTRY",
    "PRIMARY_BACKENDS",
    "SUPPORTED_BACKENDS",
    "create_pose_backend",
    "list_backend_diagnostics",
    "unavailable_message",
]
