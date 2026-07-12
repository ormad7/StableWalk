"""
Backend registry and factory.

Select backend via ``POSE_BACKEND`` environment variable or ``stablewalk.config``.
"""

from __future__ import annotations

import logging
from typing import Any, Type

from stablewalk import config
from stablewalk.pose.backends.base import BackendUnavailableError, HumanMotionBackend
from stablewalk.pose.backends.hybrik_backend import HybrIKBackend
from stablewalk.pose.backends.mediapipe_backend import MediaPipePoseBackend
from stablewalk.pose.backends.romp_backend import ROMPBackend
from stablewalk.pose.backends.wham_backend import WHAMBackend

logger = logging.getLogger(__name__)

BACKEND_REGISTRY: dict[str, Type[HumanMotionBackend]] = {
    "mediapipe": MediaPipePoseBackend,
    "romp": ROMPBackend,
    "hybrik": HybrIKBackend,
    "wham": WHAMBackend,
}

SUPPORTED_BACKENDS: tuple[str, ...] = tuple(BACKEND_REGISTRY.keys())


def list_backend_diagnostics() -> list[dict[str, Any]]:
    """Return availability diagnostics for all known backends."""
    out: list[dict[str, Any]] = []
    for name, cls in BACKEND_REGISTRY.items():
        out.append(
            {
                "name": name,
                "display_name": cls.display_name,
                "available": cls.is_available(),
                "reason": cls.availability_reason(),
                "dependencies": cls.dependency_summary(),
            }
        )
    return out


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
        One of ``mediapipe``, ``romp``, ``hybrik``, ``wham``. Defaults to
        ``config.POSE_BACKEND``.
    allow_fallback:
        When True and the requested backend is unavailable, fall back to
        MediaPipe with an explicit log message. When False, raise
        ``BackendUnavailableError``. Defaults to ``config.POSE_BACKEND_ALLOW_FALLBACK``.

    Raises
    ------
    BackendUnavailableError
        Requested backend missing and fallback disabled.
    ValueError
        Unknown backend name.
    """
    requested = (backend_name or config.POSE_BACKEND).strip().lower()
    fallback_ok = (
        config.POSE_BACKEND_ALLOW_FALLBACK
        if allow_fallback is None
        else allow_fallback
    )

    if requested not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown POSE_BACKEND '{requested}'. "
            f"Supported: {', '.join(SUPPORTED_BACKENDS)}"
        )

    cls = BACKEND_REGISTRY[requested]

    if requested == "mediapipe":
        if not cls.is_available():
            raise BackendUnavailableError("mediapipe", cls.availability_reason())
        return cls(**kwargs)

    if not cls.is_available():
        msg = (
            f"Backend unavailable: {cls.display_name} — "
            f"{cls.availability_reason()}"
        )
        if fallback_ok and requested != "mediapipe":
            logger.warning("%s — falling back to MediaPipe (allow_fallback=True)", msg)
            if not MediaPipePoseBackend.is_available():
                raise BackendUnavailableError("mediapipe", MediaPipePoseBackend.availability_reason())
            return MediaPipePoseBackend(**kwargs)
        raise BackendUnavailableError(requested, cls.availability_reason())

    if requested != "mediapipe":
        # Package importable but adapter not wired (ROMP/HybrIK/WHAM placeholders)
        msg = cls.availability_reason()
        if fallback_ok:
            logger.warning(
                "Backend unavailable: %s — %s — falling back to MediaPipe",
                cls.display_name,
                msg,
            )
            return MediaPipePoseBackend(**kwargs)
        raise BackendUnavailableError(requested, msg)

    return cls(**kwargs)


def unavailable_message(backend_name: str) -> str:
    """Formatted message for UI / logs when a backend cannot run."""
    key = backend_name.strip().lower()
    cls = BACKEND_REGISTRY.get(key)
    if cls is None:
        return f"Unknown backend: {backend_name}"
    return f"Backend unavailable: {cls.display_name} — {cls.availability_reason()}"
