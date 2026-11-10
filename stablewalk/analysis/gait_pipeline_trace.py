"""Concise development logs for gait-analysis stages.

Statuses are ``complete``, ``failed``, ``insufficient``, or ``skipped``.
This module does not invent metric values or alter analysis math.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("stablewalk.gait_pipeline")

VALID_STATUSES = frozenset({"complete", "failed", "insufficient", "skipped"})


def log_stage(stage: str, status: str, **detail: Any) -> None:
    """Emit one line describing a pipeline stage outcome."""
    label = str(status or "insufficient").strip().lower()
    if label not in VALID_STATUSES:
        label = "insufficient"
    bits = [f"{key}={value}" for key, value in detail.items() if value is not None]
    extra = f" {' '.join(bits)}" if bits else ""
    logger.info("gait-pipeline %s: %s%s", stage, label, extra)


__all__ = ["VALID_STATUSES", "log_stage"]
