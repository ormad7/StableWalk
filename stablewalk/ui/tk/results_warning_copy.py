"""Results-tab warning presentation. Detection and pipeline status are unchanged."""

from __future__ import annotations

from typing import Literal

WarningKind = Literal["critical", "info", "none"]

# Stage notes that belong in Processing Details, not the primary strip.
_TECHNICAL_WARNING_TOKENS = (
    "coverage",
    "artifact",
    "optional",
    "worker completed",
    "tracking quality",
    "video quality",
    "sub-step",
    "derived analysis",
    "inverse kinematics",
    "marker",
    ".mot",
    ".trc",
    "technical",
)

# Raw or rewritten language that means the user should act.
_ACTION_WARNING_TOKENS = (
    "fail",
    "error",
    "could not",
    "cannot ",
    "can't ",
    "missing video",
    "no video",
    "reload",
    "re-run",
    "rerun",
    "required stage",
    "blocked",
    "abort",
    "crash",
    "needs attention",
    "action required",
    "action needed",
    "did not finish",
    "could not be opened",
    "choose another file",
    "check the video",
    "load a video",
)

# File/path misses that are processing notes, not "load another video".
_TECHNICAL_NOT_FOUND_TOKENS = (
    ".mot",
    ".trc",
    "artifact",
    "optional",
    "coverage",
    "inverse kinematics",
    "marker",
    "mot file",
    " mot",
    "mot ",
    "ik ",
    "opensim",
)


def _normalize(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def warning_is_technical(text: str) -> bool:
    lowered = _normalize(text).lower()
    return any(token in lowered for token in _TECHNICAL_WARNING_TOKENS)


def classify_results_warning(text: str, *, status: str = "") -> WarningKind:
    """Informational technical note vs a warning that needs user action."""
    if str(status or "").strip().lower() == "failed":
        return "critical"
    raw = _normalize(text)
    if not raw or raw in {"None", "—", "-"}:
        return "none"
    lowered = raw.lower()
    technical = any(token in lowered for token in _TECHNICAL_WARNING_TOKENS)
    action = any(token in lowered for token in _ACTION_WARNING_TOKENS)
    if "not found" in lowered:
        if any(token in lowered for token in _TECHNICAL_NOT_FOUND_TOKENS):
            return "info"
        return "critical"
    if action and technical:
        if any(token in lowered for token in _TECHNICAL_NOT_FOUND_TOKENS):
            return "info"
        return "critical"
    if action:
        return "critical"
    if technical:
        return "info"
    return "info"


def warning_needs_user_action(text: str, *, status: str = "") -> bool:
    """True when a warning should appear in the primary Results strip."""
    return classify_results_warning(text, status=status) == "critical"


def format_results_warning_copy(
    text: str,
    *,
    label: str = "",
    status: str = "",
    kind: WarningKind | None = None,
) -> str:
    """Rewrite detected warning text for examiners. Does not change detection."""
    raw = _normalize(text)
    resolved: WarningKind = kind or classify_results_warning(raw, status=status)
    if resolved == "none":
        return ""
    lowered = raw.lower()
    name = str(label or "").strip()

    if resolved == "critical":
        if any(
            token in lowered
            for token in (
                "missing video",
                "no video",
                "could not",
                "cannot ",
                "can't ",
                "could not be opened",
                "check the video",
                "load a video",
            )
        ):
            return (
                "The video could not be opened. Choose another file and run analysis again."
            )
        if (
            str(status or "").strip().lower() == "failed"
            or "fail" in lowered
            or "error" in lowered
            or "crash" in lowered
            or "abort" in lowered
            or "did not finish" in lowered
        ):
            if name:
                return (
                    f"{name} did not finish. Open Processing Details to see what happened."
                )
            return "A required step did not finish. Open Processing Details to see what happened."
        if "not found" in lowered:
            return (
                "A required file could not be found. Check the video source and run analysis again."
            )
        if name:
            return f"{name} needs attention. Open Processing Details to see what happened."
        return "This analysis needs attention. Open Processing Details to see what happened."

    if any(token in lowered for token in ("coverage", "marker", "inverse kinematics")):
        return "Some joints were not tracked in every frame."
    if "optional" in lowered or "artifact" in lowered or "not found" in lowered:
        return "An optional output was not produced."
    if "quality" in lowered:
        return "Tracking quality is limited. Results are still available."
    if "worker completed" in lowered:
        return "This step finished with an optional note."
    if raw and raw not in {"None", "—", "-"}:
        return raw
    return "See stage details."


def format_primary_action_warning(text: str) -> str:
    """Prefix critical copy so the examiner sees that action is required."""
    message = _normalize(text)
    if not message:
        return ""
    if message.lower().startswith("action needed"):
        return message
    return f"Action needed — {message}"
