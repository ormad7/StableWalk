"""GUI analysis window: full source vs labeled preview cap."""

from __future__ import annotations

MODE_FULL = "full"
MODE_PREVIEW = "preview"

PREVIEW_LABEL = "Preview"
FULL_LABEL = "Full analysis"


def resolve_gui_max_frames(
    *,
    mode: str,
    is_demo: bool,
    explicit_max_frames: int | None = None,
) -> tuple[int | None, str]:
    """Return (max_frames, mode) for Examples / Analyze.

    Full analysis never applies ``DEMO_MAX_FRAMES`` or ``GUI_MAX_FRAMES_PER_LOAD``.
    Preview is the capped path and must be labeled as such.
    """
    requested = str(mode or "").strip().lower()
    if requested == MODE_FULL:
        return None, MODE_FULL
    if explicit_max_frames is not None:
        return int(explicit_max_frames), MODE_PREVIEW
    from stablewalk import config

    if is_demo:
        cap = getattr(config, "DEMO_MAX_FRAMES", None)
        return (int(cap) if cap else 900), MODE_PREVIEW
    cap = getattr(config, "GUI_MAX_FRAMES_PER_LOAD", None)
    return (int(cap) if cap else 240), MODE_PREVIEW


def format_analysis_mode_label(
    mode: str | None,
    max_frames: int | None,
) -> str:
    """Short UI label. Preview always names the frame cap."""
    if str(mode or "").lower() == MODE_PREVIEW or max_frames is not None:
        n = max_frames if max_frames is not None else 900
        return f"{PREVIEW_LABEL} · first {int(n)} frames"
    return FULL_LABEL


def normalize_analysis_mode(
    mode: str | None,
    *,
    max_frames: int | None = None,
) -> str:
    """Return ``full``, ``preview``, or ``""`` when the mode is unknown."""
    key = str(mode or "").strip().lower()
    if key == MODE_FULL:
        return MODE_FULL
    if key == MODE_PREVIEW or max_frames is not None:
        return MODE_PREVIEW
    return ""


def resolve_compare_analysis_mode(
    *,
    session_a_mode: str | None = None,
    requested_mode: str | None = None,
    max_frames: int | None = None,
) -> str:
    """Mode for Session B. Never silently fall back to a preview cap."""
    requested = normalize_analysis_mode(requested_mode, max_frames=max_frames)
    if requested:
        return requested
    inherited = normalize_analysis_mode(session_a_mode)
    if inherited:
        return inherited
    return MODE_FULL


def analysis_modes_comparable(
    mode_a: str | None,
    mode_b: str | None,
    *,
    max_frames_a: int | None = None,
    max_frames_b: int | None = None,
) -> tuple[bool, str]:
    """True only when A and B are the same explicit full or preview mode."""
    a = normalize_analysis_mode(mode_a, max_frames=max_frames_a)
    b = normalize_analysis_mode(mode_b, max_frames=max_frames_b)
    if not a and not b:
        return True, ""
    if not a or not b:
        return (
            False,
            "Compare requires an explicit analysis mode on both Session A and "
            "Session B (full analysis or labeled preview). One session is unlabeled, "
            "so a mixed full-versus-preview comparison was not started.",
        )
    if a != b:
        label_a = format_analysis_mode_label(a, max_frames_a)
        label_b = format_analysis_mode_label(b, max_frames_b)
        return (
            False,
            f"Session A is {label_a} and Session B is {label_b}. "
            "Compare requires matching analysis modes; full-versus-preview "
            "comparisons are not run.",
        )
    return True, ""


def format_analysis_status(mode: str | None, *, playing: bool, paused: bool) -> str:
    """Status-bar analysis field: mode first, then playback."""
    key = str(mode or "").lower()
    if key == MODE_PREVIEW:
        prefix = PREVIEW_LABEL
    elif key == MODE_FULL:
        prefix = FULL_LABEL
    else:
        prefix = "Playback"
    if playing:
        return f"{prefix} · Playing"
    if paused:
        return f"{prefix} · Paused"
    return prefix
