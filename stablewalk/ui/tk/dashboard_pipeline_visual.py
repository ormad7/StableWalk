"""Shared Real-to-Sim pipeline visualization helpers (metrics + completion animation)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tkinter as tk

from stablewalk.monitoring.pipeline_status import (
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    STATUS_UNAVAILABLE,
    PipelineStageState,
)
from stablewalk.ui.theme import BORDER, ELEVATED, MUTED, ORANGE, PANEL, SUCCESS

# Consistent status palette for flow, engineering cards, status panel, and dialogs.
PIPELINE_STATUS_FG = {
    STATUS_COMPLETED: SUCCESS,
    STATUS_PARTIAL: ORANGE,
    STATUS_UNAVAILABLE: MUTED,
}

PIPELINE_STATUS_STRIP = {
    STATUS_COMPLETED: SUCCESS,
    STATUS_PARTIAL: ORANGE,
    STATUS_UNAVAILABLE: BORDER,
}

PIPELINE_STATUS_RING = PIPELINE_STATUS_STRIP

_COMPLETION_PULSE = ("#2ecc71", "#42be65", "#5ce1a8", SUCCESS)
_CONNECTOR_PULSE = ("#3d5a4a", "#4a7a5c", "#5ea874", SUCCESS)
_ANIM_MS = 48
_CONNECTOR_ANIM_MS = 56


@dataclass(frozen=True)
class PipelineStageMetrics:
    """Compact metrics shown on pipeline stage cards."""

    duration_text: str
    confidence_pct: int | None
    confidence_text: str
    files_text: str
    files_count: int


def parse_confidence_percent(confidence: str) -> int | None:
    """Extract a 0–100 confidence value from a human-readable confidence string."""
    if not confidence or confidence.strip() in ("—", "-"):
        return None
    match = re.search(r"(\d{1,3})\s*%", confidence)
    if match:
        return max(0, min(100, int(match.group(1))))
    ratio = re.search(r"(\d+)\s*/\s*(\d+)", confidence)
    if ratio:
        num, den = int(ratio.group(1)), int(ratio.group(2))
        if den > 0:
            return max(0, min(100, int(round(100.0 * num / den))))
    return None


def format_duration_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    if seconds < 0.05:
        return "<0.1s"
    if seconds < 10.0:
        return f"{seconds:.1f}s"
    if seconds < 120.0:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}m {rem:.0f}s"


def compact_generated_files(files_block: str, *, max_names: int = 2) -> tuple[str, int]:
    if not files_block or files_block.startswith("No matching"):
        return "No files yet", 0
    paths = [
        line.strip().lstrip("•").strip()
        for line in files_block.splitlines()
        if line.strip()
    ]
    if not paths:
        return "No files yet", 0
    names = [Path(p).name for p in paths]
    if len(names) <= max_names:
        return ", ".join(names), len(names)
    extra = len(names) - max_names
    return f"{', '.join(names[:max_names])} +{extra}", len(names)


def metrics_from_dialog_content(content: Any) -> PipelineStageMetrics:
    """Build card metrics from :class:`PipelineStageDialogContent`."""
    confidence_text = getattr(content, "confidence", "—") or "—"
    pct = parse_confidence_percent(confidence_text)
    files_short, files_count = compact_generated_files(
        getattr(content, "generated_files", "") or ""
    )
    duration_text = getattr(content, "duration_text", "—") or "—"
    return PipelineStageMetrics(
        duration_text=duration_text,
        confidence_pct=pct,
        confidence_text=confidence_text,
        files_text=files_short,
        files_count=files_count,
    )


def metrics_summary_line(metrics: PipelineStageMetrics) -> str:
    parts: list[str] = []
    if metrics.duration_text and metrics.duration_text != "—":
        parts.append(metrics.duration_text)
    if metrics.confidence_pct is not None:
        parts.append(f"{metrics.confidence_pct}% conf")
    elif metrics.confidence_text and metrics.confidence_text != "—":
        short = metrics.confidence_text.split(":")[-1].strip()
        if short and len(short) <= 28:
            parts.append(short)
    return " · ".join(parts) if parts else "—"


def _cancel_stage_animation(gui: Any, stage_key: str) -> None:
    after_ids: dict[str, str] = getattr(gui, "_pipeline_anim_after_ids", {})
    aid = after_ids.pop(stage_key, None)
    if aid is not None:
        root = getattr(gui, "root", None)
        if root is not None:
            try:
                root.after_cancel(aid)
            except tk.TclError:
                pass
    gui._pipeline_anim_after_ids = after_ids


def _set_progress_fill(fill: tk.Misc | None, fraction: float, *, color: str) -> None:
    if fill is None:
        return
    fraction = max(0.0, min(1.0, fraction))
    try:
        fill.place(relx=0.0, rely=0.0, relwidth=fraction, relheight=1.0)
        fill.configure(bg=color)
    except tk.TclError:
        pass


def animate_stage_completion(
    gui: Any,
    stage_key: str,
    widgets: dict[str, tk.Misc],
    *,
    status: PipelineStageState,
) -> None:
    """Green progress sweep + ring pulse when a stage newly completes."""
    if status != STATUS_COMPLETED:
        track = widgets.get("progress_track")
        fill = widgets.get("progress_fill")
        ring = widgets.get("ring")
        if fill is not None and track is not None:
            _set_progress_fill(fill, 0.0, color=BORDER)
        if ring is not None:
            try:
                ring.configure(bg=PIPELINE_STATUS_RING.get(status, BORDER))
            except tk.TclError:
                pass
        return

    root = getattr(gui, "root", None)
    if root is None:
        return

    _cancel_stage_animation(gui, stage_key)
    fill = widgets.get("progress_fill")
    ring = widgets.get("ring")
    if fill is None and ring is None:
        return

    after_ids: dict[str, str] = getattr(gui, "_pipeline_anim_after_ids", {})
    steps = max(len(_COMPLETION_PULSE), 8)

    def _tick(step: int = 0) -> None:
        if step >= steps:
            _set_progress_fill(fill, 1.0, color=SUCCESS)
            if ring is not None:
                try:
                    ring.configure(bg=SUCCESS)
                except tk.TclError:
                    pass
            after_ids.pop(stage_key, None)
            gui._pipeline_anim_after_ids = after_ids
            return

        frac = (step + 1) / steps
        pulse_idx = min(step, len(_COMPLETION_PULSE) - 1)
        color = _COMPLETION_PULSE[pulse_idx]
        _set_progress_fill(fill, frac, color=color)
        if ring is not None:
            try:
                ring.configure(bg=color)
            except tk.TclError:
                pass
        after_ids[stage_key] = root.after(_ANIM_MS, lambda s=step + 1: _tick(s))
        gui._pipeline_anim_after_ids = after_ids

    _tick(0)


def install_progress_track(
    parent: tk.Misc,
    *,
    row: int | None = None,
    column: int = 0,
    columnspan: int = 1,
) -> tuple[tk.Frame, tk.Frame]:
    """Thin progress track with a fill child for completion animation.

    Use ``row=`` when the parent already manages children with ``grid``;
    otherwise the track is packed along the bottom (pack-managed parents).
    """
    track = tk.Frame(parent, bg=BORDER, height=4, highlightthickness=0)
    if row is not None:
        track.grid(row=row, column=column, columnspan=columnspan, sticky="ew")
        track.grid_propagate(False)
    else:
        track.pack(fill=tk.X, side=tk.BOTTOM)
        track.pack_propagate(False)
    fill = tk.Frame(track, bg=BORDER, highlightthickness=0)
    fill.place(relx=0.0, rely=0.0, relwidth=0.0, relheight=1.0)
    return track, fill


def set_progress_fill(fill: tk.Misc | None, fraction: float, *, color: str) -> None:
    """Public alias for progress bar updates."""
    _set_progress_fill(fill, fraction, color=color)


def set_connector_state(
    widget: tk.Misc | None,
    *,
    active: bool,
    dormant_fg: str = MUTED,
    active_fg: str = SUCCESS,
) -> None:
    """Colour a static connector arrow/rail based on upstream completion."""
    if widget is None:
        return
    try:
        widget.configure(fg=active_fg if active else dormant_fg)
    except tk.TclError:
        try:
            widget.configure(bg=active_fg if active else dormant_fg)
        except tk.TclError:
            pass


def animate_connector_activation(
    gui: Any,
    connector_key: str,
    widget: tk.Misc | None,
    *,
    active: bool,
) -> None:
    """Pulse a connector when the upstream stage newly completes."""
    if widget is None:
        return
    if not active:
        set_connector_state(widget, active=False)
        return

    root = getattr(gui, "root", None)
    if root is None:
        set_connector_state(widget, active=True)
        return

    after_ids: dict[str, str] = getattr(gui, "_pipeline_connector_after_ids", {})
    prev = after_ids.pop(connector_key, None)
    if prev is not None:
        try:
            root.after_cancel(prev)
        except tk.TclError:
            pass

    steps = len(_CONNECTOR_PULSE)

    def _tick(step: int = 0) -> None:
        if step >= steps:
            set_connector_state(widget, active=True)
            after_ids.pop(connector_key, None)
            gui._pipeline_connector_after_ids = after_ids
            return
        color = _CONNECTOR_PULSE[min(step, steps - 1)]
        try:
            widget.configure(fg=color)
        except tk.TclError:
            try:
                widget.configure(bg=color)
            except tk.TclError:
                pass
        after_ids[connector_key] = root.after(
            _CONNECTOR_ANIM_MS, lambda s=step + 1: _tick(s)
        )
        gui._pipeline_connector_after_ids = after_ids

    _tick(0)


__all__ = [
    "PIPELINE_STATUS_FG",
    "PIPELINE_STATUS_RING",
    "PIPELINE_STATUS_STRIP",
    "PipelineStageMetrics",
    "animate_connector_activation",
    "animate_stage_completion",
    "compact_generated_files",
    "format_duration_seconds",
    "install_progress_track",
    "metrics_from_dialog_content",
    "metrics_summary_line",
    "parse_confidence_percent",
    "set_connector_state",
    "set_progress_fill",
]
