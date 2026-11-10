"""
Compact Analyze progress indicator — stage, bar, estimated remaining time.
"""

from __future__ import annotations

import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Any

from stablewalk.ui.theme import (
    ACCENT,
    DANGER,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    PAD_SM,
    PAD_XS,
    PANEL,
    SUCCESS,
    TEXT,
)

# Ordered Analyze stages (user-facing labels).
ANALYZE_STAGES: tuple[tuple[str, str], ...] = (
    ("loading", "Loading video..."),
    ("extracting", "Extracting frames..."),
    ("pose", "Pose estimation..."),
    ("reconstruction", "3D reconstruction..."),
    ("joint_angles", "Joint angles..."),
    ("biomechanics", "Biomechanics..."),
    ("virtual_grf", "Virtual GRF..."),
    ("opensim", "OpenSim..."),
    ("report", "Generating report..."),
    ("completed", "Completed."),
)

_STAGE_INDEX = {key: i for i, (key, _) in enumerate(ANALYZE_STAGES)}


@dataclass
class AnalyzeProgressSnapshot:
    """Live metrics for the Analyze progress panel."""

    stage_key: str = "loading"
    message: str = ""
    fraction: float = 0.0
    frames_done: int | None = None
    frames_total: int | None = None
    fps: float | None = None
    elapsed_s: float | None = None
    remaining_s: float | None = None


def stage_label(stage_key: str) -> str:
    for key, label in ANALYZE_STAGES:
        if key == stage_key:
            return label
    return stage_key.replace("_", " ").title() + "..."


def infer_stage_from_message(message: str, fraction: float) -> str:
    """Map pipeline / UI messages onto Analyze stage keys."""
    low = (message or "").lower()
    if fraction >= 0.995 or "completed" in low or low.strip() in ("done.", "done"):
        return "completed"
    if "report" in low or "summary" in low:
        return "report"
    if "opensim" in low:
        return "opensim"
    if "grf" in low or "ground reaction" in low:
        return "virtual_grf"
    if "biomech" in low or "stability" in low or "gait cycle" in low:
        return "biomechanics"
    if "joint angle" in low or "enrich" in low:
        return "joint_angles"
    if "3d" in low or "skeleton" in low or "reconstruction" in low:
        return "reconstruction"
    if "pose" in low or "mediapipe" in low or "smpl" in low:
        return "pose"
    if "extract" in low or "frame" in low or "processing video" in low:
        return "extracting"
    if "load" in low or "validat" in low or fraction < 0.12:
        return "loading"
    if fraction < 0.25:
        return "extracting"
    if fraction < 0.55:
        return "pose"
    if fraction < 0.65:
        return "reconstruction"
    if fraction < 0.72:
        return "joint_angles"
    if fraction < 0.82:
        return "biomechanics"
    if fraction < 0.88:
        return "virtual_grf"
    if fraction < 0.94:
        return "opensim"
    if fraction < 0.995:
        return "report"
    return "completed"


def stage_fraction(stage_key: str, *, within: float = 0.5) -> float:
    """Overall 0–1 progress for a named stage (optionally mid-stage)."""
    idx = _STAGE_INDEX.get(stage_key, 0)
    n = max(len(ANALYZE_STAGES) - 1, 1)
    base = idx / n
    span = 1.0 / n
    return min(1.0, base + span * max(0.0, min(1.0, within)))


def format_elapsed(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s:02d}s"


def estimate_remaining_s(elapsed_s: float | None, fraction: float) -> float | None:
    """Estimate seconds remaining from elapsed time and overall fraction."""
    try:
        frac = float(fraction)
    except (TypeError, ValueError):
        return None
    if elapsed_s is None or elapsed_s < 0.0 or not (0.0 <= frac <= 1.0):
        return None
    if frac >= 0.995:
        return 0.0
    # Need a little progress before ETA is meaningful.
    if elapsed_s < 0.75 or frac < 0.04:
        return None
    return float(elapsed_s) * (1.0 - frac) / frac


def format_remaining(seconds: float | None) -> str:
    if seconds is None:
        return "Estimating…"
    if seconds <= 0.5:
        return "Almost done"
    return f"~{format_elapsed(seconds)} left"


def format_frames(done: int | None, total: int | None) -> str:
    if done is None and total is None:
        return "—"
    if total is None:
        return f"{done}" if done is not None else "—"
    if done is None:
        return f"— / {total}"
    return f"{done} / {total}"


def format_fps(fps: float | None) -> str:
    if fps is None or fps <= 0 or fps != fps:
        return "—"
    if fps >= 100:
        return f"{fps:.0f}"
    return f"{fps:.1f}"


def build_analyze_progress_panel(gui: Any, parent: tk.Misc) -> ttk.Frame:
    """
    Build the compact Analyze indicator (stage · bar · remaining).

    Hidden until ``reset_analyze_progress``; does not block the UI thread.
    """
    frame = ttk.Frame(parent)
    gui.analyze_progress_frame = frame

    row = ttk.Frame(frame)
    row.pack(fill=tk.X, pady=(PAD_XS, 0))
    row.columnconfigure(1, weight=1)

    gui.lbl_analyze_stage = tk.Label(
        row,
        text="Stage: —",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
        width=22,
    )
    gui.lbl_analyze_stage.grid(row=0, column=0, sticky="w", padx=(0, PAD_SM))

    # Prefer a dedicated bar so the toolbar stays clean when idle.
    gui.progress = ttk.Progressbar(
        row, mode="determinate", style="Accent.Horizontal.TProgressbar"
    )
    gui.progress.grid(row=0, column=1, sticky="ew")

    gui.lbl_analyze_remaining = tk.Label(
        row,
        text="Estimating…",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="e",
        width=14,
    )
    gui.lbl_analyze_remaining.grid(row=0, column=2, sticky="e", padx=(PAD_SM, 0))

    # Lightweight secondary readout (frames / FPS) — one line, no stage checklist.
    meta = ttk.Frame(frame)
    meta.pack(fill=tk.X)
    gui.lbl_analyze_elapsed = tk.Label(
        meta,
        text="Elapsed: —",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_analyze_elapsed.pack(side=tk.LEFT, padx=(0, 12))
    gui.lbl_analyze_frames = tk.Label(
        meta,
        text="Frames: —",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_analyze_frames.pack(side=tk.LEFT, padx=(0, 12))
    gui.lbl_analyze_fps = tk.Label(
        meta,
        text="FPS: —",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_analyze_fps.pack(side=tk.LEFT)

    # Legacy attr kept for any callers that inspected the checklist.
    gui._analyze_stage_labels = {}

    # Start hidden — only visible while analysis is running.
    return frame


def show_analyze_progress(gui: Any) -> None:
    frame = getattr(gui, "analyze_progress_frame", None)
    if frame is None:
        return
    try:
        if not frame.winfo_ismapped():
            frame.pack(fill=tk.X, pady=(PAD_XS, 0))
    except tk.TclError:
        pass


def hide_analyze_progress(gui: Any, *, delay_ms: int = 0) -> None:
    frame = getattr(gui, "analyze_progress_frame", None)
    if frame is None:
        return

    def _hide() -> None:
        gui._analyze_progress_active = False
        try:
            frame.pack_forget()
        except tk.TclError:
            pass
        _sync_unfinished_controls(gui)

    if delay_ms > 0:
        root = getattr(gui, "root", None)
        if root is not None:
            root.after(delay_ms, _hide)
            return
    _hide()


def analyze_progress_active(gui: Any) -> bool:
    """True while an Analyze run is still processing.

    A lingering \"Completed.\" overlay after the worker finishes is not
    treated as active analysis, so Results can show Complete without the
    status bar staying on Analyzing / Playing-as-analysis.
    """
    if not bool(getattr(gui, "_analyze_progress_active", False)):
        return False
    snap = getattr(gui, "_analyze_progress_snapshot", None)
    stage = str(getattr(snap, "stage_key", "") or "")
    return stage != "completed"


def _sync_unfinished_controls(gui: Any) -> None:
    """Disable export / Advanced actions that require a finished analysis."""
    update = getattr(gui, "_update_analysis_export_state", None)
    if callable(update):
        try:
            update()
        except Exception:
            pass
    sync_busy = getattr(gui, "_sync_analysis_busy_controls", None)
    if callable(sync_busy):
        try:
            sync_busy()
        except Exception:
            pass


def reset_analyze_progress(gui: Any) -> None:
    """Reset indicators and start elapsed / ETA timer."""
    gui._analyze_progress_active = True
    gui._analyze_progress_t0 = time.perf_counter()
    gui._analyze_progress_anim_tick = 0
    gui._analyze_progress_snapshot = AnalyzeProgressSnapshot(stage_key="loading")
    _cancel_anim(gui)
    show_analyze_progress(gui)
    update_analyze_progress(
        gui,
        AnalyzeProgressSnapshot(
            stage_key="loading",
            message=stage_label("loading"),
            fraction=0.0,
            elapsed_s=0.0,
        ),
    )
    _start_anim(gui)
    _sync_unfinished_controls(gui)


def _cancel_anim(gui: Any) -> None:
    aid = getattr(gui, "_analyze_progress_anim_after", None)
    if aid is None:
        return
    root = getattr(gui, "root", None)
    if root is not None:
        try:
            root.after_cancel(aid)
        except tk.TclError:
            pass
    gui._analyze_progress_anim_after = None


def _start_anim(gui: Any) -> None:
    _cancel_anim(gui)
    root = getattr(gui, "root", None)
    if root is None:
        return

    def _tick() -> None:
        snap = getattr(gui, "_analyze_progress_snapshot", None)
        if snap is None or snap.stage_key == "completed":
            gui._analyze_progress_anim_after = None
            return
        tick = int(getattr(gui, "_analyze_progress_anim_tick", 0)) + 1
        gui._analyze_progress_anim_tick = tick
        dots = "." * (1 + (tick % 3))
        lbl = getattr(gui, "lbl_analyze_stage", None)
        base = stage_label(snap.stage_key).rstrip(".")
        if lbl is not None:
            try:
                lbl.configure(text=f"{base}{dots}", fg=ACCENT)
            except tk.TclError:
                pass
        t0 = getattr(gui, "_analyze_progress_t0", None)
        if t0 is not None:
            elapsed = time.perf_counter() - t0
            snap.elapsed_s = elapsed
            elbl = getattr(gui, "lbl_analyze_elapsed", None)
            if elbl is not None:
                try:
                    elbl.configure(text=f"Elapsed: {format_elapsed(elapsed)}")
                except tk.TclError:
                    pass
            remaining = estimate_remaining_s(elapsed, snap.fraction)
            snap.remaining_s = remaining
            rlbl = getattr(gui, "lbl_analyze_remaining", None)
            if rlbl is not None:
                try:
                    rlbl.configure(text=format_remaining(remaining), fg=MUTED)
                except tk.TclError:
                    pass
        gui._analyze_progress_anim_after = root.after(400, _tick)

    gui._analyze_progress_anim_after = root.after(400, _tick)


def update_analyze_progress(gui: Any, snap: AnalyzeProgressSnapshot) -> None:
    """Apply a progress snapshot to the compact indicator widgets."""
    if not getattr(gui, "_analyze_progress_active", False):
        return
    t0 = getattr(gui, "_analyze_progress_t0", None)
    if snap.elapsed_s is None and t0 is not None:
        snap.elapsed_s = time.perf_counter() - t0
    if snap.remaining_s is None:
        snap.remaining_s = estimate_remaining_s(snap.elapsed_s, snap.fraction)
    gui._analyze_progress_snapshot = snap

    progress = getattr(gui, "progress", None)
    if progress is not None:
        try:
            progress["value"] = max(0.0, min(100.0, snap.fraction * 100.0))
        except tk.TclError:
            pass

    stage_lbl = getattr(gui, "lbl_analyze_stage", None)
    if stage_lbl is not None:
        msg = (snap.message or stage_label(snap.stage_key)).strip().rstrip(".")
        try:
            color = SUCCESS if snap.stage_key == "completed" else ACCENT
            stage_lbl.configure(text=msg, fg=color)
        except tk.TclError:
            pass

    rlbl = getattr(gui, "lbl_analyze_remaining", None)
    if rlbl is not None:
        try:
            if snap.stage_key == "completed":
                rlbl.configure(text="Done", fg=SUCCESS)
            else:
                rlbl.configure(text=format_remaining(snap.remaining_s), fg=MUTED)
        except tk.TclError:
            pass

    elbl = getattr(gui, "lbl_analyze_elapsed", None)
    if elbl is not None:
        try:
            elbl.configure(text=f"Elapsed: {format_elapsed(snap.elapsed_s)}")
        except tk.TclError:
            pass

    flbl = getattr(gui, "lbl_analyze_frames", None)
    if flbl is not None:
        try:
            flbl.configure(
                text=f"Frames: {format_frames(snap.frames_done, snap.frames_total)}"
            )
        except tk.TclError:
            pass

    fps_lbl = getattr(gui, "lbl_analyze_fps", None)
    if fps_lbl is not None:
        try:
            fps_lbl.configure(text=f"FPS: {format_fps(snap.fps)}")
        except tk.TclError:
            pass

    if snap.stage_key == "completed":
        _cancel_anim(gui)


def complete_analyze_progress(
    gui: Any,
    *,
    frames_done: int | None = None,
    frames_total: int | None = None,
    fps: float | None = None,
    hide_after_ms: int = 2000,
) -> None:
    """Mark Analyze as completed and optionally hide the panel."""
    if not getattr(gui, "_analyze_progress_active", False):
        return
    t0 = getattr(gui, "_analyze_progress_t0", None)
    elapsed = (time.perf_counter() - t0) if t0 is not None else None
    update_analyze_progress(
        gui,
        AnalyzeProgressSnapshot(
            stage_key="completed",
            message="Completed.",
            fraction=1.0,
            frames_done=frames_done,
            frames_total=frames_total,
            fps=fps,
            elapsed_s=elapsed,
            remaining_s=0.0,
        ),
    )
    _sync_unfinished_controls(gui)
    hide_analyze_progress(gui, delay_ms=hide_after_ms)


def fail_analyze_progress(gui: Any, message: str = "Failed") -> None:
    if not getattr(gui, "_analyze_progress_active", False):
        return
    _cancel_anim(gui)
    stage_lbl = getattr(gui, "lbl_analyze_stage", None)
    if stage_lbl is not None:
        try:
            stage_lbl.configure(text=message, fg=DANGER)
        except tk.TclError:
            pass
    rlbl = getattr(gui, "lbl_analyze_remaining", None)
    if rlbl is not None:
        try:
            rlbl.configure(text="—", fg=DANGER)
        except tk.TclError:
            pass
    hide_analyze_progress(gui, delay_ms=2500)


__all__ = [
    "ANALYZE_STAGES",
    "AnalyzeProgressSnapshot",
    "analyze_progress_active",
    "build_analyze_progress_panel",
    "complete_analyze_progress",
    "estimate_remaining_s",
    "fail_analyze_progress",
    "format_elapsed",
    "format_fps",
    "format_frames",
    "format_remaining",
    "hide_analyze_progress",
    "infer_stage_from_message",
    "reset_analyze_progress",
    "show_analyze_progress",
    "stage_fraction",
    "stage_label",
    "update_analyze_progress",
]
