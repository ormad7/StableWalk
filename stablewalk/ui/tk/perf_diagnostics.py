"""Performance Diagnostics — rolling timing, system counters, UI budgets.

Records decode / render / chart / compare / OpenSim durations and system
snapshots (RSS, threads, after-callbacks, Matplotlib figures). Does not
change analysis formulas.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Soft budgets (milliseconds) — diagnostics warn; callers should defer work.
UI_CALLBACK_BUDGET_MS = 16.0
CHART_REFRESH_BUDGET_MS = 50.0

_PERF_SAMPLE_CAP = 240
_PERF_LOG_INTERVAL_S = 2.0

# Timing sections shown in the diagnostics panel.
# StableWalk is Tkinter + Matplotlib Agg (not Qt / native OpenGL).
# Aliases map common GPU/Qt vocabulary onto the real paint paths.
PERF_SECTIONS: tuple[str, ...] = (
    "decode",
    "video",
    "skeleton",
    "trajectory",
    "3d",
    "mpl_draw",
    "tk_paint",
    "chart",
    "compare_prep",
    "opensim",
    "tab_switch",
    "tab_activate",
    "startup",
    "total",
)

_SECTION_ALIASES = {
    "graph": "chart",
    "frame_decode": "decode",
    "matplotlib": "mpl_draw",
    "compare": "compare_prep",
    "tab": "tab_switch",
    "trajectory_tip": "trajectory",
    "path": "trajectory",
    # User-facing / cross-toolkit vocabulary → actual backends.
    "qt_paint": "tk_paint",
    "qt": "tk_paint",
    "opengl": "3d",
    "gpu": "3d",
}


@dataclass
class PerfSectionStats:
    """Rolling statistics for one timed section."""

    samples: list[float] = field(default_factory=list)
    cap: int = _PERF_SAMPLE_CAP

    def add(self, ms: float) -> None:
        self.samples.append(float(ms))
        if len(self.samples) > self.cap:
            del self.samples[: len(self.samples) - self.cap]

    def clear(self) -> None:
        self.samples.clear()

    @property
    def count(self) -> int:
        return len(self.samples)

    def average(self) -> float | None:
        if not self.samples:
            return None
        return sum(self.samples) / len(self.samples)

    def maximum(self) -> float | None:
        if not self.samples:
            return None
        return max(self.samples)

    def p95(self) -> float | None:
        return percentile(self.samples, 95.0)

    def format_line(self, name: str) -> str:
        if not self.samples:
            return f"{name}: —"
        avg = self.average() or 0.0
        mx = self.maximum() or 0.0
        p95 = self.p95() or 0.0
        return (
            f"{name}: avg={avg:.1f} ms  max={mx:.1f} ms  "
            f"p95={p95:.1f} ms  n={len(self.samples)}"
        )


def percentile(samples: list[float], pct: float) -> float | None:
    """Inclusive nearest-rank percentile (pct in 0..100)."""
    if not samples:
        return None
    ordered = sorted(float(x) for x in samples)
    if len(ordered) == 1:
        return ordered[0]
    pct = max(0.0, min(100.0, float(pct)))
    # Nearest-rank: index = ceil(p/100 * n) - 1
    rank = max(1, int((pct / 100.0) * len(ordered) + 0.999999))
    return ordered[min(len(ordered), rank) - 1]


@dataclass
class SystemSnapshot:
    rss_mb: float | None
    thread_count: int
    after_callback_count: int
    figure_count: int
    playback_timer_count: int
    analysis_worker_count: int
    timestamp: float = field(default_factory=time.monotonic)

    def format_lines(self) -> list[str]:
        rss = f"{self.rss_mb:.1f} MiB" if self.rss_mb is not None else "—"
        return [
            f"memory (RSS):           {rss}",
            f"active threads:         {self.thread_count}",
            f"active after-callbacks: {self.after_callback_count}",
            f"Matplotlib figures:     {self.figure_count}",
            f"playback timers:        {self.playback_timer_count}",
            f"analysis workers:       {self.analysis_worker_count}",
        ]


@dataclass
class PerfDiagnosticsState:
    """Mutable diagnostics state stored on the GUI object."""

    sections: dict[str, PerfSectionStats] = field(default_factory=dict)
    system_history: list[SystemSnapshot] = field(default_factory=list)
    budget_warnings: list[str] = field(default_factory=list)
    enabled: bool = False
    last_log_ts: float = 0.0

    def ensure_sections(self) -> None:
        for key in PERF_SECTIONS:
            if key not in self.sections:
                self.sections[key] = PerfSectionStats()

    def record(self, section: str, ms: float) -> None:
        self.ensure_sections()
        key = _SECTION_ALIASES.get(section, section)
        bucket = self.sections.setdefault(key, PerfSectionStats())
        bucket.add(ms)

    def record_budget_warning(self, message: str) -> None:
        self.budget_warnings.append(message)
        if len(self.budget_warnings) > 80:
            del self.budget_warnings[: len(self.budget_warnings) - 80]

    def clear(self) -> None:
        for bucket in self.sections.values():
            bucket.clear()
        self.system_history.clear()
        self.budget_warnings.clear()
        self.last_log_ts = 0.0


def perf_diagnostics_enabled(gui: Any | None = None) -> bool:
    if gui is not None and getattr(gui, "_perf_diagnostics", False):
        return True
    if gui is not None and getattr(gui, "_perf_log", False):
        return True
    env = os.environ.get("STABLEWALK_PERF_DIAG", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    return os.environ.get("STABLEWALK_PERF_LOG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def get_perf_state(gui: Any) -> PerfDiagnosticsState:
    state = getattr(gui, "_perf_diag_state", None)
    if not isinstance(state, PerfDiagnosticsState):
        state = PerfDiagnosticsState()
        # Migrate legacy _perf_timings lists if present.
        legacy = getattr(gui, "_perf_timings", None)
        if isinstance(legacy, dict):
            state.ensure_sections()
            for key, samples in legacy.items():
                canon = _SECTION_ALIASES.get(key, key)
                bucket = state.sections.setdefault(canon, PerfSectionStats())
                for ms in list(samples or []):
                    bucket.add(float(ms))
        try:
            gui._perf_diag_state = state
        except Exception:
            pass
    state.ensure_sections()
    return state


def set_perf_diagnostics(gui: Any, enabled: bool) -> None:
    gui._perf_diagnostics = bool(enabled)
    gui._perf_log = bool(enabled)  # reuse periodic logger
    state = get_perf_state(gui)
    state.enabled = bool(enabled)
    if enabled:
        state.clear()


@contextmanager
def timed_section(
    gui: Any,
    section: str,
    *,
    budget_ms: float | None = None,
    warn_label: str | None = None,
) -> Iterator[None]:
    """Time a section; optionally warn when ``budget_ms`` is exceeded."""
    if not perf_diagnostics_enabled(gui) and budget_ms is None:
        # Still allow timing when budgets are requested for enforcement paths.
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # Keep legacy store warm for maybe_log_perf_timings.
            store = getattr(gui, "_perf_timings", None)
            if store is None:
                store = {}
                try:
                    gui._perf_timings = store
                except Exception:
                    pass
            key = _SECTION_ALIASES.get(section, section)
            bucket = store.setdefault(key, [])
            bucket.append(elapsed_ms)
            if len(bucket) > _PERF_SAMPLE_CAP:
                del bucket[: len(bucket) - _PERF_SAMPLE_CAP]
        return

    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        state = get_perf_state(gui)
        state.record(section, elapsed_ms)
        # Mirror into legacy list for existing maybe_log_perf_timings.
        store = getattr(gui, "_perf_timings", None)
        if store is None:
            store = {}
            try:
                gui._perf_timings = store
            except Exception:
                pass
        key = _SECTION_ALIASES.get(section, section)
        bucket = store.setdefault(key, [])
        bucket.append(elapsed_ms)
        if len(bucket) > _PERF_SAMPLE_CAP:
            del bucket[: len(bucket) - _PERF_SAMPLE_CAP]
        if budget_ms is not None and elapsed_ms > budget_ms:
            label = warn_label or section
            msg = (
                f"[Perf budget] {label}: {elapsed_ms:.1f} ms "
                f"> {budget_ms:.0f} ms budget"
            )
            state.record_budget_warning(msg)
            if perf_diagnostics_enabled(gui):
                print(msg, flush=True)
                logger.warning(msg)


def count_playback_timers(gui: Any) -> int:
    """How many playback tick chains are currently scheduled (should be ≤1)."""
    count = 0
    if getattr(gui, "_after_id", None) is not None:
        count += 1
    compare = getattr(gui, "_comparison_mode", None)
    if compare is not None and getattr(compare, "_play_job", None) is not None:
        count += 1
    return count


def count_analysis_workers(gui: Any) -> int:
    """Best-effort count of in-flight analysis workers (should be ≤1)."""
    count = 0
    if getattr(gui, "_pipeline_callback", None) is not None:
        count += 1
    if getattr(gui, "_analysis_compute_active", False):
        count += 1
    worker = getattr(gui, "_analysis_worker", None)
    if worker is not None and getattr(worker, "is_alive", lambda: False)():
        # Already counted via _analysis_compute_active when set correctly.
        if not getattr(gui, "_analysis_compute_active", False):
            count += 1
    if getattr(gui, "_session_save_in_progress", False):
        count += 1
    opensim = getattr(gui, "_opensim_worker", None)
    if opensim is not None and getattr(opensim, "is_alive", lambda: False)():
        count += 1
    compare = getattr(gui, "_comparison_mode", None)
    if compare is not None:
        if getattr(compare, "_compare_worker", None) is not None:
            cw = compare._compare_worker
            if cw is not None and getattr(cw, "is_alive", lambda: False)():
                count += 1
        elif getattr(compare, "_loading_keys", None):
            count += 1
    return count


def collect_system_snapshot(gui: Any) -> SystemSnapshot:
    from stablewalk.ui.tk.gui_memory import (
        approximate_rss_mb,
        collect_active_timer_ids,
        collect_owned_figures,
    )

    timers = collect_active_timer_ids(gui)
    figures = collect_owned_figures(gui)
    return SystemSnapshot(
        rss_mb=approximate_rss_mb(),
        thread_count=int(threading.active_count()),
        after_callback_count=len(timers),
        figure_count=len(figures),
        playback_timer_count=count_playback_timers(gui),
        analysis_worker_count=count_analysis_workers(gui),
    )


def record_system_snapshot(gui: Any) -> SystemSnapshot | None:
    if not perf_diagnostics_enabled(gui):
        return None
    snap = collect_system_snapshot(gui)
    state = get_perf_state(gui)
    state.system_history.append(snap)
    if len(state.system_history) > 60:
        del state.system_history[: len(state.system_history) - 60]
    # Soft integrity rules.
    if snap.playback_timer_count > 1:
        state.record_budget_warning(
            f"[Perf budget] multiple playback timers: {snap.playback_timer_count}"
        )
    if snap.analysis_worker_count > 1:
        state.record_budget_warning(
            f"[Perf budget] multiple analysis workers: {snap.analysis_worker_count}"
        )
    return snap


def format_perf_diagnostics_report(gui: Any) -> str:
    state = get_perf_state(gui)
    state.ensure_sections()
    lines = [
        "PERFORMANCE DIAGNOSTICS",
        "=======================",
        f"Mode: {'ON' if perf_diagnostics_enabled(gui) else 'OFF'}",
        "Backend: Tkinter + Matplotlib Agg (Qt/OpenGL labels map to Tk paint / Agg)",
        f"UI callback budget: {UI_CALLBACK_BUDGET_MS:.0f} ms (playback)",
        f"Chart refresh budget: {CHART_REFRESH_BUDGET_MS:.0f} ms",
        "Paint policy: skip widget redraw when paint signature unchanged",
        "",
        "Timing (rolling avg / max / p95)",
        "--------------------------------",
    ]
    for key in PERF_SECTIONS:
        bucket = state.sections.get(key) or PerfSectionStats()
        lines.append(bucket.format_line(key))
    lines.append("")
    lines.append("System")
    lines.append("------")
    snap = collect_system_snapshot(gui)
    lines.extend(snap.format_lines())
    if state.system_history:
        rss_vals = [s.rss_mb for s in state.system_history if s.rss_mb is not None]
        if rss_vals:
            lines.append(
                f"RSS trend (samples): min={min(rss_vals):.1f}  "
                f"max={max(rss_vals):.1f}  last={rss_vals[-1]:.1f} MiB"
            )
    if state.budget_warnings:
        lines.append("")
        lines.append("Recent budget warnings")
        lines.append("----------------------")
        for warn in state.budget_warnings[-12:]:
            lines.append(warn)
    return "\n".join(lines)


def maybe_log_perf_diagnostics(gui: Any) -> None:
    """Periodic console dump when Performance Diagnostics is enabled."""
    if not perf_diagnostics_enabled(gui):
        return
    state = get_perf_state(gui)
    now = time.monotonic()
    if now - float(state.last_log_ts or 0.0) < _PERF_LOG_INTERVAL_S:
        return
    state.last_log_ts = now
    record_system_snapshot(gui)
    report = format_perf_diagnostics_report(gui)
    print(report, flush=True)
    logger.debug("perf diagnostics logged")


__all__ = [
    "CHART_REFRESH_BUDGET_MS",
    "PERF_SECTIONS",
    "PerfDiagnosticsState",
    "PerfSectionStats",
    "SystemSnapshot",
    "UI_CALLBACK_BUDGET_MS",
    "collect_system_snapshot",
    "count_analysis_workers",
    "count_playback_timers",
    "format_perf_diagnostics_report",
    "get_perf_state",
    "maybe_log_perf_diagnostics",
    "percentile",
    "perf_diagnostics_enabled",
    "record_system_snapshot",
    "set_perf_diagnostics",
    "timed_section",
]
