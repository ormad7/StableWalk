"""Optional UI freeze diagnostics for Compare / Tk responsiveness.

Enable with environment variable::

    STABLEWALK_UI_FREEZE_DEBUG=1

Writes:
  - logs/ui_freeze_debug.log   — operations, workers, after(), figures, cv2, RSS
  - logs/ui_freeze_threads.log — faulthandler dumps when UI stalls ≥ 5s

High-frequency wraps are opt-in only; keep them off in normal releases.
"""

from __future__ import annotations

import atexit
import faulthandler
import functools
import logging
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

_ENABLED: bool | None = None
_UI_THREAD_ID: int | None = None
_LOCK = threading.Lock()
_AFTER_LIVE: dict[int, str] = {}
_WATCHDOG: threading.Event | None = None
_WATCHDOG_THREAD: threading.Thread | None = None
_LAST_HEARTBEAT = 0.0
_LAST_OP = "idle"
_SEQ = 0

STALL_S = 5.0


def freeze_debug_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        flag = os.environ.get("STABLEWALK_UI_FREEZE_DEBUG", "").strip().lower()
        _ENABLED = flag in ("1", "true", "yes", "on")
    return bool(_ENABLED)


def _project_logs_dir() -> Path:
    try:
        from stablewalk.config import PROJECT_ROOT

        base = Path(PROJECT_ROOT)
    except Exception:
        base = Path(__file__).resolve().parents[3]
    path = base / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def debug_log_path() -> Path:
    return _project_logs_dir() / "ui_freeze_debug.log"


def threads_log_path() -> Path:
    return _project_logs_dir() / "ui_freeze_threads.log"


def _rss_mb() -> float | None:
    try:
        from stablewalk.ui.tk.gui_memory import approximate_rss_mb

        return approximate_rss_mb()
    except Exception:
        return None


def _write(path: Path, line: str) -> None:
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            if not line.endswith("\n"):
                fh.write("\n")
    except Exception:
        logger.debug("ui_freeze_debug write failed", exc_info=True)


def log_debug(event: str, **fields: Any) -> None:
    if not freeze_debug_enabled():
        return
    global _SEQ
    with _LOCK:
        _SEQ += 1
        seq = _SEQ
    parts = [
        f"ts={time.time():.3f}",
        f"seq={seq}",
        f"event={event}",
        f"tid={threading.get_ident()}",
        f"ui_tid={_UI_THREAD_ID}",
        f"threads={threading.active_count()}",
    ]
    rss = _rss_mb()
    if rss is not None:
        parts.append(f"rss_mb={rss:.1f}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    _write(debug_log_path(), " | ".join(str(p) for p in parts))


def mark_ui_thread(root: Any | None = None) -> None:
    """Record the Tk UI thread id and start the 5s stall watchdog."""
    global _UI_THREAD_ID, _LAST_HEARTBEAT
    if not freeze_debug_enabled():
        return
    _UI_THREAD_ID = threading.get_ident()
    _LAST_HEARTBEAT = time.monotonic()
    log_debug("ui_thread", note="marked")
    _ensure_watchdog(root)
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:
        pass


def heartbeat(op: str | None = None) -> None:
    global _LAST_HEARTBEAT, _LAST_OP
    if not freeze_debug_enabled():
        return
    _LAST_HEARTBEAT = time.monotonic()
    if op:
        _LAST_OP = op


def _ensure_watchdog(root: Any | None) -> None:
    global _WATCHDOG, _WATCHDOG_THREAD
    if _WATCHDOG is not None:
        return
    stop = threading.Event()
    _WATCHDOG = stop

    def _loop() -> None:
        while not stop.wait(1.0):
            idle = time.monotonic() - _LAST_HEARTBEAT
            if idle < STALL_S:
                continue
            path = threads_log_path()
            try:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"\n=== UI stall {idle:.1f}s last_op={_LAST_OP!r} "
                        f"ts={time.time():.3f} ===\n"
                    )
                    faulthandler.dump_traceback(file=fh, all_threads=True)
            except Exception:
                logger.exception("faulthandler dump failed")
            log_debug("ui_stall_dump", idle_s=f"{idle:.1f}", last_op=_LAST_OP)
            # Avoid dump spam while still wedged.
            time.sleep(STALL_S)

    t = threading.Thread(target=_loop, name="ui-freeze-watchdog", daemon=True)
    _WATCHDOG_THREAD = t
    t.start()
    atexit.register(lambda: stop.set())
    if root is not None:
        _arm_ui_heartbeat(root)


def _arm_ui_heartbeat(root: Any) -> None:
    """Pulse heartbeat from the UI thread so stalls are detectable."""

    def _pulse() -> None:
        if not freeze_debug_enabled() or _WATCHDOG is None or _WATCHDOG.is_set():
            return
        heartbeat("ui_idle_pulse")
        try:
            root.after(500, _pulse)
        except Exception:
            pass

    try:
        root.after(500, _pulse)
    except Exception:
        pass


@contextmanager
def timed_op(name: str, **fields: Any) -> Iterator[None]:
    """Log before/after an operation (the primary freeze-localization tool)."""
    if not freeze_debug_enabled():
        yield
        return
    heartbeat(f"begin:{name}")
    log_debug("op_begin", op=name, **fields)
    t0 = time.perf_counter()
    try:
        yield
    except Exception as exc:
        dt = (time.perf_counter() - t0) * 1000.0
        log_debug(
            "op_error",
            op=name,
            ms=f"{dt:.1f}",
            error=type(exc).__name__,
            detail=str(exc)[:200],
        )
        raise
    else:
        dt = (time.perf_counter() - t0) * 1000.0
        heartbeat(f"end:{name}")
        log_debug("op_end", op=name, ms=f"{dt:.1f}", **fields)


def log_compare_enter_exit(phase: str, **fields: Any) -> None:
    log_debug(f"compare_{phase}", **fields)


def log_worker(phase: str, **fields: Any) -> None:
    log_debug(f"worker_{phase}", **fields)


def log_figure(phase: str, **fields: Any) -> None:
    log_debug(f"mpl_{phase}", **fields)


def log_capture(phase: str, **fields: Any) -> None:
    log_debug(f"cv2_{phase}", **fields)


def wrap_root_after(root: Any) -> None:
    """Monkey-patch ``root.after`` / ``after_cancel`` for create/cancel logging."""
    if not freeze_debug_enabled():
        return
    if getattr(root, "_ui_freeze_after_wrapped", False):
        return

    orig_after = root.after
    orig_cancel = root.after_cancel

    @functools.wraps(orig_after)
    def _after(ms, func=None, *args):  # type: ignore[no-untyped-def]
        if func is None:
            return orig_after(ms)

        def _wrapped(*a, **k):  # type: ignore[no-untyped-def]
            with _LOCK:
                _AFTER_LIVE.pop(id(func), None)
            heartbeat(f"after:{getattr(func, '__name__', func)}")
            log_debug("after_run", cb=getattr(func, "__name__", repr(func)))
            return func(*a, **k)

        job = orig_after(ms, _wrapped, *args)
        with _LOCK:
            _AFTER_LIVE[id(func)] = getattr(func, "__name__", repr(func))
        log_debug(
            "after_create",
            job=job,
            ms=ms,
            cb=getattr(func, "__name__", repr(func)),
            live=len(_AFTER_LIVE),
        )
        return job

    @functools.wraps(orig_cancel)
    def _cancel(job):  # type: ignore[no-untyped-def]
        log_debug("after_cancel", job=job, live=len(_AFTER_LIVE))
        return orig_cancel(job)

    root.after = _after  # type: ignore[method-assign]
    root.after_cancel = _cancel  # type: ignore[method-assign]
    root._ui_freeze_after_wrapped = True
    log_debug("after_wrapped")


def install_on_gui(gui: Any) -> None:
    """Call once from app startup when debug is enabled."""
    if not freeze_debug_enabled():
        return
    root = getattr(gui, "root", None)
    mark_ui_thread(root)
    if root is not None:
        wrap_root_after(root)
    log_debug("install_on_gui")


__all__ = [
    "STALL_S",
    "debug_log_path",
    "freeze_debug_enabled",
    "heartbeat",
    "install_on_gui",
    "log_capture",
    "log_compare_enter_exit",
    "log_debug",
    "log_figure",
    "log_worker",
    "mark_ui_thread",
    "threads_log_path",
    "timed_op",
    "wrap_root_after",
]
