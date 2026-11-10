"""Reproduce Advanced-tab runtime errors with full exception capture."""
from __future__ import annotations

import logging
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tkinter as tk
from tkinter import messagebox

from stablewalk.ui.theme import GLYPH_COLLAPSE, GLYPH_EXPAND
from stablewalk.ui.tk.app import StableWalkGUI
from stablewalk.ui.tk.dashboard_notebook import (
    TAB_ADVANCED,
    TAB_BIOMECHANICS,
    TAB_MOTION,
    TAB_OVERVIEW,
    TAB_RESULTS_SUMMARY,
    dashboard_tab_widget,
    select_dashboard_tab,
)
from stablewalk.ui.tk.layout_scale import apply_process_dpi_awareness

ERRS: list[str] = []
LOG_ERRS: list[str] = []


def _record(kind: str, text: str) -> None:
    ERRS.append(f"{kind}\n{text}")
    print(f"{kind}\n{text}", flush=True)


def _hook(exc, val, tb) -> None:
    _record("CALLBACK_EXCEPTION", "".join(traceback.format_exception(exc, val, tb)))


def _excepthook(exc, val, tb) -> None:
    _record("SYS_EXCEPTHOOK", "".join(traceback.format_exception(exc, val, tb)))


class _ErrorLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.ERROR and not record.exc_info:
            return
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        if record.exc_info and not record.exc_text:
            msg += "\n" + "".join(traceback.format_exception(*record.exc_info))
        LOG_ERRS.append(msg)
        print("LOG_ERROR\n" + msg, flush=True)


def _pump(root: tk.Misc, seconds: float = 0.35) -> None:
    end = time.time() + seconds
    while time.time() < end:
        try:
            root.update()
        except tk.TclError:
            return
        time.sleep(0.02)


def _safe(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        _record(f"RAISED {label}", traceback.format_exc())


def _user_select(app, tab: str) -> None:
    page = dashboard_tab_widget(app, tab)
    if page is not None:
        app._dashboard_notebook.select(page)
        app.root.update()
        return
    select_dashboard_tab(app, tab)


def _walk_buttons(widget) -> list[tk.Misc]:
    found: list[tk.Misc] = []

    def walk(node) -> None:
        try:
            children = node.winfo_children()
        except tk.TclError:
            return
        cls = type(node).__name__
        if cls in ("Button", "TButton") or isinstance(node, tk.Button):
            found.append(node)
        for child in children:
            walk(child)

    walk(widget)
    return found


def _toggle_category(host) -> None:
    for btn in _walk_buttons(host):
        try:
            text = str(btn.cget("text") or "")
        except tk.TclError:
            continue
        if text in (GLYPH_EXPAND, GLYPH_COLLAPSE, "▶", "▼"):
            btn.invoke()
            return
    # Fallback: first tk.Button in the card header.
    for btn in _walk_buttons(host):
        if isinstance(btn, tk.Button):
            btn.invoke()
            return


def _close_toplevels(root: tk.Misc) -> None:
    for child in list(root.winfo_children()):
        try:
            if isinstance(child, tk.Toplevel):
                child.destroy()
        except tk.TclError:
            pass


def main() -> int:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger().addHandler(_ErrorLogHandler())
    sys.excepthook = _excepthook
    tk.Tk.report_callback_exception = staticmethod(_hook)

    orig_error = messagebox.showerror
    orig_warning = messagebox.showwarning

    def _trap_error(title, message, **kwargs):
        _record("MESSAGEBOX_ERROR", f"{title}: {message}")
        return "ok"

    def _trap_warning(title, message, **kwargs):
        _record("MESSAGEBOX_WARNING", f"{title}: {message}")
        return "ok"

    messagebox.showerror = _trap_error  # type: ignore[assignment]
    messagebox.showwarning = _trap_warning  # type: ignore[assignment]

    apply_process_dpi_awareness()
    root = tk.Tk()
    root.geometry("1366x768+40+40")
    app = StableWalkGUI(root=root)
    root.update()
    _pump(root, 0.4)

    poses = _ROOT / "data" / "output" / "poses" / "normal_gait_poses.json"
    video = _ROOT / "data" / "input" / "normal_gait.mp4"
    print(f"poses={poses.is_file()} video={video.is_file()}", flush=True)
    if video.is_file() and hasattr(app, "load_video"):
        # Prefer attaching the real clip so playback/OpenSim paths match the GUI.
        try:
            app.video_path = str(video)
        except Exception:
            pass
    if poses.is_file():
        print(f"load_poses {poses}", flush=True)
        try:
            app.load_poses(poses, fresh=True, wait_for_analysis=True)
        except TypeError:
            app.load_poses(poses, fresh=True)
        except Exception:
            _record("load_poses", traceback.format_exc())
    elif video.is_file():
        print(f"load video {video}", flush=True)
        _safe("load_video", lambda: app.load_video(str(video)))
    _pump(root, 1.5)

    print("=== open Advanced paused ===", flush=True)
    _safe("select Advanced", lambda: _user_select(app, TAB_ADVANCED))
    _pump(root, 1.2)

    hosts = getattr(app, "_advanced_category_hosts", {}) or {}
    print("hosts", list(hosts), flush=True)
    print(
        "content_built",
        getattr(app, "_advanced_content_built", None),
        "empty",
        getattr(getattr(app, "_advanced_empty_state", None), "cget", lambda *_: None)("text")
        if getattr(app, "_advanced_empty_state", None)
        else None,
        flush=True,
    )

    for key in ("session", "opensim", "real_to_sim", "exports", "reports"):
        host = hosts.get(key)
        print(f"toggle {key} host={host is not None}", flush=True)
        if host is None:
            continue
        _safe(f"toggle {key} close", lambda h=host: _toggle_category(h))
        _pump(root, 0.2)
        _safe(f"toggle {key} open", lambda h=host: _toggle_category(h))
        _pump(root, 0.2)

    btn_refresh = getattr(app, "btn_generated_files_refresh", None)
    if btn_refresh is not None:
        print("click Refresh Generated Files", flush=True)
        _safe("refresh files invoke", btn_refresh.invoke)
        _pump(root, 0.4)
    btn_show = getattr(app, "btn_generated_files_toggle", None)
    if btn_show is not None:
        print("click Show Generated Files", flush=True)
        _safe("show files invoke", btn_show.invoke)
        _pump(root, 0.5)
        _safe("hide files invoke", btn_show.invoke)
        _pump(root, 0.2)
        _safe("show files again", btn_show.invoke)
        _pump(root, 0.3)

    notify = getattr(app, "_notify_generated_files_changed", None)
    if callable(notify):
        _safe("notify generated files", lambda: notify(force=True))
        _pump(root, 0.3)

    _safe("refresh opensim status", app._refresh_opensim_status)
    _safe("refresh pipeline force", lambda: app._refresh_pipeline_status_panel(force=True))
    from stablewalk.ui.tk.dashboard_advanced import update_engineering_dashboard

    _safe("update engineering dashboard", lambda: update_engineering_dashboard(app))
    _pump(root, 0.4)

    cards = getattr(app, "_adv_engineering_cards", {}) or {}
    for key in list(cards):
        toggle = (cards.get(key) or {}).get("toggle")
        if toggle is None:
            continue
        print(f"engineering toggle {key}", flush=True)
        _safe(f"eng toggle {key}", lambda w=toggle: w.event_generate("<Button-1>"))
        _pump(root, 0.15)
        _close_toplevels(root)

    flow = getattr(app, "_flow_stage_widgets", {}) or {}
    for key, widgets in list(flow.items())[:3]:
        card = (widgets or {}).get("card")
        if card is None:
            continue
        print(f"click flow stage {key}", flush=True)
        _safe(f"flow click {key}", lambda w=card: w.event_generate("<Button-1>"))
        _pump(root, 0.25)
        _close_toplevels(root)

    btn_joint = getattr(app, "btn_view_detailed_data", None)
    if btn_joint is not None:
        print("click View Joint Data", flush=True)
        _safe("view joint invoke", btn_joint.invoke)
        _pump(root, 0.4)
        _close_toplevels(root)
        app._collected_data_dialog = None

    btn_adv = getattr(app, "btn_advanced_analysis", None)
    if btn_adv is not None:
        print("click Advanced Analysis", flush=True)
        _safe("advanced analysis invoke", btn_adv.invoke)
        _pump(root, 0.4)
        _close_toplevels(root)

    print("=== playback on Advanced ===", flush=True)
    _safe("play", app._toggle_play)
    _pump(root, 1.0)
    _safe("select Advanced while playing", lambda: _user_select(app, TAB_ADVANCED))
    _pump(root, 0.8)
    for _ in range(12):
        _pump(root, 0.05)
    _safe("refresh files while playing", lambda: getattr(app, "btn_generated_files_refresh", None) and app.btn_generated_files_refresh.invoke())
    _pump(root, 0.2)
    if app.playing:
        _safe("pause", app._toggle_play)
    _pump(root, 0.4)

    print("=== repeated tab navigation ===", flush=True)
    for i in range(4):
        for tab in (
            TAB_OVERVIEW,
            TAB_MOTION,
            TAB_BIOMECHANICS,
            TAB_RESULTS_SUMMARY,
            TAB_ADVANCED,
        ):
            _safe(f"nav {i} {tab}", lambda t=tab: _user_select(app, t))
            _pump(root, 0.12)
        _safe(f"play nav {i}", app._toggle_play)
        _pump(root, 0.2)
        _safe(f"nav playing {i}", lambda: _user_select(app, TAB_ADVANCED))
        _pump(root, 0.25)
        if app.playing:
            _safe(f"stop nav {i}", app._toggle_play)
        _pump(root, 0.12)

    print(f"ERRS {len(ERRS)} LOG_ERRS {len(LOG_ERRS)}", flush=True)
    if ERRS:
        print("===== FIRST ERROR =====", flush=True)
        print(ERRS[0][:4000], flush=True)
    ws = getattr(app, "_advanced_workspace", None)
    if ws is not None:
        print(
            "workspace",
            ws.winfo_width(),
            ws.winfo_height(),
            ws.winfo_viewable(),
            flush=True,
        )
    try:
        root.destroy()
    except tk.TclError:
        pass
    messagebox.showerror = orig_error
    messagebox.showwarning = orig_warning
    out = _ROOT / "docs" / "_advanced_runtime_debug.txt"
    out.write_text(
        f"error_count={len(ERRS)}\nlog_error_count={len(LOG_ERRS)}\n\n"
        + "\n\n==== CALLBACK/RAISED ====\n\n".join(ERRS)
        + "\n\n==== LOG ERRORS ====\n\n"
        + "\n\n----\n\n".join(LOG_ERRS),
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)
    return 1 if ERRS else 0


if __name__ == "__main__":
    raise SystemExit(main())
