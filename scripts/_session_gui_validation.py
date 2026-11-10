#!/usr/bin/env python3
"""Scripted Tk GUI checks for the current UI session. Does not retune analysis."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "validation_outputs" / "gui_session.json"
EXAMPLES = (
    ("abnormal", "Abnormal Gait", "abnormal_gait_v2.mp4"),
    ("normal", "Normal Gait", "normal_gait_v2.mp4"),
    ("athletic", "Fast Walking", "fast_walking_v2.mp4"),
)
SIZES = ((1920, 1080), (1366, 768))


def _silence_dialogs() -> None:
    import tkinter.messagebox as mb

    mb.showinfo = lambda *a, **k: "ok"  # type: ignore[assignment]
    mb.showwarning = lambda *a, **k: "ok"  # type: ignore[assignment]
    mb.showerror = lambda *a, **k: print("dialog error:", a)  # type: ignore[assignment]


def _pump(root, seconds: float = 0.08) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            root.update()
        except Exception:
            return
        time.sleep(0.01)


def _label(widget) -> str:
    if widget is None:
        return ""
    try:
        return str(widget.cget("text") or "")
    except Exception:
        return ""


def _latest_poses(filename: str) -> Path | None:
    from stablewalk import config

    matches = []
    for path in config.POSES_DIR.glob("*_poses.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if Path(str(raw.get("source_video") or "")).name == filename:
            matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _select_inner(nb, title: str) -> bool:
    if nb is None:
        return False
    try:
        for tid in nb.tabs():
            if title.lower() in str(nb.tab(tid, "text")).strip().lower():
                nb.select(tid)
                return True
    except Exception:
        return False
    return False


def _clipped(widget) -> dict | None:
    if widget is None:
        return None
    try:
        if not widget.winfo_ismapped():
            return None
        w = int(widget.winfo_width() or 0)
        h = int(widget.winfo_height() or 0)
        req_w = int(widget.winfo_reqwidth() or 0)
        req_h = int(widget.winfo_reqheight() or 0)
        rw = int(widget.winfo_rootx() or 0)
        rh = int(widget.winfo_rooty() or 0)
        root = widget.winfo_toplevel()
        rw0 = int(root.winfo_rootx() or 0)
        rh0 = int(root.winfo_rooty() or 0)
        rw1 = rw0 + int(root.winfo_width() or 0)
        rh1 = rh0 + int(root.winfo_height() or 0)
        overflow = (rw + w > rw1 + 8) or (rh + h > rh1 + 8) or (rw < rw0 - 8)
        return {
            "w": w,
            "h": h,
            "req_w": req_w,
            "req_h": req_h,
            "overflow_window": overflow,
            "zero_size": w <= 1 or h <= 1,
        }
    except Exception as exc:
        return {"error": str(exc)}


def check_example(app, key: str, label: str, filename: str) -> dict:
    from stablewalk.ui.media.demo_gait import demo_path, example_by_key
    from stablewalk.ui.playback_state import (
        collect_playback_sync_probe,
        read_playback_state,
        sample_sync_frame_indices,
    )
    from stablewalk.ui.tk.dashboard_notebook import select_dashboard_tab
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout
    from stablewalk.opensim_ik_status import resolve_opensim_status_pair_from_gui

    rec: dict = {"key": key, "display_name": label, "filename": filename}
    ex = example_by_key(key)
    poses = _latest_poses(filename)
    rec["pose_json"] = str(poses) if poses else None
    rec["video_exists"] = bool(ex and demo_path(ex).is_file())
    if poses is None:
        rec["error"] = "cached pose JSON missing"
        return rec

    app._active_demo_gait = ex
    highlight = getattr(app, "_highlight_demo_button", None)
    if callable(highlight) and ex is not None:
        highlight(key)
    source = str(demo_path(ex)) if ex else filename
    app.url_var.set(source)
    print(f"  load_poses {label}...", flush=True)
    t0 = time.perf_counter()
    ok = app.load_poses(poses, fresh=True, expected_source=source, wait_for_analysis=False)
    rec["load_ok"] = bool(ok)
    if not ok:
        rec["error"] = "load_poses returned False"
        rec["load_s"] = round(time.perf_counter() - t0, 2)
        return rec
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        drain = getattr(app, "_drain_pending_pipeline", None)
        if callable(drain):
            try:
                drain()
            except Exception:
                pass
        _pump(app.root, seconds=0.08)
        busy = False
        check = getattr(app, "_analysis_workflow_busy", None)
        if callable(check):
            try:
                busy = bool(check())
            except Exception:
                busy = False
        if (
            not busy
            and app.sequence is not None
            and app.skeleton_player is not None
            and getattr(app.skeleton_player, "frame_count", 0)
        ):
            break
    else:
        rec["error"] = "analysis did not finish within 180s"
        rec["load_s"] = round(time.perf_counter() - t0, 2)
        return rec
    rec["load_s"] = round(time.perf_counter() - t0, 2)
    _pump(app.root, seconds=0.3)
    if app.sequence is None or app.skeleton_player is None:
        rec["error"] = "load_poses failed or no skeleton player"
        return rec

    n = int(app.skeleton_player.frame_count or 0)
    rec["frames"] = n
    sync_idxs = sample_sync_frame_indices(n)
    rec["playback"] = []
    for idx in sync_idxs:
        app._go_to(idx)
        _pump(app.root, seconds=0.05)
        state = read_playback_state(app)
        probe = collect_playback_sync_probe(app, state)
        rec["playback"].append(
            {
                "index": idx,
                "display_frame": state.display_frame_number if state else None,
                "time_s": state.current_time_seconds if state else None,
                "consistent": probe.consistent(),
                "debug": probe.as_debug_lines(),
            }
        )

    # Pause / play / step
    if app.playing:
        app._toggle_play()
        _pump(app.root, seconds=0.05)
    rec["paused"] = not bool(app.playing)
    app._toggle_play()
    _pump(app.root, seconds=0.25)
    rec["playing_after_toggle"] = bool(app.playing)
    app._toggle_play()
    _pump(app.root, seconds=0.1)
    rec["paused_after_toggle"] = not bool(app.playing)
    before = int(app.current_pos)
    app._step(1)
    _pump(app.root, seconds=0.05)
    rec["stepped"] = int(app.current_pos) != before

    select_dashboard_tab(app, "Biomechanics")
    _pump(app.root, seconds=0.2)
    activate = getattr(app, "_activate_biomech_workspace_tab", None)
    notes = {}
    if callable(activate):
        for tab_id in ("grf", "contact", "com", "stability"):
            activate(tab_id)
            _pump(app.root, seconds=0.25)
            notes[tab_id] = _label(getattr(app, "lbl_biomech_chart_note", None))
    rec["biomech_notes"] = notes
    rec["grf_peak_on_contact"] = "peak" in notes.get("contact", "").lower() or "BW" in notes.get(
        "contact", ""
    )
    rec["grf_note_has_force"] = "peak" in notes.get("grf", "").lower() or "BW" in notes.get(
        "grf", ""
    )

    select_dashboard_tab(app, "Results")
    _pump(app.root, seconds=0.15)
    select_dashboard_tab(app, "Advanced")
    _pump(app.root, seconds=0.15)
    pair = resolve_opensim_status_pair_from_gui(app)
    rec["opensim_short"] = pair.short_label
    rec["opensim_tooltip"] = pair.tooltip
    ik_value = ""
    widgets = getattr(app, "_adv_opensim_status_values", None) or {}
    if "ik_output" in widgets:
        ik_value = _label(widgets["ik_output"])
    rec["ik_mot_row"] = ik_value
    rec["view_log_button"] = getattr(app, "btn_adv_opensim_view_log", None) is not None

    rec["resize"] = []
    for width, height in SIZES:
        app.root.geometry(f"{width}x{height}+20+20")
        apply_responsive_layout(app, width=width, height=height)
        _pump(app.root, seconds=0.2)
        select_dashboard_tab(app, "Overview")
        _pump(app.root, seconds=0.1)
        select_dashboard_tab(app, "Biomechanics")
        _pump(app.root, seconds=0.1)
        widgets = {
            "note": _clipped(getattr(app, "lbl_biomech_chart_note", None)),
            "play": _clipped(getattr(app, "btn_play", None)),
            "notebook": _clipped(getattr(app, "_dashboard_notebook", None)),
        }
        rec["resize"].append({"size": f"{width}x{height}", "widgets": widgets})

    rec["ok_sync"] = all(item.get("consistent") for item in rec["playback"])
    rec["ok_contact_isolation"] = not rec["grf_peak_on_contact"]
    return rec


def main() -> int:
    import tkinter as tk

    from stablewalk.ui.tk.app import StableWalkGUI

    _silence_dialogs()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = {"gui_available": True, "examples": [], "failures": []}
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        report["gui_available"] = False
        report["detail"] = str(exc)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("GUI unavailable:", exc, flush=True)
        return 2

    root.title("StableWalk session GUI validation")
    root.geometry("1366x768+20+20")
    app = StableWalkGUI(root=root)
    _pump(root, seconds=0.4)
    try:
        for key, label, filename in EXAMPLES:
            print(f"GUI {label}...", flush=True)
            rec = check_example(app, key, label, filename)
            report["examples"].append(rec)
            if rec.get("error"):
                report["failures"].append(f"{key}: {rec['error']}")
            if rec.get("ok_sync") is False:
                report["failures"].append(f"{key}: playback surfaces disagreed")
            if rec.get("ok_contact_isolation") is False:
                report["failures"].append(f"{key}: GRF note on Contact tab")
        report["launch"] = "ok"
    except Exception as exc:
        report["failures"].append(f"gui exception: {exc}")
        print("GUI exception:", exc, flush=True)
    finally:
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {OUT}", flush=True)
        try:
            root.destroy()
        except Exception:
            pass
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
