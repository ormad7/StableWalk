#!/usr/bin/env python3
"""Validate Overview on the three official v2 demo videos.

Checks playback, pause, seeking, example switching, resize, fullscreen,
fresh analysis, OpenSim stage wording, and comparison context. Does not
change scoring thresholds.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/output/example_v2_verification"
KEYS = ("abnormal", "normal", "athletic")


def _silence_dialogs(captured: list) -> None:
    import tkinter.messagebox as mb

    def _ok(*a, **k):
        captured.append((a, k))
        return "ok"

    mb.showinfo = _ok  # type: ignore[assignment]
    mb.showwarning = _ok  # type: ignore[assignment]
    mb.showerror = _ok  # type: ignore[assignment]


def _label_text(widget) -> str:
    if widget is None:
        return ""
    try:
        return str(widget.cget("text"))
    except Exception:
        return ""


def snapshot_overview(app) -> dict:
    from stablewalk.opensim_ik_status import resolve_opensim_status_pair_from_gui

    hud_labels = getattr(app, "_overview_hud_value_labels", {}) or {}
    path_stats = getattr(app, "overview_path_metric_values", {}) or {}
    kpis = getattr(app, "_overview_clinical_value_labels", {}) or {}
    context = getattr(app, "_overview_context_labels", {}) or {}
    pair = resolve_opensim_status_pair_from_gui(app)
    tree = getattr(app, "overview_comparison_tree", None)
    tree_rows = []
    if tree is not None:
        try:
            for item in tree.get_children():
                tree_rows.append(list(tree.item(item, "values")))
        except Exception:
            pass
    seq = getattr(app, "sequence", None)
    return {
        "demo_key": getattr(getattr(app, "_active_demo_gait", None), "key", None),
        "sequence_id": id(seq) if seq is not None else 0,
        "gait_motion": app.gait_motion is not None,
        "hud": {key: _label_text(lbl) for key, lbl in hud_labels.items()},
        "path": {key: _label_text(lbl) for key, lbl in path_stats.items()},
        "kpis": {key: _label_text(lbl) for key, lbl in kpis.items()},
        "context": {key: _label_text(lbl) for key, lbl in context.items()},
        "opensim_short": pair.short_label,
        "opensim_tooltip": pair.tooltip,
        "comparison_rows": tree_rows,
    }


def _computed_still_visible(snap: dict) -> bool:
    pending = {"", "—", "Not available", "N/A"}
    for group in ("hud", "path", "kpis"):
        for text in (snap.get(group) or {}).values():
            value = str(text or "").strip()
            if value and value not in pending and "not computed" not in value.lower():
                if value not in {
                    "Load and analyze a video to see the clinical gait overview.",
                    "Key findings appear after analysis.",
                }:
                    return True
    ctx = snap.get("context") or {}
    for key in ("duration", "cycles", "tracking"):
        value = str(ctx.get(key) or "").strip()
        if value and value not in pending:
            return True
    return False


class OverviewV2Validation:
    def __init__(self, app, root) -> None:
        self.app = app
        self.root = root
        self.report: dict = {
            "examples": [],
            "switch_cleared": [],
            "resize": {},
            "fullscreen": {},
            "gui_available": True,
        }
        self._idx = 0
        self._saw_busy = False
        self._prev_seq = 0
        self._deadline = 0.0
        self.exit_code = 0
        self._last_loaded: dict | None = None

    def start(self) -> None:
        self.root.after(250, self._begin)

    def _begin(self) -> None:
        from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES, demo_path

        examples = [ex for ex in DEMO_GAIT_EXAMPLES if ex.key in KEYS]
        if self._idx >= len(examples):
            self._exercise_resize_and_fullscreen()
            self._finish()
            return
        ex = examples[self._idx]
        official = demo_path(ex)
        print(f"Load {ex.display_name} {official.name}", flush=True)
        if not official.is_file():
            self.report["examples"].append(
                {"key": ex.key, "exists": False, "blocker": "missing file"}
            )
            self.exit_code = 1
            self._idx += 1
            self.root.after(80, self._begin)
            return
        self._current = {
            "key": ex.key,
            "display_name": ex.display_name,
            "path": str(official),
            "filename": official.name,
            "exists": True,
        }
        self._saw_busy = False
        self._prev_seq = id(self.app.sequence) if self.app.sequence is not None else 0
        self._deadline = time.time() + 480.0
        before = snapshot_overview(self.app)
        self.app._load_demo_gait(ex.key)
        self.root.update_idletasks()
        after_switch = snapshot_overview(self.app)
        stale = _computed_still_visible(after_switch)
        self._current["stale_after_switch"] = stale
        self._current["switch_context"] = after_switch.get("context")
        if stale and self._idx > 0:
            self.exit_code = 1
        self.report["switch_cleared"].append(
            {
                "to": ex.key,
                "stale": stale,
                "previous_demo": before.get("demo_key"),
                "hud": after_switch.get("hud"),
                "kpis": after_switch.get("kpis"),
                "context_computed": {
                    key: after_switch.get("context", {}).get(key)
                    for key in ("duration", "cycles", "tracking")
                },
            }
        )
        self.root.after(80, self._poll_load)

    def _ready(self) -> bool:
        app = self.app
        if hasattr(app, "_drain_pending_pipeline"):
            try:
                app._drain_pending_pipeline()
            except Exception:
                pass
        busy = bool(app._analysis_workflow_busy())
        if busy:
            self._saw_busy = True
        seq_id = id(app.sequence) if app.sequence is not None else 0
        return bool(
            self._saw_busy
            and not busy
            and app.sequence is not None
            and getattr(app, "pose_indices", None)
            and seq_id != self._prev_seq
        )

    def _poll_load(self) -> None:
        if self._ready():
            self._record_loaded()
            return
        if time.time() > self._deadline:
            self._current["timeout"] = True
            self.report["examples"].append(self._current)
            self.exit_code = 1
            self._idx += 1
            self.root.after(80, self._begin)
            return
        self.root.after(80, self._poll_load)

    def _record_loaded(self) -> None:
        from stablewalk.ui.playback_state import (
            assert_playback_surfaces_synced,
            commit_playback_index,
            collect_playback_sync_probe,
            read_playback_state,
        )
        from stablewalk.ui.tk.dashboard_overview_view_mode import (
            toggle_overview_full_screen,
        )
        from stablewalk.ui.tk.dashboard_panel_maximize import current_maximized_panel_id

        app = self.app
        app._update_overview_playback_hud(include_clinical=True)
        app._refresh_overview_recording_context()
        snap = snapshot_overview(app)
        pair_tip = str(snap.get("opensim_tooltip") or "")
        opensim_ok = (
            "Succeeded:" in pair_tip
            and "Unavailable:" in pair_tip
            and "IK Partial" not in str(snap.get("opensim_short") or "")
        )
        ctx = snap.get("context") or {}
        documented_ok = bool(ctx.get("source_label")) and ctx.get("source_label") not in {
            ctx.get("duration"),
            ctx.get("cycles"),
            ctx.get("tracking"),
        }
        n = len(app.pose_indices or [])
        seeks = sorted({0, n // 3, (2 * n) // 3, max(0, n - 1)})
        sync_ok = []
        for idx in seeks:
            commit_playback_index(app, idx)
            if hasattr(app, "frame_var"):
                try:
                    app.frame_var.set(int(idx))
                except Exception:
                    pass
            app._show_pose_at(idx, force_draw=True)
            app._update_overview_path_tip_only()
            if hasattr(app, "_sync_transport_labels"):
                try:
                    app._sync_transport_labels()
                except Exception:
                    pass
            self.root.update_idletasks()
            state = read_playback_state(app)
            probe = collect_playback_sync_probe(app, state)
            try:
                assert_playback_surfaces_synced(app, state)
                ok = True
                err = None
            except AssertionError as exc:
                ok = False
                err = str(exc)
                self.exit_code = 1
            sync_ok.append(
                {
                    "list_pos": idx,
                    "source_frame": None if state is None else state.source_frame_index,
                    "time_s": None if state is None else state.current_time_seconds,
                    "consistent": probe.consistent() and ok,
                    "error": err,
                }
            )

        play_ok = False
        pause_ok = False
        try:
            if not app.playing:
                app._toggle_play()
            self.root.update_idletasks()
            play_ok = bool(app.playing)
            if app.playing:
                app._toggle_play()
            self.root.update_idletasks()
            pause_ok = not bool(app.playing)
        except Exception as exc:
            self._current["playback_error"] = str(exc)
            self.exit_code = 1

        coverage = app._overview_hud_tracking_coverage()
        self._current.update(
            {
                "snapshot": snap,
                "opensim_short": snap.get("opensim_short"),
                "opensim_names_stages": opensim_ok,
                "documented_separate": documented_ok,
                "context": ctx,
                "sync": sync_ok,
                "play_ok": play_ok,
                "pause_ok": pause_ok,
                "same_video": Path(getattr(app.sequence, "source_video", "")).name
                == Path(self._current["path"]).name,
                "detection_pct": None if coverage is None else coverage.detection_pct,
                "usable_gait_frames": None
                if coverage is None
                else coverage.usable_gait_frames,
            }
        )
        if not opensim_ok or not documented_ok or not play_ok or not pause_ok:
            self.exit_code = 1
        if not all(row["consistent"] for row in sync_ok):
            self.exit_code = 1
        print(
            f"  opensim={snap.get('opensim_short')!r} "
            f"duration={ctx.get('duration')!r} cycles={ctx.get('cycles')!r} "
            f"play={play_ok} pause={pause_ok} sync={all(r['consistent'] for r in sync_ok)}",
            flush=True,
        )
        self.report["examples"].append(self._current)
        self._last_loaded = self._current
        self._idx += 1
        _ = toggle_overview_full_screen
        _ = current_maximized_panel_id
        self.root.after(80, self._begin)

    def _exercise_resize_and_fullscreen(self) -> None:
        from stablewalk.ui.tk.dashboard_overview_view_mode import (
            toggle_overview_full_screen,
        )
        from stablewalk.ui.tk.dashboard_panel_maximize import current_maximized_panel_id

        app = self.app
        try:
            before = (self.root.winfo_width(), self.root.winfo_height())
            self.root.geometry("1180x720+20+20")
            self.root.update_idletasks()
            self.root.update()
            mid = (self.root.winfo_width(), self.root.winfo_height())
            self.root.geometry("1520x900+20+20")
            self.root.update_idletasks()
            self.root.update()
            after = (self.root.winfo_width(), self.root.winfo_height())
            self.report["resize"] = {
                "ok": True,
                "before": before,
                "mid": mid,
                "after": after,
            }
        except Exception as exc:
            self.report["resize"] = {"ok": False, "error": str(exc)}
            self.exit_code = 1
        try:
            toggle_overview_full_screen(app)
            self.root.update_idletasks()
            entered = current_maximized_panel_id(app) is not None
            toggle_overview_full_screen(app)
            self.root.update_idletasks()
            restored = current_maximized_panel_id(app) is None
            self.report["fullscreen"] = {
                "ok": bool(entered and restored),
                "entered": entered,
                "restored": restored,
            }
            if not (entered and restored):
                self.exit_code = 1
        except Exception as exc:
            self.report["fullscreen"] = {"ok": False, "error": str(exc)}
            self.exit_code = 1

    def _finish(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        dest = OUT / "overview_v2_validation.json"
        dest.write_text(json.dumps(self.report, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {dest}", flush=True)
        self.root.after(50, self.root.quit)


def main() -> int:
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        dest = OUT / "overview_v2_validation.json"
        dest.write_text(
            json.dumps(
                {
                    "gui_available": False,
                    "blocker": str(exc),
                    "examples": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"GUI unavailable: {exc}", flush=True)
        print(f"Wrote {dest}", flush=True)
        return 2

    from stablewalk.ui.tk.app import StableWalkGUI

    dialogs: list = []
    _silence_dialogs(dialogs)
    root.title("Overview v2 validation")
    root.geometry("1280x720+20+20")
    app = StableWalkGUI(root=root)
    check = OverviewV2Validation(app, root)
    check.start()
    root.mainloop()
    try:
        root.destroy()
    except Exception:
        pass
    return check.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
