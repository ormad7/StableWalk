#!/usr/bin/env python3
"""Validate Overview tracking / OpenSim / playback sync on both example videos."""

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


class OverviewFixCheck:
    def __init__(self, app, root) -> None:
        self.app = app
        self.root = root
        self.report: dict = {"examples": []}
        self._idx = 0
        self._saw_busy = False
        self._prev_seq = 0
        self._deadline = 0.0
        self._seeks: list[int] = []
        self._seek_i = 0
        self.exit_code = 0

    def start(self) -> None:
        self.root.after(250, self._begin)

    def _begin(self) -> None:
        from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES, demo_path

        examples = [ex for ex in DEMO_GAIT_EXAMPLES if ex.key in KEYS]
        if self._idx >= len(examples):
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
            "exists": True,
        }
        self._saw_busy = False
        self._prev_seq = id(self.app.sequence) if self.app.sequence is not None else 0
        self._deadline = time.time() + 480.0
        self.app._load_demo_gait(ex.key)
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
        from stablewalk.opensim_ik_status import resolve_opensim_status_pair_from_gui
        from stablewalk.ui.playback_state import (
            assert_playback_surfaces_synced,
            commit_playback_index,
            collect_playback_sync_probe,
            read_playback_state,
        )

        app = self.app
        app._update_overview_playback_hud(include_clinical=True)
        tracking = app._overview_hud_tracking_confidence()
        coverage = app._overview_hud_tracking_coverage()
        hud_labels = getattr(app, "_overview_hud_value_labels", {}) or {}
        hud_caps = getattr(app, "_overview_hud_caption_labels", {}) or {}
        hud_values = {
            key: str(hud_labels[key].cget("text"))
            for key in ("detected", "joints", "usable")
            if key in hud_labels
        }
        hud_captions = {
            key: str(hud_caps[key].cget("text"))
            for key in ("detected", "joints", "usable")
            if key in hud_caps
        }
        path_stats = getattr(app, "overview_path_metric_values", {}) or {}
        path_values = {
            key: str(path_stats[key].cget("text"))
            for key in ("path_length", "displacement", "xyz")
            if key in path_stats
        }
        ax_skel = getattr(app, "ax_3d", None)
        fixed_box = getattr(ax_skel, "_sw_fixed_view_box", None) if ax_skel is not None else None
        scale_spans = []
        if app.skeleton_player is not None and ax_skel is not None:
            from stablewalk.ui.viewers.gait_skeleton_renderer import _apply_view_limits

            rec = getattr(app, "gait_motion", None)
            snaps = list(getattr(rec, "snapshots", None) or [])
            sample = snaps[:1] + snaps[len(snaps)//2:len(snaps)//2+1] + snaps[-1:]
            for snap in sample:
                try:
                    _apply_view_limits(ax_skel, snap)
                    xlim, ylim = ax_skel.get_xlim(), ax_skel.get_ylim()
                    scale_spans.append((xlim[1] - xlim[0], ylim[1] - ylim[0]))
                except Exception:
                    pass
        scale_stable = (
            len(scale_spans) >= 2
            and all(abs(sx - scale_spans[0][0]) < 1e-6 and abs(sy - scale_spans[0][1]) < 1e-6
                    for sx, sy in scale_spans[1:])
        )
        pair = resolve_opensim_status_pair_from_gui(app)
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
                    "probe": probe.as_debug_lines(),
                    "consistent": probe.consistent() and ok,
                    "error": err,
                }
            )
        self._current.update(
            {
                "tracking_hud": tracking,
                "hud_values": hud_values,
                "hud_captions": hud_captions,
                "path_stats": path_values,
                "path_stats_collapsed": not bool(
                    getattr(app, "_overview_path_stats_expanded", True)
                ),
                "inspector_collapsed": not bool(
                    getattr(app, "_overview_inspector_expanded", True)
                ),
                "skeleton_fixed_box": fixed_box is not None,
                "skeleton_scale_stable": scale_stable,
                "skeleton_spans": scale_spans,
                "tracking_is_single_percent": tracking.strip().endswith("%")
                and "\n" not in tracking,
                "detection_pct": None if coverage is None else coverage.detection_pct,
                "joint_visibility_pct": None
                if coverage is None
                else coverage.joint_visibility_pct,
                "usable_gait_frames": None
                if coverage is None
                else coverage.usable_gait_frames,
                "opensim_short": pair.short_label,
                "opensim_has_outputs": "Outputs:" in pair.tooltip,
                "opensim_ik": pair.ik.label,
                "opensim_pipeline": pair.pipeline.label,
                "sync": sync_ok,
                "same_video": Path(getattr(app.sequence, "source_video", "")).name
                == Path(self._current["path"]).name,
            }
        )
        print(
            f"  tracking={tracking!r} hud={hud_values} path={path_values} "
            f"skel_stable={scale_stable} inspector_collapsed="
            f"{self._current.get('inspector_collapsed')} "
            f"opensim={pair.short_label} "
            f"sync={all(r['consistent'] for r in sync_ok)}",
            flush=True,
        )
        self.report["examples"].append(self._current)
        self._idx += 1
        self.root.after(80, self._begin)

    def _finish(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        dest = OUT / "overview_fix_check.json"
        dest.write_text(json.dumps(self.report, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {dest}", flush=True)
        self.root.after(50, self.root.quit)


def main() -> int:
    import tkinter as tk

    from stablewalk.ui.tk.app import StableWalkGUI

    dialogs: list = []
    _silence_dialogs(dialogs)
    root = tk.Tk()
    root.title("Overview fix check")
    root.geometry("1280x720+20+20")
    app = StableWalkGUI(root=root)
    check = OverviewFixCheck(app, root)
    check.start()
    root.mainloop()
    try:
        root.destroy()
    except Exception:
        pass
    return check.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
