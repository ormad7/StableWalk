#!/usr/bin/env python3
"""Re-inspect v2 poses with keypoints + cycle analysis, then drive the GUI."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POSE_FILES = {
    "abnormal": ROOT / "data/output/poses/run_8a254c44e19fc08b_1793524712974_poses.json",
    "normal": ROOT / "data/output/poses/run_90d21a9367838db7_1793524779679_poses.json",
}
NEED = ("left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle")
FEET = ("left_ankle", "right_ankle", "left_heel", "right_heel")
OUT = ROOT / "data/output/example_v2_verification"


def inspect_and_cycles() -> dict:
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles_from_pose_sequence
    from stablewalk.io.pose_loader import load_pose_sequence

    results = {}
    for key, path in POSE_FILES.items():
        raw = json.loads(path.read_text(encoding="utf-8"))
        seq = load_pose_sequence(path)
        usable = fail = left = right = feet = vis_fail = 0
        for fr in seq.frames:
            kps = {kp.name: kp for kp in fr.keypoints}
            if not fr.detected or not kps:
                fail += 1
                continue
            usable += 1
            if all(n in kps for n in ("left_hip", "left_knee", "left_ankle")):
                left += 1
            if all(n in kps for n in ("right_hip", "right_knee", "right_ankle")):
                right += 1
            if all(n in kps for n in FEET):
                feet += 1
            if any(kps[n].visibility < 0.5 for n in NEED if n in kps):
                vis_fail += 1
        cycles = analyze_gait_cycles_from_pose_sequence(seq)
        metrics = cycles.metrics.to_dict() if cycles.metrics else {}
        results[key] = {
            "poses_path": str(path),
            "source_video": raw.get("source_video"),
            "json_frame_count": raw.get("frame_count"),
            "json_detected_count": raw.get("detected_count"),
            "json_fps": raw.get("fps"),
            "usable_detected_frames": usable,
            "tracking_failures": fail,
            "left_limb_complete": left,
            "right_limb_complete": right,
            "feet_present": feet,
            "low_visibility_on_core_joints": vis_fail,
            "detected_cycles": len(cycles.cycles),
            "event_count": len(cycles.events),
            "event_sufficiency": cycles.event_sufficiency,
            "event_sufficiency_reason": cycles.event_sufficiency_reason,
            "warnings": list(cycles.warnings or []),
            "metrics": metrics,
        }
        print(f"{key}: usable={usable} fail={fail} cycles={len(cycles.cycles)} "
              f"suff={cycles.event_sufficiency} warn={cycles.warnings[:3]}", flush=True)
    dest = OUT / "pose_inspect_corrected.json"
    dest.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {dest}", flush=True)
    return results


def _silence_dialogs(captured: list) -> None:
    import tkinter.messagebox as mb

    def _info(*a, **k):
        captured.append(("info", a, k))
        return "ok"

    def _warn(*a, **k):
        captured.append(("warning", a, k))
        return "ok"

    def _error(*a, **k):
        captured.append(("error", a, k))
        print("dialog error:", a, flush=True)
        return "ok"

    mb.showinfo = _info  # type: ignore[assignment]
    mb.showwarning = _warn  # type: ignore[assignment]
    mb.showerror = _error  # type: ignore[assignment]


class V2GuiCheck:
    def __init__(self, app, root, dialogs: list) -> None:
        from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES, demo_path, format_about_active_example

        self.app = app
        self.root = root
        self.dialogs = dialogs
        self.examples = list(DEMO_GAIT_EXAMPLES)
        self.report: dict = {"examples": [], "gui_launched": True}
        self._idx = 0
        self._saw_busy = False
        self._prev_seq_id = 0
        self._deadline = 0.0
        self._current = None
        self.exit_code = 0
        self._demo_path = demo_path
        self._about = format_about_active_example

    def start(self) -> None:
        self.root.after(250, self._begin)

    def _begin(self) -> None:
        if self._idx >= len(self.examples):
            self._finish()
            return
        ex = self.examples[self._idx]
        official = self._demo_path(ex)
        about = self._about(ex)
        rec = {
            "key": ex.key,
            "display_name": ex.display_name,
            "official_path": str(official),
            "exists": official.is_file(),
            "about_source_url": ex.source_url,
            "about_has_source_url": "Source URL:" in about,
            "about_has_scores": "/100" in about or "usable cycle" in about.lower(),
            "about_excerpt": about.splitlines()[:10],
        }
        print(f"GUI {ex.display_name}: exists={official.is_file()} path={official.name}", flush=True)
        if not official.is_file():
            before = len(self.dialogs)
            self.app._load_demo_gait(ex.key)
            rec["missing_dialog"] = [
                {"kind": k, "title": a[0] if a else "", "text": str(a[1])[:400] if len(a) > 1 else ""}
                for k, a, _ in self.dialogs[before:]
            ]
            rec["url_after"] = self.app.url_var.get() if hasattr(self.app, "url_var") else None
            rec["active_after"] = getattr(getattr(self.app, "_active_demo_gait", None), "key", None)
            rec["gui_loaded"] = False
            rec["replacement_verified"] = False
            self.report["examples"].append(rec)
            self._idx += 1
            self.root.after(80, self._begin)
            return

        self._current = rec
        self._saw_busy = False
        self._prev_seq_id = id(self.app.sequence) if self.app.sequence is not None else 0
        self._deadline = time.time() + 420.0
        self.app._load_demo_gait(ex.key)
        self.root.after(80, self._poll)

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
            and seq_id != self._prev_seq_id
        )

    def _poll(self) -> None:
        rec = self._current
        if rec is None:
            self._idx += 1
            self.root.after(80, self._begin)
            return
        if self._ready():
            self._record_loaded(rec, ok=True)
            self._idx += 1
            self.root.after(120, self._begin)
            return
        if time.time() > self._deadline:
            rec["gui_loaded"] = False
            rec["timeout"] = True
            rec["url_after"] = self.app.url_var.get() if hasattr(self.app, "url_var") else None
            self.report["examples"].append(rec)
            print(f"  TIMEOUT {rec['key']}", flush=True)
            self.exit_code = 1
            self._idx += 1
            self.root.after(80, self._begin)
            return
        self.root.after(80, self._poll)

    def _record_loaded(self, rec: dict, *, ok: bool) -> None:
        app = self.app
        seq = app.sequence
        src = getattr(seq, "source_video", None) if seq is not None else None
        rec.update(
            {
                "gui_loaded": ok,
                "url_after": app.url_var.get() if hasattr(app, "url_var") else None,
                "active_after": getattr(getattr(app, "_active_demo_gait", None), "filename", None),
                "sequence_source": src,
                "sequence_frames": len(getattr(seq, "frames", []) or []) if seq else 0,
                "pose_indices": len(app.pose_indices or []),
                "same_video": bool(
                    src
                    and Path(src).resolve() == Path(rec["official_path"]).resolve()
                ),
            }
        )
        n = len(app.pose_indices or [])
        for idx in sorted({0, n // 2, max(0, n - 1)}):
            app._show_pose_at(idx)
            self.root.update_idletasks()
        rec["playback_indices"] = [0, n // 2, max(0, n - 1)]
        usable, detected = app._resolved_gait_cycle_count()
        rec["usable_cycles"] = usable
        rec["detected_cycles"] = detected
        print(
            f"  loaded url={rec['url_after']} src={src} frames={n} "
            f"same={rec['same_video']} cycles={usable}/{detected}",
            flush=True,
        )
        self.report["examples"].append(rec)

    def _finish(self) -> None:
        dest = OUT / "gui_v2_check.json"
        dest.write_text(json.dumps(self.report, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {dest}", flush=True)
        self.root.after(50, self.root.quit)


def run_gui() -> int:
    import tkinter as tk

    from stablewalk.ui.tk.app import StableWalkGUI

    dialogs: list = []
    _silence_dialogs(dialogs)
    root = tk.Tk()
    root.title("StableWalk v2 replacement check")
    root.geometry("960x640+40+40")
    app = StableWalkGUI(root=root)
    check = V2GuiCheck(app, root, dialogs)
    check.start()
    root.mainloop()
    try:
        root.destroy()
    except Exception:
        pass
    return check.exit_code


if __name__ == "__main__":
    inspect_and_cycles()
    raise SystemExit(run_gui())
