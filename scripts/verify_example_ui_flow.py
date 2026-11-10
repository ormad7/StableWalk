#!/usr/bin/env python3
"""Load each Examples clip, play frames, switch, and export analysis.

Drives the real Tk main loop so the pipeline worker can schedule UI completion.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _silence_dialogs() -> None:
    import tkinter.messagebox as mb

    mb.showinfo = lambda *a, **k: "ok"  # type: ignore[assignment]
    mb.showwarning = lambda *a, **k: "ok"  # type: ignore[assignment]
    mb.showerror = lambda *a, **k: print("dialog error:", a)  # type: ignore[assignment]


class ExampleFlow:
    def __init__(self, app, root, out_dir: Path) -> None:
        from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES

        self.app = app
        self.root = root
        self.out_dir = out_dir
        self.keys = [ex.key for ex in DEMO_GAIT_EXAMPLES]
        self.switch_keys = list(reversed(self.keys))
        self.report: dict[str, object] = {"examples": [], "switching": []}
        self.failures: list[str] = []
        self.exit_code = 0
        self._phase = "load"
        self._idx = 0
        self._prev_sid = 0
        self._saw_busy = False
        self._prev_seq_id = 0
        self._deadline = 0.0
        self._current_key = ""

    def start(self) -> None:
        self.root.after(250, self._begin_load)

    def _ready(self, key: str) -> bool:
        app = self.app
        if hasattr(app, "_drain_pending_pipeline"):
            try:
                app._drain_pending_pipeline()
            except Exception:
                pass
        busy = bool(app._analysis_workflow_busy())
        if busy:
            self._saw_busy = True
        active = getattr(getattr(app, "_active_demo_gait", None), "key", None)
        seq_id = id(app.sequence) if app.sequence is not None else 0
        return bool(
            self._saw_busy
            and not busy
            and app.sequence is not None
            and getattr(app, "pose_indices", None)
            and seq_id != self._prev_seq_id
            and active == key
        )

    def _begin_load(self) -> None:
        if self._idx >= len(self.keys):
            self._phase = "switch"
            self._idx = 0
            self._begin_switch()
            return
        key = self.keys[self._idx]
        self._current_key = key
        print(f"Loading {key}...", flush=True)
        self._prev_sid = int(getattr(self.app, "_session_id", 0) or 0)
        self._saw_busy = False
        self._prev_seq_id = id(self.app.sequence) if self.app.sequence is not None else 0
        self._deadline = time.time() + 300.0
        self.app._load_demo_gait(key)
        self.root.after(80, self._poll)

    def _begin_switch(self) -> None:
        if self._idx >= len(self.switch_keys):
            self._finish()
            return
        key = self.switch_keys[self._idx]
        self._current_key = key
        print(f"Switching to {key}...", flush=True)
        self._prev_sid = int(getattr(self.app, "_session_id", 0) or 0)
        self._saw_busy = False
        self._prev_seq_id = id(self.app.sequence) if self.app.sequence is not None else 0
        self._deadline = time.time() + 300.0
        self.app._load_demo_gait(key)
        self.root.after(80, self._poll)

    def _poll(self) -> None:
        key = self._current_key
        if self._ready(key):
            if self._phase == "load":
                self._record_load(key, ok=True)
                self._idx += 1
                self.root.after(120, self._begin_load)
            else:
                active = getattr(self.app._active_demo_gait, "key", None)
                self.report["switching"].append(  # type: ignore[union-attr]
                    {"key": key, "ok": True, "active": active}
                )
                if active != key:
                    self.failures.append(f"switch {key}: active={active}")
                self._idx += 1
                self.root.after(120, self._begin_switch)
            return
        if time.time() > self._deadline:
            if self._phase == "load":
                self._record_load(key, ok=False)
                self.failures.append(f"{key}: load/analysis did not finish")
                print(f"  FAIL load {key}", flush=True)
                self._idx += 1
                self.root.after(120, self._begin_load)
            else:
                active = getattr(getattr(self.app, "_active_demo_gait", None), "key", None)
                self.report["switching"].append(  # type: ignore[union-attr]
                    {"key": key, "ok": False, "active": active}
                )
                self.failures.append(f"switch {key}: ok=False active={active}")
                self._idx += 1
                self.root.after(120, self._begin_switch)
            return
        self.root.after(80, self._poll)

    def _record_load(self, key: str, *, ok: bool) -> None:
        from stablewalk.data_export import export_analysis
        from stablewalk.ui.media.demo_gait import (
            demo_gait_interpretation,
            format_about_active_example,
        )

        app = self.app
        ex = app._active_demo_gait
        about = format_about_active_example(ex) if ex else ""
        rec: dict[str, object] = {
            "key": key,
            "display_name": ex.display_name if ex else None,
            "loaded": ok,
            "frames": len(app.pose_indices) if app.pose_indices else 0,
            "session_id": int(getattr(app, "_session_id", 0) or 0),
            "about_has_source_url": "Source URL:" in about,
            "about_has_dataset": "Dataset:" in about,
            "about_has_condition": "Documented condition / activity:" in about,
            "about_has_license": "License / access:" in about,
            "about_has_fps": "Original FPS" in about,
            "about_has_duration": "Duration" in about,
            "about_has_interval": "Selected time interval:" in about,
            "about_has_scores": "/100" in about or "usable cycle" in about.lower(),
        }
        if not ok:
            self.report["examples"].append(rec)  # type: ignore[union-attr]
            return

        n = len(app.pose_indices or [])
        for idx in sorted({0, n // 2, max(0, n - 1)}):
            app._show_pose_at(idx)
            self.root.update_idletasks()
        rec["playback_indices"] = [0, n // 2, max(0, n - 1)]

        export_path = self.out_dir / f"{key}_analysis.json"
        try:
            written = export_analysis(app.sequence, export_path)
            rec["export"] = str(written)
            rec["export_ok"] = written.is_file() and written.stat().st_size > 100
            if not rec["export_ok"]:
                self.failures.append(f"{key}: export empty")
        except Exception as exc:
            rec["export_ok"] = False
            rec["export_error"] = str(exc)
            self.failures.append(f"{key}: export failed ({exc})")

        biomech = getattr(app, "_biomech", None)
        usable, detected = app._resolved_gait_cycle_count()
        rec["usable_cycles"] = usable
        rec["detected_cycles"] = detected
        rec["completeness_pct"] = getattr(biomech, "completeness_pct", None)
        interp = demo_gait_interpretation(
            key,
            usable_cycles=int(usable or 0),
            detected_cycles=int(detected or 0),
            completeness_pct=float(rec["completeness_pct"] or 0.0),
        )
        rec["insufficient_data_warning"] = bool(
            interp
            and "Insufficient data" in (interp.gait_quality + interp.analysis_confidence)
        )
        if rec["about_has_scores"]:
            self.failures.append(f"{key}: About mixed in computed scores")
        print(
            f"  {ex.display_name}: frames={n} export={rec.get('export_ok')} "
            f"cycles={rec.get('usable_cycles')}",
            flush=True,
        )
        self.report["examples"].append(rec)  # type: ignore[union-attr]

    def _finish(self) -> None:
        self.report["failures"] = self.failures
        dest = self.out_dir / "flow_report.json"
        dest.write_text(json.dumps(self.report, indent=2), encoding="utf-8")
        print(f"Wrote {dest}", flush=True)
        if self.failures:
            print("FAILURES:", flush=True)
            for item in self.failures:
                print(" -", item, flush=True)
            self.exit_code = 1
        else:
            print("All example load / playback / switch / export checks passed.", flush=True)
            self.exit_code = 0
        self.root.after(50, self.root.quit)


def main() -> int:
    import tkinter as tk

    from stablewalk.ui.tk.app import StableWalkGUI

    _silence_dialogs()
    out_dir = ROOT / "data" / "output" / "example_ui_flow"
    out_dir.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    root.title("StableWalk example flow")
    root.geometry("960x640+40+40")
    app = StableWalkGUI(root=root)
    flow = ExampleFlow(app, root, out_dir)
    flow.start()
    root.mainloop()
    try:
        root.destroy()
    except Exception:
        pass
    return flow.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
