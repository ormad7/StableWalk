#!/usr/bin/env python3
"""Load Fast Walking from the Examples menu and record identity checks."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/output/example_v2_verification"


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


class FastWalkingGuiCheck:
    def __init__(self, app, root, dialogs: list) -> None:
        from stablewalk.ui.media.demo_gait import (
            example_by_key,
            demo_path,
            format_about_active_example,
        )

        self.app = app
        self.root = root
        self.dialogs = dialogs
        self.ex = example_by_key("athletic")
        official = demo_path(self.ex)
        about = format_about_active_example(self.ex)
        self.report = {
            "gui_launched": True,
            "key": self.ex.key,
            "display_name": self.ex.display_name,
            "official_path": str(official),
            "exists": official.is_file(),
            "source_url": self.ex.source_url,
            "about_has_source_url": "Source URL:" in about,
            "about_has_walk_54": "walk_54" in about,
            "about_has_scores": "/100" in about or "usable cycle" in about.lower(),
            "about_excerpt": about.splitlines()[:12],
            "dialogs": [],
        }
        self._saw_busy = False
        self._prev_seq_id = 0
        self._deadline = 0.0
        self.exit_code = 0

    def start(self) -> None:
        self.root.after(250, self._begin)

    def _begin(self) -> None:
        official = Path(self.report["official_path"])
        print(f"GUI Fast Walking: exists={official.is_file()} path={official}", flush=True)
        if not official.is_file():
            self.report["gui_loaded"] = False
            self.report["blocker"] = "fast_walking_v2.mp4 missing"
            self.exit_code = 1
            self._finish()
            return
        self._prev_seq_id = id(self.app.sequence) if self.app.sequence is not None else 0
        self._deadline = time.time() + 600.0
        before = len(self.dialogs)
        self.app._load_demo_gait("athletic")
        self.report["dialogs"] = [
            {"kind": k, "title": a[0] if a else "", "text": str(a[1])[:400] if len(a) > 1 else ""}
            for k, a, _ in self.dialogs[before:]
        ]
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
        if self._ready():
            self._record_loaded()
            self._finish()
            return
        if time.time() > self._deadline:
            self.report["gui_loaded"] = False
            self.report["timeout"] = True
            self.report["url_after"] = self.app.url_var.get() if hasattr(self.app, "url_var") else None
            print("TIMEOUT Fast Walking GUI", flush=True)
            self.exit_code = 1
            self._finish()
            return
        self.root.after(80, self._poll)

    def _record_loaded(self) -> None:
        app = self.app
        seq = app.sequence
        src = getattr(seq, "source_video", None) if seq is not None else None
        official = Path(self.report["official_path"]).resolve()
        n = len(app.pose_indices or [])
        for idx in sorted({0, n // 2, max(0, n - 1)}):
            app._show_pose_at(idx)
            self.root.update_idletasks()
        usable, detected = app._resolved_gait_cycle_count()
        self.report.update(
            {
                "gui_loaded": True,
                "url_after": app.url_var.get() if hasattr(app, "url_var") else None,
                "active_after": getattr(getattr(app, "_active_demo_gait", None), "filename", None),
                "sequence_source": src,
                "sequence_frames": len(getattr(seq, "frames", []) or []) if seq else 0,
                "pose_indices": n,
                "same_video": bool(src and Path(src).resolve() == official),
                "playback_indices": [0, n // 2, max(0, n - 1)],
                "usable_cycles": usable,
                "detected_cycles": detected,
                "legacy_not_loaded": "athletic_walking.mp4" not in str(src or ""),
            }
        )
        print(
            f"  loaded url={self.report['url_after']} src={src} frames={n} "
            f"same={self.report['same_video']} cycles={usable}/{detected}",
            flush=True,
        )

    def _finish(self) -> None:
        dest = OUT / "fast_walking_gui_check.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.report, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {dest}", flush=True)
        self.root.after(50, self.root.quit)


def main() -> int:
    import tkinter as tk

    from stablewalk.ui.tk.app import StableWalkGUI

    dialogs: list = []
    _silence_dialogs(dialogs)
    root = tk.Tk()
    root.title("StableWalk Fast Walking example check")
    root.geometry("960x640+40+40")
    app = StableWalkGUI(root=root)
    check = FastWalkingGuiCheck(app, root, dialogs)
    check.start()
    root.mainloop()
    try:
        root.destroy()
    except Exception:
        pass
    return check.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
