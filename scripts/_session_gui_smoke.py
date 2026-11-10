#!/usr/bin/env python3
"""Launch-only GUI smoke test. Does not load examples (load hung in this agent)."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "validation_outputs" / "gui_smoke.json"


def main() -> int:
    import tkinter as tk

    report = {"attempted": True, "available": False}
    def _watchdog() -> None:
        time.sleep(40)
        print("WATCHDOG: GUI smoke did not finish", flush=True)
        os._exit(3)

    watchdog = threading.Thread(target=_watchdog, daemon=True)

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        report["detail"] = str(exc)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("GUI unavailable:", exc, flush=True)
        return 2

    report["available"] = True
    watchdog.start()
    from stablewalk.ui.tk.app import StableWalkGUI

    t0 = time.perf_counter()
    print("constructing StableWalkGUI...", flush=True)
    app = StableWalkGUI(root=root)
    report["construct_s"] = round(time.perf_counter() - t0, 2)
    print(f"constructed in {report['construct_s']}s", flush=True)
    root.update()
    report["geometry"] = root.winfo_geometry()
    report["mapped"] = bool(root.winfo_ismapped())
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout

    sizes = []
    for width, height in ((1920, 1080), (1366, 768)):
        root.geometry(f"{width}x{height}+20+20")
        apply_responsive_layout(app, width=width, height=height)
        root.update()
        sizes.append(
            {
                "requested": f"{width}x{height}",
                "geometry": root.winfo_geometry(),
                "notebook_w": int(getattr(app, "_dashboard_notebook", root).winfo_width() or 0),
                "notebook_h": int(getattr(app, "_dashboard_notebook", root).winfo_height() or 0),
            }
        )
    report["resize"] = sizes
    report["ok"] = True
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    try:
        root.destroy()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
