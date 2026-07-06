"""Quick user-flow test using cached poses (no full re-analyze)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tkinter as tk

POSES = ROOT / "data" / "output" / "poses"


def main() -> int:
    from stablewalk.ui.tk.app import StableWalkGUI
    from scripts.final_validation_report import _run_gui_interaction

    root = tk.Tk()
    root.withdraw()
    app = StableWalkGUI(root=root)
    root.deiconify()

    failures: list[str] = []
    for key in ("athletic", "normal", "abnormal"):
        pp = POSES / f"validation_{key}_poses.json"
        fails = _run_gui_interaction(app, key, poses_path=pp)
        # Collected data modal
        app._open_collected_data_dialog()
        root.update()
        dlg = getattr(app, "_collected_data_dialog", None)
        if dlg is None or not dlg.winfo_exists():
            fails.append(f"{key}: collected data modal failed")
        else:
            dlg.destroy()
            app._collected_data_dialog = None
            root.update()
        # Walk summary details button
        btn = getattr(app, "btn_walk_summary_details", None)
        if btn is None:
            fails.append(f"{key}: walk summary details button missing")
        elif str(btn.cget("state")) == str(tk.DISABLED):
            fails.append(f"{key}: walk summary details disabled after load")

        status = "PASS" if not fails else "FAIL"
        print(f"{key.capitalize()}: {status}")
        for f in fails:
            print(f"  - {f}")
        failures.extend(fails)

    root.destroy()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
