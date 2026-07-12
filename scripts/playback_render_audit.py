"""Quick 120-frame render diagnostic smoke script."""
from __future__ import annotations

import tkinter as tk

from stablewalk.ui.tk.app import StableWalkGUI
from stablewalk.ui.tk.render_diagnostics import run_playback_render_stress_test


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    app = StableWalkGUI(root=root)
    root.update_idletasks()
    app._render_debug = True
    results = run_playback_render_stress_test(app, frames=120, scroll_during_playback=True)
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status} ({detail})")
    root.destroy()


if __name__ == "__main__":
    main()
