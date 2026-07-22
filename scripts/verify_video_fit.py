"""Verify the Video & Pose Overlay uses a no-crop 'contain' fit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import tkinter as tk

import numpy as np

from stablewalk.ui.tk.app import StableWalkGUI


def main() -> int:
    root = tk.Tk()
    root.geometry("1400x900")
    app = StableWalkGUI(root=root)
    app.root.update_idletasks()
    app.video_label.update_idletasks()

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if ok:
            print(f"PASS  {name}")
        else:
            failures.append(f"{name}: {detail}")
            print(f"FAIL  {name}: {detail}")

    lw = max(app.video_label.winfo_width(), 1)
    lh = max(app.video_label.winfo_height(), 1)
    box_w = max(lw, 64)
    box_h = max(lh, 64)

    cases = {
        "portrait 720x1280": (1280, 720),  # (h, w)
        "landscape 1080x1920": (1080, 1920),
        "square 800x800": (800, 800),
    }
    for name, (h, w) in cases.items():
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        app._set_video_image(rgb)
        pw = app._photo.width()
        ph = app._photo.height()

        check(f"{name}: fits width (no crop)", pw <= box_w + 1, f"{pw} > {box_w}")
        check(f"{name}: fits height (no crop)", ph <= box_h + 1, f"{ph} > {box_h}")

        src_ratio = w / h
        out_ratio = pw / ph
        check(
            f"{name}: aspect preserved",
            abs(src_ratio - out_ratio) < 0.02,
            f"src={src_ratio:.3f} out={out_ratio:.3f}",
        )
        # The image should fill at least one dimension of the box (largest fit).
        fills = (pw >= box_w - 2) or (ph >= box_h - 2)
        check(f"{name}: fills the panel (largest contain)", fills, f"{pw}x{ph} in {box_w}x{box_h}")

    root.destroy()

    if failures:
        print("\nFAILED", len(failures), "check(s):")
        for f in failures:
            print(" -", f)
        return 1
    print("\nAll video-fit checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
