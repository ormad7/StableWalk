"""Render Spectral Analysis at 1366×768 and 1920×1080 and capture the live Tk UI."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

OUT = ROOT / "docs" / "spectral_layout_verify"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    import tkinter as tk

    from stablewalk.analysis.spectral.report import analyze_spectral
    from stablewalk.ui.tk.dashboard_spectral import (
        build_spectral_analysis_workspace,
        open_spectral_plot_window,
        relayout_spectral_plots,
        render_spectral_workspace,
        spectral_figure_min_px,
        spectral_split_minsizes,
    )
    from stablewalk.ui.tk.layout_scale import dpi_scale_factor, SCALE_TARGETS
    from test_spectral_analysis import _sine_recording

    try:
        from PIL import ImageGrab
    except Exception:
        ImageGrab = None  # type: ignore[assignment]

    root = tk.Tk()
    root.title("StableWalk Spectral layout verify")
    root.attributes("-topmost", True)
    scale = dpi_scale_factor(root)
    gui = SimpleNamespace(root=root, _motion_page_notebook=None, _layout_dpi_scale=1.0)
    parent = tk.Frame(root)
    host = build_spectral_analysis_workspace(gui, parent)
    gui._spectral_report = analyze_spectral(_sine_recording())
    gui._spectral_channel_id = "dof:left_knee_flexion"
    render_spectral_workspace(gui)

    payload: dict[str, object] = {
        "dpi_scale": scale,
        "scale_targets": list(SCALE_TARGETS),
        "split_minsizes": {
            str(s): list(spectral_split_minsizes(s)) for s in SCALE_TARGETS
        },
        "figure_mins": {str(s): list(spectral_figure_min_px(s)) for s in SCALE_TARGETS},
        "sizes": {},
    }
    parent.pack_forget()
    for width, height in ((1366, 768), (1920, 1080)):
        root.geometry(f"{width}x{height}+40+40")
        parent.place(x=0, y=0, width=width, height=height)
        root.deiconify()
        root.lift()
        root.update_idletasks()
        root.update()
        gui._spectral_sash_for_width = None
        place = getattr(gui, "_place_spectral_sash", None)
        if callable(place):
            place()
        relayout_spectral_plots(gui)
        root.update()
        time.sleep(0.25)
        root.update()
        row = {
            "host": [int(host.winfo_width()), int(host.winfo_height())],
            "plot_pane": [
                int(gui._spectral_plot_pane.winfo_width()),
                int(gui._spectral_plot_pane.winfo_height()),
            ],
            "table_pane": [
                int(gui._spectral_side.winfo_width()),
                int(gui._spectral_side.winfo_height()),
            ],
            "graph_host": [
                int(gui._spectral_graph_host.winfo_width()),
                int(gui._spectral_graph_host.winfo_height()),
            ],
            "figure_px": [
                int(round(gui.fig_spectral.get_size_inches()[0] * gui.fig_spectral.get_dpi())),
                int(round(gui.fig_spectral.get_size_inches()[1] * gui.fig_spectral.get_dpi())),
            ],
            "scrollregion": str(gui._spectral_scroll_canvas.cget("scrollregion")),
            "expand_buttons": len(gui._spectral_expand_bar.winfo_children()),
            "tree_mapped": bool(gui._spectral_tree.winfo_ismapped()),
        }
        fig_path = OUT / f"spectral_{width}x{height}.png"
        gui.fig_spectral.savefig(fig_path, dpi=100, facecolor=gui.fig_spectral.get_facecolor())
        row["figure_png"] = str(fig_path)
        if ImageGrab is not None:
            rx, ry = int(root.winfo_rootx()), int(root.winfo_rooty())
            rw, rh = int(root.winfo_width()), int(root.winfo_height())
            grab = ImageGrab.grab(bbox=(rx, ry, rx + rw, ry + rh))
            ui_path = OUT / f"spectral_ui_{width}x{height}.png"
            grab.save(ui_path)
            row["ui_png"] = str(ui_path)
            row["grab_box"] = [rx, ry, rw, rh]
        payload["sizes"][f"{width}x{height}"] = row
        print(json.dumps({f"{width}x{height}": row}, indent=2), flush=True)

    open_spectral_plot_window(gui, "acf")
    root.update()
    payload["expand_acf_title"] = str(gui._spectral_expand_by_id["acf"]["ax"].get_title())
    (OUT / "measurements.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    root.destroy()
    print(f"dpi_scale={scale:.3f} wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
