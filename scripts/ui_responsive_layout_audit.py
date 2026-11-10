"""Live resize audit for the responsive layout pass.

Opens the real Tk GUI, resizes to supported desktops that fit this screen,
visits every main tab, and writes measurements (+ screenshots when Pillow
is available). Does not recompute analysis.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tkinter as tk

from stablewalk.ui.tk.app import StableWalkGUI
from stablewalk.ui.tk.dashboard_notebook import (
    TAB_ADVANCED,
    TAB_BIOMECHANICS,
    TAB_COMPARE,
    TAB_MOTION,
    TAB_OVERVIEW,
    TAB_RESULTS_SUMMARY,
    select_dashboard_tab,
)
from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout, classify_layout
from stablewalk.ui.tk.layout_scale import SCALE_TARGETS, dpi_scale_factor

MOTION_INNER = (
    "Kinematics",
    "Gait Correlation & Spectral Analysis",
    "Spectral Analysis",
    "Spectral Features",
)
BIOMECH_INNER = (
    "COM",
    "Ground Reaction Force",
    "Balance",
    "Contact",
    "Gait Phase",
)

SIZES = (
    (1366, 768),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
)
TABS = (
    TAB_OVERVIEW,
    TAB_MOTION,
    TAB_BIOMECHANICS,
    TAB_RESULTS_SUMMARY,
    TAB_COMPARE,
    TAB_ADVANCED,
)
SHOT_DIR = _ROOT / "docs" / "ui_responsive_screenshots"
OUT_JSON = _ROOT / "docs" / "_ui_responsive_audit.json"


def _tk_widget(widget):
    if widget is None:
        return None
    if hasattr(widget, "winfo_width"):
        return widget
    frame = getattr(widget, "frame", None)
    if hasattr(frame, "winfo_width"):
        return frame
    return None


def _size(widget) -> tuple[int, int]:
    widget = _tk_widget(widget)
    if widget is None:
        return 0, 0
    try:
        return int(widget.winfo_width() or 0), int(widget.winfo_height() or 0)
    except (tk.TclError, AttributeError):
        return 0, 0


def _mapped(widget) -> bool:
    widget = _tk_widget(widget)
    if widget is None:
        return False
    try:
        return bool(widget.winfo_ismapped()) and int(widget.winfo_width() or 0) > 1
    except (tk.TclError, AttributeError):
        return False


def _grab(root: tk.Misc, path: Path) -> bool:
    try:
        from PIL import ImageGrab
    except Exception:
        return False
    try:
        try:
            root.attributes("-topmost", True)
            root.deiconify()
            root.lift()
            root.focus_force()
        except tk.TclError:
            pass
        root.update_idletasks()
        root.update()
        time.sleep(0.15)
        x = int(root.winfo_rootx())
        y = int(root.winfo_rooty())
        w = int(root.winfo_width())
        h = int(root.winfo_height())
        if w < 80 or h < 80:
            return False
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return True
    except Exception:
        return False


def _select_inner(notebook, title: str) -> bool:
    if notebook is None:
        return False
    try:
        for tab_id in notebook.tabs():
            if str(notebook.tab(tab_id, "text")) == title:
                notebook.select(tab_id)
                return True
    except tk.TclError:
        return False
    return False


def _win_dpi() -> dict[str, object]:
    info: dict[str, object] = {}
    try:
        import ctypes

        info["GetDpiForSystem"] = int(ctypes.windll.user32.GetDpiForSystem())
        info["windows_scale_pct"] = round(int(info["GetDpiForSystem"]) / 96.0 * 100)
    except Exception:
        pass
    return info


def _measure(app, tab: str) -> dict[str, object]:
    row: dict[str, object] = {"tab": tab}
    if tab == TAB_OVERVIEW:
        for name, widget in (
            ("video", getattr(app, "video_frame", None)),
            ("skeleton", getattr(app, "skel_frame", None)),
            ("path", getattr(app, "overview_traj_panel", None)),
        ):
            w, h = _size(widget)
            row[name] = {"w": w, "h": h, "mapped": _mapped(widget)}
        row["viz_layout"] = str(getattr(app, "_overview_viz_layout", ""))
    elif tab == TAB_MOTION:
        gh = getattr(app, "motion_graph_host", None)
        live = getattr(app, "_motion_stats_panel", None)
        graphs = getattr(app, "_motion_graphs_panel", None)
        gw, gh_h = _size(gh)
        lw, lh = _size(live)
        row["graph_host"] = {"w": gw, "h": gh_h, "mapped": _mapped(gh)}
        row["live_panel"] = {"w": lw, "h": lh, "mapped": _mapped(live)}
        if graphs is not None:
            from stablewalk.ui.tk.dashboard_motion_workspace import (
                _MOTION_GRAPH_CANVAS_ROW,
            )

            row["graphs_canvas_row_weight"] = int(
                graphs.grid_rowconfigure(_MOTION_GRAPH_CANVAS_ROW)["weight"] or 0
            )
        combo = getattr(app, "cmb_motion_joint", None)
        if combo is not None:
            row["joint_combo_width"] = int(combo.cget("width") or 0)
            row["joint_combo_px"] = _size(combo)[0]
        row["graph_above_live"] = bool(gh_h > 0 and (lh == 0 or gh_h >= lh))
    elif tab == TAB_BIOMECHANICS:
        chart = getattr(app, "_biomech_chart_panel", None)
        findings = getattr(app, "_biomech_findings_section", None) or getattr(
            app, "_biomech_findings_host", None
        )
        clinical = getattr(app, "_biomech_clinical_section", None) or getattr(
            app, "_biomech_clinical_host", None
        )
        cw, ch = _size(chart)
        row["chart"] = {"w": cw, "h": ch, "mapped": _mapped(chart)}
        row["findings"] = {"w": _size(findings)[0], "mapped": _mapped(findings)}
        row["clinical"] = {"w": _size(clinical)[0], "mapped": _mapped(clinical)}
        row["stacked"] = bool(getattr(app, "_biomech_stacked", False))
    elif tab == TAB_RESULTS_SUMMARY:
        preview = getattr(app, "_results_preview_host", None)
        details = getattr(app, "_results_processing_section", None)
        row["preview"] = {"w": _size(preview)[0], "h": _size(preview)[1], "mapped": _mapped(preview)}
        row["processing"] = {"mapped": _mapped(getattr(details, "frame", details))}
    elif tab == TAB_COMPARE:
        workspace = getattr(app, "_compare_workspace", None)
        row["workspace"] = {"w": _size(workspace)[0], "h": _size(workspace)[1], "mapped": _mapped(workspace)}
    elif tab == TAB_ADVANCED:
        ws = getattr(app, "_advanced_workspace", None)
        row["workspace"] = {"w": _size(ws)[0], "h": _size(ws)[1], "mapped": _mapped(ws)}
        row["two_col"] = bool(getattr(app, "_advanced_two_col", False))
        veil = getattr(app, "_tab_motion_veil", None)
        row["veil_placed"] = False
        if veil is not None:
            try:
                row["veil_placed"] = str(veil.winfo_manager() or "") == "place"
            except tk.TclError:
                pass
    return row


def main() -> int:
    root = tk.Tk()
    screen_w = int(root.winfo_screenwidth() or 0)
    screen_h = int(root.winfo_screenheight() or 0)
    scale = dpi_scale_factor(root)
    app = StableWalkGUI(root=root)
    root.update_idletasks()
    shots = 0
    payload: dict[str, object] = {
        "screen": [screen_w, screen_h],
        "dpi_scale": round(scale, 3),
        "windows_dpi": _win_dpi(),
        "tk_fpixels_per_inch": round(float(root.winfo_fpixels("1i")), 2),
        "scale_targets": list(SCALE_TARGETS),
        "sizes": [],
    }
    seen_applied: set[tuple[int, int]] = set()
    try:
        for width, height in SIZES:
            live_w = min(width, max(960, screen_w - 24))
            live_h = min(height, max(600, screen_h - 48))
            fits = width <= screen_w + 8 and height <= screen_h + 8
            wmode, hmode = classify_layout(live_w, live_h)
            entry: dict[str, object] = {
                "requested": [width, height],
                "applied": [live_w, live_h],
                "fits_screen": fits,
                "width_mode": str(wmode),
                "height_mode": str(hmode),
                "tabs": [],
            }
            applied_key = (live_w, live_h)
            if applied_key in seen_applied:
                entry["note"] = (
                    f"logical classify only — live geometry already audited at {live_w}x{live_h}"
                )
                payload["sizes"].append(entry)
                print(f"{width}x{height}  skip duplicate {live_w}x{live_h}")
                continue
            seen_applied.add(applied_key)
            if not fits:
                entry["note"] = (
                    f"clamped to {live_w}x{live_h} — requested size exceeds this display"
                )
            root.geometry(f"{live_w}x{live_h}+8+8")
            root.update_idletasks()
            root.update()
            apply_responsive_layout(app, width=live_w, height=live_h)
            root.update()
            for tab in TABS:
                select_dashboard_tab(app, tab)
                root.update_idletasks()
                root.update()
                apply_responsive_layout(app, width=live_w, height=live_h)
                root.update()
                try:
                    from stablewalk.ui.tk.ui_motion import dismiss_tab_transition_veil

                    dismiss_tab_transition_veil(app)
                except Exception:
                    pass
                time.sleep(0.22)
                root.update()
                row = _measure(app, tab)
                if tab == TAB_MOTION:
                    inner = []
                    mnb = getattr(app, "_motion_page_notebook", None)
                    for title in MOTION_INNER:
                        _select_inner(mnb, title)
                        root.update_idletasks()
                        root.update()
                        apply_responsive_layout(app, width=live_w, height=live_h)
                        root.update()
                        time.sleep(0.05)
                        item = {"inner": title}
                        if title == "Kinematics":
                            item.update(_measure(app, TAB_MOTION))
                        else:
                            hosts = {
                                "Spectral Analysis": "_spectral_graph_host",
                                "Spectral Features": "_spectral_features_graph_host",
                                "Gait Correlation & Spectral Analysis": "_four_gait_graph_host",
                            }
                            host = getattr(app, hosts.get(title, ""), None)
                            item["graph"] = {
                                "w": _size(host)[0],
                                "h": _size(host)[1],
                                "mapped": _mapped(host),
                            }
                        inner.append(item)
                        shot_inner = SHOT_DIR / (
                            f"{width}x{height}_Motion_{title.replace(' ', '_')}.png"
                        )
                        if _grab(root, shot_inner):
                            item["screenshot"] = str(shot_inner.relative_to(_ROOT))
                            shots += 1
                    row["inner"] = inner
                elif tab == TAB_BIOMECHANICS:
                    inner_b = []
                    bnb = getattr(app, "_biomech_notebook", None)
                    for title in BIOMECH_INNER:
                        ok = _select_inner(bnb, title)
                        root.update_idletasks()
                        root.update()
                        chart = getattr(app, "_biomech_chart_panel", None)
                        inner_b.append(
                            {
                                "inner": title,
                                "selected": ok,
                                "chart": {
                                    "w": _size(chart)[0],
                                    "h": _size(chart)[1],
                                    "mapped": _mapped(chart),
                                },
                            }
                        )
                    row["inner"] = inner_b
                entry["tabs"].append(row)
                shot = SHOT_DIR / f"{width}x{height}_{tab.replace(' ', '_')}.png"
                if _grab(root, shot):
                    row["screenshot"] = str(shot.relative_to(_ROOT))
                    shots += 1
            payload["sizes"].append(entry)
            print(f"{width}x{height}  {wmode.value}/{hmode.value}  tabs={len(entry['tabs'])}")
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    payload["screenshots"] = shots
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT_JSON}  screenshots={shots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
