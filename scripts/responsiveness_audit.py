"""Live GUI responsiveness audit at supported desktop sizes.

Presentation only — no analysis. Prints issues for Overview, Motion,
Biomechanics, Results, and Advanced.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tkinter as tk
from typing import Any

# Live Tk sizes. 2560×1440 / 3840×2160 are classified in unit tests; opening
# them here overflows a laptop display and is not a reliable GUI check.
SIZES = (
    (1920, 1080),
    (1600, 900),
    (1366, 768),
    (1280, 720),
)

TABS = (
    ("Overview", "Overview"),
    ("Motion Analysis", "Motion"),
    ("Biomechanics", "Biomechanics"),
    ("Results", "Results"),
    ("Compare", "Compare"),
    ("Advanced", "Advanced"),
)


def _box(widget: Any) -> tuple[int, int, int, int] | None:
    try:
        widget.update_idletasks()
        x, y = int(widget.winfo_rootx()), int(widget.winfo_rooty())
        w, h = int(widget.winfo_width()), int(widget.winfo_height())
        if w <= 1 or h <= 1:
            return None
        return (x, y, x + w, y + h)
    except Exception:
        return None


def _size(widget: Any) -> tuple[int, int]:
    try:
        return int(widget.winfo_width() or 0), int(widget.winfo_height() or 0)
    except Exception:
        return 0, 0


def _mapped(widget: Any) -> bool:
    if widget is None:
        return False
    try:
        return bool(widget.winfo_ismapped()) and int(widget.winfo_width() or 0) > 1
    except Exception:
        return False


def _chrome_present(widget: Any) -> bool:
    """Global bars stay packed even while a notebook page is remapping."""
    if widget is None:
        return False
    try:
        if bool(widget.winfo_ismapped()) and int(widget.winfo_width() or 0) > 1:
            return True
        return str(widget.winfo_manager() or "") in {"pack", "grid", "place"}
    except Exception:
        return False


def _outside_root(widget: Any, root: tk.Misc, *, slop: int = 8) -> str | None:
    rb = _box(root)
    wb = _box(widget)
    if not rb or not wb:
        return None
    parts = []
    if wb[2] > rb[2] + slop:
        parts.append(f"right+{wb[2] - rb[2]}")
    if wb[3] > rb[3] + slop:
        parts.append(f"bottom+{wb[3] - rb[3]}")
    if wb[0] < rb[0] - slop:
        parts.append(f"left-{rb[0] - wb[0]}")
    if wb[1] < rb[1] - slop:
        parts.append(f"top-{rb[1] - wb[1]}")
    return ",".join(parts) if parts else None


def _overflows(widget: Any, parent: Any, *, slop: int = 6) -> bool:
    wb = _box(widget)
    pb = _box(parent)
    if not wb or not pb:
        return False
    return wb[2] > pb[2] + slop or wb[3] > pb[3] + slop


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return (a[0] + 2) < b[2] and (b[0] + 2) < a[2] and (a[1] + 2) < b[3] and (b[1] + 2) < a[3]


def audit_tab(app: Any, *, width: int, height: int, tab: str) -> list[str]:
    issues: list[str] = []
    prefix = f"{width}x{height} {tab}"
    root = app.root

    transport = getattr(app, "_transport_bar", None) or getattr(app, "_transport_row", None)
    if not _chrome_present(transport):
        issues.append(f"{prefix}: playback bar not mapped")
    else:
        clip = _outside_root(transport, root)
        if clip:
            issues.append(f"{prefix}: playback clipped ({clip})")
        play = getattr(app, "btn_play", None)
        slider = getattr(app, "slider", None)
        if play is not None and not _chrome_present(play) and not _mapped(play):
            issues.append(f"{prefix}: Play hidden")
        if slider is not None and not _chrome_present(slider) and not _mapped(slider):
            issues.append(f"{prefix}: timeline hidden")

    source = getattr(app, "_input_bar", None)
    if source is not None:
        if not _chrome_present(source):
            issues.append(f"{prefix}: source bar not mapped")
        else:
            clip = _outside_root(source, root)
            if clip:
                issues.append(f"{prefix}: source bar clipped ({clip})")
            w, h = _size(source)
            if h < 20 and _mapped(source):
                issues.append(f"{prefix}: source bar too short ({h}px)")

    notebook = getattr(app, "_dashboard_notebook", None)
    if notebook is not None and _mapped(notebook):
        clip = _outside_root(notebook, root, slop=12)
        if clip:
            issues.append(f"{prefix}: notebook clipped ({clip})")
        try:
            texts = [str(notebook.tab(tid, "text")) for tid in notebook.tabs()]
        except tk.TclError:
            texts = []
        expected = {"Overview", "Motion Analysis", "Biomechanics", "Results", "Advanced"}
        if "Compare" in texts:
            expected.add("Compare")
        missing = expected - set(texts)
        if missing:
            issues.append(f"{prefix}: missing notebook tabs {sorted(missing)}")

    # Horizontal overflow of the selected notebook page.
    tab_widget = {
        "Overview": getattr(app, "_tab_overview", None),
        "Motion": getattr(app, "_tab_motion", None),
        "Biomechanics": getattr(app, "_tab_biomechanics", None),
        "Results": getattr(app, "_tab_results_summary", None),
        "Compare": getattr(app, "_tab_compare", None),
        "Advanced": getattr(app, "_tab_advanced_outer", None),
    }.get(tab)
    if tab_widget is not None and _mapped(tab_widget):
        tw, th = _size(tab_widget)
        if tw > width + 24:
            issues.append(f"{prefix}: tab wider than window ({tw}>{width})")
        try:
            req_w = int(tab_widget.winfo_reqwidth() or 0)
        except tk.TclError:
            req_w = 0
        if req_w > width + 48 and tw > width + 8:
            issues.append(f"{prefix}: tab reqwidth overflow ({req_w}>{width})")

    if tab == "Overview":
        for name, widget, min_w, min_h in (
            ("Video", getattr(app, "video_frame", None), 140, 120),
            ("Skeleton", getattr(app, "skel_frame", None), 140, 120),
            ("3D Path", getattr(app, "overview_traj_panel", None), 140, 120),
        ):
            if not _mapped(widget):
                issues.append(f"{prefix}: {name} hidden")
                continue
            w, h = _size(widget)
            if w < min_w or h < min_h:
                issues.append(f"{prefix}: {name} unusable ({w}x{h})")
            clip = _outside_root(widget, root, slop=10)
            if clip:
                issues.append(f"{prefix}: {name} clipped ({clip})")
        kpis = getattr(app, "_overview_metrics_row", None)
        if kpis is not None and _mapped(kpis):
            kw, kh = _size(kpis)
            if kh < 28:
                issues.append(f"{prefix}: KPI row crushed ({kh}px)")

    if tab == "Motion":
        host = getattr(app, "motion_graph_host", None) or getattr(
            app, "_motion_graphs_panel", None
        )
        if not _mapped(host):
            issues.append(f"{prefix}: motion graph host hidden")
        else:
            w, h = _size(host)
            min_h = 140 if height >= 768 else 100
            if w < 200 or h < min_h:
                issues.append(f"{prefix}: motion graph unusable ({w}x{h})")
            clip = _outside_root(host, root, slop=10)
            if clip:
                issues.append(f"{prefix}: motion graph clipped ({clip})")
        canvas = getattr(app, "canvas_motion_workspace", None)
        if canvas is not None:
            try:
                widget = canvas.get_tk_widget()
            except Exception:
                widget = None
            if widget is not None and _mapped(widget):
                w, h = _size(widget)
                if w < 40 or h < 40:
                    issues.append(f"{prefix}: motion canvas disappeared ({w}x{h})")
        joint = getattr(app, "cmb_motion_joint", None) or getattr(
            app, "cmb_selected_joint", None
        )
        if joint is not None and not _mapped(joint):
            issues.append(f"{prefix}: Selected Joint control hidden")

    if tab == "Biomechanics":
        chart = getattr(app, "_biomech_chart_panel", None)
        if not _mapped(chart):
            issues.append(f"{prefix}: biomech chart hidden")
        else:
            w, h = _size(chart)
            min_h = 120 if height >= 768 else 90
            if w < 180 or h < min_h:
                issues.append(f"{prefix}: biomech chart unusable ({w}x{h})")
            clip = _outside_root(chart, root, slop=10)
            if clip:
                issues.append(f"{prefix}: biomech chart clipped ({clip})")
        canvas = getattr(app, "canvas_biomech", None) or getattr(
            app, "biomech_canvas", None
        )
        host = getattr(app, "biomech_chart_host", None)
        if host is not None and _mapped(host):
            w, h = _size(host)
            if w < 40 or h < 40:
                issues.append(f"{prefix}: biomech chart host crushed ({w}x{h})")
        body = getattr(app, "_biomech_body", None)
        sidebar = getattr(app, "_biomech_sidebar", None)
        if body is not None and sidebar is not None and _mapped(body) and _mapped(sidebar):
            _bw, bh = _size(body)
            _sw, sh = _size(sidebar)
            if sh > bh + 24 and not bool(getattr(app, "_biomech_stacked", False)):
                issues.append(f"{prefix}: Key Findings taller than body ({sh}>{bh})")
        findings = getattr(app, "_biomech_sidebar", None)
        clinical = getattr(app, "_biomech_clinical_host", None)
        if chart is not None and findings is not None and _mapped(chart) and _mapped(findings):
            cb, fb = _box(chart), _box(findings)
            if cb and fb and _boxes_overlap(cb, fb):
                issues.append(f"{prefix}: Key Findings overlaps graph")
        if chart is not None and clinical is not None and _mapped(chart) and _mapped(clinical):
            cb, ib = _box(chart), _box(clinical)
            if cb and ib and _boxes_overlap(cb, ib):
                issues.append(f"{prefix}: Interpretation overlaps graph")

    if tab == "Results":
        primary = getattr(app, "_results_primary_host", None)
        if primary is not None and not _mapped(primary):
            issues.append(f"{prefix}: Results primary findings hidden")
        exports = (
            getattr(app, "btn_export_summary_report", None),
            getattr(app, "btn_export_summary_json", None),
        )
        for btn in exports:
            if btn is not None and not _mapped(btn):
                issues.append(f"{prefix}: Results export button hidden ({btn})")
        preview = getattr(app, "_results_preview_host", None)
        if preview is not None and _mapped(preview):
            clip = _outside_root(preview, root, slop=10)
            if clip:
                issues.append(f"{prefix}: Report Preview clipped ({clip})")
        pipeline = getattr(app, "_results_pipeline_host", None) or getattr(
            app, "_results_pipeline_slot", None
        )
        tab_page = getattr(app, "_tab_results_summary", None)
        if pipeline is not None and tab_page is not None and _mapped(pipeline):
            try:
                req_h = int(pipeline.winfo_reqheight() or 0)
            except tk.TclError:
                req_h = 0
            _tw, th = _size(tab_page)
            if th > 80 and req_h > th + 80:
                issues.append(
                    f"{prefix}: Processing Details taller than Results tab ({req_h}>{th})"
                )

    if tab == "Compare":
        ctrl = getattr(app, "_comparison_mode", None)
        workspace = getattr(ctrl, "_compare_workspace", None) if ctrl else None
        if workspace is None or not _mapped(workspace):
            issues.append(f"{prefix}: Compare workspace hidden")
        else:
            clip = _outside_root(workspace, root, slop=16)
            if clip:
                issues.append(f"{prefix}: Compare workspace clipped ({clip})")

    if tab == "Advanced":
        outer = getattr(app, "_tab_advanced_outer", None)
        if outer is None or not _mapped(outer):
            issues.append(f"{prefix}: Advanced page hidden")
        else:
            clip = _outside_root(outer, root, slop=16)
            if clip:
                issues.append(f"{prefix}: Advanced page clipped ({clip})")
            ow, _oh = _size(outer)
            if ow > width + 24:
                issues.append(f"{prefix}: Advanced page wider than window ({ow}>{width})")
        workspace = getattr(app, "_advanced_workspace", None)
        if workspace is not None:
            ww, _wh = _size(workspace)
            if ww > width + 24:
                issues.append(f"{prefix}: Advanced inner wider than window ({ww}>{width})")
        save = getattr(app, "btn_save_session", None)
        if save is not None and not _mapped(save):
            issues.append(f"{prefix}: Save Session hidden")
        tree = getattr(app, "generated_files_tree", None)
        section = getattr(app, "_section_generated_files", None)
        host = getattr(app, "_generated_files_tree_host", None) or section
        if tree is not None and host is not None and _mapped(tree):
            tw, _th = _size(tree)
            hw, _hh = _size(host)
            if hw > 40 and tw > hw + 24:
                issues.append(
                    f"{prefix}: Generated Files tree wider than card ({tw}>{hw})"
                )

    return issues


def main() -> int:
    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.dashboard_notebook import (
        ENABLE_COMPARE_TAB,
        select_dashboard_tab,
    )
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout

    tabs = tuple(
        pair
        for pair in TABS
        if ENABLE_COMPARE_TAB or pair[0] != "Compare"
    )

    root = tk.Tk()
    app = StableWalkGUI(root=root)
    root.update_idletasks()

    all_issues: list[str] = []
    print("RESPONSIVENESS AUDIT")
    print("=" * 72)
    try:
        for width, height in SIZES:
            root.geometry(f"{width}x{height}+40+40")
            root.update_idletasks()
            root.update()
            apply_responsive_layout(app, width=width, height=height)
            root.update_idletasks()
            root.update()
            print(f"\n--- {width}x{height} ---")
            for tab_id, tab_name in tabs:
                select_dashboard_tab(app, tab_id)
                root.update_idletasks()
                root.update()
                apply_responsive_layout(app, width=width, height=height)
                root.update_idletasks()
                root.update()
                issues = audit_tab(app, width=width, height=height, tab=tab_name)
                all_issues.extend(issues)
                if issues:
                    for line in issues:
                        print(f"  FAIL  {line}")
                else:
                    print(f"  ok    {tab_name}")
    except tk.TclError as exc:
        all_issues.append(f"audit aborted: {exc}")
        print(f"  FAIL  audit aborted: {exc}")
    finally:
        try:
            info = str(root.tk.call("after", "info") or "")
            for after_id in info.split():
                try:
                    root.after_cancel(after_id)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass

    print("\n" + "=" * 72)
    if all_issues:
        print(f"{len(all_issues)} issue(s)")
        return 1
    print("No layout issues found at the four supported sizes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
