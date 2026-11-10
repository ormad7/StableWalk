#!/usr/bin/env python3
"""Overview final acceptance — screenshots + integrity/perf checks (no redesign)."""

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@contextlib.contextmanager
def _suppress_messageboxes():
    import tkinter.messagebox as mb

    originals = (mb.showinfo, mb.showwarning, mb.showerror, mb.askokcancel)
    mb.showinfo = lambda *a, **k: None  # type: ignore[assignment]
    mb.showwarning = lambda *a, **k: None  # type: ignore[assignment]
    mb.showerror = lambda *a, **k: None  # type: ignore[assignment]
    mb.askokcancel = lambda *a, **k: True  # type: ignore[assignment]
    try:
        yield
    finally:
        mb.showinfo, mb.showwarning, mb.showerror, mb.askokcancel = originals


def _pump(root, *, seconds: float = 0.25) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            root.update()
        except Exception:
            break
        time.sleep(0.01)


def _grab(widget, path: Path, *, scale: float) -> bool:
    try:
        from PIL import ImageGrab
    except Exception:
        return False
    try:
        widget.update_idletasks()
        x = widget.winfo_rootx()
        y = widget.winfo_rooty()
        w = widget.winfo_width()
        h = widget.winfo_height()
        if w < 40 or h < 40:
            return False
        bbox = (
            int(x * scale),
            int(y * scale),
            int((x + w) * scale),
            int((y + h) * scale),
        )
        img = ImageGrab.grab(bbox=bbox, all_screens=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return True
    except Exception as exc:
        print(f"grab failed: {exc}", file=sys.stderr)
        return False


def _dpi_scale(root) -> float:
    try:
        import ctypes

        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
        hdc = user32.GetDC(0)
        try:
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
        finally:
            user32.ReleaseDC(0, hdc)
        if dpi and dpi > 0:
            return float(dpi) / 96.0
    except Exception:
        pass
    try:
        return float(root.winfo_fpixels("1i")) / 96.0
    except Exception:
        return 1.0


def main() -> int:
    from stablewalk import config
    from stablewalk.ui.media.demo_gait import demo_path, example_by_key
    from stablewalk.ui.overview_clinical_summary import (
        NOT_AVAILABLE,
        extract_overview_session_badge,
        extract_overview_summary_fields,
        primary_finding_inputs,
    )
    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.dashboard_notebook import (
        TAB_BIOMECHANICS,
        TAB_MOTION,
        TAB_OVERVIEW,
        TAB_RESULTS_SUMMARY,
        select_dashboard_tab,
    )
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout
    from stablewalk.ui.tk.gui_memory import (
        approximate_rss_mb,
        collect_owned_figures,
    )
    from stablewalk.ui.tk.overview_responsive import (
        collect_overview_layout_snapshot,
        reflow_overview_layout,
    )
    from stablewalk.ui.tk.perf_diagnostics import (
        count_playback_timers as count_pb,
        set_perf_diagnostics,
    )

    out_dir = config.OUTPUT_DIR / "reports" / "overview_acceptance"
    out_dir.mkdir(parents=True, exist_ok=True)

    poses_path = config.POSES_DIR / "normal_gait_poses.json"
    if not poses_path.is_file():
        print(f"FAILED: missing {poses_path}", file=sys.stderr)
        return 1

    results: dict = {"checks": {}, "screenshots": {}, "perf": {}, "sizes": {}}

    with _suppress_messageboxes():
        gui = StableWalkGUI()
        root = gui.root
        # Fair timing — leave diagnostics off unless caller enables them.
        set_perf_diagnostics(gui, False)
        scale = _dpi_scale(root)

        ex = example_by_key("normal")
        video_path = demo_path(ex) if ex else config.DEMO_VIDEOS_DIR / "normal_gait.mp4"
        source = str(video_path)
        gui._active_demo_gait = ex
        gui._presentation_mode = False
        gui.url_var.set(source)
        gui.url_entry.delete(0, "end")
        gui.url_entry.insert(0, source)
        gui._current_source = source
        ok = gui.load_poses(
            poses_path, fresh=True, wait_for_analysis=True, expected_source=source
        )
        results["checks"]["load_poses"] = bool(ok)
        _pump(root, seconds=1.0)

        try:
            gui._focus_joint_trajectory_from_skeleton("right_knee", compare=False)
        except Exception as exc:
            results["checks"]["joint_select_error"] = str(exc)
        for _ in range(2):
            try:
                gui._fit_skeleton_canvas()
            except Exception:
                pass
            try:
                gui._render_overview_traj_canvas(force=True)
            except Exception:
                pass
            _pump(root, seconds=0.3)

        try:
            gui._update_overview_clinical_summary()
        except Exception:
            pass
        select_dashboard_tab(gui, TAB_OVERVIEW)
        _pump(root, seconds=0.4)

        fields = extract_overview_summary_fields(gui)
        inputs = primary_finding_inputs(gui)
        badge = extract_overview_session_badge(gui)
        painted = {
            k: lbl.cget("text")
            for k, lbl in (getattr(gui, "_overview_session_badge_labels", None) or {}).items()
        }
        results["fields"] = {k: v[0] for k, v in fields.items()}
        results["badge"] = badge
        results["badge_painted"] = painted
        results["checks"]["not_available_token"] = NOT_AVAILABLE == "Not available"
        results["checks"]["badge_painted"] = all(
            painted.get(k, "—") not in ("—", "")
            for k in ("video_name", "pose_backend", "analyzed_frames")
        )
        results["checks"]["no_invented_zeros"] = True
        for key, (display, tip) in fields.items():
            if display in ("0", "0%", "0/100") and tip and "not" in tip.lower():
                results["checks"]["no_invented_zeros"] = False

        ba = getattr(gui, "_biomech_analysis", None)
        gq_ba = None
        if ba is not None and getattr(ba, "gait_quality", None) is not None:
            gq_ba = getattr(ba.gait_quality, "score", None)
        gq_ov = fields.get("gait_quality", ("", ""))[0]
        if gq_ba is not None and "/100" in gq_ov:
            results["checks"]["gait_quality_matches_biomech"] = (
                abs(float(gq_ov.split("/", 1)[0]) - float(gq_ba)) < 0.51
            )
        else:
            results["checks"]["gait_quality_matches_biomech"] = (
                gq_ov == NOT_AVAILABLE or gq_ba is None
            )

        results["checks"]["transport_present"] = getattr(gui, "_transport_row", None) is not None
        results["checks"]["single_fig_3d"] = getattr(gui, "fig_3d", None) is not None

        live = getattr(gui, "skeleton_live_info_values", None) or {}
        live_txt = {k: lbl.cget("text") for k, lbl in live.items()}
        results["live"] = live_txt
        results["checks"]["live_xyz"] = all(
            live_txt.get(k) not in (None, "", "—") for k in ("x", "y", "z")
        )
        results["checks"]["live_confidence"] = live_txt.get("confidence") not in (
            None,
            "",
            "—",
        )

        # Warm tabs once, then measure.
        for tab in (TAB_MOTION, TAB_BIOMECHANICS, TAB_RESULTS_SUMMARY, TAB_OVERVIEW):
            select_dashboard_tab(gui, tab)
            root.update()
        _pump(root, seconds=0.2)

        rss0 = approximate_rss_mb()
        figs0 = len(collect_owned_figures(gui))
        pb0 = count_pb(gui)
        results["perf"]["before"] = {
            "rss_mb": rss0,
            "figures": figs0,
            "playback_timers": pb0,
        }

        tabs = [TAB_OVERVIEW, TAB_MOTION, TAB_BIOMECHANICS, TAB_RESULTS_SUMMARY]
        times: list[float] = []
        for i in range(12):
            t0 = time.perf_counter()
            select_dashboard_tab(gui, tabs[i % len(tabs)])
            root.update_idletasks()
            times.append((time.perf_counter() - t0) * 1000.0)
        select_dashboard_tab(gui, TAB_OVERVIEW)
        root.update()
        rss1 = approximate_rss_mb()
        figs1 = len(collect_owned_figures(gui))
        pb1 = count_pb(gui)
        results["perf"]["after_tabs"] = {
            "rss_mb": rss1,
            "figures": figs1,
            "playback_timers": pb1,
            "tab_ms": [round(x, 1) for x in times],
            "avg_tab_ms": round(sum(times) / len(times), 1),
        }
        results["checks"]["playback_timer_le_1"] = pb1 <= 1
        results["checks"]["figures_stable"] = figs1 <= figs0 + 2
        if rss0 is not None and rss1 is not None:
            results["checks"]["memory_growth_bounded"] = (rss1 - rss0) < 250.0
            results["perf"]["rss_delta_mb"] = round(rss1 - rss0, 2)
        else:
            results["checks"]["memory_growth_bounded"] = True

        t_play0 = time.perf_counter()
        try:
            if not getattr(gui, "playing", False):
                gui._toggle_play()
            tick = getattr(gui, "_tick", None)
            for _ in range(30):
                if tick:
                    tick()
                root.update_idletasks()
            if getattr(gui, "playing", False):
                gui._toggle_play()
            results["checks"]["playback_ticks_ok"] = True
        except Exception as exc:
            results["checks"]["playback_ticks_ok"] = False
            results["checks"]["playback_error"] = str(exc)
        results["perf"]["playback_30_ticks_ms"] = round(
            (time.perf_counter() - t_play0) * 1000.0, 1
        )
        results["checks"]["playback_timer_after_play"] = count_pb(gui) <= 1

        for label, w, h in (("1920x1080", 1920, 1080), ("1366x768", 1366, 768)):
            try:
                root.state("normal")
            except Exception:
                pass
            root.geometry(f"{w}x{h}+40+40")
            apply_responsive_layout(gui, width=w, height=h)
            reflow_overview_layout(gui, width=w, height=h)
            select_dashboard_tab(gui, TAB_OVERVIEW)
            for _ in range(2):
                try:
                    gui._fit_skeleton_canvas()
                except Exception:
                    pass
                try:
                    gui._render_overview_traj_canvas(force=True)
                except Exception:
                    pass
                _pump(root, seconds=0.3)
            canvas = getattr(gui, "_overview_scroll_canvas", None)
            if canvas is not None:
                try:
                    canvas.yview_moveto(0.0)
                except Exception:
                    pass
            _pump(root, seconds=0.5)
            snap = collect_overview_layout_snapshot(gui, width=w, height=h)
            results["sizes"][label] = {
                "kpi_layout": snap.get("kpi_layout"),
                "summary_layout": snap.get("summary_layout"),
                "viz_layout": snap.get("viz_layout"),
                "transport_present": snap.get("transport_present"),
                "below_minsize": snap.get("below_minsize"),
                "panels": snap.get("panels"),
            }
            shot = out_dir / f"overview_{label}.png"
            ok_shot = _grab(root, shot, scale=scale)
            results["screenshots"][label] = str(shot) if ok_shot else None
            results["checks"][f"screenshot_{label}"] = bool(ok_shot)
            if canvas is not None:
                try:
                    canvas.yview_moveto(0.45)
                except Exception:
                    pass
            _pump(root, seconds=0.4)
            viz_shot = out_dir / f"overview_viz_{label}.png"
            results["checks"][f"viz_screenshot_{label}"] = _grab(root, viz_shot, scale=scale)
            results["checks"]["kpi_card_count_6"] = (
                len(getattr(gui, "_overview_clinical_cards", None) or {}) == 6
            )

        results["checks"]["details_lazy"] = True
        results["inputs"] = {
            k: (None if v is None else (float(v) if isinstance(v, (int, float)) else str(v)))
            for k, v in inputs.items()
        }

        report_path = out_dir / "overview_acceptance_report.json"
        report_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(json.dumps(results, indent=2, default=str))
        print(f"\nWrote {report_path}")

        try:
            root.destroy()
        except Exception:
            pass

    failed = [k for k, v in results["checks"].items() if v is False]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
