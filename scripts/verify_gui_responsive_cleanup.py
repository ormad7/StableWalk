"""Live Tk validation for the targeted responsive GUI cleanup.

Loads real demo poses, visits every major tab, measures overflow, opens the
3D Coordinate Inspector, and compares two different videos on Research dataset.
Does not recompute scientific formulas.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util
import tkinter as tk

_shots_path = _ROOT / "scripts" / "gui_layout_screenshots.py"
_shots_spec = importlib.util.spec_from_file_location("gui_layout_screenshots", _shots_path)
_shots = importlib.util.module_from_spec(_shots_spec)
assert _shots_spec.loader is not None
_shots_spec.loader.exec_module(_shots)
_find_notebook = _shots._find_notebook
_select_inner = _shots._select_inner
_set_geometry = _shots._set_geometry
shot = _shots.shot

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
from stablewalk.ui.tk.layout_scale import apply_process_dpi_awareness, dpi_scale_factor

OUT = _ROOT / "docs" / "gui_layout_screenshots"
REPORT = _ROOT / "docs" / "gui_responsive_cleanup_results.json"

_ALL_SIZES = (
    ("1366", "1366x768+20+20"),
    ("1600", "1600x900+20+20"),
    ("1920", "1920x1080+20+20"),
    ("2560", "2560x1440+20+20"),
)
_only = str(__import__("os").environ.get("SW_GUI_SIZES", "") or "").strip()
if _only:
    wanted = {item.strip() for item in _only.split(",") if item.strip()}
    SIZES = tuple(item for item in _ALL_SIZES if item[0] in wanted)
else:
    SIZES = _ALL_SIZES

MOTION_SUBTABS = (
    "Kinematics",
    "Gait Correlation & Spectral Analysis",
    "Spectral Analysis",
    "Spectral Features",
)
BIOMECH_SUBTABS = ("COM", "Ground Reaction Force", "Balance", "Contact", "Gait Phase")

PROBE_ATTRS = (
    ("preset_combo", "header Source combo"),
    ("cmb_motion_compare_add", "Compare Joints combo"),
    ("_motion_stats_panel", "Live data panel"),
    ("_motion_joint_row", "Compare Joints row"),
    ("lbl_results_primary_warnings", "Results warnings"),
    ("_biomech_clinical_host", "Biomechanics interpretation"),
    ("_research_delta_tree", "Compare Delta table"),
    ("cmb_research_a", "Compare Video A combo"),
    ("cmb_research_b", "Compare Video B combo"),
    ("btn_open_coord_inspector", "Coordinate Inspector button"),
)


def _pump(root: tk.Misc, seconds: float = 0.2) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            root.update()
        except tk.TclError:
            return
        time.sleep(0.02)


def _overflow(widget: tk.Misc, *, pad: int = 6) -> dict[str, object] | None:
    try:
        if not widget.winfo_exists() or not widget.winfo_viewable():
            return None
        mapped_w = int(widget.winfo_width() or 0)
        mapped_h = int(widget.winfo_height() or 0)
        req_w = int(widget.winfo_reqwidth() or 0)
        req_h = int(widget.winfo_reqheight() or 0)
    except tk.TclError:
        return None
    if mapped_w < 8 or mapped_h < 8:
        return None
    clip_w = req_w > mapped_w + pad
    clip_h = req_h > mapped_h + pad
    if not clip_w and not clip_h:
        return None
    return {
        "mapped": f"{mapped_w}x{mapped_h}",
        "req": f"{req_w}x{req_h}",
        "clip_w": clip_w,
        "clip_h": clip_h,
    }


def _probe(gui, prefix: str, tab: str) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    for attr, label in PROBE_ATTRS:
        widget = getattr(gui, attr, None)
        if widget is None:
            continue
        info = _overflow(widget)
        if info is None:
            continue
        hits.append({"size": prefix, "tab": tab, "widget": label, **info})
    return hits


def _load_two_research_videos(gui) -> dict[str, object]:
    from stablewalk.io.pose_loader import load_pose_sequence
    from stablewalk.research.dataset import ClipMetadata
    from stablewalk.research.pipeline import (
        analyze_independent_clip,
        slots_are_independent,
        slots_share_source_file,
    )
    from stablewalk.ui.tk.dashboard_research import run_comparison

    poses_a = _ROOT / "data" / "output" / "poses" / "normal_gait_poses.json"
    poses_b = _ROOT / "data" / "output" / "poses" / "abnormal_gait_poses.json"
    video_a = _ROOT / "data" / "demo_videos" / "normal_gait.mp4"
    video_b = _ROOT / "data" / "demo_videos" / "abnormal_gait.mp4"
    out: dict[str, object] = {
        "poses_a": poses_a.is_file(),
        "poses_b": poses_b.is_file(),
        "video_a": video_a.is_file(),
        "video_b": video_b.is_file(),
        "independent": False,
        "same_file": None,
        "delta_rows": 0,
        "error": None,
    }
    if not poses_a.is_file() or not poses_b.is_file():
        out["error"] = "missing pose JSON"
        return out
    try:
        seq_a = load_pose_sequence(poses_a)
        seq_b = load_pose_sequence(poses_b)
        meta_a = ClipMetadata(
            video_id="normal_gait",
            gait_category="normal_gait",
            local_path=str(video_a) if video_a.is_file() else str(poses_a),
        )
        meta_b = ClipMetadata(
            video_id="abnormal_gait",
            gait_category="limping_gait",
            local_path=str(video_b) if video_b.is_file() else str(poses_b),
        )
        slot_a = analyze_independent_clip(
            sequence=seq_a,
            metadata=meta_a,
            video_id="normal_gait",
            label="normal_gait",
            run_pose=False,
        )
        slot_b = analyze_independent_clip(
            sequence=seq_b,
            metadata=meta_b,
            video_id="abnormal_gait",
            label="abnormal_gait",
            run_pose=False,
        )
        gui._research_slot_a = slot_a
        gui._research_slot_b = slot_b
        try:
            from stablewalk.ui.tk.dashboard_research import _set_slot_status

            _set_slot_status(gui, "A", f"ready · {slot_a.n_frames} frames")
            _set_slot_status(gui, "B", f"ready · {slot_b.n_frames} frames")
        except Exception:
            pass
        out["independent"] = bool(slots_are_independent(slot_a, slot_b))
        out["same_file"] = bool(slots_share_source_file(slot_a, slot_b))
        out["n_frames_a"] = int(slot_a.n_frames)
        out["n_frames_b"] = int(slot_b.n_frames)
        run_comparison(gui)
        tree = getattr(gui, "_research_delta_tree", None)
        if tree is not None:
            out["delta_rows"] = len(tree.get_children())
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    errs: list[str] = []

    def hook(exc, val, tb) -> None:
        errs.append("".join(traceback.format_exception(exc, val, tb)))

    tk.Tk.report_callback_exception = staticmethod(hook)
    apply_process_dpi_awareness()
    root = tk.Tk()
    root.geometry("1366x768+20+20")
    app = StableWalkGUI(root=root)
    root.update()
    root.deiconify()
    try:
        root.attributes("-topmost", True)
        root.lift()
    except tk.TclError:
        pass
    _pump(root, 0.3)

    scale = dpi_scale_factor(root)
    screen = (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
    report: dict[str, object] = {
        "screen": list(screen),
        "dpi_scale": round(float(scale), 3),
        "geometries": {},
        "overflow": [],
        "inspector": {},
        "compare_two_videos": {},
        "callback_errors": [],
    }
    print(f"screen={screen} dpi_scale={scale:.3f}", flush=True)

    poses = _ROOT / "data" / "output" / "poses" / "normal_gait_poses.json"
    loaded = False
    if poses.is_file():
        try:
            try:
                loaded = bool(app.load_poses(poses, fresh=True, wait_for_analysis=True))
            except TypeError:
                loaded = bool(app.load_poses(poses, fresh=True))
        except Exception as exc:
            print(f"load_poses failed: {exc}", flush=True)
        _pump(root, 0.8)
    report["poses_loaded"] = loaded

    overflow: list[dict[str, object]] = []
    geometries: dict[str, object] = {}
    report["geometries"] = geometries

    for prefix, spec in SIZES:
        actual_w, actual_h = _set_geometry(root, spec)
        geometries[prefix] = {
            "requested": spec.split("+", 1)[0],
            "actual": f"{actual_w}x{actual_h}",
            "clamped": (actual_w, actual_h) != tuple(int(x) for x in spec.split("+", 1)[0].split("x")),
        }
        print(f"geometry {spec} actual={actual_w}x{actual_h}", flush=True)

        for tab, name in (
            (TAB_OVERVIEW, "overview"),
            (TAB_MOTION, "motion"),
            (TAB_BIOMECHANICS, "biomech"),
            (TAB_RESULTS_SUMMARY, "results"),
            (TAB_ADVANCED, "advanced"),
        ):
            try:
                select_dashboard_tab(app, tab)
                wait = 0.45 if name in {"biomech", "motion"} else 0.18
                _pump(root, wait)
                shot(root, f"cleanup_{prefix}_{name}.png")
                overflow.extend(_probe(app, prefix, name))
            except Exception as exc:
                print(f"{prefix} {name} failed: {exc}", flush=True)
                overflow.append(
                    {"size": prefix, "tab": name, "widget": "TAB", "error": str(exc)}
                )

        if prefix == "1366":
            select_dashboard_tab(app, TAB_MOTION)
            inner = _find_notebook(
                getattr(app, "_tab_motion"),
                ("Spectral Analysis", "Gait Correlation & Spectral Analysis", "Kinematics"),
            )
            for title in MOTION_SUBTABS:
                _select_inner(inner, title)
                _pump(root, 0.12)
                shot(root, f"cleanup_1366_motion_{title.replace(' ', '_')}.png")
                overflow.extend(_probe(app, prefix, f"motion:{title}"))
            select_dashboard_tab(app, TAB_BIOMECHANICS)
            bnb = _find_notebook(
                getattr(app, "_tab_biomechanics"),
                ("COM", "Balance", "Contact"),
            )
            for title in BIOMECH_SUBTABS:
                _select_inner(bnb, title)
                _pump(root, 0.12)
                shot(root, f"cleanup_1366_biomech_{title.replace(' ', '_')}.png")
                overflow.extend(_probe(app, prefix, f"biomech:{title}"))

        try:
            app._skip_next_notebook_tab_changed = True
            app._active_dashboard_tab_id = TAB_COMPARE
            nb = getattr(app, "_dashboard_notebook", None)
            page = getattr(app, "_tab_compare", None)
            if nb is not None and page is not None:
                nb.select(page)
            _pump(root, 0.2)
            shot(root, f"cleanup_{prefix}_compare.png")
            overflow.extend(_probe(app, prefix, "compare"))
        except Exception as exc:
            print(f"{prefix} compare failed: {exc}", flush=True)

    select_dashboard_tab(app, TAB_ADVANCED)
    _pump(root, 0.2)
    inspector = {"opened": False, "exists": False, "frame": None, "error": None}
    try:
        app._open_coordinate_inspector_window()
        _pump(root, 0.5)
        win = getattr(app, "_coord_inspector_window", None)
        inspector["opened"] = True
        inspector["exists"] = bool(win is not None and win.winfo_exists())
        vals = getattr(app, "_coord_inspector_window_values", None) or {}
        frame_lbl = vals.get("frame")
        if frame_lbl is not None:
            inspector["frame"] = str(frame_lbl.cget("text") or "")
        if inspector["exists"]:
            shot(win, "cleanup_inspector.png")
            n = int(getattr(getattr(app, "skeleton_player", None), "frame_count", 0) or 0)
            if n > 2:
                target = min(n - 1, 40)
                app._go_to(target)
                _pump(root, 0.6)
                inspector["frame_after_seek"] = (
                    str(frame_lbl.cget("text") or "") if frame_lbl else None
                )
                slider_lbl = getattr(app, "_coord_inspector_window_frame_label", None)
                if slider_lbl is not None:
                    inspector["slider_after_seek"] = str(slider_lbl.cget("text") or "")
    except Exception as exc:
        inspector["error"] = f"{type(exc).__name__}: {exc}"
        print(f"inspector failed: {exc}", flush=True)
    report["inspector"] = inspector

    results_check: dict[str, object] = {"matched": False, "error": None}
    try:
        select_dashboard_tab(app, TAB_RESULTS_SUMMARY)
        _pump(root, 0.2)
        summary = getattr(app, "_analysis_summary_cache", None)
        cards = getattr(app, "_results_kpi_cards", None) or {}
        overview = getattr(app, "_overview_clinical_value_labels", None) or {}
        gq_ov = overview.get("gait_quality")
        gq_ov_text = str(gq_ov.cget("text") or "") if gq_ov is not None else ""
        cache_gq = ""
        if summary is not None:
            field = getattr(summary, "gait_quality", None)
            cache_gq = str(getattr(field, "value", "") or "")
        results_check = {
            "has_cache": summary is not None,
            "overview_gait_quality": gq_ov_text,
            "cache_gait_quality": cache_gq,
            "results_cards": sorted(str(k) for k in cards),
            "matched": bool(summary is not None and (not cache_gq or cache_gq.split()[0] in gq_ov_text or gq_ov_text in cache_gq or gq_ov_text in ("—", "-", ""))),
        }
    except Exception as exc:
        results_check["error"] = f"{type(exc).__name__}: {exc}"
    report["results_consistency"] = results_check

    advanced_check = {
        "save_session": callable(getattr(app, "_save_session_to_files", None)),
        "open_inspector": callable(getattr(app, "_open_coordinate_inspector_window", None)),
        "export_opensim": callable(getattr(app, "_export_opensim", None)),
        "real_to_sim": callable(getattr(app, "_run_real_to_sim_pipeline", None)),
        "inspector_opened": bool(inspector.get("exists")),
    }
    report["advanced_tools"] = advanced_check

    abnormal = _ROOT / "data" / "output" / "poses" / "abnormal_gait_poses.json"
    if abnormal.is_file():
        try:
            try:
                app.load_poses(abnormal, fresh=True, wait_for_analysis=True)
            except TypeError:
                app.load_poses(abnormal, fresh=True)
            _pump(root, 0.8)
            select_dashboard_tab(app, TAB_OVERVIEW)
            _pump(root, 0.25)
            shot(root, "cleanup_abnormal_overview.png")
            select_dashboard_tab(app, TAB_MOTION)
            _pump(root, 0.25)
            shot(root, "cleanup_abnormal_motion.png")
            report["abnormal_loaded"] = True
        except Exception as exc:
            report["abnormal_loaded"] = False
            report["abnormal_error"] = f"{type(exc).__name__}: {exc}"

    select_dashboard_tab(app, TAB_COMPARE)
    _pump(root, 0.25)
    report["compare_two_videos"] = _load_two_research_videos(app)
    _pump(root, 0.4)
    shot(root, "cleanup_compare_two_videos.png")

    report["overflow"] = overflow
    report["callback_errors"] = errs[:8]
    report["callback_error_count"] = len(errs)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:4000], flush=True)
    print(f"wrote {REPORT}", flush=True)
    try:
        root.destroy()
    except tk.TclError:
        pass
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
