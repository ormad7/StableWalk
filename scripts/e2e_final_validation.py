"""End-to-end validation: stability scores, UI sizes, user flow."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tkinter as tk

POSES_DIR = ROOT / "data" / "output" / "poses"


def _audit_label_scoring() -> tuple[bool, list[str]]:
    scoring_files = [
        ROOT / "stablewalk" / "analysis" / "biomech_stability.py",
        ROOT / "stablewalk" / "pose" / "gait_step_detection.py",
    ]
    hits: list[str] = []
    patterns = [
        re.compile(r'["\']abnormal["\']', re.I),
        re.compile(r'["\']athletic["\']', re.I),
        re.compile(r'["\']normal["\']', re.I),
        re.compile(r"example\.key", re.I),
        re.compile(r"demo_gait", re.I),
    ]
    for path in scoring_files:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pat in patterns:
                if pat.search(line):
                    hits.append(f"{path.name}:{i}: {stripped[:90]}")
                    break
    return len(hits) == 0, hits


def _score_table() -> dict[str, dict]:
    from scripts.final_validation_report import _analyze_demo
    from stablewalk.io.pose_loader import load_pose_sequence
    from stablewalk.pose.enrichment import enrich_pose_sequence
    from stablewalk.analysis.biomech_stability import analyze_biomech_stability
    from stablewalk.pose.events import analyze_gait_sequence
    from stablewalk.ui.media.demo_gait import example_by_key

    out: dict[str, dict] = {}
    for key in ("abnormal", "normal", "athletic"):
        d = _analyze_demo(key)
        seq = load_pose_sequence(d["poses_path"])
        enrich_pose_sequence(seq)
        result = analyze_biomech_stability(seq)
        mm = {m.key: m for m in result.metrics}
        events, _ = analyze_gait_sequence(seq.frames, fps=seq.fps)
        heel_strikes = sum(1 for e in events if e.event_type == "heel_strike")
        cycles = heel_strikes // 2 if heel_strikes else 0
        vq = d["video_quality"]
        step_m = mm.get("step_consistency")
        regularity = step_m.score if step_m and step_m.score is not None else d["components"].get("step_regularity")
        ex = example_by_key(key)

        out[key] = {
            "stability": result.score,
            "classification": result.classification,
            "symmetry": mm["symmetry"].score,
            "regularity": regularity,
            "rom": mm["range_of_motion"].score,
            "body": mm["body_stability"].score,
            "metrics": mm,
            "result": result,
            "pose_confidence": vq["avg_pose_confidence"],
            "foot_confidence": vq["avg_heel_visibility"],
            "gait_cycles": cycles,
            "steps": d["steps"]["total"],
            "step_confidence": d["steps"]["confidence"],
            "source": ex.source_url if ex else "",
            "display": ex.display_name if ex else key,
        }
    return out


def _explain_normal_vs_athletic(scores: dict[str, dict]) -> str:
    n, a = scores["normal"], scores["athletic"]
    lines = [
        f"Normal {n['stability']:.1f} vs Athletic {a['stability']:.1f} "
        f"(delta {n['stability'] - a['stability']:+.1f})"
    ]
    if a["stability"] >= n["stability"]:
        lines.append("Athletic is NOT lower than Normal.")
        if a["stability"] > n["stability"]:
            contributors = []
            for label, nk, ak in (
                ("Symmetry", "symmetry", "symmetry"),
                ("Regularity", "regularity", "regularity"),
                ("ROM", "rom", "rom"),
                ("Body stability", "body", "body"),
            ):
                nv, av = n.get(nk), a.get(ak)
                if nv is not None and av is not None and av - nv > 0.5:
                    contributors.append((av - nv, label, nv, av))
            contributors.sort(reverse=True)
            if contributors:
                d, label, nv, av = contributors[0]
                lines.append(f"Athletic leads primarily on {label} ({av:.0f} vs {nv:.0f}).")
        return "\n".join(lines)

    contributors = []
    for label, nk, ak in (
        ("Symmetry", "symmetry", "symmetry"),
        ("Regularity", "regularity", "regularity"),
        ("ROM", "rom", "rom"),
        ("Body stability", "body", "body"),
    ):
        nv, av = n.get(nk), a.get(ak)
        if nv is not None and av is not None and nv - av > 0.5:
            contributors.append((nv - av, label, nv, av))

    contributors.sort(reverse=True)
    if contributors:
        d, label, nv, av = contributors[0]
        lines.append(
            f"Primary metric: {label} (Normal {nv:.0f} vs Athletic {av:.0f}, "
            f"weighted gap ~{d * 0.22:.1f} pts on symmetry-weighted path)."
        )
        for extra in contributors[1:3]:
            d2, lbl, nv2, av2 = extra
            lines.append(f"  Also: {lbl} Normal {nv2:.0f} vs Athletic {av2:.0f}.")

    sym_n = n["metrics"]["symmetry"]
    sym_a = a["metrics"]["symmetry"]
    if sym_n.values and sym_a.values:
        lines.append(
            f"Symmetry detail — knee L/R diff: Normal "
            f"{sym_n.values.get('knee_mean_abs_diff_deg', 'N/A')}° vs Athletic "
            f"{sym_a.values.get('knee_mean_abs_diff_deg', 'N/A')}°; "
            f"ankle L/R diff: Normal {sym_n.values.get('ankle_mean_abs_diff_deg', 'N/A')}° vs "
            f"Athletic {sym_a.values.get('ankle_mean_abs_diff_deg', 'N/A')}°."
        )

    lines.append(
        f"Pose quality: Normal {n['pose_confidence']:.3f} vs Athletic {a['pose_confidence']:.3f} "
        f"(lower athletic pose confidence reduces tracking reliability)."
    )
    lines.append(
        f"Steps: Normal {n['steps']} ({n['step_confidence']}) vs "
        f"Athletic {a['steps']} ({a['step_confidence']}); "
        f"gait cycles: Normal {n['gait_cycles']} vs Athletic {a['gait_cycles']}."
    )
    return "\n".join(lines)


def _measure_ui_1600(app) -> dict[str, object]:
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout
    from scripts.final_validation_report import _simulate_analyzed_demo

    w, h = 1600, 900
    app.root.geometry(f"{w}x{h}+-3200+0")
    apply_responsive_layout(app, width=w, height=h)
    _simulate_analyzed_demo(
        app, "athletic", poses_path=POSES_DIR / "validation_athletic_poses.json"
    )
    app._ensure_default_dof_selection()
    app._show_pose_at(0)
    app._fit_skeleton_canvas()
    app._fit_dof_traj_canvas()
    app._render_dof_traj_canvas(force=True)
    app.root.update()

    def wh(widget):
        widget.update_idletasks()
        return widget.winfo_width(), widget.winfo_height()

    def lframe_ancestor(w):
        n = w
        for _ in range(6):
            if n is None:
                break
            if isinstance(n, tk.Widget) and n.winfo_class() == "TLabelframe":
                return n
            n = n.master
        return w

    vp_w, vp_h = wh(lframe_ancestor(app.video_label))
    vl_w, vl_h = wh(app.video_label)
    img_w = img_h = 0
    try:
        img = app.video_label.cget("image")
        if img:
            img_w = int(app.video_label.tk.call("image", "width", img))
            img_h = int(app.video_label.tk.call("image", "height", img))
    except tk.TclError:
        pass

    sk_w, sk_h = wh(app.skel_frame)
    sh_w, sh_h = wh(app.skel_canvas_host)
    sc_w, sc_h = wh(app.canvas_3d.get_tk_widget())

    gh_w, gh_h = wh(app.dof_analysis_graph_canvas_host)
    tw_w, tw_h = wh(app.canvas_dof_traj.get_tk_widget())

    ax_skel = app.ax_3d
    sk_bbox = ax_skel.get_window_extent(app.canvas_3d.renderer).transformed(
        app.fig_3d.dpi_scale_trans.inverted()
    )
    ax_traj = app.ax_dof_traj
    app.canvas_dof_traj.draw()
    tr_bbox = ax_traj.get_window_extent(app.canvas_dof_traj.renderer).transformed(
        app.fig_dof_traj.dpi_scale_trans.inverted()
    )

    video_fill = min(img_w / max(vl_w, 1), img_h / max(vl_h, 1)) if img_w else 0
    skel_fill = min(sc_w / max(sh_w, 1), sc_h / max(sh_h, 1))
    graph_fill = min(tw_w / max(gh_w, 1), tw_h / max(gh_h, 1))

    return {
        "video_panel": (vp_w, vp_h),
        "video_label": (vl_w, vl_h),
        "video_image": (img_w, img_h),
        "video_fill": video_fill,
        "skel_panel": (sk_w, sk_h),
        "skel_host": (sh_w, sh_h),
        "skel_canvas": (sc_w, sc_h),
        "skel_axes_in": (sk_bbox.width, sk_bbox.height),
        "graph_host": (gh_w, gh_h),
        "graph_canvas": (tw_w, tw_h),
        "graph_axes_in": (tr_bbox.width, tr_bbox.height),
        "skel_fill": skel_fill,
        "graph_fill": graph_fill,
    }


def _user_flow(app, key: str) -> list[str]:
    from scripts.final_validation_report import _simulate_analyzed_demo

    failures: list[str] = []
    poses = POSES_DIR / f"validation_{key}_poses.json"
    failures.extend(_simulate_analyzed_demo(app, key, poses_path=poses))
    if failures:
        return failures

    app.root.update()
    app._ensure_default_dof_selection()
    app.root.update()

    if "right_knee" not in app.selection.selected:
        var = app._dof_checkbox_vars.get("right_knee")
        if var:
            var.set(True)
            app._on_dof_checkbox_changed("right_knee")
        else:
            failures.append(f"{key}: right_knee checkbox missing")

    if not app.skeleton_player:
        failures.append(f"{key}: skeleton player not ready")
        return failures

    app._toggle_play()
    app.root.update_idletasks()
    if not app.playing:
        failures.append(f"{key}: play failed")
    app._stop_playback()
    app.root.update_idletasks()
    if app.playing:
        failures.append(f"{key}: stop after play failed")

    btn_adv = getattr(app, "btn_toggle_joint_advanced", None)
    if btn_adv:
        if getattr(app, "_joint_advanced_visible", False):
            app._toggle_joint_advanced_data()
            app.root.update()
        app._toggle_joint_advanced_data()
        app.root.update()
        host = getattr(app, "dof_analysis_advanced_host", None)
        if host is None or not host.winfo_ismapped():
            failures.append(f"{key}: Detailed Joint Data did not expand")

    app._toggle_collected_data_table()
    app.root.update()
    dlg = getattr(app, "_collected_data_dialog", None)
    if dlg is None or not dlg.winfo_exists():
        failures.append(f"{key}: Collected Data modal did not open")
    else:
        app._toggle_collected_data_table()
        app.root.update()

    sections = getattr(app, "_sidebar_sections", ())
    cmp_sec = next((s for s in sections if s[0] == "Gait Comparison"), None)
    if cmp_sec:
        cmp_sec[4]()
        app.root.update()
    else:
        failures.append(f"{key}: Gait Comparison section missing")

    os_sec = next((s for s in sections if s[0] == "OpenSim"), None)
    if os_sec:
        os_sec[4]()
        app._refresh_opensim_status()
        app.root.update()
    else:
        failures.append(f"{key}: OpenSim section missing")

    detail_btn = getattr(app, "btn_stab_steps_details", None)
    if detail_btn and detail_btn.winfo_exists():
        detail_btn.invoke()
        app.root.update()

    app._toggle_play()
    app.root.update()
    if app.playing:
        failures.append(f"{key}: pause failed")

    before = app.current_pos
    app._step(1)
    app.root.update()
    if app.current_pos == before and len(app.pose_indices) > 1:
        failures.append(f"{key}: step forward unchanged frame")

    app._step(-1)
    app.root.update()
    app._reset_playback()
    app.root.update()
    if app.playing:
        failures.append(f"{key}: reset did not stop playback")

    return failures


def _visual_checks(app, width: int, height: int) -> dict[str, bool]:
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout
    from scripts.final_validation_report import _simulate_analyzed_demo, _check_responsive_at

    _simulate_analyzed_demo(
        app, "athletic", poses_path=POSES_DIR / "validation_athletic_poses.json"
    )
    app._ensure_default_dof_selection()
    app._show_pose_at(0)
    app._fit_skeleton_canvas()
    app._fit_dof_traj_canvas()
    app.root.update()
    checks = _check_responsive_at(app, width, height)
    checks.pop("Vertical scrolling works", None)
    checks["No main scroll canvas (fixed layout)"] = getattr(app, "_dash_scroll_canvas", None) is None

    vp = app.video_label.winfo_width(), app.video_label.winfo_height()
    sk = app.skel_canvas_host.winfo_width(), app.skel_canvas_host.winfo_height()
    gh = app.dof_analysis_graph_canvas_host.winfo_width(), app.dof_analysis_graph_canvas_host.winfo_height()

    checks["Video area >= 400x180"] = vp[0] >= 400 and vp[1] >= 180
    checks["Skeleton host >= 400x220"] = sk[0] >= 400 and sk[1] >= 220
    checks["Graph host >= 500x120"] = gh[0] >= 500 and gh[1] >= 120
    checks["Collected Data dialog button"] = getattr(app, "btn_collected_data", None) is not None
    stab_score = getattr(app, "lbl_stab_score", None)
    checks["Stability score visible"] = (
        stab_score is not None and stab_score.winfo_ismapped()
    )
    checks["Walk Summary headline visible"] = (
        stab_score is not None and stab_score.winfo_ismapped()
    )
    reason = getattr(app, "lbl_stab_reason", None)
    checks["Debug reason hidden"] = reason is None or not reason.winfo_ismapped()
    return checks


def main() -> int:
    print("=" * 72)
    print("PART A — SCORE VALIDATION")
    print("=" * 72)

    clean, hits = _audit_label_scoring()
    print(f"Scoring uses demo label: {'NO' if clean else 'YES (VIOLATION)'}")
    if hits:
        for h in hits:
            print(f"  HIT: {h}")

    scores = _score_table()
    print()
    print(f"{'Demo':<12} {'Stability':>10} {'Symmetry':>10} {'Regularity':>12} {'ROM':>8} {'Body':>8}")
    print("-" * 62)
    for key in ("abnormal", "normal", "athletic"):
        s = scores[key]
        reg = s["regularity"]
        reg_s = f"{reg:.0f}" if reg is not None else "N/A"
        print(
            f"{key.capitalize():<12} {s['stability']:>10.1f} {s['symmetry']:>10.0f} "
            f"{reg_s:>12} {s['rom']:>8.0f} {s['body']:>8.0f}"
        )

    print()
    print(_explain_normal_vs_athletic(scores))
    print()
    for key in ("athletic", "normal"):
        s = scores[key]
        label = key.capitalize()
        print(f"{label} video pose confidence: {s['pose_confidence']:.3f}")
        print(f"{label} foot landmark confidence: {s['foot_confidence']:.3f}")
        print(f"{label} usable gait cycles: {s['gait_cycles']} ({s['steps']} steps, {s['step_confidence']} step conf)")
        print()

    print("=" * 72)
    print("PART B — VIDEO VISUAL SIZE @ 1600x900")
    print("=" * 72)

    root = tk.Tk()
    root.withdraw()
    from stablewalk.ui.tk.app import StableWalkGUI

    app = StableWalkGUI(root=root)
    root.deiconify()
    m = _measure_ui_1600(app)

    vp = m["video_panel"]
    vi = m["video_image"]
    print(f"Original Video panel: {vp[0]}x{vp[1]} px")
    print(f"Displayed video: {vi[0]}x{vi[1]} px")
    print(f"Video fill (min dim vs label): {m['video_fill']:.0%}")
    sk = m["skel_panel"]
    sa = m["skel_axes_in"]
    print(f"3D Reconstruction panel: {sk[0]}x{sk[1]} px")
    print(f"Skeleton axes bbox: {sa[0]:.0f}x{sa[1]:.0f} in")
    print(f"Skeleton canvas fill: {m['skel_fill']:.0%}")
    gh = m["graph_host"]
    ga = m["graph_axes_in"]
    print(f"Movement graph host: {gh[0]}x{gh[1]} px")
    print(f"Plot axes bbox: {ga[0]:.0f}x{ga[1]:.0f} in")
    print(f"Graph canvas fill: {m['graph_fill']:.0%}")

    print()
    print("=" * 72)
    print("PART C — VISUAL INSPECTION")
    print("=" * 72)
    for res in ((1920, 1080), (1600, 900), (1366, 768)):
        checks = _visual_checks(app, *res)
        failed = [k for k, v in checks.items() if not v]
        status = "PASS" if not failed else "FAIL"
        print(f"\n{res[0]}x{res[1]}: {status}")
        for k, v in checks.items():
            print(f"  [{'OK' if v else 'XX'}] {k}")
        if failed:
            print("  Failed:", ", ".join(failed))

    print()
    print("=" * 72)
    print("PART D — USER FLOW TEST")
    print("=" * 72)
    all_flow_failures: list[str] = []
    for key in ("athletic", "normal", "abnormal"):
        fails = _user_flow(app, key)
        status = "PASS" if not fails else "FAIL"
        print(f"{key.capitalize()}: {status}")
        for f in fails:
            print(f"  - {f}")
        all_flow_failures.extend(fails)

    root.destroy()

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    issues = []
    if not clean:
        issues.append("Demo label references in scoring code")
    if scores["athletic"]["stability"] < scores["normal"]["stability"]:
        issues.append("Athletic still below Normal (see metric breakdown)")
    if m["video_fill"] < 0.35:
        issues.append(f"Video visually small ({m['video_fill']:.0%} fill)")
    if m["graph_fill"] < 0.55:
        issues.append(f"Graph canvas small ({m['graph_fill']:.0%} fill)")
    if all_flow_failures:
        issues.append(f"{len(all_flow_failures)} user-flow failure(s)")

    if issues:
        print("Remaining issues:")
        for i in issues:
            print(f"  - {i}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
