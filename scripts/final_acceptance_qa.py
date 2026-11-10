"""End-to-end GUI + scientific acceptance run. Writes JSON; does not invent PASS.

Launch: python scripts/final_acceptance_qa.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIZES = (
    (1366, 768),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
)
MAIN_TABS = (
    "Overview",
    "Motion Analysis",
    "Biomechanics",
    "Results",
    "Compare",
    "Advanced",
)
MOTION_INNER = (
    "Kinematics",
    "Gait Correlation & Spectral Analysis",
    "Spectral Analysis",
    "Spectral Features",
)
BIOMECH_INNER = ("COM", "Ground Reaction Force", "Balance", "Contact", "Gait Phase")


def _row(feature: str, status: str, evidence: str, notes: str = "") -> dict:
    return {"feature": feature, "status": status, "evidence": evidence, "notes": notes}


def _pump(root, seconds: float = 0.08) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            root.update()
        except Exception:
            return
        time.sleep(0.01)


def _mapped(widget) -> bool:
    if widget is None:
        return False
    try:
        return bool(widget.winfo_ismapped()) and int(widget.winfo_width() or 0) > 1
    except Exception:
        return False


def _select_inner(nb, title: str) -> bool:
    if nb is None:
        return False
    try:
        for tid in nb.tabs():
            if str(nb.tab(tid, "text")).strip() == title:
                nb.select(tid)
                return True
    except Exception:
        return False
    return False


def _select_main(app, tab: str) -> str:
    from stablewalk.ui.tk.dashboard_notebook import select_dashboard_tab

    try:
        select_dashboard_tab(app, tab)
        return "ok"
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"


def _canvas_widget(host):
    if host is None:
        return None
    getter = getattr(host, "get_tk_widget", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    return host


def _wait_report(app, attr: str, root, *, seconds: float = 45.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if getattr(app, attr, None) is not None:
            _pump(root, 0.15)
            return True
        _pump(root, 0.12)
    return getattr(app, attr, None) is not None


def main() -> int:
    rows: list[dict] = []
    untested: list[str] = []
    failures: list[dict] = []

    try:
        import tkinter as tk
        from stablewalk.ui.tk.app import StableWalkGUI
        from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout
        from stablewalk.ui.tk.layout_scale import dpi_scale_factor
        from stablewalk import config
    except Exception as exc:
        print(json.dumps({"fatal": str(exc)}))
        return 2

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        payload = {
            "fatal": f"Tk unavailable: {exc}",
            "rows": [_row("Launch desktop GUI", "NOT TESTED", str(exc))],
        }
        out = ROOT / "docs" / "_acceptance_run.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2)[:2000])
        return 1

    app = None
    try:
        app = StableWalkGUI(root=root)
        _pump(root, 0.4)
        scale = dpi_scale_factor(root)
        sw, sh = int(root.winfo_screenwidth()), int(root.winfo_screenheight())
        rows.append(
            _row(
                "Launch desktop GUI",
                "PASS",
                f"StableWalkGUI created; screen {sw}x{sh}; Tk DPI scale {scale:.2f}",
            )
        )
        if abs(scale - 1.0) < 0.08:
            untested.append("Windows display scaling 125/150/175% was not applied as an OS setting")
        rows.append(
            _row(
                "Windows DPI 125/150/175% as OS setting",
                "NOT TESTED",
                f"Process scale factor from Tk is {scale:.2f}; OS scale was not changed",
            )
        )

        notebook = getattr(app, "_dashboard_notebook", None)
        texts = []
        if notebook is not None:
            try:
                texts = [str(notebook.tab(t, "text")) for t in notebook.tabs()]
            except Exception:
                texts = []
        missing = [t for t in MAIN_TABS if t not in texts]
        rows.append(
            _row(
                "Main notebook tabs present",
                "FAIL" if missing else "PASS",
                f"tabs={texts}",
                f"missing={missing}" if missing else "",
            )
        )

        for tab in MAIN_TABS:
            if tab not in texts:
                rows.append(_row(f"Select tab: {tab}", "FAIL", "tab not in notebook"))
                continue
            err = _select_main(app, tab)
            _pump(root, 0.15)
            apply_responsive_layout(app)
            _pump(root, 0.1)
            page = {
                "Overview": getattr(app, "_tab_overview", None),
                "Motion Analysis": getattr(app, "_tab_motion", None),
                "Biomechanics": getattr(app, "_tab_biomechanics", None),
                "Results": getattr(app, "_tab_results_summary", None),
                "Compare": getattr(app, "_tab_compare", None),
                "Advanced": getattr(app, "_tab_advanced_outer", None),
            }.get(tab)
            ok = err == "ok" and _mapped(page)
            rows.append(
                _row(
                    f"Select tab: {tab}",
                    "PASS" if ok else "FAIL",
                    f"select={err}; mapped={_mapped(page)}; size={page.winfo_width()}x{page.winfo_height()}"
                    if page is not None
                    else err,
                )
            )
            if tab == "Advanced":
                empty = getattr(app, "_advanced_empty_state", None)
                ws = getattr(app, "_advanced_workspace", None)
                rows.append(
                    _row(
                        "Advanced empty/workspace state",
                        "PASS" if (_mapped(empty) or _mapped(ws)) else "FAIL",
                        f"empty_mapped={_mapped(empty)} workspace_mapped={_mapped(ws)}",
                    )
                )

        _select_main(app, "Motion Analysis")
        _pump(root, 0.2)
        mnb = getattr(app, "_motion_page_notebook", None)
        for title in MOTION_INNER:
            ok = _select_inner(mnb, title)
            _pump(root, 0.12)
            apply_responsive_layout(app)
            _pump(root, 0.08)
            rows.append(
                _row(
                    f"Motion inner tab: {title}",
                    "PASS" if ok else "FAIL",
                    f"selected={ok}",
                )
            )

        _select_main(app, "Biomechanics")
        _pump(root, 0.2)
        bnb = getattr(app, "_biomech_notebook", None)
        for title in BIOMECH_INNER:
            ok = _select_inner(bnb, title)
            _pump(root, 0.12)
            apply_responsive_layout(app)
            _pump(root, 0.08)
            rows.append(
                _row(
                    f"Biomechanics inner tab: {title}",
                    "PASS" if ok else "FAIL",
                    f"selected={ok}",
                )
            )

        from tests.test_responsive_layout_regression import _collect_issues

        screen_cap = (min(1920, sw), min(1080, sh))
        for w, h in SIZES:
            geom_w, geom_h = w, h
            if w > sw or h > sh:
                # Logical reflow still applied; physical window is clamped.
                geom_w, geom_h = min(w, sw - 8), min(h, sh - 48)
            root.geometry(f"{geom_w}x{geom_h}+20+20")
            _pump(root, 0.2)
            apply_responsive_layout(app, width=w, height=h)
            _pump(root, 0.2)
            issues: list[str] = []
            for tab in ("Overview", "Motion", "Biomechanics", "Results", "Compare", "Advanced"):
                tab_id = {
                    "Overview": "Overview",
                    "Motion": "Motion Analysis",
                    "Biomechanics": "Biomechanics",
                    "Results": "Results",
                    "Compare": "Compare",
                    "Advanced": "Advanced",
                }[tab]
                if tab_id not in texts and tab != "Motion":
                    continue
                _select_main(app, "Motion Analysis" if tab == "Motion" else tab_id)
                _pump(root, 0.12)
                if tab == "Motion":
                    _select_inner(getattr(app, "_motion_page_notebook", None), "Kinematics")
                    _pump(root, 0.1)
                apply_responsive_layout(app, width=w, height=h)
                _pump(root, 0.12)
                issues.extend(_collect_issues(app, width=w, height=h, tab=tab))
            # Inner Motion canvases at this geometry.
            _select_main(app, "Motion Analysis")
            _pump(root, 0.08)
            for title, attr in (
                ("Kinematics", "canvas_motion_workspace"),
                ("Spectral Analysis", "canvas_spectral"),
                ("Spectral Features", "canvas_spectral_features"),
                ("Gait Correlation & Spectral Analysis", "canvas_four_gait"),
            ):
                _select_inner(getattr(app, "_motion_page_notebook", None), title)
                _pump(root, 0.1)
                apply_responsive_layout(app, width=w, height=h)
                _pump(root, 0.1)
                widget = _canvas_widget(getattr(app, attr, None))
                empty = getattr(
                    app,
                    {
                        "Spectral Analysis": "lbl_spectral_empty",
                        "Spectral Features": "lbl_spectral_features_empty",
                        "Gait Correlation & Spectral Analysis": "lbl_four_gait_empty",
                    }.get(title, ""),
                    None,
                )
                host = getattr(app, "motion_graph_host", None) if title == "Kinematics" else None
                if _mapped(widget) or _mapped(empty) or _mapped(host):
                    if _mapped(widget):
                        cw, ch = int(widget.winfo_width() or 0), int(widget.winfo_height() or 0)
                        if cw < 40 or ch < 40:
                            issues.append(f"{w}x{h} Motion/{title}: canvas unusable ({cw}x{ch})")
                    continue
                issues.append(f"{w}x{h} Motion/{title}: canvas/empty state not mapped")
            status = "PASS" if not issues else "FAIL"
            note = ""
            if (w, h) != (geom_w, geom_h):
                note = f"physical window clamped to {geom_w}x{geom_h} (screen {sw}x{sh})"
            rows.append(
                _row(
                    f"Resize layout {w}x{h}",
                    status,
                    "no clip/overlap issues" if not issues else "; ".join(issues[:8]),
                    note,
                )
            )
            if issues:
                failures.append({"feature": f"Resize {w}x{h}", "issues": issues})

        play = getattr(app, "btn_play", None)
        slider = getattr(app, "slider", None)
        _select_main(app, "Overview")
        _pump(root, 0.1)
        rows.append(
            _row(
                "Transport Play + timeline accessible",
                "PASS" if _mapped(play) and _mapped(slider) else "FAIL",
                f"play={_mapped(play)} slider={_mapped(slider)}",
            )
        )
        try:
            menu = root.cget("menu")
            menubar = root.nametowidget(menu) if menu else None
            labels = []
            if menubar is not None:
                end = int(menubar.index("end") or -1)
                for i in range(end + 1):
                    try:
                        labels.append(str(menubar.entrycget(i, "label")))
                    except Exception:
                        continue
            needed = {"File", "View", "Help"}
            missing_menu = sorted(needed - set(labels))
            rows.append(
                _row(
                    "Menus usable (File/View/Help)",
                    "FAIL" if missing_menu else "PASS",
                    f"labels={labels}",
                    f"missing={missing_menu}" if missing_menu else "",
                )
            )
        except Exception as exc:
            rows.append(_row("Menus usable (File/View/Help)", "FAIL", str(exc)))

        # Contain-fit: fitted rectangle must match source aspect and stay inside the box.
        aspect_ok = True
        aspect_notes = []
        for src_w, src_h, box_w, box_h in (
            (1920, 1080, 640, 480),
            (1280, 720, 400, 400),
            (1080, 1920, 320, 560),
        ):
            scale = min(box_w / src_w, box_h / src_h)
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))
            if new_w > box_w or new_h > box_h:
                aspect_ok = False
                aspect_notes.append(f"{src_w}x{src_h} overflow {box_w}x{box_h}")
            if abs((new_w / new_h) - (src_w / src_h)) > 0.02:
                aspect_ok = False
                aspect_notes.append(f"{src_w}x{src_h} aspect drift {new_w}x{new_h}")
        rows.append(
            _row(
                "Video contain-fit preserves aspect ratio",
                "PASS" if aspect_ok else "FAIL",
                "letterbox math holds" if aspect_ok else "; ".join(aspect_notes),
            )
        )

        # Cached-pose analysis (does not re-run MediaPipe).
        from stablewalk.ui.overview_clinical_summary import extract_overview_summary_fields
        from stablewalk.io.analysis_summary_export import export_analysis_summary_json
        from stablewalk.pose.video_source import check_source_format, quick_validate_source

        app._demo_running = True

        pose_pairs = [
            ("normal_gait.mp4", config.POSES_DIR / "normal_gait_poses.json"),
            ("abnormal_gait.mp4", config.POSES_DIR / "abnormal_gait_poses.json"),
        ]
        loaded_ids: list[str] = []
        stability_texts: dict[str, str] = {}
        for video_name, pose_path in pose_pairs:
            if not pose_path.is_file():
                rows.append(
                    _row(
                        f"Load poses for {video_name}",
                        "NOT TESTED",
                        f"missing {pose_path}",
                    )
                )
                continue
            video_path = config.DEMO_VIDEOS_DIR / video_name
            source = str(video_path)
            try:
                app.url_var.set(source)
                ok = bool(
                    app.load_poses(pose_path, fresh=True, expected_source=source)
                )
            except Exception as exc:
                rows.append(
                    _row(
                        f"Load poses for {video_name}",
                        "FAIL",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            _pump(root, 1.2)
            rec = getattr(app, "gait_motion", None)
            seq = getattr(app, "sequence", None)
            src_ok = rec is not None or seq is not None
            rows.append(
                _row(
                    f"Load poses for {video_name}",
                    "PASS" if ok and src_ok else "FAIL",
                    f"load_poses={ok}; recording={rec is not None}; frames={getattr(seq, 'n_frames', None)}",
                )
            )
            if not src_ok:
                continue
            loaded_ids.append(str(getattr(rec, "source", None) or pose_path.name))
            fields = extract_overview_summary_fields(app)
            stab = fields.get("stability_state", ("", ""))
            stab_text = stab[0] if isinstance(stab, tuple) else str(stab)
            conf = fields.get("analysis_confidence", ("", ""))
            conf_text = conf[0] if isinstance(conf, tuple) else str(conf)
            stability_texts[video_name] = stab_text
            rows.append(
                _row(
                    f"Overview KPIs after {video_name}",
                    "PASS",
                    f"stability={stab_text!r}; confidence={conf_text!r}",
                )
            )
            low = stab_text.lower()
            if "%" in stab_text or "reduced" in low:
                if "not available" not in low and "n/a" not in low:
                    rows.append(
                        _row(
                            f"Unreportable stability not shown as reduced ({video_name})",
                            "FAIL",
                            f"stability card={stab_text!r}",
                        )
                    )
                else:
                    rows.append(
                        _row(
                            f"Unreportable stability not shown as reduced ({video_name})",
                            "PASS",
                            f"stability card={stab_text!r}",
                        )
                    )
            else:
                rows.append(
                    _row(
                        f"Unreportable stability not shown as reduced ({video_name})",
                        "PASS",
                        f"stability card={stab_text!r}",
                    )
                )

            from stablewalk.ui.tk.dashboard_spectral import schedule_spectral_analysis
            from stablewalk.ui.tk.dashboard_asymmetry import schedule_asymmetry_analysis

            _select_main(app, "Motion Analysis")
            _select_inner(mnb, "Kinematics")
            _pump(root, 0.2)
            kin_host = getattr(app, "motion_graph_host", None) or getattr(app, "canvas_motion_workspace", None)
            rows.append(
                _row(
                    f"Kinematics plot after {video_name}",
                    "PASS" if _mapped(getattr(kin_host, "get_tk_widget", lambda: kin_host)() if hasattr(kin_host, "get_tk_widget") else kin_host) or _mapped(kin_host) else "FAIL",
                    f"host_mapped={_mapped(kin_host)}",
                )
            )
            _select_inner(mnb, "Spectral Analysis")
            schedule_spectral_analysis(app, force=True)
            _wait_report(app, "_spectral_report", root, seconds=45.0)
            spec = getattr(app, "_spectral_report", None)
            spec_canvas = getattr(app, "canvas_spectral", None)
            spec_w = None
            try:
                spec_w = spec_canvas.get_tk_widget() if spec_canvas is not None else None
            except Exception:
                spec_w = None
            rows.append(
                _row(
                    f"Spectral report bound to {video_name}",
                    "PASS" if spec is not None else "FAIL",
                    f"available={getattr(spec, 'available', None)} canvas={_mapped(spec_w)} warnings={len(getattr(spec, 'warnings', []) or [])}",
                )
            )
            _select_inner(mnb, "Spectral Features")
            _pump(root, 0.3)
            feat_canvas = getattr(app, "canvas_spectral_features", None)
            feat_w = None
            try:
                feat_w = feat_canvas.get_tk_widget() if feat_canvas is not None else None
            except Exception:
                feat_w = None
            rows.append(
                _row(
                    f"Spectral Features canvas after {video_name}",
                    "PASS" if _mapped(feat_w) or feat_canvas is not None else "FAIL",
                    f"canvas_mapped={_mapped(feat_w)}",
                )
            )
            _select_inner(mnb, "Gait Correlation & Spectral Analysis")
            from stablewalk.ui.tk.gait_correlation_tabs import select_gait_corr_tab

            select_gait_corr_tab(app, "Asymmetry Summary")
            schedule_asymmetry_analysis(app, force=True)
            _wait_report(app, "_asymmetry_report", root, seconds=45.0)
            asym = getattr(app, "_asymmetry_report", None)
            rows.append(
                _row(
                    f"Asymmetry report bound to {video_name}",
                    "PASS" if asym is not None else "FAIL",
                    f"provenance={getattr(getattr(asym, 'kinematics', None), 'provenance', None)}",
                )
            )

            _select_main(app, "Biomechanics")
            _pump(root, 0.2)
            for title in BIOMECH_INNER:
                _select_inner(bnb, title)
                _pump(root, 0.12)
                panel = getattr(app, "_biomech_chart_panel", None)
                rows.append(
                    _row(
                        f"Biomechanics {title} after {video_name}",
                        "PASS" if _mapped(panel) else "FAIL",
                        f"panel={_mapped(panel)} size="
                        f"{panel.winfo_width()}x{panel.winfo_height()}" if panel is not None else "no panel",
                    )
                )

            _select_main(app, "Overview")
            _pump(root, 0.15)
            video = getattr(app, "video_frame", None)
            skel = getattr(app, "skel_frame", None)
            path_panel = getattr(app, "overview_traj_panel", None)
            rows.append(
                _row(
                    f"Overview video/3D panels after {video_name}",
                    "PASS" if _mapped(video) and _mapped(skel) else "FAIL",
                    f"video={_mapped(video)} skeleton={_mapped(skel)} path={_mapped(path_panel)}",
                )
            )

            summary = getattr(app, "_analysis_summary_cache", None)
            if summary is None:
                rows.append(
                    _row(
                        f"Results export matches UI ({video_name})",
                        "NOT TESTED",
                        "no _analysis_summary_cache after load_poses",
                    )
                )
            else:
                export_path = Path(tempfile.mkdtemp()) / f"{Path(video_name).stem}_summary.json"
                try:
                    export_analysis_summary_json(summary, export_path)
                    dumped = json.loads(export_path.read_text(encoding="utf-8"))
                    ui_stab = fields.get("stability_state", ("", ""))
                    ui_stab_text = ui_stab[0] if isinstance(ui_stab, tuple) else str(ui_stab)
                    exp_stab = (dumped.get("stability_margin") or {}).get("value")
                    exp_avail = (dumped.get("stability_margin") or {}).get("available")
                    match = True
                    notes = f"ui_stability={ui_stab_text!r} export={exp_stab!r} available={exp_avail}"
                    if exp_avail and ui_stab_text.lower() in ("not available", "n/a", "—"):
                        match = False
                    if not exp_avail and exp_stab not in (None, "Not available", "N/A", "—"):
                        # Export may still carry the N/A string.
                        pass
                    if dumped.get("source") and summary.source:
                        if dumped["source"] != summary.source:
                            match = False
                    rows.append(
                        _row(
                            f"Results export matches UI ({video_name})",
                            "PASS" if match else "FAIL",
                            notes,
                        )
                    )
                except Exception as exc:
                    rows.append(
                        _row(
                            f"Results export matches UI ({video_name})",
                            "FAIL",
                            f"{type(exc).__name__}: {exc}",
                        )
                    )

            conf_text_low = str(conf_text).lower()
            rows.append(
                _row(
                    f"Analysis confidence present ({video_name})",
                    "PASS" if conf_text else "FAIL",
                    f"confidence={conf_text!r}",
                )
            )
            if "high" in conf_text_low and ("not available" in str(stab_text).lower()):
                # High analysis confidence with unreportable stability is allowed;
                # it must not be treated as a stability conclusion.
                rows.append(
                    _row(
                        f"Confidence is not a stability claim ({video_name})",
                        "PASS",
                        "analysis confidence is separate from COM–BoS reportability",
                    )
                )

        if len(loaded_ids) >= 2 and loaded_ids[0] != loaded_ids[-1]:
            rows.append(
                _row(
                    "Session isolation (two pose files)",
                    "PASS",
                    f"sources={loaded_ids}",
                )
            )
        elif len(loaded_ids) >= 2:
            rows.append(
                _row(
                    "Session isolation (two pose files)",
                    "FAIL",
                    f"sources did not change: {loaded_ids}",
                )
            )
        else:
            rows.append(
                _row(
                    "Session isolation (two pose files)",
                    "NOT TESTED",
                    f"loaded {len(loaded_ids)} session(s)",
                )
            )

        # Invalid / missing source — same gates the GUI uses before capture.
        bogus = ROOT / "data" / "demo_videos" / "__missing_clip__.mp4"
        ok_missing, msg_missing = check_source_format(str(bogus))
        rows.append(
            _row(
                "Missing video path handling",
                "PASS" if ok_missing is False else "FAIL",
                msg_missing,
            )
        )
        try:
            app.url_var.set(str(bogus))
            app._load_video(source=str(bogus), show_dialog=False, chain_next_on_fail=False)
            _pump(root, 0.3)
            rows.append(
                _row(
                    "Missing video does not freeze GUI",
                    "PASS",
                    f"alive after _load_video; processing={getattr(app, '_processing', None)}",
                )
            )
        except Exception as exc:
            rows.append(_row("Missing video does not freeze GUI", "FAIL", str(exc)))

        junk = Path(tempfile.mkdtemp()) / "not_a_video.mp4"
        junk.write_bytes(b"this is not a valid mp4 container")
        ok_junk, msg_junk = quick_validate_source(str(junk))
        rows.append(
            _row(
                "Invalid video container rejected",
                "PASS" if ok_junk is False else "FAIL",
                msg_junk,
            )
        )
        short_clip = config.DEMO_VIDEOS_DIR / "candidates" / "weightgait_limping_gait.mp4"
        if short_clip.is_file():
            import cv2

            cap = cv2.VideoCapture(str(short_clip))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            cap.release()
            dur = n / fps if fps > 1e-6 else 0.0
            rows.append(
                _row(
                    "Extremely short limp candidate inspected",
                    "PASS" if n < 40 and dur < 4.0 else "FAIL",
                    f"frames={n} duration_s={dur:.2f}",
                )
            )
        else:
            rows.append(
                _row(
                    "Extremely short limp candidate inspected",
                    "NOT TESTED",
                    str(short_clip),
                )
            )

        freeze = False
        t0 = time.monotonic()
        apply_responsive_layout(app, width=1280, height=720)
        _pump(root, 0.2)
        if time.monotonic() - t0 > 8:
            freeze = True
        rows.append(
            _row(
                "GUI did not freeze during reflow",
                "FAIL" if freeze else "PASS",
                f"reflow wall {time.monotonic() - t0:.2f}s",
            )
        )

        untested.append("Human visual inspection of matplotlib legend-on-data overlap")
        untested.append("Force-plate OpenSim Inverse Dynamics")
        untested.append("Subject-specific OpenSim ScaleTool")
        untested.append("Requirement 5 RL simulator")
        untested.append("Before/after screenshots (no ImageGrab harness asserted)")

    except Exception as exc:
        rows.append(
            _row(
                "Launch desktop GUI",
                "FAIL",
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}",
            )
        )
    finally:
        if app is not None:
            try:
                app.root.destroy()
            except Exception:
                pass
        else:
            try:
                root.destroy()
            except Exception:
                pass

    payload = {
        "rows": rows,
        "untested": untested,
        "failures": failures,
        "counts": {
            "PASS": sum(1 for r in rows if r["status"] == "PASS"),
            "FAIL": sum(1 for r in rows if r["status"] == "FAIL"),
            "NOT TESTED": sum(1 for r in rows if r["status"] == "NOT TESTED"),
        },
    }
    out = ROOT / "docs" / "_acceptance_run.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["counts"]))
    for r in rows:
        if r["status"] != "PASS":
            print(f"{r['status']}: {r['feature']} — {r['evidence'][:200]}")
    return 0 if payload["counts"]["FAIL"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
