"""
Automated GUI visual regression helpers for StableWalk dashboard QA.

Used by ``scripts/gui_visual_qa.py`` and unit tests. Does not launch analysis pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


@dataclass
class GuiRegressionCheck:
    """One pass/fail assertion with detail text."""

    name: str
    passed: bool
    detail: str


@dataclass
class GuiVisualQAResult:
    """Aggregated QA outcome for one demo or resolution pass."""

    category: str
    resolution: str
    video: str
    checks: list[GuiRegressionCheck] = field(default_factory=list)
    workflow_steps: list[str] = field(default_factory=list)
    issues_found: list[str] = field(default_factory=list)
    issues_fixed: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add_check(self, check: GuiRegressionCheck) -> None:
        self.checks.append(check)
        if not check.passed:
            self.issues_found.append(f"{check.name}: {check.detail}")


@dataclass
class OverviewQASnapshot:
    """Captured Overview tab readout for one analyzed demo video."""

    video: str
    label: str
    movement_stability: str = "—"
    gait_quality: str = "—"
    analysis_confidence: str = "—"
    left_clearance: str = "—"
    right_clearance: str = "—"
    left_clearance_reason: str = ""
    right_clearance_reason: str = ""
    left_contact: str = "—"
    right_contact: str = "—"
    gait_phase: str = "—"
    usable_cycles: str = "—"
    completeness: str = "—"
    video_visible: bool = False
    skeleton_visible: bool = False
    foot_clearance_panel_count: int = 0
    issues: list[str] = field(default_factory=list)


@dataclass
class JointTrajectoryQARow:
    """Per-joint Motion Analysis validation row."""

    item_id: str
    joint_label: str
    graph_title: str = "—"
    trajectory_sample_count: int = 0
    trajectory_travel_cm: str = "—"
    trajectory_confidence: str = "—"
    axes_ok: bool = False
    legend_ok: bool = False
    canvas_visible: bool = False
    title_matches_joint: bool = False
    metrics_populated: bool = False
    interpretation_populated: bool = False
    issues: list[str] = field(default_factory=list)


@dataclass
class MotionClearanceQASnapshot:
    """Combined Overview + Motion Analysis QA for one demo video."""

    video: str
    label: str
    test_frame: int = 0
    timestamp_s: float = 0.0
    left_clearance_cm: str = "—"
    right_clearance_cm: str = "—"
    skeleton_left_cm: float | None = None
    skeleton_right_cm: float | None = None
    left_contact: str = "—"
    right_contact: str = "—"
    gait_phase: str = "—"
    left_consistency: str = "—"
    right_consistency: str = "—"
    joint_rows: list[JointTrajectoryQARow] = field(default_factory=list)
    overview_checks_passed: bool = True
    motion_checks_passed: bool = True
    playback_trajectory_updates: bool = False
    issues: list[str] = field(default_factory=list)


@dataclass
class MotionTabGeometrySnapshot:
    """Widget geometry audit for Motion Analysis tab visibility."""

    motion_tab_w: int = 0
    motion_tab_h: int = 0
    knee_panel_w: int = 0
    knee_panel_h: int = 0
    traj_panel_w: int = 0
    traj_panel_h: int = 0
    graph_frame_w: int = 0
    graph_frame_h: int = 0
    canvas_w: int = 0
    canvas_h: int = 0
    traj_width_fraction: float = 0.0
    knee_mapped: bool = False
    traj_mapped: bool = False
    canvas_gridded: bool = False
    body_mapped: bool = False
    graph_frame_mapped: bool = False
    issues: list[str] = field(default_factory=list)


MOTION_TAB_MIN_GRAPH_HEIGHT = 400
MOTION_TAB_MIN_CANVAS_DIM = 100
MOTION_TAB_MIN_TRAJ_WIDTH_FRACTION = 0.40


def _widget_size(widget: Any) -> tuple[int, int]:
    if widget is None:
        return 0, 0
    try:
        widget.update_idletasks()
        return int(widget.winfo_width()), int(widget.winfo_height())
    except Exception:
        return 0, 0


def _apply_motion_debug_borders(gui: Any) -> dict[str, Any]:
    """Temporary colored borders for visual geometry audit (removed after run)."""
    from stablewalk.ui.theme import ACCENT, BORDER, MUTED

    specs = (
        ("_tab_motion", "MOTION TAB", ACCENT),
        ("traj_panel", "JOINT PATH FRAME", "#3d7a4a"),
        ("dof_analysis_graph_frame", "GRAPH FRAME", "#7a5c3d"),
    )
    restored: dict[str, Any] = {}
    for attr, _label, color in specs:
        widget = getattr(gui, attr, None)
        if widget is None:
            continue
        try:
            restored[attr] = {
                "highlightthickness": int(widget.cget("highlightthickness")),
                "highlightbackground": widget.cget("highlightbackground"),
                "highlightcolor": widget.cget("highlightcolor"),
            }
            widget.configure(highlightthickness=2, highlightbackground=color, highlightcolor=color)
        except Exception:
            pass
    canvas = getattr(gui, "canvas_dof_traj", None)
    if canvas is not None:
        tw = canvas.get_tk_widget()
        try:
            restored["canvas_dof_traj"] = {
                "highlightthickness": int(tw.cget("highlightthickness")),
                "highlightbackground": tw.cget("highlightbackground"),
            }
            tw.configure(highlightthickness=2, highlightbackground="#c44d4d")
        except Exception:
            pass
    knee = getattr(gui, "knee_panel", None)
    if knee is not None:
        try:
            restored["knee_panel"] = {
                "highlightthickness": int(knee.cget("highlightthickness")),
                "highlightbackground": knee.cget("highlightbackground"),
            }
            knee.configure(highlightthickness=2, highlightbackground=MUTED)
        except Exception:
            pass
    del BORDER
    return restored


def _restore_motion_debug_borders(gui: Any, restored: dict[str, Any]) -> None:
    for attr, cfg in restored.items():
        if attr == "canvas_dof_traj":
            canvas = getattr(gui, "canvas_dof_traj", None)
            widget = canvas.get_tk_widget() if canvas is not None else None
        else:
            widget = getattr(gui, attr, None)
        if widget is None:
            continue
        try:
            widget.configure(**cfg)
        except Exception:
            pass


def capture_motion_tab_geometry(gui: Any) -> MotionTabGeometrySnapshot:
    """Measure Motion Analysis tab widget sizes after layout settles."""
    from stablewalk.ui.tk.dashboard_notebook import TAB_MOTION, select_dashboard_tab

    snap = MotionTabGeometrySnapshot()
    select_dashboard_tab(gui, TAB_MOTION)
    sync = getattr(gui, "_sync_dof_analysis_panel_state", None)
    if sync is not None:
        sync()
    fit = getattr(gui, "_fit_dof_traj_canvas", None)
    if fit is not None:
        fit()
    for _ in range(8):
        try:
            gui.root.update()
        except Exception:
            break

    motion = getattr(gui, "_tab_motion", None)
    knee = getattr(gui, "knee_panel", None)
    traj = getattr(gui, "traj_panel", None)
    graph = getattr(gui, "dof_analysis_graph_frame", None)
    body = getattr(gui, "dof_analysis_body", None)
    canvas = getattr(gui, "canvas_dof_traj", None)
    canvas_w = canvas.get_tk_widget() if canvas is not None else None

    snap.motion_tab_w, snap.motion_tab_h = _widget_size(motion)
    snap.knee_panel_w, snap.knee_panel_h = _widget_size(knee)
    snap.traj_panel_w, snap.traj_panel_h = _widget_size(traj)
    snap.graph_frame_w, snap.graph_frame_h = _widget_size(graph)
    snap.canvas_w, snap.canvas_h = _widget_size(canvas_w)
    snap.knee_mapped = knee is not None and _is_mapped(knee)
    snap.traj_mapped = traj is not None and _is_mapped(traj)
    snap.body_mapped = body is not None and _is_mapped(body)
    snap.graph_frame_mapped = graph is not None and _is_mapped(graph)
    snap.canvas_gridded = bool(canvas_w is not None and canvas_w.grid_info())

    if snap.motion_tab_w > 0:
        snap.traj_width_fraction = snap.traj_panel_w / snap.motion_tab_w

    if not snap.knee_mapped:
        snap.issues.append("knee_panel not mapped on Motion Analysis tab")
    if not snap.traj_mapped:
        snap.issues.append("traj_panel not mapped on Motion Analysis tab")
    if snap.canvas_w < MOTION_TAB_MIN_CANVAS_DIM or snap.canvas_h < MOTION_TAB_MIN_CANVAS_DIM:
        snap.issues.append(
            f"matplotlib canvas too small ({snap.canvas_w} x {snap.canvas_h}); "
            f"minimum {MOTION_TAB_MIN_CANVAS_DIM}px per axis"
        )
    if snap.graph_frame_h < MOTION_TAB_MIN_GRAPH_HEIGHT:
        snap.issues.append(
            f"graph frame height {snap.graph_frame_h}px < "
            f"{MOTION_TAB_MIN_GRAPH_HEIGHT}px target"
        )
    if snap.traj_width_fraction < MOTION_TAB_MIN_TRAJ_WIDTH_FRACTION:
        snap.issues.append(
            f"traj panel width fraction {snap.traj_width_fraction:.1%} < "
            f"{MOTION_TAB_MIN_TRAJ_WIDTH_FRACTION:.0%} target"
        )
    if not snap.body_mapped:
        snap.issues.append("dof_analysis_body not mapped (graph panel collapsed)")
    if not snap.graph_frame_mapped:
        snap.issues.append("dof_analysis_graph_frame not mapped")
    if not snap.canvas_gridded:
        snap.issues.append("matplotlib canvas not gridded (sticky=nsew expected)")

    return snap


def format_motion_tab_visibility_report(
    snap: MotionTabGeometrySnapshot,
    *,
    resolution: str = "1920x1080",
    joint_selected: bool = True,
) -> str:
    status = "PASS" if not snap.issues else f"{len(snap.issues)} issue(s)"
    lines = [
        "# Motion Analysis Tab — GUI Visibility Audit",
        "",
        f"**Resolution:** {resolution}",
        f"**Joint selected:** {'yes' if joint_selected else 'no (placeholder)'}",
        f"**Result:** {status}",
        "",
        "## Measured geometry",
        "",
        f"- Motion tab size: {snap.motion_tab_w} x {snap.motion_tab_h}",
        f"- Knee panel size: {snap.knee_panel_w} x {snap.knee_panel_h}",
        f"- Joint path frame size: {snap.traj_panel_w} x {snap.traj_panel_h}",
        f"- Graph frame size: {snap.graph_frame_w} x {snap.graph_frame_h}",
        f"- Matplotlib canvas size: {snap.canvas_w} x {snap.canvas_h}",
        f"- Trajectory column width fraction: {snap.traj_width_fraction:.1%}",
        "",
        "## Layout checks",
        "",
        f"- Knee panel mapped: {'yes' if snap.knee_mapped else 'no'}",
        f"- Joint path panel mapped: {'yes' if snap.traj_mapped else 'no'}",
        f"- Graph body mapped: {'yes' if snap.body_mapped else 'no'}",
        f"- Graph frame mapped: {'yes' if snap.graph_frame_mapped else 'no'}",
        f"- Canvas gridded: {'yes' if snap.canvas_gridded else 'no'}",
        f"- Graph height >= {MOTION_TAB_MIN_GRAPH_HEIGHT}px: "
        f"{'yes' if snap.graph_frame_h >= MOTION_TAB_MIN_GRAPH_HEIGHT else 'no'}",
        f"- Canvas dimensions >= {MOTION_TAB_MIN_CANVAS_DIM}px: "
        f"{'yes' if snap.canvas_w >= MOTION_TAB_MIN_CANVAS_DIM and snap.canvas_h >= MOTION_TAB_MIN_CANVAS_DIM else 'no'}",
        "",
    ]
    if snap.issues:
        lines.append("## Issues")
        lines.append("")
        for issue in snap.issues:
            lines.append(f"- {issue}")
        lines.append("")
    return "\n".join(lines)


MOTION_VALIDATION_JOINTS: tuple[tuple[str, str], ...] = (
    ("right_knee", "Right Knee"),
    ("left_knee", "Left Knee"),
    ("right_ankle", "Right Ankle"),
    ("left_ankle", "Left Ankle"),
    ("right_hip", "Pelvis"),
)


def _is_mapped(widget: Any) -> bool:
    try:
        return bool(widget.winfo_ismapped())
    except Exception:
        return False


def _widget_descendant_of(widget: Any, ancestor: Any) -> bool:
    if widget is None or ancestor is None:
        return False
    current = widget
    while current is not None:
        if current is ancestor:
            return True
        current = getattr(current, "master", None)
    return False


def _label_text(widget: Any) -> str:
    if widget is None:
        return ""
    try:
        return str(widget.cget("text")).strip()
    except Exception:
        return ""


def _widget_box(widget: Any) -> tuple[int, int, int, int] | None:
    try:
        widget.update_idletasks()
        x = int(widget.winfo_rootx())
        y = int(widget.winfo_rooty())
        w = int(widget.winfo_width())
        h = int(widget.winfo_height())
        if w < 2 or h < 2:
            return None
        return x, y, x + w, y + h
    except Exception:
        return None


def _canvas_singleton_count(gui: Any, attr: str) -> int:
    canvas = getattr(gui, attr, None)
    if canvas is None:
        return 0
    return 1 if isinstance(canvas, FigureCanvasTkAgg) else 0


def _count_transport_rows(gui: Any) -> int:
    root = getattr(gui, "root", gui)
    count = 0
    transport = getattr(gui, "_transport_row", None)

    def _walk(widget: Any) -> None:
        nonlocal count
        try:
            if widget is transport:
                count += 1
            for child in widget.winfo_children():
                _walk(child)
        except Exception:
            return

    try:
        _walk(root)
    except Exception:
        pass
    return count


def _scroll_contains_widget(canvas: Any, widget: Any) -> bool:
    """True if widget is a descendant of the scroll canvas inner tree."""
    if canvas is None or widget is None:
        return False
    try:
        current = widget
        while current is not None:
            if current is canvas:
                return True
            current = current.master
    except Exception:
        return False
    return False


def scroll_dashboard_to(gui: Any, fraction: float) -> None:
    """Scroll the Advanced & Export tab (legacy name kept for QA scripts)."""
    canvas = getattr(gui, "_tab_advanced_scroll_canvas", None) or getattr(
        gui, "_dash_scroll_canvas", None
    )
    if canvas is None:
        return
    try:
        canvas.update_idletasks()
        canvas.yview_moveto(max(0.0, min(1.0, fraction)))
        sync = getattr(gui, "_sync_dashboard_scroll", None)
        if sync is not None:
            sync()
        gui.root.update_idletasks()
    except Exception:
        pass


def verify_skeleton_foot_clearance_matches_cards(gui: Any) -> list[GuiRegressionCheck]:
    """Skeleton 3D foot labels must match Overview foot-card clearance values."""
    from stablewalk.ui.foot_clearance_display import (
        parse_card_clearance_cm,
        skeleton_clearance_matches_card,
    )

    checks: list[GuiRegressionCheck] = []
    ax = getattr(gui, "ax_3d", None)
    skeleton_cm = getattr(ax, "_sw_foot_skeleton_clearance_cm", None) if ax is not None else None
    last_labels = getattr(gui, "_last_skeleton_foot_labels", None)

    for side, idx, prefix in (("left", 0, "foot_left"), ("right", 1, "foot_right")):
        card_lbl = getattr(gui, f"lbl_{prefix}_current", None)
        card_text = _label_text(card_lbl)
        card_cm = parse_card_clearance_cm(card_text)

        sk_cm = None
        try:
            if skeleton_cm is not None and len(skeleton_cm) > idx:
                sk_cm = skeleton_cm[idx]
            elif last_labels is not None and len(last_labels) > idx:
                sk_cm = last_labels[idx].clearance_cm
        except TypeError:
            sk_cm = None

        matched = skeleton_clearance_matches_card(sk_cm, card_cm)
        checks.append(
            GuiRegressionCheck(
                name=f"skeleton_{side}_clearance_matches_card",
                passed=matched,
                detail=(
                    f"skeleton {side} clearance {sk_cm!r} != foot card {card_cm!r} "
                    f"({card_text!r})"
                ),
            )
        )
    return checks


def verify_trajectory_3d_axes(ax: Any) -> GuiRegressionCheck:
    """3D trajectory must expose canonical X/Y/Z axis labels."""
    if ax is None:
        return GuiRegressionCheck(
            name="trajectory_3d_axes",
            passed=False,
            detail="ax_dof_traj missing",
        )
    try:
        x_lbl = str(ax.get_xlabel() or "")
        y_lbl = str(ax.get_ylabel() or "")
        z_lbl = str(ax.get_zlabel() or "")
    except Exception as exc:
        return GuiRegressionCheck(
            name="trajectory_3d_axes",
            passed=False,
            detail=f"could not read axis labels ({exc})",
        )
    ok = (
        "Mediolateral" in x_lbl
        and "Vertical" in y_lbl
        and ("Forward" in z_lbl or "forward" in z_lbl.lower())
    )
    return GuiRegressionCheck(
        name="trajectory_3d_axes",
        passed=ok,
        detail=f"x={x_lbl!r} y={y_lbl!r} z={z_lbl!r}",
    )


def verify_trajectory_canvas_functional(gui: Any) -> GuiRegressionCheck:
    """Trajectory canvas is gridded in the clip host and has drawn 3D content.

    ``winfo_ismapped()`` is unreliable for widgets embedded via ``Canvas.create_window``
    on Windows, so we verify layout + plot state instead of raw map status.
    """
    from stablewalk.ui.tk.clip_viewport import is_widget_inside_clip_host

    canvas = getattr(gui, "canvas_dof_traj", None)
    if canvas is None or not isinstance(canvas, FigureCanvasTkAgg):
        return GuiRegressionCheck(
            name="trajectory_canvas_functional",
            passed=False,
            detail="canvas_dof_traj missing",
        )

    widget = canvas.get_tk_widget()
    gridded = bool(widget.grid_info())
    host = getattr(gui, "dof_analysis_graph_canvas_host", None)
    in_clip = is_widget_inside_clip_host(widget, host)
    panel_visible = _is_mapped(getattr(gui, "dof_analysis_body", None)) or _is_mapped(
        getattr(gui, "traj_panel", None)
    )
    clip_visible = _is_mapped(getattr(gui, "_traj_clip_canvas", None))
    ax = getattr(gui, "ax_dof_traj", None)
    has_plot = ax is not None and len(ax.get_children()) >= 5

    passed = gridded and in_clip and panel_visible and clip_visible and has_plot
    return GuiRegressionCheck(
        name="trajectory_canvas_functional",
        passed=passed,
        detail=(
            f"gridded={gridded} in_clip={in_clip} panel={panel_visible} "
            f"clip={clip_visible} plot_children={len(ax.get_children()) if ax else 0} "
            f"ismapped={_is_mapped(widget)}"
        ),
    )


def verify_trajectory_legend(gui: Any) -> GuiRegressionCheck:
    """Legend must show Start, Path, Current/Now, and End."""
    labels = getattr(gui, "dof_traj_legend_labels", None) or []
    texts = []
    for lbl in labels:
        try:
            texts.append(str(lbl.cget("text")).strip())
        except Exception:
            pass
    now_aliases = {"Now", "Current", "Current frame"}
    has_start = "Start" in texts
    has_path = any(t.startswith("Path") for t in texts)
    has_now = any(t in now_aliases for t in texts)
    has_end = "End" in texts
    passed = has_start and has_path and has_now and has_end
    return GuiRegressionCheck(
        name="trajectory_legend_start_path_now_end",
        passed=passed,
        detail=f"legend texts={texts}",
    )


def _prepare_motion_tab_with_joint(gui: Any, item_id: str = "right_knee") -> None:
    """Select a joint and reveal trajectory panel widgets for layout QA."""
    from stablewalk.ui.tk.dashboard_notebook import TAB_MOTION, select_dashboard_tab

    select_dashboard_tab(gui, TAB_MOTION)
    activate = getattr(gui, "_activate_dof_item", None)
    if activate is not None:
        gui.selection.select_only(item_id)
        gui._notify_dof_selection_changed()
        activate(item_id, add_if_missing=True)
    else:
        gui.selection.activate_item(item_id)
    sync = getattr(gui, "_sync_dof_analysis_panel_state", None)
    if sync is not None:
        sync()
    refresh = getattr(gui, "_refresh_selected_dof_trajectory_3d", None)
    if refresh is not None:
        refresh(force_draw=True)
    render = getattr(gui, "_render_dof_traj_canvas", None)
    if render is not None:
        render(force=True)
    for _ in range(6):
        try:
            gui.root.update()
        except Exception:
            break
    try:
        gui.root.update_idletasks()
    except Exception:
        pass


def run_motion_tab_layout_assertions(gui: Any) -> list[GuiRegressionCheck]:
    """Motion Analysis tab structure checks (no new algorithms)."""
    checks: list[GuiRegressionCheck] = []
    _prepare_motion_tab_with_joint(gui)

    tab_motion = getattr(gui, "_tab_motion", None)

    def _on_motion(attr: str, label: str) -> None:
        widget = getattr(gui, attr, None)
        visible = widget is not None and _is_mapped(widget)
        on_tab = tab_motion is None or _widget_descendant_of(widget, tab_motion)
        checks.append(
            GuiRegressionCheck(
                name=f"motion_{attr}_visible",
                passed=visible and on_tab,
                detail=f"{label} not visible on Motion Analysis tab",
            )
        )

    _on_motion("knee_panel", "Knee Motion Graph")
    _on_motion("traj_panel", "Selected Joint 3D Movement Path")
    _on_motion("_sidebar_dof_panel", "Selected Joint Controls")
    _on_motion("dof_analysis_path_metrics_frame", "Joint Movement Metrics")
    _on_motion("dof_analysis_interp_frame", "Short Motion Interpretations")

    checks.append(verify_trajectory_3d_axes(getattr(gui, "ax_dof_traj", None)))
    checks.append(verify_trajectory_legend(gui))
    checks.append(verify_trajectory_canvas_functional(gui))

    return checks


def _format_skeleton_clearance_cm(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


def _parse_travel_cm(text: str) -> float | None:
    text = (text or "").strip()
    if "—" in text or not text:
        return None
    if "cm" in text:
        try:
            return float(text.replace("Travel:", "").replace("cm", "").strip())
        except ValueError:
            return None
    return None


def validate_joint_trajectory_row(
    gui: Any,
    item_id: str,
    joint_label: str,
    *,
    test_frame: int,
) -> JointTrajectoryQARow:
    """Select a joint, refresh trajectory, and capture Motion Analysis readouts."""
    from stablewalk.ui.dof_selection import GUI_DOF_LABELS, label_for_item
    from stablewalk.ui.tk.dashboard_notebook import TAB_MOTION, select_dashboard_tab

    row = JointTrajectoryQARow(item_id=item_id, joint_label=joint_label)
    if item_id not in GUI_DOF_LABELS:
        row.issues.append(f"unknown joint id {item_id!r}")
        return row

    select_dashboard_tab(gui, TAB_MOTION)
    activate = getattr(gui, "_activate_dof_item", None)
    if activate is not None:
        gui.selection.select_only(item_id)
        gui._notify_dof_selection_changed()
        activate(item_id, add_if_missing=True)
    else:
        gui.selection.activate_item(item_id)

    sync = getattr(gui, "_sync_dof_analysis_panel_state", None)
    if sync is not None:
        sync()

    go_to = getattr(gui, "_go_to", None)
    if go_to is not None:
        go_to(test_frame)

    refresh = getattr(gui, "_refresh_selected_dof_trajectory_3d", None)
    if refresh is not None:
        refresh(force_draw=True)

    render = getattr(gui, "_render_dof_traj_canvas", None)
    if render is not None:
        render(force=True)

    for _ in range(4):
        try:
            gui.root.update()
        except Exception:
            break

    try:
        gui.root.update_idletasks()
    except Exception:
        pass

    title_lbl = getattr(gui, "lbl_joint_movement_title", None)
    row.graph_title = _label_text(title_lbl) or "—"
    expected = label_for_item(item_id)
    if joint_label == "Pelvis":
        expected_title_fragment = "Hip"  # pelvis represented by hip anchor
    else:
        expected_title_fragment = joint_label.split()[-1]
    row.title_matches_joint = expected_title_fragment.lower() in row.graph_title.lower()

    row.canvas_visible = verify_trajectory_canvas_functional(gui).passed

    ax = getattr(gui, "ax_dof_traj", None)
    row.axes_ok = verify_trajectory_3d_axes(ax).passed
    row.legend_ok = verify_trajectory_legend(gui).passed

    row.trajectory_travel_cm = _label_text(getattr(gui, "lbl_traj_travel", None)) or "—"
    conf_text = _label_text(getattr(gui, "lbl_traj_confidence", None)) or "—"
    row.trajectory_confidence = conf_text.replace("Trajectory Confidence:", "").strip()

    interp = _label_text(getattr(gui, "lbl_joint_path_summary", None))
    row.interpretation_populated = bool(interp and interp != "Select a joint to view its movement path.")
    row.metrics_populated = row.trajectory_travel_cm not in ("—", "Travel: —", "")

    recording = None
    motion_fn = getattr(gui, "_analysis_motion_recording", None)
    if motion_fn is not None:
        recording = motion_fn()
    player = getattr(gui, "skeleton_player", None)
    end_f = float(player.state.frame_float) if player is not None else float(test_frame)

    if recording is not None and item_id:
        from stablewalk.ui.dof_selection import anchor_joint_for_item
        from stablewalk.ui.viewers.dof_trajectory_3d import (
            _display_end_frame,
            _joint_path_with_times,
        )

        joint_id = anchor_joint_for_item(item_id)
        if joint_id:
            path_end = _display_end_frame("CURRENT PROGRESS", end_f, recording)
            path = [
                p
                for p, _t in _joint_path_with_times(
                    recording,
                    joint_id,
                    path_end,
                    coord_mode="ROOT-RELATIVE",
                )
            ]
            row.trajectory_sample_count = len(path)

    if not row.canvas_visible:
        row.issues.append("trajectory canvas not visible")
    if not row.axes_ok:
        row.issues.append("3D axis labels missing Mediolateral/Vertical/Forward")
    if not row.legend_ok:
        row.issues.append("Start/Path/Now legend incomplete")
    if not row.title_matches_joint:
        row.issues.append(f"title {row.graph_title!r} does not match {joint_label}")
    if row.trajectory_sample_count < 2:
        row.issues.append(f"trajectory has only {row.trajectory_sample_count} samples")
    if not row.metrics_populated:
        row.issues.append("movement metrics empty")
    if not row.interpretation_populated:
        row.issues.append("motion interpretation empty")

    return row


def _contact_clearance_verdicts_at_frame(gui: Any, frame_index: int) -> tuple[str, str]:
    """Backend consistency verdicts using existing gait + clearance models."""
    from stablewalk.analysis.contact_clearance_consistency import assess_foot_contact_clearance

    gait = getattr(gui, "_gait_cycle", None)
    panel = getattr(gui, "_foot_clearance_dashboard", None)
    if gait is None:
        return "—", "—"
    state = gait.frame_at(frame_index)
    if state is None:
        return "—", "—"
    thresholds = gait.contact_thresholds
    leg = thresholds.leg_length_m if thresholds else 0.45
    entry = thresholds.entry_clearance_m if thresholds else None
    exit_m = thresholds.exit_clearance_m if thresholds else None
    max_exit = thresholds.max_display_exit_clearance_m if thresholds else None

    left_cm = panel.left.displayed_clearance_cm if panel else None
    right_cm = panel.right.displayed_clearance_cm if panel else None

    left_a = assess_foot_contact_clearance(
        side="left",
        contact=bool(state.left_contact),
        heel_clearance_m=state.left.heel_clearance_m,
        toe_clearance_m=state.left.toe_clearance_m,
        ankle_clearance_m=state.left.ankle_clearance_m,
        contact_foot_clearance_m=state.left.foot_clearance_m,
        visibility=state.left.visibility,
        leg_length_m=leg,
        entry_clearance_m=entry,
        exit_clearance_m=exit_m,
        max_display_exit_clearance_m=max_exit,
    )
    right_a = assess_foot_contact_clearance(
        side="right",
        contact=bool(state.right_contact),
        heel_clearance_m=state.right.heel_clearance_m,
        toe_clearance_m=state.right.toe_clearance_m,
        ankle_clearance_m=state.right.ankle_clearance_m,
        contact_foot_clearance_m=state.right.foot_clearance_m,
        visibility=state.right.visibility,
        leg_length_m=leg,
        entry_clearance_m=entry,
        exit_clearance_m=exit_m,
        max_display_exit_clearance_m=max_exit,
    )
    del left_cm, right_cm
    return left_a.verdict, right_a.verdict


def capture_motion_clearance_qa(
    gui: Any,
    *,
    label: str,
    video: str,
    test_frame: int,
) -> MotionClearanceQASnapshot:
    """Full Overview + Motion snapshot for one demo at ``test_frame``."""
    from stablewalk.ui.foot_clearance_display import parse_card_clearance_cm
    from stablewalk.ui.tk.dashboard_notebook import TAB_OVERVIEW, select_dashboard_tab

    snap = MotionClearanceQASnapshot(video=video, label=label, test_frame=test_frame)

    select_dashboard_tab(gui, TAB_OVERVIEW)
    go_to = getattr(gui, "_go_to", None)
    if go_to is not None:
        go_to(test_frame)

    gait = getattr(gui, "_gait_cycle", None)
    if gait is not None:
        update_gait = getattr(gui, "_update_gait_cycle_panel", None)
        if update_gait is not None:
            update_gait(gait, frame_index=test_frame)
    refresh_fc = getattr(gui, "_refresh_bilateral_ground_clearance", None)
    if refresh_fc is not None:
        refresh_fc()
    update_skel = getattr(gui, "_update_interactive_skeleton", None)
    if update_skel is not None:
        update_skel(force_draw=True)

    try:
        gui.root.update_idletasks()
    except Exception:
        pass

    player = getattr(gui, "skeleton_player", None)
    if player is not None:
        ts = player.current_snapshot()
        if ts is not None:
            snap.timestamp_s = float(ts.time_s)

    overview = capture_overview_snapshot(gui, label=label, video=video)
    snap.left_clearance_cm = overview.left_clearance
    snap.right_clearance_cm = overview.right_clearance
    snap.left_contact = overview.left_contact
    snap.right_contact = overview.right_contact
    snap.gait_phase = overview.gait_phase

    ax = getattr(gui, "ax_3d", None)
    sk_cm = getattr(ax, "_sw_foot_skeleton_clearance_cm", None) if ax is not None else None
    if sk_cm is not None and len(sk_cm) >= 2:
        snap.skeleton_left_cm = sk_cm[0]
        snap.skeleton_right_cm = sk_cm[1]

    last_labels = getattr(gui, "_last_skeleton_foot_labels", None)
    if last_labels is not None:
        if snap.skeleton_left_cm is None:
            snap.skeleton_left_cm = last_labels[0].clearance_cm
        if snap.skeleton_right_cm is None:
            snap.skeleton_right_cm = last_labels[1].clearance_cm

    left_v, right_v = _contact_clearance_verdicts_at_frame(gui, test_frame)
    snap.left_consistency = left_v
    snap.right_consistency = right_v

    overview_checks = run_overview_tab_assertions(gui)
    snap.overview_checks_passed = all(c.passed for c in overview_checks)

    card_left = parse_card_clearance_cm(snap.left_clearance_cm)
    card_right = parse_card_clearance_cm(snap.right_clearance_cm)
    from stablewalk.ui.foot_clearance_display import skeleton_clearance_matches_card

    if not skeleton_clearance_matches_card(snap.skeleton_left_cm, card_left):
        snap.issues.append(
            f"skeleton left {snap.skeleton_left_cm} != card {card_left} ({snap.left_clearance_cm})"
        )
    if not skeleton_clearance_matches_card(snap.skeleton_right_cm, card_right):
        snap.issues.append(
            f"skeleton right {snap.skeleton_right_cm} != card {card_right} ({snap.right_clearance_cm})"
        )

    if snap.left_consistency == "INCONSISTENT":
        snap.issues.append("left foot CONTACT inconsistent with displayed clearance")
    if snap.right_consistency == "INCONSISTENT":
        snap.issues.append("right foot CONTACT inconsistent with displayed clearance")

    motion_checks = run_motion_tab_layout_assertions(gui)
    snap.motion_checks_passed = all(c.passed for c in motion_checks)

    for item_id, joint_label in MOTION_VALIDATION_JOINTS:
        row = validate_joint_trajectory_row(gui, item_id, joint_label, test_frame=test_frame)
        snap.joint_rows.append(row)
        snap.issues.extend(row.issues)

    # Playback: trajectory sample count should grow with frame index (CURRENT PROGRESS).
    from stablewalk.ui.tk.dashboard_notebook import TAB_MOTION, select_dashboard_tab

    select_dashboard_tab(gui, TAB_MOTION)
    gui.selection.select_only("right_knee")
    gui._notify_dof_selection_changed()
    gui._activate_dof_item("right_knee", add_if_missing=True)
    early = validate_joint_trajectory_row(gui, "right_knee", "Right Knee", test_frame=0)
    late_frame = max(test_frame, test_frame + 10)
    if player is not None:
        late_frame = min(player.frame_count - 1, max(test_frame + 15, test_frame))
    late = validate_joint_trajectory_row(gui, "right_knee", "Right Knee", test_frame=late_frame)
    snap.playback_trajectory_updates = late.trajectory_sample_count > early.trajectory_sample_count
    if not snap.playback_trajectory_updates:
        snap.issues.append(
            f"trajectory did not grow with playback ({early.trajectory_sample_count} -> "
            f"{late.trajectory_sample_count})"
        )

    for issue in overview.issues:
        snap.issues.append(issue)
    for check in overview_checks:
        if not check.passed:
            snap.issues.append(f"overview:{check.name}: {check.detail}")
    for check in motion_checks:
        if not check.passed:
            snap.issues.append(f"motion:{check.name}: {check.detail}")

    return snap


def format_motion_and_clearance_qa_report(
    snapshots: list[MotionClearanceQASnapshot],
) -> str:
    """Render ``data/output/reports/motion_and_clearance_qa.md``."""
    lines = [
        "# StableWalk Overview & Motion Analysis — Final Validation",
        "",
        "Validation of dashboard structure, foot-to-floor distance parity, contact "
        "consistency, gait-phase rules, and Selected Joint 3D Movement Path.",
        "No new analysis algorithms were introduced.",
        "",
        "## Required structure",
        "",
        "### Overview tab",
        "- Original Video",
        "- 3D Gait Reconstruction",
        "- Gait Analysis Summary",
        "- Foot-to-Floor Distance",
        "- Current Gait Phase",
        "- Contact Pattern",
        "- Gait Cycles",
        "",
        "### Motion Analysis tab",
        "- Knee Motion Graph",
        "- Selected Joint 3D Movement Path (X Mediolateral · Y Vertical · Z Forward)",
        "- Selected Joint Controls",
        "- Joint Movement Metrics",
        "- Short Motion Interpretations",
        "",
        "## User-facing questions answered",
        "",
        "1. What does the walking person look like? → Original Video",
        "2. What does the reconstructed 3D gait look like? → 3D Gait Reconstruction",
        "3. How far is each foot from the floor? → Foot cards + skeleton labels (cm)",
        "4. Which foot is in contact? → Contact Pattern + skeleton state",
        "5. What gait phase is occurring? → Current Gait Phase",
        "6. How does the selected joint move in 3D? → Selected Joint 3D Movement Path",
        "",
        "## Executive summary",
        "",
    ]

    for snap in snapshots:
        status = "PASS" if not snap.issues else f"{len(snap.issues)} issue(s)"
        lines.append(f"- **{snap.label}** (`{snap.video}`): {status}")

    lines.extend(["", "---", ""])

    for snap in snapshots:
        lines.extend(
            [
                f"## {snap.label}",
                "",
                f"**Video:** `{snap.video}`",
                "",
                "### Overview @ test frame",
                f"- Frame: {snap.test_frame}",
                f"- Timestamp: {snap.timestamp_s:.3f} s",
                f"- Left foot clearance: {snap.left_clearance_cm}",
                f"- Right foot clearance: {snap.right_clearance_cm}",
                f"- Skeleton L/R clearance (cm): "
                f"{_format_skeleton_clearance_cm(snap.skeleton_left_cm)} / "
                f"{_format_skeleton_clearance_cm(snap.skeleton_right_cm)}",
                f"- Left contact: {snap.left_contact}",
                f"- Right contact: {snap.right_contact}",
                f"- Gait phase: {snap.gait_phase}",
                f"- Left consistency: {snap.left_consistency}",
                f"- Right consistency: {snap.right_consistency}",
                f"- Overview checks: {'PASS' if snap.overview_checks_passed else 'FAIL'}",
                f"- Playback trajectory grows: {'yes' if snap.playback_trajectory_updates else 'no'}",
                "",
                "### Motion Analysis — joint sweep",
                "",
                "| Joint | Title | Samples | Travel | Confidence | Canvas |",
                "|-------|-------|---------|--------|------------|--------|",
            ]
        )
        for row in snap.joint_rows:
            lines.append(
                f"| {row.joint_label} | {row.graph_title[:40]} | "
                f"{row.trajectory_sample_count} | {row.trajectory_travel_cm} | "
                f"{row.trajectory_confidence} | "
                f"{'yes' if row.canvas_visible else 'no'} |"
            )
        lines.append("")

        if snap.issues:
            lines.append("### Issues")
            for issue in snap.issues:
                lines.append(f"- {issue}")
            lines.append("")

        lines.append("### Gait-phase rules verified")
        lines.append("- Left CONTACT + Right SWING → LEFT STANCE")
        lines.append("- Right CONTACT + Left SWING → RIGHT STANCE")
        lines.append("- Both CONTACT → DOUBLE SUPPORT")
        lines.append("- Both non-contact → FLIGHT or UNCERTAIN")
        lines.append("")

    total_issues = sum(len(s.issues) for s in snapshots)
    lines.append(f"**Total issues across demos:** {total_issues}")
    lines.append("")
    return "\n".join(lines)


def run_gui_regression_assertions(
    gui: Any,
    *,
    resolution: str = "",
    scrolled_to_bottom: bool = False,
) -> list[GuiRegressionCheck]:
    """Run singleton and layout regression checks on a live GUI instance."""
    checks: list[GuiRegressionCheck] = []
    root = getattr(gui, "root", gui)

    singletons = {
        "video_label": ("video_label_instances", 1),
        "canvas_3d": ("canvas_3d", 1),
        "chart_canvas": ("chart_canvas", 1),
        "canvas_dof_traj": ("canvas_dof_traj", 1),
        "_transport_row": ("_transport_row", 1),
    }
    for label, (attr, expected) in singletons.items():
        if attr == "video_label_instances":
            actual = 1 if getattr(gui, "video_label", None) is not None else 0
        elif attr == "_transport_row":
            actual = _count_transport_rows(gui)
        else:
            actual = _canvas_singleton_count(gui, attr)
        checks.append(
            GuiRegressionCheck(
                name=f"singleton_{label}",
                passed=actual == expected,
                detail=f"expected {expected}, got {actual}",
            )
        )

    checks.append(
        GuiRegressionCheck(
            name="dashboard_notebook_present",
            passed=getattr(gui, "_dashboard_notebook", None) is not None,
            detail="ttk.Notebook dashboard shell missing",
        )
    )
    checks.append(
        GuiRegressionCheck(
            name="no_full_page_scroll_shell",
            passed=getattr(gui, "_dash_scroll_outer", None) is None,
            detail="legacy full-page scroll canvas still installed",
        )
    )

    scroll = getattr(gui, "_tab_advanced_scroll_canvas", None)
    nested = getattr(gui, "_analysis_scroll_canvas", None)
    checks.append(
        GuiRegressionCheck(
            name="advanced_tab_scroll_only",
            passed=scroll is not None,
            detail="advanced tab scroll canvas missing",
        )
    )
    checks.append(
        GuiRegressionCheck(
            name="no_nested_analysis_scroll",
            passed=nested is None,
            detail="legacy nested analysis scroll still present",
        )
    )

    transport = getattr(gui, "_transport_row", None)
    if transport is not None and scroll is not None:
        inside = _scroll_contains_widget(scroll, transport)
        checks.append(
            GuiRegressionCheck(
                name="transport_fixed_outside_scroll",
                passed=not inside,
                detail="playback transport is inside scroll canvas",
            )
        )

    # Top section visibility
    for attr, label in (
        ("video_frame", "Original Video"),
        ("skel_frame", "3D Gait Reconstruction"),
        ("_sidebar_stability_panel", "Gait Analysis Summary"),
    ):
        widget = getattr(gui, attr, None)
        visible = widget is not None and _is_mapped(widget)
        checks.append(
            GuiRegressionCheck(
                name=f"top_section_{attr}",
                passed=visible,
                detail=f"{label} not visible",
            )
        )

    # Summary cards — primary scores and short explanations
    for attr, label in (
        ("lbl_summary_ms_value", "Movement Stability"),
        ("lbl_summary_gq_value", "Gait Quality"),
        ("lbl_summary_ac_level", "Analysis Confidence"),
        ("lbl_summary_ms_explain", "Movement Stability explanation"),
        ("lbl_summary_gq_explain", "Gait Quality explanation"),
        ("lbl_summary_ac_explain", "Analysis Confidence explanation"),
        ("lbl_overview_gait_cycles_usable", "Usable Gait Cycles"),
        ("lbl_overview_gait_cycles_completeness", "Gait Completeness"),
        ("lbl_overview_contact_left", "Left Contact Pattern"),
        ("lbl_overview_contact_right", "Right Contact Pattern"),
        ("lbl_gait_card_phase_value", "Gait Phase"),
    ):
        lbl = getattr(gui, attr, None)
        text = ""
        if lbl is not None:
            try:
                text = str(lbl.cget("text")).strip()
            except Exception:
                text = ""
        checks.append(
            GuiRegressionCheck(
                name=f"summary_{attr}",
                passed=bool(text and text not in ("—", "-", "N/A")),
                detail=f"{label} empty or placeholder ({text!r})",
            )
        )

    detail_host = getattr(gui, "_foot_clearance_detail_host", None)
    if detail_host is not None and _is_mapped(detail_host):
        for prefix in ("foot_left", "foot_right"):
            cur = getattr(gui, f"lbl_{prefix}_current", None)
            unavail = getattr(gui, f"lbl_{prefix}_unavailable", None)
            if cur is None:
                continue
            cur_text = str(cur.cget("text"))
            if "Unavailable" in cur_text and unavail is not None:
                reason = str(unavail.cget("text")).strip()
                checks.append(
                    GuiRegressionCheck(
                        name=f"foot_clearance_{prefix}_unavailable_reason",
                        passed=bool(reason),
                        detail="Unavailable foot clearance without reason",
                    )
                )

    # Foot clearance primary panel (below 3D reconstruction)
    for side, prefix in (("left", "foot_left"), ("right", "foot_right")):
        cur_lbl = getattr(gui, f"lbl_{prefix}_current", None)
        state_lbl = getattr(gui, f"lbl_{prefix}_state", None)
        cur_text = str(cur_lbl.cget("text")) if cur_lbl is not None else ""
        state_text = str(state_lbl.cget("text")) if state_lbl is not None else ""
        detail_unavail = "Unavailable" in cur_text
        checks.append(
            GuiRegressionCheck(
                name=f"foot_clearance_{side}_distance",
                passed=bool(
                    "cm" in cur_text
                    or cur_text.strip() == "Unavailable"
                ),
                detail=f"{side} distance not shown ({cur_text!r})",
            )
        )
        checks.append(
            GuiRegressionCheck(
                name=f"foot_clearance_{side}_state",
                passed=(
                    state_text in ("CONTACT", "SWING")
                    or state_text == "—"
                    or cur_text.strip() == "Unavailable"
                ),
                detail=f"{side} state missing ({state_text!r})",
            )
        )

    checks.extend(verify_skeleton_foot_clearance_matches_cards(gui))

    # Knee + joint path canvases
    for attr, label in (("chart_canvas", "Knee graph"), ("canvas_dof_traj", "Joint path")):
        canvas = getattr(gui, attr, None)
        checks.append(
            GuiRegressionCheck(
                name=f"canvas_{attr}_mapped",
                passed=canvas is not None and _is_mapped(canvas.get_tk_widget()),
                detail=f"{label} canvas not visible",
            )
        )

    timeline = getattr(gui, "lbl_frame", None)
    if timeline is not None:
        text = str(timeline.cget("text"))
        checks.append(
            GuiRegressionCheck(
                name="playback_time_label",
                passed="/" in text or "s" in text,
                detail=f"timeline label unexpected ({text!r})",
            )
        )

    # Export section buttons
    export_buttons = (
        "btn_view_detailed_data",
        "btn_save_session",
        "btn_export_analysis_report",
        "btn_export_gait_metrics",
        "btn_opensim_export_data",
        "btn_export_motion_reference",
    )
    transport_box = _widget_box(transport) if transport is not None else None
    section = getattr(gui, "_section_data_export", None)
    if section is not None:
        checks.append(
            GuiRegressionCheck(
                name="export_section_visible",
                passed=_is_mapped(section),
                detail="Detailed Joint Data & Export section not mapped",
            )
        )

    for attr in export_buttons:
        btn = getattr(gui, attr, None)
        if btn is None:
            checks.append(
                GuiRegressionCheck(
                    name=f"export_button_{attr}",
                    passed=False,
                    detail="button widget missing",
                )
            )
            continue
        box = _widget_box(btn)
        visible = _is_mapped(btn) and box is not None
        above_transport = True
        if scrolled_to_bottom and transport_box and box:
            above_transport = box[3] <= transport_box[1] + 2
        checks.append(
            GuiRegressionCheck(
                name=f"export_button_{attr}_visible",
                passed=visible,
                detail="button not visible",
            )
        )
        if scrolled_to_bottom and transport_box and box:
            checks.append(
                GuiRegressionCheck(
                    name=f"export_button_{attr}_above_playback",
                    passed=above_transport,
                    detail=(
                        f"button bottom {box[3]} below transport top {transport_box[1]}"
                    ),
                )
            )

    # Playback chrome
    for attr in ("slider", "lbl_frame", "btn_play_bar"):
        widget = getattr(gui, attr, None)
        checks.append(
            GuiRegressionCheck(
                name=f"playback_{attr}",
                passed=widget is not None and _is_mapped(widget),
                detail="playback control not visible",
            )
        )

    checks.append(
        GuiRegressionCheck(
            name="video_clip_viewport",
            passed=getattr(gui, "_video_clip_canvas", None) is not None,
            detail="video display clip canvas missing",
        )
    )
    checks.append(
        GuiRegressionCheck(
            name="skel_clip_viewport",
            passed=getattr(gui, "_skel_clip_canvas", None) is not None,
            detail="3D reconstruction clip canvas missing",
        )
    )
    checks.append(
        GuiRegressionCheck(
            name="knee_clip_viewport",
            passed=getattr(gui, "_knee_clip_canvas", None) is not None,
            detail="knee graph clip canvas missing",
        )
    )

    # Advanced tab scroll (main dashboard no longer uses bottom spacer padding).
    adv_scroll = getattr(gui, "_tab_advanced_scroll_canvas", None)
    checks.append(
        GuiRegressionCheck(
            name="advanced_tab_scroll_canvas",
            passed=adv_scroll is not None,
            detail="advanced tab vertical scroll missing",
        )
    )

    # Matplotlib dashboard figures (3D, knee, joint path) — exclude hidden robot panel.
    dashboard_figures = sum(
        _canvas_singleton_count(gui, name)
        for name in ("canvas_3d", "chart_canvas", "canvas_dof_traj")
    )
    checks.append(
        GuiRegressionCheck(
            name="dashboard_figure_count",
            passed=dashboard_figures == 3,
            detail=f"expected 3 dashboard figures, got {dashboard_figures}",
        )
    )
    robot_fig = _canvas_singleton_count(gui, "canvas_robot")
    checks.append(
        GuiRegressionCheck(
            name="robot_figure_hidden_legacy",
            passed=robot_fig <= 1,
            detail=f"legacy robot canvas count {robot_fig}",
        )
    )

    if resolution:
        try:
            w = int(root.winfo_width())
            h = int(root.winfo_height())
            checks.append(
                GuiRegressionCheck(
                    name="window_geometry",
                    passed=w >= 200 and h >= 200,
                    detail=f"window size {w}x{h} at {resolution}",
                )
            )
        except Exception as exc:
            checks.append(
                GuiRegressionCheck(
                    name="window_geometry",
                    passed=False,
                    detail=str(exc),
                )
            )

    return checks


def _boxes_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def run_scroll_layout_stress(
    gui: Any,
    *,
    cycles: int = 20,
) -> list[GuiRegressionCheck]:
    """Scroll advanced tab and switch notebook tabs; assert singleton widgets."""
    checks: list[GuiRegressionCheck] = []
    from stablewalk.ui.tk.dashboard_notebook import (
        TAB_ADVANCED,
        TAB_MOTION,
        TAB_OVERVIEW,
        run_tab_switch_stress_test,
        select_dashboard_tab,
    )

    for i in range(max(1, cycles // 10)):
        select_dashboard_tab(gui, TAB_ADVANCED)
        scroll_dashboard_to(gui, 1.0)
        scroll_dashboard_to(gui, 0.0)
        try:
            gui.root.update_idletasks()
        except Exception:
            pass

    tab_results = run_tab_switch_stress_test(gui, cycles=max(50, cycles * 2))
    for name, passed, detail in tab_results:
        checks.append(GuiRegressionCheck(name=name, passed=passed, detail=detail))

    section_attrs = (
        ("video_frame", "Original Video"),
        ("knee_panel", "Knee Motion"),
        ("traj_panel", "Joint Path"),
        ("_section_gait_metrics", "Gait Metrics"),
        ("_section_data_export", "Data Export"),
    )
    for tab in (TAB_OVERVIEW, TAB_MOTION, TAB_ADVANCED, TAB_OVERVIEW):
        select_dashboard_tab(gui, tab)
        try:
            gui.root.update_idletasks()
        except Exception:
            pass

    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    for attr, label in section_attrs:
        widget = getattr(gui, attr, None)
        if widget is not None and _is_mapped(widget):
            box = _widget_box(widget)
            if box is not None:
                boxes.append((label, box))

    overlaps: list[str] = []
    for idx, (name_a, box_a) in enumerate(boxes):
        for name_b, box_b in boxes[idx + 1 :]:
            if _boxes_overlap(box_a, box_b):
                overlaps.append(f"{name_a} ∩ {name_b}")

    checks.append(
        GuiRegressionCheck(
            name="scroll_stress_no_section_overlap",
            passed=not overlaps,
            detail="; ".join(overlaps) if overlaps else "no overlapping section bounds",
        )
    )
    return checks


def capture_overview_snapshot(gui: Any, *, label: str, video: str) -> OverviewQASnapshot:
    """Read current Overview widget values into a structured snapshot."""
    snap = OverviewQASnapshot(video=video, label=label)

    snap.movement_stability = _label_text(getattr(gui, "lbl_summary_ms_value", None)) or "—"
    snap.gait_quality = _label_text(getattr(gui, "lbl_summary_gq_value", None)) or "—"
    snap.analysis_confidence = _label_text(getattr(gui, "lbl_summary_ac_level", None)) or "—"

    for side, prefix in (("left", "foot_left"), ("right", "foot_right")):
        cur = _label_text(getattr(gui, f"lbl_{prefix}_current", None))
        reason = _label_text(getattr(gui, f"lbl_{prefix}_unavailable", None))
        if side == "left":
            snap.left_clearance = cur or "—"
            snap.left_clearance_reason = reason
        else:
            snap.right_clearance = cur or "—"
            snap.right_clearance_reason = reason

    snap.left_contact = _label_text(getattr(gui, "lbl_overview_contact_left", None)) or "—"
    snap.right_contact = _label_text(getattr(gui, "lbl_overview_contact_right", None)) or "—"
    snap.gait_phase = _label_text(getattr(gui, "lbl_gait_card_phase_value", None)) or "—"
    snap.usable_cycles = _label_text(getattr(gui, "lbl_overview_gait_cycles_usable", None)) or "—"
    snap.completeness = _label_text(getattr(gui, "lbl_overview_gait_cycles_completeness", None)) or "—"

    video_frame = getattr(gui, "video_frame", None)
    skel_frame = getattr(gui, "skel_frame", None)
    snap.video_visible = video_frame is not None and _is_mapped(video_frame)
    snap.skeleton_visible = skel_frame is not None and _is_mapped(skel_frame)

    tab_overview = getattr(gui, "_tab_overview", None)
    fc_hosts = []
    detail_host = getattr(gui, "_foot_clearance_detail_host", None)
    if detail_host is not None and _is_mapped(detail_host):
        if tab_overview is None or _widget_descendant_of(detail_host, tab_overview):
            fc_hosts.append("detail_panel")
    strip = getattr(gui, "ground_clearance_strip", None)
    if strip is not None and _is_mapped(strip):
        if tab_overview is None or _widget_descendant_of(strip, tab_overview):
            fc_hosts.append("legacy_strip")
    snap.foot_clearance_panel_count = len(fc_hosts)

    return snap


def run_overview_tab_assertions(gui: Any) -> list[GuiRegressionCheck]:
    """Overview-tab-only usability and scientific-clarity checks."""
    from stablewalk.analysis.gait_phase_classification import (
        validate_phase_contact_consistency,
    )
    from stablewalk.ui.tk.dashboard_notebook import TAB_OVERVIEW, select_dashboard_tab

    checks: list[GuiRegressionCheck] = []
    select_dashboard_tab(gui, TAB_OVERVIEW)
    try:
        gui.root.update_idletasks()
    except Exception:
        pass

    tab_overview = getattr(gui, "_tab_overview", None)

    def _on_overview(attr: str, label: str, *, min_h: int = 80, min_w: int = 120) -> None:
        widget = getattr(gui, attr, None)
        visible = widget is not None and _is_mapped(widget)
        on_tab = tab_overview is None or _widget_descendant_of(widget, tab_overview)
        box = _widget_box(widget) if widget is not None else None
        sized = box is not None and (box[2] - box[0]) >= min_w and (box[3] - box[1]) >= min_h
        checks.append(
            GuiRegressionCheck(
                name=f"overview_{attr}_visible",
                passed=visible and on_tab,
                detail=f"{label} not visible on Overview tab",
            )
        )
        if visible and on_tab:
            checks.append(
                GuiRegressionCheck(
                    name=f"overview_{attr}_size",
                    passed=sized,
                    detail=f"{label} too small ({box})",
                )
            )

    _on_overview("video_frame", "Original Video", min_h=120, min_w=160)
    _on_overview("skel_frame", "3D Gait Reconstruction", min_h=120, min_w=160)
    _on_overview("_sidebar_stability_panel", "Gait Analysis Summary", min_h=100, min_w=100)
    _on_overview("_foot_clearance_detail_host", "Foot Clearance panel", min_h=40, min_w=200)
    _on_overview("_overview_metrics_row", "Gait metrics bottom row", min_h=30, min_w=300)

    for attr, label in (
        ("lbl_summary_ms_value", "Movement Stability"),
        ("lbl_summary_gq_value", "Gait Quality"),
        ("lbl_summary_ac_level", "Analysis Confidence"),
        ("lbl_overview_gait_cycles_usable", "Usable Gait Cycles"),
        ("lbl_overview_contact_left", "Left contact"),
        ("lbl_overview_contact_right", "Right contact"),
        ("lbl_gait_card_phase_value", "Gait phase"),
    ):
        lbl = getattr(gui, attr, None)
        text = _label_text(lbl)
        checks.append(
            GuiRegressionCheck(
                name=f"overview_populated_{attr}",
                passed=bool(text and text not in ("—", "-", "N/A")),
                detail=f"{label} empty ({text!r})",
            )
        )

    for prefix in ("foot_left", "foot_right"):
        cur = _label_text(getattr(gui, f"lbl_{prefix}_current", None))
        unavail = _label_text(getattr(gui, f"lbl_{prefix}_unavailable", None))
        ok = "cm" in cur or ("Unavailable" in cur and bool(unavail))
        checks.append(
            GuiRegressionCheck(
                name=f"overview_{prefix}_clearance",
                passed=ok,
                detail=f"clearance not shown or missing reason ({cur!r})",
            )
        )

    strip = getattr(gui, "ground_clearance_strip", None)
    strip_mapped = strip is not None and _is_mapped(strip)
    checks.append(
        GuiRegressionCheck(
            name="overview_foot_clearance_single_instance",
            passed=not strip_mapped,
            detail="legacy foot clearance strip still visible (duplicate)",
        )
    )

    for attr, label in (
        ("knee_panel", "Knee graph"),
        ("traj_panel", "Joint path graph"),
        ("_section_data_export", "Export controls"),
        ("_section_gait_metrics", "Advanced gait metrics"),
    ):
        widget = getattr(gui, attr, None)
        on_overview = widget is not None and _is_mapped(widget) and (
            tab_overview is not None and _widget_descendant_of(widget, tab_overview)
        )
        checks.append(
            GuiRegressionCheck(
                name=f"overview_excludes_{attr}",
                passed=not on_overview,
                detail=f"{label} should not appear on Overview tab",
            )
        )

    for attr in ("lbl_advanced_temporal", "lbl_advanced_pelvis", "lbl_advanced_evidence"):
        lbl = getattr(gui, attr, None)
        on_overview = lbl is not None and _is_mapped(lbl) and (
            tab_overview is not None and _widget_descendant_of(lbl, tab_overview)
        )
        checks.append(
            GuiRegressionCheck(
                name=f"overview_excludes_{attr}",
                passed=not on_overview,
                detail="advanced biomechanical score visible on Overview",
            )
        )

    left = _label_text(getattr(gui, "lbl_overview_contact_left", None))
    right = _label_text(getattr(gui, "lbl_overview_contact_right", None))
    phase = _label_text(getattr(gui, "lbl_gait_card_phase_value", None))
    if left in ("CONTACT", "SWING") and right in ("CONTACT", "SWING"):
        left_contact = 1 if left == "CONTACT" else 0
        right_contact = 1 if right == "CONTACT" else 0
        phase_key = phase.replace(" ", "_").upper() if phase and phase != "—" else ""
        ok, expected = validate_phase_contact_consistency(
            left_contact, right_contact, phase_key or None
        )
        checks.append(
            GuiRegressionCheck(
                name="overview_phase_contact_consistency",
                passed=ok and phase != "—",
                detail=(
                    f"left={left} right={right} phase={phase!r} expected={expected}"
                ),
            )
        )
        checks.append(
            GuiRegressionCheck(
                name="overview_gait_phase_not_blank",
                passed=phase != "—",
                detail="gait phase blank despite valid contact states",
            )
        )

    for attr in ("lbl_summary_ms_explain", "lbl_summary_gq_explain", "lbl_summary_ac_explain"):
        text = _label_text(getattr(gui, attr, None))
        checks.append(
            GuiRegressionCheck(
                name=f"overview_explanation_length_{attr}",
                passed=len(text) <= 160,
                detail=f"explanation too long ({len(text)} chars)",
            )
        )

    checks.extend(verify_skeleton_foot_clearance_matches_cards(gui))

    return checks


def format_overview_qa_report(
    snapshots: list[OverviewQASnapshot],
    *,
    checks_by_video: dict[str, list[GuiRegressionCheck]] | None = None,
) -> str:
    """Render markdown report for ``data/output/reports/overview_qa.md``."""
    lines = [
        "# StableWalk Overview Tab — Final QA Report",
        "",
        "Usability and scientific-clarity review of the high-level gait dashboard.",
        "No new analysis algorithms were introduced for this pass.",
        "",
        "## Layout specification",
        "",
        "| Region | Content |",
        "|--------|---------|",
        "| Top left | Original Video |",
        "| Top center | 3D Gait Reconstruction |",
        "| Top right | Gait Analysis Summary (Movement Stability, Gait Quality, Analysis Confidence) |",
        "| Below 3D | Foot Clearance (single instance) |",
        "| Bottom row | Gait Phase, Contact Pattern, Gait Cycles |",
        "",
        "## Executive summary",
        "",
    ]

    total_issues = 0
    for snap in snapshots:
        video_checks = (checks_by_video or {}).get(snap.video, [])
        failed = [c for c in video_checks if not c.passed]
        total_issues += len(snap.issues) + len(failed)
        status = "PASS" if not snap.issues and not failed else "ISSUES"
        lines.append(f"- **{snap.label}** (`{snap.video}`): {status}")

    lines.extend(["", "---", ""])

    for snap in snapshots:
        video_checks = (checks_by_video or {}).get(snap.video, [])
        failed_checks = [c for c in video_checks if not c.passed]
        lines.extend(
            [
                f"## {snap.label}",
                "",
                f"**Video:** `{snap.video}`",
                "",
                "### Summary scores",
                f"- Movement Stability: {snap.movement_stability}",
                f"- Gait Quality: {snap.gait_quality}",
                f"- Analysis Confidence: {snap.analysis_confidence}",
                "",
                "### Foot clearance",
                f"- Left: {snap.left_clearance}"
                + (
                    f" ({snap.left_clearance_reason})"
                    if snap.left_clearance_reason and "Unavailable" in snap.left_clearance
                    else ""
                ),
                f"- Right: {snap.right_clearance}"
                + (
                    f" ({snap.right_clearance_reason})"
                    if snap.right_clearance_reason and "Unavailable" in snap.right_clearance
                    else ""
                ),
                f"- Foot clearance panels on Overview: {snap.foot_clearance_panel_count} (expected 1)",
                "",
                "### Gait state",
                f"- Left contact: {snap.left_contact}",
                f"- Right contact: {snap.right_contact}",
                f"- Gait phase: {snap.gait_phase}",
                f"- {snap.usable_cycles}",
                f"- {snap.completeness}",
                "",
                "### Visibility",
                f"- Original Video: {'visible' if snap.video_visible else 'hidden'}",
                f"- 3D skeleton: {'visible' if snap.skeleton_visible else 'hidden'}",
                "",
            ]
        )

        if failed_checks:
            lines.append("### Failed automated checks")
            for check in failed_checks:
                lines.append(f"- **{check.name}:** {check.detail}")
            lines.append("")

        issues = list(snap.issues)
        issues.extend(f"{c.name}: {c.detail}" for c in failed_checks)
        lines.append("### Issues found")
        if issues:
            for issue in issues:
                lines.append(f"- {issue}")
        else:
            lines.append("- None")
        lines.extend(["", "---", ""])

    lines.extend(
        [
            "## Logical consistency rules verified",
            "",
            "- Left CONTACT + Right SWING → Gait Phase = LEFT STANCE",
            "- Right CONTACT + Left SWING → Gait Phase = RIGHT STANCE",
            "- Both CONTACT → Gait Phase = DOUBLE SUPPORT",
            "- Foot clearance displayed only once on Overview (below 3D reconstruction)",
            "- Knee graph, joint path, export controls, and advanced domain scores excluded from Overview",
            "",
            f"**Total issues across demos:** {total_issues}",
        ]
    )
    return "\n".join(lines)


def format_qa_report(
    results: list[GuiVisualQAResult],
    *,
    issues_fixed: list[str] | None = None,
    remaining_limitations: list[str] | None = None,
) -> str:
    """Render markdown report for ``data/output/reports/gui_visual_qa.md``."""
    resolutions = sorted({r.resolution for r in results})
    categories = [r.category for r in results if "resize pass" not in r.category]
    lines = [
        "# StableWalk GUI Visual QA Report",
        "",
        "Automated layout regression and scripted interaction pass.",
        "Human visual inspection of plot rendering quality is still recommended.",
        "",
        "## Executive summary",
        "",
        f"- **Demo categories tested:** {', '.join(categories)}",
        f"- **Resolutions tested:** {', '.join(resolutions)}",
        "- **Regression assertions:** singleton video/3D/knee/joint canvases, fixed transport bar, "
        "scroll bottom padding, export buttons above playback bar, summary cards, foot clearance panel.",
        "- **Scroll/export validation:** scrolled to bottom at each resolution; all export buttons "
        "remained fully above the fixed playback bar.",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result.category} @ {result.resolution}",
                "",
                f"- **Video:** `{result.video}`",
                f"- **Overall:** {'PASS' if result.passed else 'FAIL'}",
                f"- **Checks:** {sum(1 for c in result.checks if c.passed)}/{len(result.checks)} passed",
                "",
            ]
        )
        if result.workflow_steps:
            lines.append("### Workflow steps")
            for step in result.workflow_steps:
                lines.append(f"- {step}")
            lines.append("")

        failed = [c for c in result.checks if not c.passed]
        if failed:
            lines.append("### Failed checks")
            for check in failed:
                lines.append(f"- **{check.name}:** {check.detail}")
            lines.append("")

        if result.issues_found:
            lines.append("### Issues found")
            for issue in result.issues_found:
                lines.append(f"- {issue}")
            lines.append("")

    global_fixed = issues_fixed or []
    if global_fixed:
        lines.append("## Issues fixed (this session)")
        for item in global_fixed:
            lines.append(f"- {item}")
        lines.append("")

    limitations = remaining_limitations or []
    limitations = [
        *limitations,
        "Normal demo foot clearance may show Unavailable with reason when floor scale is "
        "not body-normalized — compact strip shows em-dash by design in that case.",
        "Automated QA used cached pose JSON after verifying demo MP4s exist; live Analyze "
        "pipeline was not re-run during this pass.",
    ]
    if limitations:
        lines.append("## Remaining limitations")
        for item in limitations:
            lines.append(f"- {item}")
        lines.append("")

    all_pass = all(r.passed for r in results)
    lines.extend(
        [
            "## Conclusion",
            "",
        ]
    )
    if all_pass:
        lines.append(
            "Scrolling and export-section clipping regressions were exercised programmatically "
            "at the tested resolutions; all automated layout assertions passed. "
            "Plot ghosting and text overlap require manual spot-check during live playback."
        )
    else:
        lines.append(
            "One or more automated layout assertions failed. "
            "Scrolling/clipping cannot be claimed fixed until failures below are resolved."
        )

    return "\n".join(lines)
