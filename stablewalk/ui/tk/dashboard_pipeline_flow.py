"""Horizontal Real-to-Sim pipeline flow diagram for the Advanced tab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import tkinter as tk
from tkinter import ttk

from stablewalk import config
from stablewalk.monitoring.pipeline_status import (
    STATUS_COMPLETED,
    STATUS_LABEL,
    STATUS_PARTIAL,
    STATUS_SYMBOL,
    STATUS_UNAVAILABLE,
    PipelineStageItem,
    PipelineStatusReport,
)
from stablewalk.ui.scientific_labels import LABEL_VIRTUAL_GRF_PANEL
from stablewalk.ui.theme import (
    BORDER,
    ELEVATED,
    FONT_FLOW_ARROW,
    FONT_FLOW_ICON,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    PAD_XS,
    PANEL,
    TEXT,
    create_tooltip,
)
from stablewalk.ui.tk.dashboard_pipeline_visual import (
    PIPELINE_STATUS_FG,
    PIPELINE_STATUS_RING,
    animate_connector_activation,
    animate_stage_completion,
    install_progress_track,
    set_connector_state,
    set_progress_fill,
)
from stablewalk.ui.tk.pipeline_stage_dialog import (
    bind_pipeline_stage_click,
    resolve_stage_ui_spec,
)

if TYPE_CHECKING:
    from stablewalk.monitoring.pipeline_status import PipelineStageState
else:
    PipelineStageState = str

FLOW_STATUS_FG = PIPELINE_STATUS_FG
FLOW_STATUS_RING = PIPELINE_STATUS_RING

FLOW_STAGE_SPECS: tuple[tuple[str, str, str, tuple[str, ...], str], ...] = (
    (
        "video",
        "Video",
        "\u25b6",
        ("pose_mediapipe",),
        "Load walking video and extract 2D/3D pose landmarks.",
    ),
    (
        "smpl",
        "SMPL",
        "\u25ce",
        ("pose_smpl",),
        "Optional monocular SMPL body mesh inference.",
    ),
    (
        "retargeting",
        "Retargeting",
        "\u2194",
        ("retarget_scale", "retarget_joints"),
        "Scale human motion to humanoid robot proportions.",
    ),
    (
        "opensim",
        "OpenSim",
        "\u2699",
        ("osim_model", "osim_ik", "osim_mapping"),
        "Musculoskeletal IK and OpenSim motion export.",
    ),
    (
        "biomechanics",
        "Biomechanics",
        "\u2301",
        ("bio_contact_mask", "bio_cadence", "bio_com", "bio_bos"),
        "Gait events, COM, cadence, and stability metrics.",
    ),
    (
        "virtual_grf",
        LABEL_VIRTUAL_GRF_PANEL,
        "\u2193",
        ("bio_vgrf",),
        "Estimated vertical ground reaction force proxy.",
    ),
    (
        "amp_dataset",
        "AMP Dataset",
        "\u25a3",
        ("isaac_export_only",),
        "AMP reference motion for Isaac Lab training.",
    ),
    (
        "export",
        "Export",
        "\u21a9",
        ("retarget_export",),
        "Write stablewalk_motion.npz and session artifacts.",
    ),
)


@dataclass(frozen=True)
class FlowStageView:
    key: str
    title: str
    icon: str
    status: PipelineStageState
    detail: str
    tooltip: str


def _combine_details(items: list[PipelineStageItem], *, max_parts: int = 2) -> str:
    if not items:
        return "Not executed in this session"
    parts = [item.detail for item in items[:max_parts] if item.detail]
    if len(items) > max_parts:
        parts.append(f"+{len(items) - max_parts} more")
    return " · ".join(parts) if parts else "Not executed in this session"


def _aggregate_item_status(
    items_by_key: dict[str, PipelineStageItem],
    keys: tuple[str, ...],
) -> tuple[PipelineStageState, str]:
    matched = [items_by_key[k] for k in keys if k in items_by_key]
    if not matched:
        return STATUS_UNAVAILABLE, "Stage not assessed"

    active = [item for item in matched if item.status != STATUS_UNAVAILABLE]
    if not active:
        return STATUS_UNAVAILABLE, _combine_details(matched)

    statuses = [item.status for item in active]
    if all(status == STATUS_COMPLETED for status in statuses):
        return STATUS_COMPLETED, _combine_details(active)

    if any(status == STATUS_COMPLETED for status in statuses):
        incomplete = [item for item in active if item.status != STATUS_COMPLETED]
        return STATUS_PARTIAL, _combine_details(incomplete or active)

    if any(status == STATUS_PARTIAL for status in statuses):
        return STATUS_PARTIAL, _combine_details(active)

    return STATUS_UNAVAILABLE, _combine_details(matched)


def collect_flow_stage_views(
    report: PipelineStatusReport | None,
    *,
    gui=None,
) -> list[FlowStageView]:
    """Build horizontal flow stage views from an honest pipeline report."""
    items_by_key: dict[str, PipelineStageItem] = {}
    if report is not None:
        items_by_key = {item.key: item for item in report.all_items()}

    views: list[FlowStageView] = []
    for key, title, icon, item_keys, default_tip in FLOW_STAGE_SPECS:
        status, detail = _aggregate_item_status(items_by_key, item_keys)
        tooltip = default_tip

        if key == "amp_dataset" and gui is not None:
            run_name = "session"
            if hasattr(gui, "_resolve_session_video_source"):
                run_name = Path(gui._resolve_session_video_source() or "session").stem or "session"
            amp_path = config.MOTION_REFERENCE_EXPORT_DIR / run_name / "amp_reference_motion.npz"
            if amp_path.is_file():
                status = STATUS_COMPLETED
                detail = f"AMP reference on disk ({amp_path.name})"
            elif status == STATUS_UNAVAILABLE:
                detail = "Export AMP reference or run Real-to-Sim Pipeline"

        if key == "export" and gui is not None and report is not None:
            run_name = report.run_name or "session"
            motion_path = config.MOTION_REFERENCE_EXPORT_DIR / run_name / "stablewalk_motion.npz"
            if motion_path.is_file() and status != STATUS_COMPLETED:
                status = STATUS_PARTIAL
                detail = f"Motion bundle available ({motion_path.name})"

        matched = [items_by_key[k] for k in item_keys if k in items_by_key]
        if matched:
            tooltip = matched[0].tooltip

        views.append(
            FlowStageView(
                key=key,
                title=title,
                icon=icon,
                status=status,
                detail=detail[:96],
                tooltip=tooltip,
            )
        )
    return views


def _status_badge(status: PipelineStageState) -> str:
    return f"{STATUS_SYMBOL[status]} {STATUS_LABEL[status]}"


def _progress_fraction(status: PipelineStageState) -> float:
    if status == STATUS_COMPLETED:
        return 1.0
    if status == STATUS_PARTIAL:
        return 0.55
    return 0.0


def _build_flow_arrow(parent: tk.Misc, *, column: int, gui, connector_key: str) -> tk.Label:
    arrow = tk.Label(
        parent,
        text="\u2192",
        bg=PANEL,
        fg=MUTED,
        font=FONT_FLOW_ARROW,
    )
    arrow.grid(row=0, column=column, sticky="nsew", padx=(0, 2))
    connectors = getattr(gui, "_flow_connector_widgets", {})
    connectors[connector_key] = arrow
    gui._flow_connector_widgets = connectors
    return arrow


def _toggle_flow_expanded(gui, stage_key: str) -> None:
    widgets = getattr(gui, "_flow_stage_widgets", {}).get(stage_key)
    if widgets is None:
        return
    expanded = not bool(widgets.get("expanded", False))
    widgets["expanded"] = expanded
    detail_extra = widgets.get("detail_extra")
    toggle = widgets.get("toggle")
    if detail_extra is not None:
        if expanded:
            detail_extra.grid()
        else:
            detail_extra.grid_remove()
    if toggle is not None:
        toggle.configure(text="\u25bc" if expanded else "\u25b6")
    sync = getattr(gui, "_sync_pipeline_flow_layout", None)
    if sync is not None:
        sync()


def _build_flow_node(
    parent: tk.Misc,
    *,
    column: int,
    gui,
    stage_key: str,
    title: str,
    icon: str,
) -> tk.Frame:
    card = tk.Frame(
        parent,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
    card.grid(row=0, column=column, sticky="nsew", padx=(0, 2))
    card.columnconfigure(0, weight=1)

    body = tk.Frame(card, bg=ELEVATED, highlightthickness=0)
    body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 4))
    body.columnconfigure(0, weight=1)

    header = tk.Frame(body, bg=ELEVATED)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(1, weight=1)

    toggle = tk.Label(
        header,
        text="\u25b6",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        cursor="hand2",
    )
    toggle.grid(row=0, column=0, sticky="w", padx=(0, 2))

    icon_lbl = tk.Label(
        header,
        text=icon,
        bg=ELEVATED,
        fg=FLOW_STATUS_FG[STATUS_UNAVAILABLE],
        font=FONT_FLOW_ICON,
        anchor="center",
    )
    icon_lbl.grid(row=0, column=1, sticky="ew")

    tk.Label(
        body,
        text=title,
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="center",
    ).grid(row=1, column=0)

    status_lbl = tk.Label(
        body,
        text=_status_badge(STATUS_UNAVAILABLE),
        bg=ELEVATED,
        fg=FLOW_STATUS_FG[STATUS_UNAVAILABLE],
        font=FONT_UI_XS,
        anchor="center",
    )
    status_lbl.grid(row=2, column=0, pady=(4, 2))

    detail_lbl = tk.Label(
        body,
        text="—",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="center",
        justify=tk.CENTER,
        wraplength=88,
    )
    detail_lbl.grid(row=3, column=0, sticky="ew")

    track, fill = install_progress_track(body, row=4)
    track.grid_configure(pady=(PAD_XS, 0))

    spec = resolve_stage_ui_spec(stage_key)
    detail_extra = tk.Frame(body, bg=ELEVATED)
    detail_extra.grid(row=5, column=0, sticky="ew", pady=(4, 0))
    detail_extra.columnconfigure(0, weight=1)

    purpose_lbl = tk.Label(
        detail_extra,
        text=spec.purpose,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=100,
    )
    purpose_lbl.grid(row=0, column=0, sticky="ew")

    algo_lbl = tk.Label(
        detail_extra,
        text=f"Algorithms: {spec.algorithms}",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=100,
    )
    algo_lbl.grid(row=1, column=0, sticky="ew", pady=(2, 0))

    detail_extra.grid_remove()

    ring = tk.Frame(card, bg=FLOW_STATUS_RING[STATUS_UNAVAILABLE], height=3)
    ring.pack(fill=tk.X, side=tk.BOTTOM)

    def _on_toggle(_event: object | None = None) -> str:
        _toggle_flow_expanded(gui, stage_key)
        return "break"

    toggle.bind("<Button-1>", _on_toggle)

    gui._flow_stage_widgets[stage_key] = {
        "card": card,
        "icon": icon_lbl,
        "status": status_lbl,
        "detail": detail_lbl,
        "ring": ring,
        "progress_fill": fill,
        "toggle": toggle,
        "detail_extra": detail_extra,
        "purpose": purpose_lbl,
        "algorithms": algo_lbl,
        "expanded": False,
        "last_status": STATUS_UNAVAILABLE,
    }

    bind_pipeline_stage_click(card, gui, stage_key, skip_widgets=(toggle,))
    return card


def build_pipeline_flow_diagram(gui, parent: tk.Misc) -> tk.Frame:
    """Install the horizontal Real-to-Sim flow diagram."""
    host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    host.columnconfigure(0, weight=1)

    hint = tk.Label(
        host,
        text="Click a stage for details · ▶ expands inline summary",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    hint.grid(row=0, column=0, sticky="ew", pady=(0, 4))

    canvas = tk.Canvas(host, bg=PANEL, highlightthickness=0, borderwidth=0, height=132)
    canvas.grid(row=1, column=0, sticky="ew")

    inner = tk.Frame(canvas, bg=PANEL, highlightthickness=0)
    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    stage_count = len(FLOW_STAGE_SPECS)
    for index in range(stage_count):
        inner.columnconfigure(index * 2, weight=1, uniform="flow_stage")
        if index < stage_count - 1:
            inner.columnconfigure(index * 2 + 1, weight=0)

    gui._flow_stage_widgets: dict[str, dict[str, tk.Misc]] = {}
    gui._flow_connector_widgets: dict[str, tk.Misc] = {}

    for index, (key, title, icon, _keys, _tip) in enumerate(FLOW_STAGE_SPECS):
        col = index * 2
        _build_flow_node(
            inner,
            column=col,
            gui=gui,
            stage_key=key,
            title=title,
            icon=icon,
        )
        if index < stage_count - 1:
            next_key = FLOW_STAGE_SPECS[index + 1][0]
            _build_flow_arrow(
                inner,
                column=col + 1,
                gui=gui,
                connector_key=f"{key}->{next_key}",
            )

    def _sync_flow_layout(_event: object | None = None) -> None:
        try:
            canvas.update_idletasks()
            inner.update_idletasks()
            canvas_width = max(canvas.winfo_width(), 1)
            inner_width = max(inner.winfo_reqwidth(), canvas_width)
            canvas.itemconfigure(window_id, width=inner_width)
            canvas.configure(
                scrollregion=canvas.bbox("all"),
                height=max(128, inner.winfo_reqheight() + 4),
            )

            if inner_width > canvas_width + 4:
                canvas.configure(xscrollcommand=hbar.set)
                hbar.grid(row=2, column=0, sticky="ew")
            else:
                canvas.configure(xscrollcommand=lambda *_a: None)
                hbar.grid_remove()
                canvas.xview_moveto(0)

            cell_wrap = max(72, (canvas_width - (stage_count - 1) * 18) // stage_count - 8)
            for widgets in gui._flow_stage_widgets.values():
                for attr in ("detail", "purpose", "algorithms"):
                    lbl = widgets.get(attr)
                    if lbl is not None and hasattr(lbl, "configure"):
                        lbl.configure(wraplength=cell_wrap)
        except tk.TclError:
            pass

    hbar = ttk.Scrollbar(host, orient=tk.HORIZONTAL, command=canvas.xview)
    canvas.configure(xscrollcommand=hbar.set)
    hbar.grid(row=2, column=0, sticky="ew")
    hbar.grid_remove()

    inner.bind("<Configure>", _sync_flow_layout, add="+")
    canvas.bind("<Configure>", _sync_flow_layout, add="+")
    host.bind("<Configure>", _sync_flow_layout, add="+")

    gui._pipeline_flow_host = host
    gui._pipeline_flow_canvas = canvas
    gui._pipeline_flow_inner = inner
    gui._sync_pipeline_flow_layout = _sync_flow_layout
    return host


def _apply_flow_stage(gui, view: FlowStageView) -> None:
    widgets = getattr(gui, "_flow_stage_widgets", {}).get(view.key)
    if widgets is None:
        return

    color = FLOW_STATUS_FG[view.status]
    ring_color = FLOW_STATUS_RING[view.status]
    prev_status = widgets.get("last_status")

    icon_lbl = widgets.get("icon")
    if icon_lbl is not None:
        icon_lbl.configure(fg=color, text=view.icon)

    status_lbl = widgets.get("status")
    if status_lbl is not None:
        status_lbl.configure(text=_status_badge(view.status), fg=color)

    detail_lbl = widgets.get("detail")
    if detail_lbl is not None:
        detail_lbl.configure(text=view.detail, fg=TEXT if view.status == STATUS_COMPLETED else MUTED)

    ring = widgets.get("ring")
    if ring is not None:
        ring.configure(bg=ring_color)

    fill = widgets.get("progress_fill")
    set_progress_fill(fill, _progress_fraction(view.status), color=color)

    if view.status == STATUS_COMPLETED and prev_status != STATUS_COMPLETED:
        animate_stage_completion(gui, view.key, widgets, status=view.status)
    elif view.status != STATUS_COMPLETED:
        animate_stage_completion(gui, view.key, widgets, status=view.status)

    widgets["last_status"] = view.status

    tip = (
        f"{view.title}\n\n{view.tooltip}\n\n{view.detail}\n\n"
        "Click for full stage details · ▶ expands summary"
    )
    card = widgets.get("card")
    for widget in (card, icon_lbl, status_lbl, detail_lbl):
        if widget is not None:
            create_tooltip(widget, tip)


def _update_flow_connectors(gui, views: list[FlowStageView]) -> None:
    by_key = {view.key: view for view in views}
    connectors = getattr(gui, "_flow_connector_widgets", {})
    prev_active = getattr(gui, "_flow_connector_active", {})
    new_active: dict[str, bool] = {}

    for index in range(len(FLOW_STAGE_SPECS) - 1):
        key = FLOW_STAGE_SPECS[index][0]
        next_key = FLOW_STAGE_SPECS[index + 1][0]
        connector_key = f"{key}->{next_key}"
        widget = connectors.get(connector_key)
        upstream = by_key.get(key)
        active = upstream is not None and upstream.status == STATUS_COMPLETED
        new_active[connector_key] = active
        was_active = bool(prev_active.get(connector_key))
        if active and not was_active:
            animate_connector_activation(gui, connector_key, widget, active=True)
        else:
            set_connector_state(widget, active=active)

    gui._flow_connector_active = new_active


def update_pipeline_flow_diagram(gui) -> None:
    """Refresh flow nodes from the cached pipeline status report."""
    report = getattr(gui, "_pipeline_status_report_cache", None)
    views = collect_flow_stage_views(report, gui=gui)
    for view in views:
        _apply_flow_stage(gui, view)
    _update_flow_connectors(gui, views)

    sync = getattr(gui, "_sync_pipeline_flow_layout", None)
    if sync is not None:
        sync()


__all__ = [
    "FLOW_STAGE_SPECS",
    "FLOW_STATUS_FG",
    "FLOW_STATUS_RING",
    "FlowStageView",
    "build_pipeline_flow_diagram",
    "collect_flow_stage_views",
    "update_pipeline_flow_diagram",
]
