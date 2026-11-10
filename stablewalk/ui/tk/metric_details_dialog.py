"""Metric Details dialog — publication-style documentation for computed metrics."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from stablewalk.ui.metric_details import (
    MetricDetails,
    get_metric_details,
    list_metric_details,
)
from stablewalk.ui.theme import (
    BORDER,
    ELEVATED,
    FONT_HEADING,
    FONT_UI,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    PANEL,
    SURFACE,
    TEXT,
)


_SECTIONS: tuple[tuple[str, str], ...] = (
    ("mathematical_definition", "Mathematical definition"),
    ("coordinate_system", "Coordinate system"),
    ("units", "Units"),
    ("assumptions", "Assumptions"),
    ("limitations", "Limitations"),
    ("confidence_estimation", "Confidence estimation"),
    ("source", "Implementation"),
)


def show_overview_kpi_popup(
    parent: tk.Misc | None,
    key: str | None,
    *,
    gui: Any | None = None,
) -> None:
    """Small session popup for Overview Quality / Stability / Symmetry / Cadence."""
    from stablewalk.ui.overview_clinical_summary import build_overview_kpi_detail

    root = parent
    if root is not None:
        try:
            root = root.winfo_toplevel()
        except tk.TclError:
            root = parent

    payload = build_overview_kpi_detail(gui, key)
    dlg = tk.Toplevel(root)
    dlg.title(payload["title"])
    dlg.configure(bg=ELEVATED)
    dlg.transient(root)
    dlg.resizable(False, False)

    frame = tk.Frame(
        dlg,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

    tk.Label(
        frame,
        text=payload["title"],
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI,
        anchor="w",
    ).pack(fill=tk.X, padx=PAD_MD, pady=(PAD_SM, PAD_XS))

    tk.Label(
        frame,
        text=payload["body"],
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI_XS,
        anchor="nw",
        justify=tk.LEFT,
        wraplength=380,
    ).pack(fill=tk.X, padx=PAD_MD, pady=(0, PAD_SM))

    btn_row = tk.Frame(frame, bg=ELEVATED, highlightthickness=0)
    btn_row.pack(fill=tk.X, padx=PAD_MD, pady=(0, PAD_SM))
    ttk.Button(btn_row, text="Close", command=dlg.destroy, width=10).pack(side=tk.RIGHT)

    dlg.bind("<Escape>", lambda _e: dlg.destroy())
    try:
        dlg.update_idletasks()
        if parent is not None:
            x = parent.winfo_rootx()
            y = parent.winfo_rooty() + parent.winfo_height() + 6
            dlg.geometry(f"+{x}+{y}")
        elif root is not None:
            dlg.geometry(f"+{root.winfo_rootx() + 80}+{root.winfo_rooty() + 80}")
    except tk.TclError:
        pass
    dlg.focus_set()


def show_metric_details_dialog(
    parent: tk.Misc | None,
    key: str | None = None,
    *,
    title: str | None = None,
    gui: Any | None = None,
) -> None:
    """Open a scrollable Metric Details window for *key* (or a chooser if unknown)."""
    root = parent
    if root is not None:
        try:
            root = root.winfo_toplevel()
        except tk.TclError:
            root = parent

    details = get_metric_details(key)
    dlg = tk.Toplevel(root)
    dlg.title(title or (details.title if details else "Metric Details"))
    dlg.configure(bg=SURFACE)
    dlg.transient(root)
    dlg.resizable(True, True)
    dlg.minsize(520, 480)

    outer = tk.Frame(dlg, bg=SURFACE, highlightthickness=0)
    outer.pack(fill=tk.BOTH, expand=True, padx=PAD_MD, pady=PAD_MD)
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(3, weight=1)

    tk.Label(
        outer,
        text="Metric Details",
        bg=SURFACE,
        fg=TEXT,
        font=FONT_HEADING,
        anchor="w",
    ).grid(row=0, column=0, sticky="ew")

    tk.Label(
        outer,
        text=(
            "Formulas below match the StableWalk analysis implementation. "
            "Nothing here invents equations beyond the source code."
        ),
        bg=SURFACE,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=560,
        justify=tk.LEFT,
    ).grid(row=1, column=0, sticky="ew", pady=(2, PAD_SM))

    picker_row = tk.Frame(outer, bg=SURFACE, highlightthickness=0)
    picker_row.grid(row=2, column=0, sticky="ew", pady=(0, PAD_XS))

    catalog = list_metric_details()
    labels = [d.title for d in catalog]
    key_by_title = {d.title: d.key for d in catalog}
    initial = details.title if details else (labels[0] if labels else "")

    tk.Label(picker_row, text="Metric", bg=SURFACE, fg=MUTED, font=FONT_UI_XS).pack(
        side=tk.LEFT, padx=(0, PAD_XS)
    )
    var = tk.StringVar(value=initial)
    combo = ttk.Combobox(
        picker_row,
        textvariable=var,
        values=labels,
        state="readonly",
        width=48,
    )
    combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

    body_host = tk.Frame(outer, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
    body_host.grid(row=3, column=0, sticky="nsew", pady=(PAD_XS, PAD_SM))
    body_host.columnconfigure(0, weight=1)
    body_host.rowconfigure(0, weight=1)

    canvas = tk.Canvas(body_host, bg=PANEL, highlightthickness=0, borderwidth=0)
    vsb = ttk.Scrollbar(body_host, orient=tk.VERTICAL, command=canvas.yview)
    canvas.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    canvas.configure(yscrollcommand=vsb.set)

    content = tk.Frame(canvas, bg=PANEL, highlightthickness=0)
    win = canvas.create_window((0, 0), window=content, anchor="nw")
    content.columnconfigure(0, weight=1)

    def _on_configure(_event=None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfigure(win, width=max(canvas.winfo_width(), 1))

    content.bind("<Configure>", _on_configure)
    canvas.bind("<Configure>", _on_configure)

    section_widgets: list[tk.Widget] = []

    def _clear() -> None:
        for w in section_widgets:
            try:
                w.destroy()
            except tk.TclError:
                pass
        section_widgets.clear()

    def _render(md: MetricDetails | None) -> None:
        _clear()
        if md is None:
            lbl = tk.Label(
                content,
                text="No Metric Details are registered for this control.",
                bg=PANEL,
                fg=MUTED,
                font=FONT_UI_SM,
                anchor="w",
                justify=tk.LEFT,
                wraplength=500,
            )
            lbl.pack(fill=tk.X, padx=PAD_SM, pady=PAD_SM)
            section_widgets.append(lbl)
            return

        title_lbl = tk.Label(
            content,
            text=md.title,
            bg=PANEL,
            fg=TEXT,
            font=FONT_UI,
            anchor="w",
            justify=tk.LEFT,
            wraplength=500,
        )
        title_lbl.pack(fill=tk.X, padx=PAD_SM, pady=(PAD_SM, PAD_XS))
        section_widgets.append(title_lbl)

        for attr, heading in _SECTIONS:
            hdr = tk.Label(
                content,
                text=heading,
                bg=PANEL,
                fg=TEXT,
                font=FONT_UI_SM,
                anchor="w",
            )
            hdr.pack(fill=tk.X, padx=PAD_SM, pady=(PAD_SM, 2))
            body = tk.Label(
                content,
                text=getattr(md, attr),
                bg=PANEL,
                fg=MUTED if attr == "source" else TEXT,
                font=FONT_UI_XS,
                anchor="w",
                justify=tk.LEFT,
                wraplength=500,
            )
            body.pack(fill=tk.X, padx=PAD_SM, pady=(0, 2))
            section_widgets.extend((hdr, body))

        if gui is not None and md.key in {"gait_quality", "pelvis_stability", "symmetry"}:
            try:
                from stablewalk.ui.overview_clinical_summary import (
                    format_gait_quality_session_details,
                    format_pelvis_stability_session_details,
                    format_symmetry_session_details,
                )

                if md.key == "gait_quality":
                    live = format_gait_quality_session_details(gui)
                elif md.key == "symmetry":
                    live = format_symmetry_session_details(gui)
                else:
                    live = format_pelvis_stability_session_details(gui)
            except Exception:
                live = None
            if live:
                hdr = tk.Label(
                    content,
                    text="This session",
                    bg=PANEL,
                    fg=TEXT,
                    font=FONT_UI_SM,
                    anchor="w",
                )
                hdr.pack(fill=tk.X, padx=PAD_SM, pady=(PAD_SM, 2))
                body = tk.Label(
                    content,
                    text=live,
                    bg=PANEL,
                    fg=TEXT,
                    font=FONT_UI_XS,
                    anchor="w",
                    justify=tk.LEFT,
                    wraplength=500,
                )
                body.pack(fill=tk.X, padx=PAD_SM, pady=(0, 2))
                section_widgets.extend((hdr, body))

        canvas.yview_moveto(0.0)
        _on_configure()

    def _on_pick(_event=None) -> None:
        title_sel = var.get()
        k = key_by_title.get(title_sel)
        _render(get_metric_details(k))
        dlg.title(f"Metric Details — {title_sel}")

    combo.bind("<<ComboboxSelected>>", _on_pick)
    _render(details or (catalog[0] if catalog else None))

    btn_row = tk.Frame(outer, bg=SURFACE, highlightthickness=0)
    btn_row.grid(row=4, column=0, sticky="e")
    ttk.Button(btn_row, text="Close", command=dlg.destroy, width=10).pack(side=tk.RIGHT)

    dlg.bind("<Escape>", lambda _e: dlg.destroy())

    # Center near parent
    try:
        dlg.update_idletasks()
        if root is not None:
            x = root.winfo_rootx() + 80
            y = root.winfo_rooty() + 60
            dlg.geometry(f"560x580+{x}+{y}")
    except tk.TclError:
        pass

    dlg.focus_set()


def attach_metric_details_control(
    parent: tk.Misc,
    key: str,
    *,
    bg: str | None = None,
    side: str = tk.RIGHT,
    gui: Any | None = None,
    overview_kpi: bool = False,
) -> tk.Label:
    """Small 'Details' control.

    Overview KPI cards use a short session popup. Other surfaces keep the
    full Metric Details catalog.
    """
    fill = bg if bg is not None else ELEVATED
    btn = tk.Label(
        parent,
        text="Details",
        bg=fill,
        fg=MUTED,
        font=FONT_UI_XS,
        cursor="hand2",
        padx=4,
    )
    btn.pack(side=side)

    def _open(_event=None) -> None:
        if overview_kpi:
            show_overview_kpi_popup(parent, key, gui=gui)
            return
        show_metric_details_dialog(parent, key, gui=gui)

    btn.bind("<Button-1>", _open)
    try:
        from stablewalk.ui.theme import create_tooltip

        tip = (
            "Short explanation of this session value"
            if overview_kpi
            else "Open Metric Details (definition, units, confidence)"
        )
        create_tooltip(btn, tip, wraplength=280)
    except Exception:
        pass
    return btn


def open_metric_details_from_gui(gui: Any, key: str | None) -> None:
    """Convenience for menus / chart bars bound to a GUI instance."""
    parent = getattr(gui, "root", None) or gui
    show_metric_details_dialog(parent, key, gui=gui)


__all__ = [
    "attach_metric_details_control",
    "open_metric_details_from_gui",
    "show_metric_details_dialog",
    "show_overview_kpi_popup",
]
