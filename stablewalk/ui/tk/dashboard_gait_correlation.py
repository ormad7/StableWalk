"""Gait Correlation & Spectral Analysis — Motion inner section.

Reuses the Four Gait Signals workspace and the Asymmetry workspace. One
cached report per video; switching sub-tabs only re-renders.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from stablewalk.ui.theme import (
    ELEVATED,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT_SECONDARY,
    create_tooltip,
)
from stablewalk.ui.tk.gait_correlation_tabs import (
    ASYMMETRY_TAB,
    FOUR_GAIT_TAB_VIEWS,
    SECTION_TITLE,
    gait_corr_inner_title,
)

logger = logging.getLogger(__name__)

_SECTION_NOTE = (
    "Signals come from the currently loaded video (full-frame FPS grid). "
    "Tab switches reuse the cached report. Spectral-shape r is not time-domain "
    "C(lag). Similar dominant frequencies are not proof of symmetric gait. "
    "Not a clinical diagnosis."
)


def build_gait_correlation_section(gui: Any, parent: tk.Misc) -> tk.Frame:
    """Install the six-tab correlation / spectral section into ``parent``."""
    from stablewalk.ui.tk.dashboard_asymmetry import build_asymmetry_workspace
    from stablewalk.ui.tk.dashboard_four_gait import build_four_gait_workspace

    host = ttk.Frame(parent)
    host.grid(row=0, column=0, sticky="nsew")
    parent.rowconfigure(0, weight=1)
    parent.columnconfigure(0, weight=1)
    host.columnconfigure(0, weight=1)
    host.rowconfigure(2, weight=0)
    host.rowconfigure(3, weight=1)
    gui._gait_corr_host = host
    gui._four_gait_use_section_tabs = True

    header = tk.Frame(host, bg=ELEVATED, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_SM, PAD_XS))
    header.columnconfigure(1, weight=1)
    tk.Label(
        header,
        text=SECTION_TITLE,
        bg=ELEVATED,
        fg=TEXT_SECONDARY,
        font=FONT_UI_SM,
        anchor="w",
    ).grid(row=0, column=0, padx=(PAD_SM, PAD_XS), pady=PAD_XS)
    btn = ttk.Button(
        header,
        text="CSV",
        width=6,
        command=lambda: _export_section_csv(gui),
        takefocus=False,
    )
    btn.grid(row=0, column=2, padx=PAD_XS, pady=PAD_XS)
    create_tooltip(
        btn,
        "Export cached signals, correlation matrices, frequency metrics, "
        "and asymmetry scalars for the current video.",
    )

    note = tk.Label(
        host,
        text=_SECTION_NOTE,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=960,
    )
    note.grid(row=1, column=0, sticky="ew", padx=PAD_MD, pady=(0, PAD_XS))
    gui.lbl_gait_corr_note = note

    def _wrap(_event: object | None = None) -> None:
        try:
            width = int(host.winfo_width() or 0)
        except tk.TclError:
            return
        if width < 80:
            return
        try:
            note.configure(wraplength=max(220, width - 48))
        except tk.TclError:
            pass

    host.bind("<Configure>", _wrap, add="+")

    nb = ttk.Notebook(host)
    nb.grid(row=2, column=0, sticky="ew")
    gui._gait_corr_notebook = nb

    four_pages: dict[str, ttk.Frame] = {}
    for title in FOUR_GAIT_TAB_VIEWS:
        page = ttk.Frame(nb)
        nb.add(page, text=title)
        four_pages[title] = page
    gui._gait_corr_four_pages = four_pages

    asym_tab = ttk.Frame(nb)
    nb.add(asym_tab, text=ASYMMETRY_TAB)

    content = ttk.Frame(host)
    content.grid(row=3, column=0, sticky="nsew")
    content.columnconfigure(0, weight=1)
    content.rowconfigure(0, weight=1)
    gui._gait_corr_content = content

    four_host = ttk.Frame(content)
    four_host.grid(row=0, column=0, sticky="nsew")
    four_host.columnconfigure(0, weight=1)
    four_host.rowconfigure(0, weight=1)
    gui._gait_corr_four_host = four_host
    build_four_gait_workspace(gui, four_host)

    asym_host = ttk.Frame(content)
    asym_host.grid(row=0, column=0, sticky="nsew")
    asym_host.columnconfigure(0, weight=1)
    asym_host.rowconfigure(0, weight=1)
    build_asymmetry_workspace(gui, asym_host)
    gui._gait_corr_asym_page = asym_host

    def _on_inner(_event: object | None = None) -> None:
        _sync_section_tab(gui)

    def _shrink_tabs(_event: object | None = None) -> None:
        _shrink_tabstrip(nb)

    gui._sync_gait_corr_tab = lambda: _sync_section_tab(gui)
    nb.bind("<<NotebookTabChanged>>", _on_inner, add="+")
    nb.bind("<Map>", _shrink_tabs, add="+")
    host.bind("<Configure>", lambda _e: _shrink_tabstrip(nb), add="+")
    _sync_section_tab(gui, initial=True)
    try:
        host.after_idle(lambda: _shrink_tabstrip(nb))
    except tk.TclError:
        pass
    return host


def _shrink_tabstrip(nb: ttk.Notebook) -> None:
    """Keep the notebook as a tab strip so pages are not a second blank pane."""
    try:
        nb.update_idletasks()
        tabs = nb.tabs()
        if not tabs:
            return
        box = nb.bbox(tabs[0])
        if not box:
            return
        height = max(28, int(box[1] + box[3] + 4))
        nb.configure(height=height)
    except tk.TclError:
        pass


def _raise_section_body(gui: Any, which: str) -> None:
    four = getattr(gui, "_gait_corr_four_host", None)
    asym = getattr(gui, "_gait_corr_asym_page", None)
    target = four if which == "four" else asym
    other = asym if which == "four" else four
    if target is None:
        return
    try:
        target.tkraise()
        target.grid()
        target.update_idletasks()
    except tk.TclError:
        pass
    if other is not None and other is not target:
        try:
            other.lower()
        except tk.TclError:
            pass


def _sync_section_tab(gui: Any, *, initial: bool = False) -> None:
    title = gait_corr_inner_title(gui)
    view = FOUR_GAIT_TAB_VIEWS.get(title)
    if view is not None:
        _raise_section_body(gui, "four")
        gui._four_gait_view = view
        var = getattr(gui, "var_four_gait_view", None)
        if var is not None:
            try:
                var.set(view)
            except tk.TclError:
                pass
        rebuild = getattr(gui, "_rebuild_four_gait_figure", None)
        if callable(rebuild) and not initial:
            rebuild()
        from stablewalk.ui.tk.dashboard_four_gait import (
            render_four_gait_workspace,
            schedule_four_gait_signals,
        )

        schedule_four_gait_signals(gui, force=False)
        # Tab switches must paint immediately. NotebookTabChanged already
        # chose this view — do not wait for a later idle callback.
        if getattr(gui, "_four_gait_report", None) is not None or not initial:
            try:
                render_four_gait_workspace(gui)
            except Exception:
                logger.exception("four-gait tab render failed")
        canvas = getattr(gui, "canvas_four_gait", None)
        widget = getattr(gui, "_four_gait_canvas_widget", None)
        if widget is not None:
            try:
                widget.update_idletasks()
            except tk.TclError:
                pass
        if canvas is not None:
            try:
                canvas.draw_idle()
            except Exception:
                logger.debug("four-gait canvas draw after tab change failed", exc_info=True)
        relayout = getattr(gui, "_relayout_four_gait_plots", None)
        root = getattr(gui, "root", None)
        if callable(relayout):
            try:
                if root is not None:
                    root.after_idle(relayout)
                else:
                    relayout()
            except Exception:
                logger.debug("four-gait relayout after tab change failed", exc_info=True)
        return
    if title == ASYMMETRY_TAB:
        _raise_section_body(gui, "asym")
        from stablewalk.ui.tk.dashboard_asymmetry import schedule_asymmetry_analysis

        schedule_asymmetry_analysis(gui, force=False)


def _export_section_csv(gui: Any) -> None:
    four = getattr(gui, "_four_gait_report", None)
    asym = getattr(gui, "_asymmetry_report", None)
    if four is None and (asym is None or not getattr(asym, "available", False)):
        messagebox.showinfo(
            SECTION_TITLE,
            "Load a walking video and wait for analysis before exporting.",
        )
        return
    path = filedialog.asksaveasfilename(
        title="Export gait correlation CSV",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
    )
    if not path:
        return
    try:
        dest = path
        if four is not None:
            from stablewalk.analysis.four_gait_export import export_four_gait_csv
            from stablewalk.ui.tk.dashboard_four_gait import _mode

            dest = export_four_gait_csv(four, path, mode=_mode(gui))
        if asym is not None and getattr(asym, "available", False):
            from stablewalk.analysis.asymmetry3d.export import export_asymmetry_csv

            export_asymmetry_csv(asym, dest, append=four is not None)
        messagebox.showinfo(SECTION_TITLE, "Export saved.")
    except Exception as exc:
        logger.exception("gait correlation export failed")
        messagebox.showerror(SECTION_TITLE, f"Export failed: {exc}")
