"""Generated Files panel for the Advanced & Export tab.

Lists exported artifacts with type, status, size, and creation time, plus
Open File / Reveal / Copy Path / Delete / Export actions.

The collapsed header shows a short count such as ``5 generated files``.
UI stays lazy: no disk scan at app startup. First Advanced visit + Refresh
update the count. Filters + pagination keep the table light. Playback never
triggers a rescan.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable

from stablewalk import config
from stablewalk.ui.theme import (
    DASHBOARD_CARD_PAD,
    FONT_UI,
    FONT_UI_XS,
    GLYPH_COLLAPSE,
    GLYPH_EXPAND,
    MUTED,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
    create_tooltip,
    tk_labelframe_title,
)

SECTION_GENERATED_FILES_TITLE = "Generated Files"

_MAX_FILES = 200
_PAGE_SIZE = 25
# Avoid rescanning data/output on every Advanced tab enter.
_SCAN_CACHE_TTL_S = 2.5
_scan_cache: dict[str, tuple[float, list["GeneratedFileEntry"]]] = {}

_TYPE_FILTER_ALL = "All types"
_STATUS_FILTER_ALL = "All statuses"


@dataclass(frozen=True)
class GeneratedFileEntry:
    path: Path
    filename: str
    file_type: str
    status: str
    size_bytes: int
    created_at: datetime
    reexport_kind: str | None = None
    is_directory: bool = False

    @property
    def size_label(self) -> str:
        return format_file_size(self.size_bytes)

    @property
    def created_label(self) -> str:
        return self.created_at.strftime("%Y-%m-%d %H:%M")


def format_file_size(nbytes: int) -> str:
    if nbytes < 0:
        return "—"
    if nbytes < 1024:
        return f"{nbytes} B"
    value = float(nbytes)
    for unit in ("KB", "MB", "GB"):
        value /= 1024.0
        if value < 1024.0:
            return f"{value:.1f} {unit}"
    return f"{value / 1024.0:.1f} TB"


def format_generated_files_count(count: int) -> str:
    """Collapsed Generated Files status, e.g. ``5 generated files``."""
    n = max(0, int(count or 0))
    return "1 generated file" if n == 1 else f"{n} generated files"


def format_generated_files_summary(
    count: int,
    *,
    expanded: bool = False,
    loaded: bool = False,
) -> str:
    """Header copy: empty guidance, or a count plus expand/open cue."""
    n = max(0, int(count or 0))
    if not loaded:
        return "No generated files loaded. Click Show to scan the current session."
    if n == 0:
        return "No generated files found for this session."
    base = format_generated_files_count(n)
    if expanded:
        return f"{base} — select a file to open"
    return f"{base} — Show to open"


def format_generated_files_empty_copy(
    *,
    loaded: bool = False,
    all_count: int = 0,
    filtered_count: int = 0,
) -> str:
    """Expanded-table empty state (presentation only)."""
    if not loaded:
        return "No generated files loaded. Click Show to scan the current session."
    if int(all_count or 0) <= 0:
        return "No generated files found for this session."
    if int(filtered_count or 0) <= 0:
        return (
            "No files match the current filters. Try All types or turn off "
            "Current session only."
        )
    return ""


def generated_files_toggle_label(*, expanded: bool) -> str:
    """Disclosure control: Show / Hide, not a bare chevron."""
    return f"{GLYPH_COLLAPSE} Hide" if expanded else f"{GLYPH_EXPAND} Show"


def _classify_path(path: Path) -> tuple[str, str | None]:
    """Return (display type, re-export kind)."""
    name = path.name.lower()
    suffix = path.suffix.lower()

    if path.is_dir():
        if (path / "session_metadata.json").is_file() or name.startswith(
            "stablewalk_session"
        ):
            return "Session Bundle", "session"
        return "Folder", None

    if name.endswith("_poses.json") or name == "poses.json":
        return "Pose JSON", None
    if suffix == ".trc" or name.endswith("_opensim.json") or suffix == ".mot":
        return "OpenSim", "opensim"
    if name == "stablewalk_motion.npz":
        return "Motion Reference", "motion_reference"
    if name == "amp_reference_motion.npz":
        return "AMP Reference", "amp"
    if name.startswith("analysis_report_") and suffix == ".txt":
        return "Analysis Report", "analysis_report"
    if name.startswith("session_report_") and suffix == ".pdf":
        return "Research PDF Report", "pdf_report"
    if name.startswith("comparison_report_") and suffix == ".pdf":
        return "Comparison Report", None
    if name.startswith("gait_metrics_") and suffix == ".json":
        return "Gait Metrics", "gait_metrics"
    if name.startswith("tracking_") and suffix in {".csv", ".json"}:
        return "Tracking Export", None
    if suffix == ".npz":
        return "Motion NPZ", "motion_reference"
    if suffix == ".csv":
        return "CSV Export", None
    if suffix == ".json":
        return "JSON Export", None
    if suffix == ".txt":
        return "Text Report", "analysis_report"
    if suffix == ".pdf":
        return "PDF Report", None
    return suffix.lstrip(".").upper() or "File", None


def _dir_size(path: Path, *, limit: int = 400) -> int:
    total = 0
    count = 0
    try:
        for child in path.rglob("*"):
            if not child.is_file():
                continue
            try:
                total += child.stat().st_size
            except OSError:
                continue
            count += 1
            if count >= limit:
                break
    except OSError:
        pass
    return total


def _stat_entry(path: Path) -> GeneratedFileEntry | None:
    try:
        st = path.stat()
    except OSError:
        return None
    created = datetime.fromtimestamp(getattr(st, "st_ctime", st.st_mtime))
    is_dir = path.is_dir()
    size = _dir_size(path) if is_dir else int(st.st_size)
    file_type, kind = _classify_path(path)
    if is_dir:
        try:
            empty = not any(path.iterdir())
        except OSError:
            empty = True
        status = "Empty" if empty else "Ready"
    elif size <= 0:
        status = "Empty"
    else:
        status = "Ready"
    return GeneratedFileEntry(
        path=path.resolve(),
        filename=path.name,
        file_type=file_type,
        status=status,
        size_bytes=size,
        created_at=created,
        reexport_kind=kind,
        is_directory=is_dir,
    )


def _iter_scan_candidates(*, run_name: str | None = None) -> list[Path]:
    config.ensure_output_dirs()
    roots: list[Path] = [
        config.REPORTS_DIR,
        config.TRACKING_EXPORT_DIR,
        config.ANALYSIS_EXPORT_DIR,
        config.SESSION_EXPORT_DIR,
        config.POSES_DIR,
        config.OPENSIM_DIR,
        config.MOTION_REFERENCE_EXPORT_DIR,
    ]
    found: list[Path] = []

    def _add_file(path: Path) -> None:
        if path.is_file():
            found.append(path)

    def _add_dir_bundle(path: Path) -> None:
        if path.is_dir() and (
            (path / "session_metadata.json").is_file()
            or path.name.startswith("stablewalk_session")
            or path.name == "stablewalk_autosave"
        ):
            found.append(path)

    for root in roots:
        if not root.is_dir():
            continue
        if root in {config.OPENSIM_DIR, config.MOTION_REFERENCE_EXPORT_DIR}:
            try:
                children = sorted(
                    root.iterdir(),
                    key=lambda p: p.stat().st_mtime if p.exists() else 0,
                    reverse=True,
                )
            except OSError:
                children = []
            for child in children[:40]:
                if child.is_file():
                    _add_file(child)
                elif child.is_dir():
                    try:
                        for artifact in child.iterdir():
                            if artifact.is_file():
                                _add_file(artifact)
                    except OSError:
                        pass
            continue

        if root == config.SESSION_EXPORT_DIR:
            try:
                for child in root.iterdir():
                    _add_dir_bundle(child)
                    if child.is_file() and child.suffix.lower() in {
                        ".json",
                        ".csv",
                        ".txt",
                    }:
                        _add_file(child)
            except OSError:
                pass
            continue

        try:
            for child in root.iterdir():
                if child.is_file():
                    _add_file(child)
        except OSError:
            pass

    if run_name:
        for folder in (
            config.OPENSIM_DIR / run_name,
            config.MOTION_REFERENCE_EXPORT_DIR / run_name,
        ):
            if not folder.is_dir():
                continue
            try:
                for artifact in folder.iterdir():
                    if artifact.is_file():
                        _add_file(artifact)
            except OSError:
                pass
        poses = config.POSES_DIR / f"{run_name}_poses.json"
        if poses.is_file():
            _add_file(poses)

    unique: dict[str, Path] = {}
    for path in found:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        unique[key] = path
    return list(unique.values())


def invalidate_generated_files_scan_cache() -> None:
    """Drop the disk-scan cache (call after export / delete)."""
    _scan_cache.clear()


def scan_generated_files(
    *, run_name: str | None = None, force: bool = False
) -> list[GeneratedFileEntry]:
    """Scan output directories and return newest-first generated file entries.

    Results are cached briefly so tab switches do not repeatedly walk disk.
    """
    key = str(run_name or "")
    now = time.monotonic()
    if not force:
        hit = _scan_cache.get(key)
        if hit is not None and (now - hit[0]) < _SCAN_CACHE_TTL_S:
            return list(hit[1])
    entries: list[GeneratedFileEntry] = []
    for path in _iter_scan_candidates(run_name=run_name):
        entry = _stat_entry(path)
        if entry is not None:
            entries.append(entry)
    entries.sort(key=lambda e: e.created_at, reverse=True)
    trimmed = entries[:_MAX_FILES]
    _scan_cache[key] = (now, list(trimmed))
    return list(trimmed)


def entry_belongs_to_session(entry: GeneratedFileEntry, run_name: str | None) -> bool:
    """True when the artifact path/name is tied to the current run name."""
    if not run_name:
        return False
    token = run_name.lower()
    try:
        parts = [p.lower() for p in entry.path.parts]
    except Exception:
        parts = []
    if token in parts:
        return True
    name = entry.filename.lower()
    return token in name or name.startswith(f"{token}_") or name.startswith(
        f"stablewalk_session_{token}"
    )


def filter_generated_files(
    entries: list[GeneratedFileEntry],
    *,
    type_filter: str = _TYPE_FILTER_ALL,
    status_filter: str = _STATUS_FILTER_ALL,
    session_only: bool = False,
    run_name: str | None = None,
) -> list[GeneratedFileEntry]:
    """Apply type / status / current-session filters."""
    out: list[GeneratedFileEntry] = []
    for entry in entries:
        if type_filter and type_filter != _TYPE_FILTER_ALL and entry.file_type != type_filter:
            continue
        if (
            status_filter
            and status_filter != _STATUS_FILTER_ALL
            and entry.status != status_filter
        ):
            continue
        if session_only and run_name and not entry_belongs_to_session(entry, run_name):
            continue
        out.append(entry)
    return out


def paginate_entries(
    entries: list[GeneratedFileEntry],
    *,
    page: int = 0,
    page_size: int = _PAGE_SIZE,
) -> tuple[list[GeneratedFileEntry], int, int]:
    """Return (page_slice, page_index, total_pages)."""
    total = len(entries)
    total_pages = max(1, (total + page_size - 1) // page_size) if total else 1
    page = max(0, min(int(page), total_pages - 1))
    start = page * page_size
    return entries[start : start + page_size], page, total_pages


def open_path(path: Path) -> None:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))
    if sys.platform == "win32":
        os.startfile(str(path))  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def reveal_in_folder(path: Path) -> None:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))
    if sys.platform == "win32":
        if path.is_dir():
            os.startfile(str(path))  # noqa: S606
        else:
            subprocess.run(["explorer", f"/select,{path}"], check=False)
    elif sys.platform == "darwin":
        if path.is_dir():
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["open", "-R", str(path)], check=False)
    else:
        folder = path if path.is_dir() else path.parent
        subprocess.run(["xdg-open", str(folder)], check=False)


def copy_path_to_clipboard(gui: Any, path: Path) -> None:
    text = str(path.resolve())
    root = getattr(gui, "root", None)
    if root is None:
        raise RuntimeError("No root window")
    root.clipboard_clear()
    root.clipboard_append(text)
    try:
        root.update()
    except tk.TclError:
        pass


def delete_generated_path(path: Path) -> None:
    path = path.resolve()
    if not path.exists():
        return
    if path.is_dir():
        import shutil

        shutil.rmtree(path)
    else:
        path.unlink()


_COLUMNS = ("filename", "type", "status", "size", "created")
# Hide Size/Created before their min widths overflow a two-column Advanced card.
_GENERATED_FILES_HIDE_CREATED_W = 640
_GENERATED_FILES_HIDE_SIZE_W = 440
_GENERATED_FILES_WRAP_FILTERS_W = 520
_GENERATED_FILES_WRAP_ACTIONS_W = 500


def generated_files_tree_display_columns(width: int) -> tuple[str, ...]:
    """Visible tree columns for the current Generated Files card width."""
    width = max(0, int(width or 0))
    if width < _GENERATED_FILES_HIDE_SIZE_W:
        return ("filename", "type", "status")
    if width < _GENERATED_FILES_HIDE_CREATED_W:
        return ("filename", "type", "status", "size")
    return _COLUMNS


def generated_files_tree_column_plan(width: int) -> dict[str, dict[str, Any]]:
    """Treeview column kwargs — filename stretches; secondary columns shrink."""
    width = max(0, int(width or 0))
    plan: dict[str, dict[str, Any]] = {
        "filename": {"width": 240, "minwidth": 96, "stretch": True, "anchor": "w"},
        "type": {"width": 120, "minwidth": 72, "stretch": False, "anchor": "w"},
        "status": {"width": 70, "minwidth": 52, "stretch": False, "anchor": "center"},
        "size": {"width": 80, "minwidth": 48, "stretch": False, "anchor": "e"},
        "created": {"width": 130, "minwidth": 90, "stretch": False, "anchor": "center"},
    }
    if width < _GENERATED_FILES_HIDE_SIZE_W:
        plan["filename"]["width"] = 160
        plan["type"]["width"] = 88
        plan["status"]["width"] = 58
    elif width < _GENERATED_FILES_HIDE_CREATED_W:
        plan["filename"]["width"] = 180
        plan["type"]["width"] = 100
        plan["size"]["width"] = 56
    return plan


def reflow_generated_files_panel(gui: Any, *, width: int | None = None) -> None:
    """Fit the Generated Files table and actions to the current card width."""
    tree = getattr(gui, "generated_files_tree", None)
    if tree is None:
        return
    if width is None:
        host = getattr(gui, "_generated_files_tree_host", None) or getattr(
            gui, "_section_generated_files", None
        )
        try:
            width = int((host or tree).winfo_width() or 0)
        except tk.TclError:
            width = 0
        section = getattr(gui, "_section_generated_files", None)
        if width < 40 and section is not None:
            try:
                width = int(section.winfo_width() or 0)
            except tk.TclError:
                width = 0
    width = int(width or 0)
    if width < 40:
        return

    cols = generated_files_tree_display_columns(width)
    prev = getattr(gui, "_generated_files_display_columns", None)
    if cols != prev:
        gui._generated_files_display_columns = cols
        try:
            tree.configure(displaycolumns=cols)
            plan = generated_files_tree_column_plan(width)
            for name, kwargs in plan.items():
                if name in cols:
                    tree.column(name, **kwargs)
        except tk.TclError:
            pass

    session_chk = getattr(gui, "_generated_files_session_chk", None)
    if session_chk is not None:
        try:
            if width < _GENERATED_FILES_WRAP_FILTERS_W:
                session_chk.grid(
                    row=1,
                    column=0,
                    columnspan=5,
                    sticky="w",
                    pady=(PAD_XS, 0),
                )
            else:
                session_chk.grid(row=0, column=4, sticky="w", pady=0)
        except tk.TclError:
            pass

    buttons = list(getattr(gui, "_generated_files_action_buttons", None) or [])
    wrap = 3 if width < _GENERATED_FILES_WRAP_ACTIONS_W else max(1, len(buttons))
    prev_wrap = getattr(gui, "_generated_files_action_wrap", None)
    if buttons and wrap != prev_wrap:
        gui._generated_files_action_wrap = wrap
        actions = buttons[0].master
        compact = width < _GENERATED_FILES_WRAP_ACTIONS_W
        try:
            for col in range(max(len(buttons), 5)):
                actions.columnconfigure(col, weight=1 if compact and col < wrap else 0)
            for index, btn in enumerate(buttons):
                row, col = divmod(index, wrap)
                btn.grid(
                    row=row,
                    column=col,
                    sticky="ew" if compact else "w",
                    padx=(0, PAD_XS),
                    pady=(0, PAD_XS if compact and row else 0),
                )
        except tk.TclError:
            pass

    hint = getattr(gui, "lbl_generated_files_actions_hint", None)
    if hint is not None:
        try:
            hint.configure(wraplength=max(160, width - 24))
        except tk.TclError:
            pass
    empty = getattr(gui, "lbl_generated_files_empty", None)
    if empty is not None:
        try:
            empty.configure(wraplength=max(160, width - 36))
        except tk.TclError:
            pass


def _current_run_name(gui: Any) -> str | None:
    for attr in ("_current_run_name", "_session_run_name"):
        fn = getattr(gui, attr, None)
        if callable(fn):
            try:
                run_name = fn()
            except Exception:
                run_name = None
            if run_name:
                return str(run_name)
    return None


def build_generated_files_section(gui: Any, parent: tk.Misc) -> ttk.LabelFrame:
    """Install the Generated Files card (collapsed by default, no startup scan)."""
    title = tk_labelframe_title(parent, SECTION_GENERATED_FILES_TITLE)
    section = ttk.LabelFrame(
        parent,
        labelwidget=title,
        style="Card.TLabelframe",
        padding=DASHBOARD_CARD_PAD,
    )
    section.columnconfigure(0, weight=1)
    gui._section_generated_files = section
    gui._generated_files_expanded = False
    gui._generated_files_loaded = False
    gui._generated_files_page = 0
    gui._generated_files_all_entries: list[GeneratedFileEntry] = []
    gui._generated_files_filtered: list[GeneratedFileEntry] = []

    header = tk.Frame(section, bg=PANEL, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))
    header.columnconfigure(1, weight=1)

    toggle = ttk.Button(
        header,
        text=generated_files_toggle_label(expanded=False),
        style="Compact.TButton",
        width=8,
        command=lambda: _toggle_generated_files_body(gui),
    )
    toggle.grid(row=0, column=0, sticky="w", padx=(0, PAD_XS))
    gui.btn_generated_files_toggle = toggle
    create_tooltip(toggle, "Show the file list, then Open File or double-click a row")

    gui.lbl_generated_files_summary = tk.Label(
        header,
        text=format_generated_files_summary(0, expanded=False, loaded=False),
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        cursor="hand2",
    )
    gui.lbl_generated_files_summary.grid(row=0, column=1, sticky="ew")
    gui.lbl_generated_files_summary.bind(
        "<Button-1>", lambda _e: _toggle_generated_files_body(gui)
    )
    create_tooltip(
        gui.lbl_generated_files_summary,
        "Click to show or hide the generated-files list",
    )

    refresh_btn = ttk.Button(
        header,
        text="Refresh",
        style="Compact.TButton",
        command=lambda: refresh_generated_files_panel(gui, force=True),
    )
    refresh_btn.grid(row=0, column=2, sticky="e")
    create_tooltip(refresh_btn, "Rescan output folders for generated files")
    gui.btn_generated_files_refresh = refresh_btn

    body = tk.Frame(section, bg=PANEL, highlightthickness=0)
    body.grid(row=1, column=0, sticky="ew")
    body.columnconfigure(0, weight=1)
    gui._generated_files_body = body

    filters = tk.Frame(body, bg=PANEL, highlightthickness=0)
    filters.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))
    filters.columnconfigure(4, weight=1)
    gui._generated_files_filters = filters

    gui.generated_files_type_var = tk.StringVar(value=_TYPE_FILTER_ALL)
    gui.generated_files_status_var = tk.StringVar(value=_STATUS_FILTER_ALL)
    gui.generated_files_session_only = tk.BooleanVar(value=True)

    ttk.Label(filters, text="Type").grid(row=0, column=0, sticky="w", padx=(0, 4))
    type_combo = ttk.Combobox(
        filters,
        textvariable=gui.generated_files_type_var,
        values=(_TYPE_FILTER_ALL,),
        state="readonly",
        width=18,
    )
    type_combo.grid(row=0, column=1, sticky="w", padx=(0, PAD_SM))
    gui.generated_files_type_combo = type_combo
    type_combo.bind(
        "<<ComboboxSelected>>",
        lambda _e: _apply_generated_files_filters(gui, reset_page=True),
    )

    ttk.Label(filters, text="Status").grid(row=0, column=2, sticky="w", padx=(0, 4))
    status_combo = ttk.Combobox(
        filters,
        textvariable=gui.generated_files_status_var,
        values=(_STATUS_FILTER_ALL, "Ready", "Empty"),
        state="readonly",
        width=12,
    )
    status_combo.grid(row=0, column=3, sticky="w", padx=(0, PAD_SM))
    gui.generated_files_status_combo = status_combo
    status_combo.bind(
        "<<ComboboxSelected>>",
        lambda _e: _apply_generated_files_filters(gui, reset_page=True),
    )

    session_chk = ttk.Checkbutton(
        filters,
        text="Current session only",
        variable=gui.generated_files_session_only,
        command=lambda: _apply_generated_files_filters(gui, reset_page=True),
    )
    session_chk.grid(row=0, column=4, sticky="w")
    gui._generated_files_session_chk = session_chk
    create_tooltip(
        session_chk,
        "When checked, only files belonging to the current run/session are listed",
    )

    tree_host = ttk.Frame(body)
    tree_host.grid(row=1, column=0, sticky="nsew")
    tree_host.columnconfigure(0, weight=1)
    tree_host.rowconfigure(0, weight=1)
    body.rowconfigure(1, weight=1)
    gui._generated_files_tree_host = tree_host

    # Size the table to the current page instead of reserving six empty rows.
    tree = ttk.Treeview(
        tree_host,
        columns=_COLUMNS,
        show="headings",
        height=3,
        selectmode="browse",
    )
    tree.heading("filename", text="Filename")
    tree.heading("type", text="Type")
    tree.heading("status", text="Status")
    tree.heading("size", text="Size")
    tree.heading("created", text="Creation Time")
    tree.column("filename", width=240, stretch=True, anchor="w")
    tree.column("type", width=120, stretch=False, anchor="w")
    tree.column("status", width=70, stretch=False, anchor="center")
    tree.column("size", width=80, stretch=False, anchor="e")
    tree.column("created", width=130, stretch=False, anchor="center")
    tree.grid(row=0, column=0, sticky="nsew")

    gui.generated_files_tree = tree
    gui._generated_files_by_iid: dict[str, GeneratedFileEntry] = {}

    empty = tk.Label(
        tree_host,
        text=format_generated_files_empty_copy(loaded=False),
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI,
        anchor="nw",
        justify=tk.LEFT,
        wraplength=360,
        padx=PAD_SM,
        pady=PAD_SM,
    )
    empty.grid(row=0, column=0, sticky="nsew")
    empty.grid_remove()
    gui.lbl_generated_files_empty = empty

    pager = tk.Frame(body, bg=PANEL, highlightthickness=0)
    pager.grid(row=2, column=0, sticky="ew", pady=(PAD_XS, 0))
    gui._generated_files_pager = pager
    gui.lbl_generated_files_page = tk.Label(
        pager,
        text="Page 1 / 1",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_generated_files_page.pack(side=tk.LEFT)
    prev_btn = ttk.Button(
        pager,
        text="Prev",
        style="Compact.TButton",
        command=lambda: _change_generated_files_page(gui, -1),
    )
    prev_btn.pack(side=tk.RIGHT, padx=(PAD_XS, 0))
    next_btn = ttk.Button(
        pager,
        text="Next",
        style="Compact.TButton",
        command=lambda: _change_generated_files_page(gui, 1),
    )
    next_btn.pack(side=tk.RIGHT)
    gui.btn_generated_files_prev = prev_btn
    gui.btn_generated_files_next = next_btn

    actions = ttk.Frame(body)
    actions.grid(row=3, column=0, sticky="ew", pady=(PAD_SM, 0))

    def _selected() -> GeneratedFileEntry | None:
        sel = tree.selection()
        if not sel:
            return None
        return gui._generated_files_by_iid.get(sel[0])

    def _with_selected(
        fn: Callable[[GeneratedFileEntry], None], *, need_file: bool = False
    ) -> None:
        entry = _selected()
        if entry is None:
            messagebox.showinfo("Generated Files", "Select a file first.")
            return
        if need_file and entry.is_directory:
            messagebox.showinfo("Generated Files", "Select a file (not a folder).")
            return
        try:
            fn(entry)
        except Exception as exc:
            from stablewalk.ui.tk.user_errors import format_user_error

            messagebox.showerror("Generated Files", format_user_error(exc))

    def _do_open(entry: GeneratedFileEntry) -> None:
        open_path(entry.path)
        _status(gui, f"Opened {entry.filename}")

    def _do_reveal(entry: GeneratedFileEntry) -> None:
        reveal_in_folder(entry.path)
        _status(gui, f"Revealed {entry.filename}")

    def _do_copy(entry: GeneratedFileEntry) -> None:
        copy_path_to_clipboard(gui, entry.path)
        _status(gui, f"Copied path — {entry.filename}")

    def _do_delete(entry: GeneratedFileEntry) -> None:
        kind = "folder" if entry.is_directory else "file"
        warning = (
            f"Permanently delete this {kind}?\n\n"
            f"{entry.path}\n\n"
            "This cannot be undone. Downstream exports that depend on this "
            "artifact may need to be regenerated."
        )
        if not messagebox.askyesno("Delete generated file", warning, icon="warning"):
            return
        # Second confirmation for directories / large artifacts.
        if entry.is_directory or entry.size_bytes > 1_000_000:
            if not messagebox.askyesno(
                "Confirm delete",
                f"Are you sure you want to permanently remove:\n\n{entry.filename}",
                icon="warning",
            ):
                return
        delete_generated_path(entry.path)
        refresh_generated_files_panel(gui, force=True)
        _status(gui, f"Deleted {entry.filename}")

    def _do_reexport(entry: GeneratedFileEntry) -> None:
        kind = entry.reexport_kind
        if not kind:
            messagebox.showinfo(
                "Re-export",
                "Re-export is not available for this file type.\n"
                "Use the matching Advanced export action instead.",
            )
            return
        dispatch = {
            "opensim": "_export_opensim_session",
            "motion_reference": "_export_motion_reference",
            "amp": "_export_amp_reference",
            "analysis_report": "_export_analysis_report",
            "pdf_report": "_export_professional_pdf_report",
            "gait_metrics": "_export_gait_metrics",
            "session": "_save_session_to_files",
        }
        method_name = dispatch.get(kind)
        method = getattr(gui, method_name, None) if method_name else None
        if method is None:
            messagebox.showinfo("Re-export", "Re-export handler is unavailable.")
            return
        method()
        refresh_generated_files_panel(gui, force=True)

    btn_specs = (
        ("Open File", _do_open, False, "Open the selected file with the default app"),
        ("Reveal Folder", _do_reveal, False, "Show the file in the system file manager"),
        ("Copy Path", _do_copy, False, "Copy the absolute path to the clipboard"),
        ("Delete", _do_delete, False, "Permanently delete the selected file or session folder"),
        ("Export", _do_reexport, False, "Run the matching export again when this file type supports it"),
    )
    gui._generated_files_action_buttons = []
    gui._generated_files_actions_host = actions
    for index, (label, handler, need_file, tip) in enumerate(btn_specs):
        btn = ttk.Button(
            actions,
            text=label,
            style="Compact.TButton",
            command=lambda h=handler, nf=need_file: _with_selected(h, need_file=nf),
        )
        btn.grid(row=0, column=index, sticky="w", padx=(0, PAD_XS))
        create_tooltip(btn, tip)
        gui._generated_files_action_buttons.append(btn)

    hint = tk.Label(
        body,
        text="Select a file, then Open File — or double-click a row.",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=520,
    )
    hint.grid(row=4, column=0, sticky="ew", pady=(PAD_XS, 0))
    gui.lbl_generated_files_actions_hint = hint

    tree.bind("<Double-1>", lambda _e: _with_selected(_do_open))
    tree.bind("<<TreeviewSelect>>", lambda _e: _sync_action_states(gui))

    def _on_files_configure(_event: object | None = None) -> None:
        reflow_generated_files_panel(gui)

    section.bind("<Configure>", _on_files_configure, add="+")
    tree_host.bind("<Configure>", _on_files_configure, add="+")

    # Collapsed + unscanned by default — avoids startup / tab-build disk walks.
    body.grid_remove()
    return section


def _generated_files_visible_count(gui: Any) -> int:
    filtered = getattr(gui, "_generated_files_filtered", None)
    if filtered is not None:
        return len(filtered)
    return len(getattr(gui, "_generated_files_all_entries", []) or [])


def _update_generated_files_summary(gui: Any) -> None:
    """Header: empty guidance, or a count plus expand/open cue."""
    summary = getattr(gui, "lbl_generated_files_summary", None)
    if summary is None:
        return
    count = _generated_files_visible_count(gui)
    loaded = bool(getattr(gui, "_generated_files_loaded", False))
    expanded = bool(getattr(gui, "_generated_files_expanded", False))
    try:
        summary.configure(
            text=format_generated_files_summary(
                count, expanded=expanded, loaded=loaded
            ),
            fg=TEXT if count else MUTED,
        )
    except tk.TclError:
        pass
    _sync_generated_files_toggle(gui)
    _sync_generated_files_empty_state(gui)


def _sync_generated_files_toggle(gui: Any) -> None:
    toggle = getattr(gui, "btn_generated_files_toggle", None)
    if toggle is None:
        return
    expanded = bool(getattr(gui, "_generated_files_expanded", False))
    try:
        toggle.configure(text=generated_files_toggle_label(expanded=expanded))
    except tk.TclError:
        pass


def _sync_generated_files_empty_state(gui: Any) -> None:
    """Show a clear empty message in the table host; hide it when files exist."""
    empty = getattr(gui, "lbl_generated_files_empty", None)
    tree = getattr(gui, "generated_files_tree", None)
    pager = getattr(gui, "_generated_files_pager", None)
    hint = getattr(gui, "lbl_generated_files_actions_hint", None)
    loaded = bool(getattr(gui, "_generated_files_loaded", False))
    all_count = len(getattr(gui, "_generated_files_all_entries", []) or [])
    filtered_count = _generated_files_visible_count(gui)
    is_empty = filtered_count <= 0
    copy = format_generated_files_empty_copy(
        loaded=loaded,
        all_count=all_count,
        filtered_count=filtered_count,
    )
    if empty is not None:
        try:
            empty.configure(text=copy)
            if is_empty:
                empty.grid()
            else:
                empty.grid_remove()
        except tk.TclError:
            pass
    if tree is not None:
        try:
            if is_empty:
                tree.grid_remove()
            else:
                tree.grid()
        except tk.TclError:
            pass
    if pager is not None:
        try:
            if is_empty:
                pager.grid_remove()
            else:
                pager.grid()
        except tk.TclError:
            pass
    if hint is not None:
        try:
            if is_empty:
                hint.configure(
                    text="Save a session or run an Advanced export, then Refresh."
                    if all_count <= 0
                    else "Adjust Type / Status, or turn off Current session only."
                )
            else:
                hint.configure(
                    text="Select a file, then Open File — or double-click a row."
                )
        except tk.TclError:
            pass


def _toggle_generated_files_body(gui: Any) -> None:
    body = getattr(gui, "_generated_files_body", None)
    if body is None:
        return
    expanded = not bool(getattr(gui, "_generated_files_expanded", False))
    gui._generated_files_expanded = expanded
    if expanded:
        try:
            body.grid()
        except tk.TclError:
            pass
        _sync_generated_files_toggle(gui)
        if not getattr(gui, "_generated_files_loaded", False):
            refresh_generated_files_panel(gui, force=False)
        else:
            _render_generated_files_page(gui)
    else:
        try:
            body.grid_remove()
        except tk.TclError:
            pass
        _update_generated_files_summary(gui)


def _status(gui: Any, text: str) -> None:
    status = getattr(gui, "status", None)
    if status is not None:
        try:
            status.configure(text=text)
        except Exception:
            pass


def _sync_action_states(gui: Any) -> None:
    tree = getattr(gui, "generated_files_tree", None)
    buttons = getattr(gui, "_generated_files_action_buttons", None) or []
    if tree is None:
        return
    has_sel = bool(tree.selection())
    state = tk.NORMAL if has_sel else tk.DISABLED
    for btn in buttons:
        try:
            if str(btn.cget("text")) == "Export" and has_sel:
                iid = tree.selection()[0]
                entry = gui._generated_files_by_iid.get(iid)
                btn.configure(
                    state=tk.NORMAL if entry and entry.reexport_kind else tk.DISABLED
                )
            else:
                btn.configure(state=state)
        except tk.TclError:
            pass


def _change_generated_files_page(gui: Any, delta: int) -> None:
    page = int(getattr(gui, "_generated_files_page", 0)) + int(delta)
    gui._generated_files_page = max(0, page)
    _render_generated_files_page(gui)


def _apply_generated_files_filters(gui: Any, *, reset_page: bool = False) -> None:
    if reset_page:
        gui._generated_files_page = 0
    run_name = _current_run_name(gui)
    type_filter = str(getattr(gui, "generated_files_type_var", tk.StringVar()).get())
    status_filter = str(getattr(gui, "generated_files_status_var", tk.StringVar()).get())
    session_only = bool(
        getattr(gui, "generated_files_session_only", tk.BooleanVar(value=True)).get()
    )
    all_entries = list(getattr(gui, "_generated_files_all_entries", []) or [])
    filtered = filter_generated_files(
        all_entries,
        type_filter=type_filter,
        status_filter=status_filter,
        session_only=session_only,
        run_name=run_name,
    )
    gui._generated_files_filtered = filtered
    _render_generated_files_page(gui)


def _render_generated_files_page(gui: Any) -> None:
    tree = getattr(gui, "generated_files_tree", None)
    if tree is None:
        return
    filtered = list(getattr(gui, "_generated_files_filtered", []) or [])
    page_entries, page, total_pages = paginate_entries(
        filtered, page=int(getattr(gui, "_generated_files_page", 0))
    )
    gui._generated_files_page = page
    gui._generated_files_by_iid = {}

    try:
        for iid in tree.get_children():
            tree.delete(iid)
    except tk.TclError:
        return

    for index, entry in enumerate(page_entries):
        iid = f"gf{page}_{index}"
        gui._generated_files_by_iid[iid] = entry
        tree.insert(
            "",
            tk.END,
            iid=iid,
            values=(
                entry.filename,
                entry.file_type,
                entry.status,
                entry.size_label,
                entry.created_label,
            ),
        )

    try:
        tree.configure(height=max(2, min(6, len(page_entries) or 2)))
    except tk.TclError:
        pass

    _update_generated_files_summary(gui)

    page_lbl = getattr(gui, "lbl_generated_files_page", None)
    if page_lbl is not None:
        try:
            page_lbl.configure(text=f"Page {page + 1} / {total_pages}")
        except tk.TclError:
            pass
    for attr, enabled in (
        ("btn_generated_files_prev", page > 0),
        ("btn_generated_files_next", page < total_pages - 1 and bool(filtered)),
    ):
        btn = getattr(gui, attr, None)
        if btn is not None:
            try:
                btn.configure(state=tk.NORMAL if enabled else tk.DISABLED)
            except tk.TclError:
                pass

    _sync_action_states(gui)
    sync = getattr(gui, "_sync_tab_advanced_scroll", None)
    if sync is not None:
        try:
            sync()
        except Exception:
            pass


def refresh_generated_files_panel(gui: Any, *, force: bool = False) -> None:
    """Rescan disk and refresh the Generated Files count / table."""
    if getattr(gui, "playing", False):
        gui._generated_files_dirty = True
        if force:
            invalidate_generated_files_scan_cache()
        return

    tree = getattr(gui, "generated_files_tree", None)
    if tree is None:
        return

    loaded = bool(getattr(gui, "_generated_files_loaded", False))
    dirty = bool(getattr(gui, "_generated_files_dirty", False))
    expanded = bool(getattr(gui, "_generated_files_expanded", False))
    if loaded and not dirty and not force and not expanded:
        _update_generated_files_summary(gui)
        return

    run_name = _current_run_name(gui)
    entries = scan_generated_files(run_name=run_name, force=force)
    gui._generated_files_all_entries = entries
    gui._generated_files_loaded = True
    gui._generated_files_dirty = False

    types = sorted({e.file_type for e in entries})
    type_combo = getattr(gui, "generated_files_type_combo", None)
    if type_combo is not None:
        values = (_TYPE_FILTER_ALL, *types)
        try:
            type_combo.configure(values=values)
            current = str(gui.generated_files_type_var.get())
            if current not in values:
                gui.generated_files_type_var.set(_TYPE_FILTER_ALL)
        except tk.TclError:
            pass

    _apply_generated_files_filters(gui, reset_page=False)
    if not getattr(gui, "_generated_files_expanded", False):
        # Filters fill the tree while hidden; keep the header as a short count.
        _update_generated_files_summary(gui)


def notify_generated_files_changed(gui: Any, *, force: bool = True) -> None:
    """Call after any export so the Advanced list stays current.

    ``force=True`` (default) invalidates the scan cache after real exports.
    Tab enter should pass ``force=False`` to reuse a fresh cache.
    """
    if getattr(gui, "playing", False):
        gui._generated_files_dirty = True
        if force:
            invalidate_generated_files_scan_cache()
        return
    if force:
        invalidate_generated_files_scan_cache()
    root = getattr(gui, "root", None)
    if root is None:
        refresh_generated_files_panel(gui, force=force)
        return
    try:
        root.after_idle(lambda: refresh_generated_files_panel(gui, force=force))
    except tk.TclError:
        refresh_generated_files_panel(gui, force=force)


def flush_generated_files_dirty(gui: Any) -> None:
    """Run a deferred scan after playback pauses (if one was requested)."""
    if getattr(gui, "playing", False):
        return
    if not getattr(gui, "_generated_files_dirty", False):
        return
    refresh_generated_files_panel(gui, force=True)


__all__ = [
    "SECTION_GENERATED_FILES_TITLE",
    "GeneratedFileEntry",
    "build_generated_files_section",
    "entry_belongs_to_session",
    "filter_generated_files",
    "flush_generated_files_dirty",
    "format_file_size",
    "format_generated_files_count",
    "format_generated_files_empty_copy",
    "format_generated_files_summary",
    "generated_files_toggle_label",
    "invalidate_generated_files_scan_cache",
    "notify_generated_files_changed",
    "paginate_entries",
    "reflow_generated_files_panel",
    "refresh_generated_files_panel",
    "scan_generated_files",
    "generated_files_tree_column_plan",
    "generated_files_tree_display_columns",
]
