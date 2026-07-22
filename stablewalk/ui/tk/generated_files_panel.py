"""Generated Files panel for the Advanced & Export tab.

Lists exported artifacts with type, status, size, and creation time, plus
Open / Reveal / Copy Path / Delete / Re-export actions.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable

from stablewalk import config
from stablewalk.ui.theme import (
    DASHBOARD_CARD_PAD,
    FONT_UI_XS,
    MUTED,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
    create_tooltip,
)

SECTION_GENERATED_FILES_TITLE = "Generated Files"

_MAX_FILES = 100


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
        return "Professional PDF Report", "pdf_report"
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
                    # Prefer session-named run folders' contents.
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

    # Current-run quick paths
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

    # Deduplicate by resolved path
    unique: dict[str, Path] = {}
    for path in found:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        unique[key] = path
    return list(unique.values())


def scan_generated_files(*, run_name: str | None = None) -> list[GeneratedFileEntry]:
    """Scan output directories and return newest-first generated file entries."""
    entries: list[GeneratedFileEntry] = []
    for path in _iter_scan_candidates(run_name=run_name):
        entry = _stat_entry(path)
        if entry is not None:
            entries.append(entry)
    entries.sort(key=lambda e: e.created_at, reverse=True)
    return entries[:_MAX_FILES]


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


def build_generated_files_section(gui: Any, parent: tk.Misc) -> ttk.LabelFrame:
    """Install the Generated Files card on the Advanced & Export tab."""
    section = ttk.LabelFrame(
        parent,
        text=f"  {SECTION_GENERATED_FILES_TITLE}  ",
        style="Card.TLabelframe",
        padding=DASHBOARD_CARD_PAD,
    )
    section.columnconfigure(0, weight=1)
    gui._section_generated_files = section

    header = tk.Frame(section, bg=PANEL, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))
    header.columnconfigure(0, weight=1)

    gui.lbl_generated_files_summary = tk.Label(
        header,
        text="No generated files yet. Export analysis to see artifacts here.",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_generated_files_summary.grid(row=0, column=0, sticky="w")

    refresh_btn = ttk.Button(
        header,
        text="Refresh",
        style="Compact.TButton",
        command=lambda: refresh_generated_files_panel(gui),
    )
    refresh_btn.grid(row=0, column=1, sticky="e")
    create_tooltip(refresh_btn, "Rescan output folders for generated files")
    gui.btn_generated_files_refresh = refresh_btn

    tree_host = ttk.Frame(section)
    tree_host.grid(row=1, column=0, sticky="nsew")
    tree_host.columnconfigure(0, weight=1)
    tree_host.rowconfigure(0, weight=1)
    section.rowconfigure(1, weight=1)

    tree = ttk.Treeview(
        tree_host,
        columns=_COLUMNS,
        show="headings",
        height=8,
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

    vsb = ttk.Scrollbar(tree_host, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")

    gui.generated_files_tree = tree
    gui._generated_files_by_iid: dict[str, GeneratedFileEntry] = {}

    actions = ttk.Frame(section)
    actions.grid(row=2, column=0, sticky="ew", pady=(PAD_SM, 0))

    def _selected() -> GeneratedFileEntry | None:
        sel = tree.selection()
        if not sel:
            return None
        return gui._generated_files_by_iid.get(sel[0])

    def _with_selected(fn: Callable[[GeneratedFileEntry], None], *, need_file: bool = False) -> None:
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
            messagebox.showerror("Generated Files", str(exc))

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
        if not messagebox.askyesno(
            "Delete generated file",
            f"Delete this {kind}?\n\n{entry.path}\n\nThis cannot be undone.",
        ):
            return
        delete_generated_path(entry.path)
        refresh_generated_files_panel(gui)
        _status(gui, f"Deleted {entry.filename}")

    def _do_reexport(entry: GeneratedFileEntry) -> None:
        kind = entry.reexport_kind
        if not kind:
            messagebox.showinfo(
                "Re-export",
                "Re-export is not available for this file type.\n"
                "Use the Data & Export buttons instead.",
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
        refresh_generated_files_panel(gui)

    btn_specs = (
        ("Open", _do_open, False, "Open the selected file with the default app"),
        ("Reveal Folder", _do_reveal, False, "Show the file in the system file manager"),
        ("Copy Path", _do_copy, False, "Copy the absolute path to the clipboard"),
        ("Delete", _do_delete, False, "Delete the selected file or session folder"),
        ("Re-export", _do_reexport, False, "Run the matching export again"),
    )
    gui._generated_files_action_buttons = []
    for label, handler, need_file, tip in btn_specs:
        btn = ttk.Button(
            actions,
            text=label,
            style="Compact.TButton",
            command=lambda h=handler, nf=need_file: _with_selected(h, need_file=nf),
        )
        btn.pack(side=tk.LEFT, padx=(0, PAD_XS))
        create_tooltip(btn, tip)
        gui._generated_files_action_buttons.append(btn)

    tree.bind("<Double-1>", lambda _e: _with_selected(_do_open))
    tree.bind("<<TreeviewSelect>>", lambda _e: _sync_action_states(gui))

    refresh_generated_files_panel(gui)
    return section


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
            # Re-export stays enabled only when kind is known
            if str(btn.cget("text")) == "Re-export" and has_sel:
                iid = tree.selection()[0]
                entry = gui._generated_files_by_iid.get(iid)
                btn.configure(
                    state=tk.NORMAL if entry and entry.reexport_kind else tk.DISABLED
                )
            else:
                btn.configure(state=state)
        except tk.TclError:
            pass


def refresh_generated_files_panel(gui: Any) -> None:
    """Rescan disk and rebuild the Generated Files table."""
    tree = getattr(gui, "generated_files_tree", None)
    if tree is None:
        return

    run_name = None
    for attr in ("_current_run_name", "_session_run_name"):
        fn = getattr(gui, attr, None)
        if callable(fn):
            try:
                run_name = fn()
            except Exception:
                run_name = None
            if run_name:
                break

    entries = scan_generated_files(run_name=run_name)
    gui._generated_files_by_iid = {}

    try:
        for iid in tree.get_children():
            tree.delete(iid)
    except tk.TclError:
        return

    for index, entry in enumerate(entries):
        iid = f"gf{index}"
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

    summary = getattr(gui, "lbl_generated_files_summary", None)
    if summary is not None:
        if entries:
            text = f"{len(entries)} generated file{'s' if len(entries) != 1 else ''} in data/output/"
        else:
            text = "No generated files yet. Export analysis to see artifacts here."
        try:
            summary.configure(text=text, fg=TEXT if entries else MUTED)
        except tk.TclError:
            pass

    _sync_action_states(gui)
    sync = getattr(gui, "_sync_tab_advanced_scroll", None)
    if sync is not None:
        try:
            sync()
        except Exception:
            pass


def notify_generated_files_changed(gui: Any) -> None:
    """Call after any export so the Advanced list stays current."""
    root = getattr(gui, "root", None)
    if root is None:
        refresh_generated_files_panel(gui)
        return
    try:
        root.after_idle(lambda: refresh_generated_files_panel(gui))
    except tk.TclError:
        refresh_generated_files_panel(gui)


__all__ = [
    "SECTION_GENERATED_FILES_TITLE",
    "GeneratedFileEntry",
    "build_generated_files_section",
    "format_file_size",
    "notify_generated_files_changed",
    "refresh_generated_files_panel",
    "scan_generated_files",
]
