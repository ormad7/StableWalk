"""Professional session management for the StableWalk dashboard.

Provides New / Save / Load / Autosave / Recent / Import / Export with exact
restore of video path, results, selected joints, camera, playback, and graphs.
"""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

from stablewalk import config
from stablewalk.io.session_registry import (
    autosave_interval_s,
    autosave_root,
    is_autosave_enabled,
    list_recent_sessions,
    remember_session,
    set_autosave_enabled,
)

logger = logging.getLogger(__name__)


# ── workspace capture / restore ─────────────────────────────────────────────


def _bool_var(gui: Any, attr: str, default: bool = False) -> bool:
    var = getattr(gui, attr, None)
    if var is None:
        return default
    try:
        return bool(var.get())
    except Exception:
        return default


def _str_var(gui: Any, attr: str, default: str = "") -> str:
    var = getattr(gui, attr, None)
    if var is None:
        return default
    try:
        return str(var.get())
    except Exception:
        return default


def _capture_camera_ax(ax: Any) -> dict[str, Any] | None:
    if ax is None:
        return None
    try:
        elev = float(ax.elev)
        azim = float(ax.azim)
    except Exception:
        return None
    zoom = getattr(ax, "_stablewalk_camera_zoom", 1.0)
    pan = getattr(ax, "_stablewalk_pan_offset", (0.0, 0.0, 0.0))
    user = getattr(ax, "_stablewalk_user_camera", None)
    payload: dict[str, Any] = {
        "elev": elev,
        "azim": azim,
        "zoom": float(zoom) if zoom is not None else 1.0,
        "pan": [float(pan[0]), float(pan[1]), float(pan[2])]
        if isinstance(pan, (tuple, list)) and len(pan) == 3
        else [0.0, 0.0, 0.0],
    }
    if isinstance(user, (tuple, list)) and len(user) == 2:
        payload["user_camera"] = [float(user[0]), float(user[1])]
    return payload


def _restore_camera_ax(ax: Any, data: dict[str, Any] | None) -> None:
    if ax is None or not data:
        return
    try:
        elev = float(data.get("elev", ax.elev))
        azim = float(data.get("azim", ax.azim))
    except Exception:
        return
    try:
        ax.view_init(elev=elev, azim=azim)
    except Exception:
        pass
    user = data.get("user_camera") or (elev, azim)
    try:
        ax._stablewalk_user_camera = (float(user[0]), float(user[1]))  # type: ignore[attr-defined]
    except Exception:
        pass
    if "zoom" in data:
        try:
            ax._stablewalk_camera_zoom = float(data["zoom"])  # type: ignore[attr-defined]
        except Exception:
            pass
    pan = data.get("pan")
    if isinstance(pan, (list, tuple)) and len(pan) == 3:
        try:
            ax._stablewalk_pan_offset = (  # type: ignore[attr-defined]
                float(pan[0]),
                float(pan[1]),
                float(pan[2]),
            )
        except Exception:
            pass


def _capture_chart_xlim(fig: Any) -> list[float] | None:
    if fig is None:
        return None
    try:
        from stablewalk.ui.viewers.chart_navigation import FIG_NAV_STATE_ATTR

        state = getattr(fig, FIG_NAV_STATE_ATTR, None)
        xlim = getattr(state, "view_xlim", None) if state is not None else None
        if xlim is None and fig.axes:
            xlim = fig.axes[0].get_xlim()
        if xlim is None:
            return None
        return [float(xlim[0]), float(xlim[1])]
    except Exception:
        return None


def _restore_chart_xlim(fig: Any, canvas: Any, xlim: list[float] | None) -> None:
    if fig is None or not xlim or len(xlim) != 2:
        return
    try:
        from stablewalk.ui.viewers.chart_navigation import apply_chart_xlim

        apply_chart_xlim(fig, (float(xlim[0]), float(xlim[1])), persist=True)
        if canvas is not None:
            canvas.draw_idle()
    except Exception:
        try:
            for ax in fig.axes:
                ax.set_xlim(float(xlim[0]), float(xlim[1]))
            if canvas is not None:
                canvas.draw_idle()
        except Exception:
            pass


def capture_workspace_state(gui: Any) -> dict[str, Any]:
    """Serialize cameras, graphs, overlays, and view-mode prefs."""
    from stablewalk.ui.tk.dashboard_overview_view_mode import current_overview_view_mode

    overlays = {
        "direction": _bool_var(gui, "var_overlay_direction", True),
        "contact": _bool_var(gui, "var_overlay_contact", True),
        "com": _bool_var(gui, "var_overlay_com", True),
        "bos": _bool_var(gui, "var_overlay_bos", True),
        "com_velocity": _bool_var(gui, "var_overlay_com_velocity", False),
        "ground": _bool_var(gui, "var_overlay_ground", True),
    }
    graphs = {
        "knee_chart_axis": _str_var(gui, "var_knee_chart_axis", "Video Time"),
        "knee_angle_source": _str_var(gui, "var_knee_angle_source", "Auto"),
        "dof_projection": _str_var(gui, "var_dof_projection", "3D"),
        "dof_coord_mode": _str_var(gui, "var_dof_coord_mode", ""),
        "smooth_motion": _bool_var(gui, "smooth_motion", True),
        "show_skeleton": _bool_var(gui, "show_skeleton", True),
        "highlight_dof": _bool_var(gui, "highlight_dof", True),
        "dof_table_display_mode": _str_var(gui, "dof_table_display_mode", ""),
        "knee_xlim": _capture_chart_xlim(getattr(gui, "fig", None)),
        "biomech_xlim": _capture_chart_xlim(getattr(gui, "fig_biomech", None)),
        "contact_xlim": _capture_chart_xlim(getattr(gui, "fig_contact_gait", None)),
    }
    skeleton_mode = _str_var(gui, "skeleton_display_mode", "")
    return {
        "overview_view_mode": current_overview_view_mode(gui),
        "skeleton_display_mode": skeleton_mode,
        "overlays": overlays,
        "graphs": graphs,
        "cameras": {
            "trajectory": _capture_camera_ax(getattr(gui, "ax_dof_traj", None)),
            "trajectory_overview": _capture_camera_ax(
                getattr(gui, "ax_dof_traj_overview", None)
            ),
            "skeleton": _capture_camera_ax(getattr(gui, "ax_3d", None)),
        },
        "playing": bool(getattr(gui, "playing", False)),
    }


def capture_results_snapshot(gui: Any) -> dict[str, Any]:
    """Capture serializable analysis digests for restore verification."""
    results: dict[str, Any] = {}
    for attr, key in (
        ("_gait_cycle", "gait_cycle"),
        ("_gait_features", "gait_features"),
        ("_biomech", "biomech_stability"),
        ("_stability", "stability"),
        ("_analysis_summary_cache", "analysis_summary"),
        ("_biomech_analysis", "biomech_analysis"),
        ("_estimated_vgrf", "estimated_vgrf"),
        ("_virtual_grf", "virtual_grf"),
        ("_foot_contact", "foot_contact"),
    ):
        obj = getattr(gui, attr, None)
        if obj is None:
            continue
        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            try:
                results[key] = to_dict()
            except Exception:
                pass
    return results


def restore_workspace_state(gui: Any, workspace: dict[str, Any]) -> None:
    """Apply workspace cameras/graphs/overlays after motion is restored."""
    if not workspace:
        return

    mode = workspace.get("overview_view_mode")
    if mode:
        try:
            from stablewalk.ui.tk.dashboard_overview_view_mode import (
                apply_overview_view_mode,
            )

            apply_overview_view_mode(gui, str(mode), animate=False, persist=False)
        except Exception:
            pass

    skel_mode = workspace.get("skeleton_display_mode")
    if skel_mode and hasattr(gui, "skeleton_display_mode"):
        try:
            gui.skeleton_display_mode.set(str(skel_mode))
            on_mode = getattr(gui, "_on_skeleton_display_mode", None)
            if on_mode is not None:
                on_mode()
        except Exception:
            pass

    overlays = workspace.get("overlays") or {}
    overlay_map = {
        "direction": "var_overlay_direction",
        "contact": "var_overlay_contact",
        "com": "var_overlay_com",
        "bos": "var_overlay_bos",
        "com_velocity": "var_overlay_com_velocity",
        "ground": "var_overlay_ground",
    }
    for key, attr in overlay_map.items():
        if key not in overlays:
            continue
        var = getattr(gui, attr, None)
        if var is None:
            continue
        try:
            var.set(bool(overlays[key]))
        except Exception:
            pass
    toggle = getattr(gui, "_on_biomech_overlay_toggle", None)
    if toggle is not None and overlays:
        try:
            toggle()
        except Exception:
            pass

    graphs = workspace.get("graphs") or {}
    for key, attr in (
        ("knee_chart_axis", "var_knee_chart_axis"),
        ("knee_angle_source", "var_knee_angle_source"),
        ("dof_projection", "var_dof_projection"),
        ("dof_coord_mode", "var_dof_coord_mode"),
        ("dof_table_display_mode", "dof_table_display_mode"),
    ):
        if not graphs.get(key):
            continue
        var = getattr(gui, attr, None)
        if var is None:
            continue
        try:
            var.set(str(graphs[key]))
        except Exception:
            pass
    for key, attr in (
        ("smooth_motion", "smooth_motion"),
        ("show_skeleton", "show_skeleton"),
        ("highlight_dof", "highlight_dof"),
    ):
        if key not in graphs:
            continue
        var = getattr(gui, attr, None)
        if var is None:
            continue
        try:
            var.set(bool(graphs[key]))
        except Exception:
            pass

    if graphs.get("dof_table_display_mode"):
        sync = getattr(gui, "_sync_dof_table_mode_flag", None)
        if sync is not None:
            try:
                sync()
            except Exception:
                pass
    if graphs.get("dof_projection"):
        on_proj = getattr(gui, "_on_dof_projection_changed", None)
        if on_proj is not None:
            try:
                on_proj()
            except Exception:
                pass
    if graphs.get("knee_chart_axis"):
        on_knee = getattr(gui, "_on_knee_chart_mode_changed", None)
        if on_knee is not None:
            try:
                on_knee()
            except Exception:
                pass

    cameras = workspace.get("cameras") or {}
    _restore_camera_ax(getattr(gui, "ax_dof_traj", None), cameras.get("trajectory"))
    _restore_camera_ax(
        getattr(gui, "ax_dof_traj_overview", None), cameras.get("trajectory_overview")
    )
    _restore_camera_ax(getattr(gui, "ax_3d", None), cameras.get("skeleton"))

    _restore_chart_xlim(
        getattr(gui, "fig", None),
        getattr(gui, "chart_canvas", None),
        graphs.get("knee_xlim"),
    )
    _restore_chart_xlim(
        getattr(gui, "fig_biomech", None),
        getattr(gui, "canvas_biomech", None),
        graphs.get("biomech_xlim"),
    )
    _restore_chart_xlim(
        getattr(gui, "fig_contact_gait", None),
        getattr(gui, "canvas_contact_gait", None),
        graphs.get("contact_xlim"),
    )


# ── session manager ─────────────────────────────────────────────────────────


class SessionManager:
    """Owns current session path, dirty flag, autosave, and File menu actions."""

    def __init__(self, gui: Any) -> None:
        self.gui = gui
        self.current_path: Path | None = None
        self.dirty = False
        self._autosave_after: str | None = None
        self._autosave_var = tk.BooleanVar(value=is_autosave_enabled())
        self._recent_menu: tk.Menu | None = None
        gui._session_manager = self
        gui._session_dirty = False
        gui._session_current_path = None

    # ── dirty / autosave ────────────────────────────────────────────────────

    def mark_dirty(self) -> None:
        self.dirty = True
        self.gui._session_dirty = True
        self._update_window_title()

    def clear_dirty(self) -> None:
        self.dirty = False
        self.gui._session_dirty = False
        self._update_window_title()

    def _update_window_title(self) -> None:
        root = getattr(self.gui, "root", None)
        if root is None:
            return
        name = self.current_path.name if self.current_path else "Untitled"
        mark = " *" if self.dirty else ""
        try:
            root.title(f"StableWalk — {name}{mark}")
        except tk.TclError:
            pass

    def start_autosave(self) -> None:
        self._cancel_autosave()
        if not self._autosave_var.get():
            return
        root = getattr(self.gui, "root", None)
        if root is None:
            return
        interval_ms = autosave_interval_s() * 1000

        def _tick() -> None:
            self._autosave_after = None
            try:
                if self._autosave_var.get() and self.dirty and self._can_save():
                    self._autosave_silent()
            except Exception:
                logger.exception("Autosave failed")
            self.start_autosave()

        try:
            self._autosave_after = root.after(interval_ms, _tick)
        except tk.TclError:
            self._autosave_after = None

    def _cancel_autosave(self) -> None:
        root = getattr(self.gui, "root", None)
        if root is not None and self._autosave_after is not None:
            try:
                root.after_cancel(self._autosave_after)
            except tk.TclError:
                pass
        self._autosave_after = None

    def toggle_autosave(self) -> None:
        enabled = bool(self._autosave_var.get())
        set_autosave_enabled(enabled)
        if enabled:
            self.start_autosave()
            self._status("Autosave enabled")
        else:
            self._cancel_autosave()
            self._status("Autosave disabled")

    def _can_save(self) -> bool:
        gui = self.gui
        if getattr(gui, "_presentation_mode", False):
            return False
        if not getattr(gui, "selection", None) or not gui.selection.selected:
            return False
        snap = gui._build_session_bundle_snapshot()
        return snap is not None and snap.recording is not None

    def _enrich_snapshot(self, snapshot: Any) -> Any:
        snapshot.workspace = capture_workspace_state(self.gui)
        snapshot.results = capture_results_snapshot(self.gui)
        if self.current_path is not None:
            snapshot.display_name = self.current_path.name
        return snapshot

    def _autosave_silent(self) -> None:
        from stablewalk.io.session_bundle import SessionBundleError, export_session_bundle

        snapshot = self.gui._build_session_bundle_snapshot()
        if snapshot is None or snapshot.recording is None:
            return
        self._enrich_snapshot(snapshot)
        root = autosave_root()
        root.mkdir(parents=True, exist_ok=True)
        target = root / "stablewalk_autosave"
        try:
            export_session_bundle(
                snapshot,
                root,
                target_bundle_dir=target,
                copy_poses=True,
                bundle_name="stablewalk_autosave",
            )
            self.clear_dirty()
            self._status(f"Autosaved → {target.name}")
            remember_session(
                target,
                display_name="Autosave",
                video_source=snapshot.video_source,
                frame_count=snapshot.frame_count,
            )
            self.refresh_recent_menu()
            notify = getattr(self.gui, "_notify_generated_files_changed", None)
            if notify is not None:
                try:
                    notify()
                except Exception:
                    pass
        except (SessionBundleError, OSError) as exc:
            logger.warning("Autosave skipped: %s", exc)

    # ── user actions ────────────────────────────────────────────────────────

    def confirm_discard_if_dirty(self, action: str = "continue") -> bool:
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel(
            "Unsaved changes",
            f"The current session has unsaved changes.\n\n"
            f"Save before you {action}?",
        )
        if answer is None:
            return False
        if answer:
            return self.save_session(quiet=True)
        return True

    def new_session(self) -> None:
        if not self.confirm_discard_if_dirty("start a new session"):
            return
        reset = getattr(self.gui, "_reset_session", None)
        if reset is not None:
            reset(message="New session — load a video to begin")
        self.current_path = None
        self.gui._session_current_path = None
        self.clear_dirty()
        self._status("New session")
        self.refresh_recent_menu()

    def save_session(self, *, quiet: bool = False) -> bool:
        if self.current_path is not None:
            return self._write_to_path(self.current_path, quiet=quiet, remember=True)
        return self.save_session_as(quiet=quiet)

    def save_session_as(self, *, quiet: bool = False) -> bool:
        if getattr(self.gui, "_presentation_mode", False):
            messagebox.showinfo(
                "Save Session",
                "Load and analyze a real video before saving a session.",
            )
            return False
        if not self.gui.selection.selected:
            messagebox.showinfo(
                "Save Session",
                "Select at least one body point to save.",
            )
            return False
        snapshot = self.gui._build_session_bundle_snapshot()
        if snapshot is None or snapshot.recording is None:
            messagebox.showwarning("Save Session", "No motion data available to save.")
            return False

        config.ensure_output_dirs()
        target = filedialog.askdirectory(
            title="Choose folder for saved session",
            initialdir=str(config.SESSION_EXPORT_DIR.resolve()),
        )
        if not target:
            return False
        from stablewalk.io.session_bundle import session_bundle_stem

        bundle_dir = Path(target) / session_bundle_stem()
        return self._write_to_path(bundle_dir, quiet=quiet, remember=True, create_new=True)

    def _write_to_path(
        self,
        bundle_dir: Path,
        *,
        quiet: bool = False,
        remember: bool = True,
        create_new: bool = False,
    ) -> bool:
        from stablewalk.io.session_bundle import SessionBundleError, export_session_bundle

        snapshot = self.gui._build_session_bundle_snapshot()
        if snapshot is None or snapshot.recording is None:
            if not quiet:
                messagebox.showwarning("Save Session", "No motion data available to save.")
            return False
        self._enrich_snapshot(snapshot)
        try:
            if create_new:
                written = export_session_bundle(
                    snapshot,
                    bundle_dir.parent,
                    bundle_name=bundle_dir.name,
                    copy_poses=True,
                )
            else:
                written = export_session_bundle(
                    snapshot,
                    bundle_dir.parent,
                    target_bundle_dir=bundle_dir,
                    copy_poses=True,
                    bundle_name=bundle_dir.name,
                )
        except SessionBundleError as exc:
            if not quiet:
                messagebox.showerror("Save Session failed", str(exc))
            return False
        except OSError as exc:
            if not quiet:
                messagebox.showerror("Save Session failed", str(exc))
            return False

        self.current_path = written
        self.gui._session_current_path = written
        self.clear_dirty()
        if remember:
            remember_session(
                written,
                display_name=written.name,
                video_source=snapshot.video_source,
                frame_count=snapshot.frame_count,
            )
            self.refresh_recent_menu()
        self._status(f"Session saved → {written.name}")
        notify = getattr(self.gui, "_notify_generated_files_changed", None)
        if notify is not None:
            try:
                notify()
            except Exception:
                pass
        if not quiet:
            messagebox.showinfo(
                "Session saved",
                "Session saved successfully.\n\n"
                f"Folder:\n{written.resolve()}\n\n"
                "Includes video path, results, selected joints, camera, "
                "playback position, and graph settings.",
            )
        return True

    def load_session(self, path: str | Path | None = None) -> bool:
        if not self.confirm_discard_if_dirty("load another session"):
            return False
        if path is None:
            config.ensure_output_dirs()
            chosen = filedialog.askdirectory(
                title="Select saved session folder",
                initialdir=str(config.SESSION_EXPORT_DIR.resolve()),
            )
            if not chosen:
                return False
            path = chosen
        return self.gui._apply_session_bundle_from_path(str(path))

    def import_session(self) -> bool:
        if not self.confirm_discard_if_dirty("import a session"):
            return False
        config.ensure_output_dirs()
        initial = str(config.SESSION_EXPORT_DIR.resolve())
        path = filedialog.askopenfilename(
            title="Import session (session_metadata.json)",
            initialdir=initial,
            filetypes=[
                ("StableWalk session metadata", "session_metadata.json"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            folder = filedialog.askdirectory(
                title="Or select session folder",
                initialdir=initial,
            )
            if not folder:
                return False
            path = folder
        return self.gui._apply_session_bundle_from_path(str(path))

    def export_session(self) -> bool:
        """Export a portable copy without changing the current session path."""
        if getattr(self.gui, "_presentation_mode", False):
            messagebox.showinfo(
                "Export Session",
                "Load and analyze a real video before exporting.",
            )
            return False
        if not self.gui.selection.selected:
            messagebox.showinfo(
                "Export Session",
                "Select at least one body point to export.",
            )
            return False
        snapshot = self.gui._build_session_bundle_snapshot()
        if snapshot is None or snapshot.recording is None:
            messagebox.showwarning("Export Session", "No motion data available to export.")
            return False
        self._enrich_snapshot(snapshot)
        config.ensure_output_dirs()
        target = filedialog.askdirectory(
            title="Choose export folder",
            initialdir=str(config.SESSION_EXPORT_DIR.resolve()),
        )
        if not target:
            return False
        from stablewalk.io.session_bundle import SessionBundleError, export_session_bundle

        try:
            written = export_session_bundle(snapshot, target, copy_poses=True)
        except (SessionBundleError, OSError) as exc:
            messagebox.showerror("Export failed", str(exc))
            return False
        remember_session(
            written,
            display_name=written.name,
            video_source=snapshot.video_source,
            frame_count=snapshot.frame_count,
        )
        self.refresh_recent_menu()
        self._status(f"Session exported → {written.name}")
        notify = getattr(self.gui, "_notify_generated_files_changed", None)
        if notify is not None:
            try:
                notify()
            except Exception:
                pass
        messagebox.showinfo(
            "Session exported",
            f"Exported to:\n{written.resolve()}",
        )
        return True

    def load_recent(self, path: str) -> None:
        self.load_session(path)

    def refresh_recent_menu(self) -> None:
        menu = self._recent_menu
        if menu is None:
            return
        try:
            menu.delete(0, tk.END)
        except tk.TclError:
            return
        recent = list_recent_sessions()
        if not recent:
            menu.add_command(label="(No recent sessions)", state=tk.DISABLED)
            return
        for entry in recent:
            label = entry.display_name
            if entry.video_source:
                short = Path(entry.video_source).name
                label = f"{label}  —  {short}"
            menu.add_command(
                label=label,
                command=lambda p=entry.path: self.load_recent(p),
            )
        menu.add_separator()
        menu.add_command(label="Clear Recent…", command=self._clear_recent)

    def _clear_recent(self) -> None:
        from stablewalk.io.session_registry import load_registry, save_registry

        if not messagebox.askyesno("Clear Recent", "Clear the recent sessions list?"):
            return
        data = load_registry()
        data["recent"] = []
        save_registry(data)
        self.refresh_recent_menu()

    def _status(self, text: str) -> None:
        status = getattr(self.gui, "status", None)
        if status is not None:
            try:
                status.configure(text=text)
            except Exception:
                pass

    def install_file_menu_commands(self, file_menu: tk.Menu, menu_colors: dict) -> None:
        """Insert professional session commands at the top of the File menu."""
        file_menu.insert_command(
            0, label="New Session", command=self.new_session, accelerator="Ctrl+N"
        )
        file_menu.insert_separator(1)
        file_menu.insert_command(
            2, label="Save Session", command=self.save_session, accelerator="Ctrl+S"
        )
        file_menu.insert_command(
            3,
            label="Save Session As…",
            command=self.save_session_as,
            accelerator="Ctrl+Shift+S",
        )
        file_menu.insert_command(
            4, label="Load Session…", command=self.load_session, accelerator="Ctrl+L"
        )
        self._recent_menu = tk.Menu(file_menu, tearoff=0, **menu_colors)
        file_menu.insert_cascade(5, label="Recent Sessions", menu=self._recent_menu)
        file_menu.insert_separator(6)
        file_menu.insert_command(7, label="Import…", command=self.import_session)
        file_menu.insert_command(8, label="Export…", command=self.export_session)
        file_menu.insert_separator(9)
        file_menu.insert_checkbutton(
            10,
            label="Autosave",
            variable=self._autosave_var,
            command=self.toggle_autosave,
        )
        file_menu.insert_separator(11)
        self.refresh_recent_menu()

        root = self.gui.root
        root.bind_all("<Control-n>", lambda _e: self.new_session())
        root.bind_all("<Control-s>", lambda _e: self.save_session())
        root.bind_all("<Control-S>", lambda _e: self.save_session_as())
        root.bind_all("<Control-l>", lambda _e: self.load_session())


def install_session_manager(gui: Any) -> SessionManager:
    mgr = SessionManager(gui)
    mgr.start_autosave()
    mgr._update_window_title()
    return mgr


__all__ = [
    "SessionManager",
    "capture_results_snapshot",
    "capture_workspace_state",
    "install_session_manager",
    "restore_workspace_state",
]
