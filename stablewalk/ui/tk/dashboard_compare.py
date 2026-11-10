"""
Biomechanics Laboratory — dual-session gait Comparison Mode.

Page question: What are the differences?

Layout
------
Chrome:    Session A/B pickers · Pin · Reload · Export
Workspace: 3 columns — Video A / Skeleton A | Graph / Trajectory / Heatmap
                       | Video B / Skeleton B
           narrow — Video A+Skeleton A | Video B+Skeleton B, then the three
                    difference charts stacked full width
Transport: one shared progress clock — click any difference panel to seek
Bottom:    Stability · Symmetry · Cadence · ROM (Δ = B − A)
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from types import SimpleNamespace
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from stablewalk.analysis.session_compare import (
    CompareMetrics,
    SessionComparisonResult,
)
from stablewalk.ui.colors import ACCENT_ALT, COM, MUTED, PANEL, TEXT, WARNING as VIZ_WARNING
from stablewalk.ui.scientific_labels import (
    DISCLAIMER_COMPARE,
    DISCLAIMER_RESEARCH,
)
from stablewalk.ui.theme import (
    BORDER,
    DANGER,
    DASHBOARD_CARD_PAD,
    DASHBOARD_VIZ_CARD_PAD,
    ELEVATED,
    FONT_CAPTION,
    FONT_HEADING,
    FONT_TRANSPORT,
    FONT_SUMMARY_INTERPRETATION,
    FONT_UI_SEMIBOLD,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    ORANGE,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    SUCCESS,
    SURFACE,
    TEXT_SECONDARY,
    WARNING,
    create_tooltip,
    format_labelframe_title,
)
from stablewalk.ui.tk.session_cache import (
    AnalyzedSessionSnapshot,
    ensure_compare_cache,
    log_compare_debug,
    log_session_b_debug,
    sessions_are_identical,
)
from stablewalk.ui.tk.compare_state import (
    COMPARE_ALREADY_LOADING_MESSAGE,
    COMPARE_LOAD_TIMEOUT_S,
    COMPARE_SESSION_A_NOT_ANALYZED,
    COMPARE_SESSION_B_NOT_ANALYZED,
    COMPARE_TIMEOUT_MESSAGE,
    CompareState,
    ENABLE_DETAILED_COMPARE,
    PLACEHOLDER_DIFF_GRAPH_DETAIL,
    PLACEHOLDER_DIFF_PATH_DETAIL,
    PLACEHOLDER_SKELETON_DETAIL,
    PLACEHOLDER_SKELETON_TITLE,
    PLACEHOLDER_VIDEO_DETAIL,
    PLACEHOLDER_VIDEO_TITLE,
    SESSION_B_GUIDANCE_TEXT,
    SESSION_B_NEXT_ACTION,
    SESSION_B_WAITING_TITLE,
    SESSION_B_WORKFLOW_STEPS,
    SESSION_B_WORKFLOW_TITLE,
    log_compare_event,
    log_compare_exception,
)
from stablewalk.ui.tk.compare_workflow import (
    DEMO_DELTA_METRICS,
    DEMO_DELTA_TITLES,
    PRIMARY_COMPARE_METRICS,
    CompareUiState,
    assign_demo_comparison_slots,
    build_cached_comparison,
    is_notable_difference,
    joint_curves_ready,
    list_compare_candidates,
    notable_difference_label,
    snapshot_is_analyzed,
    validate_compare_slots,
)
from stablewalk.pose.skeleton_3d import SKELETON_3D_CONNECTIONS
from stablewalk.ui.viewers.difference_heatmap import draw_difference_heatmap
from stablewalk.ui.viewers.plot_3d import (
    apply_display_limits,
    draw_skeleton_3d,
    remember_skeleton_camera,
    setup_3d_axes,
    smooth_reset_skeleton_camera,
)

logger = logging.getLogger(__name__)

# Session A / B — research colors (not body L/R green/red).
COLOR_SESSION_A = COM
COLOR_SESSION_B = ACCENT_ALT
COLOR_LEFT = COLOR_SESSION_A  # Session A (legacy alias)
COLOR_RIGHT = COLOR_SESSION_B  # Session B (legacy alias)

# Layout floors — stretch fills leftover space; never treat these as fixed sizes.
COMPARE_METRICS_MIN_H = 200
COMPARE_VIDEO_MIN_H = 260
COMPARE_SKELETON_MIN_H = 240
COMPARE_GRAPH_MIN_H = 180
COMPARE_CHART_MIN_H = 180
COMPARE_REPORT_MIN_H = 120
COMPARE_WORKSPACE_MIN_H = 520
COMPARE_CARD_MIN_W = 168
COMPARE_METRICS_WRAP_2 = 1040
COMPARE_SUMMARY_WRAP_3 = 980
COMPARE_SUMMARY_WRAP_2 = 640
COMPARE_GRAPH_STACK_BELOW_W = 520
# Below this workspace width, stack A/B pairs and difference panels
# instead of shrinking the 3-column grid until visualizations clip.
COMPARE_STACK_BELOW_W = 1100
# Wide layout: center difference column grows faster than A/B.
COMPARE_COL_SIDE_WEIGHT = 2
COMPARE_COL_CENTER_WEIGHT = 3
COMPARE_COL_SIDE_MIN_W = 180
COMPARE_COL_CENTER_MIN_W = 240
# Target body fill of the Compare skeleton canvas (70–85%).
COMPARE_SKELETON_FILL = 0.78
COMPARE_SKELETON_BOX_ZOOM = 1.22
COMPARE_SKELETON_CAMERA_DIST = 6.4

_COMPARE_WORKER_SENTINEL = "__worker_idle__"

# Prefer gait-cycle % only when both sessions have enough usable cycles + HS times.
_COMPARE_MIN_USABLE_CYCLES = 2
_COMPARE_AXIS_GAIT = "gait_cycle"
_COMPARE_AXIS_VIDEO = "video_progress"


def fit_compare_video_frame(
    img: Image.Image,
    max_w: int,
    max_h: int,
    *,
    resample: int,
) -> Image.Image:
    """Scale a frame to fill the panel while preserving aspect ratio (may upscale)."""
    box_w = max(int(max_w) - 8, 160)
    box_h = max(int(max_h) - 8, 120)
    iw, ih = img.size
    if iw < 1 or ih < 1:
        return img.copy()
    scale = min(box_w / float(iw), box_h / float(ih))
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    if (nw, nh) == (iw, ih):
        return img.copy()
    return img.resize((nw, nh), resample)


def compare_workspace_natural_height(*, stacked: bool) -> int:
    """Readable floor for Video+Skeleton vs three difference charts."""
    pair = COMPARE_VIDEO_MIN_H + COMPARE_SKELETON_MIN_H + PAD_SM
    charts = (3 * COMPARE_CHART_MIN_H) + (2 * PAD_SM)
    if stacked:
        return pair + charts
    return max(pair, charts)

_METRIC_FIELDS = (
    "Gait Quality",
    "Cadence",
    "Speed",
    "Step Length",
    "Knee ROM",
    "Symmetry",
    "COM Excursion",
    "Mean COM–BoS Margin",
)

_DIFF_PRIORITY = (
    "Cadence difference",
    "Step length difference",
    "ROM difference",
    "Hip ROM difference",
    "Ankle ROM difference",
    "Joint angle difference",
    "COM excursion",
    "Foot clearance",
    "Stability difference",
    "Stability",
    "Walking speed",
    "Symmetry",
    "Gait quality",
    "Usable gait cycles",
    "Timeline difference",
)

def build_comparison_tab(gui: Any, parent: tk.Misc) -> ttk.Frame:
    """Install research A/B workspace plus lazy laboratory Compare."""
    from stablewalk.ui.tk.dashboard_page_frame import install_page_header
    from stablewalk.ui.tk.dashboard_research import build_research_compare_workspace

    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=0)
    parent.rowconfigure(1, weight=1)
    install_page_header(parent, "compare", row=0)

    pages = ttk.Notebook(parent)
    pages.grid(row=1, column=0, sticky="nsew")
    research_page = ttk.Frame(pages)
    laboratory_page = ttk.Frame(pages)
    research_page.columnconfigure(0, weight=1)
    research_page.rowconfigure(0, weight=1)
    laboratory_page.columnconfigure(0, weight=1)
    laboratory_page.rowconfigure(0, weight=1)
    pages.add(research_page, text="Research dataset")
    pages.add(laboratory_page, text="Laboratory")
    gui._compare_page_notebook = pages
    build_research_compare_workspace(gui, research_page)

    host = ttk.Frame(laboratory_page)
    host.grid(row=0, column=0, sticky="nsew")
    host.columnconfigure(0, weight=1)
    host.rowconfigure(0, weight=1)
    placeholder = ttk.Label(
        host,
        text=(
            "Laboratory Compare (existing session cache).\n\n"
            "Use Research dataset for imported Video A / Video B with "
            "independent pose pipelines. Open this page only when you need "
            "the classic Session A / Session B laboratory layout."
        ),
        style="EmptyState.TLabel",
        anchor="center",
        justify=tk.CENTER,
    )
    placeholder.grid(row=0, column=0, sticky="nsew", padx=PAD_SM, pady=PAD_SM)
    gui._comparison_mode = None
    gui._comparison_refresh_pending = False

    def _is_compare_active() -> bool:
        notebook = getattr(gui, "_dashboard_notebook", None)
        if notebook is None:
            return False
        try:
            return notebook.nametowidget(notebook.select()) is getattr(
                gui, "_tab_compare", None
            )
        except (tk.TclError, KeyError):
            return False

    def _ensure() -> ComparisonModeController:
        controller = getattr(gui, "_comparison_mode", None)
        if controller is not None:
            return controller
        # Build first, then drop the placeholder — avoids a black empty host flash.
        controller = ComparisonModeController(gui, host, defer_heavy=True)
        gui._comparison_mode = controller
        gui._comparison_refresh_pending = False
        try:
            placeholder.destroy()
        except tk.TclError:
            pass
        return controller

    def _request_refresh() -> None:
        controller = getattr(gui, "_comparison_mode", None)
        if controller is not None:
            request = getattr(controller, "request_refresh", None)
            if callable(request):
                request()
            else:
                controller.refresh()
        elif _is_compare_active():
            _ensure()
        else:
            gui._comparison_refresh_pending = True

    gui._ensure_comparison_mode_loaded = _ensure
    gui._refresh_comparison_mode = _request_refresh

    def _on_inner(_event: object = None) -> None:
        try:
            text = str(pages.tab(pages.select(), "text"))
        except tk.TclError:
            return
        if text.strip() == "Laboratory":
            _ensure()
        try:
            from stablewalk.ui.tk.dashboard_tab_lifecycle import (
                _sync_global_transport_for_tab,
            )

            _sync_global_transport_for_tab(gui, "Compare")
        except Exception:
            pass

    pages.bind("<<NotebookTabChanged>>", lambda e: _on_inner(), add="+")
    return host


class ComparisonModeController:
    """Owns Comparison Mode widgets and dual-session playback sync."""

    # Cap hover samples so long knee series do not create thousands of hit targets.
    _MAX_CHART_HOVER_POINTS = 64

    def __init__(
        self,
        gui: Any,
        parent: tk.Misc,
        *,
        defer_heavy: bool = True,
    ) -> None:
        self.gui = gui
        self.root = ttk.Frame(parent)
        try:
            parent.columnconfigure(0, weight=1)
            parent.rowconfigure(0, weight=1)
        except tk.TclError:
            pass
        self.root.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=0)  # session chrome
        self.root.rowconfigure(1, weight=0)  # in-tab loading banner
        self.root.rowconfigure(2, weight=1)  # side-by-side workspace
        self.root.rowconfigure(3, weight=0)  # transport
        self.root.rowconfigure(4, weight=0)  # bottom metrics
        self.root.rowconfigure(5, weight=0)  # optional details
        self._compare_stacked = False
        self._compare_needs_scroll = False

        self._progress = 0.0
        self._playing = False
        self._play_job: str | None = None
        self._speed = 1.0
        self._sample_stride = 1
        self._skeleton_mode = tk.StringVar(value="side")
        self._show_path_left = tk.BooleanVar(value=True)
        self._show_path_right = tk.BooleanVar(value=True)
        self._show_path_delta = tk.BooleanVar(value=True)
        self._photo_a: ImageTk.PhotoImage | None = None
        self._photo_b: ImageTk.PhotoImage | None = None
        self._video_resize_job: str | None = None
        self._last_paint: tuple[int, int] | None = None
        self._skel_paint_tick = 0
        self._skel_paint_stride = 1
        # Frame decode / PhotoImage caches — avoid open+seek+resize every tick.
        self._pil_frame_cache: dict[tuple[str, int], Image.Image] = {}
        self._photo_cache: dict[tuple[str, int, int, int, int], ImageTk.PhotoImage] = {}
        self._video_caps: dict[str, Any] = {}
        self._video_box: dict[str, tuple[int, int]] = {}
        self._loading_keys: set[str] = set()
        self._failed_snapshot_keys: set[str] = set()
        self._cached_comparison = None
        self._snapshot_results: queue.Queue[tuple[int, str, Any, str | None]] = (
            queue.Queue()
        )
        self._snapshot_poll_job: str | None = None
        self._snapshot_build_lock = threading.Lock()
        # Strict load state machine — never start a second worker while LOADING.
        self.compare_state = CompareState.IDLE
        self._compare_worker: threading.Thread | None = None
        self._compare_pending_keys: list[str] = []
        self._compare_load_gen = 0
        self._compare_load_started_monotonic: float | None = None
        self._compare_timeout_job: str | None = None
        self._analyzing_session_b_key: str | None = None
        self._session_b_resolved_path: str | None = None
        self._session_b_pose_backend: str | None = None
        self._session_b_max_frames: int | None = None
        self._session_b_analysis_mode: str | None = None
        self._session_b_error: str | None = None
        self._session_b_cta_analyze_buttons: list[tk.Misc] = []
        self.compare_session_b: AnalyzedSessionSnapshot | None = None
        try:
            self.gui.compare_session_b = None
        except Exception:
            pass
        self._demo_compare_load_pending = False
        self._compare_control_states: dict[tk.Misc, tuple[str, ...] | str] = {}
        self._playback_resume_after_load: bool | None = None
        self._playback_resume_when_leaving = False
        self._ui_built_once = True  # chrome/body built below; figures one-shot
        # Persistent playhead / marker artists so the synchronized cursor MOVES
        # each frame instead of forcing a full graph rebuild.
        self._path_markers: dict[str, tuple[Any, np.ndarray]] = {}
        self._heat_playhead: Any = None
        self._heat_bins = 40
        self._chart_playhead: Any = None
        self._compare_axis_mode = _COMPARE_AXIS_VIDEO
        # Keep difference playheads locked to video progress every paint tick.
        self._compare_chart_hz = 60.0
        self._last_compare_chart_ts = 0.0
        # One-shot bind guards (avoid stacking add='+' handlers on re-arm).
        self._scroll_watch_bound = False
        self._compare_enter_focus_bound = False
        self._wheel_bound_ids: set[int] = set()

        # Staged construction — chrome opens instantly; visuals fill on activate.
        self._detailed_comparison = bool(ENABLE_DETAILED_COMPARE)
        self._static_compare_only = not bool(ENABLE_DETAILED_COMPARE)
        self._videos_built = False
        self._skeletons_built = False
        self._graphs_built = False  # detailed path+heatmap (+ full graph row)
        self._lite_chart_built = False
        self._graphs_series_ready = False
        self._visuals_dirty = True
        self._paths_dirty = True
        self._chart_dirty = True
        self._heat_dirty = True
        self._deferred_jobs: list[str] = []
        self._visual_refresh_job: str | None = None
        self._scroll_watch_job: str | None = None
        self._active_chart_section: str | None = None  # path | chart | heat
        self._widgets_built = False
        self._static_refresh_job: str | None = None
        self._static_refresh_fp: tuple[Any, ...] | None = None
        self._static_refresh_inflight = False

        try:
            self._build_chrome()
            self._build_body(defer_heavy=defer_heavy)
            self._widgets_built = True
            self.lbl_status.configure(
                text=(
                    "Synchronized comparison — pin Session A/B, then Play"
                    if ENABLE_DETAILED_COMPARE
                    else "Static comparison — cached sessions only"
                ),
                fg=MUTED,
            )
        except Exception as exc:
            msg = log_compare_exception(self.gui, "controller_build", exc)
            try:
                if getattr(self, "lbl_status", None) is not None:
                    self.lbl_status.configure(text=msg, fg=DANGER)
            except Exception:
                pass

        # First paint is performed by on_activate_compare_tab / enter — do not
        # schedule a competing deferred refresh that stacks on the UI thread.
        log_compare_event(
            self.gui,
            "controller_ready",
            compare_state=self.compare_state.value,
            generation=self._compare_load_gen,
            static=self._static_compare_only,
            detailed_enabled=bool(ENABLE_DETAILED_COMPARE),
        )

    @property
    def compare_loading(self) -> bool:
        """Backward-compatible busy flag (True only while LOADING)."""
        return self.compare_state == CompareState.LOADING

    @compare_loading.setter
    def compare_loading(self, value: bool) -> None:
        if value:
            self._set_compare_state(CompareState.LOADING)
        elif self.compare_state == CompareState.LOADING:
            self._set_compare_state(CompareState.IDLE)

    def _set_compare_state(self, state: CompareState, *, reason: str = "") -> None:
        prev = getattr(self, "compare_state", CompareState.IDLE)
        self.compare_state = state
        if prev != state:
            log_compare_event(
                self.gui,
                "state",
                from_state=prev.value,
                to_state=state.value,
                reason=reason or "-",
                generation=self._compare_load_gen,
            )

    def _schedule_deferred(self, delay_ms: int, callback) -> None:
        """Queue a cancellable idle/after job (cancelled when leaving the tab)."""
        try:
            job = self.gui.root.after(max(0, int(delay_ms)), callback)
            self._deferred_jobs.append(job)
        except tk.TclError:
            try:
                callback()
            except Exception:
                logger.debug("deferred compare callback failed", exc_info=True)

    def cancel_deferred_work(self) -> None:
        """Pause hidden animations and cancel staged open/paint jobs."""
        for job in list(self._deferred_jobs):
            try:
                self.gui.root.after_cancel(job)
            except (tk.TclError, ValueError):
                pass
        self._deferred_jobs.clear()
        for attr in (
            "_visual_refresh_job",
            "_scroll_watch_job",
            "_scroll_wheel_job",
            "_static_refresh_job",
            "_snapshot_poll_job",
            "_compare_timeout_job",
            "_video_resize_job",
        ):
            job = getattr(self, attr, None)
            if job is None:
                continue
            try:
                self.gui.root.after_cancel(job)
            except (tk.TclError, ValueError):
                pass
            setattr(self, attr, None)
        if getattr(self, "_playback_resume_when_leaving", False):
            self._playback_resume_when_leaving = False
            try:
                self._resume_main_playback()
            except Exception:
                logger.debug("resume playback on Compare leave failed", exc_info=True)
        if self._playing:
            try:
                self.pause(paint=False)
            except TypeError:
                self.pause()

    def on_leave_compare_tab(self) -> None:
        """Leave-tab cleanup: stop play; keep in-flight Session B analysis."""
        try:
            from stablewalk.ui.tk.ui_freeze_debug import log_compare_enter_exit

            log_compare_enter_exit(
                "exit",
                compare_state=self.compare_state.value,
                generation=self._compare_load_gen,
            )
        except Exception:
            pass
        log_compare_event(
            self.gui,
            "cleanup_started",
            compare_state=self.compare_state.value,
            generation=self._compare_load_gen,
        )
        try:
            if self._playing:
                try:
                    self.pause(paint=False)
                except TypeError:
                    self.pause()
            # Session B analysis must keep running off-tab. Cancelling the load
            # bumps generation and orphans the worker, which then blocks every
            # later "Analyze Session B" click.
            if self.compare_state != CompareState.LOADING:
                self.cancel_deferred_work()
            else:
                job = getattr(self, "_visual_refresh_job", None)
                if job is not None:
                    try:
                        self.gui.root.after_cancel(job)
                    except (tk.TclError, ValueError):
                        pass
                    self._visual_refresh_job = None
            job = getattr(self, "_static_refresh_job", None)
            if job is not None:
                try:
                    self.gui.root.after_cancel(job)
                except (tk.TclError, ValueError):
                    pass
                self._static_refresh_job = None
            self.release_temp_buffers()
        finally:
            log_compare_event(
                self.gui,
                "cleanup_finished",
                compare_state=self.compare_state.value,
                generation=self._compare_load_gen,
            )

    def on_activate_compare_tab(self) -> None:
        """Revisit Compare: reuse existing widgets; cache-only chrome refresh."""
        try:
            try:
                from stablewalk.ui.tk.ui_freeze_debug import log_compare_enter_exit

                log_compare_enter_exit(
                    "enter",
                    compare_state=self.compare_state.value,
                    generation=self._compare_load_gen,
                    static=self._static_compare_only,
                )
            except Exception:
                pass
            log_compare_event(
                self.gui,
                "tab_clicked",
                compare_state=self.compare_state.value,
                generation=self._compare_load_gen,
                static=self._static_compare_only,
            )
            # Never recreate the tab; never start MediaPipe / OpenSim / workers.
            if self.compare_state == CompareState.LOADING:
                analyzing_b = getattr(self, "_analyzing_session_b_key", None)
                status = (
                    "Analyzing Session B..."
                    if analyzing_b
                    else COMPARE_ALREADY_LOADING_MESSAGE
                )
                try:
                    self.lbl_status.configure(text=status, fg=WARNING)
                except Exception:
                    pass
                self._sync_loading_banner()
                self._arm_snapshot_poll()
                return
            # Analysis completion already pins the current workspace snapshot.
            # Restore that cached snapshot to A without rebuilding analysis.
            self._sync_current_session_a()
            self._refresh_static_compare(force=False)
            # Finish side-by-side visuals if the first deferred pass was skipped.
            if getattr(self, "_detailed_comparison", False):
                if not getattr(self, "_videos_built", False):
                    self._schedule_deferred(0, self._deferred_build_detailed_videos)
                elif not getattr(self, "_skeletons_built", False):
                    self._schedule_deferred(0, self._deferred_build_detailed_skeletons)
                elif not getattr(self, "_lite_chart_built", False) or not getattr(
                    self, "_graphs_built", False
                ):
                    self._schedule_deferred(0, self._deferred_build_detailed_graphs)
                else:
                    # Always repaint when the workspace already exists.
                    # Static chrome refresh must not swallow Session B visuals
                    # by clearing `_visuals_dirty` before this branch runs.
                    self._queue_detailed_visual_refresh()
                try:
                    self._sync_session_b_cta()
                except Exception:
                    pass
        except Exception as exc:
            msg = log_compare_exception(self.gui, "on_activate_compare_tab", exc)
            try:
                self.lbl_status.configure(text=msg, fg=DANGER)
            except Exception:
                pass
            self._set_compare_state(CompareState.FAILED, reason="activate_exception")

    def _sync_current_session_a(self) -> bool:
        """Select the already-pinned current recording as A; never re-analyze it."""
        cache = ensure_compare_cache(self.gui)
        demo = getattr(self.gui, "_active_demo_gait", None)
        if demo is not None:
            demo_key = str(getattr(demo, "key", "") or "")
            alias = f"current:{demo_key}" if demo_key else ""
            current_key = alias if alias and cache.get(alias) is not None else demo_key
        else:
            source = getattr(self.gui, "_session_display_src", None) or getattr(
                self.gui, "_current_source", ""
            )
            current_key = Path(str(source)).stem if source else ""
        if not current_key:
            return False
        current = cache.get(current_key)
        if current is None:
            return False
        try:
            analyzed, _reason = snapshot_is_analyzed(current)
        except Exception:
            analyzed = bool(getattr(current, "ready", False))
        if not analyzed:
            return False
        if cache.left_key != current_key:
            cache.set_slot("left", current_key)
            self._cached_comparison = None
            self._static_refresh_fp = None
        if cache.right_key == current_key:
            cache.right_key = ""
            self._cached_comparison = None
            self._static_refresh_fp = None
            right_var = getattr(self, "right_key_var", None)
            if right_var is not None:
                try:
                    right_var.set("")
                except (AttributeError, tk.TclError):
                    pass
        left_var = getattr(self, "left_key_var", None)
        if left_var is not None:
            try:
                left_var.set(current_key)
            except (AttributeError, tk.TclError):
                pass
        return True

    def cancel_compare_load(self, *, reason: str = "user_cancel") -> None:
        """Cancel in-flight Compare load; keep the rest of the app usable."""
        log_compare_event(
            self.gui,
            "cancel",
            reason=reason,
            generation=self._compare_load_gen,
            compare_state=self.compare_state.value,
        )
        # Invalidate worker results still in flight.
        self._compare_load_gen += 1
        with self._snapshot_build_lock:
            self._compare_pending_keys.clear()
        self._loading_keys.clear()
        self._analyzing_session_b_key = None
        self._demo_compare_load_pending = False
        self._compare_load_started_monotonic = None
        self.cancel_deferred_work()
        self._set_compare_state(CompareState.CANCELLED, reason=reason)
        self._sync_loading_banner()
        try:
            self.lbl_status.configure(
                text="Comparison cancelled — app remains available.",
                fg=MUTED,
            )
        except Exception:
            pass
        self._restore_playback_after_compare_load()

    def _is_compare_tab_active(self) -> bool:
        notebook = getattr(self.gui, "_dashboard_notebook", None)
        if notebook is None:
            return True
        try:
            return notebook.nametowidget(notebook.select()) is getattr(
                self.gui, "_tab_compare", None
            )
        except (tk.TclError, KeyError):
            return False

    def _section_visible(self, widget: tk.Misc | None) -> bool:
        """True when *widget* intersects the Compare scroll viewport."""
        if widget is None:
            return False
        canvas = getattr(self, "_compare_scroll_canvas", None)
        if canvas is None:
            return True
        try:
            if int(widget.winfo_height()) <= 1:
                return False
            top = int(widget.winfo_rooty())
            bot = top + int(widget.winfo_height())
            ctop = int(canvas.winfo_rooty())
            cbot = ctop + max(int(canvas.winfo_height()), 1)
            return bot >= ctop - 8 and top <= cbot + 8
        except tk.TclError:
            return True

    def _arm_snapshot_poll(self) -> None:
        if self._snapshot_poll_job is not None:
            return
        # Explicit Session B analysis can start a worker even in static mode.
        if (
            getattr(self, "_static_compare_only", True)
            and not self._loading_keys
            and not getattr(self, "_analyzing_session_b_key", None)
        ):
            return
        try:
            self._snapshot_poll_job = self.gui.root.after(
                50, self._drain_snapshot_results
            )
        except tk.TclError:
            self._snapshot_poll_job = None

    @staticmethod
    def _safe_draw(canvas: Any) -> None:
        """Redraw only if the TkAgg widget still exists and is on screen."""
        from stablewalk.ui.tk.redraw_policy import skip_unmapped_canvas_paint
        from stablewalk.ui.tk.tk_lifecycle import safe_canvas_draw_idle

        if skip_unmapped_canvas_paint(canvas):
            return
        safe_canvas_draw_idle(canvas)

    def request_refresh(self) -> None:
        """Coalesced refresh of cached chrome and the detailed visual workspace."""
        if getattr(self, "_static_compare_only", True):
            if self._static_refresh_job is not None:
                return

            def _run() -> None:
                self._static_refresh_job = None
                try:
                    self._refresh_static_compare(force=False)
                except Exception as exc:
                    msg = log_compare_exception(self.gui, "request_refresh", exc)
                    try:
                        self.lbl_status.configure(text=msg, fg=DANGER)
                    except Exception:
                        pass

            try:
                self._static_refresh_job = self.gui.root.after(16, _run)
            except tk.TclError:
                _run()
            return
        try:
            self._refresh_static_compare(force=True)
        except Exception as exc:
            msg = log_compare_exception(self.gui, "request_refresh", exc)
            try:
                self.lbl_status.configure(text=msg, fg=DANGER)
            except Exception:
                pass
            return

        self._visuals_dirty = True
        self._paths_dirty = True
        self._chart_dirty = True
        self._heat_dirty = True
        if self.compare_loading or self._loading_keys:
            return
        self._queue_detailed_visual_refresh()

    def _queue_detailed_visual_refresh(self) -> None:
        """Queue one main-thread repaint after cached Session B becomes ready."""
        if getattr(self, "_static_compare_only", True):
            return
        if self._visual_refresh_job is not None:
            return
        try:
            self._visual_refresh_job = self.gui.root.after(
                0, self._run_deferred_visual_refresh
            )
        except tk.TclError:
            self._run_deferred_visual_refresh()

    def _show_compare_error(self, message: str) -> None:
        """Surface an error in the Compare status area (never a black window)."""
        lbl = getattr(self, "lbl_compare_error", None)
        if lbl is None:
            return
        try:
            lbl.configure(text=str(message or ""), fg=DANGER)
            lbl.grid()
        except tk.TclError:
            pass

    def _clear_compare_error(self) -> None:
        lbl = getattr(self, "lbl_compare_error", None)
        if lbl is None:
            return
        try:
            lbl.configure(text="")
            lbl.grid_remove()
        except tk.TclError:
            pass

    def _static_cache_fingerprint(self) -> tuple[Any, ...]:
        cache = ensure_compare_cache(self.gui)
        left = cache.left()
        right = self._session_b_snap()
        return (
            cache.left_key,
            cache.right_key,
            id(left) if left is not None else None,
            id(right) if right is not None else None,
            id(left.metrics) if left is not None else None,
            id(right.metrics) if right is not None else None,
        )

    def _refresh_static_compare(self, *, force: bool = False) -> None:
        """Cache-only metric cards + Δ table. Never starts workers or animations."""
        if not getattr(self, "_widgets_built", False):
            return
        if self._static_refresh_inflight:
            return
        fp = self._static_cache_fingerprint()
        if (
            not force
            and self.compare_state == CompareState.READY
            and fp == getattr(self, "_static_refresh_fp", None)
        ):
            return
        if self._playing:
            try:
                self.pause(paint=False)
            except Exception:
                self._playing = False
        self._static_refresh_inflight = True
        try:
            self._refresh_chrome_only(fetch_missing=False)
            self._fill_differences_table()
        except Exception as exc:
            msg = log_compare_exception(self.gui, "refresh_static_compare", exc)
            try:
                self.lbl_status.configure(text=msg, fg=DANGER)
                self._show_compare_error(msg)
            except Exception:
                pass
            self._set_compare_state(CompareState.FAILED, reason="static_refresh_failed")
            return
        finally:
            self._static_refresh_inflight = False

        cache = ensure_compare_cache(self.gui)
        left_ok = cache.get(cache.left_key) is not None if cache.left_key else False
        right_ok = self._session_b_snap() is not None
        validation = validate_compare_slots(
            cache,
            loading_keys=list(self._loading_keys),
            right_override=self._session_b_snap(),
        )
        analyzing_b = bool(getattr(self, "_analyzing_session_b_key", None))
        session_b_err = str(getattr(self, "_session_b_error", "") or "").strip()
        if self._loading_keys or analyzing_b:
            self._set_compare_state(CompareState.LOADING, reason="static_cache_loading")
        elif validation.can_compare:
            self._set_compare_state(CompareState.READY, reason="static_cache_ready")
            self._clear_compare_error()
        elif session_b_err:
            if self.compare_state != CompareState.FAILED:
                self._set_compare_state(
                    CompareState.FAILED, reason="static_cache_session_b_error"
                )
        else:
            self._set_compare_state(CompareState.IDLE, reason="static_cache_incomplete")
        self._static_refresh_fp = fp
        # Detailed Compare paints videos/skeletons/graphs on a later pass.
        # Never clear `_visuals_dirty` here — chrome already marks it True,
        # and wiping it skips Video B after Session B finishes off-tab.
        if validation.can_compare and getattr(self, "_static_compare_only", True):
            self._visuals_dirty = False
        log_compare_event(
            self.gui,
            "static_ready",
            left=cache.left_key,
            right=cache.right_key,
            left_cached=left_ok,
            right_cached=right_ok,
            generation=self._compare_load_gen,
        )

    def reload_comparison(self) -> None:
        """User-triggered re-read of cached sessions (never analyzes)."""
        cache = ensure_compare_cache(self.gui)
        validation = validate_compare_slots(
            cache,
            loading_keys=list(self._loading_keys),
            right_override=self._session_b_snap(),
        )
        if not validation.can_reload:
            try:
                self.lbl_status.configure(
                    text=validation.message or "Reload disabled until both sessions are valid.",
                    fg=WARNING,
                )
                self._show_compare_error(validation.message)
            except Exception:
                pass
            self._sync_reload_button(validation)
            return
        log_compare_event(self.gui, "reload_comparison")
        self.cancel_deferred_work()
        if self.compare_state == CompareState.LOADING:
            self.cancel_compare_load(reason="reload_while_loading")
        try:
            self._static_refresh_fp = None
            self._cached_comparison = None  # force rebuild of cached rows
            self._refresh_static_compare(force=True)
            if self.compare_state == CompareState.READY:
                self.lbl_status.configure(
                    text="Comparison reloaded from cache", fg=SUCCESS
                )
        except Exception as exc:
            msg = log_compare_exception(self.gui, "reload_comparison", exc)
            try:
                self.lbl_status.configure(text=msg, fg=DANGER)
            except Exception:
                pass

    def _run_deferred_visual_refresh(self) -> None:
        """Detailed-only path; static Compare never paints videos/charts."""
        self._visual_refresh_job = None
        if getattr(self, "_static_compare_only", True):
            self._refresh_static_compare()
            return
        if not self._is_compare_tab_active():
            self._visuals_dirty = True
            return
        if self.compare_loading or self._loading_keys:
            self._visuals_dirty = True
            return
        if self._detailed_comparison:
            if not self._videos_built:
                self._schedule_deferred(0, self._deferred_build_detailed_videos)
                return
            if not self._skeletons_built:
                self._schedule_deferred(0, self._deferred_build_detailed_skeletons)
                return
            if not self._lite_chart_built or not self._graphs_built:
                self._schedule_deferred(0, self._deferred_build_detailed_graphs)
                return
        try:
            self._refresh_visible_visuals(force=True)
        except Exception as exc:
            msg = log_compare_exception(self.gui, "deferred_visual_refresh", exc)
            try:
                self.lbl_status.configure(text=msg, fg=DANGER)
                self._show_compare_error(msg)
            except Exception:
                pass
            self._visuals_dirty = True

    def _deferred_open_stage_metrics(self) -> None:
        """Legacy hook — static Compare only refreshes cache chrome."""
        self._refresh_static_compare()

    def _deferred_open_wait_for_sessions(self) -> None:
        """Legacy hook — no background session prep in static Compare."""
        self._refresh_static_compare()

    def _deferred_open_stage_lite_chart(self) -> None:
        """Disabled in static Compare (no per-frame Matplotlib)."""
        self._refresh_static_compare()

    def _set_compare_controls_busy(self, busy: bool) -> None:
        """Disable only interactive controls owned by the Compare workspace."""
        saved = getattr(self, "_compare_control_states", None)
        if saved is None:
            saved = {}
            self._compare_control_states = saved
        if busy and saved:
            return

        def _walk(parent: tk.Misc):
            for child in parent.winfo_children():
                yield child
                yield from _walk(child)

        control_classes = {
            "TButton",
            "TCheckbutton",
            "TCombobox",
            "TRadiobutton",
            "TScale",
            "Button",
            "Checkbutton",
            "Radiobutton",
            "Scale",
        }
        cancel = getattr(self, "btn_cancel_compare", None)
        analyze_keep = set(self._iter_analyze_session_b_buttons())
        if busy:
            for widget in _walk(self.root):
                if widget is cancel or widget in analyze_keep:
                    continue
                try:
                    if str(widget.cget("text")) == "Analyze Session B":
                        continue
                except tk.TclError:
                    pass
                try:
                    if str(widget.winfo_class()) not in control_classes:
                        continue
                    state_fn = getattr(widget, "state", None)
                    if callable(state_fn):
                        saved[widget] = tuple(state_fn())
                        state_fn(["disabled"])
                    else:
                        saved[widget] = str(widget.cget("state"))
                        widget.configure(state=tk.DISABLED)
                except (tk.TclError, TypeError):
                    continue
            return

        for widget, prior in list(saved.items()):
            try:
                state_fn = getattr(widget, "state", None)
                if callable(state_fn):
                    state_fn(["!disabled"])
                    if isinstance(prior, tuple) and "disabled" in prior:
                        state_fn(["disabled"])
                else:
                    widget.configure(state=str(prior))
            except (tk.TclError, TypeError):
                pass
        saved.clear()

    def _sync_loading_banner(self, message: str | None = None) -> None:
        """Show/hide the in-tab loader (never a Toplevel / grab / second Tk)."""
        banner = getattr(self, "_loading_banner", None)
        lbl = getattr(self, "lbl_loading_banner", None)
        if banner is None or lbl is None:
            return
        loading = bool(self.compare_loading or self._loading_keys)
        if not loading:
            progress = getattr(self, "compare_analysis_progress", None)
            if progress is not None:
                try:
                    progress.stop()
                except tk.TclError:
                    pass
            self._set_compare_controls_busy(False)
            try:
                banner.grid_remove()
            except tk.TclError:
                pass
            return
        cache = ensure_compare_cache(self.gui)
        if message is None:
            right_loading = cache.right_key in self._loading_keys
            left_loading = cache.left_key in self._loading_keys
            if (
                right_loading
                and not left_loading
                and getattr(self, "_analyzing_session_b_key", None)
            ):
                message = "Analyzing Session B..."
            elif right_loading and not left_loading:
                message = "Loading Session B… UI stays responsive."
            elif left_loading and not right_loading:
                message = "Loading Session A… UI stays responsive."
            else:
                message = "Loading comparison sessions… UI stays responsive."
        try:
            self._set_compare_controls_busy(True)
            lbl.configure(text=message, fg=TEXT)
            banner.grid()
            progress = getattr(self, "compare_analysis_progress", None)
            if progress is not None:
                progress.start(12)
        except tk.TclError:
            pass
        try:
            self.lbl_status.configure(text=message.replace(" UI stays responsive.", ""), fg=MUTED)
        except Exception:
            pass
        try:
            self._sync_session_b_cta()
        except Exception:
            pass

    def _deferred_open_stage_videos(self) -> None:
        """Deprecated heavy stage — lite open never calls this."""
        if self._detailed_comparison:
            self._deferred_build_detailed_videos()

    def _deferred_open_stage_skeletons(self) -> None:
        """Deprecated heavy stage — lite open never calls this."""
        if self._detailed_comparison:
            self._deferred_build_detailed_skeletons()

    def _deferred_open_stage_graphs(self) -> None:
        """Deprecated heavy stage — lite open never calls this."""
        if self._detailed_comparison:
            self._deferred_build_detailed_graphs()

    def _arm_scroll_watch(self) -> None:
        """Scroll watch only matters for detailed multi-chart row."""
        if not self._detailed_comparison:
            return
        canvas = getattr(self, "_compare_scroll_canvas", None)
        if canvas is None:
            return

        def _check(_event: object | None = None) -> None:
            if not self._is_compare_tab_active() or not self._detailed_comparison:
                self._scroll_watch_job = None
                return
            if self._section_visible(getattr(self, "_detail_graphs_section", None)):
                self._ensure_detailed_graphs_built()
                self._refresh_detailed_charts(force=False)
            try:
                self._scroll_watch_job = canvas.after(160, _check)
            except tk.TclError:
                self._scroll_watch_job = None

        if not getattr(self, "_scroll_watch_bound", False):
            def _wheel_check(_e=None) -> None:
                job = getattr(self, "_scroll_wheel_job", None)
                if job is not None:
                    try:
                        canvas.after_cancel(job)
                    except (tk.TclError, ValueError, TypeError):
                        pass
                try:
                    self._scroll_wheel_job = canvas.after(40, _check)
                except tk.TclError:
                    self._scroll_wheel_job = None

            try:
                canvas.bind("<ButtonRelease-1>", _check, add="+")
                canvas.bind("<MouseWheel>", _wheel_check, add="+")
                self._scroll_watch_bound = True
            except tk.TclError:
                pass

        if self._scroll_watch_job is not None:
            return
        try:
            self._scroll_watch_job = canvas.after(200, _check)
        except tk.TclError:
            self._scroll_watch_job = None

    def _ensure_skeletons_built(self) -> None:
        if self._skeletons_built:
            return
        if not self._detailed_comparison:
            return
        host_a = getattr(self, "_skel_a_host", None)
        host_b = getattr(self, "_skel_b_host", None)
        if host_a is not None and host_b is not None:
            for host in (host_a, host_b):
                for child in list(host.winfo_children()):
                    try:
                        child.destroy()
                    except tk.TclError:
                        pass
            self._skel_b_cta = None
            self._build_split_skeletons(host_a, host_b)
            self._skeletons_built = True
            try:
                self._sync_session_b_cta()
            except Exception:
                pass
            return
        parent = getattr(self, "_skel_section", None)
        if parent is None:
            return
        placeholder = getattr(self, "_skel_placeholder", None)
        if placeholder is not None:
            try:
                placeholder.destroy()
            except tk.TclError:
                pass
            self._skel_placeholder = None
        for child in list(parent.winfo_children()):
            try:
                child.destroy()
            except tk.TclError:
                pass
        self._build_skeletons_row(parent)
        self._skeletons_built = True
        bind_wheel = getattr(self, "_compare_bind_wheel", None)
        if callable(bind_wheel):
            bind_wheel(parent)
        sync = getattr(self, "_sync_compare_scroll", None)
        if callable(sync):
            sync()

    def _ensure_graphs_built(self) -> None:
        """Compatibility: ensure lite knee chart (never auto-builds path/heatmap)."""
        self._ensure_lite_chart_built()

    def _refresh_detailed_charts(self, *, force: bool = False) -> None:
        """Redraw detailed path + heatmap (knee chart lives in lite section)."""
        if not self._graphs_built:
            return
        if force or self._paths_dirty:
            self._redraw_paths()
            self._paths_dirty = False
        if force or self._heat_dirty:
            self._redraw_heatmap()
            self._heat_dirty = False
        self._update_path_markers()
        self._update_heat_playhead()

    # ── chrome ──────────────────────────────────────────────────────────

    def _build_chrome(self) -> None:
        bar = ttk.Frame(self.root)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))
        bar.columnconfigure(2, weight=1)
        bar.columnconfigure(5, weight=1)

        tk.Label(
            bar,
            text="Sessions to compare",
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(0, PAD_SM))

        self.left_key_var = tk.StringVar(value="normal")
        self.right_key_var = tk.StringVar(value="abnormal")

        self.lbl_slot_a = ttk.Label(bar, text="Session A", style="Card.TLabel")
        self.lbl_slot_a.grid(
            row=0, column=1, padx=(PAD_SM, 2)
        )
        self.cmb_left = ttk.Combobox(
            bar, textvariable=self.left_key_var, width=18, state="readonly"
        )
        self.cmb_left.grid(row=0, column=2, padx=(0, 2), sticky="ew")
        self.cmb_left.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._on_slot_change(changed_slot="left"),
        )
        ttk.Button(
            bar,
            text="Browse…",
            width=8,
            style="Compact.TButton",
            command=lambda: self.open_session_picker(slot="left"),
        ).grid(row=0, column=3, padx=(0, PAD_SM))

        self.lbl_slot_b = ttk.Label(bar, text="Session B", style="Card.TLabel")
        self.lbl_slot_b.grid(
            row=0, column=4, padx=(PAD_SM, 2)
        )
        self.cmb_right = ttk.Combobox(
            bar, textvariable=self.right_key_var, width=18, state="readonly"
        )
        self.cmb_right.grid(row=0, column=5, padx=(0, 2), sticky="ew")
        self.cmb_right.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._on_slot_change(changed_slot="right"),
        )
        ttk.Button(
            bar,
            text="Browse…",
            width=8,
            style="Compact.TButton",
            command=lambda: self.open_session_picker(slot="right"),
        ).grid(row=0, column=6, padx=(0, PAD_SM))

        create_tooltip(
            self.cmb_left,
            "Session A — must be a completed analyzed session (not a filename alone)",
        )
        create_tooltip(
            self.cmb_right,
            "Session B — must be a completed analyzed session (not a filename alone)",
        )

        actions = ttk.Frame(bar)
        actions.grid(row=1, column=0, columnspan=7, sticky="ew", pady=(PAD_XS, 0))
        left_actions = ttk.Frame(actions)
        left_actions.pack(side=tk.LEFT, fill=tk.X, expand=True)
        right_actions = ttk.Frame(actions)
        right_actions.pack(side=tk.RIGHT)
        self._compare_actions_host = actions
        self._compare_left_actions = left_actions
        self._compare_right_actions = right_actions

        self.btn_analyze_b = ttk.Button(
            left_actions,
            text="Analyze Session B",
            style="Accent.TButton",
            command=self.analyze_session_b,
        )
        self.btn_analyze_b.pack(side=tk.LEFT, padx=(0, PAD_XS))
        self._register_analyze_session_b_button(self.btn_analyze_b)
        create_tooltip(
            self.btn_analyze_b,
            "Build Session B from saved poses / demo cache in the background",
        )
        ttk.Button(
            left_actions,
            text="Use current as A",
            style="Compact.TButton",
            command=lambda: self.pin_current(slot="left"),
        ).pack(side=tk.LEFT, padx=(0, PAD_XS))
        ttk.Button(
            left_actions,
            text="Use current as B",
            style="Compact.TButton",
            command=lambda: self.pin_current(slot="right"),
        ).pack(side=tk.LEFT, padx=(0, PAD_XS))

        self.btn_load_demo_comparison = ttk.Button(
            left_actions,
            text="Load demo comparison",
            style="Accent.TButton",
            command=self.load_demo_comparison,
        )
        self.btn_load_demo_comparison.pack(side=tk.LEFT, padx=(0, PAD_XS))
        create_tooltip(
            self.btn_load_demo_comparison,
            "Load Normal (Session A) vs Abnormal (Session B) and show Stability, "
            "Symmetry, Cadence, and ROM deltas.",
        )

        # Extra presets stay available only when detailed Compare is enabled.
        if ENABLE_DETAILED_COMPARE:
            examples_menu = tk.Menu(left_actions, tearoff=0)
            examples_menu.add_command(
                label="Normal vs Abnormal",
                command=lambda: self.apply_preset("normal", "abnormal"),
            )
            examples_menu.add_command(
                label="Normal vs Fast Walking",
                command=lambda: self.apply_preset("normal", "athletic"),
            )
            examples_menu.add_command(
                label="Abnormal vs Fast Walking",
                command=lambda: self.apply_preset("abnormal", "athletic"),
            )
            btn_examples = ttk.Menubutton(
                left_actions,
                text="Load examples…",
                style="Compact.TButton",
                direction="below",
            )
            btn_examples.configure(menu=examples_menu)
            btn_examples.pack(side=tk.LEFT, padx=(0, PAD_XS))

        self.btn_run_compare = ttk.Button(
            left_actions,
            text="Compare",
            style="Accent.TButton",
            command=self.reload_comparison,
            state=tk.DISABLED,
        )
        self.btn_run_compare.pack(side=tk.LEFT, padx=(0, PAD_XS))
        create_tooltip(
            self.btn_run_compare,
            "Enabled only when Session A and Session B are both analyzed and distinct.",
        )
        self.btn_reload_comparison = ttk.Button(
            left_actions,
            text="Reload Comparison",
            command=self.reload_comparison,
            state=tk.DISABLED,
        )
        self.btn_reload_comparison.pack(side=tk.LEFT, padx=(0, PAD_XS))
        create_tooltip(
            self.btn_reload_comparison,
            "Enabled only when both sessions are analyzed and distinct — cache re-read only",
        )
        self.btn_reset_comparison = ttk.Button(
            left_actions,
            text="Reset",
            command=self.reset_comparison,
            style="Compact.TButton",
        )
        self.btn_reset_comparison.pack(side=tk.LEFT, padx=(0, PAD_XS))
        create_tooltip(
            self.btn_reset_comparison,
            "Clear Session B, comparison outputs, and stale worker state. Session A is kept.",
        )

        self.btn_export_report = ttk.Button(
            right_actions,
            text="Export Lab Report",
            style="Accent.TButton",
            command=self.export_report,
            state=tk.DISABLED,
        )
        self.btn_export_report.pack(side=tk.RIGHT)
        create_tooltip(
            self.btn_export_report,
            "Export a printable lab comparison report (requires two ready sessions).\n\n"
            + (DISCLAIMER_COMPARE or ""),
            wraplength=420,
        )

        self.lbl_status = tk.Label(
            right_actions, text="", bg=PANEL, fg=MUTED, font=FONT_UI_SM, anchor="e"
        )
        self.lbl_status.pack(side=tk.RIGHT, padx=(0, PAD_SM))
        create_tooltip(
            self.lbl_status,
            (
                "Synchronized timeline — Play advances Video A/B and skeletons together"
                if ENABLE_DETAILED_COMPARE
                else "Static comparison — cached metrics only (no playback)"
            ),
        )

        # Compatibility handle — hint lives on page subtitle / status tooltip.
        self.lbl_timeline_hint = tk.Label(bar, text="", bg=PANEL, fg=MUTED)
        self.lbl_timeline_hint.grid_remove()

        self.btn_open_detailed = ttk.Button(
            bar,
            text="Open Detailed Comparison",
            width=22,
            command=self._open_detailed_comparison,
        )
        self.btn_open_detailed.grid(row=1, column=6, padx=(PAD_SM, 0), sticky="e")
        create_tooltip(
            self.btn_open_detailed,
            "Load dual video, 3D skeletons, COM paths and heatmap (heavier)",
        )
        # Side-by-side workspace is the default when detailed Compare is enabled.
        try:
            self.btn_open_detailed.grid_remove()
            self.btn_open_detailed.configure(state=tk.DISABLED)
        except tk.TclError:
            pass

        # Skeleton mode controls exist only for the optional detailed path.
        self._skel_mode_controls = ()
        if ENABLE_DETAILED_COMPARE:
            rb_side = ttk.Radiobutton(
                bar,
                text="Two Skeletons",
                variable=self._skeleton_mode,
                value="side",
                command=self._on_skeleton_mode,
            )
            rb_overlay = ttk.Radiobutton(
                bar,
                text="Overlay Skeletons",
                variable=self._skeleton_mode,
                value="overlay",
                command=self._on_skeleton_mode,
            )
            self._skel_mode_controls = (rb_side, rb_overlay)
            for w in self._skel_mode_controls:
                try:
                    w.grid_remove()
                except tk.TclError:
                    pass

        # Dedicated multi-line error/status area under selectors.
        self.lbl_compare_error = tk.Label(
            bar,
            text="",
            bg=PANEL,
            fg=DANGER,
            font=FONT_UI_XS,
            anchor="w",
            justify=tk.LEFT,
            wraplength=900,
        )
        self.lbl_compare_error.grid(
            row=2, column=0, columnspan=7, sticky="ew", pady=(PAD_XS, 0)
        )
        self.lbl_compare_error.grid_remove()

        def _sync_compare_error_wrap(_event: object | None = None) -> None:
            try:
                width = int(bar.winfo_width() or 0)
            except tk.TclError:
                return
            if width < 80:
                return
            try:
                self.lbl_compare_error.configure(wraplength=max(240, width - 24))
            except tk.TclError:
                pass

        bar.bind("<Configure>", _sync_compare_error_wrap, add="+")
        bar.bind(
            "<Configure>",
            lambda event: self._reflow_compare_chrome(int(event.width)),
            add="+",
        )

        # Non-blocking in-tab loader — never a second Tk / Toplevel / grab.
        self._loading_banner = tk.Frame(
            self.root, bg=ELEVATED, highlightthickness=1, highlightbackground=BORDER
        )
        self._loading_banner.grid(row=1, column=0, sticky="ew", padx=0, pady=(0, PAD_XS))
        self._loading_banner.grid_remove()
        self._loading_banner.columnconfigure(0, weight=1)
        self.lbl_loading_banner = tk.Label(
            self._loading_banner,
            text="Preparing comparison sessions…",
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_UI_SM,
            anchor="w",
            padx=PAD_SM,
            pady=PAD_XS,
        )
        self.lbl_loading_banner.grid(row=0, column=0, sticky="ew")
        self.btn_cancel_compare = ttk.Button(
            self._loading_banner,
            text="Cancel",
            width=10,
            command=lambda: self.cancel_compare_load(reason="user_cancel"),
        )
        self.btn_cancel_compare.grid(row=0, column=1, padx=(PAD_SM, PAD_SM), pady=PAD_XS)
        self.compare_analysis_progress = ttk.Progressbar(
            self._loading_banner,
            mode="indeterminate",
            length=180,
        )
        self.compare_analysis_progress.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=PAD_SM,
            pady=(0, PAD_XS),
        )
        create_tooltip(
            self.btn_cancel_compare,
            "Cancel this comparison load — the rest of the app stays usable",
        )

    def _build_body(self, *, defer_heavy: bool = True) -> None:
        """Side-by-side synchronized Compare workspace (or static fallback)."""
        if ENABLE_DETAILED_COMPARE:
            self._build_side_by_side_body(defer_heavy=defer_heavy)
            return
        self._build_static_body()

    def _build_static_body(self) -> None:
        """Metrics-only fallback when detailed Compare is disabled."""
        scroll_host = ttk.Frame(self.root)
        scroll_host.grid(row=2, column=0, sticky="nsew", pady=(0, PAD_XS))
        scroll_host.columnconfigure(0, weight=1)
        scroll_host.rowconfigure(0, weight=1)
        self._transport_host = ttk.Frame(self.root)

        sections = self._install_compare_scroll(scroll_host)
        sections.columnconfigure(0, weight=1)

        metrics = ttk.Frame(sections)
        metrics.grid(row=0, column=0, sticky="ew", pady=(0, PAD_SM))
        self._build_metrics_row(metrics)

        table_fr = ttk.LabelFrame(
            sections,
            text=format_labelframe_title("Key metrics (Δ = Session B − Session A)"),
            style="Card.TLabelframe",
            padding=DASHBOARD_VIZ_CARD_PAD,
        )
        table_fr.grid(row=1, column=0, sticky="ew", pady=(0, PAD_SM))
        table_fr.columnconfigure(0, weight=1)
        self._build_differences_table(table_fr)

        try:
            from stablewalk.ui.design_system.components import CollapsibleSection

            extra = CollapsibleSection(sections, "Additional metrics", expanded=False)
            extra.frame.grid(row=2, column=0, sticky="ew", pady=(0, PAD_SM))
            self._extra_metrics_host = extra.body
            self._build_extra_metrics_table(extra.body)
        except Exception:
            extra_fr = ttk.LabelFrame(
                sections,
                text=format_labelframe_title("Additional metrics"),
                style="Card.TLabelframe",
                padding=DASHBOARD_VIZ_CARD_PAD,
            )
            extra_fr.grid(row=2, column=0, sticky="ew", pady=(0, PAD_SM))
            self._extra_metrics_host = extra_fr
            self._build_extra_metrics_table(extra_fr)

        self._lite_chart_section = None
        self._graphs_section = None
        detailed = ttk.Frame(sections)
        detailed.grid(row=3, column=0, sticky="ew", pady=(0, PAD_SM))
        self._detailed_section = detailed
        detailed.grid_remove()
        self._video_section = ttk.Frame(detailed)
        self._skel_section = ttk.Frame(detailed)
        self._detail_graphs_section = ttk.Frame(detailed)

        report = ttk.Frame(sections)
        report.grid(row=4, column=0, sticky="ew", pady=(0, PAD_SM))
        self._build_report_section(report)
        bind_wheel = getattr(self, "_compare_bind_wheel", None)
        if callable(bind_wheel):
            bind_wheel(sections)
        self._sync_compare_scroll()

    def _compare_panel(
        self,
        parent: tk.Misc,
        title: str,
        *,
        row: int,
        column: int,
        padx: tuple[int, int] = (0, 0),
        pady: tuple[int, int] = (0, 0),
    ) -> tuple[ttk.LabelFrame, ttk.Frame]:
        """Compact labeled cell in the Compare 3-column workspace."""
        frame = ttk.LabelFrame(
            parent,
            text=format_labelframe_title(title),
            style="Card.TLabelframe",
            padding=(PAD_SM, PAD_XS),
        )
        frame.grid(row=row, column=column, sticky="nsew", padx=padx, pady=pady)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        body = ttk.Frame(frame)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        return frame, body

    def _build_side_by_side_body(self, *, defer_heavy: bool = True) -> None:
        """Left A / center diffs / right B columns, synchronized playback."""
        del defer_heavy
        scroll_host = ttk.Frame(self.root)
        scroll_host.grid(row=2, column=0, sticky="nsew", pady=(0, PAD_SM))
        scroll_host.columnconfigure(0, weight=1)
        scroll_host.rowconfigure(0, weight=1)
        self._compare_workspace_scroll_host = scroll_host

        canvas = tk.Canvas(scroll_host, bg=PANEL, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=canvas.yview)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        vsb.grid_remove()
        canvas.configure(yscrollcommand=vsb.set)
        self._compare_workspace_canvas = canvas
        self._compare_workspace_vsb = vsb

        workspace = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=workspace, anchor="nw")
        self._compare_workspace_window_id = inner_id
        workspace.columnconfigure(
            0, weight=COMPARE_COL_SIDE_WEIGHT, minsize=COMPARE_COL_SIDE_MIN_W
        )
        workspace.columnconfigure(
            1, weight=COMPARE_COL_CENTER_WEIGHT, minsize=COMPARE_COL_CENTER_MIN_W
        )
        workspace.columnconfigure(
            2, weight=COMPARE_COL_SIDE_WEIGHT, minsize=COMPARE_COL_SIDE_MIN_W
        )
        workspace.rowconfigure(0, weight=1, minsize=COMPARE_WORKSPACE_MIN_H)
        self._compare_workspace = workspace
        self._detailed_section = workspace
        self._detailed_comparison = True

        gutter = PAD_SM
        left = ttk.Frame(workspace)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, gutter))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=3, minsize=COMPARE_VIDEO_MIN_H)
        left.rowconfigure(1, weight=2, minsize=COMPARE_SKELETON_MIN_H)
        self._compare_left_col = left

        center = ttk.Frame(workspace)
        center.grid(row=0, column=1, sticky="nsew", padx=(0, gutter))
        center.columnconfigure(0, weight=1)
        for row in range(3):
            center.rowconfigure(row, weight=1, minsize=COMPARE_CHART_MIN_H)
        self._compare_center_col = center

        right = ttk.Frame(workspace)
        right.grid(row=0, column=2, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=3, minsize=COMPARE_VIDEO_MIN_H)
        right.rowconfigure(1, weight=2, minsize=COMPARE_SKELETON_MIN_H)
        self._compare_right_col = right

        self._session_a_column, self._video_a_host = self._compare_panel(
            left, "Video A", row=0, column=0, pady=(0, PAD_SM)
        )
        self._skel_a_panel, self._skel_a_host = self._compare_panel(
            left, "Skeleton A", row=1, column=0
        )
        self._diff_graph_panel, self._lite_chart_section = self._compare_panel(
            center, "Difference Graph", row=0, column=0, pady=(0, PAD_SM)
        )
        self._traj_panel, self._traj_host = self._compare_panel(
            center, "Difference Trajectory", row=1, column=0, pady=(0, PAD_SM)
        )
        self._heat_panel, self._heat_host = self._compare_panel(
            center, "Difference Heatmap", row=2, column=0
        )
        self._traj_heat_panel = self._traj_panel
        self._detail_graphs_section = self._traj_host
        self._graphs_section = self._traj_host
        self._session_b_column, video_b_body = self._compare_panel(
            right, "Video B", row=0, column=0, pady=(0, PAD_SM)
        )
        self._skel_b_panel, self._skel_b_host = self._compare_panel(
            right, "Skeleton B", row=1, column=0
        )
        self._compare_top_panel_frames = (
            self._session_a_column,
            self._diff_graph_panel,
            self._session_b_column,
        )
        # Keep A/B columns geometrically identical. Missing-B actions live in
        # the B panel overlay rather than consuming a permanent extra row.
        self._session_b_workflow_host = None
        self._video_b_host = ttk.Frame(video_b_body)
        self._video_b_host.grid(row=0, column=0, sticky="nsew")
        self._video_b_host.columnconfigure(0, weight=1)
        self._video_b_host.rowconfigure(0, weight=1)
        self._compare_stacked = False

        # Compatibility hosts used by older ensure helpers.
        self._video_section = workspace
        self._skel_section = workspace

        cache = ensure_compare_cache(self.gui)
        validation = validate_compare_slots(
            cache, loading_keys=list(self._loading_keys)
        )
        self._install_initial_compare_content(validation)

        # Synced transport (caption is the page subtitle — avoid repeating it).
        transport_host = ttk.Frame(self.root)
        transport_host.grid(row=3, column=0, sticky="ew", pady=(PAD_XS, PAD_SM))
        transport_host.columnconfigure(0, weight=1)
        self._transport_host = transport_host
        sync_row = ttk.Frame(transport_host)
        sync_row.grid(row=0, column=0, sticky="ew")
        sync_row.columnconfigure(0, weight=1)
        self._build_transport(sync_row)

        # Bottom: Difference Summary + comparison metrics (layout preserved).
        metrics_host = ttk.LabelFrame(
            self.root,
            text=format_labelframe_title("Comparison metrics (Δ = B − A)"),
            style="Card.TLabelframe",
            padding=(PAD_SM, PAD_SM),
        )
        metrics_host.grid(row=4, column=0, sticky="ew", pady=(PAD_SM, 0))
        metrics_host.columnconfigure(0, weight=1)
        metrics_host.rowconfigure(0, weight=1)
        metrics_host.rowconfigure(1, weight=0)
        self._compare_metrics_host = metrics_host

        from stablewalk.ui.tk.compare_delta_visuals import (
            install_difference_summary_panel,
        )

        summary_host = tk.Frame(metrics_host, bg=PANEL, highlightthickness=0)
        summary_host.grid(row=0, column=0, sticky="ew", pady=(0, PAD_SM))
        self._difference_summary = install_difference_summary_panel(
            summary_host, bg=PANEL
        )
        self._compare_summary_host = summary_host

        bottom_host = tk.Frame(metrics_host, bg=PANEL, highlightthickness=0)
        bottom_host.grid(row=1, column=0, sticky="ew")
        bottom_host.columnconfigure(0, weight=1)
        self._compare_bottom_host = bottom_host
        self._build_bottom_metrics_bar(bottom_host)

        waiting_host = tk.Frame(
            metrics_host,
            bg=ELEVATED,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        waiting_host.grid(row=0, column=0, rowspan=2, sticky="nsew")
        waiting_host.columnconfigure(0, weight=1)
        waiting_host.rowconfigure(0, weight=1)
        waiting_host.rowconfigure(1, weight=0)
        waiting_host.rowconfigure(2, weight=1)
        self._compare_waiting_host = waiting_host
        self._compare_waiting_label = tk.Label(
            waiting_host,
            text=f"{SESSION_B_WAITING_TITLE}\n{SESSION_B_GUIDANCE_TEXT}",
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_SM,
            anchor="center",
            justify=tk.LEFT,
            wraplength=640,
        )
        self._compare_waiting_label.grid(
            row=1, column=0, sticky="ew", padx=PAD_MD, pady=PAD_MD
        )
        self._bind_placeholder_wraplength(
            self._compare_waiting_label, waiting_host, pad=PAD_MD * 2
        )

        # Collapsible technical tables / report under metrics.
        details_host = ttk.Frame(self.root)
        details_host.grid(row=5, column=0, sticky="ew")
        details_host.columnconfigure(0, weight=1)
        self.root.rowconfigure(5, weight=0)
        self._compare_details_host = details_host
        try:
            from stablewalk.ui.design_system.components import CollapsibleSection

            details = CollapsibleSection(
                details_host,
                "Detailed metric table & laboratory report",
                expanded=False,
                bg=PANEL,
            )
            details.frame.grid(row=0, column=0, sticky="ew", pady=(PAD_XS, 0))
            body = details.body
        except Exception:
            body = details_host
        body.columnconfigure(0, weight=1)
        table_fr = ttk.Frame(body)
        table_fr.grid(row=0, column=0, sticky="ew")
        table_fr.columnconfigure(0, weight=1)
        self._build_differences_table(table_fr)
        try:
            from stablewalk.ui.design_system.components import CollapsibleSection as _CS

            extra = _CS(body, "Additional metrics", expanded=False, bg=PANEL)
            extra.frame.grid(row=1, column=0, sticky="ew", pady=(PAD_XS, 0))
            self._extra_metrics_host = extra.body
            self._build_extra_metrics_table(extra.body)
        except Exception:
            self._extra_metrics_host = body
            self._build_extra_metrics_table(body)
        report = ttk.Frame(body)
        report.grid(row=2, column=0, sticky="ew", pady=(PAD_XS, 0))
        self._build_report_section(report)

        # Also keep compact A | Δ | B cards available inside collapsible metrics area.
        cards = ttk.Frame(body)
        cards.grid(row=3, column=0, sticky="ew", pady=(PAD_SM, 0))
        self._build_metrics_row(cards)

        canvas.bind("<Configure>", self._schedule_compare_workspace_reflow, add="+")
        self.root.bind("<Configure>", self._schedule_compare_workspace_reflow, add="+")
        self._bind_compare_workspace_wheel(canvas, workspace)

        # Build real visuals in idle chunks; guidance exists only for missing data.
        self._schedule_deferred(0, self._deferred_build_detailed_videos)

    def _reflow_compare_chrome(self, width: int) -> None:
        """Stack action groups before they can overflow the Compare header."""
        left = getattr(self, "_compare_left_actions", None)
        right = getattr(self, "_compare_right_actions", None)
        if left is None or right is None:
            return
        compact = int(width) < 1200
        if compact != getattr(self, "_compare_chrome_compact", None):
            self._compare_chrome_compact = compact
            try:
                left.pack_forget()
                right.pack_forget()
                if compact:
                    left.pack(side=tk.TOP, fill=tk.X, expand=True)
                    right.pack(side=tk.TOP, fill=tk.X, pady=(PAD_XS, 0))
                else:
                    left.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    right.pack(side=tk.RIGHT)
            except tk.TclError:
                pass
        for combo in (getattr(self, "cmb_left", None), getattr(self, "cmb_right", None)):
            if combo is None:
                continue
            try:
                chars = 12 if compact else 18
                if int(width) >= 1600:
                    chars = 22
                combo.configure(width=chars)
            except tk.TclError:
                continue

    def _install_initial_compare_content(self, validation: Any) -> None:
        """Show guidance only for unavailable data, never over valid sessions."""
        if not bool(getattr(validation, "left_ready", False)):
            self._fill_compare_host_stub(
                self._video_a_host,
                "Video A",
                "Analyze the current recording to use it as Session A.",
            )
            self._fill_compare_host_stub(
                self._skel_a_host,
                "Skeleton A",
                "Session A pose data is not available yet.",
            )
        if not bool(getattr(validation, "can_compare", False)):
            self._fill_compare_host_stub(
                self._lite_chart_section,
                "Difference Graph",
                PLACEHOLDER_DIFF_GRAPH_DETAIL,
            )
            self._fill_compare_host_stub(
                getattr(self, "_traj_host", None) or self._detail_graphs_section,
                "Difference Trajectory",
                PLACEHOLDER_DIFF_PATH_DETAIL,
            )
            self._fill_compare_host_stub(
                getattr(self, "_heat_host", None),
                "Difference Heatmap",
                PLACEHOLDER_DIFF_PATH_DETAIL,
            )
        if not bool(getattr(validation, "right_ready", False)):
            message = getattr(validation, "right_reason", None) or getattr(
                validation, "message", None
            )
            self._fill_session_b_placeholder(
                self._video_b_host, kind="video", message=message
            )
            self._fill_session_b_placeholder(
                self._skel_b_host, kind="skeleton", message=message
            )
        # Workflow actions are already inside unavailable-B overlays.
        self._sync_session_b_workflow(show=False)

    def _compare_panel_slots(self) -> tuple[Any, Any, Any, Any, Any, Any] | None:
        """Return the six visualization frames, or None if the workspace is incomplete."""
        panels = (
            getattr(self, "_session_a_column", None),
            getattr(self, "_session_b_column", None),
            getattr(self, "_skel_a_panel", None),
            getattr(self, "_skel_b_panel", None),
            getattr(self, "_diff_graph_panel", None),
            getattr(self, "_traj_heat_panel", None),
        )
        if any(panel is None or not hasattr(panel, "grid") for panel in panels):
            return None
        return panels  # type: ignore[return-value]

    def _bind_compare_workspace_wheel(self, canvas: tk.Canvas, workspace: tk.Misc) -> None:
        """Scroll the Compare workspace when content is taller than the viewport."""

        def _wheel(event: tk.Event) -> str | None:
            if not getattr(self, "_compare_needs_scroll", False):
                return None
            if getattr(event, "delta", None):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                return "break"
            return None

        if not hasattr(self, "_wheel_bound_ids"):
            self._wheel_bound_ids = set()

        def _bind_wheel(widget: tk.Misc) -> None:
            try:
                wid = int(widget.winfo_id())
            except (tk.TclError, TypeError, ValueError):
                wid = id(widget)
            if wid not in self._wheel_bound_ids:
                self._wheel_bound_ids.add(wid)
                try:
                    widget.bind("<MouseWheel>", _wheel)
                except tk.TclError:
                    pass
            try:
                for child in widget.winfo_children():
                    _bind_wheel(child)
            except tk.TclError:
                pass

        self._compare_bind_wheel = _bind_wheel
        for widget in (canvas, workspace):
            _bind_wheel(widget)
        try:
            canvas.bind("<Enter>", lambda _e: canvas.focus_set(), add="+")
        except tk.TclError:
            pass

    def _apply_compare_panel_arrangement(self, *, stacked: bool, gap: int) -> None:
        """Wide: 3 columns. Narrow: A|B on top, difference charts below."""
        left = getattr(self, "_compare_left_col", None)
        center = getattr(self, "_compare_center_col", None)
        right = getattr(self, "_compare_right_col", None)
        if left is None or center is None or right is None:
            return

        def _place(
            panel: Any,
            *,
            row: int,
            column: int,
            columnspan: int = 1,
            padx: tuple[int, int] = (0, 0),
            pady: tuple[int, int] = (0, 0),
        ) -> None:
            try:
                panel.grid(
                    row=row,
                    column=column,
                    columnspan=columnspan,
                    sticky="nsew",
                    padx=padx,
                    pady=pady,
                )
            except tk.TclError:
                pass

        if stacked:
            _place(left, row=0, column=0, padx=(0, gap), pady=(0, gap))
            _place(right, row=0, column=1, pady=(0, gap))
            _place(center, row=1, column=0, columnspan=2)
            return
        _place(left, row=0, column=0, padx=(0, gap))
        _place(center, row=0, column=1, padx=(0, gap))
        _place(right, row=0, column=2)

    def _apply_compare_stretch_weights(self) -> None:
        """Center column expands with the window; stacked A/B columns share width."""
        workspace = getattr(self, "_compare_workspace", None)
        if workspace is None or not hasattr(workspace, "columnconfigure"):
            return
        stacked = bool(getattr(self, "_compare_stacked", False))
        pair_floor = COMPARE_VIDEO_MIN_H + COMPARE_SKELETON_MIN_H
        charts_floor = 3 * COMPARE_CHART_MIN_H
        try:
            if stacked:
                workspace.columnconfigure(
                    0, weight=1, minsize=COMPARE_COL_SIDE_MIN_W, uniform="compare_ab"
                )
                workspace.columnconfigure(
                    1, weight=1, minsize=COMPARE_COL_SIDE_MIN_W, uniform="compare_ab"
                )
                workspace.columnconfigure(2, weight=0, minsize=0, uniform="")
                workspace.rowconfigure(0, weight=2, minsize=pair_floor)
                workspace.rowconfigure(1, weight=3, minsize=charts_floor)
                workspace.rowconfigure(2, weight=0, minsize=0)
                workspace.rowconfigure(3, weight=0, minsize=0)
            else:
                workspace.columnconfigure(
                    0, weight=COMPARE_COL_SIDE_WEIGHT, minsize=COMPARE_COL_SIDE_MIN_W, uniform=""
                )
                workspace.columnconfigure(
                    1,
                    weight=COMPARE_COL_CENTER_WEIGHT,
                    minsize=COMPARE_COL_CENTER_MIN_W,
                    uniform="",
                )
                workspace.columnconfigure(
                    2, weight=COMPARE_COL_SIDE_WEIGHT, minsize=COMPARE_COL_SIDE_MIN_W, uniform=""
                )
                workspace.rowconfigure(0, weight=1, minsize=max(pair_floor, charts_floor))
                workspace.rowconfigure(1, weight=0, minsize=0)
                workspace.rowconfigure(2, weight=0, minsize=0)
                workspace.rowconfigure(3, weight=0, minsize=0)
        except tk.TclError:
            pass

    def _configure_compare_column_floors(self, usable_height: int | None = None) -> None:
        """Readable video / skeleton / chart floors; extra space is shared by weight."""
        del usable_height
        video_floor = COMPARE_VIDEO_MIN_H
        skeleton_floor = COMPARE_SKELETON_MIN_H
        chart_floor = COMPARE_CHART_MIN_H
        for col in (
            getattr(self, "_compare_left_col", None),
            getattr(self, "_compare_right_col", None),
        ):
            if col is None or not hasattr(col, "rowconfigure"):
                continue
            try:
                col.columnconfigure(0, weight=1)
                col.rowconfigure(0, weight=3, minsize=video_floor)
                col.rowconfigure(1, weight=2, minsize=skeleton_floor)
            except tk.TclError:
                pass
        center = getattr(self, "_compare_center_col", None)
        if center is not None and hasattr(center, "rowconfigure"):
            try:
                center.columnconfigure(0, weight=1)
                for row in range(3):
                    center.rowconfigure(row, weight=1, minsize=chart_floor)
            except tk.TclError:
                pass

    def _sync_compare_workspace_scroll(
        self, view_w: int, view_h: int, *, stacked: bool
    ) -> None:
        """Stretch the workspace when it fits; scroll instead of crushing plots."""
        canvas = getattr(self, "_compare_workspace_canvas", None)
        inner_id = getattr(self, "_compare_workspace_window_id", None)
        vsb = getattr(self, "_compare_workspace_vsb", None)
        if canvas is None or inner_id is None or vsb is None:
            return
        if view_w < 32 or view_h < 32:
            return
        natural = compare_workspace_natural_height(stacked=stacked)
        inner_h = max(natural, view_h)
        needs_scroll = natural > view_h + 2
        geom = (int(view_w), int(inner_h), bool(needs_scroll))
        if geom == getattr(self, "_compare_scroll_geom", None):
            return
        try:
            canvas.coords(inner_id, 0, 0)
            canvas.itemconfigure(inner_id, width=view_w, height=inner_h)
            canvas.configure(scrollregion=(0, 0, view_w, inner_h))
            if needs_scroll:
                if not vsb.grid_info():
                    vsb.grid()
            else:
                if vsb.grid_info():
                    vsb.grid_remove()
                canvas.yview_moveto(0.0)
            self._compare_needs_scroll = needs_scroll
            self._compare_scroll_geom = geom
        except tk.TclError:
            pass

    def _schedule_compare_workspace_reflow(self, _event: object | None = None) -> None:
        """Debounce Configure storms from canvas / matplotlib resizing."""
        root = getattr(self, "gui", None)
        scheduler = getattr(root, "root", None) if root is not None else None
        if scheduler is None:
            scheduler = getattr(self, "root", None)
        after = getattr(scheduler, "after", None)
        cancel = getattr(scheduler, "after_cancel", None)
        job = getattr(self, "_compare_reflow_job", None)
        if job is not None and callable(cancel):
            try:
                cancel(job)
            except (tk.TclError, ValueError):
                pass
        if callable(after):
            try:
                self._compare_reflow_job = after(24, self._reflow_side_by_side_workspace)
                return
            except tk.TclError:
                pass
        self._reflow_side_by_side_workspace()

    def _reflow_side_by_side_workspace(self, _event: object | None = None) -> None:
        """Keep every comparison visualization visible: side-by-side, or stacked."""
        self._compare_reflow_job = None
        if getattr(self, "_compare_reflow_active", False):
            return
        workspace = getattr(self, "_compare_workspace", None)
        if workspace is None:
            return
        viewport = getattr(self, "_compare_workspace_canvas", None) or workspace
        try:
            width = int(viewport.winfo_width() or 0)
            height = int(viewport.winfo_height() or 0)
            if width < 80:
                width = int(workspace.winfo_width() or self.root.winfo_width() or 0)
            if height < 40:
                height = int(workspace.winfo_height() or 0)
            root_height_fn = getattr(self.root, "winfo_height", None)
            root_height = int(root_height_fn() or 0) if callable(root_height_fn) else 0
        except tk.TclError:
            return
        if width < 80 or height < 40:
            return
        size_key = (width // 8, height // 8, root_height // 8)
        if size_key == getattr(self, "_compare_reflow_size", None):
            return
        self._compare_reflow_active = True
        compact_vertical = 0 < root_height < 620
        self._compare_vertical_compact = compact_vertical
        from stablewalk.ui.tk.layout_scale import layout_scale_from_gui, scaled_breakpoints

        scale = layout_scale_from_gui(getattr(self, "gui", self))
        stack_floor = COMPARE_STACK_BELOW_W
        if abs(scale - 1.0) >= 0.08:
            stack_floor = int(scaled_breakpoints(scale)["compare_stack"])
        stacked = width < stack_floor
        section_gap = PAD_XS if compact_vertical or stacked else PAD_SM
        self._compare_stacked = stacked
        self._apply_compare_panel_arrangement(stacked=stacked, gap=section_gap)
        try:
            self._apply_compare_stretch_weights()
            self._configure_compare_column_floors()
            self._sync_compare_workspace_scroll(width, height, stacked=stacked)
            transport = getattr(self, "_transport_host", None)
            if transport is not None:
                transport.grid_configure(pady=(PAD_XS, section_gap))
            metrics = getattr(self, "_compare_metrics_host", None)
            if metrics is not None:
                metrics.grid_configure(pady=(section_gap, 0))
            row_cfg = getattr(self.root, "rowconfigure", None)
            if callable(row_cfg):
                row_cfg(2, weight=1, minsize=200)
            waiting = getattr(self, "_compare_waiting_label", None)
            if waiting is not None and width > 80:
                waiting.configure(wraplength=max(200, min(640, width - 32)))
            self._compare_reflow_size = size_key
        except tk.TclError:
            pass
        finally:
            self._compare_reflow_active = False

    _DIFF_TABLE_MINWIDTH = {
        "metric": 108,
        "session_a": 84,
        "delta": 88,
        "session_b": 84,
        "note": 100,
    }

    def _sync_compare_tree_columns(self, tree: Any, width: int) -> None:
        """Keep Δ readable; leftover width goes to the note column."""
        if tree is None or int(width) < 80:
            return
        usable = max(80, int(width) - 22)
        mins = self._DIFF_TABLE_MINWIDTH
        floor = sum(mins.values())
        scale = 1.0 if usable >= floor else max(0.72, usable / float(floor))
        leftover = max(0, usable - int(round(floor * scale)))
        widths = {
            key: max(48, int(round(value * scale)))
            for key, value in mins.items()
        }
        widths["note"] = widths["note"] + leftover
        try:
            tree.column("metric", width=widths["metric"], minwidth=72, stretch=False)
            tree.column(
                "session_a", width=widths["session_a"], minwidth=64, stretch=False
            )
            tree.column("delta", width=widths["delta"], minwidth=72, stretch=False)
            tree.column(
                "session_b", width=widths["session_b"], minwidth=64, stretch=False
            )
            tree.column("note", width=widths["note"], minwidth=80, stretch=True)
        except tk.TclError:
            pass

    def _bind_compare_tree_columns(self, tree: Any, host: Any) -> None:
        if tree is None or host is None or not hasattr(host, "bind"):
            return

        def _on_resize(event: object | None = None) -> None:
            try:
                width = int(getattr(event, "width", 0) or host.winfo_width() or 0)
            except tk.TclError:
                return
            self._sync_compare_tree_columns(tree, width)

        try:
            host.bind("<Configure>", _on_resize, add="+")
        except tk.TclError:
            return

    def _build_differences_table(self, parent: ttk.Frame) -> None:
        """Primary 5-metric table: Session A | Δ (B−A) | Session B."""
        cols = ("metric", "session_a", "delta", "session_b", "note")
        tree = ttk.Treeview(
            parent,
            columns=cols,
            show="headings",
            height=6,
            selectmode="browse",
        )
        tree.heading("metric", text="Metric")
        tree.heading("session_a", text="Session A")
        tree.heading("delta", text="Δ (B − A)")
        tree.heading("session_b", text="Session B")
        tree.heading("note", text="Note")
        tree.column("metric", width=120, anchor="w", minwidth=72, stretch=False)
        tree.column("session_a", width=100, anchor="center", minwidth=64, stretch=False)
        tree.column("delta", width=100, anchor="center", minwidth=72, stretch=False)
        tree.column("session_b", width=100, anchor="center", minwidth=64, stretch=False)
        tree.column("note", width=280, anchor="w", minwidth=100, stretch=True)
        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.diff_table = tree
        self._bind_compare_tree_columns(tree, parent)

    def _build_extra_metrics_table(self, parent: tk.Misc) -> None:
        cols = ("metric", "session_a", "delta", "session_b", "note")
        wrap = tk.Frame(parent, bg=PANEL, highlightthickness=0)
        wrap.pack(fill=tk.BOTH, expand=True)
        from stablewalk.ui.tk.tab_grid import grid_fill, stretch_columns, stretch_rows

        stretch_columns(wrap)
        stretch_rows(wrap)
        tree = ttk.Treeview(
            wrap,
            columns=cols,
            show="headings",
            height=6,
            selectmode="browse",
        )
        tree.heading("metric", text="Metric")
        tree.heading("session_a", text="Session A")
        tree.heading("delta", text="Δ (B − A)")
        tree.heading("session_b", text="Session B")
        tree.heading("note", text="Note")
        tree.column("metric", width=120, anchor="w", minwidth=72, stretch=False)
        tree.column("session_a", width=100, anchor="center", minwidth=64, stretch=False)
        tree.column("delta", width=100, anchor="center", minwidth=72, stretch=False)
        tree.column("session_b", width=100, anchor="center", minwidth=64, stretch=False)
        tree.column("note", width=280, anchor="w", minwidth=100, stretch=True)
        grid_fill(tree)
        self.extra_diff_table = tree
        self._bind_compare_tree_columns(tree, wrap)

    def _fill_differences_table(self) -> None:
        """Fill primary + extra tables from the cached comparison (no recalc)."""
        cached = getattr(self, "_cached_comparison", None)
        if cached is None:
            cache = ensure_compare_cache(self.gui)
            cached = build_cached_comparison(
                cache,
                loading_keys=list(self._loading_keys),
                right_override=self._session_b_snap(),
            )
            self._cached_comparison = cached

        def _fill(tree: Any, rows: list) -> None:
            if tree is None:
                return
            try:
                for item in tree.get_children():
                    tree.delete(item)
            except tk.TclError:
                return
            validation = cached.validation
            if validation is not None and validation.state == CompareUiState.IDENTICAL:
                tree.insert(
                    "",
                    tk.END,
                    values=("Sessions identical", "—", "0", "—", validation.message),
                )
                return
            if validation is not None and not validation.can_compare:
                tree.insert(
                    "",
                    tk.END,
                    values=(
                        "—",
                        "—" if not validation.left_ready else "Ready",
                        "—",
                        "—" if not validation.right_ready else "Ready",
                        validation.message,
                    ),
                )
                return
            if not rows:
                tree.insert(
                    "",
                    tk.END,
                    values=("—", "—", "—", "—", "No comparable metrics yet."),
                )
                return
            for row in rows:
                note = (row.reason or "").strip()
                if row.available_both and row.directionality == "none":
                    dir_note = "Δ only — directionality not defined"
                    note = f"{note} · {dir_note}" if note else dir_note
                tree.insert(
                    "",
                    tk.END,
                    values=(
                        row.name,
                        row.display_a,
                        row.display_delta,
                        row.display_b,
                        note,
                    ),
                )

        _fill(getattr(self, "diff_table", None), list(cached.primary_rows))
        _fill(getattr(self, "extra_diff_table", None), list(cached.secondary_rows))

    def _open_detailed_comparison(self) -> None:
        """User-opted heavy visuals: dual video, skeletons, COM path, heatmap."""
        if not ENABLE_DETAILED_COMPARE:
            try:
                self.lbl_status.configure(
                    text="Detailed comparison is disabled (ENABLE_DETAILED_COMPARE=False)",
                    fg=MUTED,
                )
                self._show_compare_error(
                    "Detailed comparison is disabled for stability. "
                    "Use the static metric cards and differences table."
                )
            except Exception:
                pass
            return
        if self._detailed_comparison:
            try:
                self.lbl_status.configure(
                    text="Detailed comparison already open", fg=MUTED
                )
            except Exception:
                pass
            return
        log_compare_event(self.gui, "open_detailed_comparison")
        detailed = getattr(self, "_detailed_section", None)
        if detailed is not None:
            try:
                detailed.grid()
            except tk.TclError:
                pass
        self._detailed_comparison = True
        try:
            if getattr(self, "btn_open_detailed", None) is not None:
                self.btn_open_detailed.configure(state=tk.DISABLED)
        except tk.TclError:
            pass
        for w in getattr(self, "_skel_mode_controls", ()):
            try:
                w.grid()
            except tk.TclError:
                pass
        # Build heavy sections once on the UI thread (idle chunks).
        self._schedule_deferred(0, self._deferred_build_detailed_videos)

    def _deferred_build_detailed_videos(self) -> None:
        if not self._is_compare_tab_active():
            return
        try:
            self._ensure_videos_built()
            self._paint_videos_only(force=True)
            self._sync_session_b_cta()
        except Exception:
            logger.exception("Detailed Compare videos failed")
        self._schedule_deferred(16, self._deferred_build_detailed_skeletons)

    def _deferred_build_detailed_skeletons(self) -> None:
        if not self._is_compare_tab_active():
            return
        try:
            self._ensure_skeletons_built()
            self._paint_skeletons(*self._session_pair())
            self._sync_session_b_cta()
        except Exception:
            logger.exception("Detailed Compare skeletons failed")
        self._schedule_deferred(16, self._deferred_build_detailed_graphs)

    def _deferred_build_detailed_graphs(self) -> None:
        if not self._is_compare_tab_active():
            return
        try:
            self._ensure_lite_chart_built()
            self._ensure_detailed_graphs_built()
            self._paths_dirty = self._heat_dirty = self._chart_dirty = True
            self._refresh_detailed_charts(force=True)
            if self._lite_chart_built:
                try:
                    self._redraw_chart_series()
                except Exception:
                    pass
        except Exception:
            logger.exception("Detailed Compare graphs failed")
        try:
            ready, waiting = self._both_sessions_chart_ready()
            if ready:
                self.lbl_status.configure(
                    text="Side-by-side comparison ready — use Play to sync",
                    fg=SUCCESS,
                )
            elif waiting:
                self.lbl_status.configure(text=waiting.split("\n", 1)[0], fg=WARNING)
        except Exception:
            pass
        try:
            self._sync_session_b_cta()
        except Exception:
            pass
        sync = getattr(self, "_sync_compare_scroll", None)
        if callable(sync):
            sync()

    @staticmethod
    def _placeholder_kind_from_title(title: str) -> str:
        lower = (title or "").lower()
        if "heatmap" in lower or "trajectory" in lower:
            return "heatmap"
        if "graph" in lower or "difference" in lower:
            return "graph"
        if "skeleton" in lower:
            return "skeleton"
        if "video" in lower:
            return "video"
        return "graph"

    def _bind_placeholder_wraplength(
        self, label: tk.Misc, host: tk.Misc, *, pad: int = 32
    ) -> None:
        def _sync(event: object | None = None) -> None:
            try:
                width = int(getattr(event, "width", 0) or host.winfo_width() or 0)
                wrap = max(120, width - int(pad))
                label.configure(wraplength=wrap)
            except tk.TclError:
                pass

        try:
            host.bind("<Configure>", _sync, add="+")
        except tk.TclError:
            pass

    def _fill_compare_host_stub(
        self,
        host: tk.Misc | None,
        title: str,
        detail: str,
        *,
        session_b_cta: bool = False,
    ) -> None:
        """Fill a layout host so Compare never shows a blank panel."""
        if host is None:
            return
        if session_b_cta:
            kind = "skeleton" if "skeleton" in title.lower() else "video"
            self._fill_session_b_placeholder(host, kind=kind, message=detail)
            return
        for child in list(host.winfo_children()):
            try:
                child.destroy()
            except tk.TclError:
                pass
        try:
            host.columnconfigure(0, weight=1)
            host.rowconfigure(0, weight=1)
        except tk.TclError:
            pass
        box = tk.Frame(
            host,
            bg=PANEL,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        box.grid(row=0, column=0, sticky="nsew")
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        kind = self._placeholder_kind_from_title(title)
        graphic = tk.Canvas(
            box,
            bg=PANEL,
            highlightthickness=0,
            borderwidth=0,
        )
        graphic.grid(row=0, column=0, sticky="nsew", padx=PAD_SM, pady=(PAD_SM, PAD_XS))

        def _redraw(_event: object | None = None) -> None:
            self._draw_placeholder_graphic(graphic, kind=kind)

        graphic.bind("<Configure>", _redraw, add="+")
        title_lbl = tk.Label(
            box,
            text=title,
            bg=PANEL,
            fg=TEXT,
            font=FONT_UI_SEMIBOLD,
            anchor="center",
        )
        title_lbl.grid(row=1, column=0, sticky="ew", padx=PAD_SM)
        detail_lbl = tk.Label(
            box,
            text=detail,
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="center",
            justify=tk.CENTER,
            wraplength=240,
        )
        detail_lbl.grid(row=2, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_XS, PAD_SM))
        self._bind_placeholder_wraplength(detail_lbl, box)
        try:
            box.after(16, _redraw)
        except tk.TclError:
            pass

    def _build_session_b_cta_buttons(self, parent: tk.Misc) -> ttk.Frame:
        """Analyze / Load actions when Session B is unavailable."""
        row = ttk.Frame(parent)
        analyze = ttk.Button(
            row,
            text="Analyze Session B",
            style="Accent.TButton",
            command=self.analyze_session_b,
        )
        analyze.pack(side=tk.LEFT, padx=(0, PAD_XS))
        self._register_analyze_session_b_button(analyze)
        ttk.Button(
            row,
            text="Load recording…",
            style="Export.TButton",
            command=lambda: self.open_session_picker(slot="right"),
        ).pack(side=tk.LEFT)
        return row

    def _draw_placeholder_graphic(
        self, canvas: tk.Canvas, *, kind: str
    ) -> None:
        """Panel silhouette so empty Compare cells read as placeholders, not data."""
        try:
            canvas.delete("all")
            w = max(int(canvas.winfo_width()), 80)
            h = max(int(canvas.winfo_height()), 64)
        except tk.TclError:
            return
        canvas.configure(bg=PANEL)
        cx, cy = w * 0.5, h * 0.5
        stroke = MUTED
        if kind == "video":
            pad_x, pad_y = w * 0.10, h * 0.12
            canvas.create_rectangle(
                pad_x, pad_y, w - pad_x, h - pad_y,
                outline=stroke, width=2, fill=ELEVATED,
            )
            tri = min(w, h) * 0.14
            canvas.create_polygon(
                cx - tri * 0.45,
                cy - tri,
                cx - tri * 0.45,
                cy + tri,
                cx + tri,
                cy,
                fill=COLOR_SESSION_B,
                outline="",
            )
            return
        if kind == "skeleton":
            s = min(w, h) * 0.28
            head_r = s * 0.26
            canvas.create_oval(
                cx - head_r, cy - s * 1.25 - head_r,
                cx + head_r, cy - s * 1.25 + head_r,
                outline=stroke, width=2,
            )
            canvas.create_line(cx, cy - s * 0.98, cx, cy + s * 0.12, fill=stroke, width=2)
            canvas.create_line(
                cx - s * 0.72, cy - s * 0.48, cx + s * 0.72, cy - s * 0.48, fill=stroke, width=2
            )
            canvas.create_line(cx, cy + s * 0.12, cx - s * 0.52, cy + s * 1.05, fill=stroke, width=2)
            canvas.create_line(cx, cy + s * 0.12, cx + s * 0.52, cy + s * 1.05, fill=stroke, width=2)
            return
        if kind == "heatmap":
            gap = 8
            pad = 10
            tile_w = max(24, (w - pad * 2 - gap) / 2)
            tile_h = max(24, h - pad * 2)
            left = pad
            right = pad + tile_w + gap
            canvas.create_rectangle(
                left, pad, left + tile_w, pad + tile_h,
                outline=stroke, width=1, fill=ELEVATED,
            )
            canvas.create_rectangle(
                right, pad, right + tile_w, pad + tile_h,
                outline=stroke, width=1, fill=ELEVATED,
            )
            canvas.create_text(
                left + tile_w / 2, pad + tile_h / 2,
                text="Trajectory", fill=MUTED, font=FONT_CAPTION,
            )
            canvas.create_text(
                right + tile_w / 2, pad + tile_h / 2,
                text="Heatmap", fill=MUTED, font=FONT_CAPTION,
            )
            return
        # Empty difference-graph axes — no series, no fabricated values.
        pad_l, pad_b, pad_r, pad_t = 28, 22, 14, 14
        x0, y0 = pad_l, h - pad_b
        x1, y1 = w - pad_r, pad_t
        canvas.create_rectangle(0, 0, w, h, fill=ELEVATED, outline="")
        canvas.create_line(x0, y0, x1, y0, fill=stroke, width=1)
        canvas.create_line(x0, y0, x0, y1, fill=stroke, width=1)
        wait_text = SESSION_B_WAITING_TITLE
        if str(getattr(self, "_session_b_error", "") or "").strip():
            wait_text = "Session B analysis failed"
        canvas.create_text(
            (x0 + x1) / 2, (y0 + y1) / 2,
            text=wait_text,
            fill=MUTED,
            font=FONT_CAPTION,
        )

    def _build_session_b_cta_frame(
        self,
        parent: tk.Misc,
        *,
        kind: str = "video",
        message: str | None = None,
        show_workflow: bool = True,
        show_actions: bool = True,
    ) -> tk.Frame:
        """Full-panel Placeholder Video / Placeholder Skeleton for missing Session B."""
        title = self._session_b_placeholder_title()
        default_detail = (
            PLACEHOLDER_VIDEO_DETAIL if kind == "video" else PLACEHOLDER_SKELETON_DETAIL
        )
        wrap = tk.Frame(
            parent,
            bg=PANEL,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(2, weight=1)

        tk.Label(
            wrap,
            text=title,
            bg=PANEL,
            fg=TEXT,
            font=FONT_UI_SEMIBOLD,
            anchor="center",
        ).grid(row=0, column=0, sticky="ew", pady=(PAD_SM, 2))

        reason = (message or COMPARE_SESSION_B_NOT_ANALYZED).strip()
        reason_lbl = tk.Label(
            wrap,
            text=reason,
            bg=PANEL,
            fg=ORANGE if "unavailable" in reason.lower() or "not" in reason.lower() else MUTED,
            font=FONT_UI_XS,
            anchor="center",
            justify=tk.CENTER,
            wraplength=240,
        )
        reason_lbl.grid(row=1, column=0, sticky="ew", padx=PAD_SM)
        self._bind_placeholder_wraplength(reason_lbl, wrap)

        graphic = tk.Canvas(
            wrap,
            bg=PANEL,
            highlightthickness=0,
            borderwidth=0,
        )
        graphic.grid(row=2, column=0, sticky="nsew", padx=PAD_SM, pady=(PAD_XS, PAD_XS))

        def _redraw(_event: object | None = None) -> None:
            self._draw_placeholder_graphic(graphic, kind=kind)

        graphic.bind("<Configure>", _redraw, add="+")
        try:
            wrap.after(16, _redraw)
        except tk.TclError:
            pass

        detail_lbl = tk.Label(
            wrap,
            text=default_detail,
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="center",
            justify=tk.CENTER,
            wraplength=240,
        )
        detail_lbl.grid(row=3, column=0, sticky="ew", padx=PAD_SM)
        self._bind_placeholder_wraplength(detail_lbl, wrap)

        if show_workflow:
            steps = "\n".join(SESSION_B_WORKFLOW_STEPS)
            guide = tk.Label(
                wrap,
                text=f"{SESSION_B_WORKFLOW_TITLE}\n{steps}",
                bg=PANEL,
                fg=TEXT_SECONDARY,
                font=FONT_CAPTION,
                anchor="center",
                justify=tk.LEFT,
                wraplength=240,
            )
            guide.grid(row=4, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_XS, 0))
            self._bind_placeholder_wraplength(guide, wrap)

        if show_actions:
            btns = self._build_session_b_cta_buttons(wrap)
            btns.grid(row=5, column=0, pady=(PAD_SM, PAD_SM))
        return wrap

    def _fill_session_b_placeholder(
        self,
        host: tk.Misc | None,
        *,
        kind: str,
        message: str | None = None,
    ) -> None:
        """Install a complete Session B placeholder into a layout host."""
        if host is None:
            return
        for child in list(host.winfo_children()):
            try:
                child.destroy()
            except tk.TclError:
                pass
        try:
            host.columnconfigure(0, weight=1)
            host.rowconfigure(0, weight=1)
        except tk.TclError:
            pass
        frame = self._build_session_b_cta_frame(
            host,
            kind=kind,
            message=message,
            show_workflow=(kind == "video"),
            show_actions=(kind == "video"),
        )
        frame.grid(row=0, column=0, sticky="nsew")
        if kind == "video":
            self._video_b_placeholder = frame
        else:
            self._skel_b_cta = frame

    def _sync_session_b_workflow(self, *, show: bool) -> None:
        """Show/hide the Session B creation workflow strip above B panels."""
        host = getattr(self, "_session_b_workflow_host", None)
        if host is None:
            return
        for child in list(host.winfo_children()):
            try:
                child.destroy()
            except tk.TclError:
                pass
        if not show:
            try:
                host.grid_remove()
            except tk.TclError:
                pass
            return
        try:
            host.grid()
        except tk.TclError:
            pass
        card = tk.Frame(
            host,
            bg=ELEVATED,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        card.grid(row=0, column=0, sticky="ew")
        card.columnconfigure(0, weight=1)
        tk.Label(
            card,
            text=SESSION_B_WORKFLOW_TITLE,
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_UI_SM,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_XS, 2))
        btns = self._build_session_b_cta_buttons(card)
        btns.grid(row=1, column=0, sticky="w", padx=PAD_SM, pady=(0, PAD_XS))

    def _save_compare_session_b(self, snap: AnalyzedSessionSnapshot | None) -> bool:
        """Store Session B independently of the live Overview / Session A object."""
        from stablewalk.ui.tk.session_cache import isolate_analyzed_snapshot

        cache = ensure_compare_cache(self.gui)
        if snap is None or snap is cache.left():
            log_session_b_debug("session_b_saved", False)
            return False
        if int(getattr(snap, "n_frames", 0) or 0) <= 0:
            log_session_b_debug("session_b_saved", False)
            return False
        key = str(cache.right_key or snap.key or "").strip() or str(snap.key)
        isolated = isolate_analyzed_snapshot(snap, key=key)
        isolated.key = key
        self.compare_session_b = isolated
        try:
            self.gui.compare_session_b = isolated
        except Exception:
            pass
        pinned = snap
        if str(getattr(pinned, "key", "") or "") != key:
            from dataclasses import replace

            pinned = replace(pinned, key=key)
        cache.pin(pinned)
        cache.set_slot("right", key)
        frame_count = getattr(isolated, "n_frames", None)
        log_session_b_debug("session_b_saved", True)
        log_session_b_debug("result_type", type(isolated).__name__)
        log_session_b_debug("frame_count", frame_count)
        log_session_b_debug("fps", getattr(isolated, "fps", None))
        if getattr(self, "_analyzing_session_b_key", None):
            logger.info(
                "[COMPARE] Session B analysis complete | frames=%s | cached=True",
                frame_count,
            )
        return True

    def _clear_compare_session_b(self) -> None:
        self.compare_session_b = None
        try:
            self.gui.compare_session_b = None
        except Exception:
            pass

    def _invalidate_compare_outputs(self, *, clear_session_b: bool = False) -> None:
        """Drop stale Δ / video / graph payloads when Session A or B changes."""
        log_compare_debug(
            "invalidate_outputs",
            clear_session_b=clear_session_b,
            left=ensure_compare_cache(self.gui).left_key,
            right=ensure_compare_cache(self.gui).right_key,
        )
        self._cached_comparison = None
        self._static_refresh_fp = None
        self._last_paint = None
        self._visuals_dirty = True
        self._paths_dirty = True
        self._chart_dirty = True
        self._heat_dirty = True
        self._session_b_error = None
        if clear_session_b:
            self._clear_compare_session_b()

    def _abort_in_flight_compare_load(self, *, reason: str) -> None:
        log_compare_debug("abort_in_flight", reason=reason)
        self._compare_load_gen += 1
        with self._snapshot_build_lock:
            self._compare_pending_keys.clear()
        self._loading_keys.clear()
        self._compare_worker = None
        if self.compare_state == CompareState.LOADING:
            self._set_compare_state(CompareState.IDLE, reason=reason)
        try:
            self._sync_loading_banner()
        except Exception:
            logger.exception("Compare banner sync after abort failed")

    def _iter_analyze_session_b_buttons(self) -> list[tk.Misc]:
        buttons: list[tk.Misc] = []
        btn = getattr(self, "btn_analyze_b", None)
        if btn is not None:
            buttons.append(btn)
        tracked = getattr(self, "_session_b_cta_analyze_buttons", None) or []
        live: list[tk.Misc] = []
        for widget in tracked:
            try:
                if not widget.winfo_exists():
                    continue
            except (tk.TclError, AttributeError):
                continue
            live.append(widget)
            buttons.append(widget)
        self._session_b_cta_analyze_buttons = live
        return buttons

    def _register_analyze_session_b_button(self, button: tk.Misc | None) -> None:
        """Keep every visible Analyze Session B button bound to this callback."""
        if button is None:
            return
        try:
            button.configure(command=self.analyze_session_b)
        except tk.TclError:
            logger.exception("[COMPARE] Failed to bind Analyze Session B command")
            return
        tracked = getattr(self, "_session_b_cta_analyze_buttons", None)
        if tracked is None:
            self._session_b_cta_analyze_buttons = []
            tracked = self._session_b_cta_analyze_buttons
        if button is not getattr(self, "btn_analyze_b", None) and button not in tracked:
            tracked.append(button)

    def _session_b_analysis_running(self) -> bool:
        worker = getattr(self, "_compare_worker", None)
        alive = worker is not None and bool(getattr(worker, "is_alive", lambda: False)())
        return bool(getattr(self, "_analyzing_session_b_key", None) and alive)

    def _set_analyze_session_b_busy(self, busy: bool) -> None:
        """Disable Analyze Session B immediately; restore via reload/CTA sync."""
        for btn in self._iter_analyze_session_b_buttons():
            try:
                btn.configure(state=tk.DISABLED if busy else tk.NORMAL)
            except tk.TclError:
                logger.exception("Could not update Analyze Session B button state")

    def _log_compare_session_paths(
        self, *, key_b: str, path_b: str | None, path_b_exists: bool
    ) -> None:
        """Print and log the exact A/B recordings that will be analyzed."""
        from stablewalk.ui.tk.session_cache import resolve_session_b_video_path

        cache = ensure_compare_cache(self.gui)
        left = cache.left()
        path_a = (
            str(getattr(left, "video_path", None) or getattr(left, "source", "") or "")
            if left is not None
            else ""
        )
        if not path_a and cache.left_key:
            path_a = resolve_session_b_video_path(str(cache.left_key)) or ""
        print(
            f"[Compare] Session A key={cache.left_key!r} path={path_a or '(none)'}",
            flush=True,
        )
        print(
            f"[Compare] Session B key={key_b!r} path={path_b or '(none)'} "
            f"exists={path_b_exists}",
            flush=True,
        )
        logger.info(
            "Compare resolved Session A key=%s path=%s",
            cache.left_key,
            path_a or "(none)",
        )
        logger.info(
            "Compare resolved Session B key=%s path=%s exists=%s",
            key_b,
            path_b or "(none)",
            path_b_exists,
        )

    def _slot_needs_analysis(self, key: str) -> bool:
        cache = ensure_compare_cache(self.gui)
        snap = cache.get(key)
        if snap is None:
            return True
        if key == cache.right_key and snap is cache.left():
            return True
        try:
            ok, _reason = snapshot_is_analyzed(snap)
        except Exception:
            ok = bool(getattr(snap, "ready", False)) and int(
                getattr(snap, "n_frames", 0) or 0
            ) > 0
        return not ok

    def _demo_slot_needs_analysis(self, key: str) -> bool:
        """True when Load demo cannot reuse the cached slot as a full session."""
        from stablewalk.ui.tk.session_cache import (
            demo_snapshot_is_compare_ready,
            hydrate_demo_snapshot,
        )

        cache = ensure_compare_cache(self.gui)
        snap = cache.get(key)
        if snap is None:
            return True
        if key == cache.right_key and snap is cache.left():
            return True
        hydrated = hydrate_demo_snapshot(snap, key)
        if hydrated is not None and hydrated is not snap:
            cache.pin(hydrated)
            snap = hydrated
        other = cache.left() if key != cache.left_key else cache.right()
        if other is snap:
            other = None
        return not demo_snapshot_is_compare_ready(snap, other=other)

    def _apply_demo_comparison_ready(self) -> None:
        """Mark Normal vs Abnormal as fully loaded and refresh Compare visuals."""
        from stablewalk.ui.tk.session_cache import log_compare_step

        cache = ensure_compare_cache(self.gui)
        left = cache.get("normal") or cache.left()
        right = cache.get("abnormal")
        if left is not None:
            log_compare_step("session A loaded")
            log_compare_step("session A analyzed")
        if right is not None:
            log_compare_step("session B loaded")
            log_compare_step("session B analyzed")
            self._save_compare_session_b(right)
        self._progress = 0.0
        timeline = getattr(self, "timeline", None)
        if timeline is not None:
            try:
                timeline.set(0.0)
            except Exception:
                pass
        self._demo_compare_load_pending = False
        self._analyzing_session_b_key = None
        self._set_compare_state(CompareState.READY, reason="demo_ready")
        log_compare_debug(
            "compare_data_ready",
            left_frames=getattr(left, "n_frames", None),
            right_frames=getattr(self._session_b_snap(), "n_frames", None),
        )
        log_compare_step("metrics ready")
        try:
            self._sync_compare_play_enabled()
        except Exception:
            pass
        try:
            self._update_progress_readout()
        except Exception:
            pass
        self.request_refresh()
        try:
            if getattr(self, "video_a", None) is not None:
                self._paint_videos_only(force=True)
            if getattr(self, "ax_skel_a", None) is not None:
                self._paint_skeletons(*self._session_pair())
        except Exception:
            logger.exception("Demo comparison paint failed")
        log_compare_debug("ui_refresh", reason="demo_ready")
        log_compare_step("render ready")
        try:
            self.lbl_status.configure(
                text="Demo comparison: Normal vs Abnormal",
                fg=SUCCESS,
            )
            self._clear_compare_error()
            self._sync_session_b_cta()
        except Exception:
            pass

    def _prepare_demo_analysis_options(self, keys: list[str]) -> None:
        from stablewalk.ui.tk.session_cache import resolve_session_b_video_path

        b_key = "abnormal" if "abnormal" in keys else (keys[-1] if keys else None)
        if b_key:
            self._session_b_resolved_path = resolve_session_b_video_path(b_key)
            log_compare_debug(
                "resolved_session_b",
                selected_name=b_key,
                resolved_path=self._session_b_resolved_path,
                path_exists=bool(
                    self._session_b_resolved_path
                    and Path(self._session_b_resolved_path).is_file()
                ),
            )
        try:
            selected = getattr(self.gui, "_selected_pose_backend", None)
            self._session_b_pose_backend = selected() if callable(selected) else None
        except Exception:
            self._session_b_pose_backend = None
        from stablewalk.ui.analysis_mode import MODE_FULL

        self._apply_session_b_analysis_window(
            key=b_key, inherit_from_a=False, requested_mode=MODE_FULL
        )

    def _apply_session_b_analysis_window(
        self,
        *,
        key: str | None,
        inherit_from_a: bool = True,
        requested_mode: str | None = None,
    ) -> None:
        """Set Session B full/preview explicitly. Never apply a silent demo cap."""
        from stablewalk.ui.analysis_mode import (
            MODE_FULL,
            resolve_compare_analysis_mode,
            resolve_gui_max_frames,
        )
        from stablewalk.ui.tk.session_cache import DEMO_COMPARE_KEYS

        a_mode = requested_mode
        if inherit_from_a and not requested_mode:
            left = None
            try:
                left = ensure_compare_cache(self.gui).left()
            except Exception:
                left = None
            if left is not None:
                a_mode = str(getattr(left, "analysis_mode", "") or "") or None
            if not a_mode:
                a_mode = getattr(self.gui, "_analysis_mode", None)
        mode = resolve_compare_analysis_mode(
            session_a_mode=a_mode if inherit_from_a else None,
            requested_mode=requested_mode or (None if inherit_from_a else MODE_FULL),
        )
        is_demo = str(key or "") in DEMO_COMPARE_KEYS
        max_frames, mode = resolve_gui_max_frames(mode=mode, is_demo=is_demo)
        self._session_b_analysis_mode = mode
        self._session_b_max_frames = max_frames

    def _session_b_snap(self) -> AnalyzedSessionSnapshot | None:
        """Session B object used by Compare rendering — never Session A."""
        dedicated = getattr(self, "compare_session_b", None)
        try:
            cache = ensure_compare_cache(self.gui)
        except Exception:
            if dedicated is not None and int(getattr(dedicated, "n_frames", 0) or 0) > 0:
                return dedicated
            return None
        left = cache.left()
        if dedicated is None:
            dedicated = getattr(self.gui, "compare_session_b", None)
        if dedicated is not None and dedicated is not left:
            key = str(getattr(dedicated, "key", "") or "")
            wanted = str(cache.right_key or "")
            if not wanted or not key or key == wanted:
                if int(getattr(dedicated, "n_frames", 0) or 0) > 0:
                    return dedicated
        right = cache.right()
        if right is not None and right is left:
            return None
        return right

    def _session_pair(
        self,
    ) -> tuple[AnalyzedSessionSnapshot | None, AnalyzedSessionSnapshot | None]:
        cache = ensure_compare_cache(self.gui)
        return cache.left(), self._session_b_snap()

    def _session_b_needs_cta(self, validation: Any | None = None) -> bool:
        if validation is not None and bool(getattr(validation, "right_ready", False)):
            return False
        snap = None
        left = None
        try:
            snap = self._session_b_snap()
            left = ensure_compare_cache(self.gui).left()
        except Exception:
            snap = getattr(self, "compare_session_b", None)
        if (
            snap is not None
            and snap is not left
            and bool(getattr(snap, "ready", False))
            and int(getattr(snap, "n_frames", 0) or 0) > 0
        ):
            return False
        if validation is not None:
            return True
        if snap is None:
            return True
        try:
            ok, _reason = snapshot_is_analyzed(snap)
        except Exception:
            ok = bool(getattr(snap, "ready", False))
        return not ok

    def _session_b_placeholder_title(self) -> str:
        cache = ensure_compare_cache(self.gui)
        if str(getattr(self, "_session_b_error", "") or "").strip():
            return "Session B analysis failed"
        if cache.right_key in getattr(self, "_loading_keys", set()) or getattr(
            self, "_analyzing_session_b_key", None
        ):
            return "Analyzing Session B"
        return SESSION_B_WAITING_TITLE

    def _session_b_display_reason(self, validation: Any | None = None) -> str:
        cache = ensure_compare_cache(self.gui)
        err = str(getattr(self, "_session_b_error", "") or "").strip()
        if err:
            return err
        if cache.right_key in getattr(self, "_loading_keys", set()) or getattr(
            self, "_analyzing_session_b_key", None
        ):
            return "Analyzing Session B..."
        if validation is not None:
            reason = getattr(validation, "right_reason", None) or getattr(
                validation, "message", None
            )
            if reason:
                return str(reason).strip()
        return COMPARE_SESSION_B_NOT_ANALYZED

    def _sync_session_b_cta(self, validation: Any | None = None) -> None:
        """Show or hide Session B placeholders; replace automatically when ready."""
        show = self._session_b_needs_cta(validation)
        message = self._session_b_display_reason(validation) if show else None
        self._set_video_b_cta_visible(show, message=message)
        self._set_skel_b_cta_visible(show, message=message)
        try:
            self._sync_session_b_workflow(show=show)
        except Exception:
            pass

    def _surface_session_b_failure(self, message: str) -> None:
        """Leave Compare with a visible Session B error, never the waiting CTA."""
        text = str(message or "").strip() or "Could not analyze Session B."
        if not text.lower().startswith("session b analysis failed"):
            text = f"Session B analysis failed:\n{text}"
        self._session_b_error = text
        self._analyzing_session_b_key = None
        self._set_analyze_session_b_busy(False)
        if self.compare_state not in (CompareState.FAILED, CompareState.CANCELLED):
            self._set_compare_state(CompareState.FAILED, reason="session_b_failed")
        try:
            self.lbl_status.configure(text=text, fg=DANGER)
        except Exception:
            pass
        self._show_compare_error(text)
        try:
            self._sync_reload_button()
            self._sync_session_b_cta()
            _ready, waiting_message = self._both_sessions_chart_ready()
            self._sync_compare_ready_sections(False, message=waiting_message)
        except Exception:
            logger.exception("Session B failure chrome update failed")

    def _set_video_b_cta_visible(
        self, show: bool, *, message: str | None = None
    ) -> None:
        panel = getattr(self, "video_b", None)
        host = getattr(self, "_video_b_host", None)
        if show:
            # Prefer overlay on built video panel; otherwise fill the host stub.
            if isinstance(panel, dict) and panel.get("frame") is not None:
                lbl = panel.get("label")
                frame = panel["frame"]
                cta = panel.get("cta")
                if cta is not None:
                    try:
                        cta.destroy()
                    except tk.TclError:
                        pass
                    panel["cta"] = None
                cta = self._build_session_b_cta_frame(
                    frame, kind="video", message=message, show_workflow=False
                )
                panel["cta"] = cta
                try:
                    if lbl is not None:
                        lbl.configure(image="", text="")
                        lbl.grid_remove()
                except tk.TclError:
                    pass
                try:
                    cta.grid(row=0, column=0, sticky="nsew")
                    cta.lift()
                except tk.TclError:
                    pass
            elif host is not None:
                self._fill_session_b_placeholder(host, kind="video", message=message)
            return
        # Session B ready — remove placeholders and reveal live video.
        if isinstance(panel, dict):
            cta = panel.get("cta")
            if cta is not None:
                try:
                    cta.grid_remove()
                except tk.TclError:
                    pass
            lbl = panel.get("label")
            if lbl is not None:
                try:
                    lbl.grid(row=0, column=0, sticky="nsew")
                except tk.TclError:
                    pass
        ph = getattr(self, "_video_b_placeholder", None)
        if ph is not None:
            try:
                ph.destroy()
            except tk.TclError:
                pass
            self._video_b_placeholder = None

    def _set_skel_b_cta_visible(
        self, show: bool, *, message: str | None = None
    ) -> None:
        host = getattr(self, "_skel_b_host", None)
        if host is None:
            return
        overlay = getattr(self, "_skel_b_cta", None)
        if show:
            if overlay is not None:
                try:
                    overlay.destroy()
                except tk.TclError:
                    pass
                self._skel_b_cta = None
            overlay = self._build_session_b_cta_frame(
                host,
                kind="skeleton",
                message=message,
                show_workflow=False,
                show_actions=False,
            )
            self._skel_b_cta = overlay
            try:
                overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
                overlay.lift()
            except tk.TclError:
                # Host may only have stub children — fill the host instead.
                self._fill_session_b_placeholder(
                    host, kind="skeleton", message=message
                )
        elif overlay is not None:
            try:
                overlay.place_forget()
                overlay.destroy()
            except tk.TclError:
                pass
            self._skel_b_cta = None

    def _ensure_videos_built(self) -> None:
        if self._videos_built:
            return
        host_a = getattr(self, "_video_a_host", None)
        host_b = getattr(self, "_video_b_host", None)
        if host_a is not None and host_b is not None:
            for host in (host_a, host_b):
                for child in list(host.winfo_children()):
                    try:
                        child.destroy()
                    except tk.TclError:
                        pass
            self.video_a = self._video_panel(host_a, "Video A", wrap=False)
            self.video_b = self._video_panel(host_b, "Video B", wrap=False)
            self._videos_built = True
            try:
                self._sync_session_b_cta()
            except Exception:
                pass
            return
        parent = getattr(self, "_video_section", None)
        if parent is None:
            return
        placeholder = getattr(self, "_videos_placeholder", None)
        if placeholder is not None:
            try:
                placeholder.destroy()
            except tk.TclError:
                pass
            self._videos_placeholder = None
        for child in list(parent.winfo_children()):
            try:
                child.destroy()
            except tk.TclError:
                pass
        self._build_videos_row(parent)
        self._videos_built = True

    def _ensure_lite_chart_built(self) -> None:
        if getattr(self, "_static_compare_only", True):
            return
        if self._lite_chart_built:
            return
        parent = getattr(self, "_lite_chart_section", None)
        if parent is None:
            return
        placeholder = getattr(self, "_lite_chart_placeholder", None)
        if placeholder is not None:
            try:
                placeholder.destroy()
            except tk.TclError:
                pass
            self._lite_chart_placeholder = None
        for child in list(parent.winfo_children()):
            try:
                child.destroy()
            except tk.TclError:
                pass
        self._build_lite_knee_chart(parent)
        self._lite_chart_built = True
        self._chart_dirty = True
        self._active_chart_section = "chart"

    def _ensure_detailed_graphs_built(self) -> None:
        """Build COM path + heatmap (separate center-column panels when present)."""
        if self._graphs_built:
            return
        traj_host = getattr(self, "_traj_host", None)
        heat_host = getattr(self, "_heat_host", None)
        if traj_host is not None and heat_host is not None:
            for host in (traj_host, heat_host):
                placeholder_name = (
                    "_traj_placeholder" if host is traj_host else "_heat_placeholder"
                )
                placeholder = getattr(self, placeholder_name, None)
                if placeholder is not None:
                    try:
                        placeholder.destroy()
                    except tk.TclError:
                        pass
                    setattr(self, placeholder_name, None)
                for child in list(host.winfo_children()):
                    try:
                        child.destroy()
                    except tk.TclError:
                        pass
            self._build_difference_trajectory(traj_host)
            self._build_difference_heatmap(heat_host)
            self._graphs_built = True
            self._paths_dirty = True
            self._heat_dirty = True
            return
        parent = getattr(self, "_detail_graphs_section", None)
        if parent is None:
            return
        placeholder = getattr(self, "_detail_graphs_placeholder", None)
        if placeholder is not None:
            try:
                placeholder.destroy()
            except tk.TclError:
                pass
            self._detail_graphs_placeholder = None
        for child in list(parent.winfo_children()):
            try:
                child.destroy()
            except tk.TclError:
                pass
        self._build_detailed_extra_graphs(parent)
        self._graphs_built = True
        self._paths_dirty = True
        self._heat_dirty = True

    def _build_lite_knee_chart(self, parent: ttk.Frame) -> None:
        """Primary center chart: synchronized joint-angle difference (knee)."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        # The workspace cell already supplies the "Difference Graph" title.
        chart_fr = ttk.Frame(parent, padding=DASHBOARD_VIZ_CARD_PAD)
        chart_fr.grid(row=0, column=0, sticky="nsew")
        chart_fr.columnconfigure(0, weight=1)
        chart_fr.rowconfigure(0, weight=0)
        chart_fr.rowconfigure(1, weight=1)
        from stablewalk.ui.viewers.chart_interactions import (
            attach_chart_interactions,
            build_chart_tools_bar,
        )

        self.fig_chart = Figure(figsize=(7.2, 3.4), dpi=120, facecolor=PANEL)
        try:
            from stablewalk.ui.tk.ui_freeze_debug import log_figure

            log_figure("create", name="fig_chart")
        except Exception:
            pass
        self.ax_chart = self.fig_chart.add_subplot(111)
        self._style_chart_ax()
        tools_host = SimpleNamespace(
            fig_chart=None,
            canvas_chart=None,
            root=getattr(self, "root", None) or parent.winfo_toplevel(),
        )
        self.canvas_chart = FigureCanvasTkAgg(self.fig_chart, master=chart_fr)
        tools_host.fig_chart = self.fig_chart
        tools_host.canvas_chart = self.canvas_chart
        self._compare_chart_tools = build_chart_tools_bar(
            tools_host,
            chart_fr,
            fig_attr="fig_chart",
            canvas_attr="canvas_chart",
            export_name="compare_knee",
        )
        self._compare_chart_tools.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))
        wc = self.canvas_chart.get_tk_widget()
        wc.grid(row=1, column=0, sticky="nsew")
        self._bind_compare_canvas_resize(wc, self.fig_chart, self.canvas_chart)
        attach_chart_interactions(
            self.fig_chart,
            self.canvas_chart,
            on_seek_time=self._seek_from_normalized_x,
        )
        self._chart_playhead = None
        self._compare_axis_mode = _COMPARE_AXIS_VIDEO
        self.canvas_chart.mpl_connect("motion_notify_event", self._on_chart_hover)
        self._hover_annot = self.ax_chart.annotate(
            "",
            xy=(0, 0),
            xytext=(8, 8),
            textcoords="offset points",
            color=TEXT,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc=ELEVATED, ec=BORDER, alpha=0.95),
        )
        self._hover_annot.set_visible(False)

    def _build_detailed_extra_graphs(self, parent: ttk.Frame) -> None:
        """Fallback: trajectory + heatmap sharing one host (static Compare)."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        self._build_difference_trajectory(parent, row=0)
        self._build_difference_heatmap(parent, row=1)

    def _build_difference_trajectory(
        self, parent: ttk.Frame, *, row: int = 0
    ) -> None:
        """COM trajectory difference (Δ = B−A)."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(row, weight=1)
        path_fr = ttk.Frame(parent, padding=DASHBOARD_VIZ_CARD_PAD)
        path_fr.grid(row=row, column=0, sticky="nsew")
        path_fr.columnconfigure(0, weight=1)
        path_fr.rowconfigure(1, weight=1)
        toggles = ttk.Frame(path_fr)
        toggles.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))
        ttk.Checkbutton(
            toggles,
            text="Show A",
            variable=self._show_path_left,
            command=self._redraw_paths,
        ).pack(side=tk.LEFT, padx=(PAD_XS, 0))
        ttk.Checkbutton(
            toggles,
            text="Show B",
            variable=self._show_path_right,
            command=self._redraw_paths,
        ).pack(side=tk.LEFT, padx=(PAD_SM, 0))
        ttk.Checkbutton(
            toggles,
            text="Show Δ",
            variable=self._show_path_delta,
            command=self._redraw_paths,
        ).pack(side=tk.LEFT, padx=(PAD_SM, 0))
        self.fig_path = Figure(figsize=(6.2, 4.0), dpi=120, facecolor=PANEL)
        self.ax_path = self.fig_path.add_subplot(111, projection="3d")
        setup_3d_axes(self.ax_path)
        self.canvas_path = FigureCanvasTkAgg(self.fig_path, master=path_fr)
        wp = self.canvas_path.get_tk_widget()
        wp.grid(row=1, column=0, sticky="nsew")
        self._bind_compare_canvas_resize(wp, self.fig_path, self.canvas_path)

    def _build_difference_heatmap(
        self, parent: ttk.Frame, *, row: int = 0
    ) -> None:
        """Joint-angle difference heatmap (click to seek)."""
        from stablewalk.ui.viewers.chart_interactions import attach_chart_time_seek

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(row, weight=1)
        heat_fr = ttk.Frame(parent, padding=DASHBOARD_VIZ_CARD_PAD)
        heat_fr.grid(row=row, column=0, sticky="nsew")
        heat_fr.columnconfigure(0, weight=1)
        heat_fr.rowconfigure(0, weight=1)
        self.fig_heat = Figure(figsize=(6.2, 3.6), dpi=120, facecolor=PANEL)
        self.ax_heat = self.fig_heat.add_subplot(111)
        self.canvas_heat = FigureCanvasTkAgg(self.fig_heat, master=heat_fr)
        wh = self.canvas_heat.get_tk_widget()
        wh.grid(row=0, column=0, sticky="nsew")
        self._bind_compare_canvas_resize(wh, self.fig_heat, self.canvas_heat)
        self.fig_heat.subplots_adjust(left=0.18, right=0.88, top=0.90, bottom=0.18)
        attach_chart_time_seek(
            self.fig_heat, self.canvas_heat, self._seek_from_heatmap_x
        )

    def _install_compare_scroll(self, parent: ttk.Frame) -> ttk.Frame:
        """Local vertical scroll shell (does not touch the Advanced-tab scroll)."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        outer = ttk.Frame(parent)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        # PANEL — not near-black BG — so first layout never flashes a black window.
        canvas = tk.Canvas(outer, bg=PANEL, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        vsb.grid_remove()
        canvas.configure(yscrollcommand=vsb.set)

        scroll_content = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        scroll_content.columnconfigure(0, weight=1)

        sections = ttk.Frame(scroll_content)
        sections.grid(row=0, column=0, sticky="ew", pady=(0, PAD_SM))
        sections.columnconfigure(0, weight=1)

        self._compare_scroll_canvas = canvas
        self._compare_scroll_inner = scroll_content
        self._compare_scroll_window_id = inner_id
        self._compare_scroll_sections = sections

        def _sync(_event: object | None = None) -> None:
            if self._playing:
                return
            if getattr(canvas, "_compare_scroll_syncing", False):
                return
            setattr(canvas, "_compare_scroll_syncing", True)
            try:
                from stablewalk.ui.tk.scroll_shell import apply_overflow_scroll_geometry

                req_h = max(int(scroll_content.winfo_reqheight() or 0), 1)
                self._compare_needs_scroll = apply_overflow_scroll_geometry(
                    canvas, inner_id, vsb, content_height=req_h
                )
            except tk.TclError:
                pass
            finally:
                setattr(canvas, "_compare_scroll_syncing", False)

        self._sync_compare_scroll = _sync
        scroll_content.bind("<Configure>", _sync, add="+")
        canvas.bind("<Configure>", _sync, add="+")

        def _wheel(event: tk.Event) -> str | None:
            if not getattr(self, "_compare_needs_scroll", False):
                return None
            if getattr(event, "delta", None):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                return "break"
            return None

        if not hasattr(self, "_wheel_bound_ids"):
            self._wheel_bound_ids: set[int] = set()

        def _bind_wheel(widget: tk.Misc) -> None:
            """Bind MouseWheel once per widget — never stack add='+' handlers."""
            try:
                wid = int(widget.winfo_id())
            except (tk.TclError, TypeError, ValueError):
                wid = id(widget)
            if wid not in self._wheel_bound_ids:
                self._wheel_bound_ids.add(wid)
                try:
                    widget.bind("<MouseWheel>", _wheel)
                except tk.TclError:
                    pass
            try:
                for child in widget.winfo_children():
                    _bind_wheel(child)
            except tk.TclError:
                pass

        self._compare_bind_wheel = _bind_wheel
        for widget in (canvas, scroll_content, sections):
            _bind_wheel(widget)
        if not getattr(self, "_compare_enter_focus_bound", False):
            canvas.bind("<Enter>", lambda _e: canvas.focus_set(), add="+")
            self._compare_enter_focus_bound = True
        return sections

    def _section_min_host(self, parent: tk.Misc, *, height: int) -> tk.Frame:
        """Soft-floor host — panels keep a readable minimum and still stretch."""
        from stablewalk.ui.tk.tab_grid import grid_fill, stretch_columns, stretch_rows

        host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
        stretch_columns(parent)
        stretch_rows(parent, minsize=max(48, min(int(height), 240)))
        grid_fill(host)
        stretch_columns(host)
        stretch_rows(host)
        return host

    def _build_metrics_row(self, parent: ttk.Frame) -> None:
        host = self._section_min_host(parent, height=COMPARE_METRICS_MIN_H)
        host.columnconfigure(0, weight=1)
        host.columnconfigure(1, weight=2)
        host.columnconfigure(2, weight=1)
        host.rowconfigure(0, weight=1)

        self.card_left = self._metric_card(host, "Session A")
        self.card_left["frame"].grid(row=0, column=0, sticky="nsew", padx=(0, PAD_XS))
        self.card_summary = self._summary_card(host)
        self.card_summary.grid(row=0, column=1, sticky="nsew", padx=PAD_XS)
        self.card_right = self._metric_card(host, "Session B")
        self.card_right["frame"].grid(row=0, column=2, sticky="nsew", padx=(PAD_XS, 0))

    def _metric_card(self, parent: tk.Misc, title: str) -> dict[str, Any]:
        from stablewalk.ui.tk.kpi_cards import create_kpi_card, create_technical_details_panel

        frame = tk.Frame(parent, bg=ELEVATED, highlightbackground=BORDER, highlightthickness=1)
        hdr = tk.Label(
            frame, text=title, bg=ELEVATED, fg=TEXT, font=FONT_HEADING, anchor="w"
        )
        hdr.pack(fill=tk.X, padx=PAD_SM, pady=(PAD_SM, PAD_XS))
        fields: dict[str, tk.Label] = {}
        kpis: dict[str, Any] = {}
        body = tk.Frame(frame, bg=ELEVATED, highlightthickness=0)
        body.pack(fill=tk.BOTH, expand=True, padx=PAD_SM, pady=(0, PAD_SM))

        # At most five key metrics on the session cards; more live in Additional.
        primary_names = tuple(PRIMARY_COMPARE_METRICS)

        for name in primary_names:
            kpi = create_kpi_card(
                body,
                key=name,
                title=name,
                tooltip=f"{name} — laboratory comparison metric (Δ = B − A in center)",
                compact=False,
                show_bar=name in ("Gait Quality", "Cadence", "Symmetry"),
                fill=False,
                tier="primary",
            )
            fields[name] = kpi.value_lbl
            kpis[name] = kpi

        details = create_technical_details_panel(
            body,
            title="Session details",
            bg=ELEVATED,
            start_expanded=False,
        )
        self_meta = tk.Label(
            details.content,
            text="Select a completed analysis session via Browse…",
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_XS,
            justify=tk.LEFT,
            wraplength=220,
            anchor="w",
        )
        self_meta.pack(fill=tk.X, pady=(0, PAD_XS))
        return {
            "frame": frame,
            "header": hdr,
            "fields": fields,
            "kpis": kpis,
            "meta": self_meta,
        }

    def _summary_card(self, parent: tk.Misc) -> tk.Frame:
        from stablewalk.ui.tk.compare_delta_visuals import install_visual_diff_chrome

        frame = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        chrome = install_visual_diff_chrome(frame, bg=SURFACE)
        self.diff_host = chrome["diff_host"]
        self._diff_spark_canvas = chrome["spark_canvas"]
        self._diff_labels: list[tk.Label] = []
        return frame

    def _build_videos_row(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        host = self._section_min_host(parent, height=COMPARE_VIDEO_MIN_H)
        host.columnconfigure(0, weight=1)
        host.columnconfigure(1, weight=1)
        host.rowconfigure(0, weight=1)

        self.video_a = self._video_panel(host, "Session A — Video")
        self.video_a["frame"].grid(
            row=0, column=0, sticky="nsew", padx=(0, PAD_XS), pady=0
        )
        self.video_b = self._video_panel(host, "Session B — Video")
        self.video_b["frame"].grid(
            row=0, column=1, sticky="nsew", padx=(PAD_XS, 0), pady=0
        )

    def _build_split_skeletons(self, host_a: ttk.Frame, host_b: ttk.Frame) -> None:
        """Skeleton A under Video A, Skeleton B under Video B."""
        for host in (host_a, host_b):
            host.columnconfigure(0, weight=1)
            host.rowconfigure(0, weight=1)

        self.skel_side = None
        self.fig_skel_a = Figure(figsize=(5.2, 4.6), dpi=120, facecolor=PANEL)
        self.ax_skel_a = self.fig_skel_a.add_subplot(111, projection="3d")
        setup_3d_axes(self.ax_skel_a)
        self.canvas_skel_a = FigureCanvasTkAgg(self.fig_skel_a, master=host_a)
        wa = self.canvas_skel_a.get_tk_widget()
        wa.grid(row=0, column=0, sticky="nsew")
        self._bind_compare_canvas_resize(wa, self.fig_skel_a, self.canvas_skel_a)
        self._bind_skeleton_camera_controls(self.canvas_skel_a, self.ax_skel_a)

        self.fig_skel_b = Figure(figsize=(5.2, 4.6), dpi=120, facecolor=PANEL)
        self.ax_skel_b = self.fig_skel_b.add_subplot(111, projection="3d")
        setup_3d_axes(self.ax_skel_b)
        self.canvas_skel_b = FigureCanvasTkAgg(self.fig_skel_b, master=host_b)
        wb = self.canvas_skel_b.get_tk_widget()
        wb.grid(row=0, column=0, sticky="nsew")
        self._bind_compare_canvas_resize(wb, self.fig_skel_b, self.canvas_skel_b)
        self._bind_skeleton_camera_controls(self.canvas_skel_b, self.ax_skel_b)

        # Overlay mode host kept for mode toggle compatibility (not shown by default).
        self.skel_overlay = ttk.Frame(host_a)
        self.fig_skel_ov = Figure(figsize=(8.0, 3.0), dpi=120, facecolor=PANEL)
        self.ax_skel_ov = self.fig_skel_ov.add_subplot(111, projection="3d")
        setup_3d_axes(self.ax_skel_ov)
        self.canvas_skel_ov = FigureCanvasTkAgg(self.fig_skel_ov, master=self.skel_overlay)
        wo = self.canvas_skel_ov.get_tk_widget()
        wo.pack(fill=tk.BOTH, expand=True)
        self._bind_compare_canvas_resize(wo, self.fig_skel_ov, self.canvas_skel_ov)
        self._bind_skeleton_camera_controls(self.canvas_skel_ov, self.ax_skel_ov)

    def _build_skeletons_row(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        host = self._section_min_host(parent, height=COMPARE_SKELETON_MIN_H)
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)

        self.skel_side = ttk.Frame(host)
        self.skel_side.grid(row=0, column=0, sticky="nsew")
        self.skel_side.columnconfigure(0, weight=1)
        self.skel_side.columnconfigure(1, weight=1)
        self.skel_side.rowconfigure(0, weight=1)

        self.fig_skel_a = Figure(figsize=(4.2, 3.0), dpi=120, facecolor=PANEL)
        self.ax_skel_a = self.fig_skel_a.add_subplot(111, projection="3d")
        setup_3d_axes(self.ax_skel_a)
        self.canvas_skel_a = FigureCanvasTkAgg(self.fig_skel_a, master=self.skel_side)
        wa = self.canvas_skel_a.get_tk_widget()
        wa.grid(row=0, column=0, sticky="nsew", padx=(0, PAD_XS))
        self._bind_compare_canvas_resize(wa, self.fig_skel_a, self.canvas_skel_a)
        self._bind_skeleton_camera_controls(self.canvas_skel_a, self.ax_skel_a)

        self.fig_skel_b = Figure(figsize=(4.2, 3.0), dpi=120, facecolor=PANEL)
        self.ax_skel_b = self.fig_skel_b.add_subplot(111, projection="3d")
        setup_3d_axes(self.ax_skel_b)
        self.canvas_skel_b = FigureCanvasTkAgg(self.fig_skel_b, master=self.skel_side)
        wb = self.canvas_skel_b.get_tk_widget()
        wb.grid(row=0, column=1, sticky="nsew", padx=(PAD_XS, 0))
        self._bind_compare_canvas_resize(wb, self.fig_skel_b, self.canvas_skel_b)
        self._bind_skeleton_camera_controls(self.canvas_skel_b, self.ax_skel_b)

        self.skel_overlay = ttk.Frame(host)
        self.fig_skel_ov = Figure(figsize=(8.0, 3.0), dpi=120, facecolor=PANEL)
        self.ax_skel_ov = self.fig_skel_ov.add_subplot(111, projection="3d")
        setup_3d_axes(self.ax_skel_ov)
        self.canvas_skel_ov = FigureCanvasTkAgg(self.fig_skel_ov, master=self.skel_overlay)
        wo = self.canvas_skel_ov.get_tk_widget()
        wo.pack(fill=tk.BOTH, expand=True)
        self._bind_compare_canvas_resize(wo, self.fig_skel_ov, self.canvas_skel_ov)
        self._bind_skeleton_camera_controls(self.canvas_skel_ov, self.ax_skel_ov)

    def _build_bottom_metrics_bar(self, parent: ttk.Frame) -> None:
        """Compact Δ metrics strip under the synchronized workspace."""
        from stablewalk.ui.tk.compare_delta_visuals import (
            build_session_b_metrics_placeholder,
        )
        from stablewalk.ui.tk.kpi_cards import create_kpi_grid_card

        parent.columnconfigure(0, weight=1)
        stack = tk.Frame(parent, bg=PANEL, highlightthickness=0)
        stack.grid(row=0, column=0, sticky="ew")
        stack.columnconfigure(0, weight=1)

        placeholder = build_session_b_metrics_placeholder(stack, bg=PANEL)
        placeholder.grid(row=0, column=0, sticky="ew")
        self._bottom_metrics_placeholder = placeholder

        caption = tk.Label(
            stack,
            text="Δ metrics  ·  Stability · Symmetry · Cadence · ROM   (Δ = B − A)",
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        )
        caption.grid_remove()
        self._bottom_metrics_caption = caption

        grid = tk.Frame(stack, bg=PANEL, highlightthickness=0)
        # Shown only when both sessions are ready (placeholder covers empty state).
        self._bottom_metrics_grid = grid
        names = tuple(DEMO_DELTA_METRICS)
        for col in range(len(names)):
            grid.columnconfigure(col, weight=1, minsize=COMPARE_CARD_MIN_W)

        self._bottom_metric_kpis: dict[str, Any] = {}
        self._bottom_metric_delta_lbls: dict[str, tk.Label] = {}
        self._bottom_metric_pct_lbls: dict[str, tk.Label] = {}
        self._bottom_metric_cells: list[tk.Frame] = []
        for col, name in enumerate(names):
            cell = tk.Frame(grid, bg=ELEVATED, highlightthickness=1, highlightbackground=BORDER)
            cell.grid(
                row=0,
                column=col,
                sticky="nsew",
                padx=(0 if col == 0 else PAD_SM, 0),
            )
            cell.columnconfigure(0, weight=1)
            self._bottom_metric_cells.append(cell)
            kpi = create_kpi_grid_card(
                cell,
                key=name,
                title=DEMO_DELTA_TITLES.get(name, name),
                tooltip=f"{DEMO_DELTA_TITLES.get(name, name)} — Session A vs Session B (Δ = B − A)",
                compact=True,
                show_bar=name in ("Symmetry", "Cadence", "Stability"),
                tier="primary",
            )
            kpi.frame.grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_SM, 0))
            delta = tk.Label(
                cell,
                text="",
                bg=ELEVATED,
                fg=MUTED,
                font=FONT_UI_XS,
                anchor="w",
            )
            delta.grid(row=1, column=0, sticky="ew", padx=PAD_SM, pady=(2, 0))
            pct = tk.Label(
                cell,
                text="",
                bg=ELEVATED,
                fg=MUTED,
                font=FONT_UI_XS,
                anchor="w",
            )
            pct.grid(row=2, column=0, sticky="ew", padx=PAD_SM, pady=(0, PAD_SM))
            self._bottom_metric_kpis[name] = kpi
            self._bottom_metric_delta_lbls[name] = delta
            self._bottom_metric_pct_lbls[name] = pct

        self._bottom_metrics_wrap = 0

        def _flow_bottom_metrics(_event: object | None = None) -> None:
            cells = getattr(self, "_bottom_metric_cells", None) or []
            if not cells:
                return
            try:
                width = int(grid.winfo_width() or 0)
            except tk.TclError:
                return
            n = len(cells)
            if width < 80:
                return
            # Four cards: stay 2×2 until the row is wide enough for comfortable 4-up.
            cols = n if width >= COMPARE_METRICS_WRAP_2 else 2
            cols = max(1, min(cols, n))
            if cols == getattr(self, "_bottom_metrics_wrap", None):
                return
            self._bottom_metrics_wrap = cols
            for c in range(n):
                grid.columnconfigure(
                    c,
                    weight=1 if c < cols else 0,
                    minsize=COMPARE_CARD_MIN_W if c < cols else 0,
                )
            for index, cell in enumerate(cells):
                r, c = divmod(index, cols)
                cell.grid(
                    row=r,
                    column=c,
                    sticky="nsew",
                    padx=(0 if c == 0 else PAD_SM, 0),
                    pady=(0 if r == 0 else PAD_SM, 0),
                )

        grid.bind("<Configure>", _flow_bottom_metrics, add="+")

    def _fill_bottom_metrics_bar(
        self,
        left: CompareMetrics | None,
        right: CompareMetrics | None,
        deltas: dict[str, float | None] | None,
    ) -> None:
        """Refresh bottom Stability / Symmetry / Cadence / ROM strip."""
        import math

        from stablewalk.ui.tk.compare_delta_visuals import (
            format_pct,
            pct_change,
            tone_color,
            tone_glyph,
        )
        from stablewalk.ui.scientific_labels import UNIT_CADENCE
        from stablewalk.ui.tk.kpi_cards import update_kpi_card

        kpis = getattr(self, "_bottom_metric_kpis", None) or {}
        delta_lbls = getattr(self, "_bottom_metric_delta_lbls", None) or {}
        pct_lbls = getattr(self, "_bottom_metric_pct_lbls", None) or {}
        grid = getattr(self, "_bottom_metrics_grid", None)
        placeholder = getattr(self, "_bottom_metrics_placeholder", None)
        if not kpis:
            return

        both_ready = left is not None and right is not None and deltas is not None
        if not both_ready:
            # Replace empty metric cells with a clear requirement card.
            if grid is not None:
                try:
                    grid.grid_remove()
                except tk.TclError:
                    pass
            if placeholder is not None:
                try:
                    placeholder.grid()
                except tk.TclError:
                    pass
            return

        if placeholder is not None:
            try:
                placeholder.grid_remove()
            except tk.TclError:
                pass
        if grid is not None:
            try:
                grid.grid(row=0, column=0, sticky="ew")
            except tk.TclError:
                pass

        attr_map = {
            "Cadence": ("cadence_spm", "{:.0f}", UNIT_CADENCE),
            "Stride Length": ("stride_length_m", "{:.2f}", "m"),
            "Symmetry": ("symmetry_pct", "{:.0f}", "%"),
            "Stability": ("stability_margin_m", "{:.3f}", "m"),
            "Speed": ("walking_speed_m_s", "{:.2f}", "m/s"),
            "Joint ROM": ("knee_rom_deg", "{:.0f}", "°"),
        }

        def _raw(name: str, metrics: CompareMetrics | None) -> float | None:
            spec = attr_map.get(name)
            if metrics is None or spec is None:
                return None
            raw = getattr(metrics, spec[0], None)
            if raw is None:
                return None
            try:
                val = float(raw)
            except (TypeError, ValueError):
                return None
            return val if math.isfinite(val) else None

        def _fmt(name: str, val: float | None) -> str:
            spec = attr_map.get(name)
            if val is None or spec is None:
                return ""
            _attr, fmt, unit = spec
            separator = "" if unit in ("°", "%") else " "
            return f"{fmt.format(val)}{separator}{unit}".strip()

        for name, kpi in kpis.items():
            a_val = _raw(name, left)
            b_val = _raw(name, right)
            a_txt = _fmt(name, a_val)
            b_txt = _fmt(name, b_val)
            if a_val is not None and b_val is not None:
                value = f"{a_txt} → {b_txt}"
                available = True
            else:
                from stablewalk.ui.unavailable_labels import compare_unavailable_label

                value = compare_unavailable_label(
                    name,
                    left_available=a_val is not None,
                    right_available=b_val is not None,
                    left_session_ready=left is not None,
                    right_session_ready=right is not None,
                )
                available = False
            try:
                update_kpi_card(
                    kpi,
                    value=value,
                    available=available,
                    quality="neutral" if available else "unavailable",
                )
            except Exception:
                pass
            dlbl = delta_lbls.get(name)
            plbl = pct_lbls.get(name)
            dval = (deltas or {}).get(name)
            if dlbl is None:
                continue
            if dval is None or a_val is None or b_val is None:
                from stablewalk.ui.unavailable_labels import compare_unavailable_label

                dlbl.configure(
                    text=compare_unavailable_label(
                        name,
                        left_available=a_val is not None,
                        right_available=b_val is not None,
                        left_session_ready=left is not None,
                        right_session_ready=right is not None,
                    ),
                    fg=MUTED,
                )
                if plbl is not None:
                    plbl.configure(text="", fg=MUTED)
                continue
            # Educational directionality: higher Session B = improvement.
            if abs(dval) < 1e-9:
                tone = "neutral"
            else:
                tone = "better" if dval > 0 else "worse"
            fg = tone_color(tone)
            glyph = tone_glyph(tone, dval)
            sign = "+" if dval > 0 else ""
            notable = is_notable_difference(name, dval)
            note = " · Notable" if notable else ""
            delta_text = _fmt(name, dval)
            dlbl.configure(text=f"{glyph} Δ {sign}{delta_text}{note}", fg=fg)
            if plbl is not None:
                pct_txt = format_pct(pct_change(a_val, b_val))
                plbl.configure(
                    text=pct_txt if pct_txt else "",
                    fg=fg if pct_txt else MUTED,
                )
            tip = notable_difference_label(name, dval)
            if tip:
                try:
                    create_tooltip(dlbl, tip)
                except Exception:
                    pass

    def _build_graphs_row(self, parent: ttk.Frame) -> None:
        from stablewalk.ui.viewers.chart_interactions import attach_chart_time_seek

        parent.columnconfigure(0, weight=1)
        host = self._section_min_host(parent, height=COMPARE_GRAPH_MIN_H)
        host.columnconfigure(0, weight=1)
        host.columnconfigure(1, weight=1)
        host.columnconfigure(2, weight=1)
        host.rowconfigure(0, weight=1)

        path_fr = ttk.LabelFrame(
            host,
            text=format_labelframe_title("Difference Trajectory (COM · Δ = B−A)"),
            style="Card.TLabelframe",
            padding=DASHBOARD_VIZ_CARD_PAD,
        )
        path_fr.grid(row=0, column=0, sticky="nsew", padx=(0, PAD_XS))
        path_fr.columnconfigure(0, weight=1)
        path_fr.rowconfigure(1, weight=1)
        toggles = ttk.Frame(path_fr)
        toggles.grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(
            toggles,
            text="Show A",
            variable=self._show_path_left,
            command=self._redraw_paths,
        ).pack(side=tk.LEFT, padx=(PAD_XS, 0))
        ttk.Checkbutton(
            toggles,
            text="Show B",
            variable=self._show_path_right,
            command=self._redraw_paths,
        ).pack(side=tk.LEFT, padx=(PAD_SM, 0))
        ttk.Checkbutton(
            toggles,
            text="Show Δ",
            variable=self._show_path_delta,
            command=self._redraw_paths,
        ).pack(side=tk.LEFT, padx=(PAD_SM, 0))
        self.fig_path = Figure(figsize=(4.5, 2.4), dpi=120, facecolor=PANEL)
        self.ax_path = self.fig_path.add_subplot(111, projection="3d")
        setup_3d_axes(self.ax_path)
        self.canvas_path = FigureCanvasTkAgg(self.fig_path, master=path_fr)
        wp = self.canvas_path.get_tk_widget()
        wp.grid(row=1, column=0, sticky="nsew")
        self._bind_compare_canvas_resize(wp, self.fig_path, self.canvas_path)

        chart_fr = ttk.LabelFrame(
            host,
            text=format_labelframe_title(
                "Difference Graph (knee · Δ = B−A · click to seek)"
            ),
            style="Card.TLabelframe",
            padding=DASHBOARD_VIZ_CARD_PAD,
        )
        chart_fr.grid(row=0, column=1, sticky="nsew", padx=(0, PAD_XS))
        chart_fr.columnconfigure(0, weight=1)
        chart_fr.rowconfigure(0, weight=0)
        chart_fr.rowconfigure(1, weight=1)
        from stablewalk.ui.viewers.chart_interactions import (
            attach_chart_interactions,
            build_chart_tools_bar,
        )

        self.fig_chart = Figure(figsize=(4.5, 2.4), dpi=120, facecolor=PANEL)
        self.ax_chart = self.fig_chart.add_subplot(111)
        self._style_chart_ax()
        tools_host = SimpleNamespace(
            fig_chart=None,
            canvas_chart=None,
            root=getattr(self, "root", None) or parent.winfo_toplevel(),
        )
        self.canvas_chart = FigureCanvasTkAgg(self.fig_chart, master=chart_fr)
        tools_host.fig_chart = self.fig_chart
        tools_host.canvas_chart = self.canvas_chart
        self._compare_chart_tools = build_chart_tools_bar(
            tools_host,
            chart_fr,
            fig_attr="fig_chart",
            canvas_attr="canvas_chart",
            export_name="compare_knee",
        )
        self._compare_chart_tools.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))
        wc = self.canvas_chart.get_tk_widget()
        wc.grid(row=1, column=0, sticky="nsew")
        self._bind_compare_canvas_resize(wc, self.fig_chart, self.canvas_chart)
        attach_chart_interactions(
            self.fig_chart,
            self.canvas_chart,
            on_seek_time=self._seek_from_normalized_x,
        )
        self._chart_playhead = None
        self._compare_axis_mode = _COMPARE_AXIS_VIDEO
        self.canvas_chart.mpl_connect("motion_notify_event", self._on_chart_hover)
        self._hover_annot = self.ax_chart.annotate(
            "",
            xy=(0, 0),
            xytext=(8, 8),
            textcoords="offset points",
            color=TEXT,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc=ELEVATED, ec=BORDER, alpha=0.95),
        )
        self._hover_annot.set_visible(False)

        heat_fr = ttk.LabelFrame(
            host,
            text=format_labelframe_title(
                "Difference Heatmap (|A−B| · click to seek)"
            ),
            style="Card.TLabelframe",
            padding=DASHBOARD_VIZ_CARD_PAD,
        )
        heat_fr.grid(row=0, column=2, sticky="nsew")
        heat_fr.columnconfigure(0, weight=1)
        heat_fr.rowconfigure(0, weight=1)
        self.fig_heat = Figure(figsize=(4.8, 2.4), dpi=120, facecolor=PANEL)
        self.ax_heat = self.fig_heat.add_subplot(111)
        self.canvas_heat = FigureCanvasTkAgg(self.fig_heat, master=heat_fr)
        wh = self.canvas_heat.get_tk_widget()
        wh.grid(row=0, column=0, sticky="nsew")
        self._bind_compare_canvas_resize(wh, self.fig_heat, self.canvas_heat)
        # Leave room for y-tick labels / colorbar — never crush labels.
        self.fig_heat.subplots_adjust(left=0.24, right=0.86, top=0.88, bottom=0.18)
        attach_chart_time_seek(
            self.fig_heat, self.canvas_heat, self._seek_from_heatmap_x
        )

    def _bind_compare_canvas_resize(
        self,
        widget: tk.Misc,
        fig: Figure,
        canvas: FigureCanvasTkAgg,
    ) -> None:
        """Keep matplotlib figures filling their panel as the window resizes."""
        state: dict[str, object] = {"job": None, "last": (0, 0)}

        def _apply() -> None:
            state["job"] = None
            if getattr(canvas, "_stablewalk_fitting", False):
                return
            setattr(canvas, "_stablewalk_fitting", True)
            try:
                host = widget.master
                w = max(int(widget.winfo_width()), 0)
                h = max(int(widget.winfo_height()), 0)
                if (w < 40 or h < 40) and host is not None:
                    w = max(int(host.winfo_width()), 80)
                    h = max(int(host.winfo_height()), 80)
                w = max(w, 80)
                h = max(h, 80)
                last = state.get("last")
                if last == (w, h):
                    return
                state["last"] = (w, h)
                dpi = float(fig.get_dpi() or 100.0)
                fig.set_size_inches(w / dpi, h / dpi, forward=False)
                try:
                    canvas.resize(type("E", (), {"width": int(w), "height": int(h)})())
                except (TypeError, AttributeError, tk.TclError):
                    fig.set_size_inches(w / dpi, h / dpi, forward=True)
                self._safe_draw(canvas)
            except Exception:
                pass
            finally:
                setattr(canvas, "_stablewalk_fitting", False)

        def _on_configure(_event: object | None = None) -> None:
            if self._playing:
                return
            if getattr(canvas, "_stablewalk_fitting", False):
                return
            job = state.get("job")
            if job is not None:
                try:
                    widget.after_cancel(job)  # type: ignore[arg-type]
                except Exception:
                    pass
            try:
                state["job"] = widget.after(16, _apply)
            except Exception:
                _apply()

        # Bind host only — configuring the canvas widget would re-enter here.
        host = widget.master
        if host is not None:
            host.bind("<Configure>", _on_configure, add="+")
        else:
            widget.bind("<Configure>", _on_configure, add="+")

    def _bind_skeleton_camera_controls(self, canvas, ax) -> None:
        """Remember orbit gestures; double-click eases back to the default view."""

        def _remember(_event: object | None = None) -> None:
            remember_skeleton_camera(ax)

        def _reset(event) -> None:
            if not bool(getattr(event, "dblclick", False)):
                return
            smooth_reset_skeleton_camera(
                ax,
                canvas=canvas,
                scheduler=self.gui.root,
            )

        # Persist the orbit while dragging and after release so playback never
        # snaps back to the default Blender/OpenSim framing mid-session.
        canvas.mpl_connect("motion_notify_event", _remember)
        canvas.mpl_connect("button_release_event", _remember)
        canvas.mpl_connect("button_press_event", _reset)

    def _video_panel(
        self, parent: tk.Misc, title: str, *, wrap: bool = True
    ) -> dict[str, Any]:
        if wrap:
            frame = ttk.LabelFrame(
                parent,
                text=format_labelframe_title(title),
                style="Card.TLabelframe",
                padding=DASHBOARD_VIZ_CARD_PAD,
            )
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
            body = frame
        else:
            frame = parent
            body = parent
            try:
                parent.columnconfigure(0, weight=1)
                parent.rowconfigure(0, weight=1)
            except tk.TclError:
                pass
        # tk.Label required: PhotoImage + fg/bg configure in _paint_video.
        label = tk.Label(
            body,
            text="No session",
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_SM,
            anchor="center",
            justify=tk.CENTER,
        )
        label.grid(row=0, column=0, sticky="nsew")
        try:
            body.grid_propagate(False)
        except tk.TclError:
            pass
        panel: dict[str, Any] = {"frame": frame, "label": label, "_last_size": None}

        def _on_resize(event: tk.Event) -> None:
            if self._playing or getattr(self, "_compare_painting", False):
                return
            # Ignore tiny jitter from image swaps (prevents paint ↔ Configure loops).
            size = (int(event.width), int(event.height))
            last = panel.get("_last_size")
            if last is not None:
                try:
                    lw, lh = last  # type: ignore[misc]
                    if abs(int(lw) - size[0]) < 3 and abs(int(lh) - size[1]) < 3:
                        return
                except (TypeError, ValueError):
                    pass
            if size == panel["_last_size"] or min(size) < 20:
                return
            panel["_last_size"] = size
            if self._video_resize_job is not None:
                try:
                    self.gui.root.after_cancel(self._video_resize_job)
                except (tk.TclError, ValueError):
                    pass

            def _refit() -> None:
                self._video_resize_job = None
                if self._playing or getattr(self, "_compare_painting", False):
                    return
                self._last_paint = None
                self._video_box.clear()
                self._paint_frame(force=True)

            self._video_resize_job = self.gui.root.after(120, _refit)

        # Bind the frame, not the label — image configure on the label would
        # otherwise re-enter resize → paint → resize.
        frame.bind("<Configure>", _on_resize, add="+")
        return panel

    def _build_report_section(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        host = self._section_min_host(parent, height=COMPARE_REPORT_MIN_H)
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)
        interp = ttk.LabelFrame(
            host,
            text=format_labelframe_title("Laboratory Comparison Report"),
            style="Card.TLabelframe",
            padding=DASHBOARD_CARD_PAD,
        )
        interp.grid(row=0, column=0, sticky="nsew")
        self.txt_interp = tk.Text(
            interp,
            height=8,
            wrap=tk.WORD,
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_SUMMARY_INTERPRETATION,
            relief=tk.FLAT,
            padx=PAD_SM,
            pady=PAD_SM,
            highlightthickness=0,
        )
        self.txt_interp.pack(fill=tk.BOTH, expand=True)
        self.txt_interp.configure(state=tk.DISABLED)

    def _build_transport(self, parent: ttk.Frame) -> None:
        """Fixed playback strip — always visible under the scroll area."""
        parent.columnconfigure(0, weight=1)
        transport = ttk.Frame(parent, padding=(PAD_SM, PAD_XS))
        transport.grid(row=0, column=0, sticky="ew")

        self.btn_play = ttk.Button(
            transport, text="Play", style="Transport.TButton", width=8, command=self.toggle_play
        )
        self.btn_play.pack(side=tk.LEFT, padx=(0, PAD_XS))
        ttk.Button(
            transport, text="Reset", style="Transport.TButton", width=6, command=lambda: self.seek(0.0)
        ).pack(side=tk.LEFT, padx=(0, PAD_XS))
        ttk.Button(
            transport, text="Prev", style="Transport.TButton", width=5, command=lambda: self.step_frames(-1)
        ).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(
            transport, text="Next", style="Transport.TButton", width=5, command=lambda: self.step_frames(1)
        ).pack(side=tk.LEFT, padx=(0, PAD_XS))

        self.timeline = ttk.Scale(
            transport,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            command=self._on_timeline,
        )
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=PAD_SM)

        self.lbl_progress = tk.Label(
            transport,
            text="Gait cycle 0%",
            bg=PANEL,
            fg=TEXT,
            font=FONT_HEADING,
            width=14,
            anchor="e",
        )
        self.lbl_progress.pack(side=tk.LEFT, padx=(PAD_XS, PAD_SM))

        self.lbl_time = tk.Label(
            transport,
            text="A F—/— · —\nB F—/— · —",
            bg=PANEL,
            fg=MUTED,
            font=FONT_TRANSPORT,
            width=20,
            anchor="e",
            justify=tk.RIGHT,
        )
        self.lbl_time.pack(side=tk.LEFT, padx=(PAD_XS, PAD_SM))

        ttk.Label(transport, text="Speed").pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="1.0×")
        sp = ttk.Combobox(
            transport,
            textvariable=self.speed_var,
            values=("0.25×", "0.5×", "1.0×", "1.5×", "2.0×"),
            width=6,
            state="readonly",
        )
        sp.pack(side=tk.LEFT, padx=(2, PAD_SM))
        sp.bind("<<ComboboxSelected>>", self._on_speed)

        ttk.Label(transport, text="Sample").pack(side=tk.LEFT)
        self.sample_var = tk.StringVar(value="1")
        sm = ttk.Combobox(
            transport,
            textvariable=self.sample_var,
            values=("1", "2", "3", "5"),
            width=4,
            state="readonly",
        )
        sm.pack(side=tk.LEFT, padx=(2, 0))
        sm.bind("<<ComboboxSelected>>", self._on_sample)

    # ── public API ──────────────────────────────────────────────────────

    def enter(self) -> None:
        """Switch into Comparison Mode without restarting analysis."""
        log_compare_event(
            self.gui,
            "tab_clicked",
            via="enter",
            compare_state=self.compare_state.value,
            generation=self._compare_load_gen,
        )
        cache = ensure_compare_cache(self.gui)
        # Defaults: Session A = Normal, Session B = Abnormal (independent keys).
        if not cache.left_key:
            cache.left_key = "normal"
        if not cache.right_key:
            cache.right_key = "abnormal"
        try:
            self.left_key_var.set(cache.left_key)
            self.right_key_var.set(cache.right_key)
        except Exception:
            pass
        from stablewalk.ui.tk.dashboard_notebook import TAB_COMPARE, select_dashboard_tab

        # Tab select → activate_tab → on_activate_compare_tab performs the single
        # cache refresh. Do not pin / schedule a second load (that froze the UI).
        select_dashboard_tab(self.gui, TAB_COMPARE)
        self.gui._comparison_mode_active = True

    def _enter_async_load(self) -> None:
        """Legacy hook — static Compare refresh is owned by on_activate_compare_tab."""
        if not self._is_compare_tab_active():
            return
        if self.compare_state == CompareState.LOADING:
            try:
                self.lbl_status.configure(
                    text=COMPARE_ALREADY_LOADING_MESSAGE, fg=WARNING
                )
            except Exception:
                pass
            self._sync_loading_banner()
            return
        cache = ensure_compare_cache(self.gui)
        cache.left_key = cache.left_key or "normal"
        cache.right_key = cache.right_key or "abnormal"
        # Cache-only — never pin (knee_angle_series on UI thread) or start workers.
        self._refresh_static_compare(force=False)

    def apply_preset(self, left_key: str, right_key: str) -> None:
        """Select a named pair (e.g. Normal vs Abnormal / Fast Walking)."""
        if left_key == "normal" and right_key == "abnormal":
            self.load_demo_comparison()
            return
        cache = ensure_compare_cache(self.gui)
        self._invalidate_compare_outputs(clear_session_b=True)
        cache.set_slot("left", left_key)
        cache.set_slot("right", right_key)
        try:
            self.left_key_var.set(left_key)
            self.right_key_var.set(right_key)
        except Exception:
            pass
        log_compare_debug("loading_sessions", left=left_key, right=right_key, via="preset")
        needed = [k for k in (left_key, right_key) if self._slot_needs_analysis(k)]
        if needed:
            if right_key in needed:
                self._analyzing_session_b_key = right_key
                self._prepare_demo_analysis_options(needed)
            self._request_compare_snapshots(
                needed, allow_retry=True, explicit_analysis=True
            )
            try:
                self.lbl_status.configure(
                    text=f"Loading {', '.join(needed)}…",
                    fg=TEXT,
                )
            except Exception:
                pass
            return
        right = cache.get(right_key)
        if right is not None:
            self._save_compare_session_b(right)
        self.request_refresh()
        log_compare_debug("ui_refresh", reason="preset_cache_ready")

    def reset_comparison(self) -> None:
        """Drop Session B, in-flight work, and stale deltas. Keep Session A."""
        self.cancel_compare_load(reason="reset")
        cache = ensure_compare_cache(self.gui)
        cache.set_slot("right", "")
        try:
            self.right_key_var.set("")
        except (AttributeError, tk.TclError):
            pass
        self._session_b_error = None
        self._clear_compare_session_b()
        self._invalidate_compare_outputs(clear_session_b=True)
        try:
            self.lbl_status.configure(
                text="Comparison reset. Choose a different Session B.",
                fg=MUTED,
            )
        except Exception:
            pass
        self._refresh_static_compare(force=True)
        log_compare_event(self.gui, "reset_comparison")

    def load_demo_comparison(self) -> None:
        """Load Normal (A) vs Abnormal (B), analyzing any session that is not ready."""
        from stablewalk.ui.tk.session_cache import log_compare_step

        log_compare_step("load demo clicked")
        cache = ensure_compare_cache(self.gui)
        log_compare_debug(
            "loading_sessions",
            left="normal",
            right="abnormal",
            via="load_demo_comparison",
        )
        if self.compare_state == CompareState.LOADING or self._loading_keys:
            self._abort_in_flight_compare_load(reason="replace_with_demo_load")
        self._invalidate_compare_outputs(clear_session_b=True)
        for name in ("_pil_frame_cache", "_photo_cache", "_video_box"):
            cached = getattr(self, name, None)
            if isinstance(cached, dict):
                cached.clear()
        assign_demo_comparison_slots(cache)
        try:
            self.left_key_var.set(cache.left_key)
            self.right_key_var.set(cache.right_key)
        except Exception:
            pass
        needed = [
            key for key in ("normal", "abnormal") if self._demo_slot_needs_analysis(key)
        ]
        if needed:
            for key in needed:
                self._failed_snapshot_keys.discard(key)
            self._demo_compare_load_pending = True
            self._analyzing_session_b_key = (
                "abnormal" if "abnormal" in needed else None
            )
            self._prepare_demo_analysis_options(needed)
            log_compare_debug("analysis_start", keys=",".join(needed), via="demo")
            self._request_compare_snapshots(
                needed, allow_retry=True, explicit_analysis=True
            )
            try:
                message = (
                    "Analyzing Session B..."
                    if needed == ["abnormal"]
                    else (
                        "Loading Normal and analyzing Session B..."
                        if "abnormal" in needed
                        else "Loading Normal demo session..."
                    )
                )
                self.lbl_status.configure(text=message, fg=TEXT)
                self._sync_session_b_cta()
            except Exception:
                pass
            return
        self._apply_demo_comparison_ready()

    def apply_user_vs_reference(self) -> None:
        """Use the current analysis as A and a separate reference as B."""
        from stablewalk.ui.tk.session_cache import pin_current_session

        snap = pin_current_session(self.gui, refresh=False, slot="left")
        cache = ensure_compare_cache(self.gui)
        keys = [key for key in cache.keys() if snap is None or key != snap.key]
        ref_key = "normal" if "normal" in keys else None
        if ref_key is None:
            for candidate in keys:
                if candidate not in ("abnormal", "athletic"):
                    ref_key = candidate
                    break
        if ref_key is None and keys:
            ref_key = keys[0]
        if ref_key is None:
            self.lbl_status.configure(
                text="Analyze a Reference (Normal) session first",
                fg=DANGER,
            )
            return
        if snap is not None:
            cache.left_key = snap.key
            cache.right_key = ref_key
            self.left_key_var.set(snap.key)
            self.right_key_var.set(ref_key)
            self.request_refresh()
            self.lbl_status.configure(
                text=f"Current vs Reference: {snap.label} vs {ref_key}",
                fg=SUCCESS,
            )
            return
        self.request_refresh()
        self.lbl_status.configure(
            text="Analyze your video, then click User vs Reference",
            fg=DANGER,
        )

    def _sync_reload_button(self, validation: Any | None = None) -> None:
        btn = getattr(self, "btn_reload_comparison", None)
        export_btn = getattr(self, "btn_export_report", None)
        if validation is None:
            cache = ensure_compare_cache(self.gui)
            validation = validate_compare_slots(
                cache,
                loading_keys=list(self._loading_keys),
                right_override=self._session_b_snap(),
            )
        if btn is not None:
            try:
                btn.configure(
                    state=tk.NORMAL if validation.can_reload else tk.DISABLED
                )
            except tk.TclError:
                pass
        compare_btn = getattr(self, "btn_run_compare", None)
        if compare_btn is not None:
            try:
                compare_btn.configure(
                    state=tk.NORMAL if validation.can_compare else tk.DISABLED
                )
            except tk.TclError:
                pass
        if export_btn is not None:
            try:
                export_btn.configure(
                    state=tk.NORMAL if validation.can_reload else tk.DISABLED
                )
            except tk.TclError:
                pass
        really_busy = self._session_b_analysis_running()
        for analyze in self._iter_analyze_session_b_buttons():
            self._register_analyze_session_b_button(analyze)
            try:
                analyze.configure(
                    command=self.analyze_session_b,
                    state=tk.DISABLED if really_busy else tk.NORMAL,
                    text="Analyze Session B",
                )
            except tk.TclError:
                logger.exception("[COMPARE] Failed to update Analyze Session B button")
        try:
            self._sync_session_b_cta(validation)
        except Exception:
            logger.exception("[COMPARE] Session B CTA sync failed")
        for analyze in self._iter_analyze_session_b_buttons():
            self._register_analyze_session_b_button(analyze)
            try:
                analyze.configure(
                    command=self.analyze_session_b,
                    state=tk.DISABLED if really_busy else tk.NORMAL,
                    text="Analyze Session B",
                )
            except tk.TclError:
                logger.exception("[COMPARE] Failed to rebind Analyze Session B after CTA refresh")

    def open_session_picker(self, *, slot: str = "left") -> None:
        """Rich picker: name, video, date, backend, frames, confidence."""
        cache = ensure_compare_cache(self.gui)
        candidates = list_compare_candidates(cache)
        parent = getattr(self.gui, "root", None)
        dlg = tk.Toplevel(parent) if parent is not None else tk.Toplevel()
        dlg.title(f"Choose Session {'B' if slot == 'right' else 'A'}")
        dlg.transient(parent)
        dlg.grab_set()
        cols = (
            "session",
            "video",
            "date",
            "backend",
            "frames",
            "confidence",
            "status",
        )
        tree = ttk.Treeview(dlg, columns=cols, show="headings", height=12)
        headings = {
            "session": "Session name",
            "video": "Video",
            "date": "Analysis date",
            "backend": "Backend",
            "frames": "Valid frames",
            "confidence": "Confidence",
            "status": "Status",
        }
        for col in cols:
            tree.heading(col, text=headings[col])
            tree.column(col, width=110 if col != "session" else 160, anchor="w")
        tree.pack(fill=tk.BOTH, expand=True, padx=PAD_SM, pady=PAD_SM)
        key_by_iid: dict[str, str] = {}
        for cand in candidates:
            status = "Analyzed" if cand.is_analyzed else "Not analyzed"
            iid = tree.insert(
                "",
                tk.END,
                values=(
                    cand.session_name,
                    cand.video_name,
                    cand.analysis_date or "—",
                    cand.backend or "—",
                    cand.valid_frame_count,
                    cand.confidence or "—",
                    status,
                ),
            )
            key_by_iid[iid] = cand.key
            if not cand.is_analyzed:
                try:
                    tree.item(iid, tags=("incomplete",))
                except tk.TclError:
                    pass
        try:
            tree.tag_configure("incomplete", foreground=MUTED)
        except tk.TclError:
            pass

        def _choose() -> None:
            sel = tree.selection()
            if not sel:
                return
            key = key_by_iid.get(sel[0], "")
            cand = next((c for c in candidates if c.key == key), None)
            if cand is not None and not cand.is_analyzed:
                from stablewalk.ui.tk.tk_lifecycle import safe_messagebox

                safe_messagebox(
                    "info",
                    "Session not analyzed",
                    (
                        f"'{cand.session_name}' is not a completed analysis.\n\n"
                        f"{cand.incomplete_reason or 'Analyze this session before comparing.'}\n\n"
                        "Use Analyze Session B after selecting a demo/bundle with poses."
                    ),
                    parent=dlg,
                )
                # Still assign the key so the incomplete state is visible.
            if not key:
                return
            other = cache.right_key if slot == "left" else cache.left_key
            if key == other:
                from stablewalk.ui.tk.tk_lifecycle import safe_messagebox

                safe_messagebox(
                    "warning",
                    "Same session",
                    "Session A and Session B must be different analyses.",
                    parent=dlg,
                )
                return
            cache.set_slot("left" if slot == "left" else "right", key)
            if slot == "left":
                self.left_key_var.set(key)
            else:
                self.right_key_var.set(key)
            # Bundle keys: try to load into compare cache asynchronously.
            if key.startswith("bundle:"):
                self._load_bundle_into_slot(key, slot=slot)
            self._cached_comparison = None
            self._static_refresh_fp = None
            if getattr(self, "_detailed_comparison", False) and not getattr(
                self, "_static_compare_only", True
            ):
                self.request_refresh()
            else:
                self._refresh_static_compare(force=True)
            try:
                dlg.destroy()
            except tk.TclError:
                pass

        def _choose_recording() -> None:
            path = filedialog.askopenfilename(
                title="Select Session B recording",
                parent=dlg,
                filetypes=[
                    ("Video", "*.mp4 *.avi *.mov *.mkv *.webm"),
                    ("All files", "*.*"),
                ],
            )
            if not path:
                return
            resolved = Path(path).resolve()
            left = cache.left()
            left_source = str(
                getattr(left, "video_path", None)
                or getattr(left, "source", None)
                or ""
            )
            try:
                same_source = bool(left_source) and Path(left_source).resolve() == resolved
            except (OSError, ValueError):
                same_source = left_source == str(resolved)
            if slot == "right" and same_source:
                from stablewalk.ui.tk.tk_lifecycle import safe_messagebox

                safe_messagebox(
                    "warning",
                    "Same recording",
                    "Session B must be a different recording from Session A.",
                    parent=dlg,
                )
                return
            key = f"video:{resolved}"
            cache.set_slot(slot, key)
            if slot == "left":
                self.left_key_var.set(key)
            else:
                self.right_key_var.set(key)
                self._session_b_error = None
            try:
                dlg.destroy()
            except tk.TclError:
                pass
            if slot == "right":
                # Same Session B pipeline as the toolbar button.
                self.analyze_session_b()
                return
            self._request_compare_snapshots(
                [key],
                allow_retry=True,
                explicit_analysis=True,
            )

        btns = ttk.Frame(dlg)
        btns.pack(fill=tk.X, padx=PAD_SM, pady=(0, PAD_SM))
        if slot == "right":
            ttk.Button(
                btns,
                text="Choose recording…",
                style="Accent.TButton",
                command=_choose_recording,
            ).pack(side=tk.LEFT)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Use selection", style="Accent.TButton", command=_choose).pack(
            side=tk.RIGHT, padx=(0, 6)
        )
        tree.bind("<Double-1>", lambda _e: _choose())

    def _load_bundle_into_slot(self, key: str, *, slot: str) -> None:
        """Load a saved session bundle into the compare cache off the UI thread."""
        path = key.removeprefix("bundle:")
        if not path or not Path(path).exists():
            return
        cache = ensure_compare_cache(self.gui)
        cache.set_slot(slot, key)
        if cache.get(key) is not None:
            self.request_refresh()
            return
        if slot == "right":
            self._session_b_error = None
            self._analyzing_session_b_key = key
        self._request_compare_snapshots(
            [key],
            allow_retry=True,
            explicit_analysis=True,
        )

    def analyze_session_b(self) -> None:
        """User action: analyze / build Session B without freezing the UI."""
        logger.info("[COMPARE] Analyze Session B clicked")
        try:
            self._log_analyze_session_b_click_state()
        except Exception:
            logger.exception("[COMPARE] Analyze Session B preflight log failed")
        log_session_b_debug("analysis_handler_entered", True)
        try:
            self._analyze_session_b_impl()
        except Exception as exc:
            import traceback

            logger.exception("[COMPARE] Analyze Session B failed")
            tb = traceback.format_exc().strip()
            msg = log_compare_exception(self.gui, "analyze_session_b", exc)
            log_session_b_debug("analysis_started", False)
            log_session_b_debug("analysis_finished", False)
            log_session_b_debug("session_b_saved", False)
            self._surface_session_b_failure(f"{exc}\n{tb}\n{msg}")

    def _log_analyze_session_b_click_state(self) -> None:
        """Log Session A/B ownership before any guard or cache check."""
        from stablewalk.ui.tk.session_cache import resolve_session_b_video_path

        cache = ensure_compare_cache(self.gui)
        try:
            display = str(self.right_key_var.get() or "").strip()
        except Exception:
            display = ""
        key = str(cache.right_key or display or "").strip()
        left = cache.left()
        right = None
        try:
            right = self._session_b_snap()
            if right is cache.left():
                right = None
        except Exception:
            right = cache.get(key) if cache.get(key) is not cache.left() else None
        path_a = ""
        if left is not None:
            path_a = str(
                getattr(left, "video_path", None)
                or getattr(left, "source", "")
                or ""
            )
        if not path_a:
            path_a = resolve_session_b_video_path(str(cache.left_key or "")) or ""
        # Session B path comes from the B key / demo registry — never the
        # primary Source box / current video.
        path_b = resolve_session_b_video_path(key) or resolve_session_b_video_path(
            display
        )
        logger.info(
            "[COMPARE] Session A:\nkey=%s\nlabel=%s\npath=%s",
            cache.left_key,
            getattr(left, "label", None) or cache.left_key,
            path_a or "(none)",
        )
        logger.info(
            "[COMPARE] Session B:\nkey=%s\nlabel=%s\npath=%s",
            key or display,
            getattr(right, "label", None) or key or display,
            path_b or "(none)",
        )
        cached = bool(
            right is not None
            and right is not cache.left()
            and int(getattr(right, "n_frames", 0) or 0) > 0
        )
        right_frames = getattr(right, "n_frames", None) if right is not None else None
        try:
            validation = validate_compare_slots(
                cache,
                loading_keys=list(getattr(self, "_loading_keys", set()) or ()),
                right_override=self._session_b_snap(),
            )
        except Exception:
            validation = None
        logger.info(
            "[COMPARE] Analyze Session B clicked | "
            "display=%s | key=%s | path=%s | cached=%s | "
            "right_frames=%s | state=%s | can_compare=%s",
            display or key,
            key,
            path_b,
            cached,
            right_frames,
            getattr(validation, "state", None),
            getattr(validation, "can_compare", False),
        )

    def _analyze_session_b_impl(self) -> None:
        """Resolve Session B and start an independent analysis worker."""
        from stablewalk.ui.tk.session_cache import (
            DEMO_COMPARE_KEYS,
            resolve_session_b_video_path,
        )

        cache = ensure_compare_cache(self.gui)
        try:
            combo_key = str(self.right_key_var.get() or "").strip()
        except Exception:
            combo_key = ""
        if combo_key and combo_key != cache.right_key:
            cache.set_slot("right", combo_key)
        key = cache.right_key
        log_session_b_debug("selected_name", key)
        resolved = resolve_session_b_video_path(key)
        path_b = Path(resolved) if resolved else None
        path_exists = bool(path_b is not None and path_b.is_file())
        logger.info(
            "[COMPARE] Resolved Session B path=%s exists=%s",
            resolved or path_b,
            path_exists,
        )
        log_session_b_debug("resolved_path", resolved)
        log_session_b_debug("path_exists", path_exists)
        self._log_compare_session_paths(
            key_b=key, path_b=resolved, path_b_exists=path_exists
        )
        if not key:
            msg = "Choose a recording for Session B, then analyze it."
            logger.info("[COMPARE] Analyze Session B stopped: missing Session B key")
            try:
                self.lbl_status.configure(text=msg, fg=WARNING)
                self._show_compare_error(msg)
            except Exception:
                logger.exception("Could not show Session B missing-file status")
            try:
                from stablewalk.ui.tk.tk_lifecycle import safe_messagebox

                safe_messagebox(
                    "info",
                    "Session B",
                    msg,
                    parent=getattr(self.gui, "root", None),
                )
            except Exception:
                logger.exception("Could not show Session B missing-file dialog")
            self.open_session_picker(slot="right")
            return
        if key == cache.left_key:
            # Session A may occupy the demo key after Overview analysis.
            # Move A off that key so B can analyze the actual recording.
            freed = False
            if key in DEMO_COMPARE_KEYS and cache.left() is not None:
                new_a_key = f"current:{key}"
                if cache.get(new_a_key) is None or cache.get(new_a_key) is cache.left():
                    try:
                        cache.rekey(key, new_a_key)
                        cache.set_slot("right", key)
                        try:
                            self.left_key_var.set(cache.left_key)
                            self.right_key_var.set(key)
                        except Exception:
                            pass
                        freed = cache.left_key != key
                    except Exception:
                        logger.exception("Could not free demo key %s for Session B", key)
            if not freed:
                logger.error(
                    "[COMPARE] Analyze Session B stopped: Session A and B share key %s",
                    key,
                )
                self._surface_session_b_failure(
                    "Session A and Session B must be different."
                )
                return
            resolved = resolve_session_b_video_path(key)
            path_b = Path(resolved) if resolved else None
            path_exists = bool(path_b is not None and path_b.is_file())
            logger.info(
                "[COMPARE] Resolved Session B path=%s exists=%s",
                resolved or path_b,
                path_exists,
            )
        if key.startswith("bundle:"):
            self._load_bundle_into_slot(key, slot="right")
            return

        left = cache.left()
        path_a = (
            str(getattr(left, "video_path", None) or getattr(left, "source", "") or "")
            if left is not None
            else ""
        )
        if not path_a and cache.left_key:
            path_a = resolve_session_b_video_path(str(cache.left_key)) or ""
        logger.info("[COMPARE] Session A path=%s", path_a or "(none)")
        logger.info("[COMPARE] Session B path=%s", resolved or "(none)")
        if path_a and resolved:
            from stablewalk.ui.tk.session_cache import _recording_identity

            try:
                same_recording = _recording_identity(path_a) == _recording_identity(
                    resolved
                )
            except Exception:
                logger.exception("[COMPARE] Could not compare Session A/B paths")
                same_recording = Path(path_a).name == Path(resolved).name
            if same_recording:
                logger.error(
                    "[COMPARE] Analyze Session B stopped: Session A and B are the same file"
                )
                self._surface_session_b_failure(
                    "Session B is the same recording as Session A.\n"
                    f"Session A: {path_a}\n"
                    f"Session B: {resolved}"
                )
                return

        if not path_exists and key not in DEMO_COMPARE_KEYS:
            shown = resolved or (
                key.removeprefix("video:") if key.startswith("video:") else None
            )
            self._surface_session_b_failure(
                "Session B analysis failed:\n"
                f"Could not resolve video for '{key}'.\n"
                f"resolved_path={shown}\n"
                "path_exists=False"
            )
            log_session_b_debug("analysis_started", False)
            log_session_b_debug("analysis_finished", False)
            log_session_b_debug("session_b_saved", False)
            return

        worker = getattr(self, "_compare_worker", None)
        alive = worker is not None and bool(getattr(worker, "is_alive", lambda: False)())
        already_this_b = key in self._loading_keys or (
            getattr(self, "_analyzing_session_b_key", None) == key
        )
        if already_this_b and alive and key in self._loading_keys:
            logger.info(
                "[COMPARE] Session B analysis already in progress | key=%s",
                key,
            )
            self._analyzing_session_b_key = key
            self._set_analyze_session_b_busy(True)
            try:
                self.lbl_status.configure(
                    text="Analyzing Session B...",
                    fg=WARNING,
                )
            except Exception:
                logger.exception("Could not update Session B analyzing status")
            self._sync_loading_banner("Analyzing Session B...")
            return
        # Stuck LOADING (empty keys / dead worker) or an orphan thread used to
        # return from _request_compare_snapshots without starting analysis —
        # that made Analyze Session B look like a no-op.
        self._abort_in_flight_compare_load(reason="replace_with_session_b")

        # Reuse the controller's single-worker queue. The worker performs all
        # pose loading and analysis; queue draining and visual updates stay on Tk.
        self._session_b_error = None
        self._analyzing_session_b_key = key
        self._session_b_resolved_path = resolved
        self._set_analyze_session_b_busy(True)
        try:
            self.lbl_status.configure(text="Analyzing Session B...", fg=WARNING)
        except Exception:
            logger.exception("Could not update Session B analyzing status")
        self._invalidate_compare_outputs(clear_session_b=True)
        self._clear_compare_session_b()
        log_compare_debug("analysis_start", key=key, via="analyze_session_b")
        try:
            selected = getattr(self.gui, "_selected_pose_backend", None)
            self._session_b_pose_backend = selected() if callable(selected) else None
        except Exception:
            self._session_b_pose_backend = None
        self._apply_session_b_analysis_window(key=key, inherit_from_a=True)
        if not path_exists:
            try:
                self.lbl_status.configure(
                    text=(
                        f"Session B ('{key}') video not on disk yet.\n"
                        f"resolved_path={resolved}\n"
                        "path_exists=False\n"
                        "Analyzing Session B (download if needed)..."
                    ),
                    fg=WARNING,
                )
            except Exception:
                logger.exception("[COMPARE] Could not show Session B missing-path status")
        try:
            self._sync_compare_ready_sections(
                False, message="Analyzing Session B..."
            )
            self._sync_session_b_cta()
        except Exception:
            logger.exception("[COMPARE] Could not update Session B analyzing chrome")
        self._set_analyze_session_b_busy(True)
        logger.info("[COMPARE] Starting independent Session B analysis")
        self._request_compare_snapshots(
            [key],
            allow_retry=True,
            explicit_analysis=True,
        )
        if self.compare_state == CompareState.LOADING:
            self._sync_loading_banner("Analyzing Session B...")
            try:
                self.lbl_status.configure(text="Analyzing Session B...", fg=WARNING)
            except Exception:
                logger.exception("[COMPARE] Could not keep Analyzing Session B status")
            try:
                self._sync_session_b_cta()
                self._sync_compare_ready_sections(
                    False, message="Analyzing Session B..."
                )
            except Exception:
                logger.exception("[COMPARE] Could not refresh Session B chrome after start")
            self._set_analyze_session_b_busy(True)
        else:
            # Cache hit: no worker was needed, but all B visuals still refresh.
            snap_b = cache.get(key)
            try:
                ok, reason = snapshot_is_analyzed(snap_b)
            except Exception:
                ok = snap_b is not None and bool(getattr(snap_b, "ready", False))
                reason = ""
            independent = snap_b is not None and snap_b is not cache.left()
            if ok and independent:
                self._save_compare_session_b(snap_b)
                logger.info(
                    "[COMPARE] Session B analysis complete | frames=%s | cached=True",
                    getattr(snap_b, "n_frames", None),
                )
                self._analyzing_session_b_key = None
                self._set_analyze_session_b_busy(False)
                log_session_b_debug("analysis_finished", True)
                log_session_b_debug("compare_refresh_called", True)
                try:
                    validation = validate_compare_slots(
                        cache, right_override=self._session_b_snap()
                    )
                    if validation.can_compare:
                        self.lbl_status.configure(text="Comparison ready", fg=SUCCESS)
                except Exception:
                    logger.exception("[COMPARE] Could not set Comparison ready status")
                self.request_refresh()
            else:
                self._surface_session_b_failure(
                    reason
                    or f"Could not analyze Session B ('{key}')."
                )

    def pin_current(self, *, slot: str = "left") -> None:
        """Pin the currently analyzed session into Session A or Session B."""
        from stablewalk.ui.tk.session_cache import pin_current_session

        snap = pin_current_session(self.gui)
        if snap is None:
            from stablewalk.ui.tk.tk_lifecycle import safe_messagebox

            safe_messagebox(
                "info",
                "Comparison Mode",
                "Analyze a video first, then pin it as Session A or Session B.",
                parent=getattr(self.gui, "root", None),
            )
            return
        cache = ensure_compare_cache(self.gui)
        other_key = cache.left_key if slot == "right" else cache.right_key
        other = cache.left() if slot == "right" else cache.right()
        if snap.key == other_key or sessions_are_identical(
            snap, other, left_key=snap.key, right_key=other_key
        ):
            from stablewalk.ui.tk.tk_lifecycle import safe_messagebox

            safe_messagebox(
                "warning",
                "Same session",
                "Session A and Session B must be different analyses.",
                parent=getattr(self.gui, "root", None),
            )
            return
        self._invalidate_compare_outputs(clear_session_b=(slot == "right"))
        if slot == "right":
            cache.set_slot("right", snap.key)
            try:
                self.right_key_var.set(snap.key)
            except Exception:
                pass
            self._save_compare_session_b(snap)
        else:
            cache.set_slot("left", snap.key)
            try:
                self.left_key_var.set(snap.key)
            except Exception:
                pass
        log_compare_debug("pin_current", slot=slot, key=snap.key, label=snap.label)
        self.request_refresh()
        log_compare_debug("ui_refresh", reason=f"pin_{slot}")
        side = "B" if slot == "right" else "A"
        self.lbl_status.configure(
            text=f"Pinned {snap.label} → Session {side}",
            fg=SUCCESS,
        )

    def step_frames(self, delta: int) -> None:
        """Step both sessions using shared normalized progress (not equal frame counts)."""
        cache = ensure_compare_cache(self.gui)
        left, right = self._session_pair()
        n = max(
            (left.n_frames if left else 1),
            (right.n_frames if right else 1),
            1,
        )
        step = 1.0 / max(n - 1, 1)
        self.seek(self._progress + delta * step)

    def export_report(self) -> None:
        """Delegate to the application PDF export handler."""
        export_fn = getattr(self.gui, "_export_comparison_report", None)
        if callable(export_fn):
            export_fn()
        else:
            from stablewalk.ui.tk.tk_lifecycle import safe_messagebox

            safe_messagebox(
                "info",
                "Export Comparison Report",
                "Comparison report export is not available in this build.",
                parent=getattr(self.gui, "root", None),
            )

    def _ensure_slot_snapshot(self, key: str) -> None:
        """Static Compare never auto-loads missing sessions."""
        if not key:
            return
        if getattr(self, "_static_compare_only", True):
            cache = ensure_compare_cache(self.gui)
            if cache.get(key) is None:
                msg = (
                    COMPARE_SESSION_B_NOT_ANALYZED
                    if key == cache.right_key
                    else COMPARE_SESSION_A_NOT_ANALYZED
                )
                try:
                    self.lbl_status.configure(text=msg, fg=WARNING)
                    self._show_compare_error(msg)
                except Exception:
                    pass
            return
        self._request_compare_snapshots([key], allow_retry=True)

    def _request_compare_snapshots(
        self,
        keys: list[str],
        *,
        allow_retry: bool = False,
        explicit_analysis: bool = False,
    ) -> None:
        """Ensure each key is cached; start at most one background worker.

        Strict rules:
        - Static Compare (ENABLE_DETAILED_COMPARE=False) never starts workers.
        - LOADING → do nothing (show already-loading status).
        - Cache hits → no worker.
        - At most one daemon worker for the whole Compare controller.
        """
        if getattr(self, "_static_compare_only", True) and not explicit_analysis:
            # Cache-only path — surface missing sessions; never analyze.
            del allow_retry
            cache = ensure_compare_cache(self.gui)
            for key in keys:
                key = str(key or "").strip()
                if not key or cache.get(key) is not None:
                    continue
                if key == cache.right_key:
                    msg = COMPARE_SESSION_B_NOT_ANALYZED
                elif key == cache.left_key:
                    msg = COMPARE_SESSION_A_NOT_ANALYZED
                else:
                    msg = f"Session '{key}' has not been analyzed yet."
                try:
                    self.lbl_status.configure(text=msg, fg=WARNING)
                    self._show_compare_error(msg)
                except Exception:
                    pass
            self._refresh_static_compare()
            return

        requested = {str(k).strip() for k in keys if str(k or "").strip()}
        worker = self._compare_worker
        alive = worker is not None and bool(getattr(worker, "is_alive", lambda: False)())
        loading = set(getattr(self, "_loading_keys", set()) or ())
        still_working = bool(loading) or alive
        already_requested = bool(requested) and requested <= loading and alive

        if self.compare_state == CompareState.LOADING:
            if already_requested:
                log_compare_event(
                    self.gui,
                    "load_blocked",
                    reason="already_loading_same_keys",
                    generation=self._compare_load_gen,
                )
                try:
                    self.lbl_status.configure(
                        text=COMPARE_ALREADY_LOADING_MESSAGE, fg=WARNING
                    )
                except Exception:
                    logger.exception("Could not update already-loading status")
                self._sync_loading_banner()
                return
            if still_working and not explicit_analysis:
                log_compare_event(
                    self.gui,
                    "load_blocked",
                    reason="already_loading",
                    generation=self._compare_load_gen,
                )
                try:
                    self.lbl_status.configure(
                        text=COMPARE_ALREADY_LOADING_MESSAGE, fg=WARNING
                    )
                except Exception:
                    logger.exception("Could not update already-loading status")
                self._sync_loading_banner()
                return
            if still_working and explicit_analysis:
                self._abort_in_flight_compare_load(reason="replace_with_explicit")
            elif not still_working:
                if explicit_analysis:
                    # Stuck LOADING with no worker and no keys — Analyze B used
                    # to return here and never start the Session B pipeline.
                    self._set_compare_state(
                        CompareState.IDLE, reason="unstick_empty_loading"
                    )
                else:
                    log_compare_event(
                        self.gui,
                        "load_blocked",
                        reason="already_loading",
                        generation=self._compare_load_gen,
                    )
                    try:
                        self.lbl_status.configure(
                            text=COMPARE_ALREADY_LOADING_MESSAGE, fg=WARNING
                        )
                    except Exception:
                        logger.exception("Could not update already-loading status")
                    self._sync_loading_banner()
                    return

        worker = self._compare_worker
        if worker is not None and worker.is_alive():
            loading = set(getattr(self, "_loading_keys", set()) or ())
            analyzing_b = getattr(self, "_analyzing_session_b_key", None)
            if explicit_analysis and not (requested and requested <= loading):
                self._abort_in_flight_compare_load(reason="replace_orphan_explicit")
            elif loading or analyzing_b:
                log_compare_event(
                    self.gui,
                    "load_blocked",
                    reason="worker_still_alive",
                    generation=self._compare_load_gen,
                )
                try:
                    self.lbl_status.configure(
                        text=(
                            "Analyzing Session B..."
                            if analyzing_b
                            else COMPARE_ALREADY_LOADING_MESSAGE
                        ),
                        fg=WARNING,
                    )
                except Exception:
                    logger.exception("Could not update worker-alive status")
                self._arm_snapshot_poll()
                self._arm_compare_timeout()
                self._sync_loading_banner()
                return
            else:
                # Timed-out / cancelled worker with no in-flight keys — detach so
                # Analyze Session B can start a fresh independent analysis.
                log_compare_event(
                    self.gui,
                    "orphan_worker_detached",
                    generation=self._compare_load_gen,
                )
                self._compare_worker = None

        cache = ensure_compare_cache(self.gui)
        needed: list[str] = []
        for key in keys:
            key = str(key or "").strip()
            if not key:
                continue
            if allow_retry:
                self._failed_snapshot_keys.discard(key)
            existing = cache.get(key)
            if existing is not None:
                demo_pending = bool(
                    getattr(self, "_demo_compare_load_pending", False)
                )
                if demo_pending:
                    from stablewalk.ui.tk.session_cache import (
                        DEMO_COMPARE_KEYS,
                        demo_snapshot_is_compare_ready,
                        hydrate_demo_snapshot,
                    )

                    if key in DEMO_COMPARE_KEYS:
                        existing = hydrate_demo_snapshot(existing, key) or existing
                        if existing is not cache.get(key):
                            cache.pin(existing)
                        other = (
                            cache.left()
                            if key != cache.left_key
                            else cache.right()
                        )
                        ok = demo_snapshot_is_compare_ready(existing, other=other)
                    else:
                        try:
                            ok, _reason = snapshot_is_analyzed(existing)
                        except Exception:
                            ok = bool(getattr(existing, "ready", False))
                else:
                    try:
                        ok, _reason = snapshot_is_analyzed(existing)
                    except Exception:
                        ok = bool(getattr(existing, "ready", False))
                same_as_a = cache.left() is existing or (
                    cache.left_key == key and cache.left() is not None
                )
                if ok and not same_as_a:
                    log_compare_event(self.gui, "cache_hit", key=key)
                    continue
                if not explicit_analysis:
                    log_compare_event(self.gui, "cache_hit", key=key)
                    continue
                log_compare_event(
                    self.gui, "cache_incomplete_rebuild", key=key
                )
            if key in self._failed_snapshot_keys and not allow_retry:
                log_compare_event(self.gui, "cache_skip_failed", key=key)
                continue
            needed.append(key)

        if not needed:
            validation = validate_compare_slots(cache)
            self._set_compare_state(
                CompareState.READY if validation.can_compare else CompareState.IDLE,
                reason="all_cached" if validation.can_compare else "cached_incomplete",
            )
            self._sync_loading_banner()
            return

        try:
            from stablewalk.ui.tk.tk_lifecycle import assert_single_tk_root

            assert_single_tk_root(getattr(self.gui, "root", None))
        except Exception:
            pass

        # New load generation — stale results from prior workers are ignored.
        self._compare_load_gen += 1
        gen = self._compare_load_gen
        self._loading_keys = set(needed)
        self._set_compare_state(CompareState.LOADING, reason="start_worker")
        log_compare_event(
            self.gui,
            "worker_started",
            keys=",".join(needed),
            generation=gen,
        )
        self._sync_loading_banner()
        self._pause_playback_for_compare_load()
        self._arm_snapshot_poll()
        self._arm_compare_timeout()

        with self._snapshot_build_lock:
            self._compare_pending_keys = list(needed)

        self._compare_worker = threading.Thread(
            target=self._compare_worker_main,
            name="stablewalk-compare-load",
            daemon=True,
            kwargs={"generation": gen},
        )
        try:
            from stablewalk.ui.tk.ui_freeze_debug import log_worker

            log_worker(
                "create",
                name="stablewalk-compare-load",
                keys=",".join(needed),
                generation=gen,
            )
        except Exception:
            pass
        self._compare_worker.start()

    def _compare_worker_main(self, *, generation: int | None = None) -> None:
        """Background only: prepare snapshots. Never touch Tk / Matplotlib."""
        from stablewalk.ui.tk.perf_diagnostics import timed_section

        gen = int(generation if generation is not None else self._compare_load_gen)
        try:
            from stablewalk.ui.tk.ui_freeze_debug import log_worker

            log_worker("start", generation=gen)
        except Exception:
            pass
        try:
            while True:
                # Abort promptly when generation is cancelled / timed out.
                if gen != self._compare_load_gen:
                    break
                with self._snapshot_build_lock:
                    if not self._compare_pending_keys:
                        break
                    key = self._compare_pending_keys.pop(0)
                snap = None
                err: str | None = None
                try:
                    import matplotlib

                    matplotlib.use("Agg", force=False)
                except Exception:
                    pass
                try:
                    from stablewalk.ui.tk.session_cache import (
                        DEMO_COMPARE_KEYS,
                        build_demo_snapshot_with_reason,
                        demo_snapshot_is_compare_ready,
                        ensure_compare_cache,
                        hydrate_demo_snapshot,
                        load_or_analyze_demo_session,
                        resolve_session_b_video_path,
                        run_independent_session_b_analysis,
                        try_build_bundle_snapshot,
                        try_build_video_snapshot,
                    )

                    # Never re-analyze a complete key already pinned on the cache.
                    existing = ensure_compare_cache(self.gui).get(key)
                    left_existing = ensure_compare_cache(self.gui).left()
                    if existing is not None:
                        demo_pipeline_hit = bool(
                            getattr(self, "_demo_compare_load_pending", False)
                        ) and key in DEMO_COMPARE_KEYS
                        if demo_pipeline_hit:
                            existing = hydrate_demo_snapshot(existing, key) or existing
                            try:
                                ok = demo_snapshot_is_compare_ready(existing)
                            except Exception:
                                ok = False
                        else:
                            try:
                                ok, _reason = snapshot_is_analyzed(existing)
                            except Exception:
                                ok = bool(getattr(existing, "ready", False))
                        same_as_a = bool(
                            left_existing is not None
                            and (
                                existing is left_existing
                                or (
                                    existing.sequence is not None
                                    and existing.sequence
                                    is getattr(left_existing, "sequence", None)
                                )
                            )
                        )
                        if ok and not same_as_a:
                            snap = existing
                            log_compare_event(
                                self.gui, "cache_hit", key=key, where="worker"
                            )
                    if snap is None:
                        log_compare_event(
                            self.gui, "cache_miss", key=key, where="worker"
                        )
                        with timed_section(
                            self.gui, "compare_prep", warn_label=f"compare_prep:{key}"
                        ):
                            analyzing_b = (
                                getattr(self, "_analyzing_session_b_key", None) == key
                            )
                            demo_pipeline = bool(
                                getattr(self, "_demo_compare_load_pending", False)
                            ) and key in DEMO_COMPARE_KEYS
                            cache_now = ensure_compare_cache(self.gui)
                            is_session_b = analyzing_b or key == cache_now.right_key
                            use_independent = (
                                analyzing_b or demo_pipeline
                            ) and not key.startswith("bundle:")
                            if use_independent:
                                log_compare_debug(
                                    "analysis_start",
                                    key=key,
                                    via="independent_pipeline",
                                )
                                left_now = cache_now.left()
                                left_id = None
                                if is_session_b:
                                    from stablewalk.ui.tk.session_cache import (
                                        _recording_identity,
                                    )

                                    left_src = ""
                                    if left_now is not None:
                                        left_src = str(
                                            getattr(left_now, "video_path", None)
                                            or getattr(left_now, "source", "")
                                            or ""
                                        )
                                    if not left_src:
                                        left_src = (
                                            resolve_session_b_video_path(
                                                str(cache_now.left_key or "")
                                            )
                                            or ""
                                        )
                                    left_id = (
                                        _recording_identity(left_src)
                                        if left_src
                                        else None
                                    )
                                stored_hint = None
                                if is_session_b:
                                    stored_hint = getattr(
                                        self, "_session_b_resolved_path", None
                                    )
                                if key.startswith("video:") and not stored_hint:
                                    stored_hint = key.removeprefix("video:")
                                resolved = resolve_session_b_video_path(
                                    key, hint=stored_hint
                                )
                                if is_session_b and resolved:
                                    self._session_b_resolved_path = resolved
                                snap, err = (
                                    load_or_analyze_demo_session
                                    if demo_pipeline
                                    else run_independent_session_b_analysis
                                )(
                                    key,
                                    video_path=resolved,
                                    pose_backend=getattr(
                                        self, "_session_b_pose_backend", None
                                    ),
                                    max_frames=getattr(
                                        self, "_session_b_max_frames", None
                                    ),
                                    analysis_mode=getattr(
                                        self, "_session_b_analysis_mode", None
                                    ),
                                    left_video_identity=left_id,
                                )
                                log_compare_debug(
                                    "analysis_end",
                                    key=key,
                                    ok=snap is not None,
                                    frames=getattr(snap, "n_frames", None),
                                )
                            elif key.startswith("bundle:"):
                                snap = try_build_bundle_snapshot(key)
                                if snap is None:
                                    err = (
                                        "Could not load session bundle: "
                                        f"{key.removeprefix('bundle:')}"
                                    )
                            elif key.startswith("video:"):
                                video_path = Path(key.removeprefix("video:"))
                                snap = try_build_video_snapshot(key)
                                if snap is None:
                                    if not video_path.is_file():
                                        err = f"Recording not found: {video_path}"
                                    else:
                                        err = (
                                            "Could not analyze recording "
                                            f"'{video_path.name}'."
                                        )
                            else:
                                snap, err = build_demo_snapshot_with_reason(key)
                        if snap is not None:
                            cache_now = ensure_compare_cache(self.gui)
                            left_now = cache_now.left()
                            analyzing_b_now = (
                                getattr(self, "_analyzing_session_b_key", None) == key
                            )
                            is_right_slot = bool(
                                key == cache_now.right_key or analyzing_b_now
                            )
                            # Only Session B must not reuse Session A's recording.
                            # Rebuilding Session A (Normal) from the same demo
                            # file is expected and must not be rejected.
                            if is_right_slot:
                                if left_now is not None and snap is not left_now:
                                    from stablewalk.ui.tk.session_cache import (
                                        _recording_identity,
                                    )

                                    snap_src = str(
                                        getattr(snap, "video_path", None)
                                        or getattr(snap, "source", "")
                                        or ""
                                    )
                                    left_src = str(
                                        getattr(left_now, "video_path", None)
                                        or getattr(left_now, "source", "")
                                        or ""
                                    )
                                    if (
                                        snap_src
                                        and left_src
                                        and _recording_identity(snap_src)
                                        == _recording_identity(left_src)
                                    ):
                                        err = (
                                            f"Session B ('{key}') resolved to Session A's "
                                            f"recording ({Path(snap_src).name}) and was rejected."
                                        )
                                        logger.error("%s", err)
                                        snap = None
                                elif snap is left_now:
                                    err = (
                                        f"Session B ('{key}') reused Session A's analysis "
                                        "object and was rejected."
                                    )
                                    logger.error("%s", err)
                                    snap = None
                        if snap is None and not err:
                            err = (
                                f"Could not analyze Session B ('{key}'). "
                                "No independent analysis was produced."
                            )
                except Exception as exc:
                    import traceback

                    tb = traceback.format_exc().strip()
                    err = f"{exc}\n{tb}"
                    logger.exception("Compare worker failed for %s", key)
                    try:
                        log_compare_exception(
                            self.gui, f"session_b_worker:{key}", exc
                        )
                    except Exception:
                        pass
                if gen != self._compare_load_gen:
                    break
                self._snapshot_results.put((gen, key, snap, err))
        finally:
            try:
                from stablewalk.ui.tk.ui_freeze_debug import log_worker

                log_worker("complete", generation=gen)
            except Exception:
                pass
            self._snapshot_results.put(
                (gen, _COMPARE_WORKER_SENTINEL, None, None)
            )
            log_compare_event(
                self.gui,
                "worker_finished",
                generation=gen,
                current_generation=self._compare_load_gen,
            )

    def _arm_compare_timeout(self) -> None:
        import time

        if self._compare_timeout_job is not None:
            return
        if self._compare_load_started_monotonic is None:
            self._compare_load_started_monotonic = time.monotonic()
        try:
            self._compare_timeout_job = self.gui.root.after(
                1000, self._check_compare_timeout
            )
        except tk.TclError:
            self._compare_timeout_job = None

    def _compare_load_timeout_s(self) -> float:
        """Cached demo lookup stays short; Session B analysis may run the pipeline."""
        if getattr(self, "_analyzing_session_b_key", None):
            return 300.0
        if getattr(self, "_demo_compare_load_pending", False):
            return 300.0
        if any(
            str(key).startswith(("video:", "bundle:"))
            for key in getattr(self, "_loading_keys", set())
        ):
            return 300.0
        return COMPARE_LOAD_TIMEOUT_S

    def _check_compare_timeout(self) -> None:
        import time

        self._compare_timeout_job = None
        if self.compare_state != CompareState.LOADING and not self._loading_keys:
            self._compare_load_started_monotonic = None
            return
        started = self._compare_load_started_monotonic
        if started is None:
            self._compare_load_started_monotonic = time.monotonic()
            self._arm_compare_timeout()
            return
        elapsed = time.monotonic() - float(started)
        timeout_s = self._compare_load_timeout_s()
        if elapsed >= timeout_s:
            log_compare_event(
                self.gui,
                "timeout",
                elapsed_s=f"{elapsed:.1f}",
                keys=",".join(sorted(self._loading_keys)),
                generation=self._compare_load_gen,
            )
            # Invalidate in-flight worker results; do not join/kill the thread.
            self._compare_load_gen += 1
            with self._snapshot_build_lock:
                self._compare_pending_keys.clear()
            timed_out = list(self._loading_keys)
            self._loading_keys.clear()
            for key in timed_out:
                self._failed_snapshot_keys.add(key)
            analyzing_b = getattr(self, "_analyzing_session_b_key", None)
            if analyzing_b:
                self._session_b_error = (
                    f"Session B analysis timed out after {int(timeout_s)}s "
                    f"({analyzing_b}). {COMPARE_TIMEOUT_MESSAGE}"
                )
            self._compare_load_started_monotonic = None
            self._set_compare_state(CompareState.FAILED, reason="timeout")
            self._sync_loading_banner()
            try:
                self.lbl_status.configure(
                    text=getattr(self, "_session_b_error", None)
                    or COMPARE_TIMEOUT_MESSAGE,
                    fg=DANGER,
                )
            except Exception:
                pass
            self._finish_compare_load(timed_out=True)
            return
        try:
            self._compare_timeout_job = self.gui.root.after(
                1000, self._check_compare_timeout
            )
        except tk.TclError:
            self._compare_timeout_job = None

    def _pause_playback_for_compare_load(self) -> None:
        if self._playback_resume_after_load is not None:
            return
        gui = self.gui
        was_playing = bool(getattr(gui, "playing", False))
        self._playback_resume_after_load = was_playing
        if not was_playing:
            return
        try:
            gui.playing = False
            player = getattr(gui, "skeleton_player", None)
            if player is not None and hasattr(player, "pause"):
                player.pause()
            cancel = getattr(gui, "_cancel_timer", None)
            if callable(cancel):
                cancel()
            sync = getattr(gui, "_sync_play_buttons", None)
            if callable(sync):
                sync()
            # Stop hidden-tab chart refreshes tied to the main transport.
            cancel_redraw = getattr(gui, "_cancel_queued_redraw_callbacks", None)
            if callable(cancel_redraw):
                cancel_redraw()
        except Exception:
            logger.debug("pause playback for Compare load failed", exc_info=True)

    def _restore_playback_after_compare_load(self) -> None:
        was = self._playback_resume_after_load
        self._playback_resume_after_load = None
        if not was:
            return
        if self._is_compare_tab_active():
            # Resume when the user leaves Compare so Overview play does not
            # fight the Compare tab transport.
            self._playback_resume_when_leaving = True
            return
        self._resume_main_playback()

    def _resume_main_playback(self) -> None:
        gui = self.gui
        if getattr(gui, "skeleton_player", None) is None:
            return
        if getattr(gui, "playing", False):
            return
        try:
            toggle = getattr(gui, "_toggle_play", None)
            if callable(toggle):
                toggle()
            else:
                gui.playing = True
                schedule = getattr(gui, "_schedule_tick", None)
                if callable(schedule):
                    schedule()
                sync = getattr(gui, "_sync_play_buttons", None)
                if callable(sync):
                    sync()
        except Exception:
            logger.debug("resume main playback failed", exc_info=True)

    def _finish_compare_load(self, *, timed_out: bool = False) -> None:
        """Main-thread completion: unlock visuals and restore lifecycle state."""
        self._compare_load_started_monotonic = None
        if self._compare_timeout_job is not None:
            try:
                self.gui.root.after_cancel(self._compare_timeout_job)
            except (tk.TclError, ValueError):
                pass
            self._compare_timeout_job = None

        if timed_out:
            if self.compare_state != CompareState.FAILED:
                self._set_compare_state(CompareState.FAILED, reason="timeout_finish")
        elif self._loading_keys:
            self._set_compare_state(CompareState.LOADING, reason="still_loading")
        elif self.compare_state == CompareState.CANCELLED:
            pass
        else:
            cache = ensure_compare_cache(self.gui)
            validation = validate_compare_slots(
                cache, right_override=self._session_b_snap()
            )
            if validation.can_compare:
                self._set_compare_state(CompareState.READY, reason="load_ok")
            elif self._failed_snapshot_keys:
                self._set_compare_state(CompareState.FAILED, reason="partial_fail")
            else:
                self._set_compare_state(CompareState.IDLE, reason="load_empty")

        self._sync_loading_banner()
        self._restore_playback_after_compare_load()
        analyzed_b_key = getattr(self, "_analyzing_session_b_key", None)
        snap_b = None
        if analyzed_b_key and not self._loading_keys:
            cache = ensure_compare_cache(self.gui)
            loaded_b = self._session_b_snap() or cache.get(analyzed_b_key)
            has_data = (
                loaded_b is not None
                and loaded_b is not cache.left()
                and int(getattr(loaded_b, "n_frames", 0) or 0) > 0
                and bool(getattr(loaded_b, "ready", False))
            )
            try:
                ok, reason = snapshot_is_analyzed(loaded_b)
            except Exception:
                ok = has_data
                reason = ""
            try:
                if has_data:
                    if getattr(self, "compare_session_b", None) is None:
                        self._save_compare_session_b(loaded_b)
                    snap_b = self._session_b_snap() or loaded_b
                    logger.info(
                        "[COMPARE] Session B analysis complete | frames=%s",
                        getattr(snap_b, "n_frames", None),
                    )
                    self._session_b_error = None
                    label = getattr(snap_b, "label", None) or analyzed_b_key
                    self._clear_compare_error()
                    status_text = f"Session B ready: {label}"
                    cache_now = ensure_compare_cache(self.gui)
                    validation = validate_compare_slots(
                        cache_now, right_override=snap_b
                    )
                    logger.info("[COMPARE] right_cached=%s", True)
                    logger.info(
                        "[COMPARE] can_compare=%s",
                        bool(getattr(validation, "can_compare", False)),
                    )
                    if ok and validation.can_compare:
                        self._set_compare_state(
                            CompareState.READY, reason="session_b_ready"
                        )
                        status_text = "Comparison ready"
                    self.lbl_status.configure(text=status_text, fg=SUCCESS)
                    try:
                        self._sync_session_b_cta()
                    except Exception:
                        logger.exception("[COMPARE] Session B CTA hide after success failed")
                elif self.compare_state != CompareState.CANCELLED:
                    message = str(
                        getattr(self, "_session_b_error", "") or ""
                    ).strip()
                    if not message:
                        if timed_out:
                            message = (
                                f"Session B analysis timed out ('{analyzed_b_key}'). "
                                f"{COMPARE_TIMEOUT_MESSAGE}"
                            )
                        else:
                            message = (
                                reason
                                or f"Could not analyze Session B ('{analyzed_b_key}')."
                            )
                    self._surface_session_b_failure(message)
            except Exception:
                logger.exception("Session B completion update failed")
                if self.compare_state != CompareState.CANCELLED:
                    self._surface_session_b_failure(
                        str(getattr(self, "_session_b_error", "") or "").strip()
                        or f"Could not analyze Session B ('{analyzed_b_key}')."
                    )
            self._analyzing_session_b_key = None
            self._set_analyze_session_b_busy(False)
        demo_load = bool(getattr(self, "_demo_compare_load_pending", False))
        demo_applied = False
        if demo_load and not self._loading_keys:
            cache = ensure_compare_cache(self.gui)
            demo_needed = [
                key
                for key in ("normal", "abnormal")
                if self._demo_slot_needs_analysis(key)
            ]
            demo_failed = any(
                key in self._failed_snapshot_keys for key in demo_needed
            )
            if demo_needed and not demo_failed and not timed_out:
                # A prior unrelated Compare worker may have been finishing when
                # the demo button was clicked. Queue the demos now, without
                # ever running a second worker concurrently.
                self._analyzing_session_b_key = (
                    "abnormal" if "abnormal" in demo_needed else None
                )
                self._prepare_demo_analysis_options(demo_needed)
                log_compare_debug(
                    "analysis_start",
                    keys=",".join(demo_needed),
                    via="demo_finish_retry",
                )
                self._request_compare_snapshots(
                    demo_needed,
                    allow_retry=True,
                    explicit_analysis=True,
                )
                if self.compare_state == CompareState.LOADING:
                    return
            if not demo_needed:
                self._apply_demo_comparison_ready()
                demo_applied = True
            else:
                self._demo_compare_load_pending = False
        if demo_applied:
            log_compare_event(
                self.gui,
                "ui_update_finished",
                timed_out=timed_out,
                compare_state=self.compare_state.value,
                generation=self._compare_load_gen,
            )
            return
        log_compare_event(
            self.gui,
            "ui_update_finished",
            timed_out=timed_out,
            compare_state=self.compare_state.value,
            generation=self._compare_load_gen,
        )
        if not timed_out and (
            self.compare_state == CompareState.READY
            or snap_b is not None
            or (
                self._session_b_snap() is not None
                and int(getattr(self._session_b_snap(), "n_frames", 0) or 0) > 0
            )
        ):
            log_compare_event(
                self.gui,
                "ui_update_started",
                generation=self._compare_load_gen,
            )
            log_session_b_debug("compare_refresh_called", True)
            log_compare_debug(
                "ui_refresh",
                reason="finish_compare_load",
                compare_state=self.compare_state.value,
            )
            try:
                self._refresh_chrome_only(fetch_missing=False)
            except Exception:
                logger.exception("Compare chrome after load failed")
            self._visuals_dirty = True
            try:
                self._queue_detailed_visual_refresh()
            except Exception:
                logger.exception("Compare visual refresh after load failed")
            try:
                self.request_refresh()
            except Exception:
                logger.exception("Compare request_refresh after Session B failed")
        elif getattr(self, "_analyzing_session_b_key", None) is None:
            log_session_b_debug("compare_refresh_called", False)

    def _drain_snapshot_results(self) -> None:
        """Transfer worker results into Tk-owned state on the main thread only."""
        self._snapshot_poll_job = None
        changed = False
        worker_idle = False
        current_gen = self._compare_load_gen
        if self.compare_state == CompareState.CANCELLED:
            # Drop any queued stale payloads.
            try:
                while True:
                    self._snapshot_results.get_nowait()
            except queue.Empty:
                pass
            return
        try:
            while True:
                gen, key, snap, err = self._snapshot_results.get_nowait()
                if key == _COMPARE_WORKER_SENTINEL:
                    if gen == current_gen:
                        worker_idle = True
                    continue
                if gen != current_gen:
                    analyzing_b = getattr(self, "_analyzing_session_b_key", None)
                    accept_late_b = bool(
                        analyzing_b
                        and key == analyzing_b
                        and self.compare_state != CompareState.CANCELLED
                    )
                    if not accept_late_b:
                        log_compare_event(
                            self.gui,
                            "stale_result_ignored",
                            gen=gen,
                            current=current_gen,
                            key=key,
                        )
                        continue
                    log_compare_event(
                        self.gui,
                        "late_session_b_accepted",
                        gen=gen,
                        current=current_gen,
                        key=key,
                    )
                if snap is not None:
                    from dataclasses import replace

                    cache = ensure_compare_cache(self.gui)
                    is_session_b = bool(
                        cache.right_key and key == cache.right_key
                    )
                    left_snap = cache.left()
                    # Session B must never alias Session A's object. Re-key an
                    # independent snapshot onto the requested B slot.
                    if is_session_b and left_snap is not None and snap is left_snap:
                        log_compare_event(
                            self.gui,
                            "session_a_pin_blocked",
                            key=key,
                            left=cache.left_key,
                            generation=gen,
                        )
                        self._failed_snapshot_keys.add(key)
                        self._session_b_error = (
                            "Session B analysis produced Session A's snapshot "
                            "and was blocked."
                        )
                    else:
                        if (
                            is_session_b
                            and cache.left_key == key
                            and left_snap is not None
                            and snap is not left_snap
                        ):
                            cache.rekey(key, f"current:{key}")
                            cache.set_slot("right", key)
                        if snap.key != key:
                            snap = replace(snap, key=key)
                        cache.pin(snap)
                        self._failed_snapshot_keys.discard(key)
                        if is_session_b:
                            self._session_b_error = None
                            self._save_compare_session_b(snap)
                        log_compare_event(
                            self.gui,
                            "ui_pinned",
                            key=key,
                            frames=getattr(snap, "n_frames", None),
                            generation=gen,
                        )
                else:
                    cache = ensure_compare_cache(self.gui)
                    self._failed_snapshot_keys.add(key)
                    if cache.right_key and key == cache.right_key:
                        self._session_b_error = (
                            err or f"Could not analyze Session B ('{key}')."
                        )
                        try:
                            from stablewalk.ui.tk.compare_state import (
                                compare_error_log_path,
                            )
                            from datetime import datetime, timezone

                            path = compare_error_log_path()
                            path.parent.mkdir(parents=True, exist_ok=True)
                            with path.open("a", encoding="utf-8") as fh:
                                fh.write(
                                    f"\n--- {datetime.now(timezone.utc).isoformat()} "
                                    f"| session_b_failed:{key} ---\n"
                                )
                                fh.write(str(self._session_b_error) + "\n")
                        except Exception:
                            logger.exception("Failed to write Session B error log")
                    logger.warning(
                        "Compare main-thread UI update: %s failed (%s)",
                        key,
                        err or "incomplete",
                    )
                self._loading_keys.discard(key)
                changed = True
        except queue.Empty:
            pass

        if changed:
            try:
                self._sync_loading_banner()
            except Exception:
                logger.exception("Compare loading banner refresh failed")

        still_busy = bool(self._loading_keys) or (
            self._compare_worker is not None and self._compare_worker.is_alive()
        )
        if still_busy or not self._snapshot_results.empty():
            if self.compare_state != CompareState.CANCELLED:
                self._set_compare_state(CompareState.LOADING, reason="drain_busy")
            # Keep draining the worker queue even if the user visits another
            # tab. This callback is lightweight and leaves the detailed paint
            # deferred until Compare is visible again.
            try:
                self._snapshot_poll_job = self.gui.root.after(
                    100, self._drain_snapshot_results
                )
            except tk.TclError:
                self._snapshot_poll_job = None
            return

        self._snapshot_poll_job = None
        if worker_idle or changed or self.compare_state == CompareState.LOADING:
            self._finish_compare_load(timed_out=False)

    def refresh(self) -> None:
        """Full refresh. Static Compare is cache chrome + Δ table only."""
        if getattr(self, "_static_compare_only", True):
            self._refresh_static_compare()
            return
        self._refresh_chrome_only()
        if self.compare_loading or self._loading_keys:
            return
        self._refresh_visible_visuals(force=True)

    def _refresh_chrome_only(self, *, fetch_missing: bool = False) -> None:
        """Update pickers / KPI / Δ text without building Matplotlib figures.

        Default ``fetch_missing=False`` — chrome never starts a worker by itself.
        Static Compare never fetches missing sessions (no MediaPipe / OpenSim).
        """
        cache = ensure_compare_cache(self.gui)
        loading_keys = list(self._loading_keys)
        analyzing_b = getattr(self, "_analyzing_session_b_key", None)
        if analyzing_b and analyzing_b not in loading_keys:
            loading_keys.append(analyzing_b)
        if (
            fetch_missing
            and not getattr(self, "_static_compare_only", True)
            and self.compare_state not in (CompareState.LOADING, CompareState.READY)
            and not self._loading_keys
            and not analyzing_b
        ):
            self._request_compare_snapshots([cache.left_key, cache.right_key])

        choices = cache.choice_keys()
        self._key_choices = choices
        cmb_left = getattr(self, "cmb_left", None)
        cmb_right = getattr(self, "cmb_right", None)
        if cmb_left is not None:
            try:
                cmb_left.configure(values=choices)
            except tk.TclError:
                pass
        if cmb_right is not None:
            try:
                cmb_right.configure(values=choices)
            except tk.TclError:
                pass

        # Preserve requested keys even when a session is not yet loaded.
        # Never rewrite Session B to Session A's key.
        if cache.left_key not in choices and choices:
            # Only fall back for an empty/invalid left key — not onto right.
            if not cache.left_key:
                cache.left_key = "normal" if "normal" in choices else choices[0]
        if cache.right_key not in choices and choices:
            if not cache.right_key:
                cache.right_key = next(
                    (
                        k
                        for k in ("abnormal", "athletic", *choices)
                        if k and k != cache.left_key
                    ),
                    "",
                )
        left_var = getattr(self, "left_key_var", None)
        right_var = getattr(self, "right_key_var", None)
        if left_var is not None:
            try:
                left_var.set(cache.left_key)
            except Exception:
                pass
        if right_var is not None:
            try:
                right_var.set(cache.right_key)
            except Exception:
                pass

        left, right = self._session_pair()
        cached = build_cached_comparison(
            cache,
            loading_keys=loading_keys,
            previous=getattr(self, "_cached_comparison", None),
            right_override=right,
        )
        self._cached_comparison = cached
        validation = cached.validation
        log_compare_debug(
            "compare_data_generation",
            state=getattr(validation, "state", None),
            can_compare=getattr(validation, "can_compare", False),
            left_frames=getattr(left, "n_frames", None),
            right_frames=getattr(right, "n_frames", None),
        )
        identical = bool(validation and validation.identical)
        self._sync_reload_button(validation)

        left_status = self._session_card_status(left, key=cache.left_key)
        right_status = self._session_card_status(right, key=cache.right_key)
        both_ready, waiting_message = self._both_sessions_chart_ready()
        self._sync_compare_ready_sections(
            both_ready and not identical,
            message=waiting_message,
        )

        missing = []
        if left is None:
            missing.append(cache.left_key)
        if right is None:
            missing.append(cache.right_key)
        right_loading = right_status == "loading" or (
            right is None and cache.right_key in self._loading_keys
        )
        left_loading = left_status == "loading" or (
            left is None and cache.left_key in self._loading_keys
        )

        if validation is not None:
            fg = SUCCESS
            if validation.state in (
                CompareUiState.INCOMPLETE,
                CompareUiState.IDENTICAL,
                CompareUiState.EMPTY,
            ):
                fg = WARNING
            elif validation.state == CompareUiState.ERROR:
                fg = DANGER
            elif validation.state == CompareUiState.LOADING:
                fg = MUTED
            status_text = validation.message
            session_b_err = str(getattr(self, "_session_b_error", "") or "").strip()
            if (
                session_b_err
                and not validation.right_ready
                and not right_loading
            ):
                status_text = session_b_err
                fg = DANGER
            elif analyzing_b and not validation.right_ready:
                status_text = "Analyzing Session B..."
                fg = MUTED
            self.lbl_status.configure(text=status_text, fg=fg)
            if validation.state in (
                CompareUiState.INCOMPLETE,
                CompareUiState.ERROR,
                CompareUiState.IDENTICAL,
            ):
                self._show_compare_error(status_text)
            elif validation.state == CompareUiState.READY:
                self._clear_compare_error()
        elif (right_loading or left_loading) and not getattr(
            self, "_static_compare_only", True
        ):
            self.lbl_status.configure(text="Loading sessions…", fg=MUTED)

        self._set_video_titles(left, right)
        self._sync_compare_play_enabled()
        try:
            self._fill_differences_table()
        except Exception:
            pass

        delta_by_field: dict[str, float | None] = {}
        result: SessionComparisonResult | None = cached.result
        if identical:
            self._fill_diffs(None, identical=True)
            self._set_interp(
                "Both comparison sessions are identical.\n\n"
                "Select two different completed analyses to compute differences."
            )
        elif result is not None and validation is not None and validation.can_compare:
            self._fill_diffs(result)
            if left is not None and right is not None:
                delta_by_field = self._deltas_from_metrics(left.metrics, right.metrics)
            parts = [
                result.lab_report_summary or result.interpretation,
            ]
            if result.research_summary:
                parts.append(result.research_summary)
            parts.append(f"\n{DISCLAIMER_COMPARE}\n{DISCLAIMER_RESEARCH}")
            self._set_interp("\n\n".join(p for p in parts if p))
        else:
            self._fill_diffs(None)
            if missing and any(k in self._loading_keys for k in missing):
                self._set_interp(
                    "Loading the second session from demo cache…\n\n"
                    "Difference metrics appear when Session A and Session B "
                    "both have finished analysis."
                )
            else:
                msg = (
                    validation.message
                    if validation is not None
                    else "Select two completed analysis sessions."
                )
                action = ""
                if validation is not None and validation.action == "analyze_session_b":
                    action = "\n\nClick “Analyze Session B” to build it from cached poses, or Browse… a saved session bundle."
                elif validation is not None and validation.action == "analyze_session_a":
                    action = "\n\nAnalyze Session A (or pin the current workspace analysis as A)."
                self._set_interp(
                    f"{msg}{action}\n\n"
                    "Comparison uses only metrics available in both sessions. "
                    "Δ = B − A. Reload is enabled when both sessions are valid and distinct."
                )

        if getattr(self, "card_left", None) is not None:
            self._fill_metric_card(
                self.card_left,
                left.metrics if left else CompareMetrics(label="Session A"),
                header_prefix="Session A",
                status=left_status,
                deltas=delta_by_field if left_status == "ready" and delta_by_field else None,
            )
        if getattr(self, "card_right", None) is not None:
            self._fill_metric_card(
                self.card_right,
                right.metrics if right else CompareMetrics(label="Session B"),
                header_prefix="Session B",
                status=right_status,
                deltas=delta_by_field if right_status == "ready" and delta_by_field else None,
            )
        self._fill_bottom_metrics_bar(
            left.metrics if left else None,
            right.metrics if right else None,
            delta_by_field if left_status == "ready" and right_status == "ready" else None,
        )
        self._fill_difference_summary(
            left.metrics if left else None,
            right.metrics if right else None,
            identical=identical,
            ready=(
                left_status == "ready"
                and right_status == "ready"
                and not identical
            ),
        )

        self._paths_dirty = self._chart_dirty = self._heat_dirty = True
        self._visuals_dirty = True

    def _fill_difference_summary(
        self,
        left: CompareMetrics | None,
        right: CompareMetrics | None,
        *,
        identical: bool = False,
        ready: bool = False,
    ) -> None:
        """Auto-refresh Difference Summary (green = improvement, red = deterioration)."""
        from stablewalk.ui.tk.compare_delta_visuals import fill_difference_summary

        handles = getattr(self, "_difference_summary", None)
        fill_difference_summary(
            handles,
            left,
            right,
            identical=identical,
            ready=ready and not identical,
        )

    def _sync_compare_ready_sections(
        self,
        ready: bool,
        *,
        message: str = "",
    ) -> None:
        """Show comparison summaries only after two valid sessions are ready."""
        summary = getattr(self, "_compare_summary_host", None)
        bottom = getattr(self, "_compare_bottom_host", None)
        waiting = getattr(self, "_compare_waiting_host", None)
        waiting_label = getattr(self, "_compare_waiting_label", None)
        handles = getattr(self, "_difference_summary", None)
        try:
            if handles:
                from stablewalk.ui.tk.compare_delta_visuals import (
                    _set_difference_summary_mode,
                )

                _set_difference_summary_mode(handles, show_metrics=bool(ready))
            if ready:
                if waiting is not None:
                    waiting.grid_remove()
                if summary is not None:
                    summary.grid()
                if bottom is not None:
                    bottom.grid()
                return
            if summary is not None:
                summary.grid_remove()
            if bottom is not None:
                bottom.grid_remove()
            if waiting_label is not None:
                err = str(getattr(self, "_session_b_error", "") or "").strip()
                analyzing = bool(getattr(self, "_analyzing_session_b_key", None)) or (
                    ensure_compare_cache(self.gui).right_key
                    in getattr(self, "_loading_keys", set())
                )
                if err:
                    shown = err
                elif analyzing:
                    shown = "Analyzing Session B..."
                else:
                    shown = message or (
                        f"{SESSION_B_WAITING_TITLE}\n{SESSION_B_GUIDANCE_TEXT}"
                    )
                waiting_label.configure(text=shown)
            if waiting is not None:
                waiting.grid()
                waiting.lift()
        except tk.TclError:
            pass

    @staticmethod
    def _deltas_from_metrics(
        left: CompareMetrics,
        right: CompareMetrics,
    ) -> dict[str, float | None]:
        """Δ = B − A per card field; None when either side is unavailable."""
        pairs: dict[str, tuple[float | None, float | None]] = {
            "Gait Quality": (left.gait_quality, right.gait_quality),
            "Cadence": (left.cadence_spm, right.cadence_spm),
            "Speed": (left.walking_speed_m_s, right.walking_speed_m_s),
            "Step Length": (left.step_length_m, right.step_length_m),
            "Stride Length": (left.stride_length_m, right.stride_length_m),
            "Knee ROM": (left.knee_rom_deg, right.knee_rom_deg),
            "Joint ROM": (left.knee_rom_deg, right.knee_rom_deg),
            "Symmetry": (left.symmetry_pct, right.symmetry_pct),
            "COM Excursion": (left.com_excursion_m, right.com_excursion_m),
            "Stability": (left.stability_margin_m, right.stability_margin_m),
            "Hip ROM": (left.hip_rom_deg, right.hip_rom_deg),
            "Ankle ROM": (left.ankle_rom_deg, right.ankle_rom_deg),
            "Foot clearance": (
                getattr(left, "foot_clearance_m", None),
                getattr(right, "foot_clearance_m", None),
            ),
            "Usable gait cycles": (
                None if left.usable_gait_cycles is None else float(left.usable_gait_cycles),
                None if right.usable_gait_cycles is None else float(right.usable_gait_cycles),
            ),
        }
        out: dict[str, float | None] = {}
        for field, (a, b) in pairs.items():
            if a is None or b is None:
                out[field] = None
            else:
                try:
                    out[field] = float(b) - float(a)
                except (TypeError, ValueError):
                    out[field] = None
        return out

    def _paint_videos_only(self, *, force: bool = False) -> None:
        if not getattr(self, "_videos_built", False):
            return
        if not hasattr(self, "video_a") or not hasattr(self, "video_b"):
            return
        left, right = self._session_pair()
        ia, ib = self._index_for(left), self._index_for(right)
        if not force and self._last_paint == (ia, ib):
            self._update_progress_readout()
            return
        self._last_paint = (ia, ib)
        self._paint_video(self.video_a, left, ia)
        self._paint_video(self.video_b, right, ib)
        self._update_time_label(left, right, ia, ib)
        self._update_progress_readout()

    def _refresh_visible_visuals(self, *, force: bool = False) -> None:
        """Lite: chrome + knee chart. Detailed: also videos/skeletons/extra charts."""
        if not self._is_compare_tab_active():
            self._visuals_dirty = True
            return
        if self.compare_loading or self._loading_keys:
            self._visuals_dirty = True
            return

        # Lite path — cached metrics + one chart only (no video decode).
        if not self._detailed_comparison:
            self._ensure_lite_chart_built()
            if force or self._chart_dirty:
                self._redraw_chart_series()
                self._chart_dirty = False
            self._update_chart_playhead()
            self._update_progress_readout()
            sync = getattr(self, "_sync_compare_scroll", None)
            if callable(sync):
                sync()
            self._visuals_dirty = False
            return

        for img in list(self._pil_frame_cache.values()):
            try:
                img.close()
            except Exception:
                pass
        self._pil_frame_cache.clear()
        self._photo_cache.clear()
        self._video_box.clear()

        if self._videos_built:
            self._paint_videos_only(force=True)

        if self._skeletons_built and (
            force
            or self._section_visible(getattr(self, "_skel_section", None))
        ):
            self._paint_skeletons(*self._session_pair())

        self._ensure_lite_chart_built()
        if force or self._chart_dirty:
            self._redraw_chart_series()
            self._chart_dirty = False
        self._update_chart_playhead()

        if self._graphs_built:
            self._refresh_detailed_charts(force=force)
        elif self._section_visible(getattr(self, "_detail_graphs_section", None)):
            self._ensure_detailed_graphs_built()
            self._refresh_detailed_charts(force=True)

        sync = getattr(self, "_sync_compare_scroll", None)
        if callable(sync):
            sync()
        self._visuals_dirty = False

    def _visible_graph_panel(self) -> str | None:
        """Which graph panel is most visible: path, chart, or heat."""
        if not self._lite_chart_built and not self._graphs_built:
            return None
        if not self._detailed_comparison or not self._graphs_built:
            return "chart"
        panels = (
            ("path", getattr(self, "canvas_path", None)),
            ("chart", getattr(self, "canvas_chart", None)),
            ("heat", getattr(self, "canvas_heat", None)),
        )
        best: str | None = None
        best_overlap = 0
        canvas = getattr(self, "_compare_scroll_canvas", None)
        if canvas is None:
            return "chart"
        try:
            ctop = int(canvas.winfo_rooty())
            cbot = ctop + max(int(canvas.winfo_height()), 1)
        except tk.TclError:
            return "chart"
        for name, mpl_canvas in panels:
            if mpl_canvas is None:
                continue
            try:
                w = mpl_canvas.get_tk_widget()
                top = int(w.winfo_rooty())
                bot = top + max(int(w.winfo_height()), 1)
                overlap = max(0, min(bot, cbot) - max(top, ctop))
            except (tk.TclError, AttributeError):
                continue
            if overlap > best_overlap:
                best_overlap = overlap
                best = name
        return best or "chart"

    def _refresh_visible_chart(self, *, force: bool = False) -> None:
        """Redraw the lite knee chart; optionally detailed path/heatmap."""
        if self._lite_chart_built and (force or self._chart_dirty):
            self._redraw_chart_series()
            self._chart_dirty = False
            self._update_chart_playhead()
        elif self._lite_chart_built:
            self._update_chart_playhead()

        if not self._graphs_built:
            self._graphs_series_ready = self._lite_chart_built
            return

        if force:
            self._redraw_paths()
            self._redraw_heatmap()
            self._paths_dirty = self._heat_dirty = False
            self._graphs_series_ready = True
            self._update_path_markers()
            self._update_heat_playhead()
            return

        which = self._visible_graph_panel() or "chart"
        prev = self._active_chart_section
        self._active_chart_section = which
        panel_changed = prev is not None and prev != which

        if which == "path" and (self._paths_dirty or panel_changed):
            self._redraw_paths()
            self._paths_dirty = False
        elif which == "heat" and (self._heat_dirty or panel_changed):
            self._redraw_heatmap()
            self._heat_dirty = False

        self._graphs_series_ready = True
        if which == "path":
            self._update_path_markers()
        elif which == "heat":
            self._update_heat_playhead()
        else:
            self._update_chart_playhead()

    @staticmethod
    def _metrics_ready(snap: AnalyzedSessionSnapshot | None) -> bool:
        """True when a session finished analysis with usable compare KPIs.

        Knee-ROM alone (pose-derived filler) is not enough — at least one
        primary gait KPI must be present so Session B cards are not empty
        while trajectories unlock.
        """
        if snap is None or not bool(getattr(snap, "ready", False)):
            return False
        m = snap.metrics
        primary = any(
            v is not None
            for v in (
                m.gait_quality,
                m.cadence_spm,
                m.walking_speed_m_s,
                m.step_length_m,
                m.symmetry_pct,
                m.stability_margin_m,
            )
        )
        if primary:
            return True
        # Accept knee ROM only when duration proves a full analysis pass ran.
        return (
            m.knee_rom_deg is not None
            and m.duration_s is not None
            and float(m.duration_s) > 0.0
            and (
                m.hip_rom_deg is not None
                or m.com_excursion_m is not None
                or m.gait_quality is not None
            )
        )

    def _sessions_are_independent(
        self,
        left: AnalyzedSessionSnapshot | None,
        right: AnalyzedSessionSnapshot | None,
        *,
        left_key: str,
        right_key: str,
    ) -> bool:
        """True when A and B are distinct session objects and distinct result caches."""
        if left is None or right is None:
            return False
        if left_key == right_key:
            return False
        if left is right:
            return False
        if id(left) == id(right):
            return False
        if id(left.metrics) == id(right.metrics):
            return False
        # Never treat a shared trajectory / pose buffer as two sessions.
        if (
            left.path_xyz is not None
            and right.path_xyz is not None
            and left.path_xyz is right.path_xyz
        ):
            return False
        if (
            left.sequence is not None
            and right.sequence is not None
            and left.sequence is right.sequence
        ):
            return False
        if sessions_are_identical(
            left, right, left_key=left_key, right_key=right_key
        ):
            return False
        return True

    def _session_card_status(
        self,
        snap: AnalyzedSessionSnapshot | None,
        *,
        key: str,
    ) -> str:
        """ready | loading | unavailable | incomplete."""
        if key in self._loading_keys and (
            snap is None or not self._metrics_ready(snap)
        ):
            return "loading"
        if snap is not None and self._metrics_ready(snap):
            return "ready"
        if snap is None:
            if key in getattr(self, "_failed_snapshot_keys", set()):
                return "unavailable"
            return "unavailable"
        return "incomplete"

    def _both_sessions_chart_ready(self) -> tuple[bool, str]:
        """True only when two independent, valid sessions can be compared."""
        cache = ensure_compare_cache(self.gui)
        left, right = self._session_pair()
        if right is None and cache.right_key in self._loading_keys:
            return False, "Loading Session B..."
        if left is None and cache.left_key in self._loading_keys:
            return False, "Loading Session A..."
        if left is None or right is None:
            missing = []
            if left is None:
                missing.append("Session A")
            if right is None:
                missing.append("Session B")
            loading = any(
                k in self._loading_keys for k in (cache.left_key, cache.right_key)
            )
            if loading:
                return False, "Loading both sessions..."
            if missing == ["Session B"]:
                err = str(getattr(self, "_session_b_error", "") or "").strip()
                if err:
                    return False, err
                return (
                    False,
                    f"{SESSION_B_WAITING_TITLE}\n{SESSION_B_GUIDANCE_TEXT}",
                )
            return False, f"Waiting for {' and '.join(missing)}"
        if not self._sessions_are_independent(
            left, right, left_key=cache.left_key, right_key=cache.right_key
        ):
            return False, "Session A and Session B must use different analysis objects"
        if not left.ready or not right.ready:
            if not right.ready:
                return (
                    False,
                    "Session B analysis is incomplete.\n"
                    f"{SESSION_B_NEXT_ACTION}",
                )
            return False, "Comparison unavailable — Session A is incomplete."
        if not self._metrics_ready(left):
            return False, "Comparison unavailable — Session A is incomplete."
        if not self._metrics_ready(right):
            return (
                False,
                "Session B analysis is incomplete.\n"
                f"{SESSION_B_NEXT_ACTION}",
            )
        from stablewalk.ui.analysis_mode import analysis_modes_comparable

        comparable, mode_reason = analysis_modes_comparable(
            getattr(left, "analysis_mode", None),
            getattr(right, "analysis_mode", None),
            max_frames_a=getattr(left, "max_frames", None),
            max_frames_b=getattr(right, "max_frames", None),
        )
        if not comparable:
            return False, mode_reason
        return True, ""

    def _sync_compare_play_enabled(self) -> None:
        """Enable playback only after both sessions have valid independent metrics."""
        ready, message = self._both_sessions_chart_ready()
        btn = getattr(self, "btn_play", None)
        if btn is not None:
            try:
                btn.configure(state=tk.NORMAL if ready else tk.DISABLED)
            except tk.TclError:
                pass
        if not ready and self._playing:
            self.pause(paint=False)
            if message:
                try:
                    self.lbl_status.configure(text=message, fg=WARNING)
                except Exception:
                    pass

    @staticmethod
    def _snap_can_use_gait_axis(snap: AnalyzedSessionSnapshot | None) -> bool:
        if snap is None:
            return False
        usable = int(getattr(snap.metrics, "usable_gait_cycles", 0) or 0)
        hs = list(getattr(snap, "heel_strike_times_s", None) or [])
        return usable >= _COMPARE_MIN_USABLE_CYCLES and len(hs) >= 2

    def _compare_axis_mode_for(
        self,
        left: AnalyzedSessionSnapshot | None,
        right: AnalyzedSessionSnapshot | None,
    ) -> str:
        if self._snap_can_use_gait_axis(left) and self._snap_can_use_gait_axis(right):
            return _COMPARE_AXIS_GAIT
        return _COMPARE_AXIS_VIDEO

    @staticmethod
    def _time_to_cycle_percent(time_s: float, heel_strikes: list[float]) -> float | None:
        hs = [float(t) for t in heel_strikes if np.isfinite(t)]
        if len(hs) < 2:
            return None
        t = float(time_s)
        for i in range(len(hs) - 1):
            t0, t1 = hs[i], hs[i + 1]
            if t0 <= t <= t1 and t1 > t0:
                return max(0.0, min(100.0, 100.0 * (t - t0) / (t1 - t0)))
        if t < hs[0]:
            return 0.0
        return 100.0

    def _normalized_knee_xy(
        self,
        snap: AnalyzedSessionSnapshot,
        *,
        mode: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (x 0–100, y°) for overlay — gait-cycle % or video progress."""
        if not snap.knee_t:
            return None
        t = np.asarray(snap.knee_t, dtype=float)
        y = np.asarray(
            [np.nan if v is None else float(v) for v in snap.knee_y], dtype=float
        )
        if len(t) < 2:
            return None
        if mode == _COMPARE_AXIS_GAIT:
            hs = list(getattr(snap, "heel_strike_times_s", None) or [])
            xs: list[float] = []
            ys: list[float] = []
            for ti, yi in zip(t, y):
                if not np.isfinite(yi):
                    continue
                pct = self._time_to_cycle_percent(float(ti), hs)
                if pct is None:
                    continue
                xs.append(float(pct))
                ys.append(float(yi))
            if len(xs) < 2:
                return None
            # Average into a regular 0–100% grid so A/B share one x-axis.
            grid = np.linspace(0.0, 100.0, 101)
            order = np.argsort(xs)
            x_sorted = np.asarray(xs, dtype=float)[order]
            y_sorted = np.asarray(ys, dtype=float)[order]
            # np.interp needs unique x — collapse duplicates by mean.
            uniq_x, inv = np.unique(np.round(x_sorted, 3), return_inverse=True)
            uniq_y = np.zeros_like(uniq_x)
            for i in range(len(uniq_x)):
                uniq_y[i] = float(np.nanmean(y_sorted[inv == i]))
            if len(uniq_x) < 2:
                return None
            return grid, np.interp(grid, uniq_x, uniq_y)
        xn = (t - t[0]) / max(float(t[-1] - t[0]), 1e-6) * 100.0
        return xn, y

    @staticmethod
    def _axis_xlabel(mode: str) -> str:
        if mode == _COMPARE_AXIS_GAIT:
            return "Gait cycle (%)"
        return "Video progress (%)"

    @staticmethod
    def _axes_is_3d(ax) -> bool:
        name = str(getattr(ax, "name", "") or "")
        if name == "3d":
            return True
        return type(ax).__name__.lower().find("3d") >= 0

    def _chart_unavailable_message(self, ax, fig, canvas, message: str) -> None:
        ax.cla()
        is_3d = ax is getattr(self, "ax_path", None) or self._axes_is_3d(ax)
        if ax is getattr(self, "ax_chart", None):
            self._style_chart_ax()
        elif is_3d:
            setup_3d_axes(ax)
        else:
            ax.set_facecolor(ELEVATED)
        if is_3d:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            z0, z1 = ax.get_zlim()
            ax.text(
                0.5 * (x0 + x1),
                0.5 * (y0 + y1),
                0.5 * (z0 + z1),
                str(message),
                ha="center",
                va="center",
                color=MUTED,
                fontsize=10,
            )
        else:
            try:
                ax.set_axis_off()
            except Exception:
                pass
            ax.text(
                0.5,
                0.5,
                str(message),
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=MUTED,
                fontsize=10,
                wrap=True,
            )
        self._safe_draw(canvas)

    def _set_video_titles(
        self,
        left: AnalyzedSessionSnapshot | None,
        right: AnalyzedSessionSnapshot | None,
    ) -> None:
        if not getattr(self, "_videos_built", False):
            return
        if not hasattr(self, "video_a") or not hasattr(self, "video_b"):
            return
        from stablewalk.ui.analysis_mode import (
            format_analysis_mode_label,
            normalize_analysis_mode,
        )

        a_name = (left.label if left else None) or "Session A"
        b_name = (right.label if right else None) or "Session B"
        a_key = (
            normalize_analysis_mode(
                getattr(left, "analysis_mode", None),
                max_frames=getattr(left, "max_frames", None),
            )
            if left is not None
            else ""
        )
        b_key = (
            normalize_analysis_mode(
                getattr(right, "analysis_mode", None),
                max_frames=getattr(right, "max_frames", None),
            )
            if right is not None
            else ""
        )
        a_mode = (
            format_analysis_mode_label(a_key, getattr(left, "max_frames", None))
            if a_key
            else ""
        )
        b_mode = (
            format_analysis_mode_label(b_key, getattr(right, "max_frames", None))
            if b_key
            else ""
        )
        try:
            self.video_a["frame"].configure(
                text=f"Session A — {a_name}" + (f" ({a_mode})" if a_mode else "")
            )
            self.video_b["frame"].configure(
                text=f"Session B — {b_name}" + (f" ({b_mode})" if b_mode else "")
            )
        except Exception:
            pass
        try:
            if getattr(self, "lbl_slot_a", None) is not None:
                self.lbl_slot_a.configure(
                    text="Session A" + (f" · {a_mode}" if a_mode else "")
                )
            if getattr(self, "lbl_slot_b", None) is not None:
                self.lbl_slot_b.configure(
                    text="Session B" + (f" · {b_mode}" if b_mode else "")
                )
        except Exception:
            pass

    # ── slot / transport ────────────────────────────────────────────────

    def _on_slot_change(self, *, changed_slot: str | None = None) -> None:
        if self.compare_state == CompareState.LOADING:
            try:
                self.lbl_status.configure(
                    text=COMPARE_ALREADY_LOADING_MESSAGE, fg=WARNING
                )
            except Exception:
                pass
            return
        cache = ensure_compare_cache(self.gui)
        left_key = self.left_key_var.get()
        right_key = self.right_key_var.get()
        if right_key != cache.right_key:
            self._session_b_error = None
        left_snap = cache.get(left_key) if left_key else None
        right_snap = cache.get(right_key) if right_key else None
        same_identity = bool(left_key and left_key == right_key) or sessions_are_identical(
            left_snap,
            right_snap,
            left_key=left_key or None,
            right_key=right_key or None,
        )
        if same_identity:
            try:
                if changed_slot == "left":
                    self.left_key_var.set(cache.left_key)
                elif changed_slot == "right":
                    self.right_key_var.set(cache.right_key)
                else:
                    self.left_key_var.set(cache.left_key)
                    self.right_key_var.set(cache.right_key)
            except (AttributeError, tk.TclError):
                pass
            try:
                from stablewalk.ui.tk.tk_lifecycle import safe_messagebox

                safe_messagebox(
                    "warning",
                    "Same session",
                    "Session A and Session B must be different analyses.",
                    parent=getattr(self.gui, "root", None),
                )
                self.lbl_status.configure(
                    text="Choose a different recording for Session B.",
                    fg=DANGER,
                )
            except Exception:
                pass
            return
        old_right = cache.right_key
        cache.set_slot("left", left_key)
        cache.set_slot("right", right_key)
        self._invalidate_compare_outputs(clear_session_b=(right_key != old_right))
        log_compare_debug(
            "slot_change",
            left=left_key,
            right=right_key,
            changed_slot=changed_slot,
        )
        # Never auto-analyze missing sessions from the dropdown — refresh from cache.
        self._static_refresh_fp = None
        if getattr(self, "_detailed_comparison", False) and not getattr(
            self, "_static_compare_only", True
        ):
            self.request_refresh()
        else:
            self._refresh_static_compare(force=True)

    def _on_skeleton_mode(self) -> None:
        if not self._detailed_comparison:
            return
        if not self._skeletons_built:
            self._ensure_skeletons_built()
        if not self._skeletons_built:
            return
        # Split LEFT/RIGHT layout keeps skeletons under each video (no overlay swap).
        if getattr(self, "skel_side", None) is None:
            self._skeleton_mode.set("side")
            self._paint_frame(force=True)
            return
        mode = self._skeleton_mode.get()
        if mode == "overlay":
            self.skel_side.grid_remove()
            self.skel_overlay.grid(row=0, column=0, sticky="nsew")
        else:
            self.skel_overlay.grid_remove()
            self.skel_side.grid(row=0, column=0, sticky="nsew")
        self._paint_frame(force=True)
        sync = getattr(self, "_sync_compare_scroll", None)
        if callable(sync):
            sync()

    def _on_timeline(self, value: str) -> None:
        try:
            self._progress = float(value)
        except ValueError:
            return
        self._paint_frame()

    def _on_speed(self, _e: object | None = None) -> None:
        raw = self.speed_var.get().replace("×", "").strip()
        try:
            self._speed = max(0.1, float(raw))
        except ValueError:
            self._speed = 1.0

    def _on_sample(self, _e: object | None = None) -> None:
        try:
            self._sample_stride = max(1, int(self.sample_var.get()))
        except ValueError:
            self._sample_stride = 1

    def seek(self, progress: float) -> None:
        self._progress = float(np.clip(progress, 0.0, 1.0))
        self.timeline.set(self._progress)
        self._paint_frame(force=True)

    def _seek_from_normalized_x(self, x_percent: float) -> None:
        """Seek shared progress from a 0–100% chart / trajectory X click."""
        self.seek(float(np.clip(x_percent, 0.0, 100.0)) / 100.0)

    def _seek_from_heatmap_x(self, x_bin: float) -> None:
        """Seek shared progress from a heatmap bin-index click."""
        denom = max(float(getattr(self, "_heat_bins", 40)) - 1.0, 1.0)
        self.seek(float(np.clip(x_bin, 0.0, denom)) / denom)

    def toggle_play(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if getattr(self, "_static_compare_only", True):
            try:
                self.lbl_status.configure(
                    text="Playback disabled in static comparison",
                    fg=MUTED,
                )
            except Exception:
                pass
            return
        ready, message = self._both_sessions_chart_ready()
        if not ready:
            try:
                self.lbl_status.configure(
                    text=message or "Comparison unavailable — Session B is incomplete.",
                    fg=WARNING,
                )
            except Exception:
                pass
            self._sync_compare_play_enabled()
            return
        # Mutual exclusion: never run Compare and main Overview timers together.
        self._pause_main_playback()
        self._playing = True
        self._skel_paint_tick = 0
        self._last_compare_chart_ts = 0.0
        self.btn_play.configure(text="Pause")
        self._tick()

    def _pause_main_playback(self) -> None:
        """Stop the main dashboard playback timer if it is running."""
        gui = self.gui
        if not getattr(gui, "playing", False):
            return
        try:
            gui.playing = False
            player = getattr(gui, "skeleton_player", None)
            if player is not None and hasattr(player, "pause"):
                player.pause()
            cancel = getattr(gui, "_cancel_timer", None)
            if callable(cancel):
                cancel()
            sync = getattr(gui, "_sync_play_buttons", None)
            if callable(sync):
                sync()
        except Exception:
            logger.debug("pause main playback from Compare failed", exc_info=True)

    def pause(self, *, paint: bool = True) -> None:
        self._playing = False
        try:
            self.btn_play.configure(text="Play")
        except Exception:
            pass
        if self._play_job is not None:
            try:
                self.gui.root.after_cancel(self._play_job)
            except Exception:
                pass
            self._play_job = None
        if self._video_resize_job is not None:
            try:
                self.gui.root.after_cancel(self._video_resize_job)
            except Exception:
                pass
            self._video_resize_job = None
        # Crisp final skeleton paint at the paused frame (skip when leaving tab).
        if paint:
            self._paint_frame(force=True)

    def release_temp_buffers(self) -> None:
        """Drop decode / PhotoImage caches and release OpenCV captures.

        Session pins, timeline progress, and chart series stay intact so the
        next Compare visit restores quickly without re-running analysis.
        """
        self.cancel_deferred_work()
        for img in list(self._pil_frame_cache.values()):
            try:
                img.close()
            except Exception:
                pass
        self._pil_frame_cache.clear()
        self._photo_cache.clear()
        self._video_box.clear()
        self._photo_a = None
        self._photo_b = None
        from stablewalk.core.pipeline_reset import unregister_capture

        for path, cap in list(self._video_caps.items()):
            try:
                unregister_capture(cap)
            except Exception:
                pass
            try:
                from stablewalk.ui.tk.ui_freeze_debug import log_capture

                log_capture("release", path=path)
            except Exception:
                pass
            try:
                cap.release()
            except Exception:
                pass
        self._video_caps.clear()

    def _tick(self) -> None:
        if not self._playing:
            return
        if not self._is_compare_tab_active():
            self.pause(paint=False)
            return
        ready, message = self._both_sessions_chart_ready()
        if not ready:
            self.pause(paint=False)
            try:
                self.lbl_status.configure(
                    text=message or "Comparison unavailable — Session B is incomplete.",
                    fg=WARNING,
                )
            except Exception:
                pass
            return
        cache = ensure_compare_cache(self.gui)
        left, right = self._session_pair()
        # Normalized progress sync: advance a shared 0–1 timeline so unequal
        # frame counts stay locked by percentage, not by absolute frame index.
        n = max(
            (left.n_frames if left else 1),
            (right.n_frames if right else 1),
            1,
        )
        fps = 30.0
        if left is not None and left.fps:
            fps = float(left.fps)
        elif right is not None and right.fps:
            fps = float(right.fps)
        step = (self._sample_stride / max(n - 1, 1))
        self._progress = min(1.0, self._progress + step)
        self.timeline.set(self._progress)
        self._paint_frame()
        if self._progress >= 1.0:
            self.pause()
            return
        delay_ms = max(16, int(1000 / (max(fps, 1.0) * self._speed)))
        self._play_job = self.gui.root.after(delay_ms, self._tick)

    # ── paint ───────────────────────────────────────────────────────────

    def _index_for(self, snap: AnalyzedSessionSnapshot | None) -> int:
        if snap is None or snap.n_frames <= 0:
            return 0
        return int(round(self._progress * (snap.n_frames - 1)))

    def _pose_index(self, snap: AnalyzedSessionSnapshot | None) -> int:
        idx = self._index_for(snap)
        if snap is None:
            return 0
        if snap.pose_indices:
            return snap.pose_indices[min(idx, len(snap.pose_indices) - 1)]
        return idx

    def _paint_frame(self, *, force: bool = False) -> None:
        # Static Compare never paints video / skeletons / per-frame charts.
        if getattr(self, "_static_compare_only", True):
            return
        if getattr(self, "_compare_painting", False):
            return
        if not self._is_compare_tab_active() and not force:
            return
        cache = ensure_compare_cache(self.gui)
        left, right = self._session_pair()
        ia, ib = self._index_for(left), self._index_for(right)

        import time

        now = time.monotonic()
        chart_hz = float(getattr(self, "_compare_chart_hz", 7.0) or 7.0)
        chart_interval = 1.0 / max(chart_hz, 1.0)
        do_charts = bool(force) or (
            (now - float(getattr(self, "_last_compare_chart_ts", 0.0))) >= chart_interval
        )

        # Lite Compare: progress + one chart playhead only (no video/skel decode).
        if not self._detailed_comparison:
            if not force and self._last_paint == (ia, ib):
                if do_charts and self._lite_chart_built:
                    self._last_compare_chart_ts = now
                    self._update_chart_playhead()
                self._update_progress_readout()
                return
            self._last_paint = (ia, ib)
            self._compare_painting = True
            try:
                if self._lite_chart_built:
                    if force and self._chart_dirty:
                        self._redraw_chart_series()
                        self._chart_dirty = False
                    elif do_charts:
                        self._last_compare_chart_ts = now
                        self._update_chart_playhead()
                self._update_progress_readout()
                self._update_time_label(left, right, ia, ib)
            finally:
                self._compare_painting = False
            return

        if not force and self._last_paint == (ia, ib):
            # Frame indices unchanged but progress moved sub-frame — keep the
            # continuous cursors gliding across the graphs (throttled).
            if do_charts and self._lite_chart_built:
                self._last_compare_chart_ts = now
                self._update_chart_playhead()
                if self._graphs_built:
                    self._update_heat_playhead()
                    self._update_path_markers()
            self._update_progress_readout()
            return
        self._last_paint = (ia, ib)

        self._compare_painting = True
        try:
            if self._videos_built and hasattr(self, "video_a"):
                self._paint_video(self.video_a, left, ia)
                self._paint_video(self.video_b, right, ib)
            if self._skeletons_built and self._section_visible(
                getattr(self, "_skel_section", None)
            ):
                # Skeleton A/B paint every tick with video (no LOD stride skip).
                self._skel_paint_tick = int(getattr(self, "_skel_paint_tick", 0)) + 1
                self._paint_skeletons(left, right)
            if self._lite_chart_built:
                if force and self._chart_dirty:
                    self._redraw_chart_series()
                    self._chart_dirty = False
                    self._last_compare_chart_ts = now
                elif do_charts:
                    self._last_compare_chart_ts = now
                    self._update_chart_playhead()
            if self._graphs_built and do_charts:
                which = self._active_chart_section or self._visible_graph_panel()
                if force and (self._paths_dirty or self._heat_dirty):
                    self._refresh_detailed_charts(force=False)
                elif which == "path":
                    self._update_path_markers()
                elif which == "heat":
                    self._update_heat_playhead()
                else:
                    self._update_path_markers()
                    self._update_heat_playhead()
            self._update_progress_readout()
            self._update_time_label(left, right, ia, ib)
        finally:
            self._compare_painting = False

    def _paint_video(
        self, panel: dict[str, Any], snap: AnalyzedSessionSnapshot | None, list_idx: int
    ) -> None:
        lbl: tk.Label = panel["label"]
        is_b = panel is getattr(self, "video_b", None)
        if is_b and (snap is None or self._session_b_needs_cta()):
            self._set_video_b_cta_visible(True)
            return
        if snap is None:
            lbl.configure(
                image="",
                text="Session A not loaded\nPin or choose an analyzed recording.",
                fg=MUTED,
                bg=PANEL,
            )
            return
        if is_b:
            self._set_video_b_cta_visible(False)
        img = self._load_frame_image(snap, list_idx)
        if img is None:
            unavailable_text = (
                "Video B unavailable\n"
                "No stored video frame is available for this analyzed session."
                if is_b
                else f"{snap.label}\nframe {list_idx + 1}/{snap.n_frames}"
            )
            lbl.configure(
                image="",
                text=unavailable_text,
                fg=MUTED,
                bg=PANEL,
            )
            return
        side = "a" if panel is self.video_a else "b"
        # Measure the host frame — never the image label (it shrinks to the bitmap).
        cached_box = self._video_box.get(side)
        try:
            if self._playing and cached_box is not None:
                max_w, max_h = cached_box
            else:
                host: tk.Misc = panel["frame"]
                max_w = int(host.winfo_width())
                max_h = int(host.winfo_height())
                if max_w <= 2 or max_h <= 2:
                    max_w, max_h = 480, 300
                self._video_box[side] = (max_w, max_h)
        except tk.TclError:
            max_w, max_h = 480, 300

        photo_key = (str(snap.key), int(list_idx), int(max_w), int(max_h), int(self._playing))
        photo = self._photo_cache.get(photo_key)
        if photo is None:
            try:
                fast = Image.Resampling.BILINEAR
                hq = Image.Resampling.LANCZOS
            except AttributeError:  # Pillow < 9.1
                fast = Image.BILINEAR  # type: ignore[attr-defined]
                hq = Image.LANCZOS  # type: ignore[attr-defined]
            resample = fast if self._playing else hq
            fitted = fit_compare_video_frame(img, max_w, max_h, resample=resample)
            photo = ImageTk.PhotoImage(fitted)
            if fitted is not img:
                fitted.close()
            self._photo_cache[photo_key] = photo
            # Bound cache growth during long scrubbing sessions.
            if len(self._photo_cache) > 24:
                for old in list(self._photo_cache.keys())[:8]:
                    self._photo_cache.pop(old, None)
        if panel is self.video_a:
            self._photo_a = photo
        else:
            self._photo_b = photo
        lbl.configure(image=photo, text="", bg=PANEL)

    def _load_frame_image(
        self, snap: AnalyzedSessionSnapshot, list_idx: int
    ) -> Image.Image | None:
        cache_key = (str(snap.key), int(list_idx))
        cached = self._pil_frame_cache.get(cache_key)
        if cached is not None:
            return cached

        img: Image.Image | None = None
        if snap.frame_paths:
            path = snap.frame_paths[min(list_idx, len(snap.frame_paths) - 1)]
            try:
                with Image.open(path) as opened:
                    img = opened.convert("RGB")
            except OSError:
                img = None
        if img is None and snap.video_path and Path(str(snap.video_path)).is_file():
            try:
                import cv2
                from stablewalk.core.pipeline_reset import register_capture

                vpath = str(snap.video_path)
                cap = self._video_caps.get(vpath)
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(vpath)
                    try:
                        from stablewalk.ui.tk.ui_freeze_debug import log_capture

                        log_capture("open", path=vpath)
                    except Exception:
                        pass
                    self._video_caps[vpath] = cap
                    register_capture(cap)
                if not cap.isOpened():
                    return None
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                frame_i = int(round(self._progress * max(total - 1, 0)))
                # Prefer list index when known; progress-based seek is fallback for
                # videos without pre-extracted frame paths.
                if snap.n_frames > 0:
                    frame_i = min(list_idx, max(total - 1, 0)) if total else list_idx
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    return None
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
            except Exception:
                return None
        if img is not None:
            self._pil_frame_cache[cache_key] = img
            if len(self._pil_frame_cache) > 32:
                for old in list(self._pil_frame_cache.keys())[:8]:
                    old_img = self._pil_frame_cache.pop(old, None)
                    if old_img is not None:
                        try:
                            old_img.close()
                        except Exception:
                            pass
        return img

    def _paint_skeletons(
        self,
        left: AnalyzedSessionSnapshot | None,
        right: AnalyzedSessionSnapshot | None,
    ) -> None:
        if not self._skeletons_built:
            return
        mode = self._skeleton_mode.get()
        if mode == "overlay":
            self.ax_skel_ov.cla()
            setup_3d_axes(self.ax_skel_ov)
            drawn = False
            for snap, color, alpha in (
                (left, COLOR_LEFT, 0.55),
                (right, COLOR_RIGHT, 0.75),
            ):
                sk = self._skeleton_at(snap)
                if sk is None or not sk.joints:
                    continue
                self._draw_skel_colored(self.ax_skel_ov, sk, color=color, alpha=alpha, clear=False)
                drawn = True
            if not drawn:
                self.ax_skel_ov.text2D(
                    0.5, 0.5, "No skeletons", transform=self.ax_skel_ov.transAxes, ha="center", color=MUTED
                )
            self.ax_skel_ov.set_title("Overlay Skeletons", color=TEXT, fontsize=9)
            self._safe_draw(self.canvas_skel_ov)
            return

        # Session B missing/incomplete → Analyze / Load overlay (never a blank panel).
        b_needs_cta = self._session_b_needs_cta()
        self._set_skel_b_cta_visible(b_needs_cta)

        for ax, canvas, snap, title, color in (
            (self.ax_skel_a, self.canvas_skel_a, left, "Skeleton A", COLOR_LEFT),
            (self.ax_skel_b, self.canvas_skel_b, right, "Skeleton B", COLOR_RIGHT),
        ):
            ax.cla()
            setup_3d_axes(ax)
            sk = self._skeleton_at(snap)
            if sk is None or not sk.joints:
                if title == "Skeleton B" and b_needs_cta:
                    # The actionable B overlay already explains the missing data.
                    pass
                elif snap is None:
                    ax.text2D(
                        0.5,
                        0.5,
                        "Session A not loaded",
                        transform=ax.transAxes,
                        ha="center",
                        color=MUTED,
                    )
                else:
                    ax.text2D(
                        0.5,
                        0.5,
                        (
                            "Skeleton B unavailable\n"
                            "No reconstructable pose data was stored."
                            if title == "Skeleton B"
                            else "No skeleton"
                        ),
                        transform=ax.transAxes,
                        ha="center",
                        color=MUTED,
                    )
            else:
                display = draw_skeleton_3d(ax, sk, clear=False, frame_label=title)
                self._fit_compare_skeleton_view(ax, display)
            ax.set_title(title, color=color, fontsize=9)
            self._safe_draw(canvas)

    def _fit_compare_skeleton_view(self, ax, skeleton) -> None:
        """Frame the Compare skeleton so the body fills most of the canvas."""
        if skeleton is None or not getattr(skeleton, "joints", None):
            return
        apply_display_limits(
            ax,
            skeleton,
            padding=1.0 / COMPARE_SKELETON_FILL,
            flat_frontal=False,
        )
        try:
            ax.set_box_aspect((1, 1, 1), zoom=COMPARE_SKELETON_BOX_ZOOM)
        except TypeError:
            try:
                ax.set_box_aspect((1, 1, 1))
            except (AttributeError, ValueError):
                pass
        except (AttributeError, ValueError):
            pass
        try:
            ax.dist = COMPARE_SKELETON_CAMERA_DIST
        except Exception:
            pass
        try:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
        except Exception:
            pass
        fig = getattr(ax, "figure", None)
        if fig is not None:
            try:
                fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02)
            except Exception:
                pass

    def _skeleton_at(self, snap: AnalyzedSessionSnapshot | None) -> Any:
        if snap is None:
            return None
        pose_i = self._pose_index(snap)
        sk = snap.skeletons.get(pose_i)
        if sk is not None:
            return sk
        # Lazy rebuild from the pinned PoseSequence (pins no longer store a full
        # per-frame skeleton dict — that duplicated the live GUI cache).
        seq = snap.sequence
        if seq is not None and getattr(seq, "frames", None):
            frames = seq.frames
            fi = pose_i
            if 0 <= fi < len(frames):
                frame = frames[fi]
                sk = getattr(frame, "skeleton_3d", None)
                if sk is not None and getattr(sk, "joints", None):
                    snap.skeletons[pose_i] = sk
                    self._trim_lazy_skeletons(snap)
                    return sk
                try:
                    from stablewalk.pose.skeleton_3d import reconstruct_skeleton_3d

                    kps = getattr(frame, "keypoints", None) or []
                    if kps:
                        sk = reconstruct_skeleton_3d(kps)
                        if sk is not None and getattr(sk, "joints", None):
                            snap.skeletons[pose_i] = sk
                            self._trim_lazy_skeletons(snap)
                            return sk
                except Exception:
                    pass
        # Fallback: nearest pre-stored sample
        if snap.skeletons:
            keys = sorted(snap.skeletons.keys())
            idx = keys[min(self._index_for(snap), len(keys) - 1)]
            return snap.skeletons.get(idx)
        return None

    @staticmethod
    def _trim_lazy_skeletons(snap: AnalyzedSessionSnapshot, *, max_entries: int = 48) -> None:
        """Bound per-session lazy skeleton growth during long Compare scrubbing."""
        if len(snap.skeletons) <= max_entries:
            return
        # Drop oldest insertion order (dict preserves order on 3.7+).
        overflow = len(snap.skeletons) - max_entries
        for key in list(snap.skeletons.keys())[:overflow]:
            snap.skeletons.pop(key, None)

    def _draw_skel_colored(self, ax, skeleton, *, color: str, alpha: float, clear: bool) -> None:
        # Lightweight bone draw for overlay mode (reuse joints).
        if clear:
            ax.cla()
            setup_3d_axes(ax)
        joints = skeleton.joints
        for a, b in SKELETON_3D_CONNECTIONS:
            ja, jb = joints.get(a), joints.get(b)
            if ja is None or jb is None:
                continue
            ax.plot(
                [ja.x, jb.x],
                [ja.y, jb.y],
                [ja.z, jb.z],
                color=color,
                alpha=alpha,
                linewidth=1.45,
            )
        xs = [j.x for j in joints.values()]
        ys = [j.y for j in joints.values()]
        zs = [j.z for j in joints.values()]
        if xs:
            ax.scatter(xs, ys, zs, c=color, s=12, alpha=alpha, depthshade=False)
            apply_display_limits(ax, skeleton, padding=1.0 / COMPARE_SKELETON_FILL, flat_frontal=False)
            try:
                ax.set_box_aspect((1, 1, 1), zoom=COMPARE_SKELETON_BOX_ZOOM)
            except TypeError:
                ax.set_box_aspect((1, 1, 1))
            except (AttributeError, ValueError):
                pass
            try:
                ax.dist = COMPARE_SKELETON_CAMERA_DIST
            except Exception:
                pass

    @staticmethod
    def _path_xyz_on_progress_grid(
        snap: AnalyzedSessionSnapshot,
        *,
        n: int = 64,
    ) -> np.ndarray | None:
        """Resample COM path onto a shared 0–1 progress grid for Δ = B − A."""
        if snap is None or snap.path_xyz is None:
            return None
        p = np.asarray(snap.path_xyz, dtype=float)
        if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 3:
            return None
        n_src = int(p.shape[0])
        src_t = np.linspace(0.0, 1.0, n_src)
        grid = np.linspace(0.0, 1.0, max(int(n), 2))
        return np.column_stack(
            [np.interp(grid, src_t, p[:, i]) for i in range(3)]
        )

    def _redraw_paths(self) -> None:
        if not self._graphs_built or not hasattr(self, "ax_path"):
            self._paths_dirty = True
            return
        ready, message = self._both_sessions_chart_ready()
        if not ready:
            self._path_markers = {}
            self._chart_unavailable_message(
                self.ax_path,
                self.fig_path,
                self.canvas_path,
                f"{message}\n\nDifference trajectory needs Session A and Session B.",
            )
            return

        cache = ensure_compare_cache(self.gui)
        left, right = self._session_pair()
        # Never draw a single-session path labeled as A vs B, and never reuse A
        # trajectory data as a fallback for B.
        if (
            left is None
            or right is None
            or left.path_xyz is None
            or right.path_xyz is None
        ):
            self._path_markers = {}
            which = "Session B" if (right is None or right.path_xyz is None) else "Session A"
            self._chart_unavailable_message(
                self.ax_path,
                self.fig_path,
                self.canvas_path,
                f"Comparison unavailable — {which} COM path is incomplete.",
            )
            return

        show_a = bool(self._show_path_left.get())
        show_b = bool(self._show_path_right.get())
        delta_var = getattr(self, "_show_path_delta", None)
        show_delta = bool(delta_var.get()) if delta_var is not None else True
        if not show_a and not show_b and not show_delta:
            self._path_markers = {}
            self._chart_unavailable_message(
                self.ax_path,
                self.fig_path,
                self.canvas_path,
                "Enable Show A, Show B, or Show Δ to display trajectories.",
            )
            return

        self.ax_path.cla()
        setup_3d_axes(self.ax_path)
        self._path_markers = {}
        drawn_arrays: list[np.ndarray] = []
        alias_blocked = False
        for slot, snap, show, color, name in (
            ("A", left, show_a, COLOR_LEFT, "Session A"),
            ("B", right, show_b, COLOR_RIGHT, "Session B"),
        ):
            if not show or snap is None or snap.path_xyz is None:
                continue
            p = np.asarray(snap.path_xyz, dtype=float)
            if p.ndim != 2 or p.shape[0] == 0:
                continue
            # Guard: never plot Session A's trajectory as a fallback for B.
            if slot == "B" and left.path_xyz is not None and snap.path_xyz is left.path_xyz:
                alias_blocked = True
                continue
            drawn_arrays.append(p)
            label = f"{name} — {snap.label}" if snap.label else name
            self.ax_path.plot(
                p[:, 0],
                p[:, 1],
                p[:, 2],
                color=color,
                linewidth=1.6,
                alpha=0.85,
                label=label,
            )
            i = min(self._index_for(snap), len(p) - 1)
            marker, = self.ax_path.plot(
                [p[i, 0]],
                [p[i, 1]],
                [p[i, 2]],
                marker="o",
                color=color,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.0,
                linestyle="None",
                zorder=30,
                label="_nolegend_",
            )
            self._path_markers[slot] = (marker, p, "frame")

        if show_delta and not alias_blocked:
            pa = self._path_xyz_on_progress_grid(left)
            pb = self._path_xyz_on_progress_grid(right)
            if pa is not None and pb is not None and pa.shape == pb.shape:
                d = pb - pa
                drawn_arrays.append(d)
                self.ax_path.plot(
                    d[:, 0],
                    d[:, 1],
                    d[:, 2],
                    color=VIZ_WARNING,
                    linewidth=2.4,
                    label="Δ COM (B−A)",
                    zorder=12,
                )
                i = int(round(float(self._progress) * max(len(d) - 1, 0)))
                i = max(0, min(i, len(d) - 1))
                marker, = self.ax_path.plot(
                    [d[i, 0]],
                    [d[i, 1]],
                    [d[i, 2]],
                    marker="o",
                    color=VIZ_WARNING,
                    markersize=9,
                    markeredgecolor="white",
                    markeredgewidth=1.2,
                    linestyle="None",
                    zorder=35,
                    label="_nolegend_",
                )
                self._path_markers["Δ"] = (marker, d, "progress")

        if alias_blocked or not drawn_arrays:
            self._path_markers = {}
            self._chart_unavailable_message(
                self.ax_path,
                self.fig_path,
                self.canvas_path,
                "Comparison unavailable — Session B COM path must be independent of Session A."
                if alias_blocked
                else "COM path unavailable for one or both sessions.",
            )
            return
        ab_drawn = sum(1 for s in ("A", "B") if s in self._path_markers)
        if show_a and show_b and ab_drawn < 2 and "Δ" not in self._path_markers:
            self._path_markers = {}
            self._chart_unavailable_message(
                self.ax_path,
                self.fig_path,
                self.canvas_path,
                "COM path unavailable for one or both sessions.",
            )
            return
        self._fit_compare_path_axes(drawn_arrays)
        self.ax_path.set_title(
            "Difference trajectory — Δ COM (B−A) with A/B overlays",
            color=TEXT,
            fontsize=11,
        )
        try:
            from stablewalk.ui.viewers.chart_style import style_3d_axis_labels

            style_3d_axis_labels(self.ax_path)
        except Exception:
            pass
        if self.ax_path.get_legend_handles_labels()[0]:
            try:
                from stablewalk.ui.viewers.chart_style import style_chart_legend

                style_chart_legend(
                    self.ax_path,
                    outside=False,
                    loc="upper right",
                    fontsize=9,
                    ncol=1,
                )
            except Exception:
                pass
        self._safe_draw(self.canvas_path)

    def _fit_compare_path_axes(self, arrays: list[np.ndarray]) -> None:
        """Center, equal-scale, and fit the 3D path cube to the drawn paths.

        Equal cubic limits around the data centroid (1:1:1 box aspect) keep the
        two paths comparable without distortion, and the tight span-based extent
        fills the panel instead of floating in an oversized grid.
        """
        pts = [p for p in arrays if p is not None and getattr(p, "size", 0)]
        if not pts:
            return
        allp = np.concatenate(pts, axis=0)
        mins = allp.min(axis=0)
        maxs = allp.max(axis=0)
        center = (mins + maxs) * 0.5
        span = float(np.max(maxs - mins))
        if not np.isfinite(span) or span <= 1e-6:
            span = 0.2
        half = span * 0.58  # equal cube half-extent + ~15% breathing room
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        self.ax_path.set_xlim(cx - half, cx + half)
        self.ax_path.set_ylim(cy - half, cy + half)
        self.ax_path.set_zlim(cz - half, cz + half)
        try:
            self.ax_path.set_box_aspect((1, 1, 1))
        except (AttributeError, ValueError, TypeError):
            pass

    def _update_path_markers(self) -> None:
        """Move the current-position marker on each 3D path (no rebuild)."""
        if not self._graphs_built:
            return
        markers = getattr(self, "_path_markers", None)
        if not markers:
            return
        cache = ensure_compare_cache(self.gui)
        left, right = self._session_pair()
        snaps = {"A": left, "B": right}
        moved = False
        for slot, payload in markers.items():
            if len(payload) == 3:
                marker, p, mode = payload
            else:
                marker, p = payload[0], payload[1]
                mode = "frame"
            if p is None or getattr(p, "shape", (0,))[0] == 0:
                continue
            if mode == "progress" or slot == "Δ":
                i = int(round(float(self._progress) * max(len(p) - 1, 0)))
                i = max(0, min(i, len(p) - 1))
            else:
                snap = snaps.get(slot)
                if snap is None:
                    continue
                i = min(self._index_for(snap), len(p) - 1)
            try:
                marker.set_data_3d([p[i, 0]], [p[i, 1]], [p[i, 2]])
                moved = True
            except Exception:
                pass
        if moved:
            self._safe_draw(self.canvas_path)

    def _redraw_chart_series(self) -> None:
        if not self._lite_chart_built or not hasattr(self, "ax_chart"):
            self._chart_dirty = True
            return
        ready, message = self._both_sessions_chart_ready()
        cache = ensure_compare_cache(self.gui)
        left, right = self._session_pair()
        curves_ok, curves_reason = joint_curves_ready(left, right)
        if not ready or not curves_ok:
            self._chart_playhead = None
            self._chart_unavailable_message(
                self.ax_chart,
                self.fig_chart,
                self.canvas_chart,
                (curves_reason if not curves_ok else message)
                + "\n\nDifference graph needs valid knee trajectories on both sides.",
            )
            return

        assert left is not None and right is not None

        mode = self._compare_axis_mode_for(left, right)
        self._compare_axis_mode = mode
        xlabel = self._axis_xlabel(mode)

        self.ax_chart.cla()
        self._style_chart_ax()

        from stablewalk.ui.viewers.chart_style import (
            AXIS_LABEL_SIZE,
            TICK_LABEL_SIZE,
            series_plot_kwargs,
            style_chart_legend,
            style_chart_title,
        )

        plotted: list[tuple[np.ndarray, np.ndarray, str]] = []
        for snap, color, fallback in (
            (left, COLOR_LEFT, "Session A"),
            (right, COLOR_RIGHT, "Session B"),
        ):
            xy = self._normalized_knee_xy(snap, mode=mode)
            if xy is None:
                continue
            xn, y = xy
            label = f"{fallback} — {snap.label}" if snap.label else fallback
            # Thin overlays — primary signal is Δ below.
            self.ax_chart.plot(
                xn,
                y,
                **series_plot_kwargs(
                    color=color,
                    label=label,
                    linewidth=1.35,
                    alpha=0.42,
                ),
            )
            plotted.append((xn, y, color))

        if len(plotted) < 2:
            self._chart_playhead = None
            self._chart_unavailable_message(
                self.ax_chart,
                self.fig_chart,
                self.canvas_chart,
                "Comparison unavailable — Session B is incomplete."
                if self._normalized_knee_xy(right, mode=mode) is None
                else "Knee trajectories unavailable for one or both sessions.",
            )
            return

        t0, y0, _c0 = plotted[0]
        t1, y1, _c1 = plotted[1]
        grid = np.linspace(0.0, 100.0, 101)
        ya = np.interp(grid, t0, y0, left=np.nan, right=np.nan)
        yb = np.interp(grid, t1, y1, left=np.nan, right=np.nan)
        mask = np.isfinite(ya) & np.isfinite(yb)
        delta_y = yb - ya
        if np.any(mask):
            self.ax_chart.axhline(
                0.0, color=MUTED, linewidth=0.9, alpha=0.55, zorder=2
            )
            self.ax_chart.fill_between(
                grid[mask],
                0.0,
                delta_y[mask],
                color=VIZ_WARNING,
                alpha=0.18,
                linewidth=0,
                zorder=3,
                label="_nolegend_",
            )
            self.ax_chart.plot(
                grid[mask],
                delta_y[mask],
                **series_plot_kwargs(
                    color=VIZ_WARNING,
                    label="Δ (B−A)",
                    linewidth=2.35,
                    zorder=6,
                ),
            )

        title = (
            "Knee flexion difference Δ = B−A (gait-cycle %)"
            if mode == _COMPARE_AXIS_GAIT
            else "Knee flexion difference Δ = B−A (video progress %)"
        )
        style_chart_title(self.ax_chart, title, pad=6)
        self.ax_chart.set_xlabel(xlabel, color=MUTED, fontsize=AXIS_LABEL_SIZE)
        self.ax_chart.set_ylabel(
            "Knee / Δ (°)", color=MUTED, fontsize=AXIS_LABEL_SIZE
        )
        self.ax_chart.set_xlim(0.0, 100.0)
        self.ax_chart.tick_params(labelsize=TICK_LABEL_SIZE)
        style_chart_legend(
            self.ax_chart,
            outside=False,
            loc="upper right",
            ncol=1,
            fontsize=9,
        )
        # Keep legend inside the axes so the compact Compare panel stays readable.
        try:
            self.fig_chart.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.16)
        except Exception:
            pass

        from stablewalk.ui.viewers.chart_hover import ChartHoverPoint, append_line_hover_points
        from stablewalk.ui.viewers.chart_interactions import finalize_chart_interactions

        hover_points: list[ChartHoverPoint] = []
        max_pts = max(8, int(self._MAX_CHART_HOVER_POINTS))
        for snap, _color, fallback in (
            (left, COLOR_LEFT, "Session A"),
            (right, COLOR_RIGHT, "Session B"),
        ):
            xy = self._normalized_knee_xy(snap, mode=mode)
            if xy is None:
                continue
            xn, y = xy
            step = max(1, int(np.ceil(len(xn) / max_pts)))
            append_line_hover_points(
                self.ax_chart,
                xn[::step],
                y[::step],
                metric_name="Knee flexion",
                joint_name=snap.label or fallback,
                unit="deg",
                timestamps=xn[::step],
                hover_points=hover_points,
            )
        if np.any(mask):
            step = max(1, int(np.ceil(np.count_nonzero(mask) / max_pts)))
            g = grid[mask][::step]
            dy = delta_y[mask][::step]
            append_line_hover_points(
                self.ax_chart,
                g,
                dy,
                metric_name="Δ Knee flexion",
                joint_name="B−A",
                unit="deg",
                timestamps=g,
                hover_points=hover_points,
            )
        self._chart_playhead = self.ax_chart.axvline(
            self._progress * 100.0,
            color=TEXT,
            linewidth=1.6,
            alpha=0.9,
            linestyle="-",
            zorder=20,
        )
        finalize_chart_interactions(self.fig_chart, self.canvas_chart, hover_points=hover_points)
        self._safe_draw(self.canvas_chart)

    def _redraw_paths_and_chart(self) -> None:
        self._ensure_lite_chart_built()
        self._chart_dirty = True
        if self._detailed_comparison:
            self._ensure_detailed_graphs_built()
            self._paths_dirty = self._heat_dirty = True
        self._refresh_visible_chart(force=True)

    def _redraw_heatmap(self) -> None:
        if not self._graphs_built or not hasattr(self, "fig_heat"):
            self._heat_dirty = True
            return
        ready, message = self._both_sessions_chart_ready()
        if not ready:
            self._heat_playhead = None
            self.fig_heat.clf()
            self.ax_heat = self.fig_heat.add_subplot(111)
            self._chart_unavailable_message(
                self.ax_heat,
                self.fig_heat,
                self.canvas_heat,
                f"{message}\n\nDifference heatmap needs both valid sessions.",
            )
            return

        cache = ensure_compare_cache(self.gui)
        left, right = self._session_pair()
        mode = self._compare_axis_mode_for(left, right)
        xlabel = self._axis_xlabel(mode)
        try:
            self.fig_heat.clf()
            self.ax_heat = self.fig_heat.add_subplot(111)
            draw_difference_heatmap(
                self.ax_heat,
                left.sequence if left else None,
                right.sequence if right else None,
                progress=None,
                n_bins=self._heat_bins,
                xlabel=xlabel,
            )
            x = float(np.clip(self._progress, 0.0, 1.0)) * max(self._heat_bins - 1, 1)
            self._heat_playhead = self.ax_heat.axvline(
                x, color=TEXT, linewidth=1.6, alpha=0.9, zorder=20
            )
            self._safe_draw(self.canvas_heat)
        except Exception:
            logger.debug("Difference heatmap redraw failed", exc_info=True)
            self._heat_playhead = None

    def _update_heat_playhead(self) -> None:
        """Move the heatmap's vertical cursor to the current normalized progress."""
        if not self._graphs_built:
            return
        ph = getattr(self, "_heat_playhead", None)
        if ph is None:
            return
        try:
            x = float(np.clip(self._progress, 0.0, 1.0)) * max(self._heat_bins - 1, 1)
            ph.set_xdata([x, x])
            self._safe_draw(self.canvas_heat)
        except Exception:
            pass

    def _update_chart_playhead(self) -> None:
        if not self._lite_chart_built:
            return
        if self._chart_playhead is not None:
            try:
                x = float(self._progress) * 100.0
                self._chart_playhead.set_xdata([x, x])
                self._safe_draw(self.canvas_chart)
            except Exception:
                pass

    def _update_progress_readout(self) -> None:
        """Show the shared 0–100% progress that drives every Compare panel."""
        lbl = getattr(self, "lbl_progress", None)
        if lbl is None:
            return
        mode = getattr(self, "_compare_axis_mode", _COMPARE_AXIS_VIDEO)
        if mode == _COMPARE_AXIS_GAIT:
            prefix = "Gait cycle"
        else:
            prefix = "Video progress"
        try:
            lbl.configure(text=f"{prefix} {self._progress * 100:.0f}%")
        except tk.TclError:
            pass

    def _style_chart_ax(self) -> None:
        from stablewalk.ui.viewers.chart_style import apply_chart_grid, apply_chart_panel_style

        apply_chart_panel_style(self.ax_chart)
        self.fig_chart.patch.set_facecolor(PANEL)
        self.ax_chart.tick_params(colors=MUTED, labelsize=7)
        apply_chart_grid(self.ax_chart, y_minor=True)
        for spine in self.ax_chart.spines.values():
            spine.set_color(BORDER)

    def _on_chart_hover(self, event) -> None:
        if event.inaxes != self.ax_chart or event.xdata is None or event.ydata is None:
            if self._hover_annot.get_visible():
                self._hover_annot.set_visible(False)
                self._safe_draw(self.canvas_chart)
            return
        self._hover_annot.xy = (event.xdata, event.ydata)
        self._hover_annot.set_text(
            f"{event.xdata:.0f}%\nθ / Δ={event.ydata:.1f}°"
        )
        self._hover_annot.set_visible(True)
        self._safe_draw(self.canvas_chart)

    def _update_time_label(
        self,
        left: AnalyzedSessionSnapshot | None,
        right: AnalyzedSessionSnapshot | None,
        ia: int,
        ib: int,
    ) -> None:
        def _side(
            prefix: str,
            snap: AnalyzedSessionSnapshot | None,
            index: int,
        ) -> str:
            if snap is None or int(getattr(snap, "n_frames", 0) or 0) <= 0:
                return f"{prefix} F—/— · Unavailable"
            total = int(snap.n_frames)
            i = max(0, min(int(index), total - 1))
            fps = float(getattr(snap, "fps", 0.0) or 0.0)
            time_text = f"{i / fps:.2f} s" if fps > 1e-6 else "Unavailable"
            return f"{prefix} F{i + 1}/{total} · {time_text}"

        self.lbl_time.configure(
            text=f"{_side('A', left, ia)}\n{_side('B', right, ib)}"
        )

    def _fill_metric_card(
        self,
        card: dict[str, Any],
        metrics: CompareMetrics,
        *,
        header_prefix: str = "Session",
        status: str = "ready",
        deltas: dict[str, float | None] | None = None,
    ) -> None:
        from stablewalk.ui.tk.kpi_cards import parse_numeric, split_value_unit, update_kpi_card

        status_labels = {
            "loading": "Loading analysis…",
            "unavailable": "waiting for a finished analysis",
            "incomplete": "Analysis incomplete",
        }
        if status != "ready":
            from stablewalk.ui.unavailable_labels import session_slot_unavailable_label

            title = (
                session_slot_unavailable_label(header_prefix)
                if status == "unavailable"
                else f"{header_prefix} — {status_labels.get(status, 'waiting for data')}"
            )
            card["header"].configure(text=title)
            kpis = card.get("kpis") or {}
            from stablewalk.ui.unavailable_labels import (
                METRIC_FOR_RECORDING,
                WAITING_COMPARISON,
            )

            placeholder = {
                "loading": "Loading…",
                "unavailable": WAITING_COMPARISON,
                "incomplete": METRIC_FOR_RECORDING,
            }.get(status, METRIC_FOR_RECORDING)
            for name, kpi in kpis.items():
                update_kpi_card(
                    kpi,
                    value=placeholder,
                    unit="",
                    available=False,
                    fraction=None,
                    numeric=None,
                    quality="unavailable",
                    tooltip=f"{header_prefix}: {title}",
                )
            for name, lbl in (card.get("fields") or {}).items():
                if name in kpis:
                    continue
                try:
                    lbl.configure(text=placeholder, fg=MUTED)
                except tk.TclError:
                    pass
            return

        title = metrics.label or header_prefix
        card["header"].configure(text=f"{header_prefix} — {title}")
        kpis = card.get("kpis") or {}
        for name, val in metrics.display_map().items():
            kpi = kpis.get(name)
            if kpi is not None:
                value, unit = split_value_unit(str(val))
                available = str(val).strip() not in ("", "—", "-", "N/A")
                numeric = parse_numeric(val) if available else None
                fraction = None
                if numeric is not None and name in ("Gait Quality", "Symmetry"):
                    fraction = min(1.0, numeric / 100.0)
                elif numeric is not None and name == "Cadence":
                    fraction = min(1.0, numeric / 180.0)
                delta = None if deltas is None else deltas.get(name)
                unit_txt = unit
                tip = f"{name}: {val}"
                if delta is not None and available:
                    # Δ = B − A only when both sides contributed a numeric value.
                    sign = "+" if delta >= 0 else ""
                    if abs(delta) >= 100:
                        d_body = f"{delta:.0f}"
                    elif abs(delta) >= 10:
                        d_body = f"{delta:.1f}"
                    else:
                        d_body = f"{delta:.2f}"
                    unit_txt = f"{unit}  Δ={sign}{d_body}".strip()
                    tip = f"{name}: {val}  (Δ = B − A = {sign}{d_body})"
                na_tip = f"{name}: unavailable for this session"
                update_kpi_card(
                    kpi,
                    value=value if available else "—",
                    unit=unit_txt if available else "",
                    available=available,
                    fraction=fraction,
                    numeric=numeric,
                    quality="neutral" if available else "unavailable",
                    tooltip=tip if available else na_tip,
                )
                continue
            lbl = card["fields"].get(name)
            if lbl is not None:
                text = str(val)
                if text.strip() in ("", "—", "-", "N/A"):
                    text = "—"
                delta = None if deltas is None else deltas.get(name)
                if delta is not None and text != "—":
                    sign = "+" if delta >= 0 else ""
                    text = f"{val}  Δ={sign}{delta:.2g}"
                lbl.configure(text=text, fg=TEXT if text != "—" else MUTED)

    def _fill_diffs(
        self,
        result: SessionComparisonResult | None,
        *,
        identical: bool = False,
    ) -> None:
        from stablewalk.ui.tk.compare_delta_visuals import fill_visual_diffs

        cache = ensure_compare_cache(self.gui)
        left, right = self._session_pair()
        both_ready = bool(
            result is not None
            and left is not None
            and right is not None
            and left is not right
        )
        # Static Compare: skip sparkline (avoids after_idle paint churn every refresh).
        if getattr(self, "_static_compare_only", True):
            series_a = None
            series_b = None
            spark = None
        else:
            series_a = getattr(left, "knee_y", None) if both_ready else None
            series_b = getattr(right, "knee_y", None) if both_ready else None
            spark = getattr(self, "_diff_spark_canvas", None)
        fill_visual_diffs(
            self.diff_host,
            result,
            identical=identical,
            priority=_DIFF_PRIORITY,
            max_rows=8,
            series_a=series_a,
            series_b=series_b,
            spark_canvas=spark,
            bg=SURFACE,
        )

    def _set_interp(self, text: str) -> None:
        self.txt_interp.configure(state=tk.NORMAL)
        self.txt_interp.delete("1.0", tk.END)
        self.txt_interp.insert("1.0", text)
        self.txt_interp.configure(state=tk.DISABLED)


__all__ = ["ComparisonModeController", "build_comparison_tab"]
