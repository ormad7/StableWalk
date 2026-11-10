"""
StableWalk unified dashboard — full gait analysis pipeline in one UI.

Video → 2D pose → DOF → JSON → 3D skeleton → robot simulation

Launch:
  python -m stablewalk.ui.tk.app
  python main.py --gui
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from stablewalk import config
from stablewalk.pose.dof import DOF_LABELS, GAIT_ANGLE_FIELDS, GAIT_VELOCITY_JOINTS, LIMB_GROUP_LABELS, limb_group_keys
from stablewalk.pose.kinematics import velocity_between_frames
from stablewalk.models.pose_data import PoseFrame, PoseSequence
from stablewalk.storage.collector import BilateralFootCollector, SessionKinematicCollector
from stablewalk.storage.service import SessionStorageService
from stablewalk.core.pipeline_reset import release_all_captures
from stablewalk.pose.skeleton import FrameRgbCache, render_frame_with_skeleton
from stablewalk.pose.video_source import (
    check_source_format,
    content_cache_key,
    derive_run_name,
    normalize_source,
    quick_validate_source,
    resolve_validate_mode,
)
from stablewalk.pose.video import is_video_url
from stablewalk.pose.skeleton_3d import skeleton_from_frame_data
from stablewalk.io.pose_loader import (
    load_pose_sequence,
    playback_frame_indices,
    sequence_needs_enrichment,
)
from stablewalk.pose.interpolation import (
    interpolate_skeleton_3d,
    normalize_skeleton_height,
    smooth_skeleton_temporal,
)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stablewalk.analysis.biomech_stability import StabilityResult
    from stablewalk.analysis.biomechanical import BiomechanicalAnalysisResult
    from stablewalk.analysis.estimated_vgrf_analysis import EstimatedVGRFResult
    from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult
    from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult
    from stablewalk.analysis.stability import StabilityReport
    from stablewalk.analysis.virtual_grf import VirtualForceResult
    from stablewalk.analysis.comparison import GaitComparison
    from stablewalk.core.pipeline import PipelineResult
else:
    # Concrete analysis types load on first analysis call (keeps GUI import light).
    StabilityResult = object  # type: ignore[misc, assignment]
    BiomechanicalAnalysisResult = object  # type: ignore[misc, assignment]
    EstimatedVGRFResult = object  # type: ignore[misc, assignment]
    FootContactAnalysisResult = object  # type: ignore[misc, assignment]
    GaitCycleAnalysisResult = object  # type: ignore[misc, assignment]
    StabilityReport = object  # type: ignore[misc, assignment]
    VirtualForceResult = object  # type: ignore[misc, assignment]
    GaitComparison = object  # type: ignore[misc, assignment]
    PipelineResult = object  # type: ignore[misc, assignment]
from stablewalk.pose.skeleton_3d import Skeleton3D
from stablewalk.ui.motion_display import (
    FrameMotionSnapshot,
    _collect_joint_positions,
    build_frame_motion_snapshot,
    compute_sequence_rom,
)
from stablewalk.ui.theme import POS_PANEL_WIDTH
from stablewalk.ui.viewers.plot_3d import (
    DISPLAY_BODY_SCALE,
    compute_view_limit,
    draw_robot_geometry,
    orient_skeleton_for_display,
    scale_skeleton_uniform,
    setup_3d_axes,
)
from stablewalk.simulation.kinematic import WalkSimulation, WalkSimulator
from stablewalk.ui.demo import (
    DEMO_FINAL_HOLD_MS,
    DEMO_MAX_FRAMES,
    DEMO_PAUSE_MS,
    DEMO_PLAYBACK_MS,
    get_demo_clips,
)
from stablewalk.models.motion_series import MotionSeriesIndex
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.ui.presentation_demo import (
    DEFAULT_DEMO_DOF_IDS,
    PRESENTATION_GAIT_HINT,
    PRESENTATION_SESSION_SUMMARY,
    PRESENTATION_STABILITY_LABEL,
    PRESENTATION_VIDEO_CAPTION,
    generate_presentation_recording,
)
from stablewalk.ui.dof_step_preview import (
    STEP_PREVIEW_COLUMNS,
    STEP_PREVIEW_HEADINGS,
    STEP_PREVIEW_WIDTHS,
    StepConfig,
    compute_dof_step_previews,
    preview_table_rows,
)
from stablewalk.ui.dof_position_table import (
    DOF_TABLE_COLUMNS,
    DOF_TABLE_HEADINGS,
    DOF_TABLE_MODE_CURRENT,
    DOF_TABLE_MODE_HISTORY,
    DOF_TABLE_WIDTHS,
    DofPositionTableHistory,
    snapshot_for_next_frame,
    table_row_for_item,
    table_rows_for_selection,
    table_display_columns_for_item,
)
from stablewalk.io.tracking_export import export_tracking_bundle
from stablewalk.ui.dof_selection import anchor_joint_for_item
from stablewalk.ui.scientific_labels import (
    DISCLAIMER_MONOCULAR,
    DISCLAIMER_RESEARCH,
    LABEL_CONTACT_CONFIDENCE,
    LABEL_CONTACT_PATTERN,
    LABEL_FOOT_CLEARANCE_CONFIDENCE,
    LABEL_GAIT_QUALITY,
    LABEL_VIRTUAL_GRF_STATUS_PREFIX,
    LABEL_VIRTUAL_GRF_UNAVAILABLE_MSG,
    METHODS_OVERVIEW,
    UNIT_CADENCE,
    UNITS_COORD_LEGEND,
)
from stablewalk.ui.selection_state import (
    GUI_DOF_ITEM_IDS,
    GUI_DOF_LABELS,
    SelectionState,
    item_for_joint,
    label_for_item,
)
from stablewalk.models.joint_registry import JOINT_DISPLAY_NAMES
from stablewalk.ui.skeleton_player import SkeletonPlayer
from stablewalk.ui.viewers.dof_trajectory_3d import TRAJECTORY_COLORS
from stablewalk.ui.viewers.gait_skeleton_renderer import (
    DEFAULT_SKELETON_DISPLAY_MODE,
    LABEL_TO_SKELETON_MODE,
    compute_skeleton_view_box,
    draw_gait_skeleton,
    joint_id_from_pick,
    nearest_joint_from_event,
    setup_skeleton_axes,
)
from stablewalk.ui.media.presets import (
    CUSTOM_URL_PRESET_LABEL,
    MIN_DETECTED_POSE_FRAMES,
    STABLE_WALKING_PRESETS,
    all_preset_labels,
    list_runnable_presets,
    preset_by_label,
    preset_from_custom_url,
)
from stablewalk.ui.media.demo_gait import (
    DEMO_GAIT_EXAMPLES,
    DemoGaitExample,
    demo_category_tagline,
    demo_default_dof_item,
    demo_gait_interpretation,
    demo_exists,
    demo_path,
    demo_validation_status,
    example_by_key,
    format_about_active_example,
    is_demo_video_path,
    missing_file_message,
    missing_file_placeholder,
)
from stablewalk.ui.media.demo_comparison import (
    COMPUTED_PENDING,
    DOCUMENTED_UNAVAILABLE,
    DemoGaitComparisonRow,
    build_comparison_row,
    comparison_tree_values,
    documented_camera_view,
    documented_locomotion,
    documented_placeholder_row,
    documented_source_label,
    example_comparison_order,
    format_duration_display,
    format_usable_cycles_display,
)
from stablewalk.ui.theme import (
    ACCENT,
    ACCENT_ALT,
    BG,
    BORDER,
    BORDER_SUBTLE,
    ELEVATED,
    EMPTY_OVERVIEW_JOINT_INSPECT,
    EMPTY_SELECT_DOF_CHART,
    EMPTY_VIDEO_TEXT,
    INFO,
    MUTED,
    DANGER,
    DASHBOARD_CARD_PAD,
    DASHBOARD_SUBTITLE,
    ORANGE,
    SUCCESS,
    PANEL,
    PANEL_HOVER,
    PAD_LG,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    REFRESH_INTERVAL_CHOICES,
    REFRESH_INTERVAL_DEFAULT,
    SIDEBAR_WIDTH,
    SURFACE,
    TEXT,
    TEXT_SECONDARY,
    WARNING,
    apply_theme,
    configure_demo_overlay,
    configure_video_placeholder,
    create_tooltip,
    format_labelframe_title,
    format_stability_short,
    FONT_CAPTION,
    FONT_DISPLAY,
    FONT_TITLE,
    FONT_UI,
    FONT_METRIC,
    FONT_MONO,
    FONT_MONO_SM,
    FONT_UI_SEMIBOLD,
    FONT_UI_SM,
    HEADER_HEIGHT,
    FONT_UI_XS,
    menu_colors,
)

logger = logging.getLogger(__name__)

# OpenSim export panel — user-facing status strings
_OPENSIM_NO_SESSION_MSG = (
    "No video/session loaded. Please load and analyze a video before exporting OpenSim files."
)
_OPENSIM_POSE_JSON_MISSING_MSG = "Cannot export OpenSim files: pose JSON is missing."
_OPENSIM_NO_VALID_FRAMES_MSG = (
    "Cannot export OpenSim files: no valid pose frames found."
)

PipelineCallback = Callable[[Path | None, Exception | None], None]


class StableWalkGUI:
    """Tkinter application for skeleton overlay playback and gait metrics."""

    def __init__(
        self,
        poses_path: str | Path | None = None,
        *,
        root: tk.Tk | None = None,
    ) -> None:
        _startup_t0 = time.perf_counter()
        self.root = root or tk.Tk()
        self.root.title("StableWalk — Gait Analysis Dashboard")
        # Layout helpers resolve the GUI from Configure handlers via the root.
        self.root._stablewalk_gui = self  # type: ignore[attr-defined]
        from stablewalk.ui.tk.dashboard_responsive import (
            MIN_WINDOW_HEIGHT,
            MIN_WINDOW_WIDTH,
            initial_window_geometry,
        )
        from stablewalk.ui.tk.layout_scale import apply_tk_dpi_scaling

        self._layout_dpi_scale = apply_tk_dpi_scaling(self.root)
        self.root.geometry(initial_window_geometry(self.root))
        self.root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.root.configure(bg=BG)

        self.sequence: PoseSequence | None = None
        self.pose_indices: list[int] = []
        self.current_pos = 0
        self.playing = False
        self.play_speed = 1.0
        self.show_skeleton = tk.BooleanVar(value=True)
        self.highlight_dof = tk.BooleanVar(value=True)
        self.selection = SelectionState()
        self._dof_list_syncing = False
        self._legacy_tree_syncing = False
        self._active_joint: str | None = None
        self._hover_joint: str | None = None
        self._motion_series: MotionSeriesIndex | None = None
        self.gait_motion: GaitMotionRecording | None = None
        self.gait_wireframe = None
        self._gait_wire_body_scale = 1.0
        self.skeleton_player: SkeletonPlayer | None = None
        self._data_refresh_s = 0.5
        self._last_data_refresh = 0.0
        self._last_realtime_refresh = 0.0
        self._last_dof_table_refresh = 0.0
        self._dof_table_history = DofPositionTableHistory()
        self._session_collector = SessionKinematicCollector()
        self._bilateral_foot_collector = BilateralFootCollector()
        self._ground_clearance_prev_phase: tuple[str | None, str | None] = (None, None)
        from stablewalk.ui.events import UIEventBus
        from stablewalk.ui.session_analysis_cache import SessionAnalysisCache

        # Event-driven architecture: analysis fills the cache once; UI reads it.
        self.events = UIEventBus()
        self._session_analysis = SessionAnalysisCache()
        self._analysis_compute_active = False
        self._analysis_worker: threading.Thread | None = None
        self._opensim_worker: threading.Thread | None = None
        self._advanced_export_worker: threading.Thread | None = None
        self._overview_video_frame_index: int | None = None
        self._foot_clearance_frame_index: int | None = None
        self._foot_clearance_live_cache = None
        self._gait_contact_frame_index: int | None = None
        self._gait_phase_frame_index: int | None = None
        self._session_storage = SessionStorageService()
        self._session_save_in_progress = False
        self._dof_table_track_history = True
        self._dof_table_user_prefers_current_only = False
        self._dof_step_previews: list = []
        self._checkpoints: list[tuple[str, ...]] = []
        self._photo: ImageTk.PhotoImage | None = None
        self._last_video_rgb: np.ndarray | None = None
        self._video_aspect: float | None = None
        self._video_label_box: tuple[int, int] | None = None
        self._video_photo_key: tuple | None = None
        self._video_resize_after: str | None = None
        self._after_id: str | None = None
        self._processing = False
        self._analysis_running = False
        self._pipeline_running = False
        self._overview_clinical_generation = None
        self._overview_clinical_ready = False
        self._poses_path: Path | None = None
        self._skeleton_scale: float = 1.0
        self._view_limit_3d: float = 0.55
        self._rom_cache: dict[str, tuple[float, float]] = {}
        self._3d_motion_trail: list[Skeleton3D] = []
        # Number of previous poses rendered as a fading motion echo behind the
        # live 3D skeleton (0 disables the effect). Full count when paused;
        # disabled while playing so each tick only moves pooled limb artists.
        self._skeleton_ghost_count = 12
        self._skeleton_ghost_count_play = 0
        self._anim_hz = 60
        self._walk_simulation: WalkSimulation | None = None
        self._skeleton_cache: dict[int, Skeleton3D] = {}
        # Larger RGB window reuses processed frames across tab switches / scrub.
        self._rgb_cache = FrameRgbCache(max_entries=96)
        self._opensim_auto_pending = False
        # Composed video+landmark overlays keyed by frame + draw signature.
        self._landmark_overlay_cache: dict[tuple, np.ndarray] = {}
        self._landmark_overlay_cache_max = 48
        self._panel_tick = 0
        # Dirty analytical panels deferred until their tab is visible.
        self._analysis_panel_dirty: set[str] = set()
        self._dof_table_display_dirty = False
        # Virtual window for long DOF history Treeviews (tail only).
        self._dof_table_virtual_window = 80
        self._traj_full_rebuild_cooldown_until = 0.0
        # Shared playback clock targeting ~60 UI Hz (spend-compensated).
        # Video decode still runs only when the discrete frame index changes.
        from stablewalk.ui.tk.playback_render_policy import TARGET_PLAY_HZ

        self._video_anim_hz = int(TARGET_PLAY_HZ)
        self._play_anim_hz = int(TARGET_PLAY_HZ)
        self._ui_anim_hz = int(TARGET_PLAY_HZ)
        self._ui_sync_interval_s = 0.0
        self._last_ui_sync_ts = 0.0
        # Tip / inspector follow every discrete frame change (dirty-gated).
        self._traj_redraw_interval_s = 0.0
        self._last_traj_redraw_ts = 0.0
        self._last_traj_tip_frame: int | None = None
        self._last_traj_tip_item: str | None = None
        self._last_positions: list[tuple[str, float, float, float]] = []
        self._3d_flush_scheduled = False
        self._pending_3d_args: tuple | None = None
        # Paint skeleton every play tick (reuse pooled limb meshes).
        self._3d_play_stride = 1
        self._skel_play_tick = 0
        self._skel_ui_budget_ms = 1e9
        self._skeleton_artists_dirty = True
        self._skeleton_artists_mode: str | None = None
        self._transport_label_cache: tuple[str, str] | None = None
        self._biomech_static_sig: object | None = None
        self._biomech_com_index: dict[int, int] | None = None
        self._sync_all_guard = False
        self._playback_commit_gen = 0
        self._committed_playback = None
        self._committed_playback_gen = 0
        self._live_panel_list_pos: int | None = None
        self._status_bar_list_pos: int | None = None
        self._playback_sync_immediate = False
        # Timeline scrub / step seek — graphs stay playhead-only (no geometry rebuild).
        self._scrubbing = False
        self._rate_limit_ts: dict[str, float] = {}
        self._last_live_widget_ts = 0.0
        # Playback: live surfaces only — graphs/metrics stay frozen until pause.
        self._pb_heavy = True
        self._pb_ui_tick = True
        self._heavy_chart_interval_s = 0.0
        self._last_heavy_chart_ts = 0.0
        # Joint Information / HUD chrome — never per play tick (inspector is separate).
        self._joint_info_play_interval_s = 1e9
        self._last_joint_info_play_ts = 0.0
        self._tick_last_wall: float | None = None
        self._overview_hud_frame_i: int | None = None
        self._skel_fit_size_sig: tuple[int, int] | None = None
        self._overview_media_refit_after: str | None = None
        self._overview_tab_enter_ts: float | None = None
        self._perf_timings: dict[str, list[float]] = {
            "video": [],
            "skeleton": [],
            "3d": [],
            "chart": [],
            "total": [],
        }
        self._last_perf_log_ts = 0.0
        self._overview_traj_draw_cache_key: object | None = None
        self._overview_path_cache: dict[tuple, object] = {}
        self._last_overview_info_text: str | None = None
        # Shared decoded RGB for the current playback frame (one decode / tick).
        self._pb_shared_frame_rgb = None
        self._pb_shared_frame_index: int | None = None
        self._pb_shared_frame_path: object | None = None
        # None => no visibility gating (populate all panels); a frozenset during
        # the playback sync path restricts redraws to the active tab's widgets.
        self._pb_visible: frozenset[str] | None = None
        self._frame_ts_cache: dict[int, float] = {}
        self._frame_ts_cache_key: int | None = None
        self._compare: GaitComparison | None = None
        self._stability: StabilityReport | None = None
        self._biomech: StabilityResult | None = None
        self._gait_cycle: GaitCycleAnalysisResult | None = None
        self._gait_asymmetry_moments = None
        self._selected_asymmetry_moment_index: int | None = None
        self._gait_features = None
        self._foot_contact: FootContactAnalysisResult | None = None
        self._estimated_vgrf: EstimatedVGRFResult | None = None
        self._biomech_analysis: BiomechanicalAnalysisResult | None = None
        self._analysis_summary_cache = None
        self._virtual_grf: VirtualForceResult | None = None
        self._knee_angle_series = None
        self._knee_chart_mode_user_set = False
        self._dof_traj_projection_mode: str | None = None
        self._current_source: str = ""
        self._session_display_src: str = ""
        self._session_id: int = 0
        self._demo_running: bool = False
        self._presentation_mode: bool = False
        self._active_demo_gait: DemoGaitExample | None = None
        self._demo_comparison_rows: dict[str, DemoGaitComparisonRow] = {}
        self._demo_file_duration_s: float | None = None
        self._presentation_after_id: str | None = None
        self._demo_poses_a: Path | None = None
        self._demo_stability_a: StabilityReport | None = None
        self._pipeline_callback: PipelineCallback | None = None
        self._show_success_dialog: bool = False
        self._pending_video_load: dict[str, object] | None = None
        self._fade_step: int = 0
        self._run_metadata: PipelineResult | None = None
        self._active_run_name: str | None = None
        self._opensim_export_completed: bool = False
        self._opensim_status_override: str | None = None
        self._loaded_content_key: str = ""
        self._subject_selection = None
        self._subject_selection_source: str = ""
        self._video_display_map: dict | None = None
        self._preset_cycle_index: int = 0
        self._preset_after_id: str | None = None
        self._chain_next_on_fail: bool = False
        self._next_video_attempts: int = 0
        self.auto_load_on_change = tk.BooleanVar(value=True)
        self.smooth_motion = tk.BooleanVar(value=True)
        self._smooth_motion_cached = True

        self._apply_style()
        self._build_header()
        self._build_input_bar()
        self._build_menu()
        self._build_transport_bar()
        self._build_status_bar()
        self._build_layout()
        self._bind_keys()
        self._attach_tooltips()
        self._apply_refresh_interval(
            self.refresh_var.get() if hasattr(self, "refresh_var") else REFRESH_INTERVAL_DEFAULT
        )
        try:
            from stablewalk.ui.tk.ui_freeze_debug import install_on_gui

            install_on_gui(self)
        except Exception:
            pass
        if getattr(self, "_session_mgr", None) is None:
            from stablewalk.ui.tk.session_manager import install_session_manager

            self._session_mgr = install_session_manager(self)
        else:
            self._session_mgr.start_autosave()
            self._session_mgr._update_window_title()

        # Only auto-load when CLI explicitly passes a pose file (avoid stale session on startup)
        if poses_path is not None:
            initial = self._resolve_initial_path(poses_path)
            if initial:
                self.load_poses(initial, fresh=True)
        else:
            self._show_welcome_state()
        try:
            from stablewalk.ui.tk.perf_diagnostics import get_perf_state

            get_perf_state(self).record(
                "startup", (time.perf_counter() - _startup_t0) * 1000.0
            )
        except Exception:
            pass

    def _show_welcome_state(self) -> None:
        """Clean startup screen for live presentation — no synthetic demo loaded."""
        self._cancel_presentation_autoplay()
        self._cancel_timer()
        self._presentation_mode = False
        self.playing = False
        self.gait_motion = None
        self.skeleton_player = None
        self.sequence = None
        self.pose_indices = []
        self._poses_path = None
        self._active_run_name = None
        self._dof_table_history.clear()
        self._session_collector.clear()
        self._bilateral_foot_collector.clear()
        self._ground_clearance_prev_phase = (None, None)
        self._foot_clearance_live_cache = None
        self.selection.clear()
        self._clear_demo_gait_context()
        if hasattr(self, "_clear_session_analysis_cache"):
            self._clear_session_analysis_cache()
        self._reset_stability_panel()
        self._reset_gait_cycle_panel()
        if hasattr(self, "slider"):
            self.slider.configure(to=0)
            self.frame_var.set(0)
        self._sync_play_buttons()
        self._sync_transport_labels()
        if hasattr(self, "video_label"):
            configure_video_placeholder(self.video_label)
            self.video_label.configure(
                image="",
                text=EMPTY_VIDEO_TEXT,
                fg=MUTED,
                justify=tk.CENTER,
                anchor=tk.CENTER,
                wraplength=420,
            )
        self._video_aspect = None
        from stablewalk.ui.tk.dashboard_layout import apply_top_row_aspect

        apply_top_row_aspect(self, None)
        self._update_dashboard_empty_states()
        # Skip cold 3D skeleton rebuild on empty welcome — no sequence yet.
        if hasattr(self, "ax_3d") and self.skeleton_player is not None:
            self._update_interactive_skeleton(force_draw=True)
        self._update_ground_clearance_visibility()
        self.status.configure(
            text="Ready — load a walking video or open Examples"
        )
    def _cancel_presentation_autoplay(self) -> None:
        if self._presentation_after_id:
            try:
                self.root.after_cancel(self._presentation_after_id)
            except tk.TclError:
                pass
            self._presentation_after_id = None

    def _load_presentation_demo(self) -> None:
        """Load synthetic walk + default DOF selection for professor demo (frame 0, stopped)."""
        self._cancel_presentation_autoplay()
        self._cancel_timer()
        self._presentation_mode = True
        self._active_run_name = None
        self._opensim_export_completed = False
        self._opensim_status_override = None
        self._dof_table_history.clear()
        self._session_collector.clear()
        self._bilateral_foot_collector.clear()
        self._ground_clearance_prev_phase = (None, None)
        self._foot_clearance_live_cache = None

        recording = generate_presentation_recording()
        self._clear_demo_gait_context()
        self._set_gait_motion(recording)
        self._apply_presentation_dof_selection()

        self._session_display_src = "Presentation walk"
        self._update_session_selection_overview()
        if hasattr(self, "lbl_stab_score"):
            self.lbl_stab_score.configure(text="—", fg=ACCENT)
        if hasattr(self, "lbl_stab_category"):
            self.lbl_stab_category.configure(
                text=PRESENTATION_STABILITY_LABEL, foreground=ACCENT
            )
        elif hasattr(self, "lbl_stab_headline"):
            self.lbl_stab_headline.configure(
                text=PRESENTATION_STABILITY_LABEL, foreground=ACCENT
            )
        self.video_label.configure(
            image="",
            text=PRESENTATION_VIDEO_CAPTION,
            fg=MUTED,
            justify=tk.CENTER,
            anchor=tk.CENTER,
            wraplength=360,
        )
        self._video_aspect = None
        from stablewalk.ui.tk.dashboard_layout import apply_top_row_aspect

        apply_top_row_aspect(self, None)
        self.status.configure(text="STOPPED · Presentation demo loaded · press Play")

        self._last_dof_table_refresh = 0.0
        self._last_realtime_refresh = 0.0
        self._update_interactive_skeleton(force_draw=True)
        self._maybe_refresh_dof_table(force=True)
        self._refresh_opensim_status()

    def _reset_presentation_demo(self) -> None:
        """Restore the default professor demo (e.g. after loading other data)."""
        self._cancel_timer()
        self.playing = False
        self._poses_path = None
        self.sequence = None
        self.pose_indices = []
        self._dof_table_history.clear()
        self._session_collector.clear()
        self._bilateral_foot_collector.clear()
        self._ground_clearance_prev_phase = (None, None)
        self._load_presentation_demo()

    def _apply_presentation_dof_selection(self) -> None:
        self.selection.set_selection(set(DEFAULT_DEMO_DOF_IDS))
        self.highlight_dof.set(True)
        self._sync_dof_checkboxes()
        self._notify_dof_selection_changed()

    def _apply_style(self) -> None:
        style = ttk.Style(self.root)
        apply_theme(self.root, style)
        self._style = style
        # Prevent Tk exception reporting from cascading into RecursionError
        # when a Configure feedback loop already exhausted the stack.
        reporting_callback_exception = False

        def _safe_report(exc, val, tb) -> None:  # type: ignore[no-untyped-def]
            nonlocal reporting_callback_exception
            if reporting_callback_exception:
                return
            reporting_callback_exception = True
            try:
                import traceback

                sys.stderr.write("Exception in Tkinter callback\n")
                traceback.print_exception(exc, val, tb)
            except Exception:
                pass
            finally:
                reporting_callback_exception = False

        self.root.report_callback_exception = _safe_report  # type: ignore[method-assign]

    def _build_header(self) -> None:
        self._dashboard_header = ttk.Frame(self.root)
        self._dashboard_header.pack(fill=tk.X, side=tk.TOP)
        header = tk.Frame(self._dashboard_header, bg=SURFACE, height=HEADER_HEIGHT)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        title_row = tk.Frame(header, bg=SURFACE)
        title_row.pack(side=tk.LEFT, padx=PAD_MD, pady=(PAD_SM, PAD_SM))
        tk.Label(
            title_row, text="StableWalk", bg=SURFACE, fg=TEXT, font=FONT_DISPLAY
        ).pack(side=tk.LEFT)
        tk.Label(
            title_row,
            text=DASHBOARD_SUBTITLE,
            bg=SURFACE,
            fg=MUTED,
            font=FONT_UI_SM,
        ).pack(side=tk.LEFT, padx=(PAD_SM, 0))
        # Session / trial placeholder (filled when a recording is loaded).
        self.lbl_session_header = tk.Label(
            header,
            text="",
            bg=SURFACE,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="e",
        )
        self.lbl_session_header.pack(side=tk.RIGHT, padx=PAD_MD)
        tk.Frame(
            self._dashboard_header, bg=BORDER_SUBTLE, height=1, highlightthickness=0
        ).pack(fill=tk.X)

    def _build_menu(self) -> None:
        mc = menu_colors()
        menubar = tk.Menu(self.root, **mc)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0, **mc)
        menubar.add_cascade(label="File", menu=file_menu)
        # Professional session commands are inserted at index 0 by SessionManager.
        file_menu.add_command(label="Open pose JSON…", command=self._menu_open, accelerator="Ctrl+O")
        file_menu.add_command(label="Open output folder", command=self._open_output_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Export JSON (save as…)", command=self._export_json)
        file_menu.add_command(label="Export Analysis…", command=self._export_analysis)
        file_menu.add_command(
            label="Export Research PDF Report…",
            command=self._export_professional_pdf_report,
        )
        file_menu.add_command(label="Export OpenSim (.trc + .mot)…", command=self._export_opensim)
        file_menu.add_command(label="Export Analysis Data…", command=self._export_analysis_data)
        file_menu.add_separator()
        file_menu.add_command(label="Save Session to database…", command=self._save_analysis_session)
        file_menu.add_separator()
        from stablewalk.ui.tk.dashboard_notebook import ENABLE_COMPARE_TAB

        if ENABLE_COMPARE_TAB:
            file_menu.add_command(
                label="Comparison Mode…",
                command=self._open_comparison_mode,
            )
        file_menu.add_command(label="Compare with JSON…", command=self._compare_gait)
        file_menu.add_command(label="Compare with video…", command=self._compare_video)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._request_app_close)

        from stablewalk.ui.tk.session_manager import SessionManager

        # Temporary manager so File → session items exist before full install.
        if not getattr(self, "_session_mgr", None):
            self._session_mgr = SessionManager(self)
        self._session_mgr.install_file_menu_commands(file_menu, mc)

        view_menu = tk.Menu(menubar, tearoff=0, **mc)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(
            label="Show skeleton",
            variable=self.show_skeleton,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Highlight gait DOF joints",
            variable=self.highlight_dof,
            command=self._refresh_display,
        )
        self.show_robot_panel = tk.BooleanVar(value=False)
        view_menu.add_checkbutton(
            label="Show robot simulation panel",
            variable=self.show_robot_panel,
            command=self._toggle_robot_panel,
        )
        view_menu.add_separator()
        view_menu.add_command(
            label="Fit Full Body",
            command=self._fit_full_body,
        )
        view_menu.add_command(
            label="Fit Trajectory",
            command=self._fit_overview_trajectory,
        )
        view_menu.add_command(
            label="3D Coordinate Inspector",
            command=self._open_coordinate_inspector_window,
        )
        view_menu.add_command(
            label="Skeleton Model Inspector",
            command=self._open_skeleton_model_inspector_window,
        )
        view_menu.add_command(
            label="Joint Data…",
            command=self._open_collected_data_dialog,
        )

        help_menu = tk.Menu(menubar, tearoff=0, **mc)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(
            label="Analysis methods…",
            command=self._show_analysis_methods,
            accelerator="F1",
        )
        help_menu.add_command(
            label="Metric Details…",
            command=self._show_metric_details_catalog,
        )
        help_menu.add_command(label="Keyboard shortcuts", command=self._show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About StableWalk", command=self._show_about)

        # Debug menu is opt-in (STABLEWALK_DEBUG=1) — keep release UI clean.
        import os as _os

        self._perf_diagnostics = False
        self._perf_diag_var = tk.BooleanVar(value=False)
        if _os.environ.get("STABLEWALK_DEBUG", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            debug_menu = tk.Menu(menubar, tearoff=0, **mc)
            menubar.add_cascade(label="Debug", menu=debug_menu)
            debug_menu.add_checkbutton(
                label="Performance Diagnostics",
                variable=self._perf_diag_var,
                command=self._debug_toggle_perf_diagnostics,
            )
            debug_menu.add_command(
                label="Performance Diagnostics report…",
                command=self._debug_show_perf_diagnostics,
            )
            debug_menu.add_separator()
            debug_menu.add_command(
                label="Inspect value provenance…",
                command=self._debug_inspect_value_provenance,
            )
            debug_menu.add_command(
                label="Dump memory / GUI state…",
                command=self._debug_dump_gui_memory,
            )
            debug_menu.add_command(
                label="Dump Overview Layout",
                command=self._debug_dump_overview_layout,
            )

        sim_menu = tk.Menu(menubar, tearoff=0, **mc)
        menubar.add_cascade(label="Simulation", menu=sim_menu)
        sim_menu.add_command(
            label="Run OpenSim IK (experimental)…",
            command=self._run_stablewalk_ik_experimental,
        )
        sim_menu.add_command(
            label="Run OpenSim demo IK…",
            command=self._run_opensim_demo_ik,
        )

        demo_menu = tk.Menu(menubar, tearoff=0, **mc)
        menubar.add_cascade(label="Demo", menu=demo_menu)
        demo_menu.add_command(
            label="Reset presentation demo",
            command=self._reset_presentation_demo,
        )
        demo_menu.add_separator()
        for ex in DEMO_GAIT_EXAMPLES:
            demo_menu.add_command(
                label=ex.display_name,
                command=lambda k=ex.key: self._load_demo_gait(k),
            )
        demo_menu.add_separator()
        demo_menu.add_command(label="Full video comparison demo", command=self._run_demo)
        demo_menu.add_command(label="Edit demo URL list…", command=self._open_url_list)

    def _build_input_bar(self) -> None:
        """Compact toolbar: video source, analyze, and demo gait shortcuts."""
        parent = getattr(self, "_dashboard_header", self.root)
        bar = ttk.Frame(parent, padding=(PAD_MD, PAD_SM, PAD_MD, PAD_XS))
        bar.pack(fill=tk.X)

        row = ttk.Frame(bar)
        row.pack(fill=tk.X)
        self._input_bar = bar
        self._input_bar_row = row

        _verified = list_runnable_presets(verified_only=True, urls_only=True)
        _default_label = (
            _verified[0].label
            if _verified
            else (STABLE_WALKING_PRESETS[0].label if STABLE_WALKING_PRESETS else "")
        )
        self.preset_var = tk.StringVar(value=_default_label)
        self.url_var = tk.StringVar(value=config.DEFAULT_WALKING_VIDEO_URL)

        ttk.Label(row, text="Source", style="Card.TLabel").pack(side=tk.LEFT, padx=(0, PAD_XS))
        from stablewalk.ui.tk.layout_scale import combobox_char_width

        self.preset_combo = ttk.Combobox(
            row,
            textvariable=self.preset_var,
            values=all_preset_labels(),
            width=combobox_char_width(
                all_preset_labels() or (_default_label,),
                minimum=24,
                maximum=48,
            ),
            state="readonly",
        )
        self.preset_combo.pack(side=tk.LEFT, padx=(0, PAD_XS))
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)
        create_tooltip(
            self.preset_combo,
            "Video source preset. Full title is in the dropdown list.",
            wraplength=360,
        )

        self.url_entry = ttk.Entry(row, textvariable=self.url_var)
        self.url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, PAD_XS))
        self.url_entry.bind("<Return>", lambda _e: self._load_video())

        actions = ttk.Frame(row)
        actions.pack(side=tk.LEFT)
        self._input_bar_actions = actions

        self.btn_browse = ttk.Button(
            actions,
            text="Browse",
            style="Secondary.TButton",
            width=8,
            command=self._browse_video,
        )
        self.btn_browse.pack(side=tk.LEFT, padx=(0, PAD_XS))

        self.btn_full = ttk.Button(
            actions,
            text="Analyze",
            style="Accent.TButton",
            width=9,
            command=self._run_full_analysis,
        )
        self.btn_full.pack(side=tk.LEFT, padx=(0, PAD_XS))
        self.btn_load = self.btn_full
        create_tooltip(
            self.btn_full,
            "Run full analysis of the current source (every decoded frame). "
            "Examples load a labeled preview only.",
            wraplength=280,
        )

        self.btn_input_settings = ttk.Button(
            actions,
            text="\u2699",
            style="Secondary.TButton",
            width=3,
            command=self._show_input_settings_menu,
        )
        self.btn_input_settings.pack(side=tk.LEFT, padx=(0, PAD_XS))
        create_tooltip(self.btn_input_settings, "Input options (smooth motion, …)")

        # Examples menubutton — demos live here / Demo menu, not as primary chrome.
        self._demo_gait_buttons: dict[str, ttk.Button] = {}
        examples_menu = tk.Menu(row, tearoff=0)
        for ex in DEMO_GAIT_EXAMPLES:
            examples_menu.add_command(
                label=ex.display_name,
                command=lambda k=ex.key: self._load_demo_gait(k),
            )
        examples_menu.add_separator()
        examples_menu.add_command(
            label="About active example…",
            command=self._show_demo_video_details,
        )
        self.btn_examples = ttk.Menubutton(
            actions,
            text="Examples",
            style="Secondary.TButton",
            direction="below",
        )
        self.btn_examples.configure(menu=examples_menu)
        self.btn_examples.pack(side=tk.LEFT, padx=(0, PAD_XS))
        create_tooltip(
            self.btn_examples,
            "Preview a verified gait example (first 900 frames). "
            "Click Analyze for the full source.",
            wraplength=280,
        )
        # Hidden legacy buttons kept for highlight / lock_ui API compatibility.
        for ex in DEMO_GAIT_EXAMPLES:
            btn = ttk.Button(
                row,
                text=ex.button_label,
                style="Secondary.TButton",
                command=lambda k=ex.key: self._load_demo_gait(k),
            )
            btn.pack_forget()
            self._demo_gait_buttons[ex.key] = btn

        self.btn_next_video = ttk.Button(
            row, text="Next ▶", style="Secondary.TButton", command=self._next_video
        )
        self.btn_next_video.pack_forget()
        self.btn_custom_url = ttk.Button(
            row, text="URL…", style="Secondary.TButton", command=self._enter_custom_url
        )
        self.btn_custom_url.pack_forget()

        self.btn_demo = ttk.Button(row, text="", command=self._run_demo)
        self.btn_demo.pack_forget()
        self.btn_export = ttk.Button(row, text="", command=self._export_analysis)
        self.btn_export.pack_forget()

        self.btn_save_demo_comparison = ttk.Button(
            row,
            text="Save",
            style="Compact.TButton",
            width=5,
            command=self._save_demo_comparison,
            state=tk.DISABLED,
        )
        # Save comparison lives on the Compare tab — keep widget for API, hide chrome.
        create_tooltip(
            self.btn_save_demo_comparison,
            "Save this demo row for Compare (use the Compare tab)",
            wraplength=280,
        )

        self.lbl_demo_category_hint = tk.Label(
            row,
            text="",
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
            justify=tk.LEFT,
            wraplength=280,
        )

        self._demo_info_btn = ttk.Label(row, text="", style="Card.TLabel")
        # Info action moved into Examples menu.

        # Compact Analyze indicator (stage · bar · ETA) — shown only while running.
        self._input_row2 = None
        from stablewalk.ui.tk.analyze_progress import build_analyze_progress_panel

        build_analyze_progress_panel(self, bar)

    def _show_input_settings_menu(self) -> None:
        """Popup for secondary input options (smooth motion, etc.)."""
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_checkbutton(
            label="Smooth motion",
            variable=self.smooth_motion,
            command=self._on_smooth_toggle,
        )
        try:
            x = self.btn_input_settings.winfo_rootx()
            y = self.btn_input_settings.winfo_rooty() + self.btn_input_settings.winfo_height()
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()

    def _load_demo_gait(self, key: str) -> None:
        """Load a labeled preview of a predefined demo (capped). Analyze runs the full source."""
        ex = example_by_key(key)
        if ex is None:
            return
        config.ensure_output_dirs()

        self._active_demo_gait = ex
        self._presentation_mode = False
        self._cancel_presentation_autoplay()
        self._cache_demo_file_duration()

        resolved = demo_path(ex)
        logger.info("Demo gait resolved path: %s (exists=%s)", resolved, resolved.is_file())

        from stablewalk.ui.media.demo_gait import is_legacy_demo_filename
        from stablewalk.ui.media.utah_abnormal import opencv_can_decode

        if is_legacy_demo_filename(resolved.name):
            messagebox.showwarning(ex.display_name, missing_file_message(ex))
            return
        if not resolved.is_file() or not opencv_can_decode(resolved):
            messagebox.showwarning(ex.display_name, missing_file_message(ex))
            return
        source = str(resolved)

        self.url_var.set(source)
        self.url_entry.delete(0, tk.END)
        self.url_entry.insert(0, source)
        self.preset_var.set(CUSTOM_URL_PRESET_LABEL)
        self._update_demo_analysis_title(ex)
        self._highlight_demo_button(key)
        self._sync_demo_category_hint()
        self._load_video(
            source=source,
            show_dialog=False,
            unique_session=True,
            analysis_mode="preview",
        )

    def _sync_demo_category_hint(self) -> None:
        """Put category context on demo-button tooltips (no permanent tagline)."""
        demo = getattr(self, "_active_demo_gait", None)
        hint = getattr(self, "lbl_demo_category_hint", None)
        if hint is not None:
            hint.configure(text="")
        # Refresh tooltips on demo gait buttons.
        from stablewalk.ui.theme import create_tooltip

        for key, btn in getattr(self, "_demo_gait_buttons", {}).items():
            ex = example_by_key(key)
            if ex is None:
                continue
            tagline = demo_category_tagline(key)
            tip = f"{ex.display_name}"
            if tagline:
                tip = f"{tip}\n{tagline}"
            if demo is not None and demo.key == key:
                tip = f"{tip}\n(active)"
            create_tooltip(btn, tip, wraplength=320)
        self._sync_demo_category_compare_strip()

    def _highlight_demo_button(self, active_key: str | None) -> None:
        for key, btn in getattr(self, "_demo_gait_buttons", {}).items():
            try:
                btn.configure(
                    style="Accent.TButton" if key == active_key else "Secondary.TButton"
                )
            except tk.TclError:
                pass

    def _clear_demo_gait_context(self) -> None:
        self._active_demo_gait = None
        self._demo_file_duration_s = None
        self._update_demo_analysis_title(None)
        self._highlight_demo_button(None)
        hint = getattr(self, "lbl_demo_category_hint", None)
        if hint is not None:
            hint.configure(text="")
        self._sync_demo_category_compare_strip()
        self._sync_demo_save_button()
        self._refresh_overview_recording_context()
        self._refresh_demo_comparison_panel()

    def _update_demo_analysis_title(self, example: DemoGaitExample | None = None) -> None:
        """Keep the video panel clean — demo context lives in the Info dialog."""
        lbl = getattr(self, "lbl_demo_analysis_title", None)
        meta_row = getattr(self, "_demo_meta_row", None)
        info_btn = getattr(self, "btn_demo_video_info", None)
        if lbl is not None:
            lbl.configure(text="")
            lbl.grid_remove()
        if meta_row is not None:
            meta_row.grid_remove()
        if info_btn is not None:
            info_btn.grid_remove()

    def _show_demo_video_details(self) -> None:
        ex = self._active_demo_gait
        if ex is None:
            messagebox.showinfo(
                "About active example",
                "No example is active. Load Abnormal Gait, Normal Gait, or Fast Walking first.",
            )
            return
        messagebox.showinfo("About active example", format_about_active_example(ex))

    def _apply_demo_presentation(self) -> None:
        """Refresh demo title and session label after a demo video loads."""
        if not self._active_demo_gait:
            return
        self._update_demo_analysis_title()
        self._session_display_src = self._active_demo_gait.display_name
        self._cache_demo_file_duration()
        self.status.configure(
            text=f"{self._active_demo_gait.display_name} — press Play"
        )
        self._sync_demo_category_hint()
        self._sync_demo_save_button()
        self._refresh_overview_recording_context()
        self._refresh_demo_comparison_panel()

    def _sync_demo_category_compare_strip(self) -> None:
        """Hide the redundant COMPARE banner — category already shows in the header."""
        lbl = getattr(self, "lbl_overview_demo_compare", None)
        if lbl is None:
            return
        try:
            lbl.configure(text="")
            lbl.grid_remove()
        except tk.TclError:
            pass

    def _sync_demo_save_button(self) -> None:
        btn = getattr(self, "btn_save_demo_comparison", None)
        if btn is None:
            return
        can_save = (
            self._active_demo_gait is not None
            and self.gait_motion is not None
            and self.selection.active_item_id is not None
        )
        btn.configure(state=tk.NORMAL if can_save else tk.DISABLED)

    def _save_demo_comparison(self) -> None:
        if not self._active_demo_gait or not self.gait_motion:
            return
        item_id = self.selection.active_item_id
        if not item_id:
            messagebox.showinfo(
                "Select a joint",
                "Select a joint or DOF in the sidebar, then click Save comparison.",
            )
            return
        self._capture_demo_comparison_snapshot(item_id=item_id)
        row = self._demo_comparison_rows.get(self._active_demo_gait.key)
        self._comparison_expanded = True
        self._refresh_demo_comparison_panel()
        if row is not None:
            self.status.configure(
                text=f"Comparison saved — {row.demo_type} · {row.joint_label}"
            )

    def _cache_demo_file_duration(self) -> None:
        ex = getattr(self, "_active_demo_gait", None)
        if ex is None:
            self._demo_file_duration_s = None
            return
        try:
            from stablewalk.ui.media.demo_gait import probe_demo_timing

            timing = probe_demo_timing(demo_path(ex))
            duration = float(timing.get("duration_s") or 0.0)
            self._demo_file_duration_s = duration if duration > 0 else None
        except Exception:
            self._demo_file_duration_s = None

    def _capture_demo_comparison_snapshot(self, *, item_id: str | None = None) -> None:
        """Store computed comparison facts for the active example only."""
        ex = getattr(self, "_active_demo_gait", None)
        recording = getattr(self, "gait_motion", None)
        if ex is None or recording is None:
            return
        joint_id = item_id or getattr(self.selection, "active_item_id", None)
        if not joint_id:
            joint_id = demo_default_dof_item(ex.key)
        usable, _detected = self._resolved_gait_cycle_count()
        coverage = None
        try:
            coverage = self._overview_hud_tracking_coverage()
        except Exception:
            coverage = None
        tracking = COMPUTED_PENDING
        if coverage is not None:
            tracking = coverage.compact_text()
        self._demo_comparison_rows[ex.key] = build_comparison_row(
            recording,
            ex,
            joint_id,
            usable_cycles=usable,
            tracking_quality=tracking,
            recording_duration_s=getattr(self, "_demo_file_duration_s", None),
        )
        self._refresh_overview_recording_context()
        self._refresh_demo_comparison_panel()

    def _refresh_overview_recording_context(self) -> None:
        """Current-video documented labels vs computed duration/cycles/tracking."""
        labels = getattr(self, "_overview_context_labels", None) or {}
        if not labels:
            return
        ex = getattr(self, "_active_demo_gait", None)
        source = documented_source_label(ex) if ex is not None else DOCUMENTED_UNAVAILABLE
        camera = documented_camera_view(ex) if ex is not None else DOCUMENTED_UNAVAILABLE
        condition = documented_locomotion(ex) if ex is not None else DOCUMENTED_UNAVAILABLE
        duration = COMPUTED_PENDING
        cycles = COMPUTED_PENDING
        tracking = COMPUTED_PENDING
        recording = getattr(self, "gait_motion", None)
        if recording is not None:
            from stablewalk.ui.media.demo_comparison import analyzed_duration_s

            usable_s = None
            meta = getattr(recording, "metadata", None) or {}
            if meta.get("usable_duration_s") is not None:
                try:
                    usable_s = float(meta.get("usable_duration_s"))
                except (TypeError, ValueError):
                    usable_s = None
            duration = format_duration_display(
                analyzed_s=analyzed_duration_s(recording),
                recording_s=getattr(self, "_demo_file_duration_s", None),
                usable_s=usable_s,
            )
            usable, _detected = self._resolved_gait_cycle_count()
            cycles = format_usable_cycles_display(usable)
            try:
                coverage = self._overview_hud_tracking_coverage()
            except Exception:
                coverage = None
            if coverage is not None:
                tracking = coverage.compact_text()
        mapping = {
            "source_label": source,
            "camera": camera,
            "condition": condition,
            "duration": duration,
            "cycles": cycles,
            "tracking": tracking,
        }
        for key, text in mapping.items():
            lbl = labels.get(key)
            if lbl is None:
                continue
            try:
                lbl.configure(text=text)
            except tk.TclError:
                pass

    def _refresh_demo_comparison_panel(self) -> None:
        trees = [
            getattr(self, "demo_comparison_tree", None),
            getattr(self, "overview_comparison_tree", None),
        ]
        trees = [tree for tree in trees if tree is not None]
        if not trees:
            return
        has_computed = False
        rows: list[tuple[str, ...]] = []
        for example in example_comparison_order():
            stored = self._demo_comparison_rows.get(example.key)
            if stored is None:
                stored = documented_placeholder_row(example)
            else:
                has_computed = True
            rows.append(comparison_tree_values(stored))
        for tree in trees:
            try:
                tree.delete(*tree.get_children())
                for values in rows:
                    tree.insert("", tk.END, values=values)
            except tk.TclError:
                pass
        self._sync_gait_comparison_visibility(has_rows=has_computed)

    def _sync_gait_comparison_visibility(self, *, has_rows: bool | None = None) -> None:
        toggle = getattr(self, "btn_toggle_comparison", None)
        body = getattr(self, "comparison_body", None)
        if toggle is None:
            return
        if body is not None:
            body.pack_forget()
        toggle.configure(text="Compare")

    def _open_comparison_mode(self) -> None:
        """Open professional dual-session Comparison Mode."""
        from stablewalk.ui.tk.dashboard_notebook import ENABLE_COMPARE_TAB

        if not ENABLE_COMPARE_TAB:
            return
        controller = getattr(self, "_comparison_mode", None)
        if controller is None:
            ensure = getattr(self, "_ensure_comparison_mode_loaded", None)
            if callable(ensure):
                controller = ensure()
        if controller is None:
            messagebox.showinfo(
                "Comparison Mode",
                "Comparison Mode is not available in this layout.",
            )
            return
        # Pin happens asynchronously inside enter() so the tab switch stays instant.
        controller.enter()

    def _export_comparison_report(self) -> None:
        """Export a multi-page PDF comparison report for the active left/right pair."""
        from stablewalk.io.comparison_report_export import (
            export_comparison_report_from_cache,
        )
        from stablewalk.ui.tk.session_cache import ensure_compare_cache

        cache = ensure_compare_cache(self)
        if cache.left() is None or cache.right() is None:
            messagebox.showinfo(
                "Export Comparison Report",
                "Pin or load two analyzed sessions (Left and Right) before exporting.",
            )
            return
        from datetime import datetime

        config.ensure_output_dirs()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = config.REPORTS_DIR / f"comparison_report_{stamp}.pdf"
        path = filedialog.asksaveasfilename(
            title="Export Comparison Report",
            initialdir=str(config.REPORTS_DIR.resolve()),
            initialfile=default.name,
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            written = export_comparison_report_from_cache(cache, path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Export Comparison Report", str(exc))
            return
        self.status.configure(text=f"Comparison report → {Path(written).name}")
        notify = getattr(self, "_notify_generated_files_changed", None)
        if notify is not None:
            try:
                notify()
            except Exception:
                pass
        messagebox.showinfo(
            "Export Comparison Report",
            f"Comparison report saved:\n{Path(written).resolve()}",
        )

    def _export_professional_pdf_report(self) -> None:
        """Export a multi-page Research PDF Report for the current session."""
        if getattr(self, "_presentation_mode", False):
            messagebox.showinfo(
                "Export Research PDF Report",
                "Load and analyze a real video before exporting a PDF report.",
            )
            return
        if not self.sequence and getattr(self, "_biomech_analysis", None) is None:
            messagebox.showinfo(
                "Export Research PDF Report",
                "Load pose data first (Run Full Analysis).",
            )
            return

        from datetime import datetime

        from stablewalk.io.session_pdf_report import export_session_pdf_report_from_gui

        config.ensure_output_dirs()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = self._current_run_name() or "session"
        default = config.REPORTS_DIR / f"research_report_{run}_{stamp}.pdf"
        path = filedialog.asksaveasfilename(
            title="Export Research PDF Report",
            initialdir=str(config.REPORTS_DIR.resolve()),
            initialfile=default.name,
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            written = export_session_pdf_report_from_gui(self, path)
        except ImportError as exc:
            messagebox.showerror("Export Research PDF Report", str(exc))
            return
        except (OSError, ValueError) as exc:
            messagebox.showerror("Export Research PDF Report", str(exc))
            return
        except Exception as exc:
            logger.exception("Research PDF export failed")
            messagebox.showerror("Export Research PDF Report", f"Export failed:\n{exc}")
            return

        self.status.configure(text=f"Research PDF → {Path(written).name}")
        notify = getattr(self, "_notify_generated_files_changed", None)
        if notify is not None:
            try:
                notify()
            except Exception:
                pass
        messagebox.showinfo(
            "Export Research PDF Report",
            f"Research PDF report saved:\n{Path(written).resolve()}",
        )

    def _open_gait_comparison_dialog(self) -> None:
        if not self._demo_comparison_rows:
            messagebox.showinfo(
                "Gait comparison",
                "Load demo gaits and click Save on each, then open Compare Gaits.",
            )
            return
        existing = getattr(self, "_gait_comparison_dialog", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    existing.focus_force()
                    return
            except tk.TclError:
                pass

        dlg = tk.Toplevel(self.root)
        dlg.title("Compare Gaits")
        dlg.geometry("680x340")
        dlg.minsize(480, 260)
        dlg.transient(self.root)
        self._gait_comparison_dialog = dlg

        container = ttk.Frame(dlg, padding=8)
        container.pack(fill=tk.BOTH, expand=True)

        body = getattr(self, "comparison_body", None)
        if body is not None:
            body.pack(in_=container, fill=tk.BOTH, expand=True)

        footer = ttk.Frame(container)
        footer.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(
            footer,
            text="Clear",
            style="Compact.TButton",
            command=self._clear_demo_comparison,
        ).pack(side=tk.LEFT)
        ttk.Button(footer, text="Close", command=lambda: _close()).pack(side=tk.RIGHT)

        def _close() -> None:
            if body is not None:
                body.pack_forget()
            self._gait_comparison_dialog = None
            try:
                dlg.destroy()
            except tk.TclError:
                pass

        dlg.protocol("WM_DELETE_WINDOW", _close)

    def _toggle_gait_comparison_panel(self) -> None:
        self._open_gait_comparison_dialog()

    def _clear_demo_comparison(self) -> None:
        self._demo_comparison_rows.clear()
        self._comparison_expanded = False
        self._refresh_demo_comparison_panel()
        self.status.configure(text="Gait comparison cleared")

    def _build_layout(self) -> None:
        from stablewalk.ui.tk.dashboard_layout import build_dashboard_layout

        build_dashboard_layout(self)
        self._refresh_knee_source_selector()
        self._sync_dof_table_mode_flag()
        self._configure_dof_table_columns()
        self._init_opensim_panel()
        self._build_legacy_hidden_panels()
        self._update_export_output_status()
        self._update_export_section_status()
        self._update_dashboard_empty_states()
        self._refresh_demo_comparison_panel()
        self._refresh_overview_recording_context()
        # Bind video panel host (not the label) so frame paints do not fire resize.
        video_host = getattr(self, "video_frame", None) or getattr(
            self, "_video_clip_canvas", None
        )
        if video_host is not None:
            video_host.bind("<Configure>", self._on_video_label_resize, add="+")
        elif hasattr(self, "video_label"):
            self.video_label.bind("<Configure>", self._on_video_label_resize, add="+")
        if hasattr(self, "video_label"):
            self.video_label.bind("<Button-1>", self._on_video_click_select_subject, add="+")
        from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout

        self.root.after_idle(lambda: apply_responsive_layout(self))
        from stablewalk.ui.tk.dashboard_shell import (
            assert_dashboard_widget_singletons,
            print_structural_diagnostic_report,
        )
        from stablewalk.ui.tk.dashboard_notebook import reflow_tab_canvases
        from stablewalk.ui.tk.gui_layout_debug import log_gui_layout_audit_if_enabled

        # Structural dump is opt-in — printing it every launch costs layout work.
        import os

        if os.environ.get("STABLEWALK_STRUCTURAL_DIAG", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            print_structural_diagnostic_report(self)
        assert_dashboard_widget_singletons(self)
        self.root.after_idle(reflow_tab_canvases, self)
        self.root.after_idle(
            lambda: log_gui_layout_audit_if_enabled(self, context="build_layout")
        )

    def _build_legacy_hidden_panels(self) -> None:
        """Hidden legacy tables and optional robot panel (not shown in dashboard)."""
        frame_panel = ttk.Frame(self.root)
        self._frame_panel = frame_panel

        self.lbl_frame_data = ttk.Label(
            frame_panel,
            text="Frame —",
            style="Heading.TLabel",
            wraplength=POS_PANEL_WIDTH,
        )

        self.frame_notebook = ttk.Notebook(frame_panel)

        def _legacy_tree_frame(
            parent: ttk.Frame,
            columns: tuple[str, ...],
            headings: dict[str, str],
            *,
            col_widths: dict[str, int] | None = None,
            height: int = 18,
        ) -> ttk.Treeview:
            fr = ttk.Frame(parent)
            fr.pack(fill=tk.BOTH, expand=True)
            tree = ttk.Treeview(
                fr,
                columns=columns,
                show="headings",
                height=height,
                style="Large.Treeview",
            )
            widths = col_widths or {}
            for col in columns:
                tree.heading(col, text=headings.get(col, col))
                w = widths.get(col, 72)
                anchor = tk.W if col in ("joint", "name", "label", "dof") else tk.E
                stretch = col not in ("joint", "dof")
                tree.column(col, width=w, anchor=anchor, minwidth=48, stretch=stretch)
            tree._sw_column_widths = dict(widths)  # type: ignore[attr-defined]
            tree.grid(row=0, column=0, sticky="nsew")
            vsb = ttk.Scrollbar(fr, orient=tk.VERTICAL, command=tree.yview)
            hsb = ttk.Scrollbar(fr, orient=tk.HORIZONTAL, command=tree.xview)
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")
            fr.grid_rowconfigure(0, weight=1)
            fr.grid_columnconfigure(0, weight=1)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            return tree

        tab_pos = ttk.Frame(self.frame_notebook)
        self.frame_notebook.add(tab_pos, text="Positions")
        self.pos_3d_tree = _legacy_tree_frame(
            tab_pos,
            ("joint", "x", "y", "z"),
            {"joint": "Joint", "x": "X", "y": "Y", "z": "Z"},
            col_widths={"joint": 140, "x": 88, "y": 88, "z": 88},
            height=14,
        )

        tab_ang = ttk.Frame(self.frame_notebook)
        self.frame_notebook.add(tab_ang, text="Angles")
        self.dof_tree = _legacy_tree_frame(
            tab_ang,
            ("joint", "angle", "omega", "rom", "speed"),
            {
                "joint": "Joint",
                "angle": "°",
                "omega": "ω°/s",
                "rom": "Full-session ROM",
                "speed": "|v|",
            },
            col_widths={"joint": 118, "angle": 56, "omega": 62, "rom": 72, "speed": 56},
        )

        tab_vel = ttk.Frame(self.frame_notebook)
        self.frame_notebook.add(tab_vel, text="Velocity")
        self.vel_tree = _legacy_tree_frame(
            tab_vel,
            ("joint", "vx", "vy", "speed", "dir"),
            {
                "joint": "Joint",
                "vx": "vx",
                "vy": "vy",
                "speed": "|v|",
                "dir": "dir°",
            },
            col_widths={"joint": 118, "vx": 62, "vy": 62, "speed": 58, "dir": 52},
        )

        tab_sum = ttk.Frame(self.frame_notebook)
        self.frame_notebook.add(tab_sum, text="Summary")
        self.motion_text = tk.Text(tab_sum, height=16, wrap=tk.WORD, state=tk.DISABLED)
        self.motion_text.configure(
            bg=PANEL, fg=TEXT, font=FONT_UI_SM, relief=tk.FLAT, highlightthickness=0
        )
        self.motion_text.pack(fill=tk.BOTH, expand=True, padx=PAD_XS // 2, pady=PAD_XS // 2)

        self.dof_tree.bind("<<TreeviewSelect>>", self._on_dof_tree_select)
        self.pos_3d_tree.bind("<<TreeviewSelect>>", self._on_pos_tree_select)

        self.robot_frame = ttk.LabelFrame(
            self.root,
            text=format_labelframe_title("Robot simulation"),
            style="Card.TLabelframe",
            padding=PAD_SM,
        )
        self.fig_robot = Figure(figsize=(3.2, 3.5), dpi=100, facecolor=PANEL)
        self.ax_robot = self.fig_robot.add_subplot(111, projection="3d")
        setup_3d_axes(self.ax_robot)
        self.canvas_robot = FigureCanvasTkAgg(self.fig_robot, master=self.robot_frame)
        self.canvas_robot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_transport_bar(self) -> None:
        """Playback controls and timeline scrubber."""
        transport = ttk.Frame(self.root, padding=(PAD_MD, PAD_XS, PAD_MD, PAD_XS))
        transport.pack(side=tk.BOTTOM, fill=tk.X, padx=PAD_XS, pady=(0, PAD_XS))
        self._transport_bar = transport

        row = ttk.Frame(transport)
        row.pack(fill=tk.X)
        row.columnconfigure(5, weight=1)
        self._transport_row = row

        playback = ttk.Frame(row)
        playback.grid(row=0, column=0, sticky="w")
        self._transport_playback = playback

        self.btn_play_bar = ttk.Button(
            playback,
            text="Play",
            width=8,
            style="Accent.TButton",
            command=self._toggle_play,
        )
        self.btn_play_bar.pack(side=tk.LEFT, padx=(0, PAD_XS))
        self.btn_play = self.btn_play_bar

        self.btn_stop_bar = ttk.Button(
            playback,
            text="Stop",
            width=7,
            style="Transport.TButton",
            command=self._stop_playback,
        )
        self.btn_stop_bar.pack(side=tk.LEFT, padx=(0, PAD_XS))

        self.btn_reset_bar = ttk.Button(
            playback,
            text="Reset",
            width=7,
            style="Transport.TButton",
            command=self._reset_playback,
        )
        self.btn_reset_bar.pack(side=tk.LEFT)

        self._transport_sep1 = ttk.Separator(row, orient=tk.VERTICAL)
        self._transport_sep1.grid(row=0, column=1, sticky="ns", padx=PAD_MD)

        frame_grp = ttk.Frame(row)
        frame_grp.grid(row=0, column=2, sticky="w")
        self._transport_frame = frame_grp

        self._btn_step_back = ttk.Button(
            frame_grp,
            text="◀ Frame",
            width=8,
            style="Transport.TButton",
            command=lambda: self._step(-1),
        )
        self._btn_step_fwd = ttk.Button(
            frame_grp,
            text="Frame ▶",
            width=8,
            style="Transport.TButton",
            command=lambda: self._step(1),
        )
        self._btn_step_back.pack(side=tk.LEFT, padx=(0, PAD_XS))
        self._btn_step_fwd.pack(side=tk.LEFT)

        self._transport_sep2 = ttk.Separator(row, orient=tk.VERTICAL)
        self._transport_sep2.grid(row=0, column=3, sticky="ns", padx=PAD_MD)

        analysis = ttk.Frame(row)
        analysis.grid(row=0, column=4, sticky="w")
        self._transport_analysis = analysis

        self._lbl_sampling = ttk.Label(analysis, text="Sampling", style="Card.TLabel")
        self._lbl_sampling.pack(side=tk.LEFT, padx=(0, PAD_XS))
        self.refresh_var = tk.StringVar(value=REFRESH_INTERVAL_DEFAULT)
        self.cmb_sampling = ttk.Combobox(
            analysis,
            textvariable=self.refresh_var,
            values=REFRESH_INTERVAL_CHOICES,
            state="readonly",
            width=6,
        )
        self.cmb_sampling.pack(side=tk.LEFT, padx=(0, PAD_MD))
        self.cmb_sampling.bind("<<ComboboxSelected>>", self._on_refresh_interval)
        try:
            from stablewalk.ui.theme import create_tooltip

            _sampling_tip = (
                "Display look-ahead for the next-step preview table only. "
                "Pose extraction and gait analysis always use the original "
                "video timestamps at the source frame rate."
            )
            create_tooltip(self._lbl_sampling, _sampling_tip, wraplength=360)
            create_tooltip(self.cmb_sampling, _sampling_tip, wraplength=360)
        except Exception:
            pass

        self._lbl_speed = ttk.Label(analysis, text="Speed", style="Card.TLabel")
        self._lbl_speed.pack(side=tk.LEFT, padx=(0, PAD_XS))
        self.speed_var = tk.DoubleVar(value=1.0)
        self._speed_scale = ttk.Scale(
            analysis,
            from_=0.25,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            length=140,
            style="Transport.Horizontal.TScale",
            command=self._on_speed_change,
        )
        self._speed_scale.pack(side=tk.LEFT, padx=(0, PAD_XS))
        self.lbl_speed = ttk.Label(analysis, text="1.00×", style="Card.TLabel", width=5)
        self.lbl_speed.pack(side=tk.LEFT, padx=(0, PAD_MD))

        timeline = ttk.Frame(row)
        timeline.grid(row=0, column=5, sticky="ew")
        row.columnconfigure(5, weight=1)
        self._transport_timeline = timeline

        self.frame_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(
            timeline,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.frame_var,
            style="Transport.Horizontal.TScale",
            command=self._on_slider,
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, PAD_XS))

        self.lbl_frame = ttk.Label(timeline, text="0.00s / 0.00s", style="Heading.TLabel", width=13)
        self.lbl_frame.pack(side=tk.RIGHT)
        # Frame index lives on the status bar — keep a hidden alias for callers.
        self.lbl_frame_idx = ttk.Label(timeline, text="F—/—", style="Card.TLabel")
        self.lbl_frame_idx.pack_forget()

    def _build_status_bar(self) -> None:
        from stablewalk.ui.tk.dashboard_status_bar import build_dashboard_status_bar

        build_dashboard_status_bar(self)

    @staticmethod
    def _play_button_text(playing: bool) -> str:
        return "Pause" if playing else "Play"

    def _sync_play_buttons(self) -> None:
        text = self._play_button_text(self.playing)
        self.btn_play.configure(text=text)
        if hasattr(self, "btn_play_bar"):
            self.btn_play_bar.configure(text=text)
        skel_play = getattr(self, "btn_play_skeleton", None)
        if skel_play is not None:
            try:
                skel_play.configure(text=text)
            except tk.TclError:
                pass

    @staticmethod
    def _short_source_name(src: str) -> str:
        p = Path(src)
        if p.suffix and len(src) < 120:
            return p.name
        return src[:40] + "…" if len(src) > 43 else src

    def _show_compare_note(self, text: str) -> None:
        if text:
            self.lbl_compare.configure(text=text)
            self.lbl_compare.pack(fill=tk.X, pady=(PAD_XS, 0))
        else:
            self.lbl_compare.pack_forget()

    def _attach_tooltips(self) -> None:
        for widget, tip in {
            self.preset_combo: "Video preset",
            self.url_entry: "URL or file (Enter)",
            self.btn_full: "Run analysis",
            self.btn_browse: "Local video file",
            self.btn_input_settings: "Input settings (smooth motion, etc.)",
            self.btn_next_video: "Next clip",
            self.btn_play: "Play / pause (Space)",
            self.btn_play_bar: "Play / pause (Space)",
            self.btn_stop_bar: "Stop (keeps the current frame)",
            self.btn_reset_bar: "Reset to the first frame",
            self.slider: "Timeline scrubber — drag to scrub frame-by-frame",
            getattr(self, "cmb_sampling", None): "Playback sampling / refresh interval",
            getattr(self, "btn_step_back", None): "Step one frame backward (←)",
            getattr(self, "btn_step_fwd", None): "Step one frame forward (→)",
        }.items():
            if widget is not None:
                create_tooltip(widget, tip)
        if hasattr(self, "cmb_skeleton_mode"):
            create_tooltip(
                self.cmb_skeleton_mode,
                "Skeleton type: Walking / 2D / 3D / Side / Biomechanical stay; "
                "Full Skeleton or 16-Joint Gait Model for the dedicated wireframe.",
            )
        explain = getattr(self, "lbl_dof_graph_explain_body", None)
        if explain is not None:
            create_tooltip(
                explain,
                "3D path colored from dim (start) to bright (current position)",
            )
        gc_left = getattr(self, "lbl_ground_clearance_left", None)
        gc_right = getattr(self, "lbl_ground_clearance_right", None)
        from stablewalk.ui.metric_tooltips import get_metric_tooltip

        clearance_tip = get_metric_tooltip("foot_clearance")
        if gc_left is not None and clearance_tip:
            create_tooltip(gc_left, clearance_tip, wraplength=340)
        if gc_right is not None and clearance_tip:
            create_tooltip(gc_right, clearance_tip, wraplength=340)
        self._attach_opensim_tooltips()
        self._attach_export_tooltips()
        from stablewalk.ui.metric_tooltips import attach_professional_metric_tooltips

        attach_professional_metric_tooltips(self)

    def _attach_export_tooltips(self) -> None:
        """Tooltips for the Detailed Joint Data & Export section."""
        if getattr(self, "_export_tooltips_attached", False):
            return
        tips = {
            "btn_view_detailed_data": (
                "Open a per-frame joint table (frame, time, XYZ, angle, velocity, "
                "acceleration, confidence, gait phase when available)."
            ),
            "btn_export_joint_csv": (
                "Export per-frame joint data as CSV and JSON to the output folder."
            ),
            "btn_open_coord_inspector": (
                "Open the 3D Coordinate Inspector window (joint, frame, XYZ, motion)."
            ),
            "btn_clear_dof_history": (
                "Clear collected playback samples for the current session."
            ),
            "btn_save_session": (
                "Save the full analysis session (poses, selection, metrics) to a folder."
            ),
            "btn_export_analysis_report": (
                "Write a text report with gait metrics, forces, and stability summary."
            ),
            "btn_export_pdf_report": (
                "Export the StableWalk Research PDF Report (session info, metrics, "
                "graphs, heel strikes, contact timing, confidence, processing)."
            ),
            "btn_export_gait_metrics": (
                "Export structured gait and stability metrics as JSON."
            ),
            "btn_opensim_export_data": (
                "Export OpenSim .trc, .mot/.csv, JSON, and mapped TRC for this session."
            ),
            "btn_export_motion_reference": (
                "Export stablewalk_motion.npz for Real-to-Sim / retargeting workflows."
            ),
            "btn_real_to_sim_pipeline": (
                "Run full Real-to-Sim pipeline: gait style, retargeting, AMP export, vGRF."
            ),
            "btn_export_amp_reference": (
                "Export AMP reference motion clip for Isaac Lab imitation learning."
            ),
        }
        attached = False
        for attr, tip in tips.items():
            widget = getattr(self, attr, None)
            if widget is not None:
                create_tooltip(widget, tip)
                attached = True
        if attached:
            self._export_tooltips_attached = True

    def _attach_opensim_tooltips(self) -> None:
        """One-time OpenSim sidebar tooltips (SDK version, etc.)."""
        if getattr(self, "_opensim_tooltips_attached", False):
            return
        if not hasattr(self, "lbl_opensim_sdk"):
            return
        self._opensim_tooltips_attached = True
        ver = getattr(self, "_opensim_sdk_version", None)
        from stablewalk.ui.tk.sidebar_display import sdk_tooltip_line

        create_tooltip(
            self.lbl_opensim_sdk,
            sdk_tooltip_line(version=ver),
        )
        from stablewalk.ui.tk.sidebar_display import OPENSIM_MODEL_DROPDOWN_HINT

        if hasattr(self, "opensim_model_combo"):
            create_tooltip(self.opensim_model_combo, OPENSIM_MODEL_DROPDOWN_HINT)
        if hasattr(self, "btn_opensim_local_model"):
            create_tooltip(self.btn_opensim_local_model, OPENSIM_MODEL_DROPDOWN_HINT)
        export_btn = getattr(self, "btn_opensim_export", None) or getattr(
            self, "btn_opensim_export_data", None
        )
        if export_btn is not None:
            create_tooltip(
                export_btn,
                "Export .trc, .mot/.csv, JSON, and mapped TRC for this session",
            )
        if hasattr(self, "btn_opensim_run_ik"):
            create_tooltip(
                self.btn_opensim_run_ik,
                "Run Export OpenSim Files on an analyzed video first.",
            )
        if hasattr(self, "lbl_opensim_reliability"):
            from stablewalk.ui.tk.sidebar_display import RELIABILITY_TOOLTIP

            create_tooltip(self.lbl_opensim_reliability, RELIABILITY_TOOLTIP)

    def _set_opensim_presentation_banner(self, visible: bool) -> None:
        """Show or hide presentation-demo labels at the top of OpenSim Status."""
        mode = getattr(self, "lbl_opensim_demo_mode", None)
        subtitle = getattr(self, "lbl_opensim_demo_subtitle", None)
        note = getattr(self, "lbl_opensim_demo_note", None)
        anchor = getattr(self, "lbl_opensim_sdk", None)
        for widget in (mode, subtitle, note):
            if widget is None:
                continue
            if visible:
                if not widget.winfo_ismapped():
                    widget.pack(
                        anchor=tk.W,
                        pady=(0, 2 if widget is note else 3),
                        before=anchor,
                    )
            else:
                widget.pack_forget()

    def _init_chart(self) -> None:
        from stablewalk.ui.viewers.chart_style import style_single_time_series_chart

        if getattr(self, "ax_chart", None) is None or getattr(self, "fig", None) is None:
            return
        self.ax_chart.clear()
        style_single_time_series_chart(self.ax_chart, ylabel="Knee flexion (°)")
        self.ax_chart.set_title(
            "Knee flexion",
            color=TEXT,
            fontsize=9.5,
            fontweight="medium",
            pad=5,
        )
        self.fig.tight_layout(pad=1.2)

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda _e: self._toggle_play())
        self.root.bind("<Left>", lambda _e: self._step(-1))
        self.root.bind("<Right>", lambda _e: self._step(1))
        self.root.bind("<Home>", lambda _e: self._go_to(0))
        self.root.bind(
            "<End>",
            lambda _e: (
                self._go_to(self.skeleton_player.frame_count - 1)
                if self.skeleton_player and self.skeleton_player.frame_count
                else None
            ),
        )
        self.root.bind("<Control-o>", lambda _e: self._menu_open())
        self.root.bind("<F1>", lambda _e: self._show_analysis_methods())

    def _set_gait_motion(self, recording: GaitMotionRecording) -> None:
        """Attach a gait recording and reset the skeleton player."""
        recording.build_time_series()
        self._cached_source_video_frames = None
        self.gait_motion = recording
        from stablewalk.models.gait_wireframe import build_gait_wireframe

        self.gait_wireframe = build_gait_wireframe(recording)
        self._gait_wire_body_scale = float(self.gait_wireframe.body_scale)
        self._motion_series_cache_key = None
        self._motion_frame_series_cache = None
        self._overview_traj_draw_cache_key = None
        self._overview_path_cache = {}
        try:
            from stablewalk.ui.viewers.dof_trajectory_3d import clear_trajectory_path_cache

            clear_trajectory_path_cache()
        except Exception:
            pass
        from stablewalk.ui.skeleton_display_filter import ensure_display_filter

        ensure_display_filter(self).reset()
        self.skeleton_player = SkeletonPlayer(recording, smooth=self.smooth_motion.get())
        self.skeleton_player.stop(reset=True)
        self._skeleton_artists_dirty = True
        self._sync_playback_hz_from_source()
        n = max(recording.frame_count - 1, 0)
        self.slider.configure(to=max(n, 0))
        self.frame_var.set(0)
        self.current_pos = 0
        self._playback_pos = 0.0
        self.playing = False
        self._sync_play_buttons()
        self._cancel_timer()
        self._sync_transport_labels()
        self._update_skeleton_view_box()
        self._update_interactive_skeleton(force_draw=True)
        self._update_session_selection_overview()
        self._ensure_demo_dof_selection()
        # Always rebuild Overview / Motion 3D paths after a new recording attach.
        # Default-DOF selection is a no-op when a joint is already selected, so
        # progressive re-attach would otherwise leave a cleared/empty scene.
        try:
            self._overview_traj_draw_cache_key = None
            self._traj_draw_cache_key = None
            if getattr(self, "_overview_traj_dock_visible", False):
                self._refresh_overview_trajectory_dock(force_draw=True)
            if getattr(self, "canvas_dof_traj", None) is not None or getattr(
                self, "ax_dof_traj", None
            ) is not None:
                self._refresh_selected_dof_trajectory_3d(force_draw=True)
            self._fit_overview_viewers(reset_camera=True)
        except Exception:
            logger.debug("trajectory redraw after gait attach failed", exc_info=True)
        sched = getattr(self, "_schedule_spectral_analysis", None)
        if callable(sched):
            try:
                sched(True)
            except Exception:
                logger.debug("spectral schedule after gait attach failed", exc_info=True)
        sched_f = getattr(self, "_schedule_four_gait_signals", None)
        if callable(sched_f):
            try:
                sched_f(True)
            except Exception:
                logger.debug("four-gait schedule after gait attach failed", exc_info=True)
        sched_a = getattr(self, "_schedule_asymmetry_analysis", None)
        if callable(sched_a):
            try:
                sched_a(True)
            except Exception:
                logger.debug("asymmetry schedule after gait attach failed", exc_info=True)

    def _ensure_demo_dof_selection(self) -> None:
        """Pick the category-appropriate joint so the three demos compare clearly."""
        if not hasattr(self, "_dof_checkbox_vars"):
            return
        demo = getattr(self, "_active_demo_gait", None)
        if demo is None:
            self._ensure_default_dof_selection()
            return
        item_id = demo_default_dof_item(demo.key)
        if item_id not in self._dof_checkbox_vars:
            self._ensure_default_dof_selection()
            return
        self.selection.select_only(item_id)
        self._sync_dof_checkboxes()
        self._notify_dof_selection_changed()

    def _update_skeleton_view_box(self) -> None:
        """Compute a stable, sequence-global view box for the skeleton panel.

        Called whenever the recording or display mode changes. The box encloses
        the full body across every frame, so playback keeps a constant scale and
        never crops the person. Stored on the axes and consumed by
        ``draw_gait_skeleton`` -> ``_apply_view_limits``.
        """
        ax = getattr(self, "ax_3d", None)
        if ax is None:
            return
        recording = getattr(self, "gait_motion", None)
        if recording is None or not getattr(recording, "snapshots", None):
            ax._sw_fixed_view_box = None  # type: ignore[attr-defined]
            return
        try:
            box = compute_skeleton_view_box(
                recording, self._skeleton_display_mode_key()
            )
        except Exception:
            box = None
        ax._sw_fixed_view_box = box  # type: ignore[attr-defined]
        for attr in ("_sw_view_cache", "_sw_view_scale_cache"):
            if hasattr(ax, attr):
                try:
                    delattr(ax, attr)
                except Exception:
                    setattr(ax, attr, None)

    def _fit_full_body(self) -> None:
        """Fit the skeleton viewer to the full head-to-feet motion range."""
        from stablewalk.ui.viewers.gait_skeleton_renderer import (
            clear_skeleton_user_view,
            fit_full_body_view,
        )

        recording = self._analysis_motion_recording()
        snap = None
        if self.skeleton_player is not None:
            try:
                snap = self.skeleton_player.current_snapshot()
            except Exception:
                snap = None
        ax = getattr(self, "ax_3d", None)
        if ax is None:
            return
        clear_skeleton_user_view(ax)
        fitted = fit_full_body_view(
            ax,
            snap,
            recording,
            self._skeleton_display_mode_key(),
        )
        self._skel_fit_size_sig = None
        try:
            self._fit_skeleton_canvas(force=True)
        except Exception:
            pass
        if fitted:
            self._paint_canvas_now(getattr(self, "canvas_3d", None), idle=True)
            try:
                self.status.configure(text="Skeleton fitted to full body")
            except tk.TclError:
                pass

    def _fit_overview_viewers(self, *, reset_camera: bool = False) -> None:
        """Fit skeleton + trajectory after load / joint change. Skip during play."""
        if bool(getattr(self, "playing", False)):
            return
        try:
            self._fit_skeleton_canvas(force=True)
        except Exception:
            pass
        try:
            self._fit_overview_trajectory(keep_camera=not reset_camera)
        except Exception:
            pass
        try:
            rebuild = getattr(self, "_rebuild_overview_camera_view_menu", None)
            if callable(rebuild):
                rebuild()
        except Exception:
            pass
        try:
            self._refresh_overview_coord_caption()
        except Exception:
            pass

    def _refresh_overview_coord_caption(self) -> None:
        """Keep the Overview path footer origin / axis / units line current."""
        from stablewalk.ui.overview_path_display import coordinate_frame_caption

        lbl = getattr(self, "lbl_overview_coord_caption", None)
        if lbl is None:
            return
        coord_mode = "Root-relative"
        var = getattr(self, "var_dof_coord_mode", None)
        if var is not None:
            try:
                coord_mode = str(var.get() or coord_mode)
            except Exception:
                pass
        try:
            lbl.configure(
                text=coordinate_frame_caption(
                    coord_mode=coord_mode,
                    recording=self._analysis_motion_recording(),
                )
            )
        except tk.TclError:
            pass

    def _fit_overview_trajectory(self, *, keep_camera: bool = True) -> None:
        """Fit the Overview 3D Path to the full selected-joint trajectory."""
        from stablewalk.ui.viewers.dof_trajectory_3d import fit_trajectory_viewport

        ax = getattr(self, "ax_dof_traj_overview", None)
        if ax is None:
            ax = getattr(self, "ax_dof_traj", None)
        fit_trajectory_viewport(ax, keep_camera=keep_camera)
        try:
            self._refresh_overview_trajectory_dock(force_draw=True)
        except Exception:
            pass
        try:
            self._render_overview_traj_canvas(force=True)
        except Exception:
            pass
        try:
            self.status.configure(text="Trajectory fitted to full path")
        except tk.TclError:
            pass

    def _sync_transport_labels(self, state=None) -> None:
        from stablewalk.ui.playback_state import committed_playback_state

        resolved = state if state is not None else committed_playback_state(self)
        if resolved is None:
            key = ("0.00s / 0.00s", "F—/—")
        else:
            frame_lbl = resolved.format_frame_label_prefixed()
            src_lbl = resolved.format_source_frame_label()
            if src_lbl and src_lbl != "—":
                frame_lbl = f"{frame_lbl} · {src_lbl}"
            key = (resolved.format_time_label(), frame_lbl)
        if self._transport_label_cache != key:
            self._transport_label_cache = key
            self.lbl_frame.configure(text=key[0])
            if hasattr(self, "lbl_frame_idx"):
                self.lbl_frame_idx.configure(text=key[1])
        try:
            from stablewalk.ui.tk.dashboard_status_bar import update_dashboard_status_bar

            # Frame/time chrome must follow the same commit as Live data.
            # Do not 8 Hz-throttle the status bar — that left it two frames behind.
            update_dashboard_status_bar(self, state=resolved)
        except Exception:
            pass

    def _configure_label_if_changed(self, lbl, **kwargs) -> bool:
        """Apply label kwargs only when at least one value actually changed."""
        if lbl is None:
            return False
        try:
            changed = False
            for key, value in kwargs.items():
                if str(lbl.cget(key)) != str(value):
                    changed = True
                    break
            if not changed:
                return False
            lbl.configure(**kwargs)
            return True
        except tk.TclError:
            return False

    def _paint_canvas_now(self, canvas, *, idle: bool = False) -> None:
        """Matplotlib paint — Overview always uses ``draw_idle`` (never rebuild)."""
        if canvas is None:
            return
        from stablewalk.ui.tk.redraw_policy import skip_unmapped_canvas_paint

        if skip_unmapped_canvas_paint(canvas):
            return
        painted = getattr(self, "_pb_painted_canvas_ids", None)
        # One paint per canvas per play tick — skip redundant draw_idle queues.
        if painted is not None and id(canvas) in painted:
            return
        from stablewalk.ui.tk.render_diagnostics import timed_render

        overview_canvas = canvas is getattr(self, "canvas_3d", None) or canvas is getattr(
            self, "canvas_dof_traj_overview", None
        )
        with timed_render(self, "mpl_draw"):
            try:
                if overview_canvas or self.playing or idle:
                    canvas.draw_idle()
                else:
                    canvas.draw()
                if painted is not None:
                    painted.add(id(canvas))
            except Exception:
                try:
                    canvas.draw_idle()
                    if painted is not None:
                        painted.add(id(canvas))
                except Exception:
                    pass
        # One-shot soft reveal after tab enter / panel load (never during playback).
        if not idle and not self.playing:
            self._maybe_reveal_painted_canvas(canvas)

    def _maybe_reveal_painted_canvas(self, canvas) -> None:
        """Consume a pending viz-reveal mark for this canvas, if any."""
        try:
            from stablewalk.ui.tk.ui_motion import consume_viz_reveal, reveal_canvas
        except Exception:
            return
        kind = None
        if canvas is getattr(self, "canvas_3d", None):
            kind = "3d"
        elif canvas is getattr(self, "canvas_dof_traj", None):
            kind = "traj"
        elif canvas is getattr(self, "chart_canvas", None):
            kind = "graph"
        elif canvas is getattr(self, "canvas_biomech", None):
            kind = "biomech"
        elif canvas is getattr(self, "canvas_joint_motion", None):
            kind = "graph"
        if not kind or not consume_viz_reveal(self, kind):
            return
        try:
            reveal_canvas(self, canvas, key=kind)
        except Exception:
            pass

    def _overview_hud_gait_phase(self, frame_index: int | None) -> str:
        """Live gait phase string for the video info card (contact-derived)."""
        result = getattr(self, "_gait_cycle", None)
        from stablewalk.ui.overview_clinical_summary import NOT_AVAILABLE

        busy = False
        try:
            busy = bool(self._analysis_workflow_busy())
        except Exception:
            busy = False
        has_session = bool(
            getattr(self, "skeleton_player", None)
            and getattr(self.skeleton_player, "frame_count", 0) > 0
        ) or bool(getattr(self, "sequence", None))
        # After a session exists, never leave a stuck "Estimating..." chip.
        pending = "Estimating..." if (busy and not has_session) else NOT_AVAILABLE
        if result is None or not result.per_frame:
            return pending
        state = None
        if frame_index is not None:
            state = result.frame_at(frame_index)
        if state is None:
            state = result.per_frame[-1]
        if state is None:
            return pending
        from stablewalk.analysis.foot_contact_analysis import (
            foot_contact_observation_known,
        )
        from stablewalk.analysis.gait_phase_classification import (
            classify_gait_phase_from_contacts,
        )

        m = result.metrics
        phase = classify_gait_phase_from_contacts(
            state.left_contact,
            state.right_contact,
            contact_confidence=m.contact_confidence,
            confidence_tier=m.confidence_tier,
            left_foot_clearance_m=state.left.foot_clearance_m,
            right_foot_clearance_m=state.right.foot_clearance_m,
            left_known=foot_contact_observation_known(
                state.left, state.left_contact
            ),
            right_known=foot_contact_observation_known(
                state.right, state.right_contact
            ),
        )
        if phase is None:
            return pending
        return self._format_dashboard_gait_phase(phase)

    def _overview_hud_tracking_coverage(self):
        """Person detection vs joint visibility vs usable gait frames."""
        from stablewalk.ui.tracking_coverage import measure_tracking_coverage

        return measure_tracking_coverage(
            getattr(self, "sequence", None),
            getattr(self, "pose_indices", None),
            video_source_frames=self._source_video_frame_count(),
        )

    def _source_video_frame_count(self) -> int | None:
        """Original video file frame count when it differs from the analyzed clip."""
        cached = getattr(self, "_cached_source_video_frames", None)
        if cached is not None:
            return cached if cached > 0 else None
        path = getattr(self, "video_path", None) or getattr(self, "_video_path", None)
        if not path:
            demo = getattr(self, "_active_demo_gait", None)
            if demo is not None:
                from stablewalk.ui.media.demo_gait import demo_path

                try:
                    path = demo_path(demo)
                except Exception:
                    path = None
        if not path:
            try:
                path = self.url_var.get() if getattr(self, "url_var", None) is not None else None
            except Exception:
                path = None
        if not path:
            self._cached_source_video_frames = 0
            return None
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        except Exception:
            n = 0
        self._cached_source_video_frames = n
        return n if n > 0 else None

    def _overview_hud_tracking_confidence(self) -> str:
        """HUD Tracking field: coverage breakdown, not a saturated 100% score."""
        from stablewalk.ui.overview_clinical_summary import NOT_AVAILABLE

        coverage = self._overview_hud_tracking_coverage()
        if coverage is not None:
            return coverage.hud_text()
        return NOT_AVAILABLE

    def _overview_hud_detected_frames(self) -> str:
        """Detected pose frames vs total frames in the loaded sequence."""
        from stablewalk.ui.overview_clinical_summary import NOT_AVAILABLE

        seq = getattr(self, "sequence", None)
        idxs = getattr(self, "pose_indices", None)
        if seq is None or not getattr(seq, "frames", None):
            return NOT_AVAILABLE
        total = len(seq.frames)
        detected = len(idxs) if idxs else sum(
            1 for fr in seq.frames if getattr(fr, "detected", False)
        )
        return f"{int(detected)} / {int(total)}"

    def _overview_hud_detected_cycles(self) -> str:
        """Detected gait-cycle count for the video info card."""
        from stablewalk.ui.overview_clinical_summary import NOT_AVAILABLE

        usable = detected = None
        resolver = getattr(self, "_resolved_gait_cycle_count", None)
        if callable(resolver):
            try:
                usable, detected = resolver()
            except Exception:
                usable = detected = None
        if detected is None:
            gait = getattr(self, "_gait_cycle", None)
            if gait is not None:
                detected = getattr(gait, "detected_cycle_count", None)
                if detected is None and getattr(gait, "cycles", None) is not None:
                    try:
                        detected = len(gait.cycles)
                    except Exception:
                        detected = None
        if detected is not None:
            return str(int(detected))
        if usable is not None:
            return str(int(usable))
        return NOT_AVAILABLE

    def _overview_hud_video_fps(self) -> str:
        """Source video / pose sequence frame rate."""
        from stablewalk.ui.overview_clinical_summary import NOT_AVAILABLE

        seq = getattr(self, "sequence", None)
        fps = getattr(seq, "fps", None) if seq is not None else None
        if fps is None:
            meta = getattr(self, "_run_metadata", None)
            fps = getattr(meta, "fps", None) if meta is not None else None
        if fps is None:
            return NOT_AVAILABLE
        try:
            value = float(fps)
        except (TypeError, ValueError):
            return NOT_AVAILABLE
        if value <= 0:
            return NOT_AVAILABLE
        if abs(value - round(value)) < 1e-6:
            return f"{value:.0f}"
        return f"{value:.2f}"

    def _overview_hud_selected_rom(self, item_id: str | None) -> str:
        """Full-session robust flexion ROM (deg) for the selected joint.

        Knee joints use ``knee_flexion_rom_deg`` — the same definition as
        Motion Analysis. Other joints use ``analyze_joint_rom`` when present.
        Never invents ROM.
        """
        from stablewalk.ui.overview_clinical_summary import NOT_AVAILABLE
        from stablewalk.ui.value_provenance import (
            ValueProvenance,
            format_na,
            store_provenance,
        )

        if not item_id:
            text = NOT_AVAILABLE
            store_provenance(
                self,
                "rom_session_robust",
                ValueProvenance(
                    metric_name="Full-session robust ROM (estimated)",
                    display_value=text,
                    raw_value=None,
                    units="degrees",
                    source_frame=None,
                    source_function="app._overview_hud_selected_rom",
                    source_array="",
                    coordinate_system="n/a",
                    filtering="",
                    scope="session",
                    tier="estimated",
                    unavailable_reason="no joint selected",
                ),
            )
            return text

        # Knees: Motion Analysis parity — knee_flexion_rom_deg on session series.
        if "knee" in item_id:
            text = self._overview_hud_rom_fallback(item_id)
            if text.startswith("N/A"):
                # Normalize to Overview wording.
                text = NOT_AVAILABLE
            store_provenance(
                self,
                "rom_session_robust",
                ValueProvenance(
                    metric_name="Full-session robust ROM (estimated)",
                    display_value=text,
                    raw_value=self._overview_hud_rom_fallback_value(item_id),
                    units="degrees (knee flexion series P10–P90)",
                    source_frame=None,
                    source_function="knee_angle_chart.knee_flexion_rom_deg",
                    source_array="_knee_angle_series",
                    coordinate_system="Pose flexion (estimated)",
                    filtering="session series robust span (P10–P90)",
                    scope="session",
                    tier="estimated",
                    unavailable_reason=(
                        "full-session ROM not available"
                        if text == NOT_AVAILABLE
                        else None
                    ),
                ),
            )
            return text

        ba = getattr(self, "_biomech_analysis", None)
        if ba is None or ba.joint_rom is None or not ba.joint_rom.joints:
            text = NOT_AVAILABLE
            store_provenance(
                self,
                "rom_session_robust",
                ValueProvenance(
                    metric_name="Full-session robust ROM (estimated)",
                    display_value=text,
                    raw_value=None,
                    units="degrees",
                    source_frame=None,
                    source_function="app._overview_hud_selected_rom",
                    source_array="BiomechanicalAnalysisResult.joint_rom",
                    coordinate_system="Root-relative pose flexion",
                    filtering="",
                    scope="session",
                    tier="estimated",
                    unavailable_reason="joint ROM analysis unavailable",
                ),
            )
            return text
        parts = item_id.split("_", 1)
        if len(parts) != 2:
            return NOT_AVAILABLE
        side, joint = parts[0], parts[1]
        entry = next(
            (j for j in ba.joint_rom.joints if j.joint == joint and j.side == side),
            None,
        )
        if entry is None or entry.rom_deg is None:
            return NOT_AVAILABLE
        rom = float(entry.rom_deg)
        conf = float(getattr(entry, "confidence", 1.0) or 0.0)
        if conf < 0.40:
            text = NOT_AVAILABLE
            store_provenance(
                self,
                "rom_session_robust",
                ValueProvenance(
                    metric_name="Full-session robust ROM (estimated)",
                    display_value=text,
                    raw_value=rom,
                    units="degrees (robust per-cycle flexion ROM)",
                    source_frame=None,
                    source_function="biomechanical.joint_rom.analyze_joint_rom",
                    source_array="BiomechanicalAnalysisResult.joint_rom",
                    coordinate_system="Root-relative pose flexion",
                    filtering="P5–P95 per cycle, session mean",
                    confidence=conf,
                    scope="session",
                    tier="estimated",
                    unavailable_reason="low ROM confidence",
                ),
            )
            return text
        text = f"{rom:.0f}\u00b0"
        store_provenance(
            self,
            "rom_session_robust",
            ValueProvenance(
                metric_name="Full-session robust ROM (estimated)",
                display_value=text,
                raw_value=rom,
                units="degrees (robust per-cycle flexion ROM)",
                source_frame=None,
                source_function="app._overview_hud_selected_rom",
                source_array="BiomechanicalAnalysisResult.joint_rom",
                coordinate_system="Root-relative pose flexion",
                filtering="P5–P95 per cycle, session mean",
                confidence=conf,
                scope="session",
                tier="estimated",
            ),
        )
        return text

    def _overview_hud_rom_fallback_value(self, item_id: str | None) -> float | None:
        if not item_id or "knee" not in item_id:
            return None
        series = getattr(self, "_knee_angle_series", None)
        if series is None:
            return None
        from stablewalk.ui.viewers.knee_angle_chart import knee_flexion_rom_deg

        if item_id.startswith("left"):
            return knee_flexion_rom_deg(series.left_deg)
        if item_id.startswith("right"):
            return knee_flexion_rom_deg(series.right_deg)
        return None

    def _overview_hud_rom_fallback(self, item_id: str | None) -> str:
        from stablewalk.ui.overview_clinical_summary import NOT_AVAILABLE

        value = self._overview_hud_rom_fallback_value(item_id)
        if value is None:
            return NOT_AVAILABLE
        return f"{value:.0f}\u00b0"

    def _update_overview_playback_hud(self, *, include_clinical: bool = True) -> None:
        """Refresh compact HUD under Original Video (text metrics + optional clinical)."""
        from stablewalk.ui.overview_clinical_summary import NOT_AVAILABLE
        from stablewalk.ui.theme import MUTED, TEXT, create_tooltip

        labels = getattr(self, "_overview_hud_value_labels", None)
        hud = getattr(self, "_overview_playback_hud", None)
        if not labels or hud is None:
            return

        # Cheap gate: only compute/paint while the Overview tab owns the sync.
        if not self._group_visible("overview"):
            return

        self._overview_hud_shown = True

        # Prefer transport playhead so phase matches the displayed video frame.
        video_frame_index = None
        try:
            from stablewalk.ui.tk.dashboard_status_bar import playhead_frame_0based

            video_frame_index = playhead_frame_0based(self)
        except Exception:
            video_frame_index = None
        player = self.skeleton_player
        if video_frame_index is None and player is not None:
            snap = player.current_snapshot()
            video_frame_index = getattr(snap, "frame_index", None) if snap else None

        item_id = getattr(self.selection, "active_item_id", None)
        if hasattr(self, "_active_dof_item_id"):
            try:
                item_id = self._active_dof_item_id() or item_id
            except Exception:
                pass

        coverage = self._overview_hud_tracking_coverage()
        if coverage is not None:
            detected_text = coverage.detected_display()
            joints_text = coverage.joint_confidence_display()
            usable_text = coverage.usable_display()
            detected_tip = coverage.detection_tooltip()
            joints_tip = coverage.joint_confidence_tooltip()
            usable_tip = coverage.usable_tooltip()
        else:
            detected_text = joints_text = usable_text = NOT_AVAILABLE
            detected_tip = (
                "Detected frames / total frames. The percentage is person-detection "
                "coverage, not accuracy."
            )
            joints_tip = (
                "Mean landmark visibility on detected frames. This is a visibility "
                "summary, not joint-presence accuracy."
            )
            usable_tip = (
                "Frames with hips, knees, and ankles visible enough for gait "
                "analysis. The percentage is a usable-frame share, not accuracy."
            )
        values = {
            "rom": self._overview_hud_selected_rom(item_id),
            "phase": self._overview_hud_gait_phase(video_frame_index),
            "detected": detected_text,
            "joints": joints_text,
            "usable": usable_text,
        }
        tips = {
            "rom": "Full-session flexion range for the selected joint.",
            "phase": "Gait phase at the current video frame (contact-derived).",
            "detected": detected_tip,
            "joints": joints_tip,
            "usable": usable_tip,
        }

        cards = getattr(self, "_overview_hud_cards", None) or {}
        captions = getattr(self, "_overview_hud_caption_labels", None) or {}
        tracking_keys = {"detected", "joints", "usable"}
        for key, text in values.items():
            lbl = labels.get(key)
            card = cards.get(key)
            if lbl is None:
                continue
            estimating = key == "phase" and text == "Estimating..."
            unavailable = text == NOT_AVAILABLE or text in ("—", "", "N/A")
            # Tracking fields stay mapped so the coverage readout remains readable.
            if card is not None and key not in tracking_keys:
                try:
                    if unavailable and not estimating:
                        card.grid_remove()
                        continue
                    card.grid()
                except tk.TclError:
                    pass
            self._configure_label_if_changed(lbl, text=text)
            try:
                lbl.configure(fg=MUTED if (unavailable or estimating) else TEXT)
                tip = tips.get(key, "")
                if estimating:
                    tip = "Phase is being estimated from contact / gait-cycle analysis."
                elif unavailable:
                    tip = f"{tips.get(key, 'Value')} — not available for this session."
                create_tooltip(lbl, tip, wraplength=360)
                cap = captions.get(key)
                if cap is not None:
                    create_tooltip(cap, tip, wraplength=360)
                if card is not None:
                    create_tooltip(card, tip, wraplength=360)
            except tk.TclError:
                pass

        # Ensure the compact HUD is mapped while Overview is visible.
        try:
            if not hud.winfo_ismapped():
                hud.grid()
        except tk.TclError:
            pass

        # Static KPI cards: never restyle during playback.
        if not include_clinical or getattr(self, "playing", False):
            return

        import time

        now = time.monotonic()
        last = float(getattr(self, "_last_overview_clinical_ts", 0.0) or 0.0)
        if (now - last) >= 0.5:
            self._last_overview_clinical_ts = now
            try:
                self._update_overview_clinical_summary()
            except Exception:
                pass

    def _move_existing_chart_playheads(
        self,
        fig,
        canvas,
        playhead_time_s: float | None,
        *,
        list_pos: int | None = None,
    ) -> bool:
        """Move cached playhead artists without rebuilding static chart data."""
        if fig is None or canvas is None or playhead_time_s is None:
            return False
        from stablewalk.ui.tk.redraw_policy import skip_unmapped_canvas_paint

        if skip_unmapped_canvas_paint(canvas):
            return False
        axes = [
            ax
            for ax in getattr(fig, "axes", ())
            if getattr(ax, "_stablewalk_playhead_artists", None)
        ]
        if not axes:
            return False
        from stablewalk.ui.viewers.chart_playhead import (
            PlayheadState,
            blit_playhead_artists,
            cache_static_chart_background,
            playhead_from_gui,
            update_chart_playhead,
            update_playhead_on_axes,
        )

        animating = bool(getattr(self, "playing", False))
        phase = float(getattr(self, "_joint_motion_playhead_phase", 0.0) or 0.0)
        if animating:
            from stablewalk.ui.viewers.chart_playhead import advance_playhead_anim_phase

            phase = advance_playhead_anim_phase(phase)
            self._joint_motion_playhead_phase = phase
        state = playhead_from_gui(
            self,
            time_s=float(playhead_time_s),
            anim_phase=phase,
            animating=animating,
        )
        if state is None:
            frame_index = 0
            player = getattr(self, "skeleton_player", None)
            if player is not None:
                try:
                    frame_index = int(player.state.frame_index) + 1
                except Exception:
                    idx = getattr(player, "current_index", None)
                    if idx is not None:
                        frame_index = int(idx) + 1
            state = PlayheadState(
                time_s=float(playhead_time_s),
                frame_index=frame_index,
                list_pos=list_pos,
                anim_phase=phase,
                animating=animating,
            )
        elif list_pos is None:
            list_pos = state.list_pos
        # Knee chart: move L/R markers + refresh value callout together.
        knee_ax = None
        for ax in axes:
            if getattr(ax, "_stablewalk_knee_series", None) is not None:
                knee_ax = ax
                break
        if knee_ax is not None and list_pos is not None:
            from stablewalk.ui.viewers.knee_angle_chart import update_knee_playhead_markers

            value_label, value_y = update_knee_playhead_markers(knee_ax, list_pos=int(list_pos))
            artists = getattr(knee_ax, "_stablewalk_playhead_artists", None)
            show_label = bool(artists and len(artists) >= 5 and artists[4] is not None)
            update_chart_playhead(
                knee_ax,
                state,
                show_label=show_label,
                value_label=value_label,
                value_y=value_y,
            )
            other = [ax for ax in axes if ax is not knee_ax]
            if other:
                update_playhead_on_axes(other, state)
        else:
            update_playhead_on_axes(axes, state)

        # Prefer blit (static pixels + cursor) over a full canvas redraw.
        if getattr(self, "playing", False):
            if blit_playhead_artists(fig, canvas, axes):
                painted = getattr(self, "_pb_painted_canvas_ids", None)
                if painted is not None:
                    painted.add(id(canvas))
                return True
            # First play tick (or after resize): snapshot static+cursor for next blits.
            if cache_static_chart_background(fig, canvas, axes):
                painted = getattr(self, "_pb_painted_canvas_ids", None)
                if painted is not None:
                    painted.add(id(canvas))
                return True
        self._paint_canvas_now(canvas, idle=True)
        return True

    # Canvas attribute -> panel-visibility group (see ``_visible_panel_groups``).
    _CANVAS_GROUPS = (
        ("canvas_dof_traj", "trajectory"),
        ("canvas_dof_traj_overview", "trajectory"),
        ("chart_canvas", "knee"),
        ("canvas_contact_gait", "contact"),
        ("canvas_biomech", "biomech"),
        ("canvas_joint_motion", "joint_motion"),
        ("canvas_motion_workspace", "joint_motion"),
        ("canvas_robot", "skeleton"),
    )

    def _active_tab_widget(self):
        """Return the currently selected dashboard notebook tab widget (or None)."""
        notebook = getattr(self, "_dashboard_notebook", None)
        if notebook is None:
            return None
        try:
            return notebook.nametowidget(notebook.select())
        except (tk.TclError, KeyError):
            return None

    def _visible_panel_groups(self) -> frozenset[str]:
        """Panel groups on the active notebook tab (playback never paints the rest)."""
        from stablewalk.ui.tk.playback_active_tab import playback_groups_for_active_tab

        return playback_groups_for_active_tab(self)

    def _widget_mapped(self, widget) -> bool:
        """True when a Tk widget is currently mapped on screen."""
        if widget is None:
            return False
        try:
            return bool(widget.winfo_ismapped())
        except tk.TclError:
            return False

    def _group_visible(self, group: str) -> bool:
        """True if ``group`` should be redrawn on the active tab (or playback gate).

        During playback ``_pb_visible`` is the authority. Outside that gate we
        still consult ``_visible_panel_groups`` so progressive paints and
        background refreshes never rebuild charts on inactive notebook tabs.
        Unmapped canvases never paint even if their group is tab-active.
        """
        vis = getattr(self, "_pb_visible", None)
        in_set = False
        if vis is not None:
            in_set = group in vis
        else:
            try:
                in_set = group in self._visible_panel_groups()
            except Exception:
                return True
        if not in_set:
            return False
        # Hard gate: never rasterize widgets the user cannot see.
        if group == "video":
            label = getattr(self, "video_label", None)
            host = getattr(self, "video_display_host", None)
            if label is None and host is None:
                return True
            return self._widget_mapped(label) or self._widget_mapped(host)
        if group == "skeleton":
            canvas = getattr(self, "canvas_3d", None)
            host = getattr(self, "skel_canvas_host", None)
            widget = None
            if canvas is not None:
                try:
                    widget = canvas.get_tk_widget()
                except Exception:
                    widget = None
            if widget is None and host is None:
                return True
            return self._widget_mapped(widget) or self._widget_mapped(host)
        if group == "trajectory":
            saw_canvas = False
            for attr in ("canvas_dof_traj", "canvas_dof_traj_overview"):
                canvas = getattr(self, attr, None)
                if canvas is None:
                    continue
                saw_canvas = True
                try:
                    if self._widget_mapped(canvas.get_tk_widget()):
                        return True
                except Exception:
                    continue
            # Lazy ensure: allow until a canvas exists and is unmapped.
            return not saw_canvas
        if group in ("knee", "biomech", "contact", "joint_motion", "motion"):
            want = "joint_motion" if group == "motion" else group
            saw_canvas = False
            for attr, mapped_group in self._CANVAS_GROUPS:
                if mapped_group != want:
                    continue
                canvas = getattr(self, attr, None)
                if canvas is None:
                    continue
                saw_canvas = True
                try:
                    if self._widget_mapped(canvas.get_tk_widget()):
                        return True
                except Exception:
                    continue
            if not saw_canvas:
                return True
            # First paint: the workspace canvas is created in the same turn and
            # is often still unmapped when refresh runs. ``in_set`` already
            # means this notebook tab owns the group.
            if group in ("motion", "joint_motion"):
                try:
                    from stablewalk.ui.tk.dashboard_notebook import (
                        is_motion_tab_selected,
                    )

                    if is_motion_tab_selected(self):
                        return True
                except Exception:
                    pass
            if group in ("biomech", "contact"):
                return True
            return False
        return True

    def refresh_motion_analysis_tab(self) -> None:
        """Restore and repaint Motion Analysis without rebuilding the tab shell.

        Layout is created once at dashboard build; lazy chart canvases are ensured
        at most once. Call when analysis completes, the selected joint changes, or
        the user opens the Motion Analysis tab. Never call from the per-frame
        playback path — that only moves playheads / artists.
        """
        if getattr(self, "_refreshing_motion_analysis_tab", False):
            return
        if getattr(self, "_sync_all_guard", False):
            return
        self._refreshing_motion_analysis_tab = True
        try:
            from stablewalk.ui.tk.dashboard_layout import (
                _ensure_trajectory_canvas_gridded,
                _hide_trajectory_debug_placeholder,
            )
            from stablewalk.ui.tk.dashboard_notebook import is_motion_tab_selected
            from stablewalk.ui.tk.layout_playback import force_layout_pass

            # Allow scroll / figure fit while playing so opening Motion mid-play
            # does not leave the canvas window at 1x1 (blank tab).
            with force_layout_pass(self):
                ensure = getattr(self, "_ensure_motion_tab_charts", None)
                if callable(ensure):
                    try:
                        ensure()
                    except Exception:
                        pass

                # Propagate current session + selected joint into panel chrome.
                try:
                    self._sync_dof_analysis_panel_state()
                except Exception:
                    pass

                sync_motion = getattr(self, "_sync_motion_scroll", None)
                if callable(sync_motion):
                    try:
                        sync_motion()
                    except Exception:
                        pass

                # Restore trajectory canvas mapping if it was ungridded.
                if getattr(self, "canvas_dof_traj", None) is not None:
                    try:
                        _ensure_trajectory_canvas_gridded(self.canvas_dof_traj)
                        _hide_trajectory_debug_placeholder(self)
                    except Exception:
                        pass
                    lift = getattr(self, "_lift_trajectory_canvas", None)
                    if callable(lift):
                        try:
                            lift()
                        except Exception:
                            pass

                has_session = bool(
                    getattr(self, "skeleton_player", None)
                    and getattr(self.skeleton_player, "frame_count", 0) > 0
                )
                if has_session:
                    try:
                        ensure_ws = getattr(self, "_ensure_motion_workspace_chart", None)
                        if callable(ensure_ws):
                            ensure_ws()
                    except Exception:
                        pass
                    try:
                        self._update_joint_motion_analysis_chart(force_rebuild=True)
                    except Exception:
                        pass
                    try:
                        self._update_chart()
                    except Exception:
                        pass
                    try:
                        self._refresh_selected_dof_trajectory_3d(force_draw=True)
                    except Exception:
                        pass
                    try:
                        self._render_dof_traj_canvas(force=True)
                    except Exception:
                        pass
                    try:
                        reflow_motion = getattr(
                            self, "_reflow_motion_workspace", None
                        )
                        if callable(reflow_motion):
                            reflow_motion()
                    except Exception:
                        pass
                elif (
                    getattr(self, "canvas_dof_traj", None) is not None
                    and not getattr(self, "_traj_startup_test_drawn", False)
                ):
                    try:
                        from stablewalk.ui.tk.dashboard_layout import (
                            _draw_trajectory_startup_test,
                        )

                        _draw_trajectory_startup_test(self)
                    except Exception:
                        pass

                # Fit figures only when the tab is the one on screen (or just
                # became visible) so we do not thrash layout on every joint
                # change while the user stays on Overview.
                if is_motion_tab_selected(self):
                    fit = getattr(self, "_fit_dof_traj_canvas", None)
                    if callable(fit):
                        try:
                            fit()
                        except Exception:
                            pass
                    if callable(sync_motion):
                        try:
                            sync_motion()
                        except Exception:
                            pass
        finally:
            self._refreshing_motion_analysis_tab = False

    def _refresh_active_tab_charts(self) -> None:
        """Bring the newly visible tab's charts to the current frame.

        Because playback only redraws widgets on the active tab, switching tabs
        must repaint the destination tab once so it is never left stale.
        While playing, never rebuild full charts — only sync artists from cache.
        """
        if getattr(self, "_sync_all_guard", False):
            return
        from stablewalk.ui.tk.perf_diagnostics import timed_section

        with timed_section(self, "tab_activate", warn_label="tab activate paint"):
            try:
                tab = self._active_tab_widget()
                # Compare is cache/metrics-only — a full playback sync here freezes
                # the UI on every Overview↔Compare switch (black / Not Responding).
                if tab is not None and tab is getattr(self, "_tab_compare", None):
                    try:
                        self._flush_dirty_analysis_panels({"compare"})
                    except Exception:
                        pass
                    return
                playing = bool(getattr(self, "playing", False))
                motion = getattr(self, "_tab_motion", None)
                if tab is not None and motion is not None and tab is motion:
                    if playing:
                        # Canvases ensured in activate_tab; move playheads only.
                        try:
                            self._sync_all_to_frame(force_draw=False)
                        except Exception:
                            pass
                        return
                    # Flush deferred knee/motion graphs before a full refresh.
                    try:
                        if "knee" in getattr(self, "_analysis_panel_dirty", set()):
                            self._rebuild_knee_angle_series()
                            self._apply_default_knee_chart_mode()
                            self._update_chart()
                            self._analysis_panel_dirty.discard("knee")
                    except Exception:
                        pass
                    self.refresh_motion_analysis_tab()
                    self._analysis_panel_dirty.discard("motion")
                    self._analysis_panel_dirty.discard("joint_motion")
                    try:
                        self._flush_dof_position_table_if_dirty()
                    except Exception:
                        pass
                    return
                # Overview while playing: reuse the lite path (no Motion/Biomech charts).
                if (
                    tab is not None
                    and tab is getattr(self, "_tab_overview", None)
                    and playing
                ):
                    try:
                        self._sync_overview_playback_lite(force_draw=True)
                    except Exception:
                        pass
                    return
                biomech = getattr(self, "_tab_biomechanics", None)
                if tab is not None and biomech is not None and tab is biomech:
                    if playing:
                        try:
                            self._sync_all_to_frame(force_draw=False)
                        except Exception:
                            pass
                        return
                    try:
                        self._flush_dirty_analysis_panels({"biomech", "contact"})
                    except Exception:
                        pass
                results = getattr(self, "_tab_results_summary", None)
                if tab is not None and results is not None and tab is results:
                    try:
                        self._flush_dirty_analysis_panels({"summary"})
                    except Exception:
                        pass
                advanced = getattr(self, "_tab_advanced_outer", None)
                if tab is not None and advanced is not None and tab is advanced:
                    try:
                        from stablewalk.ui.tk.dashboard_notebook import (
                            realize_advanced_tab_shell,
                        )

                        realize_advanced_tab_shell(self)
                    except Exception:
                        pass
                    return
            except Exception:
                pass
            player = getattr(self, "skeleton_player", None)
            if player is None or getattr(player, "frame_count", 0) <= 0:
                return
            try:
                self._sync_all_to_frame(force_draw=True)
            except Exception:
                pass

    def _current_chart_playhead_time_s(self) -> float | None:
        """Timestamp for the paused/playing pose frame, including the last frame."""
        from stablewalk.ui.viewers.chart_playhead import playhead_from_gui

        cursor = playhead_from_gui(self)
        if cursor is not None:
            return float(cursor.time_s)
        player = getattr(self, "skeleton_player", None)
        idx = None
        if player is not None:
            try:
                idx = int(player.state.frame_index)
            except Exception:
                idx = getattr(player, "current_index", None)
        if idx is None:
            idx = getattr(self, "current_pos", None)
        if idx is None:
            return None
        try:
            index = int(idx)
        except (TypeError, ValueError):
            return None
        pose_indices = getattr(self, "pose_indices", None) or ()
        if pose_indices:
            if index < 0:
                index = 0
            if index >= len(pose_indices):
                index = len(pose_indices) - 1
            frame_index = pose_indices[index]
        else:
            frame_index = index
        ts = self._timestamp_for_frame_index(int(frame_index))
        if ts is not None:
            return float(ts)
        seq = getattr(self, "sequence", None)
        if seq is None:
            return None
        for frame in getattr(seq, "frames", ()) or ():
            if int(getattr(frame, "frame_index", -1)) == int(frame_index):
                try:
                    return float(frame.timestamp_s)
                except (TypeError, ValueError):
                    return None
        return None

    def _timestamp_for_frame_index(self, frame_index: int) -> float | None:
        """Timestamp (s) for a pose frame index, using a cached index map.

        Replaces a per-frame O(n) linear scan over ``sequence.frames`` with an
        O(1) dict lookup built once per loaded sequence.
        """
        seq = self.sequence
        if not seq:
            return None
        key = id(seq)
        if self._frame_ts_cache_key != key:
            self._frame_ts_cache = {
                f.frame_index: f.timestamp_s for f in seq.frames
            }
            self._frame_ts_cache_key = key
        return self._frame_ts_cache.get(frame_index)

    def _flush_playback_canvases(self) -> None:
        """Force playback-linked canvases to paint the current artists now.

        The 3D skeleton canvas is already painted inside
        ``_update_interactive_skeleton`` every UI tick, so it is intentionally not
        repainted here (that would double the most expensive draw). Analytical
        canvases paint only on UI ticks and only when their tab is visible.
        """
        if not getattr(self, "_pb_ui_tick", True):
            return
        if not getattr(self, "_pb_heavy", True):
            return
        for attr, group in self._CANVAS_GROUPS:
            if not self._group_visible(group):
                continue
            canvas = getattr(self, attr, None)
            if id(canvas) in getattr(self, "_pb_painted_canvas_ids", set()):
                continue
            # Coalesce secondary chart paints while playing.
            self._paint_canvas_now(canvas, idle=True)

    def _sync_playback_video_only(self) -> None:
        """Advance-linked video + frame chrome only (no graph / panel repaint)."""
        if getattr(self, "_sync_all_guard", False):
            return
        player = self.skeleton_player
        if player is None:
            return
        self._sync_all_guard = True
        try:
            from stablewalk.ui.playback_state import mirror_playback_mirrors, read_playback_state

            self._pb_ui_tick = False
            self._pb_heavy = False
            self._pb_visible = self._visible_panel_groups()
            state = read_playback_state(self)
            if state is None:
                return
            frame_f = float(state.frame_float)
            frame_i = int(state.current_frame_index)
            mirror_playback_mirrors(self, state)
            self._last_video_list_pos = frame_i
            try:
                if int(round(float(self.frame_var.get()))) != frame_i:
                    self.frame_var.set(frame_i)
            except (tk.TclError, TypeError, ValueError):
                pass
            if self.sequence and self.pose_indices:
                self._show_pose_at(
                    frame_f,
                    force_draw=False,
                    skeleton_only=False,
                    video_only=True,
                )
            self._sync_transport_labels()
        finally:
            self._sync_all_guard = False
            self._pb_ui_tick = True
            self._pb_heavy = True
            self._pb_visible = None

    def _update_overview_path_tip_only(self) -> bool:
        """Move Overview 3D current marker + partial path without rebuilding axes.

        Full trajectory / limits / ticks are built once on joint change
        (``_refresh_overview_trajectory_dock``). Playback only updates artists.
        """
        if not getattr(self, "_overview_traj_dock_visible", False):
            return False
        ax = getattr(self, "ax_dof_traj_overview", None)
        canvas = getattr(self, "canvas_dof_traj_overview", None)
        if ax is None or canvas is None:
            return False
        item_id = self._active_dof_item_id()
        if not item_id or not self.selection.selected:
            return False
        recording = self._analysis_motion_recording()
        if recording is None or self.skeleton_player is None:
            return False
        now = time.monotonic()
        interval = float(getattr(self, "_traj_redraw_interval_s", 0.0) or 0.0)
        if interval > 0.0 and (
            now - float(getattr(self, "_last_traj_redraw_ts", 0.0))
        ) < interval:
            return True
        self._last_traj_redraw_ts = now

        from stablewalk.ui.tk.dashboard_status_bar import playhead_frame_0based
        from stablewalk.ui.dof_selection import anchor_joint_for_item
        from stablewalk.ui.viewers.dof_trajectory_3d import update_trajectory_current_tip

        frame_i = playhead_frame_0based(self)
        if frame_i is None:
            frame_i = int(self.skeleton_player.state.frame_index)
        # Dirty-region: same discrete frame → tip/inspector already current.
        if (
            self.playing
            and getattr(self, "_last_traj_tip_frame", None) == int(frame_i)
            and getattr(self, "_last_traj_tip_item", None) == item_id
        ):
            return True
        tip_snapshot = recording.snapshot_at(frame_i)
        if tip_snapshot is None:
            tip_snapshot = self.skeleton_player.current_snapshot()
        if tip_snapshot is None:
            return False
        joint_id = anchor_joint_for_item(item_id)
        if not joint_id:
            return False
        # Overview dock uses Root-relative (same as full dock draw).
        if not update_trajectory_current_tip(
            ax,
            None,
            tip_snapshot=tip_snapshot,
            joint_id=joint_id,
            recording=recording,
            coord_mode="Root-relative",
            list_pos=frame_i,
            update_path=True,
        ):
            # Artists / static cache not ready — one full dock draw, then tip-only.
            # Cooldown prevents mid-play full rebuild storms when tip misses.
            now_cd = time.monotonic()
            if now_cd < float(
                getattr(self, "_traj_full_rebuild_cooldown_until", 0.0) or 0.0
            ):
                return False
            self._traj_full_rebuild_cooldown_until = now_cd + 0.75
            try:
                self._refresh_overview_trajectory_dock(force_draw=False)
            except Exception:
                return False
            return True
        self._path_marker_list_pos = int(frame_i)
        self._overview_path_list_pos = int(frame_i)
        self._last_traj_tip_frame = int(frame_i)
        self._last_traj_tip_item = item_id
        # Coordinate inspector only — never Joint Information / HUD chrome.
        try:
            self._refresh_overview_trajectory_readout(item_id)
        except Exception:
            pass
        self._paint_canvas_now(canvas, idle=True)
        return True

    def _sync_overview_playback_lite(self, *, force_draw: bool = True) -> None:
        """Overview play: only live surfaces (video/skeleton/tip/timeline/inspector).

        Metric cards, graphs, Joint Information, HUD KPIs, legends, and layouts
        stay frozen — see ``playback_render_policy.LIVE_PLAY_SURFACES``.
        Video/skeleton track ``TARGET_PLAY_HZ``; tip/inspector/phase ≤ ``MAX_LIVE_WIDGET_HZ``.
        """
        from stablewalk.ui.playback_state import mirror_playback_mirrors, read_playback_state
        from stablewalk.ui.tk.playback_render_policy import live_widgets_due

        player = self.skeleton_player
        if player is None:
            return
        state = read_playback_state(self)
        if state is None:
            return
        frame_f = float(state.frame_float)
        frame_i = int(state.current_frame_index)
        mirror_playback_mirrors(self, state)
        self._graph_cursor_list_pos = frame_i
        self._path_marker_list_pos = frame_i
        self._overview_path_list_pos = frame_i
        try:
            if int(round(float(self.frame_var.get()))) != frame_i:
                self.frame_var.set(frame_i)
        except (tk.TclError, TypeError, ValueError):
            pass
        self._playback_sync_immediate = True
        # Charts / trees / HUD off; video + skeleton + tip stay on.
        self._pb_heavy = False
        self._pb_ui_tick = True
        # Tip / inspector / phase capped at 30 FPS; video + skeleton stay at play Hz.
        update_live = live_widgets_due(self)
        try:
            if self.sequence and self.pose_indices:
                self._show_pose_at(
                    frame_f, force_draw=force_draw, skeleton_only=False
                )
            else:
                self._update_interactive_skeleton(force_draw=force_draw)
            if update_live and self._group_visible("trajectory"):
                from stablewalk.ui.tk.render_diagnostics import timed_render

                with timed_render(self, "trajectory"):
                    # Tip move + coordinate inspector (cached path mesh).
                    self._update_overview_path_tip_only()
            # Timeline cursor only — no metric / Joint Information / HUD chrome.
            self._sync_transport_labels()
            # Live gait phase (precomputed cache lookup — O(1)), ≤30 FPS.
            if update_live:
                try:
                    from stablewalk.ui.tk.gait_phase_indicator import (
                        update_gait_phase_indicator,
                    )

                    update_gait_phase_indicator(self, frame_i)
                    self._update_overview_playback_hud(include_clinical=False)
                except Exception:
                    pass
                # O(1) current-frame lookup using the cached floor plane and
                # session statistics. This keeps the Overview readouts live
                # without adding analysis work to playback.
                try:
                    snapshot = player.current_snapshot()
                    if snapshot is not None:
                        self._refresh_bilateral_ground_clearance(
                            snapshot=snapshot,
                            current_frame_only=True,
                        )
                except Exception:
                    pass
                if bool(getattr(self, "_overview_signal_spectral_expanded", False)):
                    from stablewalk.ui.tk.playback_render_policy import (
                        PLAY_CHART_HZ,
                        rate_limited_due,
                    )

                    if rate_limited_due(
                        self, "overview_sigspec_playhead", max_hz=PLAY_CHART_HZ
                    ):
                        upd_os = getattr(
                            self, "_update_overview_signal_spectral_playhead", None
                        )
                        if callable(upd_os):
                            try:
                                upd_os()
                            except Exception:
                                pass
        finally:
            self._playback_sync_immediate = False
            self._pb_heavy = True

    def _sync_all_to_frame(self, *, force_draw: bool = True) -> None:
        """Refresh visible playback panels to skeleton_player's current frame.

        During play this runs every timer tick so video, skeleton, trajectory,
        chart playheads, and live metrics stay on one shared PlaybackState.
        Overview play uses a lite path (video + skeleton + path tip + live HUD)
        and never rebuilds off-tab Motion/Biomech charts.
        """
        if getattr(self, "_sync_all_guard", False):
            return
        player = self.skeleton_player
        if player is None:
            return

        self._sync_all_guard = True
        try:
            # Every play tick is a UI tick — keep playheads/tips moving.
            if self.playing:
                now = time.monotonic()
                self._last_heavy_chart_ts = now
                self._last_ui_sync_ts = now
            self._pb_ui_tick = True
            self._pb_heavy = True
            self._pb_painted_canvas_ids = set()
            # Restrict redraws to the widgets on the active tab.
            self._pb_visible = self._visible_panel_groups()
            if self.playing:
                from stablewalk.ui.tk.playback_active_tab import (
                    mark_hidden_tab_panels_dirty,
                )

                mark_hidden_tab_panels_dirty(self)

            from stablewalk.ui.playback_state import (
                assert_playback_synced_if_debug,
                begin_playback_commit,
                mirror_playback_mirrors,
                quantize_playback_to_discrete_frame,
                read_playback_state,
                store_committed_playback,
            )

            if not self.playing:
                quantize_playback_to_discrete_frame(self)
            generation = begin_playback_commit(self)
            state = read_playback_state(self)
            if state is None:
                return
            store_committed_playback(self, state, generation)
            frame_f = float(state.current_frame_index if not self.playing else state.frame_float)
            frame_i = int(state.current_frame_index)
            mirror_playback_mirrors(self, state)
            try:
                if int(round(float(self.frame_var.get()))) != frame_i:
                    self.frame_var.set(frame_i)
            except (tk.TclError, TypeError, ValueError):
                pass

            if not self._pb_visible:
                # Advanced (and other non-playback pages) own no live canvases.
                # Drawing hidden TkAgg widgets from here can leave the selected
                # tab as a solid black pane. Still refresh the detachable
                # Coordinate Inspector so frame seeks from Advanced work.
                self._sync_transport_labels(state)
                try:
                    self._refresh_overview_trajectory_readout(self._active_dof_item_id())
                except Exception:
                    pass
                return

            # Overview play: live surfaces only (video/skeleton/tip/timeline/inspector).
            if self.playing and self._group_visible("overview"):
                self._sync_overview_playback_lite(force_draw=False)
                self._sync_transport_labels(state)
                from stablewalk.ui.tk.render_diagnostics import maybe_log_perf_timings

                maybe_log_perf_timings(self)
                return
            if not self.playing:
                self._publish_frame_changed(frame_f, frame_i)
            self._graph_cursor_list_pos = frame_i
            self._path_marker_list_pos = frame_i
            self._overview_path_list_pos = frame_i

            if not self.playing:
                self._last_data_refresh = 0.0
                self._last_realtime_refresh = 0.0
                self._last_dof_table_refresh = 0.0
            self._playback_sync_immediate = True
            # Off-Overview play: tip/playhead only — never graph geometry rebuilds.
            if self.playing:
                self._pb_heavy = False

            need_pose = self._group_visible("video") or self._group_visible("skeleton")
            if need_pose:
                if self.sequence and self.pose_indices:
                    self._show_pose_at(
                        frame_f,
                        force_draw=force_draw and not self.playing,
                        skeleton_only=False,
                    )
                else:
                    self._update_interactive_skeleton(
                        force_draw=force_draw and not self.playing
                    )

            # Timeline cursor on cached chart geometries (no full redraw).
            if self.playing:
                from stablewalk.ui.tk.playback_render_policy import (
                    PLAY_CHART_HZ,
                    rate_limited_due,
                )

                move_charts = rate_limited_due(self, "chart_playheads", max_hz=PLAY_CHART_HZ)
                if move_charts:
                    if self._group_visible("knee"):
                        try:
                            self._update_chart(playhead_list_pos=frame_i)
                        except Exception:
                            pass
                    if self._group_visible("joint_motion") or self._group_visible("motion"):
                        try:
                            self._update_joint_motion_analysis_chart(
                                playhead_list_pos=frame_i, force_rebuild=False
                            )
                        except Exception:
                            pass
                    if self._group_visible("biomech") or self._group_visible("contact"):
                        try:
                            from stablewalk.ui.playback_state import (
                                source_frame_for_list_pos,
                            )

                            source_frame_i = source_frame_for_list_pos(self, frame_i)
                            playhead_t = self._timestamp_for_frame_index(
                                frame_i if source_frame_i is None else source_frame_i
                            )
                            if self._group_visible("contact"):
                                self._update_contact_gait_chart(playhead_time_s=playhead_t)
                            if self._group_visible("biomech"):
                                self._update_biomechanics_chart(playhead_time_s=playhead_t)
                        except Exception:
                            pass

            # Trajectory tip from cached path mesh (full dock only when paused).
            if self._group_visible("trajectory"):
                from stablewalk.ui.tk.render_diagnostics import timed_render

                with timed_render(self, "trajectory"):
                    if self._group_visible("motion"):
                        self._refresh_motion_trajectory_on_frame(
                            force_draw=force_draw and not self.playing,
                            refresh_overview=False,
                        )
                    if (
                        self._group_visible("overview")
                        and getattr(self, "_overview_traj_dock_visible", False)
                    ):
                        try:
                            if not self._update_overview_path_tip_only():
                                if not self.playing:
                                    self._refresh_overview_trajectory_dock(
                                        force_draw=False
                                    )
                        except Exception:
                            pass
            elif (
                not self.playing
                and self._group_visible("overview")
                and getattr(self, "_overview_traj_dock_visible", False)
            ):
                try:
                    if not self._update_overview_path_tip_only():
                        self._sync_overview_traj_playhead_labels()
                except Exception:
                    pass

            self._sync_transport_labels(state)
            if self._group_visible("motion") or self._group_visible("joint_motion"):
                try:
                    self._update_motion_workspace_stats()
                except Exception:
                    pass
            if self._group_visible("summary"):
                try:
                    from stablewalk.ui.tk.dashboard_summary import (
                        sync_results_playhead,
                    )

                    sync_results_playhead(self)
                except Exception:
                    pass
            # Gait phase indicator: O(1) cache lookup (Overview scrub + other tabs).
            if self._group_visible("overview"):
                try:
                    from stablewalk.ui.tk.gait_phase_indicator import (
                        update_gait_phase_indicator,
                    )

                    update_gait_phase_indicator(self, frame_i, force=not self.playing)
                except Exception:
                    pass
                if bool(getattr(self, "_overview_signal_spectral_expanded", False)):
                    upd_os = getattr(
                        self, "_update_overview_signal_spectral_playhead", None
                    )
                    if callable(upd_os):
                        try:
                            upd_os()
                        except Exception:
                            pass
            # Status bar follows the playhead even while playing (cheap text).
            # Overview HUD stays pause/scrub only to keep the play timer light.
            if not self.playing:
                try:
                    self._update_overview_playback_hud()
                except Exception:
                    pass
            if force_draw or (self.playing and getattr(self, "_pb_heavy", False)):
                self._flush_playback_canvases()
            from stablewalk.ui.tk.render_diagnostics import maybe_log_perf_timings

            maybe_log_perf_timings(self)
            assert_playback_synced_if_debug(self, state)
        finally:
            self._playback_sync_immediate = False
            self._sync_all_guard = False
            self._pb_heavy = True
            self._pb_ui_tick = True
            self._pb_visible = None
            self._pb_painted_canvas_ids = None

    def _ghost_snapshots_for(self, current_index: int):
        """Previous poses (oldest → newest) for the fading motion echo."""
        from stablewalk.ui.skeleton_display_filter import ensure_display_filter

        if self.playing:
            n = int(getattr(self, "_skeleton_ghost_count_play", 4) or 0)
        else:
            n = int(getattr(self, "_skeleton_ghost_count", 0) or 0)
        if n <= 0:
            return []
        filtered = ensure_display_filter(self).ghost_snapshots(n)
        if filtered:
            return filtered
        player = self.skeleton_player
        if player is None:
            return []
        ghosts = []
        step = 2 if self.playing else 1
        for k in range(n * step, 0, -step):
            idx = current_index - k
            if idx < 0:
                continue
            g = player.snapshot_at(idx)
            if g is not None:
                ghosts.append(g)
        return ghosts

    def _update_interactive_skeleton(self, *, force_draw: bool = False) -> None:
        """Draw the human skeleton for the current playback position."""
        if self.playing and not self._group_visible("skeleton"):
            return
        ensure = getattr(self, "_ensure_skeleton_figure", None)
        if callable(ensure):
            try:
                if not ensure():
                    return
            except Exception:
                return
        if not self.skeleton_player:
            self.ax_3d.cla()
            setup_skeleton_axes(self.ax_3d, display_mode=self._skeleton_display_mode_key())
            self.ax_3d.text(
                0.5, 0.5, "Load a walking video to begin", transform=self.ax_3d.transAxes,
                ha="center", color=MUTED, fontsize=11,
            )
            self.canvas_3d.draw_idle()
            self._dof_step_previews = []
            self._refresh_dof_step_panel()
            self._skeleton_artists_dirty = True
            return

        snap = self.skeleton_player.current_snapshot()
        if not snap:
            return
        list_pos = int(self.skeleton_player.state.frame_index)

        is_playing = self.skeleton_player.state.playing
        is_stopped = self.skeleton_player.state.stopped
        show_detail = not is_playing  # Pause / Stop: crisp draw
        if is_playing:
            mode = "PLAYING"
        elif is_stopped:
            mode = "STOPPED"
        else:
            mode = "PAUSED"

        labeled = self._skeleton_joint_labels()
        skel_status = getattr(self, "lbl_skeleton_status", None)
        if skel_status is not None:
            self._configure_label_if_changed(skel_status, text=mode)

        highlight = self._resolve_highlight_joints()
        heavy = getattr(self, "_pb_heavy", True)
        ui_tick = getattr(self, "_pb_ui_tick", True)
        # During play: skip table rebuilds and non-graph chrome. Keep selected-
        # joint / visible-graph work on UI ticks only.
        tables_visible = self._group_visible("overview") or self._group_visible("motion")
        if heavy and ui_tick and tables_visible and not is_playing:
            self._refresh_realtime_analysis(snapshot=snap, force_draw=force_draw)
        self._overview_video_frame_index = snap.frame_index
        playhead_t = self._timestamp_for_frame_index(snap.frame_index)
        if heavy and ui_tick and self._group_visible("contact"):
            from stablewalk.ui.tk.render_diagnostics import timed_render

            with timed_render(self, "chart"):
                self._update_contact_gait_chart(playhead_time_s=playhead_t)
        if heavy and ui_tick and self._group_visible("biomech"):
            from stablewalk.ui.tk.render_diagnostics import timed_render

            with timed_render(self, "chart"):
                self._update_gait_cycle_panel(self._gait_cycle, frame_index=list_pos)
                self._update_biomechanics_chart(playhead_time_s=playhead_t)
                # Panel: full bind on analysis change; lite stability update while playing.
                self._update_biomechanics_panel(frame_index=list_pos)
        # Results / debug overlays remain pause-only. Foot clearance uses its
        # cached O(1) current-frame path during Overview playback.
        if heavy and not is_playing and self._group_visible("summary"):
            self._update_analysis_summary_panel(frame_index=list_pos, timestamp_s=playhead_t)
        if (
            not is_playing
            and (self._group_visible("overview") or self._group_visible("motion"))
        ):
            self._refresh_bilateral_ground_clearance(
                snapshot=snap,
                current_frame_only=getattr(self, "_foot_clearance_live_cache", None)
                is not None,
            )
        if heavy and not is_playing and self._group_visible("skeleton"):
            from stablewalk.ui.overview_frame_consistency import (
                log_overview_frame_consistency_if_enabled,
            )

            log_overview_frame_consistency_if_enabled(
                self,
                snapshot=snap,
                gait_result=self._gait_cycle,
                clearance_frame_index=getattr(self, "_foot_clearance_frame_index", None),
            )
            self._assert_overview_contact_clearance_sync_debug(snap)
        # Skeleton labels share the same current-frame clearance model.
        if not is_playing:
            self._last_skeleton_foot_labels = self._foot_skeleton_labels_for_frame(
                snap.frame_index
            )
        # The 3D skeleton is the single most expensive draw. Skip it entirely
        # when its host is not mapped / not on the active tab.
        if not self._group_visible("skeleton"):
            return

        from stablewalk.ui.skeleton_display_filter import ensure_display_filter

        # Always advance the display filter so temporal smoothing stays continuous,
        # even on ticks where we skip the Matplotlib paint.
        display_snap = ensure_display_filter(self).filter_snapshot(snap)

        # Keep skeleton in lockstep with video — never skip a play tick.
        if is_playing and not ui_tick:
            return
        if is_playing:
            self._skel_play_tick = int(getattr(self, "_skel_play_tick", 0)) + 1

        com_ov, poly_ov, dir_ov, foot_c, support_type, stability_state, com_proj, com_trail, com_vel = (
            self._biomech_overlays_for_frame(snap.frame_index)
        )
        ghosts = self._ghost_snapshots_for(snap.frame_index)
        # Rebuild axes only when dirty (load / mode change); reuse limb meshes.
        mode_key = self._skeleton_display_mode_key()
        if mode_key == "gait_16":
            from stablewalk.models.gait_wireframe import (
                project_snapshot_to_gait_wire,
                session_body_scale,
            )

            scale = float(getattr(self, "_gait_wire_body_scale", 0.0) or 0.0)
            if scale <= 1e-8:
                scale = session_body_scale(getattr(self, "gait_motion", None))
                self._gait_wire_body_scale = scale
            display_snap = project_snapshot_to_gait_wire(
                display_snap, body_scale=scale
            )
        skel_dirty = bool(getattr(self, "_skeleton_artists_dirty", True))
        if mode_key != getattr(self, "_skeleton_artists_mode", None):
            skel_dirty = True
        moment = self._selected_asymmetric_moment()
        highlight_color = None
        dim_unhighlighted = True
        match_snap = None
        diff_pairs = None
        motion_arrows = self._motion_arrows_from_previews() if show_detail else None
        if moment is not None:
            from stablewalk.ui.asymmetric_moments_view import (
                ASYMMETRY_HIGHLIGHT_COLOR,
                moment_difference_pairs,
                moment_measurement_labels,
            )

            highlight_color = ASYMMETRY_HIGHLIGHT_COLOR
            dim_unhighlighted = False
            labeled = moment_measurement_labels(moment, mode_key)
            motion_arrows = None
            if show_detail:
                raw_match = self._snapshot_nearest_time(float(moment.match_time_s))
                if raw_match is not None:
                    match_snap = raw_match
                    if mode_key == "gait_16":
                        match_snap = project_snapshot_to_gait_wire(
                            raw_match, body_scale=scale
                        )
                    diff_pairs = moment_difference_pairs(moment, mode_key)
        from stablewalk.ui.tk.render_diagnostics import timed_render

        with timed_render(self, "skeleton"):
            draw_gait_skeleton(
                self.ax_3d,
                display_snap,
                clear=skel_dirty,
                paused=show_detail,
                # Hands skipped while playing; limb capsules stay (pooled artists).
                lite_draw=bool(is_playing),
                ghost_snapshots=ghosts,
                trail_joints=None if moment is not None else (highlight if show_detail else None),
                show_labels=False,
                show_legend=False,
                highlight_joints=highlight,
                hover_joint=self._hover_joint if show_detail else None,
                labeled_joints=labeled,
                motion_arrows=motion_arrows,
                title="",
                display_mode=mode_key,
                ground_floor_y=self._ground_floor_y_for_skeleton(),
                foot_skeleton_labels=None,
                foot_contact=foot_c,
                com_overlay=com_ov,
                com_projection_xz=com_proj,
                com_trail=com_trail,
                com_velocity=com_vel,
                support_polygon=poly_ov,
                support_type=support_type,
                gait_direction=dir_ov,
                stability_state=stability_state,
                show_ground_plane=self._overlay_enabled("var_overlay_ground"),
                show_com=self._overlay_enabled("var_overlay_com"),
                show_com_velocity=self._overlay_enabled("var_overlay_com_velocity"),
                show_support_polygon=self._overlay_enabled("var_overlay_bos"),
                show_contact_points=self._overlay_enabled("var_overlay_contact"),
                show_gait_direction=self._overlay_enabled("var_overlay_direction")
                and show_detail,
                show_dof_pick_targets=bool(
                    getattr(self, "var_skeleton_pick_dof", None)
                    and self.var_skeleton_pick_dof.get()
                ),
                highlight_color=highlight_color,
                dim_unhighlighted=dim_unhighlighted,
                difference_match_snapshot=match_snap,
                difference_joint_pairs=diff_pairs,
                difference_arrow_color=highlight_color,
            )
            self._skeleton_artists_dirty = False
            self._skeleton_artists_mode = mode_key
            # While playing always coalesce — sync draw() stalls the UI thread.
            self._paint_canvas_now(self.canvas_3d, idle=bool(is_playing) or not force_draw)
            if not is_playing and force_draw and not getattr(
                self, "_playback_sync_immediate", False
            ):
                self.root.after_idle(self._fit_skeleton_canvas)

    def _resolve_initial_path(self, poses_path: str | Path | None) -> Path | None:
        if poses_path:
            p = Path(poses_path)
            if p.is_file():
                return p
            candidate = config.POSES_DIR / f"{poses_path}_poses.json"
            if candidate.is_file():
                return candidate

        default = config.POSES_DIR / "walk_stream_poses.json"
        if default.is_file():
            return default

        if config.POSES_DIR.is_dir():
            jsons = sorted(config.POSES_DIR.glob("*_poses.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if jsons:
                return jsons[0]
        return None

    def _show_overlay(self, text: str) -> None:
        self._demo_overlay.configure(text=text)
        self._demo_overlay.grid(row=0, column=0, sticky="n", pady=(4, 0))

    def _hide_overlay(self) -> None:
        self._demo_overlay.grid_remove()

    def _fade_transition(self, *, steps: int = 6, on_done: Callable[[], None] | None = None) -> None:
        """Brief dim flash when switching videos (presentation polish)."""
        self._fade_step = 0
        shades = (PANEL, ELEVATED, SURFACE, ELEVATED, PANEL, PANEL)

        def step() -> None:
            if self._fade_step < len(shades):
                self.video_label.configure(bg=shades[self._fade_step])
                self._fade_step += 1
                self.root.after(45, step)
            else:
                self.video_label.configure(bg=PANEL)
                if on_done:
                    on_done()

        step()

    def _set_load_feedback(self, message: str) -> None:
        # Professional lab status — plain text only (no emoji decoration).
        self.status.configure(text=message)
        if self._demo_running:
            self._show_overlay(message)

    def _analysis_workflow_busy(self) -> bool:
        """True while pipeline / session analysis is still running (UI stays responsive)."""
        if (
            getattr(self, "_pending_pipeline_finish", None) is not None
            and not getattr(self, "_draining_pipeline", False)
            and threading.current_thread() is threading.main_thread()
        ):
            self._draining_pipeline = True
            try:
                self._drain_pending_pipeline()
            finally:
                self._draining_pipeline = False
        if bool(getattr(self, "_processing", False)):
            return True
        if bool(getattr(self, "_analysis_compute_active", False)):
            return True
        try:
            from stablewalk.ui.tk.analyze_progress import analyze_progress_active

            if analyze_progress_active(self):
                return True
        except Exception:
            pass
        worker = getattr(self, "_analysis_worker", None)
        if worker is not None and worker.is_alive():
            return True
        return False

    def _sync_analysis_busy_controls(self) -> None:
        """Disable unfinished load/export controls while analysis is running."""
        busy = self._analysis_workflow_busy()
        state = tk.DISABLED if busy else tk.NORMAL
        for btn in (
            getattr(self, "btn_load", None),
            getattr(self, "btn_browse", None),
            getattr(self, "btn_full", None),
            getattr(self, "btn_export", None),
            getattr(self, "btn_demo", None),
            getattr(self, "btn_next_video", None),
            getattr(self, "btn_custom_url", None),
            *getattr(self, "_demo_gait_buttons", {}).values(),
        ):
            if btn is None:
                continue
            try:
                btn.configure(state=state)
            except tk.TclError:
                pass
        url = getattr(self, "url_entry", None)
        if url is not None:
            try:
                url.configure(state="readonly" if busy else tk.NORMAL)
            except tk.TclError:
                pass
        # Playback stays available so the shell never feels frozen.
        play = getattr(self, "btn_play", None)
        if play is not None:
            try:
                play.configure(state=tk.NORMAL)
            except tk.TclError:
                pass
        try:
            self._update_analysis_export_state()
        except Exception:
            pass

    def _lock_ui(self, locked: bool) -> None:
        """Lock load chrome for a pipeline run; export gating uses workflow busy."""
        self._ui_locked = bool(locked)
        if locked:
            state = tk.DISABLED
            for btn in (
                self.btn_load,
                self.btn_browse,
                self.btn_full,
                self.btn_export,
                self.btn_demo,
                self.btn_next_video,
                self.btn_custom_url,
                *getattr(self, "_demo_gait_buttons", {}).values(),
            ):
                btn.configure(state=state)
            self.btn_play.configure(state=tk.NORMAL)
            self.url_entry.configure(state="readonly")
        else:
            # Keep unfinished controls disabled if analysis is still finishing.
            self._sync_analysis_busy_controls()
        try:
            self._update_analysis_export_state()
        except Exception:
            pass

    def _reset_session(self, *, message: str = "Loading new video...") -> None:
        """Clear ALL gait state so no data leaks between videos."""
        try:
            from stablewalk.ui.tk.gui_memory import teardown_for_new_video

            teardown_for_new_video(self)
        except Exception:
            release_all_captures()
            self._stop_all_gui_timers()
            try:
                from stablewalk.ui.tk.dashboard_tab_lifecycle import (
                    release_compare_temp_buffers,
                )

                release_compare_temp_buffers(self)
            except Exception:
                pass
        self._sync_play_buttons()
        self.sequence = None
        self.pose_indices = []
        self.current_pos = 0
        self._poses_path = None
        self._walk_simulation = None
        self._skeleton_cache = {}
        self._frame_motion_snap_cache = {}
        self._compare = None
        self._run_metadata = None
        if hasattr(self, "_clear_session_analysis_cache"):
            self._clear_session_analysis_cache()
        self._active_run_name = None
        self._opensim_export_completed = False
        self._opensim_status_override = None
        self._loaded_content_key = ""
        self._playback_pos = 0.0
        self._photo = None
        self._last_video_rgb = None
        self._video_photo_key = None
        self._traj_draw_cache_key = None
        self._overview_traj_draw_cache_key = None
        self._overview_path_cache = {}
        try:
            self._rgb_cache.clear()
        except Exception:
            pass
        try:
            cache = getattr(self, "_landmark_overlay_cache", None)
            if isinstance(cache, dict):
                cache.clear()
        except Exception:
            pass
        self._analysis_panel_dirty = set()
        self._dof_table_display_dirty = False
        self._biomech_static_sig = None
        self._biomech_stability_state_sig = None
        try:
            from stablewalk.ui.tk.playback_render_policy import clear_paint_signatures

            clear_paint_signatures(self)
        except Exception:
            self._paint_signatures = {}
        # Drop large PhotoImage / trail buffers from the previous session.
        self._last_video_list_pos = -1
        if hasattr(self, "_video_overlay_trails"):
            try:
                delattr(self, "_video_overlay_trails")
            except Exception:
                self._video_overlay_trails = None
        for ax_attr in ("ax_dof_traj", "ax_dof_traj_overview", "ax_3d"):
            ax = getattr(self, ax_attr, None)
            if ax is not None and hasattr(ax, "_stablewalk_stable_viewport"):
                try:
                    delattr(ax, "_stablewalk_stable_viewport")
                except Exception:
                    pass
        self._skeleton_scale = 1.0
        self._view_limit_3d = 0.55
        self._rom_cache = {}
        self._3d_motion_trail = []
        self._motion_series = None
        self.gait_motion = None
        self.skeleton_player = None
        incoming = getattr(self, "_active_demo_gait", None)
        if incoming is not None:
            self._demo_comparison_rows.pop(incoming.key, None)
        self._clear_overview_analysis_surfaces()
        self._last_data_refresh = 0.0
        self._last_dof_table_refresh = 0.0
        self._dof_table_history.clear()
        self._session_collector.clear()
        self._bilateral_foot_collector.clear()
        self._ground_clearance_prev_phase = (None, None)
        self._foot_clearance_live_cache = None
        self.selection = SelectionState()
        self._sync_dof_checkboxes()
        self._update_dof_selection_chrome()
        self._refresh_dof_details()
        self.slider.configure(to=0)
        self.frame_var.set(0)
        self.lbl_session_status.configure(text="No session loaded")
        self._session_display_src = ""
        self.lbl_frame_data.configure(text="Frame —")
        for tree in (self.pos_3d_tree, self.dof_tree, self.vel_tree):
            for item in tree.get_children():
                tree.delete(item)
        self._clear_dof_position_table()
        self._dof_step_previews = []
        self._refresh_dof_step_panel()
        self.motion_text.configure(state=tk.NORMAL)
        self.motion_text.delete("1.0", tk.END)
        self.motion_text.configure(state=tk.DISABLED)
        self._update_session_selection_overview()
        self._reset_stability_panel()
        self._reset_gait_cycle_panel()
        self._refresh_real_to_sim_advanced_panel()
        if hasattr(self, "lbl_opensim_stats"):
            self._update_opensim_overview(
                frames="—",
                markers="—",
                angles="—",
                export_summary="no session",
            )
            self._opensim_last_dir = None
            self._opensim_ik_state = None
            self._refresh_opensim_status()
        self._show_compare_note("")
        self._init_chart()
        # Motion knee chart is lazy — skip draw until the Motion tab has built it.
        if getattr(self, "chart_canvas", None) is not None:
            self.chart_canvas.draw_idle()
        if getattr(self, "ax_3d", None) is not None and getattr(self, "canvas_3d", None) is not None:
            self.ax_3d.cla()
            setup_skeleton_axes(self.ax_3d)
            self.canvas_3d.draw_idle()
        if getattr(self, "ax_robot", None) is not None and getattr(self, "canvas_robot", None) is not None:
            self.ax_robot.cla()
            setup_3d_axes(self.ax_robot)
            self.canvas_robot.draw_idle()
        try:
            from stablewalk.ui.tk.gui_memory import dispose_orphan_matplotlib_figures

            dispose_orphan_matplotlib_figures(self)
        except Exception:
            pass
        self.video_label.configure(
            image="",
            text=message if message.startswith("Loading") else EMPTY_VIDEO_TEXT,
            fg=MUTED if message.startswith("Loading") else TEXT,
        )
        self.lbl_frame.configure(text="—")
        self._set_load_feedback(message)
        self.root.update_idletasks()

    def load_poses(
        self,
        path: str | Path,
        *,
        fresh: bool = True,
        expected_source: str | None = None,
        metadata: PipelineResult | None = None,
        wait_for_analysis: bool = True,
    ) -> bool:
        """Load pose JSON; return False on failure. If fresh, rebuild all views from frame 0.

        When ``wait_for_analysis`` is False (GUI pipeline path), heavy analysis
        runs on the single session analysis worker and UI finalization continues
        after it completes — so the Tk thread is not blocked.
        """
        path = Path(path)
        if fresh:
            # Stop playback / deferred paints before swapping analysis objects.
            self._stop_all_gui_timers()
            self._sync_play_buttons()
            self._skeleton_cache = {}
            self._frame_motion_snap_cache = {}
            self._walk_simulation = None
            self._compare = None
            self.sequence = None
            self.pose_indices = []
            self._dof_table_history.clear()
            self._session_collector.clear()
            self._bilateral_foot_collector.clear()
            self._ground_clearance_prev_phase = (None, None)
            self._traj_draw_cache_key = None
            self._overview_traj_draw_cache_key = None
            self._overview_path_cache = {}
            try:
                from stablewalk.ui.viewers.chart_interactions import reset_all_chart_navigation

                reset_all_chart_navigation(self)
            except Exception:
                pass
        try:
            self.sequence = load_pose_sequence(path)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            if not self._demo_running:
                messagebox.showerror("Load error", f"Could not load pose data:\n{exc}")
            return False

        self._rgb_cache.clear()
        try:
            cache = getattr(self, "_landmark_overlay_cache", None)
            if isinstance(cache, dict):
                cache.clear()
        except Exception:
            pass
        self._analysis_panel_dirty = set()
        self._dof_table_display_dirty = False
        self._progressive_media_ready = False
        try:
            from stablewalk.ui.tk.panel_loading import clear_all_panel_loading

            clear_all_panel_loading(self)
        except Exception:
            pass
        self._panel_tick = 0
        self._clear_shared_frame_buffer()
        self.pose_indices = playback_frame_indices(self.sequence)
        if not self.pose_indices:
            if not self._demo_running:
                messagebox.showwarning(
                    "No frames",
                    "This file has no stored source frames.\n"
                    "Run: python main.py --url --view",
                )
            return False

        self._sync_playback_hz_from_source()
        # Real analysis session — must be set before any OpenSim status refresh.
        self._presentation_mode = False
        self._cancel_presentation_autoplay()
        self._poses_path = path
        if metadata is not None and getattr(metadata, "run_name", None):
            self._active_run_name = str(metadata.run_name)
        else:
            stem = path.stem
            self._active_run_name = (
                stem[: -len("_poses")] if stem.endswith("_poses") else stem
            ) or None
        self._opensim_export_completed = False
        self._opensim_status_override = None

        self.current_pos = 0
        self.playing = False
        self._sync_play_buttons()
        self._cancel_timer()

        n = len(self.pose_indices) - 1
        self.slider.configure(to=max(n, 0))
        self.frame_var.set(0)

        src = expected_source or self._current_source or self.sequence.source_video
        display_src = self._short_source_name(src)
        pending = self._get_input_source()
        if pending and content_cache_key(pending) != content_cache_key(src):
            self.status.configure(text="URL changed — click Analyze")

        self._loaded_content_key = content_cache_key(src)
        if metadata:
            self._run_metadata = metadata
        if expected_source and self.sequence.source_video:
            norm_expected = normalize_source(expected_source)
            norm_stored = normalize_source(self.sequence.source_video)
            if norm_stored != norm_expected and not norm_expected.endswith(
                Path(norm_stored).name
            ):
                logger.warning(
                    "Pose source_video mismatch: stored=%s expected=%s",
                    norm_stored,
                    norm_expected,
                )

        logger.info("Loaded new video: %s → %s", display_src, path.name)
        self._session_display_src = display_src
        meta_frames = None
        meta_fps = None
        if metadata is not None:
            meta_frames = getattr(metadata, "frame_count", None)
            meta_fps = getattr(metadata, "fps", None)
        n_total = meta_frames or (len(self.sequence.frames) if self.sequence else None)
        n_detected = (
            sum(1 for f in self.sequence.frames if f.detected)
            if self.sequence
            else None
        )
        try:
            self._report_analyze_stage(
                "reconstruction",
                fraction=0.62,
                frames_done=n_detected,
                frames_total=n_total,
                fps=meta_fps or getattr(self.sequence, "fps", None),
            )
            # Progressive loading: paint Video + Skeleton immediately. Heavy
            # enrich + metrics/graphs/biomech/OpenSim run on the analysis worker.
            needs_enrich = sequence_needs_enrichment(self.sequence)
            if wait_for_analysis and needs_enrich:
                from stablewalk.pose.enrichment import enrich_pose_sequence

                enrich_pose_sequence(self.sequence)
                needs_enrich = False

            self._report_analyze_stage(
                "joint_angles",
                fraction=0.70,
                frames_done=n_detected,
                frames_total=n_total,
                fps=meta_fps or getattr(self.sequence, "fps", None),
            )

            # Immediate media — never wait for full session analysis.
            try:
                self._paint_media_from_poses()
            except Exception:
                logger.debug("Early media paint failed", exc_info=True)

            from stablewalk.ui.tk.panel_loading import set_panels_loading

            set_panels_loading(
                self, ("metrics", "graphs", "biomech", "opensim"), True
            )

            def _after_analysis(error: Exception | None) -> None:
                if error is not None:
                    from stablewalk.ui.tk.analyze_progress import fail_analyze_progress
                    from stablewalk.ui.tk.panel_loading import clear_all_panel_loading

                    clear_all_panel_loading(self)
                    fail_analyze_progress(self, message="Failed")
                    if not self._demo_running:
                        messagebox.showerror(
                            "Load error",
                            f"Pose data loaded but analysis failed:\n{error}",
                        )
                    post = getattr(self, "_pipeline_post_load", None)
                    if post is not None:
                        self._pipeline_post_load = None
                        cb = post.get("callback")
                        if cb:
                            try:
                                cb(None, error)
                            except Exception:
                                pass
                    return
                try:
                    self._finalize_loaded_poses_after_analysis(
                        n_detected=n_detected,
                        n_total=n_total,
                        meta_fps=meta_fps,
                    )
                except Exception as exc:
                    logger.exception("Failed to display loaded poses")
                    from stablewalk.ui.tk.analyze_progress import fail_analyze_progress
                    from stablewalk.ui.tk.panel_loading import clear_all_panel_loading

                    clear_all_panel_loading(self)
                    fail_analyze_progress(self, message="Failed")
                    if not self._demo_running:
                        messagebox.showerror(
                            "Load error",
                            f"Pose data loaded but the dashboard could not start:\n{exc}",
                        )

            # Single analysis pass — results live in ``_session_analysis``.
            # GUI loads use the background worker so Tk stays responsive.
            self._run_session_analysis_once(
                enrich=bool(needs_enrich),
                report_stages=True,
                n_detected=n_detected,
                n_total=n_total,
                meta_fps=meta_fps,
                display_src=display_src,
                already_have_gait_motion=self.gait_motion is not None,
                blocking=bool(wait_for_analysis),
                on_done=None if wait_for_analysis else _after_analysis,
            )
            if wait_for_analysis:
                if not getattr(self._session_analysis, "ready", False):
                    _after_analysis(RuntimeError("Session analysis did not complete"))
                else:
                    _after_analysis(None)
        except Exception as exc:
            logger.exception("Failed to display loaded poses")
            from stablewalk.ui.tk.analyze_progress import fail_analyze_progress
            from stablewalk.ui.tk.panel_loading import clear_all_panel_loading

            clear_all_panel_loading(self)
            fail_analyze_progress(self, message="Failed")
            if not self._demo_running:
                messagebox.showerror(
                    "Load error",
                    f"Pose data loaded but the dashboard could not start:\n{exc}",
                )
            return False

        if wait_for_analysis:
            return self._finish_load_poses_shell()
        # Async path: media already painted; keep UI interactive while analyzing.
        try:
            self.status.configure(text=f"{display_src} — analyzing…")
            self._hide_overlay()
        except Exception:
            pass
        return True

    def _paint_media_from_poses(self) -> None:
        """Show Original Video + 3D Skeleton immediately from loaded poses.

        Metrics / graphs / biomechanics / OpenSim stay deferred with per-widget
        spinners until session analysis finishes.
        """
        if not self.sequence or not self.pose_indices:
            return
        self._progressive_media_ready = True
        # Provisional gait recording so the interactive skeleton can paint now.
        if self.skeleton_player is None or self.gait_motion is None:
            from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion

            recording = pose_sequence_to_gait_motion(self.sequence)
            self._set_gait_motion(recording)
        else:
            try:
                self._show_pose_at(0, force_draw=True)
            except Exception:
                pass
        # Ensure video frame is painted even if skeleton path already drew.
        try:
            self._show_pose_at(0, video_only=True, force_draw=True)
        except Exception:
            logger.debug("Early video paint failed", exc_info=True)

    def _finalize_loaded_poses_after_analysis(
        self,
        *,
        n_detected: int | None,
        n_total: int | None,
        meta_fps: float | None,
    ) -> None:
        """Apply analysis results; paint Metrics/Graphs/Biomech/OpenSim progressively."""
        self._report_analyze_stage(
            "opensim",
            fraction=0.90,
            frames_done=n_detected,
            frames_total=n_total,
            fps=meta_fps or getattr(self.sequence, "fps", None),
        )
        # Drop stale composed overlays from a previous session.
        cache = getattr(self, "_landmark_overlay_cache", None)
        if isinstance(cache, dict):
            cache.clear()
        self._opensim_ik_state = None
        self._update_dof_table_controls_state()
        self._knee_chart_mode_user_set = False
        # Graphs stay dirty until their progressive paint / tab open.
        self._mark_analysis_panels_dirty(
            "knee", "motion", "biomech", "contact", "summary", "compare"
        )
        self._report_analyze_stage(
            "report",
            fraction=0.96,
            frames_done=n_detected,
            frames_total=n_total,
            fps=meta_fps or getattr(self.sequence, "fps", None),
        )
        # Media was already shown progressively — refresh once after cache apply.
        if not getattr(self, "_progressive_media_ready", False):
            try:
                self._show_pose_at(0, force_draw=True)
            except Exception:
                pass
        else:
            try:
                self._update_interactive_skeleton(force_draw=True)
            except Exception:
                pass
        detected_n = sum(1 for f in self.sequence.frames if f.detected) if self.sequence else 0
        self._update_session_selection_overview()
        from stablewalk.ui.tk.analyze_progress import complete_analyze_progress

        complete_analyze_progress(
            self,
            frames_done=n_detected or detected_n,
            frames_total=n_total or (len(self.sequence.frames) if self.sequence else 0),
            fps=float(meta_fps or getattr(self.sequence, "fps", 0.0) or 0.0) or None,
        )
        try:
            self._write_completed_frame_audit()
        except Exception:
            logger.debug("Frame audit write failed", exc_info=True)
        if not self._demo_running:
            self._lock_ui(False)
        else:
            self._sync_analysis_busy_controls()
        self._finish_load_poses_shell()
        try:
            self._hide_overlay()
        except Exception:
            pass
        post = getattr(self, "_pipeline_post_load", None)
        if post is not None:
            self._pipeline_post_load = None
            try:
                self._complete_pipeline_post_load(**post)
            except Exception:
                logger.exception("pipeline post-load UI failed")

        # Stagger panel paints so Tk never freezes on a single mega-callback.
        self._schedule_progressive_panel_paints()

    def _schedule_progressive_panel_paints(self) -> None:
        """Metrics → Graphs → Biomechanics → OpenSim, each on its own after() tick."""
        from stablewalk.ui.tk.timer_registry import schedule_unique

        def _paint_metrics() -> None:
            try:
                self._paint_metrics_panels()
            finally:
                from stablewalk.ui.tk.panel_loading import set_panel_loading

                set_panel_loading(self, "metrics", False)

        def _paint_graphs() -> None:
            try:
                self._paint_graphs_panels()
            finally:
                from stablewalk.ui.tk.panel_loading import set_panel_loading

                set_panel_loading(self, "graphs", False)

        def _paint_biomech() -> None:
            try:
                self._paint_biomech_panels()
            finally:
                from stablewalk.ui.tk.panel_loading import set_panel_loading

                set_panel_loading(self, "biomech", False)

        def _paint_opensim() -> None:
            try:
                self._refresh_opensim_status()
                self._maybe_auto_opensim_pipeline()
            except Exception:
                logger.debug("progressive OpenSim stage failed", exc_info=True)
                from stablewalk.ui.tk.panel_loading import set_panel_loading

                set_panel_loading(self, "opensim", False)
            # Spinner cleared when auto-pipeline finishes (or immediately if skipped).
            if not getattr(self, "_opensim_auto_pipeline_busy", False):
                from stablewalk.ui.tk.panel_loading import set_panel_loading

                set_panel_loading(self, "opensim", False)

        schedule_unique(self, "_progressive_metrics_after", 0, _paint_metrics)
        schedule_unique(self, "_progressive_graphs_after", 20, _paint_graphs)
        schedule_unique(self, "_progressive_biomech_after", 40, _paint_biomech)
        schedule_unique(self, "_progressive_opensim_after", 80, _paint_opensim)

    def _paint_metrics_panels(self) -> None:
        """Overview KPI / gait / physics labels — after analysis cache is ready."""
        try:
            self._ensure_metrics_stage_summary()
        except Exception:
            logger.debug("Metrics stage summary failed during panel paint", exc_info=True)
        try:
            self._update_gait_cycle_panel(self._gait_cycle)
        except Exception:
            pass
        try:
            self._refresh_bilateral_ground_clearance()
        except Exception:
            pass
        overview_active = self._group_visible("overview") or self._group_visible(
            "video"
        )
        if overview_active:
            try:
                self._refresh_physics_force_panel()
            except Exception:
                pass
            try:
                self._update_motion_temporal_metrics_panel()
            except Exception:
                pass
            try:
                self._update_overview_clinical_summary()
            except Exception:
                pass
            # Overview 3D path lives next to video/skeleton — refresh with metrics so
            # progressive load never leaves an empty trajectory cube.
            try:
                if getattr(self, "_overview_traj_dock_visible", False) and self._group_visible(
                    "trajectory"
                ):
                    self._overview_traj_draw_cache_key = None
                    self._refresh_overview_trajectory_dock(force_draw=True)
            except Exception:
                logger.debug(
                    "progressive overview trajectory paint failed", exc_info=True
                )
        if self._group_visible("summary"):
            try:
                self._analysis_panel_dirty.discard("summary")
            except Exception:
                pass

    def _paint_graphs_panels(self) -> None:
        """Knee / motion charts — only when their tab is visible; else stay dirty."""
        from stablewalk.ui.tk.redraw_policy import full_tab_redraw_blocked

        if full_tab_redraw_blocked(self):
            from stablewalk.ui.tk.playback_active_tab import mark_hidden_tab_panels_dirty

            mark_hidden_tab_panels_dirty(self)
            # Playback / resize: move visible playheads only — never rebuild the tab.
            if self._group_visible("knee"):
                try:
                    self._update_chart()
                except Exception:
                    pass
            if self._group_visible("joint_motion") or self._group_visible("motion"):
                try:
                    self._update_joint_motion_analysis_chart(force_rebuild=False)
                except Exception:
                    pass
            return
        if self._group_visible("knee") or self._group_visible("motion"):
            try:
                self._rebuild_knee_angle_series()
                self._apply_default_knee_chart_mode()
                self._update_chart()
                self._analysis_panel_dirty.discard("knee")
            except Exception:
                logger.debug("progressive knee paint failed", exc_info=True)
            try:
                self.refresh_motion_analysis_tab()
                self._analysis_panel_dirty.discard("motion")
                self._analysis_panel_dirty.discard("joint_motion")
            except Exception:
                pass
        else:
            self._mark_analysis_panels_dirty("knee", "motion", "joint_motion")
        if self._group_visible("compare"):
            try:
                self._update_compare_chart()
                self._analysis_panel_dirty.discard("compare")
            except Exception:
                pass

    def _paint_biomech_panels(self) -> None:
        """Biomechanics + contact charts — active workspace family only."""
        from stablewalk.ui.tk.redraw_policy import full_tab_redraw_blocked

        if full_tab_redraw_blocked(self):
            from stablewalk.ui.tk.playback_active_tab import mark_hidden_tab_panels_dirty

            mark_hidden_tab_panels_dirty(self)
            family = str(getattr(self, "_biomech_chart_family", "biomech") or "biomech")
            try:
                if family == "contact" and self._group_visible("contact"):
                    self._update_contact_gait_chart()
                elif self._group_visible("biomech"):
                    self._update_biomechanics_chart()
            except Exception:
                pass
            return
        if self._group_visible("biomech") or self._group_visible("contact"):
            try:
                self._update_biomechanics_panel()
            except Exception:
                pass
            family = str(
                getattr(self, "_biomech_chart_family", "biomech") or "biomech"
            )
            if family == "contact":
                try:
                    self._update_contact_gait_chart()
                except Exception:
                    pass
            else:
                try:
                    self._update_biomechanics_chart()
                except Exception:
                    pass
            dirty = getattr(self, "_analysis_panel_dirty", None)
            if dirty is not None:
                dirty.discard("biomech")
                dirty.discard("contact")
        else:
            self._mark_analysis_panels_dirty("biomech", "contact")
        # Results/Advanced pipeline chrome — defer unless that tab is open.
        tab_now = None
        results_tab = "Results"
        advanced_tab = "Advanced"
        try:
            from stablewalk.ui.tk.dashboard_tab_lifecycle import active_tab_id
            from stablewalk.ui.tk.dashboard_notebook import (
                TAB_ADVANCED,
                TAB_RESULTS_SUMMARY,
            )

            results_tab = TAB_RESULTS_SUMMARY
            advanced_tab = TAB_ADVANCED
            tab_now = active_tab_id(self)
        except Exception:
            pass
        if tab_now in (results_tab, advanced_tab) or self._group_visible("summary"):
            try:
                self._refresh_pipeline_status_panel()
            except Exception:
                pass
            try:
                self._refresh_real_to_sim_advanced_panel()
            except Exception:
                pass
        else:
            self._pipeline_status_dirty_while_playing = True

    def _finish_load_poses_shell(self) -> bool:
        """Demo chrome / session dirty bookkeeping after a successful load."""
        if self._current_source:
            detected = is_demo_video_path(self._current_source)
            if detected:
                self._active_demo_gait = detected
        self._apply_demo_presentation()

        # Motion-tab trajectory only when Motion Analysis is the active surface.
        if getattr(self, "canvas_dof_traj", None) is not None and self._group_visible(
            "motion"
        ):
            self._sync_dof_analysis_panel_state()
            self._refresh_selected_dof_trajectory_3d(force_draw=True)
        elif getattr(self, "canvas_dof_traj", None) is not None:
            self._mark_analysis_panels_dirty("motion", "knee")

        self._request_dashboard_scroll_sync()
        if not getattr(self, "_session_restoring", False):
            mgr = getattr(self, "_session_mgr", None)
            if mgr is not None:
                mgr.current_path = None
                self._session_current_path = None
                mgr.mark_dirty()
        try:
            from stablewalk.ui.tk.session_cache import pin_current_session

            # The loaded/completed workspace analysis is always Compare
            # Session A. Reuse it directly; Session B remains independent.
            pin_current_session(self, slot="left")
        except Exception:
            pass
        return True

    def _smooth_motion_enabled(self) -> bool:
        """Read the smooth-motion flag without requiring the Tk event loop.

        Session analysis may run on a worker thread. BooleanVar.get() is not
        safe there, so the UI thread snapshots ``_smooth_motion_cached`` first.
        """
        if bool(getattr(self, "_analysis_compute_active", False)):
            return bool(getattr(self, "_smooth_motion_cached", False))
        var = getattr(self, "smooth_motion", None)
        if var is not None:
            try:
                return bool(var.get())
            except (tk.TclError, RuntimeError):
                pass
        return bool(getattr(self, "_smooth_motion_cached", False))

    def _prepare_3d_sequence(self) -> None:
        """Ensure 3D data exists and compute stable view limits for animation."""
        if not self.sequence:
            return

        from stablewalk.pose.enrichment import enrich_pose_frame
        from stablewalk.pose.skeleton_3d import (
            Skeleton3D,
            reconstruct_skeleton_3d,
            sequence_skeleton_scale,
        )

        detected_kps = [
            f.keypoints for f in self.sequence.frames if f.detected and f.keypoints
        ]
        self._skeleton_scale = (
            sequence_skeleton_scale(detected_kps) if detected_kps else 1.0
        )

        skeletons: list[Skeleton3D] = []
        for idx in self.pose_indices:
            frame = self.sequence.frames[idx]
            if not frame.skeleton_3d.get("joints"):
                enrich_pose_frame(frame, uniform_scale=self._skeleton_scale)
            skel = skeleton_from_frame_data(
                frame.keypoints,
                frame.skeleton_3d,
                self._skeleton_scale,
            )
            if not skel.joints and frame.keypoints:
                skel = reconstruct_skeleton_3d(
                    frame.keypoints, scale=self._skeleton_scale
                )
            skeletons.append(skel)

        if self._smooth_motion_enabled() and len(skeletons) >= 3:
            skeletons = smooth_skeleton_temporal(skeletons, window=3)

        self._view_limit_3d = compute_view_limit(skeletons) if skeletons else 0.55
        self._skeleton_cache = {
            idx: skeletons[i] for i, idx in enumerate(self.pose_indices)
        }
        self._rom_cache = compute_sequence_rom(self.sequence.frames, self.pose_indices)
        self._3d_motion_trail = []

    def _build_walk_simulation(self) -> None:
        if not self.sequence:
            self._walk_simulation = None
            return
        sim = WalkSimulator()
        self._walk_simulation = (
            sim.from_pose_sequence_smoothed(self.sequence, substeps=3)
            if self._smooth_motion_enabled()
            else sim.from_pose_sequence(self.sequence)
        )

    def _run_full_analysis(self) -> None:
        """Run complete pipeline on every source frame (no DEMO_MAX_FRAMES cap)."""
        source = self._get_input_source()
        if source and is_demo_video_path(source) is None:
            self._clear_demo_gait_context()
        self._load_video(analysis_mode="full")

    def _get_input_source(self) -> str:
        """Read the URL/file field after Tk sync (always current text)."""
        self.root.update_idletasks()
        try:
            raw = self.url_entry.get().strip()
        except tk.TclError:
            raw = self.url_var.get().strip()
        if not raw:
            raw = self.url_var.get().strip()
        return normalize_source(raw)

    def _apply_preset_to_form(self, preset) -> None:
        self.url_var.set(preset.url)
        self.url_entry.delete(0, tk.END)
        self.url_entry.insert(0, preset.url)
        self.status.configure(text=f"Preset: {preset.label}")

    def _on_preset_selected(self, _event: object = None) -> None:
        if self.preset_var.get() == CUSTOM_URL_PRESET_LABEL:
            self.url_entry.focus_set()
            self.status.configure(text="Paste URL, then Analyze")
            return
        preset = preset_by_label(self.preset_var.get())
        if not preset:
            return
        self._apply_preset_to_form(preset)
        if not self.auto_load_on_change.get():
            self.status.configure(text=f"{preset.label} — click Analyze")
            return
        if self._preset_after_id:
            self.root.after_cancel(self._preset_after_id)
        self._preset_after_id = self.root.after(
            150,
            lambda p=preset: self._load_video(source=p.url),
        )

    def _enter_custom_url(self) -> None:
        """Dialog so the user can paste any walking-video URL."""
        current = self._get_input_source()
        url = simpledialog.askstring(
            "Enter video URL",
            "Paste a direct video URL (Pexels download link works best):\n"
            "Example: https://www.pexels.com/download/video/5319095/",
            initialvalue=current if is_video_url(current) else config.DEFAULT_WALKING_VIDEO_URL,
            parent=self.root,
        )
        if not url or not url.strip():
            return
        url = normalize_source(url.strip())
        preset = preset_from_custom_url(url)
        self.preset_var.set(CUSTOM_URL_PRESET_LABEL)
        self._apply_preset_to_form(preset)
        self.status.configure(text="Custom URL ready")
        if self.auto_load_on_change.get():
            self._load_video(source=url, chain_next_on_fail=False)

    def _pick_next_verified_preset(self, presets: list) -> tuple[int, object] | tuple[None, None]:
        """Skip to the next clip that passes a quick full-body pose sample."""
        from stablewalk.pose.video_validation import validate_video_source

        if not presets:
            return None, None
        start = (self._preset_cycle_index + 1) % len(presets)
        for offset in range(len(presets)):
            idx = (start + offset) % len(presets)
            preset = presets[idx]
            passed, ratio, msg = validate_video_source(
                preset.url, sample_count=12, min_valid_ratio=0.2
            )
            logger.info("Pre-check %s: %s (%s)", preset.label, passed, msg)
            if passed:
                return idx, preset
        return None, None

    def _next_video(self) -> None:
        """Cycle pose-verified men's walk URLs only."""
        presets = list_runnable_presets(verified_only=True, urls_only=True)
        if not presets:
            messagebox.showwarning(
                "No URL videos",
                "No verified men's walk URLs.\n"
                f"Check {config.DATA_DIR / 'verified_men_walk.json'} or use Enter URL…",
            )
            return
        self._lock_ui(True)
        self._show_overlay("Finding next good walking video…")
        self.status.configure(text="Checking next clip…")

        def worker() -> None:
            idx, preset = self._pick_next_verified_preset(presets)
            self.root.after(0, lambda: self._apply_next_preset(idx, preset, presets))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_next_preset(
        self,
        idx: int | None,
        preset,
        presets: list,
    ) -> None:
        if idx is None or preset is None:
            self._lock_ui(False)
            self._hide_overlay()
            messagebox.showwarning(
                "No suitable video",
                "Could not find a walking clip with enough poses.\n"
                "Try Enter URL… with a clear full-body walker, or upload a local file.",
            )
            return
        self._preset_cycle_index = idx
        self._next_video_attempts = 0
        self._chain_next_on_fail = True
        self.preset_var.set(preset.label)
        self._apply_preset_to_form(preset)
        self._show_overlay(f"Loading next video…\n{preset.label}")
        self._load_video(source=preset.url, chain_next_on_fail=True)

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select walking video",
            initialdir=str(config.INPUT_DIR),
            filetypes=[
                ("Video", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.url_var.set(path)
            self.url_entry.delete(0, tk.END)
            self.url_entry.insert(0, path)
            self.preset_var.set(CUSTOM_URL_PRESET_LABEL)
            detected = is_demo_video_path(path)
            if detected:
                self._active_demo_gait = detected
                self._update_demo_analysis_title(detected)
                self._highlight_demo_button(detected.key)
            else:
                self._clear_demo_gait_context()
            if self.auto_load_on_change.get():
                self._load_video(source=path)
            else:
                self.status.configure(text=f"{Path(path).name} — click Analyze")

    def _load_video(
        self,
        *,
        source: str | None = None,
        on_complete: PipelineCallback | None = None,
        show_dialog: bool = True,
        max_frames: int | None = None,
        unique_session: bool = True,
        chain_next_on_fail: bool = False,
        analysis_mode: str | None = None,
    ) -> None:
        if chain_next_on_fail:
            self._chain_next_on_fail = True

        # Always use explicit source when provided (preset / next video / browse)
        source = normalize_source(source) if source else self._get_input_source()
        if not source:
            if not self._demo_running:
                messagebox.showwarning("Input", "Enter a video URL or choose a local file.")
            if on_complete:
                on_complete(None, ValueError("No video source"))
            return

        if self._processing:
            short = source if len(source) < 72 else source[:69] + "…"
            self._pending_video_load = {
                "source": source,
                "on_complete": on_complete,
                "show_dialog": show_dialog,
                "max_frames": max_frames,
                "unique_session": unique_session,
                "chain_next_on_fail": chain_next_on_fail,
                "analysis_mode": analysis_mode,
            }
            logger.info("Queued video load while pipeline active: %s", short)
            self.status.configure(text=f"Queued: {short}")
            return

        detected_demo = is_demo_video_path(source)
        if detected_demo:
            self._active_demo_gait = detected_demo
            self._presentation_mode = False
            self._cancel_presentation_autoplay()
            self._update_demo_analysis_title(detected_demo)
            self._highlight_demo_button(detected_demo.key)
        elif self._active_demo_gait is not None:
            self._clear_demo_gait_context()

        ok, fmt_msg = check_source_format(source)
        if not ok:
            self._handle_load_error(fmt_msg, on_complete=on_complete)
            return

        qok, qmsg = quick_validate_source(source)
        if not qok:
            self._handle_load_error(qmsg, on_complete=on_complete)
            return

        self._validate_mode = resolve_validate_mode(source)

        self._session_id += 1
        session_id = self._session_id
        self._current_source = source
        self._pipeline_callback = on_complete
        self._show_success_dialog = show_dialog
        self.url_var.set(source)
        self.url_entry.delete(0, tk.END)
        self.url_entry.insert(0, source)

        short = source if len(source) < 72 else source[:69] + "…"
        self.status.configure(text=f"Loading: {short}")
        new_key = content_cache_key(source)
        if new_key != getattr(self, "_subject_selection_source", ""):
            self._subject_selection = None
            self._subject_selection_source = ""
        if new_key == self._loaded_content_key and self.sequence and not unique_session:
            logger.info("Same video content key — forcing fresh session folder")
            unique_session = True
        logger.info(
            "Loading video — source=%s key=%s… session=%d",
            source[:80],
            new_key[:12],
            session_id,
        )

        from stablewalk.ui.analysis_mode import (
            MODE_FULL,
            MODE_PREVIEW,
            format_analysis_mode_label,
            resolve_gui_max_frames,
        )

        requested_mode = analysis_mode or MODE_PREVIEW
        if str(requested_mode).lower() == MODE_FULL:
            max_frames, resolved_mode = resolve_gui_max_frames(
                mode=MODE_FULL, is_demo=detected_demo is not None
            )
        elif max_frames is None and not self._demo_running:
            max_frames, resolved_mode = resolve_gui_max_frames(
                mode=MODE_PREVIEW,
                is_demo=detected_demo is not None,
            )
        else:
            resolved_mode = MODE_PREVIEW if max_frames is not None else MODE_FULL
        self._analysis_mode = resolved_mode
        self._analysis_max_frames = max_frames
        self.status.configure(
            text=f"{format_analysis_mode_label(resolved_mode, max_frames)}: {short}"
        )

        # Drop the previous video's Overview numbers before the fade so they
        # cannot remain visible while the next example is loading.
        self._reset_session(message="Loading video...")
        self._fade_transition(
            on_done=lambda: self._begin_pipeline(
                source,
                session_id,
                max_frames,
                unique_session,
                getattr(self, "_validate_mode", "quick"),
            )
        )

    def _handle_load_error(
        self,
        message: str,
        *,
        on_complete: PipelineCallback | None = None,
    ) -> None:
        """Show error, unlock UI, optionally try next preset."""
        self._processing = False
        self._hide_overlay()
        if not self._demo_running:
            self._lock_ui(False)
        self.status.configure(text="Load failed — try Next to continue")
        self.video_label.configure(
            image="",
            text=f"Could not load this video.\n{message}\n\nClick Next video ▶",
            fg=WARNING,
        )
        if on_complete:
            on_complete(None, RuntimeError(message))
        if self._chain_next_on_fail:
            presets = list_runnable_presets(verified_only=True, urls_only=True)
            self._next_video_attempts += 1
            if self._next_video_attempts < len(presets):
                self._show_overlay("Trying next verified walking video…")
                self.status.configure(text="Checking next clip…")

                def worker() -> None:
                    idx, preset = self._pick_next_verified_preset(presets)
                    self.root.after(
                        0,
                        lambda: self._retry_next_after_fail(idx, preset, presets),
                    )

                threading.Thread(target=worker, daemon=True).start()
            else:
                self._chain_next_on_fail = False
                self._next_video_attempts = 0
                messagebox.showwarning(
                    "Load failed",
                    "Could not load a suitable walking video.\n"
                    "Use Enter URL… or upload a local full-body walk clip.",
                )

    def _retry_next_after_fail(
        self,
        idx: int | None,
        preset,
        presets: list,
    ) -> None:
        if idx is None or preset is None:
            self._chain_next_on_fail = False
            self._next_video_attempts = 0
            self._hide_overlay()
            messagebox.showwarning(
                "No suitable video",
                "No more verified walking clips passed the pose check.\n"
                "Try Enter URL… or a local video.",
            )
            return
        self._preset_cycle_index = idx
        self.preset_var.set(preset.label)
        self._apply_preset_to_form(preset)
        self._show_overlay(f"Trying next video…\n{preset.label}")
        self.root.after(
            200,
            lambda p=preset: self._load_video(source=p.url, chain_next_on_fail=True),
        )

    def _begin_pipeline(
        self,
        source: str,
        session_id: int,
        max_frames: int | None,
        unique_session: bool,
        validate_mode: bool | str,
    ) -> None:
        self._reset_session(message="Loading video...")
        self._processing = True
        self._lock_ui(True)
        self.progress["value"] = 0
        from stablewalk.ui.tk.analyze_progress import reset_analyze_progress

        reset_analyze_progress(self)

        run_name = derive_run_name(source, unique_session=unique_session)
        pose_backend = self._selected_pose_backend()
        thread = threading.Thread(
            target=self._pipeline_worker,
            args=(source, run_name, session_id, max_frames, validate_mode, pose_backend),
            daemon=True,
        )
        thread.start()

    def _pipeline_worker(
        self,
        source: str,
        run_name: str,
        session_id: int,
        max_frames: int | None,
        validate_mode: bool | str,
        pose_backend: str,
    ) -> None:
        try:
            def on_progress(msg: str, frac: float, **info: object) -> None:
                try:
                    self.root.after(
                        0,
                        lambda m=msg, f=frac, i=dict(info), sid=session_id: self._set_progress(
                            m, f, sid, **i
                        ),
                    )
                except tk.TclError:
                    pass

            from stablewalk.core.pipeline import run_gait_pipeline

            result = run_gait_pipeline(
                source,
                run_name=run_name,
                validate=validate_mode,
                max_frames=max_frames,
                force_reprocess=True,
                on_progress=on_progress,
                pose_backend=pose_backend,
                subject_selection=getattr(self, "_subject_selection", None),
            )
            self._schedule_pipeline_done(result, None, session_id)
        except Exception as exc:
            self._schedule_pipeline_done(None, exc, session_id)

    def _schedule_pipeline_done(
        self,
        result: object,
        err: Exception | None,
        session_id: int,
    ) -> None:
        """Apply pipeline completion on the Tk thread; stash if after() cannot run."""
        self._pending_pipeline_finish = (result, err, session_id)
        try:
            self.root.after(0, self._drain_pending_pipeline)
        except tk.TclError:
            pass

    def _drain_pending_pipeline(self) -> None:
        pending = getattr(self, "_pending_pipeline_finish", None)
        if pending is None:
            return
        self._pending_pipeline_finish = None
        result, err, session_id = pending
        self._pipeline_done(result, err, session_id)

    def _set_progress(
        self,
        message: str,
        fraction: float,
        session_id: int,
        *,
        stage: str | None = None,
        frames_done: int | None = None,
        frames_total: int | None = None,
        fps: float | None = None,
        **_extra: object,
    ) -> None:
        if session_id != self._session_id:
            return
        from stablewalk.ui.tk.analyze_progress import (
            AnalyzeProgressSnapshot,
            infer_stage_from_message,
            stage_label,
            update_analyze_progress,
        )

        stage_key = stage or infer_stage_from_message(message, fraction)
        display = message.strip() or stage_label(stage_key)
        # Prefer canonical stage labels for the checklist animation.
        if stage_key in (
            "loading",
            "extracting",
            "pose",
            "reconstruction",
            "joint_angles",
            "biomechanics",
            "virtual_grf",
            "opensim",
            "report",
            "completed",
        ):
            canonical = stage_label(stage_key)
            if not message or message.lower().startswith(canonical.lower()[:8]):
                display = canonical

        prev = getattr(self, "_analyze_progress_snapshot", None)
        snap = AnalyzeProgressSnapshot(
            stage_key=stage_key,
            message=display,
            fraction=float(fraction),
            frames_done=frames_done
            if frames_done is not None
            else (prev.frames_done if prev else None),
            frames_total=frames_total
            if frames_total is not None
            else (prev.frames_total if prev else None),
            fps=fps if fps is not None else (prev.fps if prev else None),
        )
        update_analyze_progress(self, snap)
        self.status.configure(text=display)
        if self._demo_running:
            self._show_overlay(display)

    def _report_analyze_stage(
        self,
        stage_key: str,
        *,
        fraction: float | None = None,
        message: str | None = None,
        frames_done: int | None = None,
        frames_total: int | None = None,
        fps: float | None = None,
    ) -> None:
        """Update Analyze progress from the UI thread during post-pipeline work."""
        from stablewalk.ui.tk.analyze_progress import (
            AnalyzeProgressSnapshot,
            analyze_progress_active,
            stage_fraction,
            stage_label,
            update_analyze_progress,
        )

        if not analyze_progress_active(self) and stage_key != "completed":
            return
        prev = getattr(self, "_analyze_progress_snapshot", None)
        snap = AnalyzeProgressSnapshot(
            stage_key=stage_key,
            message=message or stage_label(stage_key),
            fraction=stage_fraction(stage_key, within=0.6)
            if fraction is None
            else float(fraction),
            frames_done=frames_done
            if frames_done is not None
            else (prev.frames_done if prev else None),
            frames_total=frames_total
            if frames_total is not None
            else (prev.frames_total if prev else None),
            fps=fps if fps is not None else (prev.fps if prev else None),
        )
        update_analyze_progress(self, snap)
        self.status.configure(text=snap.message)
        try:
            self.root.update_idletasks()
        except tk.TclError:
            pass

    def _pipeline_done(
        self,
        result: PipelineResult | None,
        error: Exception | None,
        session_id: int,
    ) -> None:
        if session_id != self._session_id:
            return

        self._processing = False
        # Keep unfinished controls disabled while post-pipeline analysis continues.
        # Unlock only on hard failure here; success unlocks after analysis completes.
        if error and not self._demo_running:
            self._lock_ui(False)
        elif not error:
            self._sync_analysis_busy_controls()
        try:
            if getattr(self, "progress", None) is not None:
                self.progress["value"] = 100 if result else 0
        except tk.TclError:
            pass

        callback = self._pipeline_callback
        self._pipeline_callback = None

        if error:
            self._hide_overlay()
            from stablewalk.ui.tk.analyze_progress import fail_analyze_progress

            fail_analyze_progress(self, message="Failed")
            if not self._demo_running:
                self._lock_ui(False)
            err_text = str(error)
            if self._chain_next_on_fail:
                self._handle_load_error(err_text, on_complete=callback)
            else:
                if not self._demo_running:
                    messagebox.showerror("Load failed", err_text)
                self._handle_load_error(err_text, on_complete=callback)
            if self._demo_running and not self._chain_next_on_fail:
                self._demo_finished(success=False)
            self._start_pending_video_load_if_any()
            return

        self._chain_next_on_fail = False
        self._next_video_attempts = 0

        assert result is not None
        self._run_metadata = result
        self._active_run_name = result.run_name
        self._set_load_feedback("Video loaded successfully")
        if result.pose_backend_fallback and result.pose_backend_fallback_reason:
            logger.warning(
                "Pose backend fallback (%s → %s): %s",
                result.pose_backend_requested,
                result.pose_backend_used,
                result.pose_backend_fallback_reason,
            )
            if not self._demo_running:
                messagebox.showwarning(
                    "Pose backend fallback",
                    (
                        f"Requested: {result.pose_backend_requested}\n"
                        f"Used: {result.pose_backend_used}\n\n"
                        f"{result.pose_backend_fallback_reason}\n\n"
                        "See docs/SMPL_BACKEND_SETUP.md to enable SMPL extraction."
                    ),
                )
        self._refresh_pose_backend_status_label()
        detected_n = 0
        try:
            import json

            pdata = json.loads(result.poses_path.read_text(encoding="utf-8"))
            detected_n = int(pdata.get("detected_count", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            detected_n = len(self.pose_indices) if self.pose_indices else 0

        if detected_n < MIN_DETECTED_POSE_FRAMES:
            from stablewalk.pose.video import is_video_url
            from stablewalk.ui.tk.analyze_progress import fail_analyze_progress

            fail_analyze_progress(self, message="Failed")
            is_url = is_video_url(result.source)
            hint = (
                "Use Preset → men's walk URL, or Next video ▶ (online clips)."
                if not is_url
                else "Try the other men's walk URL from the Preset list."
            )
            msg = f"Not enough walking poses ({detected_n} frames). {hint}"
            if self._chain_next_on_fail:
                self._handle_load_error(msg, on_complete=callback)
            else:
                messagebox.showwarning("Not a walk video", msg)
                self._handle_load_error(msg, on_complete=callback)
            self._start_pending_video_load_if_any()
            return

        if not self.load_poses(
            result.poses_path,
            fresh=True,
            expected_source=result.source,
            metadata=result,
            wait_for_analysis=False,
        ):
            from stablewalk.ui.tk.analyze_progress import fail_analyze_progress

            fail_analyze_progress(self, message="Failed")
            if self._chain_next_on_fail:
                self._handle_load_error(
                    "No full-body poses detected in this video.",
                    on_complete=callback,
                )
            elif callback:
                callback(None, RuntimeError("No poses detected"))
            else:
                messagebox.showwarning(
                    "No poses",
                    "No full-body poses detected. Try another men's walk preset.",
                )
            self._start_pending_video_load_if_any()
            return

        # Success chrome + caller callback run after analysis worker finishes
        # (see ``_finalize_loaded_poses_after_analysis``).
        self._pipeline_post_load = {
            "result": result,
            "callback": callback,
        }

    def _complete_pipeline_post_load(
        self,
        *,
        result: PipelineResult,
        callback: Callable | None,
    ) -> None:
        """Finish pipeline UI after session analysis is ready on the UI thread."""
        preset = preset_by_label(self.preset_var.get())
        title = preset.label if preset else Path(result.source).name
        if self._active_demo_gait:
            title = self._active_demo_gait.display_name
        self.status.configure(
            text=f"{title} — {result.frame_count} frames — press Play"
        )
        self._show_overlay(f"Loaded\n\n{title}")
        self.root.after(2000, self._hide_overlay)
        if self._show_success_dialog and not self._demo_running:
            messagebox.showinfo(
                "Video loaded",
                f"{title}\n{result.frame_count} frames analyzed.\nPress Play to begin.",
            )

        if callback:
            callback(result.poses_path, None)
        self._opensim_last_dir = config.OPENSIM_DIR / result.run_name
        self._refresh_opensim_status()
        self.root.update_idletasks()
        self._request_dashboard_scroll_sync()
        self._start_pending_video_load_if_any()

    def _start_pending_video_load_if_any(self) -> None:
        pending = self._pending_video_load
        if pending is None:
            return
        self._pending_video_load = None
        self.root.after(
            50,
            lambda p=pending: self._load_video(
                source=p.get("source"),
                on_complete=p.get("on_complete"),
                show_dialog=bool(p.get("show_dialog", True)),
                max_frames=p.get("max_frames"),
                unique_session=bool(p.get("unique_session", True)),
                chain_next_on_fail=bool(p.get("chain_next_on_fail", False)),
                analysis_mode=p.get("analysis_mode"),
            ),
        )

    # ------------------------------------------------------------------ demo
    def _run_demo(self) -> None:
        if self._processing or self._demo_running:
            return

        clip_a, clip_b = get_demo_clips()
        self._demo_running = True
        self._demo_poses_a = None
        self._demo_stability_a = None
        self._show_success_dialog = False
        self.highlight_dof.set(True)
        self.smooth_motion.set(True)
        self._lock_ui(True)

        self._show_overlay(
            "StableWalk Demo\n\n"
            "Step 1/2 — Loading stable walking reference…"
        )
        self._set_load_feedback("Analyzing walking pattern...")

        def after_a(path: Path | None, err: Exception | None) -> None:
            if err or not path:
                self._demo_finished(success=False)
                return
            self._demo_poses_a = path
            self._demo_stability_a = self._stability
            self._show_overlay(
                f"Step 1 complete — {clip_a.label}\n"
                f"Stability: {self._stability.label if self._stability else '—'} "
                f"({self._stability.score:.0f}/100)\n\n"
                "Playing gait preview…"
            )
            self._demo_playback_then(
                delay_ms=DEMO_PLAYBACK_MS,
                next_step=lambda: self._demo_load_b(clip_b),
            )

        self.url_var.set(clip_a.source)
        self._load_video(
            source=clip_a.source,
            on_complete=after_a,
            show_dialog=False,
            max_frames=DEMO_MAX_FRAMES,
            unique_session=True,
            analysis_mode="preview",
        )

    def _demo_playback_then(
        self,
        *,
        delay_ms: int,
        next_step: Callable[[], None],
    ) -> None:
        if not self.pose_indices:
            self.root.after(delay_ms, next_step)
            return
        self.playing = True
        self.highlight_dof.set(True)
        self._sync_play_buttons()
        self._playback_pos = 0.0
        self._show_pose_at(0)
        self._schedule_tick()

        def stop_and_continue() -> None:
            self.playing = False
            self._sync_play_buttons()
            self._cancel_timer()
            next_step()

        self.root.after(delay_ms, stop_and_continue)

    def _demo_load_b(self, clip_b) -> None:
        self._fade_transition(
            on_done=lambda: self._show_overlay(
                "Step 2/2 — Loading alternate walking pattern…\n"
                "Comparing stability…"
            ),
        )
        self.root.after(
            DEMO_PAUSE_MS // 2,
            lambda: self._set_load_feedback("Comparing stability..."),
        )

        def after_b(path: Path | None, err: Exception | None) -> None:
            if err or not path or not self._demo_poses_a:
                self._demo_finished(success=False)
                return
            try:
                from stablewalk.analysis.comparison import compare_pose_files

                self._compare = compare_pose_files(
                    str(self._demo_poses_a),
                    str(path),
                )
                self._show_compare_note(self._compare.summary)
                self._update_compare_chart()
            except Exception as exc:
                logger.warning("Demo compare failed: %s", exc)

            stab_a = self._demo_stability_a
            stab_b = self._stability
            delta = ""
            if stab_a and stab_b:
                delta = (
                    f"\n\nStability: {stab_a.label} ({stab_a.score:.0f}) → "
                    f"{stab_b.label} ({stab_b.score:.0f})"
                )
            sym = ""
            if stab_b:
                sym = (
                    f"\nLeg symmetry: {stab_b.symmetry_score:.0%}  "
                    f"(knee Δ {stab_b.knee_symmetry_deg:.1f}°)"
                )

            self._show_overlay(
                "Comparison Complete\n\n"
                "Gait extraction · DOF · 3D skeleton · Robot simulation"
                f"{delta}{sym}"
            )
            self._set_load_feedback("Comparison Complete")
            self.status.configure(text="Comparison Complete — demo finished")
            self._demo_playback_then(
                delay_ms=DEMO_FINAL_HOLD_MS,
                next_step=lambda: self._demo_finished(success=True),
            )

        self.url_var.set(clip_b.source)
        self.root.after(
            DEMO_PAUSE_MS,
            lambda: self._load_video(
                source=clip_b.source,
                on_complete=after_b,
                show_dialog=False,
                max_frames=DEMO_MAX_FRAMES,
                unique_session=True,
                analysis_mode="preview",
            ),
        )

    def _demo_finished(self, *, success: bool) -> None:
        self._demo_running = False
        self._show_success_dialog = True
        self.playing = False
        self._sync_play_buttons()
        self._cancel_timer()
        self._lock_ui(False)
        if success:
            messagebox.showinfo(
                "Comparison Complete",
                "Demo finished.\n\n"
                "Two walking patterns were analyzed and compared.\n"
                "Review stability scores and knee-angle charts.",
            )
        else:
            messagebox.showerror(
                "Demo interrupted",
                "Demo could not complete.\n"
                "Check your internet connection (URL video) or add videos to data/input/.",
            )
        self.root.after(2500, self._hide_overlay)

    def _menu_open(self) -> None:
        path = filedialog.askopenfilename(
            title="Open pose JSON",
            initialdir=str(config.POSES_DIR),
            filetypes=[("Pose JSON", "*_poses.json"), ("JSON", "*.json"), ("All", "*.*")],
        )
        if path:
            self._reset_session(message="Loading pose file...")
            self.load_poses(path, fresh=True)

    def _compare_gait(self) -> None:
        if not self.sequence:
            messagebox.showinfo("Compare", "Load primary gait data first.")
            return
        path = filedialog.askopenfilename(
            title="Select comparison pose JSON",
            initialdir=str(config.POSES_DIR),
            filetypes=[("Pose JSON", "*_poses.json"), ("JSON", "*.json")],
        )
        if not path or not self._poses_path:
            return
        try:
            from stablewalk.analysis.comparison import compare_pose_files

            self._compare = compare_pose_files(str(self._poses_path), path)
            self._show_compare_note(self._compare.summary)
            self._update_compare_chart()
        except Exception as exc:
            messagebox.showerror("Compare error", str(exc))

    def _export_json(self) -> None:
        self._export_analysis(default_ext=".json")

    def _require_analysis_complete(self, title: str) -> bool:
        if self._analysis_workflow_busy():
            messagebox.showinfo(
                title,
                "Wait until analysis finishes before generating the final report.",
            )
            return False
        return True

    def _export_analysis(self, default_ext: str | None = None) -> None:
        if not self._require_analysis_complete("Export"):
            return
        if not self.sequence:
            messagebox.showinfo("Export", "Load pose data first (Run Full Analysis).")
            return
        path = filedialog.asksaveasfilename(
            title="Export gait analysis",
            initialdir=str(config.POSES_DIR),
            initialfile="gait_analysis.json",
            defaultextension=default_ext or ".json",
            filetypes=[
                ("JSON (keypoints, angles, velocities)", "*.json"),
                ("CSV (flat table)", "*.csv"),
            ],
        )
        if not path:
            return
        try:
            out = export_analysis(self.sequence, path)
            self.status.configure(text=f"Exported → {out.name}")
            messagebox.showinfo(
                "Export complete",
                f"Saved analysis:\n{out}\n\n"
                "Includes keypoints, joint angles, and velocities.",
            )
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))

    # ── Stability breakdown panel ────────────────────────────────────────
    @staticmethod
    def _stability_color(classification: str) -> str:
        return {
            "Stable": SUCCESS,
            "Moderate": INFO,
            "Unstable": WARNING,
        }.get(classification, MUTED)

    def _update_stability_panel(self, result: "StabilityResult | None") -> None:
        """Refresh the Gait Analysis Summary with three semantic domain cards."""
        if result is None:
            return

        from stablewalk.analysis.gait_analysis_summary import (
            build_gait_analysis_summary,
            format_summary_display,
        )
        from stablewalk.analysis.stability_validity import (
            assess_stability_result_validity,
        )

        summary = result.gait_summary or build_gait_analysis_summary(result)
        display = format_summary_display(summary)
        validity = result.validity or assess_stability_result_validity(result)

        self._last_stability_result = result
        self._update_gait_summary_cards_panel(result, summary, display, validity)
        self._update_overview_clinical_summary()

        # Legacy widget aliases for scripts and the Details dialog (hidden from dashboard).
        if hasattr(self, "lbl_stab_score"):
            self.lbl_stab_score.configure(
                text=display["movement_stability"],
                fg=TEXT,
            )
        cat_lbl = getattr(self, "lbl_stab_category", None)
        if cat_lbl is not None:
            cat_lbl.configure(
                text=summary.legacy_classification,
                foreground=self._stability_color(summary.legacy_classification),
            )
        legacy_lbl = getattr(self, "lbl_stab_legacy", None)
        if legacy_lbl is not None:
            note_parts = [
                p for p in (display["legacy_note"], display.get("partial_scores_note", "")) if p
            ]
            legacy_lbl.configure(text=" · ".join(note_parts), fg=MUTED)
        explain_lbl = getattr(self, "lbl_gait_explanation", None)
        if explain_lbl is not None:
            explain_lbl.configure(text="", fg=MUTED)

        view_lbl = getattr(self, "lbl_stab_view", None)
        if view_lbl is not None:
            if result.view_display_name:
                view_lbl.configure(
                    text=f"View: {result.view_display_name}",
                    fg=TEXT,
                )
            else:
                view_lbl.configure(text="")

        slots = getattr(self, "_walk_summary_slots", None)
        if slots:
            from stablewalk.analysis.stability_scoring import MetricResult
            from stablewalk.ui.dashboard_interpretability import interpret_domain_metric

            for _title_lbl, value_lbl, interp_lbl in slots:
                value_lbl.configure(text="—", fg=TEXT)
                if interp_lbl is not None:
                    interp_lbl.configure(text="")
            metric_by_key: dict[str, MetricResult] = {
                m.key: m for m in result.metrics
            }
            metric_map = {
                "temporal_symmetry": 0,
                "spatial_symmetry": 1,
                "pelvis_stability": 2,
                "cycle_consistency": 3,
            }

            color = self._stability_color(result.classification)
            for key, idx in metric_map.items():
                if idx >= len(slots):
                    continue
                m = metric_by_key.get(key)
                _title_lbl, value_lbl, interp_lbl = slots[idx]
                if m is None or m.availability == "UNAVAILABLE" or m.score is None:
                    value_lbl.configure(text="—", fg=MUTED)
                    if interp_lbl is not None:
                        interp_lbl.configure(text="")
                    continue
                slot_display = f"{m.score:.0f}"
                if m.availability == "LOW_CONFIDENCE":
                    fg = INFO if m.score >= 60 else WARNING
                else:
                    fg = color if m.score < 60 else TEXT
                value_lbl.configure(text=slot_display, fg=fg)
                if interp_lbl is not None and key in ("temporal_symmetry", "pelvis_stability"):
                    names = {
                        "temporal_symmetry": "Temporal Symmetry",
                        "pelvis_stability": "Pelvis Stability",
                    }
                    card = interpret_domain_metric(m, display_name=names[key])
                    interp_lbl.configure(text=f'"{card.sentence}"  ·  {card.confidence}')

        btn = getattr(self, "btn_walk_summary_details", None)
        if btn is not None:
            btn.configure(state=tk.NORMAL)
        adv_btn = getattr(self, "btn_advanced_analysis", None) or getattr(
            self, "btn_gait_metrics_details", None
        )
        if adv_btn is not None:
            adv_btn.configure(state=tk.NORMAL)

        self._update_gait_metric_cards(result, summary, display, validity)

    def _update_overview_clinical_summary(self) -> None:
        """Fill Overview KPI cards, interpretation, findings, and session badge."""
        cache = getattr(self, "_session_analysis", None)
        generation = int(getattr(cache, "generation", 0) or 0)
        ready = bool(getattr(cache, "ready", False))
        last_gen = getattr(self, "_overview_clinical_generation", None)
        last_ready = bool(getattr(self, "_overview_clinical_ready", False))
        stale = last_gen != generation or last_ready != ready
        playing = bool(getattr(self, "playing", False))
        # Skip restyle churn during playback unless analysis results just landed.
        if playing and not stale:
            return
        from stablewalk.ui.overview_clinical_summary import (
            INSUFFICIENT_EVIDENCE,
            NOT_AVAILABLE,
            build_data_quality_warning,
            build_key_limitations,
            build_overview_interpretation,
            build_overview_key_findings_texts,
            card_description,
            card_provenance,
            classify_overview_metric,
            extract_confidence_visualization,
            extract_overview_session_badge,
            extract_overview_summary_fields,
            primary_finding_inputs,
            status_label_for_overview,
            walking_speed_available,
        )
        from stablewalk.ui.summary_metric_style import metric_visual_style
        from stablewalk.ui.theme import (
            BORDER,
            MUTED,
            MUTED_DIM,
            TEXT,
            create_tooltip,
        )

        inputs = primary_finding_inputs(self)
        fields = extract_overview_summary_fields(self)
        conf_viz = extract_confidence_visualization(self)

        # Shared metric contracts for Overview / Biomech / Results / exports.
        try:
            from stablewalk.metrics.session_view import build_session_metric_snapshots

            badge_name = ""
            try:
                badge_name = str(
                    (extract_overview_session_badge(self) or {}).get("video_name") or ""
                )
            except Exception:
                badge_name = ""
            stability_result = getattr(self, "_stability_result", None) or getattr(
                self, "_last_stability_result", None
            )
            self._session_metric_view = build_session_metric_snapshots(
                biomechanical=getattr(self, "_biomech_analysis", None),
                cycles=getattr(self, "_gait_cycle", None),
                gait_summary=getattr(stability_result, "gait_summary", None)
                if stability_result is not None
                else None,
                analysis_confidence_pct=conf_viz.get("pct"),
                analysis_confidence_level=inputs.get("analysis_confidence_level"),
                session_label=badge_name,
                contact=getattr(self, "_foot_contact", None),
                spectral=getattr(self, "_spectral_report", None),
            )
        except Exception:
            self._session_metric_view = None

        has_session = bool(
            getattr(self, "_run_metadata", None)
            or getattr(self, "skeleton_player", None)
            or getattr(self, "_biomech_analysis", None)
        )
        analyzing = bool(
            getattr(self, "_analysis_running", False)
            or getattr(self, "_pipeline_running", False)
            or getattr(self, "_analysis_compute_active", False)
            or getattr(self, "_processing", False)
        )

        # Session badge (page header)
        try:
            badge = extract_overview_session_badge(self)
            for key, lbl in (getattr(self, "_overview_session_badge_labels", None) or {}).items():
                text = badge.get(key, "—")
                lbl.configure(
                    text=text,
                    fg=TEXT if text not in ("—", NOT_AVAILABLE) else MUTED,
                )
        except Exception:
            logger.debug("Overview session badge update failed", exc_info=True)

        # Empty / loading / ready state strip
        try:
            state_bar = getattr(self, "_overview_state_bar", None)
            state_lbl = getattr(self, "lbl_overview_state", None)
            if state_lbl is not None and state_bar is not None:
                # Keep the Overview dense: only flash the status strip while analyzing.
                if analyzing:
                    state_lbl.configure(
                        text="Analysis in progress — indicators update when processing finishes.",
                        fg=TEXT,
                    )
                    state_bar.grid()
                else:
                    state_bar.grid_remove()
        except tk.TclError:
            pass

        cards = getattr(self, "_overview_clinical_cards", None) or {}
        labels = getattr(self, "_overview_clinical_value_labels", None) or {}
        secondary_keys = set(getattr(self, "_overview_secondary_order", ()) or ())
        if cards:
            for key, card in cards.items():
                display, tip = fields.get(
                    key, (NOT_AVAILABLE, "value not available")
                )
                level = classify_overview_metric(
                    key,
                    display=display,
                    gait_quality_score=inputs.get("gait_quality_score"),
                    symmetry_pct=inputs.get("symmetry_pct"),
                    cadence_spm=inputs.get("cadence_spm"),
                    stable_pct=inputs.get("stable_pct"),
                    stability_margin_m=inputs.get("stability_margin_m"),
                    detected_cycles=inputs.get("detected_cycles"),
                    confidence_pct=conf_viz.get("pct"),
                    confidence_level=inputs.get("analysis_confidence_level"),
                )
                style = metric_visual_style(level)
                available = display not in (NOT_AVAILABLE, INSUFFICIENT_EVIDENCE)
                try:
                    from stablewalk.ui.theme import (
                        FONT_KPI_HERO,
                        FONT_KPI_VALUE_SM,
                    )

                    value_text = display
                    value_font = (
                        FONT_KPI_VALUE_SM if key in secondary_keys else FONT_KPI_HERO
                    )
                    if available and key == "symmetry" and display.endswith("%"):
                        value_text = display[:-1]
                    elif available and key == "gait_quality" and "/100" in display:
                        value_text = display.split("/", 1)[0]
                    elif available and key == "stability_state" and display.endswith("%"):
                        value_text = display[:-1]
                    # Never show unavailable metrics as 0 / 0%.
                    if not available:
                        value_text = display if display else NOT_AVAILABLE
                    card["value"].configure(
                        text=value_text,
                        fg=style.value_fg if available else MUTED,
                        font=value_font,
                        justify=tk.LEFT,
                    )
                    if not playing:
                        try:
                            from stablewalk.ui.tk.ui_motion import flash_metric_value

                            flash_metric_value(
                                self,
                                card["value"],
                                settle_fg=style.value_fg if available else MUTED,
                                key=f"overview:{key}",
                            )
                        except Exception:
                            pass
                    card["desc"].configure(
                        text=card_provenance(key) or card_description(key),
                        fg=MUTED_DIM,
                    )
                    status_text = (
                        status_label_for_overview(
                            key,
                            level,
                            display=display,
                            cadence_spm=inputs.get("cadence_spm"),
                        )
                        if available
                        else display
                    )
                    if (
                        available
                        and level == "unavailable"
                        and "low" in display.lower()
                    ):
                        status_text = "Low confidence"
                    card["status"].configure(
                        text=status_text or (NOT_AVAILABLE if not available else ""),
                        fg=style.value_fg if available else MUTED,
                    )
                    accent_bg = style.border if available else BORDER
                    card["accent"].configure(bg=accent_bg)
                    card["frame"].configure(
                        highlightbackground=style.border if available else BORDER,
                        highlightcolor=style.border if available else BORDER,
                    )
                    full_tip = tip
                    tech = card_description(key)
                    if tech and tech not in (tip or ""):
                        full_tip = f"{tip}\n\n{tech}" if tip else tech
                    create_tooltip(card["frame"], full_tip, wraplength=360)
                    create_tooltip(card["value"], full_tip, wraplength=360)
                except tk.TclError:
                    pass
        elif labels:
            for key, lbl in labels.items():
                display, tip = fields.get(
                    key, (NOT_AVAILABLE, "value not available")
                )
                try:
                    lbl.configure(
                        text=display,
                        fg=MUTED if display == NOT_AVAILABLE else TEXT,
                    )
                    create_tooltip(lbl, tip, wraplength=360)
                except tk.TclError:
                    pass

        bullet_labels = getattr(self, "_overview_clinical_bullet_labels", None) or []
        try:
            findings = build_overview_key_findings_texts(
                usable_cycles=inputs.get("usable_cycles"),
                detected_cycles=inputs.get("detected_cycles"),
                cadence_spm=inputs.get("cadence_spm"),
                symmetry_pct=inputs.get("symmetry_pct"),
                stable_pct=inputs.get("stable_pct"),
                stability_margin_m=inputs.get("stability_margin_m"),
                analysis_confidence_level=inputs.get("analysis_confidence_level"),
                gait_quality_score=inputs.get("gait_quality_score"),
                tracking_score=inputs.get("tracking_score"),
                max_findings=3,
            )
            for i, lbl in enumerate(bullet_labels):
                if i < len(findings):
                    lbl.configure(text=findings[i], fg=TEXT)
                elif i == 0 and not findings:
                    lbl.configure(
                        text="Key findings appear after analysis.",
                        fg=MUTED,
                    )
                else:
                    lbl.configure(text="", fg=MUTED)
        except Exception:
            logger.debug("Overview key findings update failed", exc_info=True)

        finding_lbl = getattr(self, "lbl_overview_primary_finding", None)
        if finding_lbl is not None:
            try:
                finding_lbl.configure(
                    text=build_overview_interpretation(
                        usable_cycles=inputs.get("usable_cycles"),
                        detected_cycles=inputs.get("detected_cycles"),
                        cadence_spm=inputs.get("cadence_spm"),
                        symmetry_pct=inputs.get("symmetry_pct"),
                        stable_pct=inputs.get("stable_pct"),
                        stability_margin_m=inputs.get("stability_margin_m"),
                        analysis_confidence_level=inputs.get(
                            "analysis_confidence_level"
                        ),
                        gait_quality_score=inputs.get("gait_quality_score"),
                        has_session=has_session,
                        analyzing=analyzing,
                    )
                )
            except tk.TclError:
                pass
            except Exception:
                logger.debug("Overview primary finding update failed", exc_info=True)

        # Data quality warning — only when confidence/completeness is insufficient
        try:
            dq_banner = getattr(self, "_overview_data_quality_banner", None)
            warning = build_data_quality_warning(
                analysis_confidence_level=inputs.get("analysis_confidence_level"),
                confidence_pct=conf_viz.get("pct"),
                usable_cycles=inputs.get("usable_cycles"),
                detected_cycles=inputs.get("detected_cycles"),
                tracking_score=inputs.get("tracking_score"),
            )
            if dq_banner is not None:
                if warning:
                    dq_banner.label.configure(text=warning)
                    dq_banner.frame.grid()
                else:
                    dq_banner.frame.grid_remove()
        except Exception:
            logger.debug("Overview data quality banner update failed", exc_info=True)

        # Key Limitations (collapsible; keep grid when notes exist)
        note_labels = getattr(self, "_overview_system_note_labels", None) or []
        limitations_card = getattr(self, "_overview_limitations_card", None)
        try:
            ik_state = getattr(self, "_opensim_ik_state", None)
            notes = build_key_limitations(
                tracking_score=inputs.get("tracking_score"),
                pose_confidence_level=conf_viz.get("pose_level"),
                walking_speed_available=walking_speed_available(self),
                opensim_enabled=bool(getattr(self, "_opensim_sdk_available", False)),
                opensim_ik_completed=(ik_state == "Completed"),
                monocular=True,
            )
            if limitations_card is not None:
                summary_card = getattr(self, "_overview_summary_card", None)
                if notes:
                    limitations_card.grid()
                    if summary_card is not None:
                        try:
                            summary_card.grid_configure(columnspan=1)
                        except tk.TclError:
                            pass
                else:
                    limitations_card.grid_remove()
                    if summary_card is not None:
                        try:
                            summary_card.grid_configure(columnspan=2)
                        except tk.TclError:
                            pass
            for i, lbl in enumerate(note_labels):
                if i < len(notes):
                    lbl.configure(text=notes[i], fg=TEXT)
                else:
                    lbl.configure(text="", fg=MUTED)
        except Exception:
            logger.debug("Overview key limitations update failed", exc_info=True)

        try:
            host = getattr(self, "_overview_clinical_summary", None)
            if host is not None:
                width = max(480, int(host.winfo_width()) - 24)
                col_w = max(200, (width * 3) // 5 - 32)
                lim_w = max(160, (width * 2) // 5 - 32)
                for lbl in bullet_labels:
                    lbl.configure(wraplength=col_w)
                for lbl in note_labels:
                    lbl.configure(wraplength=lim_w)
                disc = getattr(self, "lbl_overview_visible_disclaimer", None)
                if disc is not None:
                    disc.configure(wraplength=max(480, width))
        except tk.TclError:
            pass

        self._overview_clinical_generation = generation
        self._overview_clinical_ready = ready
        if ready and getattr(self, "_active_demo_gait", None) is not None:
            self._capture_demo_comparison_snapshot()
        else:
            self._refresh_overview_recording_context()

    def _update_gait_summary_cards_panel(
        self,
        result: "StabilityResult",
        summary,
        display: dict[str, str],
        validity,
    ) -> None:
        """Populate Overview sidebar — three scores and short explanations."""
        from stablewalk.ui.dashboard_interpretability import (
            METRIC_HELP,
            format_analysis_confidence_level,
            format_score_over_100,
            gait_quality_evidence_badge,
            interpret_analysis_confidence,
            interpret_gait_quality,
            interpret_movement_stability,
            truncate_dashboard_explanation,
        )
        from stablewalk.ui.tk.kpi_cards import (
            split_value_unit,
            update_kpi_card,
        )

        kpis = getattr(self, "_overview_kpi_cards", None) or {}

        def _score_quality(score: float | None) -> str:
            if score is None:
                return "unavailable"
            if score >= 70:
                return "normal"
            if score >= 45:
                return "borderline"
            return "abnormal"

        ms = summary.movement_stability.score
        ms_kpi = kpis.get("movement_stability")
        ms_display = format_score_over_100(ms)
        if ms_kpi is not None:
            value, unit = split_value_unit(ms_display)
            available = ms is not None
            quality = _score_quality(ms)
            update_kpi_card(
                ms_kpi,
                value=value if available else "—",
                unit=unit if available else "",
                available=available,
                numeric=float(ms) if available else None,
                fraction=(float(ms) / 100.0) if available else None,
                quality=quality,  # type: ignore[arg-type]
                tooltip=METRIC_HELP.get("movement_stability", "Movement Stability"),
            )
        else:
            ms_lbl = getattr(self, "lbl_summary_ms_value", None)
            if ms_lbl is not None:
                ms_fg = TEXT if ms is not None else MUTED
                if ms is not None and ms >= 70:
                    ms_fg = SUCCESS
                elif ms is not None and ms < 45:
                    ms_fg = WARNING
                ms_lbl.configure(text=ms_display, fg=ms_fg)

        ms_explain = getattr(self, "lbl_summary_ms_explain", None)
        demo_interp = None
        demo = getattr(self, "_active_demo_gait", None)
        if demo is not None:
            usable, detected = self._resolved_gait_cycle_count()
            if usable is None:
                usable = result.usable_gait_cycles
            demo_interp = demo_gait_interpretation(
                demo.key,
                usable_cycles=usable or 0,
                detected_cycles=detected or 0,
                completeness_pct=result.completeness_pct,
                movement_stability_score=summary.movement_stability.score,
                gait_quality_score=summary.gait_quality.score,
            )
        headline = getattr(self, "lbl_summary_demo_headline", None)
        compare_lbl = getattr(self, "lbl_summary_demo_compare", None)
        if headline is not None:
            if demo_interp is not None:
                # Use module-level theme colors — a local ``import WARNING`` here
                # would shadow the name for the whole function and break later
                # badge styling after a failed local bind.
                from stablewalk.ui.theme import ACCENT_ALT, ORANGE

                fg = ORANGE
                if demo.key == "abnormal":
                    fg = WARNING
                elif demo.key == "athletic":
                    fg = ACCENT_ALT
                headline.configure(text=demo_interp.category_headline, fg=fg)
            else:
                headline.configure(text="")
        if compare_lbl is not None:
            if demo_interp is not None:
                compare_lbl.configure(text=demo_interp.teacher_compare, fg=MUTED)
            else:
                compare_lbl.configure(text="")
        if ms_explain is not None:
            ms_text = (
                demo_interp.movement_stability
                if demo_interp is not None
                else interpret_movement_stability(result).sentence
            )
            ms_explain.configure(
                text=truncate_dashboard_explanation(ms_text, max_len=160)
            )

        gq = summary.gait_quality.score
        gq_kpi = kpis.get("gait_quality")
        gq_display = format_score_over_100(gq)
        badge = gait_quality_evidence_badge(
            summary, usable_gait_cycles=result.usable_gait_cycles
        )
        if gq_kpi is not None:
            value, unit = split_value_unit(gq_display)
            available = gq is not None
            quality = _score_quality(gq)
            tip = METRIC_HELP.get("gait_quality", "Gait Coordination")
            if badge:
                tip = f"{tip}\n{badge}"
            update_kpi_card(
                gq_kpi,
                value=value if available else "—",
                unit=unit if available else "",
                available=available,
                numeric=float(gq) if available else None,
                fraction=(float(gq) / 100.0) if available else None,
                quality=quality,  # type: ignore[arg-type]
                tooltip=tip,
            )
            # Preserve evidence badge text on the quality line.
            if badge:
                existing = (gq_kpi.quality_lbl.cget("text") or "").strip()
                gq_kpi.quality_lbl.configure(
                    text=f"{existing} · {badge}" if existing else badge,
                    fg=WARNING,
                )
        else:
            gq_lbl = getattr(self, "lbl_summary_gq_value", None)
            if gq_lbl is not None:
                gq_fg = TEXT if gq is not None else MUTED
                if gq is not None and gq < 45:
                    gq_fg = WARNING
                elif gq is not None and gq >= 68:
                    gq_fg = SUCCESS
                gq_lbl.configure(text=gq_display, fg=gq_fg)
            badge_lbl = getattr(self, "lbl_summary_gq_badge", None)
            if badge_lbl is not None:
                badge_lbl.configure(text=badge or "", fg=WARNING if badge else MUTED)

        gq_explain = getattr(self, "lbl_summary_gq_explain", None)
        if gq_explain is not None:
            gq_text = (
                demo_interp.gait_quality
                if demo_interp is not None
                else interpret_gait_quality(result).sentence
            )
            gq_explain.configure(
                text=truncate_dashboard_explanation(gq_text, max_len=160)
            )

        level = summary.analysis_confidence.level
        ac_kpi = kpis.get("analysis_confidence")
        ac_display = format_analysis_confidence_level(level)
        ac_quality_map = {
            "HIGH": "normal",
            "MODERATE": "borderline",
            "LOW": "abnormal",
            "INSUFFICIENT": "abnormal",
        }
        if ac_kpi is not None:
            available = bool(level)
            update_kpi_card(
                ac_kpi,
                value=ac_display if available else "—",
                unit="",
                available=available,
                numeric=None,
                quality=ac_quality_map.get(level, "neutral"),  # type: ignore[arg-type]
                tooltip=METRIC_HELP.get("analysis_confidence", "Analysis Confidence"),
            )
        else:
            ac_lbl = getattr(self, "lbl_summary_ac_level", None)
            if ac_lbl is not None:
                ac_lbl.configure(
                    text=ac_display,
                    fg={
                        "HIGH": SUCCESS,
                        "MODERATE": INFO,
                        "LOW": WARNING,
                        "INSUFFICIENT": WARNING,
                    }.get(level, MUTED),
                )
        ac_explain = getattr(self, "lbl_summary_ac_explain", None)
        if ac_explain is not None:
            ac_text = (
                demo_interp.analysis_confidence
                if demo_interp is not None
                else interpret_analysis_confidence(result).sentence
            )
            ac_explain.configure(
                text=truncate_dashboard_explanation(ac_text, max_len=160)
            )

        overview_usable = getattr(self, "lbl_overview_gait_cycles_usable", None)
        overview_completeness = getattr(self, "lbl_overview_gait_cycles_completeness", None)
        usable, detected = self._resolved_gait_cycle_count()
        if usable is None:
            usable = result.usable_gait_cycles
        self._sync_gait_cycles_labels(
            usable=usable,
            detected=detected,
            completeness_pct=result.completeness_pct,
        )

    def _resolved_gait_cycle_count(self) -> tuple[int | None, int | None]:
        """
        Return (usable_cycles, detected_cycles) from stability or gait analysis.

        Stability ``usable_gait_cycles`` can be 0 while gait cycle detection still
        found cycles — prefer the best available count for the Advanced tab cards.
        """
        usable: int | None = None
        detected: int | None = None
        if self._biomech is not None:
            usable = self._biomech.usable_gait_cycles
        if self._gait_cycle is not None:
            detected = len(self._gait_cycle.cycles)
            if self._biomech and self._biomech.domain_evidence:
                for key in ("spatial_symmetry", "cycle_consistency", "temporal_symmetry"):
                    text = self._biomech.domain_evidence.get(key, "")
                    if "usable" in text.lower():
                        import re

                        m = re.search(r"(\d+)\s+usable", text, re.I)
                        if m:
                            parsed = int(m.group(1))
                            if usable is None:
                                usable = parsed
                            elif parsed < usable:
                                usable = parsed
                            break
        return usable, detected

    def _sync_gait_cycles_labels(
        self,
        *,
        usable: int | None = None,
        detected: int | None = None,
        completeness_pct: float | None = None,
    ) -> None:
        """Keep Overview + Advanced gait-cycle cards in sync."""
        from stablewalk.ui.theme import MUTED, TEXT, WARNING

        if usable is None and detected is None:
            usable, detected = self._resolved_gait_cycle_count()
        elif usable is None or detected is None:
            resolved_u, resolved_d = self._resolved_gait_cycle_count()
            if usable is None:
                usable = resolved_u
            if detected is None:
                detected = resolved_d
        if completeness_pct is None and self._biomech is not None:
            completeness_pct = self._biomech.completeness_pct

        if detected and usable is not None and detected != usable:
            cycles_text = f"{detected} detected · {usable} usable"
            cycles_fg = WARNING if usable == 0 else TEXT
        elif usable is not None and usable > 0:
            cycles_text = f"{usable} usable cycle{'s' if usable != 1 else ''}"
            cycles_fg = TEXT
        elif detected and detected > 0:
            cycles_text = f"{detected} detected · 0 usable"
            cycles_fg = WARNING
        elif self._biomech is not None or self._gait_cycle is not None:
            cycles_text = "0 usable cycles"
            cycles_fg = MUTED
        else:
            cycles_text = "Not computed yet"
            cycles_fg = MUTED

        for attr in ("lbl_gait_card_cycles_value",):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text=cycles_text, fg=cycles_fg)

        overview_usable = getattr(self, "lbl_overview_gait_cycles_usable", None)
        if overview_usable is not None:
            if usable is None and not detected:
                overview_text = "Usable: —"
            elif "detected" in cycles_text or "usable cycle" in cycles_text:
                overview_text = cycles_text
            else:
                overview_text = f"Usable: {cycles_text}"
            overview_usable.configure(text=overview_text, fg=cycles_fg)

        self._sync_demo_category_compare_strip()

        advanced_comp_text = (
            f"Completeness: {completeness_pct:.0f}%"
            if completeness_pct is not None
            else "Completeness: Available after full analysis"
        )
        advanced_comp = getattr(self, "lbl_gait_card_completeness_value", None)
        if advanced_comp is not None:
            advanced_comp.configure(text=advanced_comp_text, fg=MUTED)
        overview_comp = getattr(self, "lbl_overview_gait_cycles_completeness", None)
        if overview_comp is not None:
            overview_comp.configure(
                text=(
                    f"Completeness: {completeness_pct:.0f}%"
                    if completeness_pct is not None
                    else "Completeness: —"
                ),
                fg=MUTED,
            )

    def _load_rts_report_for_session(self) -> dict | None:
        """Return cached or on-disk Real-to-Sim pipeline report for current session."""
        cached = getattr(self, "_last_rts_report", None)
        if cached is not None:
            return cached
        run_name = Path(self._resolve_session_video_source() or "session").stem or "session"
        report_path = (
            config.MOTION_REFERENCE_EXPORT_DIR
            / run_name
            / "real_to_sim_pipeline_report.json"
        )
        if not report_path.is_file():
            return None
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _selected_pose_backend(self) -> str:
        cached = getattr(self, "_pose_backend_cached", None)
        var = getattr(self, "pose_backend_var", None)
        if var is None:
            return cached or config.POSE_BACKEND
        try:
            value = (var.get() or "mediapipe").strip().lower()
        except tk.TclError:
            return cached or config.POSE_BACKEND
        if value not in ("mediapipe", "smpl", "auto"):
            value = "mediapipe"
        self._pose_backend_cached = value
        return value

    def _refresh_pose_backend_status_label(self) -> None:
        lbl = getattr(self, "lbl_pose_backend_status", None)
        if lbl is None:
            return
        from stablewalk.ui.theme import MUTED as THEME_MUTED
        from stablewalk.pose.backends.smpl_validation import validate_smpl_assets

        mode = self._selected_pose_backend()
        if mode == "mediapipe":
            lbl.configure(text="MediaPipe (default landmark tracker)", fg=THEME_MUTED)
            return
        if mode == "smpl":
            val = validate_smpl_assets()
            lbl.configure(
                text="SMPL ready" if val.ready else val.summary()[:120],
                fg=SUCCESS if val.ready else THEME_MUTED,
            )
            return
        val = validate_smpl_assets()
        if val.ready:
            lbl.configure(text="Auto: SMPL available", fg=SUCCESS)
        else:
            lbl.configure(
                text=f"Auto: will fall back — {val.summary()[:80]}…",
                fg=THEME_MUTED,
            )

    def _refresh_real_to_sim_advanced_panel(self) -> None:
        """Update Advanced tab Real-to-Sim 4-stage summary."""
        from stablewalk.ui.scientific_labels import LABEL_VIRTUAL_GRF_PANEL
        from stablewalk.ui.theme import INFO, MUTED as THEME_MUTED, SUCCESS, TEXT, WARNING

        s1 = getattr(self, "lbl_rts_stage1", None)
        s2 = getattr(self, "lbl_rts_stage2", None)
        s3 = getattr(self, "lbl_rts_stage3", None)
        s4 = getattr(self, "lbl_rts_stage4", None)
        summary = getattr(self, "lbl_rts_summary", None)
        if s1 is None:
            return

        if self.gait_motion is None or self._gait_cycle is None:
            for lbl, default in (
                (s1, "Gait style: — (analyze a video first)"),
                (s2, "Human → Unitree G1 scale: —"),
                (s3, "AMP reference: not exported"),
                (s4, f"{LABEL_VIRTUAL_GRF_PANEL}: —"),
            ):
                if lbl is not None:
                    lbl.configure(text=default, fg=THEME_MUTED)
            if summary is not None:
                summary.configure(
                    text="Load a walking video, then use Real-to-Sim Pipeline in Data & Export.",
                    fg=THEME_MUTED,
                )
            return

        report = self._load_rts_report_for_session()
        stage_by_id = {}
        if report:
            for stage in report.get("stages", []):
                stage_by_id[stage.get("stage", "")] = stage

        # Stage 1 — perception
        try:
            if report and report.get("gait_style"):
                fp_data = report["gait_style"]
                parts: list[str] = []
                cadence = fp_data.get("cadence_steps_per_min")
                stride = fp_data.get("stride_length_m")
                hip = fp_data.get("hip_sway_m")
                if cadence is not None:
                    parts.append(f"cadence {cadence:.0f} {UNIT_CADENCE}")
                if stride is not None:
                    parts.append(f"stride {stride * 100:.0f} cm")
                if hip is not None:
                    parts.append(f"hip sway {hip * 100:.1f} cm")
                detail = " · ".join(parts) if parts else fp_data.get("style_summary", "—")
                status = stage_by_id.get("1_perception", {}).get("status", "complete")
                color = WARNING if status == "partial" else TEXT
                if s1 is not None:
                    s1.configure(text=f"Gait style: {detail}", fg=color)
            else:
                from stablewalk.real_to_sim.gait_style_extraction import (
                    extract_gait_style_fingerprint,
                )

                fp = extract_gait_style_fingerprint(
                    self.gait_motion,
                    self._gait_cycle,
                    gait_features=self._gait_features,
                )
                parts = []
                if fp.cadence_steps_per_min is not None:
                    parts.append(f"cadence {fp.cadence_steps_per_min:.0f} {UNIT_CADENCE}")
                if fp.stride_length_m is not None:
                    parts.append(f"stride {fp.stride_length_m * 100:.0f} cm")
                if fp.hip_sway_m is not None:
                    parts.append(f"hip sway {fp.hip_sway_m * 100:.1f} cm")
                detail = " · ".join(parts) if parts else fp.style_summary
                if s1 is not None:
                    s1.configure(text=f"Gait style: {detail}", fg=TEXT)
        except Exception:
            if s1 is not None:
                s1.configure(text="Gait style: —", fg=THEME_MUTED)

        # Stage 2 — retargeting
        if s2 is not None:
            s2_stage = stage_by_id.get("2_retargeting")
            if s2_stage:
                detail2 = s2_stage.get("detail", "")
                status2 = s2_stage.get("status", "complete")
                s2.configure(
                    text=f"Retargeting: {detail2[:72]}",
                    fg=SUCCESS if status2 == "complete" else WARNING,
                )
            else:
                s2.configure(
                    text="Human → Unitree G1: ready (uniform scale from leg length)",
                    fg=INFO,
                )

        run_name = Path(self._resolve_session_video_source() or "session").stem or "session"
        amp_path = config.MOTION_REFERENCE_EXPORT_DIR / run_name / "amp_reference_motion.npz"
        if s3 is not None:
            s3_stage = stage_by_id.get("3_simulation_amp")
            if amp_path.is_file() or (s3_stage and s3_stage.get("status") in ("complete", "partial")):
                status3 = (s3_stage or {}).get("status", "complete")
                s3.configure(
                    text=f"AMP reference: exported ({amp_path.name})",
                    fg=SUCCESS if status3 == "complete" else WARNING,
                )
            else:
                s3.configure(
                    text="AMP reference: click Export AMP Reference or Real-to-Sim Pipeline",
                    fg=WARNING,
                )

        vgrf = self._virtual_grf
        contact_sync = (report or {}).get("contact_sync")
        if s4 is not None:
            if contact_sync:
                mean_r = contact_sync.get("mean_reward", 0.0)
                interp = contact_sync.get("interpretation", "")
                short = interp[:60] + ("…" if len(interp) > 60 else "")
                s4.configure(
                    text=f"Contact sync: {mean_r:.0%} mean — {short}",
                    fg=SUCCESS if mean_r >= 0.65 else (WARNING if mean_r >= 0.35 else THEME_MUTED),
                )
            elif vgrf is not None and vgrf.available:
                s4.configure(
                    text=(
                        f"{LABEL_VIRTUAL_GRF_STATUS_PREFIX} {vgrf.estimation_method_label} "
                        f"({vgrf.confidence:.0%} confidence)"
                    ),
                    fg=TEXT,
                )
            else:
                s4.configure(
                    text=LABEL_VIRTUAL_GRF_UNAVAILABLE_MSG,
                    fg=THEME_MUTED,
                )

        if summary is not None:
            if report:
                summary.configure(
                    text=f"Pipeline report ready — {report.get('run_name', run_name)}",
                    fg=SUCCESS,
                )
            else:
                summary.configure(
                    text="Real-to-Sim stages update as analysis and export complete.",
                    fg=INFO,
                )

        from stablewalk.ui.tk.dashboard_advanced import update_engineering_dashboard

        update_engineering_dashboard(self)

    def _update_gait_metric_cards(
        self,
        result: "StabilityResult",
        summary,
        display: dict[str, str],
        validity,
    ) -> None:
        """Populate Section 3 gait cycles card from stability result."""
        usable, detected = self._resolved_gait_cycle_count()
        self._sync_gait_cycles_labels(
            usable=usable,
            detected=detected,
            completeness_pct=result.completeness_pct,
        )
        try:
            self._update_gait_cycle_panel(self._gait_cycle)
        except Exception:
            pass
        adv_btn = getattr(self, "btn_advanced_analysis", None)
        if adv_btn is not None:
            adv_btn.configure(state=tk.NORMAL)

        metric_by_key = {m.key: m for m in result.metrics}
        temporal = metric_by_key.get("temporal_symmetry")
        pelvis = metric_by_key.get("pelvis_stability")
        temporal_lbl = getattr(self, "lbl_advanced_temporal", None)
        if temporal_lbl is not None:
            if temporal is not None and temporal.score is not None:
                temporal_lbl.configure(text=f"Temporal Symmetry: {temporal.score:.0f}")
            else:
                temporal_lbl.configure(text="Temporal Symmetry: —")
        pelvis_lbl = getattr(self, "lbl_advanced_pelvis", None)
        if pelvis_lbl is not None:
            if pelvis is not None and pelvis.score is not None:
                pelvis_lbl.configure(text=f"Pelvis Stability: {pelvis.score:.0f}")
            else:
                pelvis_lbl.configure(text="Pelvis Stability: —")
        evidence_lbl = getattr(self, "lbl_advanced_evidence", None)
        if evidence_lbl is not None:
            snippets = []
            if result.domain_evidence:
                for key, text in list(result.domain_evidence.items())[:4]:
                    label = key.replace("_", " ").title()
                    snippets.append(f"• {label}: {text}")
            evidence_lbl.configure(
                text="\n".join(snippets) if snippets else "Evidence: —"
            )

        self._refresh_real_to_sim_advanced_panel()

    def _clear_overview_analysis_surfaces(self) -> None:
        """Wipe Overview HUD, path stats, KPIs, and computed context for a new video."""
        self._overview_clinical_generation = None
        self._overview_clinical_ready = False
        self._overview_analysis_timestamp = None
        self._cached_source_video_frames = None
        self._overview_hud_frame_i = None
        self._stability_result = None
        self._last_stability_result = None
        self._session_metric_view = None
        hud_labels = getattr(self, "_overview_hud_value_labels", None) or {}
        for lbl in hud_labels.values():
            try:
                lbl.configure(text="—")
            except tk.TclError:
                pass
        path_labels = getattr(self, "overview_path_metric_values", None) or {}
        for lbl in path_labels.values():
            try:
                lbl.configure(text="—")
            except tk.TclError:
                pass
        for card in (getattr(self, "_overview_clinical_cards", None) or {}).values():
            value = card.get("value") if isinstance(card, dict) else None
            if value is None:
                continue
            try:
                value.configure(text="—")
            except tk.TclError:
                pass
        for lbl in (getattr(self, "_overview_clinical_bullet_labels", None) or []):
            try:
                lbl.configure(text="")
            except tk.TclError:
                pass
        finding = getattr(self, "lbl_overview_primary_finding", None)
        if finding is not None:
            try:
                finding.configure(text="Load and analyze a video to see the clinical gait overview.")
            except tk.TclError:
                pass
        for key, lbl in (getattr(self, "_overview_session_badge_labels", None) or {}).items():
            try:
                lbl.configure(text="—" if key != "video_name" else "No video loaded")
            except tk.TclError:
                pass
        banner = getattr(self, "_overview_data_quality_banner", None)
        if banner is not None:
            try:
                banner.frame.grid_remove()
            except Exception:
                pass
        self._refresh_overview_recording_context()
        self._refresh_demo_comparison_panel()

    def _reset_stability_panel(self) -> None:
        self._biomech = None
        for attr, default in (
            ("lbl_summary_ms_value", "—"),
            ("lbl_summary_gq_value", "—"),
            ("lbl_summary_ac_level", "—"),
        ):
            self._set_label_text(getattr(self, attr, None), default, fg=MUTED)
        badge_lbl = getattr(self, "lbl_summary_gq_badge", None)
        if badge_lbl is not None:
            badge_lbl.configure(text="")
        for attr in (
            "lbl_summary_ms_explain",
            "lbl_summary_gq_explain",
            "lbl_summary_ac_explain",
            "lbl_summary_ms_interp",
            "lbl_summary_gq_interp",
            "lbl_summary_ac_interp",
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text="")
        for attr, default in (
            ("lbl_summary_ac_completeness", "Analysis completeness: —"),
            ("lbl_summary_ac_cycles", "Usable gait cycles: —"),
            ("lbl_summary_ac_comparable", "Comparable score: —"),
        ):
            self._set_label_text(getattr(self, attr, None), default, fg=MUTED)
        for attr, default in (
            ("lbl_movement_stability", "—"),
            ("lbl_gait_quality", "—"),
            ("lbl_analysis_confidence", "—"),
        ):
            self._set_label_text(getattr(self, attr, None), default, fg=MUTED)
        for attr in (
            "lbl_movement_stability_interp",
            "lbl_gait_quality_interp",
            "lbl_analysis_confidence_interp",
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text="")
        for attr in ("lbl_gait_explanation", "lbl_stab_legacy", "lbl_stab_comparable"):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text="")
        if hasattr(self, "lbl_stab_score"):
            self._set_label_text(self.lbl_stab_score, "—", fg=MUTED)
        if hasattr(self, "lbl_stab_category"):
            self.lbl_stab_category.configure(text="No walk analyzed yet", foreground=MUTED)
        elif hasattr(self, "lbl_stab_headline"):
            self.lbl_stab_headline.configure(text="No walk analyzed yet", foreground=MUTED)
        for attr in ("lbl_stab_completeness", "lbl_stab_confidence_badge", "lbl_stab_view", "lbl_stab_analysis_confidence", "lbl_stab_usable_cycles"):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text="")
        slots = getattr(self, "_walk_summary_slots", None)
        if slots:
            for _title_lbl, value_lbl, interp_lbl in slots:
                self._set_label_text(value_lbl, "—", fg=TEXT)
                if interp_lbl is not None:
                    interp_lbl.configure(text="")
        if hasattr(self, "lbl_stab_metrics"):
            self.lbl_stab_metrics.configure(text="")
        if hasattr(self, "lbl_stab_steps"):
            self.lbl_stab_steps.configure(text="")
        conf_lbl = getattr(self, "lbl_stab_step_confidence", None)
        if conf_lbl is not None:
            conf_lbl.configure(text="")
        detail_btn = getattr(self, "btn_stab_steps_details", None)
        if detail_btn is not None:
            detail_btn.pack_forget()
        if hasattr(self, "lbl_stab_reason"):
            self.lbl_stab_reason.configure(text="")
            self.lbl_stab_reason.pack_forget()
        btn = getattr(self, "btn_walk_summary_details", None)
        if btn is not None:
            btn.configure(state=tk.DISABLED)
        adv_btn = getattr(self, "btn_advanced_analysis", None)
        if adv_btn is not None:
            adv_btn.configure(state=tk.DISABLED)
        for attr, default in (
            ("lbl_advanced_temporal", "Temporal Symmetry (derived): —"),
            ("lbl_advanced_pelvis", "Pelvis Stability (derived): —"),
            ("lbl_advanced_evidence", "Evidence: —"),
        ):
            self._set_label_text(getattr(self, attr, None), default, fg=MUTED)
        for attr in ("lbl_summary_demo_headline", "lbl_summary_demo_compare", "lbl_overview_demo_compare"):
            self._set_label_text(getattr(self, attr, None), "", fg=MUTED)
        for attr, default in (
            ("lbl_gait_card_phase_value", "—"),
            ("lbl_overview_gait_cycles_usable", "Usable: —"),
            ("lbl_overview_gait_cycles_completeness", "Completeness: —"),
            ("lbl_gait_card_cycles_value", "Not computed yet"),
            (
                "lbl_gait_card_completeness_value",
                "Completeness: Available after full analysis",
            ),
            ("lbl_overview_contact_left", "—"),
            ("lbl_overview_contact_right", "—"),
            ("lbl_gait_card_contact_left", "Not computed yet"),
            ("lbl_gait_card_contact_right", "Not computed yet"),
        ):
            self._set_label_text(getattr(self, attr, None), default, fg=MUTED)
        self._reset_gait_cycle_panel()
        self._refresh_real_to_sim_advanced_panel()

    def _reset_gait_cycle_panel(self) -> None:
        self._gait_cycle = None
        self._gait_features = None
        self._gait_asymmetry_moments = None
        self._selected_asymmetry_moment_index = None
        self._foot_contact = None
        self._estimated_vgrf = None
        self._biomech_analysis = None
        self._biomech_frame_cache = None
        self._biomech_com_sequence = None
        self._biomech_com_index = None
        self._biomech_static_sig = None
        self._biomech_chart_data_key = None
        self._contact_chart_data_key = None
        self._biomech_chart_undersized = False
        self._contact_chart_undersized = False
        self._pipeline_status_report_cache = None
        self._virtual_grf = None
        self._analysis_summary_cache = None
        self._analysis_summary_session_key = None
        self._analysis_summary_playhead_key = None
        self._spectral_report = None
        self._spectral_token = int(getattr(self, "_spectral_token", 0)) + 1
        self._four_gait_report = None
        self._four_gait_token = int(getattr(self, "_four_gait_token", 0)) + 1
        self._four_gait_cache_key = None
        self._spectral_pending = None
        self._gait_timing = None
        self._asymmetry_report = None
        self._asymmetry_token = int(getattr(self, "_asymmetry_token", 0)) + 1
        self._asymmetry_pending = None
        self._asymmetry_cache_key = None
        self._reset_physics_force_panel()
        self._update_contact_gait_chart()
        self._update_biomechanics_panel()
        self._update_biomechanics_chart()
        self._update_motion_temporal_metrics_panel()
        self._update_analysis_summary_panel()
        phase_lbl = getattr(self, "lbl_gait_cycle_phase", None)
        if phase_lbl is not None:
            phase_lbl.configure(text="Phase: —", fg=MUTED)
        for attr, text in (
            ("lbl_gait_cycle_left_contact", "Left: —"),
            ("lbl_gait_cycle_right_contact", "Right: —"),
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text=text, fg=MUTED)
        slots = getattr(self, "_gait_cycle_metric_slots", None)
        if slots:
            for _title, value_lbl in slots:
                value_lbl.configure(text="—", fg=TEXT)
        conf = getattr(self, "lbl_gait_cycle_confidence", None)
        if conf is not None:
            conf.configure(text="")
        gait_interp = getattr(self, "lbl_gait_cycle_interp_compact", None)
        if gait_interp is not None:
            gait_interp.configure(text="")
        for attr, default in (
            ("lbl_gait_card_phase_value", "—"),
            ("lbl_gait_card_cycles_value", "Not computed yet"),
            (
                "lbl_gait_card_completeness_value",
                "Completeness: Available after full analysis",
            ),
            ("lbl_overview_gait_cycles_usable", "Usable: —"),
            ("lbl_overview_gait_cycles_completeness", "Completeness: —"),
            ("lbl_overview_contact_left", "—"),
            ("lbl_overview_contact_right", "—"),
            ("lbl_gait_card_contact_left", "Not computed yet"),
            ("lbl_gait_card_contact_right", "Not computed yet"),
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text=default, fg=MUTED)
        for side in ("left", "right"):
            state_lbl = getattr(self, f"lbl_foot_{side}_state", None)
            if state_lbl is not None:
                state_lbl.configure(text="Unavailable", fg=MUTED)
        self._gait_contact_frame_index = None
        self._gait_phase_frame_index = None
        self._overview_video_frame_index = None
        refresh_moments = getattr(self, "_refresh_asymmetric_moments_panel", None)
        if callable(refresh_moments):
            refresh_moments()

    @staticmethod
    def _set_label_text(
        widget: tk.Misc | None,
        text: str,
        *,
        fg: str | None = None,
    ) -> None:
        """Configure text/color on tk.Label or ttk.Label."""
        if widget is None:
            return
        if isinstance(widget, ttk.Label):
            if fg is not None:
                widget.configure(text=text, foreground=fg)
            else:
                widget.configure(text=text)
            return
        if fg is not None:
            widget.configure(text=text, fg=fg)
        else:
            widget.configure(text=text)

    @staticmethod
    def _format_dashboard_gait_phase(phase: str | None) -> str:
        from stablewalk.analysis.gait_phase_classification import (
            format_gait_phase_display,
        )

        return format_gait_phase_display(phase)

    @staticmethod
    def _format_contact_state(contact: int | bool | None) -> str:
        from stablewalk.analysis.gait_phase_classification import (
            contact_to_display_state,
        )

        return contact_to_display_state(contact)

    def _reset_physics_force_panel(self) -> None:
        from stablewalk.ui.scientific_labels import LABEL_CONTACT_NOT_FORCE, LABEL_VIRTUAL_GRF_PANEL

        status = getattr(self, "lbl_physics_force_status", None)
        method = getattr(self, "lbl_physics_force_method", None)
        note = getattr(self, "lbl_physics_force_note", None)
        if status is not None:
            status.configure(text="Status: Not configured", fg=MUTED)
        if method is not None:
            method.configure(text=f"Method: {LABEL_VIRTUAL_GRF_PANEL} — none", fg=MUTED)
        if note is not None:
            note.configure(text=LABEL_CONTACT_NOT_FORCE, fg=MUTED)

    def _refresh_physics_force_panel(self) -> None:
        """Update Estimated Virtual GRF sidebar — distinct from contact mask."""
        from stablewalk.ui.scientific_labels import (
            DISCLAIMER_VGRF,
            LABEL_CONTACT_NOT_FORCE,
            LABEL_VIRTUAL_GRF_PANEL,
        )
        from stablewalk.ui.theme import MUTED as THEME_MUTED, TEXT

        status = getattr(self, "lbl_physics_force_status", None)
        method = getattr(self, "lbl_physics_force_method", None)
        note = getattr(self, "lbl_physics_force_note", None)
        result = self._estimated_vgrf
        legacy = self._virtual_grf

        if status is None or method is None:
            return

        if result is not None and result.available:
            status.configure(
                text=f"Status: Available ({result.metrics.confidence:.0%} confidence)",
                fg=TEXT,
            )
            method.configure(
                text=f"Method: {LABEL_VIRTUAL_GRF_PANEL}",
                fg=TEXT,
            )
            if note is not None:
                m = result.metrics
                note.configure(
                    text=(
                        f"Peak L {m.left_peak_force_bw:.2f} BW · R {m.right_peak_force_bw:.2f} BW · "
                        f"Total {m.peak_force_bw:.2f} BW. "
                        f"Loading rate total {m.loading_rate_n_per_s:.0f} N/s. "
                        f"{DISCLAIMER_VGRF}"
                    ),
                    fg=THEME_MUTED,
                )
        elif legacy is not None and legacy.available:
            status.configure(
                text=f"Status: Available ({legacy.confidence:.0%} confidence)",
                fg=TEXT,
            )
            method.configure(
                text=f"Method: {legacy.estimation_method_label}",
                fg=TEXT,
            )
        else:
            status.configure(text="Status: Not configured", fg=THEME_MUTED)
            method.configure(text="Method: None", fg=THEME_MUTED)
            if note is not None:
                note.configure(
                    text=LABEL_CONTACT_NOT_FORCE,
                    fg=THEME_MUTED,
                )
        style_lbl = getattr(self, "lbl_real_to_sim_style", None)
        if style_lbl is not None and self.gait_motion is not None and self._gait_cycle:
            try:
                from stablewalk.real_to_sim.gait_style_extraction import (
                    extract_gait_style_fingerprint,
                )

                fp = extract_gait_style_fingerprint(
                    self.gait_motion,
                    self._gait_cycle,
                    gait_features=self._gait_features,
                )
                style_lbl.configure(text=fp.style_summary, fg=TEXT)
            except Exception:
                style_lbl.configure(text="—", fg=THEME_MUTED)
        self._refresh_real_to_sim_advanced_panel()

    def _assert_overview_contact_clearance_sync_debug(self, snapshot) -> None:
        """Debug-mode frame-index and contact/clearance consistency checks."""
        import os

        if os.environ.get("STABLEWALK_GUI_DEBUG", "").strip().lower() not in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return
        if snapshot is None:
            return
        frame = snapshot.frame_index
        contact_frame = getattr(self, "_gait_contact_frame_index", None)
        clearance_frame = getattr(self, "_foot_clearance_frame_index", None)
        phase_frame = getattr(self, "_gait_phase_frame_index", None)
        video_frame = getattr(self, "_overview_video_frame_index", None)
        checks = [
            ("display_frame_index", frame),
            ("contact_frame_index", contact_frame),
            ("clearance_frame_index", clearance_frame),
            ("phase_frame_index", phase_frame),
            ("video_frame_index", video_frame),
        ]
        for name, value in checks:
            print(f"[Overview sync] {name}={value}", flush=True)
        indices = [v for v in (frame, contact_frame, clearance_frame, phase_frame) if v is not None]
        if len(set(indices)) > 1:
            print("[Overview sync] WARNING: frame indices mismatch", flush=True)

    def _foot_contact_for_frame(
        self, frame_index: int
    ) -> tuple[int, int] | None:
        if self._foot_contact is not None:
            state = self._foot_contact.frame_at(frame_index)
            if state is not None:
                if (
                    not state.left_contact_binary
                    and not state.right_contact_binary
                    and not (
                        getattr(state, "left_contact_known", True)
                        and getattr(state, "right_contact_known", True)
                    )
                ):
                    return None
                return (state.left_contact_binary, state.right_contact_binary)
        if self._gait_cycle is None:
            return None
        state = self._gait_cycle.frame_at(frame_index)
        if state is None:
            return None
        return (state.left_contact, state.right_contact)

    def _update_contact_gait_chart(
        self, *, playhead_time_s: float | None = None, force_rebuild: bool = False
    ) -> None:
        if not force_rebuild and not self._group_visible("contact"):
            return
        canvas = getattr(self, "canvas_contact_gait", None)
        fig = getattr(self, "fig_contact_gait", None)
        if canvas is None or fig is None:
            # Welcome / reset with no session must not force-create Motion charts
            # (each create triggers a full Agg draw and dominates GUI startup).
            if (
                self._foot_contact is None
                and self._gait_cycle is None
                and self._estimated_vgrf is None
            ):
                return
            ensure = getattr(self, "_ensure_contact_gait_chart", None)
            if callable(ensure):
                try:
                    ensure()
                except Exception:
                    return
            canvas = getattr(self, "canvas_contact_gait", None)
            fig = getattr(self, "fig_contact_gait", None)
            if canvas is None or fig is None:
                return
        # Zero-size first paint (common while Overview is active) must not stick.
        try:
            widget = canvas.get_tk_widget()
            too_small = int(widget.winfo_width()) < 80 or int(widget.winfo_height()) < 80
        except Exception:
            too_small = False
        if playhead_time_s is None:
            playhead_time_s = self._current_chart_playhead_time_s()
        from stablewalk.ui.viewers.chart_playhead import playhead_from_gui

        playhead_cursor = playhead_from_gui(self, time_s=playhead_time_s)
        playhead_frame = playhead_cursor.frame_index if playhead_cursor is not None else None
        contact_frames = getattr(self._foot_contact, "per_frame", None)
        vgrf_len = 0
        if self._estimated_vgrf is not None:
            ts = getattr(self._estimated_vgrf, "timestamps", None)
            vgrf_len = len(ts) if ts is not None else 0
        panel = str(getattr(self, "_contact_chart_panel", "stacked") or "stacked")
        data_key = (
            id(self._foot_contact),
            len(contact_frames) if contact_frames is not None else 0,
            id(self._estimated_vgrf),
            vgrf_len,
            panel,
        )
        if (
            not force_rebuild
            and not too_small
            and getattr(self, "_contact_chart_data_key", None) == data_key
            and self._move_existing_chart_playheads(fig, canvas, playhead_time_s)
        ):
            return
        from stablewalk.ui.tk.playback_render_policy import (
            allow_graph_geometry_rebuild,
            mark_analysis_dirty,
        )

        if not allow_graph_geometry_rebuild(self, force=force_rebuild):
            mark_analysis_dirty(self, "contact")
            return
        from stablewalk.ui.viewers.contact_gait_viewer import draw_contact_gait_dashboard
        from stablewalk.ui.viewers.chart_playhead import cache_static_chart_background

        draw_contact_gait_dashboard(
            fig,
            self._foot_contact,
            self._estimated_vgrf,
            playhead_time_s=playhead_time_s,
            playhead_frame=playhead_frame,
            panel=panel,
        )
        from stablewalk.ui.viewers.chart_interactions import finalize_chart_interactions

        finalize_chart_interactions(fig, canvas)
        fit = getattr(self, "_fit_contact_gait_canvas", None)
        if callable(fit):
            try:
                fit()
            except Exception:
                pass
        try:
            widget = canvas.get_tk_widget()
            too_small = int(widget.winfo_width()) < 80 or int(widget.winfo_height()) < 80
        except Exception:
            pass
        self._contact_chart_undersized = bool(too_small)
        if too_small:
            self._contact_chart_data_key = None
        else:
            self._contact_chart_data_key = data_key
        self._paint_canvas_now(canvas)
        if getattr(self, "playing", False):
            cache_static_chart_background(fig, canvas, list(fig.axes))
        sync = getattr(self, "_sync_motion_scroll", None)
        if callable(sync):
            try:
                sync()
            except Exception:
                pass

    def _overlay_enabled(self, attr: str) -> bool:
        """True only when the user explicitly enabled that Advanced overlay."""
        var = getattr(self, attr, None)
        if var is None:
            return False
        try:
            return bool(var.get())
        except tk.TclError:
            return False

    def _on_biomech_overlay_toggle(self) -> None:
        self._update_interactive_skeleton(force_draw=True)

    def _build_biomech_frame_cache(self) -> None:
        """O(1) per-frame lookup for BoS/COM overlays during playback."""
        from stablewalk.ui.viewers.gait_skeleton_renderer import COM_TRAIL_FRAME_COUNT

        ba = self._biomech_analysis
        if ba is None:
            self._biomech_frame_cache = None
            self._biomech_com_sequence = None
            self._biomech_com_index = None
            return
        cache: dict[int, dict] = {}
        com_sequence: list[dict] = []
        if ba.center_of_mass:
            for f in ba.center_of_mass.per_frame:
                entry = cache.setdefault(f.frame_index, {})
                entry["com"] = f.position
                entry["velocity"] = f.velocity
                entry["time_s"] = f.time_s
                com_sequence.append(
                    {
                        "frame_index": f.frame_index,
                        "time_s": f.time_s,
                        "position": f.position,
                        "velocity": f.velocity,
                    }
                )
        com_sequence.sort(key=lambda e: (e["time_s"], e["frame_index"]))
        self._biomech_com_sequence = com_sequence
        self._biomech_com_index = {
            int(e["frame_index"]): i for i, e in enumerate(com_sequence)
        }
        self._biomech_com_trail_count = COM_TRAIL_FRAME_COUNT
        if ba.base_of_support:
            for f in ba.base_of_support.per_frame:
                entry = cache.setdefault(f.frame_index, {})
                entry["polygon"] = f.polygon_xy
                entry["support_type"] = f.support_type
        if ba.stability_margin:
            for f in ba.stability_margin.per_frame:
                entry = cache.setdefault(f.frame_index, {})
                entry["stability_state"] = f.stability_state
                entry["com_projection"] = f.com_projection
        com_frames = ba.center_of_mass.per_frame if ba.center_of_mass else []
        for i, f in enumerate(com_frames):
            if i <= 0:
                continue
            prev = com_frames[i - 1]
            p0 = prev.position
            p1 = f.position
            cache.setdefault(f.frame_index, {})["direction"] = (p1[0] - p0[0], p1[2] - p0[2])
        self._biomech_frame_cache = cache

    def _com_trail_for_frame(self, frame_index: int) -> list[tuple[float, float, float]]:
        """Recent COM positions ending at *frame_index*, ordered by timestamp."""
        n_trail = getattr(self, "_biomech_com_trail_count", 10)
        seq = getattr(self, "_biomech_com_sequence", None)
        if seq:
            index_map = getattr(self, "_biomech_com_index", None)
            if index_map is not None:
                idx = index_map.get(int(frame_index))
            else:
                idx = next(
                    (i for i, e in enumerate(seq) if e["frame_index"] == frame_index),
                    None,
                )
            if idx is not None:
                start = max(0, idx - n_trail + 1)
                return [e["position"] for e in seq[start : idx + 1]]
        ba = self._biomech_analysis
        if ba and ba.center_of_mass:
            frames = sorted(
                ba.center_of_mass.per_frame,
                key=lambda f: (f.time_s, f.frame_index),
            )
            idx = next((i for i, f in enumerate(frames) if f.frame_index == frame_index), None)
            if idx is not None:
                start = max(0, idx - n_trail + 1)
                return [f.position for f in frames[start : idx + 1]]
        return []

    @staticmethod
    def _biomech_stability_color(classification: str) -> str:
        from stablewalk.ui.colors import MUTED, STABILITY_REDUCED, STABILITY_STABLE, STABILITY_UNSTABLE

        return {
            "Stable": STABILITY_STABLE,
            "Reduced Stability": STABILITY_REDUCED,
            "Unstable": STABILITY_UNSTABLE,
        }.get(classification, MUTED)

    def _balance_classification_supported(self) -> bool:
        sm = getattr(self._biomech_analysis, "stability_margin", None)
        if sm is None:
            return False
        try:
            from stablewalk.analysis.biomechanical.stability_margin import (
                balance_display_contract,
            )

            contract = balance_display_contract(sm)
        except Exception:
            return False
        return bool(
            contract.classification_supported
            and contract.reportable
            and contract.validity == "valid"
        )

    def _biomech_overlays_for_frame(
        self, frame_index: int
    ) -> tuple[
        tuple[float, float, float] | None,
        list[tuple[float, float]] | None,
        tuple[float, float] | None,
        tuple[int, int] | None,
        str | None,
        str | None,
        tuple[float, float] | None,
        list[tuple[float, float, float]],
        tuple[float, float, float] | None,
    ]:
        com = None
        polygon = None
        direction = None
        support_type = None
        stability_state = None
        com_projection = None
        com_velocity = None
        com_trail: list[tuple[float, float, float]] = []
        contact = self._foot_contact_for_frame(frame_index)
        cache = getattr(self, "_biomech_frame_cache", None)
        if cache is None and self._biomech_analysis is not None:
            self._build_biomech_frame_cache()
            cache = getattr(self, "_biomech_frame_cache", None)
        if cache is not None:
            entry = cache.get(frame_index, {})
            com = entry.get("com")
            com_velocity = entry.get("velocity")
            polygon = entry.get("polygon")
            support_type = entry.get("support_type")
            stability_state = entry.get("stability_state")
            com_projection = entry.get("com_projection")
            direction = entry.get("direction")
            com_trail = self._com_trail_for_frame(frame_index)
            if not self._balance_classification_supported():
                stability_state = None
            return (
                com,
                polygon,
                direction,
                contact,
                support_type,
                stability_state,
                com_projection,
                com_trail,
                com_velocity,
            )

        ba = self._biomech_analysis
        if ba is None:
            return (
                com,
                polygon,
                direction,
                contact,
                support_type,
                stability_state,
                com_projection,
                com_trail,
                com_velocity,
            )
        if ba.center_of_mass:
            for f in ba.center_of_mass.per_frame:
                if f.frame_index == frame_index:
                    com = f.position
                    com_velocity = f.velocity
                    break
            com_trail = self._com_trail_for_frame(frame_index)
        if ba.base_of_support:
            for f in ba.base_of_support.per_frame:
                if f.frame_index == frame_index:
                    polygon = f.polygon_xy
                    support_type = f.support_type
                    break
        if ba.stability_margin:
            for f in ba.stability_margin.per_frame:
                if f.frame_index == frame_index:
                    if self._balance_classification_supported():
                        stability_state = f.stability_state
                    com_projection = f.com_projection
                    break
        if ba.center_of_mass and len(ba.center_of_mass.per_frame) >= 2:
            idx = next(
                (
                    i
                    for i, f in enumerate(ba.center_of_mass.per_frame)
                    if f.frame_index == frame_index
                ),
                None,
            )
            if idx is not None and idx > 0:
                p0 = ba.center_of_mass.per_frame[idx - 1].position
                p1 = ba.center_of_mass.per_frame[idx].position
                direction = (p1[0] - p0[0], p1[2] - p0[2])
        return (
            com,
            polygon,
            direction,
            contact,
            support_type,
            stability_state,
            com_projection,
            com_trail,
            com_velocity,
        )

    def _update_biomechanics_panel(self, *, frame_index: int | None = None) -> None:
        from stablewalk.ui.theme import MUTED as THEME_MUTED, TEXT
        from stablewalk.ui.tk.dashboard_biomechanics import (
            _quality_for_key,
            bind_biomech_key_findings_from_session_view,
            update_biomech_workspace_panels,
            update_biomechanics_summary_card,
        )
        from stablewalk.ui.tk.kpi_cards import (
            parse_numeric,
            update_kpi_card,
        )

        from stablewalk.ui.value_provenance import (
            NA_DISPLAY,
            ValueProvenance,
            get_provenance_bundle,
        )

        ba = self._biomech_analysis
        # Session-static KPIs only rewrite when the analysis object OR the
        # Key Findings widgets change (the tab is lazy-built after first bind).
        kpis = getattr(self, "_biomech_kpi_cards", None) or {}
        cycles = getattr(self, "_gait_cycle", None)
        result = getattr(self, "_stability_result", None) or getattr(
            self, "_last_stability_result", None
        )
        static_sig = (
            id(ba),
            id(cycles),
            id(result),
            tuple(sorted(kpis.keys())),
        )
        refresh_static = static_sig != getattr(self, "_biomech_static_sig", None)
        playing = bool(getattr(self, "playing", False))
        rom = getattr(self, "lbl_biomech_rom", None)
        view_type = getattr(self._biomech, "view_type", None) if self._biomech else None
        valid_ratio = None
        if ba is not None and ba.stability_margin is not None:
            valid_ratio = ba.stability_margin.valid_frame_ratio
        provenance = get_provenance_bundle(self)

        def _stability_state_text() -> tuple[str, bool, str]:
            """Resolve frame/session stability label without rebuilding snapshots."""
            if ba is None or not ba.stability_margin:
                return ("N/A", False, "Stability margin analysis was not computed.")
            from stablewalk.analysis.biomechanical.stability_margin import (
                balance_display_contract,
            )

            contract = balance_display_contract(ba.stability_margin)
            if (
                contract.validity != "valid"
                or not contract.reportable
                or not contract.classification_supported
            ):
                return ("N/A", False, contract.reason)
            if frame_index is not None:
                cache = getattr(self, "_biomech_frame_cache", None)
                if cache is not None:
                    entry = cache.get(frame_index, {})
                    if "stability_state" in entry:
                        return (str(entry["stability_state"]), True, contract.reason)
                for f in ba.stability_margin.per_frame or ():
                    if f.frame_index == frame_index:
                        return (str(f.stability_state), True, contract.reason)
            pct = getattr(ba.stability_margin, "stable_pct", None)
            if pct is not None:
                from stablewalk.metrics.status import build_metric_status

                st = build_metric_status("stable_pct", value=float(pct), available=True)
                return (st.display_status.value, True, contract.reason)
            return ("N/A", False, contract.reason)

        if not refresh_static:
            # Lite path: update stability_state only when the label text changes.
            state_text, available_state, state_tip = _stability_state_text()
            prev = getattr(self, "_biomech_stability_state_sig", None)
            sig = (state_text, available_state, frame_index if not playing else None)
            if sig != prev:
                self._biomech_stability_state_sig = sig
                card = kpis.get("stability_state")
                if card is not None:
                    update_kpi_card(
                        card,
                        value=state_text if available_state else NA_DISPLAY,
                        unit="",
                        quality=_quality_for_key(
                            "stability_state",
                            state_text,
                            available=available_state,
                            view_type=view_type,
                            stability_valid_ratio=valid_ratio,
                        ),
                        available=available_state,
                        tooltip=state_tip or None,
                    )
            if not playing:
                try:
                    update_biomech_workspace_panels(self)
                except Exception:
                    pass
            return

        try:
            from stablewalk.metrics.session_view import build_session_metric_snapshots

            self._session_metric_view = build_session_metric_snapshots(
                biomechanical=ba,
                cycles=cycles,
                gait_summary=getattr(result, "gait_summary", None) if result else None,
                frame_index=frame_index,
                contact=getattr(self, "_foot_contact", None),
                spectral=getattr(self, "_spectral_report", None),
            )
        except Exception:
            self._session_metric_view = None
        self._biomech_static_sig = static_sig
        update_biomechanics_summary_card(
            self, ba, getattr(self, "_gait_cycle", None), frame_index=frame_index
        )
        session_view = getattr(self, "_session_metric_view", None)

        def _record_kpi_prov(
            key: str,
            *,
            metric_name: str,
            display: str,
            raw: float | None,
            units: str,
            source_array: str,
            source_function: str,
            coordinate_system: str,
            scope: str,
            tier: str,
            available: bool,
            tip: str | None = None,
            confidence: float | None = None,
        ) -> None:
            reason = None
            if not available or display in ("—", "-", "N/A") or str(display).startswith("N/A"):
                reason = tip or "metric not available"
            provenance.put(
                key,
                ValueProvenance(
                    metric_name=metric_name,
                    display_value=f"{display}".strip(),
                    raw_value=raw,
                    units=units,
                    source_frame=frame_index,
                    source_function=source_function,
                    source_array=source_array,
                    coordinate_system=coordinate_system,
                    filtering="session analysis gates",
                    confidence=confidence,
                    confidence_note=tip or "",
                    scope=scope,  # type: ignore[arg-type]
                    tier=tier,  # type: ignore[arg-type]
                    unavailable_reason=reason,
                ),
            )

        def _set_kpi(
            key: str,
            value: str,
            *,
            unit: str = "",
            available: bool = True,
            fraction: float | None = None,
            numeric: float | None = None,
            tip: str | None = None,
        ) -> None:
            card = kpis.get(key)
            if card is None:
                return
            if not available:
                value = NA_DISPLAY
                unit = ""
            display = f"{value}{(' ' + unit) if unit and available else ''}".strip()
            update_kpi_card(
                card,
                value=value,
                unit=unit if available else "",
                quality=_quality_for_key(
                    key,
                    display,
                    available=available,
                    view_type=view_type,
                    stability_valid_ratio=valid_ratio,
                ),
                available=available,
                fraction=fraction if available else None,
                numeric=numeric if available else None,
                tooltip=tip,
            )

        # Key Findings cards ← SessionMetricView (same computed metrics as Overview).
        bind_biomech_key_findings_from_session_view(
            self,
            session_view,
            ba=ba,
            view_type=view_type,
            stability_valid_ratio=valid_ratio,
        )

        # Provenance for the bound Key Findings (inspect / export).
        if session_view is not None:
            from stablewalk.metrics.session_view import get_snapshot
            from stablewalk.ui.tk.dashboard_biomechanics import (
                _KEY_FINDINGS_SNAPSHOT_KEYS,
            )

            for card_key, snap_key in _KEY_FINDINGS_SNAPSHOT_KEYS.items():
                snap = get_snapshot(session_view.snapshots, snap_key)
                if snap is None:
                    continue
                available = snap.value is not None or (
                    snap.display_value not in ("—", "", "N/A")
                )
                _record_kpi_prov(
                    card_key,
                    metric_name=snap.display_name,
                    display=(
                        f"{snap.display_value}"
                        + (
                            f" {snap.unit}"
                            if available and snap.unit and isinstance(snap.value, (int, float))
                            else ""
                        )
                    ).strip(),
                    raw=float(snap.value) if isinstance(snap.value, (int, float)) else None,
                    units=snap.unit or "",
                    source_array=snap.analysis_path,
                    source_function="metrics.session_view.build_session_metric_snapshots",
                    coordinate_system="session metric view",
                    scope="session" if card_key != "stability_state" else "instantaneous",
                    tier="derived",
                    available=available,
                    confidence=snap.status.confidence,
                )

        # Key findings (quality / cadence / …) already came from SessionMetricView.
        # Do not wipe them just because the detailed biomechanics object is absent.
        if ba is None:
            from stablewalk.ui.tk.dashboard_biomechanics import (
                _KEY_FINDINGS_SNAPSHOT_KEYS,
            )

            findings = set(_KEY_FINDINGS_SNAPSHOT_KEYS)
            for key in kpis:
                if key in findings:
                    continue
                _set_kpi(key, NA_DISPLAY, available=False)
            checks = getattr(self, "lbl_biomech_video_checks", None)
            if checks is not None:
                checks.configure(text="", fg=THEME_MUTED)
            if rom is not None:
                rom.configure(text="—", fg=THEME_MUTED)
            try:
                update_biomech_workspace_panels(self)
            except Exception:
                pass
            return

        video_checks = getattr(self, "lbl_biomech_video_checks", None)
        if ba.video_quality:
            score = ba.video_quality.overall_quality_score
            _set_kpi(
                "video_quality",
                f"{score:.0f}",
                unit="/100",
                available=True,
                fraction=float(score) / 100.0,
                numeric=float(score),
            )
            if video_checks is not None:
                icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}
                lines = [
                    f"{icon.get(item.status, '·')} {item.label}"
                    for item in ba.video_quality.items
                ]
                video_checks.configure(text="\n".join(lines) if lines else "—", fg=TEXT)
        else:
            _set_kpi("video_quality", NA_DISPLAY, available=False)
            if video_checks is not None:
                video_checks.configure(text="", fg=THEME_MUTED)

        if rom is not None and ba.joint_rom and ba.joint_rom.joints:
            lines: list[str] = []
            order = ("hip", "knee", "ankle")
            for joint_name in order:
                for side in ("left", "right"):
                    entry = next(
                        (
                            j
                            for j in ba.joint_rom.joints
                            if j.joint == joint_name and j.side == side
                        ),
                        None,
                    )
                    if entry is None or entry.rom_deg is None:
                        continue
                    tag = f"{side[0].upper()} {joint_name}"
                    lines.append(
                        f"{tag}: {entry.flexion_min_deg:.0f}°–{entry.flexion_max_deg:.0f}° "
                        f"(ROM {entry.rom_deg:.0f}°)"
                    )
            rom.configure(text="\n".join(lines) if lines else "—", fg=TEXT)
        elif rom is not None:
            rom.configure(text="—", fg=THEME_MUTED)

        try:
            update_biomech_workspace_panels(self)
        except Exception:
            pass

        # Clinical strip lives on Overview — skip when that tab is hidden.
        if self._group_visible("overview"):
            try:
                self._update_overview_clinical_summary()
            except Exception:
                pass

    def _update_biomechanics_chart(
        self,
        *,
        playhead_time_s: float | None = None,
        force_rebuild: bool = False,
    ) -> None:
        if not force_rebuild and not self._group_visible("biomech"):
            return
        canvas = getattr(self, "canvas_biomech", None)
        fig = getattr(self, "fig_biomech", None)
        if canvas is None or fig is None:
            # Skip lazy chart construction when clearing an empty session.
            if self._biomech_analysis is None and not force_rebuild:
                return
            ensure = getattr(self, "_ensure_biomech_chart", None)
            if callable(ensure):
                try:
                    ensure()
                except Exception:
                    return
            canvas = getattr(self, "canvas_biomech", None)
            fig = getattr(self, "fig_biomech", None)
            if canvas is None or fig is None:
                return
        try:
            widget = canvas.get_tk_widget()
            too_small = int(widget.winfo_width()) < 80 or int(widget.winfo_height()) < 80
        except Exception:
            too_small = False
        if playhead_time_s is None:
            playhead_time_s = self._current_chart_playhead_time_s()
        from stablewalk.ui.viewers.chart_playhead import playhead_from_gui

        playhead_cursor = playhead_from_gui(self, time_s=playhead_time_s)
        playhead_frame = playhead_cursor.frame_index if playhead_cursor is not None else None
        ba = self._biomech_analysis
        series = str(getattr(self, "_biomech_series_key", "com") or "com")
        data_key = (
            id(ba),
            series,
            len(getattr(getattr(ba, "center_of_mass", None), "per_frame", ()) or ()),
            len(getattr(getattr(ba, "stability_margin", None), "per_frame", ()) or ()),
            len(getattr(getattr(ba, "contact", None), "per_frame", ()) or ()),
        )
        if (
            not force_rebuild
            and not too_small
            and getattr(self, "_biomech_chart_data_key", None) == data_key
            and self._move_existing_chart_playheads(fig, canvas, playhead_time_s)
        ):
            return
        from stablewalk.ui.tk.playback_render_policy import (
            allow_graph_geometry_rebuild,
            mark_analysis_dirty,
        )

        if not allow_graph_geometry_rebuild(self, force=force_rebuild):
            mark_analysis_dirty(self, "biomech")
            return
        from stablewalk.ui.viewers.biomechanics_charts import draw_biomechanics_dashboard
        from stablewalk.ui.viewers.chart_interactions import finalize_chart_interactions
        from stablewalk.ui.viewers.chart_playhead import cache_static_chart_background

        draw_kwargs = {
            "playhead_time_s": playhead_time_s,
            "series": series,
        }
        if playhead_frame is not None:
            draw_kwargs["playhead_frame"] = playhead_frame
        draw_biomechanics_dashboard(
            fig,
            self._biomech_analysis,
            **draw_kwargs,
        )
        finalize_chart_interactions(fig, canvas)
        fit = getattr(self, "_fit_biomech_canvas", None)
        if callable(fit):
            try:
                fit()
            except Exception:
                pass
        try:
            widget = canvas.get_tk_widget()
            too_small = int(widget.winfo_width()) < 80 or int(widget.winfo_height()) < 80
        except Exception:
            pass
        self._biomech_chart_undersized = bool(too_small)
        if too_small:
            # A 1×1 first paint must not stick: later seek / tab-switch would
            # only move playheads on a blank TkAgg raster.
            self._biomech_chart_data_key = None
        else:
            self._biomech_chart_data_key = data_key
        self._paint_canvas_now(canvas)
        if getattr(self, "playing", False):
            cache_static_chart_background(fig, canvas, list(fig.axes))

    def _update_motion_temporal_metrics_panel(self) -> None:
        from stablewalk.ui.tk.dashboard_motion_metrics import update_motion_temporal_metrics_panel

        update_motion_temporal_metrics_panel(self)

    def _build_pipeline_status_context(self):
        """Collect session facts for honest pipeline stage assessment."""
        from stablewalk.monitoring.pipeline_status import PipelineStatusContext
        from stablewalk.pose.backends.smpl_validation import validate_smpl_assets

        meta = getattr(self, "_run_metadata", None)
        smpl_val = validate_smpl_assets()
        total_frames = len(self.sequence.frames) if self.sequence else 0
        detected = sum(1 for f in self.sequence.frames if f.detected) if self.sequence else 0

        provider = ""
        if meta is not None:
            provider = getattr(meta, "pose_backend_used", "") or ""
        if smpl_val.ready and smpl_val.provider_name:
            provider = smpl_val.provider_name
        rts = self._load_rts_report_for_session()
        if rts and rts.get("provider_name"):
            provider = str(rts.get("provider_name"))

        smpl_path = getattr(meta, "smpl_motion_path", None) if meta else None
        mapping = getattr(self, "_marker_mapping_comparison", None)
        isaac_available: bool | None = None
        isaac_note = ""
        if rts is not None and "isaac_lab_available" in rts:
            isaac_available = bool(rts.get("isaac_lab_available"))
            isaac_note = str(rts.get("isaac_lab_note") or "")
        else:
            try:
                from stablewalk.analysis.isaac_lab_integration import check_isaac_lab_available

                isaac_available, isaac_note = check_isaac_lab_available()
            except ImportError:
                isaac_available = False
                isaac_note = "Isaac Lab integration module unavailable"

        map_pct = self._opensim_mapping_percent(mapping)
        cov_pct = None
        if mapping is not None:
            cov_pct = getattr(mapping, "raw_coverage_percent", None)
            if cov_pct is None:
                cov_pct = getattr(mapping, "coverage_percent", None)
        ik_state = getattr(self, "_opensim_ik_state", None)
        block = self._compute_opensim_ik_block_reason()
        self._opensim_ik_block_reason = block

        return PipelineStatusContext(
            source=self._session_display_src or self._current_source or "",
            run_name=self._active_run_name or getattr(meta, "run_name", None),
            pose_backend_requested=getattr(meta, "pose_backend_requested", "mediapipe")
            if meta
            else self._selected_pose_backend(),
            pose_backend_used=getattr(meta, "pose_backend_used", "mediapipe") if meta else "mediapipe",
            pose_backend_fallback=bool(getattr(meta, "pose_backend_fallback", False)) if meta else False,
            pose_backend_provider=provider,
            detected_frames=detected,
            total_frames=total_frames,
            smpl_motion_path=smpl_path,
            smpl_assets_ready=smpl_val.ready,
            romp_importable=smpl_val.romp_importable,
            gait_motion=self.gait_motion,
            cycles=self._gait_cycle,
            contact=self._foot_contact,
            biomech=self._biomech_analysis,
            estimated_vgrf=self._estimated_vgrf,
            opensim_sdk_available=bool(getattr(self, "_opensim_sdk_available", False)),
            opensim_model_loaded=bool(getattr(self, "_opensim_model_valid", False)),
            opensim_model_name=getattr(self, "_opensim_model_name", None),
            opensim_ik_completed=ik_state == "Completed",
            opensim_ik_running=ik_state == "Running",
            opensim_ik_state=ik_state,
            opensim_ik_block_reason=block,
            opensim_mapping_status=mapping.mapping_status if mapping else None,
            opensim_mapping_percent=map_pct,
            opensim_coverage_percent=float(cov_pct) if cov_pct is not None else None,
            opensim_export_files=self._opensim_export_files_on_disk()
            if self._opensim_has_session()
            else None,
            rts_report=rts,
            isaac_lab_available=isaac_available,
            isaac_lab_note=isaac_note,
        )

    def _assess_pipeline_status_report(self, *, force: bool = False):
        from stablewalk.monitoring.pipeline_status import assess_pipeline_status

        if not force:
            cached = getattr(self, "_pipeline_status_report_cache", None)
            if cached is not None:
                return cached
        report = assess_pipeline_status(self._build_pipeline_status_context())
        self._pipeline_status_report_cache = report
        return report

    def _refresh_pipeline_status_panel(self, *, force: bool = False) -> None:
        from stablewalk.ui.tk.dashboard_pipeline_status import update_pipeline_status_panel
        from stablewalk.ui.tk.dashboard_advanced import update_engineering_dashboard
        from stablewalk.ui.tk.dashboard_notebook import TAB_ADVANCED, TAB_RESULTS_SUMMARY
        from stablewalk.ui.tk.dashboard_tab_lifecycle import active_tab_id

        # Playback must not rebuild pipeline widgets or reassess disk-heavy status.
        if getattr(self, "playing", False) and not force:
            self._pipeline_status_dirty_while_playing = True
            return
        report = self._assess_pipeline_status_report(force=force)
        tab = active_tab_id(self)
        # Results cards are built at startup. Keep them aligned even if the
        # Results tab is not currently selected.
        if tab not in (TAB_RESULTS_SUMMARY, TAB_ADVANCED):
            try:
                from stablewalk.ui.tk.dashboard_results_pipeline import (
                    update_results_analysis_pipeline,
                )

                if getattr(self, "_results_pipeline_host", None) is not None:
                    update_results_analysis_pipeline(self, report, force=force)
            except Exception:
                pass
            return
        update_pipeline_status_panel(self, report)
        if getattr(self, "playing", False):
            try:
                from stablewalk.ui.tk.dashboard_results_pipeline import (
                    update_results_analysis_pipeline,
                )

                if getattr(self, "_results_pipeline_host", None) is not None:
                    update_results_analysis_pipeline(self, report, force=True)
            except Exception:
                pass
        if tab == TAB_ADVANCED and not getattr(self, "playing", False):
            update_engineering_dashboard(self)

    def _build_analysis_summary(
        self,
        *,
        frame_index: int | None = None,
        timestamp_s: float | None = None,
    ):
        from stablewalk.analysis.analysis_summary import build_analysis_summary

        source = self._session_display_src or self._current_source or ""
        report = self._assess_pipeline_status_report(force=False)
        summary = build_analysis_summary(
            source=source,
            biomechanical=self._biomech_analysis,
            estimated_vgrf=self._estimated_vgrf,
            contact=self._foot_contact,
            cycles=self._gait_cycle,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            pipeline_status=report.to_dict(),
            spectral=getattr(self, "_spectral_report", None),
        )
        return summary

    def _update_analysis_summary_panel(
        self,
        *,
        frame_index: int | None = None,
        timestamp_s: float | None = None,
    ) -> None:
        from stablewalk.ui.tk.dashboard_summary import update_results_summary_panel

        if frame_index is None and self.sequence and self.pose_indices:
            if 0 <= self.current_pos < len(self.pose_indices):
                frame_index = int(self.current_pos)
                if timestamp_s is None:
                    source_frame_i = self.pose_indices[self.current_pos]
                    timestamp_s = self._timestamp_for_frame_index(source_frame_i)
        # Pipeline snapshot is session-level. Drop it when the analyzed objects
        # change so Results cannot keep an empty "Pending" report.
        session_key = (
            getattr(self, "_active_run_name", None),
            getattr(self, "_poses_path", None),
            id(self._gait_cycle),
            id(self._biomech_analysis),
        )
        last_session = getattr(self, "_analysis_summary_session_key", None)
        if last_session != session_key:
            self._pipeline_status_report_cache = None
        playhead_key = (frame_index, None if timestamp_s is None else round(float(timestamp_s), 3))
        last_playhead = getattr(self, "_analysis_summary_playhead_key", None)
        cached = getattr(self, "_analysis_summary_cache", None)
        cache_hit = (
            cached is not None
            and last_session == session_key
            and last_playhead == playhead_key
        )
        if cache_hit and self.playing:
            return
        if cache_hit:
            summary = cached
        else:
            summary = self._build_analysis_summary(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
            )
            self._analysis_summary_cache = summary
            self._analysis_summary_session_key = session_key
            self._analysis_summary_playhead_key = playhead_key
        update_results_summary_panel(self, summary)
        # Refresh pipeline chrome only when the underlying session changed.
        if last_session != session_key:
            self._refresh_pipeline_status_panel(force=False)

    def _ensure_metrics_stage_summary(self) -> None:
        """Run the Metrics stage when its analysis inputs exist.

        Packages already-computed biomechanics / gait-cycle results into the
        session summary. Does not invent KPI values when those inputs are missing.
        """
        if (
            getattr(self, "_biomech_analysis", None) is None
            and getattr(self, "_gait_cycle", None) is None
            and getattr(self, "_foot_contact", None) is None
        ):
            return
        self._update_analysis_summary_panel()
        self._refresh_bound_analysis_panels()

    def _export_analysis_summary_json(self) -> None:
        if not self._require_analysis_complete("Export"):
            return
        summary = getattr(self, "_analysis_summary_cache", None)
        if summary is None or not summary.source:
            messagebox.showinfo("Export", "Load and analyze a video before exporting the summary.")
            return
        from stablewalk.io.analysis_summary_export import export_analysis_summary_json

        stem = Path(summary.source).stem if summary.source else "analysis_summary"
        path = filedialog.asksaveasfilename(
            title="Export Analysis Summary (JSON)",
            defaultextension=".json",
            initialfile=f"{stem}_analysis_summary.json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            export_analysis_summary_json(summary, Path(path))
            messagebox.showinfo("Export", f"Summary exported to:\n{path}")
        except OSError as exc:
            messagebox.showerror("Export error", str(exc))

    def _export_analysis_summary_report(self) -> None:
        if not self._require_analysis_complete("Export"):
            return
        summary = getattr(self, "_analysis_summary_cache", None)
        if summary is None or not summary.source:
            messagebox.showinfo("Export", "Load and analyze a video before exporting the summary.")
            return
        from stablewalk.io.analysis_summary_export import export_analysis_summary_report

        stem = Path(summary.source).stem if summary.source else "analysis_summary"
        path = filedialog.asksaveasfilename(
            title="Export Analysis Summary (Report)",
            defaultextension=".txt",
            initialfile=f"{stem}_analysis_summary.txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            export_analysis_summary_report(summary, Path(path))
            messagebox.showinfo("Export", f"Report exported to:\n{path}")
        except OSError as exc:
            messagebox.showerror("Export error", str(exc))

    def _foot_skeleton_labels_for_frame(self, frame_index: int):
        """Skeleton foot labels from the same dashboard model as Overview foot cards."""
        from stablewalk.ui.foot_clearance_display import foot_skeleton_labels_from_dashboard

        panel = getattr(self, "_foot_clearance_dashboard", None)
        contact = self._foot_contact_for_frame(frame_index)
        left_c = bool(contact[0]) if contact is not None else False
        right_c = bool(contact[1]) if contact is not None else False
        return foot_skeleton_labels_from_dashboard(
            panel,
            left_contact=left_c,
            right_contact=right_c,
        )

    def _bind_gait_contact_placeholders(self, *, computed: bool) -> None:
        """Fill Advanced/Overview contact cards; never leave them blank."""
        text = "Unavailable" if computed else "Not computed yet"
        fg = MUTED
        for attr in (
            "lbl_gait_card_contact_left",
            "lbl_gait_card_contact_right",
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                try:
                    lbl.configure(text=text, fg=fg)
                except tk.TclError:
                    pass
        overview_text = "Unavailable" if computed else "—"
        for attr in ("lbl_overview_contact_left", "lbl_overview_contact_right"):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                try:
                    lbl.configure(text=overview_text, fg=fg)
                except tk.TclError:
                    pass
        foot_text = "Unavailable"
        for side in ("left", "right"):
            state_lbl = getattr(self, f"lbl_foot_{side}_state", None)
            if state_lbl is not None:
                try:
                    heading = getattr(state_lbl, "_fc_heading", "")
                    heading = heading.strip() if isinstance(heading, str) else ""
                    state_lbl.configure(
                        text=f"{heading}: {foot_text}" if heading else foot_text,
                        fg=fg,
                    )
                except tk.TclError:
                    pass

    def _update_gait_cycle_panel(
        self,
        result: GaitCycleAnalysisResult | None = None,
        *,
        frame_index: int | None = None,
    ) -> None:
        if result is None:
            result = getattr(self, "_gait_cycle", None)
        if result is None:
            computed = (
                getattr(self, "_biomech", None) is not None
                or getattr(self, "_foot_contact", None) is not None
                or getattr(self, "_gait_cycle", None) is not None
            )
            self._bind_gait_contact_placeholders(computed=computed)
            if computed:
                usable, detected = self._resolved_gait_cycle_count()
                completeness = (
                    self._biomech.completeness_pct if self._biomech else None
                )
                self._sync_gait_cycles_labels(
                    usable=usable, detected=detected, completeness_pct=completeness
                )
            return
        if not result.per_frame:
            self._bind_gait_contact_placeholders(computed=True)
            usable, detected = self._resolved_gait_cycle_count()
            completeness = self._biomech.completeness_pct if self._biomech else None
            self._sync_gait_cycles_labels(
                usable=usable, detected=detected, completeness_pct=completeness
            )
            return

        idx = frame_index
        if idx is None and self.skeleton_player is not None:
            idx = self.skeleton_player.state.frame_index
        state = result.frame_at(idx) if idx is not None else None
        if state is None and result.per_frame:
            state = result.per_frame[-1]

        m = result.metrics
        snap = None
        cache = getattr(self, "_session_analysis", None)
        if cache is not None and cache.ready and state is not None:
            snap = cache.frame_snapshot(state.frame_index)
        if snap is not None:
            derived_phase = snap.phase_raw
            dashboard_phase = snap.phase_dashboard
        elif state is not None:
            from stablewalk.analysis.gait_phase_classification import (
                classify_gait_phase_from_contacts,
                log_phase_consistency_warning,
            )

            derived_phase = classify_gait_phase_from_contacts(
                state.left_contact,
                state.right_contact,
                contact_confidence=m.contact_confidence,
                confidence_tier=m.confidence_tier,
                left_foot_clearance_m=state.left.foot_clearance_m,
                right_foot_clearance_m=state.right.foot_clearance_m,
            )
            dashboard_phase = self._format_dashboard_gait_phase(derived_phase)
            if not self.playing:
                log_phase_consistency_warning(
                    frame_index=state.frame_index,
                    left_contact=state.left_contact,
                    right_contact=state.right_contact,
                    contact_confidence=m.contact_confidence,
                    phase=derived_phase,
                    displayed_phase=dashboard_phase,
                )
        else:
            derived_phase = None
            dashboard_phase = "—"

        phase_lbl = getattr(self, "lbl_gait_cycle_phase", None)
        if phase_lbl is not None:
            if derived_phase is not None:
                phase_text = derived_phase.replace("_", " ").title()
                phase_lbl.configure(text=f"Phase: {phase_text}", fg=TEXT)
            else:
                phase_lbl.configure(text="Phase: —", fg=MUTED)

        for attr in ("lbl_gait_card_phase_value",):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(
                    text=dashboard_phase,
                    fg=TEXT if derived_phase is not None else MUTED,
                )

        left_lbl = getattr(self, "lbl_gait_cycle_left_contact", None)
        right_lbl = getattr(self, "lbl_gait_cycle_right_contact", None)
        if state is not None:
            from stablewalk.ui.theme import ACCENT, MUTED as THEME_MUTED

            from stablewalk.analysis.foot_contact_analysis import display_contact_value

            fc = getattr(self, "_foot_contact", None)
            fc_state = fc.frame_at(state.frame_index) if fc is not None else None
            left_display = display_contact_value(
                fc_state.left_contact_binary if fc_state is not None else state.left_contact,
                state.left,
                known=getattr(fc_state, "left_contact_known", None) if fc_state else None,
            )
            right_display = display_contact_value(
                fc_state.right_contact_binary if fc_state is not None else state.right_contact,
                state.right,
                known=getattr(fc_state, "right_contact_known", None) if fc_state else None,
            )

            def _contact_line(side: str, contact: int | None) -> tuple[str, str]:
                if contact is None:
                    return f"{side}: unknown", THEME_MUTED
                if contact:
                    return f"{side}: YES", ACCENT
                return f"{side}: NO", THEME_MUTED

            if left_lbl is not None and right_lbl is not None:
                l_text, l_fg = _contact_line("Left contact", left_display)
                r_text, r_fg = _contact_line("Right contact", right_display)
                left_lbl.configure(text=l_text, fg=l_fg)
                right_lbl.configure(text=r_text, fg=r_fg)

            left_contact = getattr(self, "lbl_gait_card_contact_left", None)
            right_contact = getattr(self, "lbl_gait_card_contact_right", None)
            overview_left = getattr(self, "lbl_overview_contact_left", None)
            overview_right = getattr(self, "lbl_overview_contact_right", None)
            left_text = self._format_contact_state(left_display)
            right_text = self._format_contact_state(right_display)
            for lbl, text in (
                (left_contact, left_text),
                (right_contact, right_text),
                (overview_left, left_text),
                (overview_right, right_text),
            ):
                if lbl is not None:
                    lbl.configure(text=text, fg=TEXT)

            # Foot clearance state on Overview uses the same contact mask.
            from stablewalk.ui.foot_clearance_display import overview_state_text

            for side, contact_val in (
                ("left", left_display),
                ("right", right_display),
            ):
                label = overview_state_text(self._format_contact_state(contact_val))
                state_fg = INFO if label == "Swing" else TEXT
                state_lbl = getattr(self, f"lbl_foot_{side}_state", None)
                if state_lbl is not None:
                    heading = getattr(state_lbl, "_fc_heading", "")
                    heading = heading.strip() if isinstance(heading, str) else ""
                    state_lbl.configure(
                        text=f"{heading}: {label}" if heading else label,
                        fg=state_fg,
                    )
            self._gait_contact_frame_index = state.frame_index
            self._gait_phase_frame_index = state.frame_index

        tier = m.confidence_tier
        values = [
            f"{m.cadence_steps_per_min:.0f} steps/min"
            if m.cadence_steps_per_min is not None
            else "—",
            f"{m.left_right_stance_symmetry:.0%}"
            if m.left_right_stance_symmetry is not None
            else "—",
            f"{m.left_right_swing_symmetry:.0%}"
            if m.left_right_swing_symmetry is not None
            else "—",
            f"{m.double_support_pct:.0f}%"
            if m.double_support_pct is not None
            else "—",
        ]
        slots = getattr(self, "_gait_cycle_metric_slots", None)
        if slots:
            for (_title, value_lbl), val in zip(slots, values):
                if tier == "LOW_CONFIDENCE" and val != "—":
                    value_lbl.configure(text=f"{val} PROVISIONAL", fg=MUTED)
                else:
                    value_lbl.configure(text=val, fg=TEXT)

        conf = getattr(self, "lbl_gait_cycle_confidence", None)
        if conf is not None:
            if tier == "LOW_CONFIDENCE":
                conf.configure(
                    text=f"{LABEL_CONTACT_CONFIDENCE}: {m.contact_confidence:.0%} (LOW)",
                    fg=MUTED,
                )
            else:
                conf.configure(
                    text=f"{LABEL_CONTACT_CONFIDENCE}: {m.contact_confidence:.0%}",
                    fg=MUTED,
                )

        from stablewalk.ui.dashboard_interpretability import (
            format_compact_interpretation,
            interpret_gait_phase,
        )

        interp_lbl = getattr(self, "lbl_gait_cycle_interp_compact", None)
        if interp_lbl is not None:
            card = interpret_gait_phase(state, result)
            interp_lbl.configure(text=format_compact_interpretation(card))

        usable, detected = self._resolved_gait_cycle_count()
        completeness = self._biomech.completeness_pct if self._biomech else None
        self._sync_gait_cycles_labels(
            usable=usable, detected=detected, completeness_pct=completeness
        )
        self._refresh_real_to_sim_advanced_panel()

    def _open_opensim_details_dialog(self) -> None:
        """Show full OpenSim technical status in a dialog."""
        frame = getattr(self, "_opensim_details_frame", None)
        if frame is None:
            return
        existing = getattr(self, "_opensim_details_dialog", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    existing.focus_force()
                    return
            except tk.TclError:
                pass

        dlg = tk.Toplevel(self.root)
        dlg.title("OpenSim Status")
        dlg.geometry("520x420")
        dlg.minsize(400, 280)
        dlg.transient(self.root)
        self._opensim_details_dialog = dlg

        container = ttk.Frame(dlg, padding=8)
        container.pack(fill=tk.BOTH, expand=True)
        frame.pack(in_=container, fill=tk.BOTH, expand=True)

        footer = ttk.Frame(container)
        footer.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(
            footer,
            text="View diagnostic log",
            command=self._show_opensim_log,
        ).pack(side=tk.LEFT)
        ttk.Button(footer, text="Close", command=lambda: _close()).pack(side=tk.RIGHT)

        def _close() -> None:
            frame.pack_forget()
            self._opensim_details_dialog = None
            self._opensim_details_visible = False
            btn = getattr(self, "btn_opensim_toggle_details", None)
            if btn is not None:
                btn.configure(text="Details")
            try:
                dlg.destroy()
            except tk.TclError:
                pass

        dlg.protocol("WM_DELETE_WINDOW", _close)
        self._opensim_details_visible = True
        btn = getattr(self, "btn_opensim_toggle_details", None)
        if btn is not None:
            btn.configure(text="Details")

    def _toggle_opensim_details(self) -> None:
        """Legacy alias — open OpenSim details dialog."""
        self._open_opensim_details_dialog()

    def _apply_opensim_compact_summary(
        self,
        *,
        sdk: bool,
        model_valid: bool,
        export_complete: bool,
        has_session: bool,
        presentation: bool,
    ) -> None:
        """Refresh the compact OpenSim status line and dot indicator."""
        from stablewalk.ui.tk.sidebar_display import (
            compact_export_summary_line,
            compact_opensim_ready_line,
        )

        from stablewalk.opensim_ik_status import (
            IK_COMPLETE,
            IK_FAILED,
            IK_PARTIAL,
            IK_RUNNING,
            PIPELINE_COMPLETE,
            resolve_opensim_status_pair_from_gui,
        )

        pair = resolve_opensim_status_pair_from_gui(self)
        if presentation:
            status_text = "Example session"
            dot_color = INFO
        elif pair.pipeline.key == PIPELINE_COMPLETE and pair.ik.key == IK_COMPLETE:
            status_text = pair.short_label
            dot_color = SUCCESS
        elif pair.pipeline.key == PIPELINE_COMPLETE:
            status_text = pair.short_label
            if pair.ik.key == IK_FAILED:
                dot_color = DANGER
            elif pair.ik.key == IK_RUNNING:
                dot_color = INFO
            else:
                dot_color = WARNING
        elif pair.ik.key == IK_RUNNING:
            status_text = pair.short_label
            dot_color = INFO
        elif pair.ik.key == IK_FAILED:
            status_text = pair.short_label
            dot_color = DANGER
        elif sdk or export_complete or has_session or pair.ik.key == IK_PARTIAL:
            status_text = pair.short_label
            dot_color = WARNING
        else:
            from stablewalk.ui.unavailable_labels import opensim_unavailable_label

            status_text = opensim_unavailable_label(sdk=sdk, has_session=has_session)
            dot_color = DANGER

        ready_line = compact_opensim_ready_line(sdk=sdk, presentation=presentation)
        export_line = compact_export_summary_line(
            export_complete=export_complete,
            has_session=has_session,
            presentation=presentation,
        )
        tip = f"{pair.tooltip}\n{ready_line}\n{export_line.replace('Export: ', 'Export ')}"

        ready_lbl = getattr(self, "lbl_opensim_compact_ready", None)
        if ready_lbl is not None:
            ready_lbl.configure(text=f"OpenSim · {status_text}")
            create_tooltip(ready_lbl, tip)

        dot = getattr(self, "lbl_opensim_status_dot", None)
        if dot is not None:
            dot.configure(fg=dot_color)

        for attr in ("lbl_opensim_compact_mode", "lbl_opensim_compact_model", "lbl_opensim_compact_export"):
            widget = getattr(self, attr, None)
            if widget is not None and widget.winfo_ismapped():
                widget.pack_forget()
        try:
            from stablewalk.ui.tk.dashboard_status_bar import (
                update_dashboard_status_bar,
            )

            update_dashboard_status_bar(self)
        except Exception:
            pass

    def _format_gait_summary_details(self, result: "StabilityResult") -> str:
        """Structured gait summary for the Details dialog (no long prose blocks)."""
        from stablewalk.analysis.gait_analysis_summary import build_gait_analysis_summary
        from stablewalk.analysis.stability_validity import (
            assess_stability_result_validity,
        )

        summary = result.gait_summary or build_gait_analysis_summary(result)
        validity = result.validity or assess_stability_result_validity(result)
        lines = [
            f"{'Movement Stability':<22}"
            f"{summary.movement_stability.score:.0f}"
            if summary.movement_stability.score is not None
            else f"{'Movement Stability':<22}—",
            f"{LABEL_GAIT_QUALITY:<22}"
            f"{summary.gait_quality.score:.0f}"
            if summary.gait_quality.score is not None
            else f"{LABEL_GAIT_QUALITY:<22}—",
            f"{'Analysis Confidence':<22}{summary.analysis_confidence.level}",
            "",
            f"{'Explanation':<22}{summary.explanation}",
            "",
            f"{'Legacy composite':<22}{result.score:.0f} / 100",
            f"{'Validity status':<22}{validity.status.replace('_', ' ')}",
            f"{'Comparable score':<22}{validity.comparable_score}",
            f"{'Category':<22}{result.classification}",
            f"{'Completeness':<22}{result.completeness_pct:.0f}%",
            f"{'Usable gait cycles':<22}{result.usable_gait_cycles}",
            f"{'Repeatability tier':<22}{result.repeatability_tier}",
        ]
        if result.video_duration_s > 0:
            lines.append(f"{'Video duration':<22}{result.video_duration_s:.2f} s")
        if result.view_display_name:
            lines.append(f"{'View':<22}{result.view_display_name}")
            lines.append(
                f"{'View confidence':<22}{result.view_confidence:.0%}"
            )
        demo = getattr(self, "_active_demo_gait", None)
        if demo is not None:
            usable, detected = self._resolved_gait_cycle_count()
            if usable is None:
                usable = result.usable_gait_cycles
            interp = demo_gait_interpretation(
                demo.key,
                usable_cycles=usable or 0,
                detected_cycles=detected or 0,
                completeness_pct=result.completeness_pct,
                movement_stability_score=summary.movement_stability.score,
                gait_quality_score=summary.gait_quality.score,
            )
            if interp is not None:
                lines.extend(
                    [
                        "",
                        f"Demo category ({demo.display_name}):",
                        f"  {interp.category_headline}",
                        f"  Movement: {interp.movement_stability}",
                        f"  Gait quality: {interp.gait_quality}",
                        f"  Confidence: {interp.analysis_confidence}",
                        f"  Compare: {interp.teacher_compare}",
                    ]
                )
        lines.extend(["", "Domains:"])
        for m in result.metrics:
            if m.availability == "UNAVAILABLE" or m.score is None:
                lines.append(f"{m.name:<22}— ({m.availability})")
            else:
                view_rel = m.values.get("view_reliability")
                rel_s = f", view rel {view_rel:.0%}" if view_rel is not None else ""
                lines.append(
                    f"{m.name:<22}{m.score:.0f}  "
                    f"[{m.availability}, conf {m.confidence:.0%}{rel_s}]"
                )
        if result.view_reliability_table:
            lines.extend(["", f"{'Metric':<22}{'View reliability':>16}"])
            for name, tier in result.view_reliability_table:
                lines.append(f"{name:<22}{tier:>16}")
        if result.analysis_evidence:
            lines.extend(["", "Analysis Evidence:"])
            video = result.analysis_evidence.get("video", {})
            cycles = result.analysis_evidence.get("cycles", {})
            if video:
                lines.append(
                    f"  Video: {video.get('duration_s', 0):.2f}s, "
                    f"{video.get('fps', 0):.1f} FPS, "
                    f"{video.get('total_frames', 0)} frames "
                    f"({video.get('valid_pose_frames', 0)} valid pose)"
                )
            if cycles:
                lines.append(
                    f"  Heel strikes: {cycles.get('left_heel_strikes', 0)} left / "
                    f"{cycles.get('right_heel_strikes', 0)} right"
                )
                lines.append(
                    f"  Steps: {cycles.get('left_steps', 0)} left / "
                    f"{cycles.get('right_steps', 0)} right"
                )
                lines.append(
                    f"  Cycles: {cycles.get('complete_gait_cycles', 0)} complete, "
                    f"{cycles.get('partial_gait_cycles', 0)} partial, "
                    f"{cycles.get('usable_gait_cycles', 0)} usable"
                )
            for domain_key, summary in result.domain_evidence.items():
                domain_title = domain_key.replace("_", " ").title()
                lines.append(f"  {domain_title}: {summary}")
        lines.extend(["", result.contribution_table_text()])
        if result.data_limitations:
            lines.extend(["", "Data limitations:"])
            lines.extend(f"  • {note}" for note in result.data_limitations)
        if result.primary_issue:
            lines.extend(["", f"Note: {result.primary_issue}"])
        if result.explanation:
            lines.extend(["", "Explanation:", result.explanation.strip()])
        return "\n".join(lines)

    def _show_gait_summary_details(self) -> None:
        """Open structured gait summary details (advanced metrics and notes)."""
        if self._biomech is None:
            messagebox.showinfo(
                "Gait Summary",
                "No analyzed walking session yet.\n\n"
                "Load a video and run Analyze to see the gait summary.",
            )
            return
        existing = getattr(self, "_gait_summary_dialog", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    existing.focus_force()
                    return
            except tk.TclError:
                pass

        dlg = tk.Toplevel(self.root)
        dlg.title("Gait Summary — Details")
        dlg.geometry("520x480")
        dlg.minsize(400, 320)
        dlg.transient(self.root)
        self._gait_summary_dialog = dlg

        frame = ttk.Frame(dlg, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)
        text = tk.Text(
            frame,
            wrap=tk.WORD,
            bg=PANEL,
            fg=TEXT,
            font=FONT_MONO_SM,
            relief=tk.FLAT,
            padx=8,
            pady=8,
        )
        text.pack(fill=tk.BOTH, expand=True)
        text.insert("1.0", self._format_gait_summary_details(self._biomech))
        text.configure(state=tk.DISABLED)

        footer = ttk.Frame(frame)
        footer.pack(fill=tk.X, pady=(8, 0))

        def _close() -> None:
            self._gait_summary_dialog = None
            try:
                dlg.destroy()
            except tk.TclError:
                pass

        ttk.Button(footer, text="Close", command=_close).pack(side=tk.RIGHT)
        dlg.protocol("WM_DELETE_WINDOW", _close)

    def _show_stability_explanation(self) -> None:
        """Legacy alias for gait summary details."""
        self._show_gait_summary_details()

    def _format_advanced_analysis_details(self, result: "StabilityResult") -> str:
        """Advanced gait metrics for the Advanced Analysis dialog."""
        from stablewalk.analysis.stability_validity import (
            assess_stability_result_validity,
        )
        from stablewalk.ui.dashboard_interpretability import interpret_domain_metric

        validity = result.validity or assess_stability_result_validity(result)
        metric_by_key = {m.key: m for m in result.metrics}
        lines = [
            "Advanced Gait Analysis",
            "",
            f"{'Temporal Symmetry':<22}"
            + (
                f"{metric_by_key['temporal_symmetry'].score:.0f}"
                if metric_by_key.get("temporal_symmetry") and metric_by_key["temporal_symmetry"].score is not None
                else "—"
            ),
            f"{'Pelvis Stability':<22}"
            + (
                f"{metric_by_key['pelvis_stability'].score:.0f}"
                if metric_by_key.get("pelvis_stability") and metric_by_key["pelvis_stability"].score is not None
                else "—"
            )
            + "  (heuristic domain mean; not a clinical Acceptable grade)",
            f"{'Trunk Stability':<22}"
            + (
                f"{metric_by_key['trunk_stability'].score:.0f}"
                if metric_by_key.get("trunk_stability") and metric_by_key["trunk_stability"].score is not None
                else "—"
            ),
            f"{'Cycle Consistency':<22}"
            + (
                f"{metric_by_key['cycle_consistency'].score:.0f}"
                if metric_by_key.get("cycle_consistency") and metric_by_key["cycle_consistency"].score is not None
                else "—"
            ),
            f"{LABEL_CONTACT_PATTERN:<22}"
            + (
                f"{metric_by_key['contact_pattern'].score:.0f}"
                if metric_by_key.get("contact_pattern") and metric_by_key["contact_pattern"].score is not None
                else "—"
            ),
            "",
            f"{'Legacy composite':<22}{result.score:.0f} / 100",
            f"{'Comparable score':<22}{validity.comparable_score}",
            f"{'Analysis completeness':<22}{result.completeness_pct:.0f}%",
            "",
            "Domain confidence:",
        ]
        for key, title in (
            ("temporal_symmetry", "Temporal Symmetry"),
            ("pelvis_stability", "Pelvis Stability"),
            ("trunk_stability", "Trunk Stability"),
            ("cycle_consistency", "Cycle Consistency"),
            ("contact_pattern", LABEL_CONTACT_PATTERN),
        ):
            metric = metric_by_key.get(key)
            if metric is None or metric.score is None:
                lines.append(f"  {title:<20}—")
                continue
            card = interpret_domain_metric(metric, display_name=title)
            extra = ""
            if key == "pelvis_stability":
                extra = "  formula: mean of gait-frame sway / oscillation / jerk scores"
            lines.append(f"  {title:<20}{metric.score:.0f}  ({card.confidence}){extra}")
        if result.domain_evidence:
            lines.extend(["", "Raw gait evidence:"])
            for domain_key, evidence in result.domain_evidence.items():
                lines.append(f"  {domain_key.replace('_', ' ').title()}: {evidence}")
        if result.data_limitations:
            lines.extend(["", "Data limitations:"])
            lines.extend(f"  • {note}" for note in result.data_limitations)
        return "\n".join(lines)

    def _show_advanced_analysis(self) -> None:
        """Open advanced temporal/pelvis/domain metrics."""
        if self._biomech is None:
            messagebox.showinfo(
                "Advanced Analysis",
                "No analyzed walking session yet.\n\n"
                "Load a video and run Analyze first.",
            )
            return
        existing = getattr(self, "_advanced_analysis_dialog", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    existing.focus_force()
                    return
            except tk.TclError:
                pass

        dlg = tk.Toplevel(self.root)
        dlg.title("Advanced Analysis")
        dlg.geometry("540x520")
        dlg.minsize(420, 360)
        dlg.transient(self.root)
        self._advanced_analysis_dialog = dlg

        frame = ttk.Frame(dlg, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)
        text = tk.Text(
            frame,
            wrap=tk.WORD,
            bg=PANEL,
            fg=TEXT,
            font=FONT_MONO_SM,
            relief=tk.FLAT,
            padx=8,
            pady=8,
        )
        text.pack(fill=tk.BOTH, expand=True)
        text.insert("1.0", self._format_advanced_analysis_details(self._biomech))
        text.configure(state=tk.DISABLED)

        footer = ttk.Frame(frame)
        footer.pack(fill=tk.X, pady=(8, 0))

        def _close() -> None:
            self._advanced_analysis_dialog = None
            try:
                dlg.destroy()
            except tk.TclError:
                pass

        ttk.Button(footer, text="Close", command=_close).pack(side=tk.RIGHT)
        dlg.protocol("WM_DELETE_WINDOW", _close)

    def _show_knee_details(self) -> None:
        """Open detailed knee ROM diagnostics."""
        from stablewalk.ui.viewers.knee_chart_interpretation import (
            build_knee_motion_summary,
            format_interpretation_panel,
        )

        axis = getattr(self, "var_knee_chart_axis", None)
        mode = "gait_cycle_pct" if axis is not None and axis.get() == "Gait Cycle %" else "video_time"
        summary = build_knee_motion_summary(
            getattr(self, "_knee_angle_series", None),
            getattr(self, "_gait_features", None),
            chart_mode=mode,  # type: ignore[arg-type]
        )
        report = format_interpretation_panel(summary)
        if getattr(self, "_knee_angle_series", None) is not None:
            from stablewalk.ui.viewers.knee_chart_interpretation import format_diagnostic_report

            report = report + "\n\n" + format_diagnostic_report(self._knee_angle_series)

        existing = getattr(self, "_knee_details_dialog", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    existing.focus_force()
                    return
            except tk.TclError:
                pass

        dlg = tk.Toplevel(self.root)
        dlg.title("Knee Details")
        dlg.geometry("480x400")
        dlg.minsize(360, 280)
        dlg.transient(self.root)
        self._knee_details_dialog = dlg

        frame = ttk.Frame(dlg, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)
        text = tk.Text(
            frame,
            wrap=tk.WORD,
            bg=PANEL,
            fg=TEXT,
            font=FONT_MONO_SM,
            relief=tk.FLAT,
            padx=8,
            pady=8,
        )
        text.pack(fill=tk.BOTH, expand=True)
        text.insert("1.0", report)
        text.configure(state=tk.DISABLED)

        footer = ttk.Frame(frame)
        footer.pack(fill=tk.X, pady=(8, 0))

        def _close() -> None:
            self._knee_details_dialog = None
            try:
                dlg.destroy()
            except tk.TclError:
                pass

        ttk.Button(footer, text="Close", command=_close).pack(side=tk.RIGHT)
        dlg.protocol("WM_DELETE_WINDOW", _close)

    @staticmethod
    def _opensim_status_color(kind: str) -> str:
        """Sidebar color tokens: ok=green, warn=orange, fail=red."""
        return {
            "ok": SUCCESS,
            "warn": ORANGE,
            "fail": DANGER,
            "muted": MUTED,
            "info": INFO,
            "accent": ACCENT,
        }.get(kind, TEXT_SECONDARY)

    def _show_opensim_log(self) -> None:
        """Show accumulated OpenSim UI log lines and extended status notes."""
        parts: list[str] = []
        extra = getattr(self, "_opensim_detail_text", "").strip()
        if extra:
            parts.append(extra)
        ui_log = getattr(self, "_opensim_ui_log", [])
        if ui_log:
            if parts:
                parts.append("")
            parts.append("Recent log:")
            parts.extend(ui_log[-40:])
        status_md = config.PROJECT_ROOT / "OPENSIM_STATUS.md"
        if status_md.is_file():
            try:
                md_tail = status_md.read_text(encoding="utf-8")[-1200:]
                parts.append("")
                parts.append("OPENSIM_STATUS.md (tail):")
                parts.append(md_tail)
            except OSError:
                pass
        if not parts:
            messagebox.showinfo(
                "OpenSim log",
                "No OpenSim log entries yet.\n\n"
                "Export or run IK to populate the log (also printed in the terminal).",
            )
            return
        body = "\n".join(parts)
        if len(body) > 4500:
            body = body[-4490:] + "\n…"
        messagebox.showinfo("OpenSim log", body)

    def _show_opensim_details(self) -> None:
        """Alias for View OpenSim Log."""
        self._show_opensim_log()

    def _show_marker_mapping_report(self) -> None:
        """Show marker mapping report from disk or the latest comparison message."""
        from stablewalk.opensim_marker_mapping import MARKER_MAPPING_REPORT

        if MARKER_MAPPING_REPORT.is_file():
            try:
                body = MARKER_MAPPING_REPORT.read_text(encoding="utf-8")
            except OSError as exc:
                messagebox.showerror("Marker mapping report", f"Could not read report: {exc}")
                return
            if len(body) > 4000:
                body = body[:3990] + "\n…"
            messagebox.showinfo("Marker mapping report", body)
            return
        mapping = getattr(self, "_marker_mapping_comparison", None)
        if mapping and getattr(mapping, "message", ""):
            messagebox.showinfo("Marker mapping report", mapping.message)
            return
        messagebox.showinfo(
            "Marker mapping report",
            "No mapping report yet.\n\nExport OpenSim files for the current session first.",
        )

    # ── OpenSim integration (optional real SDK) ───────────────────────────
    # MediaPipe extracts pose from video; OpenSim represents motion biomechanically.
    # Export (.trc / .mot / JSON) always works. Real IK needs SDK + .osim + TRC.
    def _init_opensim_panel(self) -> None:
        """Detect SDK, auto-load a usable model when available, then render status."""
        self._opensim_last_dir: Path | None = None
        self._opensim_model_path: Path | None = None
        self._opensim_model_name: str | None = None
        self._opensim_model_valid: bool = False
        self._opensim_ik_state: str | None = None   # StableWalk IK: Running|Completed|Failed
        self._opensim_ik_message: str = ""
        self._opensim_ik_block_reason: str | None = None
        self._opensim_demo_ik_state: str | None = None  # Demo IK state
        self._opensim_demo_ik_message: str = ""
        self._marker_mapping_comparison = None
        self._opensim_detail_text = ""
        self._opensim_ui_log: list[str] = []
        self._opensim_model_choices: dict[str, Path] = {}
        self._opensim_auto_pipeline_busy = False
        try:
            from stablewalk.opensim_sdk import (
                check_opensim_sdk,
                log_opensim_startup_status,
            )

            log_opensim_startup_status(logger, model_path=None)
            status = check_opensim_sdk(refresh=True)
            self._opensim_sdk_available = bool(status.available)
            self._opensim_sdk_version = status.version
            if self._opensim_sdk_available:
                self._auto_load_configured_opensim_model()
        except Exception as exc:
            logger.warning("OpenSim panel init failed: %s", exc)
            self._opensim_sdk_available = False
            self._opensim_sdk_version = None
            self._opensim_model_valid = False
        self._refresh_opensim_model_list()
        self._refresh_opensim_status()
        self._attach_opensim_tooltips()

    def _auto_load_configured_opensim_model(self) -> bool:
        """Load the preferred usable .osim quietly when the SDK is installed."""
        if getattr(self, "_opensim_model_valid", False):
            return True
        if not getattr(self, "_opensim_sdk_available", False):
            return False
        from stablewalk.opensim_models import find_usable_opensim_model

        path = find_usable_opensim_model()
        if path is None or not path.is_file():
            logger.info("OpenSim SDK ready but no usable .osim model found to auto-load")
            return False
        info = self._load_opensim_model_path(path, quiet=True)
        if info is not None:
            logger.info("Auto-loaded OpenSim model: %s", path)
            return True
        return False

    def _resolve_opensim_ik_model_path(self) -> Path | None:
        """Prefer the loaded model; fall back to demo / discovered usable model."""
        from stablewalk.opensim_marker_mapping import DEMO_IK_MODEL
        from stablewalk.opensim_models import find_usable_opensim_model

        loaded = getattr(self, "_opensim_model_path", None)
        if (
            getattr(self, "_opensim_model_valid", False)
            and loaded is not None
            and Path(loaded).is_file()
        ):
            return Path(loaded)
        if DEMO_IK_MODEL.is_file():
            return DEMO_IK_MODEL
        return find_usable_opensim_model()

    def _opensim_mapping_percent(self, mapping=None) -> float | None:
        """Marker mapping percentage for status / pipeline display."""
        mapping = mapping if mapping is not None else getattr(
            self, "_marker_mapping_comparison", None
        )
        if mapping is None:
            return None
        ref = len(getattr(mapping, "opensim_reference_markers", None) or [])
        matched = len(getattr(mapping, "mapped_matching_markers", None) or [])
        if ref > 0:
            return round(100.0 * matched / ref, 1)
        cov = getattr(mapping, "coverage_percent", None)
        return float(cov) if cov is not None else None

    def _compute_opensim_ik_block_reason(self) -> str | None:
        """Exact reason IK cannot run now, or None when runnable / already done."""
        from stablewalk.opensim_sdk import SDK_NOT_INSTALLED_MESSAGE

        if getattr(self, "_opensim_ik_state", None) == "Completed":
            return None
        if getattr(self, "_opensim_ik_state", None) == "Running":
            return None
        if getattr(self, "_opensim_ik_state", None) == "Failed":
            msg = (getattr(self, "_opensim_ik_message", "") or "").strip()
            return f"IK failed: {msg}" if msg else "IK failed"

        if not getattr(self, "_opensim_sdk_available", False):
            return f"IK cannot run: {SDK_NOT_INSTALLED_MESSAGE}"
        if getattr(self, "_presentation_mode", False):
            return "IK cannot run: presentation demo has no real session"
        if not self._current_run_name():
            return "IK cannot run: no analyzed video session loaded"
        if not getattr(self, "_opensim_model_valid", False):
            model = self._resolve_opensim_ik_model_path()
            if model is None or not model.is_file():
                return "IK cannot run: no usable .osim model loaded or found"

        files = self._opensim_export_files_on_disk()
        if not files.get("TRC"):
            return "IK cannot run: TRC not exported — click Export OpenSim Files"
        if not files.get("MAPPED"):
            mapped = self._opensim_mapped_trc_path()
            if mapped is None or not mapped.is_file():
                return "IK cannot run: mapped TRC missing — click Export OpenSim Files"

        mapping = getattr(self, "_marker_mapping_comparison", None)
        if mapping is None:
            return "IK cannot run: marker mapping not assessed"
        status = getattr(mapping, "mapping_status", "missing")
        if status not in ("improved", "partial", "experimental"):
            return (
                f"IK cannot run: marker mapping not ready (status: {status}). "
                f"{getattr(mapping, 'message', '')}".strip()
            )
        if not getattr(mapping, "ik_experimental_ready", False):
            n = len(getattr(mapping, "mapped_matching_markers", None) or [])
            detail = (getattr(mapping, "message", "") or "").strip()
            base = (
                f"IK cannot run: not enough mapped markers for experimental IK "
                f"({n} matched)"
            )
            return f"{base}. {detail}" if detail else base
        return None
    def _on_opensim_suggested_model_combo(self, _event: object = None) -> None:
        """Refresh suggested-model row when the dropdown selection changes."""
        if not hasattr(self, "lbl_opensim_loaded_model"):
            return
        model_valid = bool(getattr(self, "_opensim_model_valid", False))
        self._apply_opensim_model_status_ui(
            self._opensim_model_status_ui(model_valid=model_valid),
            model_valid=model_valid,
        )

    def _opensim_suggested_combo_label(self) -> str | None:
        """Dropdown candidate label (not loaded until Select)."""
        if not hasattr(self, "opensim_model_var"):
            return None
        label = self.opensim_model_var.get().strip()
        return label or None

    def _opensim_suggested_model_rel_path(self) -> str | None:
        """Project-relative path for the combobox selection (suggested only)."""
        label = self._opensim_suggested_combo_label()
        if not label:
            return None
        path = getattr(self, "_opensim_model_choices", {}).get(label)
        if path is not None:
            return self._format_data_path(Path(path))
        return label

    def _opensim_model_status_ui(self, *, model_valid: bool):
        """Build OpenSim Status model rows (display text + tooltip paths)."""
        from dataclasses import dataclass

        from stablewalk.opensim_marker_mapping import DEMO_IK_MODEL
        from stablewalk.ui.tk.sidebar_display import (
            ik_pipeline_model_status,
            loaded_model_status,
            suggested_model_status,
        )

        @dataclass(frozen=True)
        class _UI:
            pipeline_text: str
            pipeline_tooltip: str
            suggested_text: str
            suggested_tooltip: str
            loaded_text: str
            loaded_tooltip: str

        pipeline_rel = self._format_data_path(DEMO_IK_MODEL)
        p_text, p_tip = ik_pipeline_model_status(
            rel_path=pipeline_rel,
            file_available=DEMO_IK_MODEL.is_file(),
        )
        s_text, s_tip = suggested_model_status(
            rel_path=self._opensim_suggested_model_rel_path()
        )
        loaded_rel: str | None = None
        model_path = getattr(self, "_opensim_model_path", None)
        if model_valid and model_path is not None:
            loaded_rel = self._format_data_path(Path(model_path))
        l_text, l_tip = loaded_model_status(rel_path=loaded_rel, loaded=model_valid)
        return _UI(
            pipeline_text=p_text,
            pipeline_tooltip=p_tip,
            suggested_text=s_text,
            suggested_tooltip=s_tip,
            loaded_text=l_text,
            loaded_tooltip=l_tip,
        )

    def _apply_opensim_model_status_ui(self, ui, *, model_valid: bool) -> None:
        """Refresh IK pipeline / suggested / loaded rows in OpenSim Status."""
        from stablewalk.ui.theme import INFO, MUTED, SUCCESS

        rows = (
            ("lbl_opensim_pipeline_model", ui.pipeline_text, ui.pipeline_tooltip, INFO),
            (
                "lbl_opensim_suggested_model",
                ui.suggested_text,
                ui.suggested_tooltip,
                MUTED,
            ),
            (
                "lbl_opensim_loaded_model",
                ui.loaded_text,
                ui.loaded_tooltip,
                SUCCESS if model_valid else MUTED,
            ),
        )
        for attr, text, tip, fg in rows:
            widget = getattr(self, attr, None)
            if widget is None:
                continue
            widget.configure(text=text, fg=fg)
            full_tip = (tip or "").strip() or text
            widget._tooltip_text = full_tip
            if not getattr(widget, "_opensim_status_tooltip_bound", False):
                create_tooltip(widget, full_tip)
                widget._opensim_status_tooltip_bound = True

    def _refresh_opensim_model_list(self) -> None:
        """Populate the local-model combobox from models/opensim/."""
        from stablewalk.opensim_models import list_opensim_model_choices

        self._opensim_model_choices = {
            label: path for label, path in list_opensim_model_choices()
        }
        labels = list(self._opensim_model_choices.keys())
        if hasattr(self, "opensim_model_combo"):
            self.opensim_model_combo.configure(values=labels)
            if labels:
                if self.opensim_model_var.get() not in labels:
                    self.opensim_model_var.set(labels[0])
            else:
                self.opensim_model_var.set("")
        if hasattr(self, "btn_opensim_local_model"):
            self.btn_opensim_local_model.configure(
                state="normal" if labels else "disabled"
            )
        if hasattr(self, "lbl_opensim_sdk"):
            self._refresh_opensim_status()
        if labels:
            logger.info(
                "Discovered %d selectable OpenSim model(s) under %s",
                len(labels),
                config.OPENSIM_MODELS_DIR,
            )
            for label in labels:
                logger.info("  models/opensim/%s", label)
        else:
            logger.info(
                "No .osim models under %s — run: python download_opensim_model.py",
                config.OPENSIM_MODELS_DIR,
            )

    def _show_opensim_model_load_error(self, message: str) -> None:
        messagebox.showerror("OpenSim model not usable for IK", message)

    def _show_opensim_model_loaded_success(self, info) -> None:
        from stablewalk.ui.tk.sidebar_display import opensim_model_load_success_message

        messagebox.showinfo(
            "OpenSim model loaded successfully",
            opensim_model_load_success_message(num_markers=info.num_markers),
        )

    def _load_opensim_model_path(self, path: Path, *, quiet: bool = False):
        """
        Validate an OpenSim model and update GUI state only on success.

        Returns :class:`~stablewalk.opensim_sdk.ModelInfo` on success, else ``None``.
        Failed validation leaves any previously loaded model unchanged.
        When ``quiet`` is True, skip modal dialogs (used for auto-load).
        """
        from stablewalk.opensim_models import _is_autoload_blocked
        from stablewalk.opensim_sdk import validate_opensim_model
        from stablewalk.ui.tk.sidebar_display import opensim_model_ik_error_message

        if _is_autoload_blocked(path):
            msg = opensim_model_ik_error_message(blocked_template=True)
            logger.error("Model load rejected (blocked template): %s", path)
            if not quiet:
                self._show_opensim_model_load_error(msg)
            return None

        logger.info("Loading OpenSim model...")
        logger.info("  Path: %s", path)
        info = validate_opensim_model(path)
        if not info.valid:
            err = opensim_model_ik_error_message(
                validation_message=info.message,
                num_markers=info.num_markers,
            )
            logger.error("Model load failed: %s", info.message)
            if not quiet:
                self._show_opensim_model_load_error(err)
            return None

        self._opensim_model_path = Path(path)
        self._opensim_model_name = info.name or Path(path).name
        self._opensim_model_valid = True
        self._opensim_ik_state = None
        logger.info("Model loaded successfully: %s", self._opensim_model_name)
        logger.info("Model loaded: %s", self._opensim_model_name)
        self._refresh_opensim_status()
        return info

    def _select_discovered_opensim_model(self) -> None:
        """Load a model discovered under models/opensim/."""
        if not getattr(self, "_opensim_sdk_available", False):
            from stablewalk.opensim_sdk import SDK_NOT_INSTALLED_MESSAGE

            messagebox.showerror("Select Local Model", SDK_NOT_INSTALLED_MESSAGE)
            return
        label = self.opensim_model_var.get().strip()
        path = self._opensim_model_choices.get(label)
        if path is None or not path.is_file():
            messagebox.showwarning(
                "Select Local Model",
                "No local model selected.\n\n"
                f"Download one with:\n  python download_opensim_model.py\n\n"
                f"Expected folder:\n  {config.OPENSIM_MODELS_DIR}",
            )
            return
        info = self._load_opensim_model_path(path)
        if info is not None:
            self._show_opensim_model_loaded_success(info)

    def _format_data_path(self, path: Path) -> str:
        """Project-relative path for GUI display (e.g. data/output/opensim/...)."""
        try:
            return str(path.relative_to(config.PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            return str(path)

    def _current_run_name(self) -> str | None:
        """Active run/session folder name (never hardcoded to walk_stream)."""
        if getattr(self, "_active_run_name", None):
            return self._active_run_name
        meta = getattr(self, "_run_metadata", None)
        if meta is not None and getattr(meta, "run_name", None):
            return str(meta.run_name)
        if self._poses_path:
            stem = Path(self._poses_path).stem
            if stem.endswith("_poses"):
                return stem[: -len("_poses")] or None
            return stem or None
        return None

    def _opensim_output_dir(self) -> Path | None:
        """``data/output/opensim/<run_name>/`` for the active session."""
        run_name = self._current_run_name()
        if not run_name:
            return None
        return config.OPENSIM_DIR / run_name

    def _log_export(self, message: str, *args: object) -> None:
        """Export diagnostics to logger (and optional in-app OpenSim log buffer)."""
        text = message % args if args else message
        logger.info("%s", text)
        log = getattr(self, "_opensim_ui_log", None)
        if log is not None:
            log.append(text)
            if len(log) > 100:
                del log[:-100]

    def _resolve_poses_json_path(self) -> Path | None:
        """Pose JSON for the current run: data/output/poses/<run_name>_poses.json."""
        run_name = self._current_run_name()
        if run_name:
            canonical = config.POSES_DIR / f"{run_name}_poses.json"
            if canonical.is_file():
                return canonical
        if self._poses_path and Path(self._poses_path).is_file():
            return Path(self._poses_path)
        return None

    def _opensim_has_session(self) -> bool:
        """True when a real analyzed walk is loaded (not the presentation demo)."""
        if getattr(self, "_presentation_mode", False):
            return False
        if self._current_run_name() is not None:
            return True
        if self.sequence and any(getattr(f, "detected", False) for f in self.sequence.frames):
            return True
        return self._resolve_poses_json_path() is not None

    def _opensim_has_pose_data(self) -> bool:
        """True when the current session has at least one detected pose frame."""
        if not self._opensim_has_session():
            return False
        if self.sequence:
            return any(getattr(f, "detected", False) for f in self.sequence.frames)
        poses_path = self._resolve_poses_json_path()
        if not poses_path:
            return False
        try:
            seq = load_pose_sequence(poses_path)
            return any(getattr(f, "detected", False) for f in seq.frames)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return False

    def _opensim_raw_trc_path(self) -> Path | None:
        """``data/output/opensim/<run_name>/<run_name>.trc`` for the active session."""
        name = self._current_run_name()
        if not name:
            return None
        path = config.OPENSIM_DIR / name / f"{name}.trc"
        return path if path.is_file() else None

    def _opensim_mapped_trc_path(self) -> Path | None:
        """``data/output/opensim/<run_name>/<run_name>_mapped_for_opensim.trc``."""
        raw = self._opensim_raw_trc_path()
        if not raw:
            return None
        from stablewalk.opensim_marker_mapping import mapped_trc_path_for

        mapped = mapped_trc_path_for(raw)
        return mapped if mapped.is_file() else None

    def _opensim_ik_mot_path(self) -> Path | None:
        """``data/output/opensim/<run_name>/<run_name>_ik.mot`` if IK has been run."""
        name = self._current_run_name()
        if not name:
            return None
        path = config.OPENSIM_DIR / name / f"{name}_ik.mot"
        return path if path.is_file() else None

    def _opensim_export_files_on_disk(self) -> dict[str, bool]:
        """Which OpenSim artifact files already exist for the current session."""
        if not self._opensim_has_session():
            return {}
        name = self._opensim_session_name()
        out_dir = config.OPENSIM_DIR / name
        return {
            "TRC": (out_dir / f"{name}.trc").is_file(),
            "MOT": (out_dir / f"{name}.mot").is_file()
            or (out_dir / f"{name}.csv").is_file(),
            "JSON": (out_dir / f"{name}_opensim.json").is_file(),
            "MAPPED": (out_dir / f"{name}_mapped_for_opensim.trc").is_file(),
            "IK_MOT": (out_dir / f"{name}_ik.mot").is_file(),
            "IK_ERRORS": any(out_dir.glob("*_ik_marker_errors.sto"))
            if out_dir.is_dir()
            else False,
        }

    def _update_opensim_validation_labels(
        self,
        *,
        _set,
        detail_lines: list[str],
        run_name: str | None,
    ) -> None:
        """Show IK residual / pose↔IK agreement only when those files exist."""
        from stablewalk.analysis.pipeline_validation import (
            QUALITY_LABEL,
            compare_pose_mot_to_ik_mot,
            find_ik_marker_errors_sto,
            find_ik_mot,
            find_pose_mot,
            opensim_session_dir,
            parse_ik_marker_errors,
        )

        session_dir = opensim_session_dir(run_name)
        sto = find_ik_marker_errors_sto(session_dir)
        ik_err = parse_ik_marker_errors(sto) if sto else None
        self._opensim_ik_error_summary = ik_err
        if ik_err is not None:
            self._opensim_ik_mean_rms_m = float(ik_err.mean_rms_m)
            self._opensim_ik_frames_solved = max(
                0, int(ik_err.n_frames) - int(ik_err.n_failed_frames)
            )
        else:
            self._opensim_ik_mean_rms_m = None
            self._opensim_ik_frames_solved = None
        if ik_err is not None:
            q = QUALITY_LABEL.get(ik_err.quality, ik_err.quality)
            _set(
                "lbl_opensim_ik_residuals",
                (
                    f"IK residuals: mean RMS {ik_err.mean_rms_m * 100:.1f} cm ({q}); "
                    f"{int(ik_err.n_frames) - int(ik_err.n_failed_frames)}/"
                    f"{int(ik_err.n_frames)} frames in band, "
                    f"{int(ik_err.n_failed_frames)} failed"
                ),
                "ok"
                if ik_err.quality == "acceptable"
                else ("warn" if ik_err.quality == "limited" else "fail"),
            )
            detail_lines.append(ik_err.summary_line())
            for w in ik_err.warnings[:2]:
                detail_lines.append(w)
        else:
            _set(
                "lbl_opensim_ik_residuals",
                "IK residuals: unavailable (no marker-error .sto)",
                "muted",
            )

        pose_mot = find_pose_mot(session_dir, run_name)
        ik_mot = find_ik_mot(session_dir, run_name)
        agree = (
            compare_pose_mot_to_ik_mot(pose_mot, ik_mot)
            if pose_mot is not None and ik_mot is not None
            else None
        )
        if agree is not None and agree.rmse_mean_deg is not None:
            q = QUALITY_LABEL.get(agree.quality, agree.quality)
            _set(
                "lbl_opensim_pose_ik",
                f"Pose vs IK knees: RMSE {agree.rmse_mean_deg:.1f} deg ({q})",
                "ok"
                if agree.quality == "acceptable"
                else ("warn" if agree.quality == "limited" else "fail"),
            )
            detail_lines.append(agree.summary_line())
            detail_lines.append(agree.note)
        elif pose_mot is not None and ik_mot is None:
            _set(
                "lbl_opensim_pose_ik",
                "Pose vs IK knees: IK MOT missing",
                "muted",
            )
        else:
            _set(
                "lbl_opensim_pose_ik",
                "Pose vs IK knees: — (need pose MOT + IK MOT)",
                "muted",
            )

    def _opensim_export_readiness(self) -> dict[str, bool]:
        """Backward-compatible alias for on-disk export file checks."""
        return self._opensim_export_files_on_disk()

    def _opensim_export_line(self, label: str, exported: bool) -> tuple[str, str]:
        """
        Human-readable export status for one artifact type.

        Returns (text, foreground_color). States: ready / exported / missing.
        """
        if not self._opensim_has_session() or not self._opensim_has_pose_data():
            return f"{label}: missing", MUTED
        if exported:
            return f"{label}: exported", SUCCESS
        return f"{label}: not exported", ORANGE

    def _opensim_overview_export_summary(self, files: dict[str, bool]) -> str:
        """One-line TRC / MOT / JSON summary for Session Overview."""
        if not self._opensim_has_session() or not self._opensim_has_pose_data():
            return "TRC: missing · MOT: missing · JSON: missing"
        parts: list[str] = []
        for key, label in (("TRC", "TRC"), ("MOT", "MOT"), ("JSON", "JSON")):
            if files.get(key):
                parts.append(f"{label}: exported")
            else:
                parts.append(f"{label}: not exported")
        return " · ".join(parts)

    def _opensim_status_hint(self, files: dict[str, bool], sdk: bool) -> str:
        """Bottom hint line in the OpenSim panel."""
        if not self._opensim_has_session():
            return _OPENSIM_NO_SESSION_MSG
        if not self._resolve_poses_json_path():
            return _OPENSIM_POSE_JSON_MISSING_MSG
        if not self._opensim_has_pose_data():
            return _OPENSIM_NO_VALID_FRAMES_MSG
        if files and all(files.values()):
            run_name = self._opensim_session_name()
            out_dir = getattr(self, "_opensim_last_dir", None) or (
                config.OPENSIM_DIR / run_name
            )
            return f"OpenSim output folder: {self._format_data_path(Path(out_dir))}"
        hint = 'Session ready — click "Export OpenSim Files".'
        if sdk and not getattr(self, "_opensim_model_valid", False):
            from stablewalk.opensim_sdk import NO_MODEL_LOADED_MESSAGE

            return NO_MODEL_LOADED_MESSAGE
        if not sdk:
            from stablewalk.opensim_sdk import SDK_NOT_INSTALLED_MESSAGE

            return f"{hint} ({SDK_NOT_INSTALLED_MESSAGE})"
        return hint

    def _sync_opensim_overview(self) -> None:
        """Refresh Session Overview OpenSim block from current session + files."""
        if not hasattr(self, "lbl_opensim_stats"):
            return
        if not self._opensim_has_session():
            self._update_opensim_overview(
                frames="—", markers="—", angles="—", export_summary="no session"
            )
            return
        detected_n = sum(
            1 for f in self.sequence.frames if getattr(f, "detected", False)
        )
        files = self._opensim_export_files_on_disk()
        if files.get("JSON"):
            n_markers, n_angles = self._count_opensim_export(
                config.OPENSIM_DIR
                / self._opensim_session_name()
                / f"{self._opensim_session_name()}_opensim.json"
            )
        else:
            from stablewalk.opensim_integration import build_motion_from_pose_sequence

            try:
                motion = build_motion_from_pose_sequence(self.sequence)
                n_markers = len(motion.marker_order)
                n_angles = max((len(f.angles) for f in motion.frames), default=0)
            except Exception:
                n_markers, n_angles = "—", "—"
        self._update_opensim_overview(
            frames=detected_n,
            markers=n_markers,
            angles=n_angles,
            export_summary=self._opensim_overview_export_summary(files),
        )

    def _opensim_trc_path(self) -> Path | None:
        """Path to the exported TRC for the current session, if it exists."""
        return self._opensim_raw_trc_path()

    def _cached_marker_mapping_comparison(self, trc_path: Path | None):
        """Return marker mapping, recomputing only when the TRC changes on disk."""
        from stablewalk.opensim_marker_mapping import compare_stablewalk_trc_to_opensim

        if trc_path is None:
            cache_key = None
        else:
            try:
                st = trc_path.stat()
                cache_key = (str(trc_path.resolve()), float(st.st_mtime), int(st.st_size))
            except OSError:
                cache_key = (str(trc_path), None, None)

        prev_key = getattr(self, "_marker_mapping_cache_key", object())
        prev = getattr(self, "_marker_mapping_comparison", None)
        if prev is not None and prev_key == cache_key:
            return prev

        comparison = compare_stablewalk_trc_to_opensim(trc_path)
        self._marker_mapping_cache_key = cache_key
        return comparison

    def _refresh_opensim_status(self) -> None:
        """Single source of truth for every OpenSim status field + button state."""
        if not hasattr(self, "lbl_opensim_sdk"):
            return

        from stablewalk.opensim_marker_mapping import (
            MEDIAPIPE_LIMITATION_EXPLANATION,
            demo_ik_setup_available,
            mapping_status_label,
            reliability_label,
            stablewalk_ik_status_label,
        )
        from stablewalk.opensim_sdk import check_opensim_sdk

        sdk_status = check_opensim_sdk(refresh=False)
        sdk = bool(sdk_status.available)
        self._opensim_sdk_available = sdk
        self._opensim_sdk_version = sdk_status.version

        model_valid = bool(getattr(self, "_opensim_model_valid", False))
        has_session = self._opensim_has_session()
        has_pose = self._opensim_has_pose_data()
        files = self._opensim_export_files_on_disk()
        trc_exported = bool(files.get("TRC"))
        trc_path = self._opensim_trc_path()

        # Marker mapping is expensive (rewrites mapped TRC). Cache by TRC mtime
        # so status refreshes during/after IK do not freeze the UI thread.
        self._marker_mapping_comparison = self._cached_marker_mapping_comparison(
            trc_path if trc_path and trc_path.is_file() else None
        )
        mapping = self._marker_mapping_comparison
        sw_state = getattr(self, "_opensim_ik_state", None)
        ik_status = stablewalk_ik_status_label(mapping, run_state=sw_state)
        experimental_ready = bool(mapping and mapping.ik_experimental_ready)

        from stablewalk.ui.tk.sidebar_display import (
            PRESENTATION_EXPORT_BUTTON_TOOLTIP,
            PRESENTATION_EXPORT_FILE,
            PRESENTATION_IK_BUTTON_TOOLTIP,
            PRESENTATION_IK_STATUS,
            PRESENTATION_LAST_EXPORT,
            PRESENTATION_LAST_IK,
            PRESENTATION_MAPPING_STATUS,
            PRESENTATION_MODE_SUBTITLE,
            PRESENTATION_MODE_TITLE,
            PRESENTATION_RELIABILITY,
            PRESENTATION_WORKFLOW_NOTE,
            compact_mode_line,
            compact_sdk_line,
            coverage_line,
            demo_ik_line,
            export_file_line,
            last_export_line,
            last_ik_line,
            mapping_compact_line,
            markers_count_line,
            reliability_line,
            sdk_tooltip_line,
            stablewalk_ik_line,
        )

        mapped_exported = bool(files.get("MAPPED"))
        demo_state = getattr(self, "_opensim_demo_ik_state", None)
        ik_mot_path = self._opensim_ik_mot_path()
        detail_lines: list[str] = []
        demo_ready = sdk and demo_ik_setup_available()
        presentation = bool(getattr(self, "_presentation_mode", False))
        self._set_opensim_presentation_banner(presentation)

        def _set(lbl: str, text: str, color_kind: str) -> None:
            w = getattr(self, lbl, None)
            if w is None:
                return
            color = self._opensim_status_color(color_kind)
            if lbl == "lbl_opensim_reliability":
                w.configure(text=text, fg=color)
                return
            w.configure(text=text, foreground=color)

        sdk_ver = getattr(self, "_opensim_sdk_version", None)
        if sdk:
            detail_lines.append(sdk_tooltip_line(version=sdk_ver))

        if presentation:
            if hasattr(self, "lbl_opensim_demo_mode"):
                self.lbl_opensim_demo_mode.configure(text=PRESENTATION_MODE_TITLE)
            if hasattr(self, "lbl_opensim_demo_subtitle"):
                self.lbl_opensim_demo_subtitle.configure(
                    text=PRESENTATION_MODE_SUBTITLE
                )
            if hasattr(self, "lbl_opensim_demo_note"):
                self.lbl_opensim_demo_note.configure(
                    text=PRESENTATION_WORKFLOW_NOTE
                )

            _set(
                "lbl_opensim_sdk",
                compact_sdk_line(installed=sdk, version=sdk_ver),
                "ok" if sdk else "fail",
            )
            _set("lbl_opensim_mode", compact_mode_line(sdk=sdk), "ok" if sdk else "muted")
            self._apply_opensim_model_status_ui(
                self._opensim_model_status_ui(model_valid=model_valid),
                model_valid=model_valid,
            )
            if demo_state == "Running":
                pres_demo_color = "info"
            elif demo_state == "Completed":
                pres_demo_color = "ok"
            elif demo_state == "Failed":
                pres_demo_color = "fail"
            elif demo_ready:
                pres_demo_color = "ok"
            else:
                pres_demo_color = "muted"
            _set(
                "lbl_opensim_demo_ik",
                demo_ik_line(state=demo_state, sdk_ready=demo_ready),
                pres_demo_color,
            )
            _set("lbl_opensim_stablewalk_ik", PRESENTATION_IK_STATUS, "info")
            _set("lbl_opensim_marker_mapping", PRESENTATION_MAPPING_STATUS, "info")
            _set("lbl_opensim_reliability", PRESENTATION_RELIABILITY, "muted")
            _set("lbl_opensim_coverage", "Coverage: —", "muted")
            _set("lbl_opensim_markers_count", "Markers: —", "muted")
            _set("lbl_opensim_ik_residuals", "IK residuals: —", "muted")
            _set("lbl_opensim_pose_ik", "Pose vs IK knees: —", "muted")

            for _name, _lbl in (
                ("TRC", "lbl_opensim_trc"),
                ("MOT", "lbl_opensim_mot"),
                ("JSON", "lbl_opensim_json"),
                ("Mapped TRC", "lbl_opensim_stablewalk_trc"),
            ):
                _set(_lbl, f"{_name}: {PRESENTATION_EXPORT_FILE}", "info")
            _set("lbl_opensim_last_export", PRESENTATION_LAST_EXPORT, "info")
            _set("lbl_opensim_last_ik", PRESENTATION_LAST_IK, "info")

            detail_lines.append(PRESENTATION_WORKFLOW_NOTE)
            if hasattr(self, "btn_opensim_run_demo_ik"):
                self.btn_opensim_run_demo_ik.configure(
                    state="normal" if demo_ready else "disabled"
                )
            if hasattr(self, "btn_opensim_run_ik"):
                self.btn_opensim_run_ik.configure(state="disabled")
            if hasattr(self, "btn_opensim_export"):
                self.btn_opensim_export.configure(state="disabled")
            if hasattr(self, "btn_opensim_select_model"):
                self.btn_opensim_select_model.configure(
                    state="normal" if sdk else "disabled"
                )
            if hasattr(self, "btn_opensim_local_model"):
                self.btn_opensim_local_model.configure(
                    state="normal" if self._opensim_model_choices else "disabled"
                )
            self._update_opensim_action_tooltips(presentation=True)
            self._opensim_detail_text = "\n\n".join(
                line for line in detail_lines if line and line.strip()
            )
            self._apply_opensim_compact_summary(
                sdk=sdk,
                model_valid=model_valid,
                export_complete=False,
                has_session=has_session,
                presentation=True,
            )
            self._sync_opensim_overview()
            return

        # ── 1) OpenSim Status (real analyzed video) ───────────────────────
        _set(
            "lbl_opensim_sdk",
            compact_sdk_line(installed=sdk, version=sdk_ver),
            "ok" if sdk else "fail",
        )
        _set("lbl_opensim_mode", compact_mode_line(sdk=sdk), "ok" if sdk else "muted")
        self._apply_opensim_model_status_ui(
            self._opensim_model_status_ui(model_valid=model_valid),
            model_valid=model_valid,
        )
        if demo_state == "Running":
            demo_color = "info"
        elif demo_state == "Completed":
            demo_color = "ok"
        elif demo_state == "Failed":
            demo_color = "fail"
        elif demo_ready:
            demo_color = "ok"
        else:
            demo_color = "muted"
        _set(
            "lbl_opensim_demo_ik",
            demo_ik_line(state=demo_state, sdk_ready=demo_ready),
            demo_color,
        )

        sw_ready = (
            not sw_state
            and mapped_exported
            and experimental_ready
            and sdk
            and ik_status == "Experimental but runnable"
        )
        if sw_state == "Running":
            sw_color = "info"
            sw_line = "StableWalk IK: Running"
        elif sw_state == "Completed":
            sw_color = "ok"
            sw_line = "StableWalk IK: Completed"
        elif sw_state == "Failed":
            sw_color = "fail"
            sw_line = "StableWalk IK: Failed"
        elif sw_ready:
            sw_color = "ok"
            sw_line = "StableWalk IK: Ready"
        elif ik_status in ("Experimental but runnable", "Experimental"):
            sw_color = "warn"
            sw_line = "StableWalk IK: Experimental"
        else:
            sw_color = "muted"
            sw_line = stablewalk_ik_line(run_state=sw_state, status_label=ik_status)
        _set("lbl_opensim_stablewalk_ik", sw_line, sw_color)

        map_label = mapping_status_label(mapping)
        if map_label == "Improved":
            map_color = "ok"
        elif map_label == "Partial":
            map_color = "warn"
        elif map_label == "Experimental":
            map_color = "warn"
        else:
            map_color = "muted"
        cov = mapping.coverage_percent if mapping else None
        map_pct = self._opensim_mapping_percent(mapping)
        raw_cov = None
        if mapping is not None:
            raw_cov = getattr(mapping, "raw_coverage_percent", None)
            if raw_cov is None:
                raw_cov = cov
        _set(
            "lbl_opensim_marker_mapping",
            mapping_compact_line(label=map_label, coverage_percent=map_pct),
            map_color,
        )
        matched_n = (
            len(mapping.mapped_matching_markers) if mapping is not None else None
        )
        ref_n = (
            len(mapping.opensim_reference_markers) if mapping is not None else None
        )
        _set(
            "lbl_opensim_coverage",
            coverage_line(percent=raw_cov),
            "ok"
            if raw_cov is not None and raw_cov >= 85
            else ("warn" if raw_cov is not None else "muted"),
        )
        _set(
            "lbl_opensim_markers_count",
            markers_count_line(
                matched=matched_n,
                reference=ref_n,
                mapping_percent=map_pct,
            ),
            "ok" if map_label == "Improved" else ("warn" if map_label not in ("—", "Missing", None) else "muted"),
        )
        self._update_opensim_validation_labels(
            _set=_set,
            detail_lines=detail_lines,
            run_name=self._opensim_session_name() if has_session else None,
        )
        rel = reliability_label(mapping)
        rel_color = "warn" if rel in ("experimental", "moderate", "limited") else "muted"
        _set("lbl_opensim_reliability", reliability_line(reliability=rel), rel_color)
        if rel in ("experimental", "moderate", "limited"):
            detail_lines.append(
                "Reliability: Experimental, not clinical-grade. "
                + "MediaPipe-based IK is not clinical-grade motion capture."
            )

        # ── 2) Export Status ────────────────────────────────────────────
        export_complete = bool(
            files.get("TRC") and files.get("MOT") and files.get("JSON")
        )
        self._opensim_export_completed = export_complete
        for key, lbl, name in (
            ("TRC", "lbl_opensim_trc", "TRC"),
            ("MOT", "lbl_opensim_mot", "MOT"),
            ("JSON", "lbl_opensim_json", "JSON"),
            ("MAPPED", "lbl_opensim_stablewalk_trc", "Mapped TRC"),
        ):
            exported = bool(files.get(key))
            if exported:
                file_color = "ok"
            elif has_session and has_pose:
                file_color = "fail"
            else:
                file_color = "muted"
            _set(
                lbl,
                export_file_line(
                    name, exported=exported, has_session=has_session, has_pose=has_pose
                ),
                file_color,
            )

        last_export_done = export_complete
        _set(
            "lbl_opensim_last_export",
            last_export_line(completed=last_export_done),
            "ok" if last_export_done else "muted",
        )

        ik_on_disk = bool(files.get("IK_MOT")) or (
            ik_mot_path is not None and ik_mot_path.is_file()
        )
        if sw_state == "Failed":
            last_ik_color = "fail"
        elif sw_state == "Completed":
            last_ik_color = "ok"
        elif sw_state == "Running":
            last_ik_color = "info"
        else:
            last_ik_color = "muted"
        _set(
            "lbl_opensim_last_ik",
            last_ik_line(run_state=sw_state, ik_output_on_disk=ik_on_disk),
            last_ik_color,
        )

        # Extended notes (log / details only — not in sidebar)
        block_reason = self._compute_opensim_ik_block_reason()
        self._opensim_ik_block_reason = block_reason
        if sw_state == "Completed" and ik_mot_path:
            detail_lines.append(f"IK completed — output: {self._format_data_path(ik_mot_path)}")
        elif sw_state == "Failed":
            detail_lines.append(getattr(self, "_opensim_ik_message", "") or "IK failed")
        elif block_reason:
            detail_lines.append(block_reason)
        if demo_state == "Completed" and getattr(self, "_opensim_demo_ik_message", ""):
            detail_lines.append(f"Demo IK: {self._opensim_demo_ik_message}")
        if mapping and mapping.message:
            detail_lines.append(mapping.message)
        override = getattr(self, "_opensim_status_override", None)
        if override:
            detail_lines.append(override)
        else:
            long_hint = self._opensim_status_hint(files, sdk)
            if long_hint:
                detail_lines.append(long_hint)
        if ik_status in ("Experimental but runnable", "Experimental"):
            detail_lines.append(MEDIAPIPE_LIMITATION_EXPLANATION)

        # Button states
        demo_ready = sdk and demo_ik_setup_available() and demo_state != "Running"
        if hasattr(self, "btn_opensim_run_demo_ik"):
            self.btn_opensim_run_demo_ik.configure(
                state="normal" if demo_ready else "disabled"
            )
        mapped_trc_ok = mapped_exported or self._opensim_mapped_trc_path() is not None
        mapping_ok = bool(
            mapping
            and mapping.mapping_status in ("improved", "partial", "experimental")
            and mapping.ik_experimental_ready
        )
        ik_model = self._resolve_opensim_ik_model_path()
        if hasattr(self, "btn_opensim_run_ik"):
            self.btn_opensim_run_ik.configure(
                state=(
                    "normal"
                    if (
                        sdk
                        and has_session
                        and not self._presentation_mode
                        and mapped_trc_ok
                        and mapping_ok
                        and sw_state != "Running"
                        and ik_model is not None
                        and Path(ik_model).is_file()
                    )
                    else "disabled"
                )
            )
        if hasattr(self, "btn_opensim_select_model"):
            self.btn_opensim_select_model.configure(
                state="normal" if sdk else "disabled"
            )
        if hasattr(self, "btn_opensim_local_model"):
            self.btn_opensim_local_model.configure(
                state="normal" if self._opensim_model_choices else "disabled"
            )
        # Export enablement is owned by _update_analysis_export_state (prerequisites).
        self._update_opensim_action_tooltips(presentation=False)
        self._update_export_section_status()
        self._update_analysis_export_state()
        self._opensim_detail_text = "\n\n".join(
            line for line in detail_lines if line and line.strip()
        )
        self._apply_opensim_compact_summary(
            sdk=sdk,
            model_valid=model_valid,
            export_complete=export_complete,
            has_session=has_session,
            presentation=False,
        )
        self._sync_opensim_overview()
        self._refresh_pipeline_status_panel(force=True)

    def _update_opensim_action_tooltips(self, *, presentation: bool) -> None:
        """Refresh OpenSim action button tooltips (display only)."""
        from stablewalk.ui.tk.sidebar_display import (
            PRESENTATION_EXPORT_BUTTON_TOOLTIP,
            PRESENTATION_IK_BUTTON_TOOLTIP,
        )

        export_tip = (
            PRESENTATION_EXPORT_BUTTON_TOOLTIP
            if presentation
            else "Export .trc, .mot/.csv, JSON, and mapped TRC for this session"
        )
        ik_tip = (
            PRESENTATION_IK_BUTTON_TOOLTIP
            if presentation
            else "Run StableWalk IK on exported mapped TRC (experimental)"
        )
        if hasattr(self, "btn_opensim_export"):
            self.btn_opensim_export._tooltip_text = export_tip
        if hasattr(self, "btn_opensim_run_ik"):
            self.btn_opensim_run_ik._tooltip_text = ik_tip

    def _run_opensim_demo_ik(self) -> None:
        """Run the official Gait2392 demo IK setup (proves OpenSim SDK integration)."""
        from stablewalk.opensim_sdk import check_opensim_sdk, run_opensim_demo_ik
        from stablewalk.ui.tk.perf_diagnostics import timed_section

        status = check_opensim_sdk(refresh=True)
        if not status.available:
            messagebox.showinfo("Run OpenSim Demo IK", status.message)
            return

        if self._analysis_compute_active or (
            getattr(self, "_opensim_worker", None) is not None
            and self._opensim_worker.is_alive()
        ):
            messagebox.showinfo(
                "Run OpenSim Demo IK",
                "Another analysis / OpenSim job is already running.",
            )
            return

        self._opensim_demo_ik_state = "Running"
        self._opensim_demo_ik_message = ""
        self._refresh_opensim_status()

        def _worker() -> None:
            with timed_section(self, "opensim", warn_label="OpenSim demo IK"):
                result = run_opensim_demo_ik()
            self.root.after(0, lambda: self._on_opensim_demo_ik_done(result))

        self._opensim_worker = threading.Thread(
            target=_worker, name="stablewalk-opensim-demo-ik", daemon=True
        )
        self._opensim_worker.start()

    def _on_opensim_demo_ik_done(self, result) -> None:
        self._opensim_worker = None
        if result.ran and result.output_motion_path and Path(result.output_motion_path).is_file():
            self._opensim_demo_ik_state = "Completed"
            self._opensim_demo_ik_message = Path(result.output_motion_path).name
            from stablewalk.opensim_sdk import update_opensim_status_md

            update_opensim_status_md(demo_ik_result=result)
            messagebox.showinfo(
                "OpenSim Demo IK complete",
                f"{result.message}\n\nSetup:\n{result.setup_path}\n\nOutput:\n{result.output_motion_path}",
            )
        else:
            self._opensim_demo_ik_state = "Failed"
            self._opensim_demo_ik_message = result.message[:80]
            messagebox.showerror("OpenSim Demo IK failed", result.message)
        self._refresh_opensim_status()

    def _select_opensim_model(self) -> None:
        """
        Load an OpenSim musculoskeletal model (.osim) via opensim.Model.

        Requires the real OpenSim Python SDK. Never crashes if the SDK is absent.
        """
        path = filedialog.askopenfilename(
            title="Select an OpenSim musculoskeletal model",
            initialdir=str(config.OPENSIM_MODELS_DIR),
            filetypes=[("OpenSim model", "*.osim"), ("All files", "*.*")],
        )
        if not path:
            return

        if not getattr(self, "_opensim_sdk_available", False):
            from stablewalk.opensim_sdk import SDK_NOT_INSTALLED_MESSAGE

            messagebox.showerror("Load OpenSim Model", SDK_NOT_INSTALLED_MESSAGE)
            return

        info = self._load_opensim_model_path(Path(path))
        if info is not None:
            self._show_opensim_model_loaded_success(info)

    def _run_stablewalk_ik_experimental(self, *, quiet: bool = False) -> bool:
        """Run experimental OpenSim IK on the current session mapped TRC.

        Returns True when IK completes successfully. When ``quiet`` is True,
        skip modal dialogs (used by auto pipeline after analysis).
        """
        from stablewalk.opensim_marker_mapping import (
            MEDIAPIPE_LIMITATION_EXPLANATION,
            compare_stablewalk_trc_to_opensim,
        )
        from stablewalk.opensim_sdk import check_opensim_sdk, run_stablewalk_ik_experimental

        self._log_export("Run StableWalk IK Experimental clicked")

        status = check_opensim_sdk(refresh=True)
        if not status.available:
            self._log_export("OpenSim SDK not installed")
            if not quiet:
                messagebox.showerror("Run StableWalk IK Experimental", status.message)
            self._opensim_ik_block_reason = f"IK cannot run: {status.message}"
            self._refresh_opensim_status()
            return False

        if self._presentation_mode or not self._current_run_name():
            self._log_export("IK aborted: no video/session loaded")
            if not quiet:
                messagebox.showerror(
                    "Run StableWalk IK Experimental",
                    _OPENSIM_NO_SESSION_MSG,
                )
            self._refresh_opensim_status()
            return False

        run_name = self._current_run_name()
        assert run_name is not None
        out_dir = config.OPENSIM_DIR / run_name
        raw_trc = out_dir / f"{run_name}.trc"
        mapped_trc = out_dir / f"{run_name}_mapped_for_opensim.trc"
        setup_xml = out_dir / "stablewalk_setup_ik.xml"
        ik_mot = out_dir / f"{run_name}_ik.mot"
        model_path = self._resolve_opensim_ik_model_path()

        self._log_export("Current run/session: %s", run_name)
        self._log_export("Using mapped TRC: %s", mapped_trc)
        self._log_export("Using OpenSim model: %s", model_path)
        self._log_export("IK setup XML: %s", setup_xml)
        self._log_export("IK output path: %s", ik_mot)

        if model_path is None or not Path(model_path).is_file():
            self._log_export("OpenSim model missing: %s", model_path)
            msg = (
                f"OpenSim model not found:\n{model_path}"
                if model_path
                else "No usable OpenSim .osim model is loaded or available."
            )
            if not quiet:
                messagebox.showerror("Run StableWalk IK Experimental", msg)
            self._opensim_ik_block_reason = f"IK cannot run: {msg.replace(chr(10), ' ')}"
            self._refresh_opensim_status()
            return False

        if not mapped_trc.is_file():
            self._log_export("Mapped TRC missing: %s", mapped_trc)
            if not quiet:
                messagebox.showerror(
                    "Run StableWalk IK Experimental",
                    "Please click Export OpenSim Files first.",
                )
            self._refresh_opensim_status()
            return False

        if not raw_trc.is_file():
            if not quiet:
                messagebox.showerror(
                    "Run StableWalk IK Experimental",
                    "Please click Export OpenSim Files first.",
                )
            self._refresh_opensim_status()
            return False

        mapping = compare_stablewalk_trc_to_opensim(raw_trc)
        self._marker_mapping_comparison = mapping
        try:
            st = raw_trc.stat()
            self._marker_mapping_cache_key = (
                str(raw_trc.resolve()),
                float(st.st_mtime),
                int(st.st_size),
            )
        except OSError:
            self._marker_mapping_cache_key = None
        self._log_export("Marker mapping status: %s", mapping.mapping_status)
        self._log_export("Coverage: %s%%", mapping.coverage_percent)
        self._log_export("Warning: this is experimental and not clinical-grade")

        if mapping.mapping_status not in ("improved", "partial", "experimental"):
            msg = (
                f"Marker mapping not ready (status: {mapping.mapping_status}).\n\n"
                f"{mapping.message}"
            )
            if not quiet:
                messagebox.showerror("Run StableWalk IK Experimental", msg)
            self._refresh_opensim_status()
            return False

        if not mapping.ik_experimental_ready:
            msg = (
                f"{mapping.message}\n\nMapped matches: "
                f"{len(mapping.mapped_matching_markers)}"
            )
            if not quiet:
                messagebox.showerror("Run StableWalk IK Experimental", msg)
            self._refresh_opensim_status()
            return False

        self._opensim_ik_state = "Running"
        self._opensim_ik_message = ""
        self._refresh_opensim_status()

        if (
            getattr(self, "_opensim_worker", None) is not None
            and self._opensim_worker.is_alive()
        ):
            if not quiet:
                messagebox.showinfo(
                    "Run StableWalk IK Experimental",
                    "Another OpenSim job is already running.",
                )
            self._opensim_ik_state = "Failed"
            self._refresh_opensim_status()
            return False

        def _worker() -> None:
            from stablewalk.ui.tk.perf_diagnostics import timed_section

            with timed_section(self, "opensim", warn_label="OpenSim IK"):
                result = run_stablewalk_ik_experimental(
                    mapped_trc,
                    model_path=model_path,
                    run_name=run_name,
                )
            self.root.after(
                0,
                lambda: self._on_stablewalk_ik_done(
                    result, out_dir=out_dir, ik_mot=ik_mot, quiet=quiet
                ),
            )

        self._opensim_worker = threading.Thread(
            target=_worker, name="stablewalk-opensim-ik", daemon=True
        )
        self._opensim_worker.start()
        # Async — caller must not assume completion yet. Quiet auto-pipeline
        # continues via ``_on_stablewalk_ik_done``.
        return True

    def _on_stablewalk_ik_done(
        self,
        result,
        *,
        out_dir: Path,
        ik_mot: Path,
        quiet: bool = False,
    ) -> None:
        self._opensim_worker = None
        from stablewalk.opensim_marker_mapping import MEDIAPIPE_LIMITATION_EXPLANATION

        output_path = Path(result.output_motion_path) if result.output_motion_path else None
        rel_mot = self._format_data_path(ik_mot)

        if result.ran and output_path and output_path.is_file():
            try:
                from stablewalk.analysis.gait_feature_analysis import (
                    clear_opensim_mot_timeseries_cache,
                )

                clear_opensim_mot_timeseries_cache()
            except Exception:
                pass
            self._opensim_ik_state = "Completed"
            self._opensim_ik_message = self._format_data_path(output_path)
            self._opensim_last_dir = out_dir
            self._opensim_status_override = (
                f"StableWalk IK: Completed\nIK output: {rel_mot}\n"
                f"OpenSim output folder: {self._format_data_path(out_dir)}"
            )
            if not quiet:
                messagebox.showinfo(
                    "StableWalk IK complete (experimental)",
                    f"{result.message}\n\n"
                    f"IK output:\n  {rel_mot}\n\n"
                    f"Setup:\n  {self._format_data_path(Path(result.setup_path)) if result.setup_path else out_dir / 'stablewalk_setup_ik.xml'}\n\n"
                    f"{MEDIAPIPE_LIMITATION_EXPLANATION}",
                )
            try:
                self._rebuild_knee_angle_series()
                self._refresh_knee_source_selector()
                self._update_chart()
            except Exception:
                logger.debug("post-IK chart refresh failed", exc_info=True)
        else:
            self._opensim_ik_state = "Failed"
            self._opensim_ik_message = (result.message or "IK failed")[:120]
            self._log_export("StableWalk IK failed: %s", result.message)
            if not quiet:
                messagebox.showerror(
                    "StableWalk IK failed",
                    result.message or "OpenSim IK did not produce an output motion.",
                )
        self._refresh_opensim_status()
        self._refresh_pipeline_status_panel(force=True)
        if getattr(self, "_opensim_auto_pipeline_busy", False):
            self._opensim_auto_pipeline_busy = False
        self._clear_opensim_panel_loading()

    def _advanced_export_busy(self) -> bool:
        worker = getattr(self, "_advanced_export_worker", None)
        return worker is not None and worker.is_alive()

    def _run_advanced_export_async(
        self,
        title: str,
        work: Callable[[], Any],
        on_success: Callable[[Any], None] | None = None,
    ) -> None:
        """Run a heavy Advanced-tab export off the UI thread."""
        if self._advanced_export_busy() or (
            getattr(self, "_opensim_worker", None) is not None
            and self._opensim_worker.is_alive()
        ):
            messagebox.showinfo(title, "Another export is already running.")
            return
        try:
            self.status.configure(text=f"{title}…")
        except tk.TclError:
            pass

        def worker() -> None:
            err: Exception | None = None
            result: Any = None
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001 — surfaced in UI
                err = exc

            def finish() -> None:
                self._advanced_export_worker = None
                if err is not None:
                    messagebox.showerror(title, str(err))
                    try:
                        self.status.configure(text=f"{title} failed")
                    except tk.TclError:
                        pass
                    return
                if on_success is not None:
                    on_success(result)
                self._notify_generated_files_changed()
                self._update_analysis_export_state()

            try:
                self.root.after(0, finish)
            except tk.TclError:
                finish()

        self._advanced_export_worker = threading.Thread(
            target=worker, name=f"stablewalk-{title}", daemon=True
        )
        self._advanced_export_worker.start()

    def _export_opensim_session(self, *, quiet: bool = False) -> bool:
        """Export the currently analyzed session to OpenSim files (sidebar button).

        Returns True when TRC/MOT/JSON were written. ``quiet`` skips modal dialogs.
        UI clicks (quiet=False) run asynchronously so the interface never freezes.
        """
        if (
            not quiet
            and threading.current_thread() is threading.main_thread()
            and not getattr(self, "_opensim_session_export_sync", False)
        ):
            def _opensim_work() -> bool:
                self._export_ui_suspended = True
                try:
                    return bool(self._export_opensim_session(quiet=True))
                finally:
                    self._export_ui_suspended = False

            self._run_advanced_export_async(
                "OpenSim Export",
                _opensim_work,
                on_success=lambda ok: self._on_opensim_session_export_async_done(bool(ok)),
            )
            return False

        self._log_export("Export OpenSim Files clicked")
        self._opensim_export_completed = False
        self._opensim_status_override = None
        ui_ok = not getattr(self, "_export_ui_suspended", False)

        if self._presentation_mode:
            self._log_export("Export aborted: presentation demo active (no real session)")
            if not quiet:
                messagebox.showerror("OpenSim Export", _OPENSIM_NO_SESSION_MSG)
            if ui_ok:
                self._refresh_opensim_status()
            self._opensim_status_override = _OPENSIM_NO_SESSION_MSG
            return False

        run_name = self._current_run_name()
        if not run_name:
            self._log_export("Export aborted: no current run/session name")
            if not quiet:
                messagebox.showerror("OpenSim Export", _OPENSIM_NO_SESSION_MSG)
            if ui_ok:
                self._refresh_opensim_status()
            self._opensim_status_override = _OPENSIM_NO_SESSION_MSG
            return False

        self._log_export("Current run/session: %s", run_name)

        expected_poses = config.POSES_DIR / f"{run_name}_poses.json"
        poses_path = self._resolve_poses_json_path()
        poses_exists = poses_path is not None and poses_path.is_file()
        self._log_export("Pose JSON: %s", poses_path.resolve() if poses_path else "(none)")
        self._log_export("Pose JSON exists: %s", poses_exists)
        if not poses_exists:
            self._log_export("Expected pose JSON path: %s", expected_poses)
            if not quiet:
                messagebox.showerror("OpenSim Export", _OPENSIM_POSE_JSON_MISSING_MSG)
            if ui_ok:
                self._refresh_opensim_status()
            self._opensim_status_override = _OPENSIM_POSE_JSON_MISSING_MSG
            return False

        out_dir = config.OPENSIM_DIR / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        rel_out = self._format_data_path(out_dir)
        self._log_export("Output folder: %s", out_dir.resolve())

        try:
            sequence = load_pose_sequence(poses_path)
            self.sequence = sequence
            self._poses_path = poses_path
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._log_export("Export failed: %s", exc)
            if not quiet:
                messagebox.showerror("OpenSim Export", f"Export failed: {exc}")
            self._opensim_status_override = f"Export failed: {exc}"
            if ui_ok:
                self._refresh_opensim_status()
            return False

        valid_frames = sum(1 for f in sequence.frames if getattr(f, "detected", False))
        self._log_export("Valid pose frames: %d", valid_frames)
        if valid_frames == 0:
            if not quiet:
                messagebox.showerror("OpenSim Export", _OPENSIM_NO_VALID_FRAMES_MSG)
            self._opensim_status_override = _OPENSIM_NO_VALID_FRAMES_MSG
            if ui_ok:
                self._refresh_opensim_status()
            return False

        from stablewalk.opensim_integration import export_opensim_files
        from stablewalk.opensim_marker_mapping import create_mapped_trc, mapped_trc_path_for

        motion_as_csv = self.opensim_motion_fmt.get().lower() == ".csv"
        motion_ext = ".csv" if motion_as_csv else ".mot"
        trc_target = out_dir / f"{run_name}.trc"
        mot_target = out_dir / f"{run_name}{motion_ext}"
        json_target = out_dir / f"{run_name}_opensim.json"
        mapped_target = mapped_trc_path_for(trc_target)

        stability = self._biomech.to_dict() if self._biomech else None
        selected_dof = sorted(self.selection.selected) if hasattr(self, "selection") else []
        try:
            from stablewalk.ui.tk.perf_diagnostics import timed_section

            self._log_export("Writing TRC...")
            self._log_export("Writing MOT...")
            self._log_export("Writing JSON...")
            with timed_section(self, "opensim", warn_label="OpenSim export"):
                written = export_opensim_files(
                    sequence,
                    out_dir,
                    name=run_name,
                    motion_as_csv=motion_as_csv,
                    stability=stability,
                    selected_dof=selected_dof,
                )
            self._log_export("  TRC path: %s", written["trc"].resolve())
            self._log_export("  MOT path: %s", written["motion"].resolve())
            self._log_export("  JSON path: %s", written["json"].resolve())
        except Exception as exc:
            self._log_export("Export failed: %s", exc)
            logger.exception("OpenSim export failed")
            if not quiet:
                messagebox.showerror("OpenSim Export", f"Export failed: {exc}")
            self._opensim_status_override = f"Export failed: {exc}"
            if ui_ok:
                self._refresh_opensim_status()
            return False

        missing = [
            p for p in (trc_target, mot_target, json_target) if not p.is_file()
        ]
        if missing:
            missing_rel = [self._format_data_path(p) for p in missing]
            self._log_export("Export incomplete — missing: %s", ", ".join(missing_rel))
            missing_msg = (
                "Export did not create all required files.\n\nMissing:\n"
                + "\n".join(f"  • {m}" for m in missing_rel)
            )
            if not quiet:
                messagebox.showerror("OpenSim Export", missing_msg)
            self._opensim_status_override = missing_msg
            if ui_ok:
                self._refresh_opensim_status()
            return False

        try:
            self._log_export("Writing mapped TRC for OpenSim...")
            create_mapped_trc(trc_target, mapped_target)
            self._log_export("  Mapped TRC path: %s", mapped_target.resolve())
        except Exception as exc:
            self._log_export("Mapped TRC skipped: %s", exc)
            logger.warning("Mapped TRC not created: %s", exc)

        self._log_export("Export completed successfully")
        self._log_export("  TRC: %s", trc_target.resolve())
        self._log_export("  MOT: %s", mot_target.resolve())
        self._log_export("  JSON: %s", json_target.resolve())
        if mapped_target.is_file():
            self._log_export("  Mapped TRC: %s", mapped_target.resolve())

        self._opensim_last_dir = out_dir
        self._active_run_name = run_name
        self._opensim_export_completed = True
        self._opensim_status_override = (
            f"Export completed successfully.\nOutput folder: {rel_out}"
        )
        mapped_note = (
            f"\n  {mapped_target.name}   (mapped markers)"
            if mapped_target.is_file()
            else ""
        )
        self._opensim_async_success_msg = (
            "Export completed successfully.\n\n"
            f"OpenSim output folder:\n  {rel_out}\n\n"
            f"  {run_name}.trc\n"
            f"  {run_name}{motion_ext}\n"
            f"  {run_name}_opensim.json{mapped_note}\n\n"
            "Use “Open Output Folder” to view files in Explorer."
        )
        if ui_ok:
            self.status.configure(text=f"OpenSim export → {rel_out}")
            self._refresh_opensim_status()
            self.root.update_idletasks()
            if not quiet:
                messagebox.showinfo(
                    "OpenSim export complete",
                    self._opensim_async_success_msg,
                )
            self._notify_generated_files_changed()
        return True

    def _on_opensim_session_export_async_done(self, ok: bool) -> None:
        """Main-thread finish for Advanced → Export OpenSim Files."""
        try:
            self._refresh_opensim_status()
        except Exception:
            pass
        if ok:
            msg = getattr(self, "_opensim_async_success_msg", None) or (
                getattr(self, "_opensim_status_override", None)
                or "OpenSim export completed."
            )
            messagebox.showinfo("OpenSim export complete", msg)
            try:
                self.status.configure(text="OpenSim export complete")
            except tk.TclError:
                pass
            self._refresh_pipeline_status_panel(force=True)
            return
        err = getattr(self, "_opensim_status_override", None) or (
            "OpenSim export did not complete. Check pose data and try again."
        )
        messagebox.showerror("OpenSim Export", err)
        try:
            self.status.configure(text="OpenSim export failed")
        except tk.TclError:
            pass

    def _maybe_auto_opensim_pipeline(self) -> None:
        """Queue quiet export + IK; run only when Advanced is the active tab.

        Keeps Overview / playback free of disk and OpenSim work after analysis.
        """
        if getattr(self, "_presentation_mode", False):
            return
        if getattr(self, "_demo_running", False):
            return
        if getattr(self, "_session_restoring", False):
            return
        if not getattr(self, "_opensim_sdk_available", False):
            return
        if getattr(self, "_opensim_auto_pipeline_busy", False):
            return
        try:
            from stablewalk.ui.tk.dashboard_notebook import TAB_ADVANCED
            from stablewalk.ui.tk.dashboard_tab_lifecycle import active_tab_id

            if active_tab_id(self) != TAB_ADVANCED:
                self._opensim_auto_pending = True
                return
        except Exception:
            # If tab resolution fails, still defer rather than hitch Overview.
            self._opensim_auto_pending = True
            return
        self._opensim_auto_pending = False
        # Advanced is open — short idle so the tab chrome paints first.
        try:
            self.root.after(200, self._run_auto_opensim_pipeline)
        except Exception:
            logger.debug("Could not schedule OpenSim auto pipeline", exc_info=True)

    def _clear_opensim_panel_loading(self) -> None:
        from stablewalk.ui.tk.panel_loading import set_panel_loading

        set_panel_loading(self, "opensim", False)

    def _run_auto_opensim_pipeline(self) -> None:
        """Quiet export → IK off the UI thread; never block Overview paint."""
        if getattr(self, "_opensim_auto_pipeline_busy", False):
            return
        if not getattr(self, "_opensim_sdk_available", False):
            self._refresh_opensim_status()
            self._clear_opensim_panel_loading()
            return
        if getattr(self, "_presentation_mode", False) or not self._opensim_has_session():
            self._refresh_opensim_status()
            self._clear_opensim_panel_loading()
            return

        from stablewalk.ui.tk.panel_loading import set_panel_loading

        set_panel_loading(self, "opensim", True, message="Loading OpenSim…")
        self._opensim_auto_pipeline_busy = True
        try:
            if not getattr(self, "_opensim_model_valid", False):
                self._auto_load_configured_opensim_model()

            files = self._opensim_export_files_on_disk()
            need_export = not (
                files.get("TRC") and files.get("MOT") and files.get("MAPPED")
            )
            if need_export and self._opensim_has_pose_data():
                logger.info("OpenSim auto-pipeline: exporting TRC/MOT/mapped markers")
                self._run_auto_opensim_export_async()
                return

            self._continue_auto_opensim_after_export()
        except Exception:
            logger.exception("OpenSim auto-pipeline failed")
            self._opensim_auto_pipeline_busy = False
            self._refresh_opensim_status()
            self._clear_opensim_panel_loading()
            self._refresh_pipeline_status_panel(force=True)

    def _run_auto_opensim_export_async(self) -> None:
        """Export OpenSim files on a worker; continue IK on the UI thread after."""

        def work() -> bool:
            return bool(self._export_opensim_session(quiet=True))

        def finish(ok: bool, err: Exception | None) -> None:
            if err is not None or not ok:
                logger.info("OpenSim auto-pipeline: export failed or skipped")
                self._opensim_auto_pipeline_busy = False
                self._refresh_opensim_status()
                self._clear_opensim_panel_loading()
                self._refresh_pipeline_status_panel(force=True)
                return
            self._continue_auto_opensim_after_export()

        def worker() -> None:
            err: Exception | None = None
            ok = False
            try:
                ok = work()
            except Exception as exc:  # noqa: BLE001
                err = exc

            def _ui() -> None:
                finish(ok, err)

            try:
                self.root.after(0, _ui)
            except tk.TclError:
                _ui()

        threading.Thread(
            target=worker, name="stablewalk-opensim-auto-export", daemon=True
        ).start()

    def _continue_auto_opensim_after_export(self) -> None:
        """IK step after quiet export (UI thread)."""
        try:
            if getattr(self, "_opensim_ik_state", None) == "Completed":
                self._refresh_opensim_status()
                return

            files = self._opensim_export_files_on_disk()
            if files.get("IK_MOT"):
                # Do not re-run IK, and do not mark Complete from leftover files.
                logger.info(
                    "OpenSim auto-pipeline: IK output file present; "
                    "IK process not marked complete from file presence"
                )
                self._refresh_opensim_status()
                return

            if getattr(self, "_opensim_ik_state", None) in ("Failed", "Running"):
                self._refresh_opensim_status()
                return

            block = self._compute_opensim_ik_block_reason()
            self._opensim_ik_block_reason = block
            if block is None:
                logger.info("OpenSim auto-pipeline: running StableWalk IK")
                self._run_stablewalk_ik_experimental(quiet=True)
                # IK continues on worker — spinner clears in ``_on_stablewalk_ik_done``.
                return
            logger.info("OpenSim auto-pipeline: IK skipped — %s", block)
            self._refresh_opensim_status()
        except Exception:
            logger.exception("OpenSim auto-pipeline failed after export")
            self._refresh_opensim_status()
        finally:
            if getattr(self, "_opensim_ik_state", None) != "Running":
                self._opensim_auto_pipeline_busy = False
                self._clear_opensim_panel_loading()
            self._refresh_pipeline_status_panel(force=True)

    def _opensim_session_name(self) -> str:
        return self._current_run_name() or "stablewalk_motion"

    def _update_opensim_overview(
        self,
        *,
        frames: object,
        markers: object,
        angles: object,
        export_summary: str,
    ) -> None:
        """Legacy hook — OpenSim summary lives in the OpenSim Status panel only."""
        del frames, markers, angles, export_summary

    def _notify_generated_files_changed(self, *, force: bool = True) -> None:
        """Refresh the Advanced → Generated Files list after export activity."""
        from stablewalk.ui.tk.generated_files_panel import notify_generated_files_changed

        # Never walk the output tree during playback — defer until pause.
        if getattr(self, "playing", False):
            self._generated_files_dirty = True
            if force:
                from stablewalk.ui.tk.generated_files_panel import (
                    invalidate_generated_files_scan_cache,
                )

                invalidate_generated_files_scan_cache()
            return
        notify_generated_files_changed(self, force=force)

    def _count_opensim_export(self, json_path: Path) -> tuple[int, int]:
        """Read marker / joint-angle counts back from the exported JSON."""
        try:
            import json as _json

            data = _json.loads(Path(json_path).read_text(encoding="utf-8"))
            markers = len(data.get("marker_order", []))
            angles = 0
            for frame in data.get("frames", []):
                angles = max(angles, len(frame.get("joint_angles_deg", {})))
            return markers, angles
        except Exception:
            return 0, 0

    def _open_opensim_folder(self) -> None:
        """Open the current session OpenSim output folder in the system file manager."""
        import os
        import subprocess
        import sys

        target = (
            getattr(self, "_opensim_last_dir", None)
            or self._opensim_output_dir()
            or config.OPENSIM_DIR
        )
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        folder = str(target.resolve())
        self._log_export("Open Output Folder: %s", folder)
        try:
            if sys.platform == "win32":
                os.startfile(folder)  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.run(["open", folder], check=False)
            else:
                subprocess.run(["xdg-open", folder], check=False)
        except Exception as exc:
            messagebox.showerror("Open Output Folder", f"Could not open folder:\n{folder}\n\n{exc}")

    def _export_opensim(self) -> None:
        """Export the current pose data as OpenSim-compatible files (.trc/.mot/JSON)."""
        if not self.sequence:
            messagebox.showinfo("Export", "Load pose data first (Run Full Analysis).")
            return
        from stablewalk.opensim_integration import (
            check_opensim_availability,
            export_opensim_files,
        )

        out_dir = filedialog.askdirectory(
            title="Choose output folder for OpenSim files",
            initialdir=str(config.OPENSIM_DIR),
        )
        if not out_dir:
            return
        try:
            name = Path(self._poses_path).stem.replace("_poses", "") if self._poses_path else "stablewalk_motion"
            written = export_opensim_files(self.sequence, out_dir, name=name)
            avail = check_opensim_availability()
            sdk = "installed" if avail["available"] else "not installed (files still exported)"
            self.status.configure(text=f"OpenSim export → {written['trc'].name}")
            messagebox.showinfo(
                "OpenSim export complete",
                "Exported OpenSim-compatible files:\n"
                f"  {written['trc'].name}  (markers)\n"
                f"  {written['motion'].name}  (joint angles)\n"
                f"  {written['json'].name}  (full data)\n\n"
                f"Folder: {out_dir}\n"
                f"OpenSim SDK: {sdk}",
            )
            self._notify_generated_files_changed()
        except (OSError, ValueError) as exc:
            messagebox.showerror("OpenSim export failed", str(exc))

    def _compare_video(self) -> None:
        if not self.sequence or not self._poses_path:
            messagebox.showinfo("Compare", "Load primary gait data first.")
            return
        path = filedialog.askopenfilename(
            title="Select comparison video",
            initialdir=str(config.INPUT_DIR),
            filetypes=[
                ("Video", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        ok, msg = check_source_format(path)
        if not ok:
            messagebox.showerror("Invalid video", msg)
            return
        self._show_compare_note("Comparing…")
        self.status.configure(text="Comparing…")

        def worker() -> None:
            try:
                from stablewalk.core.pipeline import run_gait_pipeline

                run_name = derive_run_name(path) + "_cmp"
                cmp_result = run_gait_pipeline(
                    path,
                    run_name=run_name,
                    validate=resolve_validate_mode(path),
                    force_reprocess=True,
                )
                self.root.after(
                    0,
                    lambda r=cmp_result: self._compare_done(
                        str(self._poses_path), str(r.poses_path), None
                    ),
                )
            except Exception as exc:
                err = exc
                self.root.after(0, lambda e=err: self._compare_done("", "", e))

        threading.Thread(target=worker, daemon=True).start()

    def _compare_done(
        self,
        ref_path: str,
        sample_path: str,
        error: Exception | None,
    ) -> None:
        if error:
            messagebox.showerror("Compare failed", str(error))
            self.status.configure(text="Compare failed")
            return
        try:
            from stablewalk.analysis.comparison import compare_pose_files

            self._compare = compare_pose_files(ref_path, sample_path)
            self._show_compare_note(self._compare.summary)
            self._update_compare_chart()
            self.status.configure(text="Comparison ready")
        except Exception as exc:
            messagebox.showerror("Compare error", str(exc))

    def _open_url_list(self) -> None:
        from stablewalk.ui.media.catalog import ensure_user_url_template

        ensure_user_url_template()
        path = config.DATA_DIR / "men_walk_urls.txt"
        import os
        import subprocess
        import sys

        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
        messagebox.showinfo(
            "URL list",
            f"Edit demo walking URLs in:\n{path}\n\n"
            "One per line:  Label | https://...\n"
            "Verified clips: data/verified_men_walk.json\n"
            "Restart the app after saving to load new URLs.",
        )

    def _open_output_folder(self) -> None:
        config.POSES_DIR.mkdir(parents=True, exist_ok=True)
        import os
        import subprocess
        import sys

        folder = str(config.OUTPUT_DIR.resolve())
        if sys.platform == "win32":
            os.startfile(folder)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", folder], check=False)
        else:
            subprocess.run(["xdg-open", folder], check=False)

    def _show_analysis_methods(self) -> None:
        from stablewalk import __version__

        backend = self._selected_pose_backend()
        messagebox.showinfo(
            "Analysis methods",
            f"StableWalk {__version__}\n\n"
            f"{METHODS_OVERVIEW}\n\n"
            f"Active pose backend: {backend}\n\n"
            f"{UNITS_COORD_LEGEND}\n\n"
            f"{DISCLAIMER_RESEARCH}",
        )

    def _show_metric_details_catalog(self) -> None:
        from stablewalk.ui.tk.metric_details_dialog import show_metric_details_dialog

        show_metric_details_dialog(self.root, "gait_quality")

    def _show_shortcuts(self) -> None:
        messagebox.showinfo(
            "Shortcuts",
            "Help\n"
            "  F1 — Analysis methods\n\n"
            "Playback\n"
            "  Space — Play / Pause\n"
            "  ← / → — Previous / Next pose frame\n"
            "  Home / End — First / Last pose\n\n"
            "Session\n"
            "  Ctrl+N — New session\n"
            "  Ctrl+O — Open pose JSON\n"
            "  Ctrl+S — Save session\n"
            "  Ctrl+L — Load session\n\n"
            "3D path\n"
            "  Ctrl+Click — Add joint for comparison",
        )

    def _debug_inspect_value_provenance(self) -> None:
        """Debug → Inspect value provenance — print live metric sources."""
        from stablewalk.ui.tk.value_provenance_dialog import show_value_provenance_dialog

        show_value_provenance_dialog(self)

    def _debug_toggle_perf_diagnostics(self) -> None:
        """Debug → Performance Diagnostics checkbutton."""
        from stablewalk.ui.tk.perf_diagnostics import set_perf_diagnostics

        enabled = bool(self._perf_diag_var.get())
        set_perf_diagnostics(self, enabled)
        state = "ON" if enabled else "OFF"
        self.status.configure(text=f"Performance Diagnostics {state}")
        if enabled:
            print(
                "[Perf] Diagnostics ON — budgets: UI ≤16 ms (play), "
                "chart ≤50 ms; one playback timer; one analysis worker",
                flush=True,
            )

    def _debug_show_perf_diagnostics(self) -> None:
        """Debug → Performance Diagnostics report…"""
        from stablewalk.ui.tk.perf_diagnostics_dialog import show_perf_diagnostics_dialog

        self._perf_diag_var.set(True)
        self._debug_toggle_perf_diagnostics()
        show_perf_diagnostics_dialog(self)

    def _debug_dump_gui_memory(self) -> None:
        """Debug → Dump memory / GUI state — timers, figures, caches, RSS."""
        from stablewalk.ui.tk.gui_memory import dump_gui_memory_state

        dump_gui_memory_state(self, to_console=True, show_dialog=True)

    def _debug_dump_overview_layout(self) -> None:
        """Debug → Dump Overview Layout — hierarchy, geometry, timers, canvases."""
        import os

        from stablewalk.ui.tk.overview_layout_dump import dump_overview_layout
        from stablewalk.ui.tk.overview_responsive import layout_diag_enabled

        debug_on = bool(getattr(self, "_perf_diagnostics", False)) or layout_diag_enabled(
            self
        ) or os.environ.get("STABLEWALK_DEBUG", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not debug_on:
            messagebox.showinfo(
                "Dump Overview Layout",
                "Enable Debug → Performance Diagnostics (or set "
                "STABLEWALK_LAYOUT_DIAG=1 / STABLEWALK_DEBUG=1) to dump the "
                "Overview widget tree.\n\n"
                "The action stays available; detailed dump runs in debug mode.",
            )
            return
        text = dump_overview_layout(self)
        messagebox.showinfo(
            "Overview layout dump",
            "Overview hierarchy written to the console / log.\n\n"
            + "\n".join(text.splitlines()[:24])
            + ("\n…" if text.count("\n") > 24 else ""),
        )

    def _show_about(self) -> None:
        from stablewalk import __version__

        messagebox.showinfo(
            "About StableWalk",
            f"StableWalk {__version__} — Gait Analysis Dashboard\n\n"
            f"{METHODS_OVERVIEW}\n\n"
            f"{DISCLAIMER_RESEARCH}\n"
            f"{DISCLAIMER_MONOCULAR}\n\n"
            "Demo → Reset presentation demo restores the professor walk-through.",
        )

    def _skeleton_display_mode_key(self) -> str:
        label = getattr(self, "skeleton_display_mode", None)
        if label is not None:
            return LABEL_TO_SKELETON_MODE.get(label.get(), DEFAULT_SKELETON_DISPLAY_MODE)
        return DEFAULT_SKELETON_DISPLAY_MODE

    def _on_skeleton_display_mode(self, _event: object = None) -> None:
        # Different display modes project coordinates differently, so recompute
        # the stable view box before redrawing.
        self._skeleton_artists_dirty = True
        self._landmark_overlay_cache = {}
        self._update_skeleton_view_box()
        if self.skeleton_player is not None:
            self._sync_all_to_frame(force_draw=True)
        else:
            self._update_interactive_skeleton(force_draw=True)

    def _on_skeleton_motion(self, event) -> None:
        if not self.skeleton_player or self.skeleton_player.state.playing:
            if self._hover_joint is not None:
                self._hover_joint = None
                self._update_interactive_skeleton(force_draw=True)
            return
        snap = self.skeleton_player.current_snapshot()
        mode = self._skeleton_display_mode_key()
        jid = nearest_joint_from_event(event, snap, display_mode=mode)
        if jid != self._hover_joint:
            self._hover_joint = jid
            self._update_interactive_skeleton(force_draw=True)

    def _active_joint_label(self) -> dict[str, str]:
        """Label dict for the skeleton — mirrors the active analysis point."""
        return self._skeleton_joint_labels()

    def _skeleton_joint_labels(self) -> dict[str, str]:
        """Single label for the active charted body point (presentation view)."""
        from stablewalk.ui.dof_selection import anchor_joint_for_item, label_for_item

        item_id = self._charted_dof_item_id()
        if not item_id:
            return {}
        anchor = anchor_joint_for_item(item_id)
        if not anchor:
            return {}
        return {anchor: label_for_item(item_id)}

    def _selected_asymmetric_moment(self):
        """Currently selected asymmetric moment, or ``None``."""
        report = getattr(self, "_gait_asymmetry_moments", None)
        index = getattr(self, "_selected_asymmetry_moment_index", None)
        if report is None or index is None:
            return None
        moments = list(getattr(report, "moments", None) or [])
        if 0 <= int(index) < len(moments):
            return moments[int(index)]
        return None

    def _select_asymmetric_moment(self, index: int) -> None:
        moments = list(getattr(getattr(self, "_gait_asymmetry_moments", None), "moments", None) or [])
        if not moments:
            self._selected_asymmetry_moment_index = None
            refresh = getattr(self, "_refresh_asymmetric_moments_panel", None)
            if callable(refresh):
                refresh()
            return
        index = int(index)
        if getattr(self, "_selected_asymmetry_moment_index", None) == index:
            self._jump_to_asymmetric_moment(index)
            return
        if 0 <= index < len(moments):
            self._jump_to_asymmetric_moment(index)

    def _jump_to_asymmetric_moment(self, index: int) -> None:
        """Pause playback, seek video + 3D skeleton, and highlight the moment."""
        moments = list(getattr(getattr(self, "_gait_asymmetry_moments", None), "moments", None) or [])
        if not (0 <= int(index) < len(moments)):
            return
        moment = moments[int(index)]
        self._selected_asymmetry_moment_index = int(index)
        refresh = getattr(self, "_refresh_asymmetric_moments_panel", None)
        if callable(refresh):
            refresh()
        self._skeleton_artists_dirty = True
        seek_time = getattr(self, "_seek_motion_workspace_time", None)
        if callable(seek_time):
            seek_time(float(moment.time_s))
            return
        self._go_to(int(moment.frame_index))

    def _snapshot_nearest_time(self, time_s: float):
        rec = getattr(self, "gait_motion", None)
        snaps = list(getattr(rec, "snapshots", None) or [])
        if not snaps:
            return None
        return min(snaps, key=lambda s: abs(float(s.time_s) - float(time_s)))

    def _resolve_highlight_joints(self) -> set[str] | None:
        """Joints drawn bright on the skeleton; others are softly de-emphasized."""
        moment = self._selected_asymmetric_moment()
        if moment is not None:
            from stablewalk.ui.asymmetric_moments_view import moment_highlight_joints

            return moment_highlight_joints(moment, self._skeleton_display_mode_key())
        if not self.highlight_dof.get():
            return None
        from stablewalk.ui.dof_selection import (
            anchor_joint_for_item,
            joints_for_item,
        )

        # Brighten the selected anchor plus its local kinematic chain so the
        # active limb reads clearly (knee → hip/ankle, toe → heel/ankle, …).
        joints: set[str] = set()
        for item_id in self.selection.selected:
            joints.update(joints_for_item(item_id))
            anchor = anchor_joint_for_item(item_id)
            if anchor:
                joints.add(anchor)
        return joints if joints else None

    @staticmethod
    def _set_panel_data_visible(
        tree: ttk.Treeview,
        visible: bool,
        empty_label: ttk.Label,
    ) -> None:
        host = getattr(tree, "_sw_host", None)
        if host is None:
            return
        if visible:
            empty_label.pack_forget()
            host.pack(fill=tk.BOTH, expand=True)
        else:
            host.pack_forget()
            empty_label.pack(fill=tk.BOTH, expand=True)

    def _update_dof_selection_chrome(self) -> None:
        if getattr(self, "lbl_dof_summary", None) is not None:
            self.lbl_dof_summary.configure(text=self.selection.count_label())
        self._update_session_selection_overview()
        self._update_dashboard_empty_states()

    def _update_session_selection_overview(self) -> None:
        """Compact session status: source, frame count, and selected points."""
        if not hasattr(self, "lbl_session_status"):
            return
        parts: list[str] = []
        src = (getattr(self, "_session_display_src", "") or "").strip()
        if src:
            parts.append(self._short_source_name(src))
        n_frames = 0
        if self.skeleton_player and self.skeleton_player.frame_count > 0:
            n_frames = self.skeleton_player.frame_count
        elif self.sequence and self.pose_indices:
            n_frames = len(self.pose_indices)
        if n_frames:
            parts.append(f"{n_frames} frames")
        n_sel = len(self.selection.selected)
        parts.append(f"{n_sel} point{'s' if n_sel != 1 else ''}")
        text = " · ".join(parts) if parts else "No session loaded"
        self.lbl_session_status.configure(text=text)
        self._update_session_header_identity()

    def _update_session_header_identity(self) -> None:
        """Header session identity for thesis / lab presentation (display only)."""
        lbl = getattr(self, "lbl_session_header", None)
        if lbl is None:
            return
        src = (getattr(self, "_session_display_src", "") or "").strip()
        if not src:
            try:
                lbl.configure(text="")
            except tk.TclError:
                pass
            return
        short = self._short_source_name(src)
        backend = ""
        try:
            backend = self._selected_pose_backend()
        except Exception:
            backend = ""
        meta = getattr(self, "_run_metadata", None)
        used = getattr(meta, "pose_backend_used", None) if meta is not None else None
        if used:
            backend = str(used)
        n_frames = 0
        if self.skeleton_player and self.skeleton_player.frame_count > 0:
            n_frames = self.skeleton_player.frame_count
        elif self.sequence and self.pose_indices:
            n_frames = len(self.pose_indices)
        bits = [short]
        if backend:
            bits.append(str(backend))
        if n_frames:
            bits.append(f"{n_frames} fr")
        try:
            lbl.configure(text="  ·  ".join(bits))
        except tk.TclError:
            pass
        try:
            self._update_overview_clinical_summary()
        except Exception:
            pass

    def _analysis_motion_recording(self) -> GaitMotionRecording | None:
        """Recording used for the selected-point 3D trajectory panel."""
        if self.gait_motion is not None:
            return self.gait_motion
        if self.skeleton_player is not None:
            return getattr(self.skeleton_player, "recording", None)
        return None

    def _clear_bilateral_ground_clearance_ui(self) -> None:
        """Reset foot clearance compact strip and detail panel."""
        for attr, text in (
            ("lbl_overview_clearance_overlay_left", "L Foot: —"),
            ("lbl_overview_clearance_overlay_right", "R Foot: —"),
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text=text)
        for attr in ("lbl_ground_clearance_left", "lbl_ground_clearance_right"):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text="\u2014", fg=TEXT)
        for attr in ("lbl_ground_clearance_left_state", "lbl_ground_clearance_right_state"):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                side = "L" if "left" in attr else "R"
                lbl.configure(text=f"{side}: —", fg=MUTED)
        for attr in (
            "lbl_ground_clearance_phase",
            "lbl_ground_clearance_scale",
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text="")
        for prefix in ("foot_left", "foot_right"):
            for suffix in ("current", "state", "unavailable", "min_swing", "max_swing"):
                lbl = getattr(self, f"lbl_{prefix}_{suffix}", None)
                if lbl is None:
                    continue
                if suffix == "unavailable":
                    lbl.configure(text="", fg=MUTED)
                    try:
                        lbl.grid_remove()
                    except tk.TclError:
                        pass
                elif suffix in ("current", "min_swing", "max_swing", "state"):
                    heading = getattr(lbl, "_fc_heading", "")
                    heading = heading.strip() if isinstance(heading, str) else ""
                    lbl.configure(
                        text=f"{heading}: Unavailable" if heading else "Unavailable",
                        fg=MUTED,
                    )
                    try:
                        info = getattr(lbl, "_fc_grid", None)
                        if isinstance(info, dict) and info:
                            lbl.grid(**info)
                        else:
                            lbl.grid()
                    except tk.TclError:
                        pass
        conf = getattr(self, "lbl_foot_clearance_confidence", None)
        if conf is not None:
            conf.configure(text=f"{LABEL_FOOT_CLEARANCE_CONFIDENCE}: —", fg=MUTED)
        self._foot_clearance_dashboard = None
        self._foot_clearance_frame_index = None
        self._foot_clearance_live_cache = None
        self._ground_clearance_prev_phase = (None, None)

    def _apply_foot_clearance_dashboard(self, panel) -> None:
        """Push foot clearance model into the primary Overview panel."""
        from stablewalk.ui.foot_clearance_display import (
            FootClearanceDashboardPanel,
            overview_distance_text,
            overview_state_text,
            overview_unavailable_reason,
            overview_value_text,
        )
        from stablewalk.ui.theme import FONT_UI_SM

        if not isinstance(panel, FootClearanceDashboardPanel):
            return
        self._foot_clearance_dashboard = panel

        conf_lbl = getattr(self, "lbl_foot_clearance_confidence", None)
        value_fg = {
            "HIGH": SUCCESS,
            "MODERATE": WARNING,
        }.get(panel.confidence, MUTED)
        if conf_lbl is not None:
            conf_lbl.configure(text=panel.confidence_label, fg=value_fg)

        def _paint_cell(lbl, text: str, *, ok: bool, value_font: bool = False) -> None:
            if lbl is None:
                return
            shown = overview_value_text(text)
            heading = getattr(lbl, "_fc_heading", "")
            heading = heading.strip() if isinstance(heading, str) else ""
            painted = f"{heading}: {shown}" if heading else shown
            try:
                cfg: dict[str, object] = {
                    "text": painted,
                    "fg": TEXT if ok else MUTED,
                    "wraplength": 0,
                }
                if value_font:
                    cfg["font"] = FONT_UI_SM
                lbl.configure(**cfg)
            except tk.TclError:
                return
            try:
                info = getattr(lbl, "_fc_grid", None)
                if isinstance(info, dict) and info:
                    lbl.grid(**info)
                else:
                    lbl.grid()
                lift = getattr(lbl, "lift", None)
                if callable(lift):
                    lift()
            except tk.TclError:
                pass

        for foot, prefix in ((panel.left, "foot_left"), (panel.right, "foot_right")):
            name_lbl = getattr(self, f"lbl_{prefix}_name", None)
            if name_lbl is not None:
                try:
                    name_lbl.grid()
                except tk.TclError:
                    pass
            dist_lbl = getattr(self, f"lbl_{prefix}_current", None)
            min_lbl = getattr(self, f"lbl_{prefix}_min_swing", None)
            max_lbl = getattr(self, f"lbl_{prefix}_max_swing", None)
            state_lbl = getattr(self, f"lbl_{prefix}_state", None)
            reason_lbl = getattr(self, f"lbl_{prefix}_unavailable", None)
            available = foot.displayed_clearance_cm is not None
            overlay_lbl = getattr(
                self,
                "lbl_overview_clearance_overlay_left"
                if prefix == "foot_left"
                else "lbl_overview_clearance_overlay_right",
                None,
            )
            if overlay_lbl is not None:
                overlay_value = (
                    f"{foot.displayed_clearance_cm:.1f} cm"
                    if foot.displayed_clearance_cm is not None
                    else "—"
                )
                overlay_lbl.configure(
                    text=f"{'L' if prefix == 'foot_left' else 'R'} Foot: {overlay_value}"
                )

            current_text = overview_distance_text(foot)
            min_text = overview_value_text(foot.min_swing_display)
            max_text = overview_value_text(foot.max_swing_display)
            state_text = overview_state_text(foot.state_display)
            _paint_cell(
                dist_lbl,
                current_text,
                ok=available,
                value_font=True,
            )
            _paint_cell(
                min_lbl,
                min_text,
                ok="cm" in min_text.lower(),
                value_font=True,
            )
            _paint_cell(
                max_lbl,
                max_text,
                ok="cm" in max_text.lower(),
                value_font=True,
            )
            _paint_cell(
                state_lbl,
                state_text,
                ok=state_text in ("Contact", "Swing"),
            )
            if reason_lbl is not None:
                extra = ""
                if not available:
                    primary = overview_distance_text(foot)
                    detail = overview_unavailable_reason(foot.unavailable_reason)
                    extra = "" if detail == primary else detail
                try:
                    reason_lbl.configure(text=extra, fg=MUTED)
                    # Keep the extra reason off the metric grid so Current /
                    # Min / Max / Gait state cannot be pushed out of view.
                    reason_lbl.grid_remove()
                except tk.TclError:
                    pass

        from stablewalk.ui.tk.gui_layout_debug import log_foot_clearance_debug_if_enabled

        log_foot_clearance_debug_if_enabled(panel)

    def _show_foot_clearance_details(self) -> None:
        """Open foot clearance methodology and per-foot measurements."""
        panel = getattr(self, "_foot_clearance_dashboard", None)
        if panel is None:
            messagebox.showinfo(
                "Estimated Foot Clearance",
                "No estimated foot clearance data yet.\n\nLoad a video and run analysis first.",
            )
            return
        from stablewalk.ui.foot_clearance_display import format_foot_clearance_details

        existing = getattr(self, "_foot_clearance_dialog", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    existing.focus_force()
                    return
            except tk.TclError:
                pass

        dlg = tk.Toplevel(self.root)
        dlg.title("Estimated Foot Clearance — Details")
        dlg.geometry("520x520")
        dlg.minsize(400, 360)
        dlg.transient(self.root)
        self._foot_clearance_dialog = dlg

        frame = ttk.Frame(dlg, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)
        text = tk.Text(
            frame,
            wrap=tk.WORD,
            bg=PANEL,
            fg=TEXT,
            font=FONT_MONO_SM,
            relief=tk.FLAT,
            padx=8,
            pady=8,
        )
        text.pack(fill=tk.BOTH, expand=True)
        text.insert("1.0", format_foot_clearance_details(panel))
        text.configure(state=tk.DISABLED)

        footer = ttk.Frame(frame)
        footer.pack(fill=tk.X, pady=(8, 0))

        def _close() -> None:
            self._foot_clearance_dialog = None
            try:
                dlg.destroy()
            except tk.TclError:
                pass

        ttk.Button(footer, text="Close", command=_close).pack(side=tk.RIGHT)
        dlg.protocol("WM_DELETE_WINDOW", _close)

    def _update_ground_clearance_visibility(self) -> None:
        """Keep Estimated Foot Clearance on Overview; hide the duplicate strip."""
        detail = getattr(self, "_foot_clearance_detail_host", None)
        strip = getattr(self, "ground_clearance_strip", None)
        if detail is not None:
            try:
                if str(detail.winfo_manager() or "") == "":
                    detail.grid(
                        row=2,
                        column=0,
                        columnspan=3,
                        sticky="nsew",
                        pady=(PAD_SM, 0),
                    )
                else:
                    detail.grid()
            except tk.TclError:
                pass
        if strip is not None:
            try:
                strip.grid_remove()
            except tk.TclError:
                pass

    def _refresh_bilateral_ground_clearance(
        self, snapshot=None, *, current_frame_only: bool = False
    ) -> None:
        """Update Estimated Foot Clearance for the current playback frame.

        ``current_frame_only`` refreshes heel/toe clearance in O(1) using a
        cached ground plane and swing min/max — intended for live playback.
        """
        from stablewalk.ui.foot_clearance_display import (
            build_foot_clearance_dashboard_with_cache,
            foot_clearance_dashboard_current_frame,
        )

        recording = self._analysis_motion_recording()
        player = self.skeleton_player
        if snapshot is None and player is not None:
            snapshot = player.current_snapshot()
        end_f = float(player.state.frame_float) if player is not None else 0.0

        if recording is None:
            self._clear_bilateral_ground_clearance_ui()
            self._update_ground_clearance_visibility()
            return
        if snapshot is None:
            self._update_ground_clearance_visibility()
            return

        prev_l, prev_r = self._ground_clearance_prev_phase
        cache = getattr(self, "_foot_clearance_live_cache", None)
        panel = None
        if (
            current_frame_only
            and cache is not None
            and getattr(cache, "recording_id", None) == id(recording)
        ):
            panel = foot_clearance_dashboard_current_frame(
                snapshot,
                cache,
                prev_left_phase=prev_l,
                prev_right_phase=prev_r,
            )
        if panel is None:
            panel, cache = build_foot_clearance_dashboard_with_cache(
                snapshot,
                recording,
                end_f,
                prev_left_phase=prev_l,
                prev_right_phase=prev_r,
            )
            self._foot_clearance_live_cache = cache
        if panel is None:
            self._clear_bilateral_ground_clearance_ui()
            self._update_ground_clearance_visibility()
            return

        self._ground_clearance_prev_phase = (panel.left_phase, panel.right_phase)
        self._foot_clearance_frame_index = snapshot.frame_index

        self._apply_foot_clearance_dashboard(panel)
        self._update_ground_clearance_visibility()
        if getattr(self, "_pb_heavy", True) or not self.playing:
            self._refresh_foot_clearance_graph_hint()

    def _refresh_foot_clearance_graph_hint(self) -> None:
        """Update optional foot-clearance range readout beside the movement graph."""
        lbl = getattr(self, "lbl_foot_clearance_graph_range", None)
        combo = getattr(self, "var_foot_clearance_graph", None)
        if lbl is None or combo is None:
            return
        choice = combo.get()
        if choice == "Off":
            lbl.configure(text="")
            return
        recording = self._analysis_motion_recording()
        player = self.skeleton_player
        if recording is None or player is None:
            lbl.configure(text="")
            return
        from stablewalk.ui.selected_point_analysis import bilateral_foot_clearance_export_rows

        rows = bilateral_foot_clearance_export_rows(
            recording,
            float(player.state.frame_float),
        )
        if not rows:
            lbl.configure(text="")
            return
        key = (
            "left_foot_ground_cm"
            if choice.startswith("Left")
            else "right_foot_ground_cm"
        )
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            lbl.configure(text="")
            return
        lo, hi = min(values), max(values)
        lbl.configure(
            text=f"Range so far: {lo:.1f}\u2013{hi:.1f} cm (estimated body-scale)"
        )

    def _ground_floor_y_for_skeleton(self) -> float | None:
        """Estimated ground plane height for the 3D skeleton overlay."""
        recording = self._analysis_motion_recording()
        player = self.skeleton_player
        if recording is None or player is None:
            return None
        from stablewalk.analysis.ground_reference import floor_reference_y

        return floor_reference_y(recording, float(player.state.frame_float))

    def _video_foot_clearance_hud_cm(
        self, frame_index: int
    ) -> tuple[float | None, float | None]:
        """Read already-calculated current-frame clearance; never analyze here."""
        panel = getattr(self, "_foot_clearance_dashboard", None)
        if (
            panel is not None
            and getattr(self, "_foot_clearance_frame_index", None) == int(frame_index)
        ):
            return (
                panel.left.displayed_clearance_cm,
                panel.right.displayed_clearance_cm,
            )
        return None, None

    def _selected_overview_foot_sides(self) -> tuple[str, ...]:
        """Left/right foot sides implied by the current DOF selection."""
        from stablewalk.analysis.ground_reference import FOOT_POINT_IDS

        selected = getattr(getattr(self, "selection", None), "selected", None) or set()
        sides: list[str] = []
        for item_id in selected:
            if item_id not in FOOT_POINT_IDS:
                continue
            side = "left" if str(item_id).startswith("left") else "right"
            if side not in sides:
                sides.append(side)
        active = getattr(getattr(self, "selection", None), "active_item_id", None)
        if active in FOOT_POINT_IDS:
            side = "left" if str(active).startswith("left") else "right"
            sides = [side] + [s for s in sides if s != side]
        return tuple(sides)

    def _position_overview_foot_clearance_overlay(self, keypoints=None) -> None:
        """Keep the video HUD hidden — numbers live under the viewer row."""
        overlay = getattr(self, "_overview_foot_clearance_overlay", None)
        if overlay is None:
            return
        try:
            overlay.place_forget()
        except tk.TclError:
            pass

    def _on_video_click_select_subject(self, event) -> None:
        """Re-analyze using the person under the click as the subject seed."""
        if getattr(self, "_processing", False) or getattr(self, "_pipeline_running", False):
            return
        source = str(getattr(self, "_current_source", "") or "")
        if not source:
            return
        mapping = getattr(self, "_video_display_map", None) or {}
        rgb = getattr(self, "_last_video_rgb", None)
        if rgb is None:
            return
        from stablewalk.pose.subject_lock import SubjectSelection
        from stablewalk.pose.video_source import content_cache_key
        from stablewalk.ui.video_display_fit import click_to_normalized_xy

        src_h, src_w = rgb.shape[:2]
        xy = click_to_normalized_xy(
            float(event.x),
            float(event.y),
            label_w=int(mapping.get("label_w") or self.video_label.winfo_width()),
            label_h=int(mapping.get("label_h") or self.video_label.winfo_height()),
            src_w=int(mapping.get("src_w") or src_w),
            src_h=int(mapping.get("src_h") or src_h),
            zoom_mode=str(mapping.get("zoom_mode") or "Fit"),
            crop_box=mapping.get("crop_box"),
            fitted_w=mapping.get("fitted_w"),
            fitted_h=mapping.get("fitted_h"),
        )
        if xy is None:
            return
        self._subject_selection = SubjectSelection(seed_xy=xy, policy="click")
        self._subject_selection_source = content_cache_key(source)
        self._loaded_content_key = ""
        self.status.configure(
            text=f"Selecting subject at ({xy[0]:.2f}, {xy[1]:.2f}) — reprocessing…"
        )
        mode = getattr(self, "_analysis_mode", None)
        self._load_video(
            source=source,
            unique_session=True,
            analysis_mode=mode,
            show_dialog=False,
        )

    def _apply_subject_identity_overlay(self, rgb, frame):
        """Draw locked-subject ROI and pose-status even when detection failed."""
        if rgb is None or frame is None:
            return rgb
        try:
            from stablewalk.pose.skeleton import draw_identity_on_bgr
            import cv2

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = draw_identity_on_bgr(bgr, frame)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return rgb

    def _write_completed_frame_audit(self) -> None:
        """Write frame_audit.csv after analysis completes — never mid-run."""
        if self._analysis_workflow_busy():
            return
        sequence = getattr(self, "sequence", None)
        if sequence is None or not getattr(sequence, "frames", None):
            return
        from stablewalk.io.frame_audit import (
            iter_frame_audit_rows,
            write_frame_audit_csv,
        )
        from stablewalk import config

        config.ensure_output_dirs()
        clip = str(getattr(self, "_active_run_name", None) or "session")
        dest = config.REPORTS_DIR / f"{clip}_frame_audit.csv"
        write_frame_audit_csv(list(iter_frame_audit_rows(sequence, clip=clip)), dest)
        logger.info("Wrote frame audit → %s", dest)

    def _apply_video_foot_clearance_hud(
        self,
        rgb: np.ndarray,
        frame_index: int,
        *,
        keypoints=None,
        ground_y_norm: float | None = None,
    ) -> np.ndarray:
        """Position the Tk clearance HUD and paint selected-foot drop lines."""
        from stablewalk.pose.video_overlay_compose import (
            draw_foot_ground_drop_indicators,
        )

        self._position_overview_foot_clearance_overlay(keypoints)

        sides = self._selected_overview_foot_sides()
        if (
            sides
            and keypoints
            and ground_y_norm is not None
            and float(ground_y_norm) == float(ground_y_norm)
        ):
            left_cm, right_cm = self._video_foot_clearance_hud_cm(frame_index)
            preferred: dict[str, str | None] = {}
            panel = getattr(self, "_foot_clearance_dashboard", None)
            if panel is not None:
                preferred["left"] = getattr(panel.left, "measuring_joint", None)
                preferred["right"] = getattr(panel.right, "measuring_joint", None)
            rgb = draw_foot_ground_drop_indicators(
                rgb,
                keypoints,
                ground_y_norm=float(ground_y_norm),
                sides=sides,
                left_cm=left_cm,
                right_cm=right_cm,
                preferred_joints=preferred,
            )
        return rgb

    def _video_foot_contact_overlay(self, frame_index: int) -> dict[str, bool] | None:
        """Left/right stance flags for video overlays, or None if unknown."""
        contact = self._foot_contact_for_frame(int(frame_index))
        if contact is None:
            return None
        return {"left": bool(contact[0]), "right": bool(contact[1])}

    def _video_estimated_ground_y_norm(self, frame) -> float | None:
        """Map the stance-foot ground plane into normalized image Y, or None."""
        keypoints = getattr(frame, "keypoints", None)
        if not keypoints:
            return None
        from stablewalk.pose.video_overlay_compose import (
            estimate_image_ground_y,
            image_y_from_vertical_correspondence,
        )

        floor_y = self._ground_floor_y_for_skeleton()
        snapshot = None
        player = self.skeleton_player
        if player is not None:
            snapshot = player.current_snapshot()
        contact = self._video_foot_contact_overlay(getattr(frame, "frame_index", 0))
        mapped: list[float] = []
        if (
            floor_y is not None
            and snapshot is not None
            and snapshot.joints
        ):
            by_name = {kp.name: kp for kp in keypoints}
            for side in ("left", "right"):
                if contact is not None and not contact.get(side, False):
                    continue
                hip_kp = by_name.get(f"{side}_hip") or by_name.get("mid_hip")
                foot_kp = (
                    by_name.get(f"{side}_heel")
                    or by_name.get(f"{side}_foot_index")
                    or by_name.get(f"{side}_ankle")
                )
                hip_j = snapshot.joints.get(f"{side}_hip") or snapshot.joints.get(
                    "pelvis"
                )
                foot_j = snapshot.joints.get(f"{side}_heel") or snapshot.joints.get(
                    f"{side}_toe"
                )
                if hip_kp is None or foot_kp is None or hip_j is None or foot_j is None:
                    continue
                img_y = image_y_from_vertical_correspondence(
                    float(hip_kp.y),
                    float(foot_kp.y),
                    float(hip_j.position.y),
                    float(foot_j.position.y),
                    floor_y,
                )
                if img_y is not None:
                    mapped.append(img_y)
        if mapped:
            return float(sum(mapped) / len(mapped))
        return estimate_image_ground_y(keypoints, foot_contact=contact)

    def _fit_dof_traj_canvas(self) -> None:
        """Resize the 3D trajectory figure to its widget so nothing is clipped."""
        if getattr(self, "canvas_dof_traj", None) is None:
            return
        from stablewalk.ui.tk.clip_viewport import sync_clipped_viewport
        from stablewalk.ui.tk.dashboard_layout import _fit_trajectory_figure

        sync_clipped_viewport(
            getattr(self, "_traj_clip_canvas", None),
            getattr(self, "_traj_clip_window_id", None),
        )

        graph_host = (
            getattr(self, "dof_analysis_graph_canvas_host", None)
            or getattr(self, "dof_analysis_graph_inner", None)
            or getattr(self, "dof_analysis_graph_section", None)
        )
        if _fit_trajectory_figure(
            self.canvas_dof_traj,
            self.fig_dof_traj,
            self.ax_dof_traj,
            graph_host=graph_host,
        ):
            from stablewalk.ui.tk.dashboard_layout import _ensure_trajectory_canvas_gridded

            _ensure_trajectory_canvas_gridded(self.canvas_dof_traj)
            self.canvas_dof_traj.draw_idle()
        self._request_dashboard_scroll_sync()

    def _lift_trajectory_canvas(self) -> None:
        """Keep the matplotlib widget above any sibling overlays in the graph host."""
        if getattr(self, "canvas_dof_traj", None) is None:
            return
        from stablewalk.ui.tk.dashboard_layout import _ensure_trajectory_canvas_gridded

        _ensure_trajectory_canvas_gridded(self.canvas_dof_traj)

    def _render_dof_traj_canvas(self, *, force: bool = False) -> None:
        """Fit the trajectory canvas to its host and paint the current figure."""
        if getattr(self, "canvas_dof_traj", None) is None:
            return
        playing = bool(
            self.skeleton_player and self.skeleton_player.state.playing
        )
        if force or not playing or getattr(self, "_traj_layout_dirty", False):
            self._fit_dof_traj_canvas()
            self._traj_layout_dirty = False
        self._lift_trajectory_canvas()
        from stablewalk.ui.tk.dashboard_notebook import is_trajectory_graph_visible

        if force or not playing:
            self._paint_canvas_now(self.canvas_dof_traj)
        elif is_trajectory_graph_visible(self):
            self._paint_canvas_now(self.canvas_dof_traj)

    def _schedule_dof_traj_reflow(self) -> None:
        """Reflow the trajectory canvas after Tk finishes mapping the graph area."""
        if getattr(self, "canvas_dof_traj", None) is None:
            return

        # During playback the canvas is already repainted every tick via
        # draw_idle. The multi-stage reflow below issues five *forced* full
        # redraws of the 3D figure; firing that storm on every frame stalls
        # the UI and makes playback feel sluggish. The reflow is only needed
        # once the layout settles (selection change / resize), so skip it
        # while frames are advancing.
        if self.skeleton_player and self.skeleton_player.state.playing:
            return

        def _reflow() -> None:
            has_session = bool(
                self.skeleton_player
                and getattr(self.skeleton_player, "frame_count", 0) > 0
            )
            if not has_session and not self.selection.selected:
                return
            self._render_dof_traj_canvas(force=True)

        from stablewalk.ui.tk.timer_registry import schedule_unique

        # One coalesced reflow — stacking five after() jobs leaked timers on resize.
        schedule_unique(self, "_dof_traj_reflow_after", 120, _reflow)

    def _fit_skeleton_canvas(self, *, force: bool = False) -> None:
        """Resize the skeleton figure and refit the body viewport after layout."""
        if not hasattr(self, "canvas_3d"):
            return
        from stablewalk.ui.tk.dashboard_layout import _SKELETON_CANVAS_PAD
        from stablewalk.ui.tk.clip_viewport import sync_clipped_viewport

        pad_l, pad_t, pad_r, pad_b = _SKELETON_CANVAS_PAD
        widget = self.canvas_3d.get_tk_widget()
        host = getattr(self, "skel_canvas_host", None)
        clip = getattr(self, "_skel_clip_canvas", None)
        width = 0
        height = 0
        # Size to the visible pane, never the figure's natural 640x560 HWND.
        # Falling back to widget.winfo_* locked the overflow that clipped
        # the skeleton under the gait-phase strip.
        measured: list[tuple[int, int]] = []
        for target in (clip, host):
            if target is None:
                continue
            try:
                target.update_idletasks()
                tw = int(target.winfo_width() or 0)
                th = int(target.winfo_height() or 0)
            except tk.TclError:
                continue
            if tw >= 80 and th >= 80:
                measured.append((tw, th))
        if measured:
            # Prefer the larger allocation. A clip canvas that still reports
            # its request height must not keep the figure at 192 px.
            width = max(item[0] for item in measured)
            height = max(item[1] for item in measured)
            if clip is not None:
                try:
                    clip.configure(width=int(width), height=int(height))
                except tk.TclError:
                    pass
        width = max(0, width - pad_l - pad_r)
        height = max(0, height - pad_t - pad_b)
        if width < 80 or height < 80:
            return

        sig = (int(width), int(height))
        if not force and getattr(self, "_skel_fit_size_sig", None) == sig:
            return
        self._skel_fit_size_sig = sig

        class _ResizeEvent:
            __slots__ = ("width", "height")

            def __init__(self, w: int, h: int) -> None:
                self.width = int(w)
                self.height = int(h)

        try:
            self.canvas_3d.resize(_ResizeEvent(width, height))
        except (TypeError, AttributeError, tk.TclError):
            dpi = self.fig_3d.get_dpi()
            self.fig_3d.set_size_inches(width / dpi, height / dpi, forward=True)
        try:
            widget.configure(width=int(width), height=int(height))
        except tk.TclError:
            pass
        sync_clipped_viewport(
            clip, getattr(self, "_skel_clip_window_id", None)
        )
        # Invalidate cached limits so fill is recomputed for the new panel size.
        for attr in ("_sw_view_cache", "_sw_view_scale_cache"):
            if hasattr(self.ax_3d, attr):
                try:
                    delattr(self.ax_3d, attr)
                except Exception:
                    setattr(self.ax_3d, attr, None)
        from stablewalk.ui.viewers.gait_skeleton_renderer import (
            _apply_view_limits,
            _single_frame_view_box,
            relayout_skeleton_viewport,
        )

        relayout_skeleton_viewport(self.ax_3d)
        # Refit scale to the new canvas; user zoom/pan stay on the axes.
        # Skip during play — the renderer already applies limits per frame.
        if self.skeleton_player is not None and not bool(getattr(self, "playing", False)):
            snap = self.skeleton_player.current_snapshot()
            if snap is not None:
                try:
                    # Empty / undetected poses must not replace the fitted view.
                    # Coordinates stay as stored; only the viewport is fitted.
                    if _single_frame_view_box(snap) is not None:
                        _apply_view_limits(self.ax_3d, snap)
                except Exception:
                    pass
        self._paint_canvas_now(self.canvas_3d, idle=True)

    def _schedule_overview_media_refit(self, *, delay_ms: int = 120) -> None:
        """Debounced Overview video + skeleton fit (no widget recreation)."""
        prev = getattr(self, "_overview_media_refit_after", None)
        if prev is not None:
            try:
                self.root.after_cancel(prev)
            except (tk.TclError, ValueError, TypeError):
                pass
            self._overview_media_refit_after = None

        def _run() -> None:
            self._overview_media_refit_after = None
            from stablewalk.ui.tk.layout_playback import should_defer_layout

            if should_defer_layout(self):
                return
            try:
                refit_video = getattr(self, "_on_video_label_resize", None)
                if callable(refit_video):
                    refit_video()
            except Exception:
                pass
            try:
                self._fit_skeleton_canvas()
            except Exception:
                pass
            canvas = getattr(self, "canvas_dof_traj_overview", None)
            if (
                canvas is not None
                and getattr(self, "_overview_traj_dock_visible", False)
            ):
                try:
                    from stablewalk.ui.tk.dashboard_layout import (
                        _ensure_trajectory_canvas_gridded,
                        _fit_trajectory_figure,
                    )

                    _ensure_trajectory_canvas_gridded(canvas, overview_dock=True)
                    if _fit_trajectory_figure(
                        canvas,
                        self.fig_dof_traj_overview,
                        self.ax_dof_traj_overview,
                        graph_host=getattr(self, "overview_traj_canvas_host", None),
                    ):
                        if not bool(getattr(self, "playing", False)):
                            from stablewalk.ui.viewers.dof_trajectory_3d import (
                                fit_trajectory_viewport,
                            )

                            fit_trajectory_viewport(
                                self.ax_dof_traj_overview, keep_camera=True
                            )
                        canvas.draw_idle()
                except Exception:
                    pass

        self._overview_media_refit_after = self.root.after(max(40, int(delay_ms)), _run)

    def _clear_dof_graph_chrome(self) -> None:
        chrome = getattr(self, "dof_graph_chrome", None)
        if chrome is not None:
            chrome.grid_remove()
        caption = getattr(self, "lbl_dof_analysis_graph_caption", None)
        if caption is not None:
            caption.configure(text="")
            caption.grid_remove()
        floor_lbl = getattr(self, "lbl_dof_graph_floor_value", None)
        if floor_lbl is not None:
            floor_lbl.configure(text="\u2014")
        floor_note = getattr(self, "lbl_dof_graph_floor_note", None)
        if floor_note is not None:
            floor_note.configure(text="")
        floor_range = getattr(self, "lbl_dof_graph_floor_range", None)
        if floor_range is not None:
            floor_range.configure(text="")

    def _layout_analysis_body(self, *, foot_mode: bool) -> None:
        """Split body: foot card left + graph right, or full-width graph for other points."""
        from stablewalk.ui.tk.dashboard_layout import _ANALYSIS_LEFT_PANEL_WIDTH

        body = getattr(self, "dof_analysis_body", None)
        left = getattr(self, "dof_analysis_left", None)
        graph = getattr(self, "dof_analysis_graph_section", None)
        if body is None or graph is None:
            return
        if foot_mode and left is not None:
            body.columnconfigure(0, weight=0, minsize=_ANALYSIS_LEFT_PANEL_WIDTH)
            body.columnconfigure(1, weight=1, minsize=0)
            left.grid(row=0, column=0, sticky="new", padx=(0, 6))
            graph.grid(row=0, column=1, sticky="nsew", padx=(0, 0))
        else:
            body.columnconfigure(0, weight=1, minsize=0)
            body.columnconfigure(1, weight=0, minsize=0)
            if left is not None:
                left.grid_remove()
            graph.grid(row=0, column=0, columnspan=2, sticky="nsew")

    def _refresh_dof_graph_chrome(
        self,
        item_id: str,
        snapshot,
        analysis,
        *,
        end_frame_float: float = 0.0,
    ) -> None:
        """Show a one-line graph caption; trajectory draws in the canvas below."""
        from stablewalk.ui.selected_point_analysis import (
            floor_distance_parts_for_panel,
            floor_distance_range_parts_for_panel,
            graph_caption_for_panel,
            is_foot_analysis_point,
        )

        chrome = getattr(self, "dof_graph_chrome", None)
        if chrome is not None:
            chrome.grid_remove()

        # Floor distance is shown for non-foot points (hip, knee, …) that have
        # no foot card. Foot points already get a calibration-aware "distance
        # from ground" in their dedicated card, so hide it there to avoid a
        # conflicting second readout.
        floor_card = getattr(self, "dof_graph_floor_card", None)
        floor_lbl = getattr(self, "lbl_dof_graph_floor_value", None)
        if floor_card is not None:
            if item_id and not is_foot_analysis_point(item_id):
                floor_card.grid()
                primary, secondary = floor_distance_parts_for_panel(
                    item_id,
                    snapshot,
                    self._analysis_motion_recording(),
                    end_frame_float,
                )
                if floor_lbl is not None:
                    floor_lbl.configure(text=primary)
                floor_note = getattr(self, "lbl_dof_graph_floor_note", None)
                if floor_note is not None:
                    floor_note.configure(text="")
                floor_range = getattr(self, "lbl_dof_graph_floor_range", None)
                if floor_range is not None:
                    range_primary, range_secondary = floor_distance_range_parts_for_panel(
                        item_id,
                        self._analysis_motion_recording(),
                        end_frame_float,
                    )
                    if range_primary:
                        range_text = (
                            f"{range_primary}  \u00b7  {range_secondary}"
                            if range_secondary
                            else range_primary
                        )
                    else:
                        range_text = ""
                    floor_range.configure(text=range_text)
            else:
                floor_card.grid_remove()

        caption = getattr(self, "lbl_dof_analysis_graph_caption", None)
        if caption is None:
            return
        text = graph_caption_for_panel(item_id) if item_id else ""
        if text:
            caption.configure(text=text)
            caption.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        else:
            caption.configure(text="")
            caption.grid_remove()

    def _refresh_dof_analysis_legend(self, item_id: str | None) -> None:
        """Show or hide the compact graph chrome when a point is selected."""
        if not item_id or not self.selection.selected:
            self._clear_dof_graph_chrome()

    def _draw_dof_traj_idle_placeholder(self) -> None:
        """Keep the 3D trajectory axes visible before motion data is loaded."""
        if getattr(self, "ax_dof_traj", None) is None:
            return
        from stablewalk.ui.viewers.dof_trajectory_3d import (
            _ensure_trajectory_plot_legend,
            relayout_single_dof_viewport,
            setup_single_dof_trajectory_axes,
        )

        self.ax_dof_traj.cla()
        if hasattr(self.ax_dof_traj, "_stablewalk_traj_artists"):
            del self.ax_dof_traj._stablewalk_traj_artists
        if hasattr(self.ax_dof_traj, "_stablewalk_stable_viewport"):
            del self.ax_dof_traj._stablewalk_stable_viewport
        self.ax_dof_traj._stablewalk_plot_legend = None
        setup_single_dof_trajectory_axes(self.ax_dof_traj)
        self.ax_dof_traj.text2D(
            0.5,
            0.5,
            "Load motion data to view the selected joint 3D movement path",
            transform=self.ax_dof_traj.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )
        _ensure_trajectory_plot_legend(self.ax_dof_traj)
        relayout_single_dof_viewport(self.ax_dof_traj)
        title_lbl = getattr(self, "lbl_joint_movement_title", None)
        if title_lbl is not None:
            title_lbl.configure(text="Selected Joint 3D Movement Path")
        self._render_dof_traj_canvas(force=True)

    def _sync_dof_analysis_panel_state(self) -> None:
        """Show joint-path graph whenever gait data is loaded; placeholder when unselected."""
        if getattr(self, "canvas_dof_traj", None) is None:
            return

        has_session = bool(
            self.skeleton_player and getattr(self.skeleton_player, "frame_count", 0) > 0
        )
        has_selection = bool(self.selection.selected)
        header = getattr(self, "dof_analysis_header", None)
        header_host = getattr(self, "traj_panel_header_host", None)
        derived = getattr(self, "dof_analysis_derived_row", None)
        body = getattr(self, "dof_analysis_body", None)
        graph_section = getattr(self, "dof_analysis_graph_section", None)
        panel_empty = getattr(self, "dof_analysis_empty_host", None)

        if panel_empty is not None:
            panel_empty.grid_remove()
        if header_host is not None:
            header_host.grid_remove()
        if header is not None:
            header.grid_remove()
        if derived is not None and not has_session:
            derived.grid_remove()
        if body is not None:
            body.grid(row=0, column=0, sticky="nsew")
        elif graph_section is not None:
            graph_section.grid(row=0, column=0, sticky="nsew", pady=(1, PAD_SM))

        from stablewalk.ui.tk.dashboard_layout import _ensure_trajectory_canvas_gridded, _hide_trajectory_debug_placeholder

        _ensure_trajectory_canvas_gridded(self.canvas_dof_traj)
        self._schedule_dof_traj_reflow()
        if has_session:
            self._refresh_dof_analysis_legend(
                self._charted_dof_item_id() if has_selection else None
            )
            _hide_trajectory_debug_placeholder(self)
        else:
            self._refresh_dof_analysis_legend(None)
            if not getattr(self, "_traj_startup_test_drawn", False):
                from stablewalk.ui.tk.dashboard_layout import _draw_trajectory_startup_test

                _draw_trajectory_startup_test(self)
            else:
                self._draw_dof_traj_idle_placeholder()
        self._update_analysis_export_state()
        self._update_ground_clearance_visibility()
        self._request_dashboard_scroll_sync()

    def _request_dashboard_scroll_sync(self) -> None:
        """Recalculate advanced-tab scroll region after content or plot size changes."""
        if not self._should_sync_dashboard_scroll():
            return
        sync = getattr(self, "_sync_dashboard_scroll", None) or getattr(
            self, "_sync_analysis_scroll", None
        )
        if sync is None:
            return
        after_id = getattr(self, "_scroll_sync_after", None)
        if after_id is not None:
            try:
                self.root.after_cancel(after_id)
            except tk.TclError:
                pass

        def _run() -> None:
            self._scroll_sync_after = None
            if sync is not None and self._should_sync_dashboard_scroll():
                sync()

        self._scroll_sync_after = self.root.after_idle(_run)

    def _should_sync_dashboard_scroll(self) -> bool:
        """Avoid scroll-region churn during playback on non-scrolling tabs."""
        if not getattr(self, "playing", False):
            return True
        from stablewalk.ui.tk.dashboard_notebook import is_advanced_tab_selected

        return is_advanced_tab_selected(self)

    def _update_export_output_status(self) -> None:
        lbl = getattr(self, "lbl_export_output_folder", None) or getattr(
            self, "lbl_export_output_status", None
        )
        if lbl is None:
            return
        from stablewalk import config

        config.ensure_output_dirs()
        out_dir = config.OUTPUT_DIR
        lbl.configure(text=f"Output Folder:\n{self._format_data_path(out_dir)}")
        self._update_export_section_status()

    def _update_export_section_status(self) -> None:
        """Refresh OpenSim SDK/model lines in the export section."""
        sdk_lbl = getattr(self, "lbl_export_opensim_sdk", None)
        model_lbl = getattr(self, "lbl_export_opensim_model", None)
        if sdk_lbl is None and model_lbl is None:
            return
        from stablewalk.opensim_sdk import check_opensim_sdk

        sdk_status = check_opensim_sdk()
        sdk_text = "Installed" if sdk_status.available else "Not Installed"
        if sdk_lbl is not None:
            sdk_lbl.configure(text=f"OpenSim SDK: {sdk_text}")
        model_valid = bool(getattr(self, "_opensim_model_valid", False))
        if model_lbl is not None:
            model_lbl.configure(
                text=f"Model: {'Loaded' if model_valid else 'Not Loaded'}"
            )

    def _update_analysis_export_state(self) -> None:
        """Enable analysis export buttons when motion data is available."""
        from stablewalk.ui.theme import create_tooltip

        busy = self._analysis_workflow_busy()
        recording = self._analysis_motion_recording()
        can_export = bool(
            self.selection.selected
            and recording is not None
            and recording.frame_count > 0
            and not self._presentation_mode
            and not busy
        )
        has_sequence = (
            bool(self.sequence) and not self._presentation_mode and not busy
        )
        has_biomech = (
            self._biomech is not None and not self._presentation_mode and not busy
        )
        has_motion = (
            bool(self.sequence or self._poses_path)
            and not self._presentation_mode
            and not busy
        )
        meta = getattr(self, "_export_button_meta", {}) or {}
        busy_reason = "wait for analysis to finish"

        def _set(attr: str, enabled: bool, missing: str) -> None:
            btn = getattr(self, attr, None)
            if btn is None:
                return
            try:
                btn.configure(state=tk.NORMAL if enabled else tk.DISABLED)
            except tk.TclError:
                return
            tip = meta.get(attr, "")
            if not enabled:
                tip = f"{tip} Missing prerequisite: {missing}".strip()
            if tip:
                create_tooltip(btn, tip)

        for attr in (
            "btn_export_analysis_report",
            "btn_export_pdf_report",
            "btn_export_gait_metrics",
            "btn_export_analysis",
        ):
            enabled = (
                has_sequence or has_biomech
                if attr != "btn_export_analysis"
                else can_export
            )
            _set(
                attr,
                enabled,
                busy_reason
                if busy
                else (
                    "analyze a video first"
                    if attr != "btn_export_analysis"
                    else "select joints and load motion"
                ),
            )
        _set(
            "btn_export_motion_reference",
            has_motion,
            busy_reason if busy else "load or analyze a pose sequence",
        )
        _set(
            "btn_export_amp_reference",
            has_motion,
            busy_reason if busy else "load or analyze a pose sequence",
        )
        _set(
            "btn_real_to_sim_pipeline",
            has_motion,
            busy_reason if busy else "load or analyze a pose sequence",
        )
        _set(
            "btn_opensim_export_data",
            has_motion,
            busy_reason if busy else "load or analyze a pose sequence",
        )
        _set(
            "btn_view_detailed_data",
            has_sequence,
            busy_reason if busy else "analyze a video first",
        )
        _set(
            "btn_save_session",
            self._has_saveable_session_data()
            and not getattr(self, "_session_save_in_progress", False)
            and not busy,
            busy_reason if busy else "analyze a session with saveable data",
        )

    def _sync_dof_traj_graph_visibility(self) -> None:
        """Backward-compatible alias for selection-based panel sync."""
        self._sync_dof_analysis_panel_state()

    def _update_dashboard_empty_states(self) -> None:
        """Toggle empty-state placeholders vs live data panels."""
        has_selection = bool(self.selection.selected)

        if hasattr(self, "step_preview_tree"):
            self._set_panel_data_visible(
                self.step_preview_tree, has_selection, self.lbl_step_empty
            )
        if hasattr(self, "dof_pos_tree"):
            self._set_panel_data_visible(
                self.dof_pos_tree, has_selection, self.lbl_table_empty
            )
        if getattr(self, "canvas_dof_traj", None) is not None:
            self._sync_dof_analysis_panel_state()
            if not has_selection:
                self._clear_selected_point_analysis_summary()

        if hasattr(self, "lbl_details_empty"):
            if has_selection:
                self.lbl_details_empty.pack_forget()
            else:
                if not self.lbl_details_empty.winfo_ismapped():
                    self.lbl_details_empty.pack(anchor=tk.W, pady=(PAD_XS, 0))
                self.lbl_details_empty.configure(
                    text=(
                        "No joints selected — pick from the dropdown or click "
                        "skeleton joints to inspect motion."
                    )
                )

    def _step_config(self) -> StepConfig:
        label = self._refresh_interval_label()
        return StepConfig.from_refresh_label(label)

    def _refresh_interval_label(self) -> str:
        return self.refresh_var.get() if hasattr(self, "refresh_var") else REFRESH_INTERVAL_DEFAULT

    def _apply_refresh_interval(self, choice: str) -> None:
        """Apply the display-step interval. Never changes pose extraction."""
        if choice == "0.25 s":
            self._data_refresh_s = 0.25
        else:
            self._data_refresh_s = 0.5
            if hasattr(self, "refresh_var"):
                self.refresh_var.set("0.5 s")
        # Keep analysis / 3D / video on the original frame clock.
        self._traj_redraw_interval_s = 0.0
        self._ui_sync_interval_s = 0.0
        self._update_refresh_chrome()

    def _update_refresh_chrome(self) -> None:
        """Show the active interval in panel hints where present."""
        if getattr(self, "lbl_dof_table_hint", None) is not None:
            self.lbl_dof_table_hint.configure(text=self._dof_table_hint_text())
        if hasattr(self, "lbl_step_hint"):
            self.lbl_step_hint.configure(text=self._step_preview_hint_text())
        if hasattr(self, "lbl_angle_hint"):
            self.lbl_angle_hint.configure(text=self._angle_analysis_hint_text())

    def _realtime_refresh_due(self, *, force: bool = False) -> bool:
        # Playback sync requires every panel on the same frame — never throttle.
        return True

    def _mark_realtime_refresh(self) -> None:
        now = time.monotonic()
        self._last_realtime_refresh = now
        self._last_dof_table_refresh = now
        self._last_data_refresh = now

    def _refresh_realtime_analysis(
        self,
        *,
        snapshot=None,
        force_draw: bool = False,
    ) -> None:
        """Refresh table, step preview, joint details, and angle panel."""
        if self.playing and not force_draw:
            return
        self._mark_realtime_refresh()
        # Step previews feed motion arrows (pause only) and the step table.
        # Skip the O(selected) kinematics pass while playing.
        if force_draw or not self.playing:
            self._dof_step_previews = self._compute_dof_step_previews()
            self._refresh_dof_step_panel()
            self._refresh_angle_analysis_if_present()
        if snapshot is None and self.skeleton_player:
            snapshot = self.skeleton_player.current_snapshot()
        # Detail cards destroy/rebuild Tk children — too expensive every heavy tick.
        if force_draw or not self.playing:
            self._refresh_dof_details(snapshot=snapshot)
        self._refresh_dof_position_table()
        self._update_refresh_chrome()

    def _refresh_angle_analysis_if_present(self) -> None:
        """Refresh Joint Angle Analysis when that panel is wired in."""
        if not hasattr(self, "angle_analysis_tree"):
            return
        compute = getattr(self, "_compute_dof_angle_analysis", None)
        refresh = getattr(self, "_refresh_dof_angle_panel", None)
        if not callable(compute) or not callable(refresh):
            return
        self._dof_angle_entries = compute()
        refresh()

    def _angle_analysis_hint_text(self) -> str:
        return (
            "Updates every frame with video playback · "
            "biomechanical flexion angles (OpenSim-style, degrees)"
        )

    def _step_preview_hint_text(self) -> str:
        return (
            "Display look-ahead only · current vs next position. "
            "Gait analysis uses original frame timestamps."
        )

    def _compute_dof_step_previews(self) -> list:
        if not self.skeleton_player or not self.gait_motion:
            return []
        return compute_dof_step_previews(
            self.gait_motion,
            self.selection.selected,
            self.skeleton_player.state.frame_float,
            config=self._step_config(),
            smooth=self.smooth_motion.get(),
        )

    def _motion_arrows_from_previews(self) -> dict[str, tuple] | None:
        if not self._dof_step_previews:
            return None
        return {
            preview.joint_id: (preview.current, preview.next)
            for preview in self._dof_step_previews
        }

    def _fill_step_preview_table(self, rows: list[tuple[str, ...]]) -> None:
        if not hasattr(self, "step_preview_tree"):
            return
        children = self.step_preview_tree.get_children()
        for index, row in enumerate(rows):
            tag = "even" if index % 2 == 0 else "odd"
            if index < len(children):
                self.step_preview_tree.item(children[index], values=row, tags=(tag,))
            else:
                self.step_preview_tree.insert("", tk.END, values=row, tags=(tag,))
        for iid in children[len(rows) :]:
            self.step_preview_tree.delete(iid)
        if not self.playing:
            self.step_preview_tree.update_idletasks()

    def _refresh_dof_step_panel(self) -> None:
        if not hasattr(self, "step_preview_tree"):
            return
        self.lbl_step_hint.configure(text=self._step_preview_hint_text())
        if not self.selection.selected:
            self._fill_step_preview_table([])
            self._update_dashboard_empty_states()
            return
        self._fill_step_preview_table(preview_table_rows(self._dof_step_previews))
        self._update_dashboard_empty_states()

    def _sync_dof_checkboxes(self) -> None:
        if not hasattr(self, "_dof_checkbox_vars"):
            return
        self._dof_list_syncing = True
        active = self.selection.active_item_id
        try:
            for item_id, var in self._dof_checkbox_vars.items():
                selected = item_id in self.selection.selected
                var.set(selected)
                row = self._dof_checkbox_rows.get(item_id)
                name_lbl = getattr(self, "_dof_checkbox_name_labels", {}).get(item_id)
                if row is None:
                    continue
                if selected and item_id == active:
                    bg, fg = PANEL_HOVER, ORANGE
                    name_font = FONT_METRIC
                elif selected:
                    bg, fg = PANEL_HOVER, ACCENT
                    name_font = FONT_METRIC
                else:
                    bg, fg = ELEVATED, TEXT
                    name_font = FONT_UI_SM
                row.configure(bg=bg)
                for child in row.winfo_children():
                    if isinstance(child, tk.Checkbutton):
                        child.configure(bg=bg, activebackground=bg, fg=fg, activeforeground=fg)
                if name_lbl is not None:
                    name_lbl.configure(bg=bg, fg=fg, font=name_font)
        finally:
            self._dof_list_syncing = False
        self._refresh_add_point_combo()
        self._refresh_dof_chips()

    def _refresh_dof_chips(self) -> None:
        """Render selected joints as compact removable chips."""
        frame = getattr(self, "dof_chips_frame", None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        active = self.selection.active_item_id
        selected_ids = [i for i in GUI_DOF_ITEM_IDS if i in self.selection.selected]
        if not selected_ids:
            tk.Label(
                frame,
                text="None — add a joint here or Ctrl+click the skeleton",
                bg=PANEL,
                fg=MUTED,
                font=FONT_UI_XS,
                anchor="w",
            ).pack(anchor=tk.W)
            return
        for item_id in selected_ids:
            chip = tk.Frame(frame, bg=ELEVATED, highlightthickness=0)
            chip.pack(side=tk.LEFT, padx=(0, 3), pady=1)
            label = GUI_DOF_LABELS.get(item_id, item_id)
            fg = ORANGE if item_id == active else ACCENT
            name_lbl = tk.Label(
                chip,
                text=label,
                bg=ELEVATED,
                fg=fg,
                font=FONT_CAPTION,
                cursor="hand2",
                padx=4,
                pady=1,
            )
            name_lbl.pack(side=tk.LEFT)
            name_lbl.bind("<Button-1>", lambda _e, i=item_id: self._on_dof_item_focus(i))
            remove_btn = tk.Label(
                chip,
                text="\u00d7",
                bg=ELEVATED,
                fg=MUTED,
                font=FONT_UI_SM,
                cursor="hand2",
                padx=3,
            )
            remove_btn.pack(side=tk.LEFT)
            remove_btn.bind(
                "<Button-1>",
                lambda _e, i=item_id: self._remove_dof_chip(i),
            )

    def _remove_dof_chip(self, item_id: str) -> None:
        if item_id not in self.selection.selected:
            return
        if item_id in self._dof_checkbox_vars:
            self._dof_checkbox_vars[item_id].set(False)
        self._on_dof_checkbox_changed(item_id)

    def _toggle_joint_advanced_data(self) -> None:
        host = getattr(self, "dof_analysis_advanced_host", None)
        btn = getattr(self, "btn_toggle_joint_advanced", None)
        if host is None or btn is None:
            return
        visible = not getattr(self, "_joint_advanced_visible", False)
        self._joint_advanced_visible = visible
        if visible:
            host.grid()
            btn.configure(text="Detailed Joint Data \u25b4")
        else:
            host.grid_remove()
            btn.configure(text="Detailed Joint Data \u25be")

    def _toggle_collected_data_table(self) -> None:
        existing = getattr(self, "_collected_data_dialog", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.destroy()
            except tk.TclError:
                pass
            self._collected_data_dialog = None
            self._joint_data_dialog_rows = []
            self._joint_data_dialog_view_rows = []
            self._update_table_summary_label()
            return
        self._open_joint_data_dialog()

    def _open_joint_data_dialog(self) -> None:
        """Per-frame joint table for every available frame–joint record."""
        from stablewalk.ui.joint_data_table import (
            JOINT_DATA_PAGE_SIZE,
            JOINT_DATA_TABLE_COLUMNS,
            JOINT_DATA_TABLE_HEADINGS,
            JOINT_DATA_TABLE_WIDTHS,
            JOINT_FILTER_ALL,
            filter_joint_data_rows,
            joint_filter_options,
            paginate_joint_data_rows,
        )
        from stablewalk.ui.tk.dashboard_layout import _make_data_tree

        existing = getattr(self, "_collected_data_dialog", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    existing.focus_force()
                    return
            except tk.TclError:
                pass

        dlg = tk.Toplevel(self.root)
        dlg.title("Joint Data")
        dlg.geometry("1180x560")
        dlg.minsize(780, 400)
        dlg.transient(self.root)
        self._collected_data_dialog = dlg

        frame = ttk.Frame(dlg, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(frame)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(toolbar, text="Joint").pack(side=tk.LEFT)
        filter_var = tk.StringVar(value=JOINT_FILTER_ALL)
        filter_combo = ttk.Combobox(
            toolbar,
            textvariable=filter_var,
            state="readonly",
            width=22,
        )
        filter_combo.pack(side=tk.LEFT, padx=(6, 0))
        count_lbl = ttk.Label(toolbar, text="")
        count_lbl.pack(side=tk.RIGHT)

        note = ttk.Label(
            frame,
            text=(
                "Confidence is detector landmark visibility (0–1), not joint "
                "presence and not a derived accuracy score. Missing visibility "
                "stays unavailable (—), never 1.00."
            ),
            wraplength=1100,
        )
        note.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        tree_host = ttk.Frame(frame)
        tree_host.grid(row=2, column=0, sticky="nsew")

        popup_tree = _make_data_tree(
            tree_host,
            JOINT_DATA_TABLE_COLUMNS,
            JOINT_DATA_TABLE_HEADINGS,
            col_widths=JOINT_DATA_TABLE_WIDTHS,
            height=16,
            text_cols=frozenset({"joint", "contact_state", "gait_phase"}),
            style="Compact.Treeview",
        )
        self._joint_data_dialog_tree = popup_tree
        rows = self._collect_joint_data_rows(all_available=True)
        self._joint_data_dialog_rows = rows
        self._joint_data_dialog_page = 0
        filter_combo.configure(values=joint_filter_options(rows))

        pager = ttk.Frame(frame)
        pager.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        page_lbl = ttk.Label(pager, text="")
        page_lbl.pack(side=tk.LEFT)

        def _fill() -> None:
            filtered = filter_joint_data_rows(
                self._joint_data_dialog_rows, filter_var.get()
            )
            self._joint_data_dialog_view_rows = list(filtered)
            page_rows, page, page_count = paginate_joint_data_rows(
                filtered,
                int(getattr(self, "_joint_data_dialog_page", 0) or 0),
                page_size=JOINT_DATA_PAGE_SIZE,
            )
            self._joint_data_dialog_page = page
            popup_tree.delete(*popup_tree.get_children())
            start = page * JOINT_DATA_PAGE_SIZE
            for i, row in enumerate(page_rows):
                tag = "even" if i % 2 == 0 else "odd"
                if row.is_low_confidence():
                    tag = "low_confidence"
                popup_tree.insert("", tk.END, values=row.as_display_tuple(), tags=(tag,))
            joints = sorted({row.item_id for row in filtered if row.item_id})
            frames = {row.frame for row in filtered}
            shown = f"{start + 1}–{start + len(page_rows)}" if page_rows else "0–0"
            count_lbl.configure(
                text=(
                    f"{len(filtered)} row{'s' if len(filtered) != 1 else ''} · "
                    f"{len(frames)} frame{'s' if len(frames) != 1 else ''} × "
                    f"{len(joints)} joint{'s' if len(joints) != 1 else ''}"
                )
            )
            page_lbl.configure(
                text=f"Showing {shown} · page {page + 1}/{page_count}"
            )
            prev_btn.configure(state=tk.NORMAL if page > 0 else tk.DISABLED)
            next_btn.configure(
                state=tk.NORMAL if page + 1 < page_count else tk.DISABLED
            )

        def _apply_filter(_event: object | None = None) -> None:
            self._joint_data_dialog_page = 0
            _fill()

        def _page(delta: int) -> None:
            self._joint_data_dialog_page = int(
                getattr(self, "_joint_data_dialog_page", 0) or 0
            ) + int(delta)
            _fill()

        prev_btn = ttk.Button(pager, text="Previous", command=lambda: _page(-1))
        next_btn = ttk.Button(pager, text="Next", command=lambda: _page(1))
        next_btn.pack(side=tk.RIGHT)
        prev_btn.pack(side=tk.RIGHT, padx=(0, 6))
        filter_combo.bind("<<ComboboxSelected>>", _apply_filter)
        _fill()

        def _close() -> None:
            self._collected_data_dialog = None
            self._joint_data_dialog_tree = None
            self._joint_data_dialog_rows = []
            self._joint_data_dialog_view_rows = []
            self._update_table_summary_label()
            try:
                dlg.destroy()
            except tk.TclError:
                pass

        dlg.protocol("WM_DELETE_WINDOW", _close)
        footer = ttk.Frame(frame)
        footer.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(footer, text="Close", command=_close).pack(side=tk.RIGHT)
        ttk.Button(
            footer,
            text="Export JSON",
            command=lambda: self._export_joint_data(fmt="json", from_dialog=True),
        ).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Button(
            footer,
            text="Export CSV",
            command=lambda: self._export_joint_data(fmt="csv", from_dialog=True),
        ).pack(side=tk.RIGHT, padx=(0, 8))

    def _open_collected_data_dialog(self) -> None:
        existing = getattr(self, "_collected_data_dialog", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    existing.focus_force()
                    return
            except tk.TclError:
                pass
        self._open_joint_data_dialog()

    def _open_coordinate_inspector_window(self) -> None:
        from stablewalk.ui.tk.coordinate_inspector_window import (
            open_coordinate_inspector_window,
        )

        open_coordinate_inspector_window(self)

    def _open_skeleton_model_inspector_window(self) -> None:
        from stablewalk.ui.tk.skeleton_model_inspector_window import (
            open_skeleton_model_inspector_window,
        )

        open_skeleton_model_inspector_window(self)

    def _joint_data_export_item_ids(self) -> set[str]:
        selected = set(self.selection.selected)
        if selected:
            return selected
        active = self._active_dof_item_id()
        return {active} if active else set()

    def _collect_joint_data_rows(self, *, all_available: bool = False):
        from stablewalk.ui.joint_data_table import (
            available_joint_item_ids,
            collect_joint_data_rows,
        )

        recording = self._analysis_motion_recording()
        samples = None
        if recording is None or recording.frame_count <= 0:
            samples = self._collect_session_samples_for_active()
        item_ids = (
            available_joint_item_ids(recording)
            if all_available
            else self._joint_data_export_item_ids()
        )
        return collect_joint_data_rows(
            recording=recording,
            item_ids=item_ids,
            sequence=getattr(self, "sequence", None),
            gait_cycle=getattr(self, "_gait_cycle", None),
            samples=samples,
        )

    def _update_table_summary_label(self) -> None:
        count = len(self._dof_table_history.rows) if self._dof_table_history.rows else 0
        if count == 0 and self._session_collector.sample_count > 0:
            count = self._session_collector.sample_count
        label = f"Samples:\n{count}"
        sample_lbl = getattr(self, "lbl_export_sample_count", None) or getattr(
            self, "lbl_data_sample_count", None
        )
        if sample_lbl is not None:
            sample_lbl.configure(text=label)
        btn = getattr(self, "btn_collected_data", None)
        if btn is not None:
            btn.configure(text=f"Data ({count})" if count else "Data (0)")
        view_btn = getattr(self, "btn_view_detailed_data", None) or getattr(
            self, "btn_view_table_data", None
        )
        if view_btn is not None:
            view_btn.configure(text="View Detailed Data")
        lbl = getattr(self, "lbl_table_summary", None)
        if lbl is None:
            return
        lbl.configure(text=f"Collected Data \u2014 {count} sample{'s' if count != 1 else ''}")

    def _refresh_add_point_combo(self) -> None:
        """Keep 'Add joint' dropdowns showing only not-yet-selected joints."""
        from stablewalk.ui.tk.dashboard_layout import ADD_POINT_PLACEHOLDER

        available = [
            GUI_DOF_LABELS[i] for i in GUI_DOF_ITEM_IDS if i not in self.selection.selected
        ]
        combos = (
            (getattr(self, "add_point_combo", None), getattr(self, "add_point_var", None)),
            (
                getattr(self, "cmb_motion_compare_add", None),
                getattr(self, "var_motion_compare_add", None),
            ),
        )
        for combo, var in combos:
            if combo is None or var is None:
                continue
            try:
                combo.configure(values=available)
                var.set(ADD_POINT_PLACEHOLDER)
                combo.configure(state="readonly" if available else "disabled")
            except tk.TclError:
                pass

    def _add_joint_from_compare_label(self, label: str) -> None:
        """Add ``label`` to the multi-joint compare set without replacing the active joint."""
        from stablewalk.ui.tk.dashboard_layout import ADD_POINT_PLACEHOLDER

        if not label or label == ADD_POINT_PLACEHOLDER:
            return
        item_id = {GUI_DOF_LABELS[i]: i for i in GUI_DOF_ITEM_IDS}.get(label)
        if not item_id or item_id in self.selection.selected:
            self._refresh_add_point_combo()
            return
        self.selection.activate_item(item_id)
        self._notify_dof_selection_changed()

    def _add_point_from_combo(self, _event: object = None) -> None:
        """Add the joint chosen in the Advanced-tab dropdown to the current selection."""
        var = getattr(self, "add_point_var", None)
        if var is None:
            return
        try:
            label = str(var.get() or "")
        except tk.TclError:
            return
        self._add_joint_from_compare_label(label)

    def _add_motion_compare_joint(self, _event: object = None) -> None:
        """Add the joint chosen in Motion Analysis Compare Joints."""
        var = getattr(self, "var_motion_compare_add", None)
        if var is None:
            return
        try:
            label = str(var.get() or "")
        except tk.TclError:
            return
        self._add_joint_from_compare_label(label)

    def _sync_panels_for_skeleton_pick(self) -> None:
        """Fast path after a skeleton joint click during playback.

        Keeps video · skeleton · trajectory tip · inspector on the newly
        selected joint without pausing or blocking the play timer.
        """
        self.selection.ensure_last_selected()
        self._sync_active_joint_from_charted()
        self._sync_dof_checkboxes()
        self._update_dof_selection_chrome()
        self._sync_dof_analysis_panel_state()
        self._traj_draw_cache_key = None
        self._coord_inspector_force_recolor = True
        # Refresh path only when the on-demand dock is already open.
        if getattr(self, "_overview_traj_dock_visible", False):
            try:
                self._refresh_overview_trajectory_dock(force_draw=True)
            except Exception:
                pass
            try:
                # draw_idle via force=False keeps the play timer breathing.
                if hasattr(self, "canvas_dof_traj_overview"):
                    self._paint_canvas_now(self.canvas_dof_traj_overview)
            except Exception:
                try:
                    self._render_overview_traj_canvas(force=False)
                except Exception:
                    pass
        try:
            self._update_interactive_skeleton(force_draw=True)
        except Exception:
            pass
        # Re-compose the current video frame so the selected joint accent updates.
        try:
            from stablewalk.ui.playback_state import read_playback_state

            state = read_playback_state(self)
            if state is not None and self.sequence and self.pose_indices:
                self._show_pose_at(
                    float(state.frame_float), force_draw=True, skeleton_only=False
                )
        except Exception:
            pass
        try:
            self._refresh_overview_trajectory_readout(self._active_dof_item_id())
        except Exception:
            pass
        try:
            self._update_overview_playback_hud()
        except Exception:
            pass
        try:
            from stablewalk.ui.tk.dashboard_status_bar import update_dashboard_status_bar

            update_dashboard_status_bar(self)
        except Exception:
            pass

    def _ensure_playback_continues_after_pick(self, *, was_playing: bool) -> None:
        """Recover if a layout/slider side-effect paused play during a joint click."""
        if not was_playing or self.skeleton_player is None:
            return
        player = self.skeleton_player
        if not self.playing or not bool(getattr(player.state, "playing", False)):
            self.playing = True
            try:
                player.state.playing = True
                player.state.stopped = False
            except Exception:
                pass
            self._sync_play_buttons()
        if self.playing and not getattr(self, "_after_id", None):
            self._schedule_tick()

    def _sync_panels_for_selected_joint(self, *, force_draw: bool = True) -> None:
        """Synchronize all joint-inspection panels for the current selection."""
        self.selection.ensure_last_selected()
        self._sync_active_joint_from_charted()
        self._sync_dof_checkboxes()
        self._update_dof_selection_chrome()
        self._sync_dof_analysis_panel_state()
        self._refresh_realtime_analysis(force_draw=force_draw)
        self._traj_draw_cache_key = None
        self._coord_inspector_force_recolor = True
        self._refresh_selected_dof_trajectory_3d(force_draw=force_draw)
        self._render_dof_traj_canvas(force=force_draw)
        if getattr(self, "_overview_traj_dock_visible", False):
            self._refresh_overview_trajectory_dock(force_draw=force_draw)
        self._update_interactive_skeleton(force_draw=force_draw)
        # Keep gait-phase / motion graphs / playheads on the same frame.
        # Skip the full lockstep refresh while playing — the next play tick
        # already syncs every panel, and doing it here stalls the UI.
        if bool(getattr(self, "playing", False)):
            try:
                from stablewalk.ui.playback_state import read_playback_state

                state = read_playback_state(self)
                if state is not None and self.sequence and self.pose_indices:
                    self._show_pose_at(
                        float(state.frame_float), force_draw=True, skeleton_only=False
                    )
            except Exception:
                pass
            try:
                self._refresh_overview_trajectory_readout(self._active_dof_item_id())
            except Exception:
                pass
            try:
                self._update_overview_playback_hud()
            except Exception:
                pass
            try:
                from stablewalk.ui.tk.dashboard_status_bar import update_dashboard_status_bar

                update_dashboard_status_bar(self)
            except Exception:
                pass
            return
        try:
            self._update_gait_cycle_panel()
        except Exception:
            pass
        try:
            if hasattr(self, "_update_contact_gait_chart"):
                self._update_contact_gait_chart()
        except Exception:
            pass
        try:
            if hasattr(self, "_update_chart"):
                self._update_chart()
        except Exception:
            pass
        try:
            if hasattr(self, "_sync_all_to_frame"):
                self._sync_all_to_frame(force_draw=True)
        except Exception:
            pass
        hint = getattr(self, "lbl_dof_table_hint", None)
        if hint is not None:
            try:
                hint.configure(text=self._dof_table_hint_text())
            except tk.TclError:
                pass
        if self.sequence:
            if self._dof_table_history.rows:
                rows = self._filtered_position_table_rows(
                    list(self._dof_table_history.rows)
                )
                emphasize = label_for_item(self._active_dof_item_id() or "")
                self._fill_dof_position_table(
                    rows,
                    emphasize_dof=emphasize or None,
                )
            self._refresh_display()

    def _notify_dof_selection_changed(self, *, lightweight: bool = False) -> None:
        """Refresh every panel that depends on the current DOF selection.

        ``lightweight=True`` (skeleton pick while playing) updates selection chrome,
        skeleton highlight, and the Overview 3D path — without rebuilding every
        analytical chart (which freezes the UI and drops play ticks).
        """
        if not getattr(self, "_session_restoring", False):
            mgr = getattr(self, "_session_mgr", None)
            if mgr is not None:
                mgr.mark_dirty()
        # The video overlay cache includes the selected-joint accent. Force one
        # recompose even when the playhead remains on the same frame.
        self._last_video_list_pos = -1
        self._video_photo_key = None
        # Keep collected table rows when adding/removing DOFs — only reset when empty.
        if not self.selection.selected:
            self._dof_table_history.clear()
            self._session_collector.clear()
        else:
            stale = [
                item_id
                for item_id in self._dof_table_history._last_frame_by_item
                if item_id not in self.selection.selected
            ]
            for item_id in stale:
                del self._dof_table_history._last_frame_by_item[item_id]
        self._update_dof_table_controls_state()
        self._configure_dof_table_columns()
        self._sync_demo_save_button()
        try:
            self._sync_overview_joint_pick_combo()
        except Exception:
            pass
        if lightweight and bool(getattr(self, "playing", False)):
            self._sync_panels_for_skeleton_pick()
        else:
            self._sync_panels_for_selected_joint(force_draw=True)
            if not bool(getattr(self, "playing", False)):
                try:
                    self._fit_overview_viewers(reset_camera=True)
                except Exception:
                    pass
        # During playback, defer hidden analytical chart rebuilds until pause
        # or tab activation; selection chrome and live surfaces are already synced.
        if lightweight and bool(getattr(self, "playing", False)):
            self._mark_analysis_panels_dirty("motion", "joint_motion", "knee")
        else:
            try:
                self.refresh_motion_analysis_tab()
            except Exception:
                pass

    _MAX_DETAIL_CARDS = 6

    def _refresh_dof_details(self, snapshot=None) -> None:
        if not hasattr(self, "dof_details_frame"):
            return
        if snapshot is None and self.skeleton_player:
            snapshot = self.skeleton_player.current_snapshot()

        count = len(self.selection.selected)
        self.lbl_dof_summary.configure(text=self.selection.count_label())

        for widget in self.dof_details_frame.winfo_children():
            widget.destroy()

        if not count or snapshot is None:
            if hasattr(self, "lbl_details_empty"):
                if not self.lbl_details_empty.winfo_ismapped():
                    self.lbl_details_empty.pack(anchor=tk.W, pady=(PAD_XS, 0))
                self.lbl_details_empty.configure(
                    text=(
                        "No points selected — use the checklist or click skeleton "
                        "joints to see angle, position, and velocity here."
                    )
                )
            self._update_dashboard_empty_states()
            return

        if hasattr(self, "lbl_details_empty"):
            self.lbl_details_empty.pack_forget()

        ordered = [i for i in GUI_DOF_ITEM_IDS if i in self.selection.selected]
        preview_by_item = {p.item_id: p for p in self._dof_step_previews}

        for index, item_id in enumerate(ordered[: self._MAX_DETAIL_CARDS]):
            color = TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)]
            self._build_point_card(
                self.dof_details_frame,
                item_id,
                snapshot,
                color,
                preview_by_item.get(item_id),
            )

        hidden = count - self._MAX_DETAIL_CARDS
        if hidden > 0:
            tk.Label(
                self.dof_details_frame,
                text=f"+{hidden} more selected — see the Position Table",
                bg=SURFACE,
                fg=MUTED,
                font=FONT_UI_XS,
                anchor="w",
                justify=tk.LEFT,
                wraplength=SIDEBAR_WIDTH - 28,
            ).pack(anchor=tk.W, fill=tk.X, pady=(PAD_XS, 0))

    def _build_point_card(self, parent, item_id, snapshot, color, preview) -> None:
        """One clean card: color dot, label, remove button, and organized metrics."""
        from stablewalk.ui.dof_position_table import (
            NA_VALUE,
            dof_detail_metrics,
        )
        from stablewalk.ui.dof_step_preview import snapshot_at_frame

        row = table_row_for_item(item_id, snapshot)
        joint_name = JOINT_DISPLAY_NAMES.get(
            anchor_joint_for_item(item_id),
            anchor_joint_for_item(item_id).replace("_", " ").title(),
        )
        pos_x, pos_y, pos_z = row[3], row[4], row[5]

        next_snapshot = None
        if preview is not None and self.gait_motion is not None:
            next_snapshot = snapshot_at_frame(
                self.gait_motion,
                float(preview.next_frame),
                smooth=self.smooth_motion.get(),
            )
        metrics_detail = dof_detail_metrics(
            item_id,
            snapshot,
            next_snapshot=next_snapshot,
        )

        card = tk.Frame(
            parent,
            bg=ELEVATED,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=BORDER,
        )
        card.pack(fill=tk.X, pady=(0, PAD_XS))
        inner = tk.Frame(card, bg=ELEVATED, padx=PAD_SM, pady=PAD_XS)
        inner.pack(fill=tk.X)
        inner.columnconfigure(0, weight=0)
        inner.columnconfigure(1, weight=1)

        # Header: dot + label + remove button
        header = tk.Frame(inner, bg=ELEVATED)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        header.columnconfigure(1, weight=1)
        tk.Label(
            header, text="\u25cf", bg=ELEVATED, fg=color, font=FONT_UI
        ).grid(row=0, column=0, sticky="w", padx=(0, 4))
        tk.Label(
            header,
            text=label_for_item(item_id),
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_UI_SEMIBOLD,
            anchor="w",
        ).grid(row=0, column=1, sticky="ew")
        remove_btn = tk.Button(
            header,
            text="\u2715",
            command=lambda i=item_id: self._remove_selected_point(i),
            bg=ELEVATED,
            fg=MUTED,
            activebackground=ELEVATED,
            activeforeground=WARNING,
            relief=tk.FLAT,
            bd=0,
            font=FONT_UI_SM,
            cursor="hand2",
            padx=4,
            pady=0,
        )
        remove_btn.grid(row=0, column=2, sticky="e")
        create_tooltip(remove_btn, f"Remove {label_for_item(item_id)}")

        tk.Label(
            inner,
            text=f"Joint: {joint_name}",
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 2))

        # Thin divider between the header and the metric rows.
        tk.Frame(inner, bg=BORDER, height=1).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(2, 4)
        )

        next_angle_label = (
            f"Next angle ({preview.step_label})"
            if preview is not None
            else "Next angle"
        )
        metrics: list[tuple[str, str, str]] = [
            ("Current angle (est., °)", metrics_detail.angle, ACCENT),
            (next_angle_label, metrics_detail.next_angle, TEXT_SECONDARY),
            ("Δ angle (°)", metrics_detail.delta_angle, ACCENT_ALT),
            ("Velocity (°/s or m/s)", metrics_detail.velocity, TEXT_SECONDARY),
            ("X (m, root-rel.)", pos_x if pos_x != "—" else NA_VALUE, TEXT_SECONDARY),
            ("Y (m, root-rel.)", pos_y if pos_y != "—" else NA_VALUE, TEXT_SECONDARY),
            ("Z (m, root-rel.)", pos_z if pos_z != "—" else NA_VALUE, TEXT_SECONDARY),
        ]
        if preview is not None:
            metrics.append(
                (
                    f"Next position ({preview.step_label})",
                    f"{preview.direction_label()} · {preview.distance_m:.4f} m",
                    ACCENT_ALT,
                )
            )

        for r, (key, value, value_color) in enumerate(metrics, start=3):
            tk.Label(
                inner,
                text=key,
                bg=ELEVATED,
                fg=MUTED,
                font=FONT_UI_XS,
                anchor="w",
            ).grid(row=r, column=0, sticky="w", pady=2, padx=(0, PAD_SM))
            tk.Label(
                inner,
                text=value,
                bg=ELEVATED,
                fg=value_color,
                font=FONT_MONO,
                anchor="e",
                justify=tk.RIGHT,
            ).grid(row=r, column=1, sticky="e", pady=2)

    def _remove_selected_point(self, item_id: str) -> None:
        """Remove a single selected point from everywhere (cards, table, graph, skeleton)."""
        if item_id not in self.selection.selected:
            return
        self.selection.selected.discard(item_id)
        if self.selection.last_selected == item_id:
            self.selection.last_selected = next(
                (i for i in GUI_DOF_ITEM_IDS if i in self.selection.selected),
                None,
            )
        self._notify_dof_selection_changed(
            lightweight=bool(getattr(self, "playing", False))
        )

    def _sync_dof_table_mode_flag(self) -> None:
        mode = getattr(self, "dof_table_display_mode", None)
        if mode is None:
            self._dof_table_track_history = True
            self._dof_table_user_prefers_current_only = False
            return
        label = mode.get().strip()
        self._dof_table_user_prefers_current_only = label == DOF_TABLE_MODE_CURRENT
        self._dof_table_track_history = not self._dof_table_user_prefers_current_only

    def _is_dof_table_history_mode(self) -> bool:
        return not getattr(self, "_dof_table_user_prefers_current_only", False)

    def _should_record_dof_table_history(self) -> bool:
        """Append rows during playback unless the user chose Current Frame Only."""
        return bool(self.selection.selected) and self._is_dof_table_history_mode()

    def _ensure_dof_table_tracking_on_play(self) -> None:
        """Default to history tracking when playback starts with a selection."""
        if not self.selection.selected:
            return
        if getattr(self, "_dof_table_user_prefers_current_only", False):
            return
        if hasattr(self, "dof_table_display_mode"):
            self.dof_table_display_mode.set(DOF_TABLE_MODE_HISTORY)
        self._dof_table_track_history = True
        if getattr(self, "lbl_dof_table_hint", None) is not None:
            self.lbl_dof_table_hint.configure(text=self._dof_table_hint_text())

    def _discrete_playback_snapshot(self):
        """Integer-frame snapshot for history rows (avoids interpolated duplicates)."""
        if not self.skeleton_player:
            return None
        return self.skeleton_player.recording.snapshot_at(
            self.skeleton_player.state.frame_index
        )

    def _append_dof_table_history_tick(self) -> bool:
        """Record the current discrete frame for each selected DOF."""
        if not self.skeleton_player or not self.playing:
            return False
        if not self._should_record_dof_table_history():
            return False
        snap = self._discrete_playback_snapshot()
        if not snap:
            return False
        next_snap = snapshot_for_next_frame(self.skeleton_player.recording, snap)
        added = self._dof_table_history.append_tick(
            snap,
            self.selection.selected,
            next_snapshot=next_snap,
            recording=self.skeleton_player.recording,
        )
        self._session_collector.append_tick(
            snap,
            self.selection.selected,
            next_snapshot=next_snap,
        )
        if added:
            # Record in memory every tick; never rebuild Treeview while playing.
            self._dof_table_display_dirty = True
            if not self.playing:
                rows = self._filtered_position_table_rows(
                    list(self._dof_table_history.rows)
                )
                emphasize = label_for_item(self._active_dof_item_id() or "")
                self._fill_dof_position_table(
                    rows,
                    scroll_to_bottom=True,
                    emphasize_dof=emphasize or None,
                )
                self._dof_table_display_dirty = False
            # Trajectory is owned by `_sync_all_to_frame` / heavy path — avoid a
            # second 3D redraw on every history tick during playback.
            self._update_dof_table_controls_state()
        return added

    def _append_bilateral_foot_tick(self) -> None:
        """Record bilateral foot ground distance — paused/scrub only.

        Continuous playback must stay display-only; ground-plane sampling is
        deferred until pause so the tick path never re-runs geometry work.
        """
        if self.playing or not self.skeleton_player:
            return
        snap = self._discrete_playback_snapshot()
        if not snap:
            return
        player = self.skeleton_player
        prev_l, prev_r = self._ground_clearance_prev_phase
        _, new_l, new_r = self._bilateral_foot_collector.append_tick(
            snap,
            player.recording,
            float(player.state.frame_float),
            prev_left_phase=prev_l,
            prev_right_phase=prev_r,
        )
        if new_l is not None and new_r is not None:
            self._ground_clearance_prev_phase = (new_l, new_r)

    def _dof_table_hint_text(self) -> str:
        charted = self._charted_dof_item_id()
        if charted:
            return label_for_item(charted)
        return "Active point"

    def _configure_dof_table_columns(self) -> None:
        """Keep joint data table columns visible and headings in sync."""
        from stablewalk.ui.joint_data_table import (
            JOINT_DATA_TABLE_COLUMNS,
            JOINT_DATA_TABLE_HEADINGS,
            JOINT_DATA_TABLE_WIDTHS,
        )

        tree = getattr(self, "dof_pos_tree", None)
        if tree is None:
            return
        try:
            tree.configure(displaycolumns=JOINT_DATA_TABLE_COLUMNS)
        except tk.TclError:
            return
        widths = getattr(tree, "_sw_column_widths", None) or JOINT_DATA_TABLE_WIDTHS
        for col in JOINT_DATA_TABLE_COLUMNS:
            try:
                tree.heading(col, text=JOINT_DATA_TABLE_HEADINGS.get(col, col))
                tree.column(col, width=widths.get(col, 72))
            except tk.TclError:
                pass

    def _update_dof_table_controls_state(self) -> None:
        has_playback_tracking = self._session_collector.sample_count > 0
        has_table_rows = bool(self._dof_table_history.rows)
        has_session_data = self._has_saveable_session_data()
        recording = self._analysis_motion_recording()
        has_recording = bool(
            recording is not None
            and recording.frame_count > 0
            and self.selection.selected
        )
        has_view_data = has_playback_tracking or has_table_rows or has_recording
        clear_state = tk.NORMAL if has_table_rows else tk.DISABLED
        export_state = tk.NORMAL if (has_playback_tracking or has_recording) else tk.DISABLED
        view_state = tk.NORMAL if has_view_data else tk.DISABLED
        inspector_state = (
            tk.NORMAL
            if (
                recording is not None
                and recording.frame_count > 0
            )
            or bool(getattr(self, "skeleton_player", None))
            else tk.DISABLED
        )
        save_state = (
            tk.DISABLED
            if self._session_save_in_progress or not has_session_data
            else tk.NORMAL
        )
        for attr, state in (
            ("btn_clear_dof_history", clear_state),
            ("btn_export_joint_csv", export_state),
            ("btn_export_tracking", export_state),
            ("btn_view_detailed_data", view_state),
            ("btn_view_table_data", view_state),
            ("btn_open_coord_inspector", inspector_state),
            ("btn_save_session", save_state),
        ):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.configure(state=state)
        self._update_analysis_export_state()

    def _has_saveable_session_data(self) -> bool:
        if self._session_collector.sample_count > 0:
            return True
        if not self.selection.selected:
            return False
        player = self.skeleton_player
        if player is None or player.recording.frame_count <= 0:
            return False
        return not self._presentation_mode

    def _resolve_session_video_source(self) -> str:
        if self._current_source:
            return self._current_source
        if self.sequence and self.sequence.source_video:
            return self.sequence.source_video
        if self.gait_motion and self.gait_motion.source:
            return self.gait_motion.source
        return ""

    def _collect_session_samples(self) -> list:
        if self._session_collector.sample_count > 0:
            return list(self._session_collector.samples)
        player = self.skeleton_player
        if player is None or not self.selection.selected:
            return []
        return self._session_collector.samples_from_recording(
            player.recording,
            self.selection.selected,
        )

    def _build_session_bundle_snapshot(self):
        """Collect current GUI state for export or save."""
        from stablewalk.io.session_bundle import SessionBundleSnapshot

        recording = self._analysis_motion_recording()
        if recording is None:
            return None

        player = self.skeleton_player
        frame_index = 0
        frame_float = 0.0
        time_s = 0.0
        if player is not None:
            frame_index = int(player.state.frame_index)
            frame_float = float(player.state.frame_float)
            snap = player.current_snapshot()
            if snap is not None:
                time_s = float(snap.time_s)

        poses_path = str(self._poses_path.resolve()) if self._poses_path else None
        display_mode = None
        if hasattr(self, "dof_table_display_mode"):
            display_mode = self.dof_table_display_mode.get()

        active_id = self._active_dof_item_id()
        from stablewalk.ui.selected_point_analysis import analysis_mode_for_item

        analysis_mode = (
            analysis_mode_for_item(active_id) if active_id else None
        )

        from stablewalk.ui.tk.session_manager import (
            capture_results_snapshot,
            capture_workspace_state,
        )

        current = getattr(self, "_session_current_path", None)
        display_name = Path(current).name if current else None
        return SessionBundleSnapshot(
            video_source=self._resolve_session_video_source(),
            poses_json_path=poses_path,
            fps=float(self.sequence.fps) if self.sequence else (
                float(recording.fps) if recording else None
            ),
            frame_count=recording.frame_count,
            selected_item_ids=set(self.selection.selected),
            last_selected=self.selection.last_selected,
            charted_item_id=active_id,
            active_item_id=active_id,
            analysis_mode=analysis_mode,
            frame_index=frame_index,
            frame_float=frame_float,
            time_s=time_s,
            dof_table_display_mode=display_mode,
            smooth_motion=bool(self.smooth_motion.get()),
            tracking_samples=self._collect_session_samples_for_active(),
            recording=recording,
            workspace=capture_workspace_state(self),
            results=capture_results_snapshot(self),
            display_name=display_name,
        )

    def _collect_session_samples_for_active(self) -> list:
        """Playback tracking samples for the active analysis point only."""
        active = self._active_dof_item_id()
        if not active:
            return []
        label = label_for_item(active)
        samples = list(self._session_collector.samples)
        filtered = [sample for sample in samples if sample.dof_name == label]
        if filtered:
            return filtered
        recording = self._analysis_motion_recording()
        if recording is None:
            return []
        return self._session_collector.samples_from_recording(recording, {active})

    def _export_analysis_data(self) -> None:
        """Export the current analysis as a structured session bundle folder."""
        if not self._require_analysis_complete("Export Analysis Data"):
            return
        if self._presentation_mode:
            messagebox.showinfo(
                "Export Analysis Data",
                "Load and analyze a real video before exporting analysis data.",
            )
            return
        if not self.selection.selected:
            messagebox.showinfo(
                "Export Analysis Data",
                "Select at least one body point to export.",
            )
            return
        if not self._active_dof_item_id():
            messagebox.showinfo(
                "Export Analysis Data",
                "Select an active body point to export.",
            )
            return

        snapshot = self._build_session_bundle_snapshot()
        if snapshot is None or snapshot.recording is None:
            messagebox.showwarning(
                "Export Analysis Data",
                "No motion data available.\n\nLoad or analyze a walking video first.",
            )
            return

        from stablewalk.io.session_bundle import SessionBundleError, export_session_bundle

        config.ensure_output_dirs()
        try:
            bundle_dir = export_session_bundle(snapshot, config.SESSION_EXPORT_DIR)
        except SessionBundleError as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return

        self.status.configure(text=f"Analysis exported → {bundle_dir.name}")
        messagebox.showinfo(
            "Export Analysis Data",
            "Analysis data exported successfully.\n\n"
            f"Folder:\n{bundle_dir.resolve()}\n\n"
            "Files:\n"
            "  • tracking_history.csv\n"
            "  • analysis_summary.json\n"
            "  • selected_point_summary.json\n"
            "  • selected_points.json\n"
            "  • session_metadata.json\n"
            "  • gait_motion.json",
        )
        self._notify_generated_files_changed()

    def _save_session_to_files(self) -> None:
        """Compatibility wrapper — prefer File → Save Session."""
        mgr = getattr(self, "_session_mgr", None)
        if mgr is not None:
            mgr.save_session()
            return
        messagebox.showinfo("Save Session", "Session manager is not available.")

    def _load_session_from_files(self) -> None:
        """Compatibility wrapper — prefer File → Load Session…"""
        mgr = getattr(self, "_session_mgr", None)
        if mgr is not None:
            mgr.load_session()

    def _import_analysis_data(self) -> None:
        """Compatibility wrapper — prefer File → Import…"""
        mgr = getattr(self, "_session_mgr", None)
        if mgr is not None:
            mgr.import_session()

    def _apply_session_bundle_from_path(self, path: str) -> bool:
        """Restore a session bundle exactly (video, results, joints, camera, playback, graphs)."""
        from stablewalk.io.session_bundle import (
            SessionBundleError,
            load_session_bundle,
            selected_ids_from_payload,
            tracking_rows_to_kinematic_samples,
            tracking_rows_to_table_rows,
        )
        from stablewalk.io.session_registry import remember_session
        from stablewalk.ui.tk.session_manager import restore_workspace_state

        try:
            loaded = load_session_bundle(path)
        except SessionBundleError as exc:
            messagebox.showerror("Load failed", str(exc))
            return False

        self._session_restoring = True
        try:
            metadata = loaded.metadata
            playback = metadata.get("playback") or {}
            ui = metadata.get("ui") or {}
            play_frame = int(playback.get("frame_index", 0) or 0)
            play_float = float(playback.get("frame_float", play_frame) or play_frame)

            selected = selected_ids_from_payload(loaded.selected_points)
            if not selected:
                messagebox.showwarning(
                    "Load Session",
                    "No valid selected points found in the saved session.",
                )
                return False

            last_selected = loaded.selected_points.get("last_selected") or ui.get(
                "last_selected"
            )
            charted = (
                loaded.selected_points.get("active_item_id")
                or loaded.selected_points.get("charted_item_id")
                or ui.get("active_item_id")
                or ui.get("charted_item_id")
            )

            video_source = str(metadata.get("video_source") or "")
            if video_source:
                self._current_source = video_source
                self._session_display_src = video_source
                try:
                    self.url_var.set(video_source)
                    self.url_entry.delete(0, tk.END)
                    self.url_entry.insert(0, video_source)
                except Exception:
                    pass

            restored_motion = False
            poses_path = metadata.get("poses_json_path")
            poses_file = Path(poses_path) if poses_path else None

            # Prefer full poses restore (video frames + analysis) when available.
            if poses_file is not None and poses_file.is_file():
                try:
                    ok = self.load_poses(
                        poses_file,
                        fresh=True,
                        expected_source=video_source or None,
                    )
                    restored_motion = bool(ok)
                except Exception as exc:
                    loaded.warnings.append(f"Could not reload poses JSON: {exc}")

            if not restored_motion and loaded.recording is not None:
                self._set_gait_motion(loaded.recording)
                restored_motion = True
                try:
                    self._recompute_results_from_gait_motion()
                except Exception as exc:
                    loaded.warnings.append(f"Could not recompute gait results: {exc}")

            if not restored_motion:
                if not messagebox.askyesno(
                    "Limited restore",
                    "Motion recording could not be restored.\n\n"
                    "Table and summary data will still be loaded, but graphs and video "
                    "require the original video or gait_motion.json.\n\n"
                    "Continue?",
                ):
                    return False

            self._restore_dof_selection(
                selected,
                last_selected=last_selected if last_selected in selected else None,
            )

            samples = tracking_rows_to_kinematic_samples(loaded.tracking_rows)
            self._session_collector.samples = samples
            self._session_collector._last_frame_by_item.clear()
            for row in loaded.tracking_rows:
                item_id = row.get("item_id") or ""
                frame_num = row.get("frame") or "0"
                try:
                    sample_frame = int(float(frame_num)) - 1
                except (TypeError, ValueError):
                    sample_frame = 0
                key = item_id or row.get("selected_point") or ""
                if key:
                    self._session_collector._last_frame_by_item[key] = sample_frame

            table_rows = tracking_rows_to_table_rows(loaded.tracking_rows)
            self._dof_table_history.clear()
            for row in table_rows[-200:]:
                self._dof_table_history.rows.append(row)

            display_mode = ui.get("dof_table_display_mode")
            if display_mode and hasattr(self, "dof_table_display_mode"):
                try:
                    self.dof_table_display_mode.set(display_mode)
                    self._sync_dof_table_mode_flag()
                except tk.TclError:
                    pass

            if ui.get("smooth_motion") is not None:
                self.smooth_motion.set(bool(ui.get("smooth_motion")))

            if charted and charted in selected:
                self.selection.set_active(charted)
            self._sync_active_joint_from_charted()

            # Workspace (cameras / graphs / overlays) then seek playback.
            workspace = loaded.workspace or metadata.get("workspace") or {}
            restore_workspace_state(self, workspace)

            if restored_motion and self.skeleton_player is not None:
                seek = play_float if play_float else float(play_frame)
                try:
                    self.skeleton_player.go_to(seek)
                except Exception:
                    self.skeleton_player.go_to(play_frame)
                frame_i = int(self.skeleton_player.state.frame_index)
                self._playback_pos = float(self.skeleton_player.state.frame_float)
                if hasattr(self, "frame_var"):
                    self.frame_var.set(frame_i)
                play_frame = frame_i
                self._sync_all_to_frame(force_draw=True)

            self._refresh_dof_position_table()
            self._update_dof_table_controls_state()
            self._update_analysis_export_state()

            # Re-apply cameras after redraw so orbit/zoom survive chart rebuilds.
            restore_workspace_state(self, workspace)
            if restored_motion:
                self._sync_all_to_frame(force_draw=True)

            mgr = getattr(self, "_session_mgr", None)
            if mgr is not None:
                mgr.current_path = loaded.bundle_dir
                self._session_current_path = loaded.bundle_dir
                mgr.clear_dirty()
                mgr.refresh_recent_menu()
            remember_session(
                loaded.bundle_dir,
                display_name=str(metadata.get("display_name") or loaded.bundle_dir.name),
                video_source=video_source,
                frame_count=int(metadata.get("frame_count") or 0),
            )

            self.status.configure(text=f"Session loaded ← {loaded.bundle_dir.name}")
            warn_text = ""
            if loaded.warnings:
                warn_text = "\n\nNotes:\n" + "\n".join(
                    f"• {w}" for w in loaded.warnings[:5]
                )
            messagebox.showinfo(
                "Session loaded",
                f"Restored session from:\n{loaded.bundle_dir.resolve()}\n\n"
                f"Selected points: {len(selected)}\n"
                f"Tracking rows: {len(loaded.tracking_rows)}\n"
                f"Frame: {play_frame + 1}"
                f"{warn_text}",
            )
            return True
        finally:
            self._session_restoring = False

    def _recompute_results_from_gait_motion(self) -> None:
        """Rebuild gait/biomechanics results from the restored motion recording."""
        if self.gait_motion is None:
            return

        def _after(exc: Exception | None) -> None:
            if exc is not None:
                logger.warning("Session recompute failed: %s", exc)
                return
            self._paint_analysis_panels_from_cache()
            self._update_analysis_summary_panel()
            self._rebuild_knee_angle_series()
            if self._group_visible("knee") or self._group_visible("motion"):
                self._update_chart()

        self._run_session_analysis_once(
            enrich=False,
            report_stages=False,
            already_have_gait_motion=True,
            blocking=False,
            on_done=_after,
        )

    def _clear_session_analysis_cache(self) -> None:
        """Discard cached analysis (welcome / before a new load)."""
        from stablewalk.ui.events import SESSION_CLEARED

        cache = getattr(self, "_session_analysis", None)
        if cache is not None:
            cache.clear()
        self._stability = None
        self._biomech = None
        self._gait_cycle = None
        self._gait_asymmetry_moments = None
        self._selected_asymmetry_moment_index = None
        self._foot_contact = None
        self._estimated_vgrf = None
        self._gait_features = None
        self._virtual_grf = None
        self._biomech_analysis = None
        self._biomech_frame_cache = None
        self._biomech_com_sequence = None
        self._biomech_com_index = None
        self._biomech_static_sig = None
        self._biomech_chart_data_key = None
        self._contact_chart_data_key = None
        self._biomech_chart_undersized = False
        self._contact_chart_undersized = False
        self._stability_result = None
        self._last_stability_result = None
        self._session_metric_view = None
        events = getattr(self, "events", None)
        if events is not None:
            events.publish(SESSION_CLEARED)
        refresh_moments = getattr(self, "_refresh_asymmetric_moments_panel", None)
        if callable(refresh_moments):
            refresh_moments()

    def _refresh_bound_analysis_panels(self) -> None:
        """Push cached analysis into Overview cards that already exist."""
        from stablewalk.analysis.gait_pipeline_trace import log_stage

        refresh_moments = getattr(self, "_refresh_asymmetric_moments_panel", None)
        if callable(refresh_moments):
            try:
                refresh_moments()
            except Exception:
                logger.debug("asymmetric moments panel refresh failed", exc_info=True)
        try:
            self._update_overview_clinical_summary()
        except Exception:
            logger.debug("overview clinical refresh failed", exc_info=True)
        cache = getattr(self, "_session_analysis", None)
        mom = getattr(self, "_gait_asymmetry_moments", None)
        ba = getattr(self, "_biomech_analysis", None)
        overall = None
        if ba is not None and getattr(ba, "symmetry", None) is not None:
            field = getattr(ba.symmetry, "overall_symmetry_pct", None)
            if field is not None:
                overall = getattr(field, "value", None)
        log_stage(
            "ui",
            "complete" if cache is not None and getattr(cache, "ready", False) else "insufficient",
            moments_bound=mom is not None,
            moments_available=bool(getattr(mom, "available", False)),
            moment_cards=len(getattr(mom, "moments", None) or []),
            symmetry="n/a" if overall is None else f"{float(overall):.0f}",
        )
        refresh_rv = getattr(self, "_refresh_research_validation", None)
        if callable(refresh_rv):
            try:
                refresh_rv()
            except Exception:
                logger.debug("research validation refresh failed", exc_info=True)

    def _apply_session_analysis_cache(self) -> None:
        """Mirror cache fields onto legacy GUI attributes used by display code."""
        cache = self._session_analysis
        self._stability = cache.stability
        self._biomech = cache.biomech_stability
        self._gait_cycle = cache.gait_cycle
        self._gait_asymmetry_moments = getattr(cache, "gait_asymmetry_moments", None)
        self._selected_asymmetry_moment_index = None
        self._foot_contact = cache.foot_contact
        self._estimated_vgrf = cache.estimated_vgrf
        self._gait_features = cache.gait_features
        self._virtual_grf = cache.virtual_grf
        self._biomech_analysis = cache.biomech_analysis
        self._biomech_frame_cache = cache.biomech_frame_cache
        self._biomech_com_sequence = cache.biomech_com_sequence
        self._biomech_com_index = cache.biomech_com_index
        self._pipeline_status_report_cache = None
        sched_s = getattr(self, "_schedule_spectral_analysis", None)
        if callable(sched_s):
            try:
                sched_s(True)
            except Exception:
                logger.debug("spectral schedule after analysis cache failed", exc_info=True)
        sched_f = getattr(self, "_schedule_four_gait_signals", None)
        if callable(sched_f):
            try:
                sched_f(True)
            except Exception:
                logger.debug("four-gait schedule after analysis cache failed", exc_info=True)
        sched_a = getattr(self, "_schedule_asymmetry_analysis", None)
        if callable(sched_a):
            try:
                sched_a(True)
            except Exception:
                logger.debug("asymmetry schedule after analysis cache failed", exc_info=True)
        self._refresh_bound_analysis_panels()
        overview_spec = getattr(self, "_render_overview_signal_spectral", None)
        if callable(overview_spec):
            try:
                overview_spec()
            except Exception:
                logger.debug(
                    "overview signal/spectral refresh after analysis cache failed",
                    exc_info=True,
                )

    def _mark_analysis_panels_dirty(self, *groups: str) -> None:
        """Defer chart paints until the owning tab is visible."""
        dirty = getattr(self, "_analysis_panel_dirty", None)
        if dirty is None:
            self._analysis_panel_dirty = set()
            dirty = self._analysis_panel_dirty
        dirty.update(groups)

    def _flush_dirty_analysis_panels(self, groups: frozenset[str] | set[str] | None = None) -> None:
        """Paint deferred analytical panels for ``groups`` (or all dirty)."""
        from stablewalk.ui.tk.panel_loading import set_panel_loading

        dirty = getattr(self, "_analysis_panel_dirty", None)
        if not dirty:
            return
        targets = set(dirty) if groups is None else (set(dirty) & set(groups))
        if not targets:
            return
        if "biomech" in targets or "contact" in targets:
            if self._group_visible("biomech") or self._group_visible("contact"):
                try:
                    self._update_biomechanics_panel()
                except Exception:
                    pass
                family = str(
                    getattr(self, "_biomech_chart_family", "biomech") or "biomech"
                )
                if family == "contact":
                    try:
                        self._update_contact_gait_chart()
                    except Exception:
                        pass
                else:
                    try:
                        self._update_biomechanics_chart()
                    except Exception:
                        pass
                dirty.discard("biomech")
                dirty.discard("contact")
                set_panel_loading(self, "biomech", False)
        if (
            "motion" in targets
            or "knee" in targets
            or "joint_motion" in targets
        ):
            if (
                self._group_visible("motion")
                or self._group_visible("knee")
                or self._group_visible("joint_motion")
            ):
                try:
                    self._rebuild_knee_angle_series()
                    self._apply_default_knee_chart_mode()
                    self._update_chart()
                except Exception:
                    pass
                try:
                    self.refresh_motion_analysis_tab()
                except Exception:
                    pass
                dirty.discard("motion")
                dirty.discard("knee")
                dirty.discard("joint_motion")
                set_panel_loading(self, "graphs", False)
        if "summary" in targets and self._group_visible("summary"):
            try:
                self._update_analysis_summary_panel()
            except Exception:
                pass
            dirty.discard("summary")
            set_panel_loading(self, "metrics", False)
        if "compare" in targets and self._group_visible("compare"):
            try:
                self._update_compare_chart()
            except Exception:
                pass
            dirty.discard("compare")

    def _paint_analysis_panels_from_cache(self) -> None:
        """Compatibility entry — progressive paints own the staged path."""
        self._paint_metrics_panels()
        self._paint_graphs_panels()
        self._paint_biomech_panels()

    def _run_session_analysis_once(
        self,
        *,
        enrich: bool = False,
        report_stages: bool = False,
        n_detected: int | None = None,
        n_total: int | None = None,
        meta_fps: float | None = None,
        display_src: str | None = None,
        already_have_gait_motion: bool = False,
        blocking: bool = True,
        on_done: Callable[[Exception | None], None] | None = None,
    ) -> None:
        """Compute the full session analysis suite exactly once.

        Results are stored in ``_session_analysis`` and mirrored to ``self._*``
        attributes. Playback must never call this method.

        When ``blocking`` is False the heavy compute runs on a single analysis
        worker thread; ``on_done`` is invoked on the UI thread afterward.
        """
        if self.playing:
            logger.warning(
                "Refusing session analysis during playback — display must use cache"
            )
            if on_done is not None:
                on_done(RuntimeError("analysis refused during playback"))
            return
        worker = getattr(self, "_analysis_worker", None)
        if self._analysis_compute_active or (
            worker is not None and worker.is_alive()
        ):
            logger.warning("Session analysis already in progress — skipping re-entry")
            if on_done is not None:
                on_done(RuntimeError("analysis already in progress"))
            return

        # Ensure no leftover playback / deferred timers run during analysis.
        self._stop_all_gui_timers()
        self._sync_play_buttons()
        # Snapshot Tk checkbox on the UI thread before marking compute active —
        # analysis workers must not call BooleanVar.get().
        self._smooth_motion_cached = self._smooth_motion_enabled()
        self._analysis_compute_active = True
        self._analysis_running = True

        kwargs = dict(
            enrich=enrich,
            report_stages=report_stages,
            n_detected=n_detected,
            n_total=n_total,
            meta_fps=meta_fps,
            display_src=display_src,
            already_have_gait_motion=already_have_gait_motion,
            smooth_motion=self._smooth_motion_cached,
        )

        def _finish(payload: dict | None, error: Exception | None) -> None:
            try:
                if error is not None:
                    logger.exception("Session analysis failed: %s", error)
                elif payload is not None:
                    self._apply_session_analysis_payload(payload)
            finally:
                self._analysis_compute_active = False
                self._analysis_running = False
                self._analysis_worker = None
                # Re-enable unfinished controls once compute finishes (progress may
                # still briefly show "Completed" before hide).
                try:
                    self._sync_analysis_busy_controls()
                except Exception:
                    pass
                try:
                    self._refresh_pipeline_status_panel(force=True)
                except Exception:
                    pass
                if on_done is not None:
                    try:
                        on_done(error)
                    except Exception:
                        logger.exception("analysis on_done callback failed")

        if blocking:
            try:
                payload = self._compute_session_analysis_payload(**kwargs)
                _finish(payload, None)
            except Exception as exc:
                _finish(None, exc)
            return

        def _ui_finish(payload: dict | None, error: Exception | None) -> None:
            root = getattr(self, "root", None)
            if root is None:
                return
            try:
                root.after(0, lambda: _finish(payload, error))
            except (tk.TclError, RuntimeError):
                pass

        def _worker() -> None:
            try:
                payload = self._compute_session_analysis_payload(**kwargs)
                _ui_finish(payload, None)
            except Exception as exc:
                _ui_finish(None, exc)

        self._analysis_worker = threading.Thread(
            target=_worker,
            name="stablewalk-session-analysis",
            daemon=True,
        )
        self._analysis_worker.start()

    def _compute_session_analysis_payload(
        self,
        *,
        enrich: bool = False,
        report_stages: bool = False,
        n_detected: int | None = None,
        n_total: int | None = None,
        meta_fps: float | None = None,
        display_src: str | None = None,
        already_have_gait_motion: bool = False,
        smooth_motion: bool | None = None,
    ) -> dict:
        """Pure-ish analysis compute. Must not touch Tk widgets."""
        if smooth_motion is not None:
            self._smooth_motion_cached = bool(smooth_motion)
        from stablewalk.ui.session_analysis_cache import (
            SessionAnalysisCache,
            build_gait_frame_display,
        )

        cache = SessionAnalysisCache(generation=self._session_analysis.generation + 1)
        gait_recording = None
        motion_series = None
        skeleton_cache = None
        view_limit_3d = None
        rom_cache = None
        walk_simulation = None

        if enrich and self.sequence is not None:
            from stablewalk.pose.enrichment import enrich_pose_sequence

            if sequence_needs_enrichment(self.sequence):
                enrich_pose_sequence(self.sequence)

        if self.sequence is not None:
            from stablewalk.analysis.biomech_stability import analyze_biomech_stability
            from stablewalk.analysis.stability import analyze_stability
            from stablewalk.analysis.gait_pipeline_trace import log_stage

            cache.stability = analyze_stability(self.sequence)
            cache.biomech_stability = analyze_biomech_stability(self.sequence)
            # 3D prep / walk sim mutate caches only (no Tk).
            self._stability = cache.stability
            self._biomech = cache.biomech_stability
            self._prepare_3d_sequence()
            self._build_walk_simulation()
            skeleton_cache = dict(self._skeleton_cache)
            view_limit_3d = self._view_limit_3d
            rom_cache = dict(self._rom_cache)
            walk_simulation = self._walk_simulation
            self._pipeline_status_report_cache = None
            motion_series = MotionSeriesIndex.from_sequence(
                self.sequence, self.pose_indices
            )
            n_detected_frames = sum(
                1 for fr in (self.sequence.frames or ()) if getattr(fr, "detected", False)
            )
            log_stage(
                "pose",
                "complete" if n_detected_frames else "insufficient",
                frames=n_detected_frames,
                total=len(self.sequence.frames or ()),
                fps=getattr(self.sequence, "fps", None),
            )
            log_stage(
                "reconstruction_3d",
                "complete" if skeleton_cache else "insufficient",
                skeleton_frames=len(skeleton_cache),
            )
            if already_have_gait_motion and self.gait_motion is not None:
                gait_recording = None
            else:
                from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion

                gait_recording = pose_sequence_to_gait_motion(self.sequence)

        if report_stages:
            # Status updates must happen on the UI thread — queue a light after.
            try:
                self.root.after(
                    0,
                    lambda: self._report_analyze_stage(
                        "biomechanics",
                        fraction=0.78,
                        frames_done=n_detected,
                        frames_total=n_total,
                        fps=meta_fps or getattr(self.sequence, "fps", None),
                    ),
                )
            except Exception:
                pass

        # Attach recording locally for analyses that read self.gait_motion.
        previous_motion = self.gait_motion
        if gait_recording is not None:
            gait_recording.build_time_series()
            self.gait_motion = gait_recording

        try:
            if self.gait_motion is not None:
                from stablewalk.analysis.biomechanical import run_biomechanical_analysis
                from stablewalk.analysis.estimated_vgrf_analysis import (
                    analyze_estimated_vgrf,
                )
                from stablewalk.analysis.foot_contact_analysis import analyze_foot_contact
                from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
                from stablewalk.analysis.gait_feature_analysis import analyze_gait_features
                from stablewalk.analysis.virtual_grf import estimate_virtual_grf

                from stablewalk.analysis.gait_pipeline_trace import log_stage
                from stablewalk.models.gait_wireframe import (
                    GAIT_WIRE_JOINT_COUNT,
                    build_gait_wireframe,
                )

                try:
                    wire = build_gait_wireframe(self.gait_motion)
                    n_wire = len(getattr(wire, "frames", None) or ())
                    missing = 0
                    if n_wire:
                        missing = GAIT_WIRE_JOINT_COUNT - len(
                            getattr(wire.frames[0], "joints", {}) or {}
                        )
                    if n_wire <= 0:
                        log_stage("wire_16", "insufficient", frames=0)
                    elif missing > 0:
                        log_stage(
                            "wire_16",
                            "insufficient",
                            frames=n_wire,
                            missing_joints=missing,
                        )
                    else:
                        log_stage(
                            "wire_16",
                            "complete",
                            frames=n_wire,
                            joints=GAIT_WIRE_JOINT_COUNT,
                        )
                except Exception as exc:
                    log_stage("wire_16", "failed", error=type(exc).__name__)
                    logger.debug("16-joint wireframe trace failed", exc_info=True)

                cache.gait_cycle = analyze_gait_cycles(self.gait_motion)
                n_cycles = len(getattr(cache.gait_cycle, "cycles", None) or ())
                log_stage(
                    "gait_cycle",
                    "complete" if n_cycles else "insufficient",
                    cycles=n_cycles,
                )
                from stablewalk.analysis.gait_asymmetry_moments import (
                    detect_gait_asymmetry_moments,
                    failed_gait_asymmetry_report,
                )

                try:
                    cache.gait_asymmetry_moments = detect_gait_asymmetry_moments(
                        self.gait_motion, cache.gait_cycle
                    )
                    mom = cache.gait_asymmetry_moments
                    n_mom = len(getattr(mom, "moments", None) or [])
                    outcome = str(getattr(mom, "outcome", "") or "")
                    if mom is None:
                        log_stage("asymmetry_moments", "failed", reason="no report")
                    elif outcome == "analysis_failed":
                        log_stage("asymmetry_moments", "failed", moments=n_mom)
                    elif outcome == "insufficient_data" or not getattr(mom, "available", False):
                        log_stage(
                            "asymmetry_moments",
                            "insufficient",
                            moments=n_mom,
                        )
                    else:
                        log_stage(
                            "asymmetry_moments",
                            "complete",
                            moments=n_mom,
                            outcome=outcome,
                        )
                except Exception as exc:
                    logger.exception("gait-pipeline asymmetry_moments: failed")
                    cache.gait_asymmetry_moments = failed_gait_asymmetry_report(
                        f"Asymmetry-moment detection failed: {type(exc).__name__}."
                    )
                    log_stage("asymmetry_moments", "failed", error=type(exc).__name__)
                log_stage(
                    "autocorrelation",
                    "skipped",
                    reason="dedicated ACF is not on the session-cache path",
                )
                log_stage(
                    "fft_psd",
                    "skipped",
                    reason="dedicated spectrum is scheduled after cache apply",
                )
                cache.foot_contact = analyze_foot_contact(
                    self.gait_motion, cycles=cache.gait_cycle
                )
                if report_stages:
                    try:
                        self.root.after(
                            0,
                            lambda: self._report_analyze_stage(
                                "virtual_grf",
                                fraction=0.85,
                                frames_done=n_detected,
                                frames_total=n_total,
                                fps=meta_fps or getattr(self.sequence, "fps", None),
                            ),
                        )
                    except Exception:
                        pass
                cache.estimated_vgrf = analyze_estimated_vgrf(
                    self.gait_motion,
                    cache.foot_contact,
                    sequence=self.sequence,
                )
                ik_mot = (
                    self._opensim_ik_mot_path()
                    if hasattr(self, "_opensim_ik_mot_path")
                    else None
                )
                cache.gait_features = analyze_gait_features(
                    self.gait_motion,
                    cache.gait_cycle,
                    sequence=self.sequence,
                    ik_mot_path=ik_mot,
                )
                cache.virtual_grf = estimate_virtual_grf(
                    self.gait_motion,
                    cache.gait_cycle,
                    gait_features=cache.gait_features,
                    sequence=self.sequence,
                )
                stability_for_ba = cache.biomech_stability
                if stability_for_ba is None:
                    stability_for_ba = self._biomech
                cache.biomech_analysis = run_biomechanical_analysis(
                    self.gait_motion,
                    self.sequence,
                    cycles=cache.gait_cycle,
                    contact=cache.foot_contact,
                    features=cache.gait_features,
                    stability=stability_for_ba,
                )
                self._biomech_analysis = cache.biomech_analysis
                self._build_biomech_frame_cache()
                cache.biomech_frame_cache = self._biomech_frame_cache
                cache.biomech_com_sequence = self._biomech_com_sequence
                cache.biomech_com_index = self._biomech_com_index
                cache.frame_display = build_gait_frame_display(
                    cache.gait_cycle,
                    format_dashboard_phase=self._format_dashboard_gait_phase,
                )
            else:
                from stablewalk.analysis.gait_pipeline_trace import log_stage

                log_stage("wire_16", "insufficient", reason="no gait recording")
                log_stage("gait_cycle", "insufficient", reason="no gait recording")
                log_stage("asymmetry_moments", "insufficient", reason="no gait recording")
                log_stage("autocorrelation", "skipped", reason="no gait recording")
                log_stage("fft_psd", "skipped", reason="no gait recording")
        finally:
            if gait_recording is not None and previous_motion is not None:
                # Leave gait_recording installed; UI apply will wire the player.
                pass

        cache.ready = True
        return {
            "cache": cache,
            "gait_recording": gait_recording,
            "already_have_gait_motion": already_have_gait_motion,
            "display_src": display_src,
            "motion_series": motion_series,
            "skeleton_cache": skeleton_cache,
            "view_limit_3d": view_limit_3d,
            "rom_cache": rom_cache,
            "walk_simulation": walk_simulation,
        }

    def _apply_session_analysis_payload(self, payload: dict) -> None:
        """Apply analysis results on the UI thread."""
        from stablewalk.ui.events import ANALYSIS_READY

        cache = payload["cache"]
        gait_recording = payload.get("gait_recording")
        already = bool(payload.get("already_have_gait_motion"))
        display_src = payload.get("display_src")

        if payload.get("skeleton_cache") is not None:
            self._skeleton_cache = payload["skeleton_cache"]
        if payload.get("view_limit_3d") is not None:
            self._view_limit_3d = payload["view_limit_3d"]
        if payload.get("rom_cache") is not None:
            self._rom_cache = payload["rom_cache"]
        if payload.get("walk_simulation") is not None:
            self._walk_simulation = payload["walk_simulation"]
        if payload.get("motion_series") is not None:
            self._motion_series = payload["motion_series"]

        if display_src:
            try:
                self.status.configure(text=f"{display_src} — press Play")
            except Exception:
                pass

        if gait_recording is not None and not already:
            self._set_gait_motion(gait_recording)
        elif cache.biomech_analysis is None and self.gait_motion is None:
            self._biomech_static_sig = None

        self._session_analysis = cache
        self._apply_session_analysis_cache()
        self._stability = cache.stability
        self._biomech = cache.biomech_stability
        if cache.stability is not None or cache.biomech_stability is not None:
            try:
                self._apply_default_dof_projection()
                self._update_stability_panel(self._biomech)
            except Exception:
                pass
        try:
            self._sync_gait_cycles_labels()
        except Exception:
            pass
        try:
            self._update_gait_cycle_panel(self._gait_cycle)
        except Exception:
            pass
        try:
            self._refresh_bilateral_ground_clearance()
        except Exception:
            pass
        try:
            self._ensure_metrics_stage_summary()
        except Exception:
            logger.exception("Metrics stage did not run after session analysis")
        try:
            from stablewalk.ui.tk.gui_memory import enforce_rss_soft_budget

            enforce_rss_soft_budget(self)
        except Exception:
            pass
        self.events.publish(ANALYSIS_READY, cache)

    def _publish_frame_changed(self, frame_float: float, frame_index: int) -> None:
        from stablewalk.ui.events import FRAME_CHANGED, FrameChangedEvent
        from stablewalk.ui.playback_state import read_playback_state

        state = read_playback_state(self)
        list_pos = int(frame_index)
        if state is not None:
            list_pos = int(state.current_frame_index)
        source = None
        if state is not None:
            source = state.source_frame_index
        elif self.pose_indices and 0 <= list_pos < len(self.pose_indices):
            source = int(self.pose_indices[list_pos])
        self.events.publish(
            FRAME_CHANGED,
            FrameChangedEvent(
                frame_float=float(state.frame_float if state else frame_float),
                frame_index=list_pos,
                playing=bool(state.playing if state else self.playing),
                list_pos=list_pos,
                source_frame_index=source,
                time_s=float(state.current_time_seconds if state else 0.0),
                total_frames=int(state.total_frames if state else 0),
                fps=float(state.fps if state else 0.0),
            ),
        )

    def _restore_dof_selection(
        self,
        item_ids: set[str],
        *,
        last_selected: str | None = None,
    ) -> None:
        """Restore selection without clearing table history (used when loading sessions)."""
        self.selection.set_selection(item_ids, last_selected=last_selected)
        self._sync_active_joint_from_charted()
        self._sync_dof_checkboxes()
        self._update_dof_selection_chrome()
        self._sync_dof_analysis_panel_state()

    def _save_analysis_session(self) -> None:
        if self._session_save_in_progress:
            return
        if self._presentation_mode:
            messagebox.showinfo(
                "Save Session",
                "Load and analyze a real video before saving a session.",
            )
            return
        if not self.selection.selected:
            messagebox.showinfo(
                "Save Session",
                "Select at least one degree of freedom to save.",
            )
            return

        samples = self._collect_session_samples()
        if not samples:
            messagebox.showinfo(
                "Save Session",
                "No kinematic samples to save.\n\n"
                "Play the video in Tracking History mode to record samples, "
                "or load an analyzed video with selected DOFs.",
            )
            return

        video_source = self._resolve_session_video_source()
        if not video_source:
            messagebox.showwarning(
                "Save Session",
                "No video source is available for this session.",
            )
            return

        fps = None
        if self.sequence is not None:
            fps = float(self.sequence.fps)
        elif self.gait_motion is not None:
            fps = float(self.gait_motion.fps)

        selected = set(self.selection.selected)
        sample_count = len(samples)
        self._session_save_in_progress = True
        self._update_dof_table_controls_state()
        self.status.configure(text=f"Saving session ({sample_count} samples)…")
        logger.info(
            "Saving analysis session: source=%s samples=%d",
            video_source,
            sample_count,
        )

        def worker() -> None:
            error: Exception | None = None
            saved = None
            try:
                saved = self._session_storage.save_session(
                    video_source=video_source,
                    samples=samples,
                    fps=fps,
                    selected_dofs=selected,
                )
            except Exception as exc:
                error = exc
            self.root.after(
                0,
                lambda: self._on_save_analysis_session_done(saved, error),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _on_save_analysis_session_done(
        self,
        saved,
        error: Exception | None,
    ) -> None:
        self._session_save_in_progress = False
        self._update_dof_table_controls_state()
        if error is not None:
            logger.exception("Save session failed")
            messagebox.showerror("Save Session failed", str(error))
            self.status.configure(text="Session save failed")
            return

        db_path = self._session_storage.db_path
        self.status.configure(text=f"Session saved → {db_path.name}")
        messagebox.showinfo(
            "Session saved",
            f"Saved {saved.sample_count} samples.\n\n"
            f"Session ID:\n{saved.session_id}\n\n"
            f"Database:\n{db_path.resolve()}",
        )
        logger.info(
            "Session saved: id=%s samples=%d db=%s",
            saved.session_id,
            saved.sample_count,
            db_path.resolve(),
        )

    def _on_dof_table_display_mode(self, _event=None) -> None:
        self._sync_dof_table_mode_flag()
        if getattr(self, "lbl_dof_table_hint", None) is not None:
            self.lbl_dof_table_hint.configure(text=self._dof_table_hint_text())
        if self.playing and self._should_record_dof_table_history():
            self._append_dof_table_history_tick()
        self._refresh_dof_position_table()

    def _clear_dof_table_history(self) -> None:
        self._dof_table_history.clear()
        self._session_collector.clear()
        self._bilateral_foot_collector.clear()
        self._ground_clearance_prev_phase = (None, None)
        self._refresh_dof_position_table()

    def _collect_playback_tracking_samples(self) -> list:
        """Return kinematic samples recorded during playback (full history, uncapped)."""
        return list(self._session_collector.samples)

    def _export_tracking_data(self) -> None:
        self._export_joint_data(fmt="both")

    def _export_joint_data(
        self,
        *,
        fmt: str = "both",
        from_dialog: bool = False,
    ) -> None:
        """Export per-frame joint rows as CSV, JSON, or both."""
        from pathlib import Path

        from stablewalk.ui.joint_data_table import (
            export_joint_data_bundle,
            write_joint_data_csv,
            write_joint_data_json,
        )

        if from_dialog:
            rows = list(getattr(self, "_joint_data_dialog_view_rows", None) or [])
        else:
            rows = self._collect_joint_data_rows()
        if not rows:
            messagebox.showwarning(
                "Export Joint Data",
                "No joint data to export.\n\n"
                "Load a walking video or pose file and select at least one body point.",
            )
            return

        config.ensure_output_dirs()
        output_dir = Path(config.TRACKING_EXPORT_DIR)
        video_source = self._resolve_session_video_source() or None
        selected_dofs = sorted(
            {row.item_id for row in rows if getattr(row, "item_id", None)}
        ) or sorted(self._joint_data_export_item_ids())
        fps = None
        if self.sequence is not None:
            fps = float(self.sequence.fps)
        elif self.gait_motion is not None:
            fps = float(self.gait_motion.fps)

        try:
            if fmt == "csv":
                from stablewalk.ui.joint_data_table import joint_data_export_stem

                written = {
                    "csv": write_joint_data_csv(
                        rows, output_dir / f"{joint_data_export_stem()}.csv"
                    )
                }
            elif fmt == "json":
                from stablewalk.ui.joint_data_table import joint_data_export_stem

                written = {
                    "json": write_joint_data_json(
                        rows,
                        output_dir / f"{joint_data_export_stem()}.json",
                        video_source=video_source,
                        selected_joints=selected_dofs,
                        fps=fps,
                    )
                }
            else:
                written = export_joint_data_bundle(
                    rows,
                    output_dir,
                    video_source=video_source,
                    selected_joints=selected_dofs,
                    fps=fps,
                    include_json=True,
                )
        except (OSError, ValueError) as exc:
            messagebox.showerror("Export failed", str(exc))
            return

        lines = [f"Exported {len(rows)} joint-data rows.", ""]
        csv_path = written.get("csv")
        json_path = written.get("json")
        if csv_path is not None:
            csv_path = csv_path.resolve()
            lines.extend([f"CSV:\n{csv_path}", ""])
        if json_path is not None:
            json_path = json_path.resolve()
            lines.extend([f"JSON:\n{json_path}"])
        status_name = (csv_path or json_path).name if (csv_path or json_path) else "joint data"
        self.status.configure(text=f"Joint data exported → {status_name}")
        messagebox.showinfo("Export Joint Data", "\n".join(lines).strip())
        logger.info(
            "Joint data exported: rows=%d csv=%s json=%s",
            len(rows),
            csv_path or "",
            json_path or "",
        )

    def _export_analysis_report(self) -> None:
        """Write a human-readable gait analysis report to the reports folder."""
        if not self._require_analysis_complete("Export Analysis Report"):
            return
        if self._presentation_mode:
            messagebox.showinfo(
                "Export Analysis Report",
                "Load and analyze a real video before exporting a report.",
            )
            return
        if not self.sequence and self._biomech is None:
            messagebox.showinfo(
                "Export Analysis Report",
                "Load pose data first (Run Full Analysis).",
            )
            return

        from datetime import datetime, timezone

        config.ensure_output_dirs()
        lines: list[str] = [
            "StableWalk Analysis Report",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
        ]
        if self._resolve_session_video_source():
            lines.extend([f"Video: {self._resolve_session_video_source()}", ""])

        if self.sequence is not None:
            from stablewalk.analysis.report import analyze_gait
            from stablewalk.pose.enrichment import enrich_pose_sequence

            seq = self.sequence
            enrich_pose_sequence(seq)
            report, _sim = analyze_gait(seq)
            lines.append(report.summary_text())

        if self._biomech is not None:
            lines.extend(
                [
                    "",
                    "=== Biomechanical stability ===",
                    self._biomech.contribution_table_text(),
                    "",
                    self._biomech.explanation or "",
                ]
            )
            if self._biomech.gait_summary is not None:
                from stablewalk.analysis.gait_analysis_summary import format_summary_display

                display = format_summary_display(self._biomech.gait_summary)
                lines.extend(["", "=== Gait analysis summary ==="])
                for key, value in display.items():
                    if value:
                        lines.append(f"{key}: {value}")

        summary_cache = getattr(self, "_analysis_summary_cache", None)
        if summary_cache is not None and getattr(summary_cache, "scientific_interpretation", ""):
            lines.extend(
                [
                    "",
                    "=== Scientific Interpretation ===",
                    summary_cache.scientific_interpretation,
                    "",
                    "Note: Describes estimated biomechanical parameters from this recording; "
                    "not a clinical diagnosis.",
                ]
            )
        elif self._biomech_analysis is not None:
            from stablewalk.analysis.analysis_summary import build_analysis_summary
            from stablewalk.analysis.scientific_interpretation import build_scientific_interpretation

            interim = build_analysis_summary(
                source=self._session_display_src or self._current_source or "",
                biomechanical=self._biomech_analysis,
                estimated_vgrf=self._estimated_vgrf,
                contact=self._foot_contact,
                cycles=self._gait_cycle,
            )
            interp = build_scientific_interpretation(
                interim,
                biomechanical=self._biomech_analysis,
            )
            if interp:
                lines.extend(
                    [
                        "",
                        "=== Scientific Interpretation ===",
                        interp,
                        "",
                        "Note: Describes estimated biomechanical parameters from this recording; "
                        "not a clinical diagnosis.",
                    ]
                )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = config.REPORTS_DIR / f"analysis_report_{stamp}.txt"
        try:
            path.write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.status.configure(text=f"Report exported → {path.name}")
        messagebox.showinfo(
            "Export Analysis Report",
            f"Analysis report saved:\n{path.resolve()}",
        )
        self._notify_generated_files_changed()

    def _export_gait_metrics(self) -> None:
        """Export structured gait and stability metrics as JSON."""
        if self._presentation_mode:
            messagebox.showinfo(
                "Export Gait Metrics",
                "Load and analyze a real video before exporting gait metrics.",
            )
            return
        if self._biomech is None and not self.sequence:
            messagebox.showinfo(
                "Export Gait Metrics",
                "No gait metrics available yet. Run analysis first.",
            )
            return

        import json
        from datetime import datetime

        config.ensure_output_dirs()
        payload: dict = {"exported_at": datetime.now().isoformat(timespec="seconds")}
        if self._biomech is not None:
            payload["stability"] = self._biomech.to_dict()
        if self.sequence is not None:
            from stablewalk.analysis.metrics import GaitMetrics
            from stablewalk.analysis.forces import ForceAnalyzer

            grf = ForceAnalyzer().analyze(self.sequence)
            payload["advanced_gait_metrics"] = GaitMetrics().compute(
                self.sequence, grf=grf
            ).to_dict()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = config.TRACKING_EXPORT_DIR / f"gait_metrics_{stamp}.json"
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.status.configure(text=f"Gait metrics exported → {path.name}")
        messagebox.showinfo(
            "Export Gait Metrics",
            f"Gait metrics JSON saved:\n{path.resolve()}",
        )
        self._notify_generated_files_changed()

    def _export_motion_reference(self) -> None:
        """Export canonical motion reference NPZ for retargeting (async)."""
        if self._presentation_mode:
            messagebox.showinfo(
                "Export Motion Reference",
                "Load and analyze a real video before exporting motion reference.",
            )
            return
        sequence = self.sequence
        poses_path = self._poses_path
        if sequence is None and not (poses_path and poses_path.is_file()):
            messagebox.showinfo(
                "Export Motion Reference",
                "Load pose data first (Run Full Analysis).",
            )
            return
        run_name = Path(self._resolve_session_video_source() or "session").stem or "session"
        config.ensure_output_dirs()

        def work():
            if sequence is not None:
                from stablewalk.io.motion_reference_export import (
                    export_motion_reference_from_sequence,
                )

                return export_motion_reference_from_sequence(
                    sequence,
                    config.MOTION_REFERENCE_EXPORT_DIR,
                    run_name=run_name,
                )
            from stablewalk.io.motion_reference_export import (
                export_motion_reference_from_poses,
            )

            return export_motion_reference_from_poses(
                poses_path,
                config.MOTION_REFERENCE_EXPORT_DIR,
            )

        def on_success(result) -> None:
            self.status.configure(text=f"Motion reference → {result.npz_path.name}")
            messagebox.showinfo(
                "Export Motion Reference",
                f"Motion reference exported:\n{result.npz_path.resolve()}",
            )

        self._run_advanced_export_async(
            "Export Motion Reference", work, on_success=on_success
        )

    def _run_real_to_sim_pipeline(self) -> None:
        """Run offline 4-stage Real-to-Sim pipeline (async)."""
        if self._presentation_mode or self.gait_motion is None:
            messagebox.showinfo(
                "Real-to-Sim Pipeline",
                "Load and analyze a real video before running the pipeline.",
            )
            return
        config.ensure_output_dirs()
        gait_motion = self.gait_motion
        sequence = self.sequence
        cycles = self._gait_cycle
        gait_features = self._gait_features
        run_name = Path(self._resolve_session_video_source() or "session").stem or "session"

        def work():
            from stablewalk.real_to_sim.pipeline import run_real_to_sim_pipeline

            report = run_real_to_sim_pipeline(
                gait_motion,
                config.MOTION_REFERENCE_EXPORT_DIR,
                run_name=run_name,
                sequence=sequence,
                cycles=cycles,
            )
            virtual_grf = None
            if cycles is not None:
                from stablewalk.analysis.virtual_grf import estimate_virtual_grf

                virtual_grf = estimate_virtual_grf(
                    gait_motion,
                    cycles,
                    gait_features=gait_features,
                    sequence=sequence,
                )
            return report, virtual_grf

        def on_success(payload) -> None:
            report, virtual_grf = payload
            self._last_rts_report = report.to_dict()
            self._pipeline_status_report_cache = None
            self._virtual_grf = virtual_grf
            self._refresh_physics_force_panel()
            self._refresh_real_to_sim_advanced_panel()
            self._update_analysis_summary_panel()
            self._refresh_pipeline_status_panel(force=True)
            stages = "\n".join(
                f"• {s.stage}: {s.status} — {s.detail[:80]}" for s in report.stages
            )
            self.status.configure(text=f"Real-to-Sim → {report.run_name}")
            messagebox.showinfo(
                "Real-to-Sim Pipeline",
                f"Pipeline complete.\n\n{stages}\n\nReport:\n{report.report_path}",
            )

        self._run_advanced_export_async(
            "Real-to-Sim Pipeline", work, on_success=on_success
        )

    def _export_amp_reference(self) -> None:
        """Export AMP reference clip for Isaac Lab (async)."""
        if self._presentation_mode or self.gait_motion is None or self._gait_cycle is None:
            messagebox.showinfo(
                "Export AMP Reference",
                "Load and analyze a video first.",
            )
            return
        config.ensure_output_dirs()
        gait_motion = self.gait_motion
        gait_cycle = self._gait_cycle
        gait_features = self._gait_features
        run_name = Path(self._resolve_session_video_source() or "session").stem or "session"

        def work():
            from stablewalk.io.motion_reference_export import export_motion_reference_npz
            from stablewalk.real_to_sim.amp_reference_export import export_amp_reference
            from stablewalk.real_to_sim.gait_style_extraction import (
                extract_gait_style_fingerprint,
            )
            from stablewalk.real_to_sim.motion_reference_loader import load_motion_reference
            from stablewalk.real_to_sim.retargeting import (
                load_retarget_config,
                retarget_motion_reference,
            )

            run_dir = config.MOTION_REFERENCE_EXPORT_DIR / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            motion_path = run_dir / "stablewalk_motion.npz"
            gait_style = extract_gait_style_fingerprint(
                gait_motion, gait_cycle, gait_features=gait_features
            )
            export_motion_reference_npz(
                gait_motion,
                gait_cycle,
                motion_path,
                gait_style=gait_style,
            )
            motion = load_motion_reference(motion_path)
            retargeted = retarget_motion_reference(motion, load_retarget_config())
            return export_amp_reference(
                motion,
                retargeted,
                config.MOTION_REFERENCE_EXPORT_DIR,
                run_name=run_name,
                gait_style=gait_style,
            )

        def on_success(result) -> None:
            self.status.configure(text=f"AMP reference → {result.npz_path.name}")
            messagebox.showinfo(
                "Export AMP Reference",
                f"AMP reference exported for Isaac Lab:\n{result.npz_path.resolve()}",
            )

        self._run_advanced_export_async(
            "Export AMP Reference", work, on_success=on_success
        )

    def _export_selected_point_analysis(self) -> None:
        """Export full analysis data for all selected points (session bundle)."""
        self._export_analysis_data()

    def _flush_dof_position_table_if_dirty(self) -> None:
        """Paint the DOF history Treeview after pause / Advanced tab open."""
        if not getattr(self, "_dof_table_display_dirty", False):
            return
        if self.playing:
            return
        rows = self._filtered_position_table_rows(list(self._dof_table_history.rows))
        emphasize = label_for_item(self._active_dof_item_id() or "")
        self._fill_dof_position_table(
            rows,
            scroll_to_bottom=True,
            emphasize_dof=emphasize or None,
        )
        self._dof_table_display_dirty = False

    def _fill_dof_position_table(
        self,
        rows: list[tuple[str, ...]],
        *,
        scroll_to_bottom: bool = False,
        emphasize_dof: str | None = None,
    ) -> None:
        """Update the selected-DOF position table with alternating row shading.

        Long histories are virtualized to the trailing window so insert/update
        cost stays O(window) instead of O(n) per paint.
        """
        if not hasattr(self, "dof_pos_tree"):
            return
        window = int(getattr(self, "_dof_table_virtual_window", 80) or 80)
        window = max(10, window)
        view_rows = rows[-window:] if len(rows) > window else rows
        dof_col = 2
        children = self.dof_pos_tree.get_children()
        for i, row in enumerate(view_rows):
            base_tag = "even" if i % 2 == 0 else "odd"
            tags = (base_tag,)
            if emphasize_dof and len(row) > dof_col and row[dof_col] == emphasize_dof:
                tags = ("selected",)
            if i < len(children):
                self.dof_pos_tree.item(children[i], values=row, tags=tags)
            else:
                self.dof_pos_tree.insert("", tk.END, values=row, tags=tags)
        for iid in children[len(view_rows) :]:
            self.dof_pos_tree.delete(iid)
        widths = getattr(self.dof_pos_tree, "_sw_column_widths", None)
        if widths:
            for col, w in widths.items():
                try:
                    self.dof_pos_tree.column(col, width=w)
                except tk.TclError:
                    pass
        try:
            visible = max(6, min(len(view_rows), 10))
            self.dof_pos_tree.configure(height=visible)
            self.dof_pos_tree.yview_moveto(1.0 if scroll_to_bottom else 0.0)
        except tk.TclError:
            pass
        self._update_table_summary_label()
        if not self.playing:
            self.dof_pos_tree.update_idletasks()

    def _clear_dof_position_table_display(self) -> None:
        """Empty the table widget without clearing stored history."""
        self._fill_dof_position_table([])

    def _clear_dof_position_table(self) -> None:
        self._clear_dof_position_table_display()
        self._refresh_selected_dof_trajectory_3d()
        if getattr(self, "lbl_dof_table_hint", None) is not None:
            self.lbl_dof_table_hint.configure(text=self._dof_table_hint_text())
        self._update_dof_table_controls_state()

    def _active_dof_item_id(self) -> str | None:
        """Single source of truth for the point driving detailed analysis."""
        if not self.selection.selected:
            return None
        self.selection.ensure_last_selected()
        return self.selection.active_item_id

    def _activate_dof_item(self, item_id: str, *, add_if_missing: bool = False) -> None:
        """Make ``item_id`` the active analysis point and sync all dependent panels."""
        from stablewalk.ui.dof_selection import GUI_DOF_LABELS

        if item_id not in GUI_DOF_LABELS:
            return
        if add_if_missing and item_id not in self.selection.selected:
            self.selection.activate_item(item_id)
            self._notify_dof_selection_changed()
            return
        if item_id not in self.selection.selected:
            return
        if self.selection.active_item_id == item_id:
            # Same joint — still refresh panels so metrics stay current.
            self._sync_panels_for_selected_joint(force_draw=True)
            return
        previous = self.selection.active_item_id
        self.selection.set_active(item_id)
        self._on_active_dof_item_changed(previous=previous)

    def _on_active_dof_item_changed(
        self,
        *,
        previous: str | None = None,
        force_draw: bool = True,
    ) -> None:
        """Refresh every panel bound to the active analysis point."""
        if previous and previous != self._active_dof_item_id():
            self._clear_selected_point_analysis_summary()
        self._sync_panels_for_selected_joint(force_draw=force_draw)
        if previous != self._active_dof_item_id():
            try:
                self._fit_overview_viewers(reset_camera=True)
            except Exception:
                pass

    def _filtered_position_table_rows(
        self,
        rows: list[tuple[str, ...]],
    ) -> list[tuple[str, ...]]:
        """Keep only rows for the active analysis point when one is set."""
        active = self._active_dof_item_id()
        if not active:
            return rows
        label = label_for_item(active)
        filtered = [row for row in rows if len(row) > 2 and row[2] == label]
        return filtered if filtered else rows

    def _sync_active_joint_from_charted(self) -> None:
        """Keep skeleton anchor joint aligned with the active analysis point."""
        from stablewalk.ui.dof_selection import anchor_joint_for_item

        active = self._active_dof_item_id()
        if active:
            self._active_joint = anchor_joint_for_item(active)
        else:
            self._active_joint = None

    def _charted_dof_item_id(self) -> str | None:
        """Legacy alias — use ``_active_dof_item_id`` for the analysis focus point."""
        return self._active_dof_item_id()

    def _table_item_ids_for_display(self) -> set[str]:
        """Rows shown in the position table follow the active charted point."""
        charted = self._charted_dof_item_id()
        if charted:
            return {charted}
        return set(self.selection.selected)

    def _refresh_charted_dof_views(self, *, force_draw: bool = False) -> None:
        """Refresh every panel that tracks the active analysis point."""
        self.selection.ensure_last_selected()
        self._on_active_dof_item_changed(force_draw=force_draw)

    def _select_charted_dof_item(self, item_id: str) -> None:
        """Ensure a body point is selected and active, then sync all panels."""
        self._activate_dof_item(item_id, add_if_missing=True)

    def _focus_charted_dof_item(self, item_id: str) -> None:
        """Switch the active analysis point without changing the selection set."""
        self._activate_dof_item(item_id)

    def _fill_analysis_metric_slots(
        self,
        slots: list[tuple] | None,
        metrics: tuple,
        *,
        accent_value_index: int | None = None,
    ) -> None:
        if not slots:
            return
        from stablewalk.ui.theme import FONT_METRIC_VALUE, FONT_METRIC_VALUE_ACCENT

        point_value = getattr(self, "dof_analysis_point_value_lbl", None)
        selected_lbl = getattr(self, "lbl_export_selected_joint", None) or getattr(
            self, "lbl_selected_joint_data", None
        )
        for index, (title_lbl, value_lbl) in enumerate(slots):
            if index < len(metrics):
                title, value = metrics[index]
                title_lbl.configure(text=title)
                value_lbl.configure(text=value)
                if index == 0 and selected_lbl is not None:
                    selected_lbl.configure(text=f"Selected Joint:\n{value}")
                if accent_value_index is not None and index == accent_value_index:
                    value_lbl.configure(
                        fg=ORANGE,
                        font=FONT_METRIC_VALUE_ACCENT,
                    )
                elif value_lbl is point_value:
                    value_lbl.configure(fg=ORANGE, font=FONT_METRIC_VALUE_ACCENT)
                else:
                    value_lbl.configure(fg=TEXT, font=FONT_METRIC_VALUE)
            else:
                title_lbl.configure(text="")
                value_lbl.configure(text="—", fg=TEXT, font=FONT_METRIC_VALUE)

    def _set_analysis_secondary_visible_slots(self, visible_count: int) -> None:
        """Show only the first N secondary metric cells (3 general, 4 foot)."""
        hosts = getattr(self, "dof_analysis_secondary_hosts", None)
        if not hosts:
            return
        for index, host in enumerate(hosts):
            if index < visible_count:
                host.grid()
            else:
                host.grid_remove()

    def _fill_analysis_secondary_slots(
        self,
        slots: list[tuple] | None,
        metrics: tuple,
        *,
        mode: str,
    ) -> None:
        if not slots:
            return
        from stablewalk.ui.theme import FONT_METRIC_VALUE, FONT_METRIC_VALUE_ACCENT

        for index, (title_lbl, value_lbl) in enumerate(slots):
            if index < len(metrics):
                title, value = metrics[index]
                title_lbl.configure(text=title)
                value_lbl.configure(text=value)
                if mode == "foot" and index == 0:
                    from stablewalk.analysis.ground_reference import CALIBRATION_CHECK_LABEL

                    is_cal = str(value) in (
                        CALIBRATION_CHECK_LABEL,
                        "Check calibration",
                    )
                    value_lbl.configure(
                        fg=DANGER if is_cal else ORANGE,
                        font=FONT_METRIC_VALUE_ACCENT,
                    )
                elif mode == "foot" and index == 1:
                    value_lbl.configure(
                        fg=self._contact_status_color(str(value)),
                        font=FONT_METRIC_VALUE_ACCENT,
                    )
                else:
                    value_lbl.configure(fg=TEXT, font=FONT_METRIC_VALUE)
            else:
                title_lbl.configure(text="")
                value_lbl.configure(text="—", fg=TEXT, font=FONT_METRIC_VALUE)

    def _set_analysis_summary_visible_slots(self, visible_count: int) -> None:
        """Show only the first N summary metric cells (foot mode uses one)."""
        hosts = getattr(self, "dof_analysis_summary_hosts", None)
        if not hosts:
            return
        for index, host in enumerate(hosts):
            if index < visible_count:
                host.grid()
            else:
                host.grid_remove()

    def _contact_status_color(self, status: str) -> str:
        from stablewalk.analysis.ground_reference import CALIBRATION_CHECK_LABEL

        if status == "On Ground":
            return SUCCESS
        if status == "Near Ground":
            return WARNING
        if status == "In Air":
            return ACCENT_ALT
        if status in ("Check calibration", CALIBRATION_CHECK_LABEL):
            return DANGER
        return TEXT

    def _update_analysis_mode_ui(self, mode: str | None) -> None:
        """Switch visible chrome between General Point Analysis and Foot Analysis."""
        from stablewalk.ui.theme import (
            ACCENT,
            DOF_ANALYSIS_MODE_FOOT,
            DOF_ANALYSIS_MODE_FOOT_HINT,
            DOF_ANALYSIS_MODE_GENERAL,
            DOF_ANALYSIS_MODE_GENERAL_HINT,
            DOF_ANALYSIS_MOVEMENT_TITLE,
            ORANGE,
        )

        mode_lbl = getattr(self, "lbl_dof_analysis_mode", None)
        hint_lbl = getattr(self, "lbl_dof_analysis_mode_hint", None)
        movement_title = getattr(self, "lbl_dof_analysis_movement_title", None)

        if mode_lbl is None:
            return

        if mode == "foot":
            mode_lbl.configure(text=DOF_ANALYSIS_MODE_FOOT, fg=ORANGE)
            if hint_lbl is not None:
                hint_lbl.configure(text=DOF_ANALYSIS_MODE_FOOT_HINT)
                if DOF_ANALYSIS_MODE_FOOT_HINT:
                    hint_lbl.grid()
                else:
                    hint_lbl.grid_remove()
            if movement_title is not None:
                movement_title.grid_remove()
        elif mode == "general":
            mode_lbl.configure(text=DOF_ANALYSIS_MODE_GENERAL, fg=ACCENT)
            if hint_lbl is not None:
                hint_lbl.configure(text=DOF_ANALYSIS_MODE_GENERAL_HINT)
                if DOF_ANALYSIS_MODE_GENERAL_HINT:
                    hint_lbl.grid()
                else:
                    hint_lbl.grid_remove()
            if movement_title is not None:
                movement_title.grid_remove()
        else:
            mode_lbl.configure(text="")
            if hint_lbl is not None:
                hint_lbl.configure(text="")
            if movement_title is not None:
                movement_title.grid_remove()
            derived_row = getattr(self, "dof_analysis_derived_row", None)
            if derived_row is not None:
                derived_row.grid_remove()

    def _refresh_foot_analysis_card(self, card) -> None:
        from stablewalk.ui.selected_point_analysis import (
            FootAnalysisCardMetrics,
            is_foot_analysis_point,
        )

        item_id = self._charted_dof_item_id()
        host = getattr(self, "dof_analysis_foot_host", None)
        if (
            not is_foot_analysis_point(item_id)
            or host is None
            or not isinstance(card, FootAnalysisCardMetrics)
        ):
            self._clear_foot_analysis_card()
            return

        self._update_analysis_mode_ui("foot")
        self._layout_analysis_body(foot_mode=True)
        host.grid()

        for attr, value in (
            ("foot_card_min_lbl", card.min_clearance_cm),
            ("foot_card_max_lbl", card.max_clearance_cm),
            ("foot_card_avg_lbl", card.avg_clearance_cm),
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text=value or "—")

        note_lbl = getattr(self, "lbl_foot_ground_note", None)
        if note_lbl is not None:
            note_lbl.configure(text=card.ground_note or "—")

    def _clear_foot_analysis_card(self) -> None:
        host = getattr(self, "dof_analysis_foot_host", None)
        if host is not None:
            host.grid_remove()
        for attr in (
            "foot_card_min_lbl",
            "foot_card_max_lbl",
            "foot_card_avg_lbl",
            "lbl_foot_ground_note",
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text="—", fg=TEXT if attr != "lbl_foot_ground_note" else MUTED)
        self._layout_analysis_body(foot_mode=False)

    def _clear_selected_point_analysis_summary(self) -> None:
        """Reset analysis summary — values stay hidden until a point is selected."""
        summary_slots = getattr(self, "dof_analysis_summary_slots", None)
        if not summary_slots:
            identity_slots = getattr(self, "dof_analysis_identity_slots", None)
            kinematics_slots = getattr(self, "dof_analysis_kinematics_slots", None)
            if not identity_slots or not kinematics_slots:
                return
            summary_slots = identity_slots + kinematics_slots
        for title_lbl, value_lbl in summary_slots:
            title_lbl.configure(text="")
            value_lbl.configure(text="", fg=TEXT, font=FONT_METRIC)
        self._set_analysis_summary_visible_slots(len(summary_slots) if summary_slots else 0)
        movement_title = getattr(self, "lbl_dof_analysis_movement_title", None)
        if movement_title is not None:
            movement_title.grid_remove()
        derived_row = getattr(self, "dof_analysis_derived_row", None)
        if derived_row is not None:
            derived_row.grid_remove()
        derived_slots = getattr(self, "dof_analysis_derived_slots", None)
        if derived_slots:
            for title_lbl, value_lbl in derived_slots:
                title_lbl.configure(text="")
                value_lbl.configure(text="—", fg=TEXT)
        coord_slots = getattr(self, "dof_analysis_advanced_coord_slots", None)
        if coord_slots:
            for title_lbl, value_lbl in coord_slots:
                title_lbl.configure(text="")
                value_lbl.configure(text="—", fg=TEXT)
        advanced_host = getattr(self, "dof_analysis_advanced_host", None)
        if advanced_host is not None:
            advanced_host.grid_remove()
        self._joint_advanced_visible = False
        toggle_btn = getattr(self, "btn_toggle_joint_advanced", None)
        if toggle_btn is not None:
            toggle_btn.configure(text="Detailed Joint Data \u25be")
        self._set_analysis_secondary_visible_slots(0)
        self._clear_foot_analysis_card()
        self._update_analysis_mode_ui(None)
        self._clear_dof_graph_chrome()
        tracked = getattr(self, "lbl_dof_analysis_tracked", None)
        if tracked is not None:
            tracked.configure(text="")
        title_lbl = getattr(self, "lbl_joint_movement_title", None)
        if title_lbl is not None:
            title_lbl.configure(text="Select a joint to view its 3D movement path")
        for attr, default in (
            ("lbl_selected_joint_value", "—"),
            ("lbl_dof_coord_mode_value", "Root-relative"),
            ("lbl_dof_traj_mode_value", "Current progress"),
            ("lbl_dof_view_mode_value", "3D"),
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text=default)
        for attr, default in (
            ("lbl_traj_travel", "Travel: —"),
            ("lbl_traj_smoothness", "Smoothness: —"),
            ("lbl_traj_max_deviation", "Maximum deviation: —"),
            ("lbl_traj_confidence", "Trajectory Confidence: —"),
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text=default)
        summary_lbl = getattr(self, "lbl_joint_path_summary", None)
        if summary_lbl is not None:
            summary_lbl.configure(text="Select a joint to view its movement path.")

    def _refresh_selected_point_analysis_summary(
        self,
        item_id: str | None,
        tip_snapshot,
        *,
        end_frame_float: float = 0.0,
    ) -> None:
        """Update the compact metrics header for the active analysis point."""
        active = self._active_dof_item_id()
        if item_id and active and item_id != active:
            return
        summary_slots = getattr(self, "dof_analysis_summary_slots", None)
        identity_slots = getattr(self, "dof_analysis_identity_slots", None)
        kinematics_slots = getattr(self, "dof_analysis_kinematics_slots", None)
        if not summary_slots and (not identity_slots or not kinematics_slots):
            return
        if not item_id or tip_snapshot is None:
            self._clear_selected_point_analysis_summary()
            return

        from stablewalk.ui.dof_position_table import snapshot_for_next_frame
        from stablewalk.ui.selected_point_analysis import (
            build_selected_point_analysis,
            is_foot_analysis_point,
            metrics_for_analysis_panel,
        )

        recording = self._analysis_motion_recording()
        next_snapshot = None
        if recording is not None:
            next_snapshot = snapshot_for_next_frame(recording, tip_snapshot)

        analysis = build_selected_point_analysis(
            item_id,
            tip_snapshot,
            recording,
            end_frame_float,
            next_snapshot=next_snapshot,
        )
        panel_metrics = metrics_for_analysis_panel(
            item_id,
            tip_snapshot,
            analysis,
            recording=recording,
            end_frame_float=end_frame_float,
        )

        is_foot = panel_metrics.mode == "foot"
        identity = panel_metrics.identity
        kinematics = panel_metrics.kinematics

        # Primary row: joint, time, speed, movement (angle or position).
        movement_metric = ("Movement", "—")
        if panel_metrics.derived:
            movement_metric = panel_metrics.derived[0]
        elif len(kinematics) >= 4:
            movement_metric = kinematics[3]
        elif kinematics:
            movement_metric = kinematics[-1]

        primary_metrics = (
            identity[0] if identity else ("Joint", "—"),
            identity[1] if len(identity) > 1 else ("Time (s)", "—"),
            kinematics[3] if len(kinematics) > 3 else ("Speed (m/s)", "—"),
            movement_metric,
        )

        if summary_slots:
            self._fill_analysis_metric_slots(
                summary_slots,
                primary_metrics,
                accent_value_index=0,
            )
            self._set_analysis_summary_visible_slots(len(primary_metrics))
        else:
            self._fill_analysis_metric_slots(
                identity_slots, panel_metrics.identity[:2], accent_value_index=0
            )
            self._fill_analysis_metric_slots(
                kinematics_slots, primary_metrics[2:]
            )
            self._set_analysis_summary_visible_slots(len(primary_metrics))

        coord_slots = getattr(self, "dof_analysis_advanced_coord_slots", None)
        if coord_slots and len(identity) > 2 and len(kinematics) >= 3:
            coord_metrics = (
                identity[2],
                kinematics[0],
                kinematics[1],
                kinematics[2],
            )
            self._fill_analysis_metric_slots(coord_slots, coord_metrics)
            hosts = getattr(self, "dof_analysis_advanced_coord_hosts", None)
            if hosts:
                for host in hosts:
                    host.grid()

        derived_row = getattr(self, "dof_analysis_derived_row", None)
        derived_slots = getattr(self, "dof_analysis_derived_slots", None)
        advanced_host = getattr(self, "dof_analysis_advanced_host", None)

        if derived_slots and panel_metrics.derived:
            start_idx = 1 if is_foot else 0
            derived_subset = panel_metrics.derived[start_idx:]
            if derived_subset:
                self._fill_analysis_secondary_slots(
                    derived_slots,
                    derived_subset,
                    mode=panel_metrics.mode,
                )
                visible_secondary = min(
                    len(derived_subset),
                    4 if is_foot else 3,
                )
                self._set_analysis_secondary_visible_slots(visible_secondary)
            elif derived_row is not None:
                self._set_analysis_secondary_visible_slots(0)
        elif derived_row is not None:
            self._set_analysis_secondary_visible_slots(0)

        if advanced_host is not None and not getattr(self, "_joint_advanced_visible", False):
            advanced_host.grid_remove()

        if is_foot:
            self._update_analysis_mode_ui("foot")
            if panel_metrics.foot_card is not None:
                self._refresh_foot_analysis_card(panel_metrics.foot_card)
            else:
                self._clear_foot_analysis_card()
        else:
            self._update_analysis_mode_ui("general")
            self._clear_foot_analysis_card()

        self._refresh_dof_graph_chrome(
            item_id,
            tip_snapshot,
            analysis,
            end_frame_float=end_frame_float,
        )

    def _on_dof_projection_changed(self) -> None:
        from stablewalk.ui.tk.trajectory_camera_toolbar import clear_all_trajectory_cameras

        clear_all_trajectory_cameras(self)
        self._dof_traj_projection_mode = None
        self._traj_draw_cache_key = None
        self._refresh_selected_dof_trajectory_3d(force_draw=True)

    def _on_dof_coord_mode_changed(self) -> None:
        self._traj_draw_cache_key = None
        self._refresh_selected_dof_trajectory_3d(force_draw=True)

    def _on_dof_traj_display_changed(self) -> None:
        self._traj_draw_cache_key = None
        self._refresh_selected_dof_trajectory_3d(force_draw=True)

    def _on_dof_body_reference_changed(self) -> None:
        self._traj_draw_cache_key = None
        self._refresh_selected_dof_trajectory_3d(force_draw=True)

    def _motion_frame_series(self):
        """Cached global/root-relative frame series for GLOBAL trajectory mode."""
        from stablewalk.analysis.motion_frames import build_motion_frame_series

        recording = self._analysis_motion_recording()
        sequence = getattr(self, "sequence", None)
        if recording is None or sequence is None or not sequence.frames:
            return None
        cache_key = (id(recording), len(sequence.frames))
        if getattr(self, "_motion_series_cache_key", None) != cache_key:
            self._motion_series_cache_key = cache_key
            self._motion_frame_series_cache = build_motion_frame_series(
                sequence.frames,
                recording,
            )
        return getattr(self, "_motion_frame_series_cache", None)

    def _ensure_dof_traj_axes(self, projection_mode: str) -> None:
        from stablewalk.ui.viewers.dof_trajectory_3d import setup_single_dof_trajectory_axes

        # Motion-tab trajectory figure is lazy — not built until that tab opens.
        if getattr(self, "fig_dof_traj", None) is None:
            return
        ax = getattr(self, "ax_dof_traj", None)
        if ax is not None:
            try:
                ax.get_zlim()
                return
            except AttributeError:
                pass
        self.fig_dof_traj.clf()
        self.ax_dof_traj = self.fig_dof_traj.add_subplot(111, projection="3d")
        self.ax_dof_traj._stablewalk_motion_dock = True
        self.ax_dof_traj._stablewalk_overview_cm_ticks = True
        setup_single_dof_trajectory_axes(self.ax_dof_traj)
        self._dof_traj_projection_mode = projection_mode
        self._traj_draw_cache_key = None

    def _apply_default_dof_projection(self) -> None:
        from stablewalk.ui.dashboard_interpretability import default_projection_for_view

        var = getattr(self, "var_dof_projection", None)
        if var is None:
            return
        view = getattr(self._biomech, "view_type", None) if self._biomech else None
        var.set(default_projection_for_view(view))
        self._dof_traj_projection_mode = None

    def _update_dof_trajectory_live_info(
        self,
        item_id: str | None,
        tip_snapshot,
        *,
        end_frame_float: float,
        coord_mode: str,
        motion_series,
    ) -> None:
        """Update the scientific path statistics strip (XYZ, velocity, angle…)."""
        import math

        from stablewalk.ui.dof_selection import anchor_joint_for_item, label_for_item
        from stablewalk.ui.value_provenance import ValueProvenance, format_na, get_provenance_bundle
        from stablewalk.ui.viewers.dof_trajectory_3d import (
            _point_distance,
            _transform_position_for_coord_mode,
            stature_display_scale,
        )

        values = {
            "lbl_traj_info_joint": label_for_item(item_id) if item_id else "—",
            "lbl_traj_info_frame": "—",
            "lbl_traj_info_time": "—",
            "lbl_traj_info_x": "—",
            "lbl_traj_info_y": "—",
            "lbl_traj_info_z": "—",
            "lbl_traj_inst_velocity": "—",
            "lbl_traj_total_distance": "—",
            "lbl_traj_displacement": "—",
            "lbl_traj_path_length": "—",
            "lbl_traj_joint_angle": "—",
            "lbl_traj_coord_units": (
                "cm est. · root-relative · not calibrated"
                if str(coord_mode).lower().startswith("root")
                else "cm est. · global · not calibrated"
            ),
        }
        from stablewalk.ui.playback_state import read_playback_state

        recording = self._analysis_motion_recording()
        state = read_playback_state(self)
        # Always show the global playback frame — never trajectory sample count
        # or a stale tip's source-video id.
        if state is not None:
            list_pos = int(state.current_frame_index)
            time_s = float(state.current_time_seconds)
            values["lbl_traj_info_frame"] = (
                f"{state.display_frame_number} / {state.total_frames}"
            )
            values["lbl_traj_info_time"] = state.format_time_label(
                include_duration=False
            )
            self._path_marker_list_pos = list_pos
        else:
            list_pos = int(max(0, end_frame_float))
            time_s = 0.0
            if tip_snapshot is not None:
                time_s = float(getattr(tip_snapshot, "time_s", 0.0) or 0.0)
            elif recording is not None and getattr(recording, "fps", 0):
                time_s = list_pos / float(recording.fps)
            total = max(int(getattr(recording, "frame_count", 0)), 0)
            values["lbl_traj_info_frame"] = (
                f"{list_pos + 1} / {total}" if total > 0 else "—"
            )
            values["lbl_traj_info_time"] = f"{time_s:.2f} s"
        frame_index = list_pos

        joint_id = anchor_joint_for_item(item_id) if item_id else None
        sample = (
            tip_snapshot.joints.get(joint_id)
            if tip_snapshot is not None and joint_id
            else None
        )
        pos_scale = stature_display_scale(recording)
        coord_label = (
            "Root-relative (pelvis origin)"
            if str(coord_mode).lower().startswith("root")
            else str(coord_mode)
        )
        tip_xyz_m: tuple[float, float, float] | None = None
        if sample is not None:
            position = _transform_position_for_coord_mode(
                sample.position,
                frame_index,
                coord_mode=coord_mode,
                motion_series=motion_series,
            )
            coords = (float(position.x), float(position.y), float(position.z))
            if all(math.isfinite(value) for value in coords):
                tip_xyz_m = (
                    coords[0] * pos_scale,
                    coords[1] * pos_scale,
                    coords[2] * pos_scale,
                )
                values["lbl_traj_info_x"] = f"{tip_xyz_m[0] * 100.0:+.1f} cm est."
                values["lbl_traj_info_y"] = f"{tip_xyz_m[1] * 100.0:+.1f} cm est."
                values["lbl_traj_info_z"] = f"{tip_xyz_m[2] * 100.0:+.1f} cm est."
            else:
                values["lbl_traj_info_x"] = format_na("non-finite coordinates")
                values["lbl_traj_info_y"] = format_na("non-finite coordinates")
                values["lbl_traj_info_z"] = format_na("non-finite coordinates")
        else:
            values["lbl_traj_info_x"] = format_na("joint position missing")
            values["lbl_traj_info_y"] = format_na("joint position missing")
            values["lbl_traj_info_z"] = format_na("joint position missing")

        # Instantaneous velocity + traveled prefix from reconstructed samples.
        path_times = getattr(self, "_path_science_path_times", None)
        if isinstance(path_times, list) and len(path_times) >= 2:
            from stablewalk.analysis.joint_path_trace import (
                gap_aware_prefix_lengths_m,
                index_nearest_time,
                instantaneous_speed_m_s,
            )

            fps = float(getattr(recording, "fps", 0.0) or 30.0) if recording else 30.0
            idx = index_nearest_time(path_times, time_s, frame_index=list_pos)
            inst_m_s = instantaneous_speed_m_s(path_times, idx, fps=fps)
            if inst_m_s is not None:
                values["lbl_traj_inst_velocity"] = f"{inst_m_s * 100.0:.1f} cm/s est."
            prefixes = gap_aware_prefix_lengths_m(path_times, fps=fps)
            if prefixes:
                from stablewalk.ui.overview_path_display import (
                    displacement_m,
                    format_est_cm,
                )

                points = [item[0] if isinstance(item, tuple) else item for item in path_times]
                values["lbl_traj_path_length"] = format_est_cm(prefixes[idx])
                values["lbl_traj_displacement"] = format_est_cm(
                    displacement_m(points, end_index=idx)
                )
                values["lbl_traj_total_distance"] = values["lbl_traj_displacement"]
        elif tip_xyz_m is not None:
            prev = getattr(self, "_path_science_prev_tip", None)
            if (
                isinstance(prev, tuple)
                and len(prev) == 4
                and float(prev[3]) < time_s
            ):
                dt = time_s - float(prev[3])
                if dt > 1e-9:
                    dist = math.sqrt(
                        (tip_xyz_m[0] - float(prev[0])) ** 2
                        + (tip_xyz_m[1] - float(prev[1])) ** 2
                        + (tip_xyz_m[2] - float(prev[2])) ** 2
                    ) * 100.0
                    values["lbl_traj_inst_velocity"] = f"{dist / dt:.1f} cm/s"
            self._path_science_prev_tip = (
                tip_xyz_m[0],
                tip_xyz_m[1],
                tip_xyz_m[2],
                time_s,
            )

        # Current joint angle (degrees) when available for this DOF.
        if tip_snapshot is not None and item_id:
            try:
                from stablewalk.ui.dof_position_table import angle_value_for_item

                deg = angle_value_for_item(item_id, tip_snapshot)
                if deg is not None and math.isfinite(float(deg)):
                    values["lbl_traj_joint_angle"] = f"{float(deg):+.1f}°"
            except Exception:
                pass

        bundle = get_provenance_bundle(self)
        for axis, attr in (("x", "lbl_traj_info_x"), ("y", "lbl_traj_info_y"), ("z", "lbl_traj_info_z")):
            bundle.put(
                f"motion_position_{axis}",
                ValueProvenance(
                    metric_name=f"Motion {axis.upper()} (path tip)",
                    display_value=values[attr],
                    raw_value=None,
                    units="cm (stature-scaled)",
                    source_frame=list_pos,
                    source_function="app._update_dof_trajectory_live_info",
                    source_array=f"tip_snapshot.joints['{joint_id}'].position.{axis}" if joint_id else "",
                    coordinate_system=f"{coord_label}; not world unless Global mode",
                    filtering=f"stature_display_scale={pos_scale:.4f}",
                    scope="instantaneous",
                    tier="estimated",
                    unavailable_reason=(
                        "joint position missing"
                        if values[attr].startswith("N/A")
                        else None
                    ),
                ),
            )

        for attr, text in values.items():
            label = getattr(self, attr, None)
            if label is not None:
                label.configure(text=text)
        try:
            from stablewalk.ui.tk.dashboard_path_science import (
                sync_path_science_compat_labels,
            )

            sync_path_science_compat_labels(self)
        except Exception:
            pass
        try:
            self._refresh_overview_coord_caption()
        except Exception:
            pass

    def _update_dof_trajectory_interpretation(
        self,
        item_id: str,
        path: list,
        *,
        projection_mode: str,
    ) -> None:
        from stablewalk.ui.dashboard_interpretability import (
            coordinate_mode_display,
            evaluate_trajectory_readiness,
            format_trajectory_confidence,
            interpret_joint_trajectory,
            movement_path_title,
            truncate_dashboard_explanation,
        )
        from stablewalk.ui.dof_selection import label_for_item

        joint_label = label_for_item(item_id) or "Joint"
        title_lbl = getattr(self, "lbl_joint_movement_title", None)
        if title_lbl is not None:
            title_lbl.configure(text=movement_path_title(joint_label))
        traj_panel = getattr(self, "traj_panel", None)
        if traj_panel is not None:
            try:
                traj_panel.configure(
                    text=format_labelframe_title(movement_path_title(joint_label))
                )
            except tk.TclError:
                pass

        joint_val = getattr(self, "lbl_selected_joint_value", None)
        if joint_val is not None:
            joint_val.configure(text=joint_label)

        coord_var = getattr(self, "var_dof_coord_mode", None)
        coord_mode = coord_var.get() if coord_var is not None else "Root-relative"
        coord_val = getattr(self, "lbl_dof_coord_mode_value", None)
        if coord_val is not None:
            coord_val.configure(text=coordinate_mode_display(coord_mode))

        display_var = getattr(self, "var_dof_traj_display", None)
        traj_mode = display_var.get() if display_var is not None else "Current progress"
        traj_val = getattr(self, "lbl_dof_traj_mode_value", None)
        if traj_val is not None:
            from stablewalk.ui.viewers.dof_trajectory_3d import normalize_display_mode

            traj_val.configure(text=normalize_display_mode(traj_mode))

        view_val = getattr(self, "lbl_dof_view_mode_value", None)
        if view_val is not None:
            view_val.configure(text=projection_mode)

        readiness = evaluate_trajectory_readiness(path, projection=projection_mode)
        metrics = readiness.metrics
        view_type = getattr(self._biomech, "view_type", None) if self._biomech else None
        interpretation = interpret_joint_trajectory(
            joint_label,
            metrics,
            projection=projection_mode,
            view_type=view_type,
        )

        travel_lbl = getattr(self, "lbl_traj_travel", None)
        smooth_lbl = getattr(self, "lbl_traj_smoothness", None)
        dev_lbl = getattr(self, "lbl_traj_max_deviation", None)
        samples_lbl = getattr(self, "lbl_traj_samples", None)
        summary_lbl = getattr(self, "lbl_joint_path_summary", None)
        metrics_lbl = getattr(self, "lbl_motion_traj_metrics", None)
        conf_lbl = getattr(self, "lbl_traj_confidence", None)

        if samples_lbl is not None:
            samples_lbl.configure(text=f"Samples: {len(path)}")

        if not readiness.sufficient:
            if travel_lbl is not None:
                travel_lbl.configure(text="Travel: —")
            for attr in (
                "lbl_traj_path_length",
                "lbl_traj_avg_velocity",
                "lbl_traj_motion_range",
                "lbl_traj_inst_velocity",
                "lbl_traj_total_distance",
                "lbl_traj_displacement",
            ):
                lbl = getattr(self, attr, None)
                if lbl is not None:
                    lbl.configure(text="—")
            if smooth_lbl is not None:
                smooth_lbl.configure(text="Smoothness: —")
            if dev_lbl is not None:
                dev_lbl.configure(text="Maximum deviation: —")
            if summary_lbl is not None:
                summary_lbl.configure(
                    text=f"Insufficient trajectory data. {readiness.reason}"
                )
            if metrics_lbl is not None:
                metrics_lbl.configure(text="")
            if conf_lbl is not None:
                conf_lbl.configure(text="Path: Insufficient")
            return

        if metrics is not None:
            travel_cm = metrics.total_travel_m * 100.0
            if travel_lbl is not None:
                travel_lbl.configure(text=f"Travel: {travel_cm:.1f} cm est.")
            # Path length / traveled are set from reconstructed samples
            # in _update_dof_trajectory_live_info (gap-aware prefix sums).
            if smooth_lbl is not None:
                smooth_lbl.configure(text=f"Smoothness: {metrics.smoothness}")
            if dev_lbl is not None:
                dev_lbl.configure(
                    text=f"Maximum deviation: {metrics.max_deviation_m * 100:.1f} cm"
                )
            span_x = span_y = span_z = rom = 0.0
            if path:
                xs = [p.x for p in path]
                ys = [p.y for p in path]
                zs = [p.z for p in path]
                span_x = (max(xs) - min(xs)) * 100.0 if xs else 0.0
                span_y = (max(ys) - min(ys)) * 100.0 if ys else 0.0
                span_z = (max(zs) - min(zs)) * 100.0 if zs else 0.0
                rom = max(span_x, span_y, span_z)
            range_lbl = getattr(self, "lbl_traj_motion_range", None)
            if range_lbl is not None:
                range_lbl.configure(text=f"{rom:.1f} cm est.")
            # Average path speed from timed samples when available.
            avg_lbl = getattr(self, "lbl_traj_avg_velocity", None)
            path_times = getattr(self, "_path_science_path_times", None)
            if avg_lbl is not None:
                avg_txt = "—"
                if isinstance(path_times, list) and len(path_times) >= 2:
                    try:
                        from stablewalk.ui.viewers.dof_trajectory_3d import (
                            _path_speed_stats_cm_s,
                        )

                        avg_speed, _peak = _path_speed_stats_cm_s(path_times)
                        if avg_speed is not None:
                            avg_txt = f"{avg_speed:.1f} cm/s est."
                    except Exception:
                        avg_txt = "—"
                avg_lbl.configure(text=avg_txt)
            if metrics_lbl is not None and path:
                metrics_lbl.configure(
                    text=(
                        f"Travel {travel_cm:.1f} cm  ·  ROM {rom:.1f} cm  ·  "
                        f"side {span_x:.1f} · up {span_y:.1f} · fwd {span_z:.1f} cm"
                    )
                )
        zoom_pct = getattr(self.ax_dof_traj, "_stablewalk_zoom_note_pct", None)
        summary_text = truncate_dashboard_explanation(interpretation.sentence, max_len=90)
        if zoom_pct is not None:
            summary_text = (
                f"Magnified view (~{zoom_pct:.1f}% BH). {summary_text}"
            )
        if summary_lbl is not None:
            summary_lbl.configure(text=summary_text)
        if conf_lbl is not None:
            conf_lbl.configure(
                text=f"Path: {format_trajectory_confidence(interpretation.confidence)}"
            )

    def _update_motion_path_tip_only(self) -> bool:
        """Move Motion-tab current marker + partial path during playback.

        Full path / limits / camera are rebuilt on analysis load, joint change,
        or coordinate-mode change — never every video frame.
        """
        ax = getattr(self, "ax_dof_traj", None)
        canvas = getattr(self, "canvas_dof_traj", None)
        if ax is None or canvas is None:
            return False
        item_id = self._active_dof_item_id()
        if not item_id or not self.selection.selected:
            return False
        recording = self._analysis_motion_recording()
        if recording is None or self.skeleton_player is None:
            return False
        now = time.monotonic()
        interval = float(getattr(self, "_traj_redraw_interval_s", 0.0) or 0.0)
        if interval > 0.0 and (
            now - float(getattr(self, "_last_traj_redraw_ts", 0.0))
        ) < interval:
            return True
        self._last_traj_redraw_ts = now

        from stablewalk.ui.dof_selection import anchor_joint_for_item
        from stablewalk.ui.tk.dashboard_status_bar import playhead_frame_0based
        from stablewalk.ui.viewers.dof_trajectory_3d import update_trajectory_current_tip

        frame_i = playhead_frame_0based(self)
        if frame_i is None:
            frame_i = int(self.skeleton_player.state.frame_index)
        tip_snapshot = self.skeleton_player.current_snapshot()
        if tip_snapshot is None:
            tip_snapshot = recording.snapshot_at(frame_i)
        if tip_snapshot is None:
            return False
        joint_id = anchor_joint_for_item(item_id)
        if not joint_id:
            return False
        coord_mode = "Root-relative"
        var_coord = getattr(self, "var_dof_coord_mode", None)
        if var_coord is not None:
            coord_mode = var_coord.get() or "Root-relative"
        from stablewalk.ui.viewers.dof_trajectory_3d import normalize_coord_mode

        motion_series = (
            self._motion_frame_series()
            if normalize_coord_mode(coord_mode) == "Global"
            else None
        )
        if not update_trajectory_current_tip(
            ax,
            None,
            tip_snapshot=tip_snapshot,
            joint_id=joint_id,
            recording=recording,
            coord_mode=coord_mode,
            motion_series=motion_series,
            list_pos=frame_i,
            update_path=True,
        ):
            return False
        self._path_marker_list_pos = int(frame_i)
        self._last_traj_tip_frame = int(frame_i)
        try:
            self._update_dof_trajectory_live_info(
                item_id,
                tip_snapshot,
                end_frame_float=float(frame_i),
                coord_mode=coord_mode,
                motion_series=motion_series,
            )
        except Exception:
            pass
        self._paint_canvas_now(canvas, idle=True)
        return True

    def _refresh_motion_trajectory_on_frame(
        self,
        *,
        force_draw: bool = False,
        refresh_overview: bool = True,
    ) -> None:
        """Keep trajectory graphs in sync with the current playback frame."""
        if getattr(self, "fig_dof_traj", None) is None:
            return
        has_session = bool(
            self.skeleton_player and getattr(self.skeleton_player, "frame_count", 0) > 0
        )
        if not has_session:
            return
        # Hidden Motion tab: keep cached path data, skip all trajectory paints.
        if self.playing and not (
            self._group_visible("trajectory") or self._group_visible("motion")
        ):
            return
        # During play: tip marker only — never rebuild the full 3D chart.
        if self.playing and not force_draw:
            if self._update_motion_path_tip_only():
                return
        self._refresh_selected_dof_trajectory_3d(
            force_draw=force_draw,
            refresh_overview=refresh_overview,
        )
        # During the playback sync pass, ``_show_pose_at`` already updated the
        # Joint Motion Analysis chart for this frame — avoid rebuilding it twice.
        if not getattr(self, "_playback_sync_immediate", False) and (
            self._group_visible("joint_motion") or self._group_visible("motion")
        ):
            try:
                self._update_joint_motion_analysis_chart(force_rebuild=force_draw)
            except Exception:
                pass
        if self._group_visible("spectral"):
            upd = getattr(self, "_update_spectral_playhead", None)
            if callable(upd):
                try:
                    upd()
                except Exception:
                    pass
        if self._group_visible("four_gait"):
            upd_f = getattr(self, "_update_four_gait_playhead", None)
            if callable(upd_f):
                try:
                    upd_f()
                except Exception:
                    pass
        if self._group_visible("asymmetry"):
            upd_a = getattr(self, "_update_asymmetry_playhead", None)
            if callable(upd_a):
                try:
                    upd_a()
                except Exception:
                    pass

    def _refresh_selected_dof_trajectory_3d(
        self,
        *,
        force_draw: bool = False,
        refresh_overview: bool = True,
    ) -> None:
        """Update the single 3D trajectory graph for the active selected point."""
        # Lazy Motion-tab widget — attribute exists as None until the tab builds it.
        if getattr(self, "fig_dof_traj", None) is None:
            return

        item_id = self._active_dof_item_id()
        has_session = bool(
            self.skeleton_player and getattr(self.skeleton_player, "frame_count", 0) > 0
        )
        if not item_id or not self.selection.selected:
            self._update_dof_trajectory_live_info(
                None,
                None,
                end_frame_float=0.0,
                coord_mode="Root-relative",
                motion_series=None,
            )
            if has_session:
                from stablewalk.ui.viewers.dof_trajectory_3d import (
                    _ensure_trajectory_plot_legend,
                    relayout_single_dof_viewport,
                    setup_single_dof_trajectory_axes,
                )

                self.ax_dof_traj.cla()
                if hasattr(self.ax_dof_traj, "_stablewalk_traj_artists"):
                    del self.ax_dof_traj._stablewalk_traj_artists
                if hasattr(self.ax_dof_traj, "_stablewalk_stable_viewport"):
                    del self.ax_dof_traj._stablewalk_stable_viewport
                self.ax_dof_traj._stablewalk_plot_legend = None
                setup_single_dof_trajectory_axes(self.ax_dof_traj)
                self.ax_dof_traj.text2D(
                    0.5,
                    0.5,
                    EMPTY_SELECT_DOF_CHART,
                    transform=self.ax_dof_traj.transAxes,
                    ha="center",
                    va="center",
                    color=MUTED,
                    fontsize=10,
                )
                _ensure_trajectory_plot_legend(self.ax_dof_traj)
                relayout_single_dof_viewport(self.ax_dof_traj)
                title_lbl = getattr(self, "lbl_joint_movement_title", None)
                if title_lbl is not None:
                    title_lbl.configure(text=EMPTY_SELECT_DOF_CHART)
                self._render_dof_traj_canvas(force=force_draw)
            else:
                self._clear_selected_point_analysis_summary()
            return

        self._sync_dof_analysis_panel_state()
        if not (self.skeleton_player and self.skeleton_player.state.playing):
            self.root.update_idletasks()

        recording = self._analysis_motion_recording()
        end_frame_float = 0.0
        tip_snapshot = None

        if self.skeleton_player and recording is not None:
            end_frame_float = self.skeleton_player.state.frame_float
            tip_snapshot = self.skeleton_player.current_snapshot()

        projection_mode = "3D"
        var_proj = getattr(self, "var_dof_projection", None)
        if var_proj is not None:
            projection_mode = var_proj.get() or "3D"

        display_mode = "Current progress"
        var_display = getattr(self, "var_dof_traj_display", None)
        if var_display is not None:
            display_mode = var_display.get() or "Current progress"

        coord_mode = "Root-relative"
        var_coord = getattr(self, "var_dof_coord_mode", None)
        if var_coord is not None:
            coord_mode = var_coord.get() or "Root-relative"

        from stablewalk.ui.viewers.dof_trajectory_3d import normalize_coord_mode

        motion_series = (
            self._motion_frame_series()
            if normalize_coord_mode(coord_mode) == "Global"
            else None
        )
        body_ref_var = getattr(self, "var_dof_show_body_reference", None)
        show_body_reference = bool(
            body_ref_var.get() if body_ref_var is not None else False
        )

        draw_key = (
            item_id,
            projection_mode,
            display_mode,
            coord_mode,
            show_body_reference,
        )
        clear_axes = force_draw or getattr(self, "_traj_draw_cache_key", None) != draw_key
        self._traj_draw_cache_key = draw_key

        self._ensure_dof_traj_axes(projection_mode)

        from stablewalk.ui.viewers.dof_trajectory_3d import (
            draw_dof_trajectory_panel,
            relayout_single_dof_viewport,
        )

        _drawn, _progression_status, path = draw_dof_trajectory_panel(
            self.ax_dof_traj,
            recording,
            item_id,
            projection_mode=projection_mode,
            end_frame_float=end_frame_float,
            tip_snapshot=tip_snapshot,
            clear=clear_axes,
            display_mode=display_mode,
            coord_mode=coord_mode,
            motion_series=motion_series,
            show_body_reference=show_body_reference,
        )

        # Full-session timed path for scientific velocity / distance KPIs.
        try:
            from stablewalk.ui.dof_selection import anchor_joint_for_item
            from stablewalk.ui.viewers.dof_trajectory_3d import (
                _joint_path_with_times,
                stature_display_scale,
            )

            jid = anchor_joint_for_item(item_id)
            if jid and recording is not None and recording.frame_count > 0:
                self._path_science_path_times = _joint_path_with_times(
                    recording,
                    jid,
                    float(recording.frame_count - 1),
                    coord_mode=coord_mode,
                    motion_series=motion_series,
                    position_scale=float(stature_display_scale(recording)),
                )
            else:
                self._path_science_path_times = None
        except Exception:
            self._path_science_path_times = None

        if projection_mode == "3D":
            relayout_single_dof_viewport(self.ax_dof_traj)
            from stablewalk.ui.viewers.dof_trajectory_3d import _style_single_dof_trajectory_ticks

            _style_single_dof_trajectory_ticks(self.ax_dof_traj)

        self._update_dof_trajectory_live_info(
            item_id,
            tip_snapshot,
            end_frame_float=end_frame_float,
            coord_mode=coord_mode,
            motion_series=motion_series,
        )
        self._update_dof_trajectory_interpretation(
            item_id,
            path,
            projection_mode=projection_mode,
        )

        self._refresh_selected_point_analysis_summary(
            item_id,
            tip_snapshot,
            end_frame_float=end_frame_float,
        )

        self._render_dof_traj_canvas(force=force_draw)
        if refresh_overview:
            self._refresh_overview_trajectory_dock(force_draw=force_draw)
        self._schedule_dof_traj_reflow()
        from stablewalk.ui.tk.dashboard_layout import _hide_trajectory_debug_placeholder

        _hide_trajectory_debug_placeholder(self)
        self._traj_startup_test_drawn = True
        self._print_trajectory_runtime_debug(item_id, path, end_frame_float)
        self._log_motion_trajectory_debug(item_id, path, end_frame_float)

    def _print_trajectory_runtime_debug(self, item_id: str | None, path: list, end_frame_float: float) -> None:
        """Print trajectory binding diagnostics for Motion Analysis debugging."""
        import os

        debug_on = os.environ.get("STABLEWALK_TRAJ_DEBUG", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        # Avoid update_idletasks (forces layout) unless diagnostics are requested.
        if not debug_on and not logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
            return

        from stablewalk.ui.dof_selection import label_for_item

        motion = getattr(self, "_tab_motion", None)
        traj = getattr(self, "traj_panel", None)
        canvas_tw = None
        canvas = getattr(self, "canvas_dof_traj", None)
        if canvas is not None:
            canvas_tw = canvas.get_tk_widget()

        def _size(w) -> str:
            if w is None:
                return "0 x 0"
            try:
                w.update_idletasks()
                return f"{int(w.winfo_width())} x {int(w.winfo_height())}"
            except Exception:
                return "? x ?"

        joint_label = label_for_item(item_id) if item_id else "—"
        lines = [
            f"Selected joint: {joint_label}",
            f"Current frame: {int(end_frame_float)}",
            f"Valid trajectory samples: {len(path)}",
            f"Motion Analysis size: {_size(motion)}",
            f"Trajectory frame size: {_size(traj)}",
            f"Canvas size: {_size(canvas_tw)}",
        ]
        if path:
            lines.append(
                f"Trajectory frame indexes: 0-{len(path) - 1} (through frame {int(end_frame_float)})"
            )
        message = "\n".join(lines)
        logging.getLogger(__name__).debug("Motion trajectory runtime:\n%s", message)
        if debug_on:
            print(message, flush=True)

    def _log_motion_trajectory_debug(
        self,
        item_id: str,
        path: list,
        end_frame_float: float,
    ) -> None:
        """Emit trajectory binding diagnostics when Motion Analysis is active."""
        import logging
        import os

        debug_on = os.environ.get("STABLEWALK_GUI_DEBUG", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if not debug_on and not logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
            return

        from stablewalk.ui.dof_selection import label_for_item

        recording = self._analysis_motion_recording()
        if recording is None:
            return

        xs = [p.x for p in path] if path else []
        ys = [p.y for p in path] if path else []
        zs = [p.z for p in path] if path else []
        lines = [
            f"Selected joint: {label_for_item(item_id)}",
            f"Total analyzed frames: {recording.frame_count}",
            f"Valid 3D samples: {len(path)}",
        ]
        if xs:
            lines.extend(
                [
                    f"Trajectory min X/max X: {min(xs):.4f} / {max(xs):.4f}",
                    f"Trajectory min Y/max Y: {min(ys):.4f} / {max(ys):.4f}",
                    f"Trajectory min Z/max Z: {min(zs):.4f} / {max(zs):.4f}",
                ]
            )
        lines.extend(
            [
                f"Current frame: {int(end_frame_float)}",
                f"Displayed trajectory samples: {len(path)}",
            ]
        )
        message = "\n".join(lines)
        logging.getLogger(__name__).debug("Motion trajectory debug:\n%s", message)
        if debug_on:
            print(message, flush=True)

    def _refresh_dof_position_table(self) -> None:
        """Update the position table for the current frame (no throttling)."""
        if not hasattr(self, "dof_pos_tree"):
            return
        if self.playing:
            # History is recorded on the play tick; Treeview stays frozen.
            return
        if not self.skeleton_player:
            self._clear_dof_position_table()
            return

        snap = self.skeleton_player.current_snapshot()
        if not snap:
            if not getattr(self, "_playback_sync_immediate", False):
                self._refresh_selected_dof_trajectory_3d(refresh_overview=False)
            self._update_dashboard_empty_states()
            return

        if not self.selection.selected:
            self._clear_dof_position_table()
            self._update_dashboard_empty_states()
            return

        recording = self.skeleton_player.recording
        next_snap = snapshot_for_next_frame(recording, snap)
        playing = self.playing and self.skeleton_player.state.playing
        track_history = self._should_record_dof_table_history()
        charted_id = self._charted_dof_item_id()
        emphasize_dof = label_for_item(charted_id) if charted_id else None
        display_ids = self._table_item_ids_for_display()

        if track_history:
            if playing:
                self._append_dof_table_history_tick()
            if self._dof_table_history.rows:
                rows = self._filtered_position_table_rows(
                    list(self._dof_table_history.rows)
                )
                self._fill_dof_position_table(
                    rows,
                    scroll_to_bottom=playing,
                    emphasize_dof=emphasize_dof,
                )
            else:
                rows = table_rows_for_selection(
                    snap,
                    display_ids,
                    next_snapshot=next_snap,
                    recording=recording,
                )
                self._fill_dof_position_table(
                    rows,
                    emphasize_dof=emphasize_dof,
                )
        else:
            rows = table_rows_for_selection(
                snap,
                display_ids,
                next_snapshot=next_snap,
                recording=recording,
            )
            self._fill_dof_position_table(
                rows,
                emphasize_dof=emphasize_dof,
            )

        # Trajectory docks are owned by ``_sync_all_to_frame`` during playback
        # (and by selection/tab handlers when idle). Refreshing them here doubled
        # the expensive 3D path rebuild on every heavy tick.
        if not getattr(self, "_playback_sync_immediate", False):
            self._refresh_selected_dof_trajectory_3d(refresh_overview=False)
        self._configure_dof_table_columns()
        self._update_dof_table_controls_state()
        self._update_dashboard_empty_states()

    def _maybe_refresh_dof_table(self, *, force: bool = False) -> None:
        """Legacy entry — delegates to unified real-time refresh when due."""
        if not self._realtime_refresh_due(force=force):
            return
        snap = self.skeleton_player.current_snapshot() if self.skeleton_player else None
        self._refresh_realtime_analysis(snapshot=snap, force_draw=force)

    def _clear_dof_selection(self) -> None:
        self.selection.clear()
        self._active_joint = None
        self._notify_dof_selection_changed()

    def _add_checkpoint(self) -> None:
        """Save the current frame for each selected body point (position, velocity, angle)."""
        if not hasattr(self, "checkpoint_tree"):
            return
        if not self.skeleton_player:
            self.lbl_checkpoint_status.configure(
                text="Load gait data first, then add a point."
            )
            return
        if not self.selection.selected:
            self.lbl_checkpoint_status.configure(
                text="Select one or more body points first, then add a checkpoint."
            )
            return

        snap = self.skeleton_player.current_snapshot()
        if not snap:
            return

        from stablewalk.ui.dof_position_table import table_row_cell

        rows = table_rows_for_selection(snap, self.selection.selected)
        if not rows:
            self.lbl_checkpoint_status.configure(
                text="No data for the current selection at this frame."
            )
            return

        added = 0
        for row in rows:
            idx = str(len(self._checkpoints) + 1)
            self._checkpoints.append(
                (
                    idx,
                    table_row_cell(row, "frame"),
                    table_row_cell(row, "time"),
                    table_row_cell(row, "dof"),
                    table_row_cell(row, "x"),
                    table_row_cell(row, "y"),
                    table_row_cell(row, "z"),
                    table_row_cell(row, "speed"),
                    table_row_cell(row, "foot_clearance"),
                )
            )
            added += 1

        self._refresh_checkpoint_table()
        labels = ", ".join(sorted({r[2] for r in rows}))
        self.lbl_checkpoint_status.configure(
            text=f"Saved {added} point(s) at frame {rows[0][1]} ({rows[0][0]}s): {labels}"
        )

    def _clear_checkpoints(self) -> None:
        self._checkpoints.clear()
        self._refresh_checkpoint_table()
        self.lbl_checkpoint_status.configure(text="No points added yet")

    def _refresh_checkpoint_table(self) -> None:
        if not hasattr(self, "checkpoint_tree"):
            return
        for iid in self.checkpoint_tree.get_children():
            self.checkpoint_tree.delete(iid)
        for i, row in enumerate(self._checkpoints):
            tag = "even" if i % 2 == 0 else "odd"
            self.checkpoint_tree.insert("", tk.END, values=row, tags=(tag,))
        has_points = bool(self._checkpoints)
        self._set_panel_data_visible(
            self.checkpoint_tree, has_points, self.lbl_checkpoint_empty
        )

    def _ensure_default_dof_selection(self) -> None:
        """Auto-select a body point so the 3D movement graph shows by default.

        Selection is otherwise entirely manual, which means a freshly loaded
        clip lands on the "Select a body point" placeholder and the 3D cube is
        hidden. Pre-selecting a sensible default (Right Hip) keeps the graph,
        floor-distance readout and position table populated out of the box.
        """
        if not hasattr(self, "_dof_checkbox_vars"):
            return
        if self.selection.selected:
            return
        default_item = (
            "right_knee"
            if "right_knee" in self._dof_checkbox_vars
            else (
                "right_hip"
                if "right_hip" in self._dof_checkbox_vars
                else next(iter(GUI_DOF_ITEM_IDS), None)
            )
        )
        if not default_item or default_item not in self._dof_checkbox_vars:
            return
        self._dof_checkbox_vars[default_item].set(True)
        self._on_dof_checkbox_changed(default_item)

    def _on_dof_item_focus(self, item_id: str) -> None:
        """Focus analysis on a body point (select if needed, or switch among multi-select)."""
        if item_id not in self.selection.selected:
            self._dof_checkbox_vars[item_id].set(True)
            self._on_dof_checkbox_changed(item_id)
            return
        self._activate_dof_item(item_id)

    def _on_dof_checkbox_changed(self, item_id: str) -> None:
        if self._dof_list_syncing:
            return
        want_selected = self._dof_checkbox_vars[item_id].get()
        is_selected = item_id in self.selection.selected
        if want_selected and not is_selected:
            self.selection.activate_item(item_id)
        elif want_selected and is_selected:
            self._activate_dof_item(item_id)
            return
        elif not want_selected and is_selected:
            self.selection.selected.discard(item_id)
            if self.selection.active_item_id == item_id:
                self.selection.last_selected = next(
                    (i for i in GUI_DOF_ITEM_IDS if i in self.selection.selected),
                    None,
                )
            self.selection.ensure_last_selected()
        else:
            self._sync_dof_checkboxes()
            return
        self._notify_dof_selection_changed(
            lightweight=bool(getattr(self, "playing", False))
        )

    def _bind_overview_path_stat_labels(self) -> None:
        """Point the 3D path metrics at the compact Overview statistics labels."""
        ax = getattr(self, "ax_dof_traj_overview", None)
        if ax is None:
            return
        ax._stablewalk_suppress_hud = True  # type: ignore[attr-defined]
        ax._stablewalk_hud_label = None  # type: ignore[attr-defined]
        ax._stablewalk_path_stat_labels = getattr(  # type: ignore[attr-defined]
            self, "overview_path_metric_values", None
        )

    def _ensure_overview_traj_figure(self) -> bool:
        """Lazily create the Overview 3D trajectory figure/axes/canvas if missing."""
        if (
            getattr(self, "fig_dof_traj_overview", None) is not None
            and getattr(self, "ax_dof_traj_overview", None) is not None
            and getattr(self, "canvas_dof_traj_overview", None) is not None
        ):
            self._bind_overview_path_stat_labels()
            return True
        host = getattr(self, "overview_traj_canvas_host", None)
        panel = getattr(self, "overview_traj_panel", None)
        if host is None or panel is None:
            return False
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure

            from stablewalk.ui.theme import EMPTY_OVERVIEW_JOINT_INSPECT, PANEL
            from stablewalk.ui.tk.dashboard_layout import _ensure_trajectory_canvas_gridded
            from stablewalk.ui.viewers.dof_trajectory_3d import (
                setup_single_dof_trajectory_axes,
            )

            fig = Figure(figsize=(3.2, 2.8), dpi=100, facecolor=PANEL)
            ax = fig.add_subplot(111, projection="3d")
            ax._stablewalk_overview_dock = True
            ax._stablewalk_overview_cm_ticks = True
            ax._stablewalk_overview_use_progress_viewport = False
            ax._stablewalk_suppress_hud = True
            ax._stablewalk_hud_label = None
            ax._stablewalk_path_stat_labels = getattr(
                self, "overview_path_metric_values", None
            )
            setup_single_dof_trajectory_axes(ax)
            ax.text2D(
                0.5,
                0.5,
                EMPTY_OVERVIEW_JOINT_INSPECT,
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=MUTED,
                fontsize=10,
            )
            canvas = FigureCanvasTkAgg(fig, master=host)
            canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
            _ensure_trajectory_canvas_gridded(canvas, overview_dock=True)
            self.fig_dof_traj_overview = fig
            self.ax_dof_traj_overview = ax
            self.canvas_dof_traj_overview = canvas
            self._bind_overview_path_stat_labels()
            try:
                from stablewalk.ui.tk.dashboard_layout import _bind_trajectory_figure_resize
                from stablewalk.ui.tk.trajectory_camera_toolbar import (
                    bind_trajectory_camera_canvas,
                )

                bind_trajectory_camera_canvas(
                    self,
                    canvas_attr="canvas_dof_traj_overview",
                    ax_attr="ax_dof_traj_overview",
                )
                _bind_trajectory_figure_resize(
                    canvas,
                    fig,
                    ax,
                    graph_host=host,
                    extra_hosts=(
                        panel,
                        host,
                        getattr(self, "overview_traj_footer", None),
                    ),
                )
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _render_overview_traj_canvas(self, *, force: bool = False) -> None:
        """Fit and paint the Overview-tab trajectory dock canvas."""
        if not self._ensure_overview_traj_figure():
            return
        from stablewalk.ui.tk.dashboard_layout import (
            _ensure_trajectory_canvas_gridded,
            _fit_trajectory_figure,
        )

        host = getattr(self, "overview_traj_canvas_host", None)
        if host is None:
            return
        # Avoid layout storms during playback; resize handlers already reflow.
        if force or not self.playing:
            try:
                host.update_idletasks()
            except tk.TclError:
                pass
        _ensure_trajectory_canvas_gridded(
            self.canvas_dof_traj_overview, overview_dock=True
        )
        fitted = _fit_trajectory_figure(
            self.canvas_dof_traj_overview,
            self.fig_dof_traj_overview,
            self.ax_dof_traj_overview,
            graph_host=host,
        )
        if fitted or force or self.playing:
            self._paint_canvas_now(self.canvas_dof_traj_overview, idle=True)

    def _overview_joint_inspect_title(self) -> str:
        """Auto title: ``Normal — Right Hip 3D Path`` (or comparison form)."""
        from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS, label_for_item

        demo = getattr(self, "_active_demo_gait", None)
        prefix = f"{demo.display_name} — " if demo is not None else ""
        selected = [
            item for item in GUI_DOF_ITEM_IDS if item in self.selection.selected
        ]
        if not selected:
            return "Joint Inspection"
        if len(selected) == 1:
            label = label_for_item(selected[0]) or "Joint"
            return f"{prefix}{label} 3D Path"
        # Prefer the active analysis point first in comparison titles.
        active = None
        if hasattr(self, "_active_dof_item_id"):
            try:
                active = self._active_dof_item_id()
            except Exception:
                active = getattr(self.selection, "active_item_id", None)
        else:
            active = getattr(self.selection, "active_item_id", None)
        ordered = []
        if active and active in selected:
            ordered.append(active)
        ordered.extend(i for i in selected if i not in ordered)
        labels = [label_for_item(i) or i for i in ordered[:3]]
        if len(selected) > 3:
            labels.append(f"+{len(selected) - 3}")
        return f"{prefix}{' vs '.join(labels)} 3D Path"

    def _overview_traj_unavailable_message(
        self,
        item_id: str | None,
        *,
        reason: str,
    ) -> str:
        from stablewalk.ui.dof_selection import label_for_item

        label = label_for_item(item_id) if item_id else "this joint"
        return (
            f"No 3D trajectory data is available for {label}.\n\n"
            f"Reason: {reason}"
        )

    def _diagnose_overview_trajectory(
        self,
        item_id: str | None,
        recording,
    ) -> str | None:
        """Return a human-readable reason when no drawable 3D path exists.

        Selected-DOF emptiness is not fatal — the renderer falls back to the
        pelvis path. Only report unavailable when that fallback also fails.
        """
        from stablewalk.ui.viewers.dof_trajectory_3d import (
            resolve_trajectory_draw_target,
            stature_display_scale,
        )

        if recording is None:
            return "analysis not completed"
        frame_count = int(getattr(recording, "frame_count", 0) or 0)
        if frame_count <= 0:
            return "invalid frame count"
        path_scale = stature_display_scale(recording)
        joint_id, _coord_mode, _fallback = resolve_trajectory_draw_target(
            recording,
            item_id,
            end_frame_float=float(frame_count - 1),
            coord_mode="Root-relative",
            motion_series=getattr(self, "_motion_series", None),
            position_scale=path_scale,
        )
        if not joint_id:
            return "joint name mapping failed" if not item_id else "trajectory array missing"
        return None

    def _paint_overview_traj_message(self, message: str) -> None:
        """Show a status message on a fully set-up 3D scene (never a blank cube)."""
        from stablewalk.ui.viewers.dof_trajectory_3d import (
            draw_single_dof_trajectory_3d,
            relayout_single_dof_viewport,
            setup_single_dof_trajectory_axes,
        )

        if not self._ensure_overview_traj_figure():
            return
        ax = self.ax_dof_traj_overview
        recording = self._analysis_motion_recording()
        # Prefer a pelvis path under the message so the scene is never empty.
        drew = False
        if recording is not None and getattr(recording, "frame_count", 0) > 0:
            try:
                end_f = float(max(0, recording.frame_count - 1))
                ok, _status = draw_single_dof_trajectory_3d(
                    ax,
                    recording,
                    None,
                    end_frame_float=end_f,
                    clear=True,
                    display_mode="Current progress",
                    coord_mode="Root-relative",
                    motion_series=getattr(self, "_motion_series", None),
                )
                drew = bool(ok)
            except Exception:
                drew = False
        if not drew:
            ax.cla()
            if hasattr(ax, "_stablewalk_traj_artists"):
                del ax._stablewalk_traj_artists
            if hasattr(ax, "_stablewalk_stable_viewport"):
                del ax._stablewalk_stable_viewport
            ax._stablewalk_plot_legend = None
            ax._stablewalk_tip_xyz_m = None
            ax._stablewalk_tip_xyz_cm = None
            ax._stablewalk_path_travel_m = None
            ax._stablewalk_path_bounds_m = None
            setup_single_dof_trajectory_axes(ax)
        ax.text2D(
            0.5,
            0.52 if drew else 0.5,
            message,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )
        relayout_single_dof_viewport(ax)
        self._paint_canvas_now(self.canvas_dof_traj_overview, idle=True)

    def update_joint_3d_graph(self, joint_id: str | None = None) -> None:
        """Public entry: rebuild/redraw the Overview 3D joint-path graph."""
        if joint_id:
            from stablewalk.ui.dof_selection import normalize_gui_dof_id

            item_id = normalize_gui_dof_id(joint_id)
            if item_id and item_id not in self.selection.selected:
                self.selection.select_only(item_id)
        self._refresh_overview_trajectory_dock(force_draw=True)
        self._render_overview_traj_canvas(force=True)

    def show_joint_3d_panel(self) -> None:
        """Public entry: make the Overview 3D joint-path panel visible."""
        self._show_overview_trajectory_dock(True)

    def _refresh_overview_joint_information(
        self, item_id: str | None = None
    ) -> None:
        """Update the live Joint Information panel to the current playback frame."""
        active = item_id if item_id is not None else self._active_dof_item_id()
        # While playing: Coordinate Inspector only (tip path owns that). Skip
        # Joint Information / Live Metrics chrome until pause.
        if getattr(self, "playing", False):
            try:
                self._refresh_overview_trajectory_readout(active)
            except Exception:
                pass
            return
        try:
            self._refresh_overview_trajectory_readout(active)
        except Exception:
            pass
        value_labels = getattr(self, "overview_joint_info_values", None) or {}
        live_labels = getattr(self, "skeleton_live_info_values", None) or {}
        if not value_labels and not live_labels:
            return

        from stablewalk.ui.joint_information import (
            build_joint_information,
            empty_joint_information,
        )

        recording = self._analysis_motion_recording()
        snapshot = None
        end_frame_float: float | None = None
        if self.skeleton_player is not None:
            try:
                from stablewalk.ui.tk.dashboard_status_bar import playhead_frame_0based

                frame_i = playhead_frame_0based(self)
                if frame_i is None:
                    frame_i = int(self.skeleton_player.state.frame_index)
                end_frame_float = float(frame_i)
                if recording is not None:
                    snapshot = recording.snapshot_at(frame_i)
                if snapshot is None:
                    snapshot = self.skeleton_player.current_snapshot()
            except Exception:
                snapshot = None
        if snapshot is None and recording is not None and recording.frame_count > 0:
            end_frame_float = float(max(0, recording.frame_count - 1))
            snapshot = recording.snapshot_at(int(end_frame_float))

        gait_phase = None
        phase_lbl = getattr(self, "lbl_gait_card_phase_value", None)
        if phase_lbl is not None:
            try:
                gait_phase = str(phase_lbl.cget("text") or "").strip()
            except tk.TclError:
                gait_phase = None

        if not active or not self.selection.selected:
            info = empty_joint_information()
            hint_text = (
                "Select a joint on the skeleton — values update with video playback."
            )
        else:
            from stablewalk.ui.value_provenance import get_provenance_bundle

            sink: dict = {}
            info = build_joint_information(
                active,
                snapshot,
                recording=recording,
                sequence=self.sequence,
                gait_phase=gait_phase,
                end_frame_float=end_frame_float,
                provenance_sink=sink,
            )
            bundle = get_provenance_bundle(self)
            for key, prov in sink.items():
                bundle.put(key, prov)
            hint_text = "Live with video playback · synchronized to current frame"

        fields = info.as_field_map()
        # Session ROM for the live strip (same source as former video HUD).
        if active:
            try:
                fields["rom"] = self._overview_hud_selected_rom(active)
            except Exception:
                fields["rom"] = fields.get("rom", "—")
        phase = fields.get("gait_phase", "—")
        contact = fields.get("contact", "—")
        if phase not in ("—", "", "N/A") and "not available" not in phase.lower():
            fields["phase_or_contact"] = phase
        elif contact not in ("—", "", "N/A") and "not available" not in contact.lower():
            fields["phase_or_contact"] = contact
        else:
            fields["phase_or_contact"] = "—"

        last_fields = getattr(self, "_last_joint_info_fields", None)
        if last_fields != fields:
            self._last_joint_info_fields = dict(fields)
            for key, lbl in value_labels.items():
                text = fields.get(key, "—")
                try:
                    if last_fields is None or last_fields.get(key) != text:
                        lbl.configure(text=text, fg=TEXT if text != "—" else MUTED)
                except tk.TclError:
                    pass
            # Horizontal Live Joint Metrics under the visualization workspace.
            for key in (
                "joint_name",
                "frame",
                "time",
                "angle",
                "rom",
                "x",
                "y",
                "z",
                "confidence",
                "phase_or_contact",
            ):
                lbl = live_labels.get(key)
                if lbl is None:
                    continue
                text = fields.get(key, "—")
                try:
                    lbl.configure(
                        text=text,
                        fg=TEXT if text not in ("—", "") else MUTED,
                    )
                except tk.TclError:
                    pass

        hint = getattr(self, "lbl_overview_joint_info_hint", None)
        if hint is not None:
            try:
                if getattr(self, "_last_joint_info_hint", None) != hint_text:
                    hint.configure(text=hint_text)
                    self._last_joint_info_hint = hint_text
            except tk.TclError:
                pass

    def _sync_overview_traj_playhead_labels(self) -> None:
        """Keep Overview path readout locked to the shared playhead."""
        item_id = self._active_dof_item_id()
        if not item_id or not self.selection.selected:
            return
        # Throttle Joint Information during light play ticks (heavy ticks refresh fully).
        if self.playing:
            now = time.monotonic()
            interval = float(getattr(self, "_joint_info_play_interval_s", 0.10) or 0.10)
            if (now - float(getattr(self, "_last_joint_info_play_ts", 0.0))) < interval:
                return
            self._last_joint_info_play_ts = now
        try:
            self._refresh_overview_joint_information(item_id)
        except Exception:
            pass

    def _refresh_overview_trajectory_readout(self, item_id: str | None) -> None:
        """Fill the Overview 3D Coordinate Inspector from cached tip / joint data."""
        value_labels = getattr(self, "overview_traj_readout_values", None) or getattr(
            self, "overview_coordinate_inspector_values", None
        )
        if not value_labels:
            return

        from stablewalk.ui.overview_coordinate_inspector import (
            apply_coordinate_inspector_labels,
            build_coordinate_inspector_fields,
            tip_xyz_cm_from_cache,
        )
        from stablewalk.ui.playback_state import read_playback_state
        from stablewalk.ui.value_provenance import ValueProvenance, get_provenance_bundle

        recording = self._analysis_motion_recording()
        state = read_playback_state(self)
        snapshot = None
        list_pos: int | None = None
        if state is not None:
            list_pos = int(state.current_frame_index)
        if self.skeleton_player is not None:
            try:
                from stablewalk.ui.tk.dashboard_status_bar import playhead_frame_0based

                frame_i = playhead_frame_0based(self)
                if frame_i is None:
                    frame_i = int(self.skeleton_player.state.frame_index)
                list_pos = int(frame_i)
                if recording is not None:
                    snapshot = recording.snapshot_at(frame_i)
                if snapshot is None:
                    snapshot = self.skeleton_player.current_snapshot()
            except Exception:
                snapshot = None
        if snapshot is None and recording is not None and recording.frame_count > 0:
            list_pos = max(0, recording.frame_count - 1)
            snapshot = recording.snapshot_at(int(list_pos))

        active = item_id if item_id and self.selection.selected else None
        tip_cm = tip_xyz_cm_from_cache(getattr(self, "ax_dof_traj_overview", None))
        rom_display = (
            self._overview_hud_selected_rom(active) if active else None
        )
        gait_phase_display = None
        cache = getattr(self, "_session_analysis", None)
        if cache is not None and list_pos is not None:
            snap_disp = cache.frame_snapshot(int(list_pos))
            if snap_disp is not None:
                gait_phase_display = (
                    getattr(snap_disp, "cycle_phase_label", None)
                    or getattr(snap_disp, "phase_dashboard", None)
                )
        fields = build_coordinate_inspector_fields(
            active,
            snapshot=snapshot,
            recording=recording,
            sequence=self.sequence,
            playback_state=state,
            tip_xyz_cm=tip_cm,
            session_rom_display=rom_display,
            list_pos=list_pos,
            gui=self,
            gait_phase_display=gait_phase_display,
        )

        if tip_cm is not None:
            bundle = get_provenance_bundle(self)
            for axis, val in zip(("x", "y", "z"), tip_cm):
                existing = bundle.get(f"position_{axis}")
                bundle.put(
                    f"position_{axis}",
                    ValueProvenance(
                        metric_name=f"{axis.upper()} position (path tip)",
                        display_value=fields.get(axis, "N/A"),
                        raw_value=float(val),
                        units="cm (stature-scaled, root-relative)",
                        source_frame=list_pos,
                        source_function="app._refresh_overview_trajectory_readout",
                        source_array="ax._stablewalk_tip_xyz_cm",
                        coordinate_system="Root-relative (pelvis origin); not world",
                        filtering=existing.filtering if existing else "path tip cache",
                        confidence=existing.confidence if existing else None,
                        scope="instantaneous",
                        tier="estimated",
                    ),
                )

        family_colors = getattr(self, "_coord_inspector_family_colors", None)
        if not family_colors:
            readout = getattr(self, "overview_traj_readout", None)
            family_colors = getattr(readout, "_coord_inspector_family_colors", None)
        if not family_colors:
            from stablewalk.ui.theme import ACCENT_ALT, SUCCESS_SOFT, WARNING

            family_colors = {
                "identity": TEXT,
                "position": ACCENT_ALT,
                "motion": WARNING,
                "quality": SUCCESS_SOFT,
            }
        from stablewalk.ui.theme import (
            COORDINATE_INSPECTOR_LIVE_HIGHLIGHT,
            SELECTED_JOINT_ACCENT,
        )

        force_recolor = bool(getattr(self, "_coord_inspector_force_recolor", False))
        self._coord_inspector_force_recolor = False
        apply_kwargs = dict(
            fields=fields,
            text_fg=TEXT,
            muted_fg=MUTED,
            family_colors=family_colors,
            highlight_fg=COORDINATE_INSPECTOR_LIVE_HIGHLIGHT,
            joint_selected=bool(active),
            selected_accent=SELECTED_JOINT_ACCENT,
            force_recolor=force_recolor,
        )
        self._last_overview_readout_fields = apply_coordinate_inspector_labels(
            value_labels,
            last_fields=getattr(self, "_last_overview_readout_fields", None),
            **apply_kwargs,
        )
        window_labels = getattr(self, "_coord_inspector_window_values", None)
        if window_labels:
            self._last_inspector_window_fields = apply_coordinate_inspector_labels(
                window_labels,
                last_fields=getattr(self, "_last_inspector_window_fields", None),
                **apply_kwargs,
            )
            try:
                from stablewalk.ui.tk.coordinate_inspector_window import (
                    sync_coordinate_inspector_window,
                )

                sync_coordinate_inspector_window(self, frame_index=list_pos)
            except Exception:
                pass
        self._refresh_overview_path_extents()
        try:
            from stablewalk.ui.tk.skeleton_model_inspector_window import (
                refresh_skeleton_model_inspector,
            )

            refresh_skeleton_model_inspector(self)
        except Exception:
            pass

    def _refresh_overview_path_extents(self) -> None:
        """Sync Path panel Bounding Box / XYZ ranges with the cached trajectory."""
        value_labels = getattr(self, "overview_path_extent_values", None)
        if not value_labels:
            return
        from stablewalk.ui.overview_coordinate_inspector import (
            apply_path_extent_labels,
            build_path_extent_fields,
        )

        fields = build_path_extent_fields(
            getattr(self, "ax_dof_traj_overview", None)
        )
        self._last_overview_path_extent_fields = apply_path_extent_labels(
            value_labels,
            fields,
            text_fg=TEXT,
            muted_fg=MUTED,
            last_fields=getattr(self, "_last_overview_path_extent_fields", None),
        )

    def _refresh_overview_trajectory_dock(self, *, force_draw: bool = False) -> None:
        """Mirror the active joint path into the Overview trajectory dock."""
        item_id = self._active_dof_item_id()
        # Keep Joint Information in lockstep with playback even if the path
        # canvas is not ready yet.
        self._refresh_overview_joint_information(item_id)

        if not getattr(self, "_overview_traj_dock_visible", False):
            return
        if not self._ensure_overview_traj_figure():
            return

        traj_panel = getattr(self, "overview_traj_panel", None)
        title_lbl = getattr(self, "lbl_overview_traj_title", None)
        if traj_panel is not None or title_lbl is not None:
            try:
                title = self._overview_joint_inspect_title()
                if title_lbl is not None:
                    if str(title_lbl.cget("text")) != title:
                        title_lbl.configure(text=title)
                elif str(traj_panel.cget("text")) != title:
                    traj_panel.configure(text=title)
            except tk.TclError:
                pass

        recording = self._analysis_motion_recording()
        has_session = bool(
            (
                self.skeleton_player
                and getattr(self.skeleton_player, "frame_count", 0) > 0
            )
            or (recording is not None and getattr(recording, "frame_count", 0) > 0)
        )
        # Auto-pick a DOF so the path dock is never stuck on "select a joint".
        if has_session and (not item_id or not self.selection.selected):
            try:
                self._ensure_default_dof_selection()
            except Exception:
                pass
            item_id = self._active_dof_item_id()

        if not has_session:
            self._paint_overview_traj_message(
                self._overview_traj_unavailable_message(
                    item_id, reason="analysis not completed"
                )
                if item_id
                else EMPTY_OVERVIEW_JOINT_INSPECT
            )
            for attr in (
                "lbl_overview_traj_metrics",
                "lbl_overview_traj_detail",
                "lbl_overview_traj_video",
                "lbl_overview_category_note",
            ):
                lbl = getattr(self, attr, None)
                if lbl is not None:
                    lbl.configure(text="")
            legend_lbl = getattr(self, "lbl_overview_traj_legend", None)
            if legend_lbl is not None:
                legend_lbl.configure(
                    text=(
                        "● Start  —  faded path = earlier motion  —  "
                        "● Current frame  —  floor plane"
                    )
                )
            self._render_overview_traj_canvas(force=force_draw)
            return

        unavailable = self._diagnose_overview_trajectory(item_id, recording)
        if unavailable is not None:
            self._paint_overview_traj_message(
                self._overview_traj_unavailable_message(
                    item_id, reason=unavailable
                )
                if item_id
                else EMPTY_OVERVIEW_JOINT_INSPECT
            )
            self._render_overview_traj_canvas(force=force_draw)
            return

        end_frame_float = 0.0
        tip_snapshot = None
        if self.skeleton_player and recording is not None:
            from stablewalk.ui.tk.dashboard_status_bar import playhead_frame_0based

            # Lock path tip + labels to the same discrete frame as the video HUD.
            frame_i = playhead_frame_0based(self)
            if frame_i is None:
                frame_i = int(self.skeleton_player.state.frame_index)
            end_frame_float = float(frame_i)
            tip_snapshot = recording.snapshot_at(frame_i)
            if tip_snapshot is None:
                tip_snapshot = self.skeleton_player.current_snapshot()
        elif recording is not None:
            end_frame_float = float(max(0, recording.frame_count - 1))

        projection_mode = "3D"
        var_proj = getattr(self, "var_dof_projection", None)
        if var_proj is not None:
            projection_mode = var_proj.get() or "3D"

        display_mode = "Current progress"

        # Pelvis-relative: shows the joint's own movement (not whole-body walking
        # translation, which inflates Global travel to hundreds of cm).
        # Renderer falls back to Global pelvis when the selected DOF has no path.
        coord_mode = "Root-relative"
        motion_series = getattr(self, "_motion_series", None)

        from stablewalk.ui.viewers.dof_trajectory_3d import (
            draw_dof_trajectories,
            draw_dof_trajectory_panel,
            relayout_single_dof_viewport,
            setup_single_dof_trajectory_axes,
        )

        view_type = getattr(self._biomech, "view_type", None) if self._biomech else None
        self.ax_dof_traj_overview._stablewalk_view_type = view_type
        body_ref_var = getattr(self, "var_dof_show_body_reference", None)
        show_body_reference = bool(
            body_ref_var.get() if body_ref_var is not None else False
        )
        compare = len(self.selection.selected) > 1

        # Reuse the existing figure. Clear axes only when the joint/mode
        # changes — never rebuild the Matplotlib figure on play/resize.
        draw_key = (
            item_id,
            frozenset(self.selection.selected),
            projection_mode,
            display_mode,
            coord_mode,
            show_body_reference,
            bool(compare),
        )
        clear_axes = getattr(self, "_overview_traj_draw_cache_key", None) != draw_key
        self._overview_traj_draw_cache_key = draw_key
        if clear_axes and hasattr(self.ax_dof_traj_overview, "_stablewalk_stable_viewport"):
            try:
                from stablewalk.ui.viewers.dof_trajectory_3d import clear_trajectory_axis_cache

                clear_trajectory_axis_cache(self.ax_dof_traj_overview)
            except Exception:
                pass

        if compare:
            # Multi-joint comparison: colored paths + legend.
            draw_dof_trajectories(
                self.ax_dof_traj_overview,
                recording,
                set(self.selection.selected),
                end_frame_float=end_frame_float,
                tip_snapshot=tip_snapshot,
                clear=clear_axes,
                show_body_reference=show_body_reference,
            )
        else:
            if clear_axes and projection_mode == "3D":
                setup_single_dof_trajectory_axes(self.ax_dof_traj_overview)
            draw_dof_trajectory_panel(
                self.ax_dof_traj_overview,
                recording,
                item_id,
                projection_mode=projection_mode,
                end_frame_float=end_frame_float,
                tip_snapshot=tip_snapshot,
                clear=clear_axes,
                display_mode=display_mode,
                coord_mode=coord_mode,
                motion_series=motion_series,
                show_body_reference=show_body_reference,
            )
            if projection_mode == "3D" and clear_axes:
                relayout_single_dof_viewport(self.ax_dof_traj_overview)

        metrics_lbl = getattr(self, "lbl_overview_traj_metrics", None)
        detail_lbl = getattr(self, "lbl_overview_traj_detail", None)
        video_lbl = getattr(self, "lbl_overview_traj_video", None)
        if compare:
            legend_lbl = getattr(self, "lbl_overview_traj_legend", None)
            if legend_lbl is not None:
                legend_lbl.configure(
                    text=(
                        "Comparison · colored paths = selected joints · "
                        "● Current frame · Ctrl+Click to add"
                    )
                )
        # Lightweight footer: tip XYZ only (skip Travel/ROM essay rebuilds).
        if metrics_lbl is not None:
            metrics_lbl.configure(text="")
        if detail_lbl is not None:
            detail_lbl.configure(text="")
        if video_lbl is not None:
            video_lbl.configure(text="")
        category_note = getattr(self, "lbl_overview_category_note", None)
        if category_note is not None:
            category_note.configure(text="")

        self._render_overview_traj_canvas(force=force_draw)
        try:
            self._refresh_overview_trajectory_readout(item_id)
        except Exception:
            pass

    def _set_overview_traj_column_weights(self, *, traj_active: bool) -> None:
        """Apply Video|Skeleton|Path weights without crushing Phase / Foot Clearance."""
        section1 = getattr(self, "_section_visual", None)
        if section1 is None:
            return
        _ = traj_active  # path column stays reserved
        from stablewalk.ui.tk.dashboard_sections import apply_overview_viz_grid_weights

        apply_overview_viz_grid_weights(section1, content_row=0)

    def _show_overview_trajectory_dock(self, show: bool) -> None:
        """Keep the 3D joint-path panel mapped and refresh its trajectory.

        The selected-joint path is required on Overview — hide requests are
        ignored so the trail never disappears during analysis or playback.
        """
        panel = getattr(self, "overview_traj_panel", None)
        if panel is None:
            return

        # Always keep the trajectory dock visible (user requirement).
        self._overview_traj_dock_visible = True
        _ = show  # hide requests are ignored

        def _sync_open_btn() -> None:
            btn = getattr(self, "_btn_overview_open_3d_path", None)
            if btn is None:
                return
            try:
                btn.configure(text="Refresh 3D Path")
            except tk.TclError:
                pass

        def _relayout_media() -> None:
            try:
                self.root.update_idletasks()
            except tk.TclError:
                pass
            try:
                self._on_video_label_resize()
            except Exception:
                pass
            fit_skel = getattr(self, "_fit_skeleton_canvas", None)
            if fit_skel is not None:
                try:
                    fit_skel()
                except Exception:
                    pass

        already_mapped = False
        try:
            already_mapped = bool(panel.winfo_ismapped())
        except tk.TclError:
            already_mapped = False

        # Fast path: path already mapped — refresh only.
        if already_mapped and getattr(self, "_overview_view_mode_active", False):
            try:
                self._set_overview_traj_column_weights(traj_active=True)
            except Exception:
                pass
            self._refresh_overview_trajectory_dock(force_draw=True)
            self._render_overview_traj_canvas(force=True)
            _sync_open_btn()
            return

        try:
            from stablewalk.ui.tk.dashboard_overview_panes import (
                layout_panels_for_view_mode,
            )
            from stablewalk.ui.tk.dashboard_overview_view_mode import (
                current_overview_view_mode,
            )

            layout_panels_for_view_mode(self, current_overview_view_mode(self))
            self._set_overview_traj_column_weights(traj_active=True)
        except Exception:
            from stablewalk.ui.tk.dashboard_sections import overview_panel_padx

            content_row = int(getattr(self, "_primary_viz_content_row", 0))
            try:
                panel.grid(
                    row=content_row,
                    column=2,
                    sticky="nsew",
                    padx=overview_panel_padx(2),
                    pady=0,
                )
            except tk.TclError:
                pass

        try:
            panel.update_idletasks()
            self.root.update_idletasks()
        except tk.TclError:
            pass
        _relayout_media()
        self._refresh_overview_trajectory_dock(force_draw=True)
        self._render_overview_traj_canvas(force=True)
        _sync_open_btn()

    def _sync_trajectory_mount_for_active_tab(self) -> None:
        """Reflow the Overview trajectory dock when its tab becomes active."""
        from stablewalk.ui.tk.dashboard_notebook import is_overview_tab_selected

        if (
            is_overview_tab_selected(self)
            and getattr(self, "_overview_traj_dock_visible", False)
        ):
            self._refresh_overview_trajectory_dock(force_draw=True)
            self._render_overview_traj_canvas(force=True)

    @staticmethod
    def _mouse_event_is_compare(mouse) -> bool:
        """True when Ctrl (or Cmd on macOS) is held during a skeleton click."""
        if mouse is None:
            return False
        key = str(getattr(mouse, "key", "") or "").lower()
        if key in ("control", "ctrl", "cmd", "meta") or key.startswith("control"):
            return True
        gui = getattr(mouse, "guiEvent", None)
        if gui is not None:
            try:
                state = int(getattr(gui, "state", 0) or 0)
            except (TypeError, ValueError):
                state = 0
            # Tk ControlMask = 0x4; Command often Mod1 = 0x8 on macOS.
            if state & 0x4:
                return True
            if sys.platform == "darwin" and (state & 0x8 or state & 0x10):
                return True
        return False

    @staticmethod
    def _pick_event_is_compare(event) -> bool:
        """True when Ctrl (or Cmd on macOS) is held during a skeleton pick."""
        return StableWalkGUI._mouse_event_is_compare(getattr(event, "mouseevent", None))

    def _expand_overview_joint_dof_analysis(self, item_id: str) -> None:
        """Open Joint Information / Details for the clicked DOF and refresh metrics."""
        body = getattr(self, "_overview_joint_info_body", None)
        toggle = getattr(self, "btn_overview_joint_info_toggle", None)
        self._overview_joint_info_expanded = True
        if body is not None:
            try:
                if not body.winfo_ismapped():
                    body.pack(fill=tk.X, pady=(PAD_XS, 0))
            except tk.TclError:
                pass
        if toggle is not None:
            try:
                toggle.configure(text="▼ Details")
            except tk.TclError:
                pass
        try:
            self._refresh_overview_joint_information(item_id)
        except Exception:
            pass
        try:
            self._sync_dof_analysis_panel_state()
        except Exception:
            pass
        try:
            self._refresh_realtime_analysis(force_draw=True)
        except Exception:
            pass
        try:
            self._refresh_dof_details()
        except Exception:
            pass

    def _focus_joint_trajectory_from_skeleton(
        self,
        item_id: str,
        *,
        compare: bool = False,
    ) -> None:
        """
        Skeleton click → select joint and show Overview 3D path (one click).

        Keeps Video + Skeleton + 3D Path visible (Side-by-Side). Does not open
        Joint Graphs / Joint Motion Analysis. Default replaces the selection;
        Ctrl+Click adds the joint for comparison.

        Must not pause playback: layout restores previously fired the frame
        slider and stopped the play timer.
        """
        from stablewalk.ui.dof_selection import (
            GUI_DOF_LABELS,
            label_for_item,
            normalize_gui_dof_id,
        )
        from stablewalk.ui.tk.dashboard_notebook import (
            TAB_OVERVIEW,
            select_dashboard_tab,
        )
        from stablewalk.ui.tk.dashboard_overview_view_mode import (
            VIEW_MODE_SIDE_BY_SIDE,
            apply_overview_view_mode,
            current_overview_view_mode,
        )

        item_id = normalize_gui_dof_id(item_id) or item_id
        if item_id not in GUI_DOF_LABELS:
            if hasattr(self, "status"):
                self.status.configure(
                    text=f"Joint click ignored — unknown id: {item_id!r}"
                )
            return

        was_playing = bool(getattr(self, "playing", False))

        if compare:
            self.selection.activate_item(item_id)
        else:
            self.selection.select_only(item_id)

        # Never auto-expand Joint Graphs — keep Video|Skeleton|Path as the default.
        apply_graphs = getattr(self, "_apply_overview_joint_motion_expanded", None)
        if callable(apply_graphs):
            try:
                apply_graphs(False)
            except Exception:
                pass

        select_dashboard_tab(self, TAB_OVERVIEW)

        # Remount only when leaving a non Side-by-Side mode (avoid play pause).
        need_layout = True
        try:
            need_layout = current_overview_view_mode(self) != VIEW_MODE_SIDE_BY_SIDE
        except Exception:
            need_layout = True
        if need_layout:
            try:
                apply_overview_view_mode(
                    self, VIEW_MODE_SIDE_BY_SIDE, animate=False, persist=False
                )
            except Exception:
                pass

        # While playing, use the light sync path so the UI stays responsive.
        self._notify_dof_selection_changed(lightweight=was_playing)

        # Keep the selected-joint 3D path dock visible and refresh the trail.
        self._expand_overview_joint_dof_analysis(item_id)
        self._overview_traj_dock_visible = True
        try:
            self.show_joint_3d_panel()
        except Exception:
            try:
                self._show_overview_trajectory_dock(True)
            except Exception:
                pass

        try:
            self._update_overview_playback_hud()
        except Exception:
            pass
        try:
            self._refresh_overview_joint_information(item_id)
        except Exception:
            pass
        try:
            self.update_joint_3d_graph(item_id)
        except Exception:
            pass
        self._schedule_dof_traj_reflow()

        self._ensure_playback_continues_after_pick(was_playing=was_playing)

        joint_label = label_for_item(item_id) or "Joint"
        angle_bit = ""
        try:
            snap = (
                self.skeleton_player.current_snapshot()
                if self.skeleton_player is not None
                else None
            )
            if snap is not None:
                from stablewalk.ui.dof_position_table import angle_value_for_item

                deg = angle_value_for_item(item_id, snap)
                if deg is not None:
                    angle_bit = f" · angle {deg:.0f}°"
        except Exception:
            angle_bit = ""
        if hasattr(self, "status"):
            if compare and len(self.selection.selected) > 1:
                self.status.configure(
                    text=(
                        f"Comparing {len(self.selection.selected)} DOFs · "
                        f"active: {joint_label}{angle_bit}"
                    )
                )
            else:
                self.status.configure(
                    text=f"DOF analysis: {joint_label}{angle_bit} · 3D path"
                )
        self._sync_overview_joint_pick_combo(item_id)

    def _on_skeleton_button_press(self, event) -> None:
        """Primary joint hit-test: nearest pickable joint under the cursor."""
        if getattr(event, "button", None) not in (1, None):
            return
        if getattr(event, "dblclick", False):
            return
        if event.inaxes is None:
            return
        if not self.skeleton_player:
            return
        pick_var = getattr(self, "var_skeleton_pick_dof", None)
        if pick_var is not None and not bool(pick_var.get()):
            if hasattr(self, "status"):
                self.status.configure(
                    text="Open Advanced on the skeleton bar, enable Select DOF, then click a joint."
                )
            return
        snap = self.skeleton_player.current_snapshot()
        mode = self._skeleton_display_mode_key()
        joint_id = nearest_joint_from_event(event, snap, display_mode=mode)
        if not joint_id:
            return
        from stablewalk.ui.dof_selection import normalize_gui_dof_id

        item_id = item_for_joint(joint_id) or normalize_gui_dof_id(joint_id)
        if not item_id:
            return
        # Debounce overlapping pick_event for the same click.
        self._skeleton_click_handled_at = time.perf_counter()
        self._focus_joint_trajectory_from_skeleton(
            item_id,
            compare=self._mouse_event_is_compare(event),
        )

    def _on_skeleton_pick_dof_toggle(self) -> None:
        """Toolbar toggle: click-to-select DOF on the 3D gait figure."""
        enabled = True
        pick_var = getattr(self, "var_skeleton_pick_dof", None)
        if pick_var is not None:
            enabled = bool(pick_var.get())
        if hasattr(self, "status"):
            if enabled:
                self.status.configure(
                    text=(
                        "Select DOF: click a joint on the figure, or choose one "
                        "from the joint list."
                    )
                )
            else:
                self.status.configure(text="Select DOF: off — enable to pick joints.")
        self._update_interactive_skeleton(force_draw=True)

    def _sync_overview_joint_pick_combo(self, item_id: str | None = None) -> None:
        """Keep the Choose-joint combobox aligned with the active DOF."""
        var = getattr(self, "var_overview_joint_pick", None)
        if var is None:
            return
        from stablewalk.ui.dof_selection import GUI_DOF_LABELS, label_for_item

        active = item_id or self._active_dof_item_id()
        if not active:
            try:
                if var.get() != "Select joint…":
                    var.set("Select joint…")
            except Exception:
                pass
            return
        label = label_for_item(active) or GUI_DOF_LABELS.get(active, active)
        try:
            if var.get() != label:
                var.set(label)
        except Exception:
            pass

    def _on_overview_joint_pick_chosen(self) -> None:
        """Combobox: choose a joint/DOF point to analyze (same as skeleton click)."""
        var = getattr(self, "var_overview_joint_pick", None)
        if var is None:
            return
        from stablewalk.ui.dof_selection import (
            GUI_DOF_ITEM_IDS,
            GUI_DOF_LABELS,
            normalize_gui_dof_id,
        )

        try:
            label = str(var.get() or "").strip()
        except Exception:
            return
        if not label or label == "Select joint…":
            return
        item_id = None
        for candidate in GUI_DOF_ITEM_IDS:
            if GUI_DOF_LABELS.get(candidate) == label:
                item_id = candidate
                break
        if item_id is None:
            item_id = normalize_gui_dof_id(label.replace(" ", "_").lower())
        if not item_id:
            return
        # Ensure pick mode is on so the selection is interactive.
        pick_var = getattr(self, "var_skeleton_pick_dof", None)
        if pick_var is not None and not bool(pick_var.get()):
            try:
                pick_var.set(True)
            except Exception:
                pass
        self._focus_joint_trajectory_from_skeleton(item_id, compare=False)
        sync = getattr(self, "_sync_spectral_channel_from_joint", None)
        if callable(sync):
            try:
                sync(item_id)
                sched = getattr(self, "_schedule_spectral_analysis", None)
                if callable(sched):
                    sched(True)
            except Exception:
                pass

    def _on_skeleton_pick(self, event) -> None:
        from stablewalk.ui.dof_selection import normalize_gui_dof_id

        pick_var = getattr(self, "var_skeleton_pick_dof", None)
        if pick_var is not None and not bool(pick_var.get()):
            return

        # Prefer button_press hit-test when it already handled this click.
        handled_at = float(getattr(self, "_skeleton_click_handled_at", 0.0) or 0.0)
        if handled_at and (time.perf_counter() - handled_at) < 0.12:
            return

        joint_id = joint_id_from_pick(event)
        if not joint_id:
            return
        item_id = item_for_joint(joint_id) or normalize_gui_dof_id(joint_id)
        if not item_id:
            if hasattr(self, "status"):
                self.status.configure(
                    text=(
                        f"No 3D trajectory mapping for pick {joint_id!r} "
                        "(joint name mapping failed)"
                    )
                )
            return
        self._skeleton_click_handled_at = time.perf_counter()
        self._focus_joint_trajectory_from_skeleton(
            item_id,
            compare=self._pick_event_is_compare(event),
        )

    def _on_dof_pos_tree_select(self, _event: object) -> None:
        """Clicking a position-table row focuses that body point everywhere."""
        tree = getattr(self, "dof_pos_tree", None)
        if tree is None:
            return
        sel = tree.selection()
        if not sel:
            return
        values = tree.item(sel[0], "values")
        if len(values) < 3:
            return
        dof_label = values[2]
        item_id = next(
            (i for i in GUI_DOF_ITEM_IDS if GUI_DOF_LABELS[i] == dof_label),
            None,
        )
        if not item_id:
            return
        if item_id not in self.selection.selected:
            self._dof_checkbox_vars[item_id].set(True)
            self._on_dof_checkbox_changed(item_id)
        else:
            self._activate_dof_item(item_id)

    def _on_refresh_interval(self, _event: object = None) -> None:
        self._apply_refresh_interval(self.refresh_var.get())
        self._last_realtime_refresh = 0.0
        self._update_interactive_skeleton(force_draw=True)

    def _on_dof_tree_select(self, _event: object) -> None:
        if self._legacy_tree_syncing:
            return
        sel = self.dof_tree.selection()
        if not sel:
            return
        label = self.dof_tree.item(sel[0], "values")[0]
        dof = next(
            (name for name in GAIT_ANGLE_FIELDS if DOF_LABELS.get(name, name) == label),
            None,
        )
        item_map = {
            "left_knee": "left_knee",
            "right_knee": "right_knee",
            "left_hip": "left_hip",
            "right_hip": "right_hip",
            "left_ankle": "left_ankle",
            "right_ankle": "right_ankle",
            "left_shoulder": "left_shoulder",
            "right_shoulder": "right_shoulder",
            "left_elbow": "left_elbow",
            "right_elbow": "right_elbow",
            "left_wrist": "left_wrist",
            "right_wrist": "right_wrist",
            "left_heel": "left_heel",
            "right_heel": "right_heel",
            "left_toe": "left_toe",
            "right_toe": "right_toe",
        }
        item_id = item_map.get(dof or "")
        if not item_id:
            return
        self._select_charted_dof_item(item_id)

    def _on_pos_tree_select(self, _event: object) -> None:
        if self._legacy_tree_syncing:
            return
        sel = self.pos_3d_tree.selection()
        if not sel:
            return
        joint_label = self.pos_3d_tree.item(sel[0], "values")[0]
        joint_name = joint_label.replace(" ", "_")
        item_id = item_for_joint(joint_name)
        if not item_id:
            return
        self._select_charted_dof_item(item_id)

    def _stop_playback(self) -> None:
        """Stop playback but keep the current frame displayed (does not reset).

        Pausing and stopping both leave every panel (video overlay, skeleton,
        position table, selected joints) on the current frame. Use ``_reset_playback`` to return to the first frame.
        """
        if not self.skeleton_player:
            return
        self.playing = False
        self.skeleton_player.stop(reset=False)  # keep current frame
        self._sync_play_buttons()
        self._cancel_timer()
        self._3d_flush_scheduled = False
        self._pending_3d_args = None
        # Keep current position; re-render every panel on that frame.
        pos_f = float(self.skeleton_player.state.frame_float)
        self._playback_pos = pos_f
        self.current_pos = int(pos_f)
        self.frame_var.set(int(round(pos_f)))
        self._sync_all_to_frame(force_draw=True)
        from stablewalk.ui.tk.generated_files_panel import flush_generated_files_dirty

        flush_generated_files_dirty(self)
        try:
            from stablewalk.ui.tk.dashboard_tab_lifecycle import cancel_pipeline_animations

            cancel_pipeline_animations(self)
        except Exception:
            pass
        try:
            self._flush_dof_position_table_if_dirty()
        except Exception:
            pass
        if getattr(self, "_pipeline_status_dirty_while_playing", False):
            self._pipeline_status_dirty_while_playing = False
            self._refresh_pipeline_status_panel(force=True)

    def _reset_playback(self) -> None:
        """Return video, skeleton, table, and slider to the first frame."""
        if not self.skeleton_player:
            return
        self._stop_all_gui_timers()
        self.skeleton_player.stop(reset=True)
        self._playback_pos = 0.0
        self.current_pos = 0
        self.frame_var.set(0)
        self._sync_play_buttons()
        self._3d_flush_scheduled = False
        self._pending_3d_args = None
        self._3d_motion_trail = []
        self._sync_all_to_frame(force_draw=True)

    def _on_smooth_toggle(self) -> None:
        if self.skeleton_player:
            self.skeleton_player.smooth = self.smooth_motion.get()
            self._sync_all_to_frame(force_draw=True)

    def _on_speed_change(self, _value: str) -> None:
        self.play_speed = self.speed_var.get()
        self.lbl_speed.configure(text=f"{self.play_speed:.2f}×")

    def _on_slider(self, _value: str) -> None:
        if not self.skeleton_player:
            return
        if getattr(self, "_sync_all_guard", False):
            return
        if self.playing:
            self.playing = False
            self.skeleton_player.pause()
            self._sync_play_buttons()
            self._cancel_timer()
        pos_f = float(self.frame_var.get())
        pos_f = max(0.0, min(pos_f, max(self.skeleton_player.frame_count - 1, 0)))
        self._playback_pos = pos_f
        self.current_pos = int(pos_f)
        self.skeleton_player.go_to(pos_f)
        # Scrub: video/skeleton/playheads only — never recompute graph geometry.
        self._scrubbing = True
        try:
            self._sync_all_to_frame(force_draw=True)
        finally:
            self._scrubbing = False

    def _toggle_play(self) -> None:
        if not self.skeleton_player:
            return
        # Mutual exclusion with Compare playback timer.
        compare = getattr(self, "_comparison_mode", None)
        if compare is not None and getattr(compare, "_playing", False):
            pause = getattr(compare, "pause", None)
            if callable(pause):
                try:
                    pause(paint=False)
                except TypeError:
                    pause()
        self.playing = self.skeleton_player.toggle_play()
        self._sync_play_buttons()
        if self.playing:
            self._playback_pos = float(self.skeleton_player.state.frame_float)
            self._3d_motion_trail = []
            self._panel_tick = 0
            self._skel_play_tick = 0
            self._last_heavy_chart_ts = 0.0
            self._last_ui_sync_ts = 0.0
            self._last_traj_redraw_ts = 0.0
            self._last_joint_info_play_ts = 0.0
            self._tick_last_wall = None
            from stablewalk.ui.tk.render_diagnostics import reset_playback_render_counters

            reset_playback_render_counters(self)
            self._ensure_dof_table_tracking_on_play()
            if self._should_record_dof_table_history():
                self._append_dof_table_history_tick()
            self._append_bilateral_foot_tick()
            self._sync_all_to_frame(force_draw=True)
            self._schedule_tick()
        else:
            self._cancel_timer()
            self._3d_flush_scheduled = False
            self._pending_3d_args = None
            self._last_traj_tip_frame = None
            self._last_traj_tip_item = None
            self._sync_all_to_frame(force_draw=True)
            # Rebuild any charts deferred during play (geometry was frozen).
            try:
                self._flush_dirty_analysis_panels()
            except Exception:
                pass
            # Capture clearance sample once after pause (never during play).
            self._append_bilateral_foot_tick()
            from stablewalk.ui.tk.layout_playback import flush_layout_dirty
            from stablewalk.ui.tk.generated_files_panel import flush_generated_files_dirty

            flush_layout_dirty(self)
            flush_generated_files_dirty(self)
            try:
                from stablewalk.ui.tk.dashboard_tab_lifecycle import (
                    cancel_pipeline_animations,
                )

                # Expensive pipeline pulses are display-only chrome — stop when paused.
                cancel_pipeline_animations(self)
            except Exception:
                pass
            try:
                self._flush_dof_position_table_if_dirty()
            except Exception:
                pass
            if getattr(self, "_pipeline_status_dirty_while_playing", False):
                self._pipeline_status_dirty_while_playing = False
                self._refresh_pipeline_status_panel(force=True)

    def _schedule_tick(self) -> None:
        # Always drop any previous after() before arming the next — one timer only.
        # requestAnimationFrame-style: subtract work already spent this frame so
        # video ticks stay near source FPS instead of stacking fixed delays.
        self._cancel_timer()
        if not self.playing or not self.skeleton_player:
            return
        hz = getattr(self, "_video_anim_hz", None) or getattr(self, "_play_anim_hz", 30)
        target_ms = 1000.0 / max(float(hz), 1.0)
        spent_ms = 0.0
        started = getattr(self, "_tick_frame_started_at", None)
        if started is not None:
            try:
                spent_ms = (time.perf_counter() - float(started)) * 1000.0
            except (TypeError, ValueError):
                spent_ms = 0.0
        delay = max(1, int(target_ms - spent_ms))
        self._after_id = self.root.after(delay, self._tick)

    def _tick(self) -> None:
        if self.playing and self.skeleton_player:
            from stablewalk.ui.tk.render_diagnostics import (
                maybe_log_perf_timings,
                record_playback_render_frame,
                timed_render,
            )

            self._tick_frame_started_at = time.perf_counter()
            with timed_render(self, "total"):
                hz = max(
                    getattr(self, "_video_anim_hz", None)
                    or getattr(self, "_play_anim_hz", 30),
                    1,
                )
                # Wall-clock dt keeps motion smooth under load; cap catch-up so
                # one slow tick cannot jump multiple frames ahead of the UI.
                wall = time.perf_counter()
                last_wall = getattr(self, "_tick_last_wall", None)
                if last_wall is None:
                    dt = 1.0 / hz
                else:
                    dt = min(max(wall - float(last_wall), 1e-4), 2.0 / hz)
                self._tick_last_wall = wall
                prev_frame = self.skeleton_player.state.frame_index
                self.skeleton_player.smooth = self.smooth_motion.get()
                self.skeleton_player.advance(speed=self.play_speed, dt=dt)
                if (
                    self._should_record_dof_table_history()
                    and self.skeleton_player.state.frame_index != prev_frame
                ):
                    self._append_dof_table_history_tick()
                if self.skeleton_player.state.frame_index != prev_frame:
                    self._append_bilateral_foot_tick()
                # Single shared sync: video, skeleton, trajectory, charts, and
                # live metrics all advance from the same PlaybackState this tick.
                self._last_ui_sync_ts = time.monotonic()
                self._sync_all_to_frame(force_draw=False)
            record_playback_render_frame(self)
            maybe_log_perf_timings(self)
            self._schedule_tick()

    def _flush_pending_3d(self) -> None:
        """Redraw 3D body after tables (legacy; playback uses _sync_all_to_frame)."""
        self._3d_flush_scheduled = False
        args = self._pending_3d_args
        if not args or not self.playing:
            return
        from stablewalk.ui.playback_state import playback_commit_is_current

        pending_gen = getattr(self, "_pending_3d_gen", None)
        if not playback_commit_is_current(self, pending_gen):
            return
        frame, frame_next, alpha, list_pos, force_draw, skeleton = args
        self._update_3d_view(
            frame,
            frame_next,
            alpha,
            list_pos=list_pos,
            force_draw=force_draw,
            skeleton=skeleton,
        )

    def _cancel_timer(self) -> None:
        """Cancel the single main-session playback timer (if any)."""
        if self._after_id:
            try:
                self.root.after_cancel(self._after_id)
            except (tk.TclError, ValueError):
                pass
            self._after_id = None

    def _cancel_matplotlib_idle_draws(self) -> None:
        """Drop queued ``canvas.draw_idle()`` callbacks on known figures."""
        canvases: list = []
        for attr in (
            "canvas_3d",
            "canvas_dof_traj",
            "canvas_dof_traj_overview",
            "chart_canvas",
            "canvas_contact_gait",
            "canvas_biomech",
            "canvas_joint_motion",
            "canvas_motion_workspace",
            "canvas_robot",
        ):
            c = getattr(self, attr, None)
            if c is not None:
                canvases.append(c)
        compare = getattr(self, "_comparison_mode", None)
        if compare is not None:
            for attr in (
                "canvas_chart",
                "canvas_heat",
                "canvas_path",
                "canvas_skel_a",
                "canvas_skel_b",
                "canvas_skel_ov",
            ):
                c = getattr(compare, attr, None)
                if c is not None:
                    canvases.append(c)
        for canvas in canvases:
            idle_id = getattr(canvas, "_idle_draw_id", None)
            if idle_id is None:
                continue
            try:
                tkcanvas = getattr(canvas, "_tkcanvas", None)
                if tkcanvas is not None:
                    tkcanvas.after_cancel(idle_id)
                else:
                    self.root.after_cancel(idle_id)
            except (tk.TclError, ValueError, TypeError):
                pass
            try:
                canvas._idle_draw_id = None
            except Exception:
                pass

    def _cancel_queued_redraw_callbacks(self) -> None:
        """Cancel deferred layout / resize / idle-draw work (keep playback timer).

        Called on tab switch and before loading another video so hidden panels
        do not flush stale paints into the next frame.
        """
        self._3d_flush_scheduled = False
        self._pending_3d_args = None
        try:
            from stablewalk.ui.tk.timer_registry import cancel_ancillary_after_jobs

            cancel_ancillary_after_jobs(self)
        except Exception:
            for attr in (
                "_presentation_after_id",
                "_preset_after_id",
                "_video_resize_after",
                "_scroll_sync_after",
                "_traj_resize_after",
                "_skel_resize_after",
                "_knee_resize_after",
                "_tab_reflow_after",
                "_layout_flush_after",
                "_dof_traj_reflow_after",
                "_contact_resize_job",
                "_biomech_resize_job",
                "_overview_sash_sync_after",
                "_overview_media_refit_after",
            ):
                job = getattr(self, attr, None)
                if job is not None:
                    try:
                        self.root.after_cancel(job)
                    except (tk.TclError, ValueError, TypeError):
                        pass
                    setattr(self, attr, None)
        self._cancel_matplotlib_idle_draws()

    def _sync_playback_hz_from_source(self) -> None:
        """Drive the video timer from the loaded clip FPS (analytical UI stays capped)."""
        fps = None
        seq = getattr(self, "sequence", None)
        if seq is not None:
            try:
                fps = float(getattr(seq, "fps", None) or 0.0)
            except (TypeError, ValueError):
                fps = None
        if not fps or fps <= 0:
            player = getattr(self, "skeleton_player", None)
            rec = getattr(player, "recording", None) if player is not None else None
            if rec is None:
                rec = self._analysis_motion_recording() if hasattr(self, "_analysis_motion_recording") else None
            try:
                fps = float(getattr(rec, "fps", None) or 0.0) if rec is not None else None
            except (TypeError, ValueError):
                fps = None
        if not fps or fps <= 0:
            fps = 30.0
        hz = int(round(float(fps)))
        hz = max(1, min(hz, 60))
        self._video_anim_hz = hz
        self._play_anim_hz = hz
        self._ui_anim_hz = hz

    def _clear_shared_frame_buffer(self) -> None:
        self._pb_shared_frame_rgb = None
        self._pb_shared_frame_index = None
        self._pb_shared_frame_path = None

    def _prefetch_upcoming_frames(self, list_pos: int, *, ahead: int = 4) -> None:
        """Decode next frames off the UI thread so play ticks stay ≤16 ms."""
        cache = getattr(self, "_rgb_cache", None)
        seq = self.sequence
        indices = self.pose_indices
        if cache is None or seq is None or not indices:
            return
        prefetch = getattr(cache, "prefetch", None)
        if not callable(prefetch):
            return
        n = len(indices)
        for step in range(1, max(1, ahead) + 1):
            j = list_pos + step
            if j >= n:
                break
            try:
                frame = seq.frames[indices[j]]
                path = getattr(frame, "image_path", None)
                if path:
                    prefetch(path)
            except Exception:
                break

    def _stop_all_gui_timers(self) -> None:
        """Stop playback and ancillary timers before a new analysis/load.

        Ensures only one playback timer can exist afterward and avoids hidden
        tabs continuing to schedule work against a cleared session.
        """
        self._cancel_timer()
        self.playing = False
        self._cancel_queued_redraw_callbacks()
        self._clear_shared_frame_buffer()
        try:
            from stablewalk.ui.tk.timer_registry import cancel_ancillary_after_jobs

            cancel_ancillary_after_jobs(self)
        except Exception:
            pass

        try:
            from stablewalk.ui.tk.dashboard_tab_lifecycle import (
                cancel_pipeline_animations,
                release_compare_temp_buffers,
            )

            cancel_pipeline_animations(self)
            release_compare_temp_buffers(self)
        except Exception:
            pass

        compare = getattr(self, "_comparison_mode", None)
        if compare is not None:
            pause = getattr(compare, "pause", None)
            if callable(pause):
                try:
                    pause(paint=False)
                except TypeError:
                    try:
                        pause()
                    except Exception:
                        pass
            cancel = getattr(compare, "cancel_deferred_work", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    pass
            # Hard-stop Compare's independent play loop if still armed.
            play_job = getattr(compare, "_play_job", None)
            if play_job is not None:
                try:
                    self.root.after_cancel(play_job)
                except (tk.TclError, ValueError, TypeError):
                    pass
                try:
                    compare._play_job = None
                    compare._playing = False
                except Exception:
                    pass

    def _step(self, delta: int) -> None:
        if not self.skeleton_player:
            return
        n = self.skeleton_player.frame_count
        if n == 0:
            return
        pos = (self.current_pos + delta) % n
        self.current_pos = pos
        self._playback_pos = float(pos)
        self.frame_var.set(pos)
        self.skeleton_player.go_to(pos)
        self._scrubbing = True
        try:
            self._sync_all_to_frame(force_draw=True)
        finally:
            self._scrubbing = False

    def _go_to(self, pos: int) -> None:
        if not self.skeleton_player:
            return
        pos = max(0, min(pos, self.skeleton_player.frame_count - 1))
        self.current_pos = pos
        self._playback_pos = float(pos)
        self.frame_var.set(pos)
        self.skeleton_player.go_to(pos)
        self._scrubbing = True
        try:
            self._sync_all_to_frame(force_draw=True)
        finally:
            self._scrubbing = False

    def _toggle_robot_panel(self) -> None:
        if self.show_robot_panel.get():
            self.robot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(PAD_XS, 0))
            if self.sequence and self.pose_indices:
                self._refresh_display()
        else:
            self.robot_frame.pack_forget()

    def _refresh_display(self) -> None:
        if self.skeleton_player:
            self._sync_all_to_frame(force_draw=True)
        elif self.sequence and self.pose_indices:
            self._show_pose_at(self.current_pos, force_draw=True, skeleton_only=False)

    def _show_pose_at(
        self,
        list_pos: int | float,
        *,
        force_draw: bool = False,
        skeleton_only: bool = False,
        video_only: bool = False,
    ) -> None:
        if not self.sequence or not self.pose_indices:
            if self.skeleton_player and not skeleton_only and not video_only:
                self._update_interactive_skeleton(force_draw=force_draw)
            return

        n = len(self.pose_indices)
        if isinstance(list_pos, float):
            pos_f = max(0.0, min(float(list_pos), n - 1 + 1e-6))
            i0 = int(pos_f)
            i1 = min(i0 + 1, n - 1)
            alpha = pos_f - i0
            self.current_pos = i0
        else:
            i0 = int(list_pos) % n
            i1 = min(i0 + 1, n - 1)
            alpha = 0.0
            self.current_pos = i0

        frame_idx = self.pose_indices[i0]
        frame = self.sequence.frames[frame_idx]
        frame_next = self.sequence.frames[self.pose_indices[i1]]
        # Video follows source timestamps, including frames with no pose.
        video_frame = frame
        player = getattr(self, "skeleton_player", None)
        clock = getattr(player, "clock_s", None) if player is not None else None
        if clock is not None:
            from stablewalk.ui.playback_state import sequence_index_for_time

            seq_i = sequence_index_for_time(self.sequence, float(clock))
            if seq_i is not None and 0 <= seq_i < len(self.sequence.frames):
                video_frame = self.sequence.frames[seq_i]
        self._graph_cursor_list_pos = i0
        self._live_panel_list_pos = i0

        # SkeletonPlayer already owns the current reconstructed snapshot during
        # playback. Do not reconstruct/interpolate a second Skeleton3D here:
        # this local value is only consumed by the pause/scrub panel refresh.
        skeleton = None
        need_skeleton = self._group_visible("skeleton") and not video_only
        if need_skeleton and not self.playing and frame.detected and frame.keypoints:
            skeleton = self._get_skeleton(frame, frame_next, alpha)

        video_src_index = int(getattr(video_frame, "frame_index", frame_idx) or frame_idx)
        refresh_video = self._group_visible("video") and (
            not self.playing
            or video_src_index != getattr(self, "_last_video_source_index", -1)
        )
        highlight_joints = self._resolve_highlight_joints()
        if refresh_video:
            # Decode once per source-video frame and reuse the buffer across panels.
            shared_idx = getattr(self, "_pb_shared_frame_index", None)
            shared_path = getattr(self, "_pb_shared_frame_path", None)
            shared_rgb = getattr(self, "_pb_shared_frame_rgb", None)
            video_path = video_frame.image_path or frame.image_path
            if (
                shared_rgb is not None
                and shared_idx == video_src_index
                and shared_path == video_path
            ):
                rgb = shared_rgb
            else:
                from stablewalk.ui.tk.render_diagnostics import timed_render

                with timed_render(self, "decode"):
                    rgb = self._rgb_cache.get_rgb(video_path)
                self._pb_shared_frame_rgb = rgb
                self._pb_shared_frame_index = video_src_index
                self._pb_shared_frame_path = video_path
            self._last_video_source_index = video_src_index
            self._overview_video_frame_index = video_src_index
            if self.playing:
                self._prefetch_upcoming_frames(i0)
            from stablewalk.ui.tk.dashboard_overview_view_mode import (
                VIEW_MODE_OVERLAY,
                VIEW_MODE_SIDE_BY_SIDE,
                VIEW_MODE_VIDEO_ONLY,
                current_overview_view_mode,
            )

            view_mode = current_overview_view_mode(self)
            # Overlay Mode paints reconstructed pose on the video. Side-by-Side
            # keeps original framing and adds only the estimated ground line.
            # Never paint a held pose onto a different source frame.
            pose_matches_video = int(getattr(video_frame, "frame_index", -1)) == int(
                getattr(frame, "frame_index", -2)
            )
            overlay_kps = frame.keypoints if pose_matches_video else None
            ground_y_norm = (
                self._video_estimated_ground_y_norm(video_frame if pose_matches_video else frame)
                if overlay_kps
                else None
            )
            foot_contact = (
                self._video_foot_contact_overlay(video_frame.frame_index)
                if overlay_kps
                else None
            )
            if view_mode == VIEW_MODE_OVERLAY and overlay_kps:
                from stablewalk.pose.video_overlay_compose import (
                    JointTrailBuffer,
                    compose_video_skeleton_overlay,
                )
                from stablewalk.ui.tk.overlay_mode_controls import (
                    overlay_layers_from_gui,
                    overlay_opacity_from_gui,
                )

                layers = overlay_layers_from_gui(self)
                opacity = overlay_opacity_from_gui(self)
                trails = None
                if layers.joint_trajectory:
                    if not hasattr(self, "_video_overlay_trails"):
                        self._video_overlay_trails = JointTrailBuffer()
                    self._video_overlay_trails.update(
                        frame.frame_index, frame.keypoints
                    )
                    trails = self._video_overlay_trails.trails()
                # Trails mutate every frame — skip landmark overlay cache for them.
                use_landmark_cache = not bool(layers.joint_trajectory)
                hl = frozenset(highlight_joints or ())
                overlay_key = (
                    int(frame.frame_index),
                    view_mode,
                    bool(self.highlight_dof.get()),
                    hl,
                    getattr(self, "_selected_asymmetry_moment_index", None),
                    round(float(opacity), 3),
                    bool(layers.skeleton),
                    bool(layers.joint_labels),
                    bool(layers.com),
                    bool(layers.ground_reaction),
                    bool(layers.ground_plane),
                    None if ground_y_norm is None else round(float(ground_y_norm), 4),
                    self._skeleton_display_mode_key(),
                )
                overlay_cache = getattr(self, "_landmark_overlay_cache", None)
                if not isinstance(overlay_cache, dict):
                    overlay_cache = {}
                    self._landmark_overlay_cache = overlay_cache
                cached_overlay = (
                    overlay_cache.get(overlay_key) if use_landmark_cache else None
                )
                if cached_overlay is not None:
                    rgb = cached_overlay
                else:
                    highlight_bgr = None
                    if self._selected_asymmetric_moment() is not None:
                        from stablewalk.ui.asymmetric_moments_view import (
                            ASYMMETRY_HIGHLIGHT_BGR,
                        )

                        highlight_bgr = ASYMMETRY_HIGHLIGHT_BGR
                    rgb = compose_video_skeleton_overlay(
                        rgb,
                        frame.keypoints,
                        layers=layers,
                        opacity=opacity,
                        highlight_dof=bool(self.highlight_dof.get()),
                        highlight_joints=highlight_joints,
                        gait_events=frame.gait_events,
                        trails=trails,
                        foot_contact=foot_contact,
                        ground_y_norm=ground_y_norm,
                        skeleton_model=self._skeleton_display_mode_key(),
                        highlight_bgr=highlight_bgr,
                    )
                    if use_landmark_cache:
                        max_entries = int(
                            getattr(self, "_landmark_overlay_cache_max", 12) or 12
                        )
                        if len(overlay_cache) >= max_entries:
                            try:
                                overlay_cache.pop(next(iter(overlay_cache)))
                            except (StopIteration, KeyError):
                                overlay_cache.clear()
                        overlay_cache[overlay_key] = rgb
            elif (
                view_mode in (VIEW_MODE_SIDE_BY_SIDE, VIEW_MODE_VIDEO_ONLY)
                and frame.keypoints
                and ground_y_norm is not None
            ):
                from stablewalk.pose.video_overlay_compose import (
                    VideoOverlayLayers,
                    compose_video_skeleton_overlay,
                )

                rgb = compose_video_skeleton_overlay(
                    rgb,
                    frame.keypoints,
                    layers=VideoOverlayLayers(
                        skeleton=False,
                        joint_labels=False,
                        com=False,
                        ground_reaction=False,
                        joint_trajectory=False,
                        ground_plane=True,
                    ),
                    opacity=1.0,
                    foot_contact=foot_contact,
                    ground_y_norm=ground_y_norm,
                )
            elif view_mode not in (
                VIEW_MODE_SIDE_BY_SIDE,
                VIEW_MODE_VIDEO_ONLY,
                VIEW_MODE_OVERLAY,
            ) and (self.show_skeleton.get() or self.highlight_dof.get()):
                # Legacy bake path when view-mode workspace is not in use.
                rgb = render_frame_with_skeleton(
                    rgb,
                    frame.keypoints,
                    show_skeleton=self.show_skeleton.get(),
                    highlight_dof=self.highlight_dof.get(),
                    highlight_joints=highlight_joints,
                    gait_events=frame.gait_events,
                )
            if self._group_visible("overview") or self._group_visible("video"):
                rgb = self._apply_video_foot_clearance_hud(
                    rgb,
                    frame.frame_index,
                    keypoints=frame.keypoints,
                    ground_y_norm=ground_y_norm,
                )
            rgb = self._apply_subject_identity_overlay(rgb, video_frame)
            self._set_video_image(rgb)
            self._last_video_list_pos = i0

        if video_only:
            # Keep temporal smoothing warm without painting Matplotlib.
            if self.skeleton_player and self._group_visible("skeleton"):
                try:
                    from stablewalk.ui.skeleton_display_filter import ensure_display_filter

                    snap = self.skeleton_player.current_snapshot()
                    if snap is not None:
                        ensure_display_filter(self).filter_snapshot(snap)
                except Exception:
                    pass
            return

        # Legacy Treeviews / metric trees: pause & scrub only (never play ticks).
        # During scrub, cap panel refreshes at MAX_LIVE_WIDGET_HZ to avoid freezes.
        run_panels = (
            not self.playing
            and (
                not skeleton_only
                or force_draw
                or getattr(self, "_playback_sync_immediate", False)
            )
        )
        if run_panels and getattr(self, "_scrubbing", False):
            from stablewalk.ui.tk.playback_render_policy import (
                MAX_LIVE_WIDGET_HZ,
                rate_limited_due,
            )

            if not rate_limited_due(
                self, "scrub_panels", max_hz=float(MAX_LIVE_WIDGET_HZ)
            ):
                run_panels = False
        if run_panels:
            self._update_panels(frame, i0, frame_idx, frame_next, alpha, skeleton=skeleton)
            self._last_data_refresh = time.monotonic()
        # Graphs: playhead cursor only — during play `_sync_all_to_frame` owns this.
        if (
            not self.playing
            and getattr(self, "_pb_heavy", True)
            and getattr(self, "_pb_ui_tick", True)
        ):
            if self._group_visible("knee"):
                from stablewalk.ui.tk.render_diagnostics import timed_render

                with timed_render(self, "chart"):
                    self._update_chart(playhead_list_pos=i0)
            if self._group_visible("joint_motion") or self._group_visible("motion"):
                try:
                    from stablewalk.ui.tk.render_diagnostics import timed_render

                    with timed_render(self, "chart"):
                        self._update_joint_motion_analysis_chart(
                            playhead_list_pos=i0, force_rebuild=False
                        )
                except Exception:
                    pass
            if self._group_visible("biomech") or self._group_visible("contact"):
                try:
                    playhead_t = self._timestamp_for_frame_index(frame_idx)
                    if self._group_visible("contact"):
                        self._update_contact_gait_chart(playhead_time_s=playhead_t)
                    if self._group_visible("biomech"):
                        self._update_biomechanics_chart(playhead_time_s=playhead_t)
                except Exception:
                    pass
        self._panel_tick += 1

        if need_skeleton:
            self._update_interactive_skeleton(
                force_draw=force_draw or getattr(self, "_playback_sync_immediate", False)
            )

        if self.show_robot_panel.get() and not self.playing:
            self._update_robot_view(frame, frame_next, alpha)

    def _get_skeleton(self, frame: PoseFrame, frame_next: PoseFrame, alpha: float) -> Skeleton3D:
        sk_a = self._skeleton_cache.get(frame.frame_index)
        if sk_a is None:
            sk_a = skeleton_from_frame_data(
                frame.keypoints, frame.skeleton_3d, self._skeleton_scale
            )
            self._skeleton_cache[frame.frame_index] = sk_a
            self._trim_skeleton_cache()
        if alpha <= 0 or not self.smooth_motion.get():
            return normalize_skeleton_height(sk_a)

        sk_b = self._skeleton_cache.get(frame_next.frame_index)
        if sk_b is None:
            sk_b = skeleton_from_frame_data(
                frame_next.keypoints, frame_next.skeleton_3d, self._skeleton_scale
            )
            self._skeleton_cache[frame_next.frame_index] = sk_b
            self._trim_skeleton_cache()
        blended = interpolate_skeleton_3d(sk_a, sk_b, alpha)
        return normalize_skeleton_height(blended)

    def _trim_skeleton_cache(self, *, max_entries: int | None = None) -> None:
        """Bound per-session skeleton growth during long playback / scrubbing."""
        from stablewalk.ui.tk.gui_memory import _SKELETON_CACHE_MAX, trim_skeleton_cache

        trim_skeleton_cache(
            self, max_entries=max_entries if max_entries is not None else _SKELETON_CACHE_MAX
        )

    def _update_3d_view(
        self,
        frame: PoseFrame,
        frame_next: PoseFrame,
        alpha: float,
        *,
        list_pos: int = 0,
        force_draw: bool = False,
        skeleton: Skeleton3D | None = None,
    ) -> None:
        """Legacy hook — interactive skeleton uses ``_update_interactive_skeleton``."""
        self._update_interactive_skeleton(force_draw=force_draw)

    def _fill_tree_rows(self, tree: ttk.Treeview, rows: list[tuple[str, ...]]) -> None:
        """Update a flat Treeview in place (all values in columns)."""
        self._legacy_tree_syncing = True
        try:
            children = tree.get_children()
            for i, row in enumerate(rows):
                if i < len(children):
                    tree.item(children[i], values=row)
                else:
                    tree.insert("", tk.END, values=row)
            for iid in children[len(rows) :]:
                tree.delete(iid)
            widths = getattr(tree, "_sw_column_widths", None)
            if widths:
                for col, w in widths.items():
                    try:
                        tree.column(col, width=w)
                    except tk.TclError:
                        pass
            try:
                tree.xview_moveto(0)
            except tk.TclError:
                pass
            # Forcing layout every frame stalls playback; defer until paused.
            if not self.playing:
                tree.update_idletasks()
        finally:
            self._legacy_tree_syncing = False

    def _fill_position_tree(
        self, positions: list[tuple[str, float, float, float]]
    ) -> None:
        """Update positions table in place (avoids flicker during playback)."""
        self._last_positions = positions
        rows = [
            (name.replace("_", " "), f"{x:.3f}", f"{y:.3f}", f"{z:.3f}")
            for name, x, y, z in positions
        ]
        self._fill_tree_rows(self.pos_3d_tree, rows)

    def _update_robot_view(
        self,
        frame: PoseFrame,
        frame_next: PoseFrame,
        alpha: float,
    ) -> None:
        if not self._walk_simulation or not frame.detected:
            self.ax_robot.cla()
            setup_3d_axes(self.ax_robot)
            self.ax_robot.text2D(
                0.5, 0.5, "Run analysis", transform=self.ax_robot.transAxes, ha="center", color=MUTED
            )
            self.canvas_robot.draw_idle()
            return

        sim = WalkSimulator()
        if alpha > 0 and self.smooth_motion.get():
            geom = sim.geometry_at_blend(frame, frame_next, alpha, self.sequence.fps)
        else:
            sf = sim.from_pose_frame(frame, self.sequence.fps)
            geom = sf.geometry

        draw_robot_geometry(self.ax_robot, geom, clear=True, title="Robot (DOF-driven)")
        # tight_layout is expensive; only run when paused/scrubbing for crisp layout.
        if not self.playing:
            self.fig_robot.tight_layout()
        self.canvas_robot.draw_idle()

    def _set_video_image(self, rgb: np.ndarray) -> None:
        from stablewalk.ui.tk.render_diagnostics import timed_render

        with timed_render(self, "video"):
            self._set_video_image_impl(rgb)

    def _set_video_image_impl(self, rgb: np.ndarray) -> None:
        self._last_video_rgb = rgb
        pil = Image.fromarray(rgb)
        src_w, src_h = pil.size
        if src_w <= 0 or src_h <= 0:
            return

        # Adapt the top-row split to the clip's shape so a portrait video fills
        # its panel instead of floating between wide empty bands. Only reflow
        # when the aspect actually changes (i.e. a new clip), not every frame.
        aspect = src_w / src_h
        if self._video_aspect is None or abs(self._video_aspect - aspect) > 0.02:
            self._video_aspect = aspect
            from stablewalk.ui.tk.dashboard_layout import apply_top_row_aspect

            apply_top_row_aspect(self, aspect)

        # Prefer the display host size (stable panel box) so letterboxing is
        # measured against the available viewport, not the previous image size.
        host = getattr(self, "video_display_host", None) or self.video_label
        cached = getattr(self, "_video_label_box", None)
        if self.playing and cached is not None:
            lw, lh = cached
        else:
            try:
                host.update_idletasks()
                lw = int(host.winfo_width())
                lh = int(host.winfo_height())
            except tk.TclError:
                lw = self.video_label.winfo_width()
                lh = self.video_label.winfo_height()
            if lw > 1 and lh > 1:
                self._video_label_box = (lw, lh)
        if lw <= 1 or lh <= 1:
            lw, lh = 640, 480
        # Minimal inset — keep the person large while preserving aspect-fit.
        pad = 1
        box_w = max(lw - pad * 2, 64)
        box_h = max(lh - pad * 2, 64)

        from stablewalk.ui.video_display_fit import contain_size, cover_size_and_crop

        zoom_mode = "Fit"
        zoom_var = getattr(self, "_video_display_zoom_var", None)
        if zoom_var is not None:
            try:
                zoom_mode = str(zoom_var.get() or "Fit")
            except tk.TclError:
                zoom_mode = "Fit"
        crop_box = None
        if zoom_mode == "Fill":
            new_w, new_h, crop_box = cover_size_and_crop(src_w, src_h, box_w, box_h)
        else:
            new_w, new_h = contain_size(src_w, src_h, box_w, box_h)
        photo_key = (id(rgb), new_w, new_h, zoom_mode, crop_box, int(self.playing))
        if getattr(self, "_video_photo_key", None) == photo_key and self._photo is not None:
            return
        # LANCZOS is high quality but slow; while playing, BILINEAR keeps the
        # frame rate high and the difference is imperceptible on moving video.
        resample = (
            Image.Resampling.BILINEAR if self.playing else Image.Resampling.LANCZOS
        )
        fitted = pil.resize((new_w, new_h), resample)
        if crop_box is not None:
            fitted = fitted.crop(crop_box)
        from stablewalk.ui.tk.render_diagnostics import timed_render

        with timed_render(self, "tk_paint"):
            self._photo = ImageTk.PhotoImage(fitted)
            self._video_photo_key = photo_key
            self.video_label.configure(
                image=self._photo,
                text="",
                anchor=tk.CENTER,
                bg=PANEL,
                compound=tk.CENTER,
            )
            # Persistent reference prevents Tk GC and stale-frame artifacts.
            self.video_label.image = self._photo  # type: ignore[attr-defined]
            self._video_display_map = {
                "src_w": src_w,
                "src_h": src_h,
                "fitted_w": new_w,
                "fitted_h": new_h,
                "zoom_mode": zoom_mode,
                "crop_box": crop_box,
                "label_w": lw,
                "label_h": lh,
            }

    def _on_video_label_resize(self, _event: object = None) -> None:
        """Re-fit the current frame when the video panel is resized."""
        from stablewalk.ui.tk.layout_playback import should_defer_layout

        if should_defer_layout(self):
            return
        if self._last_video_rgb is None:
            return
        if _event is not None:
            w = int(getattr(_event, "width", 0) or 0)
            h = int(getattr(_event, "height", 0) or 0)
            last = getattr(self, "_video_host_size", (0, 0))
            if w >= 2 and h >= 2 and abs(w - last[0]) < 6 and abs(h - last[1]) < 6:
                return
            if w >= 2 and h >= 2:
                self._video_host_size = (w, h)
        if self._video_resize_after is not None:
            try:
                self.root.after_cancel(self._video_resize_after)
            except (tk.TclError, ValueError):
                pass

        self._video_label_box = None
        self._video_photo_key = None

        def _refit() -> None:
            self._video_resize_after = None
            self._video_label_box = None
            self._video_photo_key = None
            if self._last_video_rgb is not None:
                self._set_video_image(self._last_video_rgb)

        self._video_resize_after = self.root.after(140, _refit)

    def _update_panels(
        self,
        frame: PoseFrame,
        list_pos: int,
        frame_idx: int,
        frame_next: PoseFrame,
        alpha: float,
        *,
        skeleton: Skeleton3D | None = None,
    ) -> None:
        angles = frame.joint_angles
        gait_dof = angles.gait_dof_count() if angles else 0

        prev_frame: PoseFrame | None = None
        if list_pos > 0 and self.sequence and self.pose_indices:
            prev_frame = self.sequence.frames[self.pose_indices[list_pos - 1]]

        if skeleton is None and frame.detected and frame.keypoints:
            skeleton = self._get_skeleton(frame, frame_next, alpha)

        positions = _collect_joint_positions(
            frame, skeleton if not self.playing else None, prefer_frame=True
        )
        if not positions and self._last_positions:
            positions = self._last_positions

        # Cache per discrete frame — scrubbing revisits frames constantly.
        snap_cache = getattr(self, "_frame_motion_snap_cache", None)
        if not isinstance(snap_cache, dict):
            snap_cache = {}
            self._frame_motion_snap_cache = snap_cache
        snap_key = (
            id(self.sequence) if self.sequence is not None else 0,
            int(list_pos),
            int(frame_idx),
            round(float(alpha), 2),
        )
        snap = snap_cache.get(snap_key)
        if snap is None:
            snap = build_frame_motion_snapshot(
                frame,
                skeleton=skeleton,
                prev_frame=prev_frame,
                fps=self.sequence.fps if self.sequence else 25.0,
                rom_cache=self._rom_cache,
            )
            if positions:
                snap = FrameMotionSnapshot(
                    frame_index=snap.frame_index,
                    positions_3d=positions,
                    dof_rows=snap.dof_rows,
                    velocity_rows=snap.velocity_rows,
                    summary_lines=snap.summary_lines,
                )
            if len(snap_cache) >= 96:
                try:
                    snap_cache.pop(next(iter(snap_cache)))
                except Exception:
                    snap_cache.clear()
            snap_cache[snap_key] = snap
        elif positions and not snap.positions_3d:
            snap = FrameMotionSnapshot(
                frame_index=snap.frame_index,
                positions_3d=positions,
                dof_rows=snap.dof_rows,
                velocity_rows=snap.velocity_rows,
                summary_lines=snap.summary_lines,
            )
            snap_cache[snap_key] = snap

        from stablewalk.ui.playback_state import committed_playback_state

        play = committed_playback_state(self)
        if play is not None:
            self._live_panel_list_pos = int(play.current_frame_index)
            frame_text = (
                f"Frame {play.format_frame_label()}  ·  "
                f"{play.format_time_label(include_duration=False)}  ·  "
                f"{len(snap.positions_3d)} joints  ·  {gait_dof} DoF"
            )
        else:
            self._live_panel_list_pos = int(list_pos)
            frame_text = (
                f"Frame {list_pos + 1}  ·  {len(snap.positions_3d)} joints  ·  {gait_dof} DoF"
            )
        self.lbl_frame_data.configure(text=frame_text)

        self._fill_position_tree(snap.positions_3d)

        self._fill_tree_rows(
            self.dof_tree,
            [(label, ang, omega, rom, spd) for label, ang, omega, rom, spd in snap.dof_rows],
        )
        self._fill_tree_rows(
            self.vel_tree,
            [(label, vx, vy, spd, direction) for label, vx, vy, spd, direction in snap.velocity_rows],
        )

        self.motion_text.configure(state=tk.NORMAL)
        self.motion_text.delete("1.0", tk.END)
        self.motion_text.insert(tk.END, "\n".join(snap.summary_lines))
        self.motion_text.configure(state=tk.DISABLED)

        # Frame/DoF readout lives on the transport bar — do not mirror it on
        # the status strip (avoids overlapping ghost text under DPI scaling).

    def _display_velocities(
        self,
        list_pos: int,
        frame: PoseFrame,
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """
        Real-time velocities for UI: use stored values or compute vs previous pose frame.
        """
        if list_pos > 0 and self.sequence:
            prev_frame = self.sequence.frames[self.pose_indices[list_pos - 1]]
            if prev_frame.detected and frame.detected:
                vec, scalar = velocity_between_frames(
                    prev_frame, frame, self.sequence.fps
                )
                return scalar, vec

        return dict(frame.velocity_scalar), dict(frame.velocities)

    def _rebuild_knee_angle_series(self) -> None:
        """Cache full-video knee trajectories for the chart (rebuilt on pose load)."""
        self._knee_angle_series = None
        if not self.sequence or not self.pose_indices:
            return
        from stablewalk.ui.viewers.knee_angle_chart import build_knee_angle_series

        ik_mot = self._opensim_ik_mot_path()
        pref = self._knee_angle_source_preference()
        ik_ok = self._opensim_ik_quality_ok()
        self._knee_angle_series = build_knee_angle_series(
            self.sequence,
            self.pose_indices,
            ik_mot_path=ik_mot,
            source_preference=pref,
            ik_quality_ok=ik_ok,
        )
        self._refresh_knee_source_selector()

    def _knee_angle_source_preference(self):
        var = getattr(self, "var_knee_angle_source", None)
        label = var.get() if var is not None else "Auto"
        mapping = {
            "Auto": "auto",
            "Pose-derived": "pose_derived",
            "OpenSim IK": "opensim_ik",
        }
        return mapping.get(label, "auto")  # type: ignore[return-value]

    def _opensim_ik_quality_ok(self) -> bool:
        from stablewalk.analysis.gait_feature_analysis import read_opensim_mot_timeseries
        from stablewalk.ui.viewers.knee_angle_chart import ik_mot_passes_quality_check

        ik_mot = self._opensim_ik_mot_path()
        if not ik_mot:
            return False
        ik_state = getattr(self, "_opensim_ik_state", None)
        if ik_state not in (None, "Completed"):
            return False
        mot = read_opensim_mot_timeseries(ik_mot)
        return ik_mot_passes_quality_check(mot)

    def _opensim_ik_available_for_knee_chart(self) -> bool:
        from stablewalk.ui.viewers.knee_angle_chart import opensim_ik_available

        ik_state = getattr(self, "_opensim_ik_state", None)
        ik_completed = ik_state in (None, "Completed")
        return opensim_ik_available(self._opensim_ik_mot_path(), ik_completed=ik_completed)

    def _refresh_knee_source_selector(self) -> None:
        toolbar = getattr(self, "_knee_source_toolbar", None)
        if toolbar is None:
            return
        if self._opensim_ik_available_for_knee_chart():
            toolbar.pack(side=tk.LEFT, padx=(8, 0))
        else:
            toolbar.pack_forget()
            if hasattr(self, "var_knee_angle_source"):
                self.var_knee_angle_source.set("Auto")

    def _on_knee_chart_mode_changed(self) -> None:
        self._knee_chart_mode_user_set = True
        self._update_chart()

    def _on_knee_angle_source_changed(self) -> None:
        self._rebuild_knee_angle_series()
        self._update_chart()

    def _default_knee_chart_mode(self) -> str:
        from stablewalk.ui.viewers.knee_chart_interpretation import cycle_mode_is_available

        if cycle_mode_is_available(getattr(self, "_gait_features", None)):
            return "Gait Cycle %"
        return "Video Time"

    def _apply_default_knee_chart_mode(self) -> None:
        if not hasattr(self, "var_knee_chart_axis"):
            return
        if getattr(self, "_knee_chart_mode_user_set", False):
            return
        self.var_knee_chart_axis.set(self._default_knee_chart_mode())

    def _update_knee_interpretation_panel(self, *, chart_mode: str) -> None:
        from stablewalk.ui.dashboard_interpretability import interpret_knee_motion
        from stablewalk.ui.viewers.knee_chart_interpretation import build_knee_motion_summary

        mode_key = "gait_cycle_pct" if chart_mode == "Gait Cycle %" else "video_time"
        summary = build_knee_motion_summary(
            getattr(self, "_knee_angle_series", None),
            getattr(self, "_gait_features", None),
            chart_mode=mode_key,  # type: ignore[arg-type]
        )
        card = interpret_knee_motion(summary)
        from stablewalk.ui.viewers.knee_chart_interpretation import (
            usable_knee_cycle_count,
        )

        left = f"{summary.left_rom_deg:.0f}°" if summary.left_rom_deg is not None else "—"
        right = f"{summary.right_rom_deg:.0f}°" if summary.right_rom_deg is not None else "—"
        asym = (
            f"{summary.rom_asymmetry_pct:.1f}%"
            if summary.rom_asymmetry_pct is not None
            else "—"
        )
        cycles = usable_knee_cycle_count(getattr(self, "_gait_features", None))
        cycles_txt = str(int(cycles)) if cycles else "0"
        repeat = summary.cycle_repeatability or "—"
        compact = getattr(self, "lbl_knee_summary_compact", None) or getattr(
            self, "lbl_knee_interp_compact", None
        )
        if compact is not None:
            rom_scope = "cycle" if mode_key == "gait_cycle_pct" else "session"
            compact.configure(
                text=(
                    f"L {rom_scope} ROM {left}  ·  R {rom_scope} ROM {right}  ·  "
                    f"Asymmetry {asym}  ·  Cycles {cycles_txt}  ·  "
                    f"Repeatability {repeat}"
                )
            )
        # Visible empty-state when cycle mode has no reliable cycles.
        no_cycle = getattr(self, "lbl_knee_no_cycle", None)
        if no_cycle is not None:
            show_banner = mode_key == "gait_cycle_pct" and cycles < 2
            try:
                if show_banner:
                    no_cycle.configure(
                        text=(
                            "No reliable cycle detected — switch Mode to "
                            "Video Time, or analyze a walk with clearer strides."
                        )
                    )
                    no_cycle.grid()
                else:
                    no_cycle.grid_remove()
            except tk.TclError:
                pass
        self._knee_motion_summary = summary

    def _seek_motion_workspace_time(self, time_s: float) -> None:
        """Seek global playback to ``time_s`` — updates video, skeleton, and path."""
        player = self.skeleton_player
        recording = self._analysis_motion_recording()
        if player is None and recording is None:
            return
        if self.playing:
            self.playing = False
            if player is not None:
                try:
                    player.pause()
                except Exception:
                    pass
            self._sync_play_buttons()
            self._cancel_timer()
        # Map time → list index via recording timestamps.
        frame_f = 0.0
        if recording is not None and recording.frame_count > 0:
            best_i = 0
            best_dt = 1e9
            for i in range(recording.frame_count):
                snap = recording.snapshot_at(i)
                if snap is None:
                    continue
                dt = abs(float(snap.time_s) - float(time_s))
                if dt < best_dt:
                    best_dt = dt
                    best_i = i
            frame_f = float(best_i)
        if player is not None:
            frame_f = max(0.0, min(frame_f, max(player.frame_count - 1, 0)))
            self._playback_pos = frame_f
            self.current_pos = int(frame_f)
            player.go_to(frame_f)
            try:
                self.frame_var.set(int(frame_f))
            except (tk.TclError, TypeError, ValueError):
                pass
        self._sync_all_to_frame(force_draw=True)

    def _update_motion_graph_stats_ribbon(self, *, time_s: float | None = None) -> None:
        """Refresh the compact Min/Max/Mean/ROM/Std/Current ribbon above the graph."""
        labels = getattr(self, "_motion_graph_stats_labels", None) or {}
        if not labels:
            return
        # Aggregates are cached; still rate-limit Tk label writes while playing.
        if getattr(self, "playing", False):
            from stablewalk.ui.tk.playback_render_policy import (
                MAX_LIVE_WIDGET_HZ,
                rate_limited_due,
            )

            if not rate_limited_due(
                self, "motion_ribbon", max_hz=float(MAX_LIVE_WIDGET_HZ)
            ):
                return
        from stablewalk.ui.tk.playback_render_policy import configure_text_if_changed
        from stablewalk.ui.viewers.joint_motion_analysis_chart import (
            MOTION_ANALYSIS_TABS,
            compute_metric_ribbon_stats,
        )

        bundle = getattr(self, "_joint_motion_bundle", None)
        metric_id = str(getattr(self, "_motion_active_metric", None) or "angle")
        unit = ""
        for mid, _title, ylabel in MOTION_ANALYSIS_TABS:
            if mid == metric_id:
                if "(" in ylabel and ")" in ylabel:
                    unit = ylabel.split("(")[-1].rstrip(")")
                break
        if time_s is None:
            from stablewalk.ui.playback_state import read_playback_state

            state = read_playback_state(self)
            if state is not None:
                time_s = float(state.current_time_seconds)
        if bundle is None or bundle.empty:
            for lbl in labels.values():
                configure_text_if_changed(lbl, "—")
            return
        stats = compute_metric_ribbon_stats(
            bundle, metric_id, time_s=time_s, unit=unit
        )
        for key, text in stats.as_display().items():
            configure_text_if_changed(labels.get(key), text)

    def _update_motion_workspace_stats(self) -> None:
        """Refresh Motion Analysis live kinematics at the current playhead."""
        labels = getattr(self, "_motion_stats_value_labels", None) or {}
        if not labels:
            self._update_motion_graph_stats_ribbon()
            return
        import numpy as np

        from stablewalk.ui.tk.dashboard_motion_workspace import (
            sync_motion_live_kinematics,
        )

        bundle = getattr(self, "_joint_motion_bundle", None)
        if bundle is None or bundle.empty:
            sync_motion_live_kinematics(self, {})
            self._update_motion_graph_stats_ribbon()
            return
        active_item = getattr(getattr(self, "selection", None), "active_item_id", None)
        series = next(
            (item for item in bundle.series if item.item_id == active_item),
            None,
        )
        if series is None:
            from stablewalk.ui.dof_selection import label_for_item
            from stablewalk.ui.playback_state import committed_playback_state

            state = committed_playback_state(self)
            frame = (
                state.format_frame_label()
                if state is not None
                else "Unavailable"
            )
            time_text = (
                state.format_time_label()
                if state is not None
                else "Unavailable"
            )
            if state is not None:
                self._live_panel_list_pos = int(state.current_frame_index)
            unavailable = {
                key: "Unavailable"
                for key in (
                    "angle",
                    "velocity",
                    "acceleration",
                    "jerk",
                    "angular_jerk",
                    "x",
                    "y",
                    "z",
                )
            }
            unavailable.update(
                {
                    "joint": label_for_item(active_item) if active_item else "Unavailable",
                    "frame": frame,
                    "time": time_text,
                }
            )
            sync_motion_live_kinematics(self, unavailable)
            self._update_motion_graph_stats_ribbon()
            return
        time_s = 0.0
        time_available = False
        frame_display = "Unavailable"
        time_display = "Unavailable"
        from stablewalk.ui.playback_state import committed_playback_state

        state = committed_playback_state(self)
        if state is not None:
            time_s = float(state.current_time_seconds)
            time_available = bool(np.isfinite(time_s))
            frame_display = state.format_frame_label()
            time_display = state.format_time_label()
            self._live_panel_list_pos = int(state.current_frame_index)
        elif series.times_s.size and np.isfinite(series.times_s[0]):
            time_s = float(series.times_s[0])
            time_available = True

        sample_index: int | None = None
        if series.times_s.size:
            finite_times = np.flatnonzero(np.isfinite(series.times_s))
            if finite_times.size:
                nearest = int(
                    np.argmin(np.abs(series.times_s[finite_times] - time_s))
                )
                sample_index = int(finite_times[nearest])
                if frame_display == "Unavailable" and sample_index < series.frames.size:
                    raw_frame = float(series.frames[sample_index])
                    if np.isfinite(raw_frame):
                        frame_display = str(int(raw_frame))
        if time_available and time_display == "Unavailable":
            time_display = f"{time_s:.2f} s"

        def _at(values: np.ndarray, *, places: int = 1, unit: str = "") -> str:
            if sample_index is None or sample_index >= values.size:
                return "Unavailable"
            y = float(values[sample_index])
            if not np.isfinite(y):
                return "Unavailable"
            return f"{y:.{places}f}{unit}"

        mapping = {
            "joint": series.label,
            "angle": _at(series.angle_deg, unit="°"),
            "velocity": _at(series.omega_deg_s, unit=" °/s"),
            "acceleration": _at(series.alpha_deg_s2, unit=" °/s²"),
            "jerk": _at(series.jerk_m_s3, places=3, unit=" m/s³"),
            "angular_jerk": _at(series.angular_jerk_deg_s3, places=3, unit=" °/s³"),
            "x": _at(series.x_m, places=3, unit=" m"),
            "y": _at(series.y_m, places=3, unit=" m"),
            "z": _at(series.z_m, places=3, unit=" m"),
            "frame": str(frame_display),
            "time": time_display,
        }
        sync_motion_live_kinematics(self, mapping)
        self._update_motion_graph_stats_ribbon(time_s=time_s)

    def _update_joint_motion_analysis_chart(
        self,
        *,
        playhead_list_pos: int | None = None,
        force_rebuild: bool = False,
    ) -> None:
        """Refresh Motion metric tab (one graph) or Overview Joint Graphs (6-panel)."""
        if not (
            self._group_visible("motion") or self._group_visible("joint_motion")
        ):
            return
        on_motion = self._group_visible("motion")
        active_metric = None
        metric_panel = None
        if on_motion:
            try:
                from stablewalk.ui.tk.dashboard_motion_workspace import (
                    refresh_motion_angle_convention,
                )

                refresh_motion_angle_convention(self)
            except Exception:
                pass
            if getattr(self, "canvas_motion_workspace", None) is None:
                ensure = getattr(self, "_ensure_motion_workspace_chart", None)
                if callable(ensure):
                    try:
                        ensure()
                    except Exception:
                        pass
            active_metric = getattr(self, "_motion_active_metric", None) or "angle"
            panels = getattr(self, "_motion_metric_panels", None) or {}
            metric_panel = panels.get(str(active_metric))
            if metric_panel is not None:
                fig = metric_panel.get("fig")
                canvas = metric_panel.get("canvas")
                self.fig_motion_workspace = fig
                self.canvas_motion_workspace = canvas
            else:
                fig = getattr(self, "fig_motion_workspace", None)
                canvas = getattr(self, "canvas_motion_workspace", None)
        else:
            if getattr(self, "canvas_joint_motion", None) is None:
                ensure = getattr(self, "_ensure_joint_motion_chart", None)
                if callable(ensure) and self._group_visible("joint_motion"):
                    try:
                        ensure()
                    except Exception:
                        pass
            fig = getattr(self, "fig_joint_motion", None)
            canvas = getattr(self, "canvas_joint_motion", None)
        if fig is None or canvas is None:
            return

        from stablewalk.ui.viewers.chart_interactions import finalize_chart_interactions
        from stablewalk.ui.viewers.chart_playhead import PlayheadState, playhead_from_playback
        from stablewalk.ui.viewers.joint_motion_analysis_chart import (
            build_joint_motion_bundle,
            draw_joint_motion_analysis_chart,
            draw_motion_metric_chart,
        )

        recording = self._analysis_motion_recording()
        selected = set(self.selection.selected) if self.selection else set()
        gait_cycle = getattr(self, "_gait_cycle", None)
        cache_key = (
            id(recording) if recording is not None else None,
            frozenset(selected),
            int(getattr(recording, "frame_count", 0) or 0),
            id(gait_cycle) if gait_cycle is not None else None,
        )
        bundle = getattr(self, "_joint_motion_bundle", None)
        bundle_key = cache_key[:3]
        # Bundle is shared across all metric tabs — recompute only when data changes.
        if (
            force_rebuild
            or bundle is None
            or getattr(self, "_joint_motion_bundle_key", None) != bundle_key
        ):
            bundle = build_joint_motion_bundle(recording, selected)
            self._joint_motion_bundle = bundle
            self._joint_motion_bundle_key = bundle_key
            # Invalidate per-tab geometry caches so the active tab redraws once.
            for panel in (getattr(self, "_motion_metric_panels", None) or {}).values():
                panel["render_key"] = None

        time_s = 0.0
        frame_index = 1
        list_pos_for_cursor = playhead_list_pos
        from stablewalk.ui.playback_state import committed_playback_state

        state = committed_playback_state(self)
        if state is not None:
            time_s = float(state.current_time_seconds)
            frame_index = int(state.display_frame_number)
            list_pos_for_cursor = int(state.current_frame_index)
            self._graph_cursor_list_pos = list_pos_for_cursor
        elif recording is not None and recording.frame_count > 0:
            idx = 0
            if playhead_list_pos is not None and self.pose_indices:
                i = max(0, min(int(playhead_list_pos), len(self.pose_indices) - 1))
                idx = i
            snap = recording.snapshot_at(min(idx, recording.frame_count - 1))
            if snap is not None:
                time_s = float(snap.time_s)
                frame_index = idx + 1
            list_pos_for_cursor = idx
            self._graph_cursor_list_pos = idx

        animating = bool(getattr(self, "playing", False))
        phase = float(getattr(self, "_joint_motion_playhead_phase", 0.0) or 0.0)
        if animating:
            from stablewalk.ui.viewers.chart_playhead import advance_playhead_anim_phase

            phase = advance_playhead_anim_phase(phase)
            self._joint_motion_playhead_phase = phase

        if state is not None:
            playhead = playhead_from_playback(
                state,
                time_s=time_s,
                anim_phase=phase,
                animating=animating,
            )
        else:
            playhead = PlayheadState(
                time_s=time_s,
                frame_index=frame_index,
                list_pos=list_pos_for_cursor,
                anim_phase=phase,
                animating=animating,
            )
        panel_render_key = (
            cache_key if metric_panel is None else (cache_key, str(active_metric))
        )
        panel_already_drawn = (
            metric_panel is not None
            and metric_panel.get("render_key") == panel_render_key
        )
        reused_playhead = (
            not force_rebuild
            and (
                panel_already_drawn
                or (
                    metric_panel is None
                    and getattr(self, "_joint_motion_render_key", None) == cache_key
                )
            )
            and self._move_existing_chart_playheads(
                fig, canvas, time_s, list_pos=list_pos_for_cursor
            )
        )
        if not reused_playhead:
            # First paint must never be deferred — an unstyled/white Axes is
            # created lazily, and mid-play deferral left Motion Analysis blank.
            needs_first_paint = (
                (metric_panel is not None and metric_panel.get("render_key") is None)
                or not any(
                    getattr(ax, "_stablewalk_playhead_artists", None) for ax in fig.axes
                )
            )
            from stablewalk.ui.tk.playback_render_policy import (
                allow_graph_geometry_rebuild,
                mark_analysis_dirty,
            )

            # Play / scrub: playhead only. First paint of a blank axes is allowed.
            if (
                not allow_graph_geometry_rebuild(self, force=force_rebuild)
                and not needs_first_paint
            ):
                mark_analysis_dirty(self, "joint_motion")
            else:
                from stablewalk.ui.viewers.chart_playhead import cache_static_chart_background

                workspace = bool(getattr(self, "fig_motion_workspace", None) is fig)
                if workspace and active_metric:
                    gait_events = list(gait_cycle.events) if gait_cycle is not None else None
                    hover = draw_motion_metric_chart(
                        fig,
                        bundle,
                        str(active_metric),
                        playhead=playhead,
                        gait_events=gait_events,
                        show_peaks=True,
                    )
                    if metric_panel is not None:
                        metric_panel["render_key"] = panel_render_key
                else:
                    hover = draw_joint_motion_analysis_chart(fig, bundle, playhead=playhead)
                finalize_chart_interactions(fig, canvas, hover_points=hover)
                self._joint_motion_render_key = cache_key
                self._paint_canvas_now(canvas)
                if getattr(self, "playing", False):
                    cache_static_chart_background(fig, canvas, list(fig.axes))
                # Reveal graph host once a real paint completed.
                try:
                    empty = getattr(self, "lbl_motion_graph_empty", None)
                    if empty is not None:
                        empty.grid_remove()
                except tk.TclError:
                    pass

        # Cached live values and graph ribbon both track the playback cursor.
        try:
            self._update_motion_workspace_stats()
        except Exception:
            pass
        # Timeline cursor labels only (dirty text updates).
        from stablewalk.ui.tk.playback_render_policy import configure_text_if_changed

        motion_time_text = (
            state.format_time_label() if state is not None else f"{time_s:.2f} s"
        )
        motion_frame_text = (
            f"Frame {state.format_frame_label()}"
            if state is not None
            else f"Frame {int(frame_index)}"
        )
        configure_text_if_changed(
            getattr(self, "lbl_motion_time", None), motion_time_text
        )
        configure_text_if_changed(
            getattr(self, "lbl_motion_frame", None), motion_frame_text
        )
        timeline = getattr(self, "motion_timeline", None)
        slider = getattr(self, "slider", None)
        if timeline is not None and slider is not None:
            try:
                timeline.configure(to=float(slider.cget("to")))
            except tk.TclError:
                pass
        play_btn = getattr(self, "btn_motion_play", None)
        if play_btn is not None:
            configure_text_if_changed(
                play_btn,
                "Pause" if bool(getattr(self, "playing", False)) else "Play",
            )
        if not getattr(self, "playing", False):
            hint = getattr(self, "lbl_joint_motion_hint", None)
            if hint is not None:
                if not selected:
                    text = (
                        "Select a joint — Angle / Velocity / Acceleration / Position / Jerk "
                        "tabs share one timeline."
                    )
                elif len(selected) == 1:
                    text = (
                        "1 joint · one graph at a time · tabs reuse cached series · "
                        "Export PNG / CSV"
                    )
                else:
                    text = (
                        f"{len(selected)} joints · tabbed metrics · synchronized playhead · "
                        "Export PNG / CSV"
                    )
                configure_text_if_changed(hint, text)

    def _export_joint_motion_csv(self) -> None:
        """Prompt and write Joint Motion Analysis series to CSV."""
        from tkinter import filedialog, messagebox

        from stablewalk.ui.viewers.joint_motion_analysis_chart import (
            build_joint_motion_bundle,
            write_joint_motion_csv,
        )

        recording = self._analysis_motion_recording()
        selected = set(self.selection.selected) if self.selection else set()
        bundle = getattr(self, "_joint_motion_bundle", None)
        if bundle is None or bundle.empty:
            bundle = build_joint_motion_bundle(recording, selected)
            self._joint_motion_bundle = bundle
        if bundle.empty:
            messagebox.showinfo(
                "Export CSV",
                "Select one or more joints before exporting.",
                parent=self.root,
            )
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Joint Motion Analysis CSV",
            defaultextension=".csv",
            initialfile="joint_motion_analysis.csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if not path:
            return
        out = write_joint_motion_csv(bundle, path)
        messagebox.showinfo(
            "Export CSV",
            f"Saved joint motion series to:\n{out}",
            parent=self.root,
        )

    def _update_chart(self, *, playhead_list_pos: int | None = None) -> None:
        """Plot knee flexion in Video Time or Gait Cycle % mode."""
        if not self._group_visible("knee"):
            return
        if getattr(self, "chart_canvas", None) is None or getattr(self, "ax_chart", None) is None:
            ensure = getattr(self, "_ensure_motion_knee_chart", None)
            if callable(ensure):
                try:
                    ensure()
                except Exception:
                    return
            if getattr(self, "chart_canvas", None) is None:
                return
        if not self.sequence or not self.pose_indices:
            return

        from stablewalk.ui.viewers.gait_cycle_chart import (
            draw_knee_chart_cycle_mode,
            draw_knee_chart_time_mode,
            style_gait_chart,
        )

        axis_mode = getattr(self, "var_knee_chart_axis", None)
        mode = axis_mode.get() if axis_mode is not None else "Video Time"
        if mode == "Time":
            mode = "Video Time"

        gait_events = None
        gait_cycle = getattr(self, "_gait_cycle", None)
        if gait_cycle is not None:
            gait_events = list(gait_cycle.events)

        if mode != "Gait Cycle %" and self._knee_angle_series is None:
            self._rebuild_knee_angle_series()
        if playhead_list_pos is None:
            from stablewalk.ui.playback_state import committed_playback_state

            committed = committed_playback_state(self)
            playhead_list_pos = (
                int(committed.current_frame_index)
                if committed is not None
                else self.current_pos
            )
        chart_key = (
            id(self.sequence),
            id(self._knee_angle_series),
            id(gait_cycle),
            mode,
            self._knee_angle_source_preference(),
        )
        if (
            mode != "Gait Cycle %"
            and getattr(self, "_knee_chart_data_key", None) == chart_key
            and self._knee_angle_series is not None
            and self._knee_angle_series.times_s.size
        ):
            index = max(
                0,
                min(int(playhead_list_pos), self._knee_angle_series.times_s.size - 1),
            )
            if self._move_existing_chart_playheads(
                self.fig,
                self.chart_canvas,
                float(self._knee_angle_series.times_s[index]),
                list_pos=index,
            ):
                return

        if (
            mode == "Gait Cycle %"
            and getattr(self, "_knee_chart_data_key", None) == chart_key
        ):
            from stablewalk.ui.viewers.comparison_cursor import estimate_gait_cycle_percent

            t_s = None
            if self._knee_angle_series is not None and self._knee_angle_series.times_s.size:
                index = max(
                    0,
                    min(int(playhead_list_pos), self._knee_angle_series.times_s.size - 1),
                )
                t_s = float(self._knee_angle_series.times_s[index])
            elif self.sequence and self.pose_indices:
                i = max(0, min(int(playhead_list_pos), len(self.pose_indices) - 1))
                fi = self.pose_indices[i]
                t_s = self._timestamp_for_frame_index(fi)
            frame_i = 0
            if self.skeleton_player is not None:
                frame_i = int(self.skeleton_player.state.frame_index)
            if t_s is not None:
                pct = estimate_gait_cycle_percent(
                    gait_cycle, time_s=float(t_s), frame_index=frame_i
                )
                if pct is not None and self._move_existing_chart_playheads(
                    self.fig,
                    self.chart_canvas,
                    float(pct),
                    list_pos=playhead_list_pos,
                ):
                    return

        # Never rebuild knee geometry mid-play or mid-scrub — tip/cursor only.
        from stablewalk.ui.tk.playback_render_policy import (
            allow_graph_geometry_rebuild,
            mark_analysis_dirty,
        )

        if not allow_graph_geometry_rebuild(self, force=False):
            mark_analysis_dirty(self, "knee")
            return

        self.ax_chart.clear()
        # Invalidate blit cache after a full rebuild.
        if getattr(self, "fig", None) is not None:
            self.fig._stablewalk_playhead_bg = None  # type: ignore[attr-defined]

        if mode == "Gait Cycle %":
            drawn = draw_knee_chart_cycle_mode(
                self.ax_chart,
                getattr(self, "_gait_features", None),
                show_envelope=True,
            )
            self.ax_chart._stablewalk_x_is_percent = True  # type: ignore[attr-defined]
            if drawn:
                from stablewalk.ui.viewers.chart_playhead import (
                    PlayheadState,
                    draw_chart_playhead,
                    playhead_from_gui,
                )
                from stablewalk.ui.viewers.comparison_cursor import (
                    estimate_gait_cycle_percent,
                )

                t_s = None
                if (
                    self._knee_angle_series is not None
                    and self._knee_angle_series.times_s.size
                ):
                    index = max(
                        0,
                        min(
                            int(playhead_list_pos),
                            self._knee_angle_series.times_s.size - 1,
                        ),
                    )
                    t_s = float(self._knee_angle_series.times_s[index])
                frame_i = (
                    int(self.skeleton_player.state.frame_index)
                    if self.skeleton_player is not None
                    else 0
                )
                if t_s is not None:
                    pct = estimate_gait_cycle_percent(
                        gait_cycle, time_s=float(t_s), frame_index=frame_i
                    )
                    if pct is not None:
                        playhead = playhead_from_gui(
                            self,
                            time_s=float(pct),
                        )
                        if playhead is None:
                            playhead = PlayheadState(
                                time_s=float(pct),
                                frame_index=int(playhead_list_pos) + 1
                                if playhead_list_pos is not None
                                else frame_i + 1,
                                list_pos=playhead_list_pos,
                            )
                        draw_chart_playhead(
                            self.ax_chart,
                            playhead,
                            show_label=True,
                        )
        else:
            self.ax_chart._stablewalk_x_is_percent = False  # type: ignore[attr-defined]
            draw_knee_chart_time_mode(
                self.ax_chart,
                self.sequence,
                self.pose_indices,
                playhead_list_pos=playhead_list_pos,
                series=self._knee_angle_series,
                ik_mot_path=self._opensim_ik_mot_path(),
                source_preference=self._knee_angle_source_preference(),
                ik_quality_ok=self._opensim_ik_quality_ok(),
                gait_events=gait_events,
                gait_cycle=gait_cycle,
            )
        self._knee_chart_data_key = chart_key

        style_gait_chart(self.ax_chart, self.fig)

        from stablewalk.ui.viewers.chart_hover import ChartHoverPoint
        from stablewalk.ui.viewers.chart_interactions import finalize_chart_interactions
        from stablewalk.ui.viewers.chart_playhead import cache_static_chart_background
        from stablewalk.ui.viewers.knee_angle_chart import register_knee_time_chart_hover_points

        hover_points: list[ChartHoverPoint] = []
        if mode != "Gait Cycle %" and self._knee_angle_series is not None:
            register_knee_time_chart_hover_points(
                self.ax_chart,
                self._knee_angle_series,
                self.pose_indices,
                hover_points,
            )
        finalize_chart_interactions(self.fig, self.chart_canvas, hover_points=hover_points)
        self._update_knee_interpretation_panel(chart_mode=mode)
        self._paint_canvas_now(self.chart_canvas)
        # Snapshot static pixels so the next play ticks can blit the cursor only.
        if getattr(self, "playing", False):
            cache_static_chart_background(self.fig, self.chart_canvas, [self.ax_chart])
        if not getattr(self, "playing", False) or self._should_sync_dashboard_scroll():
            self._request_dashboard_scroll_sync()

    def _update_compare_chart(self) -> None:
        if not self._compare:
            return
        self.ax_chart.clear()
        self.ax_chart.set_facecolor(PANEL)
        c = self._compare
        if c.reference_knee_left:
            self.ax_chart.plot(
                c.reference_knee_left,
                color=ACCENT,
                linestyle="--",
                label="Ref L knee",
            )
        if c.sample_knee_left:
            self.ax_chart.plot(
                c.sample_knee_left,
                color=WARNING,
                label="Sample L knee",
            )
        self.ax_chart.set_title(
            "Gait comparison — knee angles", color=TEXT, fontsize=10, fontweight="medium", pad=6
        )
        self.ax_chart.legend(
            facecolor=ELEVATED, edgecolor=BORDER, labelcolor=TEXT, fontsize=8, framealpha=0.9
        )
        self.ax_chart.tick_params(colors=MUTED, labelsize=8)
        self.ax_chart.grid(True, color=BORDER, alpha=0.35, linestyle="--", linewidth=0.6)
        for spine in self.ax_chart.spines.values():
            spine.set_color(BORDER)
        self.fig.tight_layout(pad=1.2)
        self.chart_canvas.draw_idle()

    def _request_app_close(self) -> None:
        """Stop timers and close cleanly; warn if analysis/export is still running."""
        if getattr(self, "_app_closing", False):
            return
        busy = bool(getattr(self, "_analysis_compute_active", False))
        busy = busy or bool(getattr(self, "_processing", False))
        try:
            busy = busy or self._advanced_export_busy()
        except Exception:
            pass
        worker = getattr(self, "_opensim_worker", None)
        if worker is not None and worker.is_alive():
            busy = True
        worker = getattr(self, "_analysis_worker", None)
        if worker is not None and worker.is_alive():
            busy = True
        if busy:
            if not messagebox.askyesno(
                "Close StableWalk",
                "Analysis or export is still running.\n\n"
                "Close anyway? Background work will be abandoned.",
                parent=self.root,
            ):
                return
        self._app_closing = True
        try:
            self._stop_all_gui_timers()
        except Exception:
            logger.debug("timer teardown on close failed", exc_info=True)
        try:
            self.root.destroy()
        except tk.TclError:
            try:
                self.root.quit()
            except Exception:
                pass

    def run(self) -> None:
        self.root.mainloop()


def log_opensim_startup_status_deferred(logger) -> None:
    """Defer OpenSim SDK probe so GUI chrome can appear first."""
    try:
        from stablewalk.opensim_sdk import log_opensim_startup_status

        log_opensim_startup_status(logger)
    except Exception:
        pass


def launch_gui(poses_path: str | Path | None = None) -> None:
    """Start the desktop GUI (blocking)."""
    # Avoid Windows DPI bitmap-scaling ghosts (double tabs / LabelFrame titles).
    try:
        from stablewalk.ui.tk.layout_scale import apply_process_dpi_awareness

        apply_process_dpi_awareness()
    except Exception:
        pass

    app = StableWalkGUI(poses_path=poses_path)
    try:
        app.root.protocol("WM_DELETE_WINDOW", app._request_app_close)
    except Exception:
        pass
    # Probe OpenSim after the window exists (idle) — not on the critical path.
    try:
        app.root.after(0, lambda: log_opensim_startup_status_deferred(logging.getLogger("stablewalk")))
    except Exception:
        pass
    app.run()


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="StableWalk visualization GUI")
    parser.add_argument(
        "poses",
        nargs="?",
        help="Pose JSON name or path (default: latest in data/output/poses/)",
    )
    args = parser.parse_args()
    launch_gui(args.poses)


if __name__ == "__main__":
    main()
