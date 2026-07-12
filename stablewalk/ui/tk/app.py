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
from stablewalk.core.pipeline import PipelineResult, run_gait_pipeline
from stablewalk.core.pipeline_reset import release_all_captures
from stablewalk.pose.skeleton import FrameRgbCache, render_frame_with_skeleton
from stablewalk.analysis.stability import StabilityReport, analyze_stability
from stablewalk.analysis.biomech_stability import (
    StabilityResult,
    analyze_biomech_stability,
)
from stablewalk.analysis.stability_validity import (
    assess_stability_result_validity,
    format_validity_display,
)
from stablewalk.analysis.gait_analysis_summary import (
    build_gait_analysis_summary,
    format_summary_display,
)
from stablewalk.analysis.gait_cycle_analysis import (
    GaitCycleAnalysisResult,
    analyze_gait_cycles,
)
from stablewalk.analysis.gait_feature_analysis import analyze_gait_features
from stablewalk.analysis.foot_contact_analysis import (
    FootContactAnalysisResult,
    analyze_foot_contact,
)
from stablewalk.analysis.biomechanical import BiomechanicalAnalysisResult, run_biomechanical_analysis
from stablewalk.analysis.estimated_vgrf_analysis import (
    EstimatedVGRFResult,
    analyze_estimated_vgrf,
)
from stablewalk.analysis.virtual_grf import VirtualForceResult, estimate_virtual_grf
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
    detected_frame_indices,
    load_pose_sequence,
    sequence_needs_enrichment,
)
from stablewalk.analysis.comparison import GaitComparison, compare_pose_files
from stablewalk.pose.interpolation import (
    interpolate_skeleton_3d,
    normalize_skeleton_height,
    smooth_skeleton_temporal,
)
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
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
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
from stablewalk.ui.selection_state import (
    GUI_DOF_ITEM_IDS,
    GUI_DOF_LABELS,
    SelectionState,
    item_for_joint,
    label_for_item,
)
from stablewalk.models.joint_registry import JOINT_DISPLAY_NAMES
from stablewalk.ui.skeleton_player import SkeletonPlayer
from stablewalk.ui.viewers.dof_trajectory_3d import TRAJECTORY_COLORS, draw_single_dof_trajectory_3d
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
    demo_cached_file_ready,
    demo_category_tagline,
    demo_default_dof_item,
    demo_gait_interpretation,
    demo_exists,
    demo_path,
    demo_stream_source,
    demo_validation_status,
    example_by_key,
    is_demo_video_path,
    missing_file_message,
    missing_file_placeholder,
)
from stablewalk.ui.media.demo_comparison import (
    DemoGaitComparisonRow,
    build_comparison_row,
)
from stablewalk.pose.video_validation import validate_video_source
from stablewalk.ui.theme import (
    ACCENT,
    ACCENT_ALT,
    BG,
    BORDER,
    ELEVATED,
    EMPTY_SELECT_DOF_CHART,
    EMPTY_VIDEO_TEXT,
    INFO,
    MUTED,
    DANGER,
    DASHBOARD_CARD_PAD,
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
    format_stability_short,
    FONT_TITLE,
    FONT_UI,
    FONT_METRIC,
    FONT_MONO_SM,
    FONT_UI_SM,
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
        self.root = root or tk.Tk()
        self.root.title("StableWalk — Gait Analysis Dashboard")
        from stablewalk.ui.tk.dashboard_responsive import (
            MIN_WINDOW_HEIGHT,
            MIN_WINDOW_WIDTH,
            initial_window_geometry,
        )

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
        self.skeleton_player: SkeletonPlayer | None = None
        self._data_refresh_s = 0.5
        self._last_data_refresh = 0.0
        self._last_realtime_refresh = 0.0
        self._last_dof_table_refresh = 0.0
        self._dof_table_history = DofPositionTableHistory()
        self._session_collector = SessionKinematicCollector()
        self._bilateral_foot_collector = BilateralFootCollector()
        self._ground_clearance_prev_phase: tuple[str | None, str | None] = (None, None)
        self._overview_video_frame_index: int | None = None
        self._foot_clearance_frame_index: int | None = None
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
        self._video_resize_after: str | None = None
        self._after_id: str | None = None
        self._processing = False
        self._poses_path: Path | None = None
        self._skeleton_scale: float = 1.0
        self._view_limit_3d: float = 0.55
        self._rom_cache: dict[str, tuple[float, float]] = {}
        self._3d_motion_trail: list[Skeleton3D] = []
        self._anim_hz = 60
        self._walk_simulation: WalkSimulation | None = None
        self._skeleton_cache: dict[int, Skeleton3D] = {}
        self._rgb_cache = FrameRgbCache(max_entries=128)
        self._panel_tick = 0
        self._panel_update_stride = 1
        self._play_anim_hz = 24
        self._last_positions: list[tuple[str, float, float, float]] = []
        self._3d_flush_scheduled = False
        self._pending_3d_args: tuple | None = None
        self._3d_play_stride = 2
        self._compare: GaitComparison | None = None
        self._stability: StabilityReport | None = None
        self._biomech: StabilityResult | None = None
        self._gait_cycle: GaitCycleAnalysisResult | None = None
        self._gait_features = None
        self._foot_contact: FootContactAnalysisResult | None = None
        self._estimated_vgrf: EstimatedVGRFResult | None = None
        self._biomech_analysis: BiomechanicalAnalysisResult | None = None
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
        self._preset_cycle_index: int = 0
        self._preset_after_id: str | None = None
        self._chain_next_on_fail: bool = False
        self._next_video_attempts: int = 0
        self.auto_load_on_change = tk.BooleanVar(value=True)
        self.smooth_motion = tk.BooleanVar(value=True)

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

        # Only auto-load when CLI explicitly passes a pose file (avoid stale session on startup)
        if poses_path is not None:
            initial = self._resolve_initial_path(poses_path)
            if initial:
                self.load_poses(initial, fresh=True)
        else:
            self._show_welcome_state()

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
        self.selection.clear()
        self._clear_demo_gait_context()
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
        if hasattr(self, "ax_3d"):
            self._update_interactive_skeleton(force_draw=True)
        self._update_ground_clearance_visibility()
        self.status.configure(
            text="Ready — load a walking video or choose a Demo Gait Example"
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
        self.status.configure(text="●  STOPPED · Presentation demo loaded · press Play")

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

    def _build_header(self) -> None:
        self._dashboard_header = ttk.Frame(self.root)
        self._dashboard_header.pack(fill=tk.X, side=tk.TOP)
        header = tk.Frame(self._dashboard_header, bg=SURFACE, height=42)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        title_row = tk.Frame(header, bg=SURFACE)
        title_row.pack(side=tk.LEFT, padx=PAD_MD, pady=6)
        tk.Label(
            title_row, text="StableWalk", bg=SURFACE, fg=ACCENT, font=FONT_TITLE
        ).pack(side=tk.LEFT)
        from stablewalk.ui.theme import DASHBOARD_SUBTITLE, FONT_UI_SM

        tk.Label(
            title_row,
            text=f"  ·  {DASHBOARD_SUBTITLE}",
            bg=SURFACE,
            fg=MUTED,
            font=FONT_UI_SM,
        ).pack(side=tk.LEFT, padx=(PAD_XS, 0))
        tk.Frame(self._dashboard_header, bg=BORDER, height=1).pack(fill=tk.X)

    def _build_menu(self) -> None:
        mc = menu_colors()
        menubar = tk.Menu(self.root, **mc)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0, **mc)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open pose JSON…", command=self._menu_open, accelerator="Ctrl+O")
        file_menu.add_command(label="Open output folder", command=self._open_output_folder)
        file_menu.add_command(label="Export JSON (save as…)", command=self._export_json)
        file_menu.add_command(label="Export Analysis…", command=self._export_analysis)
        file_menu.add_command(label="Export OpenSim (.trc + .mot)…", command=self._export_opensim)
        file_menu.add_command(label="Export Analysis Data…", command=self._export_analysis_data)
        file_menu.add_command(label="Save Session to files…", command=self._save_session_to_files)
        file_menu.add_command(label="Load Session…", command=self._load_session_from_files)
        file_menu.add_command(label="Import Analysis Data…", command=self._import_analysis_data)
        file_menu.add_separator()
        file_menu.add_command(label="Save Session to database…", command=self._save_analysis_session)
        file_menu.add_separator()
        file_menu.add_command(label="Compare with JSON…", command=self._compare_gait)
        file_menu.add_command(label="Compare with video…", command=self._compare_video)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

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

        help_menu = tk.Menu(menubar, tearoff=0, **mc)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Keyboard shortcuts", command=self._show_shortcuts)
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
        help_menu.add_command(label="About StableWalk", command=self._show_about)
        help_menu.add_command(label="Edit men's walk URL list…", command=self._open_url_list)

        demo_menu = tk.Menu(menubar, tearoff=0, **mc)
        menubar.add_cascade(label="Demo", menu=demo_menu)
        demo_menu.add_command(
            label="Reset presentation demo",
            command=self._reset_presentation_demo,
        )
        demo_menu.add_separator()
        demo_menu.add_command(label="▶ Full video comparison demo", command=self._run_demo)

    def _build_input_bar(self) -> None:
        """Compact toolbar: video source, analyze, and demo gait shortcuts."""
        parent = getattr(self, "_dashboard_header", self.root)
        bar = ttk.Frame(parent, padding=(PAD_MD, PAD_SM, PAD_MD, PAD_XS))
        bar.pack(fill=tk.X)

        row = ttk.Frame(bar)
        row.pack(fill=tk.X)

        _verified = list_runnable_presets(verified_only=True, urls_only=True)
        _default_label = (
            _verified[0].label
            if _verified
            else (STABLE_WALKING_PRESETS[0].label if STABLE_WALKING_PRESETS else "")
        )
        self.preset_var = tk.StringVar(value=_default_label)
        self.url_var = tk.StringVar(value=config.DEFAULT_WALKING_VIDEO_URL)

        ttk.Label(row, text="Source", style="Card.TLabel").pack(side=tk.LEFT, padx=(0, PAD_XS))
        self.preset_combo = ttk.Combobox(
            row, textvariable=self.preset_var, values=all_preset_labels(), width=18, state="readonly"
        )
        self.preset_combo.pack(side=tk.LEFT, padx=(0, PAD_XS))
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        self.url_entry = ttk.Entry(row, textvariable=self.url_var)
        self.url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, PAD_XS))
        self.url_entry.bind("<Return>", lambda _e: self._load_video())

        self.btn_browse = ttk.Button(
            row,
            text="Browse",
            style="Secondary.TButton",
            width=8,
            command=self._browse_video,
        )
        self.btn_browse.pack(side=tk.LEFT, padx=(0, PAD_XS))

        self.btn_full = ttk.Button(
            row,
            text="Analyze",
            style="Accent.TButton",
            width=9,
            command=self._run_full_analysis,
        )
        self.btn_full.pack(side=tk.LEFT, padx=(0, PAD_XS))
        self.btn_load = self.btn_full

        self.btn_input_settings = ttk.Button(
            row,
            text="\u2699",
            style="Secondary.TButton",
            width=3,
            command=self._show_input_settings_menu,
        )
        self.btn_input_settings.pack(side=tk.LEFT)

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

        row2 = ttk.Frame(bar)
        row2.pack(fill=tk.X, pady=(PAD_XS, 0))
        self.progress = ttk.Progressbar(
            row2, mode="determinate", style="Accent.Horizontal.TProgressbar"
        )
        self.progress.pack(fill=tk.X, expand=True)

        self._input_row2 = row2

        demo_row = ttk.Frame(bar)
        demo_row.pack(fill=tk.X, pady=(PAD_XS, 0))

        self._demo_gait_buttons: dict[str, ttk.Button] = {}
        for ex in DEMO_GAIT_EXAMPLES:
            btn = ttk.Button(
                demo_row,
                text=ex.button_label,
                style="Secondary.TButton",
                width=12,
                command=lambda k=ex.key: self._load_demo_gait(k),
            )
            btn.pack(side=tk.LEFT, padx=(0, PAD_XS))
            self._demo_gait_buttons[ex.key] = btn

        self.btn_save_demo_comparison = ttk.Button(
            demo_row,
            text="Save",
            style="Compact.TButton",
            width=7,
            command=self._save_demo_comparison,
            state=tk.DISABLED,
        )
        self.btn_save_demo_comparison.pack(side=tk.LEFT, padx=(PAD_XS, 0))

        self.lbl_demo_category_hint = tk.Label(
            demo_row,
            text="",
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_SM,
            anchor="w",
            justify=tk.LEFT,
            wraplength=520,
        )
        self.lbl_demo_category_hint.pack(side=tk.LEFT, padx=(PAD_SM, 0), fill=tk.X, expand=True)

        self._demo_info_btn = ttk.Label(demo_row, text="Info", style="Card.TLabel", cursor="hand2")
        self._demo_info_btn.pack(side=tk.RIGHT)
        self._demo_info_btn.bind("<Button-1>", lambda _e: self._show_demo_video_details())
        create_tooltip(self._demo_info_btn, "About the active demo video")

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
        """Load a predefined demo gait video through the standard pipeline."""
        ex = example_by_key(key)
        if ex is None:
            return
        config.ensure_output_dirs()

        self._active_demo_gait = ex
        self._presentation_mode = False
        self._cancel_presentation_autoplay()

        resolved = demo_path(ex)
        logger.info("Demo gait resolved path: %s (exists=%s)", resolved, resolved.is_file())

        if ex.pexels_video_id is None:
            if not resolved.is_file():
                from stablewalk.ui.media.demo_download import download_demo_video

                download_demo_video(key, force=False)
            if not resolved.is_file():
                messagebox.showwarning("Demo video missing", missing_file_message(ex))
                return
            from stablewalk.ui.media.utah_abnormal import opencv_can_decode

            if not opencv_can_decode(resolved):
                messagebox.showwarning(
                    "Demo video error",
                    f"OpenCV cannot decode the demo file:\n{resolved}\n\n"
                    f"Run: python scripts/download_utah_abnormal_demo.py",
                )
                return
            source = str(resolved)
        else:
            source = demo_stream_source(ex)
            if demo_exists(ex) and not demo_cached_file_ready(ex):
                def _refresh_cache() -> None:
                    from stablewalk.ui.media.demo_download import download_demo_video

                    download_demo_video(key, force=True)

                threading.Thread(target=_refresh_cache, daemon=True).start()

        self.url_var.set(source)
        self.url_entry.delete(0, tk.END)
        self.url_entry.insert(0, source)
        self.preset_var.set(CUSTOM_URL_PRESET_LABEL)
        self._update_demo_analysis_title(ex)
        self._highlight_demo_button(key)
        self._sync_demo_category_hint()
        self._load_video(source=source, show_dialog=False, unique_session=True)

    def _sync_demo_category_hint(self) -> None:
        """Update category tagline and compare strip as soon as a demo is selected."""
        demo = getattr(self, "_active_demo_gait", None)
        hint = getattr(self, "lbl_demo_category_hint", None)
        if hint is not None:
            if demo is None:
                hint.configure(text="")
            else:
                from stablewalk.ui.theme import ACCENT_ALT, ORANGE, TEXT, WARNING

                tagline = demo_category_tagline(demo.key)
                fg = TEXT
                if demo.key == "abnormal":
                    fg = WARNING
                elif demo.key == "athletic":
                    fg = ACCENT_ALT
                elif demo.key == "normal":
                    fg = ORANGE
                hint.configure(
                    text=f"{demo.button_label}: {tagline}" if tagline else "",
                    fg=fg,
                )
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
        self._update_demo_analysis_title(None)
        self._highlight_demo_button(None)
        hint = getattr(self, "lbl_demo_category_hint", None)
        if hint is not None:
            hint.configure(text="")
        self._sync_demo_category_compare_strip()
        self._sync_demo_save_button()

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
            return
        lines = [
            ex.display_name,
            "",
            ex.source_attribution or ex.source_name,
            "",
            f"Source: {ex.source_url}",
        ]
        if ex.note:
            lines.extend(["", ex.note])
        messagebox.showinfo(ex.analysis_title, "\n".join(lines))

    def _apply_demo_presentation(self) -> None:
        """Refresh demo title and session label after a demo video loads."""
        if not self._active_demo_gait:
            return
        self._update_demo_analysis_title()
        self._session_display_src = self._active_demo_gait.display_name
        self.status.configure(
            text=f"✓  {self._active_demo_gait.button_label} — press Play"
        )
        self._sync_demo_category_hint()
        self._sync_demo_save_button()

    def _sync_demo_category_compare_strip(self) -> None:
        """Teacher-facing one-line comparison across Abnormal / Normal / Performance."""
        lbl = getattr(self, "lbl_overview_demo_compare", None)
        demo = getattr(self, "_active_demo_gait", None)
        if lbl is None:
            return
        if demo is None or self.gait_motion is None:
            lbl.configure(text="")
            return
        from stablewalk.ui.theme import ACCENT_ALT, ORANGE, WARNING

        usable, detected = self._resolved_gait_cycle_count()
        completeness = (
            self._biomech.completeness_pct if self._biomech is not None else None
        )
        comp = f" · {completeness:.0f}% complete" if completeness is not None else ""
        item_id = self._active_dof_item_id()
        joint = ""
        if item_id:
            from stablewalk.ui.dof_selection import label_for_item

            joint = label_for_item(item_id) or ""
        if demo.key == "abnormal":
            text = (
                f"COMPARE — Abnormal: walker-assisted gait · tracks {joint or 'hip'} "
                f"· {detected or 0} detected / {usable or 0} usable cycles{comp}"
            )
            fg = WARNING
        elif demo.key == "normal":
            text = (
                f"COMPARE — Normal: healthy steady walking · tracks {joint or 'knee'} "
                f"· {usable or 0} usable cycles · alternating swing/stance{comp}"
            )
            fg = ORANGE
        elif demo.key == "athletic":
            text = (
                f"COMPARE — Performance: fast side-view gait · tracks {joint or 'knee'} "
                f"· larger knee swing · {usable or 0} usable cycle(s){comp}"
            )
            fg = ACCENT_ALT
        else:
            text = ""
            fg = ACCENT_ALT
        lbl.configure(text=text, fg=fg)

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
        row = build_comparison_row(self.gait_motion, self._active_demo_gait, item_id)
        self._demo_comparison_rows[self._active_demo_gait.key] = row
        self._comparison_expanded = True
        self._refresh_demo_comparison_panel()
        self.status.configure(
            text=f"Comparison saved — {row.demo_type} · {row.joint_label}"
        )

    def _refresh_demo_comparison_panel(self) -> None:
        tree = getattr(self, "demo_comparison_tree", None)
        if tree is None:
            return
        tree.delete(*tree.get_children())
        has_rows = False
        for key in ("abnormal", "normal", "athletic"):
            row = self._demo_comparison_rows.get(key)
            if row is None:
                continue
            has_rows = True
            max_a = (
                f"{row.max_angle_deg:.1f}°"
                if row.max_angle_deg is not None
                else "—"
            )
            avg_v = (
                f"{row.avg_velocity:.3f} {row.velocity_unit}"
                if row.avg_velocity is not None
                else "—"
            )
            tree.insert(
                "",
                tk.END,
                values=(row.demo_type, row.joint_label, max_a, avg_v),
            )
        self._sync_gait_comparison_visibility(has_rows=has_rows)

    def _sync_gait_comparison_visibility(self, *, has_rows: bool | None = None) -> None:
        toggle = getattr(self, "btn_toggle_comparison", None)
        body = getattr(self, "comparison_body", None)
        if toggle is None:
            return
        if body is not None:
            body.pack_forget()
        toggle.configure(text="Compare Gaits")

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
        if hasattr(self, "video_label"):
            self.video_label.bind("<Configure>", self._on_video_label_resize, add="+")
        from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout

        self.root.after_idle(lambda: apply_responsive_layout(self))
        from stablewalk.ui.tk.dashboard_shell import (
            assert_dashboard_widget_singletons,
            print_structural_diagnostic_report,
        )
        from stablewalk.ui.tk.dashboard_notebook import reflow_tab_canvases
        from stablewalk.ui.tk.gui_layout_debug import log_gui_layout_audit_if_enabled

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
                "rom": "ROM",
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
            bg=PANEL, fg=TEXT, font=("Segoe UI", 9), relief=tk.FLAT, highlightthickness=0
        )
        self.motion_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.dof_tree.bind("<<TreeviewSelect>>", self._on_dof_tree_select)
        self.pos_3d_tree.bind("<<TreeviewSelect>>", self._on_pos_tree_select)

        self.robot_frame = ttk.LabelFrame(
            self.root,
            text="  Robot simulation  ",
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
            style="Secondary.TButton",
            command=self._stop_playback,
        )
        self.btn_stop_bar.pack(side=tk.LEFT, padx=(0, PAD_XS))

        self.btn_reset_bar = ttk.Button(
            playback,
            text="Reset",
            width=7,
            style="Secondary.TButton",
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
            style="Secondary.TButton",
            command=lambda: self._step(-1),
        )
        self._btn_step_fwd = ttk.Button(
            frame_grp,
            text="Frame ▶",
            width=8,
            style="Secondary.TButton",
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

        self._lbl_speed = ttk.Label(analysis, text="Speed", style="Card.TLabel")
        self._lbl_speed.pack(side=tk.LEFT, padx=(0, PAD_XS))
        self.speed_var = tk.DoubleVar(value=1.0)
        self._speed_scale = ttk.Scale(
            analysis,
            from_=0.25,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            length=72,
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
        self.lbl_frame_idx = ttk.Label(timeline, text="", style="Card.TLabel", width=1)

    def _build_status_bar(self) -> None:
        self.status = ttk.Label(self.root, text="Ready", style="Status.TLabel", anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    @staticmethod
    def _play_button_text(playing: bool) -> str:
        return "Pause" if playing else "Play"

    def _sync_play_buttons(self) -> None:
        text = self._play_button_text(self.playing)
        self.btn_play.configure(text=text)
        if hasattr(self, "btn_play_bar"):
            self.btn_play_bar.configure(text=text)

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
            self.btn_play_bar: "Play / pause",
            self.btn_stop_bar: "Stop (keeps the current frame)",
            self.btn_reset_bar: "Reset to the first frame",
            self.slider: "Timeline scrubber",
        }.items():
            create_tooltip(widget, tip)
        if hasattr(self, "cmb_skeleton_mode"):
            create_tooltip(self.cmb_skeleton_mode, "Skeleton display mode")
        explain = getattr(self, "lbl_dof_graph_explain_body", None)
        if explain is not None:
            create_tooltip(
                explain,
                "3D path colored from dim (start) to bright (current position)",
            )
        gc_left = getattr(self, "lbl_ground_clearance_left", None)
        gc_right = getattr(self, "lbl_ground_clearance_right", None)
        if gc_left is not None:
            create_tooltip(gc_left, "Left foot ground clearance (estimated body-scale)")
        if gc_right is not None:
            create_tooltip(gc_right, "Right foot ground clearance (estimated body-scale)")
        self._attach_opensim_tooltips()
        self._attach_export_tooltips()

    def _attach_export_tooltips(self) -> None:
        """Tooltips for the Detailed Joint Data & Export section."""
        if getattr(self, "_export_tooltips_attached", False):
            return
        tips = {
            "btn_view_detailed_data": (
                "Open a scrollable table of collected samples for the active joint."
            ),
            "btn_export_joint_csv": (
                "Export tracked joint samples as CSV (and JSON) to the output folder."
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
        self.ax_chart.clear()
        self.ax_chart.set_title("Knee angles", color=TEXT, fontsize=10, fontweight="medium", pad=6)
        self.ax_chart.tick_params(colors=MUTED, labelsize=8)
        self.ax_chart.grid(True, color=BORDER, alpha=0.35, linestyle="--", linewidth=0.6)
        for spine in self.ax_chart.spines.values():
            spine.set_color(BORDER)
        self.ax_chart.set_facecolor(PANEL)
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

    def _set_gait_motion(self, recording: GaitMotionRecording) -> None:
        """Attach a gait recording and reset the skeleton player."""
        recording.build_time_series()
        self.gait_motion = recording
        self._motion_series_cache_key = None
        self._motion_frame_series_cache = None
        self.skeleton_player = SkeletonPlayer(recording, smooth=self.smooth_motion.get())
        self.skeleton_player.stop(reset=True)
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

    def _sync_transport_labels(self) -> None:
        if not self.skeleton_player:
            self.lbl_frame.configure(text="0.00s / 0.00s")
            if hasattr(self, "lbl_frame_idx"):
                self.lbl_frame_idx.configure(text="")
            return
        t = self.skeleton_player.time_at_current()
        dur = self.skeleton_player.duration_s
        self.lbl_frame.configure(text=f"{t:.2f}s / {dur:.2f}s")
        if hasattr(self, "lbl_frame_idx"):
            self.lbl_frame_idx.configure(text="")

    def _update_interactive_skeleton(self, *, force_draw: bool = False) -> None:
        """Draw the human skeleton for the current playback position."""
        if not self.skeleton_player:
            self.ax_3d.cla()
            setup_skeleton_axes(self.ax_3d, display_mode=self._skeleton_display_mode_key())
            self.ax_3d.text(
                0.5, 0.5, "No skeleton data", transform=self.ax_3d.transAxes,
                ha="center", color=MUTED, fontsize=11,
            )
            self.canvas_3d.draw_idle()
            self._dof_step_previews = []
            self._refresh_dof_step_panel()
            return

        snap = self.skeleton_player.current_snapshot()
        if not snap:
            return

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
            skel_status.configure(text=mode)

        highlight = self._resolve_highlight_joints()
        analysis_force = force_draw or not is_playing
        if self._realtime_refresh_due(force=analysis_force):
            self._refresh_realtime_analysis(snapshot=snap, force_draw=force_draw)
        self._overview_video_frame_index = snap.frame_index
        self._update_gait_cycle_panel(self._gait_cycle, frame_index=snap.frame_index)
        playhead_t = None
        if self.sequence:
            pf = next(
                (f for f in self.sequence.frames if f.frame_index == snap.frame_index),
                None,
            )
            if pf is not None:
                playhead_t = pf.timestamp_s
        self._update_contact_gait_chart(playhead_time_s=playhead_t)
        self._update_biomechanics_chart(playhead_time_s=playhead_t)
        self._refresh_bilateral_ground_clearance(snapshot=snap)
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
        foot_skeleton_labels = self._foot_skeleton_labels_for_frame(snap.frame_index)
        self._last_skeleton_foot_labels = foot_skeleton_labels
        com_ov, poly_ov, dir_ov, foot_c = self._biomech_overlays_for_frame(snap.frame_index)
        draw_gait_skeleton(
            self.ax_3d,
            snap,
            clear=True,
            paused=show_detail,
            show_labels=False,
            show_legend=False,
            highlight_joints=highlight,
            labeled_joints=labeled,
            motion_arrows=self._motion_arrows_from_previews() if show_detail else None,
            title="",
            display_mode=self._skeleton_display_mode_key(),
            ground_floor_y=self._ground_floor_y_for_skeleton(),
            foot_skeleton_labels=foot_skeleton_labels,
            foot_contact=foot_c,
            com_overlay=com_ov,
            support_polygon=poly_ov,
            gait_direction=dir_ov,
        )
        if force_draw or show_detail:
            self.canvas_3d.draw_idle()
            if not is_playing:
                self.root.after_idle(self._fit_skeleton_canvas)
        else:
            self.canvas_3d.draw_idle()
        self._sync_transport_labels()

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
        low = message.lower()
        if "success" in low or "complete" in low:
            prefix = "✓  "
        elif "fail" in low or "error" in low:
            prefix = "✕  "
        elif "load" in low or "process" in low or "analyz" in low:
            prefix = "◐  "
        else:
            prefix = "●  "
        display = prefix + message
        self.status.configure(text=display)
        if self._demo_running:
            self._show_overlay(message)

    def _lock_ui(self, locked: bool) -> None:
        state = tk.DISABLED if locked else tk.NORMAL
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
        self.url_entry.configure(state="readonly" if locked else tk.NORMAL)

    def _reset_session(self, *, message: str = "Loading new video...") -> None:
        """Clear ALL gait state so no data leaks between videos."""
        release_all_captures()
        self._cancel_timer()
        self.playing = False
        self._3d_flush_scheduled = False
        self._pending_3d_args = None
        self._sync_play_buttons()
        self.sequence = None
        self.pose_indices = []
        self.current_pos = 0
        self._poses_path = None
        self._walk_simulation = None
        self._skeleton_cache = {}
        self._compare = None
        self._stability = None
        self._run_metadata = None
        self._active_run_name = None
        self._opensim_export_completed = False
        self._opensim_status_override = None
        self._loaded_content_key = ""
        self._playback_pos = 0.0
        self._photo = None
        self._skeleton_scale = 1.0
        self._view_limit_3d = 0.55
        self._rom_cache = {}
        self._3d_motion_trail = []
        self._last_video_list_pos = -1
        self._motion_series = None
        self.gait_motion = None
        self.skeleton_player = None
        self._last_data_refresh = 0.0
        self._last_dof_table_refresh = 0.0
        self._dof_table_history.clear()
        self._session_collector.clear()
        self._bilateral_foot_collector.clear()
        self._ground_clearance_prev_phase = (None, None)
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
        self.chart_canvas.draw_idle()
        self.ax_3d.cla()
        setup_skeleton_axes(self.ax_3d)
        self.canvas_3d.draw_idle()
        self.ax_robot.cla()
        setup_3d_axes(self.ax_robot)
        self.canvas_robot.draw_idle()
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
    ) -> bool:
        """Load pose JSON; return False on failure. If fresh, rebuild all views from frame 0."""
        path = Path(path)
        if fresh:
            self._skeleton_cache = {}
            self._walk_simulation = None
            self._compare = None
            self.sequence = None
            self.pose_indices = []
            self._dof_table_history.clear()
            self._session_collector.clear()
            self._bilateral_foot_collector.clear()
            self._ground_clearance_prev_phase = (None, None)
        try:
            self.sequence = load_pose_sequence(path)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            if not self._demo_running:
                messagebox.showerror("Load error", f"Could not load pose data:\n{exc}")
            return False

        self._rgb_cache.clear()
        self._panel_tick = 0
        self.pose_indices = detected_frame_indices(self.sequence)
        if not self.pose_indices:
            if not self._demo_running:
                messagebox.showwarning(
                    "No poses",
                    "This file has no detected full-body poses.\n"
                    "Run: python main.py --url --view",
                )
            return False

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
        try:
            if sequence_needs_enrichment(self.sequence):
                from stablewalk.pose.enrichment import enrich_pose_sequence

                enrich_pose_sequence(self.sequence)

            self._stability = analyze_stability(self.sequence)
            self._biomech = analyze_biomech_stability(self.sequence)
            self._update_stability_panel(self._biomech)
            self._apply_default_dof_projection()
            self.status.configure(text=f"✓  {display_src} — press Play")
            self._prepare_3d_sequence()
            self._build_walk_simulation()
            self._motion_series = MotionSeriesIndex.from_sequence(
                self.sequence, self.pose_indices
            )
            self._set_gait_motion(pose_sequence_to_gait_motion(self.sequence))
            if self.gait_motion is not None:
                self._gait_cycle = analyze_gait_cycles(self.gait_motion)
                self._foot_contact = analyze_foot_contact(
                    self.gait_motion, cycles=self._gait_cycle
                )
                self._estimated_vgrf = analyze_estimated_vgrf(
                    self.gait_motion, self._foot_contact
                )
                ik_mot = self._opensim_ik_mot_path()
                self._gait_features = analyze_gait_features(
                    self.gait_motion,
                    self._gait_cycle,
                    sequence=self.sequence,
                    ik_mot_path=ik_mot,
                )
                self._virtual_grf = estimate_virtual_grf(
                    self.gait_motion,
                    self._gait_cycle,
                    gait_features=self._gait_features,
                    sequence=self.sequence,
                )
                self._biomech_analysis = run_biomechanical_analysis(
                    self.gait_motion,
                    self.sequence,
                    cycles=self._gait_cycle,
                    contact=self._foot_contact,
                    features=self._gait_features,
                    stability=self._biomech,
                )
            else:
                self._gait_cycle = None
                self._foot_contact = None
                self._estimated_vgrf = None
                self._biomech_analysis = None
                self._gait_features = None
                self._virtual_grf = None
            self._update_gait_cycle_panel(self._gait_cycle)
            self._refresh_physics_force_panel()
            self._update_contact_gait_chart()
            self._update_biomechanics_panel()
            self._update_biomechanics_chart()
            self._refresh_real_to_sim_advanced_panel()
            self._knee_chart_mode_user_set = False
            self._rebuild_knee_angle_series()
            self._apply_default_knee_chart_mode()
            self._update_chart()
            self._update_compare_chart()
            self._show_pose_at(0)
            detected_n = sum(1 for f in self.sequence.frames if f.detected)
            self._update_session_selection_overview()
            self._opensim_ik_state = None
            self._refresh_opensim_status()
            self._update_dof_table_controls_state()
        except Exception as exc:
            logger.exception("Failed to display loaded poses")
            if not self._demo_running:
                messagebox.showerror(
                    "Load error",
                    f"Pose data loaded but the dashboard could not start:\n{exc}",
                )
            return False

        if self._current_source:
            detected = is_demo_video_path(self._current_source)
            if detected:
                self._active_demo_gait = detected
        self._apply_demo_presentation()

        if hasattr(self, "canvas_dof_traj"):
            self._sync_dof_analysis_panel_state()
            self._refresh_selected_dof_trajectory_3d(force_draw=True)

        self._request_dashboard_scroll_sync()
        return True

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

        if self.smooth_motion.get() and len(skeletons) >= 3:
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
            if self.smooth_motion.get()
            else sim.from_pose_sequence(self.sequence)
        )

    def _run_full_analysis(self) -> None:
        """Run complete pipeline: video → pose → DOF → JSON → 3D → simulation."""
        source = self._get_input_source()
        if source and is_demo_video_path(source) is None:
            self._clear_demo_gait_context()
        self._load_video()

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
        if new_key == self._loaded_content_key and self.sequence and not unique_session:
            logger.info("Same video content key — forcing fresh session folder")
            unique_session = True
        logger.info(
            "Loading video — source=%s key=%s… session=%d",
            source[:80],
            new_key[:12],
            session_id,
        )

        if max_frames is None and not self._demo_running:
            if detected_demo is not None:
                max_frames = DEMO_MAX_FRAMES
            else:
                max_frames = config.GUI_MAX_FRAMES_PER_LOAD

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
        self.status.configure(text=f"✕  Load failed — try Next ▶")
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
        self._reset_session(message="Loading new video...")
        self._processing = True
        self._lock_ui(True)
        self.progress["value"] = 0

        run_name = derive_run_name(source, unique_session=unique_session)
        thread = threading.Thread(
            target=self._pipeline_worker,
            args=(source, run_name, session_id, max_frames, validate_mode),
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
    ) -> None:
        try:
            def on_progress(msg: str, frac: float) -> None:
                ui_msg = self._progress_to_ui_message(msg, frac)
                self.root.after(
                    0,
                    lambda m=ui_msg, f=frac, sid=session_id: self._set_progress(m, f, sid),
                )

            result = run_gait_pipeline(
                source,
                run_name=run_name,
                validate=validate_mode,
                max_frames=max_frames,
                force_reprocess=True,
                on_progress=on_progress,
                pose_backend=self._selected_pose_backend(),
            )
            self.root.after(
                0,
                lambda r=result, sid=session_id: self._pipeline_done(r, None, sid),
            )
        except Exception as exc:
            err = exc
            self.root.after(
                0,
                lambda err=err, sid=session_id: self._pipeline_done(None, err, sid),
            )

    @staticmethod
    def _progress_to_ui_message(msg: str, frac: float) -> str:
        low = msg.lower()
        if frac < 0.12:
            return "Loading new video..."
        if "extract" in low or "frame" in low or frac < 0.5:
            return "Processing frames..."
        if "pose" in low or "saving" in low:
            return "Extracting gait features..."
        if frac >= 0.95:
            return "Video loaded successfully"
        return "Analyzing walking pattern..."

    def _set_progress(self, message: str, fraction: float, session_id: int) -> None:
        if session_id != self._session_id:
            return
        self.progress["value"] = fraction * 100
        self.status.configure(text=message)
        if self._demo_running:
            self._show_overlay(message)

    def _pipeline_done(
        self,
        result: PipelineResult | None,
        error: Exception | None,
        session_id: int,
    ) -> None:
        if session_id != self._session_id:
            return

        self._processing = False
        if not self._demo_running:
            self._lock_ui(False)
        self.progress["value"] = 100 if result else 0

        callback = self._pipeline_callback
        self._pipeline_callback = None

        if error:
            self._hide_overlay()
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
        ):
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

        preset = preset_by_label(self.preset_var.get())
        title = preset.label if preset else Path(result.source).name
        if self._active_demo_gait:
            title = self._active_demo_gait.display_name
        self.status.configure(
            text=f"✓  {title} — {result.frame_count} frames — press Play"
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
            self._compare = compare_pose_files(str(self._poses_path), path)
            self._show_compare_note(self._compare.summary)
            self._update_compare_chart()
        except Exception as exc:
            messagebox.showerror("Compare error", str(exc))

    def _export_json(self) -> None:
        self._export_analysis(default_ext=".json")

    def _export_analysis(self, default_ext: str | None = None) -> None:
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

        summary = result.gait_summary or build_gait_analysis_summary(result)
        display = format_summary_display(summary)
        validity = result.validity or assess_stability_result_validity(result)

        self._update_gait_summary_cards_panel(result, summary, display, validity)

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

    def _update_gait_summary_cards_panel(
        self,
        result: "StabilityResult",
        summary,
        display: dict[str, str],
        validity,
    ) -> None:
        """Populate Overview sidebar — three scores and short explanations."""
        from stablewalk.ui.dashboard_interpretability import (
            format_analysis_confidence_level,
            format_score_over_100,
            gait_quality_evidence_badge,
            interpret_analysis_confidence,
            interpret_gait_quality,
            interpret_movement_stability,
            truncate_dashboard_explanation,
        )

        ms = summary.movement_stability.score
        ms_lbl = getattr(self, "lbl_summary_ms_value", None)
        if ms_lbl is not None:
            ms_fg = TEXT if ms is not None else MUTED
            if ms is not None and ms >= 70:
                ms_fg = SUCCESS
            elif ms is not None and ms < 45:
                ms_fg = WARNING
            ms_lbl.configure(text=format_score_over_100(ms), fg=ms_fg)
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
                from stablewalk.ui.theme import ACCENT_ALT, ORANGE, WARNING

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
        gq_lbl = getattr(self, "lbl_summary_gq_value", None)
        if gq_lbl is not None:
            gq_fg = TEXT if gq is not None else MUTED
            if gq is not None and gq < 45:
                gq_fg = WARNING
            elif gq is not None and gq >= 68:
                gq_fg = SUCCESS
            gq_lbl.configure(text=format_score_over_100(gq), fg=gq_fg)
        badge_lbl = getattr(self, "lbl_summary_gq_badge", None)
        if badge_lbl is not None:
            badge = gait_quality_evidence_badge(
                summary, usable_gait_cycles=result.usable_gait_cycles
            )
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
        ac_lbl = getattr(self, "lbl_summary_ac_level", None)
        if ac_lbl is not None:
            ac_lbl.configure(
                text=format_analysis_confidence_level(level),
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
        else:
            cycles_text = "—"
            cycles_fg = MUTED

        for attr in ("lbl_gait_card_cycles_value",):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text=cycles_text, fg=cycles_fg)

        overview_usable = getattr(self, "lbl_overview_gait_cycles_usable", None)
        if overview_usable is not None:
            if cycles_text == "—":
                overview_text = "Usable: —"
            elif "detected" in cycles_text or "usable cycle" in cycles_text:
                overview_text = cycles_text
            else:
                overview_text = f"Usable: {cycles_text}"
            overview_usable.configure(text=overview_text, fg=cycles_fg)

        self._sync_demo_category_compare_strip()

        comp_text = (
            f"Completeness: {completeness_pct:.0f}%"
            if completeness_pct is not None
            else "Completeness: —"
        )
        for attr in (
            "lbl_gait_card_completeness_value",
            "lbl_overview_gait_cycles_completeness",
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text=comp_text, fg=MUTED)

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
        var = getattr(self, "pose_backend_var", None)
        if var is None:
            return config.POSE_BACKEND
        value = (var.get() or "mediapipe").strip().lower()
        if value not in ("mediapipe", "smpl", "auto"):
            return "mediapipe"
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
                fg="#2ecc71" if val.ready else THEME_MUTED,
            )
            return
        val = validate_smpl_assets()
        if val.ready:
            lbl.configure(text="Auto: SMPL available", fg="#2ecc71")
        else:
            lbl.configure(
                text=f"Auto: will fall back — {val.summary()[:80]}…",
                fg=THEME_MUTED,
            )

    def _refresh_real_to_sim_advanced_panel(self) -> None:
        """Update Advanced tab Real-to-Sim 4-stage summary."""
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
                (s4, "Virtual GRF: —"),
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
                    parts.append(f"cadence {cadence:.0f} spm")
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
                    parts.append(f"cadence {fp.cadence_steps_per_min:.0f} spm")
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
                    text=f"AMP reference: exported ✓ ({amp_path.name})",
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
                        f"Virtual GRF: {vgrf.estimation_method_label} "
                        f"({vgrf.confidence:.0%} confidence)"
                    ),
                    fg=TEXT,
                )
            else:
                s4.configure(
                    text="Virtual GRF: pose proxy unavailable (need pose sequence)",
                    fg=THEME_MUTED,
                )

        if summary is not None:
            spec_link = "See docs/SPEC_COMPLIANCE.md and REAL_TO_SIM_PIPELINE.md."
            if report:
                summary.configure(
                    text=(
                        f"Pipeline report loaded for “{report.get('run_name', run_name)}”. "
                        f"{spec_link}"
                    ),
                    fg=SUCCESS,
                )
            else:
                summary.configure(
                    text=(
                        "Matches research spec: video gait style → retarget → "
                        f"Isaac Lab AMP → contact-sync foot forces. {spec_link}"
                    ),
                    fg=INFO,
                )

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

    def _reset_stability_panel(self) -> None:
        self._biomech = None
        for attr, default in (
            ("lbl_summary_ms_value", "—"),
            ("lbl_summary_gq_value", "—"),
            ("lbl_summary_ac_level", "—"),
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text=default, fg=MUTED)
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
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text=default, fg=MUTED)
        for attr, default in (
            ("lbl_movement_stability", "—"),
            ("lbl_gait_quality", "—"),
            ("lbl_analysis_confidence", "—"),
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text=default, fg=MUTED)
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
            self.lbl_stab_score.configure(text="—", fg=MUTED)
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
                value_lbl.configure(text="—", fg=TEXT)
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
            ("lbl_gait_card_phase_value", "—"),
            ("lbl_overview_gait_cycles_usable", "Usable: —"),
            ("lbl_overview_gait_cycles_completeness", "Completeness: —"),
            ("lbl_gait_card_cycles_value", "—"),
            ("lbl_gait_card_completeness_value", "Completeness: —"),
            ("lbl_overview_contact_left", "—"),
            ("lbl_overview_contact_right", "—"),
            ("lbl_gait_card_contact_left", "—"),
            ("lbl_gait_card_contact_right", "—"),
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text=default, fg=MUTED)
        self._reset_gait_cycle_panel()

    def _reset_gait_cycle_panel(self) -> None:
        self._gait_cycle = None
        self._gait_features = None
        self._foot_contact = None
        self._estimated_vgrf = None
        self._biomech_analysis = None
        self._virtual_grf = None
        self._reset_physics_force_panel()
        self._update_contact_gait_chart()
        self._update_biomechanics_panel()
        self._update_biomechanics_chart()
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
            ("lbl_gait_card_cycles_value", "—"),
            ("lbl_gait_card_completeness_value", "Completeness: —"),
            ("lbl_overview_gait_cycles_usable", "Usable: —"),
            ("lbl_overview_gait_cycles_completeness", "Completeness: —"),
            ("lbl_overview_contact_left", "—"),
            ("lbl_overview_contact_right", "—"),
            ("lbl_gait_card_contact_left", "—"),
            ("lbl_gait_card_contact_right", "—"),
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.configure(text=default, fg=MUTED)
        for side in ("left", "right"):
            state_lbl = getattr(self, f"lbl_foot_{side}_state", None)
            if state_lbl is not None:
                state_lbl.configure(text="—", fg=MUTED)
        self._gait_contact_frame_index = None
        self._gait_phase_frame_index = None
        self._overview_video_frame_index = None

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
        status = getattr(self, "lbl_physics_force_status", None)
        method = getattr(self, "lbl_physics_force_method", None)
        note = getattr(self, "lbl_physics_force_note", None)
        if status is not None:
            status.configure(text="Status: Not configured", fg=MUTED)
        if method is not None:
            method.configure(text="Method: None", fg=MUTED)
        if note is not None:
            note.configure(
                text="Foot contact detection ≠ virtual force estimation",
                fg=MUTED,
            )

    def _refresh_physics_force_panel(self) -> None:
        """Update Physics Force Estimation sidebar — distinct from contact mask."""
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
                text=f"Method: {result.method_name} (estimated — not force-plate)",
                fg=TEXT,
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
                text=(
                    "Foot contact = timing only. "
                    "Virtual GRF = estimated proxy — not measured kinetics."
                ),
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
                return (state.left_contact_binary, state.right_contact_binary)
        if self._gait_cycle is None:
            return None
        state = self._gait_cycle.frame_at(frame_index)
        if state is None:
            return None
        return (state.left_contact, state.right_contact)

    def _update_contact_gait_chart(self, *, playhead_time_s: float | None = None) -> None:
        canvas = getattr(self, "canvas_contact_gait", None)
        fig = getattr(self, "fig_contact_gait", None)
        if canvas is None or fig is None:
            return
        if playhead_time_s is None and self.sequence and self.pose_indices:
            idx = getattr(self.skeleton_player, "current_index", None)
            if idx is not None:
                try:
                    fi = self.pose_indices[idx]
                    frame = next((f for f in self.sequence.frames if f.frame_index == fi), None)
                    if frame is not None:
                        playhead_time_s = frame.timestamp_s
                except (IndexError, TypeError):
                    pass
        from stablewalk.ui.viewers.contact_gait_viewer import draw_contact_gait_dashboard

        draw_contact_gait_dashboard(
            fig,
            self._foot_contact,
            self._estimated_vgrf,
            playhead_time_s=playhead_time_s,
        )
        canvas.draw_idle()

    def _biomech_overlays_for_frame(
        self, frame_index: int
    ) -> tuple[
        tuple[float, float, float] | None,
        list[tuple[float, float]] | None,
        tuple[float, float] | None,
        tuple[int, int] | None,
    ]:
        com = None
        polygon = None
        direction = None
        contact = self._foot_contact_for_frame(frame_index)
        ba = self._biomech_analysis
        if ba is None:
            return com, polygon, direction, contact
        if ba.center_of_mass:
            for f in ba.center_of_mass.per_frame:
                if f.frame_index == frame_index:
                    com = f.position
                    break
        if ba.base_of_support:
            for f in ba.base_of_support.per_frame:
                if f.frame_index == frame_index:
                    polygon = f.polygon_xy
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
        return com, polygon, direction, contact

    def _update_biomechanics_panel(self) -> None:
        from stablewalk.ui.theme import MUTED as THEME_MUTED, TEXT

        ba = self._biomech_analysis
        gq = getattr(self, "lbl_biomech_gait_quality", None)
        sym = getattr(self, "lbl_biomech_symmetry", None)
        sm = getattr(self, "lbl_biomech_stability_margin", None)
        cad = getattr(self, "lbl_biomech_cadence", None)
        spd = getattr(self, "lbl_biomech_walking_speed", None)
        vq = getattr(self, "lbl_biomech_video_quality", None)
        rom = getattr(self, "lbl_biomech_rom", None)
        interp = getattr(self, "lbl_biomech_interpretation", None)
        if ba is None:
            for lbl in (gq, sym, sm, cad, spd, vq):
                if lbl is not None:
                    lbl.configure(text="—", fg=THEME_MUTED)
            if rom is not None:
                rom.configure(text="ROM: —", fg=THEME_MUTED)
            if interp is not None:
                interp.configure(text="")
            return
        if gq is not None and ba.gait_quality:
            gq.configure(text=f"{ba.gait_quality.score:.0f}/100 (est.)", fg=TEXT)
        if sym is not None and ba.symmetry and ba.symmetry.overall_symmetry_pct:
            v = ba.symmetry.overall_symmetry_pct.value
            sym.configure(text=f"{v:.0f}%" if v is not None else "—", fg=TEXT)
        if sm is not None and ba.stability_margin:
            sm.configure(text=f"{ba.stability_margin.stable_pct:.0f}% stable frames", fg=TEXT)
        if cad is not None and ba.gait_metrics and ba.gait_metrics.cadence:
            c = ba.gait_metrics.cadence.value
            cad.configure(text=f"{c:.0f} spm" if c is not None else "—", fg=TEXT)
        if spd is not None and ba.gait_metrics and ba.gait_metrics.walking_speed:
            s = ba.gait_metrics.walking_speed.value
            spd.configure(text=f"{s:.2f} m/s" if s is not None else "—", fg=TEXT)
        if vq is not None and ba.video_quality:
            vq.configure(text=f"{ba.video_quality.overall_quality_score:.0f}/100", fg=TEXT)
        if rom is not None and ba.joint_rom and ba.joint_rom.joints:
            knee = [j for j in ba.joint_rom.joints if j.joint == "knee"]
            parts = [
                f"{j.side[0].upper()} knee {j.rom_deg:.0f}°"
                for j in knee
                if j.rom_deg is not None
            ]
            rom.configure(text="ROM: " + ("  ".join(parts) if parts else "—"), fg=TEXT)
        if interp is not None:
            text = ba.gait_quality.explanation if ba.gait_quality else ""
            if ba.abnormalities:
                text += " Flags: " + "; ".join(ba.abnormalities[:2])
            interp.configure(text=text[:240], fg=THEME_MUTED)

    def _update_biomechanics_chart(self, *, playhead_time_s: float | None = None) -> None:
        canvas = getattr(self, "canvas_biomech", None)
        fig = getattr(self, "fig_biomech", None)
        if canvas is None or fig is None:
            return
        from stablewalk.ui.viewers.biomechanics_charts import draw_biomechanics_dashboard

        draw_biomechanics_dashboard(fig, self._biomech_analysis, playhead_time_s=playhead_time_s)
        canvas.draw_idle()

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

    def _update_gait_cycle_panel(
        self,
        result: GaitCycleAnalysisResult | None,
        *,
        frame_index: int | None = None,
    ) -> None:
        if result is None or not result.per_frame:
            self._reset_gait_cycle_panel()
            return

        idx = frame_index
        if idx is None and self.skeleton_player is not None:
            idx = self.skeleton_player.state.frame_index
        state = result.frame_at(idx) if idx is not None else None
        if state is None and result.per_frame:
            state = result.per_frame[-1]

        from stablewalk.analysis.gait_phase_classification import (
            classify_gait_phase_from_contacts,
            log_phase_consistency_warning,
        )

        m = result.metrics
        if state is not None:
            derived_phase = classify_gait_phase_from_contacts(
                state.left_contact,
                state.right_contact,
                contact_confidence=m.contact_confidence,
                confidence_tier=m.confidence_tier,
                left_foot_clearance_m=state.left.foot_clearance_m,
                right_foot_clearance_m=state.right.foot_clearance_m,
            )
            dashboard_phase = self._format_dashboard_gait_phase(derived_phase)
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
        if state is not None and left_lbl is not None and right_lbl is not None:
            from stablewalk.ui.theme import ACCENT, MUTED as THEME_MUTED

            def _contact_line(side: str, contact: int) -> tuple[str, str]:
                if contact:
                    return f"{side}: YES", ACCENT
                return f"{side}: NO", THEME_MUTED

            l_text, l_fg = _contact_line("Left contact", state.left_contact)
            r_text, r_fg = _contact_line("Right contact", state.right_contact)
            left_lbl.configure(text=l_text, fg=l_fg)
            right_lbl.configure(text=r_text, fg=r_fg)

            left_contact = getattr(self, "lbl_gait_card_contact_left", None)
            right_contact = getattr(self, "lbl_gait_card_contact_right", None)
            overview_left = getattr(self, "lbl_overview_contact_left", None)
            overview_right = getattr(self, "lbl_overview_contact_right", None)
            left_text = self._format_contact_state(state.left_contact)
            right_text = self._format_contact_state(state.right_contact)
            for lbl, text in (
                (left_contact, left_text),
                (right_contact, right_text),
                (overview_left, left_text),
                (overview_right, right_text),
            ):
                if lbl is not None:
                    lbl.configure(text=text, fg=TEXT)

            # Foot clearance state on Overview uses the same contact mask.
            for side, contact_val in (
                ("left", state.left_contact),
                ("right", state.right_contact),
            ):
                label = self._format_contact_state(contact_val)
                state_fg = INFO if label == "SWING" else TEXT
                state_lbl = getattr(self, f"lbl_foot_{side}_state", None)
                if state_lbl is not None:
                    state_lbl.configure(text=label, fg=state_fg)
            self._gait_contact_frame_index = state.frame_index
            self._gait_phase_frame_index = state.frame_index

        tier = m.confidence_tier
        values = [
            f"{m.cadence_steps_per_min:.0f}" if m.cadence_steps_per_min is not None else "—",
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
                    text=f"Contact confidence: {m.contact_confidence:.0%} (LOW)",
                    fg=MUTED,
                )
            else:
                conf.configure(
                    text=f"Contact confidence: {m.contact_confidence:.0%}",
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
        if usable is None and detected:
            usable = detected
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

        if presentation:
            status_text = "Demo mode"
            dot_color = INFO
        elif sdk and model_valid:
            status_text = "Ready"
            dot_color = SUCCESS
        elif sdk:
            status_text = "Partial setup"
            dot_color = WARNING
        elif export_complete or has_session:
            status_text = "Export only"
            dot_color = INFO
        else:
            status_text = "Unavailable"
            dot_color = DANGER

        ready_line = compact_opensim_ready_line(sdk=sdk, presentation=presentation)
        export_line = compact_export_summary_line(
            export_complete=export_complete,
            has_session=has_session,
            presentation=presentation,
        )
        tip = f"{ready_line}\n{export_line.replace('Export: ', 'Export ')}"

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

    def _format_gait_summary_details(self, result: "StabilityResult") -> str:
        """Structured gait summary for the Details dialog (no long prose blocks)."""
        summary = result.gait_summary or build_gait_analysis_summary(result)
        validity = result.validity or assess_stability_result_validity(result)
        lines = [
            f"{'Movement Stability':<22}"
            f"{summary.movement_stability.score:.0f}"
            if summary.movement_stability.score is not None
            else f"{'Movement Stability':<22}—",
            f"{'Gait Quality':<22}"
            f"{summary.gait_quality.score:.0f}"
            if summary.gait_quality.score is not None
            else f"{'Gait Quality':<22}—",
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
                        f"Demo category ({demo.button_label}):",
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
        from stablewalk.analysis.stability_scoring import build_gait_analysis_summary
        from stablewalk.analysis.validity import assess_stability_result_validity
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
            ),
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
            f"{'Contact Pattern':<22}"
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
            ("contact_pattern", "Contact Pattern"),
        ):
            metric = metric_by_key.get(key)
            if metric is None or metric.score is None:
                lines.append(f"  {title:<20}—")
                continue
            card = interpret_domain_metric(metric, display_name=title)
            lines.append(f"  {title:<20}{metric.score:.0f}  ({card.confidence})")
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
        """Detect SDK + optional default model once, then render the status block."""
        self._opensim_last_dir: Path | None = None
        self._opensim_model_path: Path | None = None
        self._opensim_model_name: str | None = None
        self._opensim_model_valid: bool = False
        self._opensim_ik_state: str | None = None   # StableWalk IK: Running|Completed|Failed
        self._opensim_ik_message: str = ""
        self._opensim_demo_ik_state: str | None = None  # Demo IK state
        self._opensim_demo_ik_message: str = ""
        self._marker_mapping_comparison = None
        self._opensim_detail_text = ""
        self._opensim_ui_log: list[str] = []
        self._opensim_model_choices: dict[str, Path] = {}
        try:
            from stablewalk.opensim_sdk import (
                check_opensim_sdk,
                log_opensim_startup_status,
            )

            log_opensim_startup_status(logger, model_path=None)
            status = check_opensim_sdk(refresh=True)
            self._opensim_sdk_available = bool(status.available)
            self._opensim_sdk_version = status.version
            # Do not auto-load any .osim model — user selects manually or uses Demo IK.
        except Exception as exc:
            logger.warning("OpenSim panel init failed: %s", exc)
            self._opensim_sdk_available = False
            self._opensim_sdk_version = None
            self._opensim_model_valid = False
        self._refresh_opensim_model_list()
        self._refresh_opensim_status()
        self._attach_opensim_tooltips()

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

    def _load_opensim_model_path(self, path: Path):
        """
        Validate an OpenSim model and update GUI state only on success.

        Returns :class:`~stablewalk.opensim_sdk.ModelInfo` on success, else ``None``.
        Failed validation leaves any previously loaded model unchanged.
        """
        from stablewalk.opensim_models import _is_autoload_blocked
        from stablewalk.opensim_sdk import validate_opensim_model
        from stablewalk.ui.tk.sidebar_display import opensim_model_ik_error_message

        if _is_autoload_blocked(path):
            msg = opensim_model_ik_error_message(blocked_template=True)
            logger.error("Model load rejected (blocked template): %s", path)
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
        """Mirror export diagnostics to logger and stdout (visible in terminal)."""
        text = message % args if args else message
        logger.info("%s", text)
        print(text, flush=True)
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
        }

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
        return f"{label}: ready", ACCENT

    def _opensim_overview_export_summary(self, files: dict[str, bool]) -> str:
        """One-line TRC / MOT / JSON summary for Session Overview."""
        if not self._opensim_has_session() or not self._opensim_has_pose_data():
            return "TRC: missing · MOT: missing · JSON: missing"
        parts: list[str] = []
        for key, label in (("TRC", "TRC"), ("MOT", "MOT"), ("JSON", "JSON")):
            if files.get(key):
                parts.append(f"{label}: exported")
            else:
                parts.append(f"{label}: ready")
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

    def _refresh_opensim_status(self) -> None:
        """Single source of truth for every OpenSim status field + button state."""
        if not hasattr(self, "lbl_opensim_sdk"):
            return

        from stablewalk.opensim_marker_mapping import (
            DEMO_IK_MODEL,
            MEDIAPIPE_LIMITATION_EXPLANATION,
            compare_stablewalk_trc_to_opensim,
            demo_ik_setup_available,
            mapping_status_label,
            reliability_label,
            stablewalk_ik_status_label,
        )
        from stablewalk.opensim_sdk import NO_MODEL_LOADED_MESSAGE, check_opensim_sdk

        sdk_status = check_opensim_sdk(refresh=True)
        sdk = bool(sdk_status.available)
        self._opensim_sdk_available = sdk
        self._opensim_sdk_version = sdk_status.version

        model_valid = bool(getattr(self, "_opensim_model_valid", False))
        has_session = self._opensim_has_session()
        has_pose = self._opensim_has_pose_data()
        files = self._opensim_export_files_on_disk()
        trc_exported = bool(files.get("TRC"))
        trc_path = self._opensim_trc_path()

        # Marker mapping validation (StableWalk TRC vs OpenSim demo IK setup).
        if trc_path and trc_path.is_file():
            self._marker_mapping_comparison = compare_stablewalk_trc_to_opensim(trc_path)
        else:
            self._marker_mapping_comparison = compare_stablewalk_trc_to_opensim(None)

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
            demo_ik_line,
            export_file_line,
            last_export_line,
            last_ik_line,
            mapping_compact_line,
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
        _set(
            "lbl_opensim_marker_mapping",
            mapping_compact_line(label=map_label, coverage_percent=cov),
            map_color,
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
        elif sw_state == "Completed" or ik_on_disk:
            last_ik_color = "ok"
        else:
            last_ik_color = "muted"
        _set(
            "lbl_opensim_last_ik",
            last_ik_line(run_state=sw_state, ik_output_on_disk=ik_on_disk),
            last_ik_color,
        )

        # Extended notes (log / details only — not in sidebar)
        if sw_state == "Completed" and ik_mot_path:
            detail_lines.append(f"IK output: {self._format_data_path(ik_mot_path)}")
        elif sw_state == "Failed":
            detail_lines.append(getattr(self, "_opensim_ik_message", ""))
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
                        and DEMO_IK_MODEL.is_file()
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
        if hasattr(self, "btn_opensim_export"):
            self.btn_opensim_export.configure(
                state="disabled" if self._processing else "normal"
            )
        export_data_btn = getattr(self, "btn_opensim_export_data", None)
        if export_data_btn is not None:
            export_data_btn.configure(
                state=tk.DISABLED if self._processing else tk.NORMAL
            )
        self._update_opensim_action_tooltips(presentation=False)
        self._update_export_section_status()
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

        status = check_opensim_sdk(refresh=True)
        if not status.available:
            messagebox.showinfo("Run OpenSim Demo IK", status.message)
            return

        self._opensim_demo_ik_state = "Running"
        self._opensim_demo_ik_message = ""
        self._refresh_opensim_status()
        self.root.update_idletasks()

        result = run_opensim_demo_ik()
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

    def _run_stablewalk_ik_experimental(self) -> None:
        """Run experimental OpenSim IK on the current session mapped TRC."""
        from stablewalk.opensim_marker_mapping import (
            DEMO_IK_MODEL,
            MEDIAPIPE_LIMITATION_EXPLANATION,
            compare_stablewalk_trc_to_opensim,
        )
        from stablewalk.opensim_sdk import check_opensim_sdk, run_stablewalk_ik_experimental

        self._log_export("Run StableWalk IK Experimental clicked")

        status = check_opensim_sdk(refresh=True)
        if not status.available:
            self._log_export("OpenSim SDK not installed")
            messagebox.showerror("Run StableWalk IK Experimental", status.message)
            return

        if self._presentation_mode or not self._current_run_name():
            self._log_export("IK aborted: no video/session loaded")
            messagebox.showerror(
                "Run StableWalk IK Experimental",
                _OPENSIM_NO_SESSION_MSG,
            )
            return

        run_name = self._current_run_name()
        assert run_name is not None
        out_dir = config.OPENSIM_DIR / run_name
        raw_trc = out_dir / f"{run_name}.trc"
        mapped_trc = out_dir / f"{run_name}_mapped_for_opensim.trc"
        setup_xml = out_dir / "stablewalk_setup_ik.xml"
        ik_mot = out_dir / f"{run_name}_ik.mot"
        model_path = DEMO_IK_MODEL

        self._log_export("Current run/session: %s", run_name)
        self._log_export("Using mapped TRC: %s", mapped_trc)
        self._log_export("Using OpenSim model: %s", model_path)
        self._log_export("IK setup XML: %s", setup_xml)
        self._log_export("IK output path: %s", ik_mot)

        if not model_path.is_file():
            self._log_export("OpenSim model missing: %s", model_path)
            messagebox.showerror(
                "Run StableWalk IK Experimental",
                f"OpenSim model not found:\n{model_path}",
            )
            return

        if not mapped_trc.is_file():
            self._log_export("Mapped TRC missing: %s", mapped_trc)
            messagebox.showerror(
                "Run StableWalk IK Experimental",
                "Please click Export OpenSim Files first.",
            )
            self._refresh_opensim_status()
            return

        if not raw_trc.is_file():
            messagebox.showerror(
                "Run StableWalk IK Experimental",
                "Please click Export OpenSim Files first.",
            )
            return

        mapping = compare_stablewalk_trc_to_opensim(raw_trc)
        self._marker_mapping_comparison = mapping
        self._log_export("Marker mapping status: %s", mapping.mapping_status)
        self._log_export("Coverage: %s%%", mapping.coverage_percent)
        self._log_export("Warning: this is experimental and not clinical-grade")

        if mapping.mapping_status not in ("improved", "partial", "experimental"):
            messagebox.showerror(
                "Run StableWalk IK Experimental",
                f"Marker mapping not ready (status: {mapping.mapping_status}).\n\n"
                f"{mapping.message}",
            )
            self._refresh_opensim_status()
            return

        if not mapping.ik_experimental_ready:
            messagebox.showerror(
                "Run StableWalk IK Experimental",
                f"{mapping.message}\n\nMapped matches: "
                f"{len(mapping.mapped_matching_markers)}",
            )
            self._refresh_opensim_status()
            return

        self._opensim_ik_state = "Running"
        self._opensim_ik_message = ""
        self._refresh_opensim_status()
        self.root.update_idletasks()

        result = run_stablewalk_ik_experimental(
            mapped_trc,
            model_path=model_path,
            run_name=run_name,
        )
        output_path = Path(result.output_motion_path) if result.output_motion_path else None
        rel_mot = self._format_data_path(ik_mot)

        if result.ran and output_path and output_path.is_file():
            self._opensim_ik_state = "Completed"
            self._opensim_ik_message = self._format_data_path(output_path)
            self._opensim_last_dir = out_dir
            self._opensim_status_override = (
                f"StableWalk IK: Completed\nIK output: {rel_mot}\n"
                f"OpenSim output folder: {self._format_data_path(out_dir)}"
            )
            messagebox.showinfo(
                "StableWalk IK complete (experimental)",
                f"{result.message}\n\n"
                f"IK output:\n  {rel_mot}\n\n"
                f"Setup:\n  {self._format_data_path(Path(result.setup_path)) if result.setup_path else setup_xml}\n\n"
                f"{MEDIAPIPE_LIMITATION_EXPLANATION}",
            )
            self._rebuild_knee_angle_series()
            self._refresh_knee_source_selector()
            self._update_chart()
        else:
            self._opensim_ik_state = "Failed"
            self._opensim_ik_message = result.message
            self._log_export("StableWalk IK failed: %s", result.message)
            messagebox.showerror("StableWalk IK failed", result.message)
        self._refresh_opensim_status()

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

    def _export_opensim_session(self) -> None:
        """Export the currently analyzed session to OpenSim files (sidebar button)."""
        self._log_export("Export OpenSim Files clicked")
        self._opensim_export_completed = False
        self._opensim_status_override = None

        if self._presentation_mode:
            self._log_export("Export aborted: presentation demo active (no real session)")
            messagebox.showerror("OpenSim Export", _OPENSIM_NO_SESSION_MSG)
            self._refresh_opensim_status()
            return

        run_name = self._current_run_name()
        if not run_name:
            self._log_export("Export aborted: no current run/session name")
            messagebox.showerror("OpenSim Export", _OPENSIM_NO_SESSION_MSG)
            self._refresh_opensim_status()
            return

        self._log_export("Current run/session: %s", run_name)

        expected_poses = config.POSES_DIR / f"{run_name}_poses.json"
        poses_path = self._resolve_poses_json_path()
        poses_exists = poses_path is not None and poses_path.is_file()
        self._log_export("Pose JSON: %s", poses_path.resolve() if poses_path else "(none)")
        self._log_export("Pose JSON exists: %s", poses_exists)
        if not poses_exists:
            self._log_export("Expected pose JSON path: %s", expected_poses)
            messagebox.showerror("OpenSim Export", _OPENSIM_POSE_JSON_MISSING_MSG)
            self._refresh_opensim_status()
            return

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
            messagebox.showerror("OpenSim Export", f"Export failed: {exc}")
            self._refresh_opensim_status()
            return

        valid_frames = sum(1 for f in sequence.frames if getattr(f, "detected", False))
        self._log_export("Valid pose frames: %d", valid_frames)
        if valid_frames == 0:
            messagebox.showerror("OpenSim Export", _OPENSIM_NO_VALID_FRAMES_MSG)
            self._refresh_opensim_status()
            return

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
            self._log_export("Writing TRC...")
            self._log_export("Writing MOT...")
            self._log_export("Writing JSON...")
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
            messagebox.showerror("OpenSim Export", f"Export failed: {exc}")
            self._refresh_opensim_status()
            return

        missing = [
            p for p in (trc_target, mot_target, json_target) if not p.is_file()
        ]
        if missing:
            missing_rel = [self._format_data_path(p) for p in missing]
            self._log_export("Export incomplete — missing: %s", ", ".join(missing_rel))
            messagebox.showerror(
                "OpenSim Export",
                "Export did not create all required files.\n\nMissing:\n"
                + "\n".join(f"  • {m}" for m in missing_rel),
            )
            self._refresh_opensim_status()
            return

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
        self._opensim_ik_state = None
        self._opensim_status_override = (
            f"Export completed successfully.\nOutput folder: {rel_out}"
        )
        self.status.configure(text=f"OpenSim export → {rel_out}")
        self._refresh_opensim_status()
        self.root.update_idletasks()

        mapped_note = (
            f"\n  {mapped_target.name}   (mapped markers)"
            if mapped_target.is_file()
            else ""
        )
        messagebox.showinfo(
            "OpenSim export complete",
            "Export completed successfully.\n\n"
            f"OpenSim output folder:\n  {rel_out}\n\n"
            f"  {run_name}.trc\n"
            f"  {run_name}{motion_ext}\n"
            f"  {run_name}_opensim.json{mapped_note}\n\n"
            "Use “Open Output Folder” to view files in Explorer.",
        )

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
            f"Edit men's walking URLs in:\n{path}\n\n"
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

    def _show_shortcuts(self) -> None:
        messagebox.showinfo(
            "Shortcuts",
            "Space — Play / Pause\n"
            "← / → — Previous / Next pose frame\n"
            "Home / End — First / Last pose\n"
            "Ctrl+O — Open pose JSON",
        )

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About StableWalk",
            "StableWalk — Gait Analysis Dashboard\n\n"
            "MediaPipe detects the human skeleton from each video frame "
            "(pose estimation).\n\n"
            "OpenSim is the biomechanical layer: marker/motion export (.trc, "
            ".mot, JSON) and optional musculoskeletal simulation when the "
            "OpenSim SDK is installed.\n\n"
            "Stability metrics are computed from MediaPipe-derived joint "
            "kinematics unless you run OpenSim inverse kinematics.\n\n"
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
        self._update_skeleton_view_box()
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

    def _resolve_highlight_joints(self) -> set[str] | None:
        if not self.highlight_dof.get():
            return None
        from stablewalk.ui.dof_selection import joints_for_item

        charted = self._charted_dof_item_id()
        if charted:
            joints = joints_for_item(charted)
            return joints if joints else None
        joints = self.selection.highlight_joints()
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

    def _analysis_motion_recording(self) -> GaitMotionRecording | None:
        """Recording used for the selected-point 3D trajectory panel."""
        if self.gait_motion is not None:
            return self.gait_motion
        if self.skeleton_player is not None:
            return getattr(self.skeleton_player, "recording", None)
        return None

    def _clear_bilateral_ground_clearance_ui(self) -> None:
        """Reset foot clearance compact strip and detail panel."""
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
            for suffix in ("current", "state", "unavailable"):
                lbl = getattr(self, f"lbl_{prefix}_{suffix}", None)
                if lbl is None:
                    continue
                if suffix == "unavailable":
                    lbl.configure(text="", fg=MUTED)
                elif suffix == "current":
                    lbl.configure(text="Unavailable", fg=MUTED)
                else:
                    lbl.configure(text="—", fg=MUTED)
        conf = getattr(self, "lbl_foot_clearance_confidence", None)
        if conf is not None:
            conf.configure(text="Measurement Confidence: —", fg=MUTED)
        self._foot_clearance_dashboard = None
        self._foot_clearance_frame_index = None
        self._ground_clearance_prev_phase = (None, None)

    def _apply_foot_clearance_dashboard(self, panel) -> None:
        """Push foot clearance model into the primary Overview panel."""
        from stablewalk.ui.foot_clearance_display import (
            FootClearanceDashboardPanel,
            overview_distance_text,
            overview_unavailable_reason,
        )

        if not isinstance(panel, FootClearanceDashboardPanel):
            return
        self._foot_clearance_dashboard = panel

        conf_lbl = getattr(self, "lbl_foot_clearance_confidence", None)
        if conf_lbl is not None:
            conf_fg = {
                "HIGH": SUCCESS,
                "MODERATE": INFO,
                "LOW": WARNING,
            }.get(panel.confidence, MUTED)
            conf_lbl.configure(text=panel.confidence_label, fg=conf_fg)

        _distance_font = ("Segoe UI Semibold", 15)

        for foot, prefix in ((panel.left, "foot_left"), (panel.right, "foot_right")):
            dist_lbl = getattr(self, f"lbl_{prefix}_current", None)
            reason_lbl = getattr(self, f"lbl_{prefix}_unavailable", None)
            available = foot.displayed_clearance_cm is not None
            if dist_lbl is not None:
                if available:
                    dist_lbl.configure(
                        text=overview_distance_text(foot),
                        fg=ORANGE,
                        font=_distance_font,
                    )
                else:
                    dist_lbl.configure(
                        text="Unavailable",
                        fg=MUTED,
                        font=_distance_font,
                    )
            if reason_lbl is not None:
                if available:
                    reason_lbl.configure(text="", fg=MUTED)
                else:
                    reason_lbl.configure(
                        text=overview_unavailable_reason(foot.unavailable_reason),
                        fg=MUTED,
                    )

        from stablewalk.ui.tk.gui_layout_debug import log_foot_clearance_debug_if_enabled

        log_foot_clearance_debug_if_enabled(panel)

    def _show_foot_clearance_details(self) -> None:
        """Open foot clearance methodology and per-foot measurements."""
        panel = getattr(self, "_foot_clearance_dashboard", None)
        if panel is None:
            messagebox.showinfo(
                "Foot Clearance",
                "No foot clearance data yet.\n\nLoad a video and run analysis first.",
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
        dlg.title("Foot Clearance — Details")
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
        """Show foot clearance widgets whenever motion data is loaded."""
        detail = getattr(self, "_foot_clearance_detail_host", None)
        recording = self._analysis_motion_recording()
        if recording is not None and recording.frame_count > 0:
            if detail is not None:
                try:
                    detail.grid()
                except tk.TclError:
                    pass
        else:
            if detail is not None:
                try:
                    detail.grid_remove()
                except tk.TclError:
                    pass
            self._clear_bilateral_ground_clearance_ui()

    def _refresh_bilateral_ground_clearance(self, snapshot=None) -> None:
        """Update left/right foot clearance for the current playback frame."""
        from stablewalk.ui.foot_clearance_display import foot_clearance_dashboard_for_panel

        self._update_ground_clearance_visibility()
        recording = self._analysis_motion_recording()
        player = self.skeleton_player
        if snapshot is None and player is not None:
            snapshot = player.current_snapshot()
        end_f = float(player.state.frame_float) if player is not None else 0.0

        if recording is None or snapshot is None:
            self._clear_bilateral_ground_clearance_ui()
            return

        prev_l, prev_r = self._ground_clearance_prev_phase
        panel = foot_clearance_dashboard_for_panel(
            snapshot,
            recording,
            end_f,
            prev_left_phase=prev_l,
            prev_right_phase=prev_r,
        )
        if panel is None:
            self._clear_bilateral_ground_clearance_ui()
            return

        self._ground_clearance_prev_phase = (panel.left_phase, panel.right_phase)
        self._foot_clearance_frame_index = snapshot.frame_index

        self._apply_foot_clearance_dashboard(panel)
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

    def _fit_dof_traj_canvas(self) -> None:
        """Resize the 3D trajectory figure to its widget so nothing is clipped."""
        if not hasattr(self, "canvas_dof_traj"):
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
        if not hasattr(self, "canvas_dof_traj"):
            return
        from stablewalk.ui.tk.dashboard_layout import _ensure_trajectory_canvas_gridded

        _ensure_trajectory_canvas_gridded(self.canvas_dof_traj)

    def _render_dof_traj_canvas(self, *, force: bool = False) -> None:
        """Fit the trajectory canvas to its host and paint the current figure."""
        if not hasattr(self, "canvas_dof_traj"):
            return
        self._fit_dof_traj_canvas()
        self._lift_trajectory_canvas()
        playing = bool(
            self.skeleton_player and self.skeleton_player.state.playing
        )
        from stablewalk.ui.tk.dashboard_notebook import is_trajectory_graph_visible

        needs_immediate_paint = (
            force
            or is_trajectory_graph_visible(self)
            or not playing
        )
        if needs_immediate_paint:
            self.canvas_dof_traj.draw()
        else:
            self.canvas_dof_traj.draw_idle()

    def _schedule_dof_traj_reflow(self) -> None:
        """Reflow the trajectory canvas after Tk finishes mapping the graph area."""
        if not hasattr(self, "canvas_dof_traj"):
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

        self.root.after_idle(_reflow)
        self.root.after(50, _reflow)
        self.root.after(150, _reflow)
        self.root.after(350, _reflow)
        self.root.after(600, _reflow)

    def _fit_skeleton_canvas(self) -> None:
        """Resize the skeleton figure and refit the body viewport after layout."""
        if not hasattr(self, "canvas_3d"):
            return
        from stablewalk.ui.tk.dashboard_layout import _SKELETON_CANVAS_PAD

        pad_l, pad_t, pad_r, pad_b = _SKELETON_CANVAS_PAD
        widget = self.canvas_3d.get_tk_widget()
        widget.update_idletasks()
        width = widget.winfo_width()
        height = widget.winfo_height()
        if width < 80 or height < 80:
            host = getattr(self, "skel_canvas_host", None)
            if host is not None:
                host.update_idletasks()
                width = max(width, host.winfo_width())
                height = max(height, host.winfo_height())
        width = max(80, width - pad_l - pad_r)
        height = max(80, height - pad_t - pad_b)
        if width < 80 or height < 80:
            return

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
        from stablewalk.ui.viewers.gait_skeleton_renderer import relayout_skeleton_viewport

        relayout_skeleton_viewport(self.ax_3d)

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
        if not hasattr(self, "ax_dof_traj"):
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
        if not hasattr(self, "canvas_dof_traj"):
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
        recording = self._analysis_motion_recording()
        can_export = bool(
            self.selection.selected
            and recording is not None
            and recording.frame_count > 0
            and not self._presentation_mode
        )
        has_sequence = bool(self.sequence) and not self._presentation_mode
        has_biomech = self._biomech is not None and not self._presentation_mode
        for attr in (
            "btn_export_analysis_report",
            "btn_export_gait_metrics",
            "btn_export_analysis",
        ):
            btn = getattr(self, attr, None)
            if btn is None:
                continue
            enabled = (
                has_sequence or has_biomech
                if attr != "btn_export_analysis"
                else can_export
            )
            btn.configure(state=tk.NORMAL if enabled else tk.DISABLED)
        motion_btn = getattr(self, "btn_export_motion_reference", None)
        if motion_btn is not None:
            motion_btn.configure(
                state=tk.NORMAL
                if (self.sequence or self._poses_path) and not self._presentation_mode
                else tk.DISABLED
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
        if hasattr(self, "canvas_dof_traj"):
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
        """Apply the selected update interval to all real-time analysis panels."""
        if choice == "0.25 s":
            self._data_refresh_s = 0.25
        else:
            self._data_refresh_s = 0.5
            if hasattr(self, "refresh_var"):
                self.refresh_var.set("0.5 s")
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
        if force:
            return True
        if not self.skeleton_player or not self.skeleton_player.state.playing:
            return True
        return (time.monotonic() - self._last_realtime_refresh) >= self._data_refresh_s

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
        self._mark_realtime_refresh()
        self._dof_step_previews = self._compute_dof_step_previews()
        self._refresh_dof_step_panel()
        self._refresh_angle_analysis_if_present()
        if snapshot is None and self.skeleton_player:
            snapshot = self.skeleton_player.current_snapshot()
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
        label = self._refresh_interval_label()
        return (
            f"Updates every {label} · "
            "biomechanical flexion angles (OpenSim-style, degrees)"
        )

    def _step_preview_hint_text(self) -> str:
        label = self._refresh_interval_label()
        return f"Updates every {label} · current vs next position and difference"

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
                    name_font = ("Segoe UI Semibold", 9)
                elif selected:
                    bg, fg = PANEL_HOVER, ACCENT
                    name_font = ("Segoe UI Semibold", 9)
                else:
                    bg, fg = ELEVATED, TEXT
                    name_font = ("Segoe UI", 9)
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
                text="None — add from dropdown or click skeleton",
                bg=PANEL,
                fg=MUTED,
                font=FONT_UI_XS,
                anchor="w",
            ).pack(anchor=tk.W)
            return
        for item_id in selected_ids:
            chip = tk.Frame(frame, bg=ELEVATED, highlightthickness=0)
            chip.pack(side=tk.LEFT, padx=(0, 4), pady=2)
            label = GUI_DOF_LABELS.get(item_id, item_id)
            fg = ORANGE if item_id == active else ACCENT
            name_lbl = tk.Label(
                chip,
                text=label,
                bg=ELEVATED,
                fg=fg,
                font=("Segoe UI Semibold", 8),
                cursor="hand2",
                padx=6,
                pady=2,
            )
            name_lbl.pack(side=tk.LEFT)
            name_lbl.bind("<Button-1>", lambda _e, i=item_id: self._on_dof_item_focus(i))
            remove_btn = tk.Label(
                chip,
                text="\u00d7",
                bg=ELEVATED,
                fg=MUTED,
                font=("Segoe UI", 9),
                cursor="hand2",
                padx=4,
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
        self._toggle_collected_data_table()

    def _toggle_collected_data_table(self) -> None:
        btn = getattr(self, "btn_collected_data", None) or getattr(
            self, "btn_view_table_data", None
        )
        existing = getattr(self, "_collected_data_dialog", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.destroy()
            except tk.TclError:
                pass
            self._collected_data_dialog = None
            if btn is not None:
                self._update_table_summary_label()
            return

        from stablewalk.ui.dof_position_table import (
            DOF_TABLE_COLUMNS,
            DOF_TABLE_HEADINGS,
        )
        from stablewalk.ui.tk.dashboard_layout import (
            _TABLE_COL_WIDTHS_COMPACT,
            _make_data_tree,
        )

        dlg = tk.Toplevel(self.root)
        dlg.title("Collected Data")
        dlg.geometry("760x440")
        dlg.minsize(520, 300)
        dlg.transient(self.root)
        self._collected_data_dialog = dlg

        frame = ttk.Frame(dlg, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        tree_host = ttk.Frame(frame)
        tree_host.pack(fill=tk.BOTH, expand=True)

        popup_tree = _make_data_tree(
            tree_host,
            DOF_TABLE_COLUMNS,
            DOF_TABLE_HEADINGS,
            col_widths=_TABLE_COL_WIDTHS_COMPACT,
            height=16,
            text_cols=frozenset({"dof", "contact_status"}),
            style="Compact.Treeview",
        )
        src = getattr(self, "dof_pos_tree", None)
        if src is not None:
            for iid in src.get_children():
                popup_tree.insert("", tk.END, values=src.item(iid)["values"])

        def _close() -> None:
            self._collected_data_dialog = None
            self._update_table_summary_label()
            try:
                dlg.destroy()
            except tk.TclError:
                pass

        dlg.protocol("WM_DELETE_WINDOW", _close)
        footer = ttk.Frame(frame)
        footer.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(footer, text="Close", command=_close).pack(side=tk.RIGHT)

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
        """Keep the 'Add a point' dropdown showing only not-yet-selected joints."""
        if not hasattr(self, "add_point_combo"):
            return
        from stablewalk.ui.tk.dashboard_layout import ADD_POINT_PLACEHOLDER

        available = [
            GUI_DOF_LABELS[i] for i in GUI_DOF_ITEM_IDS if i not in self.selection.selected
        ]
        self.add_point_combo.configure(values=available)
        self.add_point_var.set(ADD_POINT_PLACEHOLDER)
        self.add_point_combo.configure(state="readonly" if available else "disabled")

    def _add_point_from_combo(self, _event: object = None) -> None:
        """Add the joint chosen in the dropdown to the current selection."""
        if not hasattr(self, "add_point_combo"):
            return
        from stablewalk.ui.tk.dashboard_layout import ADD_POINT_PLACEHOLDER

        label = self.add_point_var.get()
        if not label or label == ADD_POINT_PLACEHOLDER:
            return
        item_id = {GUI_DOF_LABELS[i]: i for i in GUI_DOF_ITEM_IDS}.get(label)
        if not item_id or item_id in self.selection.selected:
            self._refresh_add_point_combo()
            return
        self.selection.activate_item(item_id)
        self._notify_dof_selection_changed()

    def _notify_dof_selection_changed(self) -> None:
        """Refresh every panel that depends on the current DOF selection."""
        self.selection.ensure_last_selected()
        self._sync_active_joint_from_charted()
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
        self._sync_dof_checkboxes()
        self._update_dof_selection_chrome()
        self._sync_dof_analysis_panel_state()
        self._refresh_realtime_analysis(force_draw=True)
        self._traj_draw_cache_key = None
        self._refresh_selected_dof_trajectory_3d(force_draw=True)
        self._render_dof_traj_canvas(force=True)
        self._update_interactive_skeleton(force_draw=True)
        self._update_dof_table_controls_state()
        self._configure_dof_table_columns()
        self._sync_demo_save_button()
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
            font=("Segoe UI Semibold", 10),
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
            ("Angle", metrics_detail.angle, ACCENT),
            (next_angle_label, metrics_detail.next_angle, TEXT_SECONDARY),
            ("Δ angle", metrics_detail.delta_angle, ACCENT_ALT),
            ("Velocity", metrics_detail.velocity, TEXT_SECONDARY),
            ("X position", pos_x if pos_x != "—" else NA_VALUE, TEXT_SECONDARY),
            ("Y position", pos_y if pos_y != "—" else NA_VALUE, TEXT_SECONDARY),
            ("Z position", pos_z if pos_z != "—" else NA_VALUE, TEXT_SECONDARY),
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
                font=("Consolas", 10),
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
        self._notify_dof_selection_changed()

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
            rows = self._filtered_position_table_rows(list(self._dof_table_history.rows))
            emphasize = label_for_item(self._active_dof_item_id() or "")
            self._fill_dof_position_table(
                rows,
                scroll_to_bottom=True,
                emphasize_dof=emphasize or None,
            )
            self._refresh_selected_dof_trajectory_3d()
            self._update_dof_table_controls_state()
        return added

    def _append_bilateral_foot_tick(self) -> None:
        """Record bilateral foot ground distance for the current playback frame."""
        if not self.playing or not self.skeleton_player:
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
        """Show foot-specific table columns only when a foot point is active."""
        tree = getattr(self, "dof_pos_tree", None)
        if tree is None:
            return
        active = self._active_dof_item_id()
        display = table_display_columns_for_item(active)
        try:
            tree.configure(displaycolumns=display)
        except tk.TclError:
            return
        widths = getattr(tree, "_sw_column_widths", None) or DOF_TABLE_WIDTHS
        for col in DOF_TABLE_COLUMNS:
            if col not in display:
                continue
            try:
                tree.heading(col, text=DOF_TABLE_HEADINGS.get(col, col))
                tree.column(col, width=widths.get(col, 72))
            except tk.TclError:
                pass

    def _update_dof_table_controls_state(self) -> None:
        has_playback_tracking = self._session_collector.sample_count > 0
        has_table_rows = bool(self._dof_table_history.rows)
        has_session_data = self._has_saveable_session_data()
        has_view_data = has_playback_tracking or has_table_rows
        clear_state = tk.NORMAL if has_table_rows else tk.DISABLED
        export_state = tk.NORMAL if has_playback_tracking else tk.DISABLED
        view_state = tk.NORMAL if has_view_data else tk.DISABLED
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

    def _save_session_to_files(self) -> None:
        """Save the full session (video path, selection, playback, analysis) to disk."""
        if self._presentation_mode:
            messagebox.showinfo(
                "Save Session",
                "Load and analyze a real video before saving a session.",
            )
            return
        if not self.selection.selected:
            messagebox.showinfo(
                "Save Session",
                "Select at least one body point to save.",
            )
            return

        snapshot = self._build_session_bundle_snapshot()
        if snapshot is None or snapshot.recording is None:
            messagebox.showwarning(
                "Save Session",
                "No motion data available to save.",
            )
            return

        config.ensure_output_dirs()
        initial = str(config.SESSION_EXPORT_DIR.resolve())
        target = filedialog.askdirectory(
            title="Choose folder for saved session",
            initialdir=initial,
        )
        if not target:
            return

        from stablewalk.io.session_bundle import SessionBundleError, export_session_bundle

        try:
            bundle_dir = export_session_bundle(snapshot, target)
        except SessionBundleError as exc:
            messagebox.showerror("Save Session failed", str(exc))
            return
        except OSError as exc:
            messagebox.showerror("Save Session failed", str(exc))
            return

        self.status.configure(text=f"Session saved → {bundle_dir.name}")
        messagebox.showinfo(
            "Session saved",
            "Session saved successfully.\n\n"
            f"Folder:\n{bundle_dir.resolve()}\n\n"
            "Use File → Load Session to reopen it later.",
        )

    def _load_session_from_files(self) -> None:
        """Load a previously saved session bundle from disk."""
        config.ensure_output_dirs()
        initial = str(config.SESSION_EXPORT_DIR.resolve())
        path = filedialog.askdirectory(
            title="Select saved session folder",
            initialdir=initial,
        )
        if not path:
            return
        self._apply_session_bundle_from_path(path)

    def _import_analysis_data(self) -> None:
        """Import analysis data from a session bundle (metadata file or folder)."""
        config.ensure_output_dirs()
        initial = str(config.SESSION_EXPORT_DIR.resolve())
        path = filedialog.askopenfilename(
            title="Import analysis data",
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
                return
            path = folder
        self._apply_session_bundle_from_path(path)

    def _apply_session_bundle_from_path(self, path: str) -> None:
        from stablewalk.io.session_bundle import (
            SessionBundleError,
            load_session_bundle,
            selected_ids_from_payload,
            tracking_rows_to_kinematic_samples,
            tracking_rows_to_table_rows,
        )

        try:
            loaded = load_session_bundle(path)
        except SessionBundleError as exc:
            messagebox.showerror("Load failed", str(exc))
            return

        metadata = loaded.metadata
        playback = metadata.get("playback") or {}
        ui = metadata.get("ui") or {}
        frame_index = int(playback.get("frame_index", 0) or 0)

        selected = selected_ids_from_payload(loaded.selected_points)
        if not selected:
            messagebox.showwarning(
                "Load Session",
                "No valid selected points found in the saved session.",
            )
            return

        last_selected = loaded.selected_points.get("last_selected") or ui.get(
            "last_selected"
        )
        charted = (
            loaded.selected_points.get("active_item_id")
            or loaded.selected_points.get("charted_item_id")
            or ui.get("active_item_id")
            or ui.get("charted_item_id")
        )

        restored_motion = False
        poses_path = metadata.get("poses_json_path")
        if loaded.recording is not None:
            self._set_gait_motion(loaded.recording)
            restored_motion = True
        elif poses_path and Path(poses_path).is_file():
            try:
                self.load_poses(poses_path, fresh=True)
                restored_motion = True
            except Exception as exc:
                loaded.warnings.append(f"Could not reload poses JSON: {exc}")

        if not restored_motion:
            if not messagebox.askyesno(
                "Limited restore",
                "Motion recording could not be restored.\n\n"
                "Table and summary data will still be loaded, but graphs and video "
                "require the original video or gait_motion.json.\n\n"
                "Continue?",
            ):
                return

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
                frame_index = int(float(frame_num)) - 1
            except (TypeError, ValueError):
                frame_index = 0
            key = item_id or row.get("selected_point") or ""
            if key:
                self._session_collector._last_frame_by_item[key] = frame_index

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

        if restored_motion and self.skeleton_player is not None:
            self.skeleton_player.go_to(frame_index)
            self._show_pose_at(frame_index)
            if hasattr(self, "frame_var"):
                self.frame_var.set(frame_index)

        if charted and charted in selected:
            self.selection.set_active(charted)
        self._sync_active_joint_from_charted()

        self._refresh_dof_position_table()
        self._refresh_selected_dof_trajectory_3d(force_draw=True)
        self._refresh_realtime_analysis(force_draw=True)
        self._update_interactive_skeleton(force_draw=True)
        self._update_dof_table_controls_state()
        self._update_analysis_export_state()

        self.status.configure(text=f"Session loaded ← {loaded.bundle_dir.name}")
        warn_text = ""
        if loaded.warnings:
            warn_text = "\n\nNotes:\n" + "\n".join(f"• {w}" for w in loaded.warnings[:5])
        messagebox.showinfo(
            "Session loaded",
            f"Restored session from:\n{loaded.bundle_dir.resolve()}\n\n"
            f"Selected points: {len(selected)}\n"
            f"Tracking rows: {len(loaded.tracking_rows)}\n"
            f"Frame: {frame_index + 1}"
            f"{warn_text}",
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
        samples = self._collect_session_samples_for_active()
        if not samples:
            messagebox.showwarning(
                "Export Tracking Data",
                "No tracking data to export.\n\n"
                "Select a body point and play the video to record samples.",
            )
            return

        config.ensure_output_dirs()
        output_dir = config.TRACKING_EXPORT_DIR

        video_source = self._resolve_session_video_source() or None
        active = self._active_dof_item_id()
        selected_dofs = [active] if active else []
        fps = None
        if self.sequence is not None:
            fps = float(self.sequence.fps)
        elif self.gait_motion is not None:
            fps = float(self.gait_motion.fps)

        try:
            written = export_tracking_bundle(
                samples,
                output_dir,
                video_source=video_source,
                selected_dofs=selected_dofs,
                fps=fps,
                include_json=True,
            )
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return

        csv_path = written["csv"].resolve()
        lines = [
            f"Exported {len(samples)} tracking samples.",
            "",
            f"CSV:\n{csv_path}",
        ]
        if "json" in written:
            json_path = written["json"].resolve()
            lines.extend(["", f"JSON:\n{json_path}"])
        message = "\n".join(lines)
        self.status.configure(text=f"Tracking data exported → {csv_path.name}")
        messagebox.showinfo("Export Joint CSV", message)
        logger.info(
            "Tracking data exported: samples=%d csv=%s json=%s",
            len(samples),
            csv_path,
            written.get("json", ""),
        )

    def _export_analysis_report(self) -> None:
        """Write a human-readable gait analysis report to the reports folder."""
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

    def _export_motion_reference(self) -> None:
        """Export canonical motion reference NPZ for retargeting."""
        if self._presentation_mode:
            messagebox.showinfo(
                "Export Motion Reference",
                "Load and analyze a real video before exporting motion reference.",
            )
            return
        config.ensure_output_dirs()
        try:
            if self.sequence is not None:
                from stablewalk.io.motion_reference_export import (
                    export_motion_reference_from_sequence,
                )

                run_name = Path(self._resolve_session_video_source() or "session").stem
                result = export_motion_reference_from_sequence(
                    self.sequence,
                    config.MOTION_REFERENCE_EXPORT_DIR,
                    run_name=run_name or "session",
                )
            elif self._poses_path and self._poses_path.is_file():
                from stablewalk.io.motion_reference_export import (
                    export_motion_reference_from_poses,
                )

                result = export_motion_reference_from_poses(
                    self._poses_path,
                    config.MOTION_REFERENCE_EXPORT_DIR,
                )
            else:
                messagebox.showinfo(
                    "Export Motion Reference",
                    "Load pose data first (Run Full Analysis).",
                )
                return
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.status.configure(text=f"Motion reference → {result.npz_path.name}")
        messagebox.showinfo(
            "Export Motion Reference",
            f"Motion reference exported:\n{result.npz_path.resolve()}",
        )

    def _run_real_to_sim_pipeline(self) -> None:
        """Run offline 4-stage Real-to-Sim pipeline."""
        if self._presentation_mode or self.gait_motion is None:
            messagebox.showinfo(
                "Real-to-Sim Pipeline",
                "Load and analyze a real video before running the pipeline.",
            )
            return
        config.ensure_output_dirs()
        try:
            from stablewalk.real_to_sim.pipeline import run_real_to_sim_pipeline

            run_name = Path(self._resolve_session_video_source() or "session").stem or "session"
            report = run_real_to_sim_pipeline(
                self.gait_motion,
                config.MOTION_REFERENCE_EXPORT_DIR,
                run_name=run_name,
                sequence=self.sequence,
                cycles=self._gait_cycle,
            )
            self._last_rts_report = report.to_dict()
            self._virtual_grf = None
            if self._gait_cycle is not None:
                from stablewalk.analysis.virtual_grf import estimate_virtual_grf

                self._virtual_grf = estimate_virtual_grf(
                    self.gait_motion,
                    self._gait_cycle,
                    gait_features=self._gait_features,
                    sequence=self.sequence,
                )
            self._refresh_physics_force_panel()
            self._refresh_real_to_sim_advanced_panel()
        except Exception as exc:
            messagebox.showerror("Real-to-Sim failed", str(exc))
            return
        stages = "\n".join(
            f"• {s.stage}: {s.status} — {s.detail[:80]}"
            for s in report.stages
        )
        self.status.configure(text=f"Real-to-Sim → {report.run_name}")
        messagebox.showinfo(
            "Real-to-Sim Pipeline",
            f"Pipeline complete.\n\n{stages}\n\nReport:\n{report.report_path}",
        )

    def _export_amp_reference(self) -> None:
        """Export AMP reference clip for Isaac Lab."""
        if self._presentation_mode or self.gait_motion is None or self._gait_cycle is None:
            messagebox.showinfo(
                "Export AMP Reference",
                "Load and analyze a video first.",
            )
            return
        config.ensure_output_dirs()
        try:
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

            run_name = Path(self._resolve_session_video_source() or "session").stem or "session"
            run_dir = config.MOTION_REFERENCE_EXPORT_DIR / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            motion_path = run_dir / "stablewalk_motion.npz"
            gait_style = extract_gait_style_fingerprint(
                self.gait_motion, self._gait_cycle, gait_features=self._gait_features
            )
            export_motion_reference_npz(
                self.gait_motion,
                self._gait_cycle,
                motion_path,
                gait_style=gait_style,
            )
            motion = load_motion_reference(motion_path)
            retargeted = retarget_motion_reference(motion, load_retarget_config())
            result = export_amp_reference(
                motion,
                retargeted,
                config.MOTION_REFERENCE_EXPORT_DIR,
                run_name=run_name,
                gait_style=gait_style,
            )
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.status.configure(text=f"AMP reference → {result.npz_path.name}")
        messagebox.showinfo(
            "Export AMP Reference",
            f"AMP reference exported for Isaac Lab:\n{result.npz_path.resolve()}",
        )

    def _export_selected_point_analysis(self) -> None:
        """Export full analysis data for all selected points (session bundle)."""
        self._export_analysis_data()

    def _fill_dof_position_table(
        self,
        rows: list[tuple[str, ...]],
        *,
        scroll_to_bottom: bool = False,
        emphasize_dof: str | None = None,
    ) -> None:
        """Update the selected-DOF position table with alternating row shading."""
        if not hasattr(self, "dof_pos_tree"):
            return
        dof_col = 2
        children = self.dof_pos_tree.get_children()
        for i, row in enumerate(rows):
            base_tag = "even" if i % 2 == 0 else "odd"
            tags = (base_tag,)
            if emphasize_dof and len(row) > dof_col and row[dof_col] == emphasize_dof:
                tags = ("selected",)
            if i < len(children):
                self.dof_pos_tree.item(children[i], values=row, tags=tags)
            else:
                self.dof_pos_tree.insert("", tk.END, values=row, tags=tags)
        for iid in children[len(rows) :]:
            self.dof_pos_tree.delete(iid)
        widths = getattr(self.dof_pos_tree, "_sw_column_widths", None)
        if widths:
            for col, w in widths.items():
                try:
                    self.dof_pos_tree.column(col, width=w)
                except tk.TclError:
                    pass
        try:
            visible = max(6, min(len(rows), 10))
            self.dof_pos_tree.configure(height=visible)
            self.dof_pos_tree.yview_moveto(1.0 if scroll_to_bottom else 0.0)
        except tk.TclError:
            pass
        self._update_table_summary_label()
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
        self._sync_active_joint_from_charted()
        self._sync_dof_checkboxes()
        self._update_dof_selection_chrome()
        self._sync_dof_analysis_panel_state()
        self._refresh_dof_position_table()
        self._traj_draw_cache_key = None
        self._refresh_selected_dof_trajectory_3d(force_draw=force_draw)
        self._render_dof_traj_canvas(force=force_draw)
        self._update_interactive_skeleton(force_draw=force_draw)
        hint = getattr(self, "lbl_dof_table_hint", None)
        if hint is not None:
            hint.configure(text=self._dof_table_hint_text())
        if self.sequence:
            self._refresh_display()

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
            ("lbl_dof_coord_mode_value", "ROOT-RELATIVE"),
            ("lbl_dof_traj_mode_value", "CURRENT PROGRESS"),
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
        self._dof_traj_projection_mode = None
        self._traj_draw_cache_key = None
        self._refresh_selected_dof_trajectory_3d(force_draw=True)

    def _on_dof_coord_mode_changed(self) -> None:
        self._traj_draw_cache_key = None
        self._refresh_selected_dof_trajectory_3d(force_draw=True)

    def _on_dof_traj_display_changed(self) -> None:
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

        ax = getattr(self, "ax_dof_traj", None)
        if ax is not None:
            try:
                ax.get_zlim()
                return
            except AttributeError:
                pass
        self.fig_dof_traj.clf()
        self.ax_dof_traj = self.fig_dof_traj.add_subplot(111, projection="3d")
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

        joint_val = getattr(self, "lbl_selected_joint_value", None)
        if joint_val is not None:
            joint_val.configure(text=joint_label)

        coord_var = getattr(self, "var_dof_coord_mode", None)
        coord_mode = coord_var.get() if coord_var is not None else "ROOT-RELATIVE"
        coord_val = getattr(self, "lbl_dof_coord_mode_value", None)
        if coord_val is not None:
            coord_val.configure(text=coordinate_mode_display(coord_mode))

        display_var = getattr(self, "var_dof_traj_display", None)
        traj_mode = display_var.get() if display_var is not None else "CURRENT PROGRESS"
        traj_val = getattr(self, "lbl_dof_traj_mode_value", None)
        if traj_val is not None:
            traj_val.configure(text=traj_mode)

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
                conf_lbl.configure(text="Trajectory Confidence: INSUFFICIENT")
            return

        if metrics is not None:
            if travel_lbl is not None:
                travel_lbl.configure(text=f"Travel: {metrics.total_travel_m * 100:.1f} cm")
            if smooth_lbl is not None:
                smooth_lbl.configure(text=f"Smoothness: {metrics.smoothness}")
            if dev_lbl is not None:
                dev_lbl.configure(
                    text=f"Maximum deviation: {metrics.max_deviation_m * 100:.1f} cm"
                )
            if metrics_lbl is not None and path:
                xs = [p.x for p in path]
                ys = [p.y for p in path]
                zs = [p.z for p in path]
                span_x = (max(xs) - min(xs)) * 100.0 if xs else 0.0
                span_y = (max(ys) - min(ys)) * 100.0 if ys else 0.0
                span_z = (max(zs) - min(zs)) * 100.0 if zs else 0.0
                rom = max(span_x, span_y, span_z)
                travel_cm = metrics.total_travel_m * 100.0
                metrics_lbl.configure(
                    text=(
                        f"Travel {travel_cm:.1f} cm  ·  ROM {rom:.1f} cm  ·  "
                        f"side {span_x:.1f} · up {span_y:.1f} · fwd {span_z:.1f} cm"
                    )
                )
        zoom_pct = getattr(self.ax_dof_traj, "_stablewalk_zoom_note_pct", None)
        summary_text = truncate_dashboard_explanation(interpretation.sentence, max_len=220)
        if zoom_pct is not None:
            summary_text = (
                f"Magnified view — true travel ≈ {zoom_pct:.1f}% of body height. "
                f"{summary_text}"
            )
        if summary_lbl is not None:
            summary_lbl.configure(text=summary_text)
        if conf_lbl is not None:
            conf_lbl.configure(
                text=(
                    "Trajectory Confidence: "
                    f"{format_trajectory_confidence(interpretation.confidence)}"
                )
            )

    def _refresh_motion_trajectory_on_frame(self, *, force_draw: bool = False) -> None:
        """Keep trajectory graphs in sync with the current playback frame."""
        if not hasattr(self, "fig_dof_traj"):
            return
        has_session = bool(
            self.skeleton_player and getattr(self.skeleton_player, "frame_count", 0) > 0
        )
        if not has_session:
            return
        self._refresh_selected_dof_trajectory_3d(force_draw=force_draw)

    def _refresh_selected_dof_trajectory_3d(self, *, force_draw: bool = False) -> None:
        """Update the single 3D trajectory graph for the active selected point."""
        if not hasattr(self, "fig_dof_traj"):
            return

        item_id = self._active_dof_item_id()
        has_session = bool(
            self.skeleton_player and getattr(self.skeleton_player, "frame_count", 0) > 0
        )
        if not item_id or not self.selection.selected:
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

        display_mode = "CURRENT PROGRESS"
        var_display = getattr(self, "var_dof_traj_display", None)
        if var_display is not None:
            display_mode = var_display.get() or "CURRENT PROGRESS"

        coord_mode = "ROOT-RELATIVE"
        var_coord = getattr(self, "var_dof_coord_mode", None)
        if var_coord is not None:
            coord_mode = var_coord.get() or "ROOT-RELATIVE"

        motion_series = self._motion_frame_series() if coord_mode == "GLOBAL" else None

        draw_key = (
            item_id,
            projection_mode,
            display_mode,
            coord_mode,
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
        )

        if projection_mode == "3D":
            relayout_single_dof_viewport(self.ax_dof_traj)
            from stablewalk.ui.viewers.dof_trajectory_3d import _style_single_dof_trajectory_ticks

            _style_single_dof_trajectory_ticks(self.ax_dof_traj)

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

        from stablewalk.ui.dof_selection import label_for_item

        motion = getattr(self, "_tab_motion", None)
        traj = getattr(self, "traj_panel", None)
        canvas_tw = None
        if hasattr(self, "canvas_dof_traj"):
            canvas_tw = self.canvas_dof_traj.get_tk_widget()

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
        if os.environ.get("STABLEWALK_TRAJ_DEBUG", "").strip().lower() in ("1", "true", "yes"):
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
        if os.environ.get("STABLEWALK_GUI_DEBUG", "").strip() in ("1", "true", "yes"):
            print(message, flush=True)

    def _refresh_dof_position_table(self) -> None:
        """Update the position table for the current frame (no throttling)."""
        if not hasattr(self, "dof_pos_tree"):
            return
        if not self.skeleton_player:
            self._clear_dof_position_table()
            return

        snap = self.skeleton_player.current_snapshot()
        if not snap:
            self._refresh_selected_dof_trajectory_3d()
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

        self._refresh_selected_dof_trajectory_3d()
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
        self._notify_dof_selection_changed()

    def _render_overview_traj_canvas(self, *, force: bool = False) -> None:
        """Fit and paint the Overview-tab trajectory dock canvas."""
        if not hasattr(self, "canvas_dof_traj_overview"):
            return
        from stablewalk.ui.tk.dashboard_layout import _fit_trajectory_figure

        host = getattr(self, "overview_traj_canvas_host", None)
        if host is None:
            return
        if _fit_trajectory_figure(
            self.canvas_dof_traj_overview,
            self.fig_dof_traj_overview,
            self.ax_dof_traj_overview,
            graph_host=host,
        ):
            if force:
                self.canvas_dof_traj_overview.draw()
            else:
                self.canvas_dof_traj_overview.draw_idle()

    def _refresh_overview_trajectory_dock(self, *, force_draw: bool = False) -> None:
        """Mirror the active joint path into the Overview trajectory dock."""
        if not getattr(self, "_overview_traj_dock_visible", False):
            return
        if not hasattr(self, "fig_dof_traj_overview"):
            return

        item_id = self._active_dof_item_id()
        traj_panel = getattr(self, "overview_traj_panel", None)
        demo = getattr(self, "_active_demo_gait", None)
        if traj_panel is not None and item_id:
            from stablewalk.ui.dof_selection import label_for_item

            joint_label = label_for_item(item_id) or "Joint"
            if demo is not None:
                traj_panel.configure(
                    text=f"  {demo.button_label} — {joint_label} 3D Path  "
                )
            else:
                traj_panel.configure(text=f"  {joint_label} 3D Path  ")

        has_session = bool(
            self.skeleton_player and getattr(self.skeleton_player, "frame_count", 0) > 0
        )
        if not item_id or not self.selection.selected or not has_session:
            from stablewalk.ui.viewers.dof_trajectory_3d import (
                relayout_single_dof_viewport,
                setup_single_dof_trajectory_axes,
            )

            self.ax_dof_traj_overview.cla()
            if hasattr(self.ax_dof_traj_overview, "_stablewalk_traj_artists"):
                del self.ax_dof_traj_overview._stablewalk_traj_artists
            if hasattr(self.ax_dof_traj_overview, "_stablewalk_stable_viewport"):
                del self.ax_dof_traj_overview._stablewalk_stable_viewport
            self.ax_dof_traj_overview._stablewalk_plot_legend = None
            setup_single_dof_trajectory_axes(self.ax_dof_traj_overview)
            self.ax_dof_traj_overview.text2D(
                0.5,
                0.5,
                EMPTY_SELECT_DOF_CHART,
                transform=self.ax_dof_traj_overview.transAxes,
                ha="center",
                va="center",
                color=MUTED,
                fontsize=10,
            )
            relayout_single_dof_viewport(self.ax_dof_traj_overview)
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
                        "● Start (green)  —  faded path = earlier steps  —  "
                        "bright end = current stride  —  ● Now (red)"
                    )
                )
            self._render_overview_traj_canvas(force=force_draw)
            return

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

        display_mode = "CURRENT PROGRESS"

        # Pelvis-relative: shows the joint's own movement (not whole-body walking
        # translation, which inflates GLOBAL travel to hundreds of cm).
        coord_mode = "ROOT-RELATIVE"
        motion_series = None

        if hasattr(self.ax_dof_traj_overview, "_stablewalk_stable_viewport"):
            del self.ax_dof_traj_overview._stablewalk_stable_viewport
        self._traj_draw_cache_key = None

        from stablewalk.ui.dof_selection import anchor_joint_for_item, label_for_item
        from stablewalk.ui.viewers.dof_trajectory_3d import (
            _joint_path_with_times,
            draw_dof_trajectory_panel,
            relayout_single_dof_viewport,
            setup_single_dof_trajectory_axes,
            summarize_overview_trajectory,
        )

        if projection_mode == "3D":
            setup_single_dof_trajectory_axes(self.ax_dof_traj_overview)

        view_type = getattr(self._biomech, "view_type", None) if self._biomech else None
        self.ax_dof_traj_overview._stablewalk_view_type = view_type

        draw_dof_trajectory_panel(
            self.ax_dof_traj_overview,
            recording,
            item_id,
            projection_mode=projection_mode,
            end_frame_float=end_frame_float,
            tip_snapshot=tip_snapshot,
            clear=True,
            display_mode=display_mode,
            coord_mode=coord_mode,
            motion_series=motion_series,
        )
        if projection_mode == "3D":
            relayout_single_dof_viewport(self.ax_dof_traj_overview)

        metrics_lbl = getattr(self, "lbl_overview_traj_metrics", None)
        detail_lbl = getattr(self, "lbl_overview_traj_detail", None)
        video_lbl = getattr(self, "lbl_overview_traj_video", None)
        if metrics_lbl is not None and recording is not None:
            joint_id = anchor_joint_for_item(item_id)
            if joint_id:
                path_pts = _joint_path_with_times(
                    recording,
                    joint_id,
                    end_frame_float,
                    coord_mode=coord_mode,
                    motion_series=motion_series,
                )
                progress_pct: float | None = None
                elapsed_s: float | None = None
                frame_index: int | None = None
                frame_count: int | None = None
                if self.skeleton_player is not None:
                    fc = max(1, int(self.skeleton_player.frame_count))
                    frame_count = fc
                    frame_index = int(
                        max(0, min(end_frame_float, float(fc - 1)))
                    )
                    progress_pct = min(
                        100.0,
                        max(0.0, (end_frame_float / max(fc - 1, 1)) * 100.0),
                    )
                    elapsed_s = float(self.skeleton_player.time_at_current())
                gait_mode: str | None = None
                demo = getattr(self, "_active_demo_gait", None)
                if demo is not None:
                    gait_mode = demo.button_label
                gait_phase: str | None = None
                phase_lbl = getattr(self, "lbl_gait_card_phase_value", None)
                if phase_lbl is not None:
                    phase_text = phase_lbl.cget("text")
                    if phase_text and phase_text != "—":
                        gait_phase = phase_text
                left_contact: str | None = None
                right_contact: str | None = None
                overview_left = getattr(self, "lbl_overview_contact_left", None)
                overview_right = getattr(self, "lbl_overview_contact_right", None)
                if overview_left is not None:
                    left_text = overview_left.cget("text")
                    if left_text and left_text != "—":
                        left_contact = left_text
                if overview_right is not None:
                    right_text = overview_right.cget("text")
                    if right_text and right_text != "—":
                        right_contact = right_text
                summary = summarize_overview_trajectory(
                    path_pts,
                    joint_label=label_for_item(item_id) or "Joint",
                    recording=recording,
                    joint_id=joint_id,
                    end_frame_float=end_frame_float,
                    gait_mode=gait_mode,
                    gait_phase=gait_phase,
                    left_contact=left_contact,
                    right_contact=right_contact,
                    progress_pct=progress_pct,
                    elapsed_s=elapsed_s,
                    frame_index=frame_index,
                    frame_count=frame_count,
                    view_type=view_type,
                )
                if summary is not None:
                    metrics_lbl.configure(text=summary.metrics_line)
                    if detail_lbl is not None:
                        detail_lbl.configure(text=summary.detail_line)
                    if video_lbl is not None:
                        video_lbl.configure(text=summary.video_line)
                    legend_lbl = getattr(self, "lbl_overview_traj_legend", None)
                    if legend_lbl is not None:
                        legend_lbl.configure(text=summary.motion_line)
                    # Low usable cycles on abnormal / walker clips
                    usable, detected = self._resolved_gait_cycle_count()
                    category_note = getattr(self, "lbl_overview_category_note", None)
                    if category_note is not None and demo is not None:
                        from stablewalk.ui.theme import ACCENT_ALT, ORANGE, WARNING

                        completeness = (
                            self._biomech.completeness_pct
                            if self._biomech is not None
                            else None
                        )
                        comp_bit = (
                            f" · {completeness:.0f}% complete"
                            if completeness is not None
                            else ""
                        )
                        if demo.key == "abnormal":
                            category_note.configure(
                                text=(
                                    "Compare: Abnormal — assisted walker gait, "
                                    f"compact hip ROM{comp_bit}"
                                ),
                                fg=WARNING,
                            )
                        elif demo.key == "normal":
                            category_note.configure(
                                text=(
                                    f"Compare: Normal — steady walking, "
                                    f"{usable or 0} usable cycles{comp_bit}"
                                ),
                                fg=ORANGE,
                            )
                        elif demo.key == "athletic":
                            category_note.configure(
                                text=(
                                    f"Compare: Performance — fast side-view gait, "
                                    f"larger knee swing{comp_bit}"
                                ),
                                fg=ACCENT_ALT,
                            )
                        else:
                            category_note.configure(text="")
                    if (
                        demo is not None
                        and "abnormal" in (demo.button_label or "").lower()
                        and (usable or 0) == 0
                        and detail_lbl is not None
                    ):
                        detail_lbl.configure(
                            text=(
                                f"{summary.detail_line}  ·  "
                                "Walker-assisted gait — expect a compact hip path "
                                "and few complete cycles."
                            )
                        )
                else:
                    metrics_lbl.configure(text="")
                    if detail_lbl is not None:
                        detail_lbl.configure(text="")
                    if video_lbl is not None:
                        video_lbl.configure(text="")
            elif detail_lbl is not None:
                detail_lbl.configure(text="")
                if video_lbl is not None:
                    video_lbl.configure(text="")

        self._render_overview_traj_canvas(force=force_draw)

    def _set_overview_traj_column_weights(self, *, traj_active: bool) -> None:
        """Give the trajectory column more width when the 3D path is visible."""
        section1 = getattr(self, "_section_visual", None)
        if section1 is None:
            return
        if traj_active:
            video_w, skel_w, traj_w = 34, 28, 38
        else:
            from stablewalk.ui.tk.dashboard_sections import (
                SEC1_SKELETON_WEIGHT,
                SEC1_SUMMARY_WEIGHT,
                SEC1_VIDEO_WEIGHT,
            )

            video_w, skel_w, traj_w = (
                SEC1_VIDEO_WEIGHT,
                SEC1_SKELETON_WEIGHT,
                SEC1_SUMMARY_WEIGHT,
            )
        section1.columnconfigure(0, weight=video_w, uniform="sec1")
        section1.columnconfigure(1, weight=skel_w, uniform="sec1")
        section1.columnconfigure(2, weight=traj_w, uniform="sec1")

    def _show_overview_trajectory_dock(self, show: bool) -> None:
        """Show joint trajectory in the right column; hide gait summary while active."""
        panel = getattr(self, "overview_traj_panel", None)
        sidebar = getattr(self, "sidebar", None)
        if panel is None:
            return
        from stablewalk.ui.tk.dashboard_layout import DASHBOARD_GUTTER

        if show:
            if sidebar is not None:
                sidebar.grid_remove()
            panel.grid(row=0, column=2, sticky="nsew", padx=(DASHBOARD_GUTTER, 0))
            self._overview_traj_dock_visible = True
            self._set_overview_traj_column_weights(traj_active=True)
            panel.update_idletasks()
            self.root.update_idletasks()
            self._on_video_label_resize()
            fit_skel = getattr(self, "_fit_skeleton_canvas", None)
            if fit_skel is not None:
                fit_skel()
            self._render_overview_traj_canvas(force=True)
        else:
            panel.grid_remove()
            if sidebar is not None:
                sidebar.grid(row=0, column=2, sticky="nsew", padx=(DASHBOARD_GUTTER, 0))
            self._overview_traj_dock_visible = False
            self._set_overview_traj_column_weights(traj_active=False)
            self.root.update_idletasks()
            self._on_video_label_resize()
            fit_skel = getattr(self, "_fit_skeleton_canvas", None)
            if fit_skel is not None:
                fit_skel()

    def _sync_trajectory_mount_for_active_tab(self) -> None:
        """Reflow the Overview trajectory dock when its tab becomes active."""
        from stablewalk.ui.tk.dashboard_notebook import is_overview_tab_selected

        if (
            is_overview_tab_selected(self)
            and getattr(self, "_overview_traj_dock_visible", False)
        ):
            self._refresh_overview_trajectory_dock(force_draw=True)
            self._render_overview_traj_canvas(force=True)

    def _focus_joint_trajectory_from_skeleton(self, item_id: str) -> None:
        """
        Skeleton click: select the joint and show its 3D path on Overview.

        Keeps the video and 3D skeleton at full height; shows the path in the right column.
        Always refreshes even when the same joint was already active (re-click).
        """
        from stablewalk.ui.dof_selection import GUI_DOF_LABELS, label_for_item
        from stablewalk.ui.tk.dashboard_notebook import (
            TAB_OVERVIEW,
            select_dashboard_tab,
        )

        if item_id not in GUI_DOF_LABELS:
            return

        if item_id not in self.selection.selected:
            self.selection.activate_item(item_id)
            self._notify_dof_selection_changed()
        else:
            self.selection.set_active(item_id)
            self._sync_active_joint_from_charted()
            self._sync_dof_checkboxes()
            self._update_dof_selection_chrome()
            self._traj_draw_cache_key = None
            self._sync_dof_analysis_panel_state()
            self._refresh_selected_dof_trajectory_3d(force_draw=True)
            self._render_dof_traj_canvas(force=True)
            self._update_interactive_skeleton(force_draw=True)

        self._overview_traj_dock_visible = True
        select_dashboard_tab(self, TAB_OVERVIEW)
        self._show_overview_trajectory_dock(True)
        self._refresh_overview_trajectory_dock(force_draw=True)
        self._schedule_dof_traj_reflow()

        joint_label = label_for_item(item_id) or "Joint"
        if hasattr(self, "status"):
            self.status.configure(
                text=f"Overview — {joint_label} 3D path (video + skeleton)"
            )

    def _on_skeleton_pick(self, event) -> None:
        joint_id = joint_id_from_pick(event)
        if not joint_id:
            return
        item_id = item_for_joint(joint_id)
        if not item_id:
            return
        self._focus_joint_trajectory_from_skeleton(item_id)

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
        # Keep current position; just re-render the current frame everywhere.
        pos_f = float(self.skeleton_player.state.frame_float)
        self._playback_pos = pos_f
        self.current_pos = int(pos_f)
        self.frame_var.set(int(round(pos_f)))  # keep slider synced to current frame
        self._last_data_refresh = 0.0
        self._last_realtime_refresh = 0.0
        self._last_dof_table_refresh = 0.0
        self._update_interactive_skeleton(force_draw=True)
        self._refresh_motion_trajectory_on_frame(force_draw=True)
        if self.sequence and self.pose_indices:
            self._show_pose_at(pos_f, force_draw=True, skeleton_only=True)

    def _reset_playback(self) -> None:
        """Return video, skeleton, table, and slider to the first frame."""
        if not self.skeleton_player:
            return
        self.playing = False
        self.skeleton_player.stop(reset=True)
        self._playback_pos = 0.0
        self.current_pos = 0
        self.frame_var.set(0)
        self._sync_play_buttons()
        self._cancel_timer()
        self._3d_flush_scheduled = False
        self._pending_3d_args = None
        self._3d_motion_trail = []
        self._last_data_refresh = 0.0
        self._last_realtime_refresh = 0.0
        self._last_dof_table_refresh = 0.0
        self._update_interactive_skeleton(force_draw=True)
        self._refresh_motion_trajectory_on_frame(force_draw=True)
        if self.sequence and self.pose_indices:
            self._show_pose_at(0, force_draw=True, skeleton_only=True)

    def _on_smooth_toggle(self) -> None:
        if self.skeleton_player:
            self.skeleton_player.smooth = self.smooth_motion.get()
            self._update_interactive_skeleton(force_draw=True)

    def _on_speed_change(self, _value: str) -> None:
        self.play_speed = self.speed_var.get()
        self.lbl_speed.configure(text=f"{self.play_speed:.2f}×")

    def _on_slider(self, _value: str) -> None:
        if not self.skeleton_player:
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
        self._update_interactive_skeleton(force_draw=True)
        self._refresh_motion_trajectory_on_frame(force_draw=True)
        if self.sequence and self.pose_indices:
            self._show_pose_at(pos_f, force_draw=True, skeleton_only=True)

    def _toggle_play(self) -> None:
        if not self.skeleton_player:
            return
        self.playing = self.skeleton_player.toggle_play()
        self._sync_play_buttons()
        if self.playing:
            self._playback_pos = float(self.skeleton_player.state.frame_float)
            self._3d_motion_trail = []
            self._panel_tick = 0
            from stablewalk.ui.tk.render_diagnostics import reset_playback_render_counters

            reset_playback_render_counters(self)
            self._ensure_dof_table_tracking_on_play()
            if self._should_record_dof_table_history():
                self._append_dof_table_history_tick()
            self._append_bilateral_foot_tick()
            self._schedule_tick()
        else:
            self._cancel_timer()
            self._3d_flush_scheduled = False
            self._pending_3d_args = None
            self._update_interactive_skeleton(force_draw=True)

    def _schedule_tick(self) -> None:
        self._cancel_timer()
        if not self.playing or not self.skeleton_player:
            return
        hz = getattr(self, "_play_anim_hz", 24)
        delay = max(16, int(1000 / hz))
        self._after_id = self.root.after(delay, self._tick)

    def _tick(self) -> None:
        if self.playing and self.skeleton_player:
            hz = max(getattr(self, "_play_anim_hz", 24), 1)
            dt = 1.0 / hz
            prev_frame = self.skeleton_player.state.frame_index
            self.skeleton_player.smooth = self.smooth_motion.get()
            self.skeleton_player.advance(speed=self.play_speed, dt=dt)
            self._playback_pos = self.skeleton_player.state.frame_float
            self.current_pos = self.skeleton_player.state.frame_index
            self.frame_var.set(int(self._playback_pos))
            if (
                self._should_record_dof_table_history()
                and self.skeleton_player.state.frame_index != prev_frame
            ):
                self._append_dof_table_history_tick()
            if self.skeleton_player.state.frame_index != prev_frame:
                self._append_bilateral_foot_tick()
                self._refresh_motion_trajectory_on_frame()
            self._update_interactive_skeleton(force_draw=False)
            if self.sequence and self.pose_indices:
                self._show_pose_at(self._playback_pos, force_draw=False, skeleton_only=True)
            from stablewalk.ui.tk.render_diagnostics import record_playback_render_frame

            record_playback_render_frame(self)
            self._schedule_tick()

    def _flush_pending_3d(self) -> None:
        """Redraw 3D body after tables (avoids matplotlib blocking Tk during play)."""
        self._3d_flush_scheduled = False
        args = self._pending_3d_args
        if not args or not self.playing:
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
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None

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
        self._update_interactive_skeleton(force_draw=True)
        self._refresh_motion_trajectory_on_frame(force_draw=True)
        if self.sequence and self.pose_indices:
            self._show_pose_at(pos, force_draw=True, skeleton_only=True)

    def _go_to(self, pos: int) -> None:
        if not self.skeleton_player:
            return
        pos = max(0, min(pos, self.skeleton_player.frame_count - 1))
        self.current_pos = pos
        self.frame_var.set(pos)
        self.skeleton_player.go_to(pos)
        self._update_interactive_skeleton(force_draw=True)
        self._refresh_motion_trajectory_on_frame(force_draw=True)
        if self.sequence and self.pose_indices:
            self._show_pose_at(pos, skeleton_only=True)

    def _toggle_robot_panel(self) -> None:
        if self.show_robot_panel.get():
            self.robot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(PAD_XS, 0))
            if self.sequence and self.pose_indices:
                self._refresh_display()
        else:
            self.robot_frame.pack_forget()

    def _refresh_display(self) -> None:
        if self.skeleton_player:
            self._update_interactive_skeleton(force_draw=True)
        if self.sequence and self.pose_indices:
            self._show_pose_at(self.current_pos, skeleton_only=True)

    def _show_pose_at(
        self,
        list_pos: int | float,
        *,
        force_draw: bool = False,
        skeleton_only: bool = False,
    ) -> None:
        if not self.sequence or not self.pose_indices:
            if self.skeleton_player and not skeleton_only:
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

        skeleton = (
            self._get_skeleton(frame, frame_next, alpha)
            if frame.detected and frame.keypoints
            else None
        )

        refresh_video = not self.playing or i0 != getattr(self, "_last_video_list_pos", -1)
        highlight_joints = self._resolve_highlight_joints()
        if refresh_video:
            rgb = self._rgb_cache.get_rgb(frame.image_path)
            if self.show_skeleton.get() or self.highlight_dof.get():
                rgb = render_frame_with_skeleton(
                    rgb,
                    frame.keypoints,
                    show_skeleton=self.show_skeleton.get(),
                    highlight_dof=self.highlight_dof.get(),
                    highlight_joints=highlight_joints,
                    gait_events=frame.gait_events,
                )
            self._set_video_image(rgb)
            self._last_video_list_pos = i0

        now = time.monotonic()
        data_due = (
            not skeleton_only
            and (
                not self.playing
                or force_draw
                or (now - self._last_data_refresh) >= self._data_refresh_s
            )
        )
        if data_due:
            self._update_panels(frame, i0, frame_idx, frame_next, alpha, skeleton=skeleton)
            self._last_data_refresh = now
        self._update_chart(playhead_list_pos=i0)
        self._panel_tick += 1

        if not skeleton_only:
            self._update_interactive_skeleton(force_draw=force_draw or not self.playing)

        if self.show_robot_panel.get() and (not self.playing or force_draw) and not skeleton_only:
            self._update_robot_view(frame, frame_next, alpha)

        self._sync_transport_labels()

    def _get_skeleton(self, frame: PoseFrame, frame_next: PoseFrame, alpha: float) -> Skeleton3D:
        sk_a = self._skeleton_cache.get(frame.frame_index)
        if sk_a is None:
            sk_a = skeleton_from_frame_data(
                frame.keypoints, frame.skeleton_3d, self._skeleton_scale
            )
        if alpha <= 0 or not self.smooth_motion.get():
            return normalize_skeleton_height(sk_a)

        sk_b = self._skeleton_cache.get(frame_next.frame_index)
        if sk_b is None:
            sk_b = skeleton_from_frame_data(
                frame_next.keypoints, frame_next.skeleton_3d, self._skeleton_scale
            )
        blended = interpolate_skeleton_3d(sk_a, sk_b, alpha)
        return normalize_skeleton_height(blended)

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
        self.fig_robot.tight_layout()
        self.canvas_robot.draw_idle()

    def _set_video_image(self, rgb: np.ndarray) -> None:
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

        # "Contain" fit: scale the whole composited frame (video + baked-in
        # pose overlay) to fit inside the visible label without cropping, so
        # the full body — head to feet — stays visible. Scaling uniformly
        # keeps the overlay points/lines aligned with the video.
        # Maximize video within the label (contain fit, minimal padding).
        self.video_label.update_idletasks()
        lw = self.video_label.winfo_width()
        lh = self.video_label.winfo_height()
        if lw <= 1 or lh <= 1:
            lw, lh = 900, 720
        pad = 4
        box_w = max(lw - pad * 2, 64)
        box_h = max(lh - pad * 2, 64)

        scale = min(box_w / src_w, box_h / src_h)
        target_scale = scale * 0.99
        new_w = max(1, int(round(src_w * target_scale)))
        new_h = max(1, int(round(src_h * target_scale)))
        fitted = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(fitted)
        self.video_label.configure(image=self._photo, text="")
        # Persistent reference prevents Tk GC and stale-frame artifacts.
        self.video_label.image = self._photo  # type: ignore[attr-defined]

    def _on_video_label_resize(self, _event: object = None) -> None:
        """Re-fit the current frame when the video panel is resized."""
        if self._last_video_rgb is None:
            return
        if self._video_resize_after is not None:
            try:
                self.root.after_cancel(self._video_resize_after)
            except (tk.TclError, ValueError):
                pass

        def _refit() -> None:
            self._video_resize_after = None
            if self._last_video_rgb is not None:
                self._set_video_image(self._last_video_rgb)

        self._video_resize_after = self.root.after(80, _refit)

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

        self.lbl_frame_data.configure(
            text=f"Frame {frame.frame_index + 1}  ·  {len(snap.positions_3d)} joints  ·  {gait_dof} DoF"
        )

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

        self.status.configure(
            text=f"Frame {frame_idx + 1}  ·  {len(snap.positions_3d)} joints  ·  {gait_dof} DoF"
        )

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
        left = f"{summary.left_rom_deg:.0f}°" if summary.left_rom_deg is not None else "—"
        right = f"{summary.right_rom_deg:.0f}°" if summary.right_rom_deg is not None else "—"
        asym = (
            f"{summary.rom_asymmetry_pct:.1f}%"
            if summary.rom_asymmetry_pct is not None
            else "—"
        )
        compact = getattr(self, "lbl_knee_summary_compact", None) or getattr(
            self, "lbl_knee_interp_compact", None
        )
        if compact is not None:
            compact.configure(
                text=f"L ROM {left} · R ROM {right} · Asymmetry {asym}"
            )
        self._knee_motion_summary = summary

    def _update_chart(self, *, playhead_list_pos: int | None = None) -> None:
        """Plot knee flexion in Video Time or Gait Cycle % mode."""
        if not self.sequence or not self.pose_indices:
            return

        from stablewalk.ui.viewers.gait_cycle_chart import (
            draw_knee_chart_cycle_mode,
            draw_knee_chart_time_mode,
            style_gait_chart,
        )

        self.ax_chart.clear()
        axis_mode = getattr(self, "var_knee_chart_axis", None)
        mode = axis_mode.get() if axis_mode is not None else "Video Time"
        if mode == "Time":
            mode = "Video Time"

        gait_events = None
        gait_cycle = getattr(self, "_gait_cycle", None)
        if gait_cycle is not None:
            gait_events = list(gait_cycle.events)

        if mode == "Gait Cycle %":
            draw_knee_chart_cycle_mode(
                self.ax_chart,
                getattr(self, "_gait_features", None),
                show_envelope=True,
            )
        else:
            if self._knee_angle_series is None:
                self._rebuild_knee_angle_series()
            if playhead_list_pos is None:
                playhead_list_pos = self.current_pos
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

        style_gait_chart(self.ax_chart, self.fig)
        self._update_knee_interpretation_panel(chart_mode=mode)
        self.chart_canvas.draw_idle()
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

    def run(self) -> None:
        self.root.mainloop()


def launch_gui(poses_path: str | Path | None = None) -> None:
    """Start the desktop GUI (blocking)."""
    from stablewalk.opensim_sdk import log_opensim_startup_status

    log_opensim_startup_status(logging.getLogger("stablewalk"))
    app = StableWalkGUI(poses_path=poses_path)
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
