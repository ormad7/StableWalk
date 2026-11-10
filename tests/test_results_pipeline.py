"""Results visual analysis pipeline stage mapping."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from stablewalk.monitoring.pipeline_status import (
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    STATUS_UNAVAILABLE,
    PipelineStageItem,
    PipelineStatusReport,
)
from stablewalk.ui.tk.dashboard_results_pipeline import (
    OPTIONAL_RESULTS_STAGE_KEYS,
    REQUIRED_RESULTS_STAGE_KEYS,
    RESULTS_PIPELINE_STAGES,
    STATUS_FAILED,
    STATUS_RUNNING,
    _STAGE_ICONS,
    _calculated_confidence_percent,
    _results_pipeline_view_signature,
    _short_partial_reason,
    _stage_details_text,
    _stage_status_label,
    _stage_warning_preview,
    _unavailable_stage_text,
    build_results_stage_views,
    compute_results_pipeline_progress,
    format_required_analysis_headline,
    results_pipeline_session_state,
    update_results_analysis_pipeline,
)
from stablewalk.monitoring.pipeline_status import STATUS_COMPLETED as _COMPLETED
from stablewalk.monitoring.pipeline_status import STATUS_UNAVAILABLE as _UNAVAILABLE


def _item(key: str, *, status: str = STATUS_COMPLETED) -> PipelineStageItem:
    return PipelineStageItem(
        key=key,
        label=key,
        group="test",
        status=status,  # type: ignore[arg-type]
        detail="ok",
        tooltip="test",
    )


def test_results_pipeline_stage_icons() -> None:
    keys = [k for k, _label, _kind in RESULTS_PIPELINE_STAGES]
    for key in keys:
        assert key in _STAGE_ICONS
        assert _STAGE_ICONS[key]


def test_results_pipeline_stage_order() -> None:
    keys = [k for k, _label, _kind in RESULTS_PIPELINE_STAGES]
    assert keys == [
        "video",
        "pose_estimation",
        "filtering",
        "reconstruction_3d",
        "opensim",
        "biomechanics",
        "metrics",
        "export",
    ]
    labels = [label for _k, label, _kind in RESULTS_PIPELINE_STAGES]
    assert labels == [
        "Video",
        "Pose Detection",
        "Filtering",
        "3D Reconstruction",
        "OpenSim",
        "Biomechanics",
        "Metrics",
        "Export",
    ]


def test_results_pipeline_does_not_treat_completion_as_confidence() -> None:
    assert _calculated_confidence_percent("Sub-step completion: 0/5") is None
    assert _calculated_confidence_percent("Pose detection coverage: 0/5 frames (0%)") is None
    assert _calculated_confidence_percent("OpenSim marker coverage: 82%") is None
    assert _calculated_confidence_percent("Foot contact confidence: 82%") == 82
    assert _calculated_confidence_percent("78%") == 78


def test_results_pipeline_unrun_and_partial_display_text() -> None:
    assert _unavailable_stage_text("opensim") == "Optional – Not run"
    assert _unavailable_stage_text("export") == "Optional – Not run"
    partial = SimpleNamespace(
        detail="Marker coverage is incomplete for inverse kinematics",
        warnings="",
    )
    reason = _short_partial_reason(partial)  # type: ignore[arg-type]
    assert reason
    assert len(reason) <= 46


def test_stage_card_preview_and_details_preserve_technical_information() -> None:
    view = SimpleNamespace(
        status=STATUS_PARTIAL,
        warnings="First actionable warning with concise context.\nSecond detailed warning.",
        detail="Partial because one optional artifact was not produced.",
        purpose="Generate analyzed outputs.",
        tooltip="",
        duration_text="2.4 s",
        output_text="Pose and metrics ready",
        quality="Moderate tracking quality",
        artifacts_text=r"C:\analysis\session\long\artifact.json",
        logs="worker completed with one optional omission",
    )
    preview = _stage_warning_preview(view)  # type: ignore[arg-type]
    assert "\n" not in preview
    assert len(preview) <= 58
    details = _stage_details_text(view)  # type: ignore[arg-type]
    assert "Second detailed warning." in details
    assert r"C:\analysis\session\long\artifact.json" in details
    assert "worker completed with one optional omission" in details


def test_results_pipeline_separates_required_and_optional_stages() -> None:
    assert OPTIONAL_RESULTS_STAGE_KEYS == {"opensim", "export"}
    assert not (OPTIONAL_RESULTS_STAGE_KEYS & REQUIRED_RESULTS_STAGE_KEYS)
    optional = SimpleNamespace(key="opensim", status=STATUS_UNAVAILABLE)
    assert _stage_status_label(optional) == "Optional – Not run"  # type: ignore[arg-type]
    required = SimpleNamespace(key="metrics", status=STATUS_UNAVAILABLE)
    assert _stage_status_label(required) == "Not run"  # type: ignore[arg-type]


def test_results_stage_views_empty_session() -> None:
    gui = SimpleNamespace(
        sequence=None,
        _analysis_summary_cache=None,
        _analysis_compute_active=False,
        _opensim_auto_pipeline_busy=False,
        _pipeline_status_context_cache=None,
        smooth_motion=SimpleNamespace(get=lambda: True),
        _load_rts_report_for_session=lambda: None,
    )
    views = build_results_stage_views(gui, None)
    assert len(views) == 8
    assert [v.key for v in views] == [
        "video",
        "pose_estimation",
        "filtering",
        "reconstruction_3d",
        "opensim",
        "biomechanics",
        "metrics",
        "export",
    ]
    assert views[0].status == STATUS_UNAVAILABLE
    assert views[2].key == "filtering"
    assert views[2].label == "Filtering"
    assert views[-2].key == "metrics"
    assert views[-1].key == "export"
    assert views[-1].label == "Export"


def test_results_stage_views_with_pose_items() -> None:
    report = PipelineStatusReport(
        source="walk.mp4",
        run_name="demo",
        groups=[
            (
                "Pose",
                [
                    _item("pose_mediapipe"),
                    _item("pose_smpl", status=STATUS_UNAVAILABLE),
                    _item("pose_romp", status=STATUS_UNAVAILABLE),
                    _item("motion_skeleton"),
                    _item("motion_angles"),
                    _item("motion_velocities"),
                    _item("bio_contact_mask"),
                    _item("bio_heel_strike"),
                    _item("bio_toe_off"),
                    _item("bio_cadence"),
                    _item("bio_com"),
                    _item("bio_bos"),
                    _item("osim_model", status=STATUS_PARTIAL),
                    _item("osim_trc", status=STATUS_UNAVAILABLE),
                    _item("osim_mot", status=STATUS_UNAVAILABLE),
                    _item("osim_ik", status=STATUS_UNAVAILABLE),
                    _item("osim_mapping", status=STATUS_UNAVAILABLE),
                    _item("osim_coverage", status=STATUS_UNAVAILABLE),
                    _item("retarget_export", status=STATUS_UNAVAILABLE),
                    _item("isaac_export_only", status=STATUS_UNAVAILABLE),
                ],
            )
        ],
    )
    summary = SimpleNamespace(
        source="walk.mp4",
        overall_gait_quality=SimpleNamespace(available=True, value="72/100"),
        cadence=SimpleNamespace(available=True, value="110"),
        walking_speed=SimpleNamespace(available=True, value="1.1 m/s"),
        symmetry=SimpleNamespace(available=False, value="—"),
        stability_margin=SimpleNamespace(available=False, value="—"),
        pipeline_confidence=SimpleNamespace(available=True, value="80%"),
        tracking_confidence=SimpleNamespace(available=True, value="75%"),
        analysis_confidence=SimpleNamespace(available=True, value="78%"),
        video_quality=SimpleNamespace(available=True, value="85/100"),
        scientific_interpretation="Gait metrics look consistent with walking.",
    )
    gui = SimpleNamespace(
        sequence=SimpleNamespace(frames=[object()] * 10),
        _analysis_summary_cache=summary,
        _analysis_compute_active=False,
        _opensim_auto_pipeline_busy=False,
        _pipeline_status_context_cache=None,
        smooth_motion=SimpleNamespace(get=lambda: True),
        _load_rts_report_for_session=lambda: None,
    )
    views = build_results_stage_views(gui, report)
    by_key = {v.key: v for v in views}
    assert by_key["video"].status == STATUS_COMPLETED
    assert by_key["pose_estimation"].status == STATUS_COMPLETED
    assert by_key["pose_estimation"].label == "Pose Detection"
    assert by_key["filtering"].status == STATUS_COMPLETED
    assert by_key["filtering"].label == "Filtering"
    assert by_key["opensim"].label == "OpenSim"
    assert by_key["opensim"].status in (STATUS_PARTIAL, STATUS_UNAVAILABLE)
    assert by_key["metrics"].status == STATUS_COMPLETED
    assert by_key["metrics"].label == "Metrics"
    assert by_key["export"].label == "Export"
    assert by_key["pose_estimation"].confidence_text
    assert by_key["biomechanics"].label == "Biomechanics"
    assert by_key["video"].artifacts_text
    assert by_key["video"].output_text
    # Each stage exposes the fields shown on the Results cards.
    for view in views:
        assert view.duration_text is not None
        assert view.confidence_text is not None
        assert view.status
        assert view.warnings is not None
        assert view.artifacts_text is not None


def test_pipeline_progress_completion_time_fps_and_remaining() -> None:
    frames = [object()] * 100
    gui = SimpleNamespace(
        sequence=SimpleNamespace(frames=frames),
        gait_motion=None,
        _load_rts_report_for_session=lambda: {
            "stages": [
                {"stage": "1_perception", "duration_s": 2.0},
                {"stage": "2_retargeting", "duration_s": 3.0},
            ]
        },
    )
    views = [
        SimpleNamespace(
            key="video",
            label="Video",
            status=_COMPLETED,
            duration_text="2.0s",
        ),
        SimpleNamespace(
            key="pose_estimation",
            label="Pose Detection",
            status=_COMPLETED,
            duration_text="2.0s",
        ),
        SimpleNamespace(
            key="opensim",
            label="OpenSim",
            status=STATUS_RUNNING,
            duration_text="—",
        ),
        SimpleNamespace(
            key="export",
            label="Export",
            status=_UNAVAILABLE,
            duration_text="—",
        ),
    ]
    progress = compute_results_pipeline_progress(gui, views)  # type: ignore[arg-type]
    assert progress.total_stages == 2
    assert progress.completed == 2
    assert progress.completion_pct == 100.0
    assert progress.total_time_s == 5.0
    assert progress.total_time_text != "—"
    assert progress.avg_fps is not None
    assert abs(progress.avg_fps - 20.0) < 0.1  # 100 frames / 5 s
    assert progress.remaining_count == 0
    assert "required analysis complete" in progress.remaining_text
    assert "Video" in progress.per_stage_text


def test_required_partial_stage_reduces_main_analysis_completion() -> None:
    gui = SimpleNamespace(
        sequence=None,
        gait_motion=None,
        _load_rts_report_for_session=lambda: None,
    )
    views = [
        SimpleNamespace(
            key="video",
            label="Video",
            status=STATUS_COMPLETED,
            duration_text="—",
        ),
        SimpleNamespace(
            key="metrics",
            label="Metrics",
            status=STATUS_PARTIAL,
            duration_text="—",
        ),
        SimpleNamespace(
            key="opensim",
            label="OpenSim",
            status=STATUS_UNAVAILABLE,
            duration_text="—",
        ),
        SimpleNamespace(
            key="export",
            label="Export",
            status=STATUS_UNAVAILABLE,
            duration_text="—",
        ),
    ]
    progress = compute_results_pipeline_progress(gui, views)  # type: ignore[arg-type]
    assert progress.completion_pct == 75.0
    assert progress.remaining_count == 1
    assert "Metrics" in progress.remaining_text
    assert "partial" in progress.remaining_text


def test_results_pipeline_session_state_labels_partial_results() -> None:
    complete = SimpleNamespace(key="video", status=STATUS_COMPLETED)
    partial = SimpleNamespace(key="metrics", status=STATUS_PARTIAL)
    optional = SimpleNamespace(key="export", status=STATUS_UNAVAILABLE)

    assert results_pipeline_session_state([complete, partial, optional]) == "partial"
    assert results_pipeline_session_state([complete]) == "complete"


def test_required_analysis_headline_is_complete_or_named_incomplete() -> None:
    complete = [
        SimpleNamespace(key="video", label="Video", status=STATUS_COMPLETED, detail="ok"),
        SimpleNamespace(
            key="metrics", label="Metrics", status=STATUS_COMPLETED, detail="ok"
        ),
        SimpleNamespace(
            key="export", label="Export", status=STATUS_UNAVAILABLE, detail=""
        ),
    ]
    headline, kind = format_required_analysis_headline(complete)  # type: ignore[arg-type]
    assert headline == "100% Complete"
    assert kind == "complete"

    stuck = [
        SimpleNamespace(key="video", label="Video", status=STATUS_COMPLETED, detail="ok"),
        SimpleNamespace(
            key="pose_estimation",
            label="Pose Detection",
            status=STATUS_COMPLETED,
            detail="ok",
        ),
        SimpleNamespace(
            key="filtering", label="Filtering", status=STATUS_COMPLETED, detail="ok"
        ),
        SimpleNamespace(
            key="reconstruction_3d",
            label="3D Reconstruction",
            status=STATUS_COMPLETED,
            detail="ok",
        ),
        SimpleNamespace(
            key="biomechanics",
            label="Biomechanics",
            status=STATUS_COMPLETED,
            detail="ok",
        ),
        SimpleNamespace(
            key="metrics",
            label="Metrics",
            status=STATUS_UNAVAILABLE,
            detail=(
                "Metrics stage not run — missing: Biomechanical analysis result"
            ),
        ),
        SimpleNamespace(
            key="opensim", label="OpenSim", status=STATUS_UNAVAILABLE, detail=""
        ),
    ]
    headline, kind = format_required_analysis_headline(stuck)  # type: ignore[arg-type]
    assert kind == "incomplete"
    assert headline.startswith("Incomplete: Metrics — ")
    assert "Biomechanical analysis result" in headline
    assert "83" not in headline
    assert "Partial results" not in headline

    waiting = [
        SimpleNamespace(
            key="metrics", label="Metrics", status=STATUS_UNAVAILABLE, detail=""
        )
    ]
    headline, kind = format_required_analysis_headline(waiting)  # type: ignore[arg-type]
    assert headline == "Waiting for analysis"
    assert kind == "waiting"

    processing = [
        SimpleNamespace(key="video", label="Video", status=STATUS_RUNNING, detail="")
    ]
    headline, kind = format_required_analysis_headline(processing)  # type: ignore[arg-type]
    assert headline == "Processing"
    assert kind == "processing"


def test_failed_required_stage_is_incomplete_not_partial_results() -> None:
    complete = SimpleNamespace(key="video", status=STATUS_COMPLETED)
    failed = SimpleNamespace(key="metrics", status=STATUS_FAILED)

    assert results_pipeline_session_state([complete, failed]) == "incomplete"


def test_available_metrics_preserve_monitored_partial_state() -> None:
    report = PipelineStatusReport(
        source="walk.mp4",
        run_name="partial",
        groups=[
            (
                "Biomechanics",
                [
                    _item("bio_cadence", status=STATUS_PARTIAL),
                    _item("bio_com", status=STATUS_UNAVAILABLE),
                ],
            )
        ],
    )
    summary = SimpleNamespace(
        source="walk.mp4",
        cadence=SimpleNamespace(available=True, value="104"),
        overall_gait_quality=None,
        walking_speed=None,
        symmetry=None,
        stability_margin=None,
        analysis_confidence=None,
        pipeline_confidence=None,
    )
    gui = SimpleNamespace(
        sequence=SimpleNamespace(frames=[object()] * 10),
        _analysis_summary_cache=summary,
        _analysis_compute_active=False,
        _opensim_auto_pipeline_busy=False,
        _pipeline_status_context_cache=None,
        smooth_motion=SimpleNamespace(get=lambda: True),
        _load_rts_report_for_session=lambda: None,
    )

    views = build_results_stage_views(gui, report)
    metrics = next(view for view in views if view.key == "metrics")
    assert metrics.status == STATUS_PARTIAL
    assert metrics.output_text == "1 derived metric outputs from partial analysis"


def test_metrics_stage_completes_from_executed_biomech_inputs_without_summary() -> None:
    """Cadence/COM/BoS/events already ran — do not leave Metrics as not-run."""
    report = PipelineStatusReport(
        source="walk.mp4",
        run_name="analyzed",
        groups=[
            (
                "Biomechanics",
                [
                    _item("bio_cadence"),
                    _item("bio_com"),
                    _item("bio_bos"),
                    _item("bio_heel_strike"),
                    _item("bio_toe_off"),
                ],
            )
        ],
    )
    gui = SimpleNamespace(
        sequence=SimpleNamespace(frames=[object()] * 10),
        _analysis_summary_cache=None,
        _biomech_analysis=object(),
        _gait_cycle=object(),
        _foot_contact=None,
        _analysis_compute_active=False,
        _opensim_auto_pipeline_busy=False,
        _pipeline_status_context_cache=None,
        smooth_motion=SimpleNamespace(get=lambda: True),
        _load_rts_report_for_session=lambda: None,
    )

    views = build_results_stage_views(gui, report)
    metrics = next(view for view in views if view.key == "metrics")
    assert metrics.status == STATUS_COMPLETED
    assert "not run" not in metrics.detail.lower()


def test_metrics_stage_lists_exact_missing_dependencies() -> None:
    from stablewalk.ui.tk.dashboard_results_pipeline import (
        metrics_stage_missing_dependencies,
    )

    report = PipelineStatusReport(
        source="walk.mp4",
        run_name="no-metrics",
        groups=[
            (
                "Biomechanics",
                [
                    _item("bio_cadence", status=STATUS_UNAVAILABLE),
                    _item("bio_com", status=STATUS_UNAVAILABLE),
                    _item("bio_bos", status=STATUS_UNAVAILABLE),
                    _item("bio_heel_strike", status=STATUS_UNAVAILABLE),
                    _item("bio_toe_off", status=STATUS_UNAVAILABLE),
                ],
            )
        ],
    )
    gui = SimpleNamespace(
        sequence=SimpleNamespace(frames=[object()] * 10),
        _analysis_summary_cache=None,
        _biomech_analysis=None,
        _gait_cycle=None,
        _foot_contact=None,
        _analysis_compute_active=False,
        _opensim_auto_pipeline_busy=False,
        _pipeline_status_context_cache=None,
        smooth_motion=SimpleNamespace(get=lambda: True),
        _load_rts_report_for_session=lambda: None,
    )

    missing = metrics_stage_missing_dependencies(gui, report)
    assert missing == [
        "Biomechanical analysis result",
        "Gait-cycle or foot-contact analysis",
    ]
    views = build_results_stage_views(gui, report)
    metrics = next(view for view in views if view.key == "metrics")
    assert metrics.status == STATUS_UNAVAILABLE
    assert "missing: Biomechanical analysis result" in metrics.detail
    assert "Gait-cycle or foot-contact analysis" in metrics.detail


def test_session_analysis_apply_runs_metrics_stage() -> None:
    import inspect

    from stablewalk.ui.tk import app as app_mod

    apply_src = inspect.getsource(app_mod.StableWalkGUI._apply_session_analysis_payload)
    ensure_src = inspect.getsource(app_mod.StableWalkGUI._ensure_metrics_stage_summary)
    paint_src = inspect.getsource(app_mod.StableWalkGUI._paint_metrics_panels)
    assert "_ensure_metrics_stage_summary()" in apply_src
    assert "_update_analysis_summary_panel()" in ensure_src
    assert "_ensure_metrics_stage_summary()" in paint_src


def test_summary_kpis_do_not_complete_an_unrun_metrics_stage() -> None:
    report = PipelineStatusReport(
        source="walk.mp4",
        run_name="not-run",
        groups=[],
    )
    summary = SimpleNamespace(
        source="walk.mp4",
        cadence=SimpleNamespace(available=True, value="104"),
        overall_gait_quality=None,
        walking_speed=None,
        symmetry=None,
        stability_margin=None,
        analysis_confidence=None,
        pipeline_confidence=None,
        video_quality=None,
    )
    gui = SimpleNamespace(
        sequence=SimpleNamespace(frames=[object()] * 10),
        _analysis_summary_cache=summary,
        _analysis_compute_active=False,
        _opensim_auto_pipeline_busy=False,
        _pipeline_status_context_cache=None,
        smooth_motion=SimpleNamespace(get=lambda: True),
        _load_rts_report_for_session=lambda: None,
    )

    views = build_results_stage_views(gui, report)
    metrics = next(view for view in views if view.key == "metrics")
    filtering = next(view for view in views if view.key == "filtering")
    video = next(view for view in views if view.key == "video")

    assert metrics.status == STATUS_UNAVAILABLE
    assert metrics.output_text == "1 derived estimates from analysis summary"
    assert "stage not run" in metrics.detail.lower()
    assert filtering.status == STATUS_UNAVAILABLE
    assert "filtering not run" in filtering.output_text.lower()
    assert "pose frames ready" not in video.output_text.lower()


def test_pipeline_refresh_resyncs_report_preview_state() -> None:
    from unittest.mock import MagicMock, patch

    report = PipelineStatusReport(
        source="walk.mp4",
        run_name="refresh",
        groups=[],
    )
    host = MagicMock()
    host.winfo_children.return_value = [object()]
    gui = SimpleNamespace(
        playing=False,
        sequence=None,
        gait_motion=None,
        _analysis_summary_cache=None,
        _analysis_compute_active=False,
        _opensim_auto_pipeline_busy=False,
        _pipeline_status_context_cache=None,
        _results_pipeline_host=host,
        _results_pipeline_summary=None,
        _results_pipeline_wrap_cols=1,
        smooth_motion=SimpleNamespace(get=lambda: True),
        _load_rts_report_for_session=lambda: None,
    )

    with (
        patch(
            "stablewalk.ui.tk.dashboard_summary.sync_results_report_preview"
        ) as sync_preview,
        patch(
            "stablewalk.ui.tk.playback_render_policy.paint_signature_changed",
            return_value=False,
        ),
    ):
        update_results_analysis_pipeline(gui, report)

    sync_preview.assert_called_once_with(gui)
    assert gui._results_pipeline_session_state in ("partial", "waiting")
    assert gui._results_metrics_stage_status == STATUS_UNAVAILABLE


def test_pipeline_refresh_rebinds_summary_without_recursive_pipeline_refresh() -> None:
    from unittest.mock import MagicMock, patch

    report = PipelineStatusReport(source="walk.mp4", run_name="refresh", groups=[])
    host = MagicMock()
    host.winfo_children.return_value = [object()]
    summary = object()
    gui = SimpleNamespace(
        playing=False,
        sequence=None,
        gait_motion=None,
        _analysis_summary_cache=summary,
        _analysis_compute_active=False,
        _opensim_auto_pipeline_busy=False,
        _pipeline_status_context_cache=None,
        _results_pipeline_host=host,
        _results_pipeline_summary=None,
        _results_pipeline_wrap_cols=1,
        smooth_motion=SimpleNamespace(get=lambda: True),
        _load_rts_report_for_session=lambda: None,
    )

    with (
        patch(
            "stablewalk.ui.tk.dashboard_summary.update_results_summary_panel"
        ) as update_summary,
        patch(
            "stablewalk.ui.tk.playback_render_policy.paint_signature_changed",
            return_value=False,
        ),
    ):
        update_results_analysis_pipeline(gui, report)

    update_summary.assert_called_once_with(
        gui,
        summary,
        refresh_pipeline=False,
    )


def test_results_stage_views_resolve_report_when_analysis_exists() -> None:
    """Existing analysis must not leave diagram stages as Pending (unavailable)."""
    report = PipelineStatusReport(
        source="walk.mp4",
        run_name="demo",
        groups=[
            (
                "Pose",
                [
                    _item("pose_mediapipe"),
                    _item("pose_smpl", status=STATUS_UNAVAILABLE),
                    _item("pose_romp", status=STATUS_UNAVAILABLE),
                    _item("motion_skeleton"),
                    _item("motion_angles"),
                    _item("motion_velocities"),
                    _item("bio_contact_mask"),
                    _item("bio_heel_strike"),
                    _item("bio_toe_off"),
                    _item("bio_cadence"),
                    _item("bio_com"),
                    _item("bio_bos"),
                    _item("osim_model", status=STATUS_UNAVAILABLE),
                    _item("osim_trc", status=STATUS_UNAVAILABLE),
                    _item("osim_mot", status=STATUS_UNAVAILABLE),
                    _item("osim_ik", status=STATUS_UNAVAILABLE),
                    _item("osim_mapping", status=STATUS_UNAVAILABLE),
                    _item("osim_coverage", status=STATUS_UNAVAILABLE),
                    _item("retarget_export", status=STATUS_UNAVAILABLE),
                    _item("isaac_export_only", status=STATUS_UNAVAILABLE),
                ],
            )
        ],
    )
    summary = SimpleNamespace(
        source="walk.mp4",
        overall_gait_quality=SimpleNamespace(available=True, value="72/100"),
        cadence=SimpleNamespace(available=True, value="110"),
        walking_speed=SimpleNamespace(available=False, value="—"),
        symmetry=SimpleNamespace(available=False, value="—"),
        stability_margin=SimpleNamespace(available=False, value="—"),
        pipeline_confidence=SimpleNamespace(available=True, value="80%"),
        analysis_confidence=SimpleNamespace(available=True, value="78%"),
        video_quality=SimpleNamespace(available=True, value="85/100"),
    )

    def _assess(*, force: bool = False):
        return report

    gui = SimpleNamespace(
        sequence=SimpleNamespace(frames=[object()] * 10),
        gait_motion=object(),
        _biomech_analysis=object(),
        _analysis_summary_cache=summary,
        _analysis_compute_active=False,
        _opensim_auto_pipeline_busy=False,
        _pipeline_status_context_cache=None,
        _pipeline_status_report_cache=None,
        smooth_motion=SimpleNamespace(get=lambda: True),
        _load_rts_report_for_session=lambda: None,
        _assess_pipeline_status_report=_assess,
    )
    views = build_results_stage_views(gui, None)
    by_key = {v.key: v for v in views}
    assert by_key["video"].status == STATUS_COMPLETED
    assert by_key["pose_estimation"].status == STATUS_COMPLETED
    assert by_key["filtering"].status == STATUS_COMPLETED
    assert by_key["reconstruction_3d"].status == STATUS_COMPLETED
    assert by_key["biomechanics"].status == STATUS_COMPLETED
    assert by_key["metrics"].status == STATUS_COMPLETED
    assert by_key["opensim"].status == STATUS_UNAVAILABLE
    assert by_key["export"].status == STATUS_UNAVAILABLE
    assert by_key["pose_estimation"].confidence_text
    assert by_key["biomechanics"].artifacts_text is not None


def test_results_pipeline_is_responsive_without_horizontal_scroll() -> None:
    from stablewalk.ui.tk import dashboard_results_pipeline as pipe

    source = Path(pipe.__file__).read_text(encoding="utf-8")
    assert "orient=tk.HORIZONTAL" not in source
    assert "xscrollcommand" not in source
    assert "_scale_results_pipeline_cards" in source
    assert pipe._STAGE_CARD_MIN_WIDTH >= 80
    assert pipe._STAGE_CARD_ROW_FLOOR <= 120
    assert pipe._STAGE_CARD_ROW_FLOOR < pipe._STAGE_CARD_MIN_HEIGHT
    assert "minsize=_STAGE_CARD_ROW_FLOOR" in source
    assert "minsize=_STAGE_CARD_MIN_HEIGHT" not in source
    assert "uniform=\"results_stage\"" in source or 'uniform="results_stage"' in source
    assert "results_pipeline_wrap_columns" in source


def test_results_pipeline_wrap_columns() -> None:
    from stablewalk.ui.tk.dashboard_results_pipeline import results_pipeline_wrap_columns

    assert results_pipeline_wrap_columns(1400, 8) == 8
    assert results_pipeline_wrap_columns(1000, 8) == 4
    assert results_pipeline_wrap_columns(600, 8) == 2
    assert results_pipeline_wrap_columns(0, 8, previous=4) == 4
    # Tightening wrap is immediate; expanding waits for hysteresis.
    assert results_pipeline_wrap_columns(1099, 8, previous=8) == 4
    assert results_pipeline_wrap_columns(1100, 8, previous=4) == 4
    assert results_pipeline_wrap_columns(1140, 8, previous=4) == 8


def test_results_pipeline_skips_rebuild_when_signature_unchanged() -> None:
    """Identical stage views must not destroy/rebuild Results cards."""

    class _Child:
        def destroy(self) -> None:
            raise AssertionError("pipeline host must not rebuild when signature matches")

    host = SimpleNamespace(winfo_children=lambda: [_Child()])
    gui = SimpleNamespace(
        playing=False,
        _results_pipeline_host=host,
        _results_pipeline_summary=None,
        sequence=None,
        _analysis_summary_cache=None,
        _analysis_compute_active=False,
        _opensim_auto_pipeline_busy=False,
        _pipeline_status_context_cache=None,
        smooth_motion=SimpleNamespace(get=lambda: True),
        _load_rts_report_for_session=lambda: None,
        _paint_signatures={},
        _results_pipeline_wrap_cols=8,
    )
    views = build_results_stage_views(gui, None)
    gui._paint_signatures["results_pipeline"] = (
        _results_pipeline_view_signature(views),
        8,
    )
    update_results_analysis_pipeline(gui, None)
