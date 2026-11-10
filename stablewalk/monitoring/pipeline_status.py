"""
Honest Real-to-Sim pipeline stage status for GUI monitoring and JSON export.

Statuses are grounded in executed analysis and on-disk artifacts — never marked
completed unless the stage actually ran.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from stablewalk import config
from stablewalk.analysis.biomechanical.orchestrator import BiomechanicalAnalysisResult
from stablewalk.analysis.estimated_vgrf_analysis import EstimatedVGRFResult
from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult
from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult
from stablewalk.analysis.pipeline_validation import (
    QUALITY_LABEL,
    PipelineValidationReport,
    assess_pipeline_validation,
)
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.ui.scientific_labels import LABEL_VIRTUAL_GRF_PANEL

PipelineStageState = Literal["completed", "partial", "unavailable"]
ValidationQualityLabel = Literal[
    "not_assessed",
    "insufficient_data",
    "poor",
    "limited",
    "acceptable",
]

STATUS_COMPLETED = "completed"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"

STATUS_SYMBOL = {
    STATUS_COMPLETED: "",
    STATUS_PARTIAL: "",
    STATUS_UNAVAILABLE: "",
}

STATUS_LABEL = {
    STATUS_COMPLETED: "Completed",
    STATUS_PARTIAL: "Partial",
    STATUS_UNAVAILABLE: "Missing",
}


@dataclass(frozen=True)
class PipelineStageItem:
    """One monitored pipeline stage."""

    key: str
    label: str
    group: str
    status: PipelineStageState
    detail: str
    tooltip: str
    quality: ValidationQualityLabel = "not_assessed"
    confidence: float | None = None
    warnings: tuple[str, ...] = ()
    validation_note: str = ""

    def display_line(self) -> str:
        return STATUS_LABEL[self.status]

    def quality_label(self) -> str:
        return QUALITY_LABEL.get(self.quality, self.quality)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "status_symbol": STATUS_SYMBOL[self.status],
            "status_label": STATUS_LABEL[self.status],
            "quality_label": self.quality_label(),
            "warnings": list(self.warnings),
        }


@dataclass
class PipelineStatusReport:
    """Full pipeline monitoring snapshot for the current session."""

    source: str = ""
    run_name: str = ""
    groups: list[tuple[str, list[PipelineStageItem]]] = field(default_factory=list)
    validation: PipelineValidationReport | None = None
    schema_version: str = "1.1"

    def all_items(self) -> list[PipelineStageItem]:
        out: list[PipelineStageItem] = []
        for _title, items in self.groups:
            out.extend(items)
        return out

    def diagram_stages(self) -> list[PipelineDiagramStage]:
        """Top-level pipeline diagram stages derived from monitored sub-items."""
        return build_pipeline_diagram(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "pipeline_status",
            "source": self.source,
            "run_name": self.run_name,
            "diagram": [stage.to_dict() for stage in self.diagram_stages()],
            "groups": [
                {
                    "title": title,
                    "stages": [item.to_dict() for item in items],
                }
                for title, items in self.groups
            ],
            "validation": None if self.validation is None else self.validation.to_dict(),
        }


@dataclass(frozen=True)
class PipelineDiagramStage:
    """One stage in the vertical Real-to-Sim pipeline diagram."""

    key: str
    label: str
    status: PipelineStageState
    detail: str
    tooltip: str
    item_keys: tuple[str, ...] = ()

    def display_line(self) -> str:
        return STATUS_LABEL[self.status]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "status_symbol": STATUS_SYMBOL[self.status],
            "status_label": STATUS_LABEL[self.status],
            "item_keys": list(self.item_keys),
        }


PIPELINE_DIAGRAM_STAGE_SPECS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "pose_estimation",
        "Pose Estimation",
        ("pose_mediapipe", "pose_smpl", "pose_romp"),
        (
            "Extract 2D/3D body pose from video using MediaPipe, SMPL, or ROMP. "
            "This is the first step — later stages depend on detected landmarks."
        ),
    ),
    (
        "reconstruction_3d",
        "3D Reconstruction",
        ("motion_skeleton", "motion_angles", "motion_velocities"),
        (
            "Build a hip-centered 3D skeleton, joint angles, and kinematic velocities "
            "from pose landmarks across the clip."
        ),
    ),
    (
        "biomechanics",
        "Biomechanics",
        (
            "bio_contact_mask",
            "bio_heel_strike",
            "bio_toe_off",
            "bio_cadence",
            "bio_com",
            "bio_bos",
        ),
        (
            "Gait event detection, foot contact, cadence, center of mass, and base of "
            "support — estimated from pose, not force-plate measured."
        ),
    ),
    (
        "opensim",
        "OpenSim",
        (
            "osim_model",
            "osim_trc",
            "osim_mot",
            "osim_ik",
            "osim_mapping",
            "osim_coverage",
        ),
        (
            "Map landmarks to OpenSim markers, run inverse kinematics, and export "
            "musculoskeletal motion files (.trc / .mot)."
        ),
    ),
    (
        "retargeting",
        "Retargeting",
        ("retarget_scale", "retarget_joints"),
        (
            "Scale human proportions and map joint angles to a target humanoid robot "
            "DOF layout for simulation."
        ),
    ),
    (
        "virtual_grf",
        LABEL_VIRTUAL_GRF_PANEL,
        ("bio_vgrf",),
        (
            "Pose-based estimated vertical ground reaction force — a virtual proxy, "
            "not measured GRF from force plates."
        ),
    ),
    (
        "export",
        "Export",
        ("retarget_export", "isaac_export_only", "osim_ik"),
        (
            "Write motion artifacts for downstream use: stablewalk_motion.npz, "
            "retargeted_motion.npz, AMP reference, and OpenSim exports."
        ),
    ),
    (
        "isaac_lab",
        "Isaac Lab",
        ("isaac_env_ready", "isaac_training_running", "isaac_training_complete"),
        (
            "NVIDIA Isaac Lab simulation environment and optional RL/AMP training. "
            "Training runs outside StableWalk; export-only mode is supported."
        ),
    ),
)


def _combine_stage_details(items: list[PipelineStageItem], *, max_parts: int = 2) -> str:
    if not items:
        return "Not executed in this session"
    parts = [item.detail for item in items[:max_parts] if item.detail]
    if len(items) > max_parts:
        parts.append(f"+{len(items) - max_parts} more")
    return " · ".join(parts) if parts else "Not executed in this session"


def _opensim_diagram_detail(
    items_by_key: dict[str, PipelineStageItem],
    *,
    status: str,
) -> str:
    """Short OpenSim card line — explain Partial / Missing clearly."""
    order = (
        "osim_model",
        "osim_trc",
        "osim_mot",
        "osim_ik",
        "osim_mapping",
        "osim_coverage",
    )
    incomplete = [
        items_by_key[k]
        for k in order
        if k in items_by_key and items_by_key[k].status != STATUS_COMPLETED
    ]
    if status == STATUS_PARTIAL:
        ik = items_by_key.get("osim_ik")
        trc = items_by_key.get("osim_trc")
        mot = items_by_key.get("osim_mot")
        export_done = bool(
            trc is not None
            and mot is not None
            and trc.status == STATUS_COMPLETED
            and mot.status == STATUS_COMPLETED
        )
        if (
            ik is not None
            and ik.status != STATUS_COMPLETED
            and export_done
        ):
            detail = str(ik.detail or "").strip().lower()
            if str(ik.status) == "failed" or "fail" in detail:
                ik_label = "Failed"
            elif "running" in detail:
                ik_label = "Running"
            else:
                ik_label = "unavailable"
            extra_detail = str(ik.detail or "").strip()
            label = f"Export Complete · IK {ik_label}"
            if extra_detail and extra_detail.lower() not in label.lower():
                return f"{label}. {extra_detail}"
            return label
        reasons = [it.detail for it in incomplete[:2] if it.detail]
        if reasons:
            return "Partial — " + " · ".join(reasons) + ". Use Export OpenSim Files to finish."
        return (
            "Partial — SDK/model or TRC/MOT export incomplete. "
            "Use Export OpenSim Files (or load a .osim model) to complete."
        )
    if status == STATUS_UNAVAILABLE:
        return (
            "Missing — OpenSim was not configured for this session "
            "(no SDK/model and no export artifacts)."
        )
    parts: list[str] = []
    for key in order:
        item = items_by_key.get(key)
        if item is None or item.status == STATUS_UNAVAILABLE:
            continue
        label = (item.label or key).replace("OpenSim ", "")
        parts.append(label)
        if len(parts) >= 3:
            break
    if not parts:
        return "OpenSim not assessed"
    return " · ".join(parts)


def _retargeting_diagram_detail(
    items_by_key: dict[str, PipelineStageItem],
    *,
    status: str,
) -> str:
    """Explain why Retargeting is Missing / Partial."""
    if status == STATUS_UNAVAILABLE:
        return (
            "Missing — Real-to-Sim retargeting has not been run. "
            "No retargeted_motion.npz. Use Real-to-Sim Pipeline after analysis."
        )
    if status == STATUS_PARTIAL:
        bits = []
        for key in ("retarget_scale", "retarget_joints", "retarget_export"):
            item = items_by_key.get(key)
            if item is not None and item.status != STATUS_COMPLETED and item.detail:
                bits.append(item.detail)
        if bits:
            return "Partial — " + " · ".join(bits[:2])
        return "Partial — motion reference exists; retarget export still pending."
    _, text = _aggregate_stage_status(
        items_by_key, ("retarget_scale", "retarget_joints"), detail_max_parts=2
    )
    return text


def _aggregate_stage_status(
    items_by_key: dict[str, PipelineStageItem],
    keys: tuple[str, ...],
    *,
    detail_max_parts: int = 2,
) -> tuple[PipelineStageState, str]:
    """Roll up sub-item statuses — never mark completed unless executed steps completed."""
    matched = [items_by_key[k] for k in keys if k in items_by_key]
    if not matched:
        return STATUS_UNAVAILABLE, "Stage not assessed"

    active = [item for item in matched if item.status != STATUS_UNAVAILABLE]
    if not active:
        return STATUS_UNAVAILABLE, _combine_stage_details(matched, max_parts=detail_max_parts)

    statuses = [item.status for item in active]
    if all(status == STATUS_COMPLETED for status in statuses):
        return STATUS_COMPLETED, _combine_stage_details(active, max_parts=detail_max_parts)

    if any(status == STATUS_COMPLETED for status in statuses):
        incomplete = [item for item in active if item.status != STATUS_COMPLETED]
        return STATUS_PARTIAL, _combine_stage_details(
            incomplete or active, max_parts=detail_max_parts
        )

    if any(status == STATUS_PARTIAL for status in statuses):
        return STATUS_PARTIAL, _combine_stage_details(active, max_parts=detail_max_parts)

    return STATUS_UNAVAILABLE, _combine_stage_details(matched, max_parts=detail_max_parts)


def build_pipeline_diagram(report: PipelineStatusReport) -> list[PipelineDiagramStage]:
    """Build ordered diagram stages from a :class:`PipelineStatusReport`."""
    items_by_key = {item.key: item for item in report.all_items()}
    stages: list[PipelineDiagramStage] = []
    for key, label, item_keys, tooltip in PIPELINE_DIAGRAM_STAGE_SPECS:
        status, detail = _aggregate_stage_status(
            items_by_key, item_keys, detail_max_parts=6 if key == "opensim" else 2
        )
        if key == "opensim":
            detail = _opensim_diagram_detail(items_by_key, status=status)
        elif key == "retargeting":
            detail = _retargeting_diagram_detail(items_by_key, status=status)
        stages.append(
            PipelineDiagramStage(
                key=key,
                label=label,
                status=status,
                detail=detail,
                tooltip=tooltip,
                item_keys=item_keys,
            )
        )
    return stages


@dataclass
class PipelineStatusContext:
    """Session inputs for pipeline status assessment."""

    source: str = ""
    run_name: str | None = None
    pose_backend_requested: str = "mediapipe"
    pose_backend_used: str = "mediapipe"
    pose_backend_fallback: bool = False
    pose_backend_provider: str = ""
    detected_frames: int = 0
    total_frames: int = 0
    smpl_motion_path: Path | None = None
    smpl_assets_ready: bool = False
    romp_importable: bool = False
    gait_motion: GaitMotionRecording | None = None
    cycles: GaitCycleAnalysisResult | None = None
    contact: FootContactAnalysisResult | None = None
    biomech: BiomechanicalAnalysisResult | None = None
    estimated_vgrf: EstimatedVGRFResult | None = None
    opensim_sdk_available: bool = False
    opensim_model_loaded: bool = False
    opensim_model_name: str | None = None
    opensim_ik_completed: bool = False
    opensim_ik_running: bool = False
    opensim_ik_state: str | None = None
    opensim_ik_block_reason: str | None = None
    opensim_mapping_status: str | None = None
    opensim_mapping_percent: float | None = None
    opensim_coverage_percent: float | None = None
    opensim_export_files: dict[str, bool] | None = None
    rts_report: dict[str, Any] | None = None
    isaac_lab_available: bool | None = None
    isaac_lab_note: str = ""


def _item(
    key: str,
    label: str,
    group: str,
    status: PipelineStageState,
    detail: str,
    tooltip: str,
) -> PipelineStageItem:
    return PipelineStageItem(
        key=key,
        label=label,
        group=group,
        status=status,
        detail=detail,
        tooltip=tooltip,
    )


def _ratio(detected: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return detected / total


def _motion_has_angles(recording: GaitMotionRecording | None) -> tuple[bool, bool]:
    if recording is None or not recording.snapshots:
        return False, False
    angle_frames = 0
    velocity_frames = 0
    for snap in recording.snapshots:
        if any(js.angle_deg is not None for js in snap.joints.values()):
            angle_frames += 1
        if any(
            js.velocity is not None or js.velocity_vector is not None
            for js in snap.joints.values()
        ):
            velocity_frames += 1
        if snap.dofs and any(d.angle_deg is not None for d in snap.dofs.values()):
            angle_frames += 1
        if snap.dofs and any(d.velocity_deg_s is not None for d in snap.dofs.values()):
            velocity_frames += 1
    n = len(recording.snapshots)
    return angle_frames >= n * 0.5, velocity_frames >= n * 0.5


def _contact_has_events(
    contact: FootContactAnalysisResult | None,
    *,
    heel: bool,
    toe: bool,
) -> bool:
    if contact is None or not contact.per_frame:
        return False
    for f in contact.per_frame:
        if heel and (f.left_heel_strike or f.right_heel_strike):
            return True
        if toe and (f.left_toe_off or f.right_toe_off):
            return True
    return False


def _rts_stage_status(report: dict[str, Any] | None, stage_id: str) -> str | None:
    if not report:
        return None
    for stage in report.get("stages") or []:
        if stage.get("stage") == stage_id:
            return str(stage.get("status") or "")
    return None


def _artifact_exists(run_name: str | None, filename: str) -> bool:
    if not run_name:
        return False
    path = config.MOTION_REFERENCE_EXPORT_DIR / run_name / filename
    return path.is_file()


def _apply_stage_validation(
    report: PipelineStatusReport,
    ctx: PipelineStatusContext,
) -> PipelineStatusReport:
    """Attach per-stage quality/confidence/warnings from available artifacts only."""
    validation = assess_pipeline_validation(ctx)
    report.validation = validation
    enriched: list[tuple[str, list[PipelineStageItem]]] = []
    for title, items in report.groups:
        new_items: list[PipelineStageItem] = []
        for item in items:
            sv = validation.stages.get(item.key)
            if sv is None:
                new_items.append(item)
                continue
            new_items.append(
                replace(
                    item,
                    quality=sv.quality,  # type: ignore[arg-type]
                    confidence=sv.confidence,
                    warnings=sv.warnings,
                    validation_note=sv.summary,
                )
            )
        enriched.append((title, new_items))
    report.groups = enriched
    return report


def assess_pipeline_status(ctx: PipelineStatusContext) -> PipelineStatusReport:
    """Build an honest pipeline status report from the current session context."""
    run_name = ctx.run_name or ""
    det_ratio = _ratio(ctx.detected_frames, ctx.total_frames)
    has_session = ctx.total_frames > 0 or ctx.gait_motion is not None

    groups: list[tuple[str, list[PipelineStageItem]]] = []

    # --- Pose Extraction ---
    pose_items: list[PipelineStageItem] = []
    mp_used = ctx.pose_backend_used == "mediapipe"
    smpl_used = ctx.pose_backend_used == "smpl"
    provider = (ctx.pose_backend_provider or "").lower()

    if not has_session:
        mp_status: PipelineStageState = STATUS_UNAVAILABLE
        mp_detail = "No video analyzed yet"
    elif mp_used and det_ratio >= 0.25:
        mp_status = STATUS_COMPLETED if det_ratio >= 0.5 else STATUS_PARTIAL
        mp_detail = f"MediaPipe poses ({ctx.detected_frames}/{ctx.total_frames} frames)"
    elif mp_used:
        mp_status = STATUS_PARTIAL
        mp_detail = f"Low detection rate ({ctx.detected_frames} frames)"
    elif ctx.pose_backend_fallback and det_ratio >= 0.25:
        mp_status = STATUS_PARTIAL
        mp_detail = "Fallback poses from MediaPipe after SMPL request failed"
    else:
        mp_status = STATUS_UNAVAILABLE
        mp_detail = "MediaPipe not used for this session"

    pose_items.append(
        _item(
            "pose_mediapipe",
            "MediaPipe",
            "Pose Extraction",
            mp_status,
            mp_detail,
            "2D/3D landmark pose extraction via MediaPipe Pose (default tracker).",
        )
    )

    smpl_path = ctx.smpl_motion_path
    if smpl_path is None and run_name:
        candidate = config.MOTION_REFERENCE_EXPORT_DIR / run_name / "smpl_motion.npz"
        if candidate.is_file():
            smpl_path = candidate

    if smpl_used and smpl_path is not None and smpl_path.is_file():
        smpl_status: PipelineStageState = STATUS_COMPLETED
        smpl_detail = f"SMPL motion exported ({smpl_path.name})"
    elif ctx.smpl_assets_ready and not smpl_used:
        smpl_status = STATUS_PARTIAL
        smpl_detail = "SMPL assets ready but inference not used for this clip"
    elif smpl_used:
        smpl_status = STATUS_PARTIAL
        smpl_detail = "SMPL backend ran without smpl_motion.npz export"
    else:
        smpl_status = STATUS_UNAVAILABLE
        smpl_detail = "SMPL inference not executed"

    pose_items.append(
        _item(
            "pose_smpl",
            "SMPL",
            "Pose Extraction",
            smpl_status,
            smpl_detail,
            "SMPL body model pose and shape estimation (requires licensed model files).",
        )
    )

    if smpl_used and provider == "romp":
        romp_status: PipelineStageState = STATUS_COMPLETED
        romp_detail = "ROMP provider executed SMPL inference"
    elif ctx.romp_importable and ctx.smpl_assets_ready:
        romp_status = STATUS_PARTIAL
        romp_detail = "ROMP importable but not used for this session"
    elif ctx.romp_importable:
        romp_status = STATUS_PARTIAL
        romp_detail = "ROMP available; SMPL assets incomplete"
    else:
        romp_status = STATUS_UNAVAILABLE
        romp_detail = "ROMP provider not available"

    pose_items.append(
        _item(
            "pose_romp",
            "ROMP",
            "Pose Extraction",
            romp_status,
            romp_detail,
            "ROMP monocular mesh recovery used as the SMPL inference provider.",
        )
    )
    groups.append(("Pose Extraction", pose_items))

    # --- Motion Reconstruction ---
    motion_items: list[PipelineStageItem] = []
    snap_count = len(ctx.gait_motion.snapshots) if ctx.gait_motion else 0
    angles_ok, velocities_ok = _motion_has_angles(ctx.gait_motion)

    if snap_count >= 3 and det_ratio >= 0.25:
        skel_status: PipelineStageState = STATUS_COMPLETED if det_ratio >= 0.5 else STATUS_PARTIAL
        skel_detail = f"{snap_count} skeleton snapshots"
    elif snap_count > 0:
        skel_status = STATUS_PARTIAL
        skel_detail = f"Sparse skeleton ({snap_count} snapshots)"
    else:
        skel_status = STATUS_UNAVAILABLE
        skel_detail = "3D skeleton not built"

    motion_items.append(
        _item(
            "motion_skeleton",
            "3D Skeleton",
            "Motion Reconstruction",
            skel_status,
            skel_detail,
            "Hip-centered canonical 3D skeleton reconstructed from pose landmarks.",
        )
    )

    if angles_ok:
        ang_status: PipelineStageState = STATUS_COMPLETED
        ang_detail = "Joint angles populated across frames"
    elif snap_count > 0:
        ang_status = STATUS_PARTIAL
        ang_detail = "Joint angles missing on many frames"
    else:
        ang_status = STATUS_UNAVAILABLE
        ang_detail = "Joint angles not computed"

    motion_items.append(
        _item(
            "motion_angles",
            "Joint Angles",
            "Motion Reconstruction",
            ang_status,
            ang_detail,
            "Segment and DOF joint angles derived from pose and enrichment pipeline.",
        )
    )

    if velocities_ok:
        vel_status: PipelineStageState = STATUS_COMPLETED
        vel_detail = "Kinematic velocities attached to snapshots"
    elif snap_count > 0:
        vel_status = STATUS_PARTIAL
        vel_detail = "Velocities sparse or incomplete"
    else:
        vel_status = STATUS_UNAVAILABLE
        vel_detail = "Joint velocities not computed"

    motion_items.append(
        _item(
            "motion_velocities",
            "Joint Velocities",
            "Motion Reconstruction",
            vel_status,
            vel_detail,
            "Finite-difference joint and landmark velocities using frame timestamps.",
        )
    )
    groups.append(("Motion Reconstruction", motion_items))

    # --- Biomechanics ---
    bio_items: list[PipelineStageItem] = []
    contact = ctx.contact
    cycles = ctx.cycles
    bio = ctx.biomech

    if contact and contact.per_frame and contact.metrics.contact_confidence > 0.1:
        cm_status: PipelineStageState = (
            STATUS_COMPLETED if contact.metrics.contact_confidence >= 0.5 else STATUS_PARTIAL
        )
        cm_detail = f"{len(contact.per_frame)} contact frames"
    else:
        cm_status = STATUS_UNAVAILABLE
        cm_detail = "Foot contact mask not computed"

    bio_items.append(
        _item(
            "bio_contact_mask",
            "Contact Mask",
            "Biomechanics",
            cm_status,
            cm_detail,
            "Per-frame left/right foot contact probabilities and binary masks.",
        )
    )

    hs = _contact_has_events(contact, heel=True, toe=False)
    bio_items.append(
        _item(
            "bio_heel_strike",
            "Heel Strike",
            "Biomechanics",
            STATUS_COMPLETED if hs else STATUS_UNAVAILABLE,
            "Heel-strike events detected" if hs else "No heel-strike events detected",
            "Heel-strike gait events from foot clearance and contact hysteresis.",
        )
    )

    to = _contact_has_events(contact, heel=False, toe=True)
    bio_items.append(
        _item(
            "bio_toe_off",
            "Toe Off",
            "Biomechanics",
            STATUS_COMPLETED if to else STATUS_UNAVAILABLE,
            "Toe-off events detected" if to else "No toe-off events detected",
            "Toe-off gait events marking swing-phase onset.",
        )
    )

    cadence = None
    if cycles and cycles.metrics.cadence_steps_per_min is not None:
        cadence = cycles.metrics.cadence_steps_per_min
    elif contact and contact.metrics.cadence_steps_per_min is not None:
        cadence = contact.metrics.cadence_steps_per_min

    if cadence is not None and cadence > 0:
        cad_status: PipelineStageState = STATUS_COMPLETED
        cad_detail = f"{cadence:.0f} steps/min"
    elif contact and contact.per_frame:
        cad_status = STATUS_PARTIAL
        cad_detail = "Insufficient heel-strike intervals for cadence"
    else:
        cad_status = STATUS_UNAVAILABLE
        cad_detail = "Cadence not computed"

    bio_items.append(
        _item(
            "bio_cadence",
            "Cadence",
            "Biomechanics",
            cad_status,
            cad_detail,
            "Steps per minute from inter-heel-strike timing.",
        )
    )

    com_ok = bio is not None and bio.center_of_mass and bio.center_of_mass.per_frame
    bio_items.append(
        _item(
            "bio_com",
            "COM",
            "Biomechanics",
            STATUS_COMPLETED if com_ok else STATUS_UNAVAILABLE,
            "Center of mass trajectory estimated" if com_ok else "COM not estimated",
            "Segment-weighted center of mass (estimated, not force-plate measured).",
        )
    )

    bos_ok = bio is not None and bio.base_of_support and bio.base_of_support.per_frame
    bio_items.append(
        _item(
            "bio_bos",
            "BoS",
            "Biomechanics",
            STATUS_COMPLETED if bos_ok else STATUS_UNAVAILABLE,
            "Base of support polygons computed" if bos_ok else "BoS not computed",
            "Horizontal support polygons from stance-foot landmarks.",
        )
    )

    vgrf = ctx.estimated_vgrf
    if vgrf is not None and vgrf.available and len(vgrf.timestamps) > 0:
        vgrf_status: PipelineStageState = STATUS_COMPLETED
        vgrf_detail = f"{len(vgrf.timestamps)} samples (estimated)"
    elif vgrf is not None and contact and contact.per_frame:
        vgrf_status = STATUS_PARTIAL
        vgrf_detail = "vGRF estimator ran with limited confidence"
    else:
        vgrf_status = STATUS_UNAVAILABLE
        vgrf_detail = "Estimated virtual GRF not available"

    bio_items.append(
        _item(
            "bio_vgrf",
            "Estimated Virtual GRF",
            "Biomechanics",
            vgrf_status,
            vgrf_detail,
            "Pose-based vertical force proxy — not measured ground reaction force.",
        )
    )
    groups.append(("Biomechanics", bio_items))

    # --- OpenSim ---
    osim_items: list[PipelineStageItem] = []
    files = ctx.opensim_export_files  # None = no session assessed
    files_dict = files or {}
    has_export_session = files is not None

    if ctx.opensim_model_loaded:
        model_status: PipelineStageState = STATUS_COMPLETED
        model_name = (ctx.opensim_model_name or "").strip()
        model_detail = f"Model loaded ({model_name})" if model_name else "Model loaded"
    elif ctx.opensim_sdk_available:
        model_status = STATUS_PARTIAL
        model_detail = "SDK available; no usable .osim model loaded"
    else:
        model_status = STATUS_UNAVAILABLE
        model_detail = "OpenSim SDK not installed"

    osim_items.append(
        _item(
            "osim_model",
            "Model Loaded",
            "OpenSim",
            model_status,
            model_detail,
            "Configured OpenSim musculoskeletal model for IK and ID.",
        )
    )

    if files_dict.get("TRC"):
        trc_status: PipelineStageState = STATUS_COMPLETED
        trc_detail = "TRC exported"
    elif has_export_session:
        trc_status = STATUS_PARTIAL
        trc_detail = "TRC not exported — run Export OpenSim Files"
    else:
        trc_status = STATUS_UNAVAILABLE
        trc_detail = "TRC not exported (no session)"

    osim_items.append(
        _item(
            "osim_trc",
            "TRC Export",
            "OpenSim",
            trc_status,
            trc_detail,
            "Marker trajectory file (.trc) for OpenSim IK.",
        )
    )

    if files_dict.get("MOT"):
        mot_status: PipelineStageState = STATUS_COMPLETED
        mot_detail = "MOT exported"
    elif has_export_session:
        mot_status = STATUS_PARTIAL
        mot_detail = "MOT not exported — run Export OpenSim Files"
    else:
        mot_status = STATUS_UNAVAILABLE
        mot_detail = "MOT not exported (no session)"

    osim_items.append(
        _item(
            "osim_mot",
            "MOT Export",
            "OpenSim",
            mot_status,
            mot_detail,
            "Joint-angle motion file (.mot / .csv) from pose export.",
        )
    )

    from stablewalk.opensim_ik_status import resolve_opensim_ik_status_from_context

    ik = resolve_opensim_ik_status_from_context(ctx)
    osim_items.append(
        _item(
            "osim_ik",
            "Inverse Kinematics",
            "OpenSim",
            ik.pipeline_status,
            ik.detail,
            ik.tooltip,
        )
    )

    mapping = (ctx.opensim_mapping_status or "").lower()
    map_pct = ctx.opensim_mapping_percent
    if map_pct is not None:
        map_pct_txt = f"{map_pct:.0f}%"
    else:
        map_pct_txt = None

    if mapping == "improved" and map_pct is not None and map_pct >= 85.0:
        map_status: PipelineStageState = STATUS_COMPLETED
        map_detail = f"Marker mapping {map_pct_txt}"
    elif mapping in ("improved", "partial", "experimental") or map_pct is not None:
        map_status = STATUS_PARTIAL
        if map_pct_txt and mapping:
            map_detail = f"Marker mapping {map_pct_txt} ({mapping})"
        elif map_pct_txt:
            map_detail = f"Marker mapping {map_pct_txt}"
        else:
            map_detail = f"Marker mapping: {mapping or 'incomplete'}"
    elif files_dict.get("TRC"):
        map_status = STATUS_PARTIAL
        map_detail = "TRC exported; marker mapping not validated"
    else:
        map_status = STATUS_UNAVAILABLE
        map_detail = "Marker mapping not performed"

    osim_items.append(
        _item(
            "osim_mapping",
            "Marker Mapping",
            "OpenSim",
            map_status,
            map_detail,
            "MediaPipe TRC markers mapped to OpenSim Gait2392 marker names.",
        )
    )

    cov = ctx.opensim_coverage_percent
    if cov is not None and cov >= 85.0 and mapping == "improved":
        cov_status: PipelineStageState = STATUS_COMPLETED
        cov_detail = f"Coverage {cov:.0f}%"
    elif cov is not None:
        cov_status = STATUS_PARTIAL
        cov_detail = f"Coverage {cov:.0f}%"
    elif files_dict.get("TRC"):
        cov_status = STATUS_PARTIAL
        cov_detail = "Coverage not assessed"
    else:
        cov_status = STATUS_UNAVAILABLE
        cov_detail = "No coverage data"

    osim_items.append(
        _item(
            "osim_coverage",
            "Coverage",
            "OpenSim",
            cov_status,
            cov_detail,
            "Fraction of required model markers present in mapped TRC.",
        )
    )
    groups.append(("OpenSim", osim_items))

    # --- Retargeting ---
    retarget_items: list[PipelineStageItem] = []
    rts = ctx.rts_report
    retarget_npz = _artifact_exists(run_name, "retargeted_motion.npz")
    motion_npz = _artifact_exists(run_name, "stablewalk_motion.npz")
    stage2 = _rts_stage_status(rts, "2_retargeting")

    if retarget_npz and stage2 == "complete":
        scale_status: PipelineStageState = STATUS_COMPLETED
        scale_detail = "Uniform body-scale retargeting applied"
    elif motion_npz:
        scale_status = STATUS_PARTIAL
        scale_detail = "Motion reference exported; retargeting not run"
    else:
        scale_status = STATUS_UNAVAILABLE
        scale_detail = "Retargeting scale step not executed"

    retarget_items.append(
        _item(
            "retarget_scale",
            "Scale Only",
            "Retargeting",
            scale_status,
            scale_detail,
            "Uniform scale from estimated leg length to humanoid proportions.",
        )
    )

    if retarget_npz:
        jmap_status: PipelineStageState = STATUS_COMPLETED
        jmap_detail = "Joint map stored in retargeted_motion.npz"
    elif stage2 == "complete":
        jmap_status = STATUS_PARTIAL
        jmap_detail = "Retargeting reported complete; NPZ missing"
    else:
        jmap_status = STATUS_UNAVAILABLE
        jmap_detail = "Human-to-robot joint mapping not exported"

    retarget_items.append(
        _item(
            "retarget_joints",
            "Joint Mapping",
            "Retargeting",
            jmap_status,
            jmap_detail,
            "Human joint angles mapped to target humanoid DOF order.",
        )
    )

    if retarget_npz and motion_npz:
        exp_status: PipelineStageState = STATUS_COMPLETED
        exp_detail = "Retargeting export ready for simulation"
    elif motion_npz:
        exp_status = STATUS_PARTIAL
        exp_detail = "Motion reference only — retarget export pending"
    else:
        exp_status = STATUS_UNAVAILABLE
        exp_detail = "Export artifacts not ready"

    retarget_items.append(
        _item(
            "retarget_export",
            "Export Ready",
            "Retargeting",
            exp_status,
            exp_detail,
            "retargeted_motion.npz ready for Isaac Lab / AMP pipelines.",
        )
    )
    groups.append(("Retargeting", retarget_items))

    # --- Isaac Lab ---
    isaac_items: list[PipelineStageItem] = []
    isaac_checked = ctx.isaac_lab_available is not None
    isaac_ok = bool(ctx.isaac_lab_available)
    amp_npz = _artifact_exists(run_name, "amp_reference_motion.npz")
    stage3 = _rts_stage_status(rts, "3_simulation_amp")

    if isaac_checked and not isaac_ok:
        ni_status: PipelineStageState = STATUS_COMPLETED
        ni_detail = "Verified: Isaac Lab not installed"
    elif isaac_checked and isaac_ok:
        ni_status = STATUS_UNAVAILABLE
        ni_detail = "Isaac Lab is installed"
    else:
        ni_status = STATUS_UNAVAILABLE
        ni_detail = "Installation status not checked"

    isaac_items.append(
        _item(
            "isaac_not_installed",
            "Not Installed",
            "Isaac Lab",
            ni_status,
            ni_detail,
            "Whether Isaac Lab import check confirmed the package is absent.",
        )
    )

    if amp_npz:
        eo_status: PipelineStageState = STATUS_COMPLETED
        eo_detail = "AMP reference motion exported (offline use)"
    elif stage3 == "partial":
        eo_status = STATUS_PARTIAL
        eo_detail = "AMP export attempted without full Isaac environment"
    else:
        eo_status = STATUS_UNAVAILABLE
        eo_detail = "No AMP export artifact"

    isaac_items.append(
        _item(
            "isaac_export_only",
            "Export Only",
            "Isaac Lab",
            eo_status,
            eo_detail,
            "Motion reference exported for external Isaac Lab / AMP training.",
        )
    )

    if isaac_ok:
        er_status: PipelineStageState = STATUS_COMPLETED
        er_detail = ctx.isaac_lab_note or "Isaac Lab import succeeded"
    else:
        er_status = STATUS_UNAVAILABLE
        er_detail = "Isaac Lab environment not ready"

    isaac_items.append(
        _item(
            "isaac_env_ready",
            "Environment Ready",
            "Isaac Lab",
            er_status,
            er_detail,
            "Python can import isaaclab (does not imply validated training setup).",
        )
    )

    isaac_items.append(
        _item(
            "isaac_training_running",
            "Training Running",
            "Isaac Lab",
            STATUS_UNAVAILABLE,
            "Training is not launched from StableWalk",
            "RL/AMP training runs outside StableWalk; no in-app runner is connected.",
        )
    )
    isaac_items.append(
        _item(
            "isaac_training_complete",
            "Training Complete",
            "Isaac Lab",
            STATUS_UNAVAILABLE,
            "No training completion signal in StableWalk",
            "Training completion must be confirmed in your external Isaac Lab workflow.",
        )
    )
    groups.append(("Isaac Lab", isaac_items))

    report = PipelineStatusReport(
        source=ctx.source,
        run_name=run_name,
        groups=groups,
    )
    return _apply_stage_validation(report, ctx)


__all__ = [
    "PipelineDiagramStage",
    "PipelineStageItem",
    "PipelineStageState",
    "PipelineStatusContext",
    "PipelineStatusReport",
    "PIPELINE_DIAGRAM_STAGE_SPECS",
    "STATUS_COMPLETED",
    "STATUS_PARTIAL",
    "STATUS_UNAVAILABLE",
    "STATUS_LABEL",
    "STATUS_SYMBOL",
    "QUALITY_LABEL",
    "assess_pipeline_status",
    "build_pipeline_diagram",
]
