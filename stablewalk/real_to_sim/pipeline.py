"""
Orchestrate the 4-stage Real-to-Sim pipeline (offline, no Isaac Lab required).

Stages:
  1. Perception — gait style fingerprint from video pose
  2. Retargeting — scale motion to humanoid morphology
  3. Simulation prep — AMP reference export for Isaac Lab
  4. Physics — contact-sync reward vs estimated virtual GRF
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult, analyze_gait_cycles
from stablewalk.analysis.isaac_lab_integration import check_isaac_lab_available
from stablewalk.io.motion_reference_export import export_motion_reference_npz
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.models.pose_data import PoseSequence
from stablewalk.real_to_sim.amp_reference_export import export_amp_reference
from stablewalk.real_to_sim.contact_sync_reward import (
    summarize_contact_sync,
)
from stablewalk.real_to_sim.gait_style_extraction import (
    GaitStyleFingerprint,
    extract_gait_style_fingerprint,
)
from stablewalk.real_to_sim.motion_reference_loader import load_motion_reference
from stablewalk.real_to_sim.retargeting import (
    export_retargeted_motion_npz,
    load_retarget_config,
    retarget_motion_reference,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineStageStatus:
    stage: str
    status: str  # "complete" | "partial" | "pending"
    detail: str
    output_path: str | None = None
    duration_s: float | None = None


@dataclass
class RealToSimPipelineReport:
    """Result of running the offline Real-to-Sim pipeline."""

    run_name: str
    stages: list[PipelineStageStatus] = field(default_factory=list)
    gait_style: GaitStyleFingerprint | None = None
    motion_npz_path: Path | None = None
    retargeted_npz_path: Path | None = None
    amp_npz_path: Path | None = None
    contact_sync_npz_path: Path | None = None
    contact_sync: dict[str, Any] | None = None
    virtual_grf: dict[str, Any] | None = None
    isaac_lab_available: bool = False
    isaac_lab_note: str = ""
    report_path: Path | None = None
    pose_backend_requested: str = "mediapipe"
    pose_backend_used: str = "mediapipe"
    pose_backend_fallback: bool = False
    pose_backend_fallback_reason: str | None = None
    smpl_motion_path: Path | None = None
    biomechanical_report_path: Path | None = None
    biomechanical: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "stages": [
                {
                    "stage": s.stage,
                    "status": s.status,
                    "detail": s.detail,
                    "output_path": s.output_path,
                    "duration_s": s.duration_s,
                }
                for s in self.stages
            ],
            "gait_style": None if self.gait_style is None else self.gait_style.to_dict(),
            "motion_npz_path": str(self.motion_npz_path) if self.motion_npz_path else None,
            "retargeted_npz_path": (
                str(self.retargeted_npz_path) if self.retargeted_npz_path else None
            ),
            "amp_npz_path": str(self.amp_npz_path) if self.amp_npz_path else None,
            "contact_sync_npz_path": (
                str(self.contact_sync_npz_path) if self.contact_sync_npz_path else None
            ),
            "contact_sync": self.contact_sync,
            "virtual_grf": self.virtual_grf,
            "isaac_lab_available": self.isaac_lab_available,
            "isaac_lab_note": self.isaac_lab_note,
            "report_path": str(self.report_path) if self.report_path else None,
            "pose_backend_requested": self.pose_backend_requested,
            "pose_backend_used": self.pose_backend_used,
            "pose_backend_fallback": self.pose_backend_fallback,
            "pose_backend_fallback_reason": self.pose_backend_fallback_reason,
            "smpl_motion_path": str(self.smpl_motion_path) if self.smpl_motion_path else None,
            "biomechanical_report_path": (
                str(self.biomechanical_report_path) if self.biomechanical_report_path else None
            ),
            "biomechanical": self.biomechanical,
        }


def run_real_to_sim_pipeline(
    recording: GaitMotionRecording,
    output_dir: Path,
    *,
    run_name: str,
    sequence: PoseSequence | None = None,
    cycles: GaitCycleAnalysisResult | None = None,
    robot_config_path: Path | None = None,
    pose_backend_used: str = "mediapipe",
    pose_backend_requested: str = "mediapipe",
    pose_backend_fallback: bool = False,
    pose_backend_fallback_reason: str | None = None,
) -> RealToSimPipelineReport:
    """
    Run offline Real-to-Sim pipeline stages 1–4.

    Stage 3 (Isaac Lab AMP training) is prepared via AMP reference export;
    actual RL training requires a separate Isaac Lab environment.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    report = RealToSimPipelineReport(run_name=run_name)
    report.pose_backend_requested = pose_backend_requested
    report.pose_backend_used = pose_backend_used
    report.pose_backend_fallback = pose_backend_fallback
    report.pose_backend_fallback_reason = pose_backend_fallback_reason

    smpl_candidate = run_dir / "smpl_motion.npz"
    if smpl_candidate.is_file():
        report.smpl_motion_path = smpl_candidate

    isaac_ok, isaac_msg = check_isaac_lab_available()
    report.isaac_lab_available = isaac_ok
    report.isaac_lab_note = isaac_msg

    if cycles is None:
        cycles = analyze_gait_cycles(recording)

    # Stage 1 — Perception
    _t0 = time.perf_counter()
    gait_style = extract_gait_style_fingerprint(recording, cycles)
    report.gait_style = gait_style
    s1_status = "complete"
    if gait_style.usable_cycles < 1 or gait_style.confidence < 0.4:
        s1_status = "partial"
    report.stages.append(
        PipelineStageStatus(
            stage="1_perception",
            status=s1_status,
            detail=(
                f"{gait_style.style_summary} "
                f"[backend: {pose_backend_used}"
                f"{', fallback from ' + pose_backend_requested if pose_backend_fallback else ''}]"
            ),
            output_path=str(run_dir / "stablewalk_motion.npz"),
            duration_s=time.perf_counter() - _t0,
        )
    )

    # Stage 2 — Motion reference + retargeting
    _t0 = time.perf_counter()
    motion_path = run_dir / "stablewalk_motion.npz"
    export_motion_reference_npz(
        recording,
        cycles,
        motion_path,
        gait_style=gait_style,
    )
    report.motion_npz_path = motion_path

    motion_data = load_motion_reference(motion_path)
    retarget_cfg = load_retarget_config(robot_config_path)
    retargeted = retarget_motion_reference(motion_data, retarget_cfg)
    retargeted_path = run_dir / "retargeted_motion.npz"
    export_retargeted_motion_npz(
        retargeted,
        retargeted_path,
        source_motion_path=motion_path,
    )
    report.retargeted_npz_path = retargeted_path
    report.stages.append(
        PipelineStageStatus(
            stage="2_retargeting",
            status="complete",
            detail=(
                f"Scaled human motion to {retarget_cfg.robot_name} "
                f"(scale ×{retargeted.scale_factor:.2f})"
            ),
            output_path=str(retargeted_path),
            duration_s=time.perf_counter() - _t0,
        )
    )

    # Stage 3 — AMP reference export (simulation prep)
    _t0 = time.perf_counter()
    amp_result = export_amp_reference(
        motion_data,
        retargeted,
        output_dir,
        run_name=run_name,
        gait_style=gait_style,
    )
    report.amp_npz_path = amp_result.npz_path
    sim_status = "partial" if not isaac_ok else "complete"
    sim_detail = (
        "AMP reference exported — run Isaac Lab training in separate env."
        if not isaac_ok
        else "AMP reference exported — Isaac Lab detected (training not auto-run)."
    )
    report.stages.append(
        PipelineStageStatus(
            stage="3_simulation_amp",
            status=sim_status,
            detail=sim_detail,
            output_path=str(amp_result.npz_path),
            duration_s=time.perf_counter() - _t0,
        )
    )

    # Stage 4 — Physics / contact-sync vs estimated virtual GRF
    _t0 = time.perf_counter()
    from stablewalk.io.foot_contact_export import export_foot_contact_artifacts

    foot_exports = export_foot_contact_artifacts(
        recording,
        run_dir,
        run_name=run_name,
        cycles=cycles,
        reference_left_mask=motion_data.left_contact_mask,
        reference_right_mask=motion_data.right_contact_mask,
        retarget_left_mask=retargeted.left_contact_mask,
        retarget_right_mask=retargeted.right_contact_mask,
    )
    vgrf_result = foot_exports.vgrf
    report.virtual_grf = {
        **vgrf_result.to_dict(),
        "npz_path": str(foot_exports.virtual_grf_path),
        "contact_mask_path": str(foot_exports.contact_mask_path),
    }

    contact_sync_dict = None
    if foot_exports.contact_sync_path is not None:
        sync = summarize_contact_sync(
            retargeted.left_contact_mask,
            retargeted.right_contact_mask,
            vgrf_result.left_vgrf_vertical,
            vgrf_result.right_vgrf_vertical,
            reference_left_mask=motion_data.left_contact_mask,
            reference_right_mask=motion_data.right_contact_mask,
            confidence=foot_exports.contact.left_contact_probability,
        )
        contact_sync_dict = sync.to_dict()
        report.contact_sync_npz_path = foot_exports.contact_sync_path
        contact_sync_dict["per_frame_npz"] = str(foot_exports.contact_sync_path)
        contact_sync_dict["frame_count"] = int(len(vgrf_result.left_vgrf_vertical))
        report.contact_sync = contact_sync_dict

    physics_detail = (
        (
            f"{contact_sync_dict['interpretation']} "
            f"(mean sync {contact_sync_dict['mean_reward']:.0%})"
        )
        if contact_sync_dict
        else "Estimated vGRF unavailable — insufficient contact or pose data."
    )
    report.stages.append(
        PipelineStageStatus(
            stage="4_physics_vgrf",
            status="complete" if vgrf_result.available else "partial",
            detail=physics_detail,
            output_path=(
                str(report.contact_sync_npz_path)
                if report.contact_sync_npz_path
                else None
            ),
            duration_s=time.perf_counter() - _t0,
        )
    )

    # Stage 5 — Biomechanical analysis (extends pipeline; does not replace stability)
    _t0 = time.perf_counter()
    from stablewalk.analysis.biomechanical import run_biomechanical_analysis
    from stablewalk.io.biomechanical_export import export_biomechanical_artifacts

    biomech = run_biomechanical_analysis(
        recording,
        sequence,
        cycles=foot_exports.contact.gait_cycles or cycles,
        contact=foot_exports.contact,
    )
    biomech_exports = export_biomechanical_artifacts(biomech, run_dir, run_name=run_name)
    report.biomechanical_report_path = biomech_exports.biomechanical_report_path
    report.biomechanical = {
        **biomech.to_dict(),
        "center_of_mass_npz": str(biomech_exports.center_of_mass_path),
        "base_of_support_npz": str(biomech_exports.base_of_support_path),
        "video_quality_json": str(biomech_exports.video_quality_path),
        "gait_quality_score": (
            None if biomech.gait_quality is None else biomech.gait_quality.score
        ),
    }
    gq = biomech.gait_quality.score if biomech.gait_quality else 0.0
    report.stages.append(
        PipelineStageStatus(
            stage="5_biomechanical",
            status="complete" if biomech.center_of_mass else "partial",
            detail=f"Biomechanical report — Gait Quality {gq:.0f}/100 (estimated)",
            output_path=str(biomech_exports.biomechanical_report_path),
            duration_s=time.perf_counter() - _t0,
        )
    )

    report_path = run_dir / "real_to_sim_pipeline_report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
    report.report_path = report_path
    logger.info("Real-to-Sim pipeline report → %s", report_path)
    return report


def load_pipeline_report(path: Path) -> dict[str, Any] | None:
    """Load a saved ``real_to_sim_pipeline_report.json`` if present."""
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "PipelineStageStatus",
    "RealToSimPipelineReport",
    "load_pipeline_report",
    "run_real_to_sim_pipeline",
]
