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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult, analyze_gait_cycles
from stablewalk.analysis.isaac_lab_integration import check_isaac_lab_available
from stablewalk.analysis.virtual_grf import VirtualForceResult, estimate_virtual_grf
from stablewalk.io.motion_reference_export import export_motion_reference_npz
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.models.pose_data import PoseSequence
from stablewalk.real_to_sim.amp_reference_export import export_amp_reference
from stablewalk.real_to_sim.contact_sync_reward import (
    export_contact_sync_reward_npz,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "stages": [
                {
                    "stage": s.stage,
                    "status": s.status,
                    "detail": s.detail,
                    "output_path": s.output_path,
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
        }


def run_real_to_sim_pipeline(
    recording: GaitMotionRecording,
    output_dir: Path,
    *,
    run_name: str,
    sequence: PoseSequence | None = None,
    cycles: GaitCycleAnalysisResult | None = None,
    robot_config_path: Path | None = None,
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
    isaac_ok, isaac_msg = check_isaac_lab_available()
    report.isaac_lab_available = isaac_ok
    report.isaac_lab_note = isaac_msg

    if cycles is None:
        cycles = analyze_gait_cycles(recording)

    # Stage 1 — Perception
    gait_style = extract_gait_style_fingerprint(recording, cycles)
    report.gait_style = gait_style
    s1_status = "complete"
    if gait_style.usable_cycles < 1 or gait_style.confidence < 0.4:
        s1_status = "partial"
    report.stages.append(
        PipelineStageStatus(
            stage="1_perception",
            status=s1_status,
            detail=gait_style.style_summary,
            output_path=str(run_dir / "stablewalk_motion.npz"),
        )
    )

    # Stage 2 — Motion reference + retargeting
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
        )
    )

    # Stage 3 — AMP reference export (simulation prep)
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
        )
    )

    # Stage 4 — Physics / contact-sync vs virtual GRF
    from stablewalk.analysis.virtual_grf import LegacyPoseProxyForceEstimator

    vgrf_result: VirtualForceResult = estimate_virtual_grf(
        recording,
        cycles,
        estimator=LegacyPoseProxyForceEstimator(),
        sequence=sequence,
    )
    report.virtual_grf = vgrf_result.to_dict()

    contact_sync_dict = None
    if vgrf_result.available and len(vgrf_result.left_vgrf_n):
        sync = summarize_contact_sync(
            motion_data.left_contact_mask,
            motion_data.right_contact_mask,
            vgrf_result.left_vgrf_n,
            vgrf_result.right_vgrf_n,
        )
        contact_sync_dict = sync.to_dict()
        contact_sync_path = run_dir / "contact_sync_reward.npz"
        export_contact_sync_reward_npz(
            motion_data.left_contact_mask,
            motion_data.right_contact_mask,
            vgrf_result.left_vgrf_n,
            vgrf_result.right_vgrf_n,
            contact_sync_path,
            timestamps=motion_data.timestamps,
        )
        report.contact_sync_npz_path = contact_sync_path
        contact_sync_dict["per_frame_npz"] = str(contact_sync_path)
        contact_sync_dict["frame_count"] = int(
            min(
                len(motion_data.left_contact_mask),
                len(vgrf_result.left_vgrf_n),
            )
        )
        report.contact_sync = contact_sync_dict

    physics_detail = (
        (
            f"{contact_sync_dict['interpretation']} "
            f"(mean sync {contact_sync_dict['mean_reward']:.0%})"
        )
        if contact_sync_dict
        else "Virtual GRF unavailable — load pose sequence for force proxy."
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
