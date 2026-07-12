"""Orchestrate foot-contact exports for Real-to-Sim motion reference runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stablewalk.analysis.estimated_vgrf_analysis import EstimatedVGRFResult, analyze_estimated_vgrf
from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult, analyze_foot_contact
from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult
from stablewalk.io.contact_mask_export import export_contact_mask_npz
from stablewalk.io.virtual_grf_export import export_virtual_grf_npz
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.real_to_sim.contact_sync_reward import export_contact_sync_reward_npz


@dataclass(frozen=True)
class FootContactExportResult:
    contact: FootContactAnalysisResult
    vgrf: EstimatedVGRFResult
    contact_mask_path: Path
    virtual_grf_path: Path
    contact_sync_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contact_mask_path": str(self.contact_mask_path),
            "virtual_grf_path": str(self.virtual_grf_path),
            "contact_sync_path": (
                str(self.contact_sync_path) if self.contact_sync_path else None
            ),
            "metrics": self.contact.metrics.to_dict(),
            "vgrf": self.vgrf.to_dict(),
        }


def export_foot_contact_artifacts(
    recording: GaitMotionRecording,
    run_dir: Path,
    *,
    run_name: str = "",
    cycles: GaitCycleAnalysisResult | None = None,
    body_mass_kg: float = 70.0,
    reference_left_mask=None,
    reference_right_mask=None,
    retarget_left_mask=None,
    retarget_right_mask=None,
) -> FootContactExportResult:
    """
    Run foot-contact + estimated vGRF analysis and export NPZ artifacts.

    Writes:
      - contact_mask.npz
      - virtual_grf.npz
      - contact_sync_reward.npz (when masks available)
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    name = run_name or run_dir.name

    contact = analyze_foot_contact(recording, cycles=cycles)
    vgrf = analyze_estimated_vgrf(recording, contact, body_mass_kg=body_mass_kg)

    contact_path = run_dir / "contact_mask.npz"
    vgrf_path = run_dir / "virtual_grf.npz"
    export_contact_mask_npz(contact, contact_path, run_name=name)
    export_virtual_grf_npz(vgrf, vgrf_path, run_name=name)

    sync_path: Path | None = None
    if vgrf.available and len(vgrf.left_vgrf_vertical):
        ref_l = reference_left_mask
        ref_r = reference_right_mask
        cmp_l = retarget_left_mask if retarget_left_mask is not None else contact.left_contact_binary
        cmp_r = (
            retarget_right_mask if retarget_right_mask is not None else contact.right_contact_binary
        )
        if ref_l is None:
            ref_l = contact.left_contact_binary
        if ref_r is None:
            ref_r = contact.right_contact_binary

        confidence = None
        if contact.per_frame:
            confidence = (
                contact.left_contact_probability + contact.right_contact_probability
            ) * 0.5 * contact.metrics.contact_confidence

        sync_path = run_dir / "contact_sync_reward.npz"
        export_contact_sync_reward_npz(
            cmp_l,
            cmp_r,
            vgrf.left_vgrf_vertical,
            vgrf.right_vgrf_vertical,
            sync_path,
            timestamps=contact.timestamps,
            reference_left_mask=ref_l,
            reference_right_mask=ref_r,
            confidence=confidence,
            run_name=name,
        )

    return FootContactExportResult(
        contact=contact,
        vgrf=vgrf,
        contact_mask_path=contact_path,
        virtual_grf_path=vgrf_path,
        contact_sync_path=sync_path,
    )


__all__ = ["FootContactExportResult", "export_foot_contact_artifacts"]
