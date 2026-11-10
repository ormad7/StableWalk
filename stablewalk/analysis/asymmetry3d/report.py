"""Orchestrate 3D kinematics, gait events, bilateral indices, and moments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stablewalk.analysis.asymmetry3d.compare import AsymmetryScalar, compare_bilateral
from stablewalk.analysis.asymmetry3d.events import GaitEventSummary, summarize_gait_events
from stablewalk.analysis.asymmetry3d.indices import index_documentation
from stablewalk.analysis.asymmetry3d.kinematics import KinematicBundle, build_kinematics
from stablewalk.analysis.asymmetry3d.moments import MomentReport, build_moment_report
from stablewalk.analysis.asymmetry3d.quality import ReconstructionQuality, assess_reconstruction_quality
from stablewalk.analysis.asymmetry3d.trajectories import AlignedTrajectoryPair, compare_phase_aligned
from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult
from stablewalk.analysis.gait_view_analysis import GaitViewEstimate
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.models.pose_data import PoseSequence
from stablewalk.pose.reconstruction import ASSUMPTIONS as RECONSTRUCTION_ASSUMPTIONS


@dataclass
class Asymmetry3DReport:
    kinematics: KinematicBundle
    events: GaitEventSummary
    scalars: list[AsymmetryScalar] = field(default_factory=list)
    trajectories: list[AlignedTrajectoryPair] = field(default_factory=list)
    moments: MomentReport | None = None
    available: bool = False
    warnings: list[str] = field(default_factory=list)
    index_definitions: dict[str, Any] = field(default_factory=index_documentation)
    reconstruction_assumptions: str = RECONSTRUCTION_ASSUMPTIONS.strip()
    quality: ReconstructionQuality = field(default_factory=ReconstructionQuality)

    def scalar_by_name(self, name: str) -> AsymmetryScalar | None:
        for row in self.scalars:
            if row.name == name:
                return row
        return None

    def trajectory_by_name(self, name: str) -> AlignedTrajectoryPair | None:
        for row in self.trajectories:
            if row.name == name:
                return row
        return None

    def to_dict(self) -> dict[str, Any]:
        kin = self.kinematics
        events = self.events
        return {
            "schema": "stablewalk-asymmetry3d",
            "schema_version": "1.0",
            "available": self.available,
            "fps": kin.fps,
            "sampling_rate_hz": kin.fps,
            "coordinate_system": kin.coordinate_system,
            "coordinate_convention": {
                "frame": kin.coordinate_system,
                "units": kin.units,
                "provenance": kin.provenance,
                "knee_angle": (
                    "anatomical_flexion_deg; pose_interior_hka converted as "
                    "180° − interior HKA"
                ),
                "cycle_alignment": events.alignment_event,
            },
            "preprocessing": {
                "source": "gait_motion snapshots (no second reconstruction)",
                "calibrated_metric": kin.provenance == "calibrated_metric",
                "units": kin.units,
                "reconstruction_note": kin.reconstruction_note,
            },
            "cycle_definition": {
                "alignment_event": events.alignment_event,
                "alignment_label": events.alignment_label,
                "n_left_cycles": len(events.left_cycles),
                "n_right_cycles": len(events.right_cycles),
                "n_heel_strikes": len(events.heel_strikes),
            },
            "formula_identifiers": {
                "robinson_si": "robinson_si",
                "asi": "asi",
                "si_norm": "si_norm",
                "peak_timing": "circular_phase_difference_pp",
            },
            "kinematics": kin.to_dict(),
            "events": events.to_dict(),
            "scalars": [s.to_dict() for s in self.scalars],
            "trajectories": [t.to_dict() for t in self.trajectories],
            "moments": None if self.moments is None else self.moments.to_dict(),
            "index_definitions": dict(self.index_definitions),
            "reconstruction_assumptions": self.reconstruction_assumptions,
            "warnings": list(self.warnings),
            "quality": self.quality.to_dict(),
            "kind": "derived",
            "clinical": False,
        }


def analyze_asymmetry3d(
    recording: GaitMotionRecording,
    *,
    cycles: GaitCycleAnalysisResult | None = None,
    sequence: PoseSequence | None = None,
    subject_mass_kg: float | None = None,
    subject_height_m: float | None = None,
    opensim_run_name: str | None = None,
    calibrated_metric: bool = False,
    view_estimate: GaitViewEstimate | None = None,
    detect_if_missing: bool = True,
) -> Asymmetry3DReport:
    """Build the requirement-4 report from an existing gait-motion recording.

    Reuses MediaPipe canonical 3D, existing gait-cycle events, and the
    OpenSim ID readiness gate. Does not create a second reconstruction.
    When ``cycles`` is omitted, the existing ``analyze_gait_cycles`` detector
    is used so phase-matched L/R metrics are not silently blank.
    """
    kin = build_kinematics(recording, calibrated_metric=calibrated_metric)
    events = summarize_gait_events(
        recording,
        cycles,
        sequence=sequence,
        view_estimate=view_estimate,
        detect_if_missing=detect_if_missing,
    )
    aligned = compare_phase_aligned(kin, events.left_cycles, events.right_cycles)
    quality = assess_reconstruction_quality(recording, kin, events, aligned)
    scalars = compare_bilateral(
        recording,
        kin,
        events,
        aligned,
        ground_reliable=quality.ground_reliable,
        floor_y=quality.floor_y,
    )
    moments = build_moment_report(
        kin,
        subject_mass_kg=subject_mass_kg,
        subject_height_m=subject_height_m,
        opensim_run_name=opensim_run_name,
    )
    warnings = (
        list(kin.warnings)
        + list(events.warnings)
        + list(moments.warnings)
        + list(quality.warnings)
    )
    available = kin.n_frames >= 4
    return Asymmetry3DReport(
        kinematics=kin,
        events=events,
        scalars=scalars,
        trajectories=aligned,
        moments=moments,
        available=available,
        warnings=warnings,
        quality=quality,
    )
