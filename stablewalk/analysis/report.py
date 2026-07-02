"""
Unified gait analysis report: metrics, forces, stability, simulation hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stablewalk.analysis.forces import GRFAnalyzer, ForceAnalysisReport, GRFTimeSeries
from stablewalk.analysis.metrics import GaitMetrics, GaitMetricsReport, GaitMetricsResult, compute_gait_metrics
from stablewalk.models.pose_data import PoseSequence
from stablewalk.analysis.stability import StabilityAnalyzer, StabilityReport
from stablewalk.simulation.kinematic import WalkSimulation, WalkSimulator


@dataclass
class GaitAnalysisReport:
    """Complete analysis bundle for export or UI display."""

    metrics: GaitMetricsReport
    forces: ForceAnalysisReport
    stability: StabilityReport
    advanced_metrics: GaitMetricsResult | None = None
    grf: GRFTimeSeries | None = None
    simulation_frame_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary_text(self) -> str:
        lines = [
            "=== Gait metrics ===",
            f"Cadence: {self._fmt(self.metrics.cadence_steps_per_min, ' steps/min')}",
            f"Stride (norm): {self._fmt(self.metrics.stride_length_normalized)}",
            f"Stride (m est.): {self._fmt(self.metrics.stride_length_m, ' m')}",
            f"Stance symmetry: {self.metrics.contact.stance_symmetry_ratio:.0%}",
            "",
            self.forces.summary,
            "",
            "=== Stability ===",
            self.stability.summary,
        ]
        if self.warnings:
            lines.extend(["", "Warnings:", *[f"  - {w}" for w in self.warnings]])
        return "\n".join(lines)

    @staticmethod
    def _fmt(val: float | None, suffix: str = "") -> str:
        if val is None:
            return "n/a"
        return f"{val:.2f}{suffix}"


def analyze_gait(
    sequence: PoseSequence,
    *,
    body_mass_kg: float = 70.0,
    build_simulation: bool = False,
) -> tuple[GaitAnalysisReport, WalkSimulation | None]:
    """
    Run spatiotemporal metrics, GRF proxies, and stability on a pose sequence.

    Sequence should already be enriched (velocities, gait phases) from
    ``enrich_pose_sequence``.
    """
    warnings: list[str] = []
    if not any(f.gait_phase for f in sequence.frames if f.detected):
        warnings.append("Gait phases missing; contact/force estimates may be weak.")

    analyzer = GRFAnalyzer(body_mass_kg=body_mass_kg)
    grf = analyzer.analyze(sequence)
    gait_metrics_result = GaitMetrics().compute(sequence, grf=grf)
    metrics = GaitMetrics().to_legacy_report(gait_metrics_result)
    forces = analyzer.to_legacy_report(grf)
    stability = StabilityAnalyzer(body_mass_kg=body_mass_kg).analyze(
        sequence, grf=grf, gait_metrics=metrics
    )

    sim = None
    sim_count = 0
    if build_simulation:
        sim = WalkSimulator().from_pose_sequence(sequence)
        sim_count = len(sim.frames)

    return GaitAnalysisReport(
        metrics=metrics,
        forces=forces,
        grf=grf,
        stability=stability,
        advanced_metrics=gait_metrics_result,
        simulation_frame_count=sim_count,
        warnings=warnings,
    ), sim


def report_to_dict(report: GaitAnalysisReport) -> dict:
    """JSON-serializable summary for export alongside pose JSON."""
    m = report.metrics
    f = report.forces
    s = report.stability
    return {
        "cadence_steps_per_min": m.cadence_steps_per_min,
        "stride_length_normalized": m.stride_length_normalized,
        "stride_length_m": m.stride_length_m,
        "stance_symmetry_ratio": m.contact.stance_symmetry_ratio,
        "double_support_fraction": m.contact.double_support_fraction,
        "force_method": f.method,
        "body_mass_kg": f.body_mass_kg,
        "mean_grf_peak_proxy_n": f.mean_peak_vertical_n,
        "double_peak_stance_fraction": f.double_peak_fraction,
        "grf_method": report.grf.method if report.grf else None,
        "mean_grf_peak_bw": report.grf.mean_peak_bw if report.grf else None,
        "stability": s.to_dict(),
        "stability_label": s.label,
        "stability_score": s.score,
        "symmetry_score": s.symmetry_score,
        "abnormal_patterns": s.abnormal_patterns,
        "warnings": report.warnings,
        "gait_metrics": report.advanced_metrics.to_dict()
        if report.advanced_metrics
        else {},
    }
