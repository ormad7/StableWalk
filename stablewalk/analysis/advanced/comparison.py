"""
Compare two walking sessions (videos / pose sequences).

Extends angle-only ``gait_comparison`` with spatiotemporal and stability deltas,
similar to pre/post rehab or left-right instrumented comparisons in gait labs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stablewalk.analysis.advanced.classification import ClassificationResult, GaitClassifier
from stablewalk.analysis.advanced.features import FEATURE_NAMES, GaitFeatureVector, extract_gait_features
from stablewalk.analysis.comparison import GaitComparison, compare_sequences
from stablewalk.analysis.report import GaitAnalysisReport, analyze_gait
from stablewalk.models.pose_data import PoseSequence


@dataclass
class MetricDelta:
    name: str
    reference: float | None
    sample: float | None
    delta: float | None
    pct_change: float | None
    interpretation: str = ""


@dataclass
class GaitSessionComparison:
    """Full comparison between two analyzed walks."""

    reference_name: str
    sample_name: str
    reference_features: GaitFeatureVector
    sample_features: GaitFeatureVector
    reference_class: ClassificationResult
    sample_class: ClassificationResult
    metric_deltas: list[MetricDelta] = field(default_factory=list)
    angle_comparison: GaitComparison | None = None
    more_stable: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_name": self.reference_name,
            "sample_name": self.sample_name,
            "more_stable": self.more_stable,
            "reference_class": self.reference_class.to_dict(),
            "sample_class": self.sample_class.to_dict(),
            "metric_deltas": [
                {
                    "name": d.name,
                    "reference": d.reference,
                    "sample": d.sample,
                    "delta": d.delta,
                    "pct_change": d.pct_change,
                    "interpretation": d.interpretation,
                }
                for d in self.metric_deltas
            ],
            "summary": self.summary,
        }


def compare_gait_sessions(
    reference: PoseSequence,
    sample: PoseSequence,
    *,
    reference_name: str = "reference",
    sample_name: str = "sample",
    body_mass_kg: float = 70.0,
    classifier: GaitClassifier | None = None,
    reference_report: GaitAnalysisReport | None = None,
    sample_report: GaitAnalysisReport | None = None,
) -> GaitSessionComparison:
    """
    Compare two enriched pose sequences end-to-end.

    Pass ``reference_report`` / ``sample_report`` when already computed to avoid
    duplicate metric/GRF/stability passes.
    """
    clf = classifier or GaitClassifier()
    ref_report = reference_report
    samp_report = sample_report
    if ref_report is None:
        ref_report, _ = analyze_gait(reference, body_mass_kg=body_mass_kg)
    if samp_report is None:
        samp_report, _ = analyze_gait(sample, body_mass_kg=body_mass_kg)

    ref_feat = extract_gait_features(reference, ref_report, source_id=reference_name)
    samp_feat = extract_gait_features(sample, samp_report, source_id=sample_name)

    ref_class = clf.classify(ref_feat, ref_report.stability)
    samp_class = clf.classify(samp_feat, samp_report.stability)

    deltas = _compute_deltas(ref_feat, samp_feat)
    angle_cmp = compare_sequences(reference, sample, reference_name=reference_name, sample_name=sample_name)

    ref_score = ref_report.stability.score
    samp_score = samp_report.stability.score
    if ref_score >= samp_score + 5:
        more_stable = reference_name
    elif samp_score >= ref_score + 5:
        more_stable = sample_name
    else:
        more_stable = "similar"

    summary = _build_summary(
        reference_name,
        sample_name,
        ref_class,
        samp_class,
        ref_score,
        samp_score,
        more_stable,
        deltas,
        angle_cmp,
    )

    return GaitSessionComparison(
        reference_name=reference_name,
        sample_name=sample_name,
        reference_features=ref_feat,
        sample_features=samp_feat,
        reference_class=ref_class,
        sample_class=samp_class,
        metric_deltas=deltas,
        angle_comparison=angle_cmp,
        more_stable=more_stable,
        summary=summary,
    )


def _compute_deltas(ref: GaitFeatureVector, samp: GaitFeatureVector) -> list[MetricDelta]:
    deltas: list[MetricDelta] = []
    key_metrics = (
        "stability_score",
        "symmetry_score",
        "cadence_steps_per_min",
        "stride_length_m",
        "step_time_variability",
        "com_lateral_std",
        "stance_symmetry_ratio",
    )
    for name in key_metrics:
        r = ref.values.get(name)
        s = samp.values.get(name)
        delta = None
        pct = None
        if r is not None and s is not None:
            delta = s - r
            if abs(r) > 1e-8:
                pct = 100.0 * delta / r
        interp = _interpret_delta(name, delta)
        deltas.append(
            MetricDelta(name=name, reference=r, sample=s, delta=delta, pct_change=pct, interpretation=interp)
        )
    return deltas


def _interpret_delta(name: str, delta: float | None) -> str:
    if delta is None:
        return "n/a"
    higher_better = {"stability_score", "symmetry_score", "stance_symmetry_ratio"}
    lower_better = {"step_time_variability", "com_lateral_std", "knee_symmetry_deg"}
    if name in higher_better:
        if delta > 5:
            return "sample improved"
        if delta < -5:
            return "sample worse"
        return "similar"
    if name in lower_better:
        if delta < -0.02:
            return "sample improved"
        if delta > 0.02:
            return "sample worse"
        return "similar"
    if abs(delta) < 0.05:
        return "similar"
    return "sample higher" if delta > 0 else "sample lower"


def _build_summary(
    ref_name: str,
    samp_name: str,
    ref_cls: ClassificationResult,
    samp_cls: ClassificationResult,
    ref_score: float,
    samp_score: float,
    more_stable: str,
    deltas: list[MetricDelta],
    angle_cmp: GaitComparison,
) -> str:
    lines = [
        f"Comparison: {ref_name} vs {samp_name}",
        f"  Reference: {ref_cls.label} ({ref_score:.0f}/100 stability)",
        f"  Sample:    {samp_cls.label} ({samp_score:.0f}/100 stability)",
        f"  More stable session: {more_stable}",
        "",
        "Key metric changes (sample - reference):",
    ]
    for d in deltas:
        if d.delta is None:
            continue
        pct = f" ({d.pct_change:+.0f}%)" if d.pct_change is not None else ""
        lines.append(f"  {d.name}: {d.delta:+.3f}{pct} — {d.interpretation}")
    lines.append("")
    lines.append(angle_cmp.summary)
    return "\n".join(lines)
