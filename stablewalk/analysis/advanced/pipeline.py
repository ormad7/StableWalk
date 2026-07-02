"""
Unified advanced analysis entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stablewalk.analysis.advanced.anomaly import AnomalyReport, GaitAnomalyDetector
from stablewalk.analysis.advanced.classification import ClassificationResult, GaitClassifier
from stablewalk.analysis.advanced.features import GaitFeatureVector, extract_gait_features
from stablewalk.analysis.advanced.ml_stability import StabilityMLModel
from stablewalk.analysis.report import GaitAnalysisReport, analyze_gait
from stablewalk.models.pose_data import PoseSequence


@dataclass
class AdvancedGaitReport:
    """Anomaly + classification + features on top of standard gait report."""

    gait: GaitAnalysisReport
    features: GaitFeatureVector
    anomaly: AnomalyReport
    classification: ClassificationResult
    ml_stability_score: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "features": self.features.to_dict(),
            "anomaly": self.anomaly.to_dict(),
            "classification": self.classification.to_dict(),
            "stability_score": self.gait.stability.score,
            "stability_label": self.gait.stability.label,
        }
        if self.ml_stability_score is not None:
            out["ml_stability_score"] = round(self.ml_stability_score, 2)
        return out

    @property
    def summary(self) -> str:
        lines = [
            f"Classification: {self.classification.label} "
            f"({self.classification.confidence:.0%})",
            f"Anomaly: {self.anomaly.severity} (score {self.anomaly.anomaly_score:.2f})",
            f"Stability: {self.gait.stability.label} ({self.gait.stability.score:.0f}/100)",
        ]
        if self.ml_stability_score is not None:
            lines.append(f"ML stability estimate: {self.ml_stability_score:.0f}/100")
        return "\n".join(lines)


def analyze_gait_advanced(
    sequence: PoseSequence,
    *,
    body_mass_kg: float = 70.0,
    source_id: str = "",
    classifier: GaitClassifier | None = None,
    anomaly_detector: GaitAnomalyDetector | None = None,
    ml_model: StabilityMLModel | None = None,
) -> AdvancedGaitReport:
    """
    Run standard gait analysis plus anomaly detection and classification.
    """
    gait_report, _ = analyze_gait(sequence, body_mass_kg=body_mass_kg)
    features = extract_gait_features(
        sequence,
        gait_report,
        source_id=source_id,
        body_mass_kg=body_mass_kg,
    )

    detector = anomaly_detector or GaitAnomalyDetector()
    clf = classifier or GaitClassifier()
    anomaly = detector.detect(features)
    classification = clf.classify(features, gait_report.stability, anomaly=anomaly)

    ml_score = None
    notes: list[str] = []
    if ml_model is not None and ml_model.is_fitted:
        try:
            ml_score = ml_model.predict(features)
            notes.append("ML stability score available (trained model).")
        except Exception as exc:
            notes.append(f"ML predict skipped: {exc}")

    return AdvancedGaitReport(
        gait=gait_report,
        features=features,
        anomaly=anomaly,
        classification=classification,
        ml_stability_score=ml_score,
        notes=notes,
    )
