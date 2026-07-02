"""
Normal vs abnormal walking classification (rule-based + optional ML).

Aligns with clinical gait screening labels (typical vs pathological gait),
not a diagnosis — a triage layer like automated fall-risk or stroke gait flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from stablewalk.analysis.advanced.anomaly import AnomalyReport, GaitAnomalyDetector
from stablewalk.analysis.advanced.features import GaitFeatureVector
from stablewalk.analysis.stability import StabilityReport

GaitClassLabel = Literal["normal", "borderline", "abnormal"]


@dataclass
class ClassificationResult:
    """Walking pattern class with confidence and rationale."""

    label: GaitClassLabel
    confidence: float  # 0-1
    method: str  # rules | ml | hybrid
    reasons: list[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "reasons": list(self.reasons),
            "explanation": self.explanation,
        }


class GaitClassifier:
    """
    Classify a walk as normal, borderline, or abnormal.

    Default: hybrid of stability score, anomaly detector, and pattern flags
    (same information clinicians use: symmetry, variability, speed).
    Optional: sklearn classifier after ``fit()`` on labeled feature rows.
    """

    def __init__(
        self,
        *,
        normal_min_stability: float = 60.0,
        borderline_min_stability: float = 45.0,
        anomaly_threshold: float = 0.45,
    ) -> None:
        self.normal_min_stability = normal_min_stability
        self.borderline_min_stability = borderline_min_stability
        self._detector = GaitAnomalyDetector(anomaly_threshold=anomaly_threshold)
        self._sk_model = None
        self._label_encoder = None
        self._class_names: list[str] = ["normal", "borderline", "abnormal"]

    @property
    def anomaly_detector(self) -> GaitAnomalyDetector:
        return self._detector

    def fit(
        self,
        vectors: list[GaitFeatureVector],
        labels: list[str],
    ) -> bool:
        """
        Optional supervised training (labels: normal | borderline | abnormal).

        Returns True if sklearn model was fitted.
        """
        try:
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder

            X = np.array([v.to_array() for v in vectors], dtype=float)
            col_mean = np.nanmean(X, axis=0)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    if np.isnan(X[i, j]):
                        X[i, j] = col_mean[j] if not np.isnan(col_mean[j]) else 0.0

            le = LabelEncoder()
            le.fit(self._class_names + list(set(labels)))
            y = le.transform(labels)

            self._sk_model = RandomForestClassifier(
                n_estimators=64,
                max_depth=8,
                random_state=42,
            )
            self._sk_model.fit(X, y)
            self._label_encoder = le
            return True
        except ImportError:
            self._sk_model = None
            return False

    def classify(
        self,
        features: GaitFeatureVector,
        stability: StabilityReport | None = None,
        *,
        anomaly: AnomalyReport | None = None,
    ) -> ClassificationResult:
        """Classify one walk."""
        if anomaly is None:
            anomaly = self._detector.detect(features)

        rule_result = self._classify_rules(features, stability, anomaly)

        if self._sk_model is not None:
            ml_result = self._classify_ml(features)
            if ml_result is not None:
                return self._merge_hybrid(rule_result, ml_result)

        return rule_result

    def _classify_rules(
        self,
        features: GaitFeatureVector,
        stability: StabilityReport | None,
        anomaly: AnomalyReport,
    ) -> ClassificationResult:
        reasons: list[str] = []
        score = features.values.get("stability_score")
        if stability is not None:
            score = stability.score
            reasons.extend(stability.abnormal_patterns[:4])

        if anomaly.is_anomaly:
            reasons.append(f"anomaly_{anomaly.severity}")

        label: GaitClassLabel = "normal"
        confidence = 0.75

        if score is not None:
            if score >= self.normal_min_stability and not anomaly.is_anomaly:
                label = "normal"
                confidence = 0.7 + 0.25 * (score - self.normal_min_stability) / 40.0
            elif score >= self.borderline_min_stability or (
                anomaly.severity in ("mild", "none") and score and score >= 50
            ):
                label = "borderline"
                confidence = 0.55
            else:
                label = "abnormal"
                confidence = 0.65 + 0.2 * min(1.0, anomaly.anomaly_score)
        elif anomaly.is_anomaly:
            label = "abnormal"
            confidence = 0.6 + 0.25 * anomaly.anomaly_score
        else:
            label = "borderline"
            confidence = 0.5

        if features.flags and label == "normal":
            label = "borderline"
            reasons.append("stability_flags_present")

        confidence = max(0.35, min(0.95, confidence))
        explanation = (
            f"Classification: {label} ({confidence:.0%} confidence, rule-based).\n"
            f"Stability score: {score:.0f}/100.\n" if score is not None else ""
        ) + anomaly.explanation

        return ClassificationResult(
            label=label,
            confidence=confidence,
            method="rules",
            reasons=reasons[:10],
            explanation=explanation.strip(),
        )

    def _classify_ml(self, features: GaitFeatureVector) -> ClassificationResult | None:
        try:
            import numpy as np

            x = np.array([features.to_array()], dtype=float)
            col_mean = np.nanmean(x, axis=0)
            for j in range(x.shape[1]):
                if np.isnan(x[0, j]):
                    x[0, j] = col_mean[j]
            pred = self._sk_model.predict(x)[0]
            proba = self._sk_model.predict_proba(x)[0]
            label = str(self._label_encoder.inverse_transform([pred])[0])
            if label not in ("normal", "borderline", "abnormal"):
                label = "borderline"
            return ClassificationResult(
                label=label,  # type: ignore[arg-type]
                confidence=float(max(proba)),
                method="ml",
                reasons=["sklearn_random_forest"],
                explanation=f"ML classifier: {label} ({max(proba):.0%}).",
            )
        except Exception:
            return None

    @staticmethod
    def _merge_hybrid(rules: ClassificationResult, ml: ClassificationResult) -> ClassificationResult:
        if rules.label == ml.label:
            return ClassificationResult(
                label=rules.label,
                confidence=(rules.confidence + ml.confidence) / 2,
                method="hybrid",
                reasons=rules.reasons + ml.reasons,
                explanation=rules.explanation + "\n" + ml.explanation,
            )
        # Disagreement → borderline
        return ClassificationResult(
            label="borderline",
            confidence=0.5,
            method="hybrid",
            reasons=rules.reasons + ["ml_rule_disagreement"] + ml.reasons,
            explanation=rules.explanation + "\n[ML disagrees → borderline]\n" + ml.explanation,
        )
