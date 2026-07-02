"""
Automatic abnormal-gait detection via statistical norms and optional isolation forest.

Mirrors clinical screening: flag deviations in cadence, variability, symmetry,
and CoM — similar to automated quality checks in instrumented gait labs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from stablewalk.analysis.advanced.features import FEATURE_NAMES, GaitFeatureVector

# Adult walking norms (approximate; literature / clinical ranges)
# Cadence ~1.2-1.4 m/s walking often 100-120 spm; symmetry >0.85 typical
NORM_RANGES: dict[str, tuple[float, float]] = {
    "cadence_hz": (0.9, 2.0),
    "cadence_steps_per_min": (54, 130),
    "symmetry_score": (0.72, 1.0),
    "stance_symmetry_ratio": (0.75, 1.0),
    "step_time_variability": (0.0, 0.15),
    "stride_length_variability": (0.0, 0.18),
    "com_lateral_std": (0.0, 0.05),
    "knee_symmetry_deg": (0.0, 15.0),
    "hip_symmetry_deg": (0.0, 15.0),
    "stability_score": (55.0, 100.0),
    "grf_symmetry": (0.65, 1.0),
}


@dataclass
class AnomalyReport:
    """Result of automatic abnormal-gait screening."""

    is_anomaly: bool
    anomaly_score: float  # 0 = typical, 1 = highly atypical
    severity: str  # none | mild | moderate | severe
    triggered_rules: list[str] = field(default_factory=list)
    method: str = "statistical"
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": round(self.anomaly_score, 4),
            "severity": self.severity,
            "triggered_rules": list(self.triggered_rules),
            "method": self.method,
            "explanation": self.explanation,
        }


class GaitAnomalyDetector:
    """
    Detect atypical gait from feature vectors.

    Primary mode: rule/statistical deviation from healthy norms (no training data).
    Optional: ``IsolationForest`` when scikit-learn is installed and
    ``fit_reference()`` was called on a cohort of typical walks.
    """

    def __init__(self, *, anomaly_threshold: float = 0.45) -> None:
        self.anomaly_threshold = anomaly_threshold
        self._iso_model = None
        self._reference_vectors: list[list[float]] = []

    def fit_reference(self, vectors: list[GaitFeatureVector]) -> None:
        """Optional: learn typical variation from a reference cohort (e.g. healthy walks)."""
        self._reference_vectors = [v.to_array() for v in vectors]
        try:
            import numpy as np
            from sklearn.ensemble import IsolationForest

            X = np.array(self._reference_vectors, dtype=float)
            # Impute column means for NaN
            col_mean = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            self._iso_model = IsolationForest(
                contamination=0.1,
                random_state=42,
            )
            self._iso_model.fit(X)
        except ImportError:
            self._iso_model = None

    def detect(self, features: GaitFeatureVector) -> AnomalyReport:
        """Run statistical rules plus optional ML outlier score."""
        rules, rule_score = self._statistical_screen(features)
        ml_score = self._isolation_score(features)
        combined = max(rule_score, ml_score)

        # Known pattern flags from stability pipeline
        if features.flags:
            combined = max(combined, min(1.0, 0.35 + 0.08 * len(features.flags)))

        severity = _severity(combined)
        is_anomaly = combined >= self.anomaly_threshold or len(rules) >= 3

        explanation = _build_explanation(rules, combined, severity, features.flags)
        method = "statistical+isolation_forest" if ml_score > 0 and self._iso_model else "statistical"

        return AnomalyReport(
            is_anomaly=is_anomaly,
            anomaly_score=combined,
            severity=severity,
            triggered_rules=rules,
            method=method,
            explanation=explanation,
        )

    def _statistical_screen(self, features: GaitFeatureVector) -> tuple[list[str], float]:
        triggered: list[str] = []
        deviations: list[float] = []

        for key, (lo, hi) in NORM_RANGES.items():
            val = features.values.get(key)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            if val < lo:
                triggered.append(f"{key}_below_norm")
                deviations.append((lo - val) / max(abs(lo), 1e-6))
            elif val > hi:
                triggered.append(f"{key}_above_norm")
                deviations.append((val - hi) / max(abs(hi), 1e-6))

        score = min(1.0, sum(deviations) / max(len(deviations), 1) * 0.35) if deviations else 0.0
        return triggered, score

    def _isolation_score(self, features: GaitFeatureVector) -> float:
        if self._iso_model is None:
            return 0.0
        try:
            import numpy as np

            x = np.array([features.to_array()], dtype=float)
            col_mean = np.nanmean(np.array(self._reference_vectors), axis=0)
            for j in range(x.shape[1]):
                if math.isnan(x[0, j]):
                    x[0, j] = col_mean[j] if not math.isnan(col_mean[j]) else 0.0
            pred = self._iso_model.predict(x)[0]
            dec = self._iso_model.decision_function(x)[0]
            if pred == -1:
                return min(1.0, 0.55 + abs(dec) * 0.15)
            return max(0.0, 0.25 - dec * 0.05)
        except Exception:
            return 0.0


def _severity(score: float) -> str:
    if score < 0.25:
        return "none"
    if score < 0.45:
        return "mild"
    if score < 0.65:
        return "moderate"
    return "severe"


def _build_explanation(rules: list[str], score: float, severity: str, flags: list[str]) -> str:
    lines = [
        f"Anomaly screening: {severity} (score {score:.2f}/1.00).",
        "Deviations from typical adult walking norms:",
    ]
    if rules:
        for r in rules[:8]:
            lines.append(f"  - {r.replace('_', ' ')}")
        if len(rules) > 8:
            lines.append(f"  - ... and {len(rules) - 8} more")
    else:
        lines.append("  - No strong statistical deviations.")
    if flags:
        lines.append("Stability flags: " + ", ".join(flags[:6]))
    return "\n".join(lines)
