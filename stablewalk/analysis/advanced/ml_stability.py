"""
Optional machine learning for stability score prediction.

Learns a mapping from gait feature vectors to stability scores (or expert labels),
analogous to data-driven fall-risk models built on wearable or vision features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stablewalk.analysis.advanced.features import FEATURE_NAMES, GaitFeatureVector


@dataclass
class MLTrainingReport:
    n_samples: int
    model_type: str
    train_r2: float | None = None
    notes: list[str] = field(default_factory=list)


class StabilityMLModel:
    """
    Optional regressor: features → stability score (0–100).

    Requires scikit-learn. Use when you have labeled clips (e.g. clinician scores
    or consensus labels from multiple raters).
    """

    def __init__(self, model_type: str = "ridge") -> None:
        self.model_type = model_type
        self._model = None
        self._fitted = False

    @property
    def is_available(self) -> bool:
        try:
            import sklearn  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def is_fitted(self) -> bool:
        return self._fitted and self._model is not None

    def fit(
        self,
        vectors: list[GaitFeatureVector],
        scores: list[float],
    ) -> MLTrainingReport:
        """Train on feature vectors and target stability scores."""
        if len(vectors) != len(scores) or len(vectors) < 3:
            raise ValueError("Need at least 3 matched (vector, score) pairs")

        try:
            import numpy as np
            from sklearn.linear_model import Ridge
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.model_selection import cross_val_score
        except ImportError as e:
            raise ImportError("pip install scikit-learn for StabilityMLModel") from e

        X = np.array([v.to_array() for v in vectors], dtype=float)
        y = np.array(scores, dtype=float)
        col_mean = np.nanmean(X, axis=0)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if np.isnan(X[i, j]):
                    X[i, j] = col_mean[j] if not np.isnan(col_mean[j]) else 0.0

        if self.model_type == "gbr":
            self._model = GradientBoostingRegressor(
                n_estimators=80,
                max_depth=4,
                random_state=42,
            )
        else:
            self._model = Ridge(alpha=1.0)

        self._model.fit(X, y)
        self._fitted = True

        r2 = None
        if len(vectors) >= 5:
            try:
                cv = cross_val_score(self._model, X, y, cv=min(5, len(vectors)), scoring="r2")
                r2 = float(cv.mean())
            except Exception:
                r2 = None

        return MLTrainingReport(
            n_samples=len(vectors),
            model_type=self.model_type,
            train_r2=r2,
            notes=[
                "Targets should be clinician or consensus stability scores.",
                "Vision-only features; not validated for clinical deployment.",
            ],
        )

    def predict(self, features: GaitFeatureVector) -> float:
        """Predict stability score 0–100."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")
        import numpy as np

        x = np.array([features.to_array()], dtype=float)
        col_mean = np.nanmean(x, axis=0)
        for j in range(x.shape[1]):
            if np.isnan(x[0, j]):
                x[0, j] = col_mean[j] if not np.isnan(col_mean[j]) else 0.0
        pred = float(self._model.predict(x)[0])
        return max(0.0, min(100.0, pred))

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        try:
            import joblib
        except ImportError:
            raise ImportError("pip install joblib to save models")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "model_type": self.model_type, "features": FEATURE_NAMES}, path)

    def load(self, path: str | Path) -> None:
        try:
            import joblib
        except ImportError:
            raise ImportError("pip install joblib to load models")
        data = joblib.load(path)
        self._model = data["model"]
        self.model_type = data.get("model_type", "ridge")
        self._fitted = True

    def save_training_manifest(self, path: str | Path, rows: list[dict[str, Any]]) -> None:
        """Export training rows for reproducibility."""
        Path(path).write_text(json.dumps(rows, indent=2), encoding="utf-8")
