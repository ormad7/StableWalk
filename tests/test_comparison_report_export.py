"""Tests for Comparison Mode PDF report export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stablewalk.analysis.session_compare import (
    CompareMetrics,
    compare_session_metrics,
)
from stablewalk.io.comparison_report_export import (
    export_comparison_report_from_cache,
    export_comparison_report_pdf,
)
from stablewalk.ui.tk.session_cache import AnalyzedSessionSnapshot, SessionCompareCache


def _metrics(label: str, key: str, *, better: bool) -> CompareMetrics:
    if better:
        return CompareMetrics(
            label=label,
            session_key=key,
            gait_quality=82.0,
            cadence_spm=118.0,
            walking_speed_m_s=1.28,
            symmetry_pct=95.0,
            stability_label="High",
            stability_margin_m=0.05,
            com_excursion_m=0.04,
            virtual_grf_peak_bw=1.2,
            usable_gait_cycles=6,
        )
    return CompareMetrics(
        label=label,
        session_key=key,
        gait_quality=61.0,
        cadence_spm=83.0,
        walking_speed_m_s=0.74,
        symmetry_pct=71.0,
        stability_label="Low",
        stability_margin_m=0.01,
        com_excursion_m=0.11,
        virtual_grf_peak_bw=0.9,
        usable_gait_cycles=2,
    )


def _snap(key: str, label: str, *, better: bool) -> AnalyzedSessionSnapshot:
    t = list(np.linspace(0.0, 2.0, 20))
    y = [40.0 + (5.0 if better else 15.0) * np.sin(i) for i in range(20)]
    path = np.column_stack(
        [
            np.linspace(0.0, 1.0, 20),
            0.05 * np.sin(np.linspace(0, 4, 20)),
            np.linspace(0.0, 2.0, 20),
        ]
    )
    return AnalyzedSessionSnapshot(
        key=key,
        label=label,
        source=f"{key}.mp4",
        fps=30.0,
        n_frames=20,
        pose_indices=list(range(20)),
        metrics=_metrics(label, key, better=better),
        knee_t=t,
        knee_y=y,
        path_xyz=path,
    )


def test_comparison_narratives_populated() -> None:
    left = _metrics("Normal Gait", "normal", better=True)
    right = _metrics("Abnormal Gait", "abnormal", better=False)
    result = compare_session_metrics(left, right)
    assert result.clinical_summary
    assert result.research_summary
    assert result.overall_recommendation
    assert "clinical" in result.clinical_summary.lower() or "pose-estimated" in result.clinical_summary.lower()
    assert "monocular" in result.research_summary.lower()
    assert "recommendation" in result.overall_recommendation.lower()


def test_export_comparison_pdf(tmp_path: Path) -> None:
    left = _snap("normal", "Normal Gait", better=True)
    right = _snap("abnormal", "Abnormal Gait", better=False)
    out = tmp_path / "comparison_report.pdf"
    path = export_comparison_report_pdf(left, right, out)
    assert path.is_file()
    assert path.stat().st_size > 1000
    # PDF magic bytes
    assert path.read_bytes()[:4] == b"%PDF"


def test_export_from_cache_requires_both(tmp_path: Path) -> None:
    cache = SessionCompareCache()
    cache.pin(_snap("normal", "Normal Gait", better=True))
    try:
        export_comparison_report_from_cache(cache, tmp_path / "x.pdf")
        raised = False
    except ValueError:
        raised = True
    assert raised

    cache.pin(_snap("abnormal", "Abnormal Gait", better=False))
    out = export_comparison_report_from_cache(cache, tmp_path / "ok.pdf")
    assert out.is_file()
