"""Unit tests for Biomechanics chart enrichment (stats, peaks, interpretation)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from matplotlib.figure import Figure

from stablewalk.ui.viewers.biomech_chart_enrichment import (
    BIOMECH_EXPORT_NAMES,
    CLINICAL_CARD_DISCLAIMER,
    CLINICAL_CARD_SECTION_KEYS,
    annotate_peaks,
    build_clinical_interpretation,
    build_tab_enrichment,
    compute_series_stats,
    enrich_numeric_series,
    local_peak_indices,
    phase_duration_stats,
)
from stablewalk.ui.viewers.biomechanics_charts import (
    BIOMECH_SERIES_COM,
    BIOMECH_SERIES_PHASE,
    BIOMECH_SERIES_STABILITY,
    draw_biomechanics_dashboard,
)
from tests.test_biomechanics_charts import _sample_result


def test_compute_series_stats_max_min_avg_std() -> None:
    stats = compute_series_stats([1.0, 2.0, 3.0, 4.0], unit="BH", peak_count=2)
    assert stats.n == 4
    assert stats.maximum == 4.0
    assert stats.minimum == 1.0
    assert stats.mean == 2.5
    assert stats.std is not None and stats.std > 0
    assert "Max" in stats.format_line()
    assert "Avg" in stats.format_line()
    assert "SD" in stats.format_line()
    assert "Peaks 2" in stats.format_line()


def test_local_peak_indices_finds_maxima() -> None:
    y = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0], dtype=float)
    peaks = local_peak_indices(y, min_sep=1)
    assert 1 in peaks and 3 in peaks


def test_enrich_numeric_series_draws_peaks_and_stats() -> None:
    fig = Figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    t = np.linspace(0, 1, 40)
    y = np.sin(2 * np.pi * t * 3) + 1.0
    ax.plot(t, y)
    stats = enrich_numeric_series(ax, t, y, unit="BH", peak_color="#4dabf7")
    assert stats.n == 40
    assert stats.peak_count >= 1
    assert any(getattr(c, "get_offsets", None) for c in ax.collections)


def test_build_tab_enrichment_com_export_and_interp() -> None:
    result = _sample_result()
    gui = SimpleNamespace(
        _biomech_analysis=result,
        _foot_contact=result.contact,
        _estimated_vgrf=None,
        _gait_cycle=None,
    )
    enr = build_tab_enrichment(gui, "com")
    assert enr.export_name == BIOMECH_EXPORT_NAMES["com"]
    assert "COM" in enr.interpretation or "height" in enr.interpretation.lower()
    assert "Max" in enr.stats_line
    assert enr.stats is not None and enr.stats.n > 0
    assert "pelvis" in enr.interpretation.lower()
    assert "floor-absolute" in enr.interpretation.lower()


def test_com_reference_is_explicitly_illustrative() -> None:
    result = _sample_result()
    gui = SimpleNamespace(
        _biomech_analysis=result,
        _foot_contact=result.contact,
        _estimated_vgrf=None,
        _gait_cycle=None,
    )
    card = build_clinical_interpretation(gui, "com")
    text = card.normal_range.lower()
    assert "illustrative" in text
    assert "not a clinical normal range" in text
    assert "floor-absolute" in text


def test_clinical_interpretation_card_sections_are_educational() -> None:
    result = _sample_result()
    gui = SimpleNamespace(
        _biomech_analysis=result,
        _foot_contact=result.contact,
        _estimated_vgrf=None,
        _gait_cycle=None,
    )
    for tab_id in ("com", "balance", "contact", "stability", "grf"):
        card = build_clinical_interpretation(gui, tab_id)
        sections = card.as_sections()
        assert set(sections) == set(CLINICAL_CARD_SECTION_KEYS)
        for key in CLINICAL_CARD_SECTION_KEYS:
            assert sections[key].strip()
        blob = " ".join(sections.values()).lower()
        assert "diagnos" in card.disclaimer.lower() or "diagnosis" in CLINICAL_CARD_DISCLAIMER.lower()
        assert (
            "diagnos" not in blob
            or "not a" in blob
            or "not provide" in blob
            or "never provides" in blob
        )
        assert "educational" in card.disclaimer.lower() or "educational" in blob


def test_unreportable_balance_does_not_claim_physical_stability() -> None:
    result = _sample_result()
    sm = getattr(result, "stability_margin", None)
    if sm is None:
        sm = SimpleNamespace(reportable=False, stable_pct=5.83, per_frame=[])
        result.stability_margin = sm
    else:
        sm.reportable = False
        sm.stable_pct = 5.83
    gui = SimpleNamespace(
        _biomech_analysis=result,
        _foot_contact=result.contact,
        _estimated_vgrf=None,
        _gait_cycle=None,
    )
    enr = build_tab_enrichment(gui, "balance")
    assert "not a physical stability score" in enr.interpretation.lower()
    assert "5.83" not in enr.interpretation
    card = build_clinical_interpretation(gui, "balance")
    blob = " ".join(card.as_sections().values()).lower()
    assert "unavailable" in blob or "experimental" in blob
    assert "confident" in blob or "do not conclude" in blob


def test_biomechanics_phase_tab_uses_accurate_label() -> None:
    from stablewalk.ui.tk.dashboard_biomechanics import _BIOMECH_WORKSPACE_TABS

    labels = {tab_id: label for tab_id, label, _family, _series in _BIOMECH_WORKSPACE_TABS}
    assert labels["stability"] == "Gait Phase"


def test_phase_duration_stats_sums_near_100() -> None:
    result = _sample_result()
    stats = phase_duration_stats(result.contact)
    assert stats.n > 0
    total = float(stats.maximum or 0) + float(stats.minimum or 0) + float(stats.mean or 0)
    assert 95.0 <= total <= 105.0


def test_dashboard_series_include_enrichment_artists() -> None:
    result = _sample_result()
    for series in (BIOMECH_SERIES_COM, BIOMECH_SERIES_STABILITY, BIOMECH_SERIES_PHASE):
        fig = Figure(figsize=(11, 4.8))
        draw_biomechanics_dashboard(fig, result, series=series)
        ax = fig.get_axes()[0]
        # Stats box and/or peak scatter should be present on enriched views.
        texts = [t.get_text() for t in ax.texts if t.get_text()]
        has_stats = any("Max" in t or "Stance" in t for t in texts)
        has_peaks = any(
            getattr(c, "get_offsets", None) is not None and len(c.get_offsets()) > 0
            for c in ax.collections
        )
        assert has_stats or has_peaks or ax.get_legend() is not None


def test_annotate_peaks_returns_count() -> None:
    fig = Figure(figsize=(4, 2))
    ax = fig.add_subplot(111)
    t = np.arange(20, dtype=float)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=float)
    n = annotate_peaks(ax, t, y, color="#fff")
    assert n >= 1
