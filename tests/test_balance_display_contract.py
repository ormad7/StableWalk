"""Balance plot and summary cards share one COM–BoS display contract."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.figure import Figure

from stablewalk.analysis.biomechanical.stability_margin import (
    BALANCE_CLASSIFICATION_REASON,
    BALANCE_UNIT,
    DISPLAY_PREVIEW,
    VALIDITY_UNVALIDATED,
    balance_display_contract,
)
from stablewalk.metrics.session_view import build_session_metric_snapshots
from stablewalk.metrics.status import ValidityState
from stablewalk.opensim_ik_status import (
    PIPELINE_COMPLETE,
    format_opensim_status_pair,
    resolve_opensim_ik_status,
    resolve_opensim_pipeline_status,
)
from stablewalk.ui.viewers.biomechanics_charts import (
    BALANCE_UNVALIDATED_BANNER,
    BIOMECH_SERIES_STABILITY,
    draw_biomechanics_dashboard,
)
from tests.test_biomechanics_charts import _sample_result


def test_monocular_balance_contract_is_unvalidated_preview() -> None:
    result = _sample_result()
    sm = result.stability_margin
    assert sm is not None
    contract = balance_display_contract(sm)
    assert contract.unit == BALANCE_UNIT
    assert contract.coordinate_frame
    assert "Hof" in contract.method or "monocular" in contract.method.lower()
    assert contract.validity == VALIDITY_UNVALIDATED
    assert contract.display_mode == DISPLAY_PREVIEW
    assert contract.reportable is False
    assert contract.classification_supported is False
    assert contract.has_series
    assert any(v is not None for v in contract.values_m)
    assert "0.04" in contract.reason
    assert BALANCE_CLASSIFICATION_REASON in contract.reason
    payload = sm.to_dict()
    assert payload["validity"] == VALIDITY_UNVALIDATED
    assert payload["display_mode"] == DISPLAY_PREVIEW
    assert payload["classification_supported"] is False


def test_balance_plot_is_unvalidated_preview_without_classification_bands() -> None:
    result = _sample_result()
    fig = Figure(figsize=(11, 4.8))
    draw_biomechanics_dashboard(fig, result, series=BIOMECH_SERIES_STABILITY)
    ax = fig.get_axes()[0]
    assert ax.lines
    legend = ax.get_legend()
    legend_text = " ".join(t.get_text() for t in legend.get_texts()) if legend else ""
    assert "Normal" not in legend_text
    assert "Stable" not in legend_text
    assert "Unstable" not in legend_text
    assert "0.04" not in legend_text
    texts = " ".join(t.get_text() or "" for t in list(ax.texts) + list(fig.texts))
    assert BALANCE_UNVALIDATED_BANNER in texts
    assert bool(getattr(fig, "_biomech_legend_below", False))
    assert bool(getattr(fig, "_biomech_preview_notice", False))
    hover = getattr(fig, "_biomech_hover_points", [])
    assert hover
    assert any("unvalidated" in (pt.text or "").lower() for pt in hover)
    assert not any("State =" in (pt.text or "") for pt in hover)
    ylabel = (ax.get_ylabel() or "").lower()
    assert "distance" in ylabel
    assert "signed" in ylabel


def test_summary_cards_stay_na_with_shared_reason() -> None:
    result = _sample_result()
    contract = balance_display_contract(result.stability_margin)
    view = build_session_metric_snapshots(biomechanical=result)
    pct = view.snapshots["stable_pct"]
    margin = view.snapshots["stability_margin_m"]
    state = view.snapshots["stability_frame_state"]
    assert pct.validity != ValidityState.VALID
    assert margin.validity != ValidityState.VALID
    assert state.validity != ValidityState.VALID
    assert pct.value is None
    assert margin.value is None
    assert state.value is None
    notes = " ".join(pct.notes)
    assert contract.reason in notes
    assert "0.04" in notes


def test_missing_analysis_hides_balance_series() -> None:
    contract = balance_display_contract(None)
    assert contract.display_mode == "hidden"
    assert contract.validity == "unavailable"
    assert contract.values_m == ()
    assert "not computed" in contract.reason.lower()


def test_opensim_export_complete_keeps_ik_status_separate() -> None:
    pipeline = resolve_opensim_pipeline_status(
        export_completed=True,
        trc_exported=True,
        mot_exported=True,
    )
    ik = resolve_opensim_ik_status(export_completed=True, mapped_trc_ready=True)
    pair = format_opensim_status_pair(pipeline, ik)
    assert pipeline.key == PIPELINE_COMPLETE
    assert pair.short_label == "Export Complete · IK unavailable"
    assert "OpenSim Pipeline: Complete" in pair.tooltip
    assert "IK: Partial" in pair.tooltip
    assert "Inverse Kinematics (not completed)" in pair.tooltip


def test_unvalidated_contract_from_partial_namespace() -> None:
    sm = SimpleNamespace(
        reportable=False,
        experimental_note="Monocular COM–BoS is experimental.",
        per_frame=[
            SimpleNamespace(time_s=0.0, stability_margin_m=0.02, stability_state="Stable"),
            SimpleNamespace(time_s=0.1, stability_margin_m=-0.01, stability_state="Unstable"),
        ],
        mean_margin_m=0.005,
    )
    contract = balance_display_contract(sm)
    assert contract.validity == VALIDITY_UNVALIDATED
    assert contract.display_mode == DISPLAY_PREVIEW
    assert contract.classification_supported is False
    assert contract.values_m == (0.02, -0.01)
    assert np.isclose(contract.mean_margin_m or 0.0, 0.005)
