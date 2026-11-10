"""Tests for value provenance and N/A integrity helpers."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.ui.value_provenance import (
    METRIC_DISPLAY_CONTRACTS,
    ProvenanceBundle,
    ValueProvenance,
    format_contract_audit,
    format_na,
    get_provenance_bundle,
    store_provenance,
)


def test_format_na_includes_reason_not_invented_number() -> None:
    text = format_na("joint position missing")
    assert text.startswith("N/A")
    assert "joint position missing" in text
    assert "0.0" not in text


def test_format_na_with_confidence() -> None:
    text = format_na("low ROM confidence", confidence=0.22)
    assert "N/A" in text
    assert "22%" in text


def test_value_provenance_inspect_fields() -> None:
    p = ValueProvenance(
        metric_name="Angular velocity (derived)",
        display_value="12.5 °/s",
        raw_value=12.5,
        units="degrees/second",
        source_frame=29,
        source_video_frame=31,
        source_function="joint_information._angular_velocity_deg_s",
        source_array="DofSample.velocity_deg_s",
        coordinate_system="Same DOF as joint angle",
        filtering="hide if |ω| > 400 °/s",
        confidence=0.81,
        scope="instantaneous",
        tier="derived",
    )
    text = p.format_inspect()
    assert "metric name:" in text
    assert "raw value:" in text
    assert "12.5" in text
    assert "degrees/second" in text
    assert "source frame:" in text
    assert "29" in text
    assert "joint_information._angular_velocity_deg_s" in text


def test_provenance_bundle_on_gui() -> None:
    gui = SimpleNamespace()
    store_provenance(
        gui,
        "cadence",
        ValueProvenance(
            metric_name="Cadence (calculated)",
            display_value="N/A — cadence not computed",
            raw_value=None,
            units="steps/min",
            source_frame=None,
            source_function="app._update_biomechanics_panel",
            unavailable_reason="cadence not computed",
            scope="session",
            tier="calculated",
        ),
    )
    bundle = get_provenance_bundle(gui)
    assert isinstance(bundle, ProvenanceBundle)
    assert "cadence" in bundle.entries
    assert "cadence not computed" in bundle.format_all()


def test_display_contracts_cover_required_metrics() -> None:
    required = {
        "position_xyz",
        "joint_angle",
        "angular_velocity",
        "rom_window",
        "rom_session_robust",
        "rom_cycle",
        "path_span",
        "cadence",
        "gait_phase",
        "contact_state",
        "symmetry",
        "stability_margin",
        "gait_quality",
    }
    assert required.issubset(set(METRIC_DISPLAY_CONTRACTS))
    audit = format_contract_audit()
    assert "pelvis" in audit.lower() or "Root-relative" in audit
    assert "degrees/second" in audit
