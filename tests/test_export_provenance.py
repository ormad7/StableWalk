"""Provenance keys on JSON exports — formulas, quality, unavailable reasons."""

from __future__ import annotations

from pathlib import Path

from stablewalk.analysis.asymmetry3d.compare import AsymmetryScalar
from stablewalk.analysis.pipeline_validation import parse_ik_marker_errors
from stablewalk.analysis.spectral.features import FeatureRecord


def test_feature_record_export_includes_formula_and_unavailable_reason() -> None:
    missing = FeatureRecord(
        name="stride_cv",
        label="Stride CV",
        value=None,
        units="1",
        definition="Coefficient of variation of detected gait-cycle durations.",
        available=False,
        quality="insufficient",
        scope="session",
        notes=["insufficient cycles"],
    )
    payload = missing.to_dict()
    assert payload["formula_id"] == "stride_cv"
    assert payload["definition"]
    assert payload["unavailable_reason"] == "insufficient cycles"
    assert payload["quality"] == "insufficient"

    ok = FeatureRecord(
        name="dominant_frequency_hz",
        label="Dominant frequency",
        value=1.0,
        units="Hz",
        definition="Dominant frequency (Hz).",
        available=True,
        quality="moderate",
        scope="session",
    )
    assert ok.to_dict()["unavailable_reason"] is None
    assert ok.to_dict()["formula_id"] == "dominant_frequency_hz"


def test_asymmetry_scalar_export_includes_formula_and_unavailable_reason() -> None:
    row = AsymmetryScalar(
        name="step_length",
        display="Step length",
        units="m",
        left=None,
        right=None,
        si_percent=None,
        asi_percent=None,
        si_norm=None,
        quality="insufficient",
        reliable=False,
        note="Spatial step length withheld: camera view is not reportable.",
        formula="SI = 200 * (L - R) / (L + R)",
        n_left_cycles=2,
        n_right_cycles=2,
    )
    payload = row.to_dict()
    assert payload["formula_id"] == "step_length"
    assert payload["formula"]
    assert payload["n_left_cycles"] == 2
    assert payload["unavailable_reason"]
    assert "view" in payload["unavailable_reason"].lower() or "unavailable" in payload[
        "unavailable_reason"
    ].lower()


def test_parse_ik_marker_errors_exposes_failed_frames(tmp_path: Path) -> None:
    sto = tmp_path / "ik_marker_errors.sto"
    sto.write_text(
        "\n".join(
            [
                "ik_marker_errors",
                "nRows=3",
                "nColumns=3",
                "endheader",
                "time marker_error_RMS marker_error_max",
                "0.00 0.02 0.03",
                "0.04 0.21 0.30",
                "0.08 0.04 0.05",
            ]
        ),
        encoding="utf-8",
    )
    summary = parse_ik_marker_errors(sto)
    assert summary is not None
    assert summary.n_frames == 3
    assert summary.n_failed_frames == 1
    assert summary.failed_frames[0]["frame_index"] == 1
    assert summary.failed_frames[0]["time_s"] == 0.04
    assert summary.highest_error_frames[0]["marker_error_rms_m"] == 0.21
    blob = summary.to_dict()
    assert blob["n_failed_frames"] == 1
    assert "frame(s) above" in blob["summary"]
