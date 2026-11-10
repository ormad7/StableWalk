"""Research Validation snapshot and report: real outputs only, no fake passes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from stablewalk.analysis.gait_asymmetry_moments import detect_gait_asymmetry_moments
from stablewalk.analysis.research_validation import collect_research_validation
from stablewalk.data.mock_gait import MockGaitConfig, generate_mock_gait
from stablewalk.io.research_report_export import (
    export_research_report_zip,
    format_research_report_markdown,
)
from stablewalk.models.gait_wireframe import GAIT_WIRE_JOINT_COUNT, build_gait_wireframe


def _session(**kwargs):
    return SimpleNamespace(**kwargs)


def test_empty_session_is_unavailable_not_passed() -> None:
    snap = collect_research_validation(_session())
    assert snap.clinical is False
    keys = [item.key for item in snap.items]
    assert keys == [
        "wire16",
        "frames_cycles",
        "acf",
        "spectrum",
        "symmetry",
        "moments",
        "reconstruction3d",
        "confidence",
    ]
    assert all(item.status in {"unavailable", "insufficient", "failed"} for item in snap.items)
    payload = snap.to_dict()
    assert "pass" not in str(payload.get("disclaimer", "")).lower() or "not" in payload["disclaimer"].lower()
    for item in snap.items:
        assert item.status != "passed"  # type: ignore[comparison-overlap]


def test_mock_walk_computes_wire_cycles_and_moments_without_inventing_spectrum() -> None:
    rec = generate_mock_gait(MockGaitConfig(fps=50.0, duration_s=6.0, cadence_hz=1.0))
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles

    cycles = analyze_gait_cycles(rec)
    wire = build_gait_wireframe(rec)
    moments = detect_gait_asymmetry_moments(rec, cycles)
    snap = collect_research_validation(
        _session(
            gait_motion=rec,
            gait_wireframe=wire,
            _gait_cycle=cycles,
            _gait_asymmetry_moments=moments,
        )
    )
    wire_item = snap.item("wire16")
    assert wire_item is not None
    assert wire_item.status == "computed"
    assert wire_item.details["joint_count"] == GAIT_WIRE_JOINT_COUNT
    assert int(wire_item.details["usable_frames"]) > 0

    cycle_item = snap.item("frames_cycles")
    assert cycle_item is not None
    assert cycle_item.status == "computed"
    assert int(cycle_item.details["gait_cycles"]) >= 1
    assert int(cycle_item.details["usable_frames"]) > 0

    acf = snap.item("acf")
    spec = snap.item("spectrum")
    assert acf is not None and spec is not None
    assert acf.status == "unavailable"
    assert spec.status == "unavailable"
    assert "not been computed" in (acf.reason or "").lower() or "not computed" in acf.summary.lower()

    mom = snap.item("moments")
    assert mom is not None
    assert mom.status in {"computed", "insufficient"}
    if mom.status == "computed" and mom.details.get("n_moments", 0) == 0:
        assert "symmetric" in mom.summary.lower()
        recon = snap.item("reconstruction3d")
        assert recon is not None
        assert recon.status != "computed"


def test_numpy_acf_series_does_not_crash_or_invent_a_period() -> None:
    import numpy as np

    acf = SimpleNamespace(
        channel_id="pos:left_ankle:y",
        lags_s=np.array([0.0, 0.02, 0.04]),
        rho=np.array([1.0, 0.4, 0.1]),
        n_samples=80,
        gait_period_s=1.04,
        stride_period_s=1.04,
        reliability_label="moderate",
        insufficient_reason="",
    )
    spec = SimpleNamespace(
        channel_id="pos:left_ankle:y",
        fft_freq_hz=np.linspace(0.0, 4.0, 32),
        fft_power=np.ones(32),
        freq_hz=np.linspace(0.0, 4.0, 16),
        psd=np.ones(16),
        method="welch",
        dominant_hz=1.0,
        dominant_power=1.0,
        spectral_centroid_hz=1.1,
        spectral_bandwidth_hz=0.4,
        spectral_entropy=0.5,
        harmonic_ratio=1.2,
    )
    spectral = SimpleNamespace(
        primary_autocorr=lambda: acf,
        autocorr={"pos:left_ankle:y": acf},
        primary_spectrum=lambda: spec,
        spectra={"pos:left_ankle:y": spec},
        features=None,
        warnings=[],
        available=True,
    )
    snap = collect_research_validation(_session(_spectral_report=spectral))
    item = snap.item("acf")
    spec_item = snap.item("spectrum")
    assert item is not None and spec_item is not None
    assert item.status == "computed"
    assert item.details.get("gait_period_s") == 1.04
    assert spec_item.status == "computed"
    assert "acf" in snap.plot_series
    assert "spectrum" in snap.plot_series


def test_insufficient_acf_does_not_report_a_period() -> None:
    acf = SimpleNamespace(
        channel_id="pos:left_ankle:y",
        lags_s=[0.0, 0.02],
        rho=[1.0, 0.2],
        n_samples=40,
        gait_period_s=None,
        stride_period_s=None,
        reliability_label="insufficient",
        insufficient_reason="insufficient cycles",
    )
    spectral = SimpleNamespace(
        primary_autocorr=lambda: acf,
        autocorr={"pos:left_ankle:y": acf},
        primary_spectrum=lambda: None,
        spectra={},
        features=None,
        warnings=["Need more gait cycles."],
        available=False,
    )
    snap = collect_research_validation(_session(_spectral_report=spectral))
    item = snap.item("acf")
    assert item is not None
    assert item.status == "insufficient"
    assert item.details.get("gait_period_s") is None
    assert "insufficient" in (item.reason or item.summary).lower()


def test_acf_without_peak_is_not_blamed_on_event_cycles() -> None:
    acf = SimpleNamespace(
        channel_id="dof:left_knee_flexion",
        lags_s=[0.0, 0.6, 1.2],
        rho=[1.0, 0.02, -0.1],
        n_samples=120,
        gait_period_s=None,
        stride_period_s=None,
        reliability_label="low",
        insufficient_reason="",
    )
    spectral = SimpleNamespace(
        primary_autocorr=lambda: acf,
        autocorr={"dof:left_knee_flexion": acf},
        primary_spectrum=lambda: None,
        spectra={},
        features=None,
        warnings=[],
        available=True,
    )
    moments = SimpleNamespace(n_left_cycles=3, n_right_cycles=2)
    cycles = SimpleNamespace(
        cycles=[object(), object(), object()],
        metrics=SimpleNamespace(
            contact_confidence=1.0,
            reliability_reason="Computed from 3 complete same-side heel-strike cycles",
        ),
    )
    snap = collect_research_validation(
        _session(
            _spectral_report=spectral,
            _gait_cycle=cycles,
            _gait_asymmetry_moments=moments,
            gait_motion=SimpleNamespace(snapshots=[]),
        )
    )
    acf_item = snap.item("acf")
    cycle_item = snap.item("frames_cycles")
    assert acf_item is not None and cycle_item is not None
    assert acf_item.status == "insufficient"
    assert "event-based" in (acf_item.summary or "").lower()
    assert "peak" in (acf_item.reason or "").lower() or "peak" in acf_item.summary.lower()
    assert "3 L / 2 R" in cycle_item.summary or "3 L / 2 R" in cycle_item.summary.replace(" ", "")
    assert "event-based" in cycle_item.summary.lower()
    assert "not ACF" in cycle_item.summary


def test_failed_spectral_is_failed_not_passed() -> None:
    snap = collect_research_validation(_session(_spectral_error="RuntimeError: boom"))
    acf = snap.item("acf")
    spec = snap.item("spectrum")
    assert acf is not None and spec is not None
    assert acf.status == "failed"
    assert spec.status == "failed"
    assert "boom" in (acf.reason or "")


def test_markdown_and_zip_export_include_metadata_and_failure_reasons(tmp_path: Path) -> None:
    rec = generate_mock_gait(MockGaitConfig(fps=40.0, duration_s=5.0, cadence_hz=1.0))
    rec.source = "demo_walk.mp4"
    wire = build_gait_wireframe(rec)
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles

    cycles = analyze_gait_cycles(rec)
    snap = collect_research_validation(
        _session(gait_motion=rec, gait_wireframe=wire, _gait_cycle=cycles)
    )
    md = format_research_report_markdown(snap)
    assert "Research Validation" in md
    assert "demo_walk.mp4" in md
    assert "**Status:** computed" in md
    assert "**Status:** pass" not in md.lower()
    assert "status: passed" not in md.lower()
    assert "not a diagnosis" in md.lower()
    zip_path = tmp_path / "research.zip"
    export_research_report_zip(snap, zip_path, session=_session(gait_motion=rec))
    with ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "research_validation.json" in names
        assert "research_report.md" in names
        assert "parameter_definitions.json" in names
        assert "video_metadata.json" in names
        assert "failure_reasons.json" in names
        reasons = zf.read("failure_reasons.json").decode("utf-8")
        assert "acf" in reasons
        assert "spectrum" in reasons


def test_results_header_has_download_research_report_action() -> None:
    text = Path("stablewalk/ui/tk/dashboard_summary.py").read_text(encoding="utf-8")
    assert "Download Research Report" in text
    assert "build_research_validation_section" in text
    # JSON/report stay tied to summary.source; research download is not force-disabled.
    assert '"btn_download_research_report",' not in text.split("btn_export_summary_report")[1][:400]
    ui = Path("stablewalk/ui/tk/dashboard_research_validation.py").read_text(encoding="utf-8")
    assert "Open strongest moment on 3D skeleton" in ui
    assert "not a requirement pass/fail" in ui.lower() or "not a requirement pass" in ui.lower()
