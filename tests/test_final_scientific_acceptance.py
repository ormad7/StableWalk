"""Scientific acceptance: real clips, export match, isolation, short video, stability honesty."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.analysis_summary import build_analysis_summary
from stablewalk.analysis.asymmetry3d.export import export_asymmetry_json
from stablewalk.analysis.asymmetry3d.kinematics import PROVENANCE_INFERRED
from stablewalk.analysis.asymmetry3d.report import analyze_asymmetry3d
from stablewalk.analysis.biomechanical.base_of_support import analyze_base_of_support
from stablewalk.analysis.biomechanical.com_estimation import analyze_center_of_mass
from stablewalk.analysis.biomechanical.stability_margin import analyze_stability_margin
from stablewalk.analysis.foot_contact_analysis import analyze_foot_contact
from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.analysis.spectral.autocorrelation import normalized_autocorrelation
from stablewalk.analysis.spectral.export import export_spectral_json
from stablewalk.analysis.spectral.features import SESSION_VECTOR_KEYS
from stablewalk.analysis.spectral.report import analyze_spectral
from stablewalk.io.analysis_summary_export import export_analysis_summary_json
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.video_source import check_source_format, quick_validate_source
from stablewalk.ui.overview_clinical_summary import NOT_AVAILABLE, extract_overview_summary_fields
from stablewalk.ui.viewers.knee_angle_chart import _interior_to_flexion_deg


def _meta(path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "cannot_open"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "fps": fps,
        "frames": n,
        "width": w,
        "height": h,
        "duration_s": n / fps if fps > 1e-6 else 0.0,
    }


def test_demo_videos_open() -> None:
    for name in ("normal_gait.mp4", "abnormal_gait.mp4"):
        path = config.DEMO_VIDEOS_DIR / name
        assert path.is_file(), name
        meta = _meta(path)
        assert "error" not in meta, name
        assert meta["frames"] >= 2, name
        assert meta["fps"] > 1.0, name
        assert meta["width"] > 0 and meta["height"] > 0, name


def test_athletic_walking_opens_if_present() -> None:
    path = config.DEMO_VIDEOS_DIR / "athletic_walking.mp4"
    if not path.is_file():
        pytest.skip("athletic_walking.mp4 not on disk")
    meta = _meta(path)
    assert "error" not in meta
    assert meta["frames"] >= 2


def test_extremely_short_candidate_is_scientifically_thin() -> None:
    path = config.DEMO_VIDEOS_DIR / "candidates" / "weightgait_limping_gait.mp4"
    if not path.is_file():
        pytest.skip("limping candidate missing")
    meta = _meta(path)
    assert int(meta["frames"]) < 40
    assert float(meta["duration_s"]) < 4.0
    ok, msg = quick_validate_source(str(path))
    assert ok, msg


def test_missing_video_file_is_detected() -> None:
    path = config.DEMO_VIDEOS_DIR / "__no_such_clip__.mp4"
    assert not path.is_file()
    ok, msg = check_source_format(str(path))
    assert ok is False
    assert "not found" in msg.lower()


def test_invalid_container_has_no_readable_frames(tmp_path: Path) -> None:
    junk = tmp_path / "not_a_video.mp4"
    junk.write_bytes(b"this is not a valid mp4 container")
    ok_fmt, _ = check_source_format(str(junk))
    assert ok_fmt is True
    ok, msg = quick_validate_source(str(junk))
    assert ok is False
    assert "could not open" in msg.lower() or "no readable frames" in msg.lower()


def _load_rec(stem: str):
    pose = config.POSES_DIR / f"{stem}_poses.json"
    if not pose.is_file():
        pytest.skip(f"no pose JSON for {stem}")
    seq = load_pose_sequence(pose)
    rec = pose_sequence_to_gait_motion(seq)
    return seq, rec


def test_normal_and_abnormal_spectral_export_matches_memory(tmp_path: Path) -> None:
    _, rec = _load_rec("normal_gait")
    report = analyze_spectral(rec)
    dest = export_spectral_json(report, tmp_path / "spec.json")
    payload = json.loads(dest.read_text(encoding="utf-8"))
    mem = report.to_dict()
    assert payload["available"] == mem["available"]
    assert payload["fps"] == pytest.approx(report.bundle.fps)
    dumped = payload.get("features", {}).get("session_vector") or {}
    expected = mem.get("features", {}).get("session_vector") or {}
    assert dumped == expected
    ac_key = report.univariate_channel_id
    if ac_key in report.autocorr and ac_key in payload.get("autocorr", {}):
        mem_rho0 = float(report.autocorr[ac_key].rho[0])
        dump_rho0 = float(payload["autocorr"][ac_key]["rho"][0])
        assert dump_rho0 == pytest.approx(mem_rho0, rel=0, abs=1e-12)


def test_two_recordings_do_not_share_spectral_objects() -> None:
    _, a = _load_rec("normal_gait")
    _, b = _load_rec("abnormal_gait")
    ra = analyze_spectral(a)
    rb = analyze_spectral(b)
    assert ra is not rb
    assert ra.bundle is not rb.bundle
    va = ra.features.session_vector if ra.features else {}
    vb = rb.features.session_vector if rb.features else {}
    assert va is not vb
    assert ra.bundle.metadata is not rb.bundle.metadata


def test_autocorrelation_lag0_is_one_on_real_pose() -> None:
    _, rec = _load_rec("normal_gait")
    report = analyze_spectral(rec)
    if not report.available:
        pytest.skip("spectral analysis unavailable on this pose file")
    ac = report.primary_autocorr()
    assert ac is not None
    assert ac.rho.size >= 1
    rho0 = float(ac.rho[0])
    assert math.isfinite(rho0)
    assert rho0 == pytest.approx(1.0, abs=1e-6)
    ch = report.bundle.channels.get(report.univariate_channel_id)
    if ch is not None:
        x = np.asarray(ch.processed, dtype=float)
        x = x[np.isfinite(x)]
        if x.size >= 8:
            x = x - x.mean()
            _lags, rho = normalized_autocorrelation(x, max_lag=min(40, x.size - 1))
            assert float(rho[0]) == pytest.approx(1.0, abs=1e-6)


def test_spectral_features_vector_keys_and_not_disease_labels() -> None:
    _, rec = _load_rec("normal_gait")
    report = analyze_spectral(rec)
    if report.features is None:
        pytest.skip("spectral features unavailable on this pose file")
    keys = set(report.features.session_vector)
    assert set(SESSION_VECTOR_KEYS) <= keys
    warnings = " ".join(report.features.warnings).lower()
    assert "not disease" in warnings or "not a clinical" in warnings or "not disease labels" in warnings


def test_monocular_stability_is_unreportable_on_real_pose() -> None:
    _, rec = _load_rec("normal_gait")
    cycles = analyze_gait_cycles(rec)
    contact = analyze_foot_contact(rec, cycles=cycles)
    com = analyze_center_of_mass(rec, contact)
    bos = analyze_base_of_support(rec, contact)
    sm = analyze_stability_margin(com, bos)
    assert getattr(sm, "reportable", True) is False
    gui = type("G", (), {})()
    gui._biomech_analysis = type("BA", (), {"stability_margin": sm})()
    gui._gait_cycle = None
    gui.gait_motion = rec
    fields = extract_overview_summary_fields(gui)
    text = fields.get("stability_state", ("", ""))[0]
    low = text.lower()
    assert text == NOT_AVAILABLE or "not available" in low
    assert "%" not in text
    ba = type("BA2", (), {"stability_margin": sm, "gait_quality": None, "symmetry": None,
                          "gait_metrics": None, "center_of_mass": None, "joint_rom": None,
                          "video_quality": None})()
    summary = build_analysis_summary(
        source="normal_gait.mp4",
        biomechanical=ba,
        estimated_vgrf=None,
        contact=contact,
        cycles=cycles,
    )
    assert summary.stability_margin is not None
    assert summary.stability_margin.available is False
    assert summary.stability_margin.value == "Not available"
    interp = (summary.scientific_interpretation or "").lower()
    assert "markedly reduced stability" not in interp


def test_exported_summary_json_matches_unreportable_stability(tmp_path: Path) -> None:
    _, rec = _load_rec("normal_gait")
    cycles = analyze_gait_cycles(rec)
    contact = analyze_foot_contact(rec, cycles=cycles)
    com = analyze_center_of_mass(rec, contact)
    bos = analyze_base_of_support(rec, contact)
    sm = analyze_stability_margin(com, bos)
    ba = type(
        "BA",
        (),
        {
            "stability_margin": sm,
            "gait_quality": None,
            "symmetry": None,
            "gait_metrics": None,
            "center_of_mass": None,
            "joint_rom": None,
            "video_quality": None,
        },
    )()
    summary = build_analysis_summary(
        source="normal_gait.mp4",
        biomechanical=ba,
        estimated_vgrf=None,
        contact=contact,
        cycles=cycles,
    )
    dest = export_analysis_summary_json(summary, tmp_path / "summary.json")
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["stability_margin"]["value"] == summary.stability_margin.value
    assert payload["stability_margin"]["available"] is False
    assert "reduced" not in payload["stability_margin"]["value"].lower()


def test_asymmetry_and_cycles_run_on_real_pose(tmp_path: Path) -> None:
    _, rec = _load_rec("normal_gait")
    cycles = analyze_gait_cycles(rec)
    asym = analyze_asymmetry3d(rec, cycles=cycles)
    assert asym is not None
    assert asym.kinematics.provenance == PROVENANCE_INFERRED
    blob = json.dumps(asym.to_dict(), default=str).lower()
    assert "monocular" in blob or "inferred" in blob
    dest = export_asymmetry_json(asym, tmp_path / "asym.json")
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["kinematics"]["provenance"] == asym.kinematics.provenance
    assert payload["available"] == asym.available
    events = getattr(cycles, "events", None) or []
    sufficiency = getattr(cycles, "event_sufficiency", None)
    assert sufficiency is not None or events is not None


def test_joint_angle_convention_interior_to_flexion() -> None:
    assert _interior_to_flexion_deg(180.0) == 0.0
    assert _interior_to_flexion_deg(90.0) == 90.0
    _, rec = _load_rec("normal_gait")
    knees: list[float] = []
    for snap in rec.snapshots:
        dof = snap.dofs.get("left_knee_flexion") if snap.dofs else None
        if dof is not None and getattr(dof, "angle_deg", None) is not None:
            knees.append(float(dof.angle_deg))
    if not knees:
        pytest.skip("no left_knee_flexion samples")
    arr = np.asarray(knees, dtype=float)
    assert np.nanmin(arr) > -30.0
    assert np.nanmax(arr) < 180.0


def test_gait_event_timing_is_monotonic_when_present() -> None:
    _, rec = _load_rec("normal_gait")
    cycles = analyze_gait_cycles(rec)
    events = list(getattr(cycles, "events", None) or [])
    if len(events) < 2:
        pytest.skip(f"insufficient events: sufficiency={getattr(cycles, 'event_sufficiency', None)}")
    times = [float(getattr(e, "time_s", 0.0)) for e in events]
    assert times == sorted(times)
    hs = [e for e in events if "heel_strike" in str(getattr(e, "event_type", ""))]
    if len(hs) >= 2:
        dt = float(hs[1].time_s) - float(hs[0].time_s)
        assert dt > 0.15
        assert dt < 3.0
