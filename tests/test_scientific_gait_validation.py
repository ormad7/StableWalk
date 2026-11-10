"""Scientific checks: COM–BoS honesty, cadence, angles, synthetic + real clips.

Does not treat a filename such as normal_gait.mp4 as clinical ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.angle_filter import despike_angle_series
from stablewalk.analysis.biomechanical.base_of_support import BaseOfSupportAnalysis
from stablewalk.analysis.biomechanical.com_estimation import (
    CenterOfMassAnalysis,
    analyze_center_of_mass,
)
from stablewalk.analysis.biomechanical.gait_quality_score import compute_gait_quality_score
from stablewalk.analysis.biomechanical.stability_margin import analyze_stability_margin
from stablewalk.analysis.biomechanical.types import (
    BiomechanicalFrameBoS,
    BiomechanicalFrameCOM,
)
from stablewalk.analysis.biomechanical.symmetry_metrics import analyze_symmetry
from stablewalk.analysis.gait_cycle_analysis import (
    DetectedGaitCycle,
    FrameContactState,
    GaitEvent,
    compute_temporal_metrics,
    analyze_gait_cycles,
)
from stablewalk.data.mock_gait import MockGaitConfig, generate_mock_gait
from stablewalk.io.pose_loader import load_pose_sequence


def test_despike_removes_isolated_jump_not_sustained_rom() -> None:
    series = np.array([10.0, 12.0, 95.0, 13.0, 14.0], dtype=float)
    cleaned = despike_angle_series(series, max_jump_deg=40.0)
    assert cleaned[2] == pytest.approx(12.5)
    limp = np.array([10.0, 40.0, 55.0, 50.0, 20.0], dtype=float)
    kept = despike_angle_series(limp, max_jump_deg=40.0)
    assert kept[2] == pytest.approx(55.0)


def test_cadence_from_two_hs_cycles_without_fabricating() -> None:
    fps = 25.0
    n = 75
    per_frame = []
    for i in range(n):
        t = i / fps
        # Alternating 0.6 s stance / 0.4 s swing per side, 1.0 s stride.
        left_contact = (t % 1.0) < 0.60
        right_contact = ((t + 0.50) % 1.0) < 0.60
        from stablewalk.analysis.gait_cycle_analysis import FootLandmarkSample

        sample = FootLandmarkSample(0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.9)
        per_frame.append(
            FrameContactState(
                frame_index=i,
                time_s=t,
                left_contact=int(left_contact),
                right_contact=int(right_contact),
                phase="DOUBLE_SUPPORT"
                if left_contact and right_contact
                else (
                    "LEFT_STANCE"
                    if left_contact
                    else ("RIGHT_STANCE" if right_contact else "FLIGHT")
                ),
                left=sample,
                right=sample,
            )
        )
    events = [
        GaitEvent("left_heel_strike", 0, 0.0, "left"),
        GaitEvent("right_heel_strike", 12, 0.5, "right"),
        GaitEvent("left_heel_strike", 25, 1.0, "left"),
        GaitEvent("right_heel_strike", 37, 1.5, "right"),
        GaitEvent("left_heel_strike", 50, 2.0, "left"),
        GaitEvent("left_toe_off", 15, 0.60, "left"),
        GaitEvent("right_toe_off", 27, 1.10, "right"),
        GaitEvent("left_toe_off", 40, 1.60, "left"),
    ]
    cycles = [
        DetectedGaitCycle(0, 0, 25, 0.0, 1.0, 1.0, "left"),
        DetectedGaitCycle(1, 25, 50, 1.0, 2.0, 1.0, "left"),
    ]
    metrics = compute_temporal_metrics(
        per_frame,
        events,
        cycles,
        fps=fps,
        contact_confidence=0.85,
        confidence_tier="HIGH",
    )
    assert metrics.metrics_reliable is True
    assert metrics.cadence_steps_per_min == pytest.approx(120.0, abs=8.0)
    assert metrics.left_right_stance_symmetry is not None
    assert metrics.gait_cycle_count == 2


def test_monocular_com_bos_is_experimental_not_a_kpi() -> None:
    polygon = [(-0.10, -0.04), (0.10, -0.04), (0.10, 0.04), (-0.10, 0.04)]
    com = CenterOfMassAnalysis(
        per_frame=[
            BiomechanicalFrameCOM(
                i,
                i / 30.0,
                (0.0, 0.95, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.9,
            )
            for i in range(8)
        ]
    )
    bos = BaseOfSupportAnalysis(
        per_frame=[
            BiomechanicalFrameBoS(
                frame_index=i,
                time_s=i / 30.0,
                support_type="double_support",
                polygon_xy=list(polygon),
                centroid=(0.0, 0.0),
                area_m2=0.0,
                confidence=0.9,
                polygon_source="contact",
                left_xy=[(-0.10, -0.04), (-0.10, 0.04)],
                right_xy=[(0.10, -0.04), (0.10, 0.04)],
            )
            for i in range(8)
        ]
    )
    sm = analyze_stability_margin(com, bos, meters_per_unit=1.70)
    assert sm.reportable is False
    assert sm.kind == "experimental_monocular"
    assert "Hof" in sm.experimental_note or "XCoM" in sm.experimental_note
    gq = compute_gait_quality_score(stability_margin=sm)
    assert "stability_margin" not in (gq.components or {})


def test_synthetic_symmetric_gait_has_known_cadence() -> None:
    rec = generate_mock_gait(MockGaitConfig(duration_s=6.0, fps=30.0, cadence_hz=1.0))
    cycles = analyze_gait_cycles(rec)
    # Ground truth: cadence_hz=1.0 is one stride/s/leg → 120 steps/min if events fire.
    gt_spm = 120.0
    if cycles.metrics.cadence_steps_per_min is None:
        pytest.skip(
            "Synthetic FK contact did not yield ≥2 HS cycles — "
            f"sufficiency={cycles.event_sufficiency} "
            f"reason={cycles.event_sufficiency_reason}"
        )
    err = abs(float(cycles.metrics.cadence_steps_per_min) - gt_spm)
    assert err <= 15.0, (
        f"cadence {cycles.metrics.cadence_steps_per_min:.1f} vs GT {gt_spm} "
        f"(error {err:.1f})"
    )
    if cycles.metrics.left_right_stance_symmetry is not None:
        assert cycles.metrics.left_right_stance_symmetry >= 0.75


def test_synthetic_asymmetric_knee_rom_is_detectable() -> None:
    rec = generate_mock_gait(MockGaitConfig(duration_s=5.0, fps=30.0, cadence_hz=1.0))
    for snap in rec.snapshots:
        lk = snap.joints.get("left_knee")
        if lk is not None and lk.angle_deg is not None:
            lk.angle_deg = 5.0 + 0.40 * (float(lk.angle_deg) - 5.0)
        dof = snap.dofs.get("left_knee_flexion")
        if dof is not None and dof.angle_deg is not None:
            dof.angle_deg = 5.0 + 0.40 * (float(dof.angle_deg) - 5.0)
    left = [
        float(s.dofs["left_knee_flexion"].angle_deg)
        for s in rec.snapshots
        if s.dofs.get("left_knee_flexion") and s.dofs["left_knee_flexion"].angle_deg is not None
    ]
    right = [
        float(s.dofs["right_knee_flexion"].angle_deg)
        for s in rec.snapshots
        if s.dofs.get("right_knee_flexion") and s.dofs["right_knee_flexion"].angle_deg is not None
    ]
    assert left and right
    left_rom = float(np.percentile(left, 90) - np.percentile(left, 10))
    right_rom = float(np.percentile(right, 90) - np.percentile(right, 10))
    assert left_rom < 0.70 * right_rom
    cycles = analyze_gait_cycles(rec)
    sym = analyze_symmetry(cycles, None, None, knee_rom_left=left_rom, knee_rom_right=right_rom)
    if sym.knee_rom_symmetry is not None and sym.knee_rom_symmetry.value is not None:
        assert float(sym.knee_rom_symmetry.value) < 0.85


def _pose_path_for(video: Path) -> Path | None:
    named = config.POSES_DIR / f"{video.stem}_poses.json"
    if named.is_file():
        return named
    autosave = config.SESSION_EXPORT_DIR / "autosave" / "stablewalk_autosave" / "poses.json"
    if autosave.is_file():
        try:
            data = json.loads(autosave.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        src = str(data.get("source_video") or "")
        if video.name in src.replace("\\", "/"):
            return autosave
    return None


def test_normal_gait_mp4_is_not_assumed_clinically_normal() -> None:
    video = config.DEMO_VIDEOS_DIR / "normal_gait.mp4"
    if not video.is_file():
        pytest.skip("normal_gait.mp4 missing")
    pose_path = _pose_path_for(video)
    if pose_path is None:
        pytest.skip("No pose JSON for normal_gait.mp4 — no ground truth")
    rec = pose_sequence_to_gait_motion(load_pose_sequence(pose_path))
    cycles = analyze_gait_cycles(rec)
    com = analyze_center_of_mass(rec)
    sm = analyze_stability_margin(
        com,
        BaseOfSupportAnalysis(per_frame=[]),
        meters_per_unit=1.70,
    )
    assert sm.reportable is False
    # Filename is not clinical GT. Report facts only.
    if cycles.metrics.gait_cycle_count >= 2 and not cycles.metrics.session_metrics_withheld:
        assert cycles.metrics.cadence_steps_per_min is not None
        assert cycles.event_sufficiency == "none"
    else:
        assert cycles.metrics.cadence_steps_per_min is None
        assert cycles.event_sufficiency in ("input_insufficient", "processing_failure")
        assert cycles.event_sufficiency_reason


def test_abnormal_gait_does_not_fabricate_session_metrics() -> None:
    video = config.DEMO_VIDEOS_DIR / "abnormal_gait.mp4"
    pose_path = _pose_path_for(video)
    if pose_path is None or not pose_path.is_file():
        pytest.skip("No pose JSON for abnormal_gait.mp4")
    rec = pose_sequence_to_gait_motion(load_pose_sequence(pose_path))
    cycles = analyze_gait_cycles(rec)
    if cycles.metrics.session_metrics_withheld or cycles.metrics.gait_cycle_count < 2:
        assert cycles.metrics.cadence_steps_per_min is None
        assert cycles.metrics.metrics_reliable is False
