"""Dimensional consistency: angular vs linear jerk, labels, export columns."""

from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path

import numpy as np

from stablewalk.analysis.asymmetry3d.moments import (
    KIND_ESTIMATED,
    KIND_KINEMATIC,
    build_moment_report,
    proximal_inertia_kg_m2,
)
from stablewalk.analysis.asymmetry3d.kinematics import build_kinematics
from stablewalk.analysis.spectral.matrices import correlation_from_covariance, lagged_matrix
from stablewalk.models.gait_motion import (
    DofSample,
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.ui.viewers.joint_motion_analysis_chart import (
    CSV_FIELDNAMES,
    MOTION_ANALYSIS_TABS,
    MOTION_WORKSPACE_METRICS,
    build_joint_motion_series,
    write_joint_motion_csv,
)


def _analytic_jerk_recording(
    *,
    fps: float = 100.0,
    t0: float = 0.20,
    t1: float = 1.80,
    j_lin: float = 12.0,
    j_ang: float = 48.0,
) -> GaitMotionRecording:
    """x = (1/6) j_lin t³, ω = (1/2) j_ang t². Interior finite differences are exact."""
    n = int(round((t1 - t0) * fps)) + 1
    snaps: list[SkeletonSnapshot] = []
    for i in range(n):
        t = t0 + i / fps
        x = (1.0 / 6.0) * j_lin * (t ** 3)
        omega = 0.5 * j_ang * (t ** 2)
        angle = (1.0 / 6.0) * j_ang * (t ** 3)
        knee = Vec3(x, 0.80, 0.10)
        hip = Vec3(x, 1.20, 0.10)
        rad = math.radians(float(angle))
        ankle = Vec3(x + 0.40 * math.sin(rad), 0.80 - 0.40 * math.cos(rad), 0.10)
        snaps.append(
            SkeletonSnapshot(
                frame_index=i,
                time_s=t,
                joints={
                    "right_hip": JointSample(joint_id="right_hip", position=hip),
                    "right_knee": JointSample(
                        joint_id="right_knee",
                        position=knee,
                        angle_deg=angle,
                    ),
                    "right_ankle": JointSample(
                        joint_id="right_ankle", position=ankle
                    ),
                },
                dofs={
                    "right_knee_flexion": DofSample(
                        dof_id="right_knee_flexion",
                        angle_deg=angle,
                        velocity_deg_s=omega,
                        joint_id="right_knee",
                    )
                },
            )
        )
    return GaitMotionRecording(source="analytic_jerk", fps=fps, snapshots=snaps)


def test_linear_jerk_is_m_s3_angular_jerk_is_deg_s3() -> None:
    j_lin, j_ang = 12.0, 48.0
    rec = _analytic_jerk_recording(j_lin=j_lin, j_ang=j_ang)
    series = build_joint_motion_series(rec, "right_knee")
    assert series is not None
    ang_ok = np.isfinite(series.angular_jerk_deg_s3)
    lin_ok = np.isfinite(series.jerk_m_s3)
    assert ang_ok.sum() >= 8 and lin_ok.sum() >= 8
    np.testing.assert_allclose(
        series.jerk_m_s3[lin_ok], j_lin, rtol=1e-3, atol=0.05
    )
    np.testing.assert_allclose(
        series.angular_jerk_deg_s3[ang_ok], j_ang, rtol=1e-3, atol=0.15
    )
    assert "jerk_m_s3" in series.__dataclass_fields__
    assert "angular_jerk_deg_s3" in series.__dataclass_fields__


def test_plot_labels_and_csv_columns_agree_on_jerk_units() -> None:
    tabs = {mid: (title, ylabel) for mid, title, ylabel in MOTION_ANALYSIS_TABS}
    assert "m/s³" in tabs["jerk"][1]
    assert "°/s³" in tabs["angular_jerk"][1] or "deg" in tabs["angular_jerk"][1].lower()
    assert "m/s³" not in tabs["angular_jerk"][1]
    assert "°" not in tabs["jerk"][1] or "m/s" in tabs["jerk"][1]
    workspace = {mid: ylabel for mid, _title, ylabel in MOTION_WORKSPACE_METRICS}
    assert "m/s³" in workspace["jerk"]
    assert "jerk_m_s3" in CSV_FIELDNAMES
    assert "angular_jerk_deg_s3" in CSV_FIELDNAMES
    rec = _analytic_jerk_recording()
    series = build_joint_motion_series(rec, "right_knee")
    assert series is not None
    from stablewalk.ui.viewers.joint_motion_analysis_chart import JointMotionBundle

    bundle = JointMotionBundle(series=(series,), times_s=series.times_s)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "jerk.csv"
        write_joint_motion_csv(bundle, path)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert "jerk_m_s3" in rows[0]
        assert "angular_jerk_deg_s3" in rows[0]
        assert "linear_accel_m_s2" in rows[0]
        assert "angular_acceleration_deg_s2" in rows[0]


def test_pearson_r0_has_unit_diagonal() -> None:
    fps = 50.0
    t = np.arange(0.0, 6.0, 1.0 / fps)
    data = np.column_stack(
        [
            np.sin(2 * np.pi * 1.0 * t),
            3.0 * np.sin(2 * np.pi * 1.0 * t + 0.2),
            0.5 * np.cos(2 * np.pi * 2.0 * t),
        ]
    )
    r0 = lagged_matrix(data, 0, fps)
    diag = np.diag(r0.correlation)
    np.testing.assert_allclose(diag, np.ones_like(diag), atol=1e-12)
    recon = correlation_from_covariance(r0.covariance)
    np.testing.assert_allclose(r0.correlation, recon, atol=1e-10, equal_nan=True)


def test_estimated_moment_is_I_times_alpha_rad_and_labeled_nm() -> None:
    from tests.test_asymmetry3d import _walking_recording

    rec, _cycles = _walking_recording(n_strides=4)
    kin = build_kinematics(rec)
    report = build_moment_report(
        kin, subject_mass_kg=70.0, subject_height_m=1.70
    )
    assert report.measured_available is False
    assert report.estimated_available is True
    inertia = proximal_inertia_kg_m2("knee", mass_kg=70.0, height_m=1.70)
    assert inertia is not None
    left = next(s for s in report.estimated if s.joint == "knee" and s.side == "left")
    assert left.kind == KIND_ESTIMATED
    idx = kin.dof_index("left_knee_flexion")
    assert idx is not None
    expected = inertia * np.deg2rad(kin.alpha_deg_s2[:, idx])
    np.testing.assert_allclose(left.moment_nm, expected, equal_nan=True, atol=1e-12)
    no_mass = build_moment_report(kin, subject_mass_kg=None)
    assert no_mass.estimated_available is False
    assert no_mass.kinematic
    assert no_mass.kinematic[0].kind == KIND_KINEMATIC


def test_occluded_joint_keeps_nan_not_zero() -> None:
    rec = _analytic_jerk_recording()
    rec.snapshots[10].joints = {}
    rec.snapshots[10].dofs["right_knee_flexion"] = DofSample(
        dof_id="right_knee_flexion", angle_deg=None, velocity_deg_s=None
    )
    series = build_joint_motion_series(rec, "right_knee")
    assert series is not None
    assert math.isnan(float(series.x_m[10]))
    assert math.isnan(float(series.angle_deg[10]))
    assert not np.all(series.x_m == 0.0)


def test_fps_does_not_change_analytic_jerk_units() -> None:
    for fps in (25.0, 50.0, 100.0):
        rec = _analytic_jerk_recording(fps=fps, j_lin=12.0, j_ang=48.0)
        series = build_joint_motion_series(rec, "right_knee")
        assert series is not None
        ang_ok = np.isfinite(series.angular_jerk_deg_s3)
        lin_ok = np.isfinite(series.jerk_m_s3)
        assert ang_ok.sum() >= 5 and lin_ok.sum() >= 5
        np.testing.assert_allclose(
            series.jerk_m_s3[lin_ok], 12.0, rtol=0.05, atol=0.4
        )
        np.testing.assert_allclose(
            series.angular_jerk_deg_s3[ang_ok], 48.0, rtol=0.05, atol=1.5
        )
