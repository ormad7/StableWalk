"""Professional camera framing for the true 3D skeleton viewer."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")
from matplotlib.figure import Figure

from stablewalk.pose.skeleton_3d import Joint3D, Skeleton3D
from stablewalk.ui.viewers.plot_3d import (
    DEFAULT_SKELETON_CAMERA_AZIM,
    DEFAULT_SKELETON_CAMERA_ELEV,
    apply_display_limits,
    remember_skeleton_camera,
    setup_3d_axes,
    smooth_reset_skeleton_camera,
)


def _skeleton() -> Skeleton3D:
    points = {
        "nose": (0.10, 1.80, 0.04),
        "mid_shoulder": (0.08, 1.52, 0.02),
        "left_shoulder": (-0.20, 1.50, 0.03),
        "right_shoulder": (0.36, 1.54, 0.01),
        "mid_hip": (0.05, 0.98, 0.00),
        "left_hip": (-0.12, 0.96, 0.02),
        "right_hip": (0.22, 1.00, -0.02),
        "left_ankle": (-0.16, 0.08, 0.10),
        "right_ankle": (0.25, 0.10, -0.08),
    }
    return Skeleton3D(
        joints={
            name: Joint3D(name=name, x=x, y=y, z=z)
            for name, (x, y, z) in points.items()
        }
    )


def test_default_camera_uses_professional_perspective() -> None:
    ax = Figure().add_subplot(111, projection="3d")
    setup_3d_axes(ax)
    assert ax.elev == pytest.approx(DEFAULT_SKELETON_CAMERA_ELEV)
    assert ax.azim == pytest.approx(DEFAULT_SKELETON_CAMERA_AZIM)
    assert ax.name == "3d"


def test_skeleton_limits_are_centered_equal_and_unclipped() -> None:
    ax = Figure().add_subplot(111, projection="3d")
    skeleton = _skeleton()
    apply_display_limits(ax, skeleton)

    limits = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())
    spans = [hi - lo for lo, hi in limits]
    assert max(spans) - min(spans) < 1e-9
    body_span = max(
        max(joint.x for joint in skeleton.joints.values())
        - min(joint.x for joint in skeleton.joints.values()),
        max(joint.y for joint in skeleton.joints.values())
        - min(joint.y for joint in skeleton.joints.values()),
        max(joint.z for joint in skeleton.joints.values())
        - min(joint.z for joint in skeleton.joints.values()),
    )
    # Auto-frame targets ~70% fill with an extra clip margin.
    fill = body_span / spans[0]
    assert 0.58 <= fill <= 0.78
    for axis, values in zip(
        limits,
        (
            [joint.x for joint in skeleton.joints.values()],
            [joint.y for joint in skeleton.joints.values()],
            [joint.z for joint in skeleton.joints.values()],
        ),
        strict=True,
    ):
        assert axis[0] < min(values)
        assert axis[1] > max(values)
        assert (axis[0] + axis[1]) * 0.5 == pytest.approx(
            (min(values) + max(values)) * 0.5
        )


def test_camera_is_remembered_and_reset_is_interpolated() -> None:
    ax = Figure().add_subplot(111, projection="3d")
    setup_3d_axes(ax)
    ax.view_init(elev=62.0, azim=-130.0)
    remember_skeleton_camera(ax)
    setup_3d_axes(ax)
    assert ax.elev == pytest.approx(62.0)
    assert ax.azim == pytest.approx(-130.0)

    class Canvas:
        draws = 0

        def draw_idle(self) -> None:
            self.draws += 1

    class Scheduler:
        calls = 0
        jobs: list[object] = []

        def after(self, _delay: int, callback) -> str:
            self.calls += 1
            job = f"job-{self.calls}"
            self.jobs.append(job)
            callback()
            return job

        def after_cancel(self, _job: object) -> None:
            return None

    canvas = Canvas()
    scheduler = Scheduler()
    smooth_reset_skeleton_camera(
        ax,
        canvas=canvas,
        scheduler=scheduler,
        duration_ms=120,
        steps=6,
    )
    assert scheduler.calls == 6
    assert canvas.draws == 7
    assert ax.elev == pytest.approx(DEFAULT_SKELETON_CAMERA_ELEV)
    assert ax.azim == pytest.approx(DEFAULT_SKELETON_CAMERA_AZIM)
    assert ax._stablewalk_skeleton_camera == (
        DEFAULT_SKELETON_CAMERA_ELEV,
        DEFAULT_SKELETON_CAMERA_AZIM,
    )

