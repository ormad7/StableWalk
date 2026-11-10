"""Motion Analysis first-paint must not leave a white / empty axes."""

from __future__ import annotations

import unittest

import numpy as np
from matplotlib.figure import Figure

from stablewalk.ui.colors import PANEL
from stablewalk.ui.viewers.chart_style import apply_chart_panel_style
from stablewalk.ui.viewers.joint_motion_analysis_chart import (
    JointMotionBundle,
    JointMotionSeries,
    draw_motion_metric_chart,
)


def _sample_bundle() -> JointMotionBundle:
    t = np.linspace(0.0, 1.0, 25)
    y = np.sin(t * 6.0) * 20.0 + 30.0
    series = JointMotionSeries(
        item_id="right_knee",
        label="Right Knee",
        color="#5b9fd4",
        times_s=t,
        frames=np.arange(1, 26),
        angle_deg=y,
        omega_deg_s=np.gradient(y, t),
        alpha_deg_s2=np.gradient(np.gradient(y, t), t),
        speed_m_s=np.abs(np.gradient(y, t)) * 0.01,
        accel_m_s2=np.zeros_like(t),
        jerk_m_s3=np.zeros_like(t),
        angular_jerk_deg_s3=np.zeros_like(t),
        x_m=np.zeros_like(t),
        y_m=np.zeros_like(t),
        z_m=np.zeros_like(t),
    )
    return JointMotionBundle(series=(series,), times_s=t)


class MotionGraphFirstPaintTests(unittest.TestCase):
    def test_lazy_axes_styled_dark_not_white(self) -> None:
        fig = Figure(facecolor=PANEL)
        ax = fig.add_subplot(1, 1, 1)
        apply_chart_panel_style(ax)
        face = ax.get_facecolor()
        # Matplotlib returns RGBA 0–1; PANEL is a hex dark blue-gray.
        self.assertLess(face[0] + face[1] + face[2], 1.5)

    def test_draw_metric_renders_series_and_playhead(self) -> None:
        from stablewalk.ui.viewers.chart_playhead import PlayheadState

        fig = Figure()
        bundle = _sample_bundle()
        draw_motion_metric_chart(
            fig,
            bundle,
            "angle",
            playhead=PlayheadState(time_s=0.4, frame_index=10, animating=True),
            show_peaks=True,
        )
        self.assertEqual(len(fig.axes), 1)
        ax = fig.axes[0]
        # Series line present (plus playhead vlines).
        self.assertGreaterEqual(len(ax.lines), 1)
        self.assertTrue(getattr(ax, "_stablewalk_playhead_artists", None))
        self.assertTrue(getattr(ax, "_stablewalk_playhead_value_source", None))


if __name__ == "__main__":
    unittest.main()
