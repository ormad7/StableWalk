"""Motion Analysis tabbed metrics — one graph at a time, shared bundle."""

from __future__ import annotations

import unittest

from stablewalk.ui.viewers.joint_motion_analysis_chart import (
    MOTION_ANALYSIS_TABS,
    MOTION_ANALYSIS_TAB_IDS,
    draw_motion_metric_chart,
    metric_tuple_for_tab,
)


class MotionAnalysisTabsTests(unittest.TestCase):
    def test_six_tabs_in_requested_order(self) -> None:
        titles = [t[1] for t in MOTION_ANALYSIS_TABS]
        self.assertEqual(
            titles,
            [
                "Knee Flexion",
                "Velocity",
                "Acceleration",
                "Position",
                "Linear jerk",
                "Angular jerk",
            ],
        )
        self.assertEqual(
            list(MOTION_ANALYSIS_TAB_IDS),
            ["angle", "speed", "accel", "position", "jerk", "angular_jerk"],
        )

    def test_metric_tuple_for_tab(self) -> None:
        mid, title, ylabel = metric_tuple_for_tab("speed")
        self.assertEqual(mid, "speed")
        self.assertEqual(title, "Velocity")
        self.assertIn("m/s", ylabel)

    def test_tab_ylabels_are_complete(self) -> None:
        expected = {
            "angle": "Knee Flexion (°)",
            "speed": "Velocity (m/s)",
            "accel": "Acceleration (m/s²)",
            "position": "Position (m)",
            "jerk": "Linear jerk (m/s³)",
            "angular_jerk": "Angular jerk (°/s³)",
        }
        for mid, title, ylabel in MOTION_ANALYSIS_TABS:
            self.assertEqual(ylabel, expected[mid])
            self.assertTrue(title in ylabel or title == "Velocity")

    def test_draw_single_metric_uses_one_axes(self) -> None:
        import numpy as np
        from matplotlib.figure import Figure

        from stablewalk.ui.viewers.joint_motion_analysis_chart import (
            JointMotionBundle,
            MOTION_METRIC_CHART_MARGINS,
            motion_metric_chart_margins,
        )

        fig = Figure()
        bundle = JointMotionBundle(series=(), times_s=np.array([]))
        draw_motion_metric_chart(fig, bundle, "angle")
        self.assertEqual(len(fig.axes), 1)

        self.assertLessEqual(fig.subplotpars.right, MOTION_METRIC_CHART_MARGINS["right"])
        self.assertGreaterEqual(fig.subplotpars.right, 0.96)
        self.assertGreaterEqual(MOTION_METRIC_CHART_MARGINS["left"], 0.12)
        self.assertLessEqual(MOTION_METRIC_CHART_MARGINS["left"], 0.16)
        self.assertGreaterEqual(motion_metric_chart_margins(fig)["left"], MOTION_METRIC_CHART_MARGINS["left"])

    def test_ylabels_stay_inside_narrow_figure(self) -> None:
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from stablewalk.ui.viewers.joint_motion_analysis_chart import (
            JointMotionBundle,
            JointMotionSeries,
        )

        t = np.linspace(0.0, 1.0, 8)
        series = JointMotionSeries(
            item_id="left_knee",
            label="Left Knee",
            color="#7eb6dc",
            times_s=t,
            frames=np.arange(len(t)),
            angle_deg=np.full(len(t), 180.0),
            omega_deg_s=np.full(len(t), 90.0),
            alpha_deg_s2=np.full(len(t), 40.0),
            speed_m_s=np.full(len(t), 1.2),
            accel_m_s2=np.full(len(t), 0.8),
            jerk_m_s3=np.full(len(t), 0.3),
            angular_jerk_deg_s3=np.full(len(t), 12.0),
            x_m=np.full(len(t), 0.2),
            y_m=np.full(len(t), 0.9),
            z_m=np.full(len(t), 0.1),
        )
        bundle = JointMotionBundle(series=(series,), times_s=t)
        for width_in in (5.0, 4.6):
            for metric_id, _title, ylabel in MOTION_ANALYSIS_TABS:
                fig = Figure(figsize=(width_in, 4.0), dpi=110)
                FigureCanvasAgg(fig)
                draw_motion_metric_chart(fig, bundle, metric_id)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                label = fig.axes[0].yaxis.label
                self.assertEqual(str(label.get_text()), ylabel)
                bbox = label.get_window_extent(renderer=renderer)
                fig_bbox = fig.bbox
                self.assertGreaterEqual(
                    bbox.x0,
                    fig_bbox.x0 - 1.0,
                    f"{ylabel} clipped on the left at {width_in} in",
                )
                self.assertLessEqual(
                    bbox.x1,
                    fig_bbox.x1 + 1.0,
                    f"{ylabel} clipped on the right at {width_in} in",
                )
                self.assertGreaterEqual(
                    bbox.y0,
                    fig_bbox.y0 - 2.0,
                    f"{ylabel} clipped below at {width_in} in",
                )
                self.assertLessEqual(
                    bbox.y1,
                    fig_bbox.y1 + 2.0,
                    f"{ylabel} clipped above at {width_in} in",
                )
                from stablewalk.ui.colors import TEXT_SECONDARY

                self.assertEqual(label.get_color(), TEXT_SECONDARY)
                title = fig.axes[0].title
                title_bbox = title.get_window_extent(renderer=renderer)
                self.assertGreaterEqual(title_bbox.y0, fig_bbox.y0 - 2.0)
                self.assertLessEqual(title_bbox.y1, fig_bbox.y1 + 4.0)


if __name__ == "__main__":
    unittest.main()
