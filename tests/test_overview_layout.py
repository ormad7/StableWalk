"""Overview tab layout rhythm — proportions, gutters, and viz/metrics split."""



from __future__ import annotations



import unittest



from stablewalk.ui.theme import OVERVIEW_COL_GUTTER, PAD_MD

from stablewalk.ui.tk.dashboard_sections import (

    SEC1_METRICS_ROW_WEIGHT,

    SEC1_SKELETON_WEIGHT,

    SEC1_SUMMARY_WEIGHT,

    SEC1_TRAJ_PATH_WEIGHT,

    SEC1_TRAJ_SKELETON_WEIGHT,

    SEC1_TRAJ_SUMMARY_WEIGHT,

    SEC1_TRAJ_VIDEO_WEIGHT,

    SEC1_TRAJ_VIZ_ROW_MINSIZE,

    SEC1_VIDEO_WEIGHT,

    SEC1_VIZ_ROW_MINSIZE,

    SEC1_VIZ_ROW_WEIGHT,

    overview_metric_padx,

    overview_panel_padx,

)





class OverviewLayoutTests(unittest.TestCase):

    def test_column_weights_are_balanced(self) -> None:

        # Default Side-by-Side: Video 34 | Skeleton 36 | Path 30.

        self.assertEqual(SEC1_VIDEO_WEIGHT, 34)

        self.assertEqual(SEC1_SKELETON_WEIGHT, 36)

        self.assertEqual(SEC1_SUMMARY_WEIGHT, 0)

        self.assertEqual(SEC1_TRAJ_VIDEO_WEIGHT, 34)

        self.assertEqual(SEC1_TRAJ_SKELETON_WEIGHT, 36)

        self.assertEqual(SEC1_TRAJ_PATH_WEIGHT, 30)

        self.assertEqual(

            SEC1_TRAJ_VIDEO_WEIGHT

            + SEC1_TRAJ_SKELETON_WEIGHT

            + SEC1_TRAJ_PATH_WEIGHT,

            100,

        )

        self.assertEqual(SEC1_TRAJ_SUMMARY_WEIGHT, 0)

        self.assertGreaterEqual(SEC1_TRAJ_VIZ_ROW_MINSIZE, SEC1_VIZ_ROW_MINSIZE)



    def test_vertical_split_prioritizes_visuals(self) -> None:

        self.assertEqual(SEC1_VIZ_ROW_WEIGHT, 78)

        self.assertEqual(SEC1_METRICS_ROW_WEIGHT, 22)

        self.assertEqual(SEC1_VIZ_ROW_WEIGHT + SEC1_METRICS_ROW_WEIGHT, 100)

        from stablewalk.ui.tk.dashboard_sections import SEC1_JOINT_MOTION_ROW_WEIGHT

        # Joint Graphs only take height when expanded; default is collapsed (0).
        self.assertGreater(SEC1_JOINT_MOTION_ROW_WEIGHT, 0)
        self.assertLess(SEC1_JOINT_MOTION_ROW_WEIGHT, SEC1_VIZ_ROW_WEIGHT)



    def test_overview_panel_padx_is_symmetric(self) -> None:

        half = OVERVIEW_COL_GUTTER // 2

        self.assertEqual(overview_panel_padx(0), (0, half))

        self.assertEqual(overview_panel_padx(1), (half, half))

        self.assertEqual(overview_panel_padx(2), (half, 0))

        self.assertEqual(

            overview_panel_padx(0)[1] + overview_panel_padx(1)[0],

            OVERVIEW_COL_GUTTER,

        )



    def test_overview_metric_padx_is_symmetric(self) -> None:

        left, right = overview_metric_padx(0)

        self.assertEqual(left, 0)

        self.assertGreater(right, 0)

        mid_left, mid_right = overview_metric_padx(1)

        self.assertEqual(mid_left, mid_right)



    def test_overview_gutter_uses_theme_md(self) -> None:

        self.assertEqual(OVERVIEW_COL_GUTTER, PAD_MD)



    def test_joint_motion_row_weight_collapsed_vs_expanded(self) -> None:
        import tkinter as tk

        from stablewalk.ui.tk.dashboard_sections import (
            SEC1_JOINT_MOTION_ROW_WEIGHT,
            apply_overview_joint_motion_row_weight,
        )

        root = tk.Tk()
        root.withdraw()
        try:
            parent = tk.Frame(root)
            apply_overview_joint_motion_row_weight(parent, expanded=False, row=2)
            self.assertEqual(int(parent.grid_rowconfigure(2).get("weight", 0)), 0)
            apply_overview_joint_motion_row_weight(parent, expanded=True, row=2)
            self.assertEqual(
                int(parent.grid_rowconfigure(2).get("weight", 0)),
                SEC1_JOINT_MOTION_ROW_WEIGHT,
            )
        finally:
            root.destroy()


if __name__ == "__main__":

    unittest.main()


