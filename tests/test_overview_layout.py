"""Overview tab layout rhythm — proportions, gutters, and viz/metrics split."""



from __future__ import annotations



import unittest



from stablewalk.ui.theme import OVERVIEW_COL_GUTTER, PAD_SM

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

        # Default Side-by-Side: Video 45 | Skeleton 55.

        self.assertEqual(SEC1_VIDEO_WEIGHT, 45)

        self.assertEqual(SEC1_SKELETON_WEIGHT, 55)

        self.assertEqual(SEC1_SUMMARY_WEIGHT, 0)

        self.assertEqual(SEC1_TRAJ_VIDEO_WEIGHT, 40)

        self.assertEqual(SEC1_TRAJ_SKELETON_WEIGHT, 30)

        self.assertEqual(SEC1_TRAJ_PATH_WEIGHT, 30)

        self.assertGreater(SEC1_TRAJ_VIDEO_WEIGHT, SEC1_TRAJ_SKELETON_WEIGHT)

        self.assertEqual(SEC1_TRAJ_SKELETON_WEIGHT, SEC1_TRAJ_PATH_WEIGHT)

        self.assertEqual(SEC1_TRAJ_SUMMARY_WEIGHT, 0)

        self.assertGreaterEqual(SEC1_TRAJ_VIZ_ROW_MINSIZE, SEC1_VIZ_ROW_MINSIZE)



    def test_overview_host_viz_row_is_the_expander(self) -> None:
        from stablewalk.ui.tk.dashboard_sections import (
            OVERVIEW_HOST_KPI_ROW,
            OVERVIEW_HOST_VIZ_ROW,
            apply_overview_host_row_weights,
            apply_overview_viz_grid_weights,
        )

        self.assertEqual(OVERVIEW_HOST_VIZ_ROW, 1)
        self.assertEqual(OVERVIEW_HOST_KPI_ROW, 2)

        try:
            root = __import__("tkinter").Tk()
        except Exception as exc:
            self.skipTest(f"Tk unavailable: {exc}")
        root.withdraw()
        try:
            host = __import__("tkinter").Frame(root)
            section = __import__("tkinter").Frame(host)
            apply_overview_host_row_weights(host)
            apply_overview_viz_grid_weights(section)
            self.assertEqual(int(host.grid_rowconfigure(OVERVIEW_HOST_VIZ_ROW)["weight"]), 1)
            self.assertEqual(int(host.grid_rowconfigure(0)["weight"]), 0)
            self.assertEqual(int(host.grid_rowconfigure(OVERVIEW_HOST_KPI_ROW)["weight"]), 0)
            self.assertEqual(int(section.grid_columnconfigure(0)["weight"]), 40)
            self.assertEqual(int(section.grid_columnconfigure(1)["weight"]), 30)
            self.assertEqual(int(section.grid_columnconfigure(2)["weight"]), 30)
            self.assertIn(section.grid_columnconfigure(0)["uniform"], ("", None))
            self.assertIn(section.grid_columnconfigure(1)["uniform"], ("", None))
            self.assertIn(section.grid_columnconfigure(2)["uniform"], ("", None))
            self.assertEqual(int(section.grid_rowconfigure(0)["weight"]), 1)
            self.assertGreaterEqual(
                int(host.grid_rowconfigure(OVERVIEW_HOST_VIZ_ROW)["minsize"]), 200
            )
            from stablewalk.ui.tk.dashboard_sections import (
                OVERVIEW_VIDEO_ROW_SOFT_MIN_H,
            )

            self.assertGreaterEqual(
                int(section.grid_rowconfigure(0)["minsize"]),
                OVERVIEW_VIDEO_ROW_SOFT_MIN_H,
            )
            # Phase / clearance moved out of the viewer section.
            self.assertEqual(int(section.grid_rowconfigure(1)["minsize"] or 0), 0)
            self.assertEqual(int(section.grid_rowconfigure(2)["minsize"] or 0), 0)
        finally:
            root.destroy()

    def test_overview_panels_cannot_collapse_to_zero_height(self) -> None:
        from types import SimpleNamespace

        from stablewalk.ui.tk.dashboard_sections import (
            apply_overview_panel_min_heights,
            apply_overview_viz_grid_weights,
        )

        try:
            root = __import__("tkinter").Tk()
        except Exception as exc:
            self.skipTest(f"Tk unavailable: {exc}")
        root.withdraw()
        try:
            section = __import__("tkinter").Frame(root)
            video = __import__("tkinter").Frame(section, height=0)
            skel = __import__("tkinter").Frame(section, height=0)
            traj = __import__("tkinter").Frame(section, height=0)
            apply_overview_viz_grid_weights(section)
            video.grid(row=0, column=0, sticky="nsew")
            skel.grid(row=0, column=1, sticky="nsew")
            traj.grid(row=0, column=2, sticky="nsew")
            gui = SimpleNamespace(
                _section_visual=section,
                _primary_viz_content_row=0,
                video_frame=video,
                skel_frame=skel,
                overview_traj_panel=traj,
            )
            apply_overview_panel_min_heights(gui)
            from stablewalk.ui.tk.dashboard_sections import OVERVIEW_VIDEO_ROW_SOFT_MIN_H

            self.assertGreaterEqual(
                int(section.grid_rowconfigure(0)["minsize"]),
                OVERVIEW_VIDEO_ROW_SOFT_MIN_H,
            )
            self.assertGreaterEqual(
                int(video.cget("height") or 0), OVERVIEW_VIDEO_ROW_SOFT_MIN_H
            )
            self.assertGreaterEqual(
                int(skel.cget("height") or 0), OVERVIEW_VIDEO_ROW_SOFT_MIN_H
            )
            self.assertGreaterEqual(
                int(traj.cget("height") or 0), OVERVIEW_VIDEO_ROW_SOFT_MIN_H
            )
        finally:
            root.destroy()

    def test_vertical_split_prioritizes_visuals(self) -> None:
        self.assertEqual(SEC1_VIZ_ROW_WEIGHT, 96)
        self.assertEqual(SEC1_METRICS_ROW_WEIGHT, 4)
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



    def test_overview_gutter_uses_theme_sm(self) -> None:

        self.assertEqual(OVERVIEW_COL_GUTTER, PAD_SM)

    def test_overview_sections_use_row_gap_between_blocks(self) -> None:
        from pathlib import Path

        from stablewalk.ui.theme import OVERVIEW_ROW_GAP

        src = (
            Path(__file__).resolve().parents[1]
            / "stablewalk"
            / "ui"
            / "tk"
            / "dashboard_layout.py"
        )
        text = src.read_text(encoding="utf-8")
        self.assertEqual(OVERVIEW_ROW_GAP, PAD_SM)
        self.assertIn("padding=_SKELETON_PANEL_PAD", text)
        self.assertIn("pady=(OVERVIEW_ROW_GAP, 0)", text)
        self.assertIn("pady=(OVERVIEW_ROW_GAP, PAD_XS)", text)
        self.assertIn("build_gait_phase_indicator", text)
        self.assertIn("build_foot_clearance_detail_panel", text)
        self.assertIn("build_overview_clinical_summary", text)
        self.assertIn("build_overview_signal_spectral_panel", text)



    def test_joint_motion_row_weight_collapsed_vs_expanded(self) -> None:
        import tkinter as tk

        from stablewalk.ui.tk.dashboard_sections import (
            SEC1_JOINT_MOTION_ROW_WEIGHT,
            apply_overview_joint_motion_row_weight,
        )

        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk unavailable: {exc}")
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

    def test_overview_viz_chrome_padding_is_shared(self) -> None:
        from stablewalk.ui.theme import (
            OVERVIEW_VIZ_TITLE_PADX,
            OVERVIEW_VIZ_TITLE_PADY,
            OVERVIEW_VIZ_TITLE_SPACER_PX,
            OVERVIEW_VIZ_TOOLBAR_PADX,
            OVERVIEW_VIZ_TOOLBAR_PADY,
            PAD_XS,
        )

        self.assertEqual(OVERVIEW_VIZ_TOOLBAR_PADX, (PAD_XS, 0))
        self.assertEqual(OVERVIEW_VIZ_TOOLBAR_PADY, (PAD_XS, PAD_XS))
        from stablewalk.ui.theme import OVERVIEW_VIZ_TOOLBAR_MIN_H

        self.assertGreaterEqual(OVERVIEW_VIZ_TOOLBAR_MIN_H, 28)
        self.assertGreaterEqual(OVERVIEW_VIZ_TITLE_SPACER_PX, 8)
        self.assertGreaterEqual(OVERVIEW_VIZ_TITLE_PADX, 2)
        self.assertGreaterEqual(OVERVIEW_VIZ_TITLE_PADY, 2)


if __name__ == "__main__":

    unittest.main()


