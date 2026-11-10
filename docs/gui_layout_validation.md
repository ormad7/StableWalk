# StableWalk GUI layout — targeted repair report

Layout-only. Pose/IK/gait math, playback, OpenSim, and exports were not changed.

GUI stack: Tkinter + ttk + Matplotlib TkAgg (`grid`, `PanedWindow`, `place` overflow scroll on Advanced).

## Defects fixed (confirmed in screenshots)

| Area | Defect | Fix |
| --- | --- | --- |
| Overview | Skeleton joint combobox clipped (`Right Knee` → `Right Kn`) and Advanced collided with Path | Independent grid toolbar; combo width capped; Advanced wraps under Joint below 300 px |
| Overview | Path camera + metrics + maximize overflowed into neighbouring chrome | Dedicated Path toolbar; metrics/maximize wrap below 420 px; numbers off the 3D cube |
| Overview | Video maximize sat in the viewport corner | Dedicated Video toolbar row; aspect via existing top-row fit |
| Motion | Compare Joints combobox clipped on 768p | Joint row `minsize` 40/72; graph 380 px floor not applied on short windows; Live data / Compare Joints keep row floors |
| Motion | Legend sat on the series | Knee legend placed above the data band (`bbox_to_anchor=(0, 1.02)`) |
| Biomechanics | Interpretation rail auto-collapsed to a header at 1366 (DPI-scaled 1180→1476) | Chart\|interpretation `PanedWindow`; Key Findings stays on the body; rails no longer auto-collapse; DPI stack cap 1200 |
| Biomechanics | Users could not widen interpretation | Draggable sash; ▶/▼ still expand/collapse each rail |
| Results | Collapsed Processing Details kept a weight-1 void | Collapsed slot `sticky=ew`, row weight 0; preview owns leftover when stacked |
| Compare | Catalog/workspace did not stay proportional; tools overlapped status copy | Sash ~34%; combo max 22 chars; status on its own row; stack floor capped at 1200 |
| Advanced | Small windows hid Session / OpenSim / exports | Existing place-scroll host preserved; one-column below ~1020; scrollbar on 1366 |

## Configurations tested (live Tk, this machine)

| Request | Actual Tk geometry | Notes |
| --- | --- | --- |
| 1366×768 | **1366×768** | Primary laptop pass; all major tabs |
| 1600×900 | **1600×900** | All major tabs |
| 1920×1080 | **1920×1030** | Work-area clamp (taskbar) |
| 2560×1440 | **1924×1030** | Display is 1920×1080; window could not exceed the screen |
| Maximized | **1920×966** | All major tabs |

Also: Windows DPI **125.1%** (`dpi_scale_factor=1.251`), screen **1920×1080**. Fonts were not shrunk to hide clipping.

Harness: `scripts/gui_layout_screenshots.py` plus a second 1366/max recapture after screenshot-driven fixes. PNGs in `docs/gui_layout_screenshots/`.

Compare was mapped with `activate_tab` skipped (that path can freeze the capture harness). Research dataset page was still painted and grabbed.

Pytest: 102 passed / 1 skipped on the layout suites (`test_biomech_layout`, `test_chart_style`, `test_motion_analysis_layout`, `test_results_layout`, `test_research_workspace`, `test_overview_layout`, `test_advanced_workspace`, `test_responsive_window_sizes`, `test_compare_layout`, `test_responsive_layout_regression`).

## Residual (visible in screenshots, not hidden)

- COM Y-axis label still clips on **1366×768** height (`…ative Y (approx. E`); it is fully readable at **1600×900** (`Pelvis-relative Y (approx. BH)`). Users can widen the chart with the new sash.
- Motion kinematics canvas can look empty at frame 0; Compare Joints / Live data / tools are fully visible.
- App header source combobox still truncates `Man walking — stead` (not a visualization-panel toolbar).
- Empty Compare 3D cubes are placeholders until A/B analysis exists; at maximized they are sized in the workspace, not overlapping the catalog.
