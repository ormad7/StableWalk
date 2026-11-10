# StableWalk responsive layout fix report

Layout-only pass. Pose, IK, gait math, OpenSim, playback, and scientific
numerics were not changed. Dark theme, tab navigation, videos, 3D skeleton,
graphs, and Compare pipelines are the same features as before.

## GUI framework (inspected)

The desktop app is **Tkinter + ttk + Matplotlib (TkAgg)**. There is no Qt or
CustomTkinter dashboard.

| Manager | Use |
| --- | --- |
| `grid` | Tab pages, Overview viz, Motion / Biomechanics / Results / Compare / Advanced |
| `pack` | Source bar, transport, some tool rows |
| `tk.PanedWindow` | Overview Video \| Skeleton \| Path; Motion optional path; Compare catalog |
| `place` | Maximize overlays and the tab-transition veil |
| `FigureCanvasTkAgg` | All scientific plots, fitted to a dedicated host frame |

Hardcoded problems found and addressed:

- Motion stretch weights still treated the kinematics **canvas as row 3** after
  live numbers moved out of the plot column (row 2, later row 1).
- Readonly comboboxes used fixed `width=` shorter than their longest label
  (skeleton type, 3D path Display/Coord/View, Overview joint, Spectral channel).
- Kinematics **live ribbon sat in the plot column**; the graph floor was 260 px.
- Window `<Configure>` height-share did not re-run when switching onto Motion,
  so the graph host stayed at its unmapped size.
- Video empty copy had **no wraplength**.
- Compare catalog was a fixed two-column grid (not sash-adjustable).

Existing infrastructure kept:

- DPI tokens and stacking breakpoints: `stablewalk/ui/tk/layout_scale.py`
- Resize debounce 140 ms: `resize_coalesce.py` (skipped while playback is active)
- Wheel zoom / pan: `stablewalk/ui/viewers/chart_navigation.py`
- Panel maximize: `dashboard_panel_maximize.py`
- 3D camera autofit: Overview / skeleton / trajectory viewers (`set_box_aspect`, trajectory autofit)

## Layout changes

### Application shell

- Window floors stay **960×600** logical pixels (usable on 1366×768 at 150%).
- Stacking breakpoints still grow with `dpi_scale_factor` (100 / 125 / 150 / 175%).
- `apply_tab_stretch_weights` now stretches the Motion canvas row
  (`_MOTION_GRAPH_CANVAS_ROW`), not a leftover empty row.
- Combobox character width helper clamps `maximum` even when it is below the
  default minimum, and treats a plain string as one value.

### Overview

- Three-column paned Video | Skeleton | 3D Path on wide layouts; two-plus-one
  or stacked below DPI-aware floors.
- Default sash fractions **0.34 / 0.66** (more even thirds). Saved user sash
  prefs are still honored if present.
- Coordinate Inspector **always stacks/collapses in three-column mode** so it
  does not cover the 3D cube.
- Video empty placeholder wraps to the panel width.
- Maximize buttons and 3D camera tools stay in chrome rows, not on the plot.

### Motion Analysis

- Live min/max/mean/ROM and joint cells live in a **separate Live data panel**
  under the graph (`sticky=ew`, max height 118 px, 24 px when empty).
- Graph figure floor **(9.2 in × 6.8 in)**; host floor **380 px** when the
  workspace is tall enough.
- Metric tabs and Graph/Reset/Export/Measure share **one chrome row** so the
  canvas is no longer stacked under two toolbars.
- Compare Joints is a compact 40 px strip (no LabelFrame title eating height).
- Height-share re-runs on window layout **and** when the Motion tab is selected.
- Spectral / Features / Asymmetry bodies **stack** below the motion-stack
  breakpoint instead of crushing the plot column.
- Channel / scope / joint comboboxes size to their longest label and stretch
  where the row has leftover width.
- Wheel zoom, drag pan, and expand remain on the existing chart tools.

### Biomechanics

- COM, Ground Reaction Force, Balance, Contact, and Gait Phase share a
  stretch chart column (grid floor 280 px).
- Key Findings / interpretation rails cap at 220 / 300 px and collapse or
  stack before they steal graph width. Stacked side rows cap at 200 px.

### Results

- Processing Details stays a **collapsible** section (collapsed on empty
  sessions).
- Report preview host `minsize=200` with stretch weight on the content row.
- Status / warning copy remains wrapping labels, not a second page scroll.

### Compare

- Research catalog | charts use an **adjustable horizontal splitter**.
- Below the compare-stack breakpoint the catalog stacks above the charts.
- Empty state: “Select two different catalog videos…” until both slots analyze.
- Catalog Video A/B combos size to catalog labels.

### Advanced

- Same dark theme, two-column cards above the DPI-aware floor, stacked below.
- Scroll shell + veil skip from the earlier Advanced empty-tab fix is unchanged.
- Live audit: workspace mapped, veil not placed.

## Performance

- `RESIZE_SETTLE_MS = 140` still coalesces window drags.
- Playback skips tab-wide figure fits (`layout_playback.is_playing`).
- `_fit_figure_to_host` resizes existing Figure objects; series are not recomputed.
- Motion metric tab switches redraw from the shared bundle (`force_rebuild=False`).

## Validation (live GUI, not source-only)

Launcher: `scripts/ui_responsive_layout_audit.py` against `StableWalkGUI`.
Measurements: `docs/_ui_responsive_audit.json`.
Screenshots: `docs/ui_responsive_screenshots/` (20 PNG files).

### Display that actually ran the app

| Item | Value |
| --- | --- |
| Tk `winfo_screenwidth/height` | **1536 × 864** |
| `GetDpiForSystem` | **96** (Windows scale **100%**) |
| Tk `winfo_fpixels("1i")` | 96.09 |
| `dpi_scale_factor` | 1.001 |

**125%, 150%, and 175% were not live-tested on this machine.** They are covered
by unit tests on `scaled_breakpoints` / `SCALE_TARGETS` and by DPI-aware
floors in code. Do not treat them as visually verified here.

### Window sizes

| Requested | Live geometry | Tabs exercised | Notes |
| --- | --- | --- | --- |
| 1366×768 | **1366×768** | Overview, Motion (Kinematics / Spectral Analysis / Spectral Features / Asymmetry), Biomechanics (COM, Ground Reaction Force, Balance, Contact, Gait Phase), Results, Compare, Advanced | Fits this display |
| 1600×900 | **1512×816** (clamped) | Same tab set | Larger than 1536×864 usable area |
| 1920×1080 | classified LARGE only | — | Duplicate of 1512×816 clamp |
| 2560×1440 | classified XLARGE only | — | Exceeds display |
| 3840×2160 | classified XLARGE only | — | Exceeds display |

Pytest (layout, not a substitute for the live pass): 72 + 72 + 61 + 47
selected layout tests passed in this session (`test_motion_analysis_layout`,
`test_responsive_window_sizes`, `test_overview_responsive`, `test_biomech_layout`,
`test_advanced_tab_open`, `test_results_layout`, `test_compare_layout`, and related).

### Measured widget sizes (live)

**1366×768**

| Surface | Measurement |
| --- | --- |
| Overview Video / Skeleton / Path | 306×248, 305×248, 687×248, `viz_layout=three_col`, all mapped |
| Kinematics `motion_graph_host` | **1294×229**, canvas row weight 1 |
| Live data panel | 1294×24 (collapsed empty), **not overlaid on the plot** |
| Joint combo | char width 28, ~195 px (label not truncated) |
| Spectral graph host | 738×298 |
| Spectral Features graph | 691×244 |
| Asymmetry graph | 763×290 |
| Biomechanics chart | 752×380, not stacked; findings 185 px, interpretation 345 px |
| Results | Processing Details mapped; preview host collapsed until a report exists |
| Compare | Research empty state visible; catalog splitter present |
| Advanced | workspace 1298×421, two-column, veil not placed |

**1512×816** (clamped 1600×900)

| Surface | Measurement |
| --- | --- |
| Overview panels | 332×260, 332×260, 730×260, three_col |
| Kinematics graph host | **1440×277** |
| Live data | 1440×24 |
| Biomechanics chart | 846×428 |

### Screenshot review (what was visible)

- Overview: Video empty copy wraps (“Browse a walking video…”). Skeleton empty
  state readable. 3D Path cube visible; camera tools stay in the Path chrome.
- Kinematics: Live data is a thin strip **below** the plot, not covering it.
  Angle/Velocity/… tabs share a row with Graph/Reset/Export/Measure.
- Spectral: “Left knee flexion” fully visible; empty-state copy centered in the
  plot; Numerical results rail on the right.
- Results: header, waiting status, COM–BoS warning, collapsed Processing
  Details — **not** a black page (earlier 50 ms screenshots caught the fade veil).
- Compare: catalog + empty-state copy; 3D placeholders, not a blank void.
- Advanced: Session | OpenSim cards, same dark theme, content painted.
- Biomechanics: large chart column, Key Findings rail, expand control present.

## Remaining limits (honest)

1. **Kinematics 380 px floor is not reachable on 768p.** Source bar + main
   tabs + page header + inner notebook + Selected Joint + transport leave about
   230–280 px for the canvas. The graph **owns leftover height** (weight 1);
   live numbers no longer cover it. On this 1536×864 screen the host was 229 px
   at 1366×768 and 277 px at 1512×816.
2. **Overview Path stayed wider than Video/Skeleton** (687 vs ~306) on this
   profile — likely saved sash fractions in `ui_preferences.json`. Defaults
   are now even thirds; dragging the sash still works.
3. **Hover-zoom hint** on the kinematics tools row can clip at 1366 px
   (“Hover scrub · wheel…”). Controls remain usable.
4. **1920 / 1440p / 4K and 125–175% DPI** were not opened as live windows here.
   Re-run `scripts/ui_responsive_layout_audit.py` on a larger or scaled display
   to fill those rows.

## How to re-verify

```text
set PYTHONPATH=<repo>\StableWalk
python scripts/ui_responsive_layout_audit.py
```

Screenshots land in `docs/ui_responsive_screenshots/`. JSON measurements in
`docs/_ui_responsive_audit.json`.
