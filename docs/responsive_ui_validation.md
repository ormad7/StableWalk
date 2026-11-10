# StableWalk responsive UI validation

Layout-only pass. Scientific numerics, pose/IK, OpenSim, export, Compare
synchronization, and video processing are unchanged.

## GUI technology

The desktop app is **Tkinter + ttk + Matplotlib (TkAgg)**. There is no
CustomTkinter, PyQt, or PySide in the dashboard. Streamlit exists as a
separate research viewer and was not modified.

Layout managers in use:

- `grid` for tab pages, Overview visualization, Motion / Biomechanics /
  Results / Compare / Advanced workspaces
- `pack` for chrome (source bar, transport) and some collapsible bodies
- `ttk.Panedwindow` for Overview Video | Skeleton | Path when space allows
- `place` only for maximize overlays and a few floating chrome controls
- Matplotlib `FigureCanvasTkAgg` fitted to a dedicated host frame

## Root causes

1. **Fixed 96-DPI stacking floors.** Wraplengths and column-stack thresholds
   were authored at 100% scale. At 150–175% Windows scaling a 1920×1080
   monitor reports a smaller *logical* Tk size, so three-column Overview and
   Biomechanics rails clipped labels and graphs.
2. **Motion inner canvases sized to the wrong parent.** Spectral Features
   and Asymmetry figures used the whole body (table + plot) as the fit host,
   so plots overflowed the graph column and legends sat on data.
3. **No per-host `<Configure>` bind** on Spectral / Features / Asymmetry, so
   those figures only resized on a full-window reflow, not when their inner
   notebook tab became visible.
4. **Biomechanics expand control was unwired.** `_biomechanics_header` is
   `None`; the maximize button never installed.
5. **Compare DPI lookup used the comparison controller** (`self.root` is a
   `ttk.Frame`). Unit-test doubles then fell through to `GetDpiForSystem`,
   which would change stack thresholds on high-DPI developer machines.
6. **Advanced empty copy** had a hardcoded `wraplength=820` and did not
   follow workspace reflow.

## Layout architecture

```
root
  source / input bar          pack TOP
  ttk.Notebook (stretch)      pack FILL
    Overview  → paned 3-col | 2+1 | stacked viz; KPI 4-col / 2×2
    Motion    → inner notebook (Kinematics / Spectral / Features / Asymmetry)
    Biomechanics → findings | chart | interpretation (collapse then stack)
    Results   → scrollable preview + wrapping report labels
    Compare   → 3-col workspace, stacks below DPI-aware floor
    Advanced  → 2-col cards, stacks below DPI-aware floor; informational empty state
  transport bar               pack BOTTOM
```

Window floors stay **960×600 logical pixels** and are *not* multiplied by
175%. Scaling the minimum window would make the app unusable on a 1366×768
laptop at 150%.

Display scale *does* grow stacking breakpoints via
`stablewalk/ui/tk/layout_scale.py` (`scaled_breakpoints`). Width mode
`XLARGE` (≥2200 px) keeps the Overview summary column inline on 1440p / 4K.

Resize work is coalesced (`resize_coalesce.py`, 140 ms) and skipped while
playback is active (`layout_playback.is_playing`). Figure fit calls
`_fit_figure_to_host` on existing Figure objects — they are not recreated
and analysis is not recomputed.

Video uses **contain-fit** (`min(box_w/src_w, box_h/src_h)`) so the subject
is not cropped; letterboxing is preferred over clipping.

## Files changed

| File | Role |
| --- | --- |
| `stablewalk/ui/tk/layout_scale.py` | DPI tokens, breakpoints, process/Tk scaling helpers |
| `stablewalk/ui/tk/dashboard_responsive.py` | `WidthMode.XLARGE`, motion inner figure reflow |
| `stablewalk/ui/tk/overview_responsive.py` | DPI-aware Overview breakpoints |
| `stablewalk/ui/tk/dashboard_spectral.py` | Expand + host-bound figure resize |
| `stablewalk/ui/tk/dashboard_spectral_features.py` | Dedicated graph host + expand |
| `stablewalk/ui/tk/dashboard_asymmetry.py` | Dedicated graph host + expand |
| `stablewalk/ui/tk/dashboard_panel_maximize.py` | Spectral/Features/Asymmetry/Biomechanics expand |
| `stablewalk/ui/tk/dashboard_biomechanics.py` | DPI-scaled stack floor |
| `stablewalk/ui/tk/dashboard_compare.py` | DPI-scaled Compare stack; scale from `gui` |
| `stablewalk/ui/tk/dashboard_advanced_workspace.py` | DPI-scaled two-column floor; empty-state wrap |
| `stablewalk/ui/tk/app.py` | Tk DPI scaling on init; video fallback 640×480 |
| `tests/test_layout_scale.py` | New DPI / inner-host tests |
| `tests/test_responsive_window_sizes.py` | XLARGE + 1440p/4K classification |
| `tests/test_overview_responsive.py` | Scale-aware Overview breakpoints |
| `scripts/responsiveness_audit.py` | Compare tab included in live audit |

## Resolutions tested

| Size | How tested | Result |
| --- | --- | --- |
| 1280×720 | Unit + `test_responsive_layout_regression` | SMALL/MEDIUM + SHORT; stacked viz as needed |
| 1366×768 | Unit + live resize smoke | MEDIUM; Overview three-col at 100% scale |
| 1600×900 | Unit + live resize smoke | LARGE |
| 1920×1080 | Unit + live resize smoke | LARGE |
| 2560×1440 | Logical `classify_layout` only | XLARGE |
| 3840×2160 | Logical `classify_layout` only | XLARGE |
| Maximized / drag-resize | Existing `test_responsive_layout_regression.py` | Debounced reflow |

Live Tk windows at 2560×1440 and 3840×2160 were **not** opened. Those sizes
overflow a typical laptop framebuffer and are not a reliable GUI check on
this host.

## Scaling configurations tested

| Scale | How tested | Result |
| --- | --- | --- |
| 100% | Default `overview_breakpoints(..., scale=1.0)` | Unchanged 96-DPI floors |
| 125% | `scaled_breakpoints(1.25)` unit test | Floors grow |
| 150% | Overview 1366-wide at `scale=1.5` | Drops three-col viz |
| 175% | Overview 1920-wide at `scale=1.75` | `two_plus_one`; 2560 stays `three_col` |

**Windows 125 / 150 / 175% were not applied as OS settings** in this
session. The process is per-monitor DPI aware
(`SetProcessDpiAwareness(2)`). Tk scaling is raised only when Tk still
reports ~96 DPI, to avoid double-scaling.

## Checks not executed

- OS-level Windows display scaling (125/150/175%) as a live desktop session
- Physical 1440p and 4K monitors
- Before/after screenshots (no Tk screenshot harness in this environment)
- GUI automation of playback, OpenSim IK, or Compare video sync
- Exhaustive visual inspection of every Matplotlib legend at every size

## Remaining limitations

- At 175% on a 1080p panel, Overview uses two-plus-one visualization; the
  third panel (3D Path) sits below Video | Skeleton. That is intentional.
- Spectral 3×2 subplots still share one figure; very short windows need the
  expand (⛶) control or inner-tab scrolling rather than six full-height axes.
- Compare default page is **Research dataset** (catalog + empty charts).
  Laboratory A/B cache is an inner tab. Global playback stays visible on
  Research; Laboratory has its own synced transport.
- Compare empty state is informative placeholders, not live video; space
  redistributes when sessions load.
- `place` remains for maximize overlays. Restoring a panel re-applies the
  original `grid`/`pack` geometry.
- Minimum practical window is 960×600. Below that, transport stays visible
  but chart hosts may drop under `_fit_figure_to_host`’s 80 px floor until
  the window grows.

## Functional preservation

This pass does not:

- Recreate gait / spectral / OpenSim analysis objects on resize
- Reload video files
- Reset Compare A/B sessions
- Change COM–BoS, cadence, or gait-quality numerics
