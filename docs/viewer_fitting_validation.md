# Viewer fitting validation (Task 2)

**Date:** 2026-10-14  
**Scope:** Overview Full-body Skeleton camera/fit and Overview 3D Joint Path viewport.  
**Not in scope:** gait/pose/IK algorithms, parent Overview splitter layout (Task 1).  
**Clips:** `data/demo_videos/normal_gait.mp4` and `abnormal_gait.mp4` (pose JSON, 120 frames each).

Joints are never moved or scaled to “fit.” Only camera, axis limits, figure margins, and canvas pixel size changed.

## What was wrong

1. **Skeleton.** `_apply_view_limits` zoomed the sequence-global box with `SKELETON_SIZE_BOOST = 1.28` and `_SKELETON_MAX_FILL = 0.945`. Rendered soles, heel ellipses, and the floor pad extend past tracked landmarks. That leftover ~2.75% per edge was smaller than the foot envelope, so **legs and feet were cropped at the bottom**. A head-weighted vertical centre could also bias the box upward.
2. **3D path.** Overview `set_box_aspect(..., zoom=1.12–1.28)` plus a close camera (`dist ≈ 4.55`) and ground-plane `pad_frac > 0` made mplot3d draw the cube, grid, and start marker **outside the axes**.

## Skeleton (Part A)

| Setting | Before | After |
|---|---|---|
| `_SKELETON_VIEW_FILL` | 0.90 | 0.78 |
| `SKELETON_SIZE_BOOST` | 1.28 | 1.00 |
| `_SKELETON_MAX_FILL` | 0.945 | 0.84 |
| Axes bottom inset | 0.005 | 0.028 |
| Envelope | landmarks | landmarks + soles + cranial oval + floor pad |
| Centre | head/feet heuristic | geometric AABB of envelopes |
| Manual view | none | scroll-wheel zoom, right-drag pan |
| Fit | existing **Fit** | still present; clears user zoom/pan and refits the full sequence |

Playback keeps a **sequence-global scale** (no zoom breathing) and recentres on the current pose. The cached half-extents **grow monotonically** if a later stride exceeds the first-frame envelope; they never shrink. Resize invalidates the scale cache and recomputes equal-aspect halves from the panel widget. Missing toes/heels are skipped; remaining landmarks still define the box.

### Quartile frames — head-to-feet inside ylim

Pose JSON from the demo videos. `inside=True` means every collected body envelope point (landmarks, soles, cranial oval, floor pad) lies strictly inside `xlim`/`ylim`.

| Clip | Frame | Label | Head–feet inside view |
|---|---|---|---|
| normal_gait | 0 | first | True |
| normal_gait | 29 | 25% | True |
| normal_gait | 59 | 50% | True |
| normal_gait | 89 | 75% | True |
| normal_gait | 119 | last | True |
| abnormal_gait | 0 | first | True |
| abnormal_gait | 29 | 25% | True |
| abnormal_gait | 59 | 50% | True |
| abnormal_gait | 89 | 75% | True |
| abnormal_gait | 119 | last | True |

![normal_gait skeleton quartiles](viewer_fitting/normal_gait_skeleton_quartiles.png)

![abnormal_gait skeleton quartiles](viewer_fitting/abnormal_gait_skeleton_quartiles.png)

Scale is constant across frames (same body height in the panel). The figure recentres slightly with the walk; feet and head stay in view. These Agg renders use the same `compute_skeleton_view_box` / `_apply_view_limits` path as the Overview canvas.

## 3D Joint Path (Part B)

| Setting | Before | After |
|---|---|---|
| Overview box zoom | 1.12–1.28 | **0.88–1.00** (never > 1) |
| Overview camera base dist | 4.55 | 5.25 |
| Figure margins | tight | extra room for X/Y/Z labels |
| Ground pad | 2–4% outside limits | **0** on Overview (plane stays inside the cube) |
| Encoding | start / time-colored path / current / floor shadow | unchanged, plus compact in-axes legend (`Start`, `Path (time)`, `Current`, `End`, `Floor`) |

Equal X/Y/Z scale is unchanged (no exaggerated motion). Toolbar Fit / orbit / zoom / pan still call `fit_trajectory_viewport` and `remember_trajectory_camera`. Direction arrows remain on the trail.

Axis labels remain **X · Lat / Y · Up / Z · Fwd (cm, est.)**. Those centimetres are **stature-scaled display units**, not calibrated world millimetres.

### Complete trajectory in the cube

After `draw_single_dof_trajectory_3d` (Overview dock), every sample of the full right-knee path and the current tip sit inside `xlim`/`ylim`/`zlim`.

![normal_gait 3D path](viewer_fitting/normal_gait_path_quartiles.png)

![abnormal_gait 3D path](viewer_fitting/abnormal_gait_path_quartiles.png)

Plasma coloring is early → late. The green/yellow disk is **Start**; the red/orange disk is **Current**; the cube floor grid is the laboratory XZ reference.

## Coordinate Inspector vs 3D point (Part C)

At frame N the Overview path current sample is the stature-scaled

`recording.snapshot_at(N).joints[joint].position`

after the selected coordinate mode (default **root-relative**). The red marker publishes `ax._stablewalk_tip_xyz_cm = meters_to_display_cm(tip)`. The Coordinate Inspector reads that same cache (`tip_xyz_cm_from_cache`) and formats each axis with `_fmt_cm` (**0.1 cm** glyph).

**Documented precision:** internal match `1e-9 m` between plotted tip meters and `cm/100`; displayed match **0.05 cm** (half of the 0.1 cm glyph). **This is not metric accuracy.** There is no checkerboard / wand calibration in these clips.

Right knee, Overview dock, actual values:

| Clip | Frame | Inspector X / Y / Z (0.1 cm) | tip_cm == 100 × tip_m | Path inside cube |
|---|---|---|---|---|
| normal_gait | 0 | −5.6 / −43.6 / −5.0 cm | True | True |
| normal_gait | 29 | −6.3 / −43.7 / −2.2 cm | True | True |
| normal_gait | 59 | −5.7 / −44.1 / −1.7 cm | True | True |
| normal_gait | 89 | −5.4 / −44.6 / 1.7 cm | True | True |
| normal_gait | 119 | −5.3 / −43.9 / 4.6 cm | True | True |
| abnormal_gait | 0 | −7.5 / −44.4 / 2.2 cm | True | True |
| abnormal_gait | 29 | −6.4 / −44.3 / 3.7 cm | True | True |
| abnormal_gait | 59 | −10.2 / −44.3 / 3.5 cm | True | True |
| abnormal_gait | 89 | −7.3 / −44.7 / 5.0 cm | True | True |
| abnormal_gait | 119 | −2.9 / −44.1 / 5.3 cm | True | True |

The on-figure HUD `XYZ` line uses the same cache (see screenshots: first/50%/last match the table).

## Automated tests

```
python -m pytest tests/test_skeleton_viewport_fit.py tests/test_viewer_fitting_validation.py tests/test_overview_traj_fit.py -q
```

**Result (2026-10-14): 43 passed.**

Related inspector + Overview path tests: **48 passed** (`test_overview_coordinate_inspector.py` included).

| Check | Data | Result |
|---|---|---|
| Head-to-feet inside ylim at 0 / 25 / 50 / 75 / 100% | mock gait + both demo poses | passed |
| Missing toe/heel landmarks still produce a valid box | mock gait | passed |
| Full right-knee path inside equal-scale cube | both demo pose files | passed |
| Drawn tip at frame N equals inspector cache; 0.1 cm rounding | both demo pose files | passed |
| Overview box zoom ≤ 1.0 | synthetic panel sizes | passed |

Screenshots were generated with `docs/viewer_fitting/_render.py` (Agg backend, same fitters as the GUI). Raw log: `docs/viewer_fitting/render_log.txt`.

## Manual GUI checklist

Live window chrome (Fit click, mouse orbit, pane drag) is not in the Agg images. After loading each demo video on the Overview tab:

1. Load `data/demo_videos/normal_gait.mp4`, Overview tab, select **right knee**.
2. Confirm the skeleton shows **head through both feet** with a sliver of stage under the soles (not cropped by the phase strip). Compare with the quartile sheet above.
3. Seek first frame, ~25%, ~50%, ~75%, last frame. Scale must stay constant; the body may recentre but feet stay in view.
4. Press **Fit**. Body remains fully visible. Scroll-wheel zoom, right-drag pan, then Fit again to restore.
5. Resize the window; skeleton and path refit without cropping the cube or the feet.
6. On the 3D path: full trail visible, green **Start**, time-colored line (plasma, early→late), **Current** marker. Inspector X/Y/Z at frame N must equal the HUD/path tip at that frame (0.1 cm) — table above.
7. Play, pause, scrub. Camera must not breathe. Orbit/zoom/pan remain usable.
8. Repeat with `abnormal_gait.mp4`.

## Remaining limitations

- True 1920×1080 client area is still clamped by the desktop work area on this machine.
- Overview centimetres are estimated/stature-scaled; do not cite them as calibrated lab units.
- Ground plane on the path is the **cube floor** (min Y) unless a foot-clearance view is active on Motion Analysis.
- Agg sheets are the fitted Matplotlib artists, not a photograph of the Tk chrome (toolbars, splitters).
