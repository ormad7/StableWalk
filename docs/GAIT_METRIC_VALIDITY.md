# Gait metric formulas and validity rules

StableWalk reports pose-derived research estimates, not instrumented clinical
measurements. Final session metrics are calculated only when at least two
complete gait cycles are bounded by consecutive heel strikes of the same foot,
contact confidence is at least 0.48, and the confidence tier is MEDIUM or HIGH.
Leading and trailing partial cycles are excluded. If these conditions fail, the
value is `N/A` and the reason is stored in the metric note and session warnings.

## Temporal metrics

For every complete cycle, binary left/right contact states are integrated using
frame timestamps.

- Left stance % = 100 × left-contact time / cycle duration.
- Left swing % = 100 × (cycle duration − left-contact time) / cycle duration.
  The right side uses the same formulas. Consequently stance + swing is 100%
  for each foot, apart from at most 0.5 percentage point numerical tolerance.
- Total double-support % = 100 × time with both feet in contact / cycle
  duration. This is **total per gait cycle**, combining both double-support
  periods; it is not a per-side value.
- Total single-support % = 100 × time with exactly one foot in contact / cycle
  duration. It combines left-only and right-only support and is calculated
  directly, not as `100 − double support`.
- Flight/uncertain % = 100 × time with neither foot in contact / cycle duration.
  Double support + single support + flight therefore partition the cycle.
- Step time = mean interval between alternating left/right heel strikes inside
  complete cycles.
- Stride time = mean interval between consecutive same-side heel strikes.
- Cadence (steps/min) = 60 / mean step time. If alternating events are complete
  but unavailable, cadence = 120 / mean stride time.

All percentages must be finite and within 0–100%. Negative support durations are
not clamped; they invalidate the metric and generate a warning.

## Spatial metrics

Canonical coordinates use X=lateral, Y=vertical, and Z=forward.

- Step length = absolute forward-axis separation between the striking and
  contralateral foot at heel strike.
- Stride length = twice the bilateral mean step length.
- Step width = median absolute lateral ankle separation.
- Walking speed uses the highest-confidence valid estimate from global pelvis
  progression, cadence × step length, or image-plane pelvis drift. Body-
  normalized distances are scaled using configured subject stature. Values
  outside the estimator's documented plausible range are rejected, not clipped.

Step/stride values use heel strikes inside complete reliable cycles only.
Monocular scale remains an assumption and is included in each metric note.

## Joint range of motion

For each hip, knee, and ankle:

1. Keep finite pose-derived angles within −180° to +180°.
2. Require at least five valid samples and at least 60% sample coverage in a
   complete cycle.
3. Per-cycle ROM = maximum angle − minimum angle.
4. Session ROM = arithmetic mean of valid per-cycle ROM values.

The reported flexion minimum and maximum are means of the corresponding
per-cycle extrema. Incomplete cycles and whole-clip extrema are not used.

## Symmetry

For non-negative bilateral quantity values L and R:

`symmetry = 2 × min(L, R) / (L + R)`

The result is 0–1 (or ×100 for percent). Negative inputs are invalid rather than
converted with absolute value. Overall symmetry is a documented weighted mean
of valid step, stride, stance, swing, cadence-consistency, and ROM symmetry
components; at least two bilateral components are required.

## Stability margin

The frame-level stability margin is the signed shortest horizontal distance
from projected COM to the base-of-support polygon edge: positive inside,
negative outside. The body-normalized geometric distance is converted to metres
using configured subject stature before applying the 0.04 m stable threshold.
Stable-frame percentage uses only frames with a finite margin
and confidence ≥0.35. At least 50% of analyzed frames must be valid; otherwise
the final mean margin and stable percentage are `N/A`. Missing polygons are
classified as unavailable, not as reduced stability.

## Validation and reporting

Invalid or non-finite final values are replaced by `N/A`, confidence is set to
zero, and the exact reason is retained. Warnings are exported in the session
PDF under **Metric Validation Warnings**. Physiologically unusual but
mathematically valid values are preserved; StableWalk does not force results
into a typical healthy range.
