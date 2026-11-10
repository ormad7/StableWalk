# Metric regression audit (Task 4)

**Date:** 2026-10-14  
**Clips:** `data/demo_videos/normal_gait.mp4` (120 frames, **25 FPS**, 4.76 s) and `abnormal_gait.mp4` (120 frames, **29.97 FPS**, 3.97 s).  
**Pose source used here:** `data/output/poses/normal_gait_poses.json` and `abnormal_gait_poses.json`.  
**Rule:** scores were not retuned to match a previous GUI snapshot. Optical mocap ground truth for these clips was **not** found in the repo. Nothing below is a clinical diagnosis.

This document answers why Gait Quality, cadence/symmetry availability, analysis confidence, and knee angular jerk *looked* different — and what was actually wrong in the software.

---

## Verdict (read this first)

| Observation | Reproduced on saved `normal_gait` poses? | Cause | Scientific status |
|-------------|------------------------------------------|--------|-------------------|
| Gait Quality ~84 → ~26 | **No.** Live composite **Q = 80.72 → 81/100** (same formula as Oct 13 live 81). | GUI could display a **different algorithm** (`gait_analysis_summary` group score) when composite Q was skipped, or a partial session. Oct 13 GUI snapshot 84 was already a different *session instance* than live 81. | Composite Q is a derived index, **not** clinical gait quality. 26 was not restored and is not the current JSON result. |
| Cadence / symmetry “unavailable” | **Not on this JSON.** Cadence **64.5 steps/min**, symmetry **86.3%**. | Code treated `LOW_CONFIDENCE` / `metrics_reliable=False` as “do not compute” and wiped values. This clip is currently `confidence_tier=HIGH`. | Cadence from ≥2 same-side HS→HS intervals is **defined**. Low contact quality is a reliability tag, not a missing number. |
| Low Analysis Confidence | **Partly.** `build_analysis_summary` → **Moderate**. Overview `gait_analysis_summary` → **LOW** when usable cycles = 2. | Two different definitions. 2 cycles / 4.76 s cannot be High. Pose tracking on this clip is ~95/100 and is **separate**. | Low/Moderate for this clip is **justified**. It was not changed to High. |
| Knee angular jerk ~25,352 deg/s³ | **Yes, as unsmoothed 3rd difference.** Raw peak **22,074 deg/s³**. After justified SG derivatives: interior peak **3,531 deg/s³**. | Live Motion panel shows **current-frame** jerk, not a session statistic. Three unsmoothed differences of 25 Hz landmark noise are not physiology. | Units (deg/s³) were already correct. Magnitude was a numerical artifact. Not mocap-validated. |

**Processing completed** on these pose files. **Analysis is not scientifically reliable as a clinical measurement** (monocular, 2 cycles, <6 s, no force plates). Several metrics are **available** as software outputs. Heel-strike times are **insufficient evidence** of true events (no independent annotation).

---

## Part A — Regression audit (Git)

Last committed revision on `main` is `a2d63c5` (2026-07-27, “Polish StableWalk UI…”). Almost all gait math, quality scoring, GUI, and docs since then are **uncommitted working-tree changes**. There is **no tagged historical source** of the 84/100 GUI snapshot inside Git.

| Area | HEAD (`a2d63c5`) | Working tree before this audit | What actually changed scores |
|------|------------------|--------------------------------|------------------------------|
| Gait Quality | Small `gait_quality_score.py`; orchestrator always scored when cycles existed | Orchestrator **skipped** `compute_gait_quality_score` when `cycles.metrics.metrics_reliable` was false | Missing composite Q → GUI could substitute another “Gait Quality” |
| Pose tracking confidence | Video-quality / landmark visibility | Still separate from analysis confidence | Mixing these produced the Oct 13 “High (95%)” snapshot |
| Gait events | Contact FSM + kinematic supplement | Same family; support-coverage floor lowered (0.98 → 0.80) so cadence cycles ≠ support samples | Event times still not mocap-validated |
| Cadence | Already withheld when `confidence_tier == LOW_CONFIDENCE` **or** too few *support-sampled* cycles | Split HS→HS cadence cycles from support samples, but still gated cadence on LOW_CONFIDENCE and **zeroed** the struct | Unavailable cadence was a **policy bug**, not a missing interval |
| Symmetry | `analyze_symmetry` required `metrics_reliable` | Same; ROM SI also gated off | One remaining component < 2 → overall symmetry None |
| Numerical derivatives | Chart used raw finite differences | Same, three times | 10⁴ deg/s³ jerk |
| Data-quality filtering | Contact confidence 0.48 / LOW tier | Same numbers used as a hard “off” switch | Conflated reliability with availability |

Limitation: the 2026-10-13 numbers in `docs/metrics_integrity_report.md` are the only recorded prior instance. They cannot be `git show`-n at a commit.

---

## Part B — What the GUI displays

| Surface | What it is | Session vs frame |
|---------|------------|------------------|
| Overview Gait Quality | `BiomechanicalAnalysisResult.gait_quality.score` from `compute_gait_quality_score` | **Session**, after one analysis pass |
| Overview cadence / symmetry | `gait_metrics.cadence` / `symmetry.overall_symmetry_pct`, then cycle metrics | **Session** |
| Analysis Confidence (Overview viz) | Mean of data-quality signals + `gait_analysis_summary.analysis_confidence.level` | **Session**; 2 usable cycles cannot display High |
| Analysis Confidence (Results summary) | `analysis_summary._analysis_confidence` | **Session**; this clip → Moderate |
| Motion “Angular jerk” value | `series.angular_jerk_deg_s3[nearest_to_playhead]` | **Current frame** |
| Motion stats ribbon peak | `nanmax(|angular_jerk|)` | **Session peak of the displayed series** |
| Rancho phase % | `gait_cycle_phase` 0–100% of current stride | **Current frame** |
| 3D path length | Gap-aware polyline (`1.5/fps` rule) | Session prefix; not tape-measure metres |

**Replay:** `SessionAnalysisCache` is filled once in the analyze path (`run_biomechanical_analysis` in `app.py`). Playback reads the cache and must not re-run the pipeline. Two live `run_biomechanical_analysis` calls on the same poses produced **identical** Q, cadence, and event lists (`delta = 0`).

**Bug found (display, not formula):** `_update_overview_clinical_summary` passed an unbound `result` into `build_session_metric_snapshots`, which always threw and left `_session_metric_view = None`. Overview KPIs still came from `extract_overview_summary_fields`. Fixed to use `_stability_result`.

**Bug found (two Gait Qualities):** if composite Q was missing, Overview and `session_view` copied `gait_analysis_summary.gait_quality` (temporal/spatial symmetry, clearance, cycle consistency, contact — **different weights**). On this JSON that group score is **60.8** (normal) / **42.8** (abnormal), not 26. A sparser active-domain set can pull it lower. That substitution is **removed**. If composite Q is missing, the card is **Not available**.

Partial results: `_update_overview_clinical_summary` returns immediately while `playing` is True (does not restyle KPIs mid-playback). Session numbers should not drift with the playhead. Frame jerk/phase **do** change with the playhead; that is intended.

---

## Part C — Numerical correctness

### Units (verified on an analytic cubic)

| Quantity | Implementation | Unit |
|----------|----------------|------|
| Joint angle | Geometric interior angle (knee ≠ anatomical flexion) | deg |
| Angular velocity | SG 1st derivative of angle, or one-sided Δθ/Δt at edges | deg/s |
| Angular acceleration | SG 2nd derivative of angle | deg/s² |
| Angular jerk | SG 3rd derivative of angle | deg/s³ |
| Position | Canonical hip-centered XYZ | m (monocular, not lab) |
| Linear speed | \|v\| | m/s |
| Linear jerk | \|d³**r**/dt³\| | m/s³ |
| Path length | Sum of connected segments; gaps > `1.5/fps` excluded | m (same monocular scale) |
| Timestamps | `snapshot.time_s`; median Δt = 0.040 s (25 FPS) / 0.0334 s (29.97 FPS); monotonic | s |

Analytic \(x = \frac{1}{6} j t^3\), \(\theta = \frac{1}{6} j_\theta t^3\) at 25 / 50 / 100 FPS: interior SG jerk matches \(j\) and \(j_\theta\). Polynomial order 3 reproduces cubics. This is **not** a validation of MediaPipe angles.

### FPS and alignment

Derivatives use recorded timestamps, not `frame_index / assumed_fps`. Gap rule matches path length (`max_connected_dt = 1.5/fps`): a missing frame is **NaN**, not a huge Δy/Δt.

### Why ~25,000 deg/s³ appeared

At 25 Hz, a 1° landmark flicker in a 3rd unsmoothed difference scales as \(O(\sigma / \Delta t^3) = O(\sigma \cdot 15625)\). Live raw peak on `normal_gait` right knee: **22,074 deg/s³**. The ~25,352 value is a **playhead sample** of that noise process, not a session clinical peak.

### What was changed (justified)

1. Isolated one-frame angle spikes despiked (sustained ROM kept).
2. Landmark visibility &lt; 0.35 treated as missing.
3. One Savitzky–Golay cubic, window ~0.52–0.64 s (min 13 samples), **direct 1st–3rd derivatives** from the local polynomial (Winter-style: filter displacement, then differentiate; do not difference noise three times).
4. 2nd/3rd derivatives **NaN** where the SG window is incomplete (edges). Filling those with raw finite differences put the 10⁴ artifact back.

After the change, `normal_gait` right-knee angular jerk peak **3,531 deg/s³** (raw 22,074). `abnormal_gait` **3,535** (raw 39,386). These remaining peaks are **not** claimed to be true biological jerk.

Gait Quality **did not move** (still 80.72). Motion-quality / joint-smoothness scoring does not use this chart jerk.

---

## Part D — Gait events (actual timestamps)

Event **confidence** is ipsilateral foot-landmark **visibility** at that frame (0–1). It is **not** a detector posterior and **not** proof of a true heel strike. No independent HS/TO annotation file exists for these demos.

### `normal_gait` (25 FPS)

Contact confidence **1.00**, tier **HIGH**, `metrics_reliable=True`, 2 complete same-side HS→HS cycles.

| t (s) | Frame | Event | Side | Visibility |
|------:|------:|-------|------|------------|
| 0.04 | 1 | heel strike | left | 1.0 |
| 0.88 | 22 | heel strike | right | 1.0 |
| 1.00 | 25 | toe-off | left | 1.0 |
| 1.60 | 40 | heel strike | left | 1.0 |
| 1.60 | 40 | toe-off | right | 1.0 |
| 2.88 | 72 | toe-off | left | 1.0 |
| 2.88 | 72 | heel strike | right | 1.0 |
| 3.76 | 94 | heel strike | left | 1.0 |
| 3.96 | 99 | toe-off | right | 1.0 |

- Left HS times 0.04, 1.60, 3.76 → two intervals of **1.56 s** and **2.16 s** (mean stride 1.86 s).  
- Cadence = \(60 / t_\mathrm{step}\) with \(t_\mathrm{step}=0.93\) s → **64.52 steps/min**.  
- Cycle consistency \(1-\mathrm{CV}\) = **0.839**.  
- Stance/swing SI 0.714 / 0.698. Double+single+flight = **51.9%** (not ~100%) → stance **percentages** are internally inconsistent; cadence from HS intervals does not use that partition.  
- HS at **0.04 s** and coincident L/R events at 1.60 s and 2.88 s are **insufficient evidence** of true contralateral timing (clip edge / monocular occlusion).

Gait phase % is Perry/Rancho bins of progress through the current same-side HS→HS stride (`gait_cycle_phase.py`). It is a **display mapping**, not a validated phase classifier.

### `abnormal_gait` (29.97 FPS)

| t (s) | Event | Side | Visibility |
|------:|-------|------|------------|
| 0.067 | heel strike | left | 1.0 |
| 0.067 | heel strike | right | 1.0 |
| 0.334 | toe-off | left | 1.0 |
| 1.001 | heel strike | left | 1.0 |
| 1.602 | toe-off | right | 1.0 |
| 2.002 | heel strike | right | 1.0 |
| 2.769 | toe-off | left | 1.0 |
| 3.537 | heel strike | left | 1.0 |

Cadence **51.87 steps/min**, stride 1.735 s, Q **70.64**, symmetry **76.9%**. Simultaneous bilateral HS at t=0.067 s is a **start artifact**, not a verified double-contact event.

---

## Part E — Result validity (definitions)

| State | Meaning in this app |
|-------|---------------------|
| **Processing completed** | Pose JSON loaded; `run_biomechanical_analysis` returned without skipping the clip. True for both demos. |
| **Analysis scientifically reliable** | Requires ≥3 complete same-side cycles, duration ≥6 s, contact tier not LOW, contact score ≥0.72, and preferably lab scale. **False** for both demos. |
| **Metric available** | A formula produced a finite value from existing events/series. Cadence, composite Q, symmetry index are available on these JSONs. |
| **Metric unavailable** | Formula inputs missing (e.g. 0–1 HS→HS cycles, saturated contact &gt;85%, unreportable COM–BoS). Display **Not available**, never 0. |
| **Insufficient evidence** | Value may exist as software output but cannot be interpreted as a clinical or mocap-grade fact (2 cycles, monocular millimetres, coincident HS/TO). |

### Gait Quality (do not confuse with tracking)

**Name:** Composite Gait Quality \(Q \in [0,100]\).  
**Formula:** weighted mean of *active* components only (missing omitted, not zero-filled).

Weights: stability 0.22, stability_margin 0.15, symmetry 0.20, cadence_consistency 0.12, cycle_consistency 0.12, contact_reliability 0.10, motion_quality 0.09.

**This clip (`normal_gait`), live:**

| Component | Value | Notes |
|-----------|------:|-------|
| stability | 75.39 | Movement-stability domain, not COM–BoS |
| motion_quality | 51.57 | Joint smoothness; LOW_CONFIDENCE repeatability (2 cycles) |
| symmetry | 86.28 | Weighted L/R ratios, not clinical symmetry |
| cadence_consistency | 83.87 | \(1-\mathrm{CV}\) of cycle duration |
| cycle_consistency | 83.87 | Same CV transform |
| contact_reliability | 100 | contact_confidence × 100 |
| stability_margin | *omitted* | `reportable=False` (monocular COM–BoS) |

\[
Q = 80.72 \;\rightarrow\; \text{displayed } 81/100.
\]

This is **not** pose-tracking quality (that is ~95/100 and labeled separately). It is **not** a clinical gait grade. A second algorithm (`gait_analysis_summary` domains: temporal/spatial symmetry, clearance, cycle consistency, contact) scored **60.8** on the same clip and must not be shown as this card.

Oct 13 GUI **84** used a slightly different event timing (cadence/cycle consistency nearer 99, symmetry 87). Oct 13 live recompute was already **81**. The ~26 observation is **not** this formula on the current JSON.

---

## Part F — Tests (actual numbers)

| Test | Result |
|------|--------|
| Analytic cubic jerk, 25/50/100 FPS | Interior SG 3rd derivative = 48 deg/s³ and 12 m/s³ (units held) |
| 1° white noise at 25 Hz | Raw 3rd-difference peak **17,253**; SG peak **&lt; 0.35 × raw** |
| `normal_gait` repeated analysis | Q **80.7199** twice; cadence **64.5161** twice; event list identical |
| `abnormal_gait` vs normal | Q **70.64** vs **80.72** (Δ ≈ 10); cadence **51.87** vs **64.52** (Δ ≈ 13 spm). Not score-tuned. |
| LOW_CONFIDENCE + 2 HS→HS cycles | Cadence **computed** (e.g. 120 spm on a 1.0 s synthetic stride); `metrics_reliable=False` |
| 1 cycle | Cadence **None** (unavailable), not 0 |
| Overview fallback | `gait_analysis_summary` score 26/71 is **not** shown as Gait Quality |
| Replay / session view | Unbound `result` NameError removed |

Tests live in `tests/test_metric_regression_audit.py`, `tests/test_dimensional_consistency.py`, `tests/test_metrics_integrity.py`, `tests/test_overview_clinical_summary.py`, `tests/test_metric_consistency.py`.

---

## Code changes made in this task (not cosmetic score restoration)

1. **Cadence/symmetry availability** — compute when ≥2 HS→HS cycles exist; `metrics_reliable` is a reliability flag. Saturated contact (&gt;85%) still withholds (events are not HS).  
2. **Downstream publish** — `analyze_advanced_gait_metrics` / `analyze_symmetry` no longer wipe finite values solely because `metrics_reliable` is false.  
3. **Composite Q** — always scored from available components; no longer skipped when temporal reliability is low.  
4. **GUI identity** — Overview / session view no longer alias `gait_analysis_summary.gait_quality` as composite Q.  
5. **Derivatives** — confidence-aware SG cubic derivatives; gap-aware finite differences; vector linear jerk.  
6. **Event visibility** — `GaitEvent.confidence` = foot landmark visibility.  
7. **Session view construction** — use `_stability_result`, not an unbound name.

**Not done (intentionally):** raising Q from 81 to 84; forcing High confidence on a 4.76 s / 2-cycle clip; treating 3,531 deg/s³ as validated biological jerk; claiming Erik / OpenSim / GRF correctness.

---

## What remains unverified

- True heel-strike / toe-off times (no force-plate or annotated video).  
- Absolute metres (stature-normalized monocular reconstruction).  
- Clinical meaning of Q, symmetry %, or cadence on these clips.  
- Whether a user’s on-screen 26 came from a *different* pose run (live MediaPipe vs this JSON) — that source was not in Git.
