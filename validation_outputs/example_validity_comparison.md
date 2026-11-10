# Example validity comparison

Generated 2026-11-03 10:55 UTC from audited `validation_outputs/*/ *validity.json` dumps.
Filename / demo key is a source label only. It is not a quality grade.

## Shared display contract

- Pose coverage, per-foot contact coverage, spectral reliability, and biomechanical validity stay independent.
- Gait-quality and symmetry scores are unvalidated heuristics. They are withheld when required contact, cycle, or spectral inputs are insufficient.
- Unequal missing-data coverage is not gait asymmetry.
- No threshold was retuned for this comparison.

## Per-example checks that were actually run

### Normal Gait (`normal_gait_v2.mp4`)

- Pose coverage: **limited** — 201/300 (67%). Presence only; not tracking accuracy.
- Left contact known: **20%** (insufficient).
- Right contact known: **37%** (insufficient).
- Cycle evidence: **4 L / 0 R** (computed).
- Spectral reliability: **insufficient**.
- Biomechanical validity: **unavailable** (monocular COM–BoS is not a physical stability score).
- Gait-quality heuristic: withheld — Withheld: L 20% / R 37% contact known, event cycles 4 L / 0 R. Unequal missing-data coverage is not a gait-laterality finding.
- Symmetry heuristic: withheld — Withheld: L 20% / R 37% contact known, event cycles 4 L / 0 R. Unequal missing-data coverage is not a gait-laterality finding.

### Fast Walking (`fast_walking_v2.mp4`)

- Pose coverage: **limited** — 693/900 (77%). Presence only; not tracking accuracy.
- Left contact known: **6%** (insufficient).
- Right contact known: **84%** (computed).
- Cycle evidence: **17 L / 0 R** (computed).
- Spectral reliability: **limited**.
- Biomechanical validity: **unavailable** (monocular COM–BoS is not a physical stability score).
- Gait-quality heuristic: withheld — Withheld: L 6% / R 84% contact known, event cycles 17 L / 0 R. Unequal missing-data coverage is not a gait-laterality finding.
- Symmetry heuristic: withheld — Withheld: L 6% / R 84% contact known, event cycles 17 L / 0 R. Unequal missing-data coverage is not a gait-laterality finding.

### Abnormal Gait (`abnormal_gait.mp4`)

- Pose coverage: **computed** — 649/649 (100%). Presence only; not tracking accuracy.
- Left contact known: **12%** (insufficient).
- Right contact known: **76%** (limited).
- Cycle evidence: **0 L / 3 R** (computed).
- Spectral reliability: **limited**.
- Biomechanical validity: **unavailable** (monocular COM–BoS is not a physical stability score).
- Gait-quality heuristic: withheld — Withheld: L 12% / R 76% contact known, event cycles 0 L / 3 R. Unequal missing-data coverage is not a gait-laterality finding.
- Symmetry heuristic: withheld — Withheld: L 12% / R 76% contact known, event cycles 0 L / 3 R. Unequal missing-data coverage is not a gait-laterality finding.

## Limitations that apply to all three examples

- Monocular 2-D/3-D pose is not marker-based kinematics.
- Contact is inferred from pose; Unknown is a missing observation, not swing and not laterality.
- Spectral ACF reliability is not interchangeable with pose coverage or video quality.
- Physical COM–BoS / XCoM / measured GRF are unreportable from these clips.
- Fast Walking analysis is truncated to the demo frame cap when the source file is longer.

## Checks that were not performed

- Force-plate or instrumented-treadmill GRF
- Marker-based optical motion capture
- IMU / wearable validation
- Clinician-scored gait quality instrument (e.g. GGI, GPS, GDI)
- Calibrated metric reconstruction / known subject stature check
- Bilateral contact with known ground truth
- Diagnosis, impairment grading, or filename-based quality boosting

## Outcome

Normal Gait does not score better because of its filename. All three examples have insufficient or unequal per-foot contact evidence, so Overview / Biomechanics / Results / exports withhold gait-quality and symmetry as unvalidated heuristics rather than Borderline or Acceptable grades.
