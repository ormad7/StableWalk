# Validation report

Generated 2026-11-03 15:54 UTC. Analysis used the existing full-analysis pose dumps (`detected_only=False`); MediaPipe was not re-run. Failed detections keep source frame IDs and timestamps. Source videos were decoded independently of playback.

Filename / demo key is a source label only. Missing L/R contact is not physical asymmetry. Thresholds were not retuned.

## Subject selection (Normal, 2026-11-03 16:38 UTC)

The Normal numbers later in this report came from the presenter-locked dump and **must not be used for further gait validation**.

Normal was reprocessed with an explicit walker seed (`xy:0.120,0.550`). Pose dump: `data/output/poses/run_90d21a9367838db7_1793723907553_poses.json` (300 frames). Diagnostic overlays: `validation_outputs/normal/identity_probe/`.

Verified from landmarks, not from the box label alone:

- Frame 80: analyzed keypoints are the white-shirt walker; presenter is Other.
- Frame 148: tracking unavailable; fused one-person detection is not analyzed; no SUBJECT landmarks.
- Frame 191: selected keypoints are the walker (hip x=0.63); presenter is Other. Frame is `filter_skip` (`insufficient_gait_dof`), so those landmarks are not gait inputs.
- Frame 220: selected keypoints remain the walker.

Contact thresholds and gait scores were not retuned.

## Missing-data handling (Normal, 2026-11-10)

Walker-seed dump reused (MediaPipe not re-run): `data/output/poses/run_90d21a9367838db7_1793723907553_poses.json`. Exports: `validation_outputs/normal/frame_audit.csv`, `validation_outputs/normal/missing_data_summary.json`, overlays in `validation_outputs/normal/missing_data_probe/`.

Landmarks vs overlap:

- Frame 80: walker pose_ok (hip x=0.33); presenter is Other.
- Frame 148: tracking unavailable (`subject_identity_ambiguous`); no SUBJECT landmarks.
- Frame 191: walker locked (hip x=0.63) but `filter_skip` / `insufficient_gait_dof` — not gait-usable.
- Frame 220: walker pose_ok.

Long gaps are **identity ambiguity**, not low confidence. The 100-frame hole (90–189) is overlap lock-skip. A second 71-frame hole (229–299) is the same policy after the walker leaves (n=1 fused person is not re-acquired). Other rejected frames: 14 `insufficient_gait_dof`, 3 `incomplete_body`. Thresholds were not relaxed.

Coverage:

| | Before | After |
|---|---|---|
| Analyzed frames | 300 | 300 |
| Detected (identity present) | 110 (previously labeled pose_ok only) | **127** |
| Usable gait frames | 110 pose_ok | **109** (HUD / Results usable; 110 pose_ok) |
| Tracking unavailable | 173 | 173 |

Kinematics lines now break at invalid samples; COM placeholders are NaN (190 missing COM samples, 0 valid zeros). Overview / Results: `127 detected · 109 usable / 300`; `0 L / 0 R accepted · 1 candidates`. Cadence is not shown (no accepted ipsilateral cycles).

## Actual tests

- pytest return code: 0
- result: passed

```
........................................................................ [ 71%]
.............................                                            [100%]
101 passed in 35.76s
```

## Per-example source vs analyzed duration

### Abnormal Gait (`abnormal_gait_v2.mp4`)

- Source decoded: **649** frames, 27.07 s @ 23.976 fps.
- Analyzed dump: **649** stored frames.
- Analyzed span: **27.03 s** (last−first original timestamp, includes gaps).
- Cumulative valid duration: **27.03 s** (sum of contiguous pose_ok runs; n−1 intervals per run).
- Coverage: **100.0%** of analyzed span; longest valid segment **27.03 s** (1 runs).
- Recording slots equal stored frames: True (detected_only=False).
- Status counts: `{'pose_ok': 649, 'filter_skip': 0, 'subject_lock_skip': 0, 'detection_failed': 0, 'undetected_unclassified': 0}`
- Source frames not in dump: 0
- Pose-dump ID holes: 0
- Left contact known: 12.3%
- Right contact known: 76.0%
- Cycle evidence (accepted, both sides published): 0 L / 0 R (candidates 10, accepted 0; accepted candidates 0 L / 0 R). Rejection counts: `{'duration_above_max': 7, 'no_known_swing': 3}`.
- Longest continuous valid pose segment: **27.03 s**. A cycle longer than that interval is accepted only when every internal pose gap is at most one missed sample; unsupported gaps are rejected as `unsupported_pose_gap` and are not interpolated.
- Laterality withhold: Withheld: L 12% / R 76% contact known, event cycles 0 L / 0 R. Unequal missing-data coverage is not a gait-laterality finding.
- Gait-quality withhold: Withheld: L 12% / R 76% contact known, event cycles 0 L / 0 R. Unequal missing-data coverage is not a gait-laterality finding.
- Summary gait quality: Not available
- Summary walking speed: 0.55 m/s (estimated) (estimated=True, reportable=True, not a measured speed).
- Phase-chart annotation: `Estimated speed 0.55 m/s (estimated) | Stance  44%
Swing   0%
Unknown 56%`
- Cursor labels ok: True; hover≠playback: True; pause/step/seek: True

### Normal Gait (`normal_gait_v2.mp4`)

- Source decoded: **300** frames, 10.01 s @ 29.970 fps.
- Analyzed dump: **300** stored frames.
- Analyzed span: **9.98 s** (last−first original timestamp, includes gaps).
- Cumulative valid duration: **6.34 s** (sum of contiguous pose_ok runs; n−1 intervals per run).
- Coverage: **63.5%** of analyzed span; longest valid segment **3.77 s** (11 runs).
- Recording slots equal stored frames: True (detected_only=False).
- Status counts: `{'pose_ok': 201, 'filter_skip': 28, 'subject_lock_skip': 71, 'detection_failed': 0, 'undetected_unclassified': 0}`
- Source frames not in dump: 0
- Pose-dump ID holes: 0
- Left contact known: 11.0%
- Right contact known: 25.0%
- Cycle evidence (accepted, both sides published): 0 L / 0 R (candidates 5, accepted 0; accepted candidates 0 L / 0 R). Rejection counts: `{'no_known_swing': 2, 'unsupported_pose_gap': 2, 'duration_below_min': 1}`.
- Longest continuous valid pose segment: **3.77 s**. A cycle longer than that interval is accepted only when every internal pose gap is at most one missed sample; unsupported gaps are rejected as `unsupported_pose_gap` and are not interpolated.
- Laterality withhold: Withheld: contact coverage is insufficient (L 11% / R 25% known). A symmetry percentage is not validated from these inputs.
- Gait-quality withhold: Withheld: contact coverage is insufficient (L 11% / R 25% known). The gait-quality heuristic is not validated from these inputs.
- Summary gait quality: Not available
- Summary walking speed: 0.55 m/s (estimated) (estimated=True, reportable=True, not a measured speed).
- Phase-chart annotation: `Estimated speed 0.55 m/s (estimated) | Stance  3%
Swing   15%
Unknown 82%`
- Cursor labels ok: True; hover≠playback: True; pause/step/seek: True

### Fast Walking (`fast_walking_v2.mp4`)

- Source decoded: **2261** frames, 68.52 s @ 33.000 fps.
- Analyzed dump: **2261** stored frames.
- Analyzed span: **68.48 s** (last−first original timestamp, includes gaps).
- Cumulative valid duration: **45.45 s** (sum of contiguous pose_ok runs; n−1 intervals per run).
- Coverage: **66.4%** of analyzed span; longest valid segment **0.91 s** (220 runs).
- Recording slots equal stored frames: True (detected_only=False).
- Status counts: `{'pose_ok': 1720, 'filter_skip': 334, 'subject_lock_skip': 0, 'detection_failed': 207, 'undetected_unclassified': 0}`
- Source frames not in dump: 0
- Pose-dump ID holes: 0
- Left contact known: 3.9%
- Right contact known: 32.8%
- Cycle evidence (accepted, both sides published): 2 L / 1 R (candidates 185, accepted 3; accepted candidates 2 L / 1 R). Rejection counts: `{'unsupported_pose_gap': 125, 'duration_below_min': 48, 'no_known_swing': 8, 'duration_above_max': 1}`.
- Longest continuous valid pose segment: **0.91 s**. A cycle longer than that interval is accepted only when every internal pose gap is at most one missed sample; unsupported gaps are rejected as `unsupported_pose_gap` and are not interpolated.
- Laterality withhold: Withheld: contact coverage is insufficient (L 4% / R 33% known). A symmetry percentage is not validated from these inputs.
- Gait-quality withhold: Withheld: contact coverage is insufficient (L 4% / R 33% known). The gait-quality heuristic is not validated from these inputs.
- Summary gait quality: Not available
- Summary walking speed: 0.32 m/s (estimated) (estimated=True, reportable=True, not a measured speed).
- Phase-chart annotation: `Estimated speed 0.32 m/s (estimated) | Stance  17%
Swing   1%
Unknown 82%`
- Cursor labels ok: True; hover≠playback: True; pause/step/seek: True

## Verified fixes

- Analyze (`_run_full_analysis`) now requests `analysis_mode=full` so `max_frames=None`. Examples (`_load_demo_gait`) request labeled preview (first 900 frames). Static source inspection of those call sites is in `tests/test_analysis_mode.py`.
- `usable_duration` is replaced by analyzed span, cumulative valid duration, coverage, and longest valid segment using original timestamps (n−1 intervals per contiguous valid run).
- Binary-contact HS–HS intervals are exported as candidates. Valid cycles require known stance, confirmed swing, and no unsupported pose gap. Accepted left and right cycles are both published.
- Cycle-candidate CSV/JSON include side, start/end source frames, duration, valid-pose coverage, longest internal gap, known stance/swing, and the acceptance or rejection reason. Gaps larger than one missed sample are not bridged.
- The 0.100 m gate is documented as body-scale +Y vs floor_y confirmed-flight. Near-ground without a prior phase is Unknown, not stance.
- Compare Session A/B now carry an explicit analysis mode. Session B no longer silently applies `DEMO_MAX_FRAMES`. Full-versus-preview comparisons are refused.

## Unequal contact coverage

Unknown contact is a missing observation (`not_confirmed_flight` dominates). Unequal L/R known percentages and unilateral cycle counts are coverage gaps, not a gait-laterality finding. Overview / Results withhold symmetry and the gait-quality heuristic for all three examples.

## Remaining causes

- Fast Walking binary-contact HS–HS intervals can far outnumber known left-contact frames. Those intervals are exported as candidates; only intervals with known stance, confirmed swing, and no unsupported pose gap count as valid cycles.
- Monocular COM–BoS remains biomechanically unavailable.
- Walking speed, when shown, is an **estimate** scaled to default stature, not a measured overground speed and not a 0.55 m/s fallback constant.
- The 0.100 m clearance gate is estimated body-scale metres on +Y vs floor_y. It confirms flight; it does not classify stance. Near-ground off bits stay Unknown.
- Fast Walking longest continuous valid pose segment is 0.91 s. 125 candidate(s) were rejected as `unsupported_pose_gap` rather than bridging those holes. Published accepted cycles: 2 L / 1 R (both sides; no silent one-side ranking).

## Checks not performed

- Force-plate or instrumented-treadmill GRF
- Marker-based optical motion capture
- IMU / wearable validation
- Clinician-scored gait quality instrument
- Live GUI click-through of Examples (preview) vs Analyze (full source) on a physical display
- Live Compare A/B click-through: pin a preview session against a full session and confirm the mixed-mode block in the UI
- Live Compare playback of Session A/B after Analyze Session B inherits full vs preview
- MediaPipe re-inference (this audit reused the 2026-11-03 13:07 UTC full-analysis dumps)

## Upload set

See the folder listing at the end of the agent reply.
