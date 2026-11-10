# Gait-analysis pipeline — end-to-end validation

**Date:** 2026-10-18  
**Method:** cached pose JSON → `pose_sequence_to_gait_motion` → 16-joint wireframe, gait cycles, ACF, FFT/Welch, half-cycle asymmetry. MediaPipe was not re-run. Cadence and gait-quality were not rewritten.  
**Checks:** 30 PASS, 0 FAIL (headless). Related pytest: 48 passed (`test_gait_wireframe`, `test_gait_acf`, `test_gait_spectrum`, `test_gait_asymmetry_moments`, `test_overview_signal_spectral`, `test_real_video_gait_validation`).  
**Machine log:** `docs/_gait_pipeline_e2e.json`

This is **not** clinical validation, laboratory inverse-dynamics, or a claim that demo videos are optically calibrated.

## Clips

| Clip | File | Kind | Video | Pose / detected | FPS | Visibility | Analysis |
|------|------|------|------:|----------------:|----:|-----------:|---------:|
| Normal symmetric | `normal_gait.mp4` | unobstructed walk | 120 @ 25.00 Hz | 120 / 120 | 25.00 | 1.00 | 2.01 s |
| Asymmetric | `abnormal_gait.mp4` | aid / asymmetric | 150 @ 29.97 Hz | 120 / 120 | 29.97 | 1.00 | 0.35 s |
| Sparse / noisy | `athletic_walking.mp4` | dropped detections | 120 @ 23.98 Hz | 120 / **27** | 23.98 | 1.00 on detected | 0.25 s |
| Low-confidence overlay | same + in-memory vis×0.22 + XY jitter | withheld-results probe | 120 | 27 | 23.98 | 0.22 | 0.09 s |

`weightgait_limping_gait.mp4` has no pose JSON, so `abnormal_gait.mp4` is the asymmetric clip.

## Checklist

| Requirement | Result | Evidence |
|-------------|--------|----------|
| 16-joint model complete | **PASS** | All 16 ids on every frame (normal 120, asymmetric 120, athletic 27). Pelvis-centered body scale. |
| Timestamps and FPS | **PASS** | Pose FPS = recording FPS = container FPS. Snapshot `time_s` matches `PoseFrame.time_seconds`. Detected-frame count = recording length. Asymmetric pose covers 120 / 150 container frames (not padded). |
| ACF realistic gait cycle | **PASS** (normal); **withheld** (asymmetric, low-conf) | Normal cycle **1.04 s** (cadence 64.5 spm ⇒ ~1.07 s/step). Asymmetric ACF unavailable (insufficient). Degraded overlay withholds cycle. |
| FFT and PSD consistent | **PASS** per signal | Median \|FFT − Welch f₀\| = **0 Hz** on every clip that produced a spectrum. |
| Phase uses equivalent gait phases | **PASS** | Right(t) vs left(t+T/2): match skew **0**. Normal T=1.86 s, asymmetric T=1.74 s. |
| Asymmetric moments vs events | **PASS** as event-aligned kinematics | Asymmetric: t=2.07 s near right heel-strike (Δ 0.07 s), t=0.27 s near left toe-off (Δ 0.07 s). Normal has one moderate moment at first left heel-strike — not inflated to “stable/normal”. No diagnostic wording. |
| Video / graphs / 3D sync | **PASS** (source + timestamps) | Same recording timestamps. Click-seek calls `_seek_motion_workspace_time` → `_sync_all_to_frame`. Playhead-only during play. **Not** a timed Tk capture this run. |
| Existing features still work | **PASS** | Cadence/cycles unchanged after ACF/FFT/moments (normal 64.52 spm, 2 cycles; asymmetric 51.87 spm, 2 cycles). Object identity of the recording unchanged. |
| No duplicate processing / UI freeze | **PASS** | `analyze_gait_autocorrelation` / `analyze_gait_spectrum` are **not** on the GUI load path (they would duplicate `analyze_spectral`). Spectral analysis is a worker thread. Session cache forbids re-analysis during playback. Overview graphs reuse `_spectral_report`. |

## Numbers that matter

**Normal (`normal_gait.mp4`)**  
16-joint complete. ACF cycle 1.04 s, regularity 0.33, confidence moderate. Session spectral f₀ **0.42 Hz** (band edge of a 4.76 s clip — not cadence). Hip channels peak at 1.04 Hz, matching the ACF step. One moment at 0.08 s / frame 2, score 61.8, at left heel-strike.

**Asymmetric (`abnormal_gait.mp4`)**  
16-joint complete. ACF withheld. Two moments (scores 76.3 and 32.9) within one frame of detected gait events. Cadence 51.87 spm unchanged.

**Sparse athletic (`athletic_walking.mp4`)**  
Only 27 detected frames. ACF cycle 1.13 s at **low** confidence. Cadence withheld (1 complete cycle). Spectrum notes dropped-detection gaps.

**Low-confidence overlay**  
ACF, spectrum, and moments all **unavailable**. No invented cycle or f₀.

## Remaining limitations

- Cached poses only. The MP4s were not re-estimated.
- Short demo clips (~4–8 s). Session FFT/Welch f₀ can sit at the walking-band floor (0.42 / 0.50 Hz) even when ACF finds a ~1 s step. That is reported, not replaced with cadence.
- Athletic pose JSON detects 27 / 120 frames. That is sparse tracking, not a fully observed noisy walk.
- No force-plate or annotated limp events. Moment-to-event Δ is vs detected heel-strike / toe-off.
- GUI playhead sync was verified in source and by shared timestamps, not by a timed on-screen capture.
- Dedicated gait ACF/FFT modules are callable and tested; the Overview Signal & Spectral panel displays `analyze_spectral` so load does not run two spectral pipelines.

## Files changed

No analysis math, metrics, or GUI playback code was changed. The timestamp FAIL on the first harness pass was a **validator** error (list index vs `frame_index / fps` on sparsely detected frames).

- `scripts/gait_pipeline_e2e_validation.py` — headless harness
- `docs/gait_pipeline_e2e_validation.md` — this report
- `docs/_gait_pipeline_e2e.json` — raw evidence
