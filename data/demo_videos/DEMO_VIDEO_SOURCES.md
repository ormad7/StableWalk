# StableWalk Demo Video Sources

Research-oriented demo protocol (v1). **Do not install a video solely because a filename already exists.**

## Categories

| UI category | Internal key | Target file | Primary dataset |
|-------------|--------------|-------------|-----------------|
| Abnormal Gait | `abnormal` | `abnormal_gait_v2.mp4` | [GAVD](https://github.com/Rahmyyy/GAVD) |
| Normal Gait | `normal` | `normal_gait_v2.mp4` | [GAVD](https://github.com/Rahmyyy/GAVD) `gait_pat=normal` |
| Fast Walking | `athletic` | `fast_walking_v2.mp4` | [Twente RGB](https://zenodo.org/records/6644593) Subj04 `walk_54` (5.4 km/h) |

The internal key `athletic` is retained for backward compatibility. The UI label is **Fast Walking**.

---

## Selection workflow

```bash
# Full protocol report (GAVD filter + Health&Gait inspection)
python scripts/demo_video_selection_workflow.py

# GAVD abnormal metadata candidates
python scripts/select_gavd_abnormal_candidate.py

# Health&Gait sample archive inspection (~62 MB)
python scripts/inspect_healthgait_samples.py

# Mandatory before installing any candidate MP4
python scripts/validate_demo_candidate.py --video path/to/candidate.mp4 --category abnormal
```

Validator verdicts:

- `ACCEPT` — suitable for demo installation
- `ACCEPT_WITH_LIMITATIONS` — usable with documented caveats in this file
- `REJECT` — do not install

---

## 1. Abnormal — GAVD

### Dataset

- **Dataset:** Gait Abnormality in Video Dataset (GAVD)
- **Source URL:** https://github.com/Rahmyyy/GAVD
- **Paper:** IEEE Access 2025 — [DOI 10.1109/ACCESS.2025.3545787](https://ieeexplore.ieee.org/document/10921672)

GAVD provides **annotations and metadata only**. Raw videos are **not** redistributed. Retrieve videos independently from public YouTube URLs in the annotation files. Comply with YouTube Terms of Service and institutional ethics requirements.

### Metadata filter (`select_gavd_abnormal_candidate.py`)

Candidates must satisfy:

- `dataset` = Abnormal Gait
- clinically annotated gait pattern (`gait_pat`) — not chosen by label alone
- **exclude** walker / cane / crutch / assistive-device keywords
- dominant camera view: frontal, sagittal (left/right side), or mild oblique
- subject bbox height ≥ ~35% of frame height (metadata bbox)
- ≥ ~90 annotated frames in sequence (~3+ estimated gait cycles)
- one dominant walking direction per `seq`

### Installation steps

1. Run `python scripts/select_gavd_abnormal_candidate.py`
2. Download top eligible YouTube URL (e.g. with `yt-dlp`, respecting source access rules)
3. Trim to annotated `seq` frame range if needed
4. Run `python scripts/validate_demo_candidate.py --video <path> --category abnormal`
5. Copy to `data/demo_videos/abnormal_gait.mp4` only after `ACCEPT` or `ACCEPT_WITH_LIMITATIONS`

### Current installed file (legacy — **REJECTED**)

| Field | Value |
|-------|-------|
| Category | Abnormal |
| Dataset | University of Utah NeuroLogic Examination (deprecated) |
| Status | **REJECT** — superseded by GAVD protocol |
| Known limitations | Walker-assisted gait; foot/walker occlusion; 0 usable gait cycles; external support stabilizes pelvis |

**Replacement status:** GAVD Parkinson RGB candidates were retrieved and validated on 2026-10-29. Both failed the existing pipeline (`gavd_parkinsons_B5hrxKe2nP8_cljan9b4p0.mp4` REJECT: 0% pose-valid / 0 cycles; `gavd_parkinsons_bmi1hYOnTHs_cljw65m080.mp4` REJECT: 20.3% pose-valid). The Utah walker clip remains installed. Parkinson’s was **not** labeled as dementia.

---

## 2. Normal — Health&Gait (UGS)

### Dataset

- **Dataset:** Health&Gait
- **Source URL:** https://zenodo.org/records/14039922
- **Paper:** Scientific Data 2025 — [DOI 10.1038/s41597-024-04327-4](https://doi.org/10.1038/s41597-024-04327-4)

### Sample archive inspection

`dataset_samples.zip` (~62 MB) contains:

- `gait_parameters.csv`, `gait_parameters_estimation.csv`, `participants_measures.csv`
- Silhouette frame exports for sample participant(s), e.g. `silhouette/PA000/UGS/...`
- **No raw RGB MP4** in the public sample archive

The full ~26.8 GB Zenodo archive provides silhouettes, pose JSON, optical flow, and semantic segmentation — **not raw RGB walking video** (stated limitation in the Scientific Data paper).

### Selection criteria (UGS = usual gait speed)

- controlled indoor recording environment
- full body visible in sourced MP4
- high pose confidence after StableWalk prevalidation
- clear feet and heels
- ≥ 3 usable gait cycles
- moderate gait speed (Velocity_UGS near cohort median when full CSV available)
- stable camera, no meaningful occlusion

### Current installed file (2026-10-29)

Health&Gait UGS remains the preferred *protocol* label, but the public release has no RGB. The installed clip is a GAVD sequence whose **documented** label is `gait_pat=normal`.

| Field | Value |
|-------|-------|
| Category | Normal |
| Dataset | GAVD |
| Sequence ID | `cljo3b2dy002l3n6l270vpzp8` |
| YouTube ID / URL | `3KnFt8bH3tE` / https://www.youtube.com/watch?v=3KnFt8bH3tE |
| Original label | `gait_pat=normal` |
| Camera view | sagittal (pipeline: SAGITTAL_LEFT) |
| Duration / FPS | 19.12 s / 29.97 |
| Pose valid frame % | 92.1 |
| Usable gait cycles | 12 (detected 12); heel strikes L 14 / R 5 |
| Reason selected | Full-body sagittal walk; validator ACCEPT; comparable side view |
| Known limitations | GAVD `dataset` column is `Abnormal Gait` (compilation source file); sequence-level label is `normal`. Left/right heel-strike counts are unequal. Not a Health&Gait UGS trial. |
| Validator verdict | ACCEPT |
| Validated on | 2026-10-29 |

Legacy Pexels `normal_gait.mp4` was backed up under `data/demo_videos/backups/examples_20261029_150528/`.

---

## 3. Fast Walking — Twente wearable-system validation RGB

### Dataset

- **Dataset:** RGB data [front and side views] of the wearable system validation study
- **Source URL:** https://zenodo.org/records/6644593
- **DOI:** [10.5281/zenodo.6644593](https://doi.org/10.5281/zenodo.6644593)
- **License:** CC BY 4.0 (anonymized RGB)
- **Companion paper:** Wang, Basu, Durandau & Sartori, *Wearable Technologies* 2023 — [10.1017/wtc.2023.7](https://doi.org/10.1017/wtc.2023.7)
- **UI label:** Fast Walking (internal key `athletic`)

Health&Gait FGS RGB is still not in the public release and was not used. The Kuopio dataset described in PMC11385067 (Zenodo 10559504) was inspected first: public files are mocap/IMU/OpenPose JSON only — the paper states no video data are released.

### Selection criteria

- documented walking (not running) speed condition
- full body and feet visible
- preferably a fixed side camera
- original FPS and timing preserved (no speed-up, loop, or interpolation)
- `athletic_walking.mp4` is never used as a substitute

### Current installed file (2026-11-01)

| Field | Value |
|-------|-------|
| Category | Fast Walking (`athletic` key) |
| Dataset | Twente wearable-system validation RGB |
| Trial ID | Subj04 / `walk_54_side_anonymized.avi` |
| Documented speed | treadmill walking 5.4 km/h (`walk_54`); running starts at 6.3 km/h |
| Camera view | fixed left-side camera |
| Duration / FPS | 68.52 s / 33.000 (2261 frames; stream-copied AVI→MP4) |
| SHA-256 | `122779bb974ec17aa579af5b860797b34d800f4dcc32aa9c2cd138b75fd621d4` |
| License | CC BY 4.0 |
| Known limitations | Treadmill (not overground). Face/body anonymized as released. Analysis uses up to `DEMO_MAX_FRAMES` (900). |
| Status | Installed as `fast_walking_v2.mp4` |

---

## Prevalidation report fields

`validate_demo_candidate.py` reports:

| Metric | Description |
|--------|-------------|
| Pose valid frame % | Frames with detected pose |
| Foot / heel / toe visibility | Landmark visibility rates |
| Mean landmark confidence | MediaPipe visibility mean |
| Subject bbox height ratio | Estimated body extent / frame height |
| Camera motion score | Hip-center displacement stability |
| Estimated usable gait cycles | StableWalk gait evidence gate |
| View type | Estimated camera view |
| Analysis completeness % | Stability v2 evidence mass |

---

## Deprecation notice (2026-07)

The following sources are **deprecated** and must not be used for new installs:

| Source | Former demo | Reason |
|--------|-------------|--------|
| Utah neuropathic + walker clips | Abnormal | External support, occlusion, 0 cycles |
| Pexels 5320110 | Normal | Distant subject, poor feet, 0 cycles |
| Pexels 27727783 / stock walking | Performance | No ground truth, short clip, heterogeneous view |

Run `python scripts/demo_video_selection_workflow.py --validate-installed` to regenerate validation reports for currently installed files.

**Replacement guide:** See [RECOMMENDED_REPLACEMENTS.md](RECOMMENDED_REPLACEMENTS.md) and run `python scripts/fetch_recommended_demo_videos.py --all` to download candidates.

---

## Final selection record (fill after ACCEPT)

When a candidate is accepted, append a completed block:

```markdown
### Abnormal (installed)
- Category: Abnormal
- Dataset: GAVD
- Sequence ID: <seq>
- YouTube ID / URL: <id> / <url>
- Original label: <gait_pat>
- Camera view: <cam_view>
- Duration / FPS: <s> / <fps>
- Pose valid frame %: <pct>
- Usable gait cycles: <n>
- Reason selected: <text>
- Known limitations: <text>
- Validator verdict: ACCEPT | ACCEPT_WITH_LIMITATIONS
- Validated on: <date>
```

Repeat for Normal and Performance.
