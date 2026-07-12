# StableWalk Demo Video Sources

Research-oriented demo protocol (v1). **Do not install a video solely because a filename already exists.**

## Categories

| UI category | Internal key | Target file | Primary dataset |
|-------------|--------------|-------------|-----------------|
| Abnormal | `abnormal` | `abnormal_gait.mp4` | [GAVD](https://github.com/Rahmyyy/GAVD) |
| Normal | `normal` | `normal_gait.mp4` | [Health&Gait](https://zenodo.org/records/14039922) UGS |
| Performance | `athletic` | `athletic_walking.mp4` | Health&Gait FGS |

The internal key `athletic` is retained for backward compatibility. The UI label is **Performance**.

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

**Replacement status:** Pending GAVD candidate validation.

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

### Current installed file (legacy — **REJECTED**)

| Field | Value |
|-------|-------|
| Category | Normal |
| Dataset | Pexels stock footage (deprecated) |
| Status | **REJECT** |
| Known limitations | Subject extremely small/distant; poor foot landmark pixels; ~26% analysis completeness; 0 usable gait cycles |

**Replacement status:** Pending Health&Gait UGS trial + validated MP4.

---

## 3. Performance — Health&Gait (FGS)

### Dataset

- **Dataset:** Health&Gait — fast gait speed (FGS)
- **Source URL:** https://zenodo.org/records/14039922
- **UI label:** Performance (not “Athletic” unless explicit athletic ground truth exists)

### Selection criteria (FGS = fast gait speed)

- same visibility/evidence gates as Normal
- select using **Velocity_FGS** / cadence metadata — controlled higher-speed locomotion
- not stock footage of a person who visually appears athletic

### Current installed file (legacy — **REJECTED**)

| Field | Value |
|-------|-------|
| Category | Performance (`athletic` key) |
| Dataset | Pexels stock footage (deprecated) |
| Status | **REJECT** |
| Known limitations | No scientific athletic label; side-view geometry differs from other demos; ~2.38 s duration; ~1 usable gait cycle; not comparable across demos |

**Replacement status:** Pending Health&Gait FGS trial + validated MP4.

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
