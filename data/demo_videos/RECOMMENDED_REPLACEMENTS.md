# Recommended Demo Video Replacements

**Status (2026-07):** The three installed legacy demos (`abnormal_gait.mp4`, `normal_gait.mp4`, `athletic_walking.mp4`) are **REJECTED** by `validate_demo_candidate.py`. Replace them before relying on demo metrics.

## Validation results (fetched 2026-07-10)

| Candidate | Category | Verdict | Pose % | Bbox H | Feet % | Cycles | Why |
|-----------|----------|---------|--------|--------|--------|--------|-----|
| `weightgait_normal_gait.mp4` | Normal | REJECT | 95.8 | 0.69 | 100 | 0 | Short clip (3.4s); still **much better** than legacy Pexels for 3D graph |
| `weightgait_limping_gait.mp4` | Abnormal | REJECT | 100 | 0.67 | 100 | 1 | Short (2.1s); simulated limp — better pose than Utah walker clip |
| `weightgait_shuffle_gait.mp4` | Abnormal | (run validator) | — | — | — | — | Shuffle gait sample |
| GAVD YouTube (#2) | Abnormal | pending | — | — | — | ~16 est. | Download manually (`yt-dlp`); automated fetch blocked (HTTP 403) |

**Interim recommendation for 3D trajectory demo:** Use WeightGait samples until GAVD/Health&Gait RGB clips are validated — full body, high pose confidence, indoor camera. Gait-cycle metrics will stay low until longer clips are installed.

```bash
# Optional interim install (improves 3D graph; gait-cycle metrics still limited)
copy data\demo_videos\candidates\weightgait_normal_gait.mp4 data\demo_videos\normal_gait.mp4
copy data\demo_videos\candidates\weightgait_limping_gait.mp4 data\demo_videos\abnormal_gait.mp4
```

Then re-run **Analyze** in the GUI for each mode.

---

```bash
python scripts/fetch_recommended_demo_videos.py --all
python scripts/validate_demo_candidate.py --video data/demo_videos/candidates/weightgait_normal_gait.mp4 --category normal
```

Install only after `ACCEPT` or `ACCEPT_WITH_LIMITATIONS`:

```bash
copy data\demo_videos\candidates\weightgait_normal_gait.mp4 data\demo_videos\normal_gait.mp4
```

Re-analyze poses from the GUI (**Analyze**) or CLI after swapping files.

---

## 1. Abnormal

| Priority | Source | How to get it | Notes |
|----------|--------|---------------|-------|
| **1 (preferred)** | [GAVD](https://github.com/Rahmyyy/GAVD) | `python scripts/select_gavd_abnormal_candidate.py` then `yt-dlp` on top URL | Clinical labels; no walker/cane; side/front view |
| **2 (interim)** | [WeightGait](https://homepages.inf.ed.ac.uk/rbf/WeightGait/) limp/shuffle samples | `fetch_recommended_demo_videos.py --category abnormal` | Simulated limp/shuffle; ~60 frames; academic license |

**Top GAVD candidates** (from latest metadata filter):

| Rank | View | Est. cycles | URL |
|------|------|-------------|-----|
| 1 | right side | ~6 | https://www.youtube.com/watch?v=0CdEPm-VYwA |
| 2 | right side | ~16 | https://www.youtube.com/watch?v=ErkLnvUHQuc |
| 3 | right side | ~15 | https://www.youtube.com/watch?v=gXws-A4op-E |

```bash
yt-dlp -f "bv*[height<=720]" -o "data/demo_videos/candidates/gavd_abnormal_%(id)s.%(ext)s" "https://www.youtube.com/watch?v=0CdEPm-VYwA"
python scripts/validate_demo_candidate.py --video data/demo_videos/candidates/gavd_abnormal_0CdEPm-VYwA.mp4 --category abnormal
```

**Do not use:** Utah walker clip (current `abnormal_gait.mp4`) — external support, occlusion, 0 usable cycles.

---

## 2. Normal

| Priority | Source | Direct sample | Notes |
|----------|--------|---------------|-------|
| **1 (protocol)** | [Health&Gait UGS](https://zenodo.org/records/14039922) | Full 26.8 GB archive (silhouettes/pose; **no raw RGB** in public sample zip) | Best scientific label when RGB trial MP4 is available |
| **2 (interim)** | [WeightGait regular walk](https://datashare.ed.ac.uk/bitstream/handle/10283/8956/1_Normal_Gait.mp4?sequence=15&isAllowed=y) | Auto-download via fetch script | Indoor, full body, ~60 frames, frontal RGB-D camera |
| **3 (alternative)** | [Toronto Older Adults Gait Archive](https://doi.org/10.6084/m9.figshare.c.5515953) | Figshare `videos/` folder (CC-BY 4.0) | Older adults; front/back walks; 60 s clips |

**Do not use:** Pexels stock (current `normal_gait.mp4`) — subject too small, poor feet, 0 cycles.

---

## 3. Performance

| Priority | Source | Notes |
|----------|--------|-------|
| **1 (protocol)** | Health&Gait **FGS** (fast gait speed) | Same Zenodo limitation — need validated RGB trial clip |
| **2 (interim)** | WeightGait obstacle walk sample | Faster stepping over obstacle; not true FGS label |
| **3 (alternative)** | Younger participant from WeightGait full dataset (`All RGB Videos/`) | 1.7 GB download; pick regular walk with highest cadence after validation |

**Do not use:** Pexels athletic stock (current `athletic_walking.mp4`) — short (~2.4 s), side view mismatch, no ground truth.

---

## Validation checklist

```bash
python scripts/demo_video_selection_workflow.py --validate-installed
python scripts/validate_demo_candidate.py --video <candidate> --category <abnormal|normal|performance>
```

Look for:

- Pose valid frame % ≥ 70%
- Subject bbox height ratio ≥ 0.35
- ≥ 3 usable gait cycles (or document limitation)
- Clear feet/heels for contact detection
- Stable camera (low hip-center drift)

---

## Citations (required for academic sources)

- **GAVD:** IEEE Access 2025 — DOI [10.1109/ACCESS.2025.3545787](https://ieeexplore.ieee.org/document/10921672)
- **WeightGait:** Lochhead & Fisher, *Computers in Biology and Medicine*, 2025
- **Health&Gait:** Scientific Data 2025 — DOI [10.1038/s41597-024-04327-4](https://doi.org/10.1038/s41597-024-04327-4)
- **Toronto OAW:** Scientific Data 2022 — DOI [10.1038/s41597-022-01495-z](https://doi.org/10.1038/s41597-022-01495-z)
