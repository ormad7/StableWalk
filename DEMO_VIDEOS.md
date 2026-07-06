# StableWalk Demo Gait Videos

See also **Demo Video Sources** in `README.md`.

## 1. Abnormal / Neuropathic Gait

**GUI label:** Abnormal / Neuropathic Gait Analysis

**Source institution:** University of Utah  
**Source:** NeuroLogic Examination  
**Page:** https://neurologicexam.med.utah.edu/adult/html/gait_abnormal.html  
**Video:** Neuropathic Gait (Utah identifier `gait_ab_10`, Kaltura `0_z1p8nsbi`)  
**Download page:** https://neurologicexam.med.utah.edu/adult/html/download_by_exam.html#gait_ab_10

**Local file:** `data/demo_videos/abnormal_gait.mp4`

Clinical walking example: right distal lower-extremity weakness with compensatory
high stepping for foot clearance. Downloaded from the official Utah MP4 mobile zip,
transcoded to H.264, trimmed to a continuous walking segment, and validated with
MediaPipe.

Download:

```
python scripts/download_utah_abnormal_demo.py
```

Metadata is written to `data/demo_videos/utah_abnormal_source.json`.

**Presentation note:** Shows neuropathic gait pattern from the Utah clinical library.
StableWalk does **not** diagnose pathology beyond the source description.

---

## 2. Normal Gait

**Source:** Pexels 5320110 — https://www.pexels.com/video/a-man-walking-towards-the-camera-5320110/  
**Local file:** `data/demo_videos/normal_gait.mp4`

---

## 3. Athletic Walking

**GUI title:** Athletic Walking Analysis

**Video title:** A man walking on a tennis court  
**Source platform:** Pexels  
**Creator:** Lola bertoncelli  
**Official source page:** https://www.pexels.com/video/a-man-walking-on-a-tennis-court-27727783/  
**Local file:** `data/demo_videos/athletic_walking.mp4`  
**Purpose:** Athletic walking gait-analysis demo

Rear-view sportswear walking on an outdoor tennis court. Replaced Pexels 5823532 on 2026-07-06
after the Athletic vs Normal investigation confirmed the previous clip was technically marginal
(low heel visibility, LOW step-detection confidence on front-view outdoor footage).

| Metric | Previous (5823532) | Current (27727783) |
|--------|-------------------|-------------------|
| Pose detection | 94% | **100%** |
| Heel visibility | 0.61 | **0.94** |
| Step detection confidence | LOW | **HIGH** |
| Mixkit 596 candidate | — | Rejected (feet occluded by railing) |

Previous backup: `data/demo_videos/athletic_walking_pexels5823532_backup_20260706.mp4`

---

## Validation

```
python scripts/validate_demo_videos.py
```
