# Presentation Checklist

Use this script for a **5–7 minute** StableWalk demo to reviewers, classmates, or stakeholders.

## Before you start (5 min setup)

- [ ] Open terminal in `StableWalk/`, activate venv
- [ ] Run `python scripts/generate_sample_outputs.py` (optional slide assets)
- [ ] Close unrelated apps; set display scaling to 100%
- [ ] Test network if using URL preset, or prepare local `data/videos/my_walk.mp4`
- [ ] Launch GUI once to confirm MediaPipe loads: `python main.py --gui`

## Slide outline (optional)

1. **Problem** — Gait stability assessment usually needs lab equipment; we estimate from video.
2. **Approach** — Pose estimation → 3D body → contact → stability + virtual forces.
3. **Live demo** — GUI walkthrough (see below).
4. **Results** — Show `docs/screenshots/*.png` or export JSON.
5. **Limitations** — Not clinical; monocular depth is approximate.
6. **Future** — Force plates calibration, multi-camera, real-time mobile.

## Live demo script (~5 min)

### 1. Intro (30 s)

> "StableWalk is an AI computer-vision lab that analyzes walking videos — pose, 3D motion, stability score, foot contact, and estimated ground reaction forces."

### 2. Launch & analyze (90 s)

- Show header: *Computer Vision Gait Analysis Laboratory*
- Select preset → **Analyze**
- While waiting: explain pipeline stages (video → pose → enrichment → metrics)

### 3. Playback (60 s)

- **Play** at 1× speed
- Point out: video overlay, 3D volumetric body, contact strip, frame slider

### 4. Analysis tabs (90 s)

| Tab | Talking point |
|-----|----------------|
| Conclusions | One-page research summary |
| Stability | Ten dimensions, score /100 |
| Foot Contact | Stance vs swing, asymmetry |
| Virtual GRF | Loading curves, impact peaks |
| Gait Charts | Cadence, symmetry, speed |

### 5. Export & wrap (30 s)

- **File → Export Analysis…**
- Restate disclaimer: screening tool, not diagnosis

## Backup plan

If analysis fails on stage:

```powershell
python main.py --gui walking_demo
```

Uses bundled pose JSON under `data/output/pose_runs/`.

## Q&A prep

| Question | Answer |
|----------|--------|
| How accurate is stability? | Relative screening from pose quality; validated against biomechanics literature heuristics, not clinical trials. |
| Why virtual GRF? | Estimates vertical loading from motion + contact when no force plate exists. |
| Can it run on phone video? | Yes, if full body visible and stable framing. |
| Open source stack? | Python, OpenCV, MediaPipe, Matplotlib, Tk/Streamlit. |

## After demo

- [ ] Save exported JSON for portfolio
- [ ] Capture GUI screenshots for report (`Win+Shift+S`)
- [ ] Run `pytest tests/ -q` to show test coverage if asked
