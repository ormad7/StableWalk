# Demo gait video files

Research-oriented demo protocol — see **`DEMO_VIDEO_SOURCES.md`** in this folder.

| File | UI category | Internal key | Installed source |
|------|-------------|----------------|------------------|
| `abnormal_gait_v2.mp4` | Abnormal Gait | `abnormal` | GAVD `gait_pat=parkinsons`, YouTube `M-_cogKwXK4` |
| `normal_gait_v2.mp4` | Normal Gait | `normal` | GAVD `gait_pat=normal`, YouTube `eCCYhDSDlDc` |
| `fast_walking_v2.mp4` | Fast Walking | `athletic` | Twente RGB Zenodo 6644593, Subj04 `walk_54_side_anonymized.avi`, 5.4 km/h treadmill walking |

**Do not install videos without running:**

```bash
python scripts/validate_demo_candidate.py --video <candidate.mp4>
```

Selection workflow:

```bash
python scripts/demo_video_selection_workflow.py
python scripts/select_gavd_abnormal_candidate.py
python scripts/inspect_healthgait_samples.py
```

Legacy Pexels / Utah stock demos are **deprecated** — see `DEMO_VIDEO_SOURCES.md`.
