# Demo gait video files

Research-oriented demo protocol — see **`DEMO_VIDEO_SOURCES.md`** in this folder.

| File | UI category | Internal key | Target source |
|------|-------------|--------------|---------------|
| `abnormal_gait.mp4` | Abnormal | `abnormal` | GAVD (YouTube URL from annotations) |
| `normal_gait.mp4` | Normal | `normal` | Health&Gait UGS |
| `athletic_walking.mp4` | Performance | `athletic` | Health&Gait FGS |

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
