# StableWalk Demo Gait Videos

**Research-oriented protocol:** see [data/demo_videos/DEMO_VIDEO_SOURCES.md](data/demo_videos/DEMO_VIDEO_SOURCES.md)

## Categories

| UI label | Internal key | File | Source |
|----------|--------------|------|--------|
| Abnormal Gait | `abnormal` | `abnormal_gait_v2.mp4` | [GAVD](https://github.com/Rahmyyy/GAVD) `gait_pat=parkinsons`, YouTube `M-_cogKwXK4` |
| Normal Gait | `normal` | `normal_gait_v2.mp4` | [GAVD](https://github.com/Rahmyyy/GAVD) `gait_pat=normal`, YouTube `eCCYhDSDlDc` |
| Fast Walking | `athletic` | `fast_walking_v2.mp4` | [Twente RGB](https://zenodo.org/records/6644593) Subj04 `walk_54` side, 5.4 km/h treadmill walking |

Legacy Pexels and Utah walker-assisted clips are **deprecated** and fail the new validator.

## Workflow

```bash
# Full selection report
python scripts/demo_video_selection_workflow.py

# Abnormal: filter GAVD annotations (metadata only)
python scripts/select_gavd_abnormal_candidate.py

# Normal / Performance: inspect Health&Gait samples first
python scripts/inspect_healthgait_samples.py

# Mandatory before installing any MP4
python scripts/validate_demo_candidate.py --video path/to/candidate.mp4 --category abnormal
```

Validator returns `ACCEPT`, `ACCEPT_WITH_LIMITATIONS`, or `REJECT`.

## Health&Gait note

The public Zenodo release provides silhouettes, pose, optical flow, and CSV metadata — **not raw RGB MP4**. Use metadata to select participant/trial (UGS for Normal, FGS for Performance), then supply a validated MP4 from an authorized/controlled source.

## GAVD note

GAVD provides annotations only. Download the referenced YouTube video independently and comply with source access requirements.
