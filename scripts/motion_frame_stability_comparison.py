"""
Compare global vs gait-frame pelvis metrics and recalculated stability scores.

Label-blind: demo keys are for report grouping only.

Usage:
  python scripts/motion_frame_stability_comparison.py --use-cached-poses

Writes:
  data/output/reports/motion_frame_stability_comparison.txt
  data/output/reports/motion_frame_stability_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.analysis.stability_scoring import (
    _build_context,
    analyze_biomech_stability,
    compute_pelvis_motion_metrics,
)
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.enrichment import enrich_pose_sequence

VIDEOS = (
    ("abnormal", "abnormal_gait"),
    ("normal", "normal_gait"),
    ("athletic", "athletic_walking"),
)

METRIC_KEYS = (
    "global_pelvis_displacement_m",
    "forward_pelvis_displacement_m",
    "mediolateral_pelvis_sway_m",
    "vertical_pelvis_oscillation_m",
    "root_relative_trunk_sway_m",
)


def _load_sequence(stem: str, *, use_cached: bool):
    path = config.POSES_DIR / f"{stem}_poses.json"
    if not path.is_file():
        if not use_cached:
            video = config.DEMO_VIDEOS_DIR / f"{stem}.mp4"
            from stablewalk.pose.estimation import PoseEstimator
            from stablewalk.video_processing import VideoProcessor

            frames_dir = config.FRAMES_DIR / stem
            PoseEstimator().estimate_video(
                video,
                output_json=path,
                frames_dir=frames_dir,
            )
        else:
            raise FileNotFoundError(path)
    seq = load_pose_sequence(path)
    enrich_pose_sequence(seq)
    return seq


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cached-poses", action="store_true")
    args = parser.parse_args()

    rows: list[dict] = []
    lines = [
        "Motion-frame stability comparison (global vs root-relative gait frame)",
        "=" * 72,
        "",
    ]

    for label, stem in VIDEOS:
        seq = _load_sequence(stem, use_cached=args.use_cached_poses)
        ctx = _build_context(seq)
        stability = analyze_biomech_stability(seq)
        pelvis = compute_pelvis_motion_metrics(ctx) if ctx else {}

        row = {
            "group_label": label,
            "video_stem": stem,
            "stability_score": stability.score,
            "classification": stability.classification,
            "metrics": {k: pelvis.get(k) for k in METRIC_KEYS},
            "pelvis_stability_score": None,
            "trunk_stability_score": None,
            "domain_scores": {},
        }
        for m in stability.metrics:
            row["domain_scores"][m.key] = m.score
            if m.key == "pelvis_stability":
                row["pelvis_stability_score"] = m.score
            if m.key == "trunk_stability":
                row["trunk_stability_score"] = m.score

        rows.append(row)

        lines.append(f"[{label}] {stem}")
        lines.append(f"  Overall stability score: {stability.score:.1f} ({stability.classification})")
        lines.append("  Pelvis decomposition:")
        lines.append(f"    GLOBAL pelvis displacement (m):     {_fmt(pelvis.get('global_pelvis_displacement_m'))}")
        lines.append(f"    FORWARD pelvis displacement (m):    {_fmt(pelvis.get('forward_pelvis_displacement_m'))}")
        lines.append(f"    MEDIOLATERAL pelvis sway (m):       {_fmt(pelvis.get('mediolateral_pelvis_sway_m'))}")
        lines.append(f"    VERTICAL pelvis oscillation (m):     {_fmt(pelvis.get('vertical_pelvis_oscillation_m'))}")
        lines.append(f"    ROOT-RELATIVE trunk sway (m std):   {_fmt(pelvis.get('root_relative_trunk_sway_m'))}")
        lines.append(f"  Domain scores: pelvis={row['pelvis_stability_score']} trunk={row['trunk_stability_score']}")
        lines.append("")

    athletic = next(r for r in rows if r["group_label"] == "athletic")
    normal = next(r for r in rows if r["group_label"] == "normal")
    if athletic["stability_score"] < normal["stability_score"]:
        lines.append("Athletic score is lower than normal. Root-relative drivers:")
        for key in ("pelvis_stability_score", "trunk_stability_score"):
            a = athletic.get(key)
            n = normal.get(key)
            if a is not None and n is not None and a < n - 5:
                lines.append(f"  - {key}: athletic={a:.1f} vs normal={n:.1f}")
        for dk, av in athletic["domain_scores"].items():
            nv = normal["domain_scores"].get(dk)
            if av is not None and nv is not None and av < nv - 8:
                lines.append(f"  - {dk}: athletic={av:.1f} vs normal={nv:.1f}")
        lines.append(
            f"  ML sway: athletic={_fmt(athletic['metrics']['mediolateral_pelvis_sway_m'])} "
            f"vs normal={_fmt(normal['metrics']['mediolateral_pelvis_sway_m'])}"
        )

    out_dir = config.REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "motion_frame_stability_comparison.txt"
    json_path = out_dir / "motion_frame_stability_comparison.json"
    text = "\n".join(lines)
    txt_path.write_text(text, encoding="utf-8")
    json_path.write_text(json.dumps({"videos": rows}, indent=2), encoding="utf-8")
    print(text)
    print(f"\nWrote {txt_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
