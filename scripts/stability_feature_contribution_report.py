"""
Stability feature classification report and per-video feature contribution analysis.

Usage:
  python scripts/stability_feature_contribution_report.py --use-cached-poses

Writes:
  data/output/reports/stability_feature_classification.txt
  data/output/reports/stability_feature_contribution.txt
  data/output/reports/stability_feature_contribution.json
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
from stablewalk.analysis.stability_feature_registry import registry_by_domain, registry_report_lines
from stablewalk.analysis.stability_scoring import analyze_biomech_stability
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.enrichment import enrich_pose_sequence

VIDEOS = (
    ("abnormal", "abnormal_gait"),
    ("normal", "normal_gait"),
    ("athletic", "athletic_walking"),
)


def _load(stem: str, *, use_cached: bool):
    path = config.POSES_DIR / f"{stem}_poses.json"
    if not path.is_file() and not use_cached:
        raise FileNotFoundError(path)
    seq = load_pose_sequence(path)
    enrich_pose_sequence(seq)
    return seq


def _feature_rows(result) -> list[dict]:
    rows: list[dict] = []
    for m in result.metrics:
        for d in m.values.get("debug_features") or []:
            rows.append(
                {
                    "domain": m.key,
                    "feature": d.get("feature"),
                    "raw": d.get("raw"),
                    "normalized": d.get("normalized"),
                    "component_score": d.get("component_score"),
                    "note": d.get("note"),
                }
            )
        for k, v in m.values.items():
            if k in ("debug_features", "controlled_motion_detail", "rom_deg"):
                continue
            if isinstance(v, (int, float, str, bool)) or v is None:
                rows.append({"domain": m.key, "feature": k, "raw": v, "normalized": v})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cached-poses", action="store_true")
    args = parser.parse_args()

    out_dir = config.REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    class_lines = [
        "StableWalk stability feature classification (v2)",
        "=" * 72,
        "",
        *registry_report_lines(),
        "",
        "MAGNITUDE features listed above are informational — they do not directly reduce scores.",
    ]
    class_path = out_dir / "stability_feature_classification.txt"
    class_path.write_text("\n".join(class_lines), encoding="utf-8")

    report: dict = {
        "feature_classification": registry_by_domain(),
        "videos": [],
    }
    text_lines = [
        "Stability feature contribution report (control-oriented scoring)",
        "=" * 72,
        "",
    ]

    for label, stem in VIDEOS:
        seq = _load(stem, use_cached=args.use_cached_poses)
        result = analyze_biomech_stability(seq)
        video_block = {
            "group_label": label,
            "video_stem": stem,
            "overall_score": result.score,
            "classification": result.classification,
            "view": result.view_display_name,
            "domains": [],
            "features": _feature_rows(result),
        }
        text_lines.append(f"[{label}] {stem}")
        text_lines.append(f"  Overall: {result.score:.1f} ({result.classification})  View: {result.view_display_name}")
        text_lines.append(f"  {'Domain':<24}{'Score':>8}{'Conf':>8}{'Contrib':>10}{'Status':>14}")
        text_lines.append("  " + "-" * 66)

        for c in result.contributions:
            video_block["domains"].append(c.to_dict())
            score_s = "N/A" if c.score is None else f"{c.score:.1f}"
            text_lines.append(
                f"  {c.name:<24}{score_s:>8}{c.confidence:>8.2f}{c.contribution:>10.2f}{c.availability:>14}"
            )

        temporal = result.metric("temporal_symmetry")
        if temporal and temporal.values:
            v = temporal.values
            text_lines.append("  Temporal breakdown:")
            text_lines.append(f"    raw_temporal_asymmetry:      {v.get('raw_temporal_asymmetry')}")
            text_lines.append(f"    normalized_temporal_penalty: {v.get('normalized_temporal_penalty')}")
            text_lines.append(f"    final_temporal_score:        {v.get('final_temporal_score')}")

        joint = result.metric("joint_smoothness")
        if joint and joint.values.get("controlled_motion_detail"):
            cm = joint.values["controlled_motion_detail"]
            text_lines.append("  Controlled motion (ROM informational, not penalized):")
            for j in cm.get("joints", []):
                text_lines.append(
                    f"    {j.get('key')}: ROM={j.get('rom_deg')}° score={j.get('controlled_motion_score')} "
                    f"repeat={j.get('repeatability_score')} smooth={j.get('smoothness_score')} "
                    f"LR={j.get('lr_symmetry_score')}"
                )
        text_lines.append("")

        report["videos"].append(video_block)

    athletic = next(v for v in report["videos"] if v["group_label"] == "athletic")
    normal = next(v for v in report["videos"] if v["group_label"] == "normal")
    if athletic["overall_score"] < normal["overall_score"]:
        text_lines.append("Athletic vs normal — biomechanical drivers (not ROM magnitude):")
        a_domains = {d["key"]: d for d in athletic["domains"]}
        n_domains = {d["key"]: d for d in normal["domains"]}
        for key in a_domains:
            a, n = a_domains.get(key), n_domains.get(key)
            if not a or not n or a.get("score") is None or n.get("score") is None:
                continue
            if a["score"] < n["score"] - 8:
                text_lines.append(
                    f"  {key}: athletic={a['score']:.1f} normal={n['score']:.1f} "
                    f"(conf {a.get('confidence')} vs {n.get('confidence')})"
                )
        a_cm = next(
            (f for f in athletic["features"] if f["feature"] == "controlled_motion_overall"), None
        )
        n_cm = next(
            (f for f in normal["features"] if f["feature"] == "controlled_motion_overall"), None
        )
        if a_cm and n_cm:
            text_lines.append(
                f"  controlled_motion_overall: athletic={a_cm.get('raw')} normal={n_cm.get('raw')}"
            )
        text_lines.append(
            "  Note: knee ROM magnitude is reported separately and does not directly reduce scores."
        )

    contrib_path = out_dir / "stability_feature_contribution.txt"
    json_path = out_dir / "stability_feature_contribution.json"
    contrib_path.write_text("\n".join(text_lines), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n".join(class_lines[:20]))
    print(f"\n... full classification: {class_path}")
    print("\n".join(text_lines))
    print(f"\nWrote {contrib_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
