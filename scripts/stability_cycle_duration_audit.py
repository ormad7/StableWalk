"""
Audit video duration and gait-cycle evidence vs stability scores.

Reports per-video inventory, domain evidence, and whether low scores
correlate with insufficient cycle data rather than biomechanical instability.

Usage:
  python scripts/stability_cycle_duration_audit.py
  python scripts/stability_cycle_duration_audit.py abnormal_gait normal_gait athletic_walking
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.analysis.stability_config import DEFAULT_STABILITY_CONFIG
from stablewalk.analysis.stability_scoring import _build_context, analyze_biomech_stability
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.enrichment import enrich_pose_sequence

DEFAULT_VIDEOS = ("abnormal_gait", "normal_gait", "athletic_walking")


def _resolve_poses(name: str) -> Path:
    stem = name.replace(".mp4", "")
    path = config.POSES_DIR / f"{stem}_poses.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing poses: {path}")
    return path


def audit_video(name: str) -> dict[str, object]:
    seq = load_pose_sequence(_resolve_poses(name))
    enrich_pose_sequence(seq)
    recording = pose_sequence_to_gait_motion(seq)
    cycles = analyze_gait_cycles(recording)
    result = analyze_biomech_stability(seq, cycles=cycles)
    ctx = _build_context(seq, cycles=cycles, recording=recording)

    print("=" * 78)
    print(f"CYCLE / DURATION AUDIT — {name}")
    print("=" * 78)

    if ctx and ctx.evidence:
        ev = ctx.evidence
        v = ev.video
        c = ev.cycles
        print("\nVideo inventory:")
        print(f"  Duration:           {v.duration_s:.3f} s")
        print(f"  FPS:                  {v.fps:.2f}")
        print(f"  Total frames:         {v.total_frames}")
        print(f"  Valid pose frames:    {v.valid_pose_frames} ({v.valid_pose_frame_pct:.1f}%)")
        print(f"  Left heel strikes:    {c.left_heel_strikes}")
        print(f"  Right heel strikes:   {c.right_heel_strikes}")
        print(f"  Left steps:           {c.left_steps}")
        print(f"  Right steps:          {c.right_steps}")
        print(f"  Complete cycles:      {c.complete_cycles}")
        print(f"  Partial cycles:       {c.partial_cycles}")
        print(f"  Usable cycles:        {c.usable_cycles}")
        print(f"  Repeatability tier:   {ev.repeatability_tier}")

        print("\nDomain evidence:")
        for key, domain_ev in ev.domain_evidence.items():
            print(f"  {key}: {domain_ev.evidence_summary} [{domain_ev.availability_hint}]")

        if ev.warnings:
            print("\nEvidence warnings:")
            for w in ev.warnings:
                print(f"  • {w}")

    print("\nStability result:")
    print(f"  Score:                {result.score:.1f}/100 ({result.classification})")
    print(f"  Confidence:           {result.confidence_badge}")
    print(f"  Completeness:         {result.completeness_pct:.0f}%")
    print(f"  Usable gait cycles:   {result.usable_gait_cycles}")

    print("\nDomain availability:")
    for m in result.metrics:
        score_s = "N/A" if m.score is None else f"{m.score:.0f}"
        print(f"  {m.name:<28} {score_s:>5}  {m.availability:<16} conf={m.confidence:.2f}")

    unavailable = [m.name for m in result.metrics if m.availability == "UNAVAILABLE"]
    low = [m.name for m in result.metrics if m.availability == "LOW_CONFIDENCE"]
    print("\nInterpretation:")
    if unavailable:
        print(f"  Excluded (no fallback score): {', '.join(unavailable)}")
    if low:
        print(f"  Reduced confidence: {', '.join(low)}")
    if result.usable_gait_cycles < DEFAULT_STABILITY_CONFIG.gait_evidence.min_usable_cycles_normal:
        print(
            "  Short clip / few usable cycles — overall score reflects available domains only; "
            "confidence is reduced, not the raw domain metrics artificially lowered."
        )

    return {
        "name": name,
        "score": result.score,
        "confidence": result.confidence_badge,
        "usable_cycles": result.usable_gait_cycles,
        "duration_s": result.video_duration_s,
        "unavailable_domains": unavailable,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit gait cycle evidence vs stability scores")
    parser.add_argument(
        "videos",
        nargs="*",
        default=list(DEFAULT_VIDEOS),
        help="Video stems (default: demo trio)",
    )
    args = parser.parse_args()

    print(DEFAULT_STABILITY_CONFIG.gait_evidence.documentation())
    print()

    summaries: list[dict[str, object]] = []
    missing: list[str] = []
    for name in args.videos:
        try:
            summaries.append(audit_video(name))
        except FileNotFoundError as exc:
            missing.append(str(exc))
            print(f"SKIP {name}: {exc}\n")

    if summaries:
        print("\n" + "=" * 78)
        print("COMPARISON SUMMARY")
        print("=" * 78)
        for s in summaries:
            print(
                f"  {s['name']:<20} score={s['score']:.1f}  "
                f"cycles={s['usable_cycles']}  duration={s['duration_s']:.2f}s  "
                f"confidence={s['confidence']}"
            )
        athletic = next((s for s in summaries if "athletic" in str(s["name"])), None)
        if athletic:
            print("\nAthletic clip attribution:")
            if athletic["usable_cycles"] < 3:
                print(
                    "  Prior athletic scores (~53) were partly driven by insufficient "
                    "cross-cycle evidence: cycle consistency and repeatability domains "
                    "are UNAVAILABLE or LOW_CONFIDENCE, reducing completeness/confidence "
                    "without inventing fallback perfect sub-scores."
                )
            else:
                print(
                    "  Athletic clip has sufficient usable cycles; lower score reflects "
                    "measured domain values, not missing cycle data."
                )

    if missing and not summaries:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
