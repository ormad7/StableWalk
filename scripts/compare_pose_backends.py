#!/usr/bin/env python3
"""
Compare pose / HMR backends on the same walking video.

Usage::

    python scripts/compare_pose_backends.py --video path/to/walk.mp4
    python scripts/compare_pose_backends.py --video path/to/walk.mp4 --max-frames 120

Runs every registered backend that is available (or reports unavailability).
Does **not** declare a winner — outputs structured metrics for research review.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.pose.backends.base import BackendUnavailableError
from stablewalk.pose.backends.comparison import BackendComparisonMetrics, compute_backend_metrics
from stablewalk.pose.backends.environment import HMR_COMPATIBILITY_NOTES, inspect_runtime_environment
from stablewalk.pose.backends.registry import BACKEND_REGISTRY, create_pose_backend, unavailable_message


def _run_backend(
    name: str,
    video: Path,
    *,
    max_frames: int | None,
    allow_fallback: bool,
) -> tuple[BackendComparisonMetrics, object | None]:
    cls = BACKEND_REGISTRY[name]
    if not cls.is_available() and name != "mediapipe":
        return (
            compute_backend_metrics(
                None,
                backend_name=name,
                available=False,
                availability_message=unavailable_message(name),
            ),
            None,
        )

    try:
        backend = create_pose_backend(name, allow_fallback=allow_fallback)
    except BackendUnavailableError as exc:
        return (
            compute_backend_metrics(
                None,
                backend_name=name,
                available=False,
                availability_message=str(exc),
            ),
            None,
        )

    try:
        sequence = backend.process_video(str(video), max_frames=max_frames)
        metrics = compute_backend_metrics(sequence, backend_name=name, available=True)
        if getattr(backend, "close", None):
            backend.close()
        return metrics, sequence
    except BackendUnavailableError as exc:
        return (
            compute_backend_metrics(
                None,
                backend_name=name,
                available=False,
                availability_message=str(exc),
            ),
            None,
        )
    except Exception as exc:
        return (
            BackendComparisonMetrics(
                backend_name=name,
                available=False,
                availability_message=f"Runtime error: {exc}",
                notes=[str(exc)],
            ),
            None,
        )


def format_report(
    video: Path,
    metrics: list[BackendComparisonMetrics],
    env_report: dict,
) -> str:
    lines = [
        "StableWalk Pose Backend Comparison Report",
        "=" * 50,
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Video: {video.resolve()}",
        f"Configured POSE_BACKEND: {config.POSE_BACKEND}",
        f"Allow fallback: {config.POSE_BACKEND_ALLOW_FALLBACK}",
        "",
        "Runtime environment",
        "-" * 30,
        f"Python: {env_report['python_version']}",
        f"Platform: {env_report['platform']}",
        f"MediaPipe: {env_report.get('mediapipe_version') or 'not installed'}",
        f"PyTorch: {env_report.get('torch_version') or 'not installed'}",
        f"CUDA available: {env_report.get('cuda_available')}",
        "",
    ]
    for w in env_report.get("warnings", []):
        lines.append(f"  Warning: {w}")
    lines.extend([
        "",
        "Recommended environments:",
        "  • stablewalk-opensim — main app (MediaPipe + OpenSim)",
        "  • stablewalk-hmr — optional ROMP / HybrIK / WHAM research stack",
        "",
        "Backend results (informational — no automatic winner)",
        "-" * 30,
    ])

    for m in metrics:
        lines.append(f"\n[{m.backend_name}] available={m.available}")
        if m.availability_message:
            lines.append(f"  Status: {m.availability_message}")
        if not m.available:
            note = HMR_COMPATIBILITY_NOTES.get(m.backend_name, "")
            if note:
                lines.append(f"  Note: {note}")
            continue
        d = m.to_dict()
        lines.append(f"  Valid frames: {d['valid_frame_ratio_percent']}% ({m.frame_count} total)")
        lines.append(f"  Mean confidence: {d['mean_landmark_confidence']}")
        lines.append(f"  Trajectory smoothness: {d['trajectory_smoothness_score']}/100")
        lines.append(f"  Joint angle consistency: {d['joint_angle_consistency_score']}/100")
        lines.append(f"  Foot-ground jitter (pseudo-mm): {d['foot_ground_jitter_mm']}")
        lines.append(f"  Pelvis trajectory consistency: {d['pelvis_trajectory_consistency']}")
        lines.append(f"  OpenSim marker confidence (mean): {d['opensim_marker_confidence_mean']}")
        lines.append(f"  OpenSim IK readiness score: {d['opensim_ik_readiness_score']}")

    lines.extend([
        "",
        "Interpretation",
        "-" * 30,
        "Higher smoothness/consistency scores suggest less frame-to-frame noise.",
        "OpenSim marker confidence reflects anatomical reconstruction quality, not clinical accuracy.",
        "Optional HMR backends require a separate stablewalk-hmr environment — see docs/POSE_BACKENDS.md.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare StableWalk pose/HMR backends")
    parser.add_argument("--video", required=True, help="Path to walking video")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")
    parser.add_argument(
        "--backends",
        nargs="*",
        default=list(BACKEND_REGISTRY.keys()),
        help="Backends to compare (default: all)",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Do not fall back to MediaPipe when optional backends fail",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write report to this path (default: data/output/reports/)",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON metrics output")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.is_file():
        print(f"Video not found: {video}", file=sys.stderr)
        return 1

    env = inspect_runtime_environment()
    allow_fallback = not args.no_fallback

    all_metrics: list[BackendComparisonMetrics] = []
    for name in args.backends:
        key = name.strip().lower()
        if key not in BACKEND_REGISTRY:
            print(f"Skipping unknown backend: {name}", file=sys.stderr)
            continue
        print(f"Running backend: {key}...")
        metrics, _seq = _run_backend(
            key,
            video,
            max_frames=args.max_frames,
            allow_fallback=allow_fallback,
        )
        all_metrics.append(metrics)

    report = format_report(video, all_metrics, env.to_dict())
    print(report)

    out = args.output
    if out is None:
        config.ensure_output_dirs()
        out = config.REPORTS_DIR / f"pose_backend_comparison_{video.stem}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Report written: {out.resolve()}")

    if args.json:
        payload = {
            "video": str(video.resolve()),
            "environment": env.to_dict(),
            "backends": [m.to_dict() for m in all_metrics],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"JSON written: {args.json.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
