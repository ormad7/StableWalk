"""
Compare stability v2 scores across Abnormal / Normal / Athletic demo videos.

Writes ``stability_v2_comparison_report.txt`` in the project root.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from stablewalk.analysis.stability_config import DEFAULT_STABILITY_CONFIG, LEGACY_V1_DOCUMENTATION
from stablewalk.analysis.stability_scoring import analyze_biomech_stability
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.pose.estimation import PoseEstimator
from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES, demo_path


def analyze_demo(key: str):
    ex = next(e for e in DEMO_GAIT_EXAMPLES if e.key == key)
    path = demo_path(ex)
    if not path.is_file():
        return None, f"Missing video: {path}"
    with PoseEstimator(video_mode=True, require_full_body=False) as est:
        seq = est.process_video(path, enrich_gait=False)
    enrich_pose_sequence(seq)
    result = analyze_biomech_stability(seq)
    return result, None


def main() -> int:
    lines: list[str] = []
    lines.append("STABILITY V2 COMPARISON REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append("LEGACY V1 SUMMARY (superseded)")
    lines.append(LEGACY_V1_DOCUMENTATION.strip())
    lines.append("")
    lines.append("V2 CONFIG WEIGHTS")
    for k, w in DEFAULT_STABILITY_CONFIG.sub_score_weights().items():
        lines.append(f"  {k}: {w:.0%}")
    lines.append("")

    results: dict[str, any] = {}
    for ex in DEMO_GAIT_EXAMPLES:
        lines.append("-" * 72)
        lines.append(f"Demo: {ex.display_name} ({ex.key})")
        result, err = analyze_demo(ex.key)
        if err:
            lines.append(f"  ERROR: {err}")
            continue
        results[ex.key] = result
        lines.append(f"Overall Stability: {result.score:.1f}/100 ({result.classification})")
        lines.append("")
        for m in result.metrics:
            if m.score is None:
                lines.append(f"  {m.name}: N/A")
            else:
                contrib = m.score * m.weight
                lines.append(
                    f"  {m.name}: {m.score:.1f}/100 "
                    f"(weight {m.weight:.0%}, contribution {contrib:.1f})"
                )
        lines.append("")
        lines.append("Why this score?")
        for line in result.explanation.splitlines()[2:6]:
            if line.strip():
                lines.append(f"  {line}")
        lines.append("")
        lines.append("Raw / normalized features (debug):")
        for m in result.metrics:
            for feat in m.values.get("debug_features", []):
                lines.append(
                    f"    [{m.key}] {feat.get('feature')}: "
                    f"raw={feat.get('raw')} norm={feat.get('normalized')} "
                    f"component={feat.get('component_score')}"
                )

    lines.append("")
    lines.append("=" * 72)
    lines.append("SUMMARY TABLE")
    lines.append("")
    header = f"{'Demo':<12} {'Overall':>8} " + " ".join(
        f"{k[:8]:>8}" for k in DEFAULT_STABILITY_CONFIG.sub_score_weights()
    )
    lines.append(header)
    lines.append("-" * len(header))
    for ex in DEMO_GAIT_EXAMPLES:
        r = results.get(ex.key)
        if r is None:
            lines.append(f"{ex.key:<12} {'N/A':>8}")
            continue
        cols = []
        for key in DEFAULT_STABILITY_CONFIG.sub_score_weights():
            m = r.metric(key)
            cols.append(f"{m.score:8.1f}" if m and m.score is not None else f"{'N/A':>8}")
        lines.append(f"{ex.key:<12} {r.score:8.1f} " + " ".join(cols))

    if len(results) >= 2:
        scores = [r.score for r in results.values()]
        spread = max(scores) - min(scores)
        lines.append("")
        lines.append(f"Score spread (max - min): {spread:.1f} points")
        if spread < 8.0:
            lines.append("")
            lines.append("DISCRIMINATION ANALYSIS")
            lines.append(
                "Scores remain similar (< 8 pt spread). Likely limiting factors:"
            )
            lines.append(
                "  - Short demo clips reduce cycle/event reliability for all videos"
            )
            lines.append(
                "  - Monocular 2D pose limits spatial stride and clearance precision"
            )
            lines.append(
                "  - Contact confidence tier affects temporal/contact domains equally"
            )
            lines.append(
                "  - Compare per-domain columns above to see which features "
                "actually differ vs stay flat"
            )
        else:
            lines.append("Scores show meaningful separation across demo gaits.")

    out = ROOT / "stability_v2_comparison_report.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    print(out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
