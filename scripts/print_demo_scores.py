"""Print final Normal vs Athletic demo metrics."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_athletic_video import validate_video
from stablewalk.ui.media.demo_gait import demo_path, example_by_key

for key in ("normal", "athletic"):
    r = validate_video(demo_path(example_by_key(key)), max_frames=120, label=key)
    print(f"{key.upper()}: stability={r['stability_score']:.1f} symmetry={r['symmetry']:.1f} "
          f"step={r['step_regularity']:.1f} body={r['body']:.1f} "
          f"det={r['detection_pct']:.1%} heel={r['heel_visibility']:.3f} conf={r['step_confidence']}")
