"""
Quick demo of the gait motion data layer (no GUI).

Run:
  python scripts/demo_gait_motion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk.adapters.opensim_schema import motion_to_opensim_table
from stablewalk.data.mock_gait import MockGaitConfig, generate_mock_gait


def main() -> None:
    recording = generate_mock_gait(
        MockGaitConfig(fps=30.0, duration_s=2.0, cadence_hz=1.0)
    )
    ts = recording.build_time_series()

    print(f"Source: {recording.source} ({recording.source_kind})")
    print(f"Frames: {recording.frame_count}  Duration: {recording.duration_s:.2f}s")
    print(f"Joints tracked: {len(recording.snapshots[0].joints)}")
    print(f"DOFs tracked: {len(recording.snapshots[0].dofs)}")

    snap = recording.snapshot_at(0)
    if snap:
        knee = snap.get_joint("left_knee")
        if knee:
            print(
                f"Frame 0 left_knee: pos={knee.position.as_tuple()} "
                f"angle={knee.angle_deg:.1f}°"
            )

    traj = ts.trajectory("left_knee")
    print(f"Left knee trajectory points: {len(traj)}")

    opensim = motion_to_opensim_table(recording)
    print(f"OpenSim export columns: {opensim['columns'][:5]}…")

    out = Path("data/output/mock_gait_motion.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recording.to_dict(), indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
