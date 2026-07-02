#!/usr/bin/env python3
"""Launch robotic walking simulation from saved pose data."""

from stablewalk.robot_simulation_viz import launch_robot_simulation
from stablewalk import config
from pathlib import Path
import sys


def main() -> int:
    poses_dir = config.POSES_DIR
    default = poses_dir / "walk_stream_poses.json"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if not path.is_file():
        print(f"Pose file not found: {path}")
        print("Run: python main.py --url")
        return 1
    launch_robot_simulation(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
