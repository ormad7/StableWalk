"""Download the University of Utah Neuropathic Gait (gait_ab_10) abnormal demo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk.ui.media.utah_abnormal import prepare_abnormal_demo


def main() -> int:
    ok, meta = prepare_abnormal_demo()
    if not ok:
        print("Failed to download/prepare Utah abnormal gait demo.")
        return 1
    print(json.dumps(meta, indent=2))
    print(f"\nSaved -> {meta.get('local_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
