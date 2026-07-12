"""Tests for stablewalk_motion.npz export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.io.motion_reference_export import (
    MOTION_REFERENCE_FILENAME,
    export_motion_reference_npz,
)
from tests.test_virtual_grf import _cycles, _recording


def test_export_motion_reference_npz(tmp_path: Path) -> None:
    rec = _recording(12)
    cycles = _cycles(rec)
    out = tmp_path / "run" / MOTION_REFERENCE_FILENAME
    result = export_motion_reference_npz(rec, cycles, out)

    assert out.is_file()
    assert result.frame_count == 12
    assert result.fps == 30.0

    data = np.load(out, allow_pickle=False)
    assert data["timestamps"].shape == (12,)
    assert data["root_positions"].shape == (12, 3)
    assert data["canonical_joint_positions"].ndim == 3
    assert data["left_contact_mask"].shape == (12,)
    assert data["right_contact_mask"].shape == (12,)
    meta = json.loads(str(data["body_scale_metadata_json"]))
    assert "contact_note" in meta
    assert "not ground reaction forces" in meta["contact_note"].lower()
    assert "joint_ids" in meta
