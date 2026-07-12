"""Tests for Real-to-Sim pipeline modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.io.motion_reference_export import export_motion_reference_npz
from stablewalk.real_to_sim.amp_reference_export import export_amp_reference
from stablewalk.real_to_sim.contact_sync_reward import (
    contact_force_sync_reward,
    summarize_contact_sync,
)
from stablewalk.real_to_sim.gait_style_extraction import extract_gait_style_fingerprint
from stablewalk.real_to_sim.motion_reference_loader import (
    load_motion_reference,
    validate_motion_reference,
)
from stablewalk.real_to_sim.retargeting import load_retarget_config, retarget_motion_reference
from tests.test_virtual_grf import _cycles, _recording


def test_gait_style_fingerprint() -> None:
    rec = _recording(20)
    cycles = _cycles(rec)
    fp = extract_gait_style_fingerprint(rec, cycles)
    assert fp.style_summary
    assert 0.0 <= fp.confidence <= 1.0
    d = fp.to_dict()
    assert "cadence_steps_per_min" in d


def test_motion_reference_with_gait_style(tmp_path: Path) -> None:
    rec = _recording(12)
    cycles = analyze_gait_cycles(rec)
    fp = extract_gait_style_fingerprint(rec, cycles)
    out = tmp_path / "stablewalk_motion.npz"
    export_motion_reference_npz(rec, cycles, out, gait_style=fp)
    ok, issues = validate_motion_reference(out)
    assert ok, issues
    data = load_motion_reference(out)
    assert data.frame_count == 12
    assert data.gait_style_json is not None
    style = json.loads(data.gait_style_json)
    assert "style_summary" in style


def test_retargeting_scale() -> None:
    rec = _recording(10)
    cycles = analyze_gait_cycles(rec)
    out_path = Path("unused")
    # use in-memory via build + save in tmp would be better - use tmp_path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "stablewalk_motion.npz"
        export_motion_reference_npz(rec, cycles, out)
        motion = load_motion_reference(out)
        cfg = load_retarget_config()
        retargeted = retarget_motion_reference(motion, cfg)
        assert retargeted.scale_factor > 0
        assert retargeted.root_positions.shape == motion.root_positions.shape


def test_amp_reference_export(tmp_path: Path) -> None:
    rec = _recording(15)
    cycles = analyze_gait_cycles(rec)
    fp = extract_gait_style_fingerprint(rec, cycles)
    motion_path = tmp_path / "stablewalk_motion.npz"
    export_motion_reference_npz(rec, cycles, motion_path, gait_style=fp)
    motion = load_motion_reference(motion_path)
    retargeted = retarget_motion_reference(motion, load_retarget_config())
    result = export_amp_reference(
        motion,
        retargeted,
        tmp_path,
        run_name="test_run",
        gait_style=fp,
    )
    assert result.npz_path.is_file()
    assert result.json_path.is_file()
    amp = np.load(result.npz_path, allow_pickle=False)
    assert "root_quaternions_wxyz" in amp
    assert len(amp["timestamps"]) == 15


def test_contact_sync_reward() -> None:
    mask = np.array([1, 1, 0, 0, 1], dtype=np.int8)
    force = np.array([50.0, 5.0, 0.0, 80.0, 30.0])
    reward = contact_force_sync_reward(mask, force, force_threshold_n=10.0)
    assert reward[0] == 1.0
    assert reward[1] == 0.0
    assert reward[4] == 1.0

    summary = summarize_contact_sync(mask, mask, force, force)
    assert 0.0 <= summary.mean_reward <= 1.0
    assert summary.interpretation
