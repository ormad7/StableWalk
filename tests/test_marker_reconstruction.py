"""Unit tests for anatomical OpenSim marker reconstruction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from stablewalk.biomechanics.marker_reconstruction import (
    DEFAULT_RECONSTRUCTION_CONFIG,
    GAIT2392_MARKER_NAMES,
    build_anatomical_frames,
    format_mapping_catalog_table,
    measure_segment_dimensions,
    reconstruct_markers_from_trc_frames,
    reconstruct_markers_single_frame,
    write_mapped_trc_from_reconstruction,
)
from stablewalk.opensim_marker_mapping import (
    create_mapped_trc,
    compare_stablewalk_trc_to_opensim,
    load_stablewalk_to_opensim_mapping,
)


def _sample_landmarks() -> dict[str, tuple[float, float, float]]:
    return {
        "L_HIP": (-100.0, 900.0, 0.0),
        "R_HIP": (100.0, 900.0, 0.0),
        "L_SHOULDER": (-120.0, 1500.0, 0.0),
        "R_SHOULDER": (120.0, 1500.0, 0.0),
        "HEAD": (0.0, 1700.0, 0.0),
        "L_KNEE": (-95.0, 550.0, 10.0),
        "R_KNEE": (95.0, 550.0, 10.0),
        "L_ANKLE": (-90.0, 120.0, 5.0),
        "R_ANKLE": (90.0, 120.0, 5.0),
        "L_HEEL": (-110.0, 80.0, -40.0),
        "R_HEEL": (110.0, 80.0, -40.0),
        "L_TOE": (-70.0, 60.0, 80.0),
        "R_TOE": (70.0, 60.0, 80.0),
    }


def _write_sample_trc(path: Path, n_frames: int = 5) -> None:
    names = list(_sample_landmarks().keys())
    lines = [
        "PathFileType\t4\t(X/Y/Z)\ttest.trc",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"30.0\t30.0\t{n_frames}\t{len(names)}\tmm\t30.0\t1\t{n_frames}",
        "Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t",
        "\t\t" + "\t".join(f"X{i}\tY{i}\tZ{i}" for i in range(1, len(names) + 1)),
        "",
    ]
    for i in range(n_frames):
        t = i / 30.0
        cells = [str(i + 1), f"{t:.6f}"]
        for name in names:
            x, y, z = _sample_landmarks()[name]
            cells.extend([f"{x:.3f}", f"{y + i:.3f}", f"{z:.3f}"])
        lines.append("\t".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_anatomical_frames_build_pelvis_and_thighs():
    lm = _sample_landmarks()
    dims = measure_segment_dimensions(lm)
    frames = build_anatomical_frames(lm, dims)
    assert "pelvis" in frames
    assert "left_thigh" in frames
    assert "right_foot" in frames
    assert dims.hip_width > 0
    assert dims.thigh_length_left > 80.0


def test_single_frame_reconstructs_all_gait2392_markers():
    results = reconstruct_markers_single_frame(_sample_landmarks())
    assert len(results) == len(GAIT2392_MARKER_NAMES)
    present = [n for n, r in results.items() if r.position is not None]
    assert len(present) >= 29
    assert results["R.ASIS"].mapping_type == "DIRECT"
    assert results["V.Sacral"].mapping_type == "DERIVED_ANATOMICAL"
    assert results["R.Thigh.Front"].mapping_type == "DERIVED_ANATOMICAL"


def test_mapping_catalog_has_all_columns():
    rows = format_mapping_catalog_table()
    assert len(rows) == len(GAIT2392_MARKER_NAMES)
    for row in rows:
        assert row["Mapping Type"] in {
            "DIRECT",
            "DERIVED_ANATOMICAL",
            "UNAVAILABLE",
        }
        assert row["OpenSim Marker"] in GAIT2392_MARKER_NAMES


def test_temporal_filtering_reduces_jump():
    lm = _sample_landmarks()
    frames = []
    for i in range(10):
        noisy = dict(lm)
        if i == 5:
            noisy["L_KNEE"] = (5000.0, 5000.0, 5000.0)
        frames.append((i + 1, i / 30.0, noisy))

    cfg = DEFAULT_RECONSTRUCTION_CONFIG.__class__(
        filter_window=3,
        max_jump_mm=80.0,
        max_velocity_mm_s=5000.0,
        max_acceleration_mm_s2=100000.0,
        high_confidence_threshold=0.65,
        ik_readiness_high=72.0,
        ik_readiness_moderate=48.0,
    )
    result = reconstruct_markers_from_trc_frames(frames, fps=30.0, config=cfg)
    l_knee_series = [
        result.frames[i][2]["L.Thigh.Upper"].position
        for i in range(len(result.frames))
    ]
    assert l_knee_series[5] is not None
    assert abs(l_knee_series[5][0]) < 1000


def test_create_mapped_trc_and_comparison(tmp_path: Path):
    source = tmp_path / "walk.trc"
    mapped = tmp_path / "walk_mapped_for_opensim.trc"
    _write_sample_trc(source)

    out_path, direct, derived, details, reconstruction = create_mapped_trc(source, mapped)
    assert out_path.is_file()
    assert len(direct) == 11
    assert len(derived) >= 18
    assert reconstruction.raw_coverage_percent >= 90.0
    assert reconstruction.ik_readiness_score > 40.0
    assert "V.Sacral" in details


def test_derived_markers_use_segment_scaling_not_fixed_offset():
    lm = _sample_landmarks()
    lm_wide = dict(lm)
    lm_wide["L_HIP"] = (-200.0, 900.0, 0.0)
    lm_wide["R_HIP"] = (200.0, 900.0, 0.0)
    sacral_narrow = reconstruct_markers_single_frame(lm)["V.Sacral"].position
    sacral_wide = reconstruct_markers_single_frame(lm_wide)["V.Sacral"].position
    assert sacral_narrow is not None and sacral_wide is not None
    assert abs(sacral_wide[2] - sacral_narrow[2]) > 1.0
