"""Export biomechanical analysis artifacts (NPZ + JSON)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stablewalk.analysis.biomechanical.orchestrator import BiomechanicalAnalysisResult

BIOMECH_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class BiomechanicalExportResult:
    run_dir: Path
    center_of_mass_path: Path | None
    base_of_support_path: Path | None
    video_quality_path: Path | None
    biomechanical_report_path: Path | None


def export_center_of_mass_npz(result: BiomechanicalAnalysisResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    com = result.center_of_mass
    if com is None or not com.per_frame:
        np.savez_compressed(path, frame_count=np.int32(0))
        return path
    pos = com.positions
    payload = {
        "schema_version": BIOMECH_SCHEMA_VERSION,
        "kind": "estimated",
        "frame_count": np.int32(len(com.per_frame)),
        "fps": np.float64(com.fps),
        "timestamps": com.timestamps,
        "com_x": pos[:, 0],
        "com_y": pos[:, 1],
        "com_z": pos[:, 2],
        "com_vx": np.array([f.velocity[0] for f in com.per_frame]),
        "com_vy": np.array([f.velocity[1] for f in com.per_frame]),
        "com_vz": np.array([f.velocity[2] for f in com.per_frame]),
        "com_ax": np.array([f.acceleration[0] for f in com.per_frame]),
        "com_ay": np.array([f.acceleration[1] for f in com.per_frame]),
        "com_az": np.array([f.acceleration[2] for f in com.per_frame]),
        "confidence": np.array([f.confidence for f in com.per_frame]),
        "frame_indices": np.array([f.frame_index for f in com.per_frame], dtype=np.int32),
    }
    np.savez_compressed(path, **payload)
    return path


def export_base_of_support_npz(result: BiomechanicalAnalysisResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bos = result.base_of_support
    if bos is None or not bos.per_frame:
        np.savez_compressed(path, frame_count=np.int32(0))
        return path

    n = len(bos.per_frame)
    max_verts = max(len(f.polygon_xy) for f in bos.per_frame) if bos.per_frame else 0
    polygons = np.full((n, max(max_verts, 1), 2), np.nan, dtype=np.float64)
    vert_counts = np.zeros(n, dtype=np.int32)
    for i, f in enumerate(bos.per_frame):
        vert_counts[i] = len(f.polygon_xy)
        for j, (x, z) in enumerate(f.polygon_xy):
            polygons[i, j, 0] = x
            polygons[i, j, 1] = z

    payload = {
        "schema_version": BIOMECH_SCHEMA_VERSION,
        "kind": "estimated",
        "frame_count": np.int32(n),
        "fps": np.float64(bos.fps),
        "timestamps": np.array([f.time_s for f in bos.per_frame]),
        "frame_indices": np.array([f.frame_index for f in bos.per_frame], dtype=np.int32),
        "support_type": np.array([f.support_type for f in bos.per_frame], dtype="U20"),
        "polygon_vertices": polygons,
        "vertex_counts": vert_counts,
        "centroid_x": np.array([f.centroid[0] for f in bos.per_frame]),
        "centroid_z": np.array([f.centroid[1] for f in bos.per_frame]),
        "area_m2": np.array([f.area_m2 for f in bos.per_frame]),
        "confidence": np.array([f.confidence for f in bos.per_frame]),
    }
    np.savez_compressed(path, **payload)
    return path


def export_stability_margin_arrays(result: BiomechanicalAnalysisResult, path: Path) -> Path:
    """Include stability margin in base_of_support or separate - embed in COM npz extras."""
    path = Path(path)
    sm = result.stability_margin
    if sm is None or not sm.per_frame:
        return path
    extra = {
        "stability_margin_m": np.array(
            [f.stability_margin_m if f.stability_margin_m is not None else np.nan for f in sm.per_frame]
        ),
        "stability_state": np.array([f.stability_state for f in sm.per_frame], dtype="U24"),
        "stability_confidence": np.array([f.confidence for f in sm.per_frame]),
    }
    # Append to existing com file if present
    if path.is_file():
        existing = dict(np.load(path))
        existing.update(extra)
        np.savez_compressed(path, **existing)
    return path


def export_video_quality_json(result: BiomechanicalAnalysisResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vq = result.video_quality
    payload = vq.to_dict() if vq else {"overall_quality_score": 0.0, "warnings": ["No sequence"]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_biomechanical_report_json(result: BiomechanicalAnalysisResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": BIOMECH_SCHEMA_VERSION,
        "terminology": {
            "measured": "Instrumented or direct sensor data (not available from monocular video alone).",
            "estimated": "Pose-derived proxy with anthropometric or heuristic models.",
            "derived": "Computed from estimated inputs (e.g., stability margin, symmetry index).",
        },
        "summary": result.to_dict(),
        "gait_quality": None if result.gait_quality is None else result.gait_quality.to_dict(),
        "abnormalities": list(result.abnormalities),
        "warnings": list(result.warnings),
        "interpretation": (
            None if result.gait_quality is None else result.gait_quality.explanation
        ),
    }
    if result.stability_margin and result.stability_margin.per_frame:
        report["stability_margin_summary"] = result.stability_margin.to_dict()
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def export_biomechanical_artifacts(
    result: BiomechanicalAnalysisResult,
    run_dir: Path,
    *,
    run_name: str = "",
) -> BiomechanicalExportResult:
    """Write all biomechanical exports to a run directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    com_path = run_dir / "center_of_mass.npz"
    bos_path = run_dir / "base_of_support.npz"
    vq_path = run_dir / "video_quality.json"
    report_path = run_dir / "biomechanical_report.json"

    export_center_of_mass_npz(result, com_path)
    export_stability_margin_arrays(result, com_path)
    export_base_of_support_npz(result, bos_path)
    export_video_quality_json(result, vq_path)
    export_biomechanical_report_json(result, report_path)

    return BiomechanicalExportResult(
        run_dir=run_dir,
        center_of_mass_path=com_path,
        base_of_support_path=bos_path,
        video_quality_path=vq_path,
        biomechanical_report_path=report_path,
    )


__all__ = [
    "BIOMECH_SCHEMA_VERSION",
    "BiomechanicalExportResult",
    "export_biomechanical_artifacts",
    "export_biomechanical_report_json",
    "export_center_of_mass_npz",
    "export_base_of_support_npz",
    "export_video_quality_json",
]
