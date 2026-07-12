"""Export estimated virtual GRF time series to ``virtual_grf.npz``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stablewalk.analysis.estimated_vgrf_analysis import EstimatedVGRFResult

VIRTUAL_GRF_SCHEMA_VERSION = "1.0"


def export_virtual_grf_npz(
    vgrf: EstimatedVGRFResult,
    output_path: Path,
    *,
    run_name: str = "",
) -> Path:
    """
    Export estimated vGRF arrays (not force-plate or PhysX data).

    Output: ``data/output/motion_reference/<run_name>/virtual_grf.npz``
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": VIRTUAL_GRF_SCHEMA_VERSION,
        "run_name": run_name or output_path.parent.name,
        "method_name": vgrf.method_name,
        "terminology": vgrf.terminology,
        "scientific_disclaimer": vgrf.scientific_disclaimer,
        "body_mass_kg": np.float64(vgrf.body_mass_kg),
        "body_weight_n": np.float64(vgrf.body_weight_n),
        "available": np.int8(1 if vgrf.available else 0),
        "frame_count": np.int32(len(vgrf.timestamps)),
        "timestamps": vgrf.timestamps.astype(np.float64),
        "left_vgrf_vertical": vgrf.left_vgrf_vertical.astype(np.float64),
        "right_vgrf_vertical": vgrf.right_vgrf_vertical.astype(np.float64),
        "total_vgrf_vertical": vgrf.total_vgrf_vertical.astype(np.float64),
        "left_vgrf_bw": vgrf.left_vgrf_bw.astype(np.float64),
        "right_vgrf_bw": vgrf.right_vgrf_bw.astype(np.float64),
        "com_accel_z": vgrf.com_accel_z.astype(np.float64),
        "confidence": vgrf.confidence.astype(np.float64),
        "peak_force_n": np.float64(vgrf.metrics.peak_force_n),
        "peak_force_bw": np.float64(vgrf.metrics.peak_force_bw),
        "loading_rate_n_per_s": np.float64(vgrf.metrics.loading_rate_n_per_s),
        "impulse_n_s": np.float64(vgrf.metrics.impulse_n_s),
        "summary_confidence": np.float64(vgrf.metrics.confidence),
    }
    if vgrf.notes:
        payload["notes"] = np.array(vgrf.notes, dtype="U256")

    np.savez_compressed(output_path, **payload)
    return output_path


__all__ = ["VIRTUAL_GRF_SCHEMA_VERSION", "export_virtual_grf_npz"]
