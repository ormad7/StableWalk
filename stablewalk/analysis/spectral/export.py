"""CSV / JSON export for spectral analysis reports."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stablewalk.analysis.spectral.report import SpectralAnalysisReport


def export_spectral_json(report: SpectralAnalysisReport, path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["exported_at_utc"] = datetime.now(timezone.utc).isoformat()
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def export_spectral_csv(report: SpectralAnalysisReport, path: str | Path) -> Path:
    """Write a multi-section CSV (parameters, series, ACF, PSD, R0, eigenvalues)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[Any]] = []

    def section(title: str, headers: list[str], data: list[list[Any]]) -> None:
        rows.append([f"# {title}"])
        rows.append(headers)
        rows.extend(data)
        rows.append([])

    cfg = report.config.to_dict()
    section(
        "parameters",
        ["key", "value", "units_or_note"],
        [
            ["fps", report.bundle.fps, "Hz"],
            ["coordinate_system", report.bundle.coordinate_system, ""],
            ["univariate_channel_id", report.univariate_channel_id, ""],
            ["visibility_min", cfg.get("visibility_min"), "0-1 landmark score"],
            ["max_gap_s", cfg.get("max_gap_s"), "s"],
            ["smooth_window_s", cfg.get("smooth_window_s"), "s"],
            ["detrend", cfg.get("detrend"), "none|mean|linear"],
            [
                "knee_angle_storage",
                (report.bundle.metadata or {}).get("knee_angle_storage"),
                "pose_interior_hka is converted to anatomical flexion for raw",
            ],
            [
                "knee_raw_signal",
                (report.bundle.metadata or {}).get("knee_raw_signal"),
                "anatomical_knee_flexion_deg; processed is detrended deg; ACF/FFT use z-score",
            ],
            ["gait_band_hz", str(cfg.get("gait_band_hz")), "Hz"],
            ["kind", "derived", "not a clinical diagnosis"],
        ],
    )

    series_rows: list[list[Any]] = []
    for cid, ch in report.bundle.channels.items():
        raw_units = ch.spec.units
        proc_units = f"detrended {ch.spec.units}"
        if "knee" in cid:
            raw_units = "anatomical_knee_flexion_deg"
            proc_units = "detrended_anatomical_knee_flexion_deg"
        for t, raw, proc in zip(ch.times_s, ch.raw, ch.processed):
            series_rows.append(
                [
                    cid,
                    ch.spec.label,
                    raw_units,
                    proc_units,
                    float(t),
                    "" if raw != raw else float(raw),
                    "" if proc != proc else float(proc),
                ]
            )
    section(
        "time_series",
        [
            "channel_id",
            "label",
            "raw_units",
            "processed_units",
            "time_s",
            "raw",
            "processed_detrended",
        ],
        series_rows,
    )

    acf_rows: list[list[Any]] = []
    for cid, ac in report.autocorr.items():
        for lag, rho in zip(ac.lags_s, ac.rho):
            acf_rows.append([cid, float(lag), "" if rho != rho else float(rho)])
    section(
        "autocorrelation",
        ["channel_id", "lag_s", "rho_normalized"],
        acf_rows,
    )

    psd_rows: list[list[Any]] = []
    for cid, sp in report.spectra.items():
        for f, p in zip(sp.freq_hz, sp.psd):
            psd_rows.append([cid, float(f), "" if p != p else float(p), sp.method])
    section(
        "welch_psd",
        ["channel_id", "freq_hz", "psd", "method"],
        psd_rows,
    )

    if report.matrices and report.matrices.r0 is not None:
        ids = report.matrices.channel_ids
        mat = report.matrices.r0.correlation
        header = ["row_channel", *ids]
        data = []
        for i, row_id in enumerate(ids):
            data.append([row_id, *[float(x) if x == x else "" for x in mat[i]]])
        section("r0_correlation", header, data)
        cov = report.matrices.r0.covariance
        data = []
        for i, row_id in enumerate(ids):
            data.append([row_id, *[float(x) if x == x else "" for x in cov[i]]])
        section("r0_covariance", header, data)

        for lag in report.matrices.lagged:
            tau = lag.tau_samples
            data = []
            for i, row_id in enumerate(ids):
                data.append(
                    [row_id, *[float(x) if x == x else "" for x in lag.correlation[i]]]
                )
            section(
                f"r_tau_correlation_tau_{tau}_samples_{lag.tau_s:.3f}s",
                header,
                data,
            )

        xc_rows: list[list[Any]] = []
        for xc in report.matrices.cross_corr:
            xc_rows.append(
                [
                    xc.channel_a,
                    xc.channel_b,
                    "" if xc.peak_lag_s is None else float(xc.peak_lag_s),
                    "" if xc.peak_value is None else float(xc.peak_value),
                ]
            )
            for lag_s, rho in zip(xc.lags_s, xc.rho):
                xc_rows.append(
                    [
                        xc.channel_a,
                        xc.channel_b,
                        float(lag_s),
                        "" if rho != rho else float(rho),
                    ]
                )
        if xc_rows:
            section(
                "left_right_cross_correlation",
                ["channel_a", "channel_b", "lag_s_or_peak_lag", "rho_or_peak"],
                xc_rows,
            )

    eig_rows: list[list[Any]] = []
    for name, eig in report.eigen.items():
        for i, (val, expl) in enumerate(zip(eig.eigenvalues, eig.explained_variance)):
            eig_rows.append([name, i + 1, float(val), float(expl), eig.explanation])
    if report.csd_eigen is not None:
        for i, (val, expl) in enumerate(
            zip(report.csd_eigen.eigenvalues, report.csd_eigen.explained_variance)
        ):
            eig_rows.append(
                ["csd", i + 1, float(val), float(expl), report.csd_eigen.explanation]
            )
    section(
        "eigenvalues",
        ["source", "mode_index", "eigenvalue", "explained_fraction", "explanation"],
        eig_rows,
    )

    feat_rows: list[list[Any]] = []
    if report.features is not None:
        for rec in report.features.records:
            feat_rows.append(
                [
                    rec.name,
                    rec.label,
                    rec.scope,
                    rec.channel_id or "",
                    rec.side or "",
                    rec.cycle_index if rec.cycle_index is not None else "",
                    rec.value if rec.available else "",
                    rec.units,
                    rec.left_value if rec.left_value is not None else "",
                    rec.right_value if rec.right_value is not None else "",
                    rec.quality,
                    "yes" if rec.available else "no",
                    rec.definition,
                    rec.kind,
                ]
            )
    section(
        "spectral_features",
        [
            "name",
            "label",
            "scope",
            "channel_id",
            "side",
            "cycle_index",
            "value",
            "units",
            "left_value",
            "right_value",
            "quality",
            "available",
            "definition",
            "kind",
        ],
        feat_rows,
    )

    vec_rows: list[list[Any]] = []
    if report.features is not None:
        for key, val in report.features.session_vector.items():
            vec_rows.append([key, "" if val is None else float(val)])
    section("session_vector", ["key", "value"], vec_rows)

    warn_rows = [[w] for w in report.warnings]
    section("warnings", ["message"], warn_rows or [["none"]])

    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    return dest
