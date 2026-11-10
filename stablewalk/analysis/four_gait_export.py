"""CSV export for four-gait signals, matrices, spectra, and coherence."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stablewalk.analysis.four_gait_signals import (
    COMPARE_PAIRS,
    FOCUS_PAIRS,
    FOUR_GAIT_CHANNEL_IDS,
    FOUR_GAIT_SHORT_LABELS,
    FourGaitSignalsReport,
    INTERPRETATION,
    LAG_INTERPRETATION,
    MATRIX_INTERPRETATION,
    SignalMode,
    SPECTRAL_COMPARE_INTERPRETATION,
    SPECTRUM_INTERPRETATION,
    XCORR_PAIRS,
)


def export_four_gait_csv(
    report: FourGaitSignalsReport,
    path: str | Path,
    *,
    mode: SignalMode = "filtered",
) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[Any]] = []

    def section(title: str, headers: list[str], data: list[list[Any]]) -> None:
        rows.append([f"# {title}"])
        rows.append(headers)
        rows.extend(data)
        rows.append([])

    times = np.asarray(report.times_s, dtype=float)
    series = report.series_for(mode)
    section(
        "parameters",
        ["key", "value", "units_or_note"],
        [
            ["fps", report.fps, "Hz"],
            ["n_samples", int(times.size), "full-frame FPS grid"],
            ["mode", mode, "filtered = SG + gap fill"],
            ["coordinate_system", report.coordinate_system, ""],
            ["kind", "derived", "not a clinical diagnosis"],
            ["exported_at_utc", datetime.now(timezone.utc).isoformat(), ""],
        ],
    )
    signal_rows: list[list[Any]] = []
    for i, t in enumerate(times):
        row: list[Any] = [float(t)]
        for cid in FOUR_GAIT_CHANNEL_IDS:
            col = np.asarray(series.get(cid), dtype=float)
            val = col[i] if i < col.size else float("nan")
            row.append(None if not np.isfinite(val) else float(val))
        signal_rows.append(row)
    section(
        "signals_deg",
        ["time_s", *FOUR_GAIT_SHORT_LABELS],
        signal_rows,
    )
    corr = np.asarray(report.correlation_for(mode), dtype=float)
    section(
        "correlation_r0",
        ["i", "j", "r", "n_paired", "note"],
        [
            [
                FOUR_GAIT_SHORT_LABELS[i],
                FOUR_GAIT_SHORT_LABELS[j],
                None if not np.isfinite(corr[i, j]) else float(corr[i, j]),
                int(report.n_paired_for(mode)[i, j]),
                "time-domain Pearson r at lag 0 — not spectral shape r",
            ]
            for i in range(4)
            for j in range(4)
        ],
    )
    spec_corr = np.asarray(report.spectral_corr_for(mode), dtype=float)
    n_bins = np.asarray(report.n_spectral_bins_for(mode), dtype=int)
    section(
        "spectral_shape_correlation",
        ["i", "j", "r", "n_bins", "note"],
        [
            [
                FOUR_GAIT_SHORT_LABELS[i],
                FOUR_GAIT_SHORT_LABELS[j],
                None if not np.isfinite(spec_corr[i, j]) else float(spec_corr[i, j]),
                int(n_bins[i, j]) if n_bins.shape == (4, 4) else 0,
                "unit-energy PSD shape r — not C(lag)",
            ]
            for i in range(min(4, spec_corr.shape[0]))
            for j in range(min(4, spec_corr.shape[1]))
        ],
    )
    spectra = report.spectrum_for(mode)
    powers = report.band_power_for(mode)
    section(
        "frequency_metrics",
        [
            "channel",
            "dominant_hz",
            "spectral_entropy",
            "band_energy",
            "method",
            "quality_status",
            "n_segments",
            "resolution_hz",
            "quality",
        ],
        [
            [
                lab,
                None if spec is None or spec.dominant_hz is None else float(spec.dominant_hz),
                None if spec is None or spec.spectral_entropy is None else float(spec.spectral_entropy),
                None
                if spec is None
                else (
                    float(spec.band_energy)
                    if getattr(spec, "band_energy", None) is not None
                    else (None if powers.get(cid) is None else float(powers[cid]))
                ),
                "" if spec is None else spec.method,
                "" if spec is None else str(getattr(spec, "quality_status", "") or ""),
                0 if spec is None else int(getattr(spec, "n_segments", 0) or 0),
                None
                if spec is None or getattr(spec, "resolution_hz", None) is None
                else float(spec.resolution_hz),
                "" if spec is None else "; ".join(spec.notes),
            ]
            for cid, lab in zip(FOUR_GAIT_CHANNEL_IDS, FOUR_GAIT_SHORT_LABELS)
            for spec in [spectra.get(cid)]
        ],
    )
    compares = report.freq_compare_for(mode)
    section(
        "dominant_frequency_compare",
        ["pair", "f_a_hz", "f_b_hz", "delta_hz", "delta_pct", "note"],
        [
            [
                cmp.label,
                cmp.f_a_hz,
                cmp.f_b_hz,
                cmp.delta_hz,
                cmp.delta_pct,
                " ".join(cmp.notes),
            ]
            for name, _a, _b, _lab in COMPARE_PAIRS
            for cmp in [compares.get(name)]
            if cmp is not None
        ],
    )
    xcorrs = report.xcorr_for(mode)
    r0 = np.asarray(report.correlation_for(mode), dtype=float)
    pair_idx = {name: (i, j) for name, i, j, _lab in FOCUS_PAIRS}

    def _r0(name: str) -> float | None:
        i, j = pair_idx.get(name, (-1, -1))
        if not (0 <= i < r0.shape[0] and 0 <= j < r0.shape[1]):
            return None
        val = float(r0[i, j])
        return None if not np.isfinite(val) else val

    section(
        "xcorr_peaks",
        ["pair", "peak_r", "peak_lag_s", "r0", "note"],
        [
            [
                lab,
                None if xc is None or xc.peak_value is None else float(xc.peak_value),
                None if xc is None or xc.peak_lag_s is None else float(xc.peak_lag_s),
                _r0(name),
                "best phase = argmax ρ(τ); r(0) is simultaneous, not instability",
            ]
            for name, _a, _b, lab in XCORR_PAIRS
            for xc in [xcorrs.get(name)]
        ],
    )
    coherences = report.coherence_for(mode)
    section(
        "coherence",
        ["pair", "sufficient", "n_segments", "mean_band", "peak_band", "peak_hz", "method", "note"],
        [
            [
                label,
                False if coh is None else bool(coh.sufficient),
                0 if coh is None else int(coh.n_segments),
                None if coh is None else coh.mean_band,
                None if coh is None else coh.peak_band,
                None if coh is None else coh.peak_hz,
                "" if coh is None else coh.method,
                "" if coh is None else "; ".join(coh.notes),
            ]
            for name, _a, _b, label in COMPARE_PAIRS
            for coh in [coherences.get(name)]
        ],
    )
    section(
        "interpretations",
        ["topic", "text"],
        [
            ["signals", INTERPRETATION],
            ["correlation_matrix", MATRIX_INTERPRETATION],
            ["autocorrelation", LAG_INTERPRETATION],
            ["spectrum", SPECTRUM_INTERPRETATION],
            ["spectral_compare", SPECTRAL_COMPARE_INTERPRETATION],
            ["warnings", " · ".join(report.warnings[:8])],
        ],
    )
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    return dest
