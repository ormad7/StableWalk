"""Orchestrate spectral / autocorrelation analysis for one gait recording."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from stablewalk.analysis.gait_timing import GaitTimingResult, compose_gait_timing

from stablewalk.analysis.spectral.autocorrelation import AutocorrResult, analyze_autocorrelation
from stablewalk.analysis.spectral.channels import (
    DEFAULT_MATRIX_CHANNEL_IDS,
    DEFAULT_UNIVARIATE_CHANNEL_ID,
    LEFT_RIGHT_PAIRS,
)
from stablewalk.analysis.spectral.config import SpectralAnalysisConfig
from stablewalk.analysis.spectral.eigen import EigenDecomposition, band_averaged_csd, decompose_r0
from stablewalk.analysis.spectral.matrices import MatrixBundle, build_matrix_bundle
from stablewalk.analysis.spectral.series import SpectralSeriesBundle, build_spectral_series
from stablewalk.analysis.spectral.features import SpectralFeatureSet, extract_spectral_features
from stablewalk.analysis.spectral.spectrum import (
    ChannelSpectrum,
    LeftRightSpectrumCompare,
    analyze_spectra,
    compare_left_right_spectra,
)
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.models.pose_data import PoseSequence


@dataclass
class SpectralAnalysisReport:
    """Full derived spectral product for GUI, tests, and export."""

    bundle: SpectralSeriesBundle
    config: SpectralAnalysisConfig
    univariate_channel_id: str
    matrix_channel_ids: list[str]
    autocorr: dict[str, AutocorrResult] = field(default_factory=dict)
    spectra: dict[str, ChannelSpectrum] = field(default_factory=dict)
    lr_compare: list[LeftRightSpectrumCompare] = field(default_factory=list)
    matrices: MatrixBundle | None = None
    eigen: dict[str, EigenDecomposition] = field(default_factory=dict)
    csd_eigen: EigenDecomposition | None = None
    features: SpectralFeatureSet | None = None
    timing: GaitTimingResult | None = None
    warnings: list[str] = field(default_factory=list)
    available: bool = False

    def primary_autocorr(self) -> AutocorrResult | None:
        return self.autocorr.get(self.univariate_channel_id)

    def primary_spectrum(self) -> ChannelSpectrum | None:
        return self.spectra.get(self.univariate_channel_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "stablewalk-spectral-analysis",
            "schema_version": "1.0",
            "kind": "derived",
            "available": self.available,
            "coordinate_system": self.bundle.coordinate_system,
            "fps": self.bundle.fps,
            "sampling_rate_hz": self.bundle.fps,
            "n_samples": int(self.bundle.times_s.size),
            "duration_s": float(self.bundle.metadata.get("duration_s", 0.0)),
            "univariate_channel_id": self.univariate_channel_id,
            "matrix_channel_ids": list(self.matrix_channel_ids),
            "config": self.config.to_dict(),
            "preprocessing": self.config.to_dict(),
            "cycle_definition": (
                "Ipsilateral heel-strike to heel-strike from analyze_gait_cycles "
                "when cycles are supplied; ACF stride lag is a spectral period estimate."
            ),
            "warnings": list(self.warnings),
            "series_metadata": dict(self.bundle.metadata),
            "channels": {
                cid: {
                    "label": ch.spec.label,
                    "units": ch.spec.units,
                    "kind": ch.spec.kind,
                    "missing_fraction": ch.missing_fraction,
                    "interpolated_fraction": ch.interpolated_fraction,
                    "mean_visibility": ch.mean_visibility,
                    "n_valid": ch.n_valid,
                    "times_s": [float(x) for x in ch.times_s],
                    "raw": [None if not _fin(v) else float(v) for v in ch.raw],
                    "processed": [None if not _fin(v) else float(v) for v in ch.processed],
                    "warnings": list(ch.warnings),
                }
                for cid, ch in self.bundle.channels.items()
            },
            "autocorr": {k: v.to_dict() for k, v in self.autocorr.items()},
            "spectra": {k: v.to_dict() for k, v in self.spectra.items()},
            "left_right": [c.to_dict() for c in self.lr_compare],
            "matrices": None if self.matrices is None else self.matrices.to_dict(),
            "eigen": {k: v.to_dict() for k, v in self.eigen.items()},
            "csd_eigen": None if self.csd_eigen is None else self.csd_eigen.to_dict(),
            "features": None if self.features is None else self.features.to_dict(),
            "timing": None if self.timing is None else self.timing.to_dict(),
            "disclaimer": (
                "Derived from monocular pose time series. Not a clinical instability "
                "diagnosis. Insufficient-data warnings must be respected."
            ),
        }


def _fin(v: float) -> bool:
    return v == v and v != float("inf") and v != float("-inf")


def analyze_spectral(
    recording: GaitMotionRecording | None,
    *,
    sequence: PoseSequence | None = None,
    config: SpectralAnalysisConfig | None = None,
    univariate_channel_id: str = DEFAULT_UNIVARIATE_CHANNEL_ID,
    matrix_channel_ids: Iterable[str] | None = None,
    cycles: Any | None = None,
) -> SpectralAnalysisReport:
    cfg = config or SpectralAnalysisConfig()
    matrix_ids = list(matrix_channel_ids or DEFAULT_MATRIX_CHANNEL_IDS)
    warnings: list[str] = []
    if recording is None or recording.frame_count <= 0:
        empty = build_spectral_series(
            GaitMotionRecording(source="", fps=30.0),
            sequence=None,
            config=cfg,
        )
        empty_report = SpectralAnalysisReport(
            bundle=empty,
            config=cfg,
            univariate_channel_id=univariate_channel_id,
            matrix_channel_ids=matrix_ids,
            warnings=["No gait recording is loaded."],
            available=False,
        )
        empty_report.features = extract_spectral_features(empty_report, cycles=cycles, config=cfg)
        empty_report.timing = compose_gait_timing(
            duration_s=0.0,
            fps=float(empty.fps or 0.0),
            cycles=cycles,
            stride_window_s=cfg.stride_lag_s,
            step_window_s=cfg.step_lag_s,
        )
        return empty_report

    bundle = build_spectral_series(
        recording,
        sequence=sequence,
        config=cfg,
        channel_ids=None,
    )
    warnings.extend(bundle.warnings)
    for ch in bundle.channels.values():
        warnings.extend(ch.warnings)

    auto: dict[str, AutocorrResult] = {}
    spec_ids = list(dict.fromkeys([univariate_channel_id, *matrix_ids]))
    for _pid, left_id, right_id in LEFT_RIGHT_PAIRS:
        spec_ids.append(left_id)
        spec_ids.append(right_id)
    spec_ids = list(dict.fromkeys(spec_ids))
    for cid in spec_ids:
        series = bundle.get(cid)
        if series is None:
            continue
        auto[cid] = analyze_autocorrelation(series, config=cfg)
    primary_ac = auto.get(univariate_channel_id)
    if primary_ac is not None and getattr(primary_ac, "insufficient_reason", None):
        warnings.append(str(primary_ac.insufficient_reason))
    spectra = analyze_spectra(bundle, spec_ids, config=cfg)
    lr = compare_left_right_spectra(spectra, LEFT_RIGHT_PAIRS)
    matrices = build_matrix_bundle(bundle, matrix_ids, config=cfg)
    eigen = decompose_r0(matrices)
    csd = band_averaged_csd(bundle, matrix_ids, config=cfg) if cfg.include_csd else None

    available = bool(bundle.channels) and bundle.times_s.size >= 8
    if not available:
        warnings.append("Spectral analysis is unavailable for this clip.")
    result = SpectralAnalysisReport(
        bundle=bundle,
        config=cfg,
        univariate_channel_id=univariate_channel_id,
        matrix_channel_ids=list(matrices.channel_ids or matrix_ids),
        autocorr=auto,
        spectra=spectra,
        lr_compare=lr,
        matrices=matrices,
        eigen=eigen,
        csd_eigen=csd,
        warnings=_unique(warnings),
        available=available,
    )
    result.features = extract_spectral_features(
        result, cycles=cycles, sequence=sequence, config=cfg
    )
    if result.features.warnings:
        result.warnings = _unique([*result.warnings, *result.features.warnings])
    duration_s = float(bundle.metadata.get("duration_s", 0.0) or 0.0)
    result.timing = compose_gait_timing(
        duration_s=duration_s,
        fps=float(bundle.fps),
        channel_acf=result.primary_autocorr(),
        cycles=cycles,
        usable_cycles=getattr(result.features, "n_valid_cycles", None),
        stride_window_s=cfg.stride_lag_s,
        step_window_s=cfg.step_lag_s,
    )
    return result


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
