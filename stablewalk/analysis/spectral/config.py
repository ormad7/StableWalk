"""Configurable parameters for spectral / autocorrelation gait analysis.

Values are signal-processing defaults, not clinical cutoffs. Nothing in this
module diagnoses instability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# Walking fundamentals typically sit well below 4 Hz; keep this band as a
# *search window* for peaks, not a claim that energy outside it is pathology.
DEFAULT_GAIT_BAND_HZ: tuple[float, float] = (0.40, 4.00)
# Slow overground walking can reach ~2.2 s/stride (≈55 steps/min).
# The previous 1.80 s ceiling hid the 4.76 s demo's ACF lobe near 2.0 s.
DEFAULT_STRIDE_LAG_S: tuple[float, float] = (0.55, 2.20)
DEFAULT_STEP_LAG_S: tuple[float, float] = (0.30, 0.90)


@dataclass
class SpectralAnalysisConfig:
    """All user-tunable spectral analysis parameters (units documented)."""

    visibility_min: float = 0.50
    max_gap_s: float = 0.15
    smooth_window_s: float = 0.12
    smooth_polyorder: int = 3
    detrend: str = "linear"  # "none" | "mean" | "linear"
    zscore_for_matrices: bool = True
    gait_band_hz: tuple[float, float] = DEFAULT_GAIT_BAND_HZ
    stride_lag_s: tuple[float, float] = DEFAULT_STRIDE_LAG_S
    step_lag_s: tuple[float, float] = DEFAULT_STEP_LAG_S
    max_acf_lag_s: float = 2.50
    fft_window: str = "hann"
    welch_nperseg: int | None = None
    welch_noverlap: int | None = None
    min_duration_s: float = 2.00
    min_samples: int = 64
    max_missing_fraction: float = 0.45
    min_cycles_for_stride: int = 2
    include_csd: bool = True
    csd_band_hz: tuple[float, float] = DEFAULT_GAIT_BAND_HZ
    # Power-ratio bands (Hz): stride-like vs step/harmonic content.
    stride_band_hz: tuple[float, float] = (0.40, 1.50)
    harmonic_band_hz: tuple[float, float] = (1.50, 3.00)
    n_eigen_modes: int = 3
    min_cycle_samples: int = 16
    # Extra lags (samples) for R(tau). ``None`` → 0 plus ± round(0.5 s * fps).
    r_tau_samples: tuple[int, ...] | None = None
    coordinate_system: str = "sw_canonical_y_up"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gait_band_hz"] = list(self.gait_band_hz)
        data["stride_lag_s"] = list(self.stride_lag_s)
        data["step_lag_s"] = list(self.step_lag_s)
        data["csd_band_hz"] = list(self.csd_band_hz)
        data["stride_band_hz"] = list(self.stride_band_hz)
        data["harmonic_band_hz"] = list(self.harmonic_band_hz)
        if self.r_tau_samples is not None:
            data["r_tau_samples"] = list(self.r_tau_samples)
        return data


def default_config(**overrides: Any) -> SpectralAnalysisConfig:
    cfg = SpectralAnalysisConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Unknown spectral config key: {key}")
        setattr(cfg, key, value)
    return cfg
