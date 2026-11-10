"""Covariance, correlation, and lagged cross-correlation matrices.

Definitions
-----------
Covariance R_cov(0)
    Sample covariance of demeaned channels at the same time index.
    Symmetric positive-semidefinite (up to numerical noise).

Correlation C(0)
    Pairwise-complete Pearson r on samples finite in both channels.
    σ is taken from that same overlap (not each channel’s own finite set).
    Symmetric, with unit diagonal when a series has at least two samples.

Lagged cross-covariance R(tau)
    R_ij(tau) = cov( x_i(t), x_j(t+tau) ).
    For tau ≠ 0 this matrix is **not** symmetric in general
    (R(tau) = R(-tau)^T). Do not feed it to a symmetric eigensolver.

Normalized lagged cross-correlation C(τ)
    C_ij(τ) = corr(x_i(t), x_j(t+τ)) on the overlapping finite samples.
    C(0) uses the same overlap Pearson as the off-lag entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stablewalk.analysis.spectral.config import SpectralAnalysisConfig
from stablewalk.analysis.spectral.series import SpectralSeriesBundle


@dataclass
class LaggedMatrix:
    tau_samples: int
    tau_s: float
    covariance: np.ndarray
    correlation: np.ndarray
    n_pairs: int
    symmetric: bool

    def to_dict(self) -> dict:
        return {
            "tau_samples": self.tau_samples,
            "tau_s": self.tau_s,
            "covariance": _mat_list(self.covariance),
            "correlation": _mat_list(self.correlation),
            "n_pairs": self.n_pairs,
            "symmetric": self.symmetric,
            "kind": "covariance_and_pearson_at_lag",
        }


@dataclass
class CrossCorrSeries:
    channel_a: str
    channel_b: str
    lags_s: np.ndarray
    rho: np.ndarray
    peak_lag_s: float | None = None
    peak_value: float | None = None

    def to_dict(self) -> dict:
        return {
            "channel_a": self.channel_a,
            "channel_b": self.channel_b,
            "lags_s": [float(x) for x in self.lags_s],
            "rho": [float(x) for x in self.rho],
            "peak_lag_s": self.peak_lag_s,
            "peak_value": self.peak_value,
            "units": {"lag": "s", "rho": "dimensionless"},
        }


@dataclass
class MatrixBundle:
    channel_ids: list[str]
    r0: LaggedMatrix | None
    lagged: list[LaggedMatrix] = field(default_factory=list)
    cross_corr: list[CrossCorrSeries] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "channel_ids": list(self.channel_ids),
            "r0": None if self.r0 is None else self.r0.to_dict(),
            "lagged": [m.to_dict() for m in self.lagged],
            "cross_corr": [c.to_dict() for c in self.cross_corr],
            "notes": list(self.notes),
        }


def _mat_list(mat: np.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in np.asarray(mat)]


_MIN_PEARSON = 4


def pearson_overlap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_paired: int = _MIN_PEARSON,
) -> tuple[float, int]:
    """Pearson r on samples that are finite in both series.

    This is C(0) and the pairwise core of C(τ). σ is taken from the same
    overlapping window, so a non-constant pair has r = 1 with itself.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = min(xa.size, ya.size)
    if n <= 0:
        return float("nan"), 0
    mask = np.isfinite(xa[:n]) & np.isfinite(ya[:n])
    count = int(mask.sum())
    if count < int(min_paired):
        return float("nan"), count
    a = xa[:n][mask]
    b = ya[:n][mask]
    sa = float(np.std(a, ddof=1))
    sb = float(np.std(b, ddof=1))
    if sa <= 1e-18 or sb <= 1e-18:
        return float("nan"), count
    return float(np.corrcoef(a, b)[0, 1]), count


def _pairwise_cov(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 4:
        return float("nan"), n
    xc = x[mask] - np.mean(x[mask])
    yc = y[mask] - np.mean(y[mask])
    return float(np.dot(xc, yc) / (n - 1)), n


def covariance_matrix(data: np.ndarray) -> tuple[np.ndarray, int]:
    """Pairwise-complete sample covariance (C, C)."""
    t, c = data.shape
    cov = np.full((c, c), np.nan, dtype=float)
    n_min = t
    for i in range(c):
        for j in range(i, c):
            val, n = _pairwise_cov(data[:, i], data[:, j])
            cov[i, j] = val
            cov[j, i] = val
            n_min = min(n_min, n)
    return cov, n_min


def correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    corr = np.full_like(cov, np.nan, dtype=float)
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    c = cov.shape[0]
    for i in range(c):
        for j in range(c):
            denom = d[i] * d[j]
            if denom > 1e-18 and np.isfinite(cov[i, j]):
                corr[i, j] = cov[i, j] / denom
    np.fill_diagonal(corr, 1.0)
    return corr


def lagged_cross_cov(x: np.ndarray, y: np.ndarray, tau: int) -> tuple[float, int]:
    """cov(x[t], y[t+tau]) for tau that may be negative."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.size, y.size)
    if abs(int(tau)) >= n:
        return float("nan"), 0
    if tau >= 0:
        a, b = x[: n - tau], y[tau:n]
    else:
        a, b = x[-tau:n], y[: n + tau]
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan"), 0
    return _pairwise_cov(a, b)


def lagged_matrix(data: np.ndarray, tau: int, fps: float) -> LaggedMatrix:
    c = data.shape[1]
    cov = np.full((c, c), np.nan, dtype=float)
    n_min = data.shape[0]
    for i in range(c):
        for j in range(c):
            val, n = lagged_cross_cov(data[:, i], data[:, j], tau)
            cov[i, j] = val
            n_min = min(n_min, n)
    corr = _lagged_corr(data, tau)
    if tau == 0:
        for i in range(c):
            n_ii = int(np.isfinite(data[:, i]).sum()) if data.shape[0] else 0
            if n_ii >= 2:
                corr[i, i] = 1.0
    return LaggedMatrix(
        tau_samples=int(tau),
        tau_s=float(tau) / max(fps, 1e-6),
        covariance=cov,
        correlation=corr,
        n_pairs=int(n_min),
        symmetric=tau == 0,
    )


def _lagged_corr(data: np.ndarray, tau: int) -> np.ndarray:
    """C_ij(τ) = corr(x_i(t), x_j(t+τ)) on the overlapping finite samples."""
    c = data.shape[1]
    out = np.full((c, c), np.nan, dtype=float)
    n = data.shape[0]
    tau = int(tau)
    if abs(tau) >= n:
        return out
    for i in range(c):
        for j in range(c):
            if tau >= 0:
                a, b = data[: n - tau, i], data[tau:n, j]
            else:
                a, b = data[-tau:n, i], data[: n + tau, j]
            r, _count = pearson_overlap(a, b)
            out[i, j] = r
    return out


def lagged_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_lag: int,
    fps: float,
) -> CrossCorrSeries:
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    rho = np.full(lags.shape, np.nan, dtype=float)
    n = min(x.size, y.size)
    for k, tau in enumerate(lags):
        if abs(int(tau)) >= n:
            continue
        if tau >= 0:
            a, b = x[: n - tau], y[tau:n]
        else:
            a, b = x[-tau:n], y[: n + tau]
        rho[k], _n = pearson_overlap(a, b)
    lags_s = lags.astype(float) / max(fps, 1e-6)
    peak_lag = peak_val = None
    finite = np.isfinite(rho)
    if finite.any():
        max_val = float(np.nanmax(rho))
        # Best phase is the lag of the true maximum. Tie-break by |τ| only
        # among exact ties — do not pull a gait half-cycle back to τ = 0.
        tied = np.where(finite & np.isclose(rho, max_val, rtol=0.0, atol=1e-12))[0]
        if tied.size:
            idx = int(tied[np.argmin(np.abs(lags_s[tied]))])
            peak_lag = float(lags_s[idx])
            peak_val = float(rho[idx])
    return CrossCorrSeries(
        channel_a="",
        channel_b="",
        lags_s=lags_s,
        rho=rho,
        peak_lag_s=peak_lag,
        peak_value=peak_val,
    )


def _maybe_zscore(data: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return data
    out = data.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        mask = np.isfinite(col)
        if mask.sum() < 4:
            continue
        mu = float(np.mean(col[mask]))
        sd = float(np.std(col[mask], ddof=1))
        if sd > 1e-18:
            out[mask, j] = (col[mask] - mu) / sd
    return out


def build_matrix_bundle(
    bundle: SpectralSeriesBundle,
    channel_ids: list[str],
    *,
    config: SpectralAnalysisConfig | None = None,
    pair_ids: list[tuple[str, str]] | None = None,
) -> MatrixBundle:
    cfg = config or SpectralAnalysisConfig()
    notes: list[str] = [
        "R(0) covariance is symmetric and suitable for eigh.",
        "R(tau) for tau≠0 is lagged cross-covariance and is not assumed symmetric.",
        "Correlation is Pearson (dimensionless); covariance keeps channel units².",
    ]
    data, kept, _row_ok = bundle.matrix_block(channel_ids, processed=True)
    if not kept:
        return MatrixBundle(channel_ids=[], r0=None, notes=["No usable channels for R(τ)."])
    data = _maybe_zscore(data, cfg.zscore_for_matrices)
    if cfg.zscore_for_matrices:
        notes.append("Channels were z-scored before matrices because units are mixed.")
    r0 = lagged_matrix(data, 0, bundle.fps)
    taus = cfg.r_tau_samples
    if taus is None:
        half = int(round(0.5 * bundle.fps))
        taus = (0, half, -half, int(round(bundle.fps)))
    lagged = []
    for tau in taus:
        if int(tau) == 0:
            continue
        lagged.append(lagged_matrix(data, int(tau), bundle.fps))

    xcorr: list[CrossCorrSeries] = []
    max_lag = int(round(cfg.max_acf_lag_s * bundle.fps))
    pairs = pair_ids or []
    if not pairs:
        # Default: every L/R name pair present in *kept*.
        left = [c for c in kept if ":left_" in c or c.startswith("dof:left_")]
        for lc in left:
            rc = lc.replace("left_", "right_", 1)
            if rc in kept:
                pairs.append((lc, rc))
    for a, b in pairs:
        if a not in kept or b not in kept:
            continue
        ia, ib = kept.index(a), kept.index(b)
        series = lagged_cross_correlation(
            data[:, ia], data[:, ib], max_lag=max_lag, fps=bundle.fps
        )
        series.channel_a = a
        series.channel_b = b
        xcorr.append(series)

    return MatrixBundle(
        channel_ids=kept,
        r0=r0,
        lagged=lagged,
        cross_corr=xcorr,
        notes=notes,
    )
