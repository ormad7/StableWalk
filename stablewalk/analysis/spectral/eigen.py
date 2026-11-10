"""Eigen-analysis of **symmetric** covariance / correlation / Hermitian CSD.

``numpy.linalg.eigh`` is used only on matrices that are constructed to be
symmetric (or Hermitian). Lagged R(τ) for τ≠0 is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stablewalk.analysis.spectral.config import SpectralAnalysisConfig
from stablewalk.analysis.spectral.matrices import MatrixBundle
from stablewalk.analysis.spectral.series import SpectralSeriesBundle


@dataclass
class EigenDecomposition:
    source: str
    explanation: str
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    explained_variance: np.ndarray
    channel_ids: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "explanation": self.explanation,
            "eigenvalues": [float(x) for x in self.eigenvalues],
            "eigenvectors": [[float(x) for x in row] for row in self.eigenvectors],
            "explained_variance": [float(x) for x in self.explained_variance],
            "channel_ids": list(self.channel_ids),
            "notes": list(self.notes),
        }


class NonsymmetricEigError(ValueError):
    """Raised when a caller asks for eigh on a lagged non-symmetric matrix."""


def eigh_symmetric(
    matrix: np.ndarray,
    *,
    channel_ids: list[str],
    source: str,
    explanation: str,
    atol: float = 1e-8,
) -> EigenDecomposition:
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Eigen-analysis requires a square matrix.")
    if not np.allclose(mat, mat.T, atol=atol, equal_nan=True):
        raise NonsymmetricEigError(
            "Refusing eigh: matrix is not symmetric. For lagged R(τ), τ≠0, "
            "use the matrix itself or SVD; do not call a symmetric eigensolver."
        )
    # Replace NaNs on the diagonal-adjacent structure with 0 for a defined operator.
    work = np.where(np.isfinite(mat), mat, 0.0)
    work = 0.5 * (work + work.T)
    w, v = np.linalg.eigh(work)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    total = float(np.sum(np.abs(w)))
    explained = (w / total) if total > 1e-18 else np.zeros_like(w)
    notes = [
        "Eigenvalues are in descending order.",
        "sum(λ) equals trace(R) up to numerical rounding for a PSD covariance.",
        explanation,
    ]
    return EigenDecomposition(
        source=source,
        explanation=explanation,
        eigenvalues=w.astype(float),
        eigenvectors=v.astype(float),
        explained_variance=explained.astype(float),
        channel_ids=list(channel_ids),
        notes=notes,
    )


def decompose_r0(matrices: MatrixBundle) -> dict[str, EigenDecomposition]:
    out: dict[str, EigenDecomposition] = {}
    if matrices.r0 is None:
        return out
    ids = matrices.channel_ids
    out["covariance"] = eigh_symmetric(
        matrices.r0.covariance,
        channel_ids=ids,
        source="R(0) covariance",
        explanation=(
            "Principal axes of contemporaneous channel covariance. "
            "Leading eigenvectors are linear mixtures of joints that vary together."
        ),
    )
    out["correlation"] = eigh_symmetric(
        matrices.r0.correlation,
        channel_ids=ids,
        source="R(0) correlation",
        explanation=(
            "Same as covariance after per-channel variance normalization. "
            "Useful when mixing degrees and metres."
        ),
    )
    return out


def band_averaged_csd(
    bundle: SpectralSeriesBundle,
    channel_ids: list[str],
    *,
    config: SpectralAnalysisConfig | None = None,
) -> EigenDecomposition | None:
    """Hermitian CSD matrix averaged over the gait band, then eigh.

    S_ij = mean_f [ X_i(f) conj(X_j(f)) ] / T  using rFFT of z-scored series.
    """
    cfg = config or SpectralAnalysisConfig()
    cols = []
    kept = []
    for cid in channel_ids:
        series = bundle.get(cid)
        if series is None:
            continue
        y = series.processed
        if np.isfinite(y).sum() < cfg.min_samples:
            continue
        cols.append(y)
        kept.append(cid)
    if len(kept) < 2:
        return None
    data = np.column_stack(cols)
    # Align on rows that are finite for every channel.
    ok = np.all(np.isfinite(data), axis=1)
    data = data[ok]
    if data.shape[0] < cfg.min_samples:
        return None
    data = data - np.mean(data, axis=0, keepdims=True)
    n = data.shape[0]
    win = np.hanning(n)
    xf = np.fft.rfft(data * win[:, None], axis=0)
    freq = np.fft.rfftfreq(n, d=1.0 / max(bundle.fps, 1e-6))
    lo, hi = cfg.csd_band_hz
    band = (freq >= lo) & (freq <= hi) & (freq > 0)
    if not np.any(band):
        return None
    xb = xf[band]
    # Hermitian CSD: (F, C) -> (C, C)
    s = (xb.conj().T @ xb) / xb.shape[0]
    s = 0.5 * (s + s.conj().T)
    w, v = np.linalg.eigh(s)
    order = np.argsort(w.real)[::-1]
    w = w.real[order]
    v = v[:, order].real
    total = float(np.sum(np.abs(w)))
    explained = w / total if total > 1e-18 else np.zeros_like(w)
    return EigenDecomposition(
        source="band-averaged CSD",
        explanation=(
            "Eigenvectors of the Hermitian cross-spectral density averaged over "
            f"{lo:.1f}–{hi:.1f} Hz. Modes describe frequency-domain co-variation "
            "in the walking band, not time-lag structure."
        ),
        eigenvalues=w.astype(float),
        eigenvectors=v.astype(float),
        explained_variance=explained.astype(float),
        channel_ids=kept,
        notes=[
            "CSD is Hermitian by construction; eigh is appropriate.",
            "This is optional and independent of R(0).",
        ],
    )
