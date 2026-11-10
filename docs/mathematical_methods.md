# Mathematical methods — spectral analysis, features, and asymmetry

StableWalk derived quantities come from joint **time series** on a uniform
FPS grid extracted from `GaitMotionRecording` (the selected video). Missing
or low-visibility landmarks are **NaN**, never `(0,0,0)`.

These methods are signal processing and kinematic comparison. They are **not**
clinical diagnoses.

---

## 1. Preprocessing

Let \(x[n]\) be a channel (flexion in degrees, or a canonical metre coordinate)
sampled at \(f_s\) (video FPS).

1. Landmark visibility \(v[n] < v_{\min}\) (default 0.50) → sample discarded.
2. Resample onto \(t_n = n / f_s\).
3. Interior gaps shorter than \(T_{\mathrm{gap}}\) (default 0.15 s) are
   linearly interpolated. Edges and long gaps stay NaN.
4. Savitzky–Golay (window ≈ 0.12 s, order 3) on finite runs.
5. Optional linear or mean detrend (default linear) on the processed copy.
   Raw samples are retained for export.

Mean landmark visibility of each channel is stored as `mean_visibility`.

---

## 2. Normalized autocorrelation

After mean removal on the longest finite run,

\[
\rho(k)=\frac{\sum_t x[t]\,x[t+k]}{\sum_t x[t]^2},\qquad k=0,\ldots,K.
\]

Implemented with `numpy.correlate`. Stride lag is searched in 0.60–1.80 s;
step lag in 0.30–0.90 s. Reliability is a **data-quality** score from peak
strength, cycle count, and missingness — not an instability label.

---

## 3. Cross-correlation and lagged matrices

For channels \(i,j\) and lag \(\tau\) (samples),

\[
R_{ij}(\tau)=\mathrm{cov}\bigl(x_i[t],\,x_j[t+\tau]\bigr)
\]

using pairwise-complete observations (\(n\ge 4\)). Pearson form divides by
\(\sigma_i\sigma_j\).

- \(R(0)\) is symmetric. Eigenanalysis uses `numpy.linalg.eigh`.
- \(R(\tau)\) for \(\tau\neq 0\) satisfies \(R(\tau)=R(-\tau)^\top\) and is
  **not** fed to a symmetric eigensolver (`NonsymmetricEigError`).

Left–right peak lag \(\tau^\star\) is the lag of maximum \(\rho_{LR}\) with
smallest \(|\tau|\) among near-maxima. Relative phase (degrees):

\[
\phi = \mathrm{wrap}_{180}\bigl(360^\circ\,\tau^\star / T_{\mathrm{stride}}\bigr).
\]

---

## 4. Spectra

**FFT periodogram** (one-sided, density, Hann):

\[
P(f)=\frac{2}{f_s\sum_n w[n]^2}\,\bigl|\mathrm{rFFT}(w\odot x)\bigr|^2
\]

(DC and Nyquist scaling as in SciPy density convention).

**Welch PSD:** overlapping Hann segments, SciPy `signal.welch` when
installed, else an equivalent NumPy implementation. Walking-band search
default \(B=[0.4,4]\,\mathrm{Hz}\).

Dominant frequency \(f_0=\arg\max_{f\in B} P(f)\). Harmonics use the nearest
bin to \(2f_0\) when it lies in \(B\).

Left–right \(\Delta f = f_{0,R}-f_{0,L}\).

---

## 5. Spectral features (requirement 3)

Let \(p_i = P_i / \sum_{j\in B} P_j\) on positive bins in \(B\).

| Feature | Definition | Units |
|---------|------------|-------|
| Dominant frequency | \(f_0\) | Hz |
| Dominant power | \(P(f_0)\) | units²/Hz |
| Band power | \(\sum_{f\in B} P(f)\,\Delta f\) | units² |
| Spectral entropy | \(H=(-\sum p_i\log p_i)/\log N\) | 1 |
| Centroid | \(\sum f_i p_i\) | Hz |
| Bandwidth | \(\sqrt{\sum p_i(f_i-c)^2}\) | Hz |
| Harmonic ratio | \(P(2f_0)/P(f_0)\) | 1 |
| Stride/harmonic power | band power \(0.4\)–\(1.5\) / \(1.5\)–\(3\) Hz | 1 |
| ACF stride peak / lag | \(\max\rho\) in stride window | 1, s |
| Regularity | \(\rho\) at stride lag | 1 |
| Stride CV | \(\mathrm{std}(T_i)/\mathrm{mean}(T_i)\) of detected cycles | 1 |
| RMS | \(\sqrt{\mathrm{mean}(x^2)}\) | deg or m |
| Leading eigenvalue | largest of \(R(0)\) correlation | 1 |
| L/R \(\Delta f\), \(\Delta H\), \(\Delta\mathrm{RMS}\) | right − left | mixed |
| Xcorr lag / phase | \(\tau^\star\), \(\phi\) | s, deg |

Unavailable inputs → `available=false`, `value=null`. Zeros are not invented.

Session vector keys are frozen in `SESSION_VECTOR_KEYS`.

---

## 6. Three-dimensional reconstruction (requirement 4)

Monocular path (`pose/reconstruction.py`):

- Image \(x\) → skeleton X; image \(y\) (down) → −Y (up).
- Origin at pelvis each frame.
- Z: anatomical depth layers + scaled MediaPipe relative \(z\).
- Height normalized ≈ 1 body length.

Provenance label: `inferred_monocular_3d`. `calibrated_metric` only when the
session is explicitly marked calibrated.

Isolated one-frame angle jumps \(>40^\circ\) are replaced by neighbour mean
(`despike_angle_series`). Sustained ROM (limp) is kept.

Hip/knee charts use anatomical flexion \(180^\circ-\)interior. Ankle stays
interior (foot landmarks are not a clean dorsi/plantar axis).

---

## 7. Gait asymmetry indices

After resampling each side’s cycles to 0–100% gait phase (101 samples):

\[
\mathrm{SI}=200\frac{L-R}{L+R},\qquad
\mathrm{ASI}=200\frac{|L-R|}{|L|+|R|},\qquad
\mathrm{SI}_{norm}=\frac{2\min(L,R)}{L+R}
\]

(SI and ASI in percent; \(\mathrm{SI}_{norm}\) in \([0,1]\) for non-negative
quantities only). Unaligned frames are never subtracted. Different walking
speeds are compared on the **phase** grid, not in raw time.

Timing: heel-strike cycles from `analyze_gait_cycles`. Stance/swing from
contact FSM. Spatial step length in metres is disabled unless calibrated
**and** the camera view supports spatial gait (frontal/unknown views gated).

---

## 8. “Moments” — three different quantities

1. **Kinematic / statistical moments** of 3D trajectories (mean, covariance
   trace). Not torques.
2. **Estimated sagittal torque** \(M=I\alpha\) with Dempster/Winter segment
   inertia, **only if** subject mass (kg) is provided. No GRF. Experimental.
3. **Measured joint moment** — OpenSim Inverse Dynamics. Requires scaled
   model, IK kinematics, and **measured** GRF/COP/`ExternalLoads.xml`.
   Video vGRF is **never** passed to `InverseDynamicsTool`. Typical
   StableWalk sessions: `ready_for_traditional_id = False`.

---

## 9. Reporting domains (kept separate)

| Domain | What it is | What it is not |
|--------------------|----------------|
| Gait movement quality | Composite Gait Quality from pose-derived movement scores | Physical balance |
| Bilateral symmetry | SI / ASI / ROM / timing | Pathology label |
| Gait regularity | ACF peak, stride CV, spectral entropy | Fall-risk score |
| Biomechanical stability | COM–BoS **unreportable** from monocular video | Session KPI |
| Analysis reliability | Pose visibility vs interpretation (cycles/events) | Clinical confidence |

Do not read “High” landmark tracking as high **interpretation** confidence.
