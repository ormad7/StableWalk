# Demonstration guide for the meeting with Erik

**Goal:** show that requirements **2–4** run on **the loaded video**, with
honest limits. Requirement **1** is the existing wire model (do not rebuild).
Requirement **5** is not implemented — say so if asked.

Restart the GUI so you are not looking at an old session.

---

## 1. Wire model (req 1) — 30 seconds

Overview 3D skeleton. That figure is **21 joints + pelvis**, not 13 DOFs.
The 13 sagittal angles are listed in Motion kinematics. If asked “12–20
joints or 12–20 DOF?”: **joints ≠ DOF**; we have 21 named joints because
heel/toe/foot are required for contact.

---

## 2. Spectral characteristics (req 2) — 3 minutes

1. Load a walking clip (demo `normal_gait.mp4` or `abnormal_gait.mp4`).
2. Wait until pose analysis finishes (status / Results).
3. **Motion Analysis → Spectral Analysis.**
4. Select left knee flexion (default). Show:
   - processed signal (playhead follows transport),
   - normalized ACF,
   - Welch PSD in the walking band,
   - L/R PSD,
   - R(0) heatmap,
   - eigenvalues.
5. CSV / JSON export: files contain `fps`, NaNs for missing samples, and
   “not a clinical diagnosis”.

If the clip is too short, the table shows **unavailable**, not zeros.

---

## 3. Spectral features (req 3) — 2 minutes

**Motion Analysis → Spectral Features.**

Point to: dominant frequency, spectral entropy, centroid, bandwidth,
harmonic ratio, ACF peak, L/R phase. Quality column is **data quality**,
not a disease grade.

Same recording as Spectral Analysis. Export includes `SESSION_VECTOR_KEYS`.

---

## 4. 3D reconstruction and asymmetry (req 4) — 3 minutes

Keep the Overview skeleton visible (this **is** the reconstruction).

**Motion Analysis → Asymmetry:**

- L/R hip, knee, ankle ROM with Robinson SI / ASI.
- Stance / swing / step timing when cycles exist.
- Phase-normalized overlay (0–100%), not raw-time subtraction.

Say explicitly:

- Coordinates are **inferred monocular 3D**, not mocap.
- **Joint moments (torques)** from OpenSim ID are **unavailable** (no force
  plates). If mass is entered, optional \(M=I\alpha\) is experimental, not ID.
- Statistical “moments” of trajectories are not torques.

**Advanced → OpenSim:** IK Partial after export is **correct**, not a bug.

---

## 5. What not to claim

- Do not present COM–BoS / “Reduced stability %” as physical balance.
  Overview Stability should be **Not available** (experimental monocular).
- Do not equate High pose tracking with High interpretation confidence.
- Do not say `normal_gait.mp4` must score as clinically normal. A short
  clip can show real but slow cadence from two strides.
- Do not say Inverse Dynamics ran.
- Do not say an RL / Isaac simulator exists.

---

## 6. If Erik asks for numbers

Synthetic tests (executed 2026-10-13): 1 Hz sine → \(f_0\approx 1\,\mathrm{Hz}\)
at 30 and 50 FPS; opposite L/R sines → ~180° phase; symmetric gait SI≈0;
asymmetric knee ROM raises ASI; missing landmarks → `value is None`.

**98** research tests passed in the main suite; **18** interpretation /
Motion-tab tests passed. Screenshots were not fabricated.

Documents to leave with him:

- `docs/erik_requirements_1_4_compliance.md` (matrix)
- `docs/mathematical_methods.md`
- `docs/research_limitations.md`
