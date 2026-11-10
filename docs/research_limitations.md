# Unresolved limitations (thesis-facing)

Honest list for Erik. None of these were hidden by inventing numbers.

1. **Monocular depth and scale.** 3D is pelvis-centred, body-normalised,
   heuristic Z. Step length and clearance in laboratory metres need
   calibration. Filename `normal_gait.mp4` does not make the scale physical.

2. **COM vs base of support.** Geometric COM-in-polygon is experimental.
   It is **not reportable** as a session stability score. Do not treat
   Stable% as Hof margin of stability.

3. **Cadence evidence can be thin.** Two HS→HS cycles on a short clip can
   yield a real but cautious cadence (e.g. ~62 spm on a 4.8 s clip). That is
   not “normal adult walking” just because the file is named normal.

4. **Walking-aid / saturated contact.** If most foot-frames stay in contact,
   session cadence/symmetry stay withheld. Cycles are not fabricated.

5. **Ankle angle convention.** Hip/knee flexion is `180° − interior`. Ankle
   remains an interior landmark angle, not a validated dorsi/plantar DOF.

6. **OpenSim Inverse Kinematics** is typically **Partial** (demo Gait2392,
   no subject ScaleTool / complete marker map). Partial is correct. File
   presence of `.mot` does not upgrade the status to Complete.

7. **OpenSim Inverse Dynamics is not executed.** No measured GRF/COP.
   Estimated \(M=I\alpha\) is optional and model-dependent; it is not ID.

8. **Spectral features are not disease markers.** Entropy, centroid, and
   eigenvalue ratios describe the pose **time series**. No clinical mapping.

9. **Visibility is a hard mask**, not a full inverse-variance weighted
   covariance. Low-confidence samples are dropped (`visibility_min`).

10. **Front-on cameras.** Spatial step length is gated; timing may remain
    usable.

11. **GUI frame cap** (240 frames) still truncates very long videos.

12. **Requirement 5** (reinforcement / deep-learning gait simulator) is
    **not implemented**. `docs/erik_requirement_5_plan.md` only.

13. **Live Tk / OpenSim ScaleTool / force-plate sessions** were not executed
    in the 2026-10-13 automated pass.

14. **No clinician-labelled heel-strike set** on `abnormal_gait.mp4` in-repo.
    Event tests use synthetic extrema and contact FSM logic.

15. **Gait Quality composite** still blends movement-domain scores. Overview
    cards keep quality, symmetry, cadence, and stability **separate**;
    stability is N/A when COM–BoS is unreportable. The composite is not a
    diagnosis.
