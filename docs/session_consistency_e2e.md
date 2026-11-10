# Session consistency end-to-end verification

Generated: 2026-10-21T14:47:59.926334+00:00

This report records checks that were actually executed on cached pose JSON.
A filename is not a diagnosis. A UI label is not proof of a consistent signal.

## Recordings

### recording_a

- Source: `normal_gait.mp4`
- Video exists: True · fps=25.0 · frames=120 · duration_s=4.8
- Pose frames: 120 · recording frames: 120
- Coordinate system: `sw_canonical_y_up` · knee storage: `pose_interior_hka`
- Cycles: {'gait_cycle_count': 2, 'left_heel_strikes': 3, 'right_heel_strikes': 2, 'metrics_reliable': True, 'reliability_reason': 'Computed from 2 complete same-side heel-strike cycles.', 'cadence_steps_per_min': 64.51612903225806, 'asymmetry_left_cycles': 2, 'asymmetry_right_cycles': 1, 'alignment_event': 'ipsilateral_heel_strike', 'spectral_detected_cycles': 2, 'spectral_valid_cycles': 2}
- Knee comparison: {'n_left_finite': 120, 'n_right_finite': 120, 'left_median': 7.722202752292233, 'right_median': 8.190652150473838, 'vs_spectral_max_abs': 0.0, 'vs_spectral_n_matched': 120, 'vs_asymmetry_max_abs': 3.1988915003679246, 'vs_asymmetry_n_matched': 120, 'spectral_n_grid': 120, 'recording_n_snapshots': 120}
- Filename diagnosis: None

### recording_b

- Source: `athletic_walking.mp4`
- Video exists: True · fps=23.976 · frames=120 · duration_s=5.005005005005005
- Pose frames: 120 · recording frames: 27
- Coordinate system: `sw_canonical_y_up` · knee storage: `pose_interior_hka`
- Cycles: {'gait_cycle_count': 1, 'left_heel_strikes': 2, 'right_heel_strikes': 1, 'metrics_reliable': False, 'reliability_reason': 'Only 1 complete same-side heel-strike cycle(s); at least 2 are required for cadence.', 'cadence_steps_per_min': None, 'asymmetry_left_cycles': 1, 'asymmetry_right_cycles': 0, 'alignment_event': 'ipsilateral_heel_strike', 'spectral_detected_cycles': 1, 'spectral_valid_cycles': 1}
- Knee comparison: {'n_left_finite': 27, 'n_right_finite': 27, 'left_median': 24.378920063740196, 'right_median': 17.568821247110918, 'vs_spectral_max_abs': 0.0, 'vs_spectral_n_matched': 27, 'vs_asymmetry_max_abs': 17.100286112978807, 'vs_asymmetry_n_matched': 27, 'spectral_n_grid': 106, 'recording_n_snapshots': 27}
- Filename diagnosis: None

### recording_c

- Source: `abnormal_gait.mp4`
- Video exists: True · fps=29.97 · frames=150 · duration_s=5.005005005005005
- Pose frames: 120 · recording frames: 120
- Coordinate system: `sw_canonical_y_up` · knee storage: `pose_interior_hka`
- Cycles: {'gait_cycle_count': 2, 'left_heel_strikes': 3, 'right_heel_strikes': 2, 'metrics_reliable': True, 'reliability_reason': 'Computed from 2 complete same-side heel-strike cycles.', 'cadence_steps_per_min': 51.871153846153845, 'asymmetry_left_cycles': 2, 'asymmetry_right_cycles': 1, 'alignment_event': 'ipsilateral_heel_strike', 'spectral_detected_cycles': 2, 'spectral_valid_cycles': 2}
- Knee comparison: {'n_left_finite': 120, 'n_right_finite': 120, 'left_median': 18.57836815319928, 'right_median': 13.978478090726867, 'vs_spectral_max_abs': 0.0, 'vs_spectral_n_matched': 120, 'vs_asymmetry_max_abs': 2.903963920253041, 'vs_asymmetry_n_matched': 120, 'spectral_n_grid': 120, 'recording_n_snapshots': 120}
- Filename diagnosis: None

## OpenSim: Pipeline complete versus IK

This harness did not run Inverse Kinematics. Session IK with ik_state=None is 'Not Started' (IK not executed). Leftover named-folder files resolve to pipeline 'Complete' and IK 'Partial' (Complete · IK Partial). app.py messagebox 'Pipeline complete' is Real-to-Sim after run_real_to_sim_pipeline; it is not IK Complete. OpenSim pipeline Complete means TRC and MOT were exported. IK Complete is allowed only when ik_state == 'Completed'. Leftover .mot/.sto files are artifacts, not a successful IK run in this session.

- Named leftover dirs: `[
  {
    "path": "C:\\Users\\ormad\\Downloads\\Stable__Walk\\StableWalk\\data\\output\\opensim\\normal_gait",
    "exists": true,
    "files": {
      "TRC": true,
      "MAPPED": true,
      "MOT": true,
      "IK_MOT": true,
      "IK_ERRORS": true,
      "JSON": true
    },
    "opensim_sdk_available": false,
    "ik_state_in_json": null,
    "source_video": "C:\\Users\\ormad\\Downloads\\Stable__Walk\\StableWalk\\data\\demo_videos\\normal_gait.mp4",
    "ik_marker_errors": {
      "path": "C:\\Users\\ormad\\Downloads\\Stable__Walk\\StableWalk\\data\\output\\opensim\\normal_gait\\normal_gait_mapped_for_opensim_ik_marker_errors.sto",
      "n_frames": 120,
      "mean_rms_m": 0.14804739525,
      "max_rms_m": 0.16120446,
      "mean_max_error_m": 0.29926912791666666,
      "peak_max_error_m": 0.33156488,
      "quality": "limited",
      "warnings": [
        "IK marker residuals are elevated (mean RMS 14.8 cm); treat IK kinematics as experimental."
      ],
      "fail_threshold_m": 0.15,
      "n_failed_frames": 60,
      "failed_frames": [
        {
          "frame_index": 0,
          "time_s": 0.0,
          "marker_error_rms_m": 0.15948621,
          "marker_error_max_m": 0.28892357,
          "reason": "rms_exceeds_0.15m"
        },
        {
          "frame_index": 1,
          "time_s": 0.04,
          "marker_error_rms_m": 0.15997474,
          "marker_error_max_m": 0.29245077,
          "reason": "rms_exceeds_0.15m"
        },
        {
          "frame_index": 2,
          "time_s": 0.08,
          "marker_error_rms_m": 0.16013988,
          "marker_error_max_m": 0.29373549,
          "reason": "rms_exceeds_0.15m"
        },
        {
          "frame_index": 3,
          "time_s": 0.12,
          "marker_error_rms_m": 0.16030673,
          "marker_error_max_m": 0.29675587,
          "reason": "rms_exceeds_0.15m"
        },
        {
          "frame_index": 4,
          "time_s": 0.16,
          "marker_error_rms_m": 0.16055872,
          "marker_er`
- Newest leftover run dirs: `['C:\\Users\\ormad\\Downloads\\Stable__Walk\\StableWalk\\data\\output\\opensim\\run_787305e91e7251c7_1792586832446', 'C:\\Users\\ormad\\Downloads\\Stable__Walk\\StableWalk\\data\\output\\opensim\\run_787305e91e7251c7_1792326338520', 'C:\\Users\\ormad\\Downloads\\Stable__Walk\\StableWalk\\data\\output\\opensim\\run_787305e91e7251c7_1791978425422']`

### Leftover IK fitting errors (on-disk, not this session)

- `C:\Users\ormad\Downloads\Stable__Walk\StableWalk\data\output\opensim\run_787305e91e7251c7_1792586832446\run_787305e91e7251c7_1792586832446_mapped_for_opensim_ik_marker_errors.sto`: n_frames=120 mean_rms_m=0.14738423883333335 n_failed=48 quality=limited
  - frame 0 t=0.0s RMS=0.16005163 max=0.28963236 (rms_exceeds_0.15m)
  - frame 1 t=0.04s RMS=0.16012821 max=0.29098537 (rms_exceeds_0.15m)
  - frame 2 t=0.08s RMS=0.16029248 max=0.29225933 (rms_exceeds_0.15m)
  - frame 3 t=0.12s RMS=0.16037235 max=0.29449415 (rms_exceeds_0.15m)
  - frame 4 t=0.16s RMS=0.16043892 max=0.29568764 (rms_exceeds_0.15m)
  - frame 5 t=0.2s RMS=0.16167178 max=0.29187088 (rms_exceeds_0.15m)
  - frame 6 t=0.24s RMS=0.1622691 max=0.29052348 (rms_exceeds_0.15m)
  - frame 7 t=0.28s RMS=0.16273105 max=0.29071448 (rms_exceeds_0.15m)
- `C:\Users\ormad\Downloads\Stable__Walk\StableWalk\data\output\opensim\run_787305e91e7251c7_1792326338520\run_787305e91e7251c7_1792326338520_mapped_for_opensim_ik_marker_errors.sto`: n_frames=120 mean_rms_m=0.14738423883333335 n_failed=48 quality=limited
  - frame 0 t=0.0s RMS=0.16005163 max=0.28963236 (rms_exceeds_0.15m)
  - frame 1 t=0.04s RMS=0.16012821 max=0.29098537 (rms_exceeds_0.15m)
  - frame 2 t=0.08s RMS=0.16029248 max=0.29225933 (rms_exceeds_0.15m)
  - frame 3 t=0.12s RMS=0.16037235 max=0.29449415 (rms_exceeds_0.15m)
  - frame 4 t=0.16s RMS=0.16043892 max=0.29568764 (rms_exceeds_0.15m)
  - frame 5 t=0.2s RMS=0.16167178 max=0.29187088 (rms_exceeds_0.15m)
  - frame 6 t=0.24s RMS=0.1622691 max=0.29052348 (rms_exceeds_0.15m)
  - frame 7 t=0.28s RMS=0.16273105 max=0.29071448 (rms_exceeds_0.15m)
- `C:\Users\ormad\Downloads\Stable__Walk\StableWalk\data\output\opensim\run_787305e91e7251c7_1791978425422\run_787305e91e7251c7_1791978425422_mapped_for_opensim_ik_marker_errors.sto`: n_frames=120 mean_rms_m=0.14738423883333335 n_failed=48 quality=limited
  - frame 0 t=0.0s RMS=0.16005163 max=0.28963236 (rms_exceeds_0.15m)
  - frame 1 t=0.04s RMS=0.16012821 max=0.29098537 (rms_exceeds_0.15m)
  - frame 2 t=0.08s RMS=0.16029248 max=0.29225933 (rms_exceeds_0.15m)
  - frame 3 t=0.12s RMS=0.16037235 max=0.29449415 (rms_exceeds_0.15m)
  - frame 4 t=0.16s RMS=0.16043892 max=0.29568764 (rms_exceeds_0.15m)
  - frame 5 t=0.2s RMS=0.16167178 max=0.29187088 (rms_exceeds_0.15m)
  - frame 6 t=0.24s RMS=0.1622691 max=0.29052348 (rms_exceeds_0.15m)
  - frame 7 t=0.28s RMS=0.16273105 max=0.29071448 (rms_exceeds_0.15m)

## Checks performed

### recording_a

- **PASS** `video_present` — normal_gait.mp4 exists=True fps=25.0 frames=120
- **PASS** `poses_present` — normal_gait_poses.json exists=True
- **PASS** `knee_storage_pose_interior` — recording_knee_storage=pose_interior_hka (pose JSON should be pose_interior_hka)
- **PASS** `knee_flexion_matches_spectral_raw` — time-aligned max |analysis_knee_flexion − spectral raw| = 0.0 (120 matched samples, tol=0.0204s)
- **PASS** `knee_flexion_asymmetry_same_convention` — time-aligned max |analysis_knee_flexion − smoothed kinematics| = 3.1988915003679246 (120 samples). Asymmetry kinematics is gap-filled + 0.12 s smoothed; this is not the raw spectral series.
- **PASS** `spectral_grid_spans_snapshots` — spectral n_grid=120 vs recording n_snapshots=120; grid [0.0, 4.76] vs snapshots [0.0, 4.76]
- **PASS** `timestamps_match_asymmetry` — max |recording time − asymmetry time| = 0.0
- **PASS** `sampling_rate_shared` — recording=25 Hz spectral=25 Hz asymmetry=25 Hz video=25 Hz
- **PASS** `recording_fps_matches_video` — recording fps 25 vs video container fps 25
- **PASS** `coordinate_system_shared` — recording='sw_canonical_y_up' spectral='sw_canonical_y_up' asymmetry='sw_canonical_y_up'
- **PASS** `cycle_definition_ipsilateral_hs` — alignment_event='ipsilateral_heel_strike'; gait_cycle_count=2 HS L/R=3/2; asymmetry cycles L/R=2/1; spectral detected/valid=2/2
- **PASS** `cycle_counts_consistent` — Asymmetry ipsilateral cycles are HS→HS subsets of detected heel strikes (L 2/3, R 1/2).
- **PASS** `knee_units_degrees` — spectral left-knee units='deg'; processed mean=-9.547918011776346e-16 (detrend ~0)
- **PASS** `spectral_processed_is_detrended` — processed (detrended flexion) mean=-9.548e-16 deg about 0
- **PASS** `spectral_json_provenance` — fps/config/coordinate/cycle keys present
- **PASS** `asymmetry_json_provenance` — fps/coordinate/preprocessing/cycle keys present
- **PASS** `research_json_provenance` — fps/coordinate/preprocessing keys present
- **PASS** `spectral_feature_records_provenance` — 68 records; key gaps=none; unavailable without reason=0
- **PASS** `asymmetry_scalar_provenance` — 35 scalars; key gaps=none; blank sides without reason=0
- **PASS** `research_items_have_reasons` — non-computed items without reason: none
- **PASS** `opensim_ik_not_inferred_from_leftover_files` — named dir IK=Partial (Mapped TRC ready; IK not completed); pipeline=Complete (TRC and MOT export completed); this-session ik_state=None → Not Started. File presence must not upgrade IK to Complete.
- **PASS** `pipeline_complete_is_not_ik_complete` — app.py 'Pipeline complete' is the Real-to-Sim dialog after run_real_to_sim_pipeline. OpenSim pipeline Complete means TRC+MOT export (Complete). IK Complete requires ik_state=='Completed' (this session: Not Started). Combined leftover readout: Complete · IK Partial.

### recording_b

- **PASS** `video_present` — athletic_walking.mp4 exists=True fps=23.976 frames=120
- **PASS** `poses_present` — athletic_walking_poses.json exists=True
- **PASS** `knee_storage_pose_interior` — recording_knee_storage=pose_interior_hka (pose JSON should be pose_interior_hka)
- **PASS** `knee_flexion_matches_spectral_raw` — time-aligned max |analysis_knee_flexion − spectral raw| = 0.0 (27 matched samples, tol=0.02127s)
- **PASS** `knee_flexion_asymmetry_same_convention` — time-aligned max |analysis_knee_flexion − smoothed kinematics| = 17.100286112978807 (27 samples). Asymmetry kinematics is gap-filled + 0.12 s smoothed; this is not the raw spectral series.
- **PASS** `spectral_grid_spans_snapshots` — spectral n_grid=106 vs recording n_snapshots=27; grid [0.12512512512512514, 4.504504504504505] vs snapshots [0.12512512512512514, 4.504504504504505]
- **PASS** `timestamps_match_asymmetry` — max |recording time − asymmetry time| = 0.0
- **PASS** `sampling_rate_shared` — recording=23.98 Hz spectral=23.98 Hz asymmetry=23.98 Hz video=23.98 Hz
- **PASS** `recording_fps_matches_video` — recording fps 23.98 vs video container fps 23.98
- **PASS** `coordinate_system_shared` — recording='sw_canonical_y_up' spectral='sw_canonical_y_up' asymmetry='sw_canonical_y_up'
- **PASS** `cycle_definition_ipsilateral_hs` — alignment_event='ipsilateral_heel_strike'; gait_cycle_count=1 HS L/R=2/1; asymmetry cycles L/R=1/0; spectral detected/valid=1/1
- **PASS** `cycle_counts_consistent` — Asymmetry ipsilateral cycles are HS→HS subsets of detected heel strikes (L 1/2, R 0/1).
- **PASS** `knee_units_degrees` — spectral left-knee units='deg'; processed mean=2.4671622769447922e-15 (detrend ~0)
- **PASS** `spectral_processed_is_detrended` — processed (detrended flexion) mean=2.467e-15 deg about 0
- **PASS** `spectral_json_provenance` — fps/config/coordinate/cycle keys present
- **PASS** `asymmetry_json_provenance` — fps/coordinate/preprocessing/cycle keys present
- **PASS** `research_json_provenance` — fps/coordinate/preprocessing keys present
- **PASS** `spectral_feature_records_provenance` — 63 records; key gaps=none; unavailable without reason=0
- **PASS** `asymmetry_scalar_provenance` — 28 scalars; key gaps=none; blank sides without reason=0
- **PASS** `research_items_have_reasons` — non-computed items without reason: none
- **PASS** `opensim_ik_not_inferred_from_leftover_files` — named dir IK=Partial (Mapped TRC ready; IK not completed); pipeline=Complete (TRC and MOT export completed); this-session ik_state=None → Not Started. File presence must not upgrade IK to Complete.
- **PASS** `pipeline_complete_is_not_ik_complete` — app.py 'Pipeline complete' is the Real-to-Sim dialog after run_real_to_sim_pipeline. OpenSim pipeline Complete means TRC+MOT export (Complete). IK Complete requires ik_state=='Completed' (this session: Not Started). Combined leftover readout: Complete · IK Partial.

### recording_c

- **PASS** `video_present` — abnormal_gait.mp4 exists=True fps=29.97 frames=150
- **PASS** `poses_present` — abnormal_gait_poses.json exists=True
- **PASS** `knee_storage_pose_interior` — recording_knee_storage=pose_interior_hka (pose JSON should be pose_interior_hka)
- **PASS** `knee_flexion_matches_spectral_raw` — time-aligned max |analysis_knee_flexion − spectral raw| = 0.0 (120 matched samples, tol=0.01702s)
- **PASS** `knee_flexion_asymmetry_same_convention` — time-aligned max |analysis_knee_flexion − smoothed kinematics| = 2.903963920253041 (120 samples). Asymmetry kinematics is gap-filled + 0.12 s smoothed; this is not the raw spectral series.
- **PASS** `spectral_grid_spans_snapshots` — spectral n_grid=120 vs recording n_snapshots=120; grid [0.0, 3.970637303970638] vs snapshots [0.0, 3.9706373039706375]
- **PASS** `timestamps_match_asymmetry` — max |recording time − asymmetry time| = 0.0
- **PASS** `sampling_rate_shared` — recording=29.97 Hz spectral=29.97 Hz asymmetry=29.97 Hz video=29.97 Hz
- **PASS** `recording_fps_matches_video` — recording fps 29.97 vs video container fps 29.97
- **PASS** `coordinate_system_shared` — recording='sw_canonical_y_up' spectral='sw_canonical_y_up' asymmetry='sw_canonical_y_up'
- **PASS** `cycle_definition_ipsilateral_hs` — alignment_event='ipsilateral_heel_strike'; gait_cycle_count=2 HS L/R=3/2; asymmetry cycles L/R=2/1; spectral detected/valid=2/2
- **PASS** `cycle_counts_consistent` — Asymmetry ipsilateral cycles are HS→HS subsets of detected heel strikes (L 2/3, R 1/2).
- **PASS** `knee_units_degrees` — spectral left-knee units='deg'; processed mean=-5.195843755245732e-15 (detrend ~0)
- **PASS** `spectral_processed_is_detrended` — processed (detrended flexion) mean=-5.196e-15 deg about 0
- **PASS** `spectral_json_provenance` — fps/config/coordinate/cycle keys present
- **PASS** `asymmetry_json_provenance` — fps/coordinate/preprocessing/cycle keys present
- **PASS** `research_json_provenance` — fps/coordinate/preprocessing keys present
- **PASS** `spectral_feature_records_provenance` — 68 records; key gaps=none; unavailable without reason=0
- **PASS** `asymmetry_scalar_provenance` — 35 scalars; key gaps=none; blank sides without reason=0
- **PASS** `research_items_have_reasons` — non-computed items without reason: none
- **PASS** `opensim_ik_not_inferred_from_leftover_files` — named dir IK=Partial (Mapped TRC ready; IK not completed); pipeline=Complete (TRC and MOT export completed); this-session ik_state=None → Not Started. File presence must not upgrade IK to Complete.
- **PASS** `pipeline_complete_is_not_ik_complete` — app.py 'Pipeline complete' is the Real-to-Sim dialog after run_real_to_sim_pipeline. OpenSim pipeline Complete means TRC+MOT export (Complete). IK Complete requires ik_state=='Completed' (this session: Not Started). Combined leftover readout: Complete · IK Partial.

## Remaining failures

The harness itself reported no failed checks. These product limits were measured and are **not** closed:

- This session did **not** run Inverse Kinematics. Session `ik_state` is `None` → **Not Started**. Leftover named folders still resolve to **Pipeline Complete · IK Partial**. File presence is not IK success.
- Leftover `normal_gait` IK marker errors: mean RMS **14.8 cm**, max **16.1 cm**, **60 / 120** frames above the 15 cm engineering band. Quality **limited**. `opensim_sdk_available` in that JSON is **false**.
- Leftover `abnormal_gait` IK marker errors: mean RMS **14.2 cm**, **33** frames above 15 cm. Quality **limited**.
- Leftover `athletic_walking` has TRC/MOT/mapped TRC only. No `_ik.mot` and no marker-error `.sto`.
- Newest leftover run folders repeat the same ~14.7 cm mean RMS with **48** frames above 15 cm. Those artifacts are not this session's IK.
- `athletic_walking.mp4`: **27 / 120** pose frames detected. One ipsilateral HS→HS cycle. Cadence withheld. Spectral resampled those 27 snapshots onto a **106**-sample FPS grid. Asymmetry kinematics (0.12 s smooth) differs from raw flexion by up to **17.1°**.
- `normal_gait.mp4` and `abnormal_gait.mp4` each have only **2** complete same-side cycles. Cycle-detector logged support+flight sums of **51.92%** and **70.63%** (not ~100%).
- Asymmetry kinematics is smoothed; Spectral raw is not. Dense clips differ by **3.2°** / **2.9°**. Tabs share the flexion convention, not the identical processed series.
- Asymmetry kinematics `unrealistic_fraction` is **1.0** on `normal_gait` (bone-length / rate flags). That quality flag is not a passed biomechanical validation.
- Tabs were checked through the analysis objects they consume (cached poses). This was not a live GUI click-through.
- Filenames were not used as diagnoses. `abnormal_gait.mp4` is only a second/third source path.

## Changed files

- `stablewalk/analysis/spectral/features.py`
- `stablewalk/analysis/spectral/report.py`
- `stablewalk/analysis/asymmetry3d/compare.py`
- `stablewalk/analysis/asymmetry3d/report.py`
- `stablewalk/analysis/research_validation.py`
- `stablewalk/analysis/pipeline_validation.py`
- `scripts/session_consistency_e2e.py`
- `tests/test_export_provenance.py`
- `docs/session_consistency_e2e.md`

## Notes

Recording B is athletic_walking.mp4 and recording C is abnormal_gait.mp4. Neither filename is treated as a diagnosis.
Tabs were verified through the analysis objects they consume, not by clicking widgets.
JSON provenance fields were added so exports carry formula ids and unavailable reasons.
