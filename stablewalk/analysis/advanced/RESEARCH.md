# Advanced gait analytics — research & biomechanics context

## Why these modules matter

Clinical and research gait labs measure **spatiotemporal parameters** (cadence, stride length, stance/swing times), **kinematics** (joint angles), and **kinetics** (ground reaction force, CoM). StableWalk’s advanced layer mirrors that pipeline using **monocular video** instead of force plates and motion-capture suits.

| StableWalk module | Biomechanics / clinical analogue |
|-------------------|----------------------------------|
| `GaitFeatureVector` | Parameter export from GaitRite, Vicon, or wearable summaries |
| `GaitAnomalyDetector` | Automated QC / fall-risk screening (deviation from normative bands) |
| `GaitClassifier` | Triage: typical vs atypical gait (not diagnosis) |
| `GaitSessionComparison` | Pre/post rehab, baseline vs current, or bilateral comparison |
| `StabilityMLModel` | Data-driven models (e.g. ML on IMU/vision features for fall risk) |

## 1. Anomaly detection

**Research relevance:** Unsupervised and rule-based outlier detection is widely used when labeled pathology data is scarce—e.g. detecting unusual stride variability in elderly cohorts or post-stroke monitoring.

**Real systems:** Instrumented walkways flag steps outside normative cadence/stride bands; wearables trigger alerts on abnormal step regularity. Our statistical norms + optional `IsolationForest` on a **healthy reference cohort** follow the same idea: learn “typical” variation, flag outliers.

**Limitation:** Norms are approximate; camera view and 2D pose add noise. Outputs support **screening**, not clinical diagnosis.

## 2. Comparison (two videos)

**Research relevance:** Longitudinal studies (before/after intervention) and case-control designs compare the same metrics across sessions—core methodology in physiotherapy and motor-control research.

**Real systems:** Gait labs report Δcadence, Δstride, Δsymmetry after training; COSMIN guidelines stress reproducible spatiotemporal endpoints. `compare_gait_sessions` reports metric deltas plus joint-angle differences (via existing `gait_comparison`).

## 3. Classification (normal vs abnormal)

**Research relevance:** Binary or three-class gait labels appear in stroke, Parkinson’s, and frailty studies—often from expert rating or structured tests (e.g. Dynamic Gait Index), sometimes automated from sensors.

**Real systems:** Hospital fall-risk tools combine slow gait, asymmetry, and variability. Our **rule hybrid** (stability score + anomaly + pattern flags) is interpretable; optional **RandomForest** on features matches supervised pipelines when labeled training clips exist.

**Limitation:** “Abnormal” here means **atypical relative to norms and stability heuristics**, not a specific pathology.

## 4. Optional ML for stability score

**Research relevance:** Regression and ensemble models map high-dimensional gait features to clinical scores (Tinetti, Berg Balance, etc.) in digital-health literature.

**Real systems:** After calibration on labeled data, models predict risk scores from IMU or pressure insoles. `StabilityMLModel` learns **features → stability score** from your dataset (Ridge or gradient boosting), useful for semester projects exploring supervised learning on exported `GaitFeatureVector` rows.

**Requirement:** `pip install scikit-learn joblib`; train with ≥3 labeled walks; validate before any real use.

## Connection summary

```
Video → Pose (MediaPipe) → Events & metrics → Features
                              ↓
         Anomaly / Classify / Compare  ←→  Clinical: screening & longitudinal charts
                              ↓
         Optional ML stability     ←→  Research: supervised risk models
```

StableWalk stays honest: **estimated** GRF and **2D** pose approximate lab-grade kinetics and 3D kinematics. The advanced layer is structurally aligned with real biomechanics workflows while remaining suitable for a CS capstone demonstration.
