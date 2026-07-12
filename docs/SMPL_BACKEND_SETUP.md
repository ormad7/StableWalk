# SMPL / Human Mesh Recovery backend setup

StableWalk supports an **optional** SMPL-based motion extraction backend alongside the
default **MediaPipe** landmark tracker.

## MediaPipe vs SMPL — important difference

| Aspect | MediaPipe (default) | SMPL (optional) |
|--------|---------------------|-----------------|
| Output | 2D/3D **landmarks** (BlazePose) | **Parametric body mesh** (β shape, θ pose) |
| Coordinates | Normalized image + heuristic depth | Camera-centered metric mesh (ROMP) |
| Dependencies | CPU-friendly, no PyTorch | PyTorch, ROMP, licensed SMPL files, GPU recommended |
| OpenSim export | Uses landmark → marker mapping (unchanged) | Same downstream path via canonical joints |
| Clinical use | Moderate kinematic proxy | Research-grade mesh recovery (still monocular) |

MediaPipe remains the **default** and **fallback**. SMPL is never synthesized when
dependencies are missing.

## Backend modes

Set via GUI (**Data & Export → Pose backend**), CLI, or environment:

```powershell
set POSE_BACKEND=mediapipe   # default
set POSE_BACKEND=smpl        # require SMPL; optional fallback if POSE_BACKEND_ALLOW_FALLBACK=true
set POSE_BACKEND=auto        # try SMPL, fall back to MediaPipe with warning
```

CLI:

```powershell
python main.py --pose-backend auto data/videos/my_walk.mp4
```

## Obtaining SMPL model files (legal requirement)

StableWalk **does not** download SMPL or SMPL-X weights automatically.

1. Register at [SMPL](https://smpl.is.tue.mpg.de/) (and optionally [SMPL-X](https://smpl-x.is.tue.mpg.de/)).
2. Download the **SMPL neutral** model (`SMPL_NEUTRAL.pkl` or equivalent).
3. Create a local directory, e.g. `C:\models\smpl\`, and place the file there.
4. Set the environment variable:

```powershell
set SMPL_MODEL_DIR=C:\models\smpl
```

Optional SMPL-X:

```powershell
set SMPLX_MODEL_DIR=C:\models\smplx
```

Expected layout:

```
SMPL_MODEL_DIR/
  SMPL_NEUTRAL.pkl
```

## Installing ROMP (separate conda environment)

Do **not** install PyTorch/ROMP into the main OpenSim/MediaPipe environment.

```powershell
conda create -n stablewalk-hmr python=3.10
conda activate stablewalk-hmr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install romp
set SMPL_MODEL_DIR=C:\path\to\smpl
```

Follow the official ROMP repository for version-specific instructions:
https://github.com/Arthur151/ROMP

## Exports

After processing with the SMPL backend:

```
data/output/motion_reference/<run_name>/
  stablewalk_motion.npz    # canonical gait motion (always when Real-to-Sim runs)
  smpl_motion.npz          # SMPL-only: root, rotations, β, joint positions (SMPL backend only)
```

`smpl_motion.npz` is written **only** when real SMPL inference ran — never for
MediaPipe or fallback sessions.

## Validation

Check readiness from Python:

```python
from stablewalk.pose.backends.smpl_validation import validate_smpl_assets
print(validate_smpl_assets().to_dict())
```

Or:

```python
from stablewalk.pose.backends.registry import list_backend_diagnostics
print(list_backend_diagnostics())
```

## Troubleshooting

| Issue | Action |
|-------|--------|
| Auto mode always uses MediaPipe | Set `SMPL_MODEL_DIR`, install torch+romp in active env |
| `smpl` mode raises error | Expected when dependencies missing; use `auto` or install stack |
| Slow inference | Enable CUDA GPU; reduce `--max-frames` for tests |
| OpenSim IK quality | OpenSim export still uses landmark mapping; SMPL does not replace Demo IK |

See also: `docs/POSE_BACKENDS.md`, `docs/REAL_TO_SIM_UPGRADE_PLAN.md`
