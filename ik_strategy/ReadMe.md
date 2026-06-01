# ik-strategy — Learning by Demonstration for Reachy

> Master's thesis project — Erasmus internship, February–July 2026  
> **Goal:** teach the Reachy humanoid robot to replicate physiotherapy exercises demonstrated by a human, using imitation learning.

---

## Overview

The system captures human upper-body movements via stereo video, extracts 3D joint positions with MediaPipe Pose, solves inverse kinematics to obtain Reachy's joint angles, and then learns to reproduce the motion through two parallel approaches:

- **Canonical trajectory** — two methods computed in parallel from the same demonstrations:
  - **Standard DBA** (DTW Barycenter Averaging via `tslearn`) → `canonical.csv`
  - **ShapeDBA** (ShapeDTW Barycenter Averaging via `aeon`) → `canonicalShape.csv`
- **Behavioral Cloning (BC)** — three neural architectures (MLP, GRU, Transformer) trained on state-action pairs from IK-solved trajectories. Multiple independent training runs can be aggregated to reduce variance.

Exercise numbering encodes the acquisition modality:

| Range | Modality |
|-------|----------|
| 001–005 | Stereo (Reachy's cameras) |
| 011–015 | Mixed (stereo + mono) |
| 021–025 | Mono (single webcam) |

---

## Setup

> Requires Python 3.9 and a virtual environment (`.venv_win39` on Windows).

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install the package in editable mode (required for absolute imports)
pip install -e .
```

---

## Project structure

```
ik_strategy/
│
├── pyproject.toml                  # Package definition (setuptools)
├── requirements.txt
│
├── run_data_acquisition.py         # Entry point: full acquisition pipeline (steps 1–4)
├── run_compute_canonical.py        # Entry point: canonical trajectory (DBA + ShapeDBA)
├── run_bc_approach.py              # Entry point: BC pipeline (build → train → test → aggregate)
├── run_simulation.py               # Entry point: send any trajectory to simulator/robot
├── run_pipeline.py                 # Legacy single-video pipeline runner
│
├── data_acquisition/               # Steps 1–4 — pose estimation & IK solving
│   ├── pose_estimation.py          #   MediaPipe Pose, mono video → pose.csv
│   ├── pose_estimation_stereo.py   #   stereo video → pose.csv (SGBM depth)
│   ├── stereo_config.py            #   camera intrinsics, extrinsics, SGBM params
│   ├── stereo_calibrate.py         #   stereo calibration tool
│   ├── data_cleaning.py            #   step 2: visibility filter + One Euro Filter
│   ├── mapping.py                  #   step 3: landmark → Reachy torso frame
│   ├── run_ik.py                   #   step 4: IK solver (L-BFGS-B) → joint_ik.csv
│   └── save_landmarks.py           #   CSV writer helper
│
├── canonical_approach/
│   └── compute_canonical.py        #   ShapeDBA + Standard DBA in parallel → canonical.csv + canonicalShape.csv
│
├── bc_approach/                    # Behavioral Cloning
│   ├── build_dataset.py            #   assembles n_XX/bc_dataset.csv (state, velocity, action)
│   ├── aggregate_runs.py           #   averages multiple training runs → bc_trajectory_mean.csv + runs_metrics.csv
│   ├── evaluate_bc.py              #   legacy per-model DTW evaluation (standalone)
│   ├── MLP/
│   │   ├── train_bc.py             #   MLP training (k-fold, early stopping)
│   │   └── test_bc.py              #   MLP autoregressive inference + FK plot
│   ├── GRU/
│   │   ├── train_bc.py             #   GRU training
│   │   └── test_bc.py              #   GRU autoregressive inference (with hidden state)
│   └── Transformer/
│       ├── train_bc.py             #   Transformer training (causal window)
│       └── test_bc.py              #   Transformer autoregressive inference
│
├── evaluation_and_comparison/      # Quantitative evaluation
│   ├── evaluate.py                 #   main entry point (per-exercise, modality, demos)
│   ├── evaluate_exercise.py        #   full evaluation of a single exercise
│   ├── evaluate_modality.py        #   Stereo vs Mixed vs Mono comparison
│   ├── evaluate_demos.py           #   sensitivity to number of demonstrations
│   ├── _config.py                  #   shared constants (architectures, palette, modality groups)
│   ├── _io.py                      #   CSV read/write helpers
│   ├── _metrics.py                 #   DTW, RMSE, Pearson, smoothness, velocity
│   └── _plots.py                   #   all plot functions (degradation chain, heatmap, spider, 3D...)
│
├── reachyController/               # Low-level robot interface (Benoît)
│   ├── reachyController.py         #   joint trajectory execution via ReachySDK
│   ├── timeSeries.py               #   time-series utilities for robot execution
│   └── config.py                   #   robot-side constants
│
└── utilities/                      # Shared helpers
    ├── config.py                   #   DATA_ROOT, JOINT_COLS, joint limits, REST_DEG
    ├── split_utils.py              #   split_name(), N_DEMOS_SPLITS
    ├── ask_inputs.py               #   interactive subject/exercise/video prompts
    ├── record_stereo.py            #   synchronized stereo recording from Reachy cameras
    ├── plot_pose.py                #   pose time-series plots (X/Y/Z per landmark)
    └── plot_joints.py              #   joint angle plots from joint_ik.csv
```

---

## Data structure

All data lives under `ik_strategy/data/` (not tracked in git):

```
data/
├── raw_data/
│   └── subject_XXX/
│       └── exercise_XXX/
│           ├── video_XXX.mp4           # mono
│           ├── video_XXX_L.mp4         # stereo left
│           └── video_XXX_R.mp4         # stereo right
│
├── landmarks/
│   └── subject_XXX/
│       └── exercise_XXX/
│           └── video_XXX/
│               ├── pose.csv            # raw MediaPipe output
│               ├── pose_cleaned.csv    # after data_cleaning.py
│               ├── poses_mapped.csv    # after mapping.py (torso frame)
│               └── joint_ik.csv        # final joint angles (degrees)
│
├── dataset/
│   └── exercise_XXX/
│       ├── baseline.csv                # reference trajectory
│       └── n_XX/                       # one folder per demo split (n=10, 25, 55)
│           ├── canonical.csv           # Standard DBA output
│           ├── canonicalShape.csv      # ShapeDBA output
│           ├── bc_dataset.csv          # state-action pairs for BC training
│           ├── MLP/
│           │   ├── bc_model_fold_N.pth
│           │   ├── scaler_fold_N.pkl
│           │   ├── bc_trajectory.csv       # single-run inference output
│           │   ├── bc_trajectory_mean.csv  # mean over multiple runs (if training-runs > 1)
│           │   ├── runs_metrics.csv        # per-run metrics + std
│           │   └── run_N/                  # subfolder for each training run
│           ├── GRU/                    # same structure as MLP/
│           ├── Transformer/            # same structure as MLP/
│           └── evaluation/
│               ├── results_summary.csv
│               ├── results_per_joint.csv
│               ├── results_per_endpoint.csv
│               └── plot_*.png
│
├── evaluation_modality/            # output of evaluate_modality.py
│   ├── results_by_exercise.csv
│   ├── results_aggregated.csv
│   └── plot_*.png
│
└── evaluation_demos/               # output of evaluate_demos.py
    ├── results_all_exercises.csv
    ├── results_aggregated.csv
    └── plot_*.png
```

---

## Pipelines

### Data acquisition

Processes one or more videos end to end. Auto-detects stereo vs mono from file names.

```bash
# Interactive
py run_data_acquisition.py

# Single video
py run_data_acquisition.py --subject 1 --exercise 2 --video 3

# All videos of all subjects for exercise 1
py run_data_acquisition.py --exercise 1
# (then answer 'a' when prompted for subject and video)
```

**Steps (in order):**

| # | Module | Input → Output |
|---|--------|----------------|
| 1 | `pose_estimation.py` / `pose_estimation_stereo.py` | `.mp4` → `pose.csv` |
| 2 | `data_cleaning.py` | `pose.csv` → `pose_cleaned.csv` |
| 3 | `mapping.py` | `pose_cleaned.csv` → `poses_mapped.csv` |
| 4 | `run_ik.py` | `poses_mapped.csv` → `joint_ik.csv` |

### Canonical approach

Computes both Standard DBA and ShapeDBA in a single run.

```bash
py run_compute_canonical.py --n-demos 55           # all exercises
py run_compute_canonical.py --exercise 1           # single exercise
py run_compute_canonical.py --exercise 1 --n-demos 10 --reach 5
```

### BC approach (build → train → test → aggregate)

```bash
py run_bc_approach.py --exercise 1                          # all three architectures, single run
py run_bc_approach.py --exercise 1 --mlp-only               # MLP only
py run_bc_approach.py --exercise 1 --training-runs 5        # 5 independent runs + aggregation
py run_bc_approach.py --exercise a --n-demos 10             # all exercises, split n=10
py run_bc_approach.py --exercise a --start 11 --end 15      # exercise range 11–15
```

When `--training-runs > 1`, each architecture is trained N times with different seeds, then `aggregate_runs.py` computes `bc_trajectory_mean.csv` and `runs_metrics.csv` automatically.

To aggregate manually:

```bash
py -m bc_approach.aggregate_runs --exercise 1 --n-demos 55
py -m bc_approach.aggregate_runs --exercise 1 --n-demos 10 25 55
```

### Simulation / robot deployment

Reads a pre-computed CSV and sends it to the Unity simulator or the real robot.

```bash
py run_simulation.py                                               # interactive mode selector
py run_simulation.py --mode canonical        --exercise 21
py run_simulation.py --mode canonical_shape  --exercise 21
py run_simulation.py --mode mlp              --exercise 21 --n-demos 55
py run_simulation.py --mode gru              --exercise 21 --n-demos 25 --runs 3
py run_simulation.py --mode transformer      --exercise 21
py run_simulation.py --mode baseline         --exercise 21 --host 10.59.1.20
py run_simulation.py --mode video            --subject 1 --exercise 21 --video 1
```

Available modes: `video`, `baseline`, `canonical`, `canonical_shape`, `mlp`, `gru`, `transformer`.

### Evaluation

```bash
# Single exercise
py -m evaluation_and_comparison.evaluate --exercise 1
py -m evaluation_and_comparison.evaluate --exercise 1 --n-demos 10 25 55

# All exercises + modality analysis
py -m evaluation_and_comparison.evaluate --all

# Modality comparison (Stereo vs Mixed vs Mono)
py -m evaluation_and_comparison.evaluate --analysis modality
py -m evaluation_and_comparison.evaluate_modality --n-demos 55

# Demos sensitivity (ablation study)
py -m evaluation_and_comparison.evaluate --analysis demos --exercise 21
py -m evaluation_and_comparison.evaluate_demos --exercise 21
py -m evaluation_and_comparison.evaluate_demos --all
```

**Metrics computed:** Cartesian DTW, RMSE per joint, RMSE wrist/elbow, peak angle error, Pearson correlation (joint-space and Cartesian), smoothness (−mean squared jerk), velocity profile.

**Full evaluation sequence (all splits, all analyses):**

```bash
py run_compute_canonical.py --n-demos 10
py run_compute_canonical.py --n-demos 25
py run_compute_canonical.py --n-demos 55

py run_bc_approach.py --exercise a --n-demos 10  --training-runs 5
py run_bc_approach.py --exercise a --n-demos 25  --training-runs 5
py run_bc_approach.py --exercise a --n-demos 55  --training-runs 5

py -m evaluation_and_comparison.evaluate --all --n-demos 10
py -m evaluation_and_comparison.evaluate --all --n-demos 25
py -m evaluation_and_comparison.evaluate --all --n-demos 55

py -m evaluation_and_comparison.evaluate_demos --all
py -m evaluation_and_comparison.evaluate_modality --n-demos 55
```

---

## Key design choices

| Choice | Rationale |
|--------|-----------|
| 8 active joints (shoulder_pitch, shoulder_roll, arm_yaw, elbow_pitch × 2 arms) | Forearm/wrist orientation not reliably recoverable from RGB pose estimation |
| `VELOCITY_LAG = 5` frames for velocity features | More robust than single-frame finite difference; dampens startup transients and accumulates less autoregressive error |
| Video-level train/val split (k-fold with `math.ceil`) | Prevents temporal data leakage; `math.ceil` avoids Python's banker's rounding (e.g. K=3 not K=2 with 10 videos) |
| ShapeDBA alongside Standard DBA | ShapeDBA aligns shape descriptors of subsequences → smoother, more shape-faithful canonical; fair comparison via shared amplitude normalization |
| Amplitude normalization before DBA | Normalize per-joint to [0,1] before averaging, then rescale using the Nth percentile of demo amplitudes — prevents low-amplitude demos from dominating the barycenter |
| Multiple training runs + aggregation | `bc_trajectory_mean.csv` averages runs frame-by-frame; `runs_metrics.csv` exposes inter-run std shown as error bars in evaluation plots |
| Stereo depth with persistence + spike rejection | SGBM alone is noisy; persistence fallback fills gaps; One Euro Filter smooths Z over time |
| Start pose saved into model checkpoint | Critical for correct BC inference initialization; avoids divergence at frame ~100 |
| FK-based stopping criterion (two-phase state machine) | Exercise-agnostic; stops when wrists return within 0.10 m of start after having departed — no fixed `n_steps` required |
| Collision check every 5 frames during playback | `_checkCollision` is called every 5 frames via `_safety_check`; `goal_position` is set every frame for smooth motion |

---

## Dependencies

```
opencv-contrib-python >= 4.13   # includes cv2.ximgproc (SGBM, WLS filter)
mediapipe
torch
aeon                            # ShapeDBA (elastic_barycenter_average)
tslearn                         # Standard DBA + DTW metrics
scipy
pandas
numpy
matplotlib
reachy-sdk
```

---

## Authors

Nicolò — master's student, Computer Engineering (Automation and Intelligent Cyber-Physical Systems)  
Erasmus thesis internship, February–July 2026