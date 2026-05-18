# ik-strategy — Learning by Demonstration for Reachy

> Master's thesis project — Erasmus internship, February–July 2026  
> **Goal:** teach the Reachy humanoid robot to replicate physiotherapy exercises demonstrated by a human, using imitation learning.

---

## Overview

The system captures human upper-body movements via stereo video, extracts 3D joint positions with MediaPipe Pose, solves inverse kinematics to obtain Reachy's joint angles, and then learns to reproduce the motion through two parallel approaches:

- **Canonical trajectory** — DTW Barycenter Averaging (ShapeDBA via `aeon`) computes a representative trajectory from multiple demonstrations.
- **Behavioral Cloning (BC)** — three neural architectures (MLP, GRU, Transformer) trained on state-action pairs from IK-solved trajectories.

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
├── run_bc_approach.py              # Entry point: BC pipeline (build → train → test)
├── run_simulation.py               # Entry point: send any trajectory to simulator/robot
├── run_pipeline.py                 # Legacy single-video pipeline runner
│
├── data_acquisition/               # Step 1 — pose estimation & IK solving
│   ├── pose_estimation.py          #   MediaPipe Pose, mono video → pose.csv
│   ├── pose_estimation_stereo.py   #   stereo video → pose.csv (SGBM depth)
│   ├── stereo_config.py            #   camera intrinsics, extrinsics, SGBM params
│   ├── stereo_calibrate.py         #   stereo calibration tool
│   ├── data_cleaning.py            #   step 2: visibility filter + One Euro Filter
│   ├── mapping.py                  #   step 3: landmark → Reachy torso frame
│   ├── run_ik.py                   #   step 4: IK solver (L-BFGS-B) → joint_ik.csv
│   └── save_landmarks.py           #   CSV writer helper
│
├── bc_approach/                    # Behavioral Cloning
│   ├── build_dataset.py            #   assembles bc_dataset.csv (state, velocity, action)
│   ├── evaluate_bc.py              #   per-model DTW evaluation vs baseline
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
├── canonical_approach/             # Canonical trajectory via ShapeDBA
│   └── compute_canonical.py        #   ShapeDTW Barycenter Averaging → canonical.csv
│
├── evaluation_and_comparison/      # Quantitative evaluation
│   └── evaluate.py                 #   DTW, RMSE, Pearson, smoothness, velocity profile
│
├── reachyController/               # Low-level robot interface
│   ├── reachyController.py         #   joint trajectory execution via ReachySDK
│   ├── timeSeries.py               #   time-series utilities for robot execution
│   └── config.py                   #   robot-side constants
│
└── utilities/                      # Shared helpers
    ├── config.py                   #   DATA_ROOT, JOINT_COLS, joint limits, REST_DEG
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
└── dataset/
    └── exercise_XXX/
        ├── bc_dataset.csv              # state-action pairs for BC training
        ├── canonical.csv               # ShapeDBA output
        ├── baseline.csv                # reference trajectory
        ├── MLP/                        # model checkpoints + scalers
        ├── GRU/
        ├── Transformer/
        ├── plot/                       # FK trajectory plots
        └── evaluation/                 # evaluate.py output (CSV + PNG)
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

```bash
py -m canonical_approach.compute_canonical                  # all exercises
py -m canonical_approach.compute_canonical --exercise 1     # single exercise
```

### BC approach (build → train → test)

```bash
py run_bc_approach.py --exercise 1          # all three architectures
py run_bc_approach.py --exercise 1 --mlp-only
py run_bc_approach.py --exercise a          # all exercises (1–25)
```

### Simulation / robot deployment

```bash
py run_simulation.py --mode canonical    --exercise 1
py run_simulation.py --mode mlp         --exercise 1
py run_simulation.py --mode gru         --exercise 1 --runs 3
py run_simulation.py --mode transformer --exercise 1 --host 10.59.1.20
py run_simulation.py --mode video       --subject 1 --exercise 1 --video 1
```

### Evaluation

```bash
py -m evaluation_and_comparison.evaluate --exercise 1
py -m evaluation_and_comparison.evaluate --exercise 1 11 21   # stereo vs mixed vs mono
py -m evaluation_and_comparison.evaluate --all
```

**Metrics computed:** DTW distance, RMSE per joint, peak angle error, Pearson correlation, smoothness (−jerk²), velocity profile.

---

## Key design choices

| Choice | Rationale |
|--------|-----------|
| 8 active joints per arm (4 solved by IK, 4 fixed at 0) | Forearm/wrist orientation not reliably recoverable from RGB pose estimation |
| `VELOCITY_LAG = 5` frames for velocity features | More robust than single-frame finite difference; dampens startup transients |
| Video-level train/val split (k-fold) | Prevents temporal data leakage across frames of the same video |
| ShapeDBA instead of standard DBA | Aligns shape descriptors of subsequences → smoother, more shape-faithful canonical |
| Stereo depth with persistence + spike rejection | SGBM alone is noisy; persistence fallback fills gaps without WLS frame drops |
| Start pose saved into model checkpoint | Critical for correct BC inference initialization; avoids divergence at frame ~100 |

---

## Dependencies

```
opencv-contrib-python >= 4.13   # includes cv2.ximgproc (SGBM, right matcher)
mediapipe
torch
aeon                            # ShapeDBA
tslearn                         # DTW metrics
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