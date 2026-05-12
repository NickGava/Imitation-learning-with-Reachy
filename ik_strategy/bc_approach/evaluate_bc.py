'''
evaluate_bc.py
=============================================================================
Evaluates a trained BC model against the baseline trajectory.
Metric: DTW on FK positions (wrist + elbow, 3D) for both arms.

Hyperparameters are read automatically from the model checkpoints.
Results are appended to a shared CSV table for cross-config comparison.

Usage:
  py evaluate_bc.py --arch mlp         --exercise 1
  py evaluate_bc.py --arch gru         --exercise 1
  py evaluate_bc.py --arch transformer --exercise 1

--- Input ---
  _data/dataset/exercise_XXX/{ARCH}/bc_model_fold_N.pth
  _data/dataset/exercise_XXX/{ARCH}/scaler_fold_N.pkl
  _data/dataset/exercise_XXX/canonical.csv    (determines n_steps)
  _data/dataset/exercise_XXX/baseline.csv     (reference trajectory)

--- Output (row appended) ---
  _data/dataset/bc_eval_results.csv
'''

import argparse
import csv
import importlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tslearn.metrics import dtw

from utilities.config import DATA_ROOT, JOINT_COLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_JOINTS    = len(JOINT_COLS)           # 16  (8 per arm, including gripper)

RESULTS_HEADER = [
    'timestamp', 'arch', 'exercise',
    # --- hyperparameters (N/A when not applicable to the arch) ---
    'hidden_size',                          # MLP, GRU
    'lr', 'batch_size', 'noise_std',        # all
    'seq_len', 'n_layers',                  # GRU, Transformer
    'd_model', 'n_heads', 'd_ff', 'dropout',# Transformer
    # --- training metrics (averaged across folds) ---
    'val_loss_mean', 'val_loss_std', 'n_folds',
    # --- DTW vs baseline FK (metres) ---
    'dtw_wrist_r', 'dtw_elbow_r',
    'dtw_wrist_l', 'dtw_elbow_l',
    'dtw_avg',
    # --- diagnostic info ---
    'n_steps_bc', 'n_steps_baseline',
]


# ---------------------------------------------------------------------------
# FK  (mirrors test_bc.py — copied here to avoid circular imports)
# Uses only the first 4 joints of each arm: shoulder_pitch, shoulder_roll,
# arm_yaw, elbow_pitch. Forearm/wrist are fixed at 0 in this approximation.
#
# JOINT_COLS layout (indices):
#   0  r_shoulder_pitch   8  l_shoulder_pitch
#   1  r_shoulder_roll    9  l_shoulder_roll
#   2  r_arm_yaw         10  l_arm_yaw
#   3  r_elbow_pitch     11  l_elbow_pitch
#   4  r_forearm_yaw     12  l_forearm_yaw
#   ...                  ...
# ---------------------------------------------------------------------------
_FK_JOINTS = [
    ('shoulder_pitch', None,                          np.array([0., 1., 0.])),
    ('shoulder_roll',  np.array([0.,   0.,    0.  ]), np.array([1., 0., 0.])),
    ('arm_yaw',        np.array([0.,   0.,    0.  ]), np.array([0., 0., 1.])),
    ('elbow_pitch',    np.array([0.,   0.,   -0.28]), np.array([0., 1., 0.])),
    ('forearm_yaw',    np.array([0.,   0.,    0.  ]), np.array([0., 0., 1.])),
    ('wrist_pitch',    np.array([0.,   0.,   -0.25]), np.array([0., 1., 0.])),
    ('wrist_roll',     np.array([0.,   0., -0.0325]), np.array([1., 0., 0.])),
]
_ELBOW_IDX  = 3
_WRIST_IDX  = 5
_SHOULDER_Y = {'right': -0.19, 'left': 0.19}


def _rot(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s   = np.cos(angle_rad), np.sin(angle_rad)
    x, y, z = axis
    return np.array([
        [c+x*x*(1-c),   x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s, c+y*y*(1-c),   y*z*(1-c)-x*s],
        [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)  ],
    ])


def _fk(q4_deg: np.ndarray, side: str):
    """Forward kinematics for one arm given 4 joint angles (degrees)."""
    q7_rad = np.deg2rad(np.append(q4_deg, [0., 0., 0.]))
    T = np.eye(4)
    elbow_pos = wrist_pos = None
    for i, (_, trans, axis) in enumerate(_FK_JOINTS):
        if i == 0:
            trans = np.array([0., _SHOULDER_Y[side], 0.])
        T[:3, 3]   += T[:3, :3] @ trans
        T[:3, :3]   = T[:3, :3] @ _rot(axis, q7_rad[i])
        if i == _ELBOW_IDX:
            elbow_pos = T[:3, 3].copy()
        if i == _WRIST_IDX:
            wrist_pos = T[:3, 3].copy()
    return elbow_pos, wrist_pos


def _compute_fk_trajectory(q_traj: np.ndarray):
    """
    q_traj : (T, N_JOINTS=16)
    Returns two dicts (right, left), each with 'elbow' and 'wrist' arrays of
    shape (T, 3) in the Reachy torso frame (metres).
    """
    T = len(q_traj)
    r_elbow = np.zeros((T, 3)); r_wrist = np.zeros((T, 3))
    l_elbow = np.zeros((T, 3)); l_wrist = np.zeros((T, 3))
    for t in range(T):
        r_elbow[t], r_wrist[t] = _fk(q_traj[t, 0:4], 'right')
        l_elbow[t], l_wrist[t] = _fk(q_traj[t, 8:12], 'left')
    r_fk = {'elbow': r_elbow, 'wrist': r_wrist}
    l_fk = {'elbow': l_elbow, 'wrist': l_wrist}
    return r_fk, l_fk


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _load_checkpoints(model_dir: Path) -> list:
    """Loads all bc_model_fold_N.pth files from model_dir."""
    ckpts, fold_idx = [], 0
    while True:
        p = model_dir / f'bc_model_fold_{fold_idx}.pth'
        if not p.exists():
            break
        ckpts.append(torch.load(p, map_location='cpu', weights_only=False))
        fold_idx += 1
    if not ckpts:
        raise FileNotFoundError(
            f'No fold checkpoints found in {model_dir}.\n'
            f'Run the corresponding train_bc.py first.')
    return ckpts


def _extract_training_info(ckpts: list) -> tuple:
    """Returns (val_loss_mean, val_loss_std, n_folds, hparams_dict)."""
    val_losses = np.array([c.get('val_loss', float('nan')) for c in ckpts])
    hparams    = ckpts[0].get('hparams', {})
    return float(val_losses.mean()), float(val_losses.std()), len(ckpts), hparams


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------
def _append_row(row: dict, RESULTS_CSV) -> None:
    """Appends one result row to RESULTS_CSV (creates file + header if needed)."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f'\nResults appended → {RESULTS_CSV}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate BC model vs baseline using DTW on FK positions.')
    parser.add_argument('--arch',     type=str, required=True,
                        choices=['mlp', 'gru', 'transformer'],
                        help='Architecture to evaluate.')
    parser.add_argument('--exercise', type=int, required=True,
                        help='Exercise number (e.g. 1).')
    args = parser.parse_args()

    arch         = args.arch.lower()
    arch_upper   = arch.upper()
    exercise_dir = DATA_ROOT / 'dataset' / f'exercise_{args.exercise:03d}'
    
    RESULTS_CSV = DATA_ROOT / 'dataset' / f'exercise_{args.exercise:03d}' / 'bc_eval_results.csv'
    model_dir    = exercise_dir / arch_upper

    print(f'\n{"="*60}')
    print(f'Evaluating BC [{arch_upper}]  —  exercise {args.exercise:03d}')
    print(f'{"="*60}')

    # ---- Import arch-specific inference functions --------------------------
    module = importlib.import_module(f'bc_approach.{arch_upper}.test_bc')
    load_ensemble   = module.load_ensemble
    run_bc_loop     = module.run_bc_loop
    load_start_pose = module._load_start_pose

    # ---- Load ensemble + start pose ----------------------------------------
    print(f'\nLoading ensemble from {model_dir} ...')
    ensemble   = load_ensemble(model_dir)
    start_pose = load_start_pose(exercise_dir)

    # ---- Resolve n_steps from canonical.csv --------------------------------
    canonical_path = exercise_dir / 'canonical.csv'
    if not canonical_path.exists():
        raise FileNotFoundError(f'canonical.csv not found in {exercise_dir}')
    n_steps = len(pd.read_csv(canonical_path))
    print(f'n_steps (from canonical.csv): {n_steps}')

    # ---- Load baseline -----------------------------------------------------
    baseline_path = exercise_dir / 'baseline.csv'
    if not baseline_path.exists():
        raise FileNotFoundError(f'baseline.csv not found in {exercise_dir}')
    df_base = pd.read_csv(baseline_path)

    missing = [c for c in JOINT_COLS if c not in df_base.columns]
    if missing:
        raise ValueError(f'baseline.csv is missing columns: {missing}')

    q_base = df_base[JOINT_COLS].values.astype(np.float32)
    n_steps_baseline = len(q_base)
    print(f'Baseline:  {n_steps_baseline} steps')

    if n_steps_baseline != n_steps:
        print(f'  [INFO] Length mismatch — BC: {n_steps}, baseline: {n_steps_baseline}.')
        print(f'  DTW will handle this automatically.')

    # ---- Run BC autoregressive loop ----------------------------------------
    print(f'\nRunning BC loop ({n_steps} steps) ...')
    q_bc = run_bc_loop(ensemble, n_steps, start_pose)   # (n_steps, N_JOINTS)
    print(f'BC output shape: {q_bc.shape}')

    # ---- Forward kinematics ------------------------------------------------
    print('Computing FK ...')
    r_fk_bc,   l_fk_bc   = _compute_fk_trajectory(q_bc)
    r_fk_base, l_fk_base = _compute_fk_trajectory(q_base)

    # ---- DTW on FK positions -----------------------------------------------
    # Each input to dtw() is shape (T, 3) — multivariate time series.
    # tslearn.metrics.dtw handles different lengths natively.
    print('Computing DTW ...')
    dtw_wrist_r = float(dtw(r_fk_bc['wrist'],  r_fk_base['wrist']))
    dtw_elbow_r = float(dtw(r_fk_bc['elbow'],  r_fk_base['elbow']))
    dtw_wrist_l = float(dtw(l_fk_bc['wrist'],  l_fk_base['wrist']))
    dtw_elbow_l = float(dtw(l_fk_bc['elbow'],  l_fk_base['elbow']))
    dtw_avg     = float(np.mean([dtw_wrist_r, dtw_elbow_r,
                                  dtw_wrist_l, dtw_elbow_l]))

    print(f'\n--- DTW vs baseline (metres) ---')
    print(f'  Wrist  R : {dtw_wrist_r:.4f}')
    print(f'  Elbow  R : {dtw_elbow_r:.4f}')
    print(f'  Wrist  L : {dtw_wrist_l:.4f}')
    print(f'  Elbow  L : {dtw_elbow_l:.4f}')
    print(f'  Average  : {dtw_avg:.4f}')

    # ---- Extract training info from checkpoints ----------------------------
    ckpts                            = _load_checkpoints(model_dir)
    val_mean, val_std, n_folds, hp   = _extract_training_info(ckpts)
    print(f'\nVal loss : {val_mean:.6f} ± {val_std:.6f}  ({n_folds} folds)')
    if hp:
        print(f'Hparams  : {hp}')
    else:
        print('  [WARN] No hparams dict found in checkpoint.')
        print('  Add hparams={"lr":..., ...} to torch.save() in train_bc.py.')

    def _hp(key):
        return hp.get(key, 'N/A')

    # ---- Build and save result row -----------------------------------------
    row = {
        'timestamp':          datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'arch':               arch_upper,
        'exercise':           args.exercise,
        # hyperparameters
        'hidden_size':        _hp('hidden_size'),
        'lr':                 _hp('lr'),
        'batch_size':         _hp('batch_size'),
        'noise_std':          _hp('noise_std'),
        'seq_len':            _hp('seq_len'),
        'n_layers':           _hp('n_layers'),
        'd_model':            _hp('d_model'),
        'n_heads':            _hp('n_heads'),
        'd_ff':               _hp('d_ff'),
        'dropout':            _hp('dropout'),
        # training
        'val_loss_mean':      round(val_mean, 6),
        'val_loss_std':       round(val_std,  6),
        'n_folds':            n_folds,
        # DTW
        'dtw_wrist_r':        round(dtw_wrist_r, 4),
        'dtw_elbow_r':        round(dtw_elbow_r, 4),
        'dtw_wrist_l':        round(dtw_wrist_l, 4),
        'dtw_elbow_l':        round(dtw_elbow_l, 4),
        'dtw_avg':            round(dtw_avg,      4),
        # diagnostic
        'n_steps_bc':         n_steps,
        'n_steps_baseline':   n_steps_baseline,
    }

    _append_row(row, RESULTS_CSV)
    print(f'\nDone.')


if __name__ == '__main__':
    main()