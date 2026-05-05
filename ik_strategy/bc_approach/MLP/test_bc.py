'''
MLP/test_bc.py
=============================================================================
Runs the BC MLP autoregressive loop for a given exercise and plots the
FK trajectories for both arms.

By default runs offline only (no simulator).
Use --sim to send the trajectory to the Unity simulator.

Usage:
  py -m bc_approach.MLP.test_bc --exercise 1
  py -m bc_approach.MLP.test_bc --exercise 1 --runs 3
  py -m bc_approach.MLP.test_bc --exercise 1 --runs 3 --sim
  py -m bc_approach.MLP.test_bc --exercise 1 --runs 3 --sim --host 10.59.1.20

--- Input ---
  data/dataset/exercise_XXX/MLP/bc_model.pth
  data/dataset/exercise_XXX/MLP/scaler.pkl
  data/dataset/exercise_XXX/baseline.csv      (for n_steps)
  data/dataset/exercise_XXX/canonical.csv     (for start pose)

--- Output ---
  data/dataset/exercise_XXX/plot/bc_MLP.png
'''

import argparse
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

import torch
import torch.nn as nn

from utilities.config import DATA_ROOT, JOINT_COLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_JOINTS        = len(JOINT_COLS)
VELOCITY_LAG    = 5
VEL_CLIP        = 10.0
SIMULATOR_HOST  = 'localhost'
MIN_FRAME_DELAY = 0.033   # ~30 Hz
GOTO_DURATION   = 2.0


# ---------------------------------------------------------------------------
# Start pose
# ---------------------------------------------------------------------------
def _load_start_pose(exercise_dir: Path) -> dict:
    canonical_path = exercise_dir / 'canonical.csv'
    if canonical_path.exists():
        first_row = pd.read_csv(canonical_path).iloc[0]
        pose = {c: float(first_row[c]) for c in JOINT_COLS if c in first_row.index}
        if len(pose) == N_JOINTS:
            print('Start pose: loaded from canonical.csv')
            return pose
    print('Start pose: canonical.csv not found, using zeros')
    return {j: 0.0 for j in JOINT_COLS}


# ---------------------------------------------------------------------------
# FK geometry
# ---------------------------------------------------------------------------
_FK_JOINTS = [
    ('shoulder_pitch', None,                      np.array([0., 1., 0.])),
    ('shoulder_roll',  np.array([0., 0., 0.]),    np.array([1., 0., 0.])),
    ('arm_yaw',        np.array([0., 0., 0.]),    np.array([0., 0., 1.])),
    ('elbow_pitch',    np.array([0., 0., -0.28]), np.array([0., 1., 0.])),
    ('forearm_yaw',    np.array([0., 0., 0.]),    np.array([0., 0., 1.])),
    ('wrist_pitch',    np.array([0., 0., -0.25]), np.array([0., 1., 0.])),
    ('wrist_roll',     np.array([0., 0., -0.0325]), np.array([1., 0., 0.])),
]
_ELBOW_IDX  = 3
_WRIST_IDX  = 5
_SHOULDER_Y = {'right': -0.19, 'left': 0.19}


def _rot(axis, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x, y, z = axis
    return np.array([
        [c+x*x*(1-c),   x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s, c+y*y*(1-c),   y*z*(1-c)-x*s],
        [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)  ],
    ])


def fk(q4_deg, side):
    q7_rad = np.deg2rad(np.append(q4_deg, [0., 0., 0.]))
    T = np.eye(4)
    elbow_pos = wrist_pos = None
    for i, (_, trans, axis) in enumerate(_FK_JOINTS):
        if i == 0:
            trans = np.array([0., _SHOULDER_Y[side], 0.])
        T[:3, 3] += T[:3, :3] @ trans
        T[:3, :3] = T[:3, :3] @ _rot(axis, q7_rad[i])
        if i == _ELBOW_IDX: elbow_pos = T[:3, 3].copy()
        if i == _WRIST_IDX: wrist_pos = T[:3, 3].copy()
    return elbow_pos, wrist_pos


def compute_fk_trajectory(q_traj):
    T = len(q_traj)
    r_elbow = np.zeros((T, 3)); r_wrist = np.zeros((T, 3))
    l_elbow = np.zeros((T, 3)); l_wrist = np.zeros((T, 3))
    for t in range(T):
        r_elbow[t], r_wrist[t] = fk(q_traj[t, :4], 'right')
        l_elbow[t], l_wrist[t] = fk(q_traj[t, 4:], 'left')
    return {'elbow': r_elbow, 'wrist': r_wrist}, {'elbow': l_elbow, 'wrist': l_wrist}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BCPolicyMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256, n_layers=2):
        super().__init__()
        layers, in_f = [], input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_f, hidden_size), nn.ReLU()]
            in_f = hidden_size
        layers.append(nn.Linear(in_f, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_ensemble(model_dir: Path):
    """
    Loads all fold models and scalers for ensemble inference.
    Returns a list of (model, scaler) pairs — one per fold.
    Falls back to single bc_model.pth if no fold files found.
    """
    fold_idx = 0
    ensemble = []
    while True:
        model_path  = model_dir / f'bc_model_fold_{fold_idx}.pth'
        scaler_path = model_dir / f'scaler_fold_{fold_idx}.pkl'
        if not model_path.exists():
            break
        ckpt  = torch.load(model_path, map_location='cpu', weights_only=False)
        model = BCPolicyMLP(ckpt['state_dim'], ckpt['action_dim'],
                            ckpt['hidden_size'], ckpt['n_layers'])
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        ensemble.append((model, scaler))
        print(f'  Fold {fold_idx} loaded  val_loss={ckpt.get("val_loss", "N/A"):.6f}')
        fold_idx += 1

    if not ensemble:
        raise FileNotFoundError(
            f'No fold models found in {model_dir}. '
            f'Run train_bc.py first.')

    print(f'Ensemble: {len(ensemble)} fold(s) loaded\n')
    return ensemble


# ---------------------------------------------------------------------------
# Autoregressive loop (ensemble)
# ---------------------------------------------------------------------------
def run_bc_loop(ensemble, n_steps: int, start_pose: dict) -> np.ndarray:
    """
    Runs the autoregressive loop using an ensemble of MLP models.
    At each step, all K models predict a delta independently.
    The final delta is the mean across all K predictions.
    """
    start_q   = np.array([start_pose[c] for c in JOINT_COLS], dtype=np.float32)
    q_history = [start_q.copy() for _ in range(VELOCITY_LAG)]
    q_current = start_q.copy()
    q_traj    = np.zeros((n_steps, N_JOINTS), dtype=np.float32)

    for step in range(n_steps):
        velocity = np.clip(q_current - q_history[0], -VEL_CLIP, VEL_CLIP)
        state    = np.concatenate([q_current, velocity]).astype(np.float32)

        # Each model predicts with its own scaler
        deltas = []
        for model, scaler in ensemble:
            state_norm = scaler.transform([state])
            with torch.no_grad():
                delta = model(torch.tensor(state_norm, dtype=torch.float32)).numpy()[0]
            deltas.append(delta)

        # Ensemble: average of all K deltas
        delta_mean = np.mean(deltas, axis=0)

        q_new = q_current + delta_mean
        q_history.pop(0)
        q_history.append(q_current.copy())
        q_current    = q_new
        q_traj[step] = q_new

    return q_traj


# ---------------------------------------------------------------------------
# Simulator helpers (only used with --sim)
# ---------------------------------------------------------------------------
def _connect(host):
    from reachy_sdk import ReachySDK
    print(f'Connecting to Reachy at {host} ...')
    return ReachySDK(host=host)


def _get_joint(reachy, col):
    arm = reachy.r_arm if col.startswith('r_') else reachy.l_arm
    return getattr(arm, col)


def _goto_pose(reachy, pose, duration):
    from reachy_sdk.trajectory import goto
    from reachy_sdk.trajectory.interpolation import InterpolationMode
    goal = {_get_joint(reachy, col): val for col, val in pose.items()}
    goto(goal, duration=duration, interpolation_mode=InterpolationMode.MINIMUM_JERK)
    time.sleep(duration + 0.1)


def _send_frame(reachy, q):
    for i, col in enumerate(JOINT_COLS):
        _get_joint(reachy, col).goal_position = float(q[i])


def run_on_simulator(reachy, q_traj):
    n_steps = len(q_traj)
    print(f'  Sending {n_steps} frames to simulator ...')
    t_start = time.perf_counter()
    for step in range(n_steps):
        t_target = t_start + step * MIN_FRAME_DELAY
        _send_frame(reachy, q_traj[step])
        sleep_time = t_target + MIN_FRAME_DELAY - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_fk_trajectories(all_fk, exercise_num, output_path: Path):
    n_runs      = len(all_fk)
    colors      = cm.tab10(np.linspace(0, 0.9, n_runs))
    axes_labels = ['X (m)', 'Y (m)', 'Z (m)']
    alpha       = 0.8 if n_runs == 1 else 0.6
    y_pad       = 0.02  # metres of padding above/below shared Y range

    # Compute shared Y limits across all runs and both arms
    all_values = np.concatenate([
        np.concatenate([fk_data['wrist'], fk_data['elbow']])
        for r_fk, l_fk in all_fk
        for fk_data in [r_fk, l_fk]
    ]).flatten()
    all_values = all_values[np.isfinite(all_values)]
    y_min = float(all_values.min()) - y_pad
    y_max = float(all_values.max()) + y_pad

    fig, axs = plt.subplots(3, 2, figsize=(13, 10))
    fig.suptitle(
        f'FK trajectories — Exercise {exercise_num:03d}  [MLP]\n'
        f'(Reachy torso frame: X=forward, Y=left, Z=up)', fontsize=13)
    axs[0, 0].set_title('Right arm', fontsize=11)
    axs[0, 1].set_title('Left arm',  fontsize=11)

    for run_idx, (r_fk, l_fk) in enumerate(all_fk):
        lw = f'wrist run {run_idx+1}' if n_runs > 1 else 'wrist'
        le = f'elbow run {run_idx+1}' if n_runs > 1 else 'elbow'
        for ax_i in range(3):
            for col_i, fk_data in enumerate([r_fk, l_fk]):
                axs[ax_i, col_i].plot(fk_data['wrist'][:, ax_i],
                    color='#3498db', lw=1.5, alpha=alpha,
                    label=lw if ax_i == 0 else '_')
                axs[ax_i, col_i].plot(fk_data['elbow'][:, ax_i],
                    color='#e67e22', lw=1.5, alpha=alpha, linestyle='--',
                    label=le if ax_i == 0 else '_')

    for ax_i in range(3):
        for col_i in range(2):
            axs[ax_i, col_i].set_ylabel(axes_labels[ax_i])
            axs[ax_i, col_i].set_ylim(y_min, y_max)
            axs[ax_i, col_i].grid(True, alpha=0.3)
    for col_i in range(2):
        axs[2, col_i].set_xlabel('Frame')
        axs[0, col_i].legend(fontsize=8, loc='upper right')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Plot saved → {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Test BC MLP.')
    parser.add_argument('--exercise', type=int, required=True)
    parser.add_argument('--runs',     type=int, default=1)
    parser.add_argument('--steps',    type=int, default=None)
    parser.add_argument('--sim',      action='store_true',
                        help='Send trajectory to Unity simulator.')
    parser.add_argument('--host',     type=str, default=SIMULATOR_HOST)
    args = parser.parse_args()

    exercise_dir = DATA_ROOT / 'dataset' / f'exercise_{args.exercise:03d}'
    model_dir    = exercise_dir / 'MLP'
    plot_path    = exercise_dir / 'plots' / 'bc_MLP.png'

    ensemble   = load_ensemble(model_dir)
    start_pose = _load_start_pose(exercise_dir)

    if args.steps is not None:
        n_steps = args.steps
        print(f'Steps: {n_steps}  (from --steps)')
    elif (exercise_dir / 'canonical.csv').exists():
        n_steps = len(pd.read_csv(exercise_dir / 'canonical.csv'))
        print(f'Steps: {n_steps}  (from canonical.csv)')
    elif (exercise_dir / 'baseline.csv').exists():
        n_steps = len(pd.read_csv(exercise_dir / 'baseline.csv'))
        print(f'Steps: {n_steps}  (from baseline.csv)')
    else:
        n_steps = 300
        print(f'Steps: {n_steps}  (default)')

    reachy = None
    if args.sim:
        plt.close('all')  # release tkinter before SDK
        reachy = _connect(args.host)
        reachy.turn_on('r_arm')
        reachy.turn_on('l_arm')

    all_fk = []
    for run in range(1, args.runs + 1):
        print(f'\n--- Run {run} / {args.runs} ---')
        q_traj = run_bc_loop(ensemble, n_steps, start_pose)
        r_fk, l_fk = compute_fk_trajectory(q_traj)
        all_fk.append((r_fk, l_fk))

        if reachy is not None:
            _goto_pose(reachy, start_pose, GOTO_DURATION)
            run_on_simulator(reachy, q_traj)

    if reachy is not None:
        _goto_pose(reachy, start_pose, GOTO_DURATION)
        reachy.turn_off_smoothly('r_arm')
        reachy.turn_off_smoothly('l_arm')

    plot_fk_trajectories(all_fk, args.exercise, plot_path)


if __name__ == '__main__':
    main()