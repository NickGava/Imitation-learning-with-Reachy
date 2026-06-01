'''
GRU/test_bc.py
=============================================================================
Runs the BC GRU autoregressive loop for a given exercise and plots the
FK trajectories for both arms.

The GRU maintains a rolling history of the last SEQ_LEN joint positions.
At each step, the history is normalized and fed to the GRU; the predicted
delta is applied to get the next joint position, which is added to history.

By default runs offline only (no simulator).
Use --sim to send the trajectory to the Unity simulator.

Usage:
  py -m bc_approach.GRU.test_bc --exercise 1
  py -m bc_approach.GRU.test_bc --exercise 1 --runs 3
  py -m bc_approach.GRU.test_bc --exercise 1 --runs 3 --sim
  py -m bc_approach.GRU.test_bc --exercise 1 --runs 3 --sim --host 10.59.1.20

--- Input ---
  data/dataset/exercise_XXX/GRU/bc_model.pth
  data/dataset/exercise_XXX/GRU/scaler.pkl
  data/dataset/exercise_XXX/baseline.csv      (for n_steps)
  data/dataset/exercise_XXX/canonical.csv     (for start pose)

--- Output ---
  data/dataset/exercise_XXX/plot/bc_GRU.png
'''

import argparse
import pickle
import time
import collections
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
from utilities.split_utils import split_name

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_JOINTS        = len(JOINT_COLS)
N_INPUT         = N_JOINTS * 2   # [q, dq] per timestep -> 32
SIMULATOR_HOST  = 'localhost'
MIN_FRAME_DELAY = 0.033
GOTO_DURATION   = 2.0

# Heuristic stopping criterion: stop when both wrists are within
# STOP_THRESHOLD_M metres of their start FK positions for STOP_WINDOW
# consecutive frames, but only after having first moved away from it.
STOP_THRESHOLD_M = 0.20   # metres
STOP_WINDOW      = 40


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
    ('shoulder_pitch', None,                        np.array([0., 1., 0.])),
    ('shoulder_roll',  np.array([0., 0., 0.]),      np.array([1., 0., 0.])),
    ('arm_yaw',        np.array([0., 0., 0.]),      np.array([0., 0., 1.])),
    ('elbow_pitch',    np.array([0., 0., -0.28]),   np.array([0., 1., 0.])),
    ('forearm_yaw',    np.array([0., 0., 0.]),      np.array([0., 0., 1.])),
    ('wrist_pitch',    np.array([0., 0., -0.25]),   np.array([0., 1., 0.])),
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
        r_elbow[t], r_wrist[t] = fk(q_traj[t, :4],  'right')
        l_elbow[t], l_wrist[t] = fk(q_traj[t, 8:12], 'left')
    return {'elbow': r_elbow, 'wrist': r_wrist}, {'elbow': l_elbow, 'wrist': l_wrist}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BCPolicyGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super().__init__()
        self.gru  = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def load_ensemble(model_dir: Path):
    """
    Loads all fold models and scalers for ensemble inference.
    Returns a list of (model, scaler, seq_len) tuples - one per fold.
    """
    fold_idx = 0
    ensemble = []
    while True:
        model_path  = model_dir / f'bc_model_fold_{fold_idx}.pth'
        scaler_path = model_dir / f'scaler_fold_{fold_idx}.pkl'
        if not model_path.exists():
            break
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        model = BCPolicyGRU(
            ckpt['input_dim'], ckpt['hidden_size'],
            ckpt['output_dim'], ckpt['n_layers']
        )
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        seq_len = ckpt['seq_len']
        ensemble.append((model, scaler, seq_len))
        print(f'  Fold {fold_idx} loaded  seq_len={seq_len}  val_loss={ckpt.get("val_loss", "N/A"):.6f}')
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
def run_bc_loop(ensemble, n_steps: int, start_pose: dict, stop_threshold: float = STOP_THRESHOLD_M, stop_window:    int   = STOP_WINDOW) -> np.ndarray:
    """
    Runs the autoregressive loop using an ensemble of GRU models.

    Each timestep in the window contains [q, dq] (32 features).
    dq at position i in the window is computed as q[i] - q[i - VELOCITY_LAG],
    mirroring the computation in build_dataset.py.

    A longer history of SEQ_LEN + VELOCITY_LAG frames is maintained so that
    dq can always be computed for every frame in the window.

    Heuristic stopping criterion (disabled if stop_threshold is None):
      Two-phase state machine:
        1. DEPARTED  - waits until at least one wrist moves more than
                       stop_threshold metres away from its start FK position.
        2. RETURNED  - once departed, counts consecutive frames where both
                       wrists are within stop_threshold of the start position.
                       Stops after stop_window such frames.
    """
    VELOCITY_LAG = 5
    VEL_CLIP     = 10.0

    start_q  = np.array([start_pose[c] for c in JOINT_COLS], dtype=np.float32)
    seq_len  = ensemble[0][2]
    hist_len = seq_len + VELOCITY_LAG

    history = collections.deque(
        [start_q.copy() for _ in range(hist_len)], maxlen=hist_len
    )
    q_traj      = np.zeros((n_steps, N_JOINTS), dtype=np.float32)
    departed    = False
    consecutive = 0

    # Pre-compute wrist FK positions at start pose
    if stop_threshold is not None:
        _, r_wrist_start = fk(start_q[0:4],  'right')
        _, l_wrist_start = fk(start_q[8:12], 'left')

    for step in range(n_steps):
        hist_arr = np.array(history, dtype=np.float32)

        window = np.zeros((seq_len, N_INPUT), dtype=np.float32)
        for i in range(seq_len):
            t  = VELOCITY_LAG + i
            q  = hist_arr[t]
            dq = np.clip(hist_arr[t] - hist_arr[t - VELOCITY_LAG], -VEL_CLIP, VEL_CLIP)
            window[i] = np.concatenate([q, dq])

        deltas = []
        for model, scaler, _ in ensemble:
            window_norm = scaler.transform(window).astype(np.float32)
            x = torch.tensor(window_norm[np.newaxis])
            with torch.no_grad():
                delta = model(x).numpy()[0]
            deltas.append(delta)

        delta_mean = np.mean(deltas, axis=0)
        q_new = history[-1] + delta_mean
        history.append(q_new)
        q_traj[step] = q_new

        # --- Heuristic stopping criterion (FK-based, departed -> returned) ---
        if stop_threshold is not None:
            _, r_wrist_cur = fk(q_new[0:4],  'right')
            _, l_wrist_cur = fk(q_new[8:12], 'left')
            dist_r = float(np.linalg.norm(r_wrist_cur - r_wrist_start))
            dist_l = float(np.linalg.norm(l_wrist_cur - l_wrist_start))
            dist   = max(dist_r, dist_l)

            if not departed:
                if dist > stop_threshold:
                    departed = True
            else:
                if dist < stop_threshold:
                    consecutive += 1
                    if consecutive >= stop_window:
                        actual_steps = step + 1
                        print(f'  [STOP] Exercise end detected at step {actual_steps} '
                              f'(wrist dist R={dist_r:.3f}m  L={dist_l:.3f}m, '
                              f'threshold={stop_threshold}m)')
                        return q_traj[:actual_steps]
                else:
                    consecutive = 0

    return q_traj



# ---------------------------------------------------------------------------
# Joint angles plot
# ---------------------------------------------------------------------------
import matplotlib.ticker as ticker
from utilities.config import JOINT_LIMITS_DEG

RIGHT_JOINTS = [
    ('r_shoulder_pitch', 0, 'right'),
    ('r_shoulder_roll',  1, 'right'),
    ('r_arm_yaw',        2, 'right'),
    ('r_elbow_pitch',    3, 'right'),
]
LEFT_JOINTS = [
    ('l_shoulder_pitch', 0, 'left'),
    ('l_shoulder_roll',  1, 'left'),
    ('l_arm_yaw',        2, 'left'),
    ('l_elbow_pitch',    3, 'left'),
]
JOINT_LABELS = {
    'r_shoulder_pitch': 'Shoulder Pitch', 'r_shoulder_roll': 'Shoulder Roll',
    'r_arm_yaw':        'Arm Yaw',        'r_elbow_pitch':   'Elbow Pitch',
    'l_shoulder_pitch': 'Shoulder Pitch', 'l_shoulder_roll': 'Shoulder Roll',
    'l_arm_yaw':        'Arm Yaw',        'l_elbow_pitch':   'Elbow Pitch',
}
LINE_COLOR = '#ff7f0e'
J_PADDING  = 5.0


def plot_joints_trajectory(q_traj: np.ndarray, exercise_num: int, output_path: Path) -> None:
    '''
    Plots joint angles of a BC trajectory using the same layout as plot_joints.py.
    Saves to output_path (typically plots_joints/joints_GRU.png).
    '''
    df     = pd.DataFrame(q_traj, columns=JOINT_COLS)
    frames = np.arange(len(df))

    n_rows = max(len(RIGHT_JOINTS), len(LEFT_JOINTS))
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(14, 2.8 * n_rows), sharex=False)
    if n_rows == 1:
        axes = np.array([axes])

    fig.suptitle(
        f'Joint angles (GRU) - exercise_{exercise_num:03d}\n'
        f'(Y scale = robot joint limits)',
        fontsize=13, fontweight='bold', y=1.01,
    )
    axes[0, 0].set_title('Right arm', fontsize=12, fontweight='bold', pad=8)
    axes[0, 1].set_title('Left arm',  fontsize=12, fontweight='bold', pad=8)

    for row_idx in range(n_rows):
        ax_r = axes[row_idx, 0]
        if row_idx < len(RIGHT_JOINTS):
            col, lim_idx, side = RIGHT_JOINTS[row_idx]
            if col in df.columns:
                ax_r.plot(frames, df[col].values, color=LINE_COLOR, linewidth=1.5)
            ax_r.set_ylabel(f'{JOINT_LABELS.get(col, col)}\n(deg)', fontsize=9)
            y_min, y_max = JOINT_LIMITS_DEG[side][lim_idx]
            ax_r.set_ylim(y_min - J_PADDING, y_max + J_PADDING)
        else:
            ax_r.set_visible(False)

        ax_l = axes[row_idx, 1]
        if row_idx < len(LEFT_JOINTS):
            col, lim_idx, side = LEFT_JOINTS[row_idx]
            if col in df.columns:
                ax_l.plot(frames, df[col].values, color=LINE_COLOR, linewidth=1.5)
            y_min, y_max = JOINT_LIMITS_DEG[side][lim_idx]
            ax_l.set_ylim(y_min - J_PADDING, y_max + J_PADDING)
        else:
            ax_l.set_visible(False)

        for ax in (ax_r, ax_l):
            if ax.get_visible():
                ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
                ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                ax.grid(True, which='minor', linestyle=':',  linewidth=0.4, alpha=0.4)
                ax.tick_params(axis='both', labelsize=8)
                if row_idx == n_rows - 1:
                    ax.set_xlabel('Frame', fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot saved -> {output_path}')

# ---------------------------------------------------------------------------
# Simulator helpers
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
        f'FK trajectories - Exercise {exercise_num:03d}  [GRU]\n'
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
    print(f'Plot saved -> {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Test BC GRU.')
    parser.add_argument('--exercise', type=int, required=True)
    parser.add_argument('--runs',     type=int, default=1)
    parser.add_argument('--steps',    type=int, default=None)
    parser.add_argument('--sim',      action='store_true', help='Send trajectory to Unity simulator.')
    parser.add_argument('--host',     type=str, default=SIMULATOR_HOST)
    parser.add_argument('--no-stop',  action='store_true', help='Disable heuristic stopping criterion (run for full n_steps).')
    parser.add_argument('--stop-threshold', type=float, default=STOP_THRESHOLD_M, metavar='M', help=f'Wrist distance threshold in metres (default: {STOP_THRESHOLD_M}).')
    parser.add_argument('--stop-window', type=int, default=STOP_WINDOW, metavar='N', help=f'Consecutive frames below threshold before stopping (default: {STOP_WINDOW}).')
    parser.add_argument('--n-demos', type=int, default=55, choices=[10,25,55])
    parser.add_argument('--run', type=int, default=None, help='Training run index. If set, loads from ARCH/run_N/.')

    args = parser.parse_args()

    exercise_dir = DATA_ROOT / 'dataset' / f'exercise_{args.exercise:03d}'
    split_dir    = exercise_dir / split_name(args.n_demos)
    _base_dir    = split_dir / 'GRU'           # o GRU / Transformer
    model_dir = _base_dir / f'run_{args.run}' if args.run is not None else _base_dir
    plot_path    = split_dir / 'plots' / 'bc_GRU.png'

    ensemble   = load_ensemble(model_dir)
    start_pose   = _load_start_pose(split_dir) # canonical da split_dir

    n_steps = 1000

    reachy = None
    if args.sim:
        plt.close('all')
        reachy = _connect(args.host)
        reachy.turn_on('r_arm')
        reachy.turn_on('l_arm')

    stop_threshold = None if args.no_stop else args.stop_threshold

    all_fk = []
    for run in range(1, args.runs + 1):
        print(f'\n--- Run {run} / {args.runs} ---')
        q_traj = run_bc_loop(ensemble, n_steps, start_pose,
                             stop_threshold = stop_threshold,
                             stop_window    = args.stop_window)
        r_fk, l_fk = compute_fk_trajectory(q_traj)
        all_fk.append((r_fk, l_fk))

        if run == 1:
            run_sfx     = f'_run{args.run}' if args.run is not None else ''
            joints_path = split_dir / 'plots_joints' / f'joints_GRU{run_sfx}.png'
            plot_joints_trajectory(q_traj, args.exercise, joints_path)
            traj_path = model_dir / 'bc_trajectory.csv'
            pd.DataFrame(q_traj, columns=JOINT_COLS).to_csv(traj_path, index=False)
            print(f'Trajectory saved -> {traj_path}')

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