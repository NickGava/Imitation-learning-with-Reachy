'''
evaluate.py
=============================================================================
Evaluates the canonical trajectory and the BC model against the baseline for a single exercise.

Pipeline:
  1. Loads baseline.csv  (ground truth)
  2. Loads canonical.csv (DBA average of training demonstrations)
  3. Generates the BC trajectory via the autoregressive loop
  4. Loads all human demonstration joint_ik.csv files (to compute the degradation chain)
  5. Computes 5 metrics for each:
       - DTW distance: total alignment cost between two sequences, lower is better.
       - RMSE per joint: root mean squared error in degrees, computed after DTW alignment via warping path. 
                        Shows which joints are reproduced well and which are not.
       - Peak angle error per joint: |max(baseline_j) - max(generated_j)| per joint; measures full range of motion.
       - Pearson correlation per joint: temporal correlation per joint, independent of scale, higher is better (-1 to 1).
       - Smoothness: -mean(jerk^2) where jerk = second derivative of angles, higher is smoother.
  6. Saves results to CSV and PNG plots

Output:
  data/dataset/exercise_XXX/evaluation/
    results_summary.csv     <- one row per method, aggregated metrics
    results_per_joint.csv   <- RMSE and peak error for every joint x method
    plot_dtw.png            <- DTW distance bar chart + degradation chain
    plot_rmse_per_joint.png <- RMSE per joint, canonical vs BC
    plot_smoothness.png     <- smoothness comparison
    plot_velocity_profile.png <- velocity profile (naturalness)

Usage:
  python evaluate.py --exercise 1

  # use a specific number of BC steps (default: length of baseline)
  python evaluate.py --exercise 1 --steps 300
'''

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
from scipy.stats import pearsonr
from tslearn.metrics import dtw_path

from config import DATA_ROOT

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
JOINT_COLS = [
    'r_shoulder_pitch', 'r_shoulder_roll', 'r_arm_yaw',
    'r_elbow_pitch',    'r_forearm_yaw',   'r_wrist_pitch', 'r_wrist_roll', 'r_gripper',
    'l_shoulder_pitch', 'l_shoulder_roll', 'l_arm_yaw',
    'l_elbow_pitch',    'l_forearm_yaw',   'l_wrist_pitch', 'l_wrist_roll', 'l_gripper',
]
HEAD_COLS   = ['head_x', 'head_y', 'head_z']
HEAD_NEUTRAL = np.array([1.0, 0.0, 0.15], dtype=np.float32)

REST_POSE = np.array([
     0., -5.,  0., -90., 0., 0., 0., 0.,
     0.,  5.,  0., -90., 0., 0., 0., 0.,
], dtype=np.float32)

# Short labels for plots
JOINT_LABELS = [
    'r_sh_p', 'r_sh_r', 'r_aw',
    'r_el_p', 'r_fw_y', 'r_wr_p', 'r_wr_r', 'r_gr',
    'l_sh_p', 'l_sh_r', 'l_aw',
    'l_el_p', 'l_fw_y', 'l_wr_p', 'l_wr_r', 'l_gr',
]


# ---------------------------------------------------------------------------
# BCModel
# ---------------------------------------------------------------------------
class BCModel(nn.Module):
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


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------
def compute_metrics(generated: np.ndarray, reference: np.ndarray, label: str = '') -> Dict:
    '''
    Computes all metrics between a generated and a reference trajectory.

    Parameters:
        generated : (N, 16) array - trajectory to evaluate
        reference : (M, 16) array - ground truth (baseline)
        label     : name used for logging

    Returns:
        dict with keys:
            dtw_distance, rmse_mean, rmse_per_joint,
            peak_error_mean, peak_error_per_joint,
            pearson_mean, pearson_per_joint,
            smoothness
    '''
    # DTW distance + warping path
    path, dtw_dist = dtw_path(generated, reference)

    # Align sequences via warping path
    gen_aligned = generated[[p[0] for p in path]]   # shape: (L, 16)
    ref_aligned = reference[[p[1] for p in path]]   # shape: (L, 16)

    # RMSE per joint
    rmse_per_joint = np.sqrt(np.mean((gen_aligned - ref_aligned) ** 2, axis=0))
    rmse_mean      = float(np.mean(rmse_per_joint))

    # Peak angle error per joint
    peak_error_per_joint = np.abs(np.max(reference, axis=0) - np.max(generated, axis=0))
    peak_error_mean = float(np.mean(peak_error_per_joint))

    # Pearson correlation per joint
    # Use aligned sequences for fair comparison
    pearson_per_joint = np.array([
        pearsonr(gen_aligned[:, j], ref_aligned[:, j])[0]
        for j in range(generated.shape[1])
    ])
    # Replace NaN (constant signal) with 0
    pearson_per_joint = np.nan_to_num(pearson_per_joint, nan=0.0)
    pearson_mean = float(np.mean(pearson_per_joint))

    # Smoothness (intrinsic, not vs baseline)
    if len(generated) > 2:
        jerk       = np.diff(generated, n=2, axis=0)
        smoothness = float(-np.mean(jerk ** 2))
    else:
        smoothness = float('nan')

    if label:
        print(f'  [{label}]  DTW={dtw_dist:.4f}  '
              f'RMSE={rmse_mean:.2f}°  '
              f'PeakErr={peak_error_mean:.2f}°  '
              f'Pearson={pearson_mean:.3f}  '
              f'Smooth={smoothness:.4f}')

    return {
        'dtw_distance'       : float(dtw_dist),
        'rmse_mean'          : rmse_mean,
        'rmse_per_joint'     : rmse_per_joint,
        'peak_error_mean'    : peak_error_mean,
        'peak_error_per_joint': peak_error_per_joint,
        'pearson_mean'       : pearson_mean,
        'pearson_per_joint'  : pearson_per_joint,
        'smoothness'         : smoothness,
    }


# ---------------------------------------------------------------------------
# BC trajectory generation (offline)
# ---------------------------------------------------------------------------
def generate_bc_trajectory(model_path: Path, scaler_path: Path, n_steps: int) -> Optional[np.ndarray]:
    '''
    Runs the BC autoregressive loop offline and returns the trajectory.

    Parameters:
        model_path  : path to bc_model.pth
        scaler_path : path to scaler.pkl
        n_steps     : number of steps to generate

    Returns:
        (n_steps+1, 16) array of joint angles, or None if model not found.
    '''
    if not model_path.exists():
        print(f'  [BC] bc_model.pth not found -> {model_path}')
        return None
    if not scaler_path.exists():
        print(f'  [BC] scaler.pkl not found -> {scaler_path}')
        return None

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = BCModel(checkpoint['state_dim'], checkpoint['action_dim'], checkpoint['hidden_size'], checkpoint['n_layers'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    state_dim  = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']
    head_in_action = action_dim > len(JOINT_COLS)

    # Initial state
    state = np.zeros(state_dim, dtype=np.float32)
    state[:16]   = REST_POSE
    state[16:32] = 0.0
    if state_dim > 32:
        state[32:35] = HEAD_NEUTRAL

    q_current   = REST_POSE.copy()
    head_current = HEAD_NEUTRAL.copy()
    trajectory  = [q_current.copy()]

    for _ in range(n_steps):
        state_norm = scaler.transform([state])
        x_t        = torch.tensor(state_norm, dtype=torch.float32)

        with torch.no_grad():
            delta = model(x_t).numpy()[0]

        q_new = q_current + delta[:16]

        if head_in_action and len(delta) >= 19:
            head_current = head_current + delta[16:19]

        state[:16]   = q_new
        state[16:32] = q_new - q_current
        if state_dim > 32:
            state[32:35] = head_current

        q_current = q_new
        trajectory.append(q_new.copy())

    print(f'  [BC] Generated {len(trajectory)} frames '
          f'(state_dim={state_dim}, action_dim={action_dim})')
    return np.array(trajectory)   # shape: (n_steps+1, 16)


# ---------------------------------------------------------------------------
# Load human demonstrations
# ---------------------------------------------------------------------------
def load_human_demos(landmarks_root: Path, exercise_num: int) -> List[np.ndarray]:
    '''
    Loads all joint_ik.csv files for the given exercise.
    Returns a list of (N_i, 16) arrays.
    '''
    exercise_name = f'exercise_{exercise_num:03d}'
    demos = []

    for subj_dir in sorted(landmarks_root.glob('subject_*')):
        exer_dir = subj_dir / exercise_name
        if not exer_dir.is_dir():
            continue
        for video_dir in sorted(exer_dir.glob('video_*')):
            ik_path = video_dir / 'joint_ik.csv'
            if not ik_path.exists():
                continue
            df = pd.read_csv(ik_path).dropna(subset=JOINT_COLS)
            if len(df) >= 2:
                demos.append(df[JOINT_COLS].values.astype(float))

    print(f'  Human demos loaded: {len(demos)}')
    return demos


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_dtw_summary(results: Dict, output_dir: Path) -> None:
    '''Degradation chain bar chart.'''
    methods = list(results.keys())
    dtw_vals = [results[m]['dtw_distance'] for m in methods]
    colors = ['#2ecc71', '#e67e22', '#3498db', '#e74c3c'][:len(methods)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, dtw_vals, color=colors, edgecolor='white', linewidth=1.5)

    # Value labels on bars
    for bar, val in zip(bars, dtw_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('DTW distance (lower is better)', fontsize=11)
    ax.set_title('DTW Distance vs Baseline - Degradation Chain', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(dtw_vals) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = output_dir / 'plot_dtw.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {path}')


def _plot_rmse_per_joint(results: Dict, output_dir: Path) -> None:
    '''RMSE per joint side-by-side bar chart for canonical and BC.'''
    methods_to_plot = {k: v for k, v in results.items()
                       if k in ('Canonical', 'BC')}
    if not methods_to_plot:
        return

    x     = np.arange(len(JOINT_LABELS))
    width = 0.35
    colors = {'Canonical': '#e67e22', 'BC': '#3498db'}

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (method, res) in enumerate(methods_to_plot.items()):
        offset = (i - 0.5) * width
        ax.bar(x + offset, res['rmse_per_joint'], width,
               label=method, color=colors.get(method, 'grey'),
               edgecolor='white', linewidth=0.8, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_LABELS, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('RMSE (degrees)', fontsize=11)
    ax.set_title('RMSE per Joint vs Baseline', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = output_dir / 'plot_rmse_per_joint.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {path}')


def _plot_smoothness(results: Dict, output_dir: Path) -> None:
    '''Smoothness bar chart.'''
    methods   = list(results.keys())
    smooth    = [results[m]['smoothness'] for m in methods]
    colors    = ['#2ecc71', '#e67e22', '#3498db', '#e74c3c'][:len(methods)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, smooth, color=colors, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, smooth):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(min(smooth)) * 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Smoothness - neg. mean squared jerk (higher is smoother)', fontsize=10)
    ax.set_title('Smoothness (intrinsic - not vs baseline)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = output_dir / 'plot_smoothness.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {path}')


def _plot_velocity_profile(baseline: np.ndarray, canonical: Optional[np.ndarray], bc: Optional[np.ndarray], output_dir: Path, joint_idx: int = 0) -> None:
    '''
    Velocity profile (first derivative) for one joint.
    Shows naturalness - bell-shaped = human-like motion.
    Not a comparison vs baseline - an intrinsic quality measure.
    '''
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    joint_name = JOINT_LABELS[joint_idx]

    for ax, (traj, label, color) in zip(axes, [
        (baseline,  'Baseline',  '#2ecc71'),
        (canonical, 'Canonical', '#e67e22'),
        (bc,        'BC',        '#3498db'),
    ]):
        if traj is None or len(traj) < 2:
            ax.set_title(f'{label}\n(not available)')
            ax.set_visible(True)
            continue
        vel = np.diff(traj[:, joint_idx])
        ax.plot(vel, color=color, linewidth=1.5)
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Frame', fontsize=9)
        ax.set_ylabel('Angular velocity (°/frame)', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f'Velocity Profile - {joint_name}\n'
        f'(bell-shaped = natural human motion; intrinsic measure, not vs baseline)',
        fontsize=11, fontweight='bold'
    )
    fig.tight_layout()
    path = output_dir / 'plot_velocity_profile.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {path}')


# ---------------------------------------------------------------------------
# Save CSVs
# ---------------------------------------------------------------------------
def _save_results(results: Dict, output_dir: Path) -> None:
    '''Saves summary and per-joint CSVs.'''

    # Summary: one row per method
    summary_rows = []
    for method, res in results.items():
        summary_rows.append({
            'method'         : method,
            'dtw_distance'   : res['dtw_distance'],
            'rmse_mean_deg'  : res['rmse_mean'],
            'peak_error_mean': res['peak_error_mean'],
            'pearson_mean'   : res['pearson_mean'],
            'smoothness'     : res['smoothness'],
        })
    df_summary = pd.DataFrame(summary_rows)
    path_summary = output_dir / 'results_summary.csv'
    df_summary.to_csv(path_summary, index=False)
    print(f'  Saved -> {path_summary}')

    # Per-joint: RMSE and peak error for each method x joint
    per_joint_rows = []
    for method, res in results.items():
        for j, jname in enumerate(JOINT_COLS):
            per_joint_rows.append({
                'method'     : method,
                'joint'      : jname,
                'rmse_deg'   : res['rmse_per_joint'][j],
                'peak_error' : res['peak_error_per_joint'][j],
                'pearson'    : res['pearson_per_joint'][j],
            })
    df_joints = pd.DataFrame(per_joint_rows)
    path_joints = output_dir / 'results_per_joint.csv'
    df_joints.to_csv(path_joints, index=False)
    print(f'  Saved -> {path_joints}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate canonical and BC against baseline for one exercise.'
    )
    parser.add_argument(
        '--exercise', type=int, required=True,
        help='Exercise number to evaluate.'
    )
    parser.add_argument(
        '--steps', type=int, default=None,
        help='Number of BC autoregressive steps. Default: length of baseline.csv.'
    )
    args = parser.parse_args()

    exercise_name  = f'exercise_{args.exercise:03d}'
    dataset_dir    = DATA_ROOT / 'dataset' / exercise_name
    landmarks_root = DATA_ROOT / 'landmarks'
    output_dir     = dataset_dir / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  Evaluation - Exercise {args.exercise:03d}')
    print(f'{"="*60}')

    # Load baseline
    baseline_path = dataset_dir / 'baseline.csv'
    if not baseline_path.exists():
        print(f'Error: baseline.csv not found -> {baseline_path}')
        return
    baseline = pd.read_csv(baseline_path)[JOINT_COLS].dropna().values.astype(float)
    print(f'\nBaseline loaded: {len(baseline)} frames')

    # Determine BC steps
    n_steps = args.steps if args.steps is not None else len(baseline)
    print(f'BC steps: {n_steps}')

    # Load canonical
    canonical_path = dataset_dir / 'canonical.csv'
    canonical = None
    if canonical_path.exists():
        canonical = pd.read_csv(canonical_path)[JOINT_COLS].dropna().values.astype(float)
        print(f'Canonical loaded: {len(canonical)} frames')
    else:
        print('[WARNING] canonical.csv not found - skipping canonical evaluation.')

    # Generate BC trajectory
    print('\nGenerating BC trajectory ...')
    bc_traj = generate_bc_trajectory(
        model_path  = dataset_dir / 'bc_model.pth',
        scaler_path = dataset_dir / 'scaler.pkl',
        n_steps     = n_steps,
    )

    # Load human demos
    print('\nLoading human demonstrations ...')
    human_demos = load_human_demos(landmarks_root, args.exercise)

    # Compute metrics
    print('\nComputing metrics ...')
    results = {}

    if human_demos:
        # Aggregate human demos: mean DTW, mean RMSE etc. over all demos
        human_metrics_list = [
            compute_metrics(demo, baseline) for demo in human_demos
        ]
        # Average scalar metrics; per-joint arrays averaged too
        results['Human copies'] = {
            'dtw_distance'         : float(np.mean([m['dtw_distance'] for m in human_metrics_list])),
            'rmse_mean'            : float(np.mean([m['rmse_mean'] for m in human_metrics_list])),
            'rmse_per_joint'       : np.mean([m['rmse_per_joint'] for m in human_metrics_list], axis=0),
            'peak_error_mean'      : float(np.mean([m['peak_error_mean'] for m in human_metrics_list])),
            'peak_error_per_joint' : np.mean([m['peak_error_per_joint'] for m in human_metrics_list], axis=0),
            'pearson_mean'         : float(np.mean([m['pearson_mean'] for m in human_metrics_list])),
            'pearson_per_joint'    : np.mean([m['pearson_per_joint'] for m in human_metrics_list], axis=0),
            'smoothness'           : float(np.mean([m['smoothness'] for m in human_metrics_list])),
        }
        print(f"  [Human copies]  DTW={results['Human copies']['dtw_distance']:.4f}  "
              f"RMSE={results['Human copies']['rmse_mean']:.2f}°  "
              f"Smooth={results['Human copies']['smoothness']:.4f}")

    if canonical is not None:
        results['Canonical'] = compute_metrics(canonical, baseline, label='Canonical')

    if bc_traj is not None:
        results['BC'] = compute_metrics(bc_traj, baseline, label='BC')

    if not results:
        print('No results to save - check that canonical.csv and bc_model.pth exist.')
        return

    # Save CSVs
    print('\nSaving results ...')
    _save_results(results, output_dir)

    # Plots
    print('\nGenerating plots ...')
    _plot_dtw_summary(results, output_dir)
    _plot_rmse_per_joint(results, output_dir)
    _plot_smoothness(results, output_dir)
    _plot_velocity_profile(
        baseline  = baseline,
        canonical = canonical,
        bc        = bc_traj,
        output_dir = output_dir,
        joint_idx  = 0,   # r_shoulder_pitch - most representative joint
    )

    # Print summary table
    print(f'\n{"="*80}')
    print(f'  SUMMARY - Exercise {args.exercise:03d}')
    print(f'{"="*80}')
    print(f'  {"Method":<16}  {"DTW":>8}  {"RMSE(°)":>8}  '
          f'{"Peak(°)":>8}  {"Pearson":>8}  {"Smooth":>10}')
    print(f'  {"-"*64}')
    for method, res in results.items():
        print(f'  {method:<16}  '
              f'{res["dtw_distance"]:>8.4f}  '
              f'{res["rmse_mean"]:>8.2f}  '
              f'{res["peak_error_mean"]:>8.2f}  '
              f'{res["pearson_mean"]:>8.3f}  '
              f'{res["smoothness"]:>10.4f}')
    print(f'{"="*80}')
    print(f'\nAll outputs saved to -> {output_dir}')
    print('Done.')


if __name__ == '__main__':
    main()