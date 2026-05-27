'''
_metrics.py
=============================================================================
Computation of evaluation metrics in joint space and Cartesian space.

FK is imported from data_acquisition.run_ik - single source of truth
across the project.

Public functions:
    compute_metrics(generated, reference)           -> joint-space metrics dict
    compute_cartesian_metrics(generated, reference) -> Cartesian metrics dict
    aggregate_metrics(metrics_list)                 -> average over multiple sequences
    print_summary(results, title)                   -> formatted terminal table
'''

from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr
from tslearn.metrics import dtw_path

from data_acquisition.run_ik import fk
from evaluation_and_comparison._config import ACTIVE_IDX, N_JOINTS


def _safe_pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    '''Pearson correlation with preventive check on constant arrays.'''
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return float(pearsonr(a, b)[0])


# ============================================================================
# Helper FK: (T, 16) degrees -> Cartesian positions di polso e gomito
# ============================================================================

def _to_cartesian(q_deg: np.ndarray):
    '''
    Applies FK to each frame of a joint-space trajectory.

    Parametres
    ----------
    q_deg : (T, 16) array in degrees, order = JOINT_COLS

    Returns
    -------
    r_wrist, r_elbow, l_wrist, l_elbow : each (T, 3) in meters
    '''
    T = len(q_deg)
    r_wrist = np.zeros((T, 3))
    r_elbow = np.zeros((T, 3))
    l_wrist = np.zeros((T, 3))
    l_elbow = np.zeros((T, 3))
    for t in range(T):
        # Indices 0-3 = right arm (shoulder_pitch/roll, arm_yaw, elbow_pitch)
        # Indices 8-11 = left arm - pad 3 zeros for forearm_yaw/wrist_pitch/wrist_roll
        q_r = np.deg2rad(np.pad(q_deg[t, :4],   (0, 3)))
        q_l = np.deg2rad(np.pad(q_deg[t, 8:12], (0, 3)))
        r_elbow[t], r_wrist[t] = fk(q_r, 'right')
        l_elbow[t], l_wrist[t] = fk(q_l, 'left')
    return r_wrist, r_elbow, l_wrist, l_elbow


# ============================================================================
# Metrics joint-space
# ============================================================================

def compute_metrics(generated: np.ndarray, reference: np.ndarray, label: str = '') -> Dict:
    '''
    Compute joint-space metrics between the generated and reference trajectories.

    Scalar averages are computed over ACTIVE_IDX only (8 mobile joints).
    Per-joint versions cover all 16 joints.

    Parameters
    ----------
    generated : (T, 16) array in degrees
    reference : (T_ref, 16) array in degrees
    label     : string for terminal output

    Returns
    -------
    dict with: dtw_distance, rmse_mean, rmse_per_joint,
               peak_error_mean, peak_error_per_joint,
               pearson_mean, pearson_per_joint, smoothness
    '''
    path, dtw_dist = dtw_path(generated, reference)
    gen_al = generated[[p[0] for p in path]]
    ref_al = reference[[p[1] for p in path]]

    # RMSE per joint (post-allignment DTW)
    rmse_pj   = np.sqrt(np.mean((gen_al - ref_al) ** 2, axis=0))
    rmse_mean = float(np.mean(rmse_pj[ACTIVE_IDX]))

    # Peak angle error
    peak_pj   = np.abs(np.max(reference, axis=0) - np.max(generated, axis=0))
    peak_mean = float(np.mean(peak_pj[ACTIVE_IDX]))

    # Pearson per joint
    pearson_pj = np.array([
        _safe_pearsonr(gen_al[:, j], ref_al[:, j])
        for j in range(N_JOINTS)
    ])
    pearson_pj   = np.nan_to_num(pearson_pj, nan=0.0)
    pearson_mean = float(np.mean(pearson_pj[ACTIVE_IDX]))

    # Smoothness: −mean(jerk²)
    if len(generated) > 2:
        jerk       = np.diff(generated[:, ACTIVE_IDX], n=2, axis=0)
        smoothness = float(-np.mean(jerk ** 2))
    else:
        smoothness = float('nan')

    if label:
        print(f'  [{label:<16}]  '
              f'DTW={dtw_dist:>9.2f}°  RMSE={rmse_mean:>6.2f}°  '
              f'r={pearson_mean:>6.3f}  Smooth={smoothness:>10.4f}')

    return {
        'dtw_distance'        : float(dtw_dist),
        'rmse_mean'           : rmse_mean,
        'rmse_per_joint'      : rmse_pj,
        'peak_error_mean'     : peak_mean,
        'peak_error_per_joint': peak_pj,
        'pearson_mean'        : pearson_mean,
        'pearson_per_joint'   : pearson_pj,
        'smoothness'          : smoothness,
    }


# ============================================================================
# Cartesian metrics
# ============================================================================

def compute_cartesian_metrics(generated: np.ndarray, reference: np.ndarray, active_side: str = 'both', label: str = '') -> Dict:
    '''
    Compute cartesian metrics only for the active arm.

    active_side : 'left' | 'right' | 'both'
        Determines which arm to consider. Metrics for the inactive arm
        are not computed (NaN). The DTW uses only the active arm (6D).
    '''
    nan = float('nan')

    def _rmse3d(a, b):
        return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))

    # FK
    r_wrist_g, r_elbow_g, l_wrist_g, l_elbow_g = _to_cartesian(generated)
    r_wrist_r, r_elbow_r, l_wrist_r, l_elbow_r = _to_cartesian(reference)

    # DTW only for the active arm (6D: wrist + elbow)
    if active_side == 'left':
        traj_g = np.hstack([l_wrist_g, l_elbow_g])
        traj_r = np.hstack([l_wrist_r, l_elbow_r])
    elif active_side == 'right':
        traj_g = np.hstack([r_wrist_g, r_elbow_g])
        traj_r = np.hstack([r_wrist_r, r_elbow_r])
    else:
        traj_g = np.hstack([r_wrist_g, r_elbow_g, l_wrist_g, l_elbow_g])
        traj_r = np.hstack([r_wrist_r, r_elbow_r, l_wrist_r, l_elbow_r])

    path, dtw_dist = dtw_path(traj_g, traj_r)
    idx_g = [p[0] for p in path]
    idx_r = [p[1] for p in path]

    # Metrics right arm
    if active_side in ('right', 'both'):
        rw_g, re_g = r_wrist_g[idx_g], r_elbow_g[idx_g]
        rw_r, re_r = r_wrist_r[idx_r], r_elbow_r[idx_r]
        rmse_r_wrist = _rmse3d(rw_g, rw_r)
        rmse_r_elbow = _rmse3d(re_g, re_r)
        peak_r       = float(np.max(np.linalg.norm(rw_g - rw_r, axis=1)))
        pearson_rw   = np.array([_safe_pearsonr(rw_g[:, c], rw_r[:, c]) for c in range(3)])
        pearson_rw   = np.nan_to_num(pearson_rw, nan=0.0)
        smooth_r     = r_wrist_g
    else:
        rmse_r_wrist = nan; rmse_r_elbow = nan; peak_r = nan
        pearson_rw   = np.full(3, nan)
        smooth_r     = None

    # Metrics left arm
    if active_side in ('left', 'both'):
        lw_g, le_g = l_wrist_g[idx_g], l_elbow_g[idx_g]
        lw_r, le_r = l_wrist_r[idx_r], l_elbow_r[idx_r]
        rmse_l_wrist = _rmse3d(lw_g, lw_r)
        rmse_l_elbow = _rmse3d(le_g, le_r)
        peak_l       = float(np.max(np.linalg.norm(lw_g - lw_r, axis=1)))
        pearson_lw   = np.array([_safe_pearsonr(lw_g[:, c], lw_r[:, c]) for c in range(3)])
        pearson_lw   = np.nan_to_num(pearson_lw, nan=0.0)
        smooth_l     = l_wrist_g
    else:
        rmse_l_wrist = nan; rmse_l_elbow = nan; peak_l = nan
        pearson_lw   = np.full(3, nan)
        smooth_l     = None

    # Pearson mean only for the active arm
    if active_side == 'left':
        pearson_mean = float(np.nanmean(pearson_lw))
    elif active_side == 'right':
        pearson_mean = float(np.nanmean(pearson_rw))
    else:
        pearson_mean = float(np.nanmean(np.concatenate([pearson_rw, pearson_lw])))

    # Smoothness only for the active arm
    wrist_active = (smooth_l if active_side == 'left'
                    else smooth_r if active_side == 'right'
                    else np.hstack([r_wrist_g, l_wrist_g]))
    if wrist_active is not None and len(wrist_active) > 2:
        jerk            = np.diff(wrist_active, n=2, axis=0)
        cart_smoothness = float(-np.mean(jerk ** 2))
    else:
        cart_smoothness = nan

    if label:
        wrist_rmse = rmse_l_wrist if active_side == 'left' else rmse_r_wrist
        print(f'  [{label:<16}]  '
              f'DTW_cart={dtw_dist:.2f}m  '
              f'RMSE_wrist={wrist_rmse:.4f}m  '
              f'Pearson={pearson_mean:.3f}')

    return {
        'cart_dtw'         : float(dtw_dist),
        'cart_rmse_r_wrist': rmse_r_wrist,
        'cart_rmse_l_wrist': rmse_l_wrist,
        'cart_rmse_r_elbow': rmse_r_elbow,
        'cart_rmse_l_elbow': rmse_l_elbow,
        'cart_peak_r_wrist': peak_r,
        'cart_peak_l_wrist': peak_l,
        'cart_pearson_rw'  : pearson_rw,   # (3,) for coordinate X, Y, Z
        'cart_pearson_lw'  : pearson_lw,
        'cart_pearson_mean': pearson_mean,
        'cart_smoothness'  : cart_smoothness,
    }


# ============================================================================
# Aggregation (average over multiple sequences)
# ============================================================================

def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    '''
    Average of a list of metric dictionaries (e.g., multiple human demos or exercises).
    Scalar keys: nanmean. Vectorial keys: nanmean per element.
    '''
    scalar_keys = [
        'dtw_distance', 'rmse_mean', 'peak_error_mean',
        'pearson_mean', 'smoothness',
        'cart_dtw',
        'cart_rmse_r_wrist', 'cart_rmse_l_wrist',
        'cart_rmse_r_elbow', 'cart_rmse_l_elbow',
        'cart_peak_r_wrist', 'cart_peak_l_wrist',
        'cart_pearson_mean', 'cart_smoothness',
    ]
    arr_keys = [
        'rmse_per_joint', 'peak_error_per_joint', 'pearson_per_joint',
        'cart_pearson_rw', 'cart_pearson_lw',
    ]
    # Take only the keys present in all dictionaries of the list
    present_scalars = [k for k in scalar_keys
                       if any(k in m for m in metrics_list)]
    present_arrs    = [k for k in arr_keys
                       if any(k in m for m in metrics_list)]

    out = {k: float(np.nanmean([m[k] for m in metrics_list if k in m]))
           for k in present_scalars}
    out.update({
        k: np.nanmean([m[k] for m in metrics_list if k in m], axis=0)
        for k in present_arrs
    })
    return out


# ============================================================================
# Print summary to terminal
# ============================================================================

def print_summary(results: Dict, title: str) -> None:
    '''Print a summary table with joint-space and cartesian metrics.'''
    w = 100
    print(f'\n{"="*w}\n  {title}\n{"="*w}')

    # Joint space
    print(f'\n  [Joint space]\n  {"─"*80}')
    print(f'  {"Method":<20}  {"DTW(°)":>9}  {"RMSE(°)":>8}  '
          f'{"Peak(°)":>8}  {"Pearson":>8}  {"Smooth":>10}')
    print(f'  {"─"*80}')
    for m, r in results.items():
        if 'dtw_distance' not in r:
            continue
        print(f'  {m:<20}  {r["dtw_distance"]:>9.2f}  {r["rmse_mean"]:>8.2f}  '
              f'{r["peak_error_mean"]:>8.2f}  {r["pearson_mean"]:>8.3f}  '
              f'{r["smoothness"]:>10.4f}')

    if any('cart_dtw' in r for r in results.values()):
        print(f'\n  [Cartesian space]\n  {"─"*80}')
        print(f'  {"Method":<20}  {"DTW(m)":>9}  {"Rw(m)":>8}  '
              f'{"Lw(m)":>8}  {"Pearson":>8}  {"Smooth":>10}')
        print(f'  {"─"*80}')
        for m, r in results.items():
            if 'cart_dtw' not in r:
                continue
            print(f'  {m:<20}  {r["cart_dtw"]:>9.2f}  '
                  f'{r["cart_rmse_r_wrist"]:>8.4f}  '
                  f'{r["cart_rmse_l_wrist"]:>8.4f}  '
                  f'{r["cart_pearson_mean"]:>8.3f}  '
                  f'{r["cart_smoothness"]:>10.6f}')
    print(f'\n{"="*w}')