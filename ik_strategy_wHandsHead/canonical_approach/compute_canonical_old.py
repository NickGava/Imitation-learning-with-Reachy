'''
compute_canonical.py
=============================================================================
Computes the canonical trajectory for each exercise using DBA (DTW Barycenter Averaging).

Input: all data/landmarks/subject_XXX/exercise_XXX/video_XXX/joint_ik.csv for the same exercise (for all exercises)

Output: data/dataset/exercise_XXX/canonical.csv

Usage:
  python compute_canonical.py                           # all exercises, all subjects
  python compute_canonical.py --exercise 1              # only exercise 1
  python compute_canonical.py --exercise 1 --subject 2  # only subject 2, exercise 1
  python compute_canonical.py --max-iter 50             # change number of DBA iterations (default: 30)
  python compute_canonical.py --amplitude-percentile 95 # use 95th percentile for rescaling (default: 95)
  python compute_canonical.py --no-amplitude-rescale    # disable amplitude rescaling (classic DBA)

Amplitude rescaling (enabled by default):
  Before DBA, each sequence is normalized per-joint to [0, 1] so that DBA captures
  only the temporal shape (timing, velocity profile). After DBA, the canonical is
  rescaled back using the Nth percentile of per-demo max values (and the (100-N)th
  percentile of per-demo min values). This prevents low-amplitude demos from
  dominating the barycenter and ensures the canonical reaches a realistic full range.
'''

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
from scipy.interpolate import interp1d
from scipy.signal import medfilt, savgol_filter
from tslearn.barycenters import dtw_barycenter_averaging

from config import DATA_ROOT, JOINT_COLS, HEAD_COLS


OUTPUT_COLS = ['frame', 'timestamp'] + JOINT_COLS + HEAD_COLS

# DBA parameter
DEFAULT_MAX_ITER = 30

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_sequences(landmarks_root: Path, exercise_num: int, filter_subject: Optional[int]) -> List[pd.DataFrame]:
    '''
    Crawls the landmarks folder and loads all joint_ik.csv files for the given exercise.
    Returns a list of DataFrames, one per video.

    Parameters:
        landmarks_root : path to data/landmarks/
        exercise_num   : exercise number to load
        filter_subject : if set, only load this subject

    Returns:
        List of DataFrames, each with JOINT_COLS + HEAD_COLS columns.
        Empty list if no valid files are found.
    '''
    exercise_name = f'exercise_{exercise_num:03d}'
    sequences     = []
    n_skipped     = 0

    for subj_dir in sorted(landmarks_root.glob('subject_*')):
        subject_num = int(subj_dir.name.split('_')[1])
        if filter_subject is not None and subject_num != filter_subject:
            continue

        exer_dir = subj_dir / exercise_name
        if not exer_dir.is_dir():
            continue

        for video_dir in sorted(exer_dir.glob('video_*')):
            video_num = int(video_dir.name.split('_')[1])
            ik_path   = video_dir / 'joint_ik.csv'

            print(f'Loading subject_{subject_num:03d} / {exercise_name} / video_{video_num:03d} ... ', end='', flush=True)

            if not ik_path.exists():
                print('MISSING')
                n_skipped += 1
                continue

            df = pd.read_csv(ik_path)

            # Keep only rows where all joint columns are present
            # df = df.dropna(subset=JOINT_COLS).reset_index(drop=True)

            if len(df) < 2:
                print(f'SKIP ({len(df)} frames)')
                n_skipped += 1
                continue

            print(f'{len(df)} frames')
            sequences.append(df)

    print(f'\nSequences loaded : {len(sequences)}  |  skipped: {n_skipped}')
    return sequences

def _run_dba(arrays: List[np.ndarray], max_iter: int, label: str) -> np.ndarray:
    '''
    Runs DBA on a list of numpy arrays and returns the barycenter sequence.

    Parameters:
        arrays   : list of (N_i, D) arrays (sequences to average)
        max_iter : maximum number of DBA iterations
        label    : name shown in progress output (e.g. 'joints', 'head')

    Returns:
        (L, D) array : the canonical sequence
    '''
    # Find the sequence with length closest to the median length
    init_seq = arrays[int(np.argmin(np.abs(np.array([len(a) for a in arrays]) - np.median([len(a) for a in arrays]))))]

    print(f'Running DBA on {label} ({len(arrays)} sequences, max_iter={max_iter}) ...', flush=True)

    barycenter = dtw_barycenter_averaging(
        arrays,
        barycenter_size = len(init_seq),   # target length = median length
        init_barycenter = init_seq,
        max_iter        = max_iter,
        tol             = 1e-5,
        verbose         = False,           # set to True for more detailed convergence info
    )

    print(f'Done. Canonical length: {len(barycenter)} frames')
    return barycenter   # shape: (L, D)

# ---------------------------------------------------------------------------
# Amplitude normalisation helpers
# ---------------------------------------------------------------------------
def _normalize_amplitude(arrays: List[np.ndarray]):
    '''
    Normalises each sequence per-joint by its maximum absolute value.

    norm[i][:, j] = arrays[i][:, j] / abs_max[i, j]

    This preserves the sign of the movement, so joints that only go negative
    (e.g. elbow: 0 to -125°) remain negative after normalisation, and
    bidirectional joints (e.g. shoulder_roll: -30° to +45°) keep their
    relative asymmetry.

    For joints that never move (abs_max == 0), the column is left at 0.0.

    Parameters:
        arrays : list of (N_i, D) arrays

    Returns:
        norm_arrays : list of (N_i, D) arrays normalised by abs_max
        abs_maxs    : (S, D) array — per-sequence per-joint abs_max values
    '''
    abs_maxs = np.array([np.max(np.abs(a), axis=0) for a in arrays])  # (S, D)

    norm_arrays = []
    for a, abs_max in zip(arrays, abs_maxs):
        abs_max_safe = np.where(abs_max == 0, 1.0, abs_max)  # avoid div-by-zero
        norm_arrays.append(a / abs_max_safe)

    return norm_arrays, abs_maxs


def _rescale_canonical(
    canonical_norm: np.ndarray,
    abs_maxs: np.ndarray,
    percentile: float,
) -> np.ndarray:
    '''
    Rescales a canonical normalised by abs_max back to physical units.

    target_abs[j] = percentile(abs_maxs[:, j], p)
    canonical_rescaled[:, j] = canonical_norm[:, j] * target_abs[j]

    The sign is preserved automatically since canonical_norm retains the
    sign of the original movement.

    Parameters:
        canonical_norm : (L, D) normalised canonical from DBA
        abs_maxs       : (S, D) per-sequence per-joint abs_max values
        percentile     : percentile p used for target_abs (0-100)

    Returns:
        (L, D) canonical in original physical units
    '''
    target_abs = np.percentile(abs_maxs, percentile, axis=0)  # (D,)
    return canonical_norm * target_abs



def _smooth_canonical(canonical: np.ndarray, window: int) -> np.ndarray:
    '''
    Smooths the canonical trajectory per-joint using a median filter followed
    by a Savitzky-Golay filter.

    Median filter (same window) removes isolated spikes.
    Savitzky-Golay (window, poly=3) then smooths residual noise while
    preserving the overall shape and peak amplitudes.

    Parameters:
        canonical : (L, D) array — canonical joint angles
        window    : filter window size (must be odd, >= 3)

    Returns:
        (L, D) smoothed array
    '''
    # Ensure window is odd and at least 3
    window = max(3, window | 1)   # bitwise OR with 1 forces odd

    # Savitzky-Golay requires window < sequence length
    sg_window = min(window, len(canonical) - (1 if len(canonical) % 2 == 0 else 0))
    sg_window = max(3, sg_window | 1)

    smoothed = np.empty_like(canonical)
    for j in range(canonical.shape[1]):
        col = canonical[:, j]
        col = medfilt(col, kernel_size=window)          # spike removal
        col = savgol_filter(col, sg_window, polyorder=3)  # shape-preserving smooth
        smoothed[:, j] = col

    return smoothed



def _process_exercise(
    exercise_num: int,
    landmarks_root: Path,
    dataset_root: Path,
    filter_subject: Optional[int],
    max_iter: int,
    amplitude_rescale: bool,
    amplitude_percentile: float,
    smooth: bool,
    smooth_window: int,
) -> None:
    '''
    Loads all joint_ik.csv files for one exercise, runs DBA separately on joint angles and head gaze, and saves the combined canonical.csv.

    Parameters:
        exercise_num         : exercise number to process
        landmarks_root       : root of the landmarks folder
        dataset_root         : root of the dataset output folder
        filter_subject       : if set, only use this subject
        max_iter             : maximum DBA iterations
        amplitude_rescale    : if True, normalise amplitude before DBA and rescale after
        amplitude_percentile : percentile used when rescaling (e.g. 95)
        smooth               : if True, apply median + Savitzky-Golay filter to canonical joints
        smooth_window        : filter window size (odd integer)
    '''
    exercise_name = f'exercise_{exercise_num:03d}'
    print(f'\n{"="*60}')
    print(f'  Exercise {exercise_num:03d}')
    print(f'{"="*60}')

    # Load sequences
    sequences = _load_sequences(landmarks_root, exercise_num, filter_subject)

    if len(sequences) < 1:
        print(f'Need at least 2 sequences for DBA, found {len(sequences)}. Skipping.')
        return

    # --- Extract arrays for DBA ---
    # Joints: shape (N_i, 16), in degrees
    joint_arrays = [df[JOINT_COLS].values.astype(float) for df in sequences]

    # Head: shape (N_i, 3), in meters; use only sequences that have head data
    head_available = [df for df in sequences if all(c in df.columns for c in HEAD_COLS) and not df[HEAD_COLS].isna().all().any()]
    head_arrays = [df[HEAD_COLS].values.astype(float) for df in head_available]

    # --- Amplitude normalisation (joints only) ---
    if amplitude_rescale:
        print(f'\nAmplitude rescaling ENABLED (percentile={amplitude_percentile})')
        norm_joint_arrays, abs_maxs = _normalize_amplitude(joint_arrays)
        canonical_joints_norm = _run_dba(norm_joint_arrays, max_iter, label='joints (normalised by abs_max)')
        canonical_joints = _rescale_canonical(canonical_joints_norm, abs_maxs, amplitude_percentile)

        # Log per-joint rescaling info
        target_abs = np.percentile(abs_maxs, amplitude_percentile, axis=0)
        print(f'  {"Joint":<30} {"TargetAbsMax":>13}')
        for j, col in enumerate(JOINT_COLS):
            print(f'  {col:<30} {target_abs[j]:>13.2f}')
    else:
        print('\nAmplitude rescaling DISABLED (classic DBA)')
        canonical_joints = _run_dba(joint_arrays, max_iter, label='joints (degrees)')

    # --- Smoothing (joint space) ---
    if smooth:
        print(f'\nSmoothing canonical joints (median + Savitzky-Golay, window={smooth_window}) ...')
        canonical_joints = _smooth_canonical(canonical_joints, smooth_window)
        print('Done.')

    if head_arrays:
        canonical_head = _run_dba(head_arrays, max_iter, label='head (meters)')
        # canonical_head may have different length than canonical_joints if not all sequences had head data; resample to match joints length
        if len(canonical_head) != len(canonical_joints):
            t_old = np.linspace(0, 1, len(canonical_head))
            t_new = np.linspace(0, 1, len(canonical_joints))
            interp = interp1d(t_old, canonical_head, axis=0, kind='linear')
            canonical_head = interp(t_new)
    else:
        print('No head data found, filling head columns with NaN.')
        canonical_head = np.full((len(canonical_joints), 3), np.nan)

    # --- Build output DataFrame ---
    L = len(canonical_joints)

    # Reconstruct timestamps: linearly spaced from 0 to mean duration
    mean_duration = np.mean([
        df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        for df in sequences
        if 'timestamp' in df.columns and len(df) > 1
    ])
    timestamps = np.linspace(0.0, float(mean_duration), L)

    out = pd.DataFrame({'frame': np.arange(L), 'timestamp': timestamps,})

    for j, col in enumerate(JOINT_COLS):
        out[col] = canonical_joints[:, j]

    for j, col in enumerate(HEAD_COLS):
        out[col] = canonical_head[:, j]

    # Save
    output_dir  = dataset_root / exercise_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'canonical.csv'
    out[OUTPUT_COLS].to_csv(output_path, index=False)

    # Summary
    mean_input_len = np.mean([len(s) for s in sequences])
    print(f'\n  Input sequences  : {len(sequences)}')
    print(f'  Mean input length: {mean_input_len:.0f} frames')
    print(f'  Canonical length : {L} frames')
    print(f'  Mean duration    : {mean_duration:.2f} s')
    print(f'  Saved -> {output_path.relative_to(DATA_ROOT)}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Compute canonical trajectory via DBA for each exercise.')
    parser.add_argument('--exercise', type=int, default=None, help='Only process this exercise number. Default: all exercises found in landmarks/.')
    parser.add_argument('--subject', type=int, default=None, help='Only use this subject. Default: all subjects.')
    parser.add_argument('--max-iter', type=int, default=DEFAULT_MAX_ITER, help=f'Maximum DBA iterations (default: {DEFAULT_MAX_ITER}).')
    parser.add_argument('--no-amplitude-rescale', action='store_true', help='Disable amplitude rescaling (use classic DBA on raw joint values).')
    parser.add_argument('--amplitude-percentile', type=float, default=90.0, help='Percentile used when rescaling amplitude (default: 95). 100 = absolute max of all demos.')
    parser.add_argument('--no-smooth', action='store_true', help='Disable post-DBA smoothing of the canonical.')
    parser.add_argument('--smooth-window', type=int, default=5, help='Window size for median + Savitzky-Golay smoothing (default: 5, must be odd).')
    args = parser.parse_args()

    landmarks_root = DATA_ROOT / 'landmarks'
    dataset_root   = DATA_ROOT / 'dataset'

    if not landmarks_root.is_dir():
        print(f'Error: landmarks folder not found -> {landmarks_root}')
        return

    print(f'Landmarks root : {landmarks_root}')
    print(f'Dataset root   : {dataset_root}')
    print(f'DBA max iter   : {args.max_iter}')
    print(f'Amplitude rescale : {not args.no_amplitude_rescale}  (percentile={args.amplitude_percentile})')
    print(f'Smoothing         : {not args.no_smooth}  (window={args.smooth_window})')

    # Collect exercise numbers to process
    if args.exercise is not None:
        exercise_nums = [args.exercise]
    else:
        found = set()
        for subj_dir in landmarks_root.glob('subject_*'):
            for exer_dir in subj_dir.glob('exercise_*'):
                found.add(int(exer_dir.name.split('_')[1]))
        exercise_nums = sorted(found)

    if not exercise_nums:
        print('No exercises found. Check that joint_ik.csv files exist.')
        return

    print(f'Exercises found: {exercise_nums}')

    for exercise_num in exercise_nums:
        _process_exercise(
            exercise_num         = exercise_num,
            landmarks_root       = landmarks_root,
            dataset_root         = dataset_root,
            filter_subject       = args.subject,
            max_iter             = args.max_iter,
            amplitude_rescale    = not args.no_amplitude_rescale,
            amplitude_percentile = args.amplitude_percentile,
            smooth               = not args.no_smooth,
            smooth_window        = args.smooth_window,
        )

    print(f'\n{"="*60}')
    print('  Done.')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()