'''
compute_canonical.py
=============================================================================
Computes canonical trajectories for each exercise using two methods in parallel:
  - Standard DBA  (DTW Barycenter Averaging via tslearn)  → canonical.csv
  - ShapeDBA      (ShapeDTW Barycenter Averaging via aeon) → canonicalShape.csv

Input: all data/landmarks/subject_XXX/exercise_XXX/video_XXX/joint_ik.csv for the same exercise (for all exercises)

Output: data/dataset/exercise_XXX/<split>/canonical.csv
        data/dataset/exercise_XXX/<split>/canonicalShape.csv

Args    
    --exercise  -> type=int, default=None, Only process this exercise number. Default: all exercises found in landmarks/
    --subject   -> type=int, default=None, Only use this subject. Default: all subjects
    --max-iter  -> type=int, default=DEFAULT_MAX_ITER, Maximum ShapeDBA iterations
    --reach     -> type=int, default=DEFAULT_REACH, ShapeDTW neighborhood size
    --no-amplitude-rescale  -> action='store_true', Disable amplitude rescaling
    --amplitude-percentile  -> type=float, default=80.0, Percentile used when rescaling amplitude
    --no-smooth -> action='store_true', Disable post-ShapeDBA smoothing of the canonical
    --smooth-window -> type=int, default=11,Window size for median + Savitzky-Golay smoothing, must be odd
    --n-demos   -> type=int, default=55, choices=[10,25,55]

Amplitude rescaling (enabled by default):
  Before DBA/ShapeDBA, each sequence is normalized per-joint to [0, 1] so that the algorithm
  captures only the temporal shape (timing, velocity profile). After averaging, the canonical
  is rescaled back using the Nth percentile of per-demo max values (and the (100-N)th
  percentile of per-demo min values). This prevents low-amplitude demos from dominating
  the barycenter and ensures the canonical reaches a realistic full range.
  The same normalization is applied to both DBA and ShapeDBA for a fair comparison.

DBA (standard):
  Uses standard DTW Barycenter Averaging (tslearn). Variable-length sequences are resampled
  to median length before averaging for consistency with ShapeDBA.
  Requires: pip install tslearn

ShapeDBA:
  Uses ShapeDTW which aligns shape descriptors of subsequences (neighborhoods of size
  `reach`) instead of raw values, producing smoother, more shape-faithful barycenters.
  All sequences are resampled to median length before averaging (aeon requires equal-length inputs).
  Requires: pip install aeon
'''

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
from scipy.interpolate import interp1d
from scipy.signal import medfilt, savgol_filter

from utilities.config import DATA_ROOT, JOINT_COLS, HEAD_COLS
from utilities.split_utils import split_name, select_subjects

OUTPUT_COLS = ['frame', 'timestamp'] + JOINT_COLS 

# ShapeDBA parameter
DEFAULT_MAX_ITER = 30
DEFAULT_REACH    = 15

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_sequences(landmarks_root, exercise_num, filter_subject=None, selected_subjects=None) -> List[pd.DataFrame]:
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
        if selected_subjects is not None and subject_num not in selected_subjects:
            continue
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
            df = df.dropna(subset=JOINT_COLS).reset_index(drop=True)

            if len(df) < 2:
                print(f'SKIP ({len(df)} frames)')
                n_skipped += 1
                continue

            print(f'{len(df)} frames')
            sequences.append(df)

    print(f'\nSequences loaded : {len(sequences)}  |  skipped: {n_skipped}')
    return sequences

def _resample_sequence(arr: np.ndarray, target_len: int) -> np.ndarray:
    '''
    Resamples a (N, D) sequence to (target_len, D) using linear interpolation.

    Parameters:
        arr        : (N, D) array
        target_len : desired output length

    Returns:
        (target_len, D) array
    '''
    if len(arr) == target_len:
        return arr
    t_old = np.linspace(0, 1, len(arr))
    t_new = np.linspace(0, 1, target_len)
    return interp1d(t_old, arr, axis=0, kind='linear')(t_new)


def _run_shape_dba(arrays: List[np.ndarray], max_iter: int, label: str, reach: int = DEFAULT_REACH) -> np.ndarray:
    '''
    Runs ShapeDBA on a list of numpy arrays and returns the barycenter sequence.

    All sequences are resampled to the median length before averaging
    (aeon's elastic_barycenter_average requires equal-length inputs).

    Parameters:
        arrays   : list of (N_i, D) arrays (sequences to average)
        max_iter : maximum number of ShapeDBA iterations
        label    : name shown in progress output (e.g. 'joints', 'head')
        reach    : ShapeDTW neighborhood size for shape descriptors

    Returns:
        (L, D) array : the canonical sequence (L = median input length)
    '''
    from aeon.clustering.averaging import elastic_barycenter_average

    lengths    = np.array([len(a) for a in arrays])
    median_len = int(np.median(lengths))

    # Sequence closest to median - used as init barycenter
    init_idx = int(np.argmin(np.abs(lengths - median_len)))
    init_seq = arrays[init_idx]

    print(f'Running ShapeDBA on {label} ({len(arrays)} sequences, '
          f'max_iter={max_iter}, reach={reach}) ...', flush=True)

    # Resample all to median length; aeon requires uniform-length 3-D input
    resampled = [_resample_sequence(a, median_len) for a in arrays]

    # aeon convention: (n_samples, n_channels, n_timepoints) → transpose (L, D) → (D, L)
    X       = np.stack([a.T for a in resampled], axis=0)   # (S, D, L)
    init_bc = _resample_sequence(init_seq, median_len).T   # (D, L)

    barycenter_t = elastic_barycenter_average(
        X,
        distance        = 'shape_dtw',
        max_iters       = max_iter,
        tol             = 1e-5,
        init_barycenter = init_bc,
        verbose         = False,
        reach           = reach,
    )
    # barycenter_t: (D, L) → transpose back to (L, D)
    barycenter = barycenter_t.T

    print(f'Done. Canonical length: {len(barycenter)} frames')
    return barycenter


def _run_dba(arrays: List[np.ndarray], max_iter: int, label: str, **_kwargs) -> np.ndarray:
    '''
    Runs standard DBA (DTW Barycenter Averaging) via tslearn.

    All sequences are resampled to median length before averaging,
    consistent with _run_shape_dba.

    Parameters:
        arrays   : list of (N_i, D) arrays (sequences to average)
        max_iter : maximum number of DBA iterations
        label    : name shown in progress output

    Returns:
        (L, D) array : the canonical sequence (L = median input length)
    '''
    from tslearn.barycenters import dtw_barycenter_averaging

    lengths    = np.array([len(a) for a in arrays])
    median_len = int(np.median(lengths))

    print(f'Running DBA on {label} ({len(arrays)} sequences, '
          f'max_iter={max_iter}) ...', flush=True)

    resampled = [_resample_sequence(a, median_len) for a in arrays]
    X = np.stack(resampled, axis=0)   # (S, L, D) - tslearn convention

    barycenter = dtw_barycenter_averaging(X, max_iter=max_iter)  # (L, D)
    print(f'Done. Canonical length: {len(barycenter)} frames')
    return barycenter


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------
def _save_canonical_csv(canonical_joints: np.ndarray, canonical_head: np.ndarray, sequences: List[pd.DataFrame], output_path: Path) -> None:
    '''
    Builds the output DataFrame from canonical joints + head and saves it.

    Parameters:
        canonical_joints : (L, len(JOINT_COLS)) array in degrees
        canonical_head   : (L, 3) array in meters (may be all-NaN)
        sequences        : input DataFrames used to compute mean duration
        output_path      : full path of the .csv file to write
    '''
    L = len(canonical_joints)

    mean_duration = np.mean([
        df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        for df in sequences
        if 'timestamp' in df.columns and len(df) > 1
    ])
    timestamps = np.linspace(0.0, float(mean_duration), L)

    out = pd.DataFrame({'frame': np.arange(L), 'timestamp': timestamps})

    for j, col in enumerate(JOINT_COLS):
        out[col] = canonical_joints[:, j]

    for j, col in enumerate(HEAD_COLS):
        out[col] = canonical_head[:, j]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out[OUTPUT_COLS].to_csv(output_path, index=False)
    print(f'  Saved -> {output_path.relative_to(DATA_ROOT)}')


# ---------------------------------------------------------------------------
# Amplitude normalisation helpers
# ---------------------------------------------------------------------------
def _normalize_amplitude(arrays: List[np.ndarray]):
    '''
    Normalises each sequence per-joint to [0, 1].

    For joints that never move in a given sequence (max == min), the column is
    left at 0.0 to avoid division by zero - DBA will produce 0.0 for that joint,
    which is then correctly rescaled back.

    Parameters:
        arrays : list of (N_i, D) arrays

    Returns:
        norm_arrays : list of (N_i, D) arrays in [0, 1]
        seq_mins    : (S, D) array - per-sequence per-joint min values
        seq_maxs    : (S, D) array - per-sequence per-joint max values
    '''
    seq_mins = np.array([a.min(axis=0) for a in arrays])   # (S, D)
    seq_maxs = np.array([a.max(axis=0) for a in arrays])   # (S, D)

    norm_arrays = []
    for a, mn, mx in zip(arrays, seq_mins, seq_maxs):
        span = mx - mn
        span_safe = np.where(span == 0, 1.0, span)         # avoid div-by-zero
        norm_arrays.append((a - mn) / span_safe)

    return norm_arrays, seq_mins, seq_maxs


def _rescale_canonical(canonical_norm: np.ndarray, seq_mins: np.ndarray, seq_maxs: np.ndarray, percentile: float) -> np.ndarray:
    '''
    Rescales a [0, 1]-normalised canonical back to physical units.

    target_max[j] = percentile(seq_maxs[:, j], p)
    target_min[j] = percentile(seq_mins[:, j], 100 - p)
    canonical_rescaled[:, j] = canonical_norm[:, j] * (target_max[j] - target_min[j]) + target_min[j]

    Using the (100-p)th percentile for min is symmetric: e.g. p=95 → 5th
    percentile for min, which trims the most extreme low-amplitude demos too.

    Parameters:
        canonical_norm : (L, D) normalised canonical from DBA
        seq_mins       : (S, D) per-sequence per-joint min values
        seq_maxs       : (S, D) per-sequence per-joint max values
        percentile     : percentile p used for target_max (0-100)

    Returns:
        (L, D) canonical in original physical units
    '''
    target_max = np.percentile(seq_maxs, percentile,       axis=0)  # (D,)
    target_min = np.percentile(seq_mins, 100 - percentile, axis=0)  # (D,)

    span = target_max - target_min
    canonical_rescaled = canonical_norm * span + target_min
    return canonical_rescaled


def _smooth_canonical(canonical: np.ndarray, window: int) -> np.ndarray:
    '''
    Smooths the canonical trajectory per-joint using a median filter followed
    by a Savitzky-Golay filter.

    Median filter (same window) removes isolated spikes.
    Savitzky-Golay (window, poly=3) then smooths residual noise while
    preserving the overall shape and peak amplitudes.

    Parameters:
        canonical : (L, D) array - canonical joint angles
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



def _process_exercise(exercise_num, landmarks_root, dataset_root,
                      filter_subject=None, n_demos=55,
                      smooth=True, smooth_window=11,
                      amplitude_percentile=80.0, amplitude_rescale=True,
                      max_iter=30, reach=15):
    '''
    Loads all joint_ik.csv files for one exercise, runs ShapeDBA separately on
    joint angles and head gaze, and saves the combined canonical.csv.

    Parameters:
        exercise_num         : exercise number to process
        landmarks_root       : root of the landmarks folder
        dataset_root         : root of the dataset output folder
        filter_subject       : if set, only use this subject
        max_iter             : maximum ShapeDBA iterations
        amplitude_rescale    : if True, normalise amplitude before ShapeDBA and rescale after
        amplitude_percentile : percentile used when rescaling (e.g. 80)
        smooth               : if True, apply median + Savitzky-Golay filter to canonical joints
        smooth_window        : filter window size (odd integer)
        reach                : ShapeDTW neighborhood size
    '''
    exercise_name = f'exercise_{exercise_num:03d}'
    
    all_subj_dirs = sorted(landmarks_root.glob('subject_*'))
    selected      = select_subjects(all_subj_dirs, n_demos)
    selected_nums = [int(d.name.split('_')[1]) for d in selected]
    print(f'  Subjects for n_{n_demos:02d}: {selected_nums}')

    print(f'\n{"="*60}')
    print(f'  Exercise {exercise_num:03d}')
    print(f'{"="*60}')

    # Load sequences
    sequences = _load_sequences(landmarks_root, exercise_num,
                                filter_subject=filter_subject,
                                selected_subjects=selected_nums)

    if len(sequences) < 1:
        print(f'Need at least 2 sequences for DBA, found {len(sequences)}. Skipping.')
        return

    # --- Extract arrays for DBA ---
    # Joints: shape (N_i, 16), in degrees
    joint_arrays = [df[JOINT_COLS].values.astype(float) for df in sequences]

    # Head: shape (N_i, 3), in meters; use only sequences that have head data
    head_available = [df for df in sequences if all(c in df.columns for c in HEAD_COLS) and not df[HEAD_COLS].isna().all().any()]
    head_arrays = [df[HEAD_COLS].values.astype(float) for df in head_available]

    split_dir = dataset_root / exercise_name / split_name(n_demos)
    split_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Shared amplitude normalisation (same preprocessing for both methods)
    # -----------------------------------------------------------------------
    if amplitude_rescale:
        print(f'\nAmplitude rescaling ENABLED (percentile={amplitude_percentile})')
        norm_joint_arrays, seq_mins, seq_maxs = _normalize_amplitude(joint_arrays)

        # Log per-joint rescaling targets
        target_max = np.percentile(seq_maxs, amplitude_percentile, axis=0)
        target_min = np.percentile(seq_mins, 100 - amplitude_percentile, axis=0)
        print(f'  {"Joint":<30} {"TargetMin":>10} {"TargetMax":>10}')
        for j, col in enumerate(JOINT_COLS):
            print(f'  {col:<30} {target_min[j]:>10.2f} {target_max[j]:>10.2f}')
    else:
        print('\nAmplitude rescaling DISABLED')
        norm_joint_arrays = joint_arrays
        seq_mins = seq_maxs = None

    def _compute_joints(run_fn, label_suffix):
        '''Amplitude-normalise → run_fn → rescale → smooth.'''
        if amplitude_rescale:
            canonical_norm = run_fn(norm_joint_arrays, max_iter,
                                    label=f'joints (normalised) [{label_suffix}]',
                                    reach=reach)
            canonical = _rescale_canonical(canonical_norm, seq_mins, seq_maxs,
                                           amplitude_percentile)
        else:
            canonical = run_fn(joint_arrays, max_iter,
                               label=f'joints [{label_suffix}]',
                               reach=reach)
        if smooth:
            print(f'\nSmoothing [{label_suffix}] joints (median + Savitzky-Golay, window={smooth_window}) ...')
            canonical = _smooth_canonical(canonical, smooth_window)
            print('Done.')
        return canonical

    def _compute_head(run_fn, label_suffix, target_len):
        '''Run run_fn on head arrays and resample to target_len if needed.'''
        if not head_arrays:
            print(f'No head data found [{label_suffix}], filling with NaN.')
            return np.full((target_len, 3), np.nan)
        canonical = run_fn(head_arrays, max_iter,
                           label=f'head [{label_suffix}]',
                           reach=reach)
        if len(canonical) != target_len:
            t_old = np.linspace(0, 1, len(canonical))
            t_new = np.linspace(0, 1, target_len)
            canonical = interp1d(t_old, canonical, axis=0, kind='linear')(t_new)
        return canonical

    # -----------------------------------------------------------------------
    # ShapeDBA → canonicalShape.csv
    # -----------------------------------------------------------------------
    print(f'\n{"─"*55}')
    print(f'  ShapeDBA  (reach={reach})')
    print(f'{"─"*55}')
    shape_joints = _compute_joints(_run_shape_dba, 'ShapeDBA')
    shape_head   = _compute_head(_run_shape_dba, 'ShapeDBA', len(shape_joints))
    _save_canonical_csv(shape_joints, shape_head, sequences,
                        split_dir / 'canonicalShape.csv')

    # -----------------------------------------------------------------------
    # Standard DBA → canonical.csv
    # -----------------------------------------------------------------------
    print(f'\n{"─"*55}')
    print(f'  Standard DBA')
    print(f'{"─"*55}')
    dba_joints = _compute_joints(_run_dba, 'DBA')
    dba_head   = _compute_head(_run_dba, 'DBA', len(dba_joints))
    _save_canonical_csv(dba_joints, dba_head, sequences,
                        split_dir / 'canonical.csv')

    # Summary
    mean_input_len = np.mean([len(s) for s in sequences])
    print(f'\n  Input sequences  : {len(sequences)}')
    print(f'  Mean input length: {mean_input_len:.0f} frames')
    print(f'  ShapeDBA length  : {len(shape_joints)} frames')
    print(f'  DBA length       : {len(dba_joints)} frames')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Compute canonical trajectory via ShapeDBA for each exercise.')
    parser.add_argument('--exercise', type=int, default=None, help='Only process this exercise number. Default: all exercises found in landmarks/.')
    parser.add_argument('--subject', type=int, default=None, help='Only use this subject. Default: all subjects.')
    parser.add_argument('--max-iter', type=int, default=DEFAULT_MAX_ITER, help=f'Maximum ShapeDBA iterations (default: {DEFAULT_MAX_ITER}).')
    parser.add_argument('--reach', type=int, default=DEFAULT_REACH, help=f'ShapeDTW neighborhood size (default: {DEFAULT_REACH}).')
    parser.add_argument('--no-amplitude-rescale', action='store_true', help='Disable amplitude rescaling.')
    parser.add_argument('--amplitude-percentile', type=float, default=80.0, help='Percentile used when rescaling amplitude (default: 80). 100 = absolute max of all demos.')
    parser.add_argument('--no-smooth', action='store_true', help='Disable post-ShapeDBA smoothing of the canonical.')
    parser.add_argument('--smooth-window', type=int, default=11, help='Window size for median + Savitzky-Golay smoothing (default: 11, must be odd).')
    parser.add_argument('--n-demos', type=int, default=55, choices=[10,25,55])
    args = parser.parse_args()

    landmarks_root = DATA_ROOT / 'landmarks'
    dataset_root   = DATA_ROOT / 'dataset'

    if not landmarks_root.is_dir():
        print(f'Error: landmarks folder not found -> {landmarks_root}')
        return

    print(f'Landmarks root    : {landmarks_root}')
    print(f'Dataset root      : {dataset_root}')
    print(f'ShapeDBA max iter : {args.max_iter}  |  reach={args.reach}')
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
            n_demos              = args.n_demos,
            max_iter             = args.max_iter,
            amplitude_rescale    = not args.no_amplitude_rescale,
            amplitude_percentile = args.amplitude_percentile,
            smooth               = not args.no_smooth,
            smooth_window        = args.smooth_window,
            reach                = args.reach,
        )

    print(f'\n{"="*60}')
    print('  Done.')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()