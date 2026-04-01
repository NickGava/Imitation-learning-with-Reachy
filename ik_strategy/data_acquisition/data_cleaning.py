'''
data_cleaning.py
=============================================================================
Data cleaning pipeline for extracted landmarks.

For each video (pose.csv, right_hand.csv, left_hand.csv, face.csv):
  1. Drop incomplete rows: frames where MediaPipe returned no landmarks
  2. Drop low-visibility frames (pose only): coordinate present but unreliable
  3. Jump detection: remove frames with sudden, physically impossible shifts
  4. Smoothing: One Euro Filter on all coordinate columns (visibility excluded)

Input:
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/{pose,right_hand,left_hand,face}.csv

Output (same folder):
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/{pose,right_hand,left_hand,face}_cleaned.csv

Note: the 'frame' column retains the original video indices (gaps are expected and intentional, they preserve traceability to the source video).
'''

import math
import numpy as np
import pandas as pd
import cv2
from ask_inputs import ask_inputs
from config import DATA_ROOT, DEFAULT_FPS

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# Visibility filter (pose only)
MIN_VISIBILITY = 0.5           # frames where any joint is below this are dropped

# Jump detection
JUMP_FACTOR = 5.0               # drop frame if displacement > JUMP_FACTOR × median displacement
                                # lower = stricter; raise if too many valid frames are removed

# One Euro Filter parameters
ONE_EURO_MINCUTOFF = 1.0        # min_cutoff: lower -> smoother but more lag at rest
ONE_EURO_BETA      = 0.007      # beta: higher -> less lag during fast motion
ONE_EURO_DCUTOFF   = 1.0        # d_cutoff: cutoff for the derivative low-pass; usually left at 1.0


# ---------------------------------------------------------------------------
# One Euro Filter
# ---------------------------------------------------------------------------
class _OneEuroFilter1D:
    """One Euro Filter for a single scalar signal."""

    def __init__(self, freq : float, min_cutoff : float, beta : float, d_cutoff : float):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x  = None
        self._dx = 0.0

    def _alpha(self, cutoff : float) -> float:
        te  = 1.0 / self.freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x : float) -> float:
        if self._x is None:
            self._x = x
            return x
        dx = (x - self._x) * self.freq
        alpha_d = self._alpha(self.d_cutoff)
        self._dx = alpha_d * dx + (1.0 - alpha_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        alpha = self._alpha(cutoff)
        self._x = alpha * x + (1.0 - alpha) * self._x
        return self._x


def _apply_one_euro(df : pd.DataFrame, xyz_cols : list, fps : float) -> pd.DataFrame:
    """Apply an independent One Euro Filter to each coordinate column."""
    df = df.copy()
    for col in xyz_cols:
        f = _OneEuroFilter1D(fps, ONE_EURO_MINCUTOFF, ONE_EURO_BETA, ONE_EURO_DCUTOFF)
        df[col] = [f(v) for v in df[col]]
    return df


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------
def _drop_incomplete(df : pd.DataFrame, coord_cols : list) -> pd.DataFrame:
    """Drop rows that have NaN in any coordinate column."""
    before = len(df)
    df = df.dropna(subset=coord_cols).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"incomplete rows dropped: {dropped}")
    return df


def _drop_low_visibility(df : pd.DataFrame, vis_cols : list) -> pd.DataFrame:
    """
    Drop rows where any visibility score is below MIN_VISIBILITY.
    Only called for pose.csv (hand and face CSVs have no visibility columns).
    """
    if not vis_cols:
        return df
    mask = (df[vis_cols] >= MIN_VISIBILITY).all(axis=1)
    dropped = (~mask).sum()
    if dropped:
        print(f"low-visibility rows dropped: {dropped} (threshold={MIN_VISIBILITY})")
    return df[mask].reset_index(drop=True)


def _drop_jumps(df: pd.DataFrame, xyz_cols: list) -> pd.DataFrame:
    """
    Remove frames where the L2 displacement (across all coordinates) from the
    previous frame exceeds JUMP_FACTOR x median displacement.

    Uses the median as the reference so outliers don't inflate the threshold.
    Gaps caused by missing frames (non-consecutive frame indices) are excluded
    from both the median computation and the jump check.
    """
    if len(df) < 2:
        return df

    coords    = df[xyz_cols].values
    frame_ids = df['frame'].values

    deltas = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    gaps   = np.diff(frame_ids) > 1    # True where frame index is not consecutive

    # Compute median only on deltas that are NOT caused by a gap
    deltas_no_gap = deltas[~gaps]
    median = np.median(deltas_no_gap) if len(deltas_no_gap) > 0 else np.median(deltas)
    thresh = JUMP_FACTOR * median

    keep = np.ones(len(df), dtype=bool)
    for i, d in enumerate(deltas):
        if gaps[i]:
            continue        # gap -> not a real jump, skip
        if d > thresh:
            keep[i + 1] = False

    dropped = (~keep).sum()
    if dropped:
        print(f"jump frames dropped: {dropped} (threshold={thresh:.5f}, median_disp={median:.5f})")
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------
def _clean_file(input_path, output_path, is_pose : bool, fps : float):
    """
    Run the full cleaning pipeline on a single CSV file and save the result.

    Parameters:
        input_path:     path to the raw landmark CSV
        output_path:    path where the cleaned CSV is written (same folder, "_cleaned" suffix)
        is_pose:        True for pose.csv (enables visibility filtering)
        fps:            frame rate of the source video (used by One Euro Filter)
    """
    df = pd.read_csv(input_path)
    n_original = len(df)

    meta_cols = ['frame', 'timestamp']
    vis_cols  = [c for c in df.columns if c.endswith('_vis')] if is_pose else []
    xyz_cols  = [c for c in df.columns if c not in meta_cols and c not in vis_cols]
    all_coord = xyz_cols + vis_cols

    print(f"rows loaded: {n_original}")

    # Step 1 — drop incomplete rows
    df = _drop_incomplete(df, all_coord)

    # Step 2 — visibility filter (pose only)
    if is_pose:
        df = _drop_low_visibility(df, vis_cols)

    # Step 3 — jump detection (on xyz only)
    df = _drop_jumps(df, xyz_cols)

    # Step 4 — smoothing (on xyz only, not visibility)
    df = _apply_one_euro(df, xyz_cols, fps)
    print(f"smoothing applied: One Euro Filter (fps={fps:.1f})")

    n_final = len(df)
    pct = 100.0 * n_final / n_original if n_original else 0
    print(f"rows kept: {n_final} / {n_original} ({pct:.1f}%)")

    df.to_csv(output_path, index=False)
    print(f"saved → {output_path.relative_to(DATA_ROOT)}")


# FPS helper
def _get_fps(subject_name : str, exercise_name : str, video_name : str) -> float:
    """Try to read FPS from the source .mp4; fall back to DEFAULT_FPS."""
    video_path = DATA_ROOT / "raw_data" / subject_name / exercise_name / f"{video_name}.mp4"
    if video_path.exists():
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps > 0:
                    return fps
        except ImportError:
            pass
    print(f"  Warning: could not read FPS from video, using default ({DEFAULT_FPS} fps)")
    return DEFAULT_FPS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    subject_name, exercise_name, video_name = ask_inputs()
    landmarks_folder = DATA_ROOT / "landmarks" / subject_name / exercise_name / video_name
    if not landmarks_folder.is_dir():
        print(f"Error: folder not found → {landmarks_folder}")
        return

    fps = _get_fps(subject_name, exercise_name, video_name)
    print(f"\nFPS: {fps:.1f}\n")

    files = {
        'pose':       True,     # is_pose=True → enables visibility filter
        'right_hand': False,
        'left_hand':  False,
        'face':       False,    
    }

    for name, is_pose in files.items():
        input_path  = landmarks_folder / f"{name}.csv"
        output_path = landmarks_folder / f"{name}_cleaned.csv"

        if not input_path.exists():
            print(f"[SKIP] {name}.csv not found.\n")
            continue

        print(f"--- {name}.csv ---")
        _clean_file(input_path, output_path, is_pose, fps)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
