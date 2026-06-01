'''
build_dataset.py
=============================================================================
Builds the behavioral cloning training dataset for a single exercise.
Shared between MLP, GRU and Transformer approaches - they consume the same CSV.

Crawls all joint_ik.csv files for the requested exercise (all subjects,
all videos) and assembles them into a single bc_dataset.csv.

Each row represents one timestep:
  - Metadata : subject, exercise, video, frame, timestamp
  - State    : current joint positions q(t)              [16 values]
               joint velocity q(t) - q(t-VELOCITY_LAG)   [16 values]
  - Action   : joint delta q(t+1) - q(t)                 [16 values]

The first VELOCITY_LAG and last 1 frames of each video are dropped.

--- Input ---
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/joint_ik.csv

--- Output ---
  data/dataset/exercise_XXX/bc_dataset.csv

Usage:
  py -m bc_approach.build_dataset 1
'''

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from utilities.config import DATA_ROOT, JOINT_COLS
from utilities.split_utils import split_name, select_subjects

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_JOINTS = len(JOINT_COLS)  # 16

# How many frames back to compute velocity.
# Must match VELOCITY_LAG in MLP/test_bc.py.
VELOCITY_LAG = 5

# Clip velocity features to this range (degrees).
VEL_CLIP = 10.0

META_COLS   = ['subject', 'exercise', 'video', 'frame', 'timestamp']
STATE_COLS  = [f'q_{j}'  for j in JOINT_COLS]   # current position
VEL_COLS    = [f'dq_{j}' for j in JOINT_COLS]   # q(t) - q(t-VELOCITY_LAG)
ACTION_COLS = [f'act_{j}' for j in JOINT_COLS]  # q(t+1) - q(t)

OUTPUT_COLS = META_COLS + STATE_COLS + VEL_COLS + ACTION_COLS


# ---------------------------------------------------------------------------
# Core: process one joint_ik.csv
# ---------------------------------------------------------------------------
def process_file(path: Path, subject: int, exercise: int, video: int):
    """
    Reads one joint_ik.csv and returns a DataFrame of (state, action) pairs.
    Returns None if the file has fewer than VELOCITY_LAG + 2 frames.
    """
    df = pd.read_csv(path)

    min_frames = VELOCITY_LAG + 2
    if len(df) < min_frames:
        print(f'  [SKIP] {path} - fewer than {min_frames} frames ({len(df)})')
        return None

    if not all(c in df.columns for c in JOINT_COLS):
        print(f'  [SKIP] {path} - missing joint columns')
        return None

    q = df[JOINT_COLS].values  # (N, 16)

    rows = []
    for t in range(VELOCITY_LAG, len(df) - 1):
        state    = q[t]
        velocity = np.clip(q[t] - q[t - VELOCITY_LAG], -VEL_CLIP, VEL_CLIP)
        action   = q[t + 1] - q[t]
        meta = [
            subject, exercise, video,
            int(df.loc[df.index[t], 'frame']),
            float(df.loc[df.index[t], 'timestamp']),
        ]
        rows.append(meta + list(state) + list(velocity) + list(action))

    return pd.DataFrame(rows, columns=OUTPUT_COLS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Build BC dataset for one exercise.')
    parser.add_argument('exercise', type=int, help='Exercise number (e.g. 1)')
    parser.add_argument('--n-demos', type=int, default=55, choices=[10,25,55])
    args = parser.parse_args()

    exercise_num   = args.exercise
    exercise_name  = f'exercise_{exercise_num:03d}'
    landmarks_root = DATA_ROOT / 'landmarks'
    output_dir     = DATA_ROOT / 'dataset' / exercise_name
    output_dir.mkdir(parents=True, exist_ok=True)
    split_d     = output_dir / split_name(args.n_demos)
    split_d.mkdir(parents=True, exist_ok=True)
    output_path = split_d / 'bc_dataset.csv'

    print(f'Building BC dataset for {exercise_name}')
    print(f'Scanning: {landmarks_root}\n')

    pattern   = f'*/exercise_{exercise_num:03d}/*/joint_ik.csv'
    all_files = sorted(landmarks_root.glob(pattern))
    subj_dirs = sorted({p.parent.parent.parent for p in all_files})
    selected  = select_subjects(subj_dirs, args.n_demos)
    all_files = [p for p in all_files if p.parent.parent.parent in selected]

    if not all_files:
        print(f'No joint_ik.csv files found for {exercise_name}.')
        print(f'Expected pattern: {landmarks_root / pattern}')
        return

    print(f'Found {len(all_files)} file(s):\n')

    all_dfs      = []
    total_frames = 0

    for path in all_files:
        parts = path.parts
        try:
            subject_num = int(parts[-4].split('_')[-1])
            video_num   = int(parts[-2].split('_')[-1])
        except (IndexError, ValueError):
            print(f'  [SKIP] {path} - unexpected folder structure')
            continue

        print(f'  subject={subject_num:03d}  video={video_num:03d}  →  {path.name}')
        df = process_file(path, subject_num, exercise_num, video_num)

        if df is not None:
            all_dfs.append(df)
            total_frames += len(df)
            print(f'           {len(df)} samples')

    if not all_dfs:
        print('\nNo valid data found. Dataset not saved.')
        return

    dataset = pd.concat(all_dfs, ignore_index=True)
    dataset.to_csv(output_path, index=False)

    print(f'\n{"="*55}')
    print(f'Dataset saved → {output_path}')
    print(f'  Files processed : {len(all_dfs)}')
    print(f'  Total samples   : {total_frames}')
    print(f'  Columns         : {len(OUTPUT_COLS)}')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()