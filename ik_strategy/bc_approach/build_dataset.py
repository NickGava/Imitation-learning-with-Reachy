'''
build_dataset.py
=============================================================================
Builds the state-action pair dataset for Behavioral Cloning training.

For each exercise, crawls all joint_ik.csv files and converts them into
state-action pairs. One output file is produced per exercise.

State-action formulation:
  state  = q(i)            shape: (16,)   joint angles
  vel    = q(i) - q(i-1)   shape: (16,)   velocity
  action = q(i+1) - q(i)   shape: (16,)   delta to apply (model target)

Head gaze (head_x/y/z) is included in the state as optional context.
By default NOT included in the action (head handled via lookAt).
Use --head-action to predict head deltas too — needed for neck exercises.
  Default:         state=[q,vel,head] 35v  action=[Dq] 16v
  --head-action:   state=[q,vel,head] 35v  action=[Dq,Dhead] 19v
  --no-head:       state=[q,vel]      32v  action=[Dq] 16v

Input: data/landmarks/subject_XXX/exercise_XXX/video_XXX/joint_ik.csv

Output: data/dataset/exercise_XXX/bc_dataset.csv

Usage:
  # process all exercises, all subjects
  python build_dataset.py

  # only exercise 1
  python build_dataset.py --exercise 1

  # only subject 2, exercise 1
  python build_dataset.py --exercise 1 --subject 2

  # exclude head gaze from state
  python build_dataset.py --no-head
'''

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from config import DATA_ROOT, JOINT_COLS, HEAD_COLS

N_JOINTS  = len(JOINT_COLS)   # 16

# ---------------------------------------------------------------------------
# Per-video builder
# ---------------------------------------------------------------------------
def _build_pairs_from_video(df: pd.DataFrame, subject: int, exercise: int, video: int, include_head: bool, head_action: bool) -> pd.DataFrame:
    '''
    Converts a single joint_ik.csv DataFrame into state-action pairs.

    Parameters:
        df           : joint_ik.csv loaded as a DataFrame
        subject      : subject number (for metadata column)
        exercise     : exercise number (for metadata column)
        video        : video number (for metadata column)
        include_head : whether to include head gaze columns in the state
        head_action  : whether to include head gaze delta in the action

    Returns:
        DataFrame with one row per valid frame (frames 1 to N-2), or empty DataFrame if fewer than 3 usable frames.
    '''
    # Drop rows where any joint column is NaN (both arms must be present)
    df = df.dropna(subset=JOINT_COLS).reset_index(drop=True)

    n = len(df)
    if n < 3:
        print(f'    [SKIP] only {n} clean frames (need >= 3)')
        return pd.DataFrame()

    q = df[JOINT_COLS].values   # shape: (N, 16)

    # Valid indices: 1 to N-2 inclusive
    idx = np.arange(1, n - 1)

    q_curr = q[idx]             # pose at i        shape: (N-2, 16)
    q_prev = q[idx - 1]         # pose at i-1      shape: (N-2, 16)
    q_next = q[idx + 1]         # pose at i+1      shape: (N-2, 16)

    vel    = q_curr - q_prev    # finite-difference velocity
    action = q_next - q_curr    # target delta

    # Assemble output DataFrame
    out = pd.DataFrame({
        'subject':   subject,
        'exercise':  exercise,
        'video':     video,
        'frame':     df['frame'].values[idx],
        'timestamp': df['timestamp'].values[idx],
    })

    # State: absolute pose
    for j, col in enumerate(JOINT_COLS):
        out[f'state_{col}'] = q_curr[:, j]

    # State: approximate velocity
    for j, col in enumerate(JOINT_COLS):
        out[f'vel_{col}'] = vel[:, j]

    # Optional: head gaze context (not predicted, just conditioning)
    if include_head:
        for col in HEAD_COLS:
            if col in df.columns:
                out[col] = df[col].values[idx]
            else:
                out[col] = np.nan

    # Action: joint deltas (degrees)
    for j, col in enumerate(JOINT_COLS):
        out[f'action_{col}'] = action[:, j]

    # Action: head gaze delta (meters), only if --head-action
    if head_action:
        for col in HEAD_COLS:
            if col in df.columns:
                head_curr = df[col].values[idx]
                head_next = df[col].values[idx + 1]
                out[f'action_{col}'] = head_next - head_curr
            else:
                out[f'action_{col}'] = np.nan

    return out


# ---------------------------------------------------------------------------
# Per-exercise processor
# ---------------------------------------------------------------------------
def _process_exercise(exercise_num: int, landmarks_root: Path, dataset_root: Path, filter_subject: Optional[int], include_head: bool, head_action: bool,) -> None:
    '''
    Crawls all subjects and videos for a single exercise, builds the
    state-action dataset and saves it to disk.

    Parameters:
        exercise_num   : exercise number to process
        landmarks_root : root of the landmarks folder
        dataset_root   : root of the dataset output folder
        filter_subject : if set, only process this subject number
        include_head   : whether to include head gaze in the state
        head_action    : whether to include head gaze delta in the action
    '''
    exercise_name = f'exercise_{exercise_num:03d}'
    print(f'\n{"="*60}')
    print(f'  Exercise {exercise_num:03d}')
    print(f'{"="*60}')

    all_dfs   = []
    n_videos  = 0
    n_skipped = 0

    subject_dirs = sorted(landmarks_root.glob('subject_*'))
    if not subject_dirs:
        print(f'  No subject folders found under {landmarks_root}')
        return

    for subj_dir in subject_dirs:
        subject_num = int(subj_dir.name.split('_')[1])

        if filter_subject is not None and subject_num != filter_subject:
            continue

        exer_dir = subj_dir / exercise_name
        if not exer_dir.is_dir():
            continue

        for video_dir in sorted(exer_dir.glob('video_*')):
            video_num = int(video_dir.name.split('_')[1])
            ik_path   = video_dir / 'joint_ik.csv'

            print(f'  subject_{subject_num:03d} / {exercise_name} / '
                  f'video_{video_num:03d} ... ', end='', flush=True)

            if not ik_path.exists():
                print('MISSING joint_ik.csv')
                n_skipped += 1
                continue

            df    = pd.read_csv(ik_path)
            pairs = _build_pairs_from_video(df, subject_num, exercise_num, video_num, include_head, head_action)

            if pairs.empty:
                n_skipped += 1
                continue

            print(f'{len(pairs)} samples')
            all_dfs.append(pairs)
            n_videos += 1

    if not all_dfs:
        print(f'  No data collected for exercise {exercise_num:03d}.')
        return

    dataset = pd.concat(all_dfs, ignore_index=True)

    # Save
    output_dir  = dataset_root / exercise_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'bc_dataset.csv'
    dataset.to_csv(output_path, index=False)

    # Summary
    state_dim  = N_JOINTS * 2 + (3 if include_head else 0)
    action_dim = N_JOINTS + (3 if head_action else 0)
    print(f'\n  Videos processed : {n_videos}  |  skipped: {n_skipped}')
    print(f'  Total samples    : {len(dataset)}')
    print(f'  Subjects         : {sorted(dataset["subject"].unique())}')
    print(f'  State dimension  : {state_dim}  '
          f'({N_JOINTS} pose + {N_JOINTS} vel'
          f'{" + 3 head" if include_head else ""})')
    print(f'  Action dimension : {action_dim}  '
          f'({N_JOINTS} joints{" + 3 head" if head_action else ""})')
    print(f'  Saved -> {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Build state-action BC dataset from joint_ik.csv files.')
    parser.add_argument('--exercise', type=int, default=None, help='Only process this exercise number. '
             'Default: all exercises found in landmarks/.')
    parser.add_argument('--subject', type=int, default=None, help='Only process this subject number. Default: all subjects.')
    parser.add_argument('--no-head', action='store_true', help='Exclude head gaze columns (head_x/y/z) from the state.')
    parser.add_argument('--head-action', action='store_true', help='Include head gaze delta (action_head_x/y/z) in the action. '
             'Required for neck exercises. Automatically includes head in the state.')
    args = parser.parse_args()

    landmarks_root = DATA_ROOT / 'landmarks'
    dataset_root   = DATA_ROOT / 'dataset'
    if not landmarks_root.is_dir():
        print(f'Error: landmarks folder not found -> {landmarks_root}')
        return

    head_action  = args.head_action
    include_head = (not args.no_head) or head_action   # head in state implied by --head-action
    print(f'Landmarks root  : {landmarks_root}')
    print(f'Dataset root    : {dataset_root}')
    print(f'Head in state   : {include_head}')
    print(f'Head in action  : {head_action}')

    # Collect exercise numbers to process
    if args.exercise is not None:
        exercise_nums = [args.exercise]
    else:
        # Auto-discover all exercises present in any subject folder
        found = set()
        for subj_dir in landmarks_root.glob('subject_*'):
            for exer_dir in subj_dir.glob('exercise_*'):
                found.add(int(exer_dir.name.split('_')[1]))
        exercise_nums = sorted(found)

    if not exercise_nums:
        print('No exercises found. Check that joint_ik.csv files exist.')
        return

    print(f'Exercises found: {exercise_nums}')

    # Process each exercise independently
    for exercise_num in exercise_nums:
        _process_exercise(
            exercise_num   = exercise_num,
            landmarks_root = landmarks_root,
            dataset_root   = dataset_root,
            filter_subject = args.subject,
            include_head   = include_head,
            head_action    = head_action,
        )

    print(f'\n{"="*60}')
    print('  Done.')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()