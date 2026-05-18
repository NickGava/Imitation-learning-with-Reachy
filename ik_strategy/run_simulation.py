'''
run_simulation.py
=============================================================================
Riproduce qualsiasi traiettoria su Reachy leggendo un CSV pre-calcolato.
Nessuna inference live: tutto il calcolo e' gia' stato fatto da test_bc.py.

Modes:
  video       — riproduce joint_ik.csv da landmarks
                (richiede --subject, --exercise, --video)
  baseline    — riproduce baseline.csv
                (richiede --exercise)
  canonical   — riproduce n_XX/canonical.csv
                (richiede --exercise, --n-demos)
  mlp         — riproduce n_XX/MLP/bc_trajectory.csv
                (richiede --exercise, --n-demos)
  gru         — riproduce n_XX/GRU/bc_trajectory.csv
  transformer — riproduce n_XX/Transformer/bc_trajectory.csv

Usage:
  py run_simulation.py
  py run_simulation.py --mode canonical  --exercise 21
  py run_simulation.py --mode mlp        --exercise 21 --n-demos 55
  py run_simulation.py --mode gru        --exercise 21 --n-demos 25 --runs 3
  py run_simulation.py --mode video      --subject 1 --exercise 21 --video 1
  py run_simulation.py --mode baseline   --exercise 21 --host 10.59.1.20
'''

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from reachy_sdk import ReachySDK
from reachyController.reachyController import ReachyController
from reachyController.timeSeries import TimeSeries

from utilities.config import DATA_ROOT, JOINT_COLS
from utilities.split_utils import split_name, N_DEMOS_SPLITS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIMULATOR_HOST  = 'localhost'
MIN_FRAME_DELAY = 0.033    # ~30 Hz
GOTO_DURATION   = 2.0

VALID_MODES = ['video', 'baseline', 'canonical', 'mlp', 'gru', 'transformer']
BC_MODES    = {'mlp', 'gru', 'transformer'}
CSV_MODES   = {'video', 'baseline', 'canonical'}

ALL_ARM_JOINTS = [
    'r_shoulder_pitch', 'r_shoulder_roll', 'r_arm_yaw', 'r_elbow_pitch',
    'r_forearm_yaw', 'r_wrist_pitch', 'r_wrist_roll', 'r_gripper',
    'l_shoulder_pitch', 'l_shoulder_roll', 'l_arm_yaw', 'l_elbow_pitch',
    'l_forearm_yaw', 'l_wrist_pitch', 'l_wrist_roll', 'l_gripper',
]


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _connect(host: str) -> ReachyController:
    print(f'Connecting to Reachy at {host} ...')
    reachy  = ReachySDK(host=host)
    reachyC = ReachyController(reachy)
    reachyC.turnOn()
    print('Connected and motors on.\n')
    return reachyC


def _shutdown(reachyC: ReachyController) -> None:
    print('\nShutting down ...')
    reachyC.turnOffSmooth()
    print('Done.')


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------

def _build_arm_dicts(reachyC: ReachyController, pose: dict) -> tuple:
    right_dict, left_dict = {}, {}
    for col, val in pose.items():
        if col.startswith('r_'):
            joint = reachyC.armRight._joints.get(col)
            if joint is not None:
                right_dict[joint] = float(val)
        elif col.startswith('l_'):
            joint = reachyC.armLeft._joints.get(col)
            if joint is not None:
                left_dict[joint] = float(val)
    return right_dict, left_dict


def _goto_pose(reachyC: ReachyController, pose: dict, duration: float) -> None:
    '''Muove entrambe le braccia simultaneamente verso la posa target.'''
    right_dict, left_dict = _build_arm_dicts(reachyC, pose)
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = []
        if right_dict:
            futures.append(ex.submit(reachyC.armRight._debug_goto, right_dict, duration))
        if left_dict:
            futures.append(ex.submit(reachyC.armLeft._debug_goto, left_dict, duration))
        for f in futures:
            f.result()
    time.sleep(0.2)


def _send_q(reachyC: ReachyController, q: np.ndarray) -> None:
    '''Invia goal_position per tutti i joint (non bloccante, usato a 30Hz).'''
    for i, col in enumerate(JOINT_COLS):
        arm   = reachyC.armRight if col.startswith('r_') else reachyC.armLeft
        joint = arm._joints.get(col)
        if joint is not None:
            joint.goal_position = float(q[i])


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_trajectory(csv_path: Path) -> np.ndarray:
    '''
    Carica un CSV di traiettoria (JOINT_COLS) come array (T, 16).
    Funziona sia con baseline/canonical/joint_ik.csv (che hanno colonne extra)
    sia con bc_trajectory.csv (solo JOINT_COLS, senza timestamp).
    '''
    if not csv_path.exists():
        raise FileNotFoundError(f'Traiettoria non trovata: {csv_path}')
    df   = pd.read_csv(csv_path)
    cols = [c for c in JOINT_COLS if c in df.columns]
    arr  = df[cols].dropna().values.astype(float)
    if len(cols) < len(JOINT_COLS):
        padded = np.zeros((len(arr), len(JOINT_COLS)))
        for i, c in enumerate(cols):
            padded[:, JOINT_COLS.index(c)] = arr[:, i]
        arr = padded
    print(f'  Loaded {len(arr)} frames from {csv_path.name}')
    return arr


def _load_start_pose(csv_path: Path) -> dict:
    '''Primo frame di un CSV come dizionario start pose.'''
    if not csv_path.exists():
        return {}
    first = pd.read_csv(csv_path).iloc[0]
    return {c: float(first[c]) for c in JOINT_COLS if c in first.index}


def _resolve_paths(mode: str, args) -> tuple[Path, dict]:
    '''
    Ritorna (traj_path, start_pose) in base al mode e agli argomenti.
    start_pose e' il dizionario {joint_col: angle} per la posa iniziale.
    '''
    exercise_num = args.exercise or _prompt_exercise()
    exercise_dir = DATA_ROOT / 'dataset' / f'exercise_{exercise_num:03d}'

    if mode == 'video':
        subject  = args.subject or _prompt_subject()
        video    = args.video   or _prompt_video()
        traj_path = (DATA_ROOT / 'landmarks'
                     / f'subject_{subject:03d}'
                     / f'exercise_{exercise_num:03d}'
                     / f'video_{video:03d}'
                     / 'joint_ik.csv')
        start_pose = _load_start_pose(traj_path)

    elif mode == 'baseline':
        traj_path  = exercise_dir / 'baseline.csv'
        start_pose = _load_start_pose(traj_path)

    elif mode == 'canonical':
        split_dir  = exercise_dir / split_name(args.n_demos)
        traj_path  = split_dir / 'canonical.csv'
        start_pose = _load_start_pose(traj_path)

    else:  # mlp / gru / transformer
        arch       = mode.upper()
        split_dir  = exercise_dir / split_name(args.n_demos)
        traj_path  = split_dir / arch / 'bc_trajectory.csv'
        # start pose dalla canonical (stessa usata da test_bc.py)
        canonical_path = split_dir / 'canonical.csv'
        start_pose = (_load_start_pose(canonical_path)
                      if canonical_path.exists()
                      else _load_start_pose(traj_path))

    return traj_path, start_pose


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def _play_trajectory(reachyC: ReachyController,
                      q_traj: np.ndarray,
                      start_pose: dict,
                      runs: int = 1) -> None:
    '''
    Invia la traiettoria al robot frame-by-frame a ~30 Hz.
    Prima di ogni run: goto start_pose.
    Dopo l'ultima run: ritorna a start_pose.
    '''
    n = len(q_traj)
    for run in range(1, runs + 1):
        if runs > 1:
            print(f'\n--- Run {run} / {runs} ---')

        if start_pose:
            print('Moving to start pose ...')
            _goto_pose(reachyC, start_pose, GOTO_DURATION)

        print(f'Sending {n} frames at ~30 Hz ...')
        t_start = time.perf_counter()
        for step in range(n):
            t_target = t_start + step * MIN_FRAME_DELAY
            _send_q(reachyC, q_traj[step])
            remaining = t_target + MIN_FRAME_DELAY - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

        elapsed = time.perf_counter() - t_start
        print(f'Complete. ({elapsed:.1f}s)')

    if start_pose:
        print('\nReturning to start pose ...')
        _goto_pose(reachyC, start_pose, GOTO_DURATION)


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def _prompt_mode() -> str:
    print('Select mode:')
    for i, m in enumerate(VALID_MODES, 1):
        print(f'  {i}. {m}')
    raw = input('Mode (name or number): ').strip().lower()
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(VALID_MODES):
            return VALID_MODES[idx]
    if raw in VALID_MODES:
        return raw
    print(f'[ERROR] Invalid mode: "{raw}"')
    sys.exit(1)


def _prompt_exercise() -> int:
    return int(input('Exercise number: ').strip())


def _prompt_subject() -> int:
    return int(input('Subject number: ').strip())


def _prompt_video() -> int:
    return int(input('Video number: ').strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Riproduce una traiettoria pre-calcolata su Reachy.')
    parser.add_argument('--mode',     choices=VALID_MODES, default=None)
    parser.add_argument('--exercise', type=int, default=None)
    parser.add_argument('--subject',  type=int, default=None,
                        help='Solo per modalita video.')
    parser.add_argument('--video',    type=int, default=None,
                        help='Solo per modalita video.')
    parser.add_argument('--n-demos',  type=int, default=55,
                        choices=N_DEMOS_SPLITS,
                        help='Split da usare per canonical/BC (default: 55).')
    parser.add_argument('--runs',     type=int, default=1,
                        help='Numero di ripetizioni (default: 1).')
    parser.add_argument('--host',     type=str, default=SIMULATOR_HOST,
                        help=f'ReachySDK host (default: {SIMULATOR_HOST}).')
    args = parser.parse_args()

    mode = args.mode or _prompt_mode()

    print(f'\nMode     : {mode}')
    if mode not in ('video',):
        print(f'Exercise : {args.exercise}')
    if mode in BC_MODES or mode == 'canonical':
        print(f'Split    : {split_name(args.n_demos)}')
    print()

    # Risolvi paths
    traj_path, start_pose = _resolve_paths(mode, args)

    # Carica traiettoria
    q_traj = _load_trajectory(traj_path)

    # Connetti e riproduci
    reachyC = _connect(args.host)
    try:
        _play_trajectory(reachyC, q_traj, start_pose, runs=args.runs)
    finally:
        _shutdown(reachyC)


if __name__ == '__main__':
    main()