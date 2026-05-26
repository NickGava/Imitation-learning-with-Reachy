"""
split_utils.py
=============================================================================
Helper functions for ablation study on numbre of demonstrations.

11 subjects x 5 demo = 55 total demo per exercise.
Subjects are selected in numerical order (deterministic).

Splits:
  n_10  ->  2 subjects  (10 demo)
  n_25  ->  5 subjects  (25 demo)
  n_55  -> 11 subjects  (55 demo)  <- default behaviour
"""

from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_DEMOS_SPLITS: List[int] = [10, 25, 55]

SUBJECTS_PER_SPLIT = {
    10:  2,
    25:  5,
    55: 11,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_name(n_demos: int) -> str:
    """
    Returns the name of the folder for a given number of demo.
      split_name(10) -> 'n_10'
      split_name(25) -> 'n_25'
      split_name(55) -> 'n_55'
    """
    return f'n_{n_demos:02d}'


def n_subjects(n_demos: int) -> int:
    """Number of subjects corresponding to n_demos."""
    if n_demos not in SUBJECTS_PER_SPLIT:
        raise ValueError(
            f'n_demos={n_demos} not supported. '
            f'Valid values: {list(SUBJECTS_PER_SPLIT.keys())}')
    return SUBJECTS_PER_SPLIT[n_demos]


def select_subjects(all_subject_dirs: List[Path], n_demos: int) -> List[Path]:
    """
    Returns the first K subject_dir (numerical order)
    K = SUBJECTS_PER_SPLIT[n_demos].

    Parameters
    ----------
    all_subject_dirs : list if Path to subject_XXX folders
    n_demos          : 10, 25 or 55

    Returns
    -------
    List of Path reduced to the first K subjects.
    """
    k      = n_subjects(n_demos)
    sorted_dirs = sorted(all_subject_dirs, key=lambda p: p.name)
    selected    = sorted_dirs[:k]
    if len(selected) < k:
        print(f'  [WARN] Requested {k} subjects, but found only {len(selected)}.')
    return selected


def get_split_dir(exercise_dir: Path, n_demos: int) -> Path:
    """
    Returns the subfolder of the split for a given exercise.
      get_split_dir(Path('.../exercise_021'), 10) -> Path('.../exercise_021/n_10')
    """
    return exercise_dir / split_name(n_demos)