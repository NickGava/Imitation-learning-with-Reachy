"""
split_utils.py
=============================================================================
Helper functions per l'ablation study sul numero di dimostrazioni.

11 soggetti x 5 demo = 55 demo totali per esercizio.
I soggetti sono selezionati in ordine alfabetico/numerico (deterministico).

Splits:
  n_10  ->  2 soggetti  (10 demo)
  n_25  ->  5 soggetti  (25 demo)
  n_55  -> 11 soggetti  (55 demo)  <- comportamento di default
"""

from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
N_DEMOS_SPLITS: List[int] = [10, 25, 55]

SUBJECTS_PER_SPLIT = {
    10:  2,
    25:  5,
    55: 11,
}


# ---------------------------------------------------------------------------
# Funzioni helper
# ---------------------------------------------------------------------------

def split_name(n_demos: int) -> str:
    """
    Restituisce il nome della cartella per un dato numero di demo.
      split_name(10) -> 'n_10'
      split_name(25) -> 'n_25'
      split_name(55) -> 'n_55'
    """
    return f'n_{n_demos:02d}'


def n_subjects(n_demos: int) -> int:
    """Numero di soggetti corrispondenti a n_demos."""
    if n_demos not in SUBJECTS_PER_SPLIT:
        raise ValueError(
            f'n_demos={n_demos} non supportato. '
            f'Valori validi: {list(SUBJECTS_PER_SPLIT.keys())}')
    return SUBJECTS_PER_SPLIT[n_demos]


def select_subjects(all_subject_dirs: List[Path], n_demos: int) -> List[Path]:
    """
    Restituisce i primi K subject_dir (ordinati alfabeticamente).
    K = SUBJECTS_PER_SPLIT[n_demos].

    Parametri
    ----------
    all_subject_dirs : lista di Path a cartelle subject_XXX
    n_demos          : 10, 25 o 55

    Ritorna
    -------
    Lista di Path ridotta ai primi K soggetti.
    """
    k      = n_subjects(n_demos)
    sorted_dirs = sorted(all_subject_dirs, key=lambda p: p.name)
    selected    = sorted_dirs[:k]
    if len(selected) < k:
        print(f'  [WARN] Richiesti {k} soggetti ma trovati solo {len(selected)}.')
    return selected


def get_split_dir(exercise_dir: Path, n_demos: int) -> Path:
    """
    Ritorna la sottocartella dello split per un dato esercizio.
      get_split_dir(Path('.../exercise_021'), 10) -> Path('.../exercise_021/n_10')
    """
    return exercise_dir / split_name(n_demos)