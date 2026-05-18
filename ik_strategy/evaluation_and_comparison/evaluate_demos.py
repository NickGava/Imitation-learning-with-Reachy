'''
evaluate_demos.py
=============================================================================
Analisi della sensibilità al numero di dimostrazioni disponibili.

Risponde alla domanda: quante demo bastano? Il sistema degrada sensibilmente
se si scende da 20 a 10 o a 5 dimostrazioni?

Strategia:
  Per ogni esercizio e ogni architettura, riaddestra (o ri-valuta) il modello
  BC usando sottoinsiemi crescenti di demo (n_demos = 5, 10, 20).
  Calcola le metriche per ogni configurazione e confronta.

Output (in data/evaluation_demos/):
    results_by_n_demos.csv   — una riga per (n_demos × architettura × esercizio)
    plot_dtw_vs_demos.png
    plot_rmse_vs_demos.png

Uso standalone:
    py -m evaluation_and_comparison.evaluate_demos --exercise 1
    py -m evaluation_and_comparison.evaluate_demos --exercise 1 --n-demos 5 10 20

TODO: questa analisi richiede di decidere se:
  (a) usare modelli già addestrati su sottoinsiemi (richiede riaddestrare con
      bc_approach/*/train_bc.py con un flag --max-demos N), oppure
  (b) valutare solo la canonical (compute_canonical con N demo) senza
      riaddestrare il BC.
  L'opzione (b) è più immediata; l'opzione (a) è più rigorosa per il BC.
  Aggiornare questo file una volta presa la decisione.
'''

import argparse
from typing import List, Optional

# from utilities.config import DATA_ROOT
# from evaluation_and_comparison._config  import ARCHITECTURES
# from evaluation_and_comparison._io      import ...
# from evaluation_and_comparison._metrics import compute_metrics
# from evaluation_and_comparison._plots   import ...

N_DEMOS_DEFAULT: List[int] = [5, 10, 20]


def run_demos_analysis(exercise_nums: List[int],
                       n_demos_list: Optional[List[int]] = None,
                       n_steps: Optional[int] = None) -> None:
    '''
    [DA IMPLEMENTARE]

    Per ogni esercizio in exercise_nums e per ogni n in n_demos_list,
    valuta le metriche BC usando solo n dimostrazioni scelte casualmente
    (media su più seed per robustezza statistica).
    '''
    if n_demos_list is None:
        n_demos_list = N_DEMOS_DEFAULT

    raise NotImplementedError(
        'evaluate_demos non ancora implementato. '
        'Vedere TODO nel docstring del modulo.')


# ============================================================================
# Standalone
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analisi sensibilità al numero di dimostrazioni.')
    parser.add_argument(
        '--exercise', type=int, nargs='+', required=True, metavar='N',
        help='Numero/i esercizio da analizzare.')
    parser.add_argument(
        '--n-demos', type=int, nargs='+', default=N_DEMOS_DEFAULT, metavar='K',
        help=f'Sottoinsiemi di demo da testare (default: {N_DEMOS_DEFAULT}).')
    parser.add_argument(
        '--steps', type=int, default=None,
        help='Override numero di step BC.')
    args = parser.parse_args()

    run_demos_analysis(args.exercise, args.n_demos, args.steps)
    print('\nDone.')


if __name__ == '__main__':
    main()
