'''
evaluate.py
=============================================================================
Entry point principale del modulo evaluation_and_comparison.

Raccoglie gli argomenti CLI e dispatcha alle analisi specifiche:
  --exercise N [N ...]  →  evaluate_exercise.run_exercise_evaluation()
  --analysis modality   →  evaluate_modality.run_modality_analysis()
  --analysis demos      →  evaluate_demos.run_demos_analysis()
  --all                 →  tutti gli esercizi trovati + analisi modality

Uso:
    py -m evaluation_and_comparison.evaluate --exercise 1
    py -m evaluation_and_comparison.evaluate --exercise 1 11 21
    py -m evaluation_and_comparison.evaluate --analysis modality
    py -m evaluation_and_comparison.evaluate --all
    py -m evaluation_and_comparison.evaluate --all --steps 200
'''

import argparse

from utilities.config import DATA_ROOT

from evaluation_and_comparison.evaluate_exercise import run_exercise_evaluation
from evaluation_and_comparison.evaluate_modality import run_modality_analysis
from evaluation_and_comparison.evaluate_demos    import run_demos_analysis


def main():
    parser = argparse.ArgumentParser(
        description='Valutazione sistema LfD su Reachy (MLP / GRU / Transformer).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Esempi:
  py -m evaluation_and_comparison.evaluate --exercise 1
  py -m evaluation_and_comparison.evaluate --exercise 1 11 21
  py -m evaluation_and_comparison.evaluate --analysis modality
  py -m evaluation_and_comparison.evaluate --analysis demos --exercise 1
  py -m evaluation_and_comparison.evaluate --all
        ''',
    )
    parser.add_argument(
        '--exercise', type=int, nargs='+', default=None, metavar='N',
        help='Numero/i esercizio da valutare.')
    parser.add_argument(
        '--analysis', type=str, default=None, choices=['modality', 'demos'],
        help='"modality": Stereo vs Mixed vs Mono.  "demos": sensibilità al numero di demo.')
    parser.add_argument(
        '--all', action='store_true',
        help='Valuta tutti gli esercizi trovati + analisi modality.')
    parser.add_argument(
        '--steps', type=int, default=None,
        help='Override numero di step BC. Default: lunghezza baseline.csv.')
    args = parser.parse_args()

    # --- Risolvi lista esercizi ---------------------------------------------
    if args.all:
        dataset_root  = DATA_ROOT / 'dataset'
        exercise_nums = sorted(
            int(d.name.split('_')[1])
            for d in dataset_root.glob('exercise_???')
            if d.is_dir()
        )
    else:
        exercise_nums = args.exercise or []

    # --- Dispatch -----------------------------------------------------------
    if exercise_nums:
        for ex in exercise_nums:
            run_exercise_evaluation(ex, args.steps)

    if args.all or args.analysis == 'modality':
        run_modality_analysis()

    if args.analysis == 'demos':
        if not exercise_nums:
            parser.error('--analysis demos richiede anche --exercise N.')
        run_demos_analysis(exercise_nums, n_steps=args.steps)

    if not exercise_nums and not args.analysis and not args.all:
        parser.print_help()

    print('\nDone.')


if __name__ == '__main__':
    main()