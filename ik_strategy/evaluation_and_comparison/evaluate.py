"""
Entry point for evaluating the LfD system on Reachy.

Supports three evaluation modes:
  - Per-exercise: DBA/BC metrics for one or more exercises and demo splits
  - Modality analysis: stereo vs. monocular comparison at a fixed split
  - Demos analysis: effect of the number of demonstrations on performance

Usage:
  py -m evaluation_and_comparison.evaluate --exercise 1
  py -m evaluation_and_comparison.evaluate --all
  py -m evaluation_and_comparison.evaluate --analysis modality
  py -m evaluation_and_comparison.evaluate --analysis demos --exercise 21
"""
import argparse

from utilities.config import DATA_ROOT
from utilities.split_utils import N_DEMOS_SPLITS

from evaluation_and_comparison.evaluate_exercise import run_exercise_evaluation
from evaluation_and_comparison.evaluate_modality import run_modality_analysis
from evaluation_and_comparison.evaluate_demos    import run_demos_analysis


def main():
    parser = argparse.ArgumentParser(description='Evaluation system LfD on Reachy.', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exercise", type=int, nargs="+", default=None, metavar="N")
    parser.add_argument("--n-demos", type=int, nargs="+", default=[55], choices=N_DEMOS_SPLITS, help="Split to be evalueted (default: 55). Possible choices: --n-demos 10 25 55")
    parser.add_argument("--analysis", type=str, default=None, choices=["modality", "demos"])
    parser.add_argument("--all", action="store_true", help="Evaluate all exercises + analisis modality.")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    if args.all:
        dataset_root  = DATA_ROOT / "dataset"
        exercise_nums = sorted(
            int(d.name.split("_")[1])
            for d in dataset_root.glob("exercise_???")
            if d.is_dir()
        )
    else:
        exercise_nums = args.exercise or []

    if exercise_nums:
        for n_demos in args.n_demos:
            for ex in exercise_nums:
                run_exercise_evaluation(ex, n_demos, args.steps)

    if args.all or args.analysis == "modality":
        n = args.n_demos[0] if args.n_demos else 55
        run_modality_analysis(n)

    if args.analysis == "demos":
        if not exercise_nums:
            parser.error("--analysis demos asks also --exercise N.")
        run_demos_analysis(exercise_nums)

    if not exercise_nums and not args.analysis and not args.all:
        parser.print_help()

    print("\nDone.")


if __name__ == "__main__":
    main()