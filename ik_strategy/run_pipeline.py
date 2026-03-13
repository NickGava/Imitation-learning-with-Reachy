'''
run_pipeline.py
=============================================================================
Master pipeline: runs all processing steps in order for a single video.

Steps (in order):
  1. pose_estimation.py   -> pose.csv, right_hand.csv, left_hand.csv
  2. data_cleaning.py     -> *_cleaned.csv
  3. face_processing.py   -> face_features.csv
  4. hand_processing.py   -> *_mapped.csv
  5. mapping.py           -> arms_mapped.csv
  6. run_ik.py            -> joint_ik.csv

Usage:
  python run_pipeline.py
    (inserisci subject/exercise/video una sola volta)

  python run_pipeline.py --subject 1 --exercise 2 --video 3
    (passa i valori direttamente)

  python run_pipeline.py --subject 1 --exercise 2 --video 3 --start 2
    (riprendi dallo step 2, salta pose_estimation)

  python run_pipeline.py --subject 1 --exercise 2 --video 3 --start 4 --stop 5
    (esegui solo gli step 4 e 5)
'''

import argparse
import builtins
import importlib
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Utility: patch input() so each module gets the right values automatically
# ---------------------------------------------------------------------------
def _make_input_stub(subject: int, exercise: int, video: int):
    answers = iter([str(subject), str(exercise), str(video)])

    def _fake_input(prompt=""):
        value = next(answers)
        print(f"{prompt}{value}")
        return value

    return _fake_input


def _run_step(step_name: str, module_name: str,
              subject: int, exercise: int, video: int) -> bool:
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}  [{module_name}.py]")
    print(f"{'='*60}")
    t0 = time.time()

    original_input = builtins.input
    try:
        builtins.input = _make_input_stub(subject, exercise, video)

        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)

        module.main()

        print(f"\n  [OK] {step_name} completato in {time.time() - t0:.1f}s")
        return True

    except Exception as e:
        print(f"\n  [FAIL] {step_name} dopo {time.time() - t0:.1f}s")
        print(f"  Errore: {e}")
        traceback.print_exc()
        return False

    finally:
        builtins.input = original_input


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
STEPS = [
    (1, "Pose Estimation",  "pose_estimation"),
    (2, "Data Cleaning",    "data_cleaning"),
    (3, "Face Processing",  "face_processing"),
    (4, "Hand Processing",  "hand_processing"),
    (5, "Mapping",          "mapping"),
    (6, "IK Solver",        "run_ik"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Full landmark-to-IK pipeline for one video."
    )
    parser.add_argument("--subject",  type=int, default=None)
    parser.add_argument("--exercise", type=int, default=None)
    parser.add_argument("--video",    type=int, default=None)
    parser.add_argument(
        "--start", type=int, default=1, metavar="N",
        help="Parti dallo step N (1-6). Default: 1"
    )
    parser.add_argument(
        "--stop", type=int, default=6, metavar="N",
        help="Fermati dopo lo step N (1-6). Default: 6"
    )
    args = parser.parse_args()

    try:
        subject  = args.subject  if args.subject  is not None else int(input("Subject number:  ").strip())
        exercise = args.exercise if args.exercise is not None else int(input("Exercise number: ").strip())
        video    = args.video    if args.video    is not None else int(input("Video number:    ").strip())
    except ValueError:
        print("Errore: i valori devono essere interi.")
        sys.exit(1)

    print(f"\nTarget : subject_{subject:03d} / exercise_{exercise:03d} / video_{video:03d}")
    print(f"Steps  : {args.start} -> {args.stop}")

    results = {}
    t_total = time.time()

    for step_num, step_name, module_name in STEPS:
        if step_num < args.start or step_num > args.stop:
            print(f"\n[SKIP] Step {step_num}: {step_name}")
            continue

        ok = _run_step(step_name, module_name, subject, exercise, video)
        results[step_num] = ok

        if not ok:
            print(f"\nPipeline interrotta allo step {step_num}.")
            print(f"Correggi l'errore e riprendi con: --start {step_num}")
            sys.exit(1)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  PIPELINE SUMMARY  (totale: {time.time() - t_total:.1f}s)")
    print(f"{'='*60}")
    for step_num, step_name, _ in STEPS:
        if step_num in results:
            status = "OK  " if results[step_num] else "FAIL"
            print(f"  [{status}] Step {step_num}: {step_name}")
        else:
            print(f"  [----] Step {step_num}: {step_name}  (skipped)")

    try:
        from config import DATA_ROOT
        folder = (DATA_ROOT / "landmarks"
                  / f"subject_{subject:03d}"
                  / f"exercise_{exercise:03d}"
                  / f"video_{video:03d}")
        print(f"\nOutput folder -> {folder}")
        print(f"Final file    -> {folder / 'joint_ik.csv'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
