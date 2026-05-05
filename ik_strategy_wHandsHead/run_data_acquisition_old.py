'''
run_data_acquisition.py
=============================================================================
Runs all processing steps in order for one or more videos.

Steps (in order):
  1. pose_estimation.py / pose_estimation_stereo.py  (auto-detected)
  2. data_cleaning.py
  3. face_processing.py
  4. hand_processing.py
  5. mapping.py
  6. run_ik.py

Stereo auto-detection:
  video_XXX.mp4        -> pose_estimation.py
  video_XXX_L/R.mp4   -> pose_estimation_stereo.py

Usage (single video):
  python run_data_acquisition.py
  python run_data_acquisition.py --subject 1 --exercise 2 --video 3
  python run_data_acquisition.py --subject 1 --exercise 2 --video 3 --start 2 --stop 5

Usage (batch):
  When prompted, type 'a' instead of a number to process all:
    Subject  'a' -> all subjects  (exercise still asked)
    Exercise 'a' -> all exercises under the subject(s), all videos
    Video    'a' -> all videos under the subject/exercise
'''

import argparse
import builtins
import importlib
import sys
import time
import traceback
from pathlib import Path

from config import DATA_ROOT

# adds data_acquisition directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "data_acquisition"))

# Pipeline steps — step 1 module is filled in dynamically per video
STEPS = [
    (1, "Pose Estimation",  None),          # module_name resolved at runtime
    (2, "Data Cleaning",    "data_cleaning_old"),
    (3, "Face Processing",  "face_processing"),
    (4, "Hand Processing",  "hand_processing"),
    (5, "Mapping",          "mapping_old"),
    (6, "IK Solver",        "run_ik_old"),
]

RAW_ROOT = DATA_ROOT / "raw_data"


# ---------------------------------------------------------------------------
# Stereo auto-detection
# ---------------------------------------------------------------------------
def _detect_stereo(subject_name: str, exercise_name: str, video_name: str) -> bool:
    """Returns True if _L/_R video pair exists, False if mono .mp4 exists."""
    ex_dir = RAW_ROOT / subject_name / exercise_name
    return (ex_dir / f"{video_name}_L.mp4").exists()


# ---------------------------------------------------------------------------
# Target collection (crawling)
# ---------------------------------------------------------------------------
def _all_subjects():
    return sorted(d.name for d in RAW_ROOT.iterdir()
                  if d.is_dir() and d.name.startswith('subject_'))

def _all_exercises(subject_name: str):
    return sorted(d.name for d in (RAW_ROOT / subject_name).iterdir()
                  if d.is_dir() and d.name.startswith('exercise_'))

def _all_videos(subject_name: str, exercise_name: str):
    """Returns unique video base names (strips _L/_R suffix)."""
    ex_dir = RAW_ROOT / subject_name / exercise_name
    names = set()
    for f in ex_dir.iterdir():
        if f.suffix != '.mp4':
            continue
        stem = f.stem
        if stem.endswith('_L') or stem.endswith('_R'):
            stem = stem[:-2]
        names.add(stem)
    return sorted(names)

def _collect_targets(subject_in, exercise_in, video_in) -> list:
    """
    Returns list of (subject_name, exercise_name, video_name, use_stereo).
    Inputs are either a zero-padded name string or 'a' (all).
    """
    subjects = _all_subjects() if subject_in == 'a' else [subject_in]
    targets  = []

    for subject in subjects:
        if not (RAW_ROOT / subject).exists():
            print(f"[SKIP] {subject} not found in raw_data.")
            continue

        exercises = _all_exercises(subject) if exercise_in == 'a' else [exercise_in]

        for exercise in exercises:
            if not (RAW_ROOT / subject / exercise).exists():
                print(f"[SKIP] {subject}/{exercise} not found.")
                continue

            videos = _all_videos(subject, exercise) if video_in == 'a' else [video_in]

            for video in videos:
                stereo = _detect_stereo(subject, exercise, video)
                # Verify at least one video file exists
                ex_dir = RAW_ROOT / subject / exercise
                mono_exists   = (ex_dir / f"{video}.mp4").exists()
                stereo_exists = (ex_dir / f"{video}_L.mp4").exists()
                if not mono_exists and not stereo_exists:
                    print(f"[SKIP] No video file found for {subject}/{exercise}/{video}.")
                    continue
                targets.append((subject, exercise, video, stereo))

    return targets


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------
def _ask(prompt: str) -> str:
    """Reads input, strips whitespace, lowercases. Returns raw string or 'a'."""
    return input(prompt).strip().lower()

def _parse_inputs(args) -> tuple:
    """
    Returns (subject_name_or_'a', exercise_name_or_'a', video_name_or_'a').
    Accepts CLI args or interactive prompts. 'a' propagates downward.
    """
    # --- Subject ---
    if args.subject is not None:
        subject_raw = str(args.subject)
    else:
        subject_raw = _ask("Subject number (or 'a' for all):  ")

    subject_all = (subject_raw == 'a')
    subject_out = 'a' if subject_all else f"subject_{int(subject_raw):03d}"

    # --- Exercise ---
    if args.exercise is not None:
        exercise_raw = str(args.exercise)
    else:
        exercise_raw = _ask("Exercise number (or 'a' for all): ")

    exercise_all = (exercise_raw == 'a')
    exercise_out = 'a' if exercise_all else f"exercise_{int(exercise_raw):03d}"

    # --- Video (not asked if exercise='a') ---
    if exercise_all:
        video_out = 'a'
    elif args.video is not None:
        video_out = f"video_{int(args.video):03d}"
    else:
        video_raw = _ask("Video number (or 'a' for all):    ")
        video_out = 'a' if video_raw == 'a' else f"video_{int(video_raw):03d}"

    return subject_out, exercise_out, video_out


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------
def _make_input_stub(subject_name: str, exercise_name: str, video_name: str):
    """Patches builtins.input so pipeline modules receive answers automatically."""
    s = subject_name.replace('subject_', '')
    e = exercise_name.replace('exercise_', '')
    v = video_name.replace('video_', '')
    answers = iter([s, e, v])

    def _fake_input(prompt=""):
        value = next(answers)
        print(f"{prompt}{value}")
        return value

    return _fake_input


def _run_step(step_name: str, module_name: str,
              subject: str, exercise: str, video: str) -> bool:
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

        print(f"\n  [OK] {step_name} completed in {time.time() - t0:.1f}s")
        return True

    except Exception as e:
        print(f"\n  [FAIL] {step_name} after {time.time() - t0:.1f}s")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False

    finally:
        builtins.input = original_input


def _run_pipeline(subject: str, exercise: str, video: str,
                  use_stereo: bool, start: int, stop: int) -> dict:
    """Runs the full pipeline for a single video. Returns {step_num: ok}."""

    step1_module = "pose_estimation_stereo_old" if use_stereo else "pose_estimation_old"
    step1_label  = "Pose Estimation (stereo)" if use_stereo else "Pose Estimation (mono)"

    steps = [(num, name if num != 1 else step1_label,
              module if num != 1 else step1_module)
             for num, name, module in STEPS]

    results = {}
    for step_num, step_name, module_name in steps:
        if step_num < start or step_num > stop:
            print(f"\n[SKIP] Step {step_num}: {step_name}")
            continue

        ok = _run_step(step_name, module_name, subject, exercise, video)
        results[step_num] = ok

        if not ok:
            print(f"\n  Pipeline stopped at step {step_num}.")
            break

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Full landmark-to-IK pipeline. Use 'a' for batch mode.")
    parser.add_argument("--subject",  type=int, default=None)
    parser.add_argument("--exercise", type=int, default=None)
    parser.add_argument("--video",    type=int, default=None)
    parser.add_argument("--start", type=int, default=1, metavar="N",
                        help="Start from step N (1-6). Default: 1")
    parser.add_argument("--stop",  type=int, default=6, metavar="N",
                        help="Stop after step N (1-6). Default: 6")
    args = parser.parse_args()

    try:
        subject_out, exercise_out, video_out = _parse_inputs(args)
    except (ValueError, StopIteration):
        print("Error: invalid input.")
        sys.exit(1)

    targets = _collect_targets(subject_out, exercise_out, video_out)

    if not targets:
        print("No valid targets found. Check subject/exercise/video numbers.")
        sys.exit(1)

    print(f"\nTargets found: {len(targets)}")
    for s, e, v, stereo in targets:
        mode = "stereo" if stereo else "mono"
        print(f"  {s} / {e} / {v}  [{mode}]")
    print(f"Steps: {args.start} -> {args.stop}\n")

    t_total   = time.time()
    all_results = {}

    for s, e, v, stereo in targets:
        label = f"{s} / {e} / {v}"
        print(f"\n{'#'*60}")
        print(f"  PROCESSING: {label}")
        print(f"{'#'*60}")

        res = _run_pipeline(s, e, v, stereo, args.start, args.stop)
        all_results[label] = res

    # --- Global summary ---
    print(f"\n{'='*60}")
    print(f"  GLOBAL SUMMARY  (total: {time.time() - t_total:.1f}s)")
    print(f"{'='*60}")
    for label, res in all_results.items():
        failed = [n for n, ok in res.items() if not ok]
        status = "OK" if not failed else f"FAIL at step(s) {failed}"
        print(f"  [{status}]  {label}")

    print()


if __name__ == "__main__":
    main()