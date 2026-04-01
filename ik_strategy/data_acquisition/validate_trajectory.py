'''
validate_trajectory.py
-------------------------------------------------------------------------------------
Reads joint_ik.csv and verifies that the trajectory is safe. If violations are found,
the trajectory is discarded and the script exits with code 1.

Two checks, in order:
  CHECK 1 -- Joint limits
    Every angle must stay within hardware limits with an additional safety
    margin (SAFETY_PADDING_DEG).

  CHECK 2 -- Maximum joint velocity
    The velocity of each joint (deg/s) is computed as Δangle / Δtime  between consecutive frames.
    If it exceeds MAX_JOINT_VEL_DEG_S the trajectory contains a movement
    that is too abrupt and is rejected.
    Warmup frames (first N_WARMUP_FRAMES) are excluded from the failure
    check but their peak velocity is reported as a warning.

Output:
  - Text report on stdout with details of each violation.
  - Exit code 0  -> trajectory SAFE.
  - Exit code 1  -> trajectory NOT SAFE (at least one critical violation).
'''

import argparse
import sys
import textwrap
import numpy as np
import pandas as pd
from pathlib import Path


from config import DATA_ROOT, JOINT_LIMITS_DEG, JOINT_COLS

# -------------------------------------------------------------------------------------
# Safety parameters -- edit here to tighten or relax the checks
# -------------------------------------------------------------------------------------
# Additional margin on hardware limits (degrees).
SAFETY_PADDING_DEG: float = 3.0

# Maximum allowed joint velocity (degrees/second).
MAX_JOINT_VEL_DEG_S: float = 1500.0

# Threshold to consider a velocity sample valid 
MIN_DT_S: float = 1e-3   # 1 ms

# Initial frames excluded from the velocity failure check.
N_WARMUP_FRAMES: int = 5

# Strict limits applied by this script
_STRICT_LIMITS = {
    side: JOINT_LIMITS_DEG[side] + np.array([SAFETY_PADDING_DEG, -SAFETY_PADDING_DEG])
    for side in ('right', 'left')
}

# -------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------
def _section(title: str) -> None:
    print(f"\n{'--' * 60}")
    print(f"  {title}")
    print('--' * 60)

def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")

def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")

def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")

# -------------------------------------------------------------------------------------
# CHECK 1 -- Joint limits
# -------------------------------------------------------------------------------------
def check_joint_limits(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Verifies that every angle respects hardware limits with an additional
    SAFETY_PADDING_DEG margin.

    Returns:
        (passed, violations)
        passed     : True if no critical violation found.
        violations : list of descriptive strings for the report.
    """
    violations: list[str] = []

    for col in JOINT_COLS:
        if col.endswith('_gripper') or col not in df.columns:
            continue

        side      = 'right' if col.startswith('r_') else 'left'
        side_cols = [c for c in JOINT_COLS if c.startswith(col[0] + '_') and not c.endswith('_gripper')]
        j_idx     = side_cols.index(col)
        lo, hi    = _STRICT_LIMITS[side][j_idx]

        angles = df[col].dropna().values
        below  = angles[angles < lo]
        above  = angles[angles > hi]

        if len(below) > 0:
            violations.append(
                f"{side:5s} | {col:<20s} | below limit: "
                f"min={below.min():.2f}° < {lo:.2f}°  ({len(below)} frames)"
            )
        if len(above) > 0:
            violations.append(
                f"{side:5s} | {col:<20s} | above limit: "
                f"max={above.max():.2f}° > {hi:.2f}°  ({len(above)} frames)"
            )

    return (len(violations) == 0), violations


# -------------------------------------------------------------------------------------
# CHECK 2 -- Maximum joint velocity
# -------------------------------------------------------------------------------------
def check_joint_velocities(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Computes joint velocity frame-by-frame (Δangle / Δtime) and verifies
    that it does not exceed MAX_JOINT_VEL_DEG_S.

    Warmup frames (first N_WARMUP_FRAMES) are excluded from the failure
    check but their peak velocity is reported as a warning if it exceeds
    the threshold.

    Returns:
        (passed, violations)
    """
    violations: list[str] = []

    timestamps = df['timestamp'].values
    dt_all     = np.diff(timestamps)

    for col in JOINT_COLS:
        if col.endswith('_gripper') or col not in df.columns:
            continue

        side   = 'right' if col.startswith('r_') else 'left'
        angles  = df[col].values
        delta_a = np.diff(angles)

        valid_all = (dt_all >= MIN_DT_S) & ~np.isnan(delta_a)

        # Warmup warning
        warmup_mask = valid_all.copy()
        warmup_mask[N_WARMUP_FRAMES:] = False
        if warmup_mask.any():
            warmup_vels  = np.abs(delta_a[warmup_mask]) / dt_all[warmup_mask]
            v_warmup_max = warmup_vels.max()
            if v_warmup_max > MAX_JOINT_VEL_DEG_S:
                peak_idx = np.where(warmup_mask)[0][warmup_vels.argmax()]
                frame_id = int(df['frame'].iloc[peak_idx + 1])
                violations.append(
                    f"[WARN] {side:5s} | {col:<20s} | warmup vel. peak={v_warmup_max:.1f} °/s "
                    f"> {MAX_JOINT_VEL_DEG_S:.0f} °/s  (frame {frame_id}, warmup -- not a failure)"
                )

        # Main check (post-warmup)
        valid = valid_all.copy()
        valid[:N_WARMUP_FRAMES] = False
        if not valid.any():
            continue

        velocities = np.abs(delta_a[valid]) / dt_all[valid]
        v_max      = velocities.max()

        if v_max > MAX_JOINT_VEL_DEG_S:
            n_over   = (velocities > MAX_JOINT_VEL_DEG_S).sum()
            peak_idx = np.where(valid)[0][velocities.argmax()]
            frame_id = int(df['frame'].iloc[peak_idx + 1])
            violations.append(
                f"{side:5s} | {col:<20s} | max vel={v_max:.1f} °/s "
                f"> {MAX_JOINT_VEL_DEG_S:.0f} °/s  "
                f"({n_over} frames, peak at frame {frame_id})"
            )

    return (len(violations) == 0 or all(v.startswith('[WARN]') for v in violations)), violations


# -------------------------------------------------------------------------------------
# Final report
# -------------------------------------------------------------------------------------
def _print_report(path: Path, check_results: list[tuple[str, bool, list[str]]]) -> bool:
    """
    Prints the full report and returns True if the trajectory is safe.

    Parameters:
        path          : path to the analysed joint_ik.csv file
        check_results : list of (check_name, passed, violations)
    """
    print(f"\n{'═' * 60}")
    print(f"  SAFETY VALIDATION REPORT")
    print(f"  {path}")
    print(f"{'═' * 60}")

    all_passed = True

    for check_name, passed, violations in check_results:
        _section(check_name)
        if passed and not violations:
            _ok("No violations.")
        else:
            for v in violations:
                if v.startswith('[WARN]'):
                    _warn(v[7:])   # strip the [WARN] prefix, already shown by _warn
                else:
                    all_passed = False
                    _fail(v)
            if passed and violations:
                _ok("No critical violations (warnings above).")

    print(f"\n{'═' * 60}")
    if all_passed:
        print("  RESULT: ✓ TRAJECTORY SAFE")
    else:
        print("  RESULT: ✗ TRAJECTORY NOT SAFE -- execution blocked")
    print(f"{'═' * 60}\n")

    return all_passed


# -------------------------------------------------------------------------------------
# Parameters log (for transparency)
# -------------------------------------------------------------------------------------
def _print_params() -> None:
    print(textwrap.dedent(f"""
    Validation parameters:
      SAFETY_PADDING_DEG   = {SAFETY_PADDING_DEG}°   (additional margin on hardware limits)
      MAX_JOINT_VEL_DEG_S  = {MAX_JOINT_VEL_DEG_S}°/s  (maximum joint velocity)
      N_WARMUP_FRAMES      = {N_WARMUP_FRAMES}       (warmup frames: warned but not failed)
    """).strip())


# -------------------------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------------------------
def validate(subject_num: int, exercise_num: int, video_num: int) -> bool:
    """
    Runs all safety checks on joint_ik.csv for the given video.

    Returns:
        True  -> trajectory is safe.
        False -> at least one critical violation detected.
    """
    subject_name  = f"subject_{subject_num:03d}"
    exercise_name = f"exercise_{exercise_num:03d}"
    video_name    = f"video_{video_num:03d}"
    folder        = DATA_ROOT / "landmarks" / subject_name / exercise_name / video_name
    ik_path       = folder / "joint_ik.csv"

    if not ik_path.exists():
        print(f"[validate_trajectory] Error: file not found -> {ik_path}")
        return False

    df = pd.read_csv(ik_path)
    print(f"[validate_trajectory] {len(df)} frames loaded from {ik_path}")
    _print_params()

    check_results = [
        ("CHECK 1 -- Joint limits", *check_joint_limits(df)),
        ("CHECK 2 -- Maximum joint velocity", *check_joint_velocities(df)),
    ]

    return _print_report(ik_path, check_results)


# -------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Offline safety validation on joint_ik.csv.
            Checks: joint limits, joint velocity.
            Exit code: 0 = safe, 1 = not safe.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--subject',  type=int, default=None)
    parser.add_argument('--exercise', type=int, default=None)
    parser.add_argument('--video',    type=int, default=None)
    args = parser.parse_args()

    try:
        subject_num  = args.subject  if args.subject  is not None else int(input("Subject number:  ").strip())
        exercise_num = args.exercise if args.exercise is not None else int(input("Exercise number: ").strip())
        video_num    = args.video    if args.video    is not None else int(input("Video number:    ").strip())
    except ValueError:
        print("Error: all values must be integers.")
        sys.exit(1)

    safe = validate(subject_num, exercise_num, video_num)
    sys.exit(0 if safe else 1)

if __name__ == "__main__":
    main()