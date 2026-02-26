"""
reproduce_offline.py
---------------------
Offline imitation pipeline: reads a recorded human motion CSV, converts it
to Reachy motor commands, and plays it back on the robot respecting the
original timing.

Full pipeline (Mese 3):
    1. Load CSV produced by get_landmark_live_bio.py
    2. Convert each row: anatomical angles → Reachy motor commands
       (via human_to_robot.py)
    3. Save a robot-space CSV for inspection / replay
    4. Safety pre-check: detect frames with large inter-frame jumps
    5. Playback: stream commands to Reachy using goto() with MINIMUM_JERK,
       respecting the original capture timestamps

Safety features
---------------
- Maximum inter-frame velocity check (degrees/second per joint).
  Frames with jumps above MAX_DEG_PER_SEC are flagged and the trajectory
  is smoothed around them before sending to the robot.
- All commands are already range-limited by human_to_robot.py (SAFETY_MARGIN).
- Robot goes to neutral before and after playback.
- Dry-run mode: prints commands without connecting to the robot.

Usage
-----
    # Dry run (no robot needed):
    python reproduce_offline.py --csv human_motion_bio.csv --dry-run

    # Live run (Unity simulator or physical robot):
    python reproduce_offline.py --csv human_motion_bio.csv

    # Custom speed:
    python reproduce_offline.py --csv human_motion_bio.csv --speed 0.5

Arguments
---------
    --csv       Path to the biomechanical CSV (default: human_motion_bio.csv)
    --speed     Playback speed multiplier (1.0 = real time, 0.5 = half speed)
    --dry-run   Print commands only, do not connect to the robot
    --host      Reachy SDK host (default: localhost)
    --arms      Which arms to move: 'both', 'left', 'right' (default: both)
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from human_to_robot import convert_trajectory, ReachyJointCommand

# ---------------------------------------------------------------------------
# Safety parameters
# ---------------------------------------------------------------------------
MAX_DEG_PER_SEC = 120.0     # max allowed joint velocity between frames
SMOOTH_WINDOW   = 5         # frames used to smooth flagged jumps (must be odd)
NEUTRAL_DURATION = 2.0      # seconds to reach neutral before/after playback


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_bio_csv(path : Path) -> list[dict]:
    """
    Reads the biomechanical CSV produced by get_landmark_live_bio.py.

    Args:
        path: Path to the CSV file.

    Returns:
        List of dicts, one per frame.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError: if required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"timestamp", "l_shoulder_flexion", "r_shoulder_flexion",
                "l_elbow_flexion", "r_elbow_flexion"}
    missing = required - set(rows[0].keys()) if rows else set()
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    print(f"Loaded {len(rows)} frames from: {path}")
    return rows


def save_robot_csv(robot_rows: list[dict], out_path: Path) -> None:
    """
    Saves the converted robot-space commands to a CSV for inspection.

    Args:
        robot_rows: List of dicts with Reachy joint angles + timestamp.
        out_path:   Destination CSV path.
    """
    if not robot_rows:
        return
    fieldnames = list(robot_rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(robot_rows)
    print(f"Robot trajectory saved → {out_path}")


# ---------------------------------------------------------------------------
# Safety check & smoothing
# ---------------------------------------------------------------------------

def smooth_trajectory(robot_rows : list[dict], max_dps : float = MAX_DEG_PER_SEC, window : int = SMOOTH_WINDOW) -> list[dict]:
    """
    Detects inter-frame velocity spikes and applies a moving-average smooth
    around them.

    A spike is any frame where any joint moves faster than max_dps degrees
    per second relative to the previous frame.

    Args:
        robot_rows: List of robot command dicts (with 'timestamp' key).
        max_dps:    Velocity threshold in degrees/second.
        window:     Half-width of the smoothing window (in frames).

    Returns:
        Smoothed list of robot command dicts.
    """
    if len(robot_rows) < 3:
        return robot_rows

    joint_keys = [k for k in robot_rows[0] if k != "timestamp"]
    n = len(robot_rows)
    values = {k: np.array([row[k] for row in robot_rows]) for k in joint_keys}
    timestamps = np.array([row["timestamp"] for row in robot_rows])
    dt = np.diff(timestamps)
    dt = np.where(dt < 1e-6, 1.0 / 30.0, dt)   # guard against zero dt

    # Find flagged frames
    flagged = set()
    for k in joint_keys:
        velocities = np.abs(np.diff(values[k])) / dt
        spike_frames = np.where(velocities > max_dps)[0] + 1
        flagged.update(spike_frames.tolist())

    if flagged:
        print(f"{len(flagged)} frames exceed {max_dps}°/s — smoothing applied.")

    # Apply moving-average around flagged frames
    half = window // 2
    smooth_frames = set()
    for f in flagged:
        for i in range(max(0, f - half), min(n, f + half + 1)):
            smooth_frames.add(i)

    for k in joint_keys:
        smoothed = values[k].copy()
        for i in sorted(smooth_frames):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            smoothed[i] = float(np.mean(values[k][lo:hi]))
        values[k] = smoothed

    result = []
    for i in range(n):
        row = {"timestamp": float(timestamps[i])}
        for k in joint_keys:
            row[k] = float(values[k][i])
        result.append(row)

    return result


def print_trajectory_stats(robot_rows: list[dict]) -> None:
    """Prints min/max/mean for each Reachy joint across the trajectory."""
    joint_keys = [k for k in robot_rows[0] if k != "timestamp"]
    timestamps  = [row["timestamp"] for row in robot_rows]
    duration    = timestamps[-1] - timestamps[0]

    print(f"\n{'='*62}")
    print(f"Robot trajectory — {len(robot_rows)} frames, {duration:.2f} s")
    print(f"{'='*62}")
    print(f"{'Joint':<22}  {'Min':>7}  {'Mean':>7}  {'Max':>7}")
    print(f"{'-'*62}")
    for k in joint_keys:
        vals = np.array([row[k] for row in robot_rows])
        print(f"{k:<22}  {vals.min():>+7.1f}  {vals.mean():>+7.1f}  {vals.max():>+7.1f}")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# Playback engine
# ---------------------------------------------------------------------------

def build_reachy_target(reachy, cmd_dict: dict, arms: str) -> dict:
    """
    Converts a command dict with string joint names to a {joint_object: value}
    dict suitable for goto(), filtering by requested arms.

    Args:
        reachy:   Connected ReachySDK instance.
        cmd_dict: Dict mapping joint name strings to angle values.
        arms:     'both', 'left', or 'right'.

    Returns:
        Dict {joint_object → angle_deg} for goto().
    """
    target = {}
    for name, value in cmd_dict.items():
        if name == "timestamp":
            continue
        if arms == "left"  and not name.startswith("l_"):
            continue
        if arms == "right" and not name.startswith("r_"):
            continue
        joint = getattr(reachy.joints, name, None)
        if joint is not None:
            target[joint] = float(value)
    return target


def playback(robot_rows : list[dict], reachy, speed : float = 1.0, arms : str = "both") -> None:
    """
    Streams robot joint commands to Reachy, respecting the original capture
    timing scaled by the speed multiplier.

    Each frame is sent as a goto() call with MINIMUM_JERK interpolation.
    The duration for each goto is the inter-frame interval from the recording,
    scaled by (1 / speed).

    Args:
        robot_rows: Smoothed, converted robot command rows (with timestamp).
        reachy:     Connected ReachySDK instance.
        speed:      Playback speed multiplier (1.0 = real time).
        arms:       'both', 'left', or 'right'.
    """
    from reachy_sdk.trajectory import goto
    from reachy_sdk.trajectory.interpolation import InterpolationMode

    print(f"Starting playback ({len(robot_rows)} frames, speed={speed}x)...")

    timestamps = [row["timestamp"] for row in robot_rows]
    total_duration = (timestamps[-1] - timestamps[0]) / speed
    print(f"Estimated duration: {total_duration:.1f} s\n")

    playback_start = time.time()
    capture_start  = timestamps[0]

    for i, row in enumerate(robot_rows):
        # How much time should have elapsed since start?
        expected_elapsed = (row["timestamp"] - capture_start) / speed
        actual_elapsed   = time.time() - playback_start
        wait = expected_elapsed - actual_elapsed

        if wait > 0.002:
            time.sleep(wait)

        # Compute goto duration: time until the NEXT frame (or 0.1 s for last)
        if i < len(robot_rows) - 1:
            dt = (timestamps[i + 1] - timestamps[i]) / speed
        else:
            dt = 0.5

        dt = max(dt, 0.05)   # never faster than 20 Hz commands

        target = build_reachy_target(reachy, row, arms)
        if target:
            goto(
                goal_positions=target,
                duration=dt,
                interpolation_mode=InterpolationMode.MINIMUM_JERK,
            )

        # Progress bar every 30 frames
        if (i + 1) % 30 == 0 or i == len(robot_rows) - 1:
            pct = (i + 1) / len(robot_rows) * 100
            print(f"   Frame {i+1:4d}/{len(robot_rows)}  [{pct:5.1f}%]")

    print("\nPlayback complete.")


def dry_run(robot_rows: list[dict], n_preview: int = 10) -> None:
    """
    Prints the first n_preview robot command frames without connecting
    to the robot. Useful for verifying the mapping before live tests.

    Args:
        robot_rows:  Converted robot command rows.
        n_preview:   Number of frames to print.
    """
    print(f"\n🔎 DRY RUN — first {n_preview} frames (no robot connection):\n")
    keys = [k for k in robot_rows[0] if k != "timestamp"]
    header = f"{'frame':>5}  {'time':>8}  " + "  ".join(f"{k:>18}" for k in keys)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(robot_rows[:n_preview]):
        t_rel = row["timestamp"] - robot_rows[0]["timestamp"]
        vals  = "  ".join(f"{row[k]:>+18.2f}" for k in keys)
        print(f"{i:>5}  {t_rel:>8.3f}  {vals}")
    print("\nDry run complete — no robot commands sent.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline human-to-Reachy motion reproduction."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).parent / "human_motion_bio.csv",
        help="Path to the biomechanical CSV (default: human_motion_bio.csv)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0 = real time)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only — no robot connection",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Reachy SDK host (default: localhost)",
    )
    parser.add_argument(
        "--arms",
        choices=["both", "left", "right"],
        default="both",
        help="Which arms to move (default: both)",
    )
    args = parser.parse_args()

    # Load bio CSV
    bio_rows = load_bio_csv(args.csv)

    # Convert to robot space
    print("Converting human angles → Reachy motor commands...")
    robot_rows = convert_trajectory(bio_rows)

    # Safety smoothing
    robot_rows = smooth_trajectory(robot_rows)

    # Stats
    print_trajectory_stats(robot_rows)

    # Save robot CSV
    out_csv = args.csv.parent / "robot_motion_bio.csv"
    save_robot_csv(robot_rows, out_csv)

    # Playback or dry run
    if args.dry_run:
        dry_run(robot_rows)
        return

    print("Connecting to Reachy...")
    try:
        from reachy_sdk import ReachySDK
        from reachy_sdk.trajectory import goto
        from reachy_sdk.trajectory.interpolation import InterpolationMode
    except ImportError:
        print("reachy_sdk not installed. Use --dry-run to test without the robot.")
        return

    reachy = ReachySDK(host=args.host)

    if reachy.r_arm is None or reachy.l_arm is None:
        print("Arms not available — is Unity running in Play mode?")
        return

    print("Connected to Reachy.\n")

    # Move to neutral before starting
    print("→ Moving to neutral position...")
    neutral = {joint: 0.0 for joint in reachy.joints.values()}
    goto(neutral, duration=NEUTRAL_DURATION,
         interpolation_mode=InterpolationMode.MINIMUM_JERK)
    time.sleep(0.5)

    # Play
    playback(robot_rows, reachy, speed=args.speed, arms=args.arms)

    # Return to neutral
    print("\n→ Returning to neutral position...")
    goto(neutral, duration=NEUTRAL_DURATION,
         interpolation_mode=InterpolationMode.MINIMUM_JERK)


if __name__ == "__main__":
    main()
