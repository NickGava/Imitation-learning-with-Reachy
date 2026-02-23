"""
human_to_robot.py
------------------
Maps human upper-body pose landmarks (recorded via get_landmark_live_ts.py)
to Reachy robot joint angles and replays the motion on the robot.

Pipeline:
    CSV (x,y,z positions) → geometric angles → robot joint angles → goto()

Coordinate systems:
    MediaPipe: x=right (person's left), y=down, z=toward camera (positive)
    Reachy:    x=forward, y=left, z=up (right-handed)

Joint mapping:
    Human elbow angle       → r/l_elbow_pitch
    Human shoulder pitch    → r/l_shoulder_pitch
    Human shoulder roll     → r/l_shoulder_roll
    Human wrist pitch       → r/l_wrist_pitch
    Human upper arm twist   → r/l_arm_yaw      (approximated)
    Human forearm twist     → r/l_forearm_yaw  (approximated)
    Human wrist twist       → r/l_wrist_roll   (approximated)

Note on axial rotations (arm_yaw, forearm_yaw, wrist_roll):
    These are rotations around the limb's own axis. Since MediaPipe only gives
    3D point positions (not segment orientations), these are geometric
    approximations and will be noisier than pitch/roll values. Tune SCALE
    factors accordingly — keeping them lower than pitch/roll is recommended.

Requirements:
    - reachy-sdk, pandas, numpy, scipy installed.
    - Reachy simulator running in Unity (Play mode) or physical robot connected.
    - A recorded CSV file from get_landmark_live_ts.py.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CSV_PATH    = Path(__file__).parent / "reachy_motion_dataset.csv"
REACHY_HOST = "localhost"

# Duration of each goto() step (seconds).
# Lower = faster replay but less smooth. Should match the CSV sampling rate.
STEP_DURATION = 0.1

# Smoothing window size (number of frames). Higher = smoother but more lag.
SMOOTH_WINDOW = 7

# Scale factors — tune these if the robot moves too much or too little.
# Axial rotations (arm_yaw, forearm_yaw, wrist_roll) are kept lower because
# they are approximated from position data and tend to be noisier.
SCALE = {
    "shoulder_pitch": 0.8,
    "shoulder_roll":  0.8,
    "elbow_pitch":    0.9,
    "wrist_pitch":    0.6,
    "arm_yaw":        0.5,   # axial — keep lower
    "forearm_yaw":    0.5,   # axial — keep lower
    "wrist_roll":     0.4,   # axial — keep lower
}

# Joint angle limits (degrees) — hard clamp to keep movements safe.
LIMITS = {
    "r_shoulder_pitch": (-150,  90),
    "r_shoulder_roll":  (-180,  10),
    "r_arm_yaw":        ( -90,  90),
    "r_elbow_pitch":    (-125,   0),
    "r_forearm_yaw":    (-100,  100),
    "r_wrist_pitch":    ( -45,  45),
    "r_wrist_roll":     ( -55,  35),

    "l_shoulder_pitch": (-150,  90),
    "l_shoulder_roll":  ( -10, 180),
    "l_arm_yaw":        ( -90,  90),
    "l_elbow_pitch":    (-125,   0),
    "l_forearm_yaw":    (-100, 100),
    "l_wrist_pitch":    ( -45,  45),
    "l_wrist_roll":     ( -35,  55),
}


# ---------------------------------------------------------------------------
# GEOMETRY HELPERS
# ---------------------------------------------------------------------------

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Returns the angle (degrees) between two vectors.
    Uses the dot product formula: cos(θ) = v1·v2 / (|v1||v2|)
    """
    v1_n = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_n = v2 / (np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(np.dot(v1_n, v2_n), -1.0, 1.0)))


def signed_angle_around_axis(v_from: np.ndarray, v_to: np.ndarray, axis: np.ndarray) -> float:
    """
    Returns the signed angle (degrees) from v_from to v_to, measured around `axis`.

    Both vectors are projected onto the plane perpendicular to `axis` before
    computing the angle. The sign follows the right-hand rule around `axis`.

    Args:
        v_from: Reference direction vector.
        v_to:   Target direction vector.
        axis:   Rotation axis (normalized).

    Returns:
        float: Signed angle in degrees.
    """
    axis_n = axis / (np.linalg.norm(axis) + 1e-8)

    # Project both vectors onto the plane perpendicular to axis
    vf = v_from - np.dot(v_from, axis_n) * axis_n
    vt = v_to   - np.dot(v_to,   axis_n) * axis_n

    vf_n = vf / (np.linalg.norm(vf) + 1e-8)
    vt_n = vt / (np.linalg.norm(vt) + 1e-8)

    cross = np.cross(vf_n, vt_n)
    sign  = np.sign(np.dot(cross, axis_n))
    return sign * angle_between(vf_n, vt_n)


# ---------------------------------------------------------------------------
# JOINT ANGLE COMPUTATION
# ---------------------------------------------------------------------------

def compute_elbow_pitch(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    """
    Computes the elbow flexion angle.

    Vectors from the elbow toward shoulder and toward wrist are used.
    - 180° = arm fully straight → Reachy elbow_pitch =    0°
    -   0° = arm fully folded  → Reachy elbow_pitch = -125°

    Returns:
        float: elbow_pitch in Reachy convention (negative = bent).
    """
    v_to_shoulder = shoulder - elbow
    v_to_wrist    = wrist    - elbow
    flexion_angle = angle_between(v_to_shoulder, v_to_wrist)
    pitch = - flexion_angle * (125.0 / 180.0)
    return pitch


def compute_shoulder_pitch(shoulder: np.ndarray, elbow: np.ndarray) -> float:
    """
    Computes the shoulder pitch (forward/backward rotation in the sagittal plane).

    - Arm hanging down → pitch =   0°
    - Arm forward      → pitch = -90°
    - Arm raised back  → pitch = +90°

    Returns:
        float: shoulder_pitch in degrees.
    """
    upper_arm = elbow - shoulder
    ua_n = upper_arm / (np.linalg.norm(upper_arm) + 1e-8)
    pitch = np.degrees(np.arctan2(ua_n[2], ua_n[1]))
    return pitch

def compute_shoulder_roll_right(shoulder: np.ndarray, elbow: np.ndarray) -> float:
    """
    Computes the right shoulder roll (lateral abduction).

    - Arm hanging down       → roll =  0°
    - Arm raised to the side → roll = negative (Reachy right arm convention)

    Returns:
        float: shoulder_roll in degrees for the right arm.
    """
    upper_arm = elbow - shoulder
    ua_n = upper_arm / (np.linalg.norm(upper_arm) + 1e-8)
    roll = np.degrees(np.arctan2(ua_n[0], ua_n[1]))
    return roll


def compute_shoulder_roll_left(shoulder: np.ndarray, elbow: np.ndarray) -> float:
    """
    Computes the left shoulder roll (lateral abduction).

    Returns:
        float: shoulder_roll in degrees for the left arm.
    """
    upper_arm = elbow - shoulder
    ua_n = upper_arm / (np.linalg.norm(upper_arm) + 1e-8)
    roll = np.degrees(np.arctan2(-ua_n[0], ua_n[1]))
    return roll


def compute_arm_yaw(shoulder: np.ndarray, elbow: np.ndarray) -> float:
    """
    Approximates the upper arm yaw (axial rotation around the shoulder-elbow axis).

    Uses the angle of the upper arm projected onto the horizontal plane (XZ
    in MediaPipe, where Y is down). When the arm hangs at the side pointing
    straight down the projection is zero; as the elbow rotates forward or
    backward the yaw increases.

    Note: this is an approximation — MediaPipe point positions do not encode
    true axial rotation of the limb segment.

    Returns:
        float: arm_yaw in degrees (positive = elbow rotated forward/inward).
    """
    upper_arm = elbow - shoulder
    ua_n = upper_arm / (np.linalg.norm(upper_arm) + 1e-8)

    horiz_mag = np.sqrt(ua_n[0]**2 + ua_n[2]**2)
    if horiz_mag < 1e-3:
        return 0.0

    yaw = np.degrees(np.arctan2(ua_n[2], ua_n[0]))
    return -yaw


def compute_forearm_yaw(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    """
    Approximates the forearm yaw (axial rotation around the elbow-wrist axis,
    i.e. pronation/supination).

    Projects the forearm vector onto the plane perpendicular to the upper arm,
    then measures its signed angle relative to a vertical reference in that plane.

    Note: this is an approximation — true pronation/supination requires hand
    orientation data not available from MediaPipe upper-body landmarks.

    Returns:
        float: forearm_yaw in degrees.
    """
    upper_arm = elbow - shoulder
    forearm   = wrist - elbow
    ua_n = upper_arm / (np.linalg.norm(upper_arm) + 1e-8)

    vertical   = np.array([0.0, 1.0, 0.0])   # down in MediaPipe
    ref_perp   = vertical - np.dot(vertical, ua_n) * ua_n
    ref_perp_n = ref_perp / (np.linalg.norm(ref_perp) + 1e-8)

    yaw = signed_angle_around_axis(ref_perp_n, forearm, ua_n)
    return yaw


def compute_wrist_pitch(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    """
    Computes the wrist pitch (flexion/extension relative to the forearm axis).

    Returns:
        float: wrist_pitch in degrees.
    """
    upper_arm = elbow - shoulder
    forearm   = wrist - elbow
    ua_n = upper_arm / (np.linalg.norm(upper_arm) + 1e-8)
    fa_n = forearm   / (np.linalg.norm(forearm)   + 1e-8)
    deviation = angle_between(ua_n, fa_n)
    pitch = deviation * 0.3
    return pitch


def compute_wrist_roll(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    """
    Approximates the wrist roll (axial rotation of the hand around the forearm axis).

    Measures the signed angle of the upper arm around the forearm axis,
    relative to a downward reference in the plane perpendicular to the forearm.
    This captures the component of forearm twist visible from wrist displacement.

    Note: a proper wrist roll requires hand landmark data (e.g. finger positions)
    not available from the 6-point upper-body model.

    Returns:
        float: wrist_roll in degrees.
    """
    forearm = wrist - elbow
    fa_n = forearm / (np.linalg.norm(forearm) + 1e-8)

    vertical   = np.array([0.0, 1.0, 0.0])
    ref_perp   = vertical - np.dot(vertical, fa_n) * fa_n
    ref_perp_n = ref_perp / (np.linalg.norm(ref_perp) + 1e-8)

    upper_arm = elbow - shoulder
    roll = signed_angle_around_axis(ref_perp_n, upper_arm, fa_n)
    return roll


# ---------------------------------------------------------------------------
# FRAME PROCESSING
# ---------------------------------------------------------------------------

def landmarks_to_robot_angles(row: pd.Series) -> dict:
    """
    Converts one frame of MediaPipe landmarks to a dictionary of Reachy joint angles.

    Args:
        row: One row of the CSV DataFrame (landmark positions for one timestep).

    Returns:
        dict: {joint_name: angle_degrees} for all mapped joints.
    """
    r_shoulder = np.array([row["r_shoulder_x"], row["r_shoulder_y"], row["r_shoulder_z"]])
    r_elbow    = np.array([row["r_elbow_x"],    row["r_elbow_y"],    row["r_elbow_z"]])
    r_wrist    = np.array([row["r_wrist_x"],    row["r_wrist_y"],    row["r_wrist_z"]])

    l_shoulder = np.array([row["l_shoulder_x"], row["l_shoulder_y"], row["l_shoulder_z"]])
    l_elbow    = np.array([row["l_elbow_x"],    row["l_elbow_y"],    row["l_elbow_z"]])
    l_wrist    = np.array([row["l_wrist_x"],    row["l_wrist_y"],    row["l_wrist_z"]])

    angles = {
        # Right arm
        "r_shoulder_pitch": compute_shoulder_pitch(r_shoulder, r_elbow)           * SCALE["shoulder_pitch"],
        "r_shoulder_roll":  compute_shoulder_roll_right(r_shoulder, r_elbow)      * SCALE["shoulder_roll"],
        "r_arm_yaw":        compute_arm_yaw(r_shoulder, r_elbow)                  * SCALE["arm_yaw"],
        "r_elbow_pitch":    compute_elbow_pitch(r_shoulder, r_elbow, r_wrist)     * SCALE["elbow_pitch"],
        "r_forearm_yaw":    compute_forearm_yaw(r_shoulder, r_elbow, r_wrist)     * SCALE["forearm_yaw"],
        "r_wrist_pitch":    compute_wrist_pitch(r_shoulder, r_elbow, r_wrist)     * SCALE["wrist_pitch"],
        "r_wrist_roll":     compute_wrist_roll(r_shoulder, r_elbow, r_wrist)      * SCALE["wrist_roll"],
        # Left arm
        "l_shoulder_pitch": compute_shoulder_pitch(l_shoulder, l_elbow)           * SCALE["shoulder_pitch"],
        "l_shoulder_roll":  compute_shoulder_roll_left(l_shoulder, l_elbow)       * SCALE["shoulder_roll"],
        "l_arm_yaw":        compute_arm_yaw(l_shoulder, l_elbow)                  * SCALE["arm_yaw"],
        "l_elbow_pitch":    compute_elbow_pitch(l_shoulder, l_elbow, l_wrist)     * SCALE["elbow_pitch"],
        "l_forearm_yaw":    compute_forearm_yaw(l_shoulder, l_elbow, l_wrist)     * SCALE["forearm_yaw"],
        "l_wrist_pitch":    compute_wrist_pitch(l_shoulder, l_elbow, l_wrist)     * SCALE["wrist_pitch"],
        "l_wrist_roll":     compute_wrist_roll(l_shoulder, l_elbow, l_wrist)      * SCALE["wrist_roll"],
    }

    # Clamp to safe limits
    for joint, value in angles.items():
        lo, hi = LIMITS[joint]
        angles[joint] = float(np.clip(value, lo, hi))

    return angles


# ---------------------------------------------------------------------------
# SMOOTHING
# ---------------------------------------------------------------------------

def smooth_angle_sequence(angles_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Applies a uniform (box) filter to each joint angle column to reduce jitter.

    Axial rotations (arm_yaw, forearm_yaw, wrist_roll) use a larger window
    because they are noisier approximations.

    Args:
        angles_df: DataFrame with one column per joint and one row per frame.
        window: Base smoothing window size (number of frames).

    Returns:
        pd.DataFrame: Smoothed angles.
    """
    axial_joints = ["arm_yaw", "forearm_yaw", "wrist_roll"]
    smoothed = angles_df.copy()

    for col in smoothed.columns:
        is_axial = any(ax in col for ax in axial_joints)
        w = window * 2 if is_axial else window
        smoothed[col] = uniform_filter1d(smoothed[col].values, size=w)

    return smoothed


# ---------------------------------------------------------------------------
# ROBOT REPLAY
# ---------------------------------------------------------------------------

def replay_on_robot(reachy: ReachySDK, angles_df: pd.DataFrame) -> None:
    """
    Sends each frame of joint angles to Reachy sequentially using goto().

    Args:
        reachy: Connected ReachySDK instance.
        angles_df: DataFrame of joint angles (one row = one timestep).
    """
    print(f"Replaying {len(angles_df)} frames on robot...")

    for i, (_, row) in enumerate(angles_df.iterrows()):
        goal_positions = {
            reachy.r_arm.r_shoulder_pitch: row["r_shoulder_pitch"],
            reachy.r_arm.r_shoulder_roll:  row["r_shoulder_roll"],
            reachy.r_arm.r_arm_yaw:        row["r_arm_yaw"],
            reachy.r_arm.r_elbow_pitch:    row["r_elbow_pitch"],
            reachy.r_arm.r_forearm_yaw:    row["r_forearm_yaw"],
            reachy.r_arm.r_wrist_pitch:    row["r_wrist_pitch"],
            reachy.r_arm.r_wrist_roll:     row["r_wrist_roll"],
            reachy.l_arm.l_shoulder_pitch: row["l_shoulder_pitch"],
            reachy.l_arm.l_shoulder_roll:  row["l_shoulder_roll"],
            reachy.l_arm.l_arm_yaw:        row["l_arm_yaw"],
            reachy.l_arm.l_elbow_pitch:    row["l_elbow_pitch"],
            reachy.l_arm.l_forearm_yaw:    row["l_forearm_yaw"],
            reachy.l_arm.l_wrist_pitch:    row["l_wrist_pitch"],
            reachy.l_arm.l_wrist_roll:     row["l_wrist_roll"],
        }

        goto(
            goal_positions=goal_positions,
            duration=STEP_DURATION,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )

        if i % 50 == 0:
            print(f"  Frame {i}/{len(angles_df)}")

    print("Replay complete.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load CSV
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} frames loaded.")

    # 2. Compute robot joint angles for each frame
    print("Computing joint angles...")
    angles_list = [landmarks_to_robot_angles(row) for _, row in df.iterrows()]
    angles_df = pd.DataFrame(angles_list)

    # 3. Smooth trajectories
    print(f"Smoothing trajectories (window={SMOOTH_WINDOW})...")
    angles_df = smooth_angle_sequence(angles_df, SMOOTH_WINDOW)

    # 4. Connect to robot
    print(f"Connecting to Reachy at {REACHY_HOST}...")
    reachy = ReachySDK(host=REACHY_HOST)

    if reachy.r_arm is None or reachy.l_arm is None:
        print("❌ Arms not available — is Unity running in Play mode?")
        exit()

    print("✅ Connected.")

    # 5. Replay motion
    replay_on_robot(reachy, angles_df)

    # 6. Return to neutral
    print("Returning to neutral position...")
    goto(
        goal_positions={joint: 0.0 for joint in reachy.joints.values()},
        duration=2.0,
        interpolation_mode=InterpolationMode.MINIMUM_JERK,
    )