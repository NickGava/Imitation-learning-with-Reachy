"""
human_to_robot.py
------------------
Maps anatomical human joint angles (from biomechanics.py) to Reachy motor
commands using proportional scaling between the two ROMs.

Why NOT a direct 1:1 mapping
------------------------------
Human shoulder abduction: 0° → 180°
Reachy shoulder roll    : 0° → ~120° (hardware limit)

A direct copy would command the robot beyond its limits. Instead we use
linear interpolation that maps the full human ROM onto the full robot ROM,
preserving the *shape* of the trajectory while staying safe.

Reachy joint naming convention (from SDK)
------------------------------------------
  {side}_shoulder_pitch   — flexion / extension  (negative = forward)
  {side}_shoulder_roll    — abduction / adduction
  {side}_arm_yaw          — upper arm axial rotation
  {side}_elbow_pitch      — elbow flexion         (negative = flexed)
  {side}_forearm_yaw      — forearm pronation/supination
  {side}_wrist_pitch      — wrist flexion / extension
  {side}_wrist_roll       — wrist radial / ulnar deviation

Sign conventions
---------------------------------------------------
  r_shoulder_pitch: 0=neutral, negative=forward flexion   → multiply flexion by -1
  l_shoulder_pitch: same sign convention                  → multiply flexion by -1
  r_shoulder_roll : 0=neutral, negative=abduction         → multiply abduction by -1
  l_shoulder_roll : 0=neutral, positive=abduction         → keep sign
  r_elbow_pitch   : 0=full extension, negative=flexion    → multiply flexion by -1
  l_elbow_pitch   : same                                  → multiply flexion by -1
  {side}_arm_yaw  : positive=internal rotation            → keep sign   # da rivedere
  {side}_wrist_pitch: positive=flexion                    → keep sign
"""

from dataclasses import dataclass
import numpy as np
from biomechanics import JointAngles

# ---------------------------------------------------------------------------
# Human ROM (degrees) — from JOINT_LIMITS in biomechanics.py
# Used as the "source" interval for linear scaling.
# ---------------------------------------------------------------------------
HUMAN_ROM : dict[str, tuple[float, float]] = {
    "shoulder_flexion":      (-60.0,  180.0),
    "shoulder_abduction":    (  0.0,  180.0),
    "shoulder_internal_rot": (-90.0,   90.0),
    "elbow_flexion":         (  0.0,  145.0),
    "wrist_flexion":         (-70.0,   80.0),
}

# ---------------------------------------------------------------------------
# Reachy ROM (degrees) — hardware limits per joint.
# These are the "destination" intervals for linear scaling.
# ---------------------------------------------------------------------------
REACHY_ROM : dict[str, tuple[float, float]] = {
    # pitch: 0=neutral, negative=forward flexion
    # We store it as [min_pitch, max_pitch] where more negative = more flexion
    "shoulder_pitch":    (-150.0,   90.0),   # ~240° total range
    # roll: abduction direction differs per side — handled in sign logic below
    "shoulder_roll_r":   (-180.0,   10.0),   # right: 0=neutral, negative=abduction
    "shoulder_roll_l":   ( -10.0,  180.0),   # left:  0=neutral, positive=abduction
    "arm_yaw":           ( -90.0,   90.0),
    "elbow_pitch":       (-125.0,    0.0),   # 0=extended, negative=flexed
    "wrist_pitch":       ( -45.0,   45.0),
}

# ---------------------------------------------------------------------------
# Safety margins (degrees) — we never command within this margin of the limit.
# Prevents banging against hardware stops.
# ---------------------------------------------------------------------------
SAFETY_MARGIN = 3.0   


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ReachyJointCommand:
    """
    Target joint angles (degrees) for Reachy's arm motors.
    Matches the attribute names used by reachy-sdk.
    """
    # Left arm
    l_shoulder_pitch : float = 0.0
    l_shoulder_roll : float = 0.0
    l_arm_yaw : float = 0.0
    l_elbow_pitch : float = 0.0
    l_wrist_pitch : float = 0.0

    # Right arm
    r_shoulder_pitch : float = 0.0
    r_shoulder_roll : float = 0.0
    r_arm_yaw : float = 0.0
    r_elbow_pitch : float = 0.0
    r_wrist_pitch : float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "l_shoulder_pitch": self.l_shoulder_pitch,
            "l_shoulder_roll": self.l_shoulder_roll,
            "l_arm_yaw": self.l_arm_yaw,
            "l_elbow_pitch": self.l_elbow_pitch,
            "l_wrist_pitch": self.l_wrist_pitch,
            "r_shoulder_pitch": self.r_shoulder_pitch,
            "r_shoulder_roll": self.r_shoulder_roll,
            "r_arm_yaw": self.r_arm_yaw,
            "r_elbow_pitch": self.r_elbow_pitch,
            "r_wrist_pitch": self.r_wrist_pitch,
        }

    @staticmethod
    def csv_header() -> list[str]:
        return list(ReachyJointCommand().to_dict().keys())


# ---------------------------------------------------------------------------
# Core scaling function
# ---------------------------------------------------------------------------

def scale(value : float, src_min : float, src_max : float, dst_min : float, dst_max : float, margin : float = SAFETY_MARGIN) -> float:
    """
    Linearly maps a value from [src_min, src_max] to [dst_min, dst_max],
    then clamps the output to [dst_min+margin, dst_max-margin].

    This is the core of the proportional scaling approach:
        output = dst_min + [(value - src_min) / (src_max - src_min)] * (dst_max - dst_min)

    Args:
        value:   Input angle in source (human) range.
        src_min: Minimum of source range.
        src_max: Maximum of source range.
        dst_min: Minimum of destination (robot) range.
        dst_max: Maximum of destination (robot) range.
        margin:  Safety margin applied at both ends of the robot range.

    Returns:
        Scaled and clamped robot angle in degrees.
    """
    if abs(src_max - src_min) < 1e-8:
        return (dst_min + dst_max) / 2.0

    ratio = (value - src_min) / (src_max - src_min)
    output = dst_min + ratio * (dst_max - dst_min)

    lo = min(dst_min, dst_max) + margin
    hi = max(dst_min, dst_max) - margin
    return float(np.clip(output, lo, hi))


# ---------------------------------------------------------------------------
# Per-joint mapping functions
# ---------------------------------------------------------------------------

def _map_shoulder_pitch(flexion_deg: float) -> float:
    """
    Maps human shoulder flexion → Reachy shoulder_pitch.

    Uses a dual-slope linear mapping anchored at neutral (0°):
        Flexion  (human 0→+180°) : robot  0° → -150°  (arm forward)
        Extension(human 0→ -60°) : robot  0° →  +90°   (arm slightly back)

    Anchoring at neutral guarantees: human 0° → robot 0°.
    """
    if flexion_deg >= 0:
        # Flexion branch 
        return scale(flexion_deg, 0.0, 180.0, 0.0, -150.0)
    else:
        # Extension branch
        return scale(flexion_deg, 0.0, -60.0, 0.0, +90.0)


def _map_shoulder_roll_right(abduction_deg: float) -> float:
    """
    Maps human shoulder abduction → Reachy r_shoulder_roll.

    Human:  0° (arm at side) → 180° (arm straight up)
    Reachy: 0° (neutral)     → -180° (full abduction, with margin)

    Mapping anchored at neutral: human 0° → robot 0°.
    """
    return scale(abduction_deg, 0.0, 180.0, 0.0, -180.0)


def _map_shoulder_roll_left(abduction_deg: float) -> float:
    """
    Maps human shoulder abduction → Reachy l_shoulder_roll.

    Human:  0° (arm at side) → 180° (arm straight up)
    Reachy: 0° (neutral)     → +180° (full abduction, with margin)

    Mapping anchored at neutral: human 0° → robot 0°.
    """
    return scale(abduction_deg, 0.0, 180.0, 0.0, +180.0)


def _map_arm_yaw(internal_rot_deg: float) -> float:
    """
    Maps human shoulder internal rotation → Reachy arm_yaw.

    Human:  -90° (external) → +90° (internal)
    Reachy: -90° (external) → +90° (internal)
    Ranges match — direct proportional scaling.
    """
    h_lo, h_hi = HUMAN_ROM["shoulder_internal_rot"]
    r_lo, r_hi = REACHY_ROM["arm_yaw"]
    return scale(internal_rot_deg, h_lo, h_hi, r_lo, r_hi)


def _map_elbow_pitch(flexion_deg: float) -> float:
    """
    Maps human elbow flexion → Reachy elbow_pitch.

    Human:   0° (fully extended) → 145° (fully flexed)
    Reachy:  0° (fully extended) → -125° (fully flexed, with margin)

    Anchored at 0°: full extension in human = full extension on robot.
    """
    return scale(flexion_deg, 0.0, 145.0, 0.0, -125.0)


def _map_wrist_pitch(wrist_flexion_deg: float) -> float:
    """
    Maps human wrist flexion → Reachy wrist_pitch.

    Human:  -70° (extension) → +80° (flexion)
    Reachy: -45° (extension) → +45° (flexion)
    """
    h_lo, h_hi = HUMAN_ROM["wrist_flexion"]
    r_lo, r_hi = REACHY_ROM["wrist_pitch"]
    return scale(wrist_flexion_deg, h_lo, h_hi, r_lo, r_hi)


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def human_to_robot(angles: JointAngles) -> ReachyJointCommand:
    """
    Converts a full set of anatomical human joint angles into Reachy motor
    commands using proportional scaling per joint.

    Args:
        angles: JointAngles from biomechanics.process_frame(), already
                constrained to anatomical limits.

    Returns:
        ReachyJointCommand with all angles in Reachy's motor space,
        clamped with safety margins.
    """
    return ReachyJointCommand(
        # Left arm
        l_shoulder_pitch = _map_shoulder_pitch(angles.l_shoulder_flexion),
        l_shoulder_roll = _map_shoulder_roll_left(angles.l_shoulder_abduction),
        l_arm_yaw = _map_arm_yaw(angles.l_shoulder_internal_rot),
        l_elbow_pitch = _map_elbow_pitch(angles.l_elbow_flexion),
        l_wrist_pitch = _map_wrist_pitch(angles.l_wrist_flexion),

        # Right arm
        r_shoulder_pitch = _map_shoulder_pitch(angles.r_shoulder_flexion),
        r_shoulder_roll = _map_shoulder_roll_right(angles.r_shoulder_abduction),
        r_arm_yaw = _map_arm_yaw(angles.r_shoulder_internal_rot),
        r_elbow_pitch = _map_elbow_pitch(angles.r_elbow_flexion),
        r_wrist_pitch = _map_wrist_pitch(angles.r_wrist_flexion),
    )


# ---------------------------------------------------------------------------
# Batch conversion (entire CSV / trajectory)
# ---------------------------------------------------------------------------

def convert_trajectory(bio_rows: list[dict]) -> list[dict]:
    """
    Converts a full list of biomechanical angle rows (from CSV) to
    Reachy motor command rows.

    Each input row must contain the keys from JointAngles.csv_header()
    plus 'timestamp'.

    Args:
        bio_rows: List of dicts, one per frame, as read from the CSV
                  produced by get_landmark_live_bio.py.

    Returns:
        List of dicts with keys from ReachyJointCommand.csv_header()
        plus 'timestamp', ready to save or feed to the player.
    """
    result = []
    for row in bio_rows:
        angles = JointAngles(
            l_shoulder_flexion = float(row["l_shoulder_flexion"]),
            l_shoulder_abduction = float(row["l_shoulder_abduction"]),
            l_shoulder_internal_rot = float(row["l_shoulder_internal_rot"]),
            l_elbow_flexion = float(row["l_elbow_flexion"]),
            l_wrist_flexion = float(row["l_wrist_flexion"]),
            r_shoulder_flexion = float(row["r_shoulder_flexion"]),
            r_shoulder_abduction = float(row["r_shoulder_abduction"]),
            r_shoulder_internal_rot = float(row["r_shoulder_internal_rot"]),
            r_elbow_flexion = float(row["r_elbow_flexion"]),
            r_wrist_flexion = float(row["r_wrist_flexion"]),
        )
        cmd = human_to_robot(angles)
        out = {"timestamp": float(row["timestamp"])}
        out.update(cmd.to_dict())
        result.append(out)
    return result
