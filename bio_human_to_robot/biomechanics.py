"""
biomechanics.py
----------------
Rigorous biomechanical processing of MediaPipe pose landmarks.

Pipeline:
    1. Torso reference frame  — removes dependency on person's position in the scene.
    2. Anthropometric normalization — removes dependency on body size.
    3. Anatomical joint angle computation — shoulder (flexion, abduction, rotation),
       elbow flexion, wrist flexion.
    4. Butterworth low-pass filter — removes tremor and MediaPipe estimation noise.
    5. Joint limit enforcement — clamps angles to anatomically feasible ranges.

Coordinate convention (MediaPipe world landmarks):
    x → increases to the right of the image  (= person's LEFT)
    y → increases downward
    z → depth (negative = closer to camera)

All output angles are in degrees, following anatomical convention:
    - 0° = anatomical neutral position
    - positive = flexion / abduction / internal rotation
    - negative = extension / adduction / external rotation

References:
    - Wu et al. (2005). ISB recommendations for joint coordinate systems of
      the shoulder, elbow, wrist and hand. J. Biomech.
    - MediaPipe Pose Landmark documentation.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Anatomical joint limits (degrees) — based on clinical reference ranges
# ---------------------------------------------------------------------------
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    # Shoulder
    "shoulder_flexion":          (-60.0,  180.0),  # extension → flexion
    "shoulder_abduction":        (  0.0,  180.0),  # adduction → abduction
    "shoulder_internal_rot":     (-90.0,   90.0),  # external → internal
    # Elbow
    "elbow_flexion":             (  0.0,  145.0),  # full extension → full flexion
    # Wrist
    "wrist_flexion":             (-70.0,   80.0),  # extension → flexion
}

# MediaPipe landmark indices used in this module
LM = {
    "nose":       0,
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_elbow":    13,
    "r_elbow":    14,
    "l_wrist":    15,
    "r_wrist":    16,
    "l_hip":      23,
    "r_hip":      24,
    "l_index":    19,
    "r_index":    20,
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class JointAngles:
    """
    Container for computed anatomical joint angles (degrees).

    Naming convention: positive = flexion / abduction / internal rotation.
    """
    # Left arm
    l_shoulder_flexion : float = 0.0
    l_shoulder_abduction : float = 0.0
    l_shoulder_internal_rot : float = 0.0
    l_elbow_flexion : float = 0.0
    l_wrist_flexion : float = 0.0

    # Right arm
    r_shoulder_flexion : float = 0.0
    r_shoulder_abduction : float = 0.0
    r_shoulder_internal_rot : float = 0.0
    r_elbow_flexion : float = 0.0
    r_wrist_flexion : float = 0.0

    def to_dict(self) -> dict[str, float]:
        # Returns a flat dict suitable for a CSV row.
        return {
            "l_shoulder_flexion": self.l_shoulder_flexion,
            "l_shoulder_abduction": self.l_shoulder_abduction,
            "l_shoulder_internal_rot": self.l_shoulder_internal_rot,
            "l_elbow_flexion": self.l_elbow_flexion,
            "l_wrist_flexion": self.l_wrist_flexion,
            "r_shoulder_flexion": self.r_shoulder_flexion,
            "r_shoulder_abduction": self.r_shoulder_abduction,
            "r_shoulder_internal_rot": self.r_shoulder_internal_rot,
            "r_elbow_flexion": self.r_elbow_flexion,
            "r_wrist_flexion": self.r_wrist_flexion,
        }

    @staticmethod
    def csv_header() -> list[str]:
        return list(JointAngles().to_dict().keys())


# ---------------------------------------------------------------------------
# Torso reference frame
# ---------------------------------------------------------------------------

class TorsoFrame:
    """
    Defines a right-handed body-fixed coordinate system anchored at the torso.

    Origin: midpoint between left and right shoulders.
    X axis: points from right shoulder to left shoulder  (mediolateral, person's left)
    Y axis: points upward along the torso spine direction
    Z axis: X × Y  (points anteriorly, i.e., forward out of the chest)

    All landmark positions expressed in this frame are invariant to the
    person's global position and orientation in the scene.
    """

    def __init__(self, landmarks: list) -> None:
        """
        Builds the torso frame from a MediaPipe pose landmark list.

        Args:
            landmarks: Full MediaPipe pose landmark list (33 points).
        """
        l_sh = self._lm(landmarks, "l_shoulder")        # takes the coordinates of the left shoulder landmark [x, y, z]
        r_sh = self._lm(landmarks, "r_shoulder")
        l_hp = self._lm(landmarks, "l_hip")
        r_hp = self._lm(landmarks, "r_hip")

        # Origin: shoulder midpoint
        self.origin = (l_sh + r_sh) / 2.0

        # X: right→left shoulder (mediolateral)
        x_raw = l_sh - r_sh
        self.x_axis = self._unit(x_raw)

        # Y: hip midpoint → shoulder midpoint (longitudinal, upward)
        hip_mid = (l_hp + r_hp) / 2.0
        y_raw = self.origin - hip_mid
        self.y_axis = self._unit(y_raw)

        # Z: anterior (forward), by right-hand rule
        self.z_axis = self._unit(np.cross(self.x_axis, self.y_axis))

        # Re-orthogonalize Y to guarantee orthonormal frame
        self.y_axis = self._unit(np.cross(self.z_axis, self.x_axis))

        # Rotation matrix: rows are the axes expressed in world frame
        # To project world coords → torso frame: R @ (p - origin)
        self.R = np.stack([self.x_axis, self.y_axis, self.z_axis], axis=0)

    def to_local(self, world_point : np.ndarray) -> np.ndarray:
        # Projects a world-space point into the torso frame.
        return self.R @ (world_point - self.origin)

    @staticmethod
    def _lm(landmarks : list, name : str) -> np.ndarray:
        lm = landmarks[LM[name]]
        return np.array([lm.x, lm.y, lm.z])

    @staticmethod
    def _unit(v : np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else v


# ---------------------------------------------------------------------------
# Anthropometric normalization
# ---------------------------------------------------------------------------

def compute_segment_lengths(landmarks: list) -> dict[str, float]:
    """
    Measures the lengths of arm segments (in MediaPipe normalized units).

    These lengths depend on the person's actual body size. They are used
    to normalize positions so that different subjects are comparable.

    Args:
        landmarks: MediaPipe pose landmark list.

    Returns:
        dict with keys: 'l_upper_arm', 'l_forearm', 'r_upper_arm', 'r_forearm'.
        Values are in MediaPipe coordinate units.
    """
    def dist(a_name : str, b_name : str) -> float:
        a = np.array([landmarks[LM[a_name]].x, landmarks[LM[a_name]].y, landmarks[LM[a_name]].z])
        b = np.array([landmarks[LM[b_name]].x, landmarks[LM[b_name]].y, landmarks[LM[b_name]].z])
        return float(np.linalg.norm(a - b))

    return {
        "l_upper_arm": dist("l_shoulder", "l_elbow"),
        "l_forearm": dist("l_elbow", "l_wrist"),
        "r_upper_arm": dist("r_shoulder", "r_elbow"),
        "r_forearm": dist("r_elbow", "r_wrist"),
    }


# ---------------------------------------------------------------------------
# Anatomical joint angle computation
# ---------------------------------------------------------------------------

def _angle_between(v1 : np.ndarray, v2 : np.ndarray) -> float:
    """
    Returns the angle (degrees) between two 3D vectors.
    Numerically stable via atan2 instead of acos.
    """
    cross = np.linalg.norm(np.cross(v1, v2))   # |v1||v2|sin(θ)
    dot   = np.dot(v1, v2)                      # |v1||v2|cos(θ)
    return float(np.degrees(np.arctan2(cross, dot)))


def _signed_angle(v1 : np.ndarray, v2 : np.ndarray, plane_normal : np.ndarray) -> float:
    """
    Returns the signed angle (degrees) from v1 to v2 around plane_normal.

    Sign convention: positive = counterclockwise when viewed from the normal tip.
    """
    cross = np.cross(v1, v2)
    sin_a = np.linalg.norm(cross) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    angle = float(np.degrees(np.arctan2(sin_a, cos_a)))
    if np.dot(cross, plane_normal) < 0:
        angle = -angle
    return angle

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n

def compute_shoulder_angles(shoulder_local : np.ndarray, elbow_local : np.ndarray, wrist_local : np.ndarray, side : str) -> tuple[float, float, float]:
    """
    Computes shoulder flexion, abduction, and internal rotation angles
    following the ISB coordinate system convention.

    All positions must be expressed in the torso reference frame.

    Axes in torso frame:
        X → mediolateral (positive = person's left)
        Y → longitudinal (positive = up)
        Z → anterior     (positive = forward)

    Definitions:
        Flexion: angle of the upper arm projected onto the sagittal plane (XY → YZ)
                 positive = arm forward, negative = arm behind torso
        Abduction: angle of the upper arm relative to the Y (longitudinal) axis
                   projected onto the frontal plane (XZ) -- positive = arm lifted away from body
        Int. Rot.: axial rotation of the upper arm around its own long axis
                   estimated from the elbow pointing direction

    Args:
        shoulder_local: shoulder position in torso frame (origin for upper arm).
        elbow_local: elbow position in torso frame.
        side: 'l' or 'r'.

    Returns:
        (flexion_deg, abduction_deg, internal_rotation_deg)
    """
    # Upper arm vector (from shoulder to elbow), in torso frame
    upper_arm = elbow_local - shoulder_local
    if np.linalg.norm(upper_arm) < 1e-8:
        return 0.0, 0.0, 0.0

    ua = upper_arm / np.linalg.norm(upper_arm)

    # Torso frame axes
    Y = np.array([0.0, 1.0, 0.0])   # up
    Z = np.array([0.0, 0.0, 1.0])   # forward / anterior
    X = np.array([1.0, 0.0, 0.0])   # mediolateral

    # --- Flexion (sagittal plane YZ) ---
    ua_sag = ua.copy()
    ua_sag[0] = 0.0  # remove mediolateral component

    norm_sag = np.linalg.norm(ua_sag)
    if norm_sag < 1e-8:
        flexion = 0.0
    else:
        ua_sag /= norm_sag
        # Neutral = -Y (arm down)
        # Positive = forward (+Z)
        flexion = np.degrees(np.arctan2(
            np.dot(ua_sag, Z),       # forward component
            -np.dot(ua_sag, Y)       # vertical reference
        ))

    # --- Abduction: angle between upper arm and the longitudinal axis (Y) ---
    # projected onto frontal (XY) plane, remove Z component
    lateral        = abs(ua[0])
    forward_longit = np.sqrt(ua[1] ** 2 + ua[2] ** 2)
    abduction = float(np.degrees(np.arctan2(lateral, forward_longit)))
    # Sign: positive = away from body for both sides
    if side == "l" and ua[0] < 0:
        abduction = -abduction   # left arm going toward body (adduction)
    if side == "r" and ua[0] > 0:
        abduction = -abduction   # right arm going toward body (adduction)

    # --- Internal rotation: axial rotation of upper arm ---
    # Approximate: angle of the upper arm vector projected onto the XZ plane
    # relative to the mediolateral axis (X). Positive = internal rotation.
    forearm = wrist_local - elbow_local
    y_h = _normalize(upper_arm) 
    if np.linalg.norm(forearm) < 1e-8:
        int_rot = 0.0
    else:
        f = _normalize(forearm)

        # Costruzione asse X dell'omero
        # normale al piano (forearm, upper_arm)
        x_h = np.cross(f, y_h)

        if np.linalg.norm(x_h) < 1e-8:
            int_rot = 0.0
        else:
            x_h = _normalize(x_h)

            # Asse di riferimento torso (mediolaterale)
            ref = X if side == "l" else -X

            # Proiettiamo ref sul piano ortogonale a y_h
            ref_proj = ref - np.dot(ref, y_h) * y_h

            if np.linalg.norm(ref_proj) < 1e-8:
                int_rot = 0.0
            else:
                ref_proj = _normalize(ref_proj)

                # Angolo attorno all'asse longitudinale dell'omero
                int_rot = _signed_angle(ref_proj, x_h, y_h)

    return flexion, abduction, int_rot

def compute_elbow_flexion(shoulder_local : np.ndarray, elbow_local : np.ndarray, wrist_local : np.ndarray) -> float:
    """
    Computes elbow flexion angle (degrees) from the three joint centers.

    Definition: angle at the elbow between the upper arm and forearm vectors.
        0°   = full extension (arm straight)
        145° = full flexion

    Args:
        shoulder_local: shoulder position in torso frame.
        elbow_local:    elbow position in torso frame.
        wrist_local:    wrist position in torso frame.

    Returns:
        Elbow flexion in degrees [0, 145].
    """
    upper_arm = shoulder_local - elbow_local   # pointing from elbow toward shoulder
    forearm = wrist_local - elbow_local        # pointing from elbow toward wrist
    # Geometric angle at elbow is 180° when fully extended.
    # Anatomical flexion convention: 0° = full extension, 145° = full flexion.
    return 180.0 - _angle_between(upper_arm, forearm)


def compute_wrist_flexion(elbow_local : np.ndarray, wrist_local : np.ndarray, finger_local : np.ndarray) -> float:
    """
    Computes wrist flexion/extension angle (degrees).

    Definition: angle between forearm axis and hand axis, in the sagittal plane.
        0°  = neutral (straight)
        > 0 = wrist flexion
        < 0 = wrist extension

    Args:
        elbow_local:  elbow position in torso frame.
        wrist_local:  wrist position in torso frame.
        finger_local: index finger MCP position in torso frame.

    Returns:
        Wrist flexion in degrees [-70, 80].
    """
    forearm = wrist_local - elbow_local     # elbow → wrist
    hand = finger_local - wrist_local       # wrist → finger

    if np.linalg.norm(forearm) < 1e-8 or np.linalg.norm(hand) < 1e-8:
        return 0.0

    # Flexion = deviation from straight (angle = 0 means perfect alignment)
    raw_angle = _angle_between(forearm, hand)
    return 180.0 - raw_angle   # 180° when straight → remap to 0° = neutral


# ---------------------------------------------------------------------------
# Butterworth low-pass filter
# ---------------------------------------------------------------------------

def butterworth_filter(signal : np.ndarray, cutoff_hz : float = 6.0, fs_hz : float = 30.0, order : int = 4) -> np.ndarray:
    """
    Applies a zero-phase Butterworth low-pass filter to a 1-D signal.

    Using zero-phase (forward + backward pass via filtfilt) ensures no
    time delay is introduced — critical for real-time imitation.

    Typical cutoff for human voluntary arm movements: 5-8 Hz.
    MediaPipe noise is mostly above 8 Hz.

    Args:
        signal:     1-D numpy array of angle values over time.
        cutoff_hz:  Low-pass cutoff frequency in Hz. Default 6.0 Hz.
        fs_hz:      Sampling frequency in Hz. Default 30.0 (webcam FPS).
        order:      Filter order. Higher = sharper rolloff, default 4.

    Returns:
        Filtered signal as numpy array, same length as input.
    """
    if len(signal) < 3 * order:
        # Not enough samples to filter — return as it is
        return signal.copy()

    nyquist = fs_hz / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normalized_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal)


def filter_joint_angles_dataframe(angles_list : list[dict], cutoff_hz : float = 6.0, fs_hz : float = 30.0) -> list[dict]:
    """
    Applies Butterworth filtering to each angle column across a time series.

    Args:
        angles_list: List of dicts, one per frame, with angle values.
        cutoff_hz:   Low-pass cutoff frequency in Hz.
        fs_hz:       Sampling frequency in Hz.

    Returns:
        List of dicts with filtered angle values (same structure).
    """
    if not angles_list:
        return angles_list

    keys = list(angles_list[0].keys())
    filtered = {k: np.array([row[k] for row in angles_list]) for k in keys}

    for k in keys:
        if k != "timestamp":
            filtered[k] = butterworth_filter(filtered[k], cutoff_hz, fs_hz)

    return [
        {k: float(filtered[k][i]) for k in keys}
        for i in range(len(angles_list))
    ]


# ---------------------------------------------------------------------------
# Joint limit enforcement
# ---------------------------------------------------------------------------

def apply_joint_limits(angles : JointAngles) -> JointAngles:
    """
    Clamps all joint angles to their anatomically feasible ranges.

    Prevents downstream IK commands from requesting impossible postures
    that could damage the robot or produce nonsensical trajectories.

    Args:
        angles: Computed JointAngles (unconstrained).

    Returns:
        JointAngles with all values clamped to JOINT_LIMITS.
    """
    def clamp(value: float, key: str) -> float:
        lo, hi = JOINT_LIMITS[key]
        return float(np.clip(value, lo, hi))

    return JointAngles(
        l_shoulder_flexion = clamp(angles.l_shoulder_flexion, "shoulder_flexion"),
        l_shoulder_abduction = clamp(angles.l_shoulder_abduction, "shoulder_abduction"),
        l_shoulder_internal_rot = clamp(angles.l_shoulder_internal_rot, "shoulder_internal_rot"),
        l_elbow_flexion = clamp(angles.l_elbow_flexion, "elbow_flexion"),
        l_wrist_flexion = clamp(angles.l_wrist_flexion, "wrist_flexion"),

        r_shoulder_flexion = clamp(angles.r_shoulder_flexion, "shoulder_flexion"),
        r_shoulder_abduction = clamp(angles.r_shoulder_abduction, "shoulder_abduction"),
        r_shoulder_internal_rot = clamp(angles.r_shoulder_internal_rot, "shoulder_internal_rot"),
        r_elbow_flexion = clamp(angles.r_elbow_flexion, "elbow_flexion"),
        r_wrist_flexion = clamp(angles.r_wrist_flexion, "wrist_flexion"),
    )


# ---------------------------------------------------------------------------
# Main entry point — process one frame
# ---------------------------------------------------------------------------

def process_frame(landmarks: list) -> Optional[JointAngles]:
    """
    Processes a single MediaPipe pose landmark frame through the full
    biomechanical pipeline and returns anatomical joint angles.

    Pipeline:
        1. Build torso reference frame.
        2. Express all joints in the torso frame.
        3. Compute anatomical angles.
        4. Enforce joint limits.

    Args:
        landmarks: Full MediaPipe pose landmark list (33 points).
                   Expects NormalizedLandmark objects with .x, .y, .z attributes.

    Returns:
        JointAngles with constrained anatomical angles, or None if
        essential landmarks are missing.
    """
    try:
        # Build torso frame
        frame = TorsoFrame(landmarks)

        # Project joints into torso frame
        def lm_local(name : str) -> np.ndarray:
            lm = landmarks[LM[name]]
            return frame.to_local(np.array([lm.x, lm.y, lm.z]))

        l_sh = lm_local("l_shoulder")
        l_el = lm_local("l_elbow")
        l_wr = lm_local("l_wrist")
        l_fi = lm_local("l_index")

        r_sh = lm_local("r_shoulder")
        r_el = lm_local("r_elbow")
        r_wr = lm_local("r_wrist")
        r_fi = lm_local("r_index")

        # Compute angles
        l_flex, l_abd, l_rot = compute_shoulder_angles(l_sh, l_el, l_wr, "l")
        r_flex, r_abd, r_rot = compute_shoulder_angles(r_sh, r_el, r_wr, "r")

        angles = JointAngles(
            l_shoulder_flexion = l_flex,
            l_shoulder_abduction = l_abd,
            l_shoulder_internal_rot = l_rot,
            l_elbow_flexion = compute_elbow_flexion(l_sh, l_el, l_wr),
            l_wrist_flexion = compute_wrist_flexion(l_el, l_wr, l_fi),

            r_shoulder_flexion = r_flex,
            r_shoulder_abduction = r_abd,
            r_shoulder_internal_rot = r_rot,
            r_elbow_flexion = compute_elbow_flexion(r_sh, r_el, r_wr),
            r_wrist_flexion = compute_wrist_flexion(r_el, r_wr, r_fi),
        )

        # Enforce limits
        return apply_joint_limits(angles)

    except (IndexError, AttributeError, ValueError):
        return None
