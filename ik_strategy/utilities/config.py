'''
config.py
=============================================================================
Usage:
    from config import DATA_ROOT
'''

from pathlib import Path
from typing import Dict
import numpy as np

# Absolute path to the shared data folder
DATA_ROOT = Path(__file__).resolve().parent.parent / "_data"

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
# Default frames per second for the videos
DEFAULT_FPS = 30

# Default neutral looking position for the head (x, y, z)
HEAD_NEUTRAL = [1.0, 0.0, 0.15]     # 1m ahead, at shoulder height

# Default open/closed gripper motor angles
GRIPPER_RANGE = {
    'right_hand': {'open': -40.0, 'closed':  20.0},
    'left_hand':  {'open':  40.0, 'closed': -20.0},
}

# ---------------------------------------------------------------------------
# Names of the joints of the robot
# ---------------------------------------------------------------------------
JOINT_COLS = [
    'r_shoulder_pitch', 'r_shoulder_roll', 'r_arm_yaw', 'r_elbow_pitch',
    'r_forearm_yaw', 'r_wrist_pitch', 'r_wrist_roll', 'r_gripper',
    'l_shoulder_pitch', 'l_shoulder_roll', 'l_arm_yaw', 'l_elbow_pitch',
    'l_forearm_yaw', 'l_wrist_pitch', 'l_wrist_roll', 'l_gripper',
]
# Names of the coordinates of the head of the robot
HEAD_COLS = ['head_x', 'head_y', 'head_z']

# ---------------------------------------------------------------------------
# Default rest pose for the robot (in degrees)
# ---------------------------------------------------------------------------
STARTING_POSE: Dict[str, float] = {
    'r_shoulder_pitch':  -20.0,  'r_shoulder_roll': 0.0,  'r_arm_yaw':     0.0,
    'r_elbow_pitch':   0.0,  'r_forearm_yaw':    0.0,  'r_wrist_pitch':  0.0,
    'r_wrist_roll':      0.0,  'r_gripper':      0.0,
    'l_shoulder_pitch':  -20.0,  'l_shoulder_roll':  0.0,  'l_arm_yaw':     0.0,
    'l_elbow_pitch':   0.0,  'l_forearm_yaw':    0.0,  'l_wrist_pitch':  0.0,
    'l_wrist_roll':      0.0,  'l_gripper':       0.0,
}
# Rest pose as a numpy array, without gripper
REST_DEG = {
    'right': np.array([0., -5., 0., -90., 0., 0., 0.]),
    'left':  np.array([0.,  5., 0., -90., 0., 0., 0.]),
}
# ---------------------------------------------------------------------------
# Joint limits for the robot (in degrees)
# ---------------------------------------------------------------------------
JOINT_LIMITS_DEG = {
    'right': np.array([
        [-150,  90],    # shoulder_pitch
        [-180,  10],    # shoulder_roll
        [ -90,  90],    # arm_yaw
        [-125,   0],    # elbow_pitch
        [-100, 100],    # forearm_yaw
        [ -45,  45],    # wrist_pitch
        [ -55,  35],    # wrist_roll
    ], dtype=float),
    'left': np.array([
        [-150,  90],
        [ -10, 180],
        [ -90,  90],
        [-125,   0],
        [-100, 100],
        [ -45,  45],
        [ -35,  55],
    ], dtype=float),
}
# Padding to apply to the joint limits
JOINT_LIMIT_PADDING_DEG = 3.0

# ---------------------------------------------------------------------------
# Indices of the keypoints in the pose estimation 
# ---------------------------------------------------------------------------
POSE_INDICES = {
    'nose':            0,
    'left_shoulder':  11,
    'right_shoulder': 12,
    'left_elbow':     13,
    'right_elbow':    14,
    'left_wrist':     15,
    'right_wrist':    16,
    'left_hip':       23,
    'right_hip':      24,
}
