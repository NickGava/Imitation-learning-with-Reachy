import numpy as np
import math
from joint_mapping import HUMAN_LANDMARKS, ROBOT_JOINTS


# ----------------------------
# UTILITY
# ----------------------------

def to_np(lm):
    return np.array([lm.x, lm.y, lm.z])


def normalize(v):
    return v / np.linalg.norm(v)


def angle_between(v1, v2):
    v1 = normalize(v1)
    v2 = normalize(v2)

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.acos(dot)


# ----------------------------
# HUMAN → ANGOLI UMANI
# ----------------------------
def compute_human_arm_angles(pose_landmarks, side="left"):

    ids = HUMAN_LANDMARKS[side]

    shoulder = to_np(pose_landmarks[ids["shoulder"]])
    elbow = to_np(pose_landmarks[ids["elbow"]])
    wrist = to_np(pose_landmarks[ids["wrist"]])

    upper_arm = elbow - shoulder
    forearm = wrist - elbow

    # 1️⃣ elbow angle
    elbow_angle = angle_between(upper_arm, forearm)

    # 2️⃣ shoulder orientation (camera frame)
    shoulder_pitch = math.atan2(upper_arm[1], upper_arm[2])
    shoulder_roll = math.atan2(upper_arm[0], upper_arm[2])

    return {
        "shoulder_pitch": shoulder_pitch,
        "shoulder_roll": shoulder_roll,
        "elbow": elbow_angle,
    }


# ----------------------------
# ANGOLI UMANI → GIUNTI ROBOT
# ----------------------------
def convert_to_robot_commands(human_angles, side="left"):

    robot_map = ROBOT_JOINTS[side]

    robot_cmd = {
        robot_map["shoulder_pitch"]: human_angles["shoulder_pitch"],
        robot_map["shoulder_roll"]: human_angles["shoulder_roll"],
        robot_map["elbow"]: human_angles["elbow"],
    }

    return robot_cmd


# ----------------------------
# PIPELINE COMPLETA
# ----------------------------
def human_pose_to_robot_commands(pose_landmarks):

    left_angles = compute_human_arm_angles(pose_landmarks, "left")
    right_angles = compute_human_arm_angles(pose_landmarks, "right")

    left_cmd = convert_to_robot_commands(left_angles, "left")
    right_cmd = convert_to_robot_commands(right_angles, "right")

    return {**left_cmd, **right_cmd}