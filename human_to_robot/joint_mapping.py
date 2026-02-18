# mapping tra features umane e giunti robot

HUMAN_LANDMARKS = {
    "left": {
        "shoulder": 11,
        "elbow": 13,
        "wrist": 15,
    },
    "right": {
        "shoulder": 12,
        "elbow": 14,
        "wrist": 16,
    }
}

ROBOT_JOINTS = {
    "left": {
        "shoulder_pitch": "l_shoulder_pitch",
        "shoulder_roll": "l_shoulder_roll",
        "elbow": "l_elbow_pitch",
    },
    "right": {
        "shoulder_pitch": "r_shoulder_pitch",
        "shoulder_roll": "r_shoulder_roll",
        "elbow": "r_elbow_pitch",
    }
}