
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
import time
import numpy as np

# Connessione a Reachy
host = '10.59.1.20'
reachy = ReachySDK(host=host)  # oppure IP del robot

if reachy.r_arm is None or reachy.l_arm is None:
    print("❌ Impossible to connect")
    exit()

print("✅ Connection OK")

reachy.turn_on('r_arm')

# Neutral position
neutral_pos = {
    "r_shoulder_pitch": 0,
    "r_shoulder_roll": 0,
    "r_arm_yaw": 0,
    "r_elbow_pitch": -100,
    "r_forearm_yaw": 0,
    "r_gripper": -69,
    "r_wrist_pitch": 0,
    "r_wrist_roll": 0
}

# Simulation positions

first_pos = {
    reachy.r_arm.r_elbow_pitch: -120,
    reachy.r_arm.r_shoulder_pitch: 70,
    reachy.r_arm.r_arm_yaw: 0,
    reachy.r_arm.r_forearm_yaw: 0,
}

second_pos = {
    reachy.r_arm.r_elbow_pitch: -120,
    reachy.r_arm.r_shoulder_pitch: -60,
}

third_pos = {
    reachy.r_arm.r_elbow_pitch: -10,
    reachy.r_arm.r_gripper: -69,
    # reachy.r_arm.r_shoulder_pitch: -67,
    # reachy.r_arm.r_wrist_pitch: 45
}

back_pos = {
    reachy.r_arm.r_arm_yaw: 50,
    reachy.r_arm.r_elbow_pitch: -50,
    reachy.r_arm.r_shoulder_pitch: -40,
    reachy.r_arm.r_forearm_yaw: -40,
}

# Real robot positions
approach_pos = {
    "r_shoulder_pitch": -40,
    "r_shoulder_roll": -60,
    "r_arm_yaw": 78,
    "r_elbow_pitch": -105,
    "r_forearm_yaw": -35,
    "r_gripper": -69,
    "r_wrist_pitch": -10,
    "r_wrist_roll": 0
}
pick_pos = {
    "r_shoulder_pitch": -40,
    "r_shoulder_roll": -20,
    "r_arm_yaw": 78,
    "r_elbow_pitch": -90,
    "r_forearm_yaw": -35,
    "r_gripper": -69,
    "r_wrist_pitch": -10,
    "r_wrist_roll": 0
}

place_pos = {
    "r_shoulder_pitch": -30,
    "r_shoulder_roll": 0,
    "r_arm_yaw": 0,
    "r_elbow_pitch": -65,
    "r_forearm_yaw": 0,
    "r_gripper": -30,
    "r_wrist_pitch": 0,
    "r_wrist_roll": 0
}


approach_pos_joints = {getattr(reachy.joints, name): val for name, val in approach_pos.items()}
pick_pos_joints = {getattr(reachy.joints, name): val for name, val in pick_pos.items()}
place_pos_joints = {getattr(reachy.joints, name): val for name, val in place_pos.items()}
neutral_pos_joints = {getattr(reachy.joints, name): val for name, val in neutral_pos.items()}

# goto(
#     goal_positions=neutral_pos_joints,
#     duration=3.0,
#     interpolation_mode = InterpolationMode.MINIMUM_JERK
# )

# Start to move
if host == 'localhost':
    goto(
        goal_positions=first_pos,
        duration=1.0,
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    )

    goto(
        goal_positions=second_pos,
        duration=1.0,
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    )
        
    goto(
        goal_positions=third_pos,
        duration=1.0,
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    )

    # preso l'oggetto
    goto(
        goal_positions={reachy.r_arm.r_gripper: 0},
        duration=2.0,
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    )


    goto(
        goal_positions=second_pos,
        duration=1.0,
        interpolation_mode = InterpolationMode.MINIMUM_JERK)

    goto(
        goal_positions=back_pos,
        duration=1.0,
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    )

    # lascio l'oggetto
    goto(
        goal_positions={reachy.r_arm.r_gripper: -69},
        duration=2.0,
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    )

    goto(
        goal_positions={reachy.r_arm.r_elbow_pitch: -120, 
        reachy.r_arm.r_arm_yaw: 0,
        reachy.r_arm.r_forearm_yaw: 0,},
        duration=1.0,
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    )

    goto(
        goal_positions=first_pos,
        duration=1.0,
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    )

else:
    goto(
        goal_positions = approach_pos_joints,
        duration = 1.5,
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    )
    
    goto(
        goal_positions = pick_pos_joints,
        duration = 1.5,
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    )
    
    # preso l'oggetto
    goto(
        goal_positions={reachy.r_arm.r_gripper: -30},
        duration=0.5,
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    )
    goto(
        goal_positions = place_pos_joints,
        duration = 1.5,
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    )
    goto(
        goal_positions={reachy.r_arm.r_gripper: -69,
                        reachy.r_arm.r_wrist_roll: 10},
        duration=0.5,
        interpolation_mode = InterpolationMode.MINIMUM_JERK
    )
    time.sleep(1)


goto(
    goal_positions=neutral_pos_joints,
    duration=3.0,
    interpolation_mode = InterpolationMode.MINIMUM_JERK
)
goto(
    goal_positions={reachy.r_arm.r_gripper: 0},
    duration=0.5,
    interpolation_mode = InterpolationMode.MINIMUM_JERK
)

reachy.turn_off_smoothly('r_arm')
