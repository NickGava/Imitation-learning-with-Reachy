
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
import time
import numpy as np

# Connessione a Reachy
reachy = ReachySDK(host="localhost")  # DO NOT USE ON REAL ROBOT

if reachy.l_arm is None or reachy.l_arm is None:
    print("❌ Impossible to connect")
    exit()

print("✅ Connection OK")

reachy.turn_on('l_arm')

# Simulation positions

first_pos = {
    reachy.l_arm.l_elbow_pitch: -120,
    reachy.l_arm.l_shoulder_pitch: 70,
    reachy.l_arm.l_arm_yaw: 0,
    reachy.l_arm.l_forearm_yaw: 0,
}

second_pos = {
    reachy.l_arm.l_elbow_pitch: -120,
    reachy.l_arm.l_shoulder_pitch: -60,
}

third_pos = {
    reachy.l_arm.l_elbow_pitch: -10,
    reachy.l_arm.l_gripper: 69,
}

back_pos = {
    reachy.l_arm.l_arm_yaw: -50,
    reachy.l_arm.l_elbow_pitch: -50,
    reachy.l_arm.l_shoulder_pitch: -40,
    reachy.l_arm.l_forearm_yaw: 40,
    reachy.l_arm.l_wrist_pitch: -10,
}

# Start to move
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

# object taken
goto(
    goal_positions={reachy.l_arm.l_gripper: 10},
    duration=2.0,
    interpolation_mode = InterpolationMode.MINIMUM_JERK
)


goto(
    goal_positions=second_pos,
    duration=1.0,
    interpolation_mode = InterpolationMode.MINIMUM_JERK)

goto(
    goal_positions=back_pos,
    duration=2.0,
    interpolation_mode = InterpolationMode.MINIMUM_JERK
)

# leaves the object
goto(
    goal_positions={reachy.l_arm.l_gripper: 69},
    duration=1.0,
    interpolation_mode = InterpolationMode.MINIMUM_JERK
)

goto(
    goal_positions={reachy.l_arm.l_elbow_pitch: -120, 
    reachy.l_arm.l_arm_yaw: 0,
    reachy.l_arm.l_forearm_yaw: 0,},
    duration=1.0,
    interpolation_mode = InterpolationMode.MINIMUM_JERK
)

goto(
    goal_positions=first_pos,
    duration=1.0,
    interpolation_mode = InterpolationMode.MINIMUM_JERK
)

goto(
    goal_positions={getattr(reachy.joints, name): 0.0 for name in reachy.joints.keys()},
    duration=3.0,
    interpolation_mode = InterpolationMode.MINIMUM_JERK
)

reachy.turn_off_smoothly('l_arm')
