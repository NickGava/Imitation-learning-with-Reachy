
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
import time
import numpy as np

# Connessione a Reachy
reachy = ReachySDK(host='localhost')  # oppure IP del robot

first_pos = {
    reachy.r_arm.r_elbow_pitch: -120,
    reachy.r_arm.r_shoulder_pitch: 50,
}

second_pos = {
    reachy.r_arm.r_elbow_pitch: -60,
    reachy.r_arm.r_shoulder_pitch: -60,
}

third_pos = {
    reachy.r_arm.r_elbow_pitch: 0,
    reachy.r_arm.r_gripper: -69,
    reachy.r_arm.r_shoulder_pitch: -67,
    reachy.r_arm.r_wrist_pitch: 45
}

back_pos = {
    reachy.r_arm.r_elbow_pitch: -180,
    reachy.r_arm.r_shoulder_pitch: 0,
}

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
    duration=3.0,
    interpolation_mode = InterpolationMode.MINIMUM_JERK
)

goto(
    goal_positions={reachy.r_arm.r_gripper: 20},
    duration=2.0,
    interpolation_mode = InterpolationMode.MINIMUM_JERK
)


goto(
    goal_positions=back_pos,
    duration=3.0,
    interpolation_mode = InterpolationMode.MINIMUM_JERK
)

# goto(
#     goal_positions={getattr(reachy.joints, name): 0.0 for name in reachy.joints.keys()},
#     duration=3.0,
#     interpolation_mode = InterpolationMode.MINIMUM_JERK
# )
