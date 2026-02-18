from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
import time
import threading
import numpy as np

print("Connessione a Reachy...")
reachy = ReachySDK(host="localhost")

if reachy.r_arm is None or reachy.l_arm is None:
    print("❌ Braccia non disponibili (Unity in Play?)")
    exit()

print("✅ Connessione OK")

goto(
    goal_positions={getattr(reachy.joints, name): 0.0 for name in reachy.joints.keys()},
    duration=1.0,
    interpolation_mode = InterpolationMode.MINIMUM_JERK
)
time.sleep(1)


#reachy.r_arm.r_shoulder_roll.goal_position = -90    # -0.8376 = max -y
reachy.r_arm.r_shoulder_pitch.goal_position = -90    # 0.6474 = max x
# reachy.r_arm.r_shoulder_pitch.goal_position = -150    # 0.5587 = max z


time.sleep(2)

p = reachy.r_arm.forward_kinematics()
print("Posizione finale del gripper:", p[:3, 3])