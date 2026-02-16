from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
import time

print("Connessione a Reachy...")
reachy = ReachySDK(host="localhost")

if reachy.r_arm is None or reachy.l_arm is None:
    print("❌ Braccia non disponibili (Unity in Play?)")
    exit()

print("✅ Connessione OK")



dx = reachy.r_arm
sx = reachy.l_arm

zero_pos = {joint: 0 for joint in reachy.joints.values()}

heart_pos = {
    dx.r_shoulder_pitch: -80,
    dx.r_elbow_pitch: -85,
    dx.r_shoulder_roll: -25,
    dx.r_arm_yaw: 50,
    sx.l_shoulder_pitch: -80,
    sx.l_elbow_pitch: -85,
    sx.l_shoulder_roll: 25,
    sx.l_arm_yaw: -50,
    dx.r_wrist_roll: 55,
    sx.l_wrist_roll: -55,
    dx.r_gripper: -9,
    sx.l_gripper: 9
}


goto(
    goal_positions=heart_pos,
    duration=1.0,
    interpolation_mode=InterpolationMode.MINIMUM_JERK
)

time.sleep(5)

goto(
    goal_positions=zero_pos,
    duration=2.0,
    interpolation_mode=InterpolationMode.MINIMUM_JERK
)