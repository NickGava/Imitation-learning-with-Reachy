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

antennas_pos = {
    reachy.head.l_antenna: 90,
    reachy.head.r_antenna: -90
}

r_arm_pos = {
    dx.r_shoulder_pitch: -80,
    dx.r_elbow_pitch: -125,
    dx.r_shoulder_roll: 10,
    dx.r_arm_yaw: 50
}
l_arm_pos = {
    sx.l_shoulder_pitch: -80,
    sx.l_elbow_pitch: -125,
    sx.l_shoulder_roll: -10,
    sx.l_arm_yaw: -50
}


goto(
	goal_positions=antennas_pos,
	duration=0.5,
	interpolation_mode=InterpolationMode.MINIMUM_JERK
	)

time.sleep(2)

goto(
	goal_positions=r_arm_pos,
	duration=2.0,
	interpolation_mode=InterpolationMode.MINIMUM_JERK
	)

goto(
	goal_positions=l_arm_pos,
	duration=2.0,
	interpolation_mode=InterpolationMode.MINIMUM_JERK
	)

time.sleep(5)



reachy.head.l_antenna.goal_position = 0
reachy.head.r_antenna.goal_position = 0
dx.r_shoulder_pitch.goal_position = 0
dx.r_elbow_pitch.goal_position = 0
dx.r_shoulder_roll.goal_position = 0
dx.r_arm_yaw.goal_position = 0
sx.l_shoulder_pitch.goal_position = 0
sx.l_elbow_pitch.goal_position = 0
sx.l_shoulder_roll.goal_position = 0
sx.l_arm_yaw.goal_position = 0