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

open_arms_pos = {
    dx.r_shoulder_roll: -90,
    sx.l_shoulder_roll: 90,
    dx.r_shoulder_pitch: -90,
    sx.l_shoulder_pitch: -90,
    dx.r_forearm_yaw: -90,
    sx.l_forearm_yaw: -90,
}
goto(
    goal_positions=zero_pos,
    duration=1.0,
    interpolation_mode=InterpolationMode.MINIMUM_JERK
)

goto(
    goal_positions=open_arms_pos,  
    duration=1.0,
    interpolation_mode=InterpolationMode.MINIMUM_JERK
)

sx.l_wrist_roll.goal_position = 55
time.sleep(0.5)

sx.l_wrist_roll.goal_position = -35
sx.l_elbow_pitch.goal_position = -20
time.sleep(0.5)

sx.l_wrist_roll.goal_position = 0
sx.l_shoulder_pitch.goal_position = 0
sx.l_arm_yaw.goal_position = -90
sx.l_forearm_yaw.goal_position = 90
sx.l_shoulder_roll.goal_position = 110
time.sleep(0.5)

# riporto in orizzontale il braccio sinistro
sx.l_elbow_pitch.goal_position = 0
sx.l_arm_yaw.goal_position = 0
sx.l_shoulder_roll.goal_position = 90
sx.l_shoulder_pitch.goal_position = -90
sx.l_forearm_yaw.goal_position = -90


dx.r_elbow_pitch.goal_position = -20
dx.r_wrist_roll.goal_position = 0
dx.r_shoulder_pitch.goal_position = 0
dx.r_arm_yaw.goal_position = 90
dx.r_forearm_yaw.goal_position = 90
dx.r_shoulder_roll.goal_position = -110
time.sleep(0.5)

dx.r_shoulder_pitch.goal_position = -90
dx.r_shoulder_roll.goal_position = -90
dx.r_arm_yaw.goal_position = 0
dx.r_wrist_roll.goal_position = 35
time.sleep(0.5)

dx.r_elbow_pitch.goal_position = 0
dx.r_wrist_roll.goal_position = -55
time.sleep(0.5)
dx.r_wrist_roll.goal_position = -55




time.sleep(3)
