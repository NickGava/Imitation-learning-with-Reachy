'''
Simulates a pick and place task with one arm of Reachy.
The head looks at the pick position at the beginning, then follows the gripper during the movement, and finally looks at neutral position.
'''

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
import time
import threading

print("Connessione a Reachy...")
reachy = ReachySDK(host="localhost")

if reachy.r_arm is None or reachy.l_arm is None:
    print("❌ Braccia non disponibili (Unity in Play?)")
    exit()

print("✅ Connessione OK")

dx = reachy.r_arm
sx = reachy.l_arm

# Neutral position
neutral_pos = {joint.name: 0.0 for joint in reachy.joints.values()}

# Pick position
pick_pos = {
    "r_shoulder_pitch": -50,
    "r_shoulder_roll": -20,
    "r_arm_yaw": 78,
    "r_elbow_pitch": -20,
    "r_forearm_yaw": -78,
    "r_gripper": -69,
    "r_wrist_pitch": 0,
    "r_wrist_roll": 0
}

# Place position
place_pos = {
    "r_shoulder_pitch": -110,
    "r_shoulder_roll": -70,
    "r_arm_yaw": 90,
    "r_elbow_pitch": -45,
    "r_forearm_yaw": -80
}

# Convert positions to joint objects so that function goto is happy
pick_pos_joints = {getattr(reachy.joints, name): val for name, val in pick_pos.items()}
place_pos_joints = {getattr(reachy.joints, name): val for name, val in place_pos.items()}
neutral_pos_joints = {getattr(reachy.joints, name): val for name, val in neutral_pos.items()}

stop_follow = threading.Event()     # Initialization of event to signal the head following thread to stop

# Function to continuously update head orientation to follow the gripper
def follow_gripper():
    while not stop_follow.is_set():
        p = dx.forward_kinematics()
        reachy.head.look_at(p[0,3], p[1,3], p[2,3], duration=0.1)

# Get the coordinates of the pick position to orient the head towards it
arm_joints_names = [j.name for j in dx.joints.values() if 'gripper' not in j.name]
pick_values = [float(pick_pos[name]) for name in arm_joints_names]
pick_coord = dx.forward_kinematics(pick_values)
reachy.head.look_at(pick_coord[0,3], pick_coord[1,3], pick_coord[2,3], duration=0.5)

# Move to pick position and close gripper
goto(goal_positions=pick_pos_joints, duration=2.0, interpolation_mode=InterpolationMode.MINIMUM_JERK)
dx.r_gripper.goal_position = 20
time.sleep(1)

# Start gripper following thread
t =threading.Thread(target = follow_gripper, daemon=True)
t.start()

# Move to place position and open gripper
goto(goal_positions=place_pos_joints, duration=2.0, interpolation_mode=InterpolationMode.MINIMUM_JERK)
dx.r_gripper.goal_position = -69
time.sleep(2)

stop_follow.set()       # Signal the head following thread to stop

# Return to neutral position
goto(goal_positions=neutral_pos_joints, duration=2.0, interpolation_mode=InterpolationMode.MINIMUM_JERK)