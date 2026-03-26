import time

import cv2
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory.interpolation import InterpolationMode
import threading
from reachy_sdk.camera import ZoomLevel


# IP del robot
ROBOT_IP = "10.59.1.20"   # cambia con il tuo IP

# connessione al robot
reachy = ReachySDK(host=ROBOT_IP)

if reachy.r_arm is None or reachy.l_arm is None:
    print("❌ Impossible to connect")
    exit()

print("Connesso a Reachy!")

reachy.turn_on('head')

head = reachy.head

stop_watch = threading.Event()     # Initialization of event to signal the cameras to stop

# Function to continuously update head orientation to follow the gripper
def watching():
    while not stop_watch.is_set():
        # frame dalle camere
        left_frame = reachy.left_camera.last_frame
        right_frame = reachy.right_camera.last_frame

        if left_frame is not None:
            cv2.imshow("Left Camera", left_frame)

        if right_frame is not None:
            cv2.imshow("Right Camera", right_frame)

        cv2.waitKey(1)
        time.sleep(0.01)

# Start watching
t = threading.Thread(target = watching, daemon=True)
t.start()

# head.look_at(0.5, 0.5, 0, 1.8, interpolation_mode=InterpolationMode.MINIMUM_JERK)       # look left
head.look_at(0.5, 0, 0, 1, interpolation_mode=InterpolationMode.MINIMUM_JERK)         
time.sleep(5)

reachy.turn_off_smoothly('head')

stop_watch.set()       # Signal the head following thread to stop

cv2.destroyAllWindows()

