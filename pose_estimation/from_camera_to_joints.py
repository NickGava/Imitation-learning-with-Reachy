import cv2
import mediapipe as mp
import numpy as np
import math
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ----------------------------
# CONFIG MODELLO
# ----------------------------
model_path = Path(__file__).parent / "pose_landmarker_heavy.task"

base_options = python.BaseOptions(model_asset_path=str(model_path))

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)


# ----------------------------
# UTILITY MATEMATICHE
# ----------------------------

def to_np(lm):
    return np.array([lm.x, lm.y, lm.z])


def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.acos(dot)  # radianti


# ----------------------------
# HUMAN → ROBOT MAPPING
# ----------------------------
def compute_arm_angles(pose_landmarks, side="left"):
    if side == "left":
        s, e, w = 11, 13, 15
    else:
        s, e, w = 12, 14, 16

    shoulder = to_np(pose_landmarks[s])
    elbow = to_np(pose_landmarks[e])
    wrist = to_np(pose_landmarks[w])

    # vettori braccio
    upper_arm = elbow - shoulder
    forearm = wrist - elbow

    # ----------------------------
    # 1️⃣ angolo gomito
    # ----------------------------
    elbow_angle = angle_between(upper_arm, forearm)

    # ----------------------------
    # 2️⃣ orientamento spalla
    # ----------------------------
    # proiezione sul piano camera
    shoulder_pitch = math.atan2(upper_arm[1], upper_arm[2])
    shoulder_roll = math.atan2(upper_arm[0], upper_arm[2])

    return {
        "shoulder_pitch": shoulder_pitch,
        "shoulder_roll": shoulder_roll,
        "elbow": elbow_angle,
    }


# ----------------------------
# WEBCAM
# ----------------------------
cap = cv2.VideoCapture(0)

frame_timestamp = 0

print("Mapping attivo — premi Q per uscire")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    if result.pose_landmarks:
        pose = result.pose_landmarks[0]

        left_angles = compute_arm_angles(pose, "left")
        right_angles = compute_arm_angles(pose, "right")

        print("\nLEFT ARM:", left_angles)
        print("RIGHT ARM:", right_angles)

        # ----------------------------
        # QUI potresti comandare il robot
        # ----------------------------
        # reachy.r_arm.goto(...)
        # reachy.l_arm.goto(...)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()