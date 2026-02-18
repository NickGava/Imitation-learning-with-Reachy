import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

from human_to_robot.human_to_robot_converter import human_pose_to_robot_commands


# ----------------------------
# CONFIG
# ----------------------------
model_path = Path(__file__).parent / "pose_landmarker_heavy.task"

# MediaPipe pose landmark indices utili per robot
JOINTS = {
    "l_shoulder": 11,
    "l_elbow": 13,
    "l_wrist": 15,
    "r_shoulder": 12,
    "r_elbow": 14,
    "r_wrist": 16,
}


# ----------------------------
# DISEGNO LANDMARK
# ----------------------------
def draw_landmarks_on_image(rgb_image, detection_result):
    annotated = np.copy(rgb_image)

    if detection_result.pose_landmarks:
        for pose_landmarks in detection_result.pose_landmarks:
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
            )

    return annotated


# ----------------------------
# CONFIGURA MODELLO
# ----------------------------
base_options = python.BaseOptions(model_asset_path=str(model_path))

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)


# ----------------------------
# CREA FILE CSV
# ----------------------------
csv_file = open("reachy_motion_dataset.csv", "w", newline="")
writer = csv.writer(csv_file)

# header
header = ["timestamp"]
for joint in JOINTS.keys():
    header += [f"{joint}_x", f"{joint}_y", f"{joint}_z"]

writer.writerow(header)


# ----------------------------
# WEBCAM
# ----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Errore apertura webcam")
    exit()

print("Recording motion dataset... premi Q per uscire")

frame_timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    # ----------------------------
    # SALVA TIME SERIES
    # ----------------------------
    if result.pose_landmarks:
        pose = result.pose_landmarks[0]

        row = [time.time()]  # timestamp reale

        for joint_name, idx in JOINTS.items():
            lm = pose[idx]
            row += [lm.x, lm.y, lm.z]

        robot_commands = human_pose_to_robot_commands(row[1:])  # escludi timestamp
        print("Robot commands:", robot_commands)

        writer.writerow(row)

    # ----------------------------
    # VISUALIZZAZIONE
    # ----------------------------
    annotated = draw_landmarks_on_image(rgb_frame, result)
    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    cv2.imshow("Motion Capture", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ----------------------------
# CLEANUP
# ----------------------------
csv_file.close()
cap.release()
cv2.destroyAllWindows()
detector.close()