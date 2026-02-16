import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from pathlib import Path


# ----------------------------
# PATH MODELLO (robusto)
# ----------------------------
model_path = Path(__file__).parent / "pose_landmarker_heavy.task"


# ----------------------------
# FUNZIONE PER DISEGNARE LANDMARK
# ----------------------------
def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)

    if detection_result.pose_landmarks:
        for pose_landmarks in detection_result.pose_landmarks:
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
            )

    return annotated_image


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
# FILE OUTPUT
# ----------------------------
output_file = open("pose_output.txt", "w")


# ----------------------------
# WEBCAM
# ----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Errore apertura webcam")
    exit()

print("Premi Q per uscire")

frame_timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Converti in MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Pose detection
    result = detector.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    # Salva landmark su file
    if result.pose_landmarks:
        for pose in result.pose_landmarks:
            output_file.write(str(pose) + "\n")

    # Disegna landmark
    annotated = draw_landmarks_on_image(rgb_frame, result)

    # RGB → BGR per OpenCV
    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    # Mostra webcam
    cv2.imshow("Pose Estimation", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ----------------------------
# CLEANUP
# ----------------------------
output_file.close()
cap.release()
cv2.destroyAllWindows()
detector.close()