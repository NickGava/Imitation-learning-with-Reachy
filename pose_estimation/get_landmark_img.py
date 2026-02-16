# 1. import
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 2. configura modello
base_options = python.BaseOptions(model_asset_path="pose_landmarker_heavy.task")

options = vision.PoseLandmarkerOptions(
    base_options=base_options
)

detector = vision.PoseLandmarker.create_from_options(options)

# 3. carica immagine
image = mp.Image.create_from_file("image.jpg")

# 4. detection
result = detector.detect(image)

# salva su file
with open("pose_output.txt", "w") as f:
    for i, pose in enumerate(result.pose_landmarks):
        f.write(f"Pose {i}\n")

        for j, landmark in enumerate(pose):
            f.write(
                f"Landmark {j}: "
                f"x={landmark.x}, y={landmark.y}, z={landmark.z}, visibility={landmark.visibility}\n"
            )