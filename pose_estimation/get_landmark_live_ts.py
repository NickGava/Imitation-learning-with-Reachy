"""
get_landmark_live_ts.py
------------------------
Records a time series of human upper-body pose landmarks from a webcam
and saves them to a CSV file for offline analysis and learning.

For each detected frame, the (x, y, z) coordinates of the following
MediaPipe landmarks are recorded:
    - Left/right shoulder (11, 12)
    - Left/right elbow    (13, 14)
    - Left/right wrist    (15, 16)

Output CSV format:
    timestamp, l_shoulder_x, l_shoulder_y, l_shoulder_z, l_elbow_x, ...

Press Q to stop recording and close the window.

Requirements:
    - mediapipe, opencv-python installed.
    - pose_landmarker_heavy.task model file in the same directory.
"""

import csv
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from pathlib import Path


# --- Paths ---
SCRIPT_DIR  = Path(__file__).parent
MODEL_PATH  = SCRIPT_DIR / "pose_landmarker_heavy.task"
OUTPUT_PATH = SCRIPT_DIR / "reachy_motion_dataset.csv"

# MediaPipe landmark indices for upper-body joints relevant to Reachy
JOINTS = {
    "head":       0,
    "l_shoulder": 11,
    "l_elbow":    13,
    "l_wrist":    15,
    "l_finger":   19,
    "l_thumb":    21,
    "r_shoulder": 12,
    "r_elbow":    14,
    "r_wrist":    16,
    "r_finger":   20,
    "r_thumb":    22,
}


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """
    Draws pose landmarks and skeleton connections on a copy of the input image.

    Args:
        rgb_image: Input image in RGB format (H x W x 3 numpy array).
        detection_result: MediaPipe PoseLandmarker detection result.

    Returns:
        np.ndarray: Annotated image in RGB format.
    """
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


def build_csv_header() -> list:
    """
    Builds the CSV header row based on the JOINTS dictionary.

    Returns:
        list: Header row with 'timestamp' followed by '<joint>_x/y/z' columns.
    """
    header = ["timestamp"]
    for joint_name in JOINTS:
        header += [f"{joint_name}_x", f"{joint_name}_y", f"{joint_name}_z"]
    return header


def extract_landmark_row(pose_landmarks, timestamp: float) -> list:
    """
    Extracts (x, y, z) coordinates for each tracked joint from a pose detection result
    and prepends the timestamp.

    Args:
        pose_landmarks: List of landmark objects from MediaPipe (one pose).
        timestamp: Unix timestamp (float) of the captured frame.

    Returns:
        list: A flat row [timestamp, x, y, z, x, y, z, ...] ready to write to CSV.
    """
    row = [timestamp]
    for joint_name, idx in JOINTS.items():
        lm = pose_landmarks[idx]
        row += [lm.x, lm.y, lm.z]
    return row


def run_motion_recording(model_path: Path, output_path: Path) -> None:
    """
    Runs real-time pose estimation and records landmark time series to a CSV file.

    Opens the default webcam, detects upper-body landmarks on each frame,
    displays the annotated video, and writes one CSV row per detected frame.
    A frame counter is displayed on screen during recording.

    Args:
        model_path: Path to the MediaPipe pose landmarker model (.task file).
        output_path: Path to the output CSV file.
    """
    # Set up MediaPipe detector
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print(f"Recording started — press Q to stop. Saving to: {output_path}")

    frame_timestamp = 0
    recorded_frames = 0

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(build_csv_header())

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = detector.detect_for_video(mp_image, frame_timestamp)
            frame_timestamp += 1

            # Record landmarks if a pose is detected
            if result.pose_landmarks:
                pose = result.pose_landmarks[0]
                row = extract_landmark_row(pose, time.time())
                writer.writerow(row)
                recorded_frames += 1

            # Draw skeleton overlay
            annotated = draw_landmarks_on_image(rgb_frame, result)
            display = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

            # Show frame counter on screen
            cv2.putText(
                display,
                f"Recorded frames: {recorded_frames}",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 200, 0),
                thickness=2,
            )

            cv2.imshow("Motion Recording — press Q to quit", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"✅ Recording complete. {recorded_frames} frames saved to: {output_path}")


if __name__ == "__main__":
    run_motion_recording(MODEL_PATH, OUTPUT_PATH)
