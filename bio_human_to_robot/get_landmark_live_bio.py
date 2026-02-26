"""
get_landmark_live_bio.py
-------------------------
Records a biomechanically rigorous time series of anatomical joint angles
from a webcam, using the full processing pipeline defined in biomechanics.py.

What changes vs. get_landmark_live_ts.py:
    - Raw MediaPipe (x, y, z) coordinates → anatomical joint angles (degrees)
    - All angles are expressed in a body-fixed torso reference frame
      (invariant to position in scene and body size)
    - Low-pass Butterworth filter applied post-recording (offline, on the full
      buffer) to remove noise without introducing real-time latency
    - Joint limits enforced — no physically impossible postures saved

Output CSV columns:
    timestamp,
    l_shoulder_flexion, l_shoulder_abduction, l_shoulder_internal_rot,
    l_elbow_flexion, l_wrist_flexion,
    r_shoulder_flexion, r_shoulder_abduction, r_shoulder_internal_rot,
    r_elbow_flexion, r_wrist_flexion

Press Q to stop recording. Filtering and saving happen automatically at the end.

Requirements:
    - mediapipe, opencv-python, scipy installed
    - pose_landmarker_heavy.task model file in the same directory
    - biomechanics.py in the same directory
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

from biomechanics import (
    process_frame,
    filter_joint_angles_dataframe,
    JointAngles,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
MODEL_PATH  = SCRIPT_DIR / "pose_landmarker_heavy.task"
OUTPUT_PATH = SCRIPT_DIR / "human_motion_bio.csv"

# ---------------------------------------------------------------------------
# Recording parameters
# ---------------------------------------------------------------------------
WEBCAM_FPS    = 30.0    # expected webcam framerate (used for filter)
FILTER_CUTOFF = 6.0     # Butterworth low-pass cutoff in Hz
FILTER_ORDER  = 4       # Butterworth filter order


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """
    Draws pose landmarks and skeleton connections on a copy of the input image.

    Args:
        rgb_image: Input image in RGB format.
        detection_result: MediaPipe PoseLandmarker detection result.

    Returns:
        Annotated image in RGB format.
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


def overlay_angles(display: np.ndarray, angles: JointAngles) -> np.ndarray:
    """
    Overlays key joint angle values on the display frame for live monitoring.

    Displayed angles (left column = left arm, right column = right arm):
        SH F  = shoulder flexion
        SH AB = shoulder abduction
        EL    = elbow flexion

    Args:
        display: BGR image to annotate.
        angles:  Current frame JointAngles.

    Returns:
        Annotated BGR image.
    """
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    color_l    = (100, 220, 100)   # green = left
    color_r    = (100, 100, 220)   # red-ish = right
    thickness  = 1

    lines_l = [
        f"L SH Flex : {angles.l_shoulder_flexion:+.1f}",
        f"L SH Abd  : {angles.l_shoulder_abduction:+.1f}",
        f"L Elbow   : {angles.l_elbow_flexion:.1f}",
        f"L Wrist   : {angles.l_wrist_flexion:+.1f}",
    ]
    lines_r = [
        f"R SH Flex : {angles.r_shoulder_flexion:+.1f}",
        f"R SH Abd  : {angles.r_shoulder_abduction:+.1f}",
        f"R Elbow   : {angles.r_elbow_flexion:.1f}",
        f"R Wrist   : {angles.r_wrist_flexion:+.1f}",
    ]

    for i, (l_line, r_line) in enumerate(zip(lines_l, lines_r)):
        y = 30 + i * 22
        cv2.putText(display, l_line, (10,  y), font, font_scale, color_l, thickness)
        cv2.putText(display, r_line, (320, y), font, font_scale, color_r, thickness)

    return display


def run_biomechanical_recording(model_path : Path, output_path : Path, webcam_fps : float = WEBCAM_FPS, filter_cutoff : float = FILTER_CUTOFF) -> None:
    """
    Captures webcam video, computes anatomical joint angles per frame via the
    biomechanical pipeline, and saves a filtered CSV on exit.

    The Butterworth filter is applied offline on the full buffer after
    recording stops — this avoids the phase shift that online filtering
    would introduce on real-time data.

    Args:
        model_path:    Path to the MediaPipe pose landmarker .task model.
        output_path:   Path for the output CSV file.
        webcam_fps:    Expected frame rate of the webcam (for filter design).
        filter_cutoff: Low-pass cutoff frequency in Hz.
    """
    # MediaPipe setup
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print(f"Recording started — press Q to stop.")
    print(f"Output: {output_path}")
    print(f"Filter: Butterworth {filter_cutoff} Hz (applied post-recording)\n")

    frame_idx = 0
    recorded_frames = 0
    raw_buffer: list[dict] = []   # accumulates unfiltered angle rows

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect_for_video(mp_image, frame_idx)
        frame_idx += 1

        display = cv2.cvtColor(
            draw_landmarks_on_image(rgb_frame, result),
            cv2.COLOR_RGB2BGR,
        )

        if result.pose_landmarks:
            angles = process_frame(result.pose_landmarks[0])
            if angles is not None:
                row = {"timestamp": time.time()}
                row.update(angles.to_dict())
                raw_buffer.append(row)
                recorded_frames += 1
                display = overlay_angles(display, angles)

        # Status overlay
        status = (
            f"Frames: {recorded_frames}  |  "
            f"Filter: {filter_cutoff} Hz Butterworth (offline)"
        )
        cv2.putText(
            display, status, (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )
        cv2.imshow("Biomechanical Recording — press Q to quit", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    if not raw_buffer:
        print("No frames recorded — nothing to save.")
        return

    # Apply Butterworth filter offline on the full buffer
    print(f"\nApplying Butterworth filter ({filter_cutoff} Hz) to {recorded_frames} frames...")
    filtered_buffer = filter_joint_angles_dataframe(
        raw_buffer,
        cutoff_hz=filter_cutoff,
        fs_hz=webcam_fps,
    )

    # Write CSV
    header = ["timestamp"] + JointAngles.csv_header()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(filtered_buffer)

    print(f"Saved {recorded_frames} frames → {output_path}")
    print(f"Columns: {', '.join(header)}")


if __name__ == "__main__":
    run_biomechanical_recording(MODEL_PATH, OUTPUT_PATH)
