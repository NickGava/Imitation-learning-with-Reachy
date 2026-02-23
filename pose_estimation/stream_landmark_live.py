"""
stream_landmark_live.py
------------------------
Real-time human pose estimation from a webcam feed using MediaPipe.
Annotated frames (with skeleton overlay) are sent via TCP socket to Unity
for display in the RawImage panel.

Unity must be running and listening on the configured HOST/PORT before
starting this script.

Press Q in the local OpenCV preview window to stop.

Requirements:
    - mediapipe, opencv-python installed.
    - pose_landmarker_heavy.task model file in the same directory.
"""

import socket
import struct
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from pathlib import Path


# --- Config ---
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH  = SCRIPT_DIR / "pose_landmarker_heavy.task"

UNITY_HOST = "127.0.0.1"   # localhost — change to robot IP if needed
UNITY_PORT = 5001
JPEG_QUALITY = 80           # 0-100, lower = faster but more compressed


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """
    Draws pose landmarks and skeleton connections on a copy of the input image.

    Args:
        rgb_image: Input image in RGB format.
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


def encode_frame(bgr_frame: np.ndarray) -> bytes:
    """
    Encodes a BGR frame as a JPEG byte string.

    Args:
        bgr_frame: Frame in BGR format (as returned by OpenCV).

    Returns:
        bytes: JPEG-encoded frame.
    """
    _, buffer = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buffer.tobytes()


def send_frame(sock: socket.socket, data: bytes) -> bool:
    """
    Sends a frame over the socket, prefixed by its size as a 4-byte big-endian integer.

    Protocol:
        [4 bytes: frame size] [N bytes: JPEG data]

    Args:
        sock: Connected TCP socket.
        data: JPEG-encoded frame bytes.

    Returns:
        bool: True if successful, False if the connection was lost.
    """
    try:
        size = struct.pack(">I", len(data))  # 4-byte big-endian unsigned int
        sock.sendall(size + data)
        return True
    except (BrokenPipeError, ConnectionResetError, OSError):
        return False


def run_stream(model_path: Path) -> None:
    """
    Main loop: opens webcam, detects pose, annotates frames, and streams to Unity.

    Args:
        model_path: Path to the MediaPipe pose landmarker model (.task file).
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

    # Connect to Unity
    print(f"Connecting to Unity at {UNITY_HOST}:{UNITY_PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((UNITY_HOST, UNITY_PORT))
    except ConnectionRefusedError:
        print("Connection refused — make sure Unity is running and in Play mode.")
        cap.release()
        return

    print("Connected. Streaming... press Q in preview window to stop.")

    frame_timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect_for_video(mp_image, frame_timestamp)
        frame_timestamp += 1

        # Draw skeleton on frame
        annotated_rgb = draw_landmarks_on_image(rgb_frame, result)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        # Send frame to Unity
        jpeg_data = encode_frame(annotated_bgr)
        if not send_frame(sock, jpeg_data):
            print("Connection to Unity lost.")
            break

        # Local preview window
        #cv2.imshow("Streaming to Unity — press Q to quit", annotated_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    sock.close()
    detector.close()
    print("Stream ended.")


if __name__ == "__main__":
    run_stream(MODEL_PATH)
