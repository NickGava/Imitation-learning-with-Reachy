'''
pose_estimation.py
=============================================================================
Main entry point for the landmark extraction pipeline.

Loads a video file, runs MediaPipe Holistic on each frame and saves the
detected landmarks to CSV via save_landmarks.py.

Controls:
  P : pause / resume
  Q : quit

Every frame is processed and saved, each video corresponds to a single exercise.

Input:
  data/raw_data/subject_XXX/exercise_XXX/video_XXX.mp4

Output:
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/pose.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/right_hand.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/left_hand.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/face.csv
'''

import cv2
import mediapipe as mp
from config import DATA_ROOT
from save_landmarks import init_csv_files, save_frame

# Setup MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Display window width in pixels
DISPLAY_WIDTH = 800

# Constants used in the helper
_POSE_FACE_INDICES     = set(range(11))
_FACE_PIPELINE_INDICES = {4, 152, 10, 33, 263}
_HIDDEN_SPEC           = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=0, circle_radius=0)  # Indices 0-10 are face landmarks (nose, eyes, ears, mouth), they are
                                                                                                # hidden because the Face Mesh already covers them with higher quality.
_GREY_CONNECTION_SPEC  = mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=1)
_DEFAULT_POSE_STYLE    = mp_drawing_styles.get_default_pose_landmarks_style()
_DEFAULT_HAND_LM_STYLE = mp_drawing_styles.get_default_hand_landmarks_style()
_DEFAULT_HAND_CN_STYLE = mp_drawing_styles.get_default_hand_connections_style()

_BODY_CONNECTIONS = [c for c in mp_holistic.POSE_CONNECTIONS if c[0] not in _POSE_FACE_INDICES and c[1] not in _POSE_FACE_INDICES]
_POSE_LANDMARK_SPEC = {idx: (_HIDDEN_SPEC if idx in _POSE_FACE_INDICES else _DEFAULT_POSE_STYLE.get(idx, mp_drawing.DrawingSpec())) for idx in range(33)}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def draw_landmarks(image, results):
    """
    Draws pose, hand and face landmarks on the frame.

    Face drawing:
      - Full Face Mesh tesselation (grey, semi-transparent) for context
      - 5 pipeline landmarks highlighted in cyan:
          nose_tip (4), chin (152), forehead (10),
          left_eye (33), right_eye (263)
    """
    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, _BODY_CONNECTIONS, landmark_drawing_spec=_POSE_LANDMARK_SPEC)

    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, _DEFAULT_HAND_LM_STYLE, _DEFAULT_HAND_CN_STYLE)

    # Face
    if results.face_landmarks:
        # Full tesselation (grey, thin lines, no dots)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, landmark_drawing_spec=None, 
                                  connection_drawing_spec = _GREY_CONNECTION_SPEC)

        # 5 useful landmarks highlighted in cyan
        h, w = image.shape[:2]
        for idx in _FACE_PIPELINE_INDICES:
            lm = results.face_landmarks.landmark[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 4, (255, 255, 0), -1)   # cyan filled dot

    # Left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, _DEFAULT_HAND_LM_STYLE, _DEFAULT_HAND_CN_STYLE)

    return image

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # User input 
    try:
        subject_num  = int(input("Subject number:  ").strip())
        exercise_num = int(input("Exercise number: ").strip())
        video_num    = int(input("Video number:    ").strip())
    except ValueError:
        print("Error: all values must be integers.")
        return

    subject_name  = f"subject_{subject_num:03d}"
    exercise_name = f"exercise_{exercise_num:03d}"
    video_name    = f"video_{video_num:03d}"

    video_path = DATA_ROOT / "raw_data" / subject_name / exercise_name / f"{video_name}.mp4"

    if not video_path.exists():
        print(f"Error: video not found at {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video loaded: {video_path}  ({total_frames} frames @ {fps:.1f} fps)")

    # Initialize CSV files for this video
    landmarks_folder = DATA_ROOT / "landmarks" / subject_name / exercise_name / video_name
    csv_paths = init_csv_files(landmarks_folder)

    frame_idx = 0
    paused    = False

    print("Press 'P' to pause/resume, 'Q' to quit.")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,    # if the tracking goes under this threshold, MediaPipe restarts the complete scanner
        model_complexity=1              # 0=lite, 1=full, 2=heavy
    ) as holistic:

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Video ended.")
                    break

                # MediaPipe works in RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False       # avoids internal memory copy

                results = holistic.process(frame_rgb)   # returns pose, hands, face

                frame_rgb.flags.writeable = True
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                save_frame(results, frame_idx, csv_paths, fps)
                frame_idx += 1

                frame_bgr = draw_landmarks(frame_bgr, results)

            # HUD
            pause_label = "[PAUSED]" if paused else ""
            cv2.putText(frame_bgr, f"Frame: {frame_idx} / {total_frames}" + pause_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"{subject_name} / {exercise_name} / {video_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show processing video
            h, w = frame_bgr.shape[:2]
            display = cv2.resize(frame_bgr, (DISPLAY_WIDTH, int(h * DISPLAY_WIDTH / w)))
            #cv2.imshow("Reachy - Landmark Extraction", display)

            # waitKey(1) when playing, waitKey(0) when paused (blocks until keypress)
            key = cv2.waitKey(1 if not paused else 0) & 0xFF

            if key == ord('p'):
                paused = not paused
                print("[PAUSED]" if paused else "[RESUMED]")

            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Frames processed: {frame_idx} / {total_frames}")

if __name__ == "__main__":
    main()