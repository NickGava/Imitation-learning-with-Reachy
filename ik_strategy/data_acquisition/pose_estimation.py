'''
pose_estimation.py
=============================================================================
Main entry point for the landmark extraction pipeline.

Loads a video file, runs MediaPipe Pose on each frame and saves the
detected pose landmarks to CSV via save_landmarks.py.

Controls:
  P : pause / resume
  Q : quit

Every frame is processed and saved in a line, each video corresponds to a single exercise.

Input:
  data/raw_data/subject_XXX/exercise_XXX/video_XXX.mp4

Output:
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/pose.csv
'''

import cv2
import mediapipe as mp

from utilities.config import DATA_ROOT
from save_landmarks import init_csv_files, save_frame
from utilities.ask_inputs import ask_inputs

# Setup MediaPipe Pose
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Display window width in pixels
DISPLAY_WIDTH = 500

# Hide face landmarks (indices 0-10) from the overlay - not used in pipeline
_POSE_FACE_INDICES  = set(range(11))
_HIDDEN_SPEC        = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=0, circle_radius=0)
_DEFAULT_POSE_STYLE = mp_drawing_styles.get_default_pose_landmarks_style()

_BODY_CONNECTIONS   = [c for c in mp_pose.POSE_CONNECTIONS
                       if c[0] not in _POSE_FACE_INDICES and c[1] not in _POSE_FACE_INDICES]
_POSE_LANDMARK_SPEC = {idx: (_HIDDEN_SPEC if idx in _POSE_FACE_INDICES
                              else _DEFAULT_POSE_STYLE.get(idx, mp_drawing.DrawingSpec()))
                       for idx in range(33)}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def draw_landmarks(image, results):
    """Draws body pose landmarks on the frame (face landmarks hidden)."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, _BODY_CONNECTIONS,
            landmark_drawing_spec=_POSE_LANDMARK_SPEC,
        )
    return image

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    subject_name, exercise_name, video_name = ask_inputs()
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
    print(f"Video loaded: {video_path.relative_to(DATA_ROOT.parent)}  ({total_frames} frames @ {fps:.1f} fps)")

    # __________ Initialize CSV files for this video __________ 
    landmarks_folder = DATA_ROOT / "landmarks" / subject_name / exercise_name / video_name
    csv_paths = init_csv_files(landmarks_folder)

    frame_idx = 0
    paused    = False

    print(f"\nPress 'P' to pause/resume, 'Q' to quit.")

    with mp_pose.Pose(
        model_complexity=2,             # 0=lite, 1=full, 2=heavy (max precision)
        smooth_landmarks=True,          # temporal smoothing across frames
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Video ended.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe works in RGB
                frame_rgb.flags.writeable = False       # avoids internal memory copy

                results = pose.process(frame_rgb)   # returns pose landmarks

                frame_rgb.flags.writeable = True
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                save_frame(results, frame_idx, csv_paths, fps)
                frame_idx += 1

                frame_bgr = draw_landmarks(frame_bgr, results)

            # _____ HUD _____
            pause_label = "[PAUSED]" if paused else ""
            cv2.putText(frame_bgr, f"Frame: {frame_idx} / {total_frames}" + pause_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (17, 106, 1), 2)
            cv2.putText(frame_bgr, f"{subject_name} / {exercise_name} / {video_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (17, 106, 1), 2)

            # _____ Show processing video _____ 
            h, w = frame_bgr.shape[:2]
            display = cv2.resize(frame_bgr, (DISPLAY_WIDTH, int(h * DISPLAY_WIDTH / w)))
            # cv2.imshow("Reachy - Landmark Extraction", display)

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