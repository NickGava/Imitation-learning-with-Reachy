'''
pose_estimation.py
=============================================================================
Main entry point for the landmark extraction pipeline.

Loads a video file, runs MediaPipe Holistic on each frame and saves the
detected landmarks to CSV via save_landmarks.py.

Controls:
  P  — pause / resume
  Q  — quit

Every frame is processed and saved — each video corresponds to a single
movement, so no gesture marking is needed.

Input:
  data/raw_data/subject_XXX/exercise_XXX/video_XXX.mp4

Output:
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/pose.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/right_hand.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/left_hand.csv
'''

import cv2
import mediapipe as mp
from config import DATA_ROOT
from save_landmarks import init_csv_files, save_frame

# Setup MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Display window width in pixels — height is scaled automatically to keep aspect ratio
DISPLAY_WIDTH = 800

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def draw_landmarks(image, results):
    """
    Draws only pose and hand landmarks on the frame (no face).
    """
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- User input ---
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
        min_tracking_confidence=0.5,    # se il tracking scende sotto questa soglia,
                                        # MediaPipe rilancia il rilevamento completo
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

                results = holistic.process(frame_rgb)   # restituisce pose, mani, viso

                frame_rgb.flags.writeable = True
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                save_frame(results, frame_idx, csv_paths)
                frame_idx += 1

                frame_bgr = draw_landmarks(frame_bgr, results)

            # HUD
            pause_label = "[PAUSED]" if paused else ""
            cv2.putText(frame_bgr, f"Frame: {frame_idx} / {total_frames}" + pause_label,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"{subject_name} / {exercise_name} / {video_name}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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