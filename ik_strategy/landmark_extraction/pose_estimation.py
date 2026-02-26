'''
pose_estimation.py
=============================================================================
Main entry point for the landmark extraction pipeline.

Opens the webcam, runs MediaPipe Holistic on each frame and saves the
detected landmarks to CSV via save_landmarks.py.

Controls:
  S  — start recording a gesture  (gesture_id incremented automatically)
  E  — stop  recording the gesture
  Q  — quit the session

While idle all frames are still saved (gesture = 0) so the full session
timeline is preserved. Only the frames flagged with gesture = 1 will be
used for training.

Output:
  data/session_XXX/pose.csv
  data/session_XXX/right_hand.csv
  data/session_XXX/left_hand.csv
'''

import cv2
import mediapipe as mp

from save_landmarks import init_csv_files, save_frame

# Setup MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Function to draw landmarks on the frame (for visualization)
def draw_landmarks(image, results):
    """
    Draws only pose and hand landmarks on the frame (no face).
    """
    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    # Left hand
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
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    # Initialization CSV for current session
    csv_paths = init_csv_files("data/session_001")
    frame_idx = 0

    # Gesture recording state
    gesture_active = False  # True while recording a gesture (S pressed, E not yet pressed)
    gesture_id = 0          # incremented at each new gesture recording

    print("Webcam started. Press 'S' to start recording, 'E' to stop, 'Q' to quit.")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,   # soglia di confidenza per rilevamento iniziale 
        min_tracking_confidence=0.5,    # soglia di confidenza per tracking continuo (dopo il ril. iniziale usa un tracker leggero,
                                        # se il tracking fallisce, cioè è sotto questa soglia torna al rilevamento completo)
        model_complexity=1  # 0=lite, 1=full, 2=heavy
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: frame not received.")
                break

            # MediaPipe works in RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False               # Impostare il frame come non scrivibile prima di passarlo a MediaPipe serve a evitare una copia interna dei dati in memoria 

            results = holistic.process(frame_rgb)           # Restituisce un oggetto con i landmark rilevati (pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks)

            frame_rgb.flags.writeable = True
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Saves the frame landmarks into the CSV files.
            save_frame(results, frame_idx, csv_paths, gesture_id, gesture_active)
            frame_idx += 1

            # Draw landmarks on the frame
            frame_bgr = draw_landmarks(frame_bgr, results)

            # Show info on the frame
            rec_color = (0, 0, 255) if gesture_active else (0, 255, 0)
            rec_label = f"REC gesture_id={gesture_id}" if gesture_active else "IDLE  (S=start)"
            cv2.putText(frame_bgr, rec_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)
            cv2.putText(frame_bgr, f"Frame: {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Reachy - Landmark Extraction", frame_bgr)

            key = cv2.waitKey(1) & 0xFF         # restituisce il codice ASCII del tasto premuto (-1 se nessun tasto premuto), & 0xFF maschera per limitare il valore a 8 bit (0-255)

            if key == ord('s') and not gesture_active:
                gesture_active = True
                gesture_id += 1
                print(f"[REC] Started gesture_id={gesture_id}")

            elif key == ord('e') and gesture_active:
                gesture_active = False
                print(f"[REC] Stopped gesture_id={gesture_id}")

            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended. Total frames saved: {frame_idx}  |  Gestures recorded: {gesture_id}")


if __name__ == "__main__":
    main()