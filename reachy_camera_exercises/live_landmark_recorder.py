'''
live_landmark_recorder.py
=============================================================================
Visualizza in tempo reale i landmarks di MediaPipe Holistic sulle camere
stereo di Reachy e salva i joint di interesse negli stessi file CSV prodotti
da pose_estimation.py (pose.csv, right_hand.csv, left_hand.csv, face.csv).

Output (struttura identica a save_landmarks.py):
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/pose.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/right_hand.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/left_hand.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/face.csv

Controlli:
  R  — avvia / metti in pausa la registrazione
  Q  — chiudi, finalizza i CSV e disconnetti

Note:
  - MediaPipe gira solo sulla camera sinistra (default) per mantenere
    la latenza bassa. Imposta PROCESS_BOTH = True per elaborare entrambe.
  - I CSV vengono scritti dal thread principale dopo ogni elaborazione
    MediaPipe, quindi non serve sincronizzazione su file.
  - Se la cartella di output esiste già, i CSV vengono sovrascritti
    (stessa convenzione di pose_estimation.py).
'''

import sys
import time
import threading
from pathlib import Path

import cv2
import mediapipe as mp

# ---------------------------------------------------------------------------
# Importa save_landmarks dalla cartella del progetto
# (modifica il path se necessario)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent   # ← cartella di questo script
sys.path.insert(0, str(PROJECT_DIR))

try:
    from save_landmarks import init_csv_files, save_frame
    from config import DATA_ROOT
except ImportError:
    print("❌  Impossibile importare save_landmarks.py / config.py.")
    print("    Assicurati che questo script sia nella stessa cartella del progetto.")
    sys.exit(1)

try:
    from reachy_sdk import ReachySDK
    from reachy_sdk.trajectory.interpolation import InterpolationMode
except ImportError:
    print("❌  reachy_sdk non trovato. Installa con:  pip install reachy-sdk")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
ROBOT_IP     = "10.59.1.20"   # ← modifica con il tuo IP
PROCESS_BOTH = False           # True → MediaPipe anche sulla camera destra
DISPLAY_W    = 500             # larghezza finestra display (px)

# ---------------------------------------------------------------------------
# Setup MediaPipe Holistic
# ---------------------------------------------------------------------------
mp_holistic       = mp.solutions.holistic
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

_POSE_FACE_INDICES  = set(range(11))    # punti viso nella pose (nascosti)
_FACE_PIPELINE_IDXS = {4, 152, 10, 33, 263}   # 5 punti pipeline (ciano)


# ---------------------------------------------------------------------------
# Disegno dei landmark
# ---------------------------------------------------------------------------
def draw_landmarks(image, results):
    """
    Disegna pose (corpo), mani e viso sull'immagine BGR.
    Logica identica a pose_estimation.py.
    """
    h, w = image.shape[:2]

    # ---- Corpo ----
    if results.pose_landmarks:
        _hidden   = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=0, circle_radius=0)
        def_style = mp_drawing_styles.get_default_pose_landmarks_style()
        lm_spec = {
            idx: (_hidden if idx in _POSE_FACE_INDICES
                  else def_style.get(idx, mp_drawing.DrawingSpec()))
            for idx in range(33)
        }
        body_connections = [
            c for c in mp_holistic.POSE_CONNECTIONS
            if c[0] not in _POSE_FACE_INDICES and c[1] not in _POSE_FACE_INDICES
        ]
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, body_connections,
            landmark_drawing_spec=lm_spec,
        )

    # ---- Mano destra ----
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    # ---- Mano sinistra ----
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    # ---- Viso: tesselation + 5 punti ciano ----
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(150, 150, 150), thickness=1,
            ),
        )
        for idx in _FACE_PIPELINE_IDXS:
            lm = results.face_landmarks.landmark[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (255, 255, 0), -1)

    return image


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
def draw_hud(image, frame_idx: int, rec_frame: int, fps: float,
             recording: bool, label: str) -> None:
    """
    Sovrappone informazioni sull'immagine.
      - Verde  : stato normale / in preview
      - Rosso  : REC attivo  (con contatore frame registrati)
    """
    color = (0, 0, 255) if recording else (0, 255, 0)
    rec_label = f"● REC  {rec_frame} frames" if recording else "■ STANDBY  (R = avvia)"

    cv2.putText(image, rec_label,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(image, f"Frame tot: {frame_idx}   {fps:.1f} FPS",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
    cv2.putText(image, label,
                (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 2)


# ---------------------------------------------------------------------------
# Resize mantenendo aspect ratio
# ---------------------------------------------------------------------------
def _resize(frame, target_w: int):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (target_w, int(h * target_w / w)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Input subject / exercise / video ---
    print("=== Live Landmark Recorder ===")
    try:
        subject_num  = int(input("Subject number:  ").strip())
        exercise_num = int(input("Exercise number: ").strip())
        video_num    = int(input("Video number:    ").strip())
    except ValueError:
        print("Errore: i valori devono essere interi.")
        return

    subject_name  = f"subject_{subject_num:03d}"
    exercise_name = f"exercise_{exercise_num:03d}"
    video_name    = f"video_{video_num:03d}"

    landmarks_folder = DATA_ROOT / "landmarks" / subject_name / exercise_name / video_name

    # --- Inizializza i CSV (sovrascrive se esistono) ---
    csv_paths = init_csv_files(landmarks_folder)
    print(f"Output → {landmarks_folder}\n")

    # --- Connessione a Reachy ---
    print(f"Connessione a Reachy ({ROBOT_IP})…")
    reachy = ReachySDK(host=ROBOT_IP)

    if reachy.r_arm is None or reachy.l_arm is None:
        print("❌ Impossibile connettersi al robot.")
        return

    print("✅ Connesso!")

    reachy.turn_on('head')
    reachy.head.look_at(0.5, 0.0, 0.0, 1.0,
                        interpolation_mode=InterpolationMode.MINIMUM_JERK)
    time.sleep(1.5)

    # --- Stato condiviso: cattura frame ---
    latest    = {'left': None, 'right': None}
    lock      = threading.Lock()
    stop_flag = threading.Event()

    def camera_thread():
        while not stop_flag.is_set():
            lf = reachy.left_camera.last_frame
            rf = reachy.right_camera.last_frame
            with lock:
                latest['left']  = lf
                latest['right'] = rf
            time.sleep(0.01)

    t_cam = threading.Thread(target=camera_thread, daemon=True)
    t_cam.start()

    # --- Stato registrazione ---
    recording  = False      # toggle con R
    frame_idx  = 0          # frame totali elaborati da MediaPipe
    rec_frame  = 0          # frame scritti nei CSV durante la sessione REC corrente
    fps_disp   = 0.0
    t_prev     = time.perf_counter()

    print("Premi  R  per avviare/fermare la registrazione.")
    print("Premi  Q  per uscire.\n")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:

        while True:
            with lock:
                left_raw  = latest['left']
                right_raw = latest['right']

            if left_raw is None and right_raw is None:
                time.sleep(0.01)
                continue

            # ---- Elaborazione camera SINISTRA (MediaPipe) ----
            results_left = None
            if left_raw is not None:
                left_rgb = cv2.cvtColor(left_raw, cv2.COLOR_BGR2RGB)
                left_rgb.flags.writeable = False
                results_left = holistic.process(left_rgb)
                left_rgb.flags.writeable = True
                left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)

                draw_landmarks(left_bgr, results_left)
                draw_hud(left_bgr, frame_idx, rec_frame, fps_disp,
                         recording, "Left Camera  —  MediaPipe Holistic")
                cv2.imshow("Left Camera  —  MediaPipe", _resize(left_bgr, DISPLAY_W))

            # ---- Camera DESTRA ----
            if right_raw is not None:
                if PROCESS_BOTH and right_raw is not None:
                    right_rgb = cv2.cvtColor(right_raw, cv2.COLOR_BGR2RGB)
                    right_rgb.flags.writeable = False
                    results_right = holistic.process(right_rgb)
                    right_rgb.flags.writeable = True
                    right_bgr = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)
                    draw_landmarks(right_bgr, results_right)
                    label_r = "Right Camera  —  MediaPipe Holistic"
                else:
                    right_bgr = right_raw.copy()
                    label_r   = "Right Camera  —  Raw"

                draw_hud(right_bgr, frame_idx, rec_frame, fps_disp,
                         recording, label_r)
                cv2.imshow("Right Camera", _resize(right_bgr, DISPLAY_W))

            # ---- Salvataggio CSV (solo se in REC e risultati disponibili) ----
            if recording and results_left is not None:
                save_frame(results_left, frame_idx, csv_paths)
                rec_frame += 1

            frame_idx += 1

            # ---- FPS smoothed ----
            t_now  = time.perf_counter()
            dt     = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_disp = 0.9 * fps_disp + 0.1 / dt

            # ---- Input tasti ----
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                recording = not recording
                if recording:
                    print(f"▶ REC avviata  (frame_idx={frame_idx})")
                else:
                    print(f"■ REC fermata  — frames salvati: {rec_frame}")

            elif key == ord('q'):
                print("Uscita…")
                break

    # --- Riepilogo finale ---
    print(f"\n{'='*50}")
    print(f"  Frames totali elaborati : {frame_idx}")
    print(f"  Frames salvati nei CSV  : {rec_frame}")
    print(f"  Output → {landmarks_folder}")
    print(f"{'='*50}")

    # --- Pulizia ---
    stop_flag.set()
    cv2.destroyAllWindows()

    reachy.turn_off_smoothly('head')
    print("Disconnesso.")


if __name__ == "__main__":
    main()