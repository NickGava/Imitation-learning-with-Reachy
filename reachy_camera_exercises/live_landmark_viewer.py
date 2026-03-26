'''
live_landmark_viewer.py
=============================================================================
Visualizza in tempo reale i landmarks di MediaPipe Holistic sulle immagini
delle camere stereo di Reachy.

Due finestre vengono aperte:
  - "Left Camera  — MediaPipe"  : camera sinistra con landmark overlay
  - "Right Camera — Raw"        : camera destra (raw, senza elaborazione)
    → MediaPipe gira solo sulla camera sinistra per mantenere la latenza bassa.
    → Puoi abilitare l'elaborazione anche sulla destra cambiando PROCESS_BOTH.

Landmark disegnati (stessa logica di pose_estimation.py):
  - Corpo  : pose landmarks (esclusi i 11 punti del viso, coperti dalla Face Mesh)
  - Mani   : left_hand e right_hand landmarks
  - Viso   : Face Mesh tesselation (grigio, semitrasparente) +
             5 punti pipeline evidenziati in ciano
             (nose_tip=4, chin=152, forehead=10, left_eye=33, right_eye=263)

Controlli:
  Q  — chiudi e disconnetti

Requisiti:
  pip install mediapipe opencv-python reachy-sdk
'''

import time
import threading

import cv2
import mediapipe as mp
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory.interpolation import InterpolationMode

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

# Indici dei landmark del viso nella pose (nascosti perché coperti dalla Face Mesh)
_POSE_FACE_INDICES   = set(range(11))
_FACE_PIPELINE_IDXS  = {4, 152, 10, 33, 263}   # i 5 usati dalla pipeline


# ---------------------------------------------------------------------------
# Disegno dei landmark (identico a pose_estimation.py)
# ---------------------------------------------------------------------------
def draw_landmarks(image, results):
    """
    Disegna pose, mani e viso sull'immagine.
    - Corpo  : joint visibili (punti viso nascosti, coperti dalla Face Mesh)
    - Mani   : left & right hand
    - Viso   : tesselation grigia + 5 punti ciano della pipeline
    """
    h, w = image.shape[:2]

    # ---- Pose (corpo) ----
    if results.pose_landmarks:
        _hidden  = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=0, circle_radius=0)
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
            landmark_drawing_spec=lm_spec
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

    # ---- Viso: tesselation + 5 punti pipeline ----
    if results.face_landmarks:
        # Tesselation grigia, semi-trasparente
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(150, 150, 150), thickness=1
            ),
        )
        # 5 punti ciano evidenziati
        for idx in _FACE_PIPELINE_IDXS:
            lm = results.face_landmarks.landmark[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (255, 255, 0), -1)   # ciano pieno

    return image


# ---------------------------------------------------------------------------
# HUD (frame counter + info)
# ---------------------------------------------------------------------------
def draw_hud(image, frame_idx: int, fps: float, label: str) -> None:
    cv2.putText(image, f"Frame: {frame_idx}   {fps:.1f} FPS",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, label,
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


# ---------------------------------------------------------------------------
# Resize mantenendo aspect ratio
# ---------------------------------------------------------------------------
def _resize(frame, target_w: int):
    h, w = frame.shape[:2]
    target_h = int(h * target_w / w)
    return cv2.resize(frame, (target_w, target_h))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Connessione a Reachy ---
    print(f"Connessione a Reachy ({ROBOT_IP})…")
    reachy = ReachySDK(host=ROBOT_IP)

    if reachy.r_arm is None or reachy.l_arm is None:
        print("❌ Impossibile connettersi al robot.")
        return

    print("✅ Connesso!")

    # Testa la testa verso avanti prima di partire
    reachy.turn_on('head')
    reachy.head.look_at(0.5, 0.0, 0.0, 1.0,
                        interpolation_mode=InterpolationMode.MINIMUM_JERK)
    time.sleep(1.5)

    # --- Stato condiviso tra thread e loop principale ---
    latest = {
        'left':    None,   # frame BGR grezzo
        'right':   None,
        'results_left':  None,   # output MediaPipe camera sinistra
        'results_right': None,
    }
    lock      = threading.Lock()
    stop_flag = threading.Event()

    # ---------------------------------------------------------------------------
    # Thread: acquisizione frame dalle camere
    # ---------------------------------------------------------------------------
    def camera_thread():
        while not stop_flag.is_set():
            lf = reachy.left_camera.last_frame
            rf = reachy.right_camera.last_frame
            with lock:
                latest['left']  = lf
                latest['right'] = rf
            time.sleep(0.01)   # ~100 Hz max acquisizione

    t_cam = threading.Thread(target=camera_thread, daemon=True)
    t_cam.start()

    # ---------------------------------------------------------------------------
    # Loop principale: MediaPipe + display (gira nel thread principale per OpenCV)
    # ---------------------------------------------------------------------------
    frame_idx = 0
    t_prev    = time.perf_counter()
    fps_disp  = 0.0

    print("\nAvviato. Premi  Q  per uscire.\n")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:

        while True:
            with lock:
                left_raw  = latest['left']
                right_raw = latest['right']

            # Aspetta il primo frame valido
            if left_raw is None and right_raw is None:
                time.sleep(0.01)
                continue

            # ---- Elaborazione camera SINISTRA ----
            if left_raw is not None:
                left_rgb = cv2.cvtColor(left_raw, cv2.COLOR_BGR2RGB)
                left_rgb.flags.writeable = False
                results_left = holistic.process(left_rgb)
                left_rgb.flags.writeable = True
                left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)

                draw_landmarks(left_bgr, results_left)
                draw_hud(left_bgr, frame_idx, fps_disp, "Left Camera  —  MediaPipe Holistic")
                cv2.imshow("Left Camera  —  MediaPipe", _resize(left_bgr, DISPLAY_W))

            # ---- Elaborazione camera DESTRA ----
            if right_raw is not None:
                if PROCESS_BOTH:
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

                draw_hud(right_bgr, frame_idx, fps_disp, label_r)
                cv2.imshow("Right Camera", _resize(right_bgr, DISPLAY_W))

            # ---- FPS ----
            frame_idx += 1
            t_now = time.perf_counter()
            dt    = t_now - t_prev
            t_prev = t_now
            fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / dt if dt > 0 else fps_disp)

            # ---- Input ----
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Uscita…")
                break

    # --- Pulizia ---
    stop_flag.set()
    cv2.destroyAllWindows()

    reachy.turn_off_smoothly('head')
    print("Disconnesso.")


if __name__ == "__main__":
    main()