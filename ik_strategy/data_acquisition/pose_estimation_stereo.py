"""
TODO: docstring
"""
# __________ Imports __________
import csv
import cv2
import math
import mediapipe as mp
import numpy as np

from utilities.ask_inputs import ask_inputs
from utilities.config import DATA_ROOT, POSE_INDICES
from save_landmarks import init_csv_files
from stereo_config import (
    LEFT_K, LEFT_D, RIGHT_K, RIGHT_D, R, T,
    SGBM_MIN_DISP, SGBM_NUM_DISP, SGBM_BLOCK_SIZE,
    SGBM_P1, SGBM_P2, SGBM_DISP12DIFF, SGBM_UNIQUENESS,
    SGBM_SPECKLE_WIN, SGBM_SPECKLE_RNG,
)

# __________ Tunable variables __________
DISPLAY_WIDTH = 400
Z_CONF_THRESHOLD = 0.2        # confidence threshold for z stereo
MAX_Z_DELTA_M = 0.05          # Threshold for spikes of z stereo
MAX_PERSISTENCE_FRAMES = 25   # Threshold for old values of z stereo
BODY_Z_MAX = 1.2              # Threshold for body center z coordinate
OEF_MIN_CUTOFF = 1.0          # One Euro Filter: base cutoff frequency (Hz) — lower = smoother
OEF_BETA = 0.1                # One Euro Filter: speed coefficient — higher = less lag on fast motion

Z_MIN = 0.1   # m - minimo atteso
Z_MAX = 1.8   # m - massimo atteso (taglia lo sfondo lontano)

# __________ MediaPipe setup __________
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# -------------------------------------------------------
# One Euro Filter (applied to stereo z of elbow/wrist)
# -------------------------------------------------------
class _OneEuroFilter:
    """
    Adaptive low-pass filter that reduces lag on fast motion.
    Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter", CHI 2012.
    """
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x_prev    = None
        self._dx_prev   = 0.0
        self._t_prev    = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._t_prev is None:            # primo campione: inizializza e restituisci invariato
            self._x_prev = x
            self._t_prev = t
            return x
        dt = t - self._t_prev
        if dt <= 0:
            return self._x_prev
        # Stima derivata e filtraggio adattivo
        dx      = (x - self._x_prev) / dt
        a_d     = self._alpha(self.d_cutoff, dt)
        dx_hat  = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff  = self.min_cutoff + self.beta * abs(dx_hat)
        a       = self._alpha(cutoff, dt)
        x_hat   = a * x + (1.0 - a) * self._x_prev
        # Aggiorno stato
        self._x_prev  = x_hat
        self._dx_prev = dx_hat
        self._t_prev  = t
        return x_hat


# -------------------------------------------------------
# Stereo rectification and SGBM matcher
# -------------------------------------------------------
def _build_rectification_map(img_size):
    """
    Build rectification maps for stereo vision.    
    Args:
        img_size: tuple of (height, width)
    Returns:
        tuple: (map_L1, map_L2, map_R1, map_R2, P_L, P_R, Q)
    """
    R1, R2, P_L, P_R, Q, _, _ = cv2.stereoRectify(
        LEFT_K, LEFT_D, RIGHT_K, RIGHT_D, img_size,
        R, T, flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.3
    )
    map_L1, map_L2 = cv2.initUndistortRectifyMap(LEFT_K, LEFT_D, R1, P_L, img_size, cv2.CV_32FC1)
    map_R1, map_R2 = cv2.initUndistortRectifyMap(RIGHT_K, RIGHT_D, R2, P_R, img_size, cv2.CV_32FC1)
    return map_L1, map_L2, map_R1, map_R2, P_L, P_R, Q

def _build_sgbm():
    """
    Creates and returns a configured StereoSGBM matcher
    """
    return cv2.StereoSGBM_create(
        minDisparity      = SGBM_MIN_DISP,
        numDisparities    = SGBM_NUM_DISP,
        blockSize         = SGBM_BLOCK_SIZE,
        P1                = SGBM_P1,
        P2                = SGBM_P2,
        disp12MaxDiff     = SGBM_DISP12DIFF,
        uniquenessRatio   = SGBM_UNIQUENESS,
        speckleWindowSize = SGBM_SPECKLE_WIN,
        speckleRange      = SGBM_SPECKLE_RNG,
        mode              = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


# -------------------------------------------------------
# Depth helpers
# -------------------------------------------------------
def _disparity_to_depth_map(disp_L, P_L):
    """
    int16 SGBM disparity (x16) -> float32 depth map in meters. Nan where invalid
    """
    disp = disp_L.astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    fx = P_L[0, 0]
    baseline = abs(T.ravel()[0])
    return fx * baseline / disp

def _stereo_confidence(disp_L, disp_R, u, v, threshold=1.0):
    """
    Left-Right Consistency check.
    Conf = 1 se i due matcher concordano, 0 se discordano.
    """
    ui, vi = int(round(u)), int(round(v))
    h, w = disp_L.shape

    if not (0 <= ui < w and 0 <= vi < h):
        return 0.0

    d_L = disp_L[vi, ui] / 16.0
    if d_L <= 0:
        return 0.0

    ui_R = int(round(ui - d_L))
    if not (0 <= ui_R < w):
        return 0.0

    d_R = -disp_R[vi, ui_R] / 16.0
    diff = abs(d_L - d_R)
    return max(0.0, 1.0 - diff / threshold)


# -------------------------------------------------------
# Temporal persistence with spike rejection and max-age expiry
# -------------------------------------------------------
def _resolve_z(name, new_z, last_z, last_z_age, frame_idx):
    """
    Returns the Z to use for this frame, or None.
    Accepts new_z only if not None and within MAX_Z_DELTA_M of cached value.
    Falls back to cached Z if not older than MAX_PERSISTENCE_FRAMES.
    """
    # __________ Prevent spikes __________
    if new_z is not None:
        prev = last_z.get(name)
        if prev is None or abs(new_z - prev) <= MAX_Z_DELTA_M: 
            last_z[name]     = new_z
            last_z_age[name] = frame_idx
            return new_z

    # __________ Prevent using too old values __________
    if name in last_z:
        age = frame_idx - last_z_age.get(name, frame_idx)
        if age <= MAX_PERSISTENCE_FRAMES:
            return last_z[name]
        del last_z[name]
        last_z_age.pop(name, None)

    return None


# -------------------------------------------------------
# Draw landmarks
# -------------------------------------------------------
def _draw_landmarks(image, results):
    if results.pose_landmarks:
        default = mp_drawing_styles.get_default_pose_landmarks_style()
        spec = {i: default.get(i, mp_drawing.DrawingSpec()) for i in range(33)}
        conns = [c for c in mp_pose.POSE_CONNECTIONS]
        mp_drawing.draw_landmarks(image, results.pose_landmarks, conns, landmark_drawing_spec=spec)
    return image


# -------------------------------------------------------
# Extract Pose Row
# -------------------------------------------------------
def _extract_pos_row(results, i_frame, timestamp, depth_map, P_L, frame_w, frame_h, last_z_pose, last_z_age, stereo_oef, disp_L, disp_R):
    """
    TODO: docstring 
    """
    # __________ Setup __________
    base = [i_frame, timestamp]
    lms = results.pose_landmarks            # Get standard landmarks
    w_lms = results.pose_world_landmarks    # Get world landmarks
    row = list(base)
    if not lms or not w_lms:
        return row
    
    # __________ Get shoulder center (z stereo values) __________
    sh_z_stereo = None 
    for name in ('left_shoulder', 'right_shoulder'):
        i = POSE_INDICES[name]
        lm = lms.landmark[i]
        z_raw = None

        # _____ Calculate stereo confidence _____
        u = lm.x * frame_w          # pixel coordinate (x)
        v = lm.y * frame_h
        confidence = _stereo_confidence(disp_L, disp_R, u, v)

        # _____ Using (or not) z_stereo _____
        if confidence >= Z_CONF_THRESHOLD:      # Se la rilevazione è abbastanza affidabile usa la z stereo
            z_raw = depth_map[int(round(v)), int(round(u))]
            if np.isnan(z_raw) or z_raw < 0.1 or z_raw > 4.0:
                z_raw = None
        z = _resolve_z(name, z_raw, last_z_pose, last_z_age, i_frame)

        # _____ Save shoulder_z_stereo _____
        if z is None:
            continue
        if sh_z_stereo is None:
            sh_z_stereo = z                            # Se una delle due non è stata rilevata usiamo solo l'altra
        else:
            sh_z_stereo = (sh_z_stereo + z) / 2.0     # Se entrambe sono state rilevate usiamo la media


    # __________ Get all landmarks of the frame __________
    STEREO_NAMES = {'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'}

    for name, i in POSE_INDICES.items():
        lm = lms.landmark[i]
        w_lm = w_lms.landmark[i]

        x   = w_lm.x
        y   = w_lm.y
        z   = w_lm.z    # default: pure MediaPipe for all landmarks
        vis = w_lm.visibility

        # _____ Stereo z for shoulder/elbow/wrist landmarks _____
        if name in STEREO_NAMES and sh_z_stereo is not None:
            u = int(round(lm.x * frame_w))
            v = int(round(lm.y * frame_h))
            confidence = _stereo_confidence(disp_L, disp_R, u, v)
            if confidence >= Z_CONF_THRESHOLD:
                z_raw = depth_map[v, u]
                if not np.isnan(z_raw) and 0.1 <= z_raw <= 4.0:
                    z_body = z_raw - sh_z_stereo
                    if abs(z_body) <= BODY_Z_MAX:
                        z = stereo_oef[name](z_body, timestamp)   # One Euro Filter

        # _____ Adding coords of the joint to the row _____
        row += [x, y, z, vis]
      
    return row


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    # __________ Inputs __________
    subject, exercise, video = ask_inputs()
    input_path = DATA_ROOT / "raw_data" / subject / exercise 
    input_path_L = input_path / f"{video}_L.mp4"
    input_path_R = input_path / f"{video}_R.mp4"
    output_path = DATA_ROOT / "landmarks" / subject / exercise / video
    if not input_path_L.exists() or not input_path_R.exists():
        print("Impossible to find video")
        return

    # __________ Opening videos and get fps __________
    cap_L = cv2.VideoCapture(input_path_L)
    cap_R = cv2.VideoCapture(input_path_R)
    if not cap_L.isOpened() or not cap_R.isOpened():
        print("Impossible to open the video")
        return    
    tot_frames = int(cap_L.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_L.get(cv2.CAP_PROP_FPS)
    
    # __________ Stereo rectification and SGBM matcher __________
    ret, frame = cap_L.read()
    if not ret:
        print("Error: cannot read first frame")
        return
    frame_h, frame_w = frame.shape[:2] 
    cap_L.set(cv2.CAP_PROP_POS_FRAMES, 0)
    img_size = (frame_w, frame_h)

    print("Building rectification map...")
    map_L1, map_L2, map_R1, map_R2, P_L, P_R, Q = _build_rectification_map(img_size)
    sgbm = _build_sgbm()
    right_matcher = cv2.ximgproc.createRightMatcher(sgbm)   # creato una volta sola
    wls = cv2.ximgproc.createDisparityWLSFilter(sgbm)       # creato una volta sola
    wls.setLambda(2000)
    wls.setSigmaColor(1.5)
    print("Ready. Press 'P' to pause/resume, 'Q' to quit\n")

    # __________ Initialization __________
    csv_path = init_csv_files(output_path)
    last_z_pose = {}
    last_z_age  = {}
    stereo_oef  = {name: _OneEuroFilter(min_cutoff=OEF_MIN_CUTOFF, beta=OEF_BETA)
                   for name in ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist')}
    i_frame = 0
    paused  = False

    with mp_pose.Pose(
        model_complexity = 2,
        smooth_landmarks = True,
        enable_segmentation = False,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.6
    ) as pose:

        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # serve ad aumentare il contrasto in modo locale
        display_bgr = None

        # __________ Main loop __________
        while True:
            if not paused:
                ret_L, raw_L = cap_L.read()
                ret_R, raw_R = cap_R.read()
                if not ret_L or not ret_R:
                    print("Video ended")
                    break
                
                # Trasformo l'immagine grezza in un'immagine rettificata (valutare INTER_CUBIC, più preciso ma lento, fonte ChatGPT)
                rect_L = cv2.remap(raw_L, map_L1, map_L2, cv2.INTER_CUBIC)
                rect_R = cv2.remap(raw_R, map_R1, map_R2, cv2.INTER_CUBIC)
        
                # Trasformo in scala di grigio e applico clahe
                gray_L = _clahe.apply(cv2.cvtColor(rect_L, cv2.COLOR_BGR2GRAY))
                gray_R = _clahe.apply(cv2.cvtColor(rect_R, cv2.COLOR_BGR2GRAY))

                # Creo la disparity map (raw, usata per LRC confidence)
                disp_L_raw = sgbm.compute(gray_L, gray_R)

                # Filtraggio WLS (disp_L filtrata, usata per la depth map)
                disp_R = right_matcher.compute(gray_R, gray_L)
                disp_L = wls.filter(disp_L_raw, gray_L, disparity_map_right=disp_R)

                # Creo depth map dalla disparity filtrata (più smooth)
                depth_map = _disparity_to_depth_map(disp_L, P_L)

                # Show disparity and depth maps
                disp_vis = cv2.normalize(disp_L, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
                # cv2.imshow("Disparity", disp_color)
                d_vis = depth_map.copy()
                d_vis = np.clip(d_vis, Z_MIN, Z_MAX)
                d_vis = ((d_vis - Z_MIN) / (Z_MAX - Z_MIN) * 255).astype(np.uint8)
                # cv2.imshow("Depth", d_vis)

                # Prendo i landmark da MediaPipe e li salvo in 'results'
                rgb_L = cv2.cvtColor(rect_L, cv2.COLOR_BGR2RGB)
                rgb_L.flags.writeable = False
                results = pose.process(rgb_L)
                rgb_L.flags.writeable = True

                # Estraggo le coordinate dei giunti e scrivo la riga nel file output (pose.csv)
                timestamp = i_frame / fps
                pose_row = _extract_pos_row(results, i_frame, timestamp, depth_map, P_L, frame_w, frame_h, last_z_pose, last_z_age, stereo_oef, disp_L_raw, disp_R)
                with open(csv_path['pose'], 'a', newline='') as f:
                    csv.writer(f).writerow(pose_row)

                # Disegno i landmarks sul frame
                display_bgr = cv2.cvtColor(rgb_L, cv2.COLOR_RGB2BGR)
                display_bgr = _draw_landmarks(display_bgr, results)
                i_frame += 1

            # Mostra la finestra con il video processato e i landmark disegnati
            if display_bgr is None:
                continue
            paused_label = "  [PAUSED]  " if paused else ""
            cv2.putText(display_bgr, f"Frame {i_frame}/{tot_frames}{paused_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_bgr, f"{subject} / {exercise} / {video}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            dh = int(frame_h * DISPLAY_WIDTH / frame_w)
            shown = cv2.resize(display_bgr, (DISPLAY_WIDTH, dh))
            # cv2.imshow("Reachy - Stereo Landmark Extraction", shown)

            # Gestisci comandi da tastiera (pause and quit)
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('p'):
                paused = not paused
                print("[PAUSED]" if paused else "[RESUMED]")
            elif key == ord('q'):
                break

    # __________ Clearing __________    
    cap_L.release()
    cap_R.release()
    cv2.destroyAllWindows()
    
    # __________ Print output info __________
    print(f"Done. Frame processed: {i_frame} / {tot_frames}")
    print(f"Output -> {output_path.relative_to(DATA_ROOT)}")
    
if __name__ == "__main__":
    main()