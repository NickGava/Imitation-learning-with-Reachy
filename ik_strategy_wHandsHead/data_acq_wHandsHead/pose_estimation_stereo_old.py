'''
pose_estimation_stereo.py
=============================================================================
Stereo-vision landmark extraction pipeline.

Input:
  data/raw_data/subject_XXX/exercise_XXX/video_XXX_L.mp4
  data/raw_data/subject_XXX/exercise_XXX/video_XXX_R.mp4

Output (same format as pose_estimation.py):
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/pose.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/right_hand.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/left_hand.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/face.csv

Controls (live preview):
  P - pause / resume
  Q - quit

Depth filtering pipeline (three layers):
  [L1] Pixel-level range gate : 0.3 m <= Z <= 4.0 m  (_pixel_to_3d)
  [L2] Temporal spike rejection: |new_Z - last_Z| <= MAX_Z_DELTA_M  (_resolve_z)
       + max-age expiry after MAX_PERSISTENCE_FRAMES frames without a fresh Z
  [L3] Body-space plausibility : |body_x|, |body_y| <= BODY_XY_MAX_M
                                  |body_z|          <= BODY_Z_MAX_M
       after subtracting the hip-midpoint origin.
       A point that fails L3 is written as None AND its cached Z is invalidated
       so a bad bootstrap value cannot propagate via persistence.
'''

import cv2
import numpy as np
import mediapipe as mp
import csv

from config import DATA_ROOT, POSE_INDICES, HAND_INDICES, FACE_INDICES
from ask_inputs import ask_inputs
from save_landmarks_old import init_csv_files
from stereo_config import (
    LEFT_K, LEFT_D, RIGHT_K, RIGHT_D, R as R_stereo, T as T_stereo,
    SGBM_MIN_DISP, SGBM_NUM_DISP, SGBM_BLOCK_SIZE,
    SGBM_P1, SGBM_P2, SGBM_DISP12DIFF, SGBM_UNIQUENESS,
    SGBM_SPECKLE_WIN, SGBM_SPECKLE_RNG,
)

DISPLAY_WIDTH = 500

# ---------------------------------------------------------------------------
# Depth filter parameters
# ---------------------------------------------------------------------------

# [L2] Max frame-to-frame Z change accepted as valid (metres).
# Wrist at ~1 m/s at 30 fps -> ~0.033 m/frame; 0.15 m is a generous bound.
MAX_Z_DELTA_M = 0.15

# [L2] After this many frames without a fresh accepted Z, cached value expires.
MAX_PERSISTENCE_FRAMES = 45

# [L3] Body-space plausibility bounds (metres, relative to hip midpoint).
# Human arm span ~1.8 m -> max lateral/vertical reach from hip centre ~0.9 m.
# Depth variation of body parts relative to hips rarely exceeds 0.8 m.
BODY_XY_MAX_M = 1.0
BODY_Z_MAX_M  = 1.0

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
mp_holistic       = mp.solutions.holistic
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ---------------------------------------------------------------------------
# Stereo rectification
# ---------------------------------------------------------------------------
def _build_rectification(image_size):
    """
    It aligns the two views horizontally
    """
    R1, R2, P_L, P_R, Q, _, _ = cv2.stereoRectify(
        LEFT_K, LEFT_D,
        RIGHT_K, RIGHT_D,
        image_size,
        R_stereo, T_stereo,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.55,
    )
    map_L1, map_L2 = cv2.initUndistortRectifyMap(LEFT_K, LEFT_D, R1, P_L, image_size, cv2.CV_32FC1)
    map_R1, map_R2 = cv2.initUndistortRectifyMap(RIGHT_K, RIGHT_D, R2, P_R, image_size, cv2.CV_32FC1)
    return map_L1, map_L2, map_R1, map_R2, P_L, P_R, Q


# ---------------------------------------------------------------------------
# SGBM matcher - single matcher (WLS removed to avoid frame drops)
# ---------------------------------------------------------------------------
def _build_sgbm():
    '''
    Creates and returns a configured StereoSGBM matcher.
    SBMG (Semi-Global Block Matching) is an algorithm for computing disparity maps: for every pixel of the left image, it finds the corresponding pixel 
    in the right image and computes the disparity (horizontal shift) between them
    '''
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


# ---------------------------------------------------------------------------
# Depth helpers
# ---------------------------------------------------------------------------
def _disparity_to_depth_map(disp_raw, P_L):
    """int16 SGBM disparity (x16) -> float32 depth map in metres. NaN where invalid."""
    disp = disp_raw.astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    fx       = P_L[0, 0]
    baseline = abs(float(T_stereo[0]))
    return fx * baseline / disp


def _pixel_to_3d(u, v, depth_map, P_L):
    """
    [L1] Back-projects pixel (u, v) using P_L.
    Returns (X, Y, Z) in metres, or None if Z is NaN / outside [0.3, 4.0] m.
    No fallback - temporal persistence is handled by the caller.
    """
    h, w = depth_map.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    Z = float(depth_map[vi, ui])
    if np.isnan(Z) or Z < 0.3 or Z > 4.0:
        return None
    fx, fy = P_L[0, 0], P_L[1, 1]
    cx, cy = P_L[0, 2], P_L[1, 2]
    return (ui - cx) * Z / fx, (vi - cy) * Z / fy, Z


def _backproject_xy(u, v, Z, P_L):
    """Given pixel (u, v) and depth Z, returns (X, Y, Z) via P_L intrinsics."""
    fx, fy = P_L[0, 0], P_L[1, 1]
    cx, cy = P_L[0, 2], P_L[1, 2]
    return (u - cx) * Z / fx, (v - cy) * Z / fy, Z


# ---------------------------------------------------------------------------
# [L2] Temporal persistence with spike rejection and max-age expiry
# ---------------------------------------------------------------------------
def _resolve_z(name, new_z, last_z, last_z_age, frame_idx):
    """
    Returns the Z to use for this frame, or None.

    Accept new_z only if:
      - it is not None (stereo gave a valid depth), AND
      - |new_z - last_z| <= MAX_Z_DELTA_M  (no spike relative to cached value)
        (first measurement for this landmark is always accepted)

    If new_z is rejected or absent, fall back to the cached Z provided it is
    no older than MAX_PERSISTENCE_FRAMES frames; otherwise expire the cache.

    Mutates last_z and last_z_age in-place.
    """
    if new_z is not None:
        prev = last_z.get(name)
        if prev is None or abs(new_z - prev) <= MAX_Z_DELTA_M:
            last_z[name]     = new_z
            last_z_age[name] = frame_idx
            return new_z
        # spike -> fall through to persistence without updating cache

    if name in last_z:
        age = frame_idx - last_z_age.get(name, frame_idx)
        if age <= MAX_PERSISTENCE_FRAMES:
            return last_z[name]
        del last_z[name]
        last_z_age.pop(name, None)

    return None


def _invalidate_z(name, last_z, last_z_age):
    """Clears the cached Z for a landmark (called when L3 fails)."""
    last_z.pop(name, None)
    last_z_age.pop(name, None)


# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------
def _estimate_z_scale(pts_stereo, pw_lms):
    """
    Stima Z_stereo = a * Z_mp + b dai landmark con stereo valido.
    pts_stereo: dict {name: (X, Y, Z_assoluto)} — solo punti validi
    pw_lms:     results.pose_world_landmarks
    Restituisce (a, b) o None se meno di 2 punti disponibili.
    """
    pairs = []
    for name, idx in POSE_INDICES.items():
        if pts_stereo.get(name) is None:
            continue
        z_stereo = pts_stereo[name][2]          # Z assoluta in camera space
        z_mp     = pw_lms.landmark[idx].z       # Z MediaPipe world (hip-centrata, metri)
        pairs.append((z_mp, z_stereo))

    if len(pairs) < 2:
        return None

    z_mp = np.array([p[0] for p in pairs])
    z_st = np.array([p[1] for p in pairs])
    A = np.column_stack([z_mp, np.ones(len(pairs))])
    try:
        (a, b), residuals, _, _ = np.linalg.lstsq(A, z_st, rcond=None)
        return float(a), float(b)
    except np.linalg.LinAlgError:
        return None


def _extract_pose_row(results, frame_idx, timestamp, depth_map, P_L,
                      frame_w, frame_h, last_z, last_z_age):
    base = [frame_idx, timestamp]
    if not results.pose_landmarks:
        return base

    lms    = results.pose_landmarks.landmark
    pw_lms = results.pose_world_landmarks      # ← MediaPipe world (per fallback)

    # --- L1 + L2: stereo depth per ogni landmark ---
    pts = {}
    for name, idx in POSE_INDICES.items():
        lm    = lms[idx]
        u     = lm.x * frame_w
        v     = lm.y * frame_h
        p3d   = _pixel_to_3d(u, v, depth_map, P_L)
        raw_z = p3d[2] if p3d is not None else None
        z     = _resolve_z(name, raw_z, last_z, last_z_age, frame_idx)
        if z is not None:
            X, Y, _ = _backproject_xy(u, v, z, P_L)
            pts[name] = (X, Y, z)
        else:
            pts[name] = None

    # --- Fallback MediaPipe Z dove stereo fallisce ---
    if pw_lms is not None:
        scale_params = _estimate_z_scale(pts, pw_lms)
        if scale_params is not None:
            a, b = scale_params
            for name, idx in POSE_INDICES.items():
                if pts[name] is not None:
                    continue                            # stereo già valido
                lm      = lms[idx]
                u       = lm.x * frame_w
                v       = lm.y * frame_h
                z_mp    = pw_lms.landmark[idx].z
                z_fallback = a * z_mp + b
                # Passa comunque per L2 (spike rejection + persistence)
                z = _resolve_z(name, z_fallback, last_z, last_z_age, frame_idx)
                if z is not None:
                    X, Y, _ = _backproject_xy(u, v, z, P_L)
                    pts[name] = (X, Y, z)

    # --- Origine: midpoint anche ---
    l_hip = pts.get('left_hip')
    r_hip = pts.get('right_hip')
    if l_hip is not None and r_hip is not None:
        origin = np.array([(l_hip[0] + r_hip[0]) / 2,
                           (l_hip[1] + r_hip[1]) / 2,
                           (l_hip[2] + r_hip[2]) / 2])
    elif l_hip is not None:
        origin = np.array(l_hip)
    elif r_hip is not None:
        origin = np.array(r_hip)
    else:
        return base

    # --- L3 + costruzione riga ---
    row = list(base)
    for name, idx in POSE_INDICES.items():
        p = pts[name]
        if p is None:
            row += [None, None, None, None]
            continue
        body_xyz = np.array(p) - origin
        if (abs(body_xyz[0]) > BODY_XY_MAX_M or
                abs(body_xyz[1]) > BODY_XY_MAX_M or
                abs(body_xyz[2]) > BODY_Z_MAX_M):
            _invalidate_z(name, last_z, last_z_age)
            row += [None, None, None, None]
            continue
        vis = float(lms[POSE_INDICES[name]].visibility)
        row += [body_xyz[0], body_xyz[1], body_xyz[2], vis]

    return row
'''
def _extract_pose_row(results, frame_idx, timestamp, depth_map, P_L, frame_w, frame_h, last_z, last_z_age):
    """
    Builds a pose CSV row: MediaPipe 2D + stereo depth, body-centred on hips.

    Three-layer depth filter:
      L1: pixel range gate  (in _pixel_to_3d)
      L2: spike rejection + max-age  (in _resolve_z)
      L3: body-space plausibility after subtracting hip-midpoint origin;
          a failing landmark is written as None and its cache is invalidated
          so a bad bootstrap value cannot persist into subsequent frames.
    """
    base = [frame_idx, timestamp]
    if not results.pose_landmarks:
        return base

    lms = results.pose_landmarks.landmark

    # --- Resolve 3D position for every joint (L1 + L2) ---------------------
    pts = {}
    for name, idx in POSE_INDICES.items():
        lm    = lms[idx]
        u     = lm.x * frame_w
        v     = lm.y * frame_h
        p3d   = _pixel_to_3d(u, v, depth_map, P_L)   # L1
        raw_z = p3d[2] if p3d is not None else None
        z     = _resolve_z(name, raw_z, last_z, last_z_age, frame_idx)  # L2

        if z is not None:
            X, Y, _ = _backproject_xy(u, v, z, P_L)
            pts[name] = (X, Y, z)
        else:
            pts[name] = None

    # --- Body-centred origin: midpoint of hips ------------------------------
    l_hip = pts.get('left_hip')
    r_hip = pts.get('right_hip')
    if l_hip is not None and r_hip is not None:
        origin = np.array([(l_hip[0] + r_hip[0]) / 2,
                           (l_hip[1] + r_hip[1]) / 2,
                           (l_hip[2] + r_hip[2]) / 2])
    elif l_hip is not None:
        origin = np.array(l_hip)
    elif r_hip is not None:
        origin = np.array(r_hip)
    else:
        return base

    # --- Build row with L3 body-space check ---------------------------------
    row = list(base)
    for name, idx in POSE_INDICES.items():
        p = pts[name]
        if p is None:
            row += [None, None, None, None]
            continue

        body_xyz = np.array(p) - origin

        # [L3] Physical plausibility in body space
        if (abs(body_xyz[0]) > BODY_XY_MAX_M or
                abs(body_xyz[1]) > BODY_XY_MAX_M or
                abs(body_xyz[2]) > BODY_Z_MAX_M):
            # Invalidate cached Z so this bad bootstrap doesn't persist
            _invalidate_z(name, last_z, last_z_age)
            row += [None, None, None, None]
            continue

        vis = float(lms[POSE_INDICES[name]].visibility)
        row += [body_xyz[0], body_xyz[1], body_xyz[2], vis]

    return row
'''

def _extract_hand_row(hand_landmarks, frame_idx, timestamp,
                      depth_map, P_L, frame_w, frame_h,
                      last_z, last_z_age):
    """
    Builds a hand CSV row with the same L1+L2 depth filtering.
    No body-space check for hands (no fixed origin to centre against).
    """
    base = [frame_idx, timestamp]
    if hand_landmarks is None:
        return base

    lms = hand_landmarks.landmark
    row = list(base)
    for name, idx in HAND_INDICES.items():
        lm    = lms[idx]
        u     = lm.x * frame_w
        v     = lm.y * frame_h
        p3d   = _pixel_to_3d(u, v, depth_map, P_L)
        raw_z = p3d[2] if p3d is not None else None
        z     = _resolve_z(name, raw_z, last_z, last_z_age, frame_idx)

        if z is not None:
            X, Y, _ = _backproject_xy(u, v, z, P_L)
            row += [X, -Y, z]
        else:
            row += [None, None, None]
    return row


def _extract_face_row(results, frame_idx, timestamp):
    """Face: keep normalised 2D coords (face_processing.py uses 2D ratios)."""
    base = [frame_idx, timestamp]
    if not results.face_landmarks:
        return base
    lms = results.face_landmarks.landmark
    row = list(base)
    for _, idx in FACE_INDICES.items():
        lm = lms[idx]
        row += [lm.x, lm.y, lm.z]
    return row


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------
def _draw_landmarks(image, results):
    if results.pose_landmarks:
        _POSE_FACE = set(range(11))
        _hidden    = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=0, circle_radius=0)
        default    = mp_drawing_styles.get_default_pose_landmarks_style()
        spec  = {i: (_hidden if i in _POSE_FACE else default.get(i, mp_drawing.DrawingSpec()))
                 for i in range(33)}
        conns = [c for c in mp_holistic.POSE_CONNECTIONS
                 if c[0] not in _POSE_FACE and c[1] not in _POSE_FACE]
        mp_drawing.draw_landmarks(image, results.pose_landmarks, conns,
                                  landmark_drawing_spec=spec)
    for lm_set, conns in [
        (results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS),
        (results.left_hand_landmarks,  mp_holistic.HAND_CONNECTIONS),
    ]:
        if lm_set:
            mp_drawing.draw_landmarks(
                image, lm_set, conns,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    subject_name, exercise_name, video_name = ask_inputs()
    raw_dir = DATA_ROOT / "raw_data" / subject_name / exercise_name
    path_L  = raw_dir / f"{video_name}_L.mp4"
    path_R  = raw_dir / f"{video_name}_R.mp4"
    for p in (path_L, path_R):
        if not p.exists():
            print(f"Error: video not found -> {p}")
            return

    cap_L = cv2.VideoCapture(str(path_L))
    cap_R = cv2.VideoCapture(str(path_R))
    if not cap_L.isOpened() or not cap_R.isOpened():
        print("Error: cannot open video files.")
        return

    total_frames = int(cap_L.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap_L.get(cv2.CAP_PROP_FPS) or 30.0
    ret, probe   = cap_L.read()
    if not ret:
        print("Error: cannot read first frame.")
        return
    frame_h, frame_w = probe.shape[:2]
    cap_L.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"Left  : {path_L}  ({total_frames} frames @ {fps:.1f} fps, {frame_w}x{frame_h})")
    print(f"Right : {path_R}")

    img_size = (frame_w, frame_h)
    print("Building rectification maps…")
    map_L1, map_L2, map_R1, map_R2, P_L, P_R, Q = _build_rectification(img_size)

    sgbm = _build_sgbm()
    print("Ready.\nPress 'P' to pause/resume, 'Q' to quit.\n")

    landmarks_folder = DATA_ROOT / "landmarks" / subject_name / exercise_name / video_name
    csv_paths = init_csv_files(landmarks_folder)

    # Temporal persistence buffers: {landmark_name -> Z} and {name -> frame_idx}
    last_z_pose      = {}; last_z_age_pose  = {}
    last_z_rhand     = {}; last_z_age_rhand = {}
    last_z_lhand     = {}; last_z_age_lhand = {}

    frame_idx = 0
    paused    = False

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2,
    ) as holistic:


        _clahe = cv2.createCLAHE(clipLimit=0.4, tileGridSize=(8, 8))
        while True:
            if not paused:
                ret_L, raw_L = cap_L.read()
                ret_R, raw_R = cap_R.read()
                if not ret_L or not ret_R:
                    print("Video ended.")
                    break

                rect_L = cv2.remap(raw_L, map_L1, map_L2, cv2.INTER_LINEAR)
                rect_R = cv2.remap(raw_R, map_R1, map_R2, cv2.INTER_LINEAR)

                # gray_L    = cv2.cvtColor(rect_L, cv2.COLOR_BGR2GRAY)
                # gray_R    = cv2.cvtColor(rect_R, cv2.COLOR_BGR2GRAY)

                gray_L = _clahe.apply(cv2.cvtColor(rect_L, cv2.COLOR_BGR2GRAY))
                gray_R = _clahe.apply(cv2.cvtColor(rect_R, cv2.COLOR_BGR2GRAY))
                disp_raw  = sgbm.compute(gray_L, gray_R)
                # Debug
                disp_vis = cv2.normalize(disp_raw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
                cv2.imshow("Disparity", disp_color)
                # End debug
                depth_map = _disparity_to_depth_map(disp_raw, P_L)

                rgb_L = cv2.cvtColor(rect_L, cv2.COLOR_BGR2RGB)
                rgb_L.flags.writeable = False
                results = holistic.process(rgb_L)
                rgb_L.flags.writeable = True

                ts = frame_idx / fps

                pose_row = _extract_pose_row(results, frame_idx, ts, depth_map, P_L, frame_w, frame_h, last_z_pose, last_z_age_pose)
                rhand_row = _extract_hand_row(results.right_hand_landmarks, frame_idx, ts, depth_map, P_L, frame_w, frame_h, last_z_rhand, last_z_age_rhand)
                lhand_row = _extract_hand_row(results.left_hand_landmarks, frame_idx, ts, depth_map, P_L, frame_w, frame_h, last_z_lhand, last_z_age_lhand)
                face_row = _extract_face_row(results, frame_idx, ts)

                for path_key, row in [('pose', pose_row), ('right_hand', rhand_row),
                                      ('left_hand', lhand_row), ('face', face_row)]:
                    with open(csv_paths[path_key], 'a', newline='') as f:
                        csv.writer(f).writerow(row)

                display_bgr = cv2.cvtColor(rgb_L, cv2.COLOR_RGB2BGR)
                display_bgr = _draw_landmarks(display_bgr, results)
                frame_idx  += 1

            pause_label = "  [PAUSED]" if paused else ""
            cv2.putText(display_bgr, f"Frame {frame_idx}/{total_frames}{pause_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_bgr, f"{subject_name} / {exercise_name} / {video_name}  (STEREO)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            dh    = int(frame_h * DISPLAY_WIDTH / frame_w)
            shown = cv2.resize(display_bgr, (DISPLAY_WIDTH, dh))
            cv2.imshow("Reachy - Stereo Landmark Extraction", shown)

            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('p'):
                paused = not paused
                print("[PAUSED]" if paused else "[RESUMED]")
            elif key == ord('q'):
                break

    cap_L.release()
    cap_R.release()
    cv2.destroyAllWindows()
    print(f"Done. Frames processed: {frame_idx} / {total_frames}")
    print(f"Output -> {landmarks_folder}")


if __name__ == "__main__":
    main()