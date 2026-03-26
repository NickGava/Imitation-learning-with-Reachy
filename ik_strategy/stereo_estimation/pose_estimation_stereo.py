'''
pose_estimation_stereo.py
=============================================================================
Stereo-vision landmark extraction pipeline.
Replaces MediaPipe's unreliable Z-estimate with stereo-triangulated depth.

Input:
  data/raw_data/subject_XXX/exercise_XXX/video_XXX_L.mp4
  data/raw_data/subject_XXX/exercise_XXX/video_XXX_R.mp4

Output (identical format to pose_estimation.py):
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/pose.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/right_hand.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/left_hand.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/face.csv

Controls (live preview):
  P - pause / resume
  Q - quit
'''

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

from config import DATA_ROOT
from save_landmarks import init_csv_files, POSE_INDICES, HAND_INDICES, FACE_INDICES
from stereo_config import (
    LEFT_K, LEFT_D, RIGHT_K, RIGHT_D, R as R_stereo, T as T_stereo, IMAGE_SIZE,
    SGBM_MIN_DISP, SGBM_NUM_DISP, SGBM_BLOCK_SIZE,
    SGBM_P1, SGBM_P2, SGBM_DISP12DIFF, SGBM_UNIQUENESS,
    SGBM_SPECKLE_WIN, SGBM_SPECKLE_RNG,
)
from stereo_config import T_stereo as _T
import csv
import time

DISPLAY_WIDTH = 500

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
mp_holistic       = mp.solutions.holistic
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ---------------------------------------------------------------------------
# Stereo rectification maps
# ---------------------------------------------------------------------------
def _build_rectification(image_size):
    """
    Computes the rectification maps for the stereo pair.

    Returns:
        map_L1, map_L2  : remap arrays for the left  image
        map_R1, map_R2  : remap arrays for the right image
        P_L, P_R        : rectified projection matrices
        Q               : disparity-to-depth reprojection matrix (4x4)
    """
    R1, R2, P_L, P_R, Q, _, _ = cv2.stereoRectify(
        LEFT_K,  LEFT_D,
        RIGHT_K, RIGHT_D,
        image_size,
        R_stereo, T_stereo,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,           # crop away black borders
    )
    map_L1, map_L2 = cv2.initUndistortRectifyMap(
        LEFT_K,  LEFT_D,  R1, P_L, image_size, cv2.CV_32FC1)
    map_R1, map_R2 = cv2.initUndistortRectifyMap(
        RIGHT_K, RIGHT_D, R2, P_R, image_size, cv2.CV_32FC1)

    return map_L1, map_L2, map_R1, map_R2, P_L, P_R, Q


def _build_sgbm():
    """Creates the StereoSGBM matcher with parameters from stereo_config."""
    return cv2.StereoSGBM_create(
        minDisparity    = SGBM_MIN_DISP,
        numDisparities  = SGBM_NUM_DISP,
        blockSize       = SGBM_BLOCK_SIZE,
        P1              = SGBM_P1,
        P2              = SGBM_P2,
        disp12MaxDiff   = SGBM_DISP12DIFF,
        uniquenessRatio = SGBM_UNIQUENESS,
        speckleWindowSize = SGBM_SPECKLE_WIN,
        speckleRange    = SGBM_SPECKLE_RNG,
        mode            = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


# ---------------------------------------------------------------------------
# Depth helpers
# ---------------------------------------------------------------------------
def _disparity_to_depth_map(disp_raw, P_L):
    """
    Converts the raw SGBM disparity (integer, scaled x16) to a float32 depth map in meters.

        Z = fx * B / d
    where fx is the focal length of the rectified left camera and B is the
    baseline (derived from P_R[0,3] = -fx * B ;     B = -P_R[0,3] / fx).

    Pixels with invalid disparity (d ≤ 0) are set to NaN.
    """
    # disp_raw is int16, values are disparity * 16
    disp = disp_raw.astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan

    fx = P_L[0, 0]
    # P_R[0, 3] = -fx * Tx  where Tx = baseline (positive). Baseline = abs(T_stereo[0])
    baseline = abs(float(_T[0]))

    depth = fx * baseline / disp    # metres
    return depth                    # float32, NaN where invalid


def _pixel_to_3d(u, v, depth_map, P_L):
    """
    Back-projects a pixel (u, v) to 3D using the rectified left projection
    matrix P_L = [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]].

    Returns (X, Y, Z) in meters in the left-camera frame, or None if the
    depth at that pixel is invalid.
    """
    h, w = depth_map.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None

    Z = float(depth_map[vi, ui])
    if np.isnan(Z) or Z <= 0 or Z > 10.0:      # sanity: 10 m max range
        return None

    fx, fy = P_L[0, 0], P_L[1, 1]
    cx, cy = P_L[0, 2], P_L[1, 2]

    X = (ui - cx) * Z / fx
    Y = (vi - cy) * Z / fy
    return X, Y, Z


# ---------------------------------------------------------------------------
# Landmark extraction with stereo depth
# ---------------------------------------------------------------------------
def _extract_pose_row(results, frame_idx, timestamp, depth_map, P_L, frame_w, frame_h):
    """
    Builds a pose CSV row using MediaPipe's 2D pixel positions + stereo depth.

    Returns a list ready to write to pose.csv, or [frame_idx, timestamp]
    if landmarks are missing / depth is invalid for too many joints.
    """
    base = [frame_idx, timestamp]

    if not results.pose_landmarks:
        return base

    lms = results.pose_landmarks.landmark

    # Collect raw 3D positions for all 9 joints
    pts = {}       # name -> (X, Y, Z) in camera frame  or  None
    for name, idx in POSE_INDICES.items():
        lm = lms[idx]
        u  = lm.x * frame_w
        v  = lm.y * frame_h
        pts[name] = _pixel_to_3d(u, v, depth_map, P_L)

    # Compute body-centred origin (midpoint hips)
    l_hip = pts.get('left_hip')
    r_hip = pts.get('right_hip')
    if l_hip is None or r_hip is None:
        return base             # can't centre without hips

    origin = np.array([
        (l_hip[0] + r_hip[0]) / 2,
        (l_hip[1] + r_hip[1]) / 2,
        (l_hip[2] + r_hip[2]) / 2,
    ])

    # Build row
    row = list(base)
    for name, idx in POSE_INDICES.items():
        p = pts[name]
        if p is None:
            row += [None, None, None, None]     # NaN -> dropped in data_cleaning
        else:
            cam_xyz = np.array(p) - origin
            # Invert Y: camera Y is down, world Y is up
            x =  cam_xyz[0]
            y = -cam_xyz[1]
            z =  cam_xyz[2]
            vis = float(lms[POSE_INDICES[name]].visibility)
            row += [x, y, z, vis]

    return row


def _extract_hand_row(hand_landmarks, frame_idx, timestamp, depth_map, P_L, frame_w, frame_h):
    """
    Builds a hand CSV row for one hand.
    Positions are in meters in the left-camera frame (no body-centring,
    consistent with what hand_processing.py expects - it only uses relative
    distances between landmarks, not absolute positions).
    Y is inverted to match the world convention.
    """
    base = [frame_idx, timestamp]
    if hand_landmarks is None:
        return base

    lms = hand_landmarks.landmark
    row = list(base)
    for name, idx in HAND_INDICES.items():
        lm = lms[idx]
        u  = lm.x * frame_w
        v  = lm.y * frame_h
        p  = _pixel_to_3d(u, v, depth_map, P_L)
        if p is None:
            row += [None, None, None]
        else:
            row += [p[0], -p[1], p[2]]
    return row


def _extract_face_row(results, frame_idx, timestamp):
    """Face landmarks: no depth needed - face_processing.py only uses 2D
    ratios to build the head orientation, so we keep normalized image coords
    exactly as save_landmarks.py does."""
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
# Drawing helper (same as pose_estimation.py)
# ---------------------------------------------------------------------------
def _draw_landmarks(image, results):
    if results.pose_landmarks:
        _POSE_FACE = set(range(11))
        _hidden    = mp_drawing.DrawingSpec(color=(0,0,0), thickness=0, circle_radius=0)
        default    = mp_drawing_styles.get_default_pose_landmarks_style()
        spec = {i: (_hidden if i in _POSE_FACE else default.get(i, mp_drawing.DrawingSpec()))
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

    # Stereo rectification
    img_size = (frame_w, frame_h)
    print("Building rectification maps…")
    map_L1, map_L2, map_R1, map_R2, P_L, P_R, Q = _build_rectification(img_size)
    sgbm = _build_sgbm()
    print("Ready.\nPress 'P' to pause/resume, 'Q' to quit.\n")

    # Output CSV files
    landmarks_folder = DATA_ROOT / "landmarks" / subject_name / exercise_name / video_name
    csv_paths = init_csv_files(landmarks_folder)

    frame_idx = 0
    paused    = False

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:

        while True:
            if not paused:
                ret_L, raw_L = cap_L.read()
                ret_R, raw_R = cap_R.read()
                if not ret_L or not ret_R:
                    print("Video ended.")
                    break

                # Rectify
                rect_L = cv2.remap(raw_L, map_L1, map_L2, cv2.INTER_LINEAR)
                rect_R = cv2.remap(raw_R, map_R1, map_R2, cv2.INTER_LINEAR)

                # Disparity -> depth
                gray_L = cv2.cvtColor(rect_L, cv2.COLOR_BGR2GRAY)
                gray_R = cv2.cvtColor(rect_R, cv2.COLOR_BGR2GRAY)
                disp_raw  = sgbm.compute(gray_L, gray_R)
                depth_map = _disparity_to_depth_map(disp_raw, P_L)

                # MediaPipe on rectified left frame
                rgb_L = cv2.cvtColor(rect_L, cv2.COLOR_BGR2RGB)
                rgb_L.flags.writeable = False
                results = holistic.process(rgb_L)
                rgb_L.flags.writeable = True

                ts = time.time()

                # Extract landmarks with stereo depth
                pose_row  = _extract_pose_row(
                    results, frame_idx, ts, depth_map, P_L, frame_w, frame_h)
                rhand_row = _extract_hand_row(
                    results.right_hand_landmarks, frame_idx, ts,
                    depth_map, P_L, frame_w, frame_h)
                lhand_row = _extract_hand_row(
                    results.left_hand_landmarks, frame_idx, ts,
                    depth_map, P_L, frame_w, frame_h)
                face_row  = _extract_face_row(results, frame_idx, ts)

                # Write to CSV
                for path_key, row in [
                    ('pose',       pose_row),
                    ('right_hand', rhand_row),
                    ('left_hand',  lhand_row),
                    ('face',       face_row),
                ]:
                    with open(csv_paths[path_key], 'a', newline='') as f:
                        csv.writer(f).writerow(row)

                # Draw landmarks for preview
                display_bgr = cv2.cvtColor(rgb_L, cv2.COLOR_RGB2BGR)
                display_bgr = _draw_landmarks(display_bgr, results)
                frame_idx  += 1

            # HUD
            pause_label = "  [PAUSED]" if paused else ""
            cv2.putText(display_bgr,
                        f"Frame {frame_idx}/{total_frames}{pause_label}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_bgr,
                        f"{subject_name} / {exercise_name} / {video_name}  (STEREO)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            dh   = int(frame_h * DISPLAY_WIDTH / frame_w)
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
