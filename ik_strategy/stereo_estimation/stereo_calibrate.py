'''
stereo_calibrate.py
=============================================================================
One-time stereo calibration using a printed checkerboard pattern.

Recommended checkerboard : 9×6 inner corners, 25 mm square size.
Print it, glue it to a rigid flat board, and measure the actual square size
with a ruler — enter it as SQUARE_SIZE_M below.

After calibration the script:
  1. Prints the matrices you need to copy into stereo_config.py.
  2. Saves everything to stereo_calib.npz for reference.

How to capture calibration frames
----------------------------------
Run this script and hold the checkerboard in front of Reachy's cameras.
Move it to cover as many positions and orientations as possible
(tilted, corners, centre, close, far).  The script captures a frame pair
every time it detects the pattern in BOTH images — aim for 20–40 pairs.
Press Q when done.

Usage:
  python stereo_calibrate.py
'''

import time
import cv2
import numpy as np
from pathlib import Path
from reachy_sdk import ReachySDK

# ---------------------------------------------------------------------------
# Settings — adjust before running
# ---------------------------------------------------------------------------
ROBOT_IP       = "10.59.1.20"
CHECKERBOARD   = (9, 6)          # inner corners (cols, rows)
SQUARE_SIZE_M  = 0.022           # metres — measure your printed board!
MIN_PAIRS      = 15              # minimum valid pairs before calibration runs
SAVE_PATH      = Path("stereo_calib.npz")
DISPLAY_WIDTH  = 500
# ---------------------------------------------------------------------------

def main():
    print(f"Connecting to Reachy at {ROBOT_IP}…")
    reachy = ReachySDK(host=ROBOT_IP)
    print("Connected.\n")
    print(f"Checkerboard : {CHECKERBOARD[0]}×{CHECKERBOARD[1]} inner corners, "
          f"{SQUARE_SIZE_M*1000:.0f} mm squares")
    print("Move the board in front of both cameras.")
    print("A frame pair is captured automatically when both cameras see the pattern.")
    print("Press Q when you have enough pairs (target: 20–40).\n")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cols, rows = CHECKERBOARD

    # 3D object points for one board view
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQUARE_SIZE_M

    obj_pts, img_pts_L, img_pts_R = [], [], []
    img_size = None

    last_capture = 0.0
    CAPTURE_COOLDOWN = 1.0      # seconds between captures to avoid duplicates

    while True:
        left  = reachy.left_camera.last_frame
        right = reachy.right_camera.last_frame

        if left is None or right is None:
            time.sleep(0.03)
            continue

        if img_size is None:
            h, w  = left.shape[:2]
            img_size = (w, h)

        gray_L = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
        gray_R = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        ret_L, corners_L = cv2.findChessboardCorners(gray_L, CHECKERBOARD, None)
        ret_R, corners_R = cv2.findChessboardCorners(gray_R, CHECKERBOARD, None)

        preview_L = left.copy()
        preview_R = right.copy()

        now = time.time()
        if ret_L and ret_R and (now - last_capture) > CAPTURE_COOLDOWN:
            # Sub-pixel refinement
            corners_L = cv2.cornerSubPix(gray_L, corners_L, (11,11), (-1,-1), criteria)
            corners_R = cv2.cornerSubPix(gray_R, corners_R, (11,11), (-1,-1), criteria)
            obj_pts.append(objp)
            img_pts_L.append(corners_L)
            img_pts_R.append(corners_R)
            last_capture = now
            print(f"  Captured pair {len(obj_pts):3d}")

        cv2.drawChessboardCorners(preview_L, CHECKERBOARD, corners_L, ret_L)
        cv2.drawChessboardCorners(preview_R, CHECKERBOARD, corners_R, ret_R)

        n = len(obj_pts)
        color = (0, 200, 0) if n >= MIN_PAIRS else (0, 165, 255)
        for img in (preview_L, preview_R):
            cv2.putText(img, f"Pairs: {n}  (need {MIN_PAIRS})  Q=done",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        side_by_side = np.hstack([preview_L, preview_R])
        dh = int(left.shape[0] * DISPLAY_WIDTH / left.shape[1])
        shown = cv2.resize(side_by_side, (DISPLAY_WIDTH * 2, dh))
        cv2.imshow("Stereo Calibration — Left | Right", shown)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(obj_pts) < MIN_PAIRS:
        print(f"\nNot enough pairs ({len(obj_pts)} < {MIN_PAIRS}). Capture more and retry.")
        return

    print(f"\nCalibrating with {len(obj_pts)} pairs…")

    # Individual camera calibration (needed as initial estimate)
    _, K_L, D_L, _, _ = cv2.calibrateCamera(obj_pts, img_pts_L, img_size, None, None)
    _, K_R, D_R, _, _ = cv2.calibrateCamera(obj_pts, img_pts_R, img_size, None, None)

    # Stereo calibration
    flags = (cv2.CALIB_FIX_INTRINSIC)      # keep individual calibrations fixed
    rms, K_L, D_L, K_R, D_R, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_pts_L, img_pts_R,
        K_L, D_L, K_R, D_R,
        img_size,
        flags=flags,
        criteria=criteria,
    )
    print(f"Stereo calibration RMS reprojection error: {rms:.4f} px")
    if rms > 1.0:
        print("  ⚠  RMS > 1.0 — consider recapturing with better coverage.")

    # Save
    np.savez(str(SAVE_PATH),
             K_L=K_L, D_L=D_L, K_R=K_R, D_R=D_R,
             R=R, T=T, E=E, F=F,
             image_size=np.array(img_size))
    print(f"Saved to {SAVE_PATH}\n")

    # Print values to copy into stereo_config.py
    print("=" * 60)
    print("Copy these values into stereo_config.py")
    print("=" * 60)
    _fmt = lambda m: np.array2string(m, separator=', ', precision=6)
    print(f"\nIMAGE_SIZE = {img_size}")
    print(f"\nLEFT_K  = np.array({_fmt(K_L)}, dtype=np.float64)")
    print(f"\nLEFT_D  = np.array({_fmt(D_L.ravel())}, dtype=np.float64)")
    print(f"\nRIGHT_K = np.array({_fmt(K_R)}, dtype=np.float64)")
    print(f"\nRIGHT_D = np.array({_fmt(D_R.ravel())}, dtype=np.float64)")
    print(f"\nR = np.array({_fmt(R)}, dtype=np.float64)")
    print(f"\nT = np.array({_fmt(T.ravel())}, dtype=np.float64)")
    print(f"\n# Baseline: {abs(T[0,0])*1000:.1f} mm  →  BASELINE_M = {abs(T[0,0]):.5f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
