'''
stereo_config.py
=============================================================================
Stereo camera calibration parameters for Reachy's left and right cameras.

HOW TO FILL THIS FILE
---------------------
Run stereo_calibrate.py once with a checkerboard pattern (9×6, 25 mm squares
is a good default).  That script will print the matrices below and save them
to stereo_calib.npz.  Copy the printed values here so every other script can
import them without re-running calibration.

Coordinate convention (output 3D points):
  Origin : principal point of the left camera
  X      : right  (image x direction)
  Y      : down   (image y direction) — same as OpenCV / MediaPipe image space
  Z      : forward (depth, positive away from camera)

mapping.py already converts from MediaPipe camera frame to the Reachy torso
frame; this module only needs to describe the raw left-camera frame.

Usage:
    from stereo_config import LEFT_K, LEFT_D, RIGHT_K, RIGHT_D, R, T, IMAGE_SIZE
'''

import numpy as np

# ---------------------------------------------------------------------------
# Image resolution (width, height) — must match what the cameras actually stream
# ---------------------------------------------------------------------------
IMAGE_SIZE = (480, 640)     # ← adjust if Reachy streams at a different resolution

# ---------------------------------------------------------------------------
# Left camera intrinsics
#   K : 3×3 camera matrix  [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
#   D : distortion coefficients [k1, k2, p1, p2, k3]
# ---------------------------------------------------------------------------

LEFT_K  = np.array([[402.168823,   0.      , 224.341125],
 [  0.      , 402.231721, 308.735249],
 [  0.      ,   0.      ,   1.      ]], dtype=np.float64)

LEFT_D  = np.array([-3.593085e-01,  2.013986e-01,  1.416095e-03,  3.445533e-04, -6.501238e-02], dtype=np.float64)

# ---------------------------------------------------------------------------
# Right camera intrinsics
# ---------------------------------------------------------------------------
RIGHT_K = np.array([[397.034718,   0.      , 255.346365],
 [  0.      , 397.328189, 297.138257],
 [  0.      ,   0.      ,   1.      ]], dtype=np.float64)

RIGHT_D = np.array([-3.609302e-01,  1.469785e-01,  1.274684e-03,  1.297635e-04, -3.082630e-02], dtype=np.float64)

# ---------------------------------------------------------------------------
# Stereo extrinsics
#   R : 3×3 rotation matrix from left to right camera frame
#   T : (3,) translation vector (meters) from left to right camera origin
#       For a typical horizontal stereo rig: T ≈ [-baseline, 0, 0]
#       Reachy's stereo baseline is approximately 65 mm → T[0] ≈ -0.065
# ---------------------------------------------------------------------------
BASELINE_M = 0.07429      # meters 

R = np.array([[ 0.999975,  0.004879,  0.005028],
 [-0.004782,  0.999807, -0.019065],
 [-0.00512 ,  0.01904 ,  0.999806]], dtype=np.float64)
T = np.array([-0.074286,  0.00185, -0.001255], dtype=np.float64)

# ---------------------------------------------------------------------------
# StereoSGBM parameters (tune for your scene / lighting)
# ---------------------------------------------------------------------------
SGBM_MIN_DISP    =  0
SGBM_NUM_DISP    =  32    # must be divisible by 16 (più alto = più profondità ma più lento)
SGBM_BLOCK_SIZE  =  11    # large block = robust on low-texture clothing/skin
SGBM_P1          =   8 * 3 * SGBM_BLOCK_SIZE ** 2   # scales with blockSize²
SGBM_P2          =  32 * 3 * SGBM_BLOCK_SIZE ** 2
SGBM_DISP12DIFF  =  -1    # disabled — L-R check was rejecting too many valid pixels
SGBM_UNIQUENESS  =  25    # very lenient — maximise coverage on low-texture areas
SGBM_SPECKLE_WIN = 100
SGBM_SPECKLE_RNG =   2    # tolleranza per considerare pixel come “rumore”
