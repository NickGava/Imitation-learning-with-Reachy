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

LEFT_K  = np.array([[395.485253,   0.      , 221.397047],
 [  0.      , 396.711611, 319.389497],
 [  0.      ,   0.      ,   1.      ]], dtype=np.float64)

LEFT_D  = np.array([-0.361544,  0.206271, -0.001457,  0.002352, -0.071953,  0.      ,
  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,
  0.      ,  0.      ], dtype=np.float64)

RIGHT_K = np.array([[394.772601,   0.      , 264.669681],
 [  0.      , 394.859592, 302.766583],
 [  0.      ,   0.      ,   1.      ]], dtype=np.float64)

RIGHT_D = np.array([-3.711103e-01,  1.556408e-01, -2.514907e-04, -2.325755e-03, -3.301015e-02,
  0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
  0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00], dtype=np.float64)

R = np.array([[ 0.999541,  0.006904, -0.029492],
 [-0.006973,  0.999973, -0.002215],
 [ 0.029476,  0.002419,  0.999563]], dtype=np.float64)

T = np.array([-7.371286e-02,  4.751034e-04,  7.869438e-06], dtype=np.float64)

# Baseline: 73.7 mm  →  
BASELINE_M = 0.07371
# ---------------------------------------------------------------------------
# StereoSGBM parameters (tune for your scene / lighting)
# ---------------------------------------------------------------------------
SGBM_MIN_DISP    =  5
SGBM_NUM_DISP    =  48    # must be divisible by 16 (più alto = più profondità ma più lento)
SGBM_BLOCK_SIZE  =  15    # large block = robust on low-texture clothing/skin
SGBM_P1          =   8 * 3 * SGBM_BLOCK_SIZE ** 2   # scales with blockSize²
SGBM_P2          =  32 * 3 * SGBM_BLOCK_SIZE ** 2
SGBM_DISP12DIFF  =  -1    # disabled — L-R check was rejecting too many valid pixels
SGBM_UNIQUENESS  =  21    # very lenient — maximise coverage on low-texture areas
SGBM_SPECKLE_WIN = 100
SGBM_SPECKLE_RNG =   2    # tolleranza per considerare pixel come “rumore”
