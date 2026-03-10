"""
debug_orientation.py
=============================================================================
Simulates the full hand-orientation pipeline with fake but controlled data.

Pipeline tested:
  1. Build hand frame from 3 landmarks (camera frame)          [hand_processing.py]
  2. Convert rotation matrix → quaternion                      [hand_processing.py]
  3. Build torso frame from shoulder/hip landmarks             [mapping.py]
  4. Rotate quaternion: camera frame → torso frame             [mapping.py]
  5. Rebuild rotation matrix from the final quaternion         [run_ik.py]

Each test case has a KNOWN expected result so you can spot errors immediately.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# MATH HELPERS  (same formulas used in the real pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)

def rotation_matrix_to_quaternion(R):
    """Shepperd method — same as hand_processing.py"""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s;  x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s;  z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s;  x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s;                z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s;  x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s;  z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

def quaternion_to_rotation_matrix(q):
    """Same as run_ik.py"""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

def build_hand_frame(wrist, index_mcp, pinky_mcp):
    """
    Same as hand_processing._compute_orientation()
    Returns R_hand_in_camera (3x3) and the quaternion.
    """
    v_forward = index_mcp - wrist
    v_lateral = pinky_mcp - wrist

    e1 = normalize(v_forward)           # forward (finger direction)
    e3 = normalize(np.cross(e1, v_lateral))   # palm normal
    e2 = np.cross(e3, e1)               # lateral

    if np.linalg.norm(e3) < 1e-9:
        return np.eye(3), np.array([1., 0., 0., 0.])

    R = np.column_stack([e1, e2, e3])   # columns = hand axes in camera frame
    q = rotation_matrix_to_quaternion(R)
    return R, q

def build_torso_frame(l_sh, r_sh, l_hip, r_hip):
    """
    Same as mapping._build_torso_rotation_matrix()
    Returns R_torso (columns = torso axes in camera frame).
    """
    mid_sh  = (l_sh + r_sh)   * 0.5
    mid_hip = (l_hip + r_hip) * 0.5

    y = normalize(l_sh - r_sh)          # lateral (right → left)
    z = normalize(mid_sh - mid_hip)     # up
    z = normalize(z - np.dot(z, y) * y) # re-orthogonalize
    x = np.cross(y, z)                  # forward (toward camera)

    return np.column_stack([x, y, z])   # columns = torso axes in camera frame

def rotate_quaternion_to_torso(q_camera, R_torso):
    """
    Same as mapping._map_arm():
        R_reachy = R_torso.T @ R_hand_camera
    """
    R_hand_camera = quaternion_to_rotation_matrix(q_camera)
    R_hand_torso  = R_torso.T @ R_hand_camera
    return rotation_matrix_to_quaternion(R_hand_torso), R_hand_torso

# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt_mat(R, indent=4):
    pad = " " * indent
    rows = []
    for row in R:
        rows.append(pad + "  ".join(f"{v:+.4f}" for v in row))
    return "\n".join(rows)

def fmt_q(q):
    return f"w={q[0]:+.4f}  x={q[1]:+.4f}  y={q[2]:+.4f}  z={q[3]:+.4f}"

def check_orthogonal(R, name):
    err = np.max(np.abs(R.T @ R - np.eye(3)))
    det = np.linalg.det(R)
    ok_orth = err < 1e-6
    ok_det  = abs(det - 1.0) < 1e-6
    status  = "✅" if (ok_orth and ok_det) else "❌"
    print(f"  {status} {name}: orthogonal_err={err:.2e}, det={det:.6f}")

def check_roundtrip(R_original, q, label):
    R_back = quaternion_to_rotation_matrix(q)
    err = np.max(np.abs(R_original - R_back))
    status = "✅" if err < 1e-6 else "❌"
    print(f"  {status} {label} R→q→R roundtrip error: {err:.2e}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

def run_test(name, wrist, index_mcp, pinky_mcp,
             l_sh, r_sh, l_hip, r_hip,
             expected_note):
    print(f"\n{'═'*65}")
    print(f"  TEST: {name}")
    print(f"  Expected: {expected_note}")
    print(f"{'═'*65}")

    # ── STEP 1-2: hand frame in camera coords ──────────────────────────────
    R_hand_cam, q_cam = build_hand_frame(wrist, index_mcp, pinky_mcp)

    print(f"\n[Step 1-2] Hand frame in CAMERA coords")
    print(f"  R_hand_cam (columns = e1_forward, e2_lateral, e3_normal):")
    print(fmt_mat(R_hand_cam))
    print(f"  q_cam: {fmt_q(q_cam)}")
    check_orthogonal(R_hand_cam, "R_hand_cam")
    check_roundtrip(R_hand_cam, q_cam, "q_cam")

    # ── STEP 3: torso frame ────────────────────────────────────────────────
    R_torso = build_torso_frame(l_sh, r_sh, l_hip, r_hip)

    print(f"\n[Step 3] Torso frame (columns = x_fwd, y_lateral, z_up in camera):")
    print(fmt_mat(R_torso))
    check_orthogonal(R_torso, "R_torso")

    # ── STEP 4: rotate to torso frame ─────────────────────────────────────
    q_torso, R_hand_torso = rotate_quaternion_to_torso(q_cam, R_torso)

    print(f"\n[Step 4] Hand orientation in TORSO frame")
    print(f"  R_hand_torso:")
    print(fmt_mat(R_hand_torso))
    print(f"  q_torso: {fmt_q(q_torso)}")
    check_orthogonal(R_hand_torso, "R_hand_torso")
    check_roundtrip(R_hand_torso, q_torso, "q_torso")

    # ── STEP 5: rebuild R from final quaternion (as run_ik.py does) ────────
    R_final = quaternion_to_rotation_matrix(q_torso)
    err_final = np.max(np.abs(R_final - R_hand_torso))
    print(f"\n[Step 5] R rebuilt from q_torso (as in run_ik.py):")
    print(fmt_mat(R_final))
    status = "✅" if err_final < 1e-6 else "❌"
    print(f"  {status} Max deviation from R_hand_torso: {err_final:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
# Camera frame convention (MediaPipe image coords):
#   x → right,  y → down,  z → toward camera (out of screen)
#
# Torso frame (Reachy):
#   x → forward (toward camera),  y → left,  z → up
#
# Person stands facing the camera, arms at sides.

def main():
    print("=" * 65)
    print("  HAND ORIENTATION PIPELINE — DEBUG SIMULATION")
    print("=" * 65)

    # ── Shared body landmarks (person standing, facing camera) ──────────────
    # These are in camera/world frame: x=right, y=down, z=depth
    l_sh  = np.array([-0.2,  0.0, 0.0])   # left shoulder  (person's left = camera right)
    r_sh  = np.array([ 0.2,  0.0, 0.0])   # right shoulder
    l_hip = np.array([-0.15, 0.5, 0.0])
    r_hip = np.array([ 0.15, 0.5, 0.0])

    # ── TEST 1: Palm facing camera, fingers pointing up ─────────────────────
    # In camera frame: fingers go in -y (up), palm normal toward +z (camera)
    wrist     = np.array([0.3, 0.3, 0.0])
    index_mcp = wrist + np.array([0.0, -0.08, 0.0])  # fingers point up (−y)
    pinky_mcp = wrist + np.array([-0.04, 0.0, 0.0])  # pinky to the left
    run_test(
        name="Palm facing camera, fingers up",
        wrist=wrist, index_mcp=index_mcp, pinky_mcp=pinky_mcp,
        l_sh=l_sh, r_sh=r_sh, l_hip=l_hip, r_hip=r_hip,
        expected_note="In torso frame: fingers should point UP (+z), palm normal toward +x (forward)"
    )

    # ── TEST 2: Identity — hand frame aligned with torso frame ──────────────
    # In camera frame: torso x=forward(+z toward cam? no, let's align directly)
    # Torso frame: x=cross(y,z), y=normalize(l_sh-r_sh)=(-1,0,0), z=up=(0,-1,0)
    # Let's build a hand that has the exact same axes as torso in camera coords
    # First compute the torso frame to know its axes
    R_t = build_torso_frame(l_sh, r_sh, l_hip, r_hip)
    # Hand aligned with torso: wrist→index along torso-x, lateral toward torso-y
    wrist2     = np.array([0.3, 0.3, 0.0])
    index_mcp2 = wrist2 + R_t[:, 0] * 0.08    # forward along torso x
    pinky_mcp2 = wrist2 + R_t[:, 1] * 0.05    # lateral along torso y
    run_test(
        name="Hand frame = Torso frame (identity case)",
        wrist=wrist2, index_mcp=index_mcp2, pinky_mcp=pinky_mcp2,
        l_sh=l_sh, r_sh=r_sh, l_hip=l_hip, r_hip=r_hip,
        expected_note="q_torso should be identity: w≈1, x≈y≈z≈0; R_hand_torso ≈ I"
    )

    # ── TEST 3: Hand rotated 90° around torso Z (up axis) ───────────────────
    # Fingers point left (torso -y), palm normal still forward (+x)
    wrist3     = np.array([0.3, 0.3, 0.0])
    index_mcp3 = wrist3 - R_t[:, 1] * 0.08    # fingers along -torso_y (left→right in image)
    pinky_mcp3 = wrist3 + R_t[:, 0] * 0.05    # lateral toward torso_x
    run_test(
        name="Hand rotated 90° around torso Z",
        wrist=wrist3, index_mcp=index_mcp3, pinky_mcp=pinky_mcp3,
        l_sh=l_sh, r_sh=r_sh, l_hip=l_hip, r_hip=r_hip,
        expected_note="q_torso should be 90° around z: w≈0.707, x≈0, y≈0, z≈±0.707"
    )

    # ── TEST 4: Noisy/degenerate — pinky_mcp collinear with index_mcp ───────
    wrist4     = np.array([0.3, 0.3, 0.0])
    index_mcp4 = wrist4 + np.array([0.0, -0.08, 0.0])
    pinky_mcp4 = wrist4 + np.array([0.0, -0.04, 0.0])  # collinear! degenerate
    run_test(
        name="Degenerate: pinky collinear with index (bad detection)",
        wrist=wrist4, index_mcp=index_mcp4, pinky_mcp=pinky_mcp4,
        l_sh=l_sh, r_sh=r_sh, l_hip=l_hip, r_hip=r_hip,
        expected_note="Should fall back to identity quaternion without crashing"
    )

    print(f"\n{'═'*65}")
    print("  DONE")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()