'''
_config.py
=============================================================================
Constants shared among all the evaluation_and_comparison moduls.

Usage:
    from evaluation_and_comparison._config import ARCHITECTURES, PALETTE, ...
'''

from typing import Dict, List
from utilities.config import JOINT_COLS

# ---------------------------------------------------------------------------
# Dimensions ad parameters BC
# ---------------------------------------------------------------------------
N_JOINTS     = len(JOINT_COLS)   # 16
VELOCITY_LAG = 5
VEL_CLIP     = 10.0

# ---------------------------------------------------------------------------
# Architectures and relatives subfolders in dataset
# ---------------------------------------------------------------------------
ARCHITECTURES: Dict[str, str] = {
    'MLP'        : 'MLP',
    'GRU'        : 'GRU',
    'Transformer': 'Transformer',
}

# ---------------------------------------------------------------------------
# Grouping exercises by acquisition modality
# ---------------------------------------------------------------------------
MODALITY_GROUPS: Dict[str, List[int]] = {
    'Stereo': list(range(1,  6)),   # 001-005
    'Mixed' : list(range(11, 16)),  # 011-015
    'Mono'  : list(range(21, 26)),  # 021-025
}


def get_modality(exercise_num: int) -> str:
    if  1 <= exercise_num <=  5: return 'Stereo'
    if 11 <= exercise_num <= 15: return 'Mixed'
    if 21 <= exercise_num <= 25: return 'Mono'
    return 'Unknown'


def get_exercise_type(exercise_num: int) -> int:
    '''Exercise type (1-5), indipendent from modality acquisition.'''
    if  1 <= exercise_num <=  5: return exercise_num
    if 11 <= exercise_num <= 15: return exercise_num - 10
    if 21 <= exercise_num <= 25: return exercise_num - 20
    return exercise_num


# ---------------------------------------------------------------------------
# Joint labels
# ---------------------------------------------------------------------------

# Short labels for the 16 joints (order = JOINT_COLS)
JOINT_LABELS = [
    'r_sh_p', 'r_sh_r', 'r_aw', 'r_el_p', 'r_fw_y', 'r_wr_p', 'r_wr_r', 'r_gr',
    'l_sh_p', 'l_sh_r', 'l_aw', 'l_el_p', 'l_fw_y', 'l_wr_p', 'l_wr_r', 'l_gr',
]

# Active joints: shoulder_pitch/roll, arm_yaw, elbow_pitch for each arm.
# Wrist and gripper are fixed to 0 - excluded from scalar averages and plots.
ACTIVE_IDX    = [0, 1, 2, 3, 8, 9, 10, 11]
ACTIVE_LABELS = [
    'r_sh_p', 'r_sh_r', 'r_aw', 'r_el_p',
    'l_sh_p', 'l_sh_r', 'l_aw', 'l_el_p',
]

# Joint used for velocity profile 
VELOCITY_JOINTS: Dict[str, int] = {
    'r_sh_p': 0,
    'l_sh_r': 9,
}

# ---------------------------------------------------------------------------
# Color palette (shared among plots)
# ---------------------------------------------------------------------------
PALETTE: Dict[str, str] = {
    'Human demos'   : '#e74c3c',
    'Canonical'     : '#e67e22',   
    'CanonicalShape': '#16a085',   
    'MLP'           : '#3498db',
    'GRU'           : '#9b59b6',
    'Transformer'   : '#1abc9c',
    'Stereo'        : '#2ecc71',
    'Mixed'         : '#f39c12',
    'Mono'          : '#8e44ad',
}