'''
_config.py
=============================================================================
Costanti condivise da tutti i moduli di evaluation_and_comparison.

Importare con:
    from evaluation_and_comparison._config import ARCHITECTURES, PALETTE, ...
'''

from typing import Dict, List
from utilities.config import JOINT_COLS

# ---------------------------------------------------------------------------
# Dimensioni e parametri BC
# ---------------------------------------------------------------------------
N_JOINTS     = len(JOINT_COLS)   # 16
VELOCITY_LAG = 5
VEL_CLIP     = 10.0

# ---------------------------------------------------------------------------
# Architetture e relative sottocartelle nel dataset
# ---------------------------------------------------------------------------
ARCHITECTURES: Dict[str, str] = {
    'MLP'        : 'MLP',
    'GRU'        : 'GRU',
    'Transformer': 'Transformer',
}

# ---------------------------------------------------------------------------
# Raggruppamento esercizi per modalità di acquisizione
# ---------------------------------------------------------------------------
MODALITY_GROUPS: Dict[str, List[int]] = {
    'Stereo': list(range(1,  6)),   # 001–005
    'Mixed' : list(range(11, 16)),  # 011–015
    'Mono'  : list(range(21, 26)),  # 021–025
}


def get_modality(exercise_num: int) -> str:
    if  1 <= exercise_num <=  5: return 'Stereo'
    if 11 <= exercise_num <= 15: return 'Mixed'
    if 21 <= exercise_num <= 25: return 'Mono'
    return 'Unknown'


def get_exercise_type(exercise_num: int) -> int:
    '''Tipo di esercizio (1–5), indipendente dalla modalità di acquisizione.'''
    if  1 <= exercise_num <=  5: return exercise_num
    if 11 <= exercise_num <= 15: return exercise_num - 10
    if 21 <= exercise_num <= 25: return exercise_num - 20
    return exercise_num


# ---------------------------------------------------------------------------
# Etichette joint
# ---------------------------------------------------------------------------

# Short labels per i 16 joint (ordine = JOINT_COLS)
JOINT_LABELS = [
    'r_sh_p', 'r_sh_r', 'r_aw', 'r_el_p', 'r_fw_y', 'r_wr_p', 'r_wr_r', 'r_gr',
    'l_sh_p', 'l_sh_r', 'l_aw', 'l_el_p', 'l_fw_y', 'l_wr_p', 'l_wr_r', 'l_gr',
]

# Active joints: shoulder_pitch/roll, arm_yaw, elbow_pitch per braccio.
# Wrist e gripper sono fissi a 0 — esclusi dalle medie scalari e dai plot.
ACTIVE_IDX    = [0, 1, 2, 3, 8, 9, 10, 11]
ACTIVE_LABELS = [
    'r_sh_p', 'r_sh_r', 'r_aw', 'r_el_p',
    'l_sh_p', 'l_sh_r', 'l_aw', 'l_el_p',
]

# Joint usati per il velocity profile (uno per braccio, rappresentativi)
VELOCITY_JOINTS: Dict[str, int] = {
    'r_sh_p': 0,
    'l_sh_r': 9,
}

# ---------------------------------------------------------------------------
# Palette colori (condivisa tra tutti i plot)
# ---------------------------------------------------------------------------
PALETTE: Dict[str, str] = {
    'Human demos': '#e74c3c',
    'Canonical'  : '#e67e22',
    'MLP'        : '#3498db',
    'GRU'        : '#9b59b6',
    'Transformer': '#1abc9c',
    'Stereo'     : '#2ecc71',
    'Mixed'      : '#f39c12',
    'Mono'       : '#8e44ad',
}
