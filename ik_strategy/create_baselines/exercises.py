from typing import List, Dict
from config import STARTING_POSE
# ===========================================================================
# EXERCISES DEFINITION
# ===========================================================================
# Each exercise is a list of keyframes. Every keyframe must have 'duration' (seconds to reach it from the previous keyframe).
EXERCISES: Dict[int, List[Dict[str, float]]] = {

    # -----------------------------------------------------------------------
    # Exercise 1 - Right arm lateral raise
    # -----------------------------------------------------------------------
    1: [
        # Strarting frame
        STARTING_POSE,  
        # Phase 1: raise right arm laterally to 90°
        {
            'duration':       2.0,
            'r_shoulder_roll': -90.0,   # lateral raise (away from body)
        },
        # Phase 2: hold
        {
            'duration':       1.0,
        },
        # Phase 3: return 
        {
            'duration':       2.0,
            'r_shoulder_roll': 0.0,
        },
    ],

    # -----------------------------------------------------------------------
    # Exercise 2 - Left arm lateral raise
    # -----------------------------------------------------------------------
    2: [
        # Starting frame
        STARTING_POSE,
        # Phase 1: raise left arm laterally to 90°
        {
            'duration':       2.0,
            'l_shoulder_roll': 90.0,   # lateral raise (away from body)
        },
        # Phase 2: hold 
        {
            'duration':       0.5,
        },
        # Phase 3: return 
        {
            'duration':       2.0,
            'l_shoulder_roll': 0.0,
        },
    ],

    # -----------------------------------------------------------------------
    # Exercise 3 - Bilateral arm raise (both arms forward)
    # -----------------------------------------------------------------------
    3: [
        # Starting frame
        STARTING_POSE,
        # Phase 1: raise both arms forward to 90°
        {
            'duration':         2.0,
            'r_shoulder_pitch': -90.0,
            'l_shoulder_pitch': -90.0,
        },
        # Phase 2: hold
        {
            'duration':         0.5,
        },
        # Phase 3: return
        {
            'duration':         2.0,
            'r_shoulder_pitch': -20.0,
            'l_shoulder_pitch': -20.0,
        }
    ],

    # -----------------------------------------------------------------------
    # Exercise 4 - Elbow + shoulder motion (right arm)
    # -----------------------------------------------------------------------
    4: [
        # Starting frame
        {
            'r_elbow_pitch': -90.0,
            'r_shoulder_pitch': 0.0,
            'r_arm_yaw':     15.0,
        },
        # Phase 1: rotate arm
        {
            'duration':      1.5,
            'r_arm_yaw':    -60.0,
        },
        # Phase 2: hold
        {
            'duration':      0.5,
        },
        # Phase 3: back
        {
            'duration':      1.5,
            'r_arm_yaw':     15.0,
        },
    ],

    # -----------------------------------------------------------------------
    # Exercise 5 - Elbow + shoulder motion (left arm)
    # -----------------------------------------------------------------------
    5: [
        # Starting frame
        {
            'l_elbow_pitch': -90.0,
            'l_shoulder_pitch': 0.0,
            'l_arm_yaw':     -15.0,
        },
        # Phase 1: rotate arm
        {
            'duration':      1.5,
            'l_arm_yaw':    60.0,
        },
        # Phase 2: hold
        {
            'duration':      0.5,
        },
        # Phase 3: back
        {
            'duration':      1.5,
            'l_arm_yaw':     -15.0,
        },
    ],

    # -----------------------------------------------------------------------
    # Exercise 6 - Shoulder rotation (right arm)
    # -----------------------------------------------------------------------
    6: [
        # Starting frame
        {
            'r_shoulder_roll': -90.0,
            'r_shoulder_pitch': -90.0,
            'r_elbow_pitch': -90.0,
        },
        # Phase 1: middle
        {
            'duration':      2.0,
            'r_shoulder_pitch': 0.0,
        },
        # Phase 2: hold
        {
            'duration':      0.5,
        },
        # Phase 3: back
        {
            'duration':     2.0,
            'r_shoulder_pitch': -90.0,
        },
    ],

    # -----------------------------------------------------------------------
    # Exercise 7 - Shoulder rotation (left arm)
    # -----------------------------------------------------------------------
    7: [
        # Starting frame
        {
            'l_shoulder_roll': 90.0,
            'l_shoulder_pitch': -90.0,
            'l_elbow_pitch': -90.0,
        },
        # Phase 1: middle
        {
            'duration':      2.0,
            'l_shoulder_pitch': 0.0,
        },
        # Phase 2: hold
        {
            'duration':      0.5,
        },
        # Phase 3: back
        {
            'duration':     2.0,
            'l_shoulder_pitch': -90.0,
        },
    ],

    # -----------------------------------------------------------------------
    # Exercise 8 - Single arm open (right arm)
    # -----------------------------------------------------------------------
    8: [
        # Starting frame
        {
            'r_shoulder_pitch': -90.0,
            'r_elbow_pitch':    -90.0,
        },
        # Phase 1: open
        {
            'duration':         2.0,
            'r_shoulder_roll': -90.0,
        },
        # Phase 2: hold
        {
            'duration':         0.5,
        },
        # Phase 3: close
        {
            'duration':         2.0,
            'r_shoulder_roll':  0.0,
        },    
    ],
    # -----------------------------------------------------------------------
    # Exercise 9 - Single arm open (left arm)
    # -----------------------------------------------------------------------
    9: [
        # Starting frame
        {
            'l_shoulder_pitch': -90.0,
            'l_elbow_pitch': -90.0,
        },
        # Phase 1: open
        {
            'duration':         2.0,
            'l_shoulder_roll':  90.0,
        },
        # Phase 2: hold
        {
            'duration':         0.5,
        },
        # Phase 3: close
        {
            'duration':         2.0,
            'l_shoulder_roll':  0.0,
        },    
    ],
}