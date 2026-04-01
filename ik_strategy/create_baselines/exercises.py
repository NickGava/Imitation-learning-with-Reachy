from typing import List, Dict
# ===========================================================================
# EXERCISES DEFINITION
# ===========================================================================
# Each exercise is a list of keyframes. Every keyframe must have 'duration' (seconds to reach it from the previous keyframe).
EXERCISES: Dict[int, List[Dict[str, float]]] = {

    # -----------------------------------------------------------------------
    # Exercise 1 - Right arm lateral raise
    # -----------------------------------------------------------------------
    1: [
        # Phase 1: raise right arm laterally to 90°
        {
            'duration':       2.0,
            'r_shoulder_roll': -90.0,   # lateral raise (away from body)
        },
        # Phase 2: hold at top for 1 second
        {
            'duration':       1.0,
            'r_shoulder_roll': -90.0,
        },
        # Phase 3: return to rest
        {
            'duration':       2.0,
            'r_shoulder_roll': -5.0,
        },
    ],

    # -----------------------------------------------------------------------
    # Exercise 2 - Left arm lateral raise
    # -----------------------------------------------------------------------
    2: [
        # Phase 1: raise left arm laterally to 90°
        {
            'duration':       2.0,
            'l_shoulder_roll': 90.0,   # lateral raise (away from body)
        },
        # Phase 2: hold at top for 1 second
        {
            'duration':       1.0,
            'l_shoulder_roll': 90.0,
        },
        # Phase 3: return to rest
        {
            'duration':       2.0,
            'l_shoulder_roll': 5.0,
        },
    ],

    # -----------------------------------------------------------------------
    # Exercise 3 - Bilateral arm raise (both arms forward)
    # -----------------------------------------------------------------------
    3: [
        # Phase 1: raise both arms forward to 90°
        {
            'duration':         2.0,
            'r_shoulder_pitch': -90.0,
            'l_shoulder_pitch': -90.0,
        },
        # Phase 2: hold
        {
            'duration':         1.0,
            'r_shoulder_pitch': -90.0,
            'l_shoulder_pitch': -90.0,
        },
        # Phase 3: return
        {
            'duration':         2.0,
            'r_shoulder_pitch':  0.0,
            'l_shoulder_pitch':  0.0,
        },
    ],

    # -----------------------------------------------------------------------
    # Exercise 4 - Elbow + shoulder motion (right arm)
    # -----------------------------------------------------------------------
    4: [
        # Phase 1: back
        {
            'duration':      2.0,
            'r_elbow_pitch': -125.0,  
            'r_shoulder_pitch': 40.0,
        },
        # Phase 2: hold
        {
            'duration':      1.0,
        },
        # Phase 3: front
        {
            'duration':      2.0,
            'r_elbow_pitch': -5.0,
            'r_shoulder_pitch': -75.0,
        },
        # Phase 4: hold
        {
            'duration':      1.0,
        },
        # Phase 5: return 
        {
            'duration':      2.0,
            'r_elbow_pitch': -90.0,
            'r_shoulder_pitch': 0.0,
        }
    ],

    # -----------------------------------------------------------------------
    # Exercise 5 - Elbow + shoulder motion (left arm)
    # -----------------------------------------------------------------------
    5: [
        # Phase 1: back
        {
            'duration':      2.0,
            'l_elbow_pitch': -125.0,  
            'l_shoulder_pitch': 40.0,
        },
        # Phase 2: hold
        {
            'duration':      1.0,
        },
        # Phase 3: front
        {
            'duration':      2.0,
            'l_elbow_pitch': -5.0,
            'l_shoulder_pitch': -75.0,
        },
        # Phase 4: hold
        {
            'duration':      1.0,
        },
        # Phase 5: return 
        {
            'duration':      2.0,
            'l_elbow_pitch': -90.0,
            'l_shoulder_pitch': 0.0,
        }
    ],

    # -----------------------------------------------------------------------
    # Exercise 6 - Shoulder rotation (right arm)
    # -----------------------------------------------------------------------
    6: [
        # Phase 1: start
        {
            'duration':      2.0,
            'r_shoulder_roll': -90.0,
            'r_shoulder_pitch': -90.0,
        },
        # Phase 2: hold
        {
            'duration':      0.5,
        },
        # Phase 3: middle
        {
            'duration':      2.0,
            'r_shoulder_pitch': 0.0,
        },
        # Phase 4: hold
        {
            'duration':      0.5,
        },
        # Phase 5: back
        {
            'duration':     2.0,
            'r_shoulder_pitch': -90.0,
        },
        # Phase 6: return 
        {
            'duration':      2.0,
            'r_elbow_pitch': -90.0,
            'r_shoulder_pitch': 0.0,
            'r_shoulder_roll': 0.0,
        }
    ],

    # -----------------------------------------------------------------------
    # Exercise 7 - Shoulder rotation (left arm)
    # -----------------------------------------------------------------------
    7: [
        # Phase 1: start
        {
            'duration':      2.0,
            'l_shoulder_roll': 90.0,
            'l_shoulder_pitch': -90.0,
        },
        # Phase 2: hold
        {
            'duration':      0.5,
        },
        # Phase 3: middle
        {
            'duration':      2.0,
            'l_shoulder_pitch': 0.0,
        },
        # Phase 4: hold
        {
            'duration':      0.5,
        },
        # Phase 5: back
        {
            'duration':     2.0,
            'l_shoulder_pitch': -90.0,
        },
        # Phase 5: return 
        {
            'duration':      2.0,
            'l_elbow_pitch': -90.0,
            'l_shoulder_pitch': 0.0,
            'l_shoulder_roll': 0.0,
        }
    ],

    # -----------------------------------------------------------------------
    # Exercise 8 - Single arm open (right arm)
    # -----------------------------------------------------------------------
    8: [
        # Phase 1: raise arm forward
        {
            'duration':         1.0,
            'r_shoulder_pitch': -90.0,
        },
        # Phase 2: hold
        {
            'duration':         1.0,
        },
        # Phase 3: open
        {
            'duration':         2.0,
            'r_shoulder_roll':  -90.0,
        },
        # Phase 4: hold
        {
            'duration':         1.0,
        },
        # Phase 5: close
        {
            'duration':         2.0,
            'r_shoulder_roll':  0.0,
        },
        # Phase 6: return
        {
            'duration':     1.0,
            'r_shoulder_pitch': 0.0,
        }        
    ],
    # -----------------------------------------------------------------------
    # Exercise 9 - Single arm open (left arm)
    # -----------------------------------------------------------------------
    9: [
        # Phase 1: raise arm forward
        {
            'duration':         1.0,
            'l_shoulder_pitch': -90.0,
        },
        # Phase 2: hold
        {
            'duration':         1.0,
        },
        # Phase 3: open
        {
            'duration':         2.0,
            'l_shoulder_roll':  90.0,
        },
        # Phase 4: hold
        {
            'duration':         1.0,
        },
        # Phase 5: close
        {
            'duration':         2.0,
            'l_shoulder_roll':  0.0,
        },
        # Phase 6: return
        {
            'duration':     1.0,
            'l_shoulder_pitch': 0.0,
        }        
    ],
}