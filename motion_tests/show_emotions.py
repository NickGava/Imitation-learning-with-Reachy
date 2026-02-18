from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
import time

print("Connessione a Reachy...")
reachy = ReachySDK(host="localhost")

if reachy.r_arm is None or reachy.l_arm is None:
    print("❌ Braccia non disponibili (Unity in Play?)")
    exit()

print("✅ Connessione OK")


head = reachy.head

def be_sad():
    head.l_antenna.goal_position = 90
    head.r_antenna.goal_position = -90
    head.look_at(0.5, 0, -0.5, 1.8, interpolation_mode=InterpolationMode.MINIMUM_JERK)


def be_happy():
    head.look_at(0.5, 0, 0.2, 1.8, interpolation_mode=InterpolationMode.MINIMUM_JERK)   # look up

    # Raise arms
    goto(
        goal_positions={reachy.r_arm.r_shoulder_pitch: -130,
                        reachy.l_arm.l_shoulder_pitch: -130,
                        reachy.r_arm.r_shoulder_roll: -20,
                        reachy.l_arm.l_shoulder_roll: 20
                        },
        duration=1.0,
        interpolation_mode=InterpolationMode.MINIMUM_JERK
    )
    
    # Antennae waving
    for _ in range(5):
        goto(
            goal_positions={head.l_antenna: 30, head.r_antenna: -30},
            duration=0.3,   
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )
        time.sleep(0.3)
        goto(
            goal_positions={head.l_antenna: 0, head.r_antenna: 0},
            duration=0.3,   
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )
        time.sleep(0.3)
    
    # Return arms to neutral position
    goto(
        goal_positions={reachy.r_arm.r_shoulder_pitch: 0,
                        reachy.l_arm.l_shoulder_pitch: 0,
                        reachy.r_arm.r_shoulder_roll: 0,
                        reachy.l_arm.l_shoulder_roll: 0
                        },
        duration=1.0,
        interpolation_mode=InterpolationMode.MINIMUM_JERK
    )
    
    
be_happy()
time.sleep(3)   
be_sad()
time.sleep(3)

# Back to neutral position
head.look_at(0.5, 0, 0, 1, interpolation_mode=InterpolationMode.MINIMUM_JERK)
goto(
    goal_positions={head.l_antenna: 0, head.r_antenna: 0},
    duration=1.0,   
    interpolation_mode=InterpolationMode.MINIMUM_JERK
)

