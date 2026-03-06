from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
import time

print("Connessione a Reachy...")
reachy = ReachySDK(host="10.59.1.20")

if reachy.r_arm is None or reachy.l_arm is None:
    print("❌ Impossible to connect")
    exit()

print("✅ Connection OK")

reachy.turn_on('r_arm')
reachy.turn_on('l_arm')
reachy.turn_on('head')


head = reachy.head

def be_sad():
    # Low arms
    goto(
        goal_positions={reachy.r_arm.r_elbow_pitch: -100,
                        reachy.l_arm.l_elbow_pitch: -100,
                        reachy.r_arm.r_shoulder_pitch: 0,
                        reachy.l_arm.l_shoulder_pitch: 0,
                        reachy.r_arm.r_shoulder_roll: 0,
                        reachy.l_arm.l_shoulder_roll: 0,
                        },
        duration=1.0,
        interpolation_mode=InterpolationMode.MINIMUM_JERK
    )
    head.look_at(0.5, 0, -0.5, 1.8, interpolation_mode=InterpolationMode.MINIMUM_JERK)
    goto(
        goal_positions={
            head.l_antenna: 90,
            head.r_antenna: -90
        },
        duration=1.0,
        interpolation_mode=InterpolationMode.MINIMUM_JERK
    )


def be_happy():
    head.look_at(0.5, 0, 0.2, 1.8, interpolation_mode=InterpolationMode.MINIMUM_JERK)   # look up

    # Raise arms
    goto(
        goal_positions={reachy.r_arm.r_shoulder_pitch: -130,
                        reachy.l_arm.l_shoulder_pitch: -130,
                        reachy.r_arm.r_shoulder_roll: -20,
                        reachy.l_arm.l_shoulder_roll: 20,
                        reachy.r_arm.r_elbow_pitch: -5,
                        reachy.l_arm.l_elbow_pitch: -5
                        },
        duration=1.0,
        interpolation_mode=InterpolationMode.MINIMUM_JERK
    )
    
    # Antennae waving
    reachy.head.l_antenna.speed_limit = 0.0
    reachy.head.r_antenna.speed_limit = 0.0
    
    for _ in range(9):
        reachy.head.l_antenna.goal_position = 10.0
        reachy.head.r_antenna.goal_position = -10.0

        time.sleep(0.1)

        reachy.head.l_antenna.goal_position = -10.0
        reachy.head.r_antenna.goal_position = 10.0

        time.sleep(0.1)
    
    reachy.head.l_antenna.goal_position = 0.0
    reachy.head.r_antenna.goal_position = 0.0
    
    
be_happy()
time.sleep(2)   
be_sad()
time.sleep(2)

# Back to neutral position
head.look_at(0.5, 0, 0, 1, interpolation_mode=InterpolationMode.MINIMUM_JERK)
goto(
    goal_positions={head.l_antenna: 0, head.r_antenna: 0},
    duration=1.0,   
    interpolation_mode=InterpolationMode.MINIMUM_JERK
)


reachy.turn_off_smoothly('r_arm')
reachy.turn_off_smoothly('l_arm')
reachy.turn_off_smoothly('head')
