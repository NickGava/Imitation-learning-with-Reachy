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

#print(reachy.head)

head = reachy.head

head.look_at(0.5, 0.6, 0, 1.8, interpolation_mode=InterpolationMode.MINIMUM_JERK)       # look left
head.look_at(0.5, 0, 0.6, 1.8, interpolation_mode=InterpolationMode.MINIMUM_JERK)       # look up
head.look_at(0.5, -0.6, 0, 1.8, interpolation_mode=InterpolationMode.MINIMUM_JERK)      # look right
head.look_at(0.5, 0, -0.6, 1.8, interpolation_mode=InterpolationMode.MINIMUM_JERK)      # look down
head.look_at(0.5, 0, 0, 1.8, interpolation_mode=InterpolationMode.MINIMUM_JERK)         # look center


