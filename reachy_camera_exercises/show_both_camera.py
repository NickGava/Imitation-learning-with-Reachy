import cv2
from reachy_sdk import ReachySDK

# IP del robot
ROBOT_IP = "192.168.1.42"   # cambia con il tuo IP

# connessione al robot
reachy = ReachySDK(host=ROBOT_IP)

if not reachy.is_connected():
    print("Connessione fallita")
    exit()

print("Connesso a Reachy!")

# loop video
while True:
    # frame dalle camere
    left_frame = reachy.cameras.left.get_frame()
    right_frame = reachy.cameras.right.get_frame()

    if left_frame is not None:
        cv2.imshow("Left Camera", left_frame)

    if right_frame is not None:
        cv2.imshow("Right Camera", right_frame)

    # premi q per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()