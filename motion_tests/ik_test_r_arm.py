from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
import time
import threading
import numpy as np

print("Connessione a Reachy...")
reachy = ReachySDK(host="localhost")

if reachy.r_arm is None or reachy.l_arm is None:
    print("❌ Braccia non disponibili (Unity in Play?)")
    exit()

print("✅ Connessione OK")

def unity_to_robot_coords(unity_coords, offset=[0, 0, 0]):
    """
    Converte le coordinate da Unity a Reachy SDK.
    unity_coords: [x, y, z] presi dall'Inspector di Unity
    offset: la posizione del torso di Reachy in Unity (x, y, z)
    """
    rel_x = (unity_coords[0] - offset[0])
    rel_y = (unity_coords[1] - offset[1])
    rel_z = (unity_coords[2] - offset[2])  
    
    robot_x = -rel_x 
    robot_y = -rel_z 
    robot_z = rel_y 

    return np.array([robot_x, robot_y, robot_z])

def get_target_pose(x, y, z, roll_deg, pitch_deg, yaw_deg):
    # Converti i gradi in radianti
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y_rad = np.radians(yaw_deg)

    # Matrici di rotazione per asse
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])

    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])

    Rz = np.array([[np.cos(y_rad), -np.sin(y_rad), 0],
                   [np.sin(y_rad), np.cos(y_rad), 0],
                   [0, 0, 1]])

    # Combina le rotazioni (Ordine standard: Rz * Ry * Rx)
    Rotation = Rz @ Ry @ Rx

    # Crea la matrice target_pose 4x4
    target_pose = np.eye(4)
    target_pose[:3, :3] = Rotation # Inserisce la rotazione
    target_pose[:3, 3] = [x, y, z] # Inserisce la traslazione
    
    return target_pose

def get_sphere(x, y, z):
    # Coordinate sfera
    unity_pos = [x, y, z]
    robot_pos = unity_to_robot_coords(unity_pos)
    # print("Coordinate convertite per Reachy:", robot_pos)

    reachy.head.look_at(robot_pos[0], robot_pos[1], robot_pos[2], duration=0.5)

    matrice = get_target_pose(robot_pos[0], robot_pos[1], robot_pos[2], 0, -90, 0)

    # Calcoliamo la Cinematica Inversa
    # Il metodo restituisce una lista di angoli per i giunti
    joint_angles = reachy.r_arm.inverse_kinematics(matrice)
    # print("Angoli dei giunti calcolati dall'IK:", joint_angles)

    # Se l'IK ha trovato una soluzione, muoviamo il braccio
    if joint_angles:
        # Creiamo il dizionario per il comando goto
        target_position = dict(zip(reachy.r_arm.joints.values(), joint_angles))
        
        # Eseguiamo il movimento
        goto(target_position, duration=1.0, interpolation_mode=InterpolationMode.MINIMUM_JERK)
        time.sleep(1)  # Aspettiamo che il movimento sia completato
        p = reachy.r_arm.forward_kinematics()
        # print("Posizione finale del gripper:", p[:3, 3])
    else:
        print("Errore: Posizione irraggiungibile!")

get_sphere(-0.3, 0, 0.4)
get_sphere(-0.4, 0.2, 0.5)
get_sphere(-0.4, -0.2, 0.1)
get_sphere(-0.2, 0.5, 0.4)

# Torniamo in posizione neutra
goto(
    goal_positions={getattr(reachy.joints, name): 0.0 for name in reachy.joints.keys()},
    duration=2.0,
    interpolation_mode = InterpolationMode.MINIMUM_JERK
)