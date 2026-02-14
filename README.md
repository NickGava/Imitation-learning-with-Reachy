# Imitation-learning-with-Reachy
This project aims at enabling Reachy to learn and reproduce simple upperbody movements demonstrated by humans.

Objectives:
1. Human motion acquisition: Capture human upper-body movements
using a vision-based pose estimation system and represent them as time
series.
2. Movement representation and processing: Develop robust representations of human gestures from multiple demonstrations (normalization,
alignment, averaging, etc.).
3. Human-to-robot motion transfer: Map human motion trajectories to
the robot morphology and generate feasible robot trajectories.
4. Learning from demonstration: Implement methods to learn a generalized movement model from several demonstrations and reproduce it on
the robot.
5. Demonstration scenario: Produce a final demonstrator where the robot
learns a gesture from human demonstrations and reproduces it.


## Installations
### Simulator
- Unity 2020.3 LTS
- 3D Core project
- import reachy2021-simulator.unitypackage: <br>
dowload package from https://github.com/pollen-robotics/reachy2021-unity-package release's page, then in Unity: Assets -> Import Package -> Custom Package -> select reachy2021-simulator.unitypackage
- get grpc unity package: Pollen Robotics -> Install GRPC

### Code Enviroment
- Python 3.9
- use a virtual enviroment: <br>
    `py -3.9 -m venv reachy_env`<br>
    `reachy_env\Scripts\activate`
- required library:<br>
    `pip install reachy-sdk`

## Configuration
