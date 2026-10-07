# src/contact_prediction_rl/assets/data/franka_hx5d20/SOURCE.md

Franka Emika Panda arm with the ROBOTIS HX5-D20 right hand, as a single USD (`franka_hx5d20.usd`).

Sources:
- Franka Emika Panda: Isaac Lab's Panda asset (arm links `panda_link0`-`panda_link7` and `panda_hand`)
- ROBOTIS HX5-D20 right hand: https://github.com/ROBOTIS-GIT/robotis_hand, model `hx5_d20_rev2`,
  MJCF file `robotis_hand_description/mujoco/hx5_d20_rev2/hx5_d20_right.xml`

How it was built:
- the hand MJCF was converted to USD with Isaac Lab's `scripts/tools/convert_mjcf.py --self_collision`
- the hand base (`/World/ConvertedAsset/Geometry/hx5_d20_right_base`) was attached to `/World/panda_hand` with a fixed
  joint, at the identity transform: the hand base frame and `panda_hand` coincide

Local changes:
- set the default prim to `/World`, so the file can be referenced as a robot
- removed the articulation-root schemas of the hand base (`PhysicsArticulationRootAPI`, `NewtonArticulationRootAPI`,
  `PhysxArticulationAPI`), so the arm and the hand form one articulation rooted at `/World`

Notes:
- the hand in this file comes from the MJCF model, not from the URDF-generated `hx5_d20/right` USD of this repository;
  the calibrated values of the lift task (closed posture, grasp centre, start poses) were measured on this file
- the file is self-contained (no external references)
- the finger joints carry no drive gains in the USD; they are set by the actuator configuration in
  `contact_prediction_rl/robots/franka_hx5d20.py`
