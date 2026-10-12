# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika Panda arm with the Robotis HX5-D20 right hand.

The following configurations are available:

* :obj:`FRANKA_HX5D20_CFG`: Franka Panda arm (7 DoF) with the 20-DoF Robotis HX5-D20 right hand
* :obj:`FRANKA_HX5D20_HAND_OPEN` / :obj:`FRANKA_HX5D20_HAND_CLOSED`: open and closed hand postures
* :obj:`FRANKA_HX5D20_GRASP_CENTER`: where an object sits in the closed hand
* :obj:`FRANKA_HX5D20_HAND_SYNERGIES`: hand-designed finger postures the policy can blend (curl, thumb, spread)

The USD is stored in ``assets/data/franka_hx5d20`` (see its ``SOURCE.md``). The hand is attached to ``panda_hand``
with a fixed joint, and the whole robot is a single articulation rooted at ``/World``.

Reference: https://github.com/ROBOTIS-GIT/robotis_hand
"""

import re
from collections.abc import Sequence
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

FRANKA_HX5D20_USD_PATH = Path(__file__).resolve().parents[1] / "assets" / "data" / "franka_hx5d20" / "franka_hx5d20.usd"
"""Path of the combined Franka + HX5-D20 USD."""

##
# Configuration
##

FRANKA_HX5D20_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(FRANKA_HX5D20_USD_PATH),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "finger_r_joint.*": 0.0,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            joint_effort_limit=87.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            joint_effort_limit=12.0,
            stiffness=400.0,
            damping=80.0,
        ),
        # The finger gains are untuned starting values; the USD does not author drives for the hand joints.
        "hx5_d20_fingers": ImplicitActuatorCfg(
            joint_names_expr=["finger_r_joint.*"],
            joint_effort_limit=10.0,
            stiffness=20.0,
            damping=2.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the Franka Emika Panda arm with the Robotis HX5-D20 right hand.

The arm gains match the drives authored in the USD (equivalent to
Isaac Lab's ``FRANKA_PANDA_HIGH_PD_CFG``).
"""


FRANKA_HX5D20_HAND_OPEN = {"finger_r_joint.*": 0.0}
"""Open hand posture [rad]: all finger joints at zero. Keys are joint name regular expressions."""

FRANKA_HX5D20_HAND_CLOSED = {
    # thumb opposition (joint 1 at its limit) and flexion
    "finger_r_joint1": 1.571,
    "finger_r_joint2": -0.524,
    "finger_r_joint[34]": 0.785,
    # finger abduction stays neutral; flexion is 75 % of the joint range
    "finger_r_joint(5|9|13|17)": 0.0,
    "finger_r_joint(6|10|14|18)": 1.505,
    "finger_r_joint(7|8|11|12|15|16|19|20)": 1.178,
}
"""Closed hand posture [rad] that squeezes a cube. Keys are joint name regular expressions.

Fitted with forward kinematics of the hand's joint frames (thumb opposition and finger flexion) and checked with
the scripted grasp search ``scripts/franka_hx5d20/search_franka_hx5d20_table_grasp.py``.
"""

FRANKA_HX5D20_GRASP_CENTER = (0.05, 0.0, 0.10)
"""Position [m] in the ``panda_hand`` frame where a 6 cm cube is held by the closed hand.

The fingers extend along the frame's +z axis and curl toward +x, so the palm faces +x.
"""


FRANKA_HX5D20_HAND_SYNERGIES = {
    "curl": {
        # finger flexion at 75 % of the joint range, as in the closed posture
        "finger_r_joint(6|10|14|18)": 1.505,
        "finger_r_joint(7|8|11|12|15|16|19|20)": 1.178,
    },
    "thumb": {
        # thumb opposition (joint 1 at its limit) and flexion, as in the closed posture
        "finger_r_joint1": 1.571,
        "finger_r_joint2": -0.524,
        "finger_r_joint[34]": 0.785,
    },
    "spread": {
        # fan the fingers apart: the outer fingers move twice as far as the inner ones. The fingers lie side by side
        # along the hand's y axis (index at +y), and a positive abduction angle moves a fingertip toward -y, so the
        # index and middle fingers take negative angles and the ring and little fingers positive ones. At full
        # spread the index and little fingertips are about 4.7 cm further apart.
        "finger_r_joint5": -0.4,
        "finger_r_joint9": -0.2,
        "finger_r_joint13": 0.2,
        "finger_r_joint17": 0.4,
    },
}
"""Hand synergies: joint postures [rad], each a displacement from the open hand for a closure of 1.

Keys of the inner dictionaries are joint name regular expressions. The hand target is
``open + sum_i closure_i * synergy_i`` with every closure in [0, 1]. Curl and thumb together at full closure give
:obj:`FRANKA_HX5D20_HAND_CLOSED`, the posture of the single-closure lift task, with the fingers parallel (no spread).
"""


def resolve_joint_posture(posture: dict[str, float], joint_names: Sequence[str]) -> list[float]:
    """Joint values [rad] of ``posture`` (regular expression keys) for ``joint_names``; unmatched joints are 0."""
    values = [0.0] * len(joint_names)
    for pattern, value in posture.items():
        for j, name in enumerate(joint_names):
            if re.fullmatch(pattern, name):
                values[j] = value
    return values
