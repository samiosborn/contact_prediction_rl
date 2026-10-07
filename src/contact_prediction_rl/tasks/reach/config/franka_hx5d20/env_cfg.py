# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka + HX5-D20 reach environment."""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.core.reach.reach_env_cfg import ReachEnvCfg

from contact_prediction_rl.robots.franka_hx5d20 import FRANKA_HX5D20_CFG

ARM_JOINTS = ["panda_joint.*"]


@configclass
class FrankaHX5D20ReachEnvCfg(ReachEnvCfg):
    """Reach a random end-effector pose with the arm while the hand stays open.

    The policy only sees and commands the seven arm joints. The 20 finger joints are held at their default
    (open) position by their implicit PD drives. The hand base frame coincides with ``panda_hand``, so the pose
    command and rewards reuse the Franka reach conventions.
    """

    def __post_init__(self):
        super().__post_init__()

        # PhysX only: the base task defaults to Newton, which this robot has not been validated with
        self.sim.physics = PhysxCfg(bounce_threshold_velocity=0.2)

        self.scene.robot = FRANKA_HX5D20_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # actions and observations cover the arm only
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=ARM_JOINTS, scale=0.5, use_default_offset=True
        )
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ARM_JOINTS)
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ARM_JOINTS)

        # rewards and command are evaluated on the hand frame; its z-axis points along the end effector
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.joint_vel.params["asset_cfg"].joint_names = ARM_JOINTS
        self.commands.ee_pose.body_name = "panda_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
