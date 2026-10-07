# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.core.reach.config.franka.agents.rsl_rl_ppo_cfg import FrankaReachPPORunnerCfg


@configclass
class FrankaHX5D20ReachPPORunnerCfg(FrankaReachPPORunnerCfg):
    """PPO settings of the Franka reach task; only the logging name differs."""

    experiment_name = "reach_franka_hx5d20"
