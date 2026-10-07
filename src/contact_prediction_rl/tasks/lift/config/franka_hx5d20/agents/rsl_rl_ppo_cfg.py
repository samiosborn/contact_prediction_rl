# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.lift.config.franka.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg


@configclass
class FrankaHX5D20BlockLiftPPORunnerCfg(LiftCubePPORunnerCfg):
    """PPO settings of the Franka cube-lift task, adjusted for the slow hand of this task.

    * Initial action std 0.2: the side grasp mostly fails under the arm noise a std of 0.5 gives (about 0.1 rad), so a
      noisier policy rarely sees a lift to learn from.
    * Entropy bonus 0.005: with 0.001 the std fell to about 0.03 within 400 iterations, and a policy holding the block
      on the table stopped exploring the lift.
    * Discount factor 0.99 (a horizon of about 100 steps, 2 s), enough to connect the 2 s hand closing to the lift.
      With 0.995 a run reached 64 % of episodes ending with the block at the goal and then collapsed, with the value
      loss swinging between 3 and 30.
    * Observation normalisation for the actor and the critic, to steady the value estimates.
    """

    experiment_name = "lift_block_franka_hx5d20"

    def __post_init__(self):
        super().__post_init__()
        self.actor.distribution_cfg.init_std = 0.2
        self.algorithm.entropy_coef = 0.005
        self.algorithm.gamma = 0.99
        self.actor.obs_normalization = True
        self.critic.obs_normalization = True
