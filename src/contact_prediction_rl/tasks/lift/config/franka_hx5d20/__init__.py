# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka + Robotis HX5-D20 hand: lift a 6x6x12 cm block from a table and hold it 10 cm higher.

Train with (the task uses PhysX, which needs the ``isaacsim`` extra)::

    uv run --extra isaacsim python scripts/train.py --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FrankaHX5D20BlockLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaHX5D20BlockLiftPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
