# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka + Robotis HX5-D20 hand: reach random end-effector poses with the hand held open.

Train with (the task uses PhysX, which needs the ``isaacsim`` extra)::

    uv run --extra isaacsim python scripts/train_ppo.py --task ContactPredictionRl-Reach-Franka-HX5-D20-Right
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="ContactPredictionRl-Reach-Franka-HX5-D20-Right",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FrankaHX5D20ReachEnvCfg",
        "ppo_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20ReachPPOCfg"
        ),
        "default_agent": "ppo",
    },
)
