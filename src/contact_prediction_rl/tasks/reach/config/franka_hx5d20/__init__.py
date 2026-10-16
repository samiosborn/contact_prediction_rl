# src/contact_prediction_rl/tasks/reach/config/franka_hx5d20/__init__.py

"""Franka + Robotis HX5-D20 hand: reach random end-effector poses with the hand held open.

Train with (the task uses PhysX, which needs the ``isaacsim`` extra)::

    uv run python scripts/train.py --task ContactPredictionRl-Reach-Franka-HX5-D20-Right
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
        "ppo_cfg_entry_point": (f"{agents.__name__}.ppo_cfg:FrankaHX5D20ReachPPOCfg"),
        "default_agent": "ppo",
    },
)
