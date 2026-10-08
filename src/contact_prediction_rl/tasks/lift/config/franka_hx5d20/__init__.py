# src/contact_prediction_rl/tasks/lift/config/franka_hx5d20/__init__.py

"""Franka + Robotis HX5-D20 hand: lift a 6x6x12 cm block from a table and hold it 10 cm higher.

Train with (the task uses PhysX, which needs the ``isaacsim`` extra)::

    uv run --extra isaacsim python scripts/train_ppo.py --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right
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
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:"
            "FrankaHX5D20BlockLiftEnvCfg"
        ),
        "ppo_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20BlockLiftPPOCfg"
        ),
        "default_agent": "ppo",
    },
)