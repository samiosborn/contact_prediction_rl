# src/contact_prediction_rl/tasks/lift/config/franka_hx5d20/__init__.py

"""Franka + Robotis HX5-D20 hand: lift a 6x6x12 cm block from a table and hold it 10 cm higher.

Train with (the task uses PhysX and Isaac Sim)::

    uv run python scripts/train.py --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right

The ``-Synergy-`` variant lets the policy blend three hand-designed finger shapes (curl, thumb, spread) instead of
one open/close value (7 + 3 actions)::

    uv run python scripts/train.py --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Synergy-Right

The ``-Curl-`` variant is the control of the synergy task: one open/close value, closing the fingers only with the
thumb left open (7 + 1 actions)::

    uv run python scripts/train.py --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Curl-Right
    uv run --extra isaacsim python scripts/train.py --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Curl-Right

The ``-AllJoints-`` variant lets the policy control every finger joint, with no designed posture (7 + 20 actions)::

    uv run --extra isaacsim python scripts/train.py --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-AllJoints-Right
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
        "env_cfg_entry_point": (f"{__name__}.env_cfg:FrankaHX5D20BlockLiftEnvCfg"),
        "ppo_cfg_entry_point": (f"{agents.__name__}.ppo_cfg:FrankaHX5D20BlockLiftPPOCfg"),
        "default_agent": "ppo",
    },
)

gym.register(
    id="ContactPredictionRl-Lift-Block-Franka-HX5-D20-Synergy-Right",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (f"{__name__}.env_cfg:FrankaHX5D20BlockLiftSynergyEnvCfg"),
        "ppo_cfg_entry_point": (f"{agents.__name__}.ppo_cfg:FrankaHX5D20BlockLiftSynergyPPOCfg"),
        "default_agent": "ppo",
    },
)

gym.register(
    id="ContactPredictionRl-Lift-Block-Franka-HX5-D20-Curl-Right",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (f"{__name__}.env_cfg:FrankaHX5D20BlockLiftCurlEnvCfg"),
        "ppo_cfg_entry_point": (f"{agents.__name__}.ppo_cfg:FrankaHX5D20BlockLiftCurlPPOCfg"),
        "default_agent": "ppo",
    },
)

gym.register(
    id="ContactPredictionRl-Lift-Block-Franka-HX5-D20-AllJoints-Right",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:"
            "FrankaHX5D20BlockLiftAllJointsEnvCfg"
        ),
        "ppo_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20BlockLiftAllJointsPPOCfg"
        ),
        "default_agent": "ppo",
    },
)

gym.register(
    id="ContactPredictionRl-Lift-Block-Franka-HX5-D20-AllJoints-Right",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:"
            "FrankaHX5D20BlockLiftAllJointsEnvCfg"
        ),
        "ppo_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20BlockLiftAllJointsPPOCfg"
        ),
        "default_agent": "ppo",
    },
)
