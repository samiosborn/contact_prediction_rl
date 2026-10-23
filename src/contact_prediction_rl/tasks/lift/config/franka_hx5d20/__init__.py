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

The ``-AllJoints-`` variant lets the policy control every finger joint, with no designed posture (7 + 20 actions)::

    uv run python scripts/train.py --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-AllJoints-Right

It also has a PPO configuration with a lower entropy bonus and a near-zero initial action, selected with
``--agent ppo_low_entropy_cfg_entry_point``.

The ``-PriorResidual-`` variant keeps the thumb-open curl posture as a prior and lets the policy correct every finger
joint around it within a bounded range (7 + 1 + 20 actions)::

    uv run python scripts/train.py --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-PriorResidual-Right

The ``-PriorResidualHistory-`` and ``-PriorResidualTactile-`` variants follow the tactile-RL paper (Miller et al.,
NeurIPS 2025): the policy observation stacks the last 4 steps, and the agent has one encoder shared by the policy and the
value function. The first is the proprioception-only baseline, the second adds the five binary fingertip contacts to the
policy input.
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
        "ppo_low_entropy_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20BlockLiftAllJointsLowEntropyPPOCfg"
        ),
        "default_agent": "ppo",
    },
)

gym.register(
    id="ContactPredictionRl-Lift-Block-Franka-HX5-D20-PriorResidual-Right",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:"
            "FrankaHX5D20BlockLiftPriorResidualEnvCfg"
        ),
        "ppo_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20BlockLiftPriorResidualPPOCfg"
        ),
        "ppo_low_entropy_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20BlockLiftPriorResidualLowEntropyPPOCfg"
        ),
        "default_agent": "ppo",
    },
)

# Tactile-RL paper set-up: the observation stacks the last 4 steps and the agent has one shared encoder.
gym.register(
    id="ContactPredictionRl-Lift-Block-Franka-HX5-D20-PriorResidualHistory-Right",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:"
            "FrankaHX5D20BlockLiftPriorResidualHistoryEnvCfg"
        ),
        "ppo_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20BlockLiftHistoryPaperPPOCfg"
        ),
        "default_agent": "ppo",
    },
)

gym.register(
    id="ContactPredictionRl-Lift-Block-Franka-HX5-D20-PriorResidualTactile-Right",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:"
            "FrankaHX5D20BlockLiftPriorResidualTactileEnvCfg"
        ),
        "ppo_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20BlockLiftTactilePaperPPOCfg"
        ),
        "ppo_tr_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20BlockLiftTactileTRPaperPPOCfg"
        ),
        "ppo_fr_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20BlockLiftTactileFRPaperPPOCfg"
        ),
        "default_agent": "ppo",
    },
)

# The same two paper-style tasks, with the open hand starting 2 cm from the grasp pose and a narrower closed-hand reward, so
# that contacts appear early in every episode that closes the hand.
gym.register(
    id="ContactPredictionRl-Lift-Block-Franka-HX5-D20-GraspStartHistory-Right",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:"
            "FrankaHX5D20BlockLiftGraspStartHistoryEnvCfg"
        ),
        "ppo_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20GraspStartHistoryPaperPPOCfg"
        ),
        "default_agent": "ppo",
    },
)

gym.register(
    id="ContactPredictionRl-Lift-Block-Franka-HX5-D20-GraspStartTactile-Right",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:"
            "FrankaHX5D20BlockLiftGraspStartTactileEnvCfg"
        ),
        "ppo_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20GraspStartTactilePaperPPOCfg"
        ),
        "ppo_tr_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20GraspStartTactileTRPaperPPOCfg"
        ),
        "ppo_fr_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20GraspStartTactileFRPaperPPOCfg"
        ),
        "ppo_raw_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20GraspStartTactileRawPaperPPOCfg"
        ),
        "ppo_raw_tr_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20GraspStartTactileRawTRPaperPPOCfg"
        ),
        "ppo_raw_fr_cfg_entry_point": (
            f"{agents.__name__}.ppo_cfg:"
            "FrankaHX5D20GraspStartTactileRawFRPaperPPOCfg"
        ),
        "default_agent": "ppo",
    },
)
