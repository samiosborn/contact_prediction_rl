# src/contact_prediction_rl/tasks/reorientation/config/hx5_d20_right/__init__.py

import os

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="ContactPredictionRl-Reorientation-HX5-D20-Right",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ReorientationEnvCfg",
        "default_agent": "ppo",
        "ppo_cfg_entry_point": f"{agents.__name__}.ppo_cfg:PPORunnerCfg",
    },
)


# The training task is explicitly opt-in so contact-validation fixture tests remain unchanged.
if os.environ.get("REORIENTATION_GRASP_SNAPSHOT"):
    gym.register(
        id="ContactPredictionRl-Reorientation-Train-HX5-D20-Right",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.train_env_cfg:ReorientationTrainEnvCfg",
            "default_agent": "ppo",
            "ppo_cfg_entry_point": f"{__name__}.train_ppo_cfg:ReorientationTrainPPORunnerCfg",
        },
    )
