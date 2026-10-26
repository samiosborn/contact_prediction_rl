# src/contact_prediction_rl/tasks/reorientation/config/hx5_d20_right/__init__.py

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


# Register independently of the runtime grasp snapshot. ReorientationTrainEnvCfg
# validates the snapshot when it is actually constructed; discovery must not
# depend on which environment variables were set at module import time.
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
