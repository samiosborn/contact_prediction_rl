# src/contact_prediction_rl/tasks/reorientation/config/hx5_d20_right/agents/ppo_cfg.py

from isaaclab.utils import configclass

import contact_prediction_rl.learning.configs.ppo as CONFIG


@configclass
class PPORunnerCfg:
    """PPO configuration for the Franka + HX5-D20 reorientation task"""

    # Runner
    num_steps_per_env = CONFIG.NUM_STEPS_PER_ENV
    max_iterations = 1000
    save_interval = 100
    experiment_name = "hx5_d20_reorientation"

    # Actor
    actor = {
        "hidden_dims": [128, 128],
        "latent_dim": 32,
        "activation": CONFIG.ACTOR_ACTIVATION,
        "observation_normalisation": CONFIG.ACTOR_OBS_NORMALIZATION,
        "initial_std": 1.0,
    }

    # Critic
    critic = {
        "hidden_dims": [128, 128, 32],
        "activation": CONFIG.CRITIC_ACTIVATION,
        "observation_normalisation": CONFIG.CRITIC_OBS_NORMALIZATION,
    }

    # PPO
    algorithm = {
        "value_loss_coef": CONFIG.PPO_VALUE_LOSS_COEF,
        "use_clipped_value_loss": CONFIG.PPO_USE_CLIPPED_VALUE_LOSS,
        "clip_param": CONFIG.PPO_CLIP_PARAM,
        "entropy_coef": CONFIG.PPO_ENTROPY_COEF,
        "num_learning_epochs": CONFIG.PPO_NUM_LEARNING_EPOCHS,
        "num_mini_batches": CONFIG.PPO_NUM_MINI_BATCHES,
        "learning_rate": 1.0e-3,
        "optimizer": CONFIG.PPO_OPTIMIZER,
        "schedule": CONFIG.PPO_SCHEDULE,
        "gamma": CONFIG.PPO_GAMMA,
        "lam": CONFIG.PPO_LAM,
        "desired_kl": CONFIG.PPO_DESIRED_KL,
        "max_grad_norm": CONFIG.PPO_MAX_GRAD_NORM,
        "normalize_advantage_per_mini_batch": (CONFIG.PPO_NORMALIZE_ADVANTAGE_PER_MINI_BATCH),
    }
