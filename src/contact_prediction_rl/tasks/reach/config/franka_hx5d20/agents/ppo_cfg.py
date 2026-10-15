# src/contact_prediction_rl/tasks/reach/config/franka_hx5d20/agents/ppo_cfg.py

from isaaclab.utils import configclass

import contact_prediction_rl.learning.configs.ppo as CONFIG


@configclass
class FrankaHX5D20ReachPPOCfg:
    """PPO configuration for the Franka + HX5-D20 reach task"""

    # Runner
    num_steps_per_env = CONFIG.NUM_STEPS_PER_ENV
    max_iterations = 1000
    save_interval = CONFIG.SAVE_INTERVAL
    experiment_name = "reach_franka_hx5d20"

    # Actor
    # The encoder latent layer supplies the second 64-unit ELU layer.
    actor = {
        "hidden_dims": [64],
        "latent_dim": 64,
        "activation": CONFIG.ACTOR_ACTIVATION,
        "observation_normalisation": False,
        "initial_std": 1.0,
    }

    # Critic
    critic = {
        "hidden_dims": [64, 64],
        "activation": CONFIG.CRITIC_ACTIVATION,
        "observation_normalisation": False,
    }

    # PPO
    algorithm = {
        "value_loss_coef": CONFIG.PPO_VALUE_LOSS_COEF,
        "use_clipped_value_loss": CONFIG.PPO_USE_CLIPPED_VALUE_LOSS,
        "clip_param": CONFIG.PPO_CLIP_PARAM,
        "entropy_coef": 0.001,
        "num_learning_epochs": 8,
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
