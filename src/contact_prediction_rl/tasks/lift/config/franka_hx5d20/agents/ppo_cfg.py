# src/contact_prediction_rl/tasks/lift/config/franka_hx5d20/agents/ppo_cfg.py

from isaaclab.utils import configclass

import contact_prediction_rl.learning.configs.ppo as CONFIG


@configclass
class FrankaHX5D20BlockLiftPPOCfg:
    """PPO configuration for the Franka + HX5-D20 block-lift task"""

    # Runner
    num_steps_per_env = CONFIG.NUM_STEPS_PER_ENV
    max_iterations = CONFIG.MAX_ITERATIONS
    save_interval = CONFIG.SAVE_INTERVAL
    experiment_name = "lift_block_franka_hx5d20"

    # Actor
    actor = {
        "hidden_dims": CONFIG.ACTOR_ENCODER_HIDDEN_DIMS,
        "latent_dim": CONFIG.ACTOR_LATENT_DIM,
        "activation": CONFIG.ACTOR_ACTIVATION,
        "observation_normalisation": CONFIG.ACTOR_OBS_NORMALIZATION,
        "initial_std": CONFIG.ACTOR_INIT_STD,
    }

    # Critic
    critic = {
        "hidden_dims": CONFIG.CRITIC_HIDDEN_DIMS,
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
        "learning_rate": CONFIG.PPO_LEARNING_RATE,
        "optimizer": CONFIG.PPO_OPTIMIZER,
        "schedule": CONFIG.PPO_SCHEDULE,
        "gamma": CONFIG.PPO_GAMMA,
        "lam": CONFIG.PPO_LAM,
        "desired_kl": CONFIG.PPO_DESIRED_KL,
        "max_grad_norm": CONFIG.PPO_MAX_GRAD_NORM,
        "normalize_advantage_per_mini_batch": (
            CONFIG.PPO_NORMALIZE_ADVANTAGE_PER_MINI_BATCH
        ),
    }


@configclass
class FrankaHX5D20BlockLiftSynergyPPOCfg(FrankaHX5D20BlockLiftPPOCfg):
    """PPO configuration of the synergy block-lift task: the same settings, logged under its own experiment name"""

    experiment_name = "lift_block_franka_hx5d20_synergy"


@configclass
class FrankaHX5D20BlockLiftCurlPPOCfg(FrankaHX5D20BlockLiftPPOCfg):
    """PPO configuration of the curl-only block-lift task: the same settings, logged under its own experiment name"""

    experiment_name = "lift_block_franka_hx5d20_curl"
