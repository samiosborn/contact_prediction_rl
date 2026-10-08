# src/contact_prediction_rl/tasks/lift/config/franka_hx5d20/agents/rsl_rl_ppo_cfg.py

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

import contact_prediction_rl.learning.configs.ppo as CONFIG


@configclass
class FrankaHX5D20BlockLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for the Franka + HX5-D20 block-lift task"""

    class_name = "contact_prediction_rl.learning.runners.on_policy_runner:OnPolicyRunner"

    # Runner
    num_steps_per_env = CONFIG.NUM_STEPS_PER_ENV
    max_iterations = CONFIG.MAX_ITERATIONS
    save_interval = CONFIG.SAVE_INTERVAL
    experiment_name = "lift_block_franka_hx5d20"

    obs_groups = CONFIG.OBS_GROUPS.copy()

    # Actor
    actor = RslRlMLPModelCfg(
        class_name="contact_prediction_rl.learning.models.actor:Actor",
        hidden_dims=[
            *CONFIG.ACTOR_ENCODER_HIDDEN_DIMS,
            CONFIG.ACTOR_LATENT_DIM,
        ],
        activation=CONFIG.ACTOR_ACTIVATION,
        obs_normalization=CONFIG.ACTOR_OBS_NORMALIZATION,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=CONFIG.ACTOR_INIT_STD,
        ),
    )

    # Critic
    critic = RslRlMLPModelCfg(
        hidden_dims=CONFIG.CRITIC_HIDDEN_DIMS,
        activation=CONFIG.CRITIC_ACTIVATION,
        obs_normalization=CONFIG.CRITIC_OBS_NORMALIZATION,
    )

    # PPO
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="contact_prediction_rl.learning.algorithms.ppo:PPO",
        value_loss_coef=CONFIG.PPO_VALUE_LOSS_COEF,
        use_clipped_value_loss=CONFIG.PPO_USE_CLIPPED_VALUE_LOSS,
        clip_param=CONFIG.PPO_CLIP_PARAM,
        entropy_coef=CONFIG.PPO_ENTROPY_COEF,
        num_learning_epochs=CONFIG.PPO_NUM_LEARNING_EPOCHS,
        num_mini_batches=CONFIG.PPO_NUM_MINI_BATCHES,
        learning_rate=CONFIG.PPO_LEARNING_RATE,
        schedule=CONFIG.PPO_SCHEDULE,
        gamma=CONFIG.PPO_GAMMA,
        lam=CONFIG.PPO_LAM,
        desired_kl=CONFIG.PPO_DESIRED_KL,
        max_grad_norm=CONFIG.PPO_MAX_GRAD_NORM,
    )
