# src/contact_prediction_rl/tasks/reorientation/config/hx5_d20_right/agents/rsl_rl_ppo_cfg.py

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

import contact_prediction_rl.learning.config as CONFIG


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 100
    experiment_name = "hx5_d20_reorientation"

    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    actor = RslRlMLPModelCfg(
        hidden_dims=[
            *CONFIG.ACTOR_ENCODER_HIDDEN_DIMS,
            CONFIG.ACTOR_LATENT_DIM,
        ],
        activation=CONFIG.ACTIVATION,
        obs_normalization=CONFIG.OBS_NORMALIZATION,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
        ),
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=CONFIG.CRITIC_HIDDEN_DIMS,
        activation=CONFIG.ACTIVATION,
        obs_normalization=CONFIG.OBS_NORMALIZATION,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
