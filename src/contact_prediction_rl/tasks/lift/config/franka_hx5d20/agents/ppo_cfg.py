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
        "normalize_advantage_per_mini_batch": (CONFIG.PPO_NORMALIZE_ADVANTAGE_PER_MINI_BATCH),
    }


@configclass
class FrankaHX5D20BlockLiftSynergyPPOCfg(FrankaHX5D20BlockLiftPPOCfg):
    """PPO configuration of the synergy block-lift task: the same settings, logged under its own experiment name"""

    experiment_name = "lift_block_franka_hx5d20_synergy"


@configclass
class FrankaHX5D20BlockLiftCurlPPOCfg(FrankaHX5D20BlockLiftPPOCfg):
    """PPO configuration of the curl-only block-lift task: the same settings, logged under its own experiment name"""

    experiment_name = "lift_block_franka_hx5d20_curl"


@configclass
class FrankaHX5D20BlockLiftAllJointsPPOCfg(FrankaHX5D20BlockLiftPPOCfg):
    """PPO configuration of the all-joints block-lift task: the same settings, logged under its own experiment name"""

    experiment_name = "lift_block_franka_hx5d20_all_joints"


ALL_JOINTS_ENTROPY_COEF = 0.002
"""Entropy coefficient of the low-entropy all-joints configuration (0.005 in the shared settings).

With 20 free finger joints, the action std rose steadily from 0.2 to about 0.4 over 600 iterations, and the policy
that had learned to lift degraded after iteration 400. The entropy bonus acts on every action dimension and nothing
opposes it on the joints that do not matter for the grasp. A value of 0.001 collapsed the std of the single-closure
hand in earlier runs, so this is between the two.
"""

ALL_JOINTS_ACTION_HEAD_INIT_SCALE = 0.01
"""Scale of the initial weights of the actor's output layer (and a zero bias).

The hand action integrates its input into joint targets, so an untrained policy that outputs a constant non-zero action
drives the joints to their limits within an episode. Small initial outputs start the hand open and still.
"""


@configclass
class FrankaHX5D20BlockLiftAllJointsLowEntropyPPOCfg(FrankaHX5D20BlockLiftAllJointsPPOCfg):
    """PPO settings of the all-joints task with a lower entropy bonus and a near-zero initial action.

    Select it with ``--agent ppo_low_entropy_cfg_entry_point``. Only these two settings differ from
    :class:`FrankaHX5D20BlockLiftAllJointsPPOCfg`.
    """

    experiment_name = "lift_block_franka_hx5d20_all_joints_low_entropy"

    # Actor
    actor = {
        "hidden_dims": CONFIG.ACTOR_ENCODER_HIDDEN_DIMS,
        "latent_dim": CONFIG.ACTOR_LATENT_DIM,
        "activation": CONFIG.ACTOR_ACTIVATION,
        "observation_normalisation": CONFIG.ACTOR_OBS_NORMALIZATION,
        "initial_std": CONFIG.ACTOR_INIT_STD,
        "action_head_init_scale": ALL_JOINTS_ACTION_HEAD_INIT_SCALE,
    }

    # PPO
    algorithm = {
        "value_loss_coef": CONFIG.PPO_VALUE_LOSS_COEF,
        "use_clipped_value_loss": CONFIG.PPO_USE_CLIPPED_VALUE_LOSS,
        "clip_param": CONFIG.PPO_CLIP_PARAM,
        "entropy_coef": ALL_JOINTS_ENTROPY_COEF,
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
