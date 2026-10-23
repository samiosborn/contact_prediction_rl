# src/contact_prediction_rl/tasks/lift/config/franka_hx5d20/agents/ppo_cfg.py

from isaaclab.utils import configclass

import contact_prediction_rl.learning.configs.ppo as CONFIG
import contact_prediction_rl.learning.configs.tactile_rl as TACTILE_RL


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


@configclass
class FrankaHX5D20BlockLiftPriorResidualPPOCfg(FrankaHX5D20BlockLiftPPOCfg):
    """PPO configuration of the prior-residual block-lift task: the shared settings, under its own experiment name"""

    experiment_name = "lift_block_franka_hx5d20_prior_residual"


@configclass
class FrankaHX5D20BlockLiftPriorResidualLowEntropyPPOCfg(FrankaHX5D20BlockLiftAllJointsLowEntropyPPOCfg):
    """Prior-residual task with the lower entropy bonus and the near-zero initial action of the all-joints task.

    Select it with ``--agent ppo_low_entropy_cfg_entry_point``.
    """

    experiment_name = "lift_block_franka_hx5d20_prior_residual_low_entropy"


@configclass
class FrankaHX5D20BlockLiftPaperPPOCfg:
    """PPO configuration with the shared-encoder agent of the tactile-RL paper (Miller et al., NeurIPS 2025).

    One encoder ``o -> 1024 -> 512 -> 256 = z`` (layer normalisation and ELU after every layer) is read by the policy
    (``z -> 128 -> 64 -> actions``, tanh output) and by the value function (``z -> 128 -> 64 -> 1``), so both losses
    train it. The layer sizes are the paper's.

    The PPO settings are those of the low-entropy prior-residual configuration, not the paper's: the paper tuned every
    experiment with its own 20-trial sweep. Using the same settings for every method keeps the comparison between them
    fair, and the first check is that the proprioception-only agent still learns the task as before.
    """

    # Runner
    num_steps_per_env = CONFIG.NUM_STEPS_PER_ENV
    max_iterations = CONFIG.MAX_ITERATIONS
    save_interval = CONFIG.SAVE_INTERVAL
    experiment_name = "lift_block_franka_hx5d20_prior_residual_paper"

    # Agent
    architecture = "shared"

    encoder = {
        "hidden_dims": [1024, 512],
        "latent_dim": 256,
        "activation": "elu",
        "layer_norm": True,
        "observation_normalisation": True,
    }

    actor = {
        "hidden_dims": [128, 64],
        "activation": "elu",
        "initial_std": CONFIG.ACTOR_INIT_STD,
        "output_activation": "tanh",
        "action_head_init_scale": ALL_JOINTS_ACTION_HEAD_INIT_SCALE,
    }

    critic = {
        "hidden_dims": [128, 64],
        "activation": "elu",
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


@configclass
class FrankaHX5D20BlockLiftHistoryPaperPPOCfg(FrankaHX5D20BlockLiftPaperPPOCfg):
    """Paper agent for the history task: PPO(prop), the proprioception-only baseline"""

    experiment_name = "lift_block_franka_hx5d20_prior_residual_history"


@configclass
class FrankaHX5D20BlockLiftTactilePaperPPOCfg(FrankaHX5D20BlockLiftPaperPPOCfg):
    """Paper agent for the tactile task: PPO(prop + tactile), the base of the auxiliary losses"""

    experiment_name = "lift_block_franka_hx5d20_prior_residual_tactile"


TACTILE_DIM = 20
"""Size of the tactile part at the end of the policy observation of the tactile task: 5 fingertips over the last 4 steps
(``OBSERVATION_HISTORY`` x number of fingertips in ``env_cfg.py``, checked by a test)."""


@configclass
class FrankaHX5D20BlockLiftTactileTRPaperPPOCfg(FrankaHX5D20BlockLiftTactilePaperPPOCfg):
    """Paper agent for the tactile task with the tactile reconstruction (TR) auxiliary loss: PPO(prop + tactile) + TR.

    The decoder ``z -> 512 -> 512 -> 20`` has to reproduce the stacked tactile readings of the observation, with a
    binary cross-entropy of positive weight 10. It has its own optimiser. Select it with
    ``--agent ppo_tr_cfg_entry_point``. Everything else is the same as the tactile baseline.
    """

    experiment_name = "lift_block_franka_hx5d20_prior_residual_tactile_tr"

    auxiliary = {
        "objective": "tactile_reconstruction",
        "tactile_dim": TACTILE_DIM,
        "hidden_dims": TACTILE_RL.DECODER_HIDDEN_DIMS,
        "activation": TACTILE_RL.DECODER_ACTIVATION,
        "pos_weight": TACTILE_RL.TR_POS_WEIGHT,
        "loss_weight": TACTILE_RL.AUX_LOSS_WEIGHT,
        "learning_rate": TACTILE_RL.AUX_LEARNING_RATE,
    }


@configclass
class FrankaHX5D20BlockLiftTactileFRPaperPPOCfg(FrankaHX5D20BlockLiftTactilePaperPPOCfg):
    """Paper agent for the tactile task with the full reconstruction (FR) auxiliary loss: PPO(prop + tactile) + FR.

    The decoder ``z -> 512 -> 512 -> observation size`` has to rebuild the whole observation: the stacked contacts (binary
    cross-entropy, positive weight 10) and the normalised proprioception (squared error). It has its own optimiser. Select
    it with ``--agent ppo_fr_cfg_entry_point``. Everything else is the same as the tactile baseline.
    """

    experiment_name = "lift_block_franka_hx5d20_prior_residual_tactile_fr"

    auxiliary = {
        "objective": "full_reconstruction",
        "tactile_dim": TACTILE_DIM,
        "hidden_dims": TACTILE_RL.DECODER_HIDDEN_DIMS,
        "activation": TACTILE_RL.DECODER_ACTIVATION,
        "pos_weight": TACTILE_RL.TR_POS_WEIGHT,
        "loss_weight": TACTILE_RL.FR_AUX_LOSS_WEIGHT,
        "learning_rate": TACTILE_RL.FR_AUX_LEARNING_RATE,
    }


# Agents of the grasp-start tasks: the same as their counterparts, logged under their own experiment names
@configclass
class FrankaHX5D20GraspStartHistoryPaperPPOCfg(FrankaHX5D20BlockLiftHistoryPaperPPOCfg):
    """PPO(prop) on the grasp-start task"""

    experiment_name = "lift_block_franka_hx5d20_grasp_start_history"


@configclass
class FrankaHX5D20GraspStartTactilePaperPPOCfg(FrankaHX5D20BlockLiftTactilePaperPPOCfg):
    """PPO(prop + tactile) on the grasp-start task"""

    experiment_name = "lift_block_franka_hx5d20_grasp_start_tactile"


@configclass
class FrankaHX5D20GraspStartTactileTRPaperPPOCfg(FrankaHX5D20BlockLiftTactileTRPaperPPOCfg):
    """PPO(prop + tactile) + TR on the grasp-start task"""

    experiment_name = "lift_block_franka_hx5d20_grasp_start_tactile_tr"


@configclass
class FrankaHX5D20GraspStartTactileFRPaperPPOCfg(FrankaHX5D20BlockLiftTactileFRPaperPPOCfg):
    """PPO(prop + tactile) + FR on the grasp-start task"""

    experiment_name = "lift_block_franka_hx5d20_grasp_start_tactile_fr"
