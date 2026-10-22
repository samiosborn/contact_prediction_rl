# src/contact_prediction_rl/learning/configs/tactile_rl.py

"""Hyperparameters of the auxiliary losses of the tactile-RL paper (Miller et al., NeurIPS 2025), independent of PPO.

The starting values are the paper's tuned ones for the Bounce task (Table A4, "+TR"), the task with the sparsest
contacts. The paper tuned every task and method separately (20-trial sweeps), so these are starting points.
"""

# Decoder of the reconstruction losses: z -> 512 -> 512 -> output (Appendix F)
DECODER_HIDDEN_DIMS = [512, 512]
DECODER_ACTIVATION = "elu"

# Positive weight of the binary cross-entropy: contacts are sparse, so a missed contact costs 10 times a false alarm
TR_POS_WEIGHT = 10.0

# Learning rate of the optimiser of the auxiliary loss, which is separate from PPO's (lr_aux in Table A4)
AUX_LEARNING_RATE = 5.13e-5

# Weight of the auxiliary loss (c_aux in Table A4). The auxiliary optimiser is Adam, whose step size does not depend on
# the scale of the loss, so this weight matters little for the auxiliary parameters
AUX_LOSS_WEIGHT = 0.00384

# Full reconstruction (FR): the paper's tuned values for Bounce (Table A4, "+FR")
FR_AUX_LEARNING_RATE = 2.77e-5
FR_AUX_LOSS_WEIGHT = 0.05669
