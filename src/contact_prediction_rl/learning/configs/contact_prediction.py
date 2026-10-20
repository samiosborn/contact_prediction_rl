# src/contact_prediction_rl/learning/configs/contact_prediction.py

"""Contact-prediction head and BCE hyperparameters, independent of PPO."""

HEAD_HIDDEN_DIMS = [64]
HEAD_ACTIVATION = "elu"

# 0.0: baseline PPO with the same instantiated contact head.
# >0.0: PPO + gradients from privileged geometric patch-contact labels.
LOSS_WEIGHT = 0.0

# Multiplicative weight on positive contact bits in BCEWithLogits.
# Set from empirical contact prevalence if class imbalance merits it.
POS_WEIGHT = 1.0
