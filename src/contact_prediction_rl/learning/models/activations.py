# src/contact_prediction_rl/learning/models/activations.py

import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """Return an activation module from its name"""

    if name == "elu":
        return nn.ELU()

    if name == "relu":
        return nn.ReLU()

    if name == "tanh":
        return nn.Tanh()

    raise ValueError(f"Unsupported activation: {name}")
