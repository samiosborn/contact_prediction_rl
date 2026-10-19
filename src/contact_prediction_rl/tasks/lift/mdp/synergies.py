# src/contact_prediction_rl/tasks/lift/mdp/synergies.py

"""Closure dynamics of the hand synergies, in plain torch (no simulator needed)."""

from __future__ import annotations

import torch


def target_closures(actions: torch.Tensor) -> torch.Tensor:
    """Closure each synergy is commanded to reach, in [0, 1].

    A negative action closes the synergy by its magnitude and a non-negative action opens it, which generalises the
    binary hand action (negative closes, otherwise open): an action of -1 commands full closure and 0 or more none.
    """
    return (-actions).clamp(0.0, 1.0)


def step_closures(closures: torch.Tensor, actions: torch.Tensor, max_change: float) -> torch.Tensor:
    """Move the closures toward their commanded values by at most ``max_change`` each.

    Limiting the rate keeps all joints moving in proportion: a fast close of the whole hand pushes a tall block over
    before the fingers surround it.
    """
    return closures + (target_closures(actions) - closures).clamp(-max_change, max_change)


def synergy_posture(open_posture: torch.Tensor, shapes: torch.Tensor, closures: torch.Tensor) -> torch.Tensor:
    """Joint targets ``open + sum_i closure_i * shape_i``.

    Args:
        open_posture: Joint values of the open hand, shape (num_joints,).
        shapes: Joint displacement of each synergy at closure 1, shape (num_synergies, num_joints).
        closures: Closure of each synergy, shape (num_envs, num_synergies).

    Returns:
        Joint targets, shape (num_envs, num_joints).
    """
    return open_posture + closures @ shapes
