# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the rate-limited hand action.

Kept apart from the implementation in :mod:`.actions` so that building the task configuration does not import the
action implementation, which loads USD (``pxr``) before Isaac Sim has started.
"""

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.utils import configclass


@configclass
class RateLimitedBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Configuration of :class:`RateLimitedBinaryJointPositionAction`."""

    class_type: type | str = "{DIR}.actions:RateLimitedBinaryJointPositionAction"

    closing_time: float = 2.0
    """Time [s] the hand takes to go from fully open to fully closed (and back)."""
