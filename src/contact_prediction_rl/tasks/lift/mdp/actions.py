# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hand action that opens and closes all finger joints together at a limited rate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import RateLimitedBinaryJointPositionActionCfg


class RateLimitedBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary open/close action whose joint targets move between the two postures at a limited rate.

    The command is binary as in :class:`BinaryJointPositionAction`: negative closes the hand, non-negative opens it.
    Instead of jumping to the commanded posture, the joint targets follow ``open + closure * (close - open)``, where
    the closure fraction moves toward 0 (open) or 1 (closed) by at most ``step_dt / closing_time`` per step. All joints
    therefore move in proportion and reach the closed posture at the same time, which a per-joint speed limit does not
    give. A fast close of the whole hand pushes a tall block over before the fingers surround it.
    """

    cfg: RateLimitedBinaryJointPositionActionCfg

    def __init__(self, cfg: RateLimitedBinaryJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._closure = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def closure(self) -> torch.Tensor:
        """Fraction of the way from the open (0) to the closed (1) posture, shape (num_envs, 1)."""
        return self._closure

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        close = ~actions if actions.dtype == torch.bool else actions < 0
        max_change = self._env.step_dt / self.cfg.closing_time
        self._closure += (close.float() - self._closure).clamp(-max_change, max_change)
        self._processed_actions = self._open_command + self._closure * (self._close_command - self._open_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._closure[:] = 0.0
        else:
            self._closure[env_ids] = 0.0
