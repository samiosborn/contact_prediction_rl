# src/contact_prediction_rl/tasks/lift/mdp/actions.py

"""Hand actions that open and close the finger joints at a limited rate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers.action_manager import ActionTerm

from .joint_targets import closure_along_shape, max_target_change, step_joint_targets
from .synergies import step_closures, synergy_posture

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import (
        PerJointPositionActionCfg,
        RateLimitedBinaryJointPositionActionCfg,
        SynergyJointPositionActionCfg,
    )


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


class SynergyJointPositionAction(ActionTerm):
    """Hand action with one continuous value per synergy, each blending in a hand-designed finger shape.

    Every synergy ``i`` has a closure fraction ``c_i`` in [0, 1]. The action ``a_i`` commands a closure of
    ``clamp(-a_i, 0, 1)`` (negative closes, non-negative opens, as in the binary hand action) and ``c_i`` moves toward
    it by at most ``step_dt / closing_time`` per step. The joint targets are ``open + sum_i c_i * shape_i``, so all
    joints of a synergy move in proportion and reach their closed posture together.

    With one synergy and actions of -1 or 1 this reproduces :class:`RateLimitedBinaryJointPositionAction`.
    """

    cfg: SynergyJointPositionActionCfg

    def __init__(self, cfg: SynergyJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names, as_proxy=True)
        self._num_joints = len(joint_ids)
        self._joint_ids = joint_ids.warp

        self._synergy_names = list(self.cfg.synergies)
        self._open_command = self._resolve_posture(self.cfg.open_command_expr)
        self._shapes = torch.stack([self._resolve_posture(p) for p in self.cfg.synergies.values()])

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = self._open_command.repeat(self.num_envs, 1)
        self._closure = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        if self.cfg.clip is not None:
            if not isinstance(self.cfg.clip, dict):
                raise ValueError(f"Unsupported clip type: {type(self.cfg.clip)}. Supported types are dict.")
            self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(self.num_envs, self._num_joints, 1)
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
            self._clip[:, index_list] = torch.tensor(value_list, device=self.device)

    def _resolve_posture(self, posture: dict[str, float]) -> torch.Tensor:
        """Joint values of a posture given by regular expressions, with 0 for the joints it does not name."""
        values = torch.zeros(self._num_joints, device=self.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(posture, self._joint_names)
        values[index_list] = torch.tensor(value_list, device=self.device)
        return values

    @property
    def action_dim(self) -> int:
        return len(self._synergy_names)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def closure(self) -> torch.Tensor:
        """Closure of each synergy in [0, 1], shape (num_envs, num_synergies), in the order of the config."""
        return self._closure

    @property
    def synergy_names(self) -> list[str]:
        return self._synergy_names

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        max_change = self._env.step_dt / self.cfg.closing_time
        self._closure = step_closures(self._closure, actions, max_change)
        self._processed_actions = synergy_posture(self._open_command, self._shapes, self._closure)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._closure[env_ids] = 0.0


class PerJointPositionAction(ActionTerm):
    """Hand action with one continuous value per joint, integrated into rate-limited position targets.

    The joint targets start at the open posture. At every step each target moves by ``action * max_change`` and is
    clipped to the joint limits, where ``max_change`` is the joint's whole range divided by ``closing_time`` per
    second. An action of 0 holds the posture and a positive action raises the joint angle, so the policy builds the
    hand shape itself, joint by joint, at the same speed limit as the single-closure hand.

    The ``closure`` property reports how far the targets are along ``closure_shape``, so that rewards written for the
    single-closure hand work unchanged.
    """

    cfg: PerJointPositionActionCfg

    def __init__(self, cfg: PerJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names, as_proxy=True)
        self._num_joints = len(joint_ids)
        self._joint_ids = joint_ids.warp

        self._open_command = self._resolve_posture(self.cfg.open_command_expr)
        self._closure_shape = self._resolve_posture(self.cfg.closure_shape)

        limits = self._asset.data.joint_pos_limits.torch[0, joint_ids.torch].to(self.device)
        self._lower, self._upper = limits[:, 0].clone(), limits[:, 1].clone()
        self._max_change = max_target_change(self._lower, self._upper, env.step_dt, self.cfg.closing_time)

        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._joint_targets = self._open_command.repeat(self.num_envs, 1)
        self._processed_actions = self._joint_targets.clone()

    def _resolve_posture(self, posture: dict[str, float]) -> torch.Tensor:
        """Joint values of a posture given by regular expressions, with 0 for the joints it does not name."""
        values = torch.zeros(self._num_joints, device=self.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(posture, self._joint_names)
        values[index_list] = torch.tensor(value_list, device=self.device)
        return values

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def joint_names(self) -> list[str]:
        """Names of the controlled joints, in the order of the action dimensions."""
        return self._joint_names

    @property
    def joint_targets(self) -> torch.Tensor:
        """Current joint position targets [rad], shape (num_envs, num_joints)."""
        return self._joint_targets

    @property
    def closure(self) -> torch.Tensor:
        """Fraction of ``closure_shape`` the targets have reached, shape (num_envs, 1)."""
        return closure_along_shape(self._joint_targets, self._closure_shape)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._joint_targets = step_joint_targets(
            self._joint_targets, actions, self._max_change, self._lower, self._upper
        )
        self._processed_actions = self._joint_targets

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._joint_targets[env_ids] = self._open_command
