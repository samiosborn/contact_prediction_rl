# src/contact_prediction_rl/tasks/lift/mdp/actions_cfg.py

"""Configurations of the rate-limited hand actions.

Kept apart from the implementation in :mod:`.actions` so that building the task configuration does not import the
action implementation, which loads USD (``pxr``) before Isaac Sim has started.
"""

from dataclasses import MISSING

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass


@configclass
class RateLimitedBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Configuration of :class:`RateLimitedBinaryJointPositionAction`."""

    class_type: type | str = "{DIR}.actions:RateLimitedBinaryJointPositionAction"

    closing_time: float = 2.0
    """Time [s] the hand takes to go from fully open to fully closed (and back)."""


@configclass
class SynergyJointPositionActionCfg(ActionTermCfg):
    """Configuration of :class:`SynergyJointPositionAction`."""

    class_type: type | str = "{DIR}.actions:SynergyJointPositionAction"

    joint_names: list[str] = MISSING
    """List of joint names or regular expressions the action is mapped to."""

    open_command_expr: dict[str, float] = MISSING
    """Joint values of the open hand. Keys are joint name regular expressions."""

    synergies: dict[str, dict[str, float]] = MISSING
    """Synergy name to joint displacement [rad] at closure 1 (keys are joint name regular expressions).

    The order is the order of the action dimensions: the first synergy is the one reported as the hand's ``closure``
    to the rewards (the finger curl).
    """

    closing_time: float = 2.0
    """Time [s] a synergy takes to go from fully open to fully closed (and back)."""


@configclass
class PerJointPositionActionCfg(ActionTermCfg):
    """Configuration of :class:`PerJointPositionAction`."""

    class_type: type | str = "{DIR}.actions:PerJointPositionAction"

    joint_names: list[str] = MISSING
    """List of joint names or regular expressions the action is mapped to; one action dimension per joint."""

    open_command_expr: dict[str, float] = MISSING
    """Joint values of the open hand, where the targets start. Keys are joint name regular expressions."""

    closure_shape: dict[str, float] = MISSING
    """Posture (joint displacement [rad] from the open hand) that counts as a closed hand for the rewards.

    Only used to report a ``closure`` fraction: how far the joint targets are along this posture. Keys are joint name
    regular expressions.
    """

    closing_time: float = 2.0
    """Time [s] a joint takes to go through its whole range at the largest action."""


@configclass
class PriorResidualPositionActionCfg(ActionTermCfg):
    """Configuration of :class:`PriorResidualPositionAction`."""

    class_type: type | str = "{DIR}.actions:PriorResidualPositionAction"

    joint_names: list[str] = MISSING
    """List of joint names or regular expressions the action is mapped to."""

    open_command_expr: dict[str, float] = MISSING
    """Joint values of the open hand. Keys are joint name regular expressions."""

    prior_shape: dict[str, float] = MISSING
    """Posture (joint displacement [rad] from the open hand) that the closure action blends in.

    Keys are joint name regular expressions.
    """

    residual_authority: float = 0.5
    """Largest per-joint correction as a fraction of the joint's range (0.5 allows +-50 % of the range)."""

    joint_ranges: dict[str, tuple[float, float]] | None = None
    """Optional allowed range [rad] of individual joints, narrower than their limits (keys are regular expressions).

    Used to keep a joint on the natural side of its range, for example the thumb opposition joint.
    """

    closing_time: float = 2.0
    """Time [s] the closure goes from open to closed, and a joint correction through its whole range."""
