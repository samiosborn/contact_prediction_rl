"""Read-only episode metrics logged through Isaac Lab's reward-manager lifecycle.

This is a metrics *observer*, not a shaping reward. Its return is identically
zero regardless of cube state. The nonzero config weight ensures the manager
invokes it each step and calls reset() for completed environments; the term
makes no contribution to PPO returns.
"""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase

from contact_prediction_rl.robots.hx5_d20_patches import fingertip_patch_contact
from contact_prediction_rl.tasks.reorientation.diagnostics import ReorientationEpisodeMetrics
from contact_prediction_rl.tasks.reorientation.metrics import ReorientationSuccessThresholds, reorientation_step_metrics


class ReorientationMetricsTerm(ManagerTermBase):
    """Measure sustained success per episode without affecting rewards."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.tracker = ReorientationEpisodeMetrics(
            env.num_envs,
            sustained_steps=int(cfg.params.get("sustained_steps", 10)),
            device=env.device,
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        summary = self.tracker.summarise_completed(env_ids)
        if summary:
            self._env.extras.setdefault("log", {}).update(summary)
        self.tracker.reset(env_ids)

    def __call__(
        self,
        env,
        goal_xyzw: tuple[float, float, float, float],
        start_pos_w: tuple[float, float, float],
        goal_tolerance_deg: float = 5.0,
        max_hold_displacement_m: float = 0.02,
        max_grasp_displacement_m: float = 0.04,
        max_linear_speed_m_s: float = 0.10,
        max_angular_speed_rad_s: float = 0.50,
        sustained_steps: int = 10,
    ) -> torch.Tensor:
        thresholds = ReorientationSuccessThresholds(
            goal_tolerance_deg=goal_tolerance_deg,
            max_hold_displacement_m=max_hold_displacement_m,
            max_grasp_displacement_m=max_grasp_displacement_m,
            max_linear_speed_m_s=max_linear_speed_m_s,
            max_angular_speed_rad_s=max_angular_speed_rad_s,
            sustained_steps=sustained_steps,
        )
        cube = env.scene["object"].data
        pos = cube.root_pos_w.torch
        state = reorientation_step_metrics(
            pos,
            cube.root_quat_w.torch,
            cube.root_lin_vel_w.torch,
            cube.root_ang_vel_w.torch,
            fingertip_patch_contact(env),
            start_pos_w=pos.new_tensor(start_pos_w) + env.scene.env_origins,
            goal_quat_w=pos.new_tensor(goal_xyzw),
            thresholds=thresholds,
        )
        self.tracker.observe(state)
        return torch.zeros_like(state["goal_error_rad"])
