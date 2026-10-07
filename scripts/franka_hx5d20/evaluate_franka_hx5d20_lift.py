# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained policy of the Franka + HX5-D20 cube-lift task with deterministic rollouts.

Every environment runs one full episode with the policy's mean action (no exploration noise) and a fixed seed. The
script reports stricter measures than the ``Metrics/success_rate`` logged in training, which counts an episode as a
success as soon as the cube has passed near the goal once:

* ``held at goal at end``: at the end of the episode the cube is held in the hand and within the goal threshold.
* ``lifted in hand at end``: at the end the cube is held in the hand and above the lift height.
* ``reached goal ever``: the training-style success.
* ``dropped``: the episode ended early because the cube fell off the post.
* times to the first lift and to the first arrival at the goal, regrasps, and the share of the episode spent holding the
  cube at the goal.

The "in hand", "lifted" and "goal" thresholds are read from the reward terms of the environment config.

.. code-block:: bash

    uv run --extra isaacsim python scripts/franka_hx5d20/evaluate_franka_hx5d20_lift.py --checkpoint latest

"""

import argparse
import importlib.metadata as metadata
import math
import os
import sys

import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import (
    CHECKPOINT_SELECTORS,
    add_frontend_args,
    create_isaaclab_env,
    resolve_checkpoint_selector,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

import contact_prediction_rl.tasks  # noqa: F401  (registers this project's tasks)

TASK = "ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right"


def parse_args() -> argparse.Namespace:
    """Parse the evaluation arguments; the remainder is left for Hydra overrides."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=str, default=TASK, help="Name of the task.")
    parser.add_argument("--num_envs", type=int, default=256, help="Number of environments, one episode each.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
    )
    parser.add_argument(
        "--no_obs_normalization",
        action="store_true",
        help="Build the policy without observation normalisation, for checkpoints trained before it was enabled.",
    )
    parser.add_argument("--goal_threshold", type=float, default=None, help="Goal distance [m]; defaults to the reward.")
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    add_frontend_args(parser)
    args_cli, remaining = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + remaining
    return args_cli


def resolve_checkpoint(args_cli: argparse.Namespace, agent_cfg, log_root_path: str) -> str:
    """Resolve ``--checkpoint`` (latest/best, a run directory or a file) as the RSL-RL play script does."""
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="rsl_rl",
            task=args_cli.task,
            checkpoint_pattern=r"model_.*\.pt",
            metadata={"agent": args_cli.agent},
        )
    if args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
        return get_checkpoint_path(
            os.path.dirname(args_cli.checkpoint), os.path.basename(args_cli.checkpoint), agent_cfg.load_checkpoint
        )
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def main() -> None:
    args_cli = parse_args()
    installed_version = metadata.version("rsl-rl-lib")
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)

    lift_params = env_cfg.rewards.lifting_object.params
    goal_params = env_cfg.rewards.object_goal_tracking.params
    lift_height, in_hand_distance = lift_params["minimal_height"], lift_params["max_distance"]
    grasp_offset_z = lift_params.get("grasp_offset_z", 0.0)
    goal_threshold = args_cli.goal_threshold or goal_params["success_threshold"]

    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        if args_cli.no_obs_normalization:
            agent_cfg.actor.obs_normalization = False
            agent_cfg.critic.obs_normalization = False
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        checkpoint = resolve_checkpoint(args_cli, agent_cfg, log_root_path)
        print(f"[INFO] Evaluating {checkpoint} on task {args_cli.task}")

        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        u = env.unwrapped
        n, device = u.num_envs, u.device
        step_dt = u.step_dt

        def snapshot() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Whether the cube is held and lifted, whether it is also at the goal, and its distance [m] to the goal."""
            robot, cube, ee = u.scene["robot"], u.scene["object"], u.scene["ee_frame"]
            cube_pos = cube.data.root_pos_w.torch
            held_part = cube_pos.clone()
            held_part[:, 2] += grasp_offset_z
            hand_distance = torch.linalg.norm(held_part - ee.data.target_pos_w.torch[:, 0, :], dim=1)
            goal_w, _ = combine_frame_transforms(
                robot.data.root_pos_w.torch,
                robot.data.root_quat_w.torch,
                u.command_manager.get_command("object_pose")[:, :3],
            )
            goal_distance = torch.linalg.norm(cube_pos - goal_w, dim=1)
            error = cube_pos - goal_w
            goal_b = u.command_manager.get_command("object_pose")[:, :3]
            lifted = (cube_pos[:, 2] > lift_height) & (hand_distance < in_hand_distance)
            return lifted, lifted & (goal_distance < goal_threshold), goal_distance, error, goal_b

        nan = torch.full((n,), float("nan"), device=device)
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        dropped = torch.zeros_like(finished)
        reached_ever = torch.zeros_like(finished)
        lifted_end = torch.zeros_like(finished)
        held_end = torch.zeros_like(finished)
        prev_lifted = torch.zeros_like(finished)
        closest, final_distance = torch.full((n,), float("inf"), device=device), nan.clone()
        first_lift, first_goal = nan.clone(), nan.clone()
        rises = torch.zeros(n, device=device)
        held_steps = torch.zeros(n, device=device)
        steps = torch.zeros(n, device=device)

        final_error, final_goal = torch.zeros(n, 3, device=device), torch.zeros(n, 3, device=device)
        hand_term = u.action_manager.get_term("gripper_action")
        final_tilt = nan.clone()
        gap_when_closed = nan.clone()
        up_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)

        def tilt_and_gap() -> tuple[torch.Tensor, torch.Tensor]:
            """Tilt [deg] of the object from vertical, and distance [m] from the grasp centre to the held part."""
            cube, ee = u.scene["object"], u.scene["ee_frame"]
            up = math_utils.quat_apply(cube.data.root_quat_w.torch, up_axis)
            tilt = torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))
            held_part = cube.data.root_pos_w.torch.clone()
            held_part[:, 2] += grasp_offset_z
            return tilt, torch.linalg.norm(held_part - ee.data.target_pos_w.torch[:, 0, :], dim=1)

        obs = env.get_observations()
        for step in range(u.max_episode_length + 5):
            with torch.inference_mode():
                lifted, held, goal_distance, error, goal_b = snapshot()
                active = ~finished
                steps += active
                held_steps += (held & active).float()
                rises += (lifted & ~prev_lifted & active).float()
                first_lift = torch.where(active & lifted & first_lift.isnan(), float(step), first_lift)
                first_goal = torch.where(active & held & first_goal.isnan(), float(step), first_goal)
                reached_ever |= held & active
                prev_lifted = lifted
                closest = torch.where(active & lifted, torch.minimum(closest, goal_distance), closest)
                final_distance = torch.where(active, goal_distance, final_distance)
                final_error = torch.where(active[:, None], error, final_error)
                tilt, gap = tilt_and_gap()
                final_tilt = torch.where(active, tilt, final_tilt)
                if hasattr(hand_term, "closure"):
                    closed_now = active & (hand_term.closure[:, 0] > 0.99) & gap_when_closed.isnan()
                    gap_when_closed = torch.where(closed_now, gap, gap_when_closed)
                final_goal = torch.where(active[:, None], goal_b, final_goal)
                obs, _, dones, extras = env.step(policy(obs))
                policy.reset(dones)
                # ``lifted`` and ``held`` describe the state just before this step, i.e. the last state of the episode
                ended = dones.bool() & active
                timed_out = extras["time_outs"].bool() if "time_outs" in extras else torch.zeros_like(finished)
                dropped |= ended & ~timed_out
                lifted_end |= ended & lifted
                held_end |= ended & held
                finished |= ended
            if bool(finished.all()):
                break

        def rate(mask: torch.Tensor) -> str:
            p = mask.float().mean().item()
            return f"{100 * p:5.1f} %  (+/- {100 * math.sqrt(max(p * (1 - p), 1e-12) / n):.1f})"

        def timing(values: torch.Tensor) -> str:
            seen = values[~values.isnan()] * step_dt
            if seen.numel() == 0:
                return "never"
            median, pct90 = seen.median().item(), seen.quantile(0.9).item()
            return f"median {median:.2f} s, 90th pct {pct90:.2f} s  ({seen.numel()} envs)"

        lifted_envs = ~first_lift.isnan()
        print(f"\n=== {args_cli.task}: {n} deterministic episodes, seed {agent_cfg.seed} ===")
        print(f"held at goal at end    {rate(held_end)}")
        print(f"lifted in hand at end  {rate(lifted_end)}")
        print(f"reached goal ever      {rate(reached_ever)}   (training-style success)")
        print(f"dropped (episode ended early) {rate(dropped)}")
        print(f"time to first lift     {timing(first_lift)}")
        print(f"time to first goal     {timing(first_goal)}")
        for name, values in (("closest to goal while held", closest), ("distance to goal at end", final_distance)):
            seen = values[torch.isfinite(values)] * 100
            print(
                f"{name}: median {seen.median().item():.1f} cm, 90th pct {seen.quantile(0.9).item():.1f} cm, "
                f"worst {seen.max().item():.1f} cm"
            )
        print("signed error at end (cube - goal) [cm]:")
        for axis, name in enumerate("xyz"):
            e = final_error[:, axis] * 100
            print(f"  {name}: mean {e.mean().item():+.1f}, std {e.std().item():.1f}")
        # how closely does the final cube position follow the goal? slope 1 = tracks it, 0 = ignores it
        print("slope of final cube position on goal position (1 = tracks, 0 = ignores):")
        for axis, name in enumerate("xyz"):
            goal = final_goal[:, axis] - final_goal[:, axis].mean()
            if goal.std().item() < 1e-4:
                print(f"  {name}: n/a (the goal does not vary along {name})")
                continue
            slope = 1.0 + (goal * final_error[:, axis]).mean() / (goal * goal).mean()
            print(f"  {name}: {slope.item():.2f}   (goal std {goal.std().item() * 100:.1f} cm)")
        # does the error depend on where the goal is?
        for axis, name in enumerate("xyz"):
            goal = final_goal[:, axis]
            if goal.std().item() < 1e-4:
                continue
            low = goal <= goal.median()
            print(
                f"  goal {name} below/above its median ({goal.median().item():.2f} m): distance at end "
                f"{final_distance[low].median().item() * 100:.1f} / {final_distance[~low].median().item() * 100:.1f} cm"
            )
        regrasps = (rises - 1).clamp(min=0)[lifted_envs]
        tilted = final_tilt[lifted_end]
        if tilted.numel():
            print(
                f"tilt of the held object at end: median {tilted.median().item():.0f} deg, 90th pct "
                f"{tilted.quantile(0.9).item():.0f} deg; "
                f"upright (<10 deg) {100 * (tilted < 10).float().mean().item():.0f} %"
            )
        seen = gap_when_closed[~gap_when_closed.isnan()] * 1000
        if seen.numel():
            print(
                f"grasp centre to held part when the hand finished closing: median {seen.median().item():.0f} mm, "
                f"90th pct {seen.quantile(0.9).item():.0f} mm  ({seen.numel()} envs)"
            )
        print(f"regrasps per env that lifted: mean {regrasps.mean().item() if regrasps.numel() else 0.0:.2f}")
        print(
            f"share of the episode holding the cube at the goal: mean {100 * (held_steps / steps).mean().item():.1f} %"
        )
        print(
            f"thresholds: lift above {lift_height:.3f} m, in hand within {in_hand_distance} m, "
            f"goal within {goal_threshold} m"
        )
        env.close()


if __name__ == "__main__":
    main()
