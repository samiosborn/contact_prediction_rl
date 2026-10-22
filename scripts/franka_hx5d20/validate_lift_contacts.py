# scripts/franka_hx5d20/validate_lift_contacts.py

"""Check that the fingertip contact sensors respond to real grasps on a Lift task (diagnostic, no training).

A trained policy that holds the block is replayed. The five fingertip contact sensors of
``robots/hx5_d20_contacts.py`` are added to the scene *in memory only*, and the contact labels are compared with
what the hand physically does: how closed the hand is, and whether the block has left the table.

Nothing in the task or in the policy input changes, so existing checkpoints load as they are.

What a working sensor should show:

* labels near 0 while the hand is open and the block rests on the table;
* labels turning on for the fingertips that touch the block when the hand closes and lifts it;
* the contact starting at about the time the hand closes, before or with the lift;
* labels near 0 for a policy that never grasps (use one as a negative control).

.. code-block:: bash

    uv run python scripts/franka_hx5d20/validate_lift_contacts.py \
      --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-PriorResidual-Right \
      --agent ppo_low_entropy_cfg_entry_point \
      --checkpoint logs/ppo/<experiment>/<run>/model_599.pt
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import gymnasium as gym
import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import contact_prediction_rl.tasks  # noqa: F401
from contact_prediction_rl.learning.checkpoints import get_log_root_path, resolve_checkpoint
from contact_prediction_rl.learning.inference import load_policy
from contact_prediction_rl.learning.isaaclab_env import IsaacLabPPOEnvWrapper
from contact_prediction_rl.robots.hx5_d20_contacts import (
    CONTACT_FORCE_THRESHOLD_N,
    FINGERTIP_NAMES,
    FINGERTIP_SENSOR_NAMES,
    fingertip_contact_sensor_cfg,
    fingertip_object_contact,
)

# On PhysX the total filtered force is not available, so ``force_matrix_w`` returns the normal force and warns at every
# call. The normal force is what the 0.1 N threshold needs, so the warning only fills the log.
warnings.filterwarnings("ignore", message="PhysX does not return a total contact force")


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=str, required=True, help="Lift task name.")
    parser.add_argument(
        "--agent", type=str, default="ppo_cfg_entry_point", help="Name of the PPO agent configuration entry point."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file, run directory, or 'latest'.")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments, one episode each.")
    parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
    parser.add_argument(
        "--lift_margin",
        type=float,
        default=0.02,
        help="Height [m] above its resting height at which the block counts as lifted.",
    )
    parser.add_argument("--bins", type=int, default=8, help="Number of time bins in the timeline table.")
    add_launcher_args(parser)
    return setup_preset_cli(parser, argv)


def _fingertip_force_magnitudes(env) -> torch.Tensor:
    """Normal-force magnitude [N] between each fingertip and the block, shape (num_envs, 5)."""
    magnitudes = []
    for name in FINGERTIP_SENSOR_NAMES:
        matrix = env.scene[name].data.normal_force_matrix_w
        forces = matrix.torch if hasattr(matrix, "torch") else matrix
        magnitudes.append(torch.linalg.vector_norm(forces[:, 0, 0, :], dim=-1))
    return torch.stack(magnitudes, dim=-1)


def _first_true(mask: torch.Tensor) -> torch.Tensor:
    """Index of the first True along the time axis (dim 0) per environment, -1 where there is none."""
    first = torch.argmax(mask.int(), dim=0)
    return torch.where(mask.any(dim=0), first, torch.full_like(first, -1))


def _median_seconds(first: torch.Tensor, step_dt: float) -> str:
    seen = first[first >= 0].float()
    if seen.numel() == 0:
        return "never"
    return f"{seen.median().item() * step_dt:.2f} s  ({seen.numel()}/{first.numel()} envs)"


def _report(
    labels: torch.Tensor,
    forces: torch.Tensor,
    height: torch.Tensor,
    closure: torch.Tensor,
    valid: torch.Tensor,
    args_cli: argparse.Namespace,
    step_dt: float,
    checkpoint_name: str,
) -> None:
    """Print how the contact labels relate to the physical grasp. All inputs are CPU tensors (steps, envs, ...)."""
    steps, n = valid.shape

    # ---------------------------------------------------------------------------------------------------------
    # analysis (plain tensors on the CPU)
    # ---------------------------------------------------------------------------------------------------------

    lifted = (height > args_cli.lift_margin) & valid
    resting = (~lifted) & valid
    n_contacts = labels.sum(dim=-1)
    mismatch = int(((forces > CONTACT_FORCE_THRESHOLD_N).float() != labels)[valid.unsqueeze(-1).expand_as(labels)].sum())

    print(f"\n=== Fingertip contacts: {n} episodes of {steps} steps ({steps * step_dt:.0f} s), {checkpoint_name} ===")
    print(f"valid steps: {int(valid.sum())}   lifted steps (block > {args_cli.lift_margin * 100:.0f} cm up): {int(lifted.sum())}")
    print(f"label vs force-threshold ({CONTACT_FORCE_THRESHOLD_N} N) mismatches: {mismatch}  (expected 0)")

    header = f"{'':<28}" + "".join(f"{name:>9}" for name in FINGERTIP_NAMES) + f"{'any':>9}{'mean #':>9}"

    def row(title: str, mask: torch.Tensor) -> None:
        if not mask.any():
            print(f"{title:<28}   (no steps)")
            return
        rates = labels[mask].mean(dim=0)
        any_rate = (n_contacts[mask] > 0).float().mean()
        print(
            f"{title:<28}" + "".join(f"{100 * r:>8.1f}%" for r in rates) + f"{100 * any_rate:>8.1f}%"
            f"{n_contacts[mask].mean():>9.2f}"
        )

    print("\n--- Contact rate by situation (share of steps with the fingertip in contact) ---")
    print(header)
    row("block lifted", lifted)
    row("block on the table", resting)
    row("hand open (closure < 0.05)", valid & (closure < 0.05))
    row("hand closed (closure > 0.5)", valid & (closure > 0.5))
    row("all valid steps", valid)

    print("\n--- Timeline (mean over episodes) ---")
    print(f"{'time':<14}{'closure':>9}{'lifted':>9}" + "".join(f"{name:>9}" for name in FINGERTIP_NAMES) + f"{'mean #':>9}")
    edges = torch.linspace(0, steps, args_cli.bins + 1).long()
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = valid[lo:hi]
        if not v.any():
            continue
        rates = labels[lo:hi][v].mean(dim=0)
        print(
            f"{lo * step_dt:4.1f}-{hi * step_dt:4.1f} s  "
            f"{closure[lo:hi][v].mean():>9.2f}{lifted[lo:hi][v].float().mean():>9.2f}"
            + "".join(f"{100 * r:>8.0f}%" for r in rates)
            + f"{n_contacts[lo:hi][v].mean():>9.2f}"
        )

    print("\n--- Timing of events (median over episodes) ---")
    print(f"hand closes (closure > 0.5)   {_median_seconds(_first_true((closure > 0.5) & valid), step_dt)}")
    print(f"first fingertip contact       {_median_seconds(_first_true((n_contacts > 0) & valid), step_dt)}")
    print(f"block lifted                  {_median_seconds(_first_true(lifted), step_dt)}")

    if lifted.any():
        lifted_no_contact = ((n_contacts == 0) & lifted).sum().float() / lifted.sum()
        print(
            f"\nlifted steps with NO fingertip contact: {100 * lifted_no_contact:.1f}%  "
            "(the block may be held by the palm or the finger bases instead)"
        )

        in_contact = forces[lifted.unsqueeze(-1).expand_as(forces) & (forces > 0)]
        if in_contact.numel() > 0:
            print(
                f"force on the fingertips while lifted [N]: median {in_contact.median().item():.2f}, "
                f"90th percentile {torch.quantile(in_contact, 0.9).item():.2f}"
            )
        else:
            print("no fingertip force above 0 while the block was lifted")
    else:
        print("\nThe block was never lifted in this run (negative control), so only the 'table' rows are informative.")


def main() -> None:
    args_cli, hydra_args = _parse_args()

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, overrides=hydra_args, play_mode=True)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device

    # Add the five fingertip sensors to the scene and make PhysX report contacts on the robot's bodies. Both are only
    # in this script: no observation uses the sensors, so the policy input is unchanged.
    for finger, sensor_name in zip(FINGERTIP_NAMES, FINGERTIP_SENSOR_NAMES):
        setattr(env_cfg.scene, sensor_name, fingertip_contact_sensor_cfg(finger))
    env_cfg.scene.robot.spawn.activate_contact_sensors = True

    checkpoint = resolve_checkpoint(args_cli.checkpoint, get_log_root_path(agent_cfg.experiment_name))
    print(f"[INFO] Checking contacts with {checkpoint} on task {args_cli.task}")

    with launch_simulation(env_cfg, args_cli):
        env = IsaacLabPPOEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
        policy = load_policy(env, agent_cfg, str(checkpoint))

        u = env.unwrapped
        n, steps, step_dt = u.num_envs, u.max_episode_length, u.step_dt
        hand = u.action_manager.get_term("gripper_action")
        block = u.scene["object"]

        labels = torch.zeros(steps, n, 5)
        forces = torch.zeros(steps, n, 5)
        height = torch.zeros(steps, n)
        closure = torch.full((steps, n), float("nan"))
        valid = torch.zeros(steps, n, dtype=torch.bool)

        with torch.inference_mode():
            obs, _ = env.reset()
            rest_z = block.data.root_pos_w.torch[:, 2].clone().cpu()
            finished = torch.zeros(n, dtype=torch.bool, device=u.device)

            for t in range(steps):
                obs, _, dones, _ = env.step(policy(obs["policy"]))

                # an environment that finished at this step has already been reset, so its readings are not valid
                valid[t] = (~(finished | dones.bool())).cpu()
                finished |= dones.bool()

                labels[t] = fingertip_object_contact(u).cpu()
                forces[t] = _fingertip_force_magnitudes(u).cpu()
                height[t] = block.data.root_pos_w.torch[:, 2].cpu() - rest_z
                if hasattr(hand, "closure"):
                    closure[t] = hand.closure[:, 0].cpu()

        _report(labels, forces, height, closure, valid, args_cli, step_dt, str(checkpoint.name))

        env.close()

if __name__ == "__main__":
    main()
