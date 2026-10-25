"""Inspect zero, Gaussian or untrained-policy finger actions without altering PPO.

This is a *continuous physics probe*, not a training/evaluation episode.
Success/drop terminations are disabled only in the local diagnostic configuration
so the cube can be followed for a whole trajectory. Keep steps below the 180-step
episode limit to avoid auto-reset. The simulation, gravity and rewards are unchanged.

Example:
  uv run python scripts/reorientation/diagnose_training.py --mode random --steps 150
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import contact_prediction_rl.tasks  # noqa: E402, F401
from contact_prediction_rl.diagnostics.contact import ContactTracker  # noqa: E402
from contact_prediction_rl.diagnostics.control import action_statistics, joint_tracking_statistics  # noqa: E402
from contact_prediction_rl.learning.models.actor import Actor  # noqa: E402
from contact_prediction_rl.tasks.reorientation.diagnostics import (  # noqa: E402
    ReorientationTracker,
    measure_reorientation,
)
from contact_prediction_rl.tasks.reorientation.grasp_snapshot import load_grasp_snapshot  # noqa: E402

TASK = "ContactPredictionRl-Reorientation-Train-HX5-D20-Right"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("zero", "random", "untrained"), default="random")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--action-std", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1 or args.action_std < 0:
        parser.error("Positive environment/step counts and nonnegative action std required")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.terminations.success = None
    cfg.terminations.dropped = None
    if args.steps >= round(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation)):
        raise ValueError("Use fewer than a full episode to avoid auto-reset in diagnostic mode")
    snapshot = load_grasp_snapshot(cfg.grasp_snapshot_path)
    goal = cfg.rewards.orientation.params["goal_xyzw"]

    with launch_simulation(cfg, args):
        from isaaclab.envs import ManagerBasedRLEnv

        from contact_prediction_rl.tasks.reorientation.config.hx5_d20_right.train_ppo_cfg import ReorientationTrainPPORunnerCfg

        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            obs, _ = env.reset()
            n = env.num_envs
            device = env.device
            action_dim = int(env.action_manager.total_action_dim)
            actor = None
            if args.mode == "untrained":
                actor_cfg = ReorientationTrainPPORunnerCfg().actor
                actor = Actor(obs["policy"].shape[-1], action_dim, **actor_cfg).to(device)
                actor.eval()
            contacts = ContactTracker(n, 5, device=device)
            motion = ReorientationTracker(n, device=device)
            start_pos = torch.as_tensor(snapshot["cube_pos_w"], device=device).expand(n, 3) + env.scene.env_origins
            start_quat = torch.as_tensor(snapshot["cube_quat_w"], device=device)
            axis = torch.as_tensor(snapshot["axis_world"], device=device)
            goal_quat = torch.as_tensor(goal, device=device)
            controls: list[dict[str, float]] = []
            tracking: list[dict[str, float]] = []
            term = env.action_manager.get_term("joint_pos")
            indices = [env.scene["robot"].joint_names.index(name) for name in term._joint_names]
            for _ in range(args.steps):
                with torch.inference_mode():
                    if args.mode == "zero":
                        action = torch.zeros((n, action_dim), device=device)
                    elif args.mode == "random":
                        action = args.action_std * torch.randn((n, action_dim), device=device)
                    else:
                        action = actor(obs["policy"], stochastic_output=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    if bool((terminated | truncated).any()):
                        raise RuntimeError("Unexpected reset during continuous diagnostic rollout")
                    controls.append(action_statistics(action))
                    cube = env.scene["object"].data
                    metrics = measure_reorientation(
                        cube.root_pos_w.torch, cube.root_quat_w.torch, obs["contact"],
                        start_pos_w=start_pos, start_quat_w=start_quat,
                        goal_quat_w=goal_quat, axis_world=axis,
                    )
                    motion.observe(metrics)
                    contacts.observe(obs["contact"])
                    processed = getattr(term, "processed_actions", None)
                    if processed is not None and processed.shape == (n, action_dim):
                        measured = env.scene["robot"].data.joint_pos.torch[:, indices]
                        tracking.append(joint_tracking_statistics(measured, processed))
            average_controls = {key: sum(item[key] for item in controls) / len(controls) for key in controls[0]}
            average_tracking = (
                {key: sum(item[key] for item in tracking) / len(tracking) for key in tracking[0]}
                if tracking else {"joint_tracking": "unavailable_from_action_term"}
            )
            print(json.dumps({
                "mode": args.mode, "num_envs": n, "steps": args.steps,
                "note": "continuous probe; success/drop terminations disabled, no reset",
                "motion": motion.summary(),
                "contact": contacts.summary(fixed_pattern=(1, 1, 1, 0, 0)),
                "control": average_controls,
                "joint_tracking": average_tracking,
            }, indent=2), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
