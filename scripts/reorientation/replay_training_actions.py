"""Replay the scripted finger gait through the actual 20D Reorientation action term.

Unlike the feasibility script, this uses the snapshotted TRAINING environment
and its normal action-processing path. For uninterrupted physical observation,
success/drop terminations are disabled in this diagnostic-only configuration;
gravity, sensors and actuator gains are unchanged. Joint targets are clipped
to actual limits exactly as in the standalone feasibility script.

Example:
  uv run python scripts/reorientation/replay_training_actions.py --controller gait --pattern spread --amplitude -0.6
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
from contact_prediction_rl.tasks.reorientation.diagnostics import ReorientationTracker, measure_reorientation  # noqa: E402
from contact_prediction_rl.tasks.reorientation.grasp_snapshot import load_grasp_snapshot  # noqa: E402
from contact_prediction_rl.tasks.reorientation.replay import rotation_waypoints, targets_to_actions  # noqa: E402

TASK = "ContactPredictionRl-Reorientation-Train-HX5-D20-Right"


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("gait", "offset"), default="gait")
    parser.add_argument("--pattern", choices=("spread", "differential_curl", "thumb_index"), default="spread")
    parser.add_argument("--amplitude", type=float, default=-0.6, help="Signed pattern amplitude in radians")
    parser.add_argument("--rotate-steps", type=int, default=150)
    parser.add_argument("--ramp-steps", type=int, default=90)
    parser.add_argument("--gait-cycles", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    add_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_envs <= 0:
        raise ValueError("num-envs must be positive")
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.terminations.success = None
    cfg.terminations.dropped = None
    max_steps = round(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
    if args.rotate_steps >= max_steps:
        raise ValueError(f"Replay is limited to fewer than {max_steps} steps to avoid auto-reset")
    snapshot = load_grasp_snapshot(cfg.grasp_snapshot_path)
    goal = cfg.rewards.orientation.params["goal_xyzw"]

    with launch_simulation(cfg, args):
        from isaaclab.envs import ManagerBasedRLEnv

        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            env.reset()
            device, n = env.device, env.num_envs
            term = env.action_manager.get_term("joint_pos")
            names = list(term._joint_names)
            if len(names) != 20 or set(names) != {f"finger_r_joint{i}" for i in range(1, 21)}:
                raise RuntimeError(f"Unexpected training action order: {names}")
            if not hasattr(term, "_offset") or not hasattr(term, "_scale"):
                raise RuntimeError("Expected a resolved Isaac Lab joint-position offset and scale")
            # Isaac Lab permits scalar floats as action scales/offsets.
            offset = torch.as_tensor(_tensor(term._offset), device=device, dtype=torch.float32)
            scale = torch.as_tensor(_tensor(term._scale), device=device, dtype=torch.float32)
            robot = env.scene["robot"]
            ids = [robot.joint_names.index(name) for name in names]
            limits = robot.data.joint_pos_limits.torch[:, ids]
            closed = [snapshot["finger_target_pos"][name] for name in names]
            commands = rotation_waypoints(
                closed, names, controller=args.controller, pattern=args.pattern,
                amplitude_rad=args.amplitude, rotate_steps=args.rotate_steps,
                ramp_steps=args.ramp_steps, gait_cycles=args.gait_cycles,
            )
            start_pos = torch.as_tensor(snapshot["cube_pos_w"], device=device).expand(n, 3) + env.scene.env_origins
            start_quat = torch.as_tensor(snapshot["cube_quat_w"], device=device)
            axis = torch.as_tensor(snapshot["axis_world"], device=device)
            goal_quat = torch.as_tensor(goal, device=device)
            tracker = ReorientationTracker(n, device=device)
            contact_tracker = ContactTracker(n, 5, device=device)
            max_raw_action = 0.0
            max_processed_target_error = 0.0
            total_clipped = 0
            error_sums = 0.0
            for values in commands:
                with torch.inference_mode():
                    requested = torch.tensor(values, device=device, dtype=offset.dtype).unsqueeze(0).expand(n, -1)
                    target = torch.minimum(torch.maximum(requested, limits[..., 0]), limits[..., 1])
                    total_clipped += int((target != requested).sum().item())
                    actions = targets_to_actions(target, offset=offset, scale=scale)
                    max_raw_action = max(max_raw_action, action_statistics(actions)["max_abs_action"])
                    obs, _, terminated, truncated, _ = env.step(actions)
                    if bool((terminated | truncated).any()):
                        raise RuntimeError("Unexpected auto-reset during scripted replay")
                    actual_processed = getattr(term, "processed_actions", None)
                    if actual_processed is not None:
                        processed = _tensor(actual_processed)
                        max_processed_target_error = max(
                            max_processed_target_error, (processed - target).abs().max().item()
                        )
                    measured = robot.data.joint_pos.torch[:, ids]
                    error_sums += joint_tracking_statistics(measured, target)["mean_abs_joint_target_error_rad"]
                    cube = env.scene["object"].data
                    metrics = measure_reorientation(
                        cube.root_pos_w.torch, cube.root_quat_w.torch, obs["contact"],
                        start_pos_w=start_pos, start_quat_w=start_quat,
                        goal_quat_w=goal_quat, axis_world=axis,
                    )
                    tracker.observe(metrics)
                    contact_tracker.observe(obs["contact"])
            print(json.dumps({
                "controller": args.controller, "pattern": args.pattern,
                "amplitude_rad": args.amplitude, "steps": len(commands), "num_envs": n,
                "note": "training action interface, physics unchanged, success/drop terminations disabled",
                "max_raw_action": max_raw_action,
                "joint_target_clipped_fraction": total_clipped / (len(commands) * n * 20),
                "max_processed_target_error_rad": max_processed_target_error if hasattr(term, "processed_actions") else None,
                "mean_abs_measured_joint_target_error_rad": error_sums / len(commands),
                "motion": tracker.summary(),
                "contact": contact_tracker.summary(fixed_pattern=(1, 1, 1, 0, 0)),
            }, indent=2), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
