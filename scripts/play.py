# scripts/play.py

"""Run a trained PPO policy in the simulator

Use ``--visualizer kit`` to watch the rollout in the Isaac Sim viewport, or ``--video`` to record it

.. code-block:: bash

    uv run python scripts/play.py \
      --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right \
      --checkpoint latest --visualizer kit
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.entrypoints.common import (
    apply_video_recording,
    enable_cameras_for_video,
    pre_launch_video_config,
)

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import contact_prediction_rl.tasks  # noqa: F401
from contact_prediction_rl.learning.checkpoints import (
    get_log_root_path,
    resolve_checkpoint,
)
from contact_prediction_rl.learning.inference import load_policy
from contact_prediction_rl.learning.isaaclab_env import IsaacLabPPOEnvWrapper


# Camera presets for the recorded video, in world coordinates (the robot base is at the origin facing +x, the table top
# is at z = 0 and the block stands at x = 0.45). Each is (eye, lookat).
VIEWS = {
    "front": ((1.5, 0.0, 0.55), (0.45, 0.0, 0.2)),  # in front of the block, looking back toward the robot
    "side": ((0.45, -1.2, 0.5), (0.45, 0.0, 0.2)),  # from the side of the block
    "top": ((0.2, 0.0, 1.3), (0.45, 0.0, 0.0)),  # looking down at the block
}


def _set_camera(env_cfg, args_cli: argparse.Namespace) -> str | None:
    """Point the recording camera at the requested view. Returns a name for the video folder, or None."""

    eye = tuple(args_cli.camera_eye) if args_cli.camera_eye else None
    lookat = tuple(args_cli.camera_lookat) if args_cli.camera_lookat else None
    name = None

    if args_cli.view is not None:
        eye, lookat = VIEWS[args_cli.view]
        name = args_cli.view

    if eye is None and lookat is None:
        return None

    for visualizer_cfg in getattr(env_cfg.sim, "visualizer_cfgs", None) or []:
        if getattr(visualizer_cfg, "visualizer_type", None) == "kit":
            visualizer_cfg.origin_type = "world"
            if eye is not None:
                visualizer_cfg.eye = eye
            if lookat is not None:
                visualizer_cfg.lookat = lookat
            return name or "custom_view"

    print("[WARN] No Kit visualizer is configured, so the camera view was not applied.")
    return None


def _parse_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:
    """Parse playback arguments"""

    parser = argparse.ArgumentParser(description="Play a trained contact_prediction_rl PPO policy.")
    parser.add_argument("--task", type=str, required=True, help="Isaac Lab task name.")
    parser.add_argument(
        "--agent",
        type=str,
        default="ppo_cfg_entry_point",
        help="Gym registry key containing the PPO agent configuration.",
    )
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint file, run directory, or 'latest' (the newest checkpoint under logs/ppo/<experiment>).",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of environment steps to run.")
    parser.add_argument(
        "--real_time",
        action="store_true",
        help="Slow the loop down to the environment step time, for watching in a viewer.",
    )
    parser.add_argument("--video", action="store_true", help="Record a video of the rollout.")
    parser.add_argument(
        "--view",
        choices=sorted(VIEWS),
        default=None,
        help="Camera view of the video: 'front' (in front of the block), 'side' or 'top'. The video goes to "
        "<run directory>/videos/play_<view>/. Default: the visualizer's default camera.",
    )
    parser.add_argument("--camera_eye", type=float, nargs=3, default=None, help="Camera position [m], world frame.")
    parser.add_argument("--camera_lookat", type=float, nargs=3, default=None, help="Camera target [m], world frame.")
    parser.add_argument("--video_length", type=int, default=None, help="Length of the video in environment steps (default 200).")
    parser.add_argument("--video_interval", type=int, default=None, help="Start a new video clip every this many environment steps.")

    add_launcher_args(parser)

    args_cli, hydra_args = setup_preset_cli(parser, argv)

    enable_cameras_for_video(args_cli)

    return args_cli, hydra_args


def run(
    argv: list[str],
) -> None:
    """Play a trained policy"""

    args_cli, hydra_args = _parse_args(argv)

    env_cfg, agent_cfg = resolve_task_config(
        args_cli.task,
        args_cli.agent,
        overrides=hydra_args,
        play_mode=True,
    )

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    args_cli.device = env_cfg.sim.device

    checkpoint = resolve_checkpoint(
        args_cli.checkpoint,
        get_log_root_path(agent_cfg.experiment_name),
    )
    print(f"[INFO] Playing {checkpoint} on task {args_cli.task}")

    # the recorder needs a Kit visualizer, which must be requested before the simulation launches
    pre_launch_video_config(env_cfg, args_cli=args_cli)
    view_name = _set_camera(env_cfg, args_cli) if args_cli.video else None

    with launch_simulation(env_cfg, args_cli):
        # videos go to <run directory>/videos/play, or play_<view> for a chosen camera view
        apply_video_recording(
            env_cfg,
            str(Path(checkpoint).parent),
            args_cli,
            subdir="play" if view_name is None else f"play_{view_name}",
            checkpoint_path=str(checkpoint),
        )

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = IsaacLabPPOEnvWrapper(env)
        policy = load_policy(env, agent_cfg, str(checkpoint))

        step_dt = env.unwrapped.step_dt
        # with --video, run just past the clip so that it is finished and written
        steps = (args_cli.video_length or 200) + 1 if args_cli.video else args_cli.steps

        obs = env.get_observations()

        try:
            for _ in range(steps):
                start = time.time()

                with torch.inference_mode():
                    obs, _, _, _ = env.step(policy(obs["policy"]))

                if args_cli.real_time:
                    time.sleep(max(0.0, step_dt - (time.time() - start)))

        except KeyboardInterrupt:
            print("\n[INFO]: Playback interrupted")

        finally:
            env.close()


if __name__ == "__main__":
    run(sys.argv[1:])
