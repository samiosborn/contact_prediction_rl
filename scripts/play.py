# scripts/play.py

"""Run a trained PPO policy in the simulator

Use ``--visualizer kit`` to watch the rollout in the Isaac Sim viewport, or ``--video`` to record it

.. code-block:: bash

    uv run --extra isaacsim python scripts/play.py \
      --task ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right \
      --checkpoint latest --visualizer kit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import contact_prediction_rl.tasks  # noqa: F401
from contact_prediction_rl.learning.checkpoints import (
    get_log_root_path,
    resolve_checkpoint,
)
from contact_prediction_rl.learning.inference import load_policy
from contact_prediction_rl.learning.isaaclab_env import IsaacLabPPOEnvWrapper


def _parse_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:
    """Parse playback arguments"""

    parser = argparse.ArgumentParser(
        description="Play a trained contact_prediction_rl PPO policy."
    )
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
    parser.add_argument("--video_length", type=int, default=200, help="Length of the video in environment steps.")

    add_launcher_args(parser)

    args_cli, hydra_args = setup_preset_cli(parser, argv)

    if args_cli.video:
        args_cli.enable_cameras = True

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

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None,
        )

        if args_cli.video:
            video_dir = Path(checkpoint).parent / "videos" / "play"
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_dir),
                step_trigger=lambda step: step == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
            )

        env = IsaacLabPPOEnvWrapper(env)
        policy = load_policy(env, agent_cfg, str(checkpoint))

        step_dt = env.unwrapped.step_dt
        steps = args_cli.video_length if args_cli.video else args_cli.steps

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
