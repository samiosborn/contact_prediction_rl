# scripts/train.py

"""PPO training entrypoint"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym

from isaaclab.app import add_launcher_args, launch_simulation, report_activity
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.seed import configure_seed

import contact_prediction_rl.tasks  # noqa: F401
from contact_prediction_rl.learning.checkpoints import (
    create_log_directory,
    resolve_checkpoint,
)
from contact_prediction_rl.learning.isaaclab_env import (
    IsaacLabPPOEnvWrapper,
)
import isaaclab_tasks  # noqa: F401

from isaaclab_rl.entrypoints.common import (
    apply_video_recording,
    enable_cameras_for_video,
    pre_launch_video_config,
)
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli


DEFAULT_SEED = 42


# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:
    """Parse PPO training arguments"""

    parser = argparse.ArgumentParser(
        description="Train the contact_prediction_rl PPO agent."
    )

    # Task
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Isaac Lab task name.",
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="ppo_cfg_entry_point",
        help="Gym registry key containing the PPO agent configuration.",
    )

    # Training
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of parallel environments.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training seed.",
    )

    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Maximum number of PPO iterations.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file, run directory, or 'latest'.",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional suffix appended to the run directory.",
    )

    parser.add_argument(
        "--no_random_episode_lengths",
        action="store_true",
        help="Start all environments at episode step zero.",
    )

    # Video
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record training videos.",
    )

    parser.add_argument(
        "--video_length",
        type=int,
        default=None,
        help="Length of each recorded video in environment steps (default 200).",
    )

    parser.add_argument(
        "--video_interval",
        type=int,
        default=None,
        help="Interval between recorded videos in environment steps (default: one clip).",
    )

    # Isaac Lab simulator arguments
    add_launcher_args(
        parser
    )

    args_cli, hydra_args = (
        setup_preset_cli(
            parser,
            argv,
        )
    )

    enable_cameras_for_video(
        args_cli
    )

    return (
        args_cli,
        hydra_args,
    )



def _dump_configs(
    log_dir: Path,
    env_cfg: Any,
    agent_cfg: Any,
) -> None:
    """Store environment, agent, and command configuration"""

    params_dir = (
        log_dir
        / "params"
    )

    params_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dump_yaml(
        str(
            params_dir
            / "env.yaml"
        ),
        env_cfg,
    )

    dump_yaml(
        str(
            params_dir
            / "agent.yaml"
        ),
        agent_cfg,
    )

    (
        log_dir
        / "command.txt"
    ).write_text(
        " ".join(
            sys.orig_argv
        ),
        encoding="utf-8",
    )


def run(
    argv: list[str],
) -> None:
    """Train PPO"""

    from contact_prediction_rl.learning.runners.on_policy_runner import (
        OnPolicyRunner,
    )

    args_cli, hydra_args = (
        _parse_args(
            argv
        )
    )

    # Load the environment and our PPO configuration from Gym registration
    env_cfg, agent_cfg = (
        resolve_task_config(
            args_cli.task,
            args_cli.agent,
            overrides=hydra_args,
        )
    )

    # ----------------------------------------------------------
    # CLI overrides
    # ----------------------------------------------------------

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = (
            args_cli.num_envs
        )

    if args_cli.device is not None:
        env_cfg.sim.device = (
            args_cli.device
        )

    # Ensure the simulator launcher uses the resolved task device
    args_cli.device = (
        env_cfg.sim.device
    )

    seed = (
        args_cli.seed
        if args_cli.seed is not None
        else DEFAULT_SEED
    )

    env_cfg.seed = seed

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = (
            args_cli.max_iterations
        )

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------

    log_root_path, log_dir = (
        create_log_directory(
            experiment_name=(
                agent_cfg.experiment_name
            ),
            run_name=args_cli.run_name,
        )
    )

    resume_path = (
        resolve_checkpoint(
            args_cli.checkpoint,
            log_root_path,
        )
    )

    _dump_configs(
        log_dir,
        env_cfg,
        agent_cfg,
    )

    # ----------------------------------------------------------
    # Simulation
    # ----------------------------------------------------------

    print(
        "[1/3] Launching simulation"
    )

    # The recorder needs a Kit visualizer, which must be requested
    # before the simulation launches
    pre_launch_video_config(
        env_cfg,
        args_cli=args_cli,
    )

    with launch_simulation(
        env_cfg,
        args_cli,
    ):
        print(
            "[2/3] Creating environment"
        )

        # Videos go to <run directory>/videos/train
        apply_video_recording(
            env_cfg,
            str(log_dir),
            args_cli,
        )

        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
        )

        # Convert the Gymnasium step API into the small interface
        # expected by our runner
        report_activity(
            "Wrapping environment"
        )

        env = IsaacLabPPOEnvWrapper(
            env
        )

        report_activity(
            None
        )

        print(
            "[3/3] Preparing agent"
        )

        start_time = time.time()

        report_activity(
            "Building policy"
        )

        runner = OnPolicyRunner(
            env=env,
            train_cfg=agent_cfg.to_dict(),
            log_dir=str(
                log_dir
            ),
            device=env.device,
        )

        report_activity(
            None
        )

        # Configure deterministic Torch behaviour only after model
        # construction so it does not alter network initialisation
        if args_cli.deterministic:
            configure_seed(
                seed,
                torch_deterministic=True,
            )

        runner.add_git_repo_to_log(
            __file__
        )

        if resume_path is not None:
            print(
                f"[INFO]: Loading model checkpoint from: "
                f"{resume_path}"
            )

            runner.load(
                str(
                    resume_path
                )
            )

        try:
            runner.learn(
                num_learning_iterations=(
                    agent_cfg.max_iterations
                ),
                init_at_random_ep_len=(
                    not args_cli.no_random_episode_lengths
                ),
            )

            print(
                f"Training time: "
                f"{round(time.time() - start_time, 2)} seconds"
            )

        except KeyboardInterrupt:
            print(
                "\n[INFO]: Training interrupted"
            )

        finally:
            env.close()


if __name__ == "__main__":
    run(
        sys.argv[1:]
    )