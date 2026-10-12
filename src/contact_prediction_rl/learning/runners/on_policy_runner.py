# src/contact_prediction_rl/learning/runners/on_policy_runner.py

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from contact_prediction_rl.learning.algorithms.ppo import PPO
from contact_prediction_rl.learning.models.actor import Actor


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class OnPolicyRunner:
    """Run rollout collection and PPO optimisation"""

    def __init__(
        self,
        env: Any,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.log_dir = log_dir


        # Build PPO from the environment observation and action dimensions
        obs = self.env.get_observations()

        self.alg = PPO.construct_algorithm(
            obs=obs,
            env=self.env,
            cfg=self.cfg,
            device=self.device,
        )

        # Current training iteration, restored when loading a checkpoint
        self.current_learning_iteration = 0

        # Number of environment transitions processed so far
        self.total_steps = 0

        # Create experiment directory
        if self.log_dir is not None:
            os.makedirs(
                self.log_dir,
                exist_ok=True,
            )

        # TensorBoard writer is created when learning starts
        self.writer = None

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """Run rollout collection and PPO updates"""

        # Randomise initial episode lengths to diversify initial states
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(
                    self.env.max_episode_length
                ),
            )

        # Initial observations
        obs = self.env.get_observations().to(
            self.device
        )

        self.alg.train_mode()

        # Create TensorBoard writer if logging is enabled
        if (
            self.log_dir is not None
            and SummaryWriter is not None
        ):
            self.writer = SummaryWriter(
                log_dir=self.log_dir
            )

        # Track unfinished episodes across rollout iterations
        current_episode_rewards = torch.zeros(
            self.env.num_envs,
            device=self.device,
        )

        current_episode_lengths = torch.zeros(
            self.env.num_envs,
            device=self.device,
        )

        start_iteration = (
            self.current_learning_iteration
        )

        total_iteration = (
            start_iteration
            + num_learning_iterations
        )

        for iteration in range(
            start_iteration,
            total_iteration,
        ):
            iteration_start = time.time()

            completed_episode_rewards = []
            completed_episode_lengths = []

            metric_sums: dict[str, float] = {}
            metric_counts: dict[str, int] = {}

            # ----------------------------------------------------------
            # Rollout collection
            # ----------------------------------------------------------

            with torch.inference_mode():
                collection_start = time.time()

                for _ in range(
                    self.cfg["num_steps_per_env"]
                ):
                    # Sample actions from the current policy
                    actions = self.alg.act(
                        obs
                    )

                    # Step the environment
                    obs, rewards, dones, extras = (
                        self.env.step(
                            actions.to(
                                self.env.device
                            )
                        )
                    )

                    # Check environment outputs before storing them
                    if self.cfg.get(
                        "check_for_nan",
                        True,
                    ):
                        self._check_finite(
                            obs,
                            rewards,
                            dones,
                        )

                    # Move tensors to the training device
                    obs = obs.to(
                        self.device
                    )

                    rewards = rewards.to(
                        self.device
                    )

                    dones = dones.to(
                        self.device
                    )

                    # Store transition in PPO rollout storage
                    self.alg.process_env_step(
                        obs,
                        rewards,
                        dones,
                        extras,
                    )

                    # Episode statistics
                    rewards_flat = rewards.view(
                        -1
                    )

                    dones_flat = dones.view(
                        -1
                    ).bool()

                    current_episode_rewards += (
                        rewards_flat
                    )

                    current_episode_lengths += 1

                    if dones_flat.any():
                        completed_episode_rewards.extend(
                            current_episode_rewards[
                                dones_flat
                            ]
                            .detach()
                            .cpu()
                            .tolist()
                        )

                        completed_episode_lengths.extend(
                            current_episode_lengths[
                                dones_flat
                            ]
                            .detach()
                            .cpu()
                            .tolist()
                        )

                        current_episode_rewards[
                            dones_flat
                        ] = 0

                        current_episode_lengths[
                            dones_flat
                        ] = 0

                    # Environment metrics
                    self._accumulate_metrics(
                        extras,
                        metric_sums,
                        metric_counts,
                    )

                collection_time = (
                    time.time()
                    - collection_start
                )

                # Compute return and advantage targets
                self.alg.compute_returns(
                    obs
                )

            # ----------------------------------------------------------
            # PPO update
            # ----------------------------------------------------------

            learning_start = time.time()

            loss_dict = self.alg.update()

            learning_time = (
                time.time()
                - learning_start
            )

            # Number of environment transitions in this rollout
            rollout_steps = (
                self.cfg["num_steps_per_env"]
                * self.env.num_envs
            )

            self.total_steps += rollout_steps

            self.current_learning_iteration = (
                iteration
            )

            # Average metrics collected during the rollout
            metrics = {
                name: (
                    metric_sums[name]
                    / metric_counts[name]
                )
                for name in metric_sums
                if metric_counts[name] > 0
            }

            mean_episode_reward = (
                sum(completed_episode_rewards)
                / len(completed_episode_rewards)
                if completed_episode_rewards
                else None
            )

            mean_episode_length = (
                sum(completed_episode_lengths)
                / len(completed_episode_lengths)
                if completed_episode_lengths
                else None
            )

            iteration_time = (
                time.time()
                - iteration_start
            )

            # Log iteration
            self._log_iteration(
                iteration=iteration,
                total_iteration=total_iteration,
                rollout_steps=rollout_steps,
                iteration_time=iteration_time,
                collection_time=collection_time,
                learning_time=learning_time,
                loss_dict=loss_dict,
                mean_episode_reward=mean_episode_reward,
                mean_episode_length=mean_episode_length,
                metrics=metrics,
            )

            # Periodic checkpoint
            if (
                self.log_dir is not None
                and iteration
                % self.cfg["save_interval"]
                == 0
            ):
                self.save(
                    os.path.join(
                        self.log_dir,
                        f"model_{iteration}.pt",
                    )
                )

        # Save final checkpoint
        if self.log_dir is not None:
            self.save(
                os.path.join(
                    self.log_dir,
                    f"model_{self.current_learning_iteration}.pt",
                )
            )

        if self.writer is not None:
            self.writer.close()

    def save(
        self,
        path: str,
        infos: dict | None = None,
    ) -> None:
        """Save model and training state"""

        saved_dict = self.alg.save()

        saved_dict[
            "iter"
        ] = self.current_learning_iteration

        saved_dict[
            "infos"
        ] = infos

        torch.save(
            saved_dict,
            path,
        )

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict | None:
        """Load model and training state"""

        loaded_dict = torch.load(
            path,
            weights_only=False,
            map_location=map_location,
        )

        load_iteration = self.alg.load(
            loaded_dict,
            load_cfg,
            strict,
        )

        if load_iteration:
            self.current_learning_iteration = (
                loaded_dict["iter"]
            )

        return loaded_dict.get(
            "infos"
        )

    def get_inference_policy(
        self,
        device: str | None = None,
    ) -> Actor:
        """Return the actor in evaluation mode"""

        self.alg.eval_mode()

        policy = self.alg.get_policy()

        if device is not None:
            policy = policy.to(
                device
            )

        return policy

    def add_git_repo_to_log(
        self,
        repo_file_path: str,
    ) -> None:
        """Store the current repository diff alongside the experiment"""

        if self.log_dir is None:
            return

        file_path = Path(
            repo_file_path
        ).resolve()

        try:
            repo_root = subprocess.run(
                [
                    "git",
                    "-C",
                    str(file_path.parent),
                    "rev-parse",
                    "--show-toplevel",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            diff = subprocess.run(
                [
                    "git",
                    "-C",
                    repo_root,
                    "diff",
                    "HEAD",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            return

        git_log_dir = os.path.join(
            self.log_dir,
            "git",
        )

        os.makedirs(
            git_log_dir,
            exist_ok=True,
        )

        repo_name = Path(
            repo_root
        ).name

        diff_path = os.path.join(
            git_log_dir,
            f"{repo_name}.diff",
        )

        with open(
            diff_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                diff
            )

        print(
            f"Storing git diff for '{repo_name}' in: {diff_path}"
        )

    def _check_finite(
        self,
        obs: Any,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Raise an error if environment outputs contain NaN or infinity"""

        if hasattr(
            obs,
            "items",
        ):
            for name, value in obs.items():
                if (
                    isinstance(
                        value,
                        torch.Tensor,
                    )
                    and not torch.isfinite(
                        value
                    ).all()
                ):
                    raise FloatingPointError(
                        f"Non-finite values found in observation '{name}'"
                    )

        if not torch.isfinite(
            rewards
        ).all():
            raise FloatingPointError(
                "Non-finite values found in rewards"
            )

        if (
            dones.is_floating_point()
            and not torch.isfinite(
                dones
            ).all()
        ):
            raise FloatingPointError(
                "Non-finite values found in done flags"
            )

    def _accumulate_metrics(
        self,
        extras: dict,
        metric_sums: dict[str, float],
        metric_counts: dict[str, int],
    ) -> None:
        """Accumulate scalar environment metrics for logging"""

        log_data = extras.get(
            "log"
        )

        if not isinstance(
            log_data,
            dict,
        ):
            return

        self._accumulate_metric_dict(
            log_data,
            metric_sums,
            metric_counts,
        )

    def _accumulate_metric_dict(
        self,
        values: dict,
        metric_sums: dict[str, float],
        metric_counts: dict[str, int],
        prefix: str = "",
    ) -> None:
        """Recursively accumulate scalar values from a dictionary"""

        for name, value in values.items():
            metric_name = (
                f"{prefix}/{name}"
                if prefix
                else name
            )

            if isinstance(
                value,
                dict,
            ):
                self._accumulate_metric_dict(
                    value,
                    metric_sums,
                    metric_counts,
                    prefix=metric_name,
                )

                continue

            if isinstance(
                value,
                torch.Tensor,
            ):
                if value.numel() == 0:
                    continue

                scalar = (
                    value.detach()
                    .float()
                    .mean()
                    .item()
                )

            elif isinstance(
                value,
                (int, float),
            ):
                scalar = float(
                    value
                )

            else:
                continue

            metric_sums[
                metric_name
            ] = (
                metric_sums.get(
                    metric_name,
                    0.0,
                )
                + scalar
            )

            metric_counts[
                metric_name
            ] = (
                metric_counts.get(
                    metric_name,
                    0,
                )
                + 1
            )

    def _log_iteration(
        self,
        iteration: int,
        total_iteration: int,
        rollout_steps: int,
        iteration_time: float,
        collection_time: float,
        learning_time: float,
        loss_dict: dict[str, float],
        mean_episode_reward: float | None,
        mean_episode_length: float | None,
        metrics: dict[str, float],
    ) -> None:
        """Print and record one training iteration"""

        steps_per_second = int(
            rollout_steps
            / max(
                iteration_time,
                1e-8,
            )
        )

        action_std = (
            self.alg.get_policy()
            .output_std
            .detach()
            .mean()
            .item()
        )

        print(
            "\n"
            + "#" * 80
        )

        print(
            f"{f'Learning iteration {iteration}/{total_iteration}':^80}"
        )

        print()

        print(
            f"{'Total steps:':>35} {self.total_steps}"
        )

        print(
            f"{'Steps per second:':>35} {steps_per_second}"
        )

        print(
            f"{'Collection time:':>35} {collection_time:.3f}s"
        )

        print(
            f"{'Learning time:':>35} {learning_time:.3f}s"
        )

        print(
            f"{'Mean value loss:':>35} {loss_dict['value']:.4f}"
        )

        print(
            f"{'Mean surrogate loss:':>35} {loss_dict['surrogate']:.4f}"
        )

        print(
            f"{'Mean entropy loss:':>35} {loss_dict['entropy']:.4f}"
        )

        print(
            f"{'Learning rate:':>35} {self.alg.learning_rate:.6g}"
        )

        print(
            f"{'Mean action std:':>35} {action_std:.4f}"
        )

        if mean_episode_reward is not None:
            print(
                f"{'Mean reward:':>35} {mean_episode_reward:.2f}"
            )

        if mean_episode_length is not None:
            print(
                f"{'Mean episode length:':>35} {mean_episode_length:.2f}"
            )

        for name, value in metrics.items():
            print(
                f"{name:>35}: {value:.4f}"
            )

        print(
            "-" * 80
        )

        print(
            f"{'Iteration time:':>35} {iteration_time:.2f}s"
        )

        # TensorBoard logging
        if self.writer is None:
            return

        self.writer.add_scalar(
            "Loss/value",
            loss_dict["value"],
            iteration,
        )

        self.writer.add_scalar(
            "Loss/surrogate",
            loss_dict["surrogate"],
            iteration,
        )

        self.writer.add_scalar(
            "Loss/entropy",
            loss_dict["entropy"],
            iteration,
        )

        self.writer.add_scalar(
            "Policy/learning_rate",
            self.alg.learning_rate,
            iteration,
        )

        self.writer.add_scalar(
            "Policy/action_std",
            action_std,
            iteration,
        )

        self.writer.add_scalar(
            "Performance/steps_per_second",
            steps_per_second,
            iteration,
        )

        if mean_episode_reward is not None:
            self.writer.add_scalar(
                "Train/mean_episode_reward",
                mean_episode_reward,
                iteration,
            )

        if mean_episode_length is not None:
            self.writer.add_scalar(
                "Train/mean_episode_length",
                mean_episode_length,
                iteration,
            )

        for name, value in metrics.items():
            self.writer.add_scalar(
                name,
                value,
                iteration,
            )