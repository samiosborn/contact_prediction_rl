# src/contact_prediction_rl/learning/storage/rollout_storage.py

from __future__ import annotations

from collections.abc import Generator

import torch
from tensordict import TensorDict


class RolloutStorage:
    """Storage for feed-forward PPO rollout data"""

    class Transition:
        """Data collected for one environment step"""

        def __init__(self) -> None:
            self.observations: TensorDict | None = None
            self.actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None

            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor | None = None
            self.distribution_params: tuple[torch.Tensor, ...] | None = None

        def clear(self) -> None:
            """Reset the transition after it has been stored"""

            self.__init__()

    class Batch:
        """Mini-batch of rollout data used for a PPO update"""

        def __init__(
            self,
            observations: TensorDict,
            actions: torch.Tensor,
            values: torch.Tensor,
            advantages: torch.Tensor,
            returns: torch.Tensor,
            old_actions_log_prob: torch.Tensor,
            old_distribution_params: tuple[torch.Tensor, ...],
        ) -> None:
            self.observations = observations
            self.actions = actions
            self.values = values
            self.advantages = advantages
            self.returns = returns
            self.old_actions_log_prob = old_actions_log_prob
            self.old_distribution_params = old_distribution_params

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
    ) -> None:
        """Allocate buffers for one PPO rollout"""

        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.actions_shape = tuple(actions_shape)

        # Observations
        self.observations = TensorDict(
            {
                key: torch.zeros(
                    num_transitions_per_env,
                    *value.shape,
                    dtype=value.dtype,
                    device=device,
                )
                for key, value in obs.items()
            },
            batch_size=[
                num_transitions_per_env,
                num_envs,
            ],
            device=device,
        )

        # Environment transitions
        self.actions = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *self.actions_shape,
            device=device,
        )

        self.rewards = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=device,
        )

        self.dones = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            dtype=torch.bool,
            device=device,
        )

        # PPO quantities
        self.values = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=device,
        )

        self.actions_log_prob = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=device,
        )

        self.returns = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=device,
        )

        self.advantages = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=device,
        )

        # Lazily allocated when the first policy distribution is observed
        self.distribution_params: tuple[torch.Tensor, ...] | None = None

        # Current write position within the rollout
        self.step = 0

    def add_transition(
        self,
        transition: Transition,
    ) -> None:
        """Store one transition at the current rollout step"""

        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow. Call clear() before adding more transitions.")

        if transition.observations is None:
            raise ValueError("Transition observations are missing")

        if transition.actions is None:
            raise ValueError("Transition actions are missing")

        if transition.rewards is None:
            raise ValueError("Transition rewards are missing")

        if transition.dones is None:
            raise ValueError("Transition dones are missing")

        if transition.values is None:
            raise ValueError("Transition values are missing")

        if transition.actions_log_prob is None:
            raise ValueError("Transition action log probabilities are missing")

        if transition.distribution_params is None:
            raise ValueError("Transition distribution parameters are missing")

        # Environment data
        self.observations[self.step].copy_(transition.observations)

        self.actions[self.step].copy_(transition.actions)

        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))

        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # PPO data
        self.values[self.step].copy_(transition.values)

        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))

        # Allocate policy-distribution buffers on the first step
        if self.distribution_params is None:
            self.distribution_params = tuple(
                torch.zeros(
                    self.num_transitions_per_env,
                    *parameter.shape,
                    dtype=parameter.dtype,
                    device=self.device,
                )
                for parameter in transition.distribution_params
            )

        # Store the policy distribution parameters
        for index, parameter in enumerate(transition.distribution_params):
            self.distribution_params[index][self.step].copy_(parameter)

        self.step += 1

    def clear(
        self,
    ) -> None:
        """Reset the write position for the next rollout"""

        self.step = 0

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int,
    ) -> Generator[Batch, None, None]:
        """Yield shuffled feed-forward PPO mini-batches"""

        if self.step != self.num_transitions_per_env:
            raise RuntimeError("Cannot generate mini-batches before the rollout buffer is full")

        if self.distribution_params is None:
            raise RuntimeError("Policy distribution parameters have not been stored")

        batch_size = self.num_envs * self.num_transitions_per_env

        if batch_size % num_mini_batches != 0:
            raise ValueError("Rollout batch size must be divisible by the number of mini-batches")

        mini_batch_size = batch_size // num_mini_batches

        # Flatten time and environment dimensions
        observations = self.observations.flatten(
            0,
            1,
        )

        actions = self.actions.flatten(
            0,
            1,
        )

        values = self.values.flatten(
            0,
            1,
        )

        advantages = self.advantages.flatten(
            0,
            1,
        )

        returns = self.returns.flatten(
            0,
            1,
        )

        old_actions_log_prob = self.actions_log_prob.flatten(
            0,
            1,
        )

        old_distribution_params = tuple(
            parameter.flatten(
                0,
                1,
            )
            for parameter in self.distribution_params
        )

        for _ in range(num_epochs):
            # Shuffle samples independently for each PPO epoch
            indices = torch.randperm(
                batch_size,
                device=self.device,
            )

            for mini_batch_index in range(num_mini_batches):
                start = mini_batch_index * mini_batch_size

                stop = start + mini_batch_size

                batch_indices = indices[start:stop]

                yield RolloutStorage.Batch(
                    observations=observations[batch_indices],
                    actions=actions[batch_indices],
                    values=values[batch_indices],
                    advantages=advantages[batch_indices],
                    returns=returns[batch_indices],
                    old_actions_log_prob=old_actions_log_prob[batch_indices],
                    old_distribution_params=tuple(parameter[batch_indices] for parameter in old_distribution_params),
                )
