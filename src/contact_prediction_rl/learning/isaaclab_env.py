# src/contact_prediction_rl/learning/isaaclab_env.py

"""Adapter between Isaac Lab environments and the local PPO runner"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict


class IsaacLabPPOEnvWrapper:
    """Minimal adapter between an Isaac Lab environment and our PPO runner"""

    def __init__(
        self,
        env: Any,
        clip_actions: float | None = None,
    ) -> None:
        self.env = env
        self.clip_actions = clip_actions

        # Our runner expects the environment to already be reset
        self.reset()

    @property
    def unwrapped(
        self,
    ) -> Any:
        """Return the underlying Isaac Lab environment"""

        return self.env.unwrapped

    @property
    def num_envs(
        self,
    ) -> int:
        """Number of parallel environments"""

        return self.unwrapped.num_envs

    @property
    def num_actions(
        self,
    ) -> int:
        """Number of policy action dimensions"""

        if hasattr(
            self.unwrapped,
            "action_manager",
        ):
            return int(self.unwrapped.action_manager.total_action_dim)

        return int(self.unwrapped.single_action_space.shape[-1])

    @property
    def device(
        self,
    ) -> str:
        """Environment simulation device"""

        return self.unwrapped.device

    @property
    def cfg(
        self,
    ) -> Any:
        """Environment configuration"""

        return self.unwrapped.cfg

    @property
    def max_episode_length(
        self,
    ) -> int:
        """Maximum number of environment steps per episode"""

        return self.unwrapped.max_episode_length

    @property
    def episode_length_buf(
        self,
    ) -> torch.Tensor:
        """Current episode lengths for all environments"""

        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(
        self,
        value: torch.Tensor,
    ) -> None:
        self.unwrapped.episode_length_buf = value

    def reset(
        self,
    ) -> tuple[TensorDict, dict]:
        """Reset the environment"""

        observations, extras = self.env.reset()

        return (
            self._to_tensordict(observations),
            extras,
        )

    def get_observations(
        self,
    ) -> TensorDict:
        """Return the current environment observations"""

        observations = self.unwrapped.observation_manager.compute() if hasattr(self.unwrapped, "observation_manager") else self.unwrapped._get_observations()

        return self._to_tensordict(observations)

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[
        TensorDict,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        """Step the environment"""

        if self.clip_actions is not None:
            actions = torch.clamp(
                actions,
                -self.clip_actions,
                self.clip_actions,
            )

        (
            observations,
            rewards,
            terminated,
            truncated,
            extras,
        ) = self.env.step(actions)

        # PPO treats both environment termination and truncation as done
        dones = terminated | truncated

        # Store truncation information separately so PPO can bootstrap
        # time-limit terminations when using an infinite-horizon task
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return (
            self._to_tensordict(observations),
            rewards,
            dones,
            extras,
        )

    def close(
        self,
    ) -> None:
        """Close the environment"""

        self.env.close()

    def _to_tensordict(
        self,
        observations: Any,
    ) -> TensorDict:
        """Convert Isaac Lab observations to a TensorDict"""

        if isinstance(
            observations,
            TensorDict,
        ):
            return observations

        return TensorDict(
            observations,
            batch_size=[self.num_envs],
        )
