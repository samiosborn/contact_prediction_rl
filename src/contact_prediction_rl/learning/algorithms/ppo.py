# src/contact_prediction_rl/learning/algorithms/ppo.py

from __future__ import annotations

from itertools import chain
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from contact_prediction_rl.learning.models.actor import Actor
from contact_prediction_rl.learning.models.critic import Critic
from contact_prediction_rl.learning.storage.rollout_storage import RolloutStorage


class PPO:
    """Proximal Policy Optimisation"""

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
    ) -> None:
        self.device = device

        # Models
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Keep raw models for checkpointing and export
        self._raw_actor = self.actor
        self._raw_critic = self.critic

        # Optimiser
        if optimizer != "adam":
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        self.optimizer = torch.optim.Adam(
            chain(
                self.actor.parameters(),
                self.critic.parameters(),
            ),
            lr=learning_rate,
        )

        # Storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.gamma = gamma
        self.lam = lam

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.learning_rate = learning_rate
        self.schedule = schedule
        self.desired_kl = desired_kl

        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Kept temporarily for runner compatibility
        self.rnd = None

    def act(
        self,
        obs: TensorDict,
    ) -> torch.Tensor:
        """Sample actions and store transition data"""

        observations = obs["policy"]

        # Sample actions from the policy
        self.transition.actions = self.actor(
            observations,
            stochastic_output=True,
        ).detach()

        # Estimate the state value
        self.transition.values = self.critic(
            observations
        ).detach()

        # Store the log probability under the current policy
        self.transition.actions_log_prob = (
            self.actor.get_output_log_prob(
                self.transition.actions
            ).detach()
        )

        # Store the parameters of the action distribution
        self.transition.distribution_params = tuple(
            parameter.detach()
            for parameter in self.actor.output_distribution_params
        )

        # Store observations before stepping the environment
        self.transition.observations = obs

        return self.transition.actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Store one environment transition"""

        observations = obs["policy"]

        # Update observation normalisation statistics
        self.actor.update_normalisation(
            observations
        )

        self.critic.update_normalisation(
            observations
        )

        # Store rewards and termination state
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrap episodes truncated by a time limit
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Store the completed transition
        self.storage.add_transition(
            self.transition
        )

        self.transition.clear()

    def compute_returns(
        self,
        obs: TensorDict,
    ) -> None:
        """Compute GAE advantages and return targets"""

        storage = self.storage

        # Bootstrap the value of the state after the final rollout step
        last_values = self.critic(
            obs["policy"]
        ).detach()

        advantage = 0

        for step in reversed(
            range(storage.num_transitions_per_env)
        ):
            # V(s_{t+1})
            if step == storage.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = storage.values[
                    step + 1
                ]

            # 1 for non-terminal transitions and 0 for terminal transitions
            next_is_not_terminal = (
                1.0
                - storage.dones[
                    step
                ].float()
            )

            # One-step TD error
            delta = (
                storage.rewards[step]
                + next_is_not_terminal
                * self.gamma
                * next_values
                - storage.values[step]
            )

            # Generalised Advantage Estimation
            advantage = (
                delta
                + next_is_not_terminal
                * self.gamma
                * self.lam
                * advantage
            )

            # Return target
            storage.returns[step] = (
                advantage
                + storage.values[step]
            )

        # A_t = R_t - V(s_t)
        storage.advantages = (
            storage.returns
            - storage.values
        )

        # Normalise advantages over the full rollout
        if not self.normalize_advantage_per_mini_batch:
            storage.advantages = (
                storage.advantages
                - storage.advantages.mean()
            ) / (
                storage.advantages.std()
                + 1e-8
            )

    def update(
        self,
    ) -> dict[str, float]:
        """Run PPO optimisation over the collected rollout"""

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0

        # Generate shuffled feed-forward mini-batches
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
        )

        for batch in generator:
            # Optionally normalise advantages separately in each mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (
                        batch.advantages
                        - batch.advantages.mean()
                    ) / (
                        batch.advantages.std()
                        + 1e-8
                    )

            observations = batch.observations[
                "policy"
            ]

            # Recompute the current policy distribution
            self.actor(
                observations,
                stochastic_output=True,
            )

            # Log probability of stored actions under the new policy
            actions_log_prob = (
                self.actor.get_output_log_prob(
                    batch.actions
                )
            )

            # Policy entropy
            entropy = (
                self.actor.output_entropy
            )

            # Parameters of the current action distribution
            distribution_params = tuple(
                parameter
                for parameter
                in self.actor.output_distribution_params
            )

            # Current critic estimates
            values = self.critic(
                observations
            )

            # Adaptive learning-rate schedule
            if (
                self.desired_kl is not None
                and self.schedule == "adaptive"
            ):
                with torch.inference_mode():
                    kl = (
                        self.actor.get_kl_divergence(
                            batch.old_distribution_params,
                            distribution_params,
                        )
                    )

                    kl_mean = torch.mean(
                        kl
                    )

                    if (
                        kl_mean
                        > self.desired_kl * 2.0
                    ):
                        self.learning_rate = max(
                            1e-5,
                            self.learning_rate
                            / 1.5,
                        )

                    elif (
                        kl_mean
                        < self.desired_kl
                        / 2.0
                        and kl_mean > 0.0
                    ):
                        self.learning_rate = min(
                            1e-2,
                            self.learning_rate
                            * 1.5,
                        )

                    for param_group in self.optimizer.param_groups:
                        param_group[
                            "lr"
                        ] = self.learning_rate

            # Probability ratio between the new and old policies
            ratio = torch.exp(
                actions_log_prob
                - torch.squeeze(
                    batch.old_actions_log_prob
                )
            )

            # Unclipped PPO objective
            surrogate = (
                -torch.squeeze(
                    batch.advantages
                )
                * ratio
            )

            # Clipped PPO objective
            surrogate_clipped = (
                -torch.squeeze(
                    batch.advantages
                )
                * torch.clamp(
                    ratio,
                    1.0
                    - self.clip_param,
                    1.0
                    + self.clip_param,
                )
            )

            # PPO uses the more pessimistic objective
            surrogate_loss = torch.max(
                surrogate,
                surrogate_clipped,
            ).mean()

            # Critic value loss
            if self.use_clipped_value_loss:
                value_clipped = (
                    batch.values
                    + (
                        values
                        - batch.values
                    ).clamp(
                        -self.clip_param,
                        self.clip_param,
                    )
                )

                value_losses = (
                    values
                    - batch.returns
                ).pow(2)

                value_losses_clipped = (
                    value_clipped
                    - batch.returns
                ).pow(2)

                value_loss = torch.max(
                    value_losses,
                    value_losses_clipped,
                ).mean()

            else:
                value_loss = (
                    batch.returns
                    - values
                ).pow(2).mean()

            # Combined PPO loss
            loss = (
                surrogate_loss
                + self.value_loss_coef
                * value_loss
                - self.entropy_coef
                * entropy.mean()
            )

            # Backpropagation
            self.optimizer.zero_grad()

            loss.backward()

            # Clip actor and critic gradients independently
            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.max_grad_norm,
            )

            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.max_grad_norm,
            )

            # Apply the parameter update
            self.optimizer.step()

            # Accumulate logging statistics
            mean_value_loss += (
                value_loss.item()
            )

            mean_surrogate_loss += (
                surrogate_loss.item()
            )

            mean_entropy += (
                entropy.mean().item()
            )

        num_updates = (
            self.num_learning_epochs
            * self.num_mini_batches
        )

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        # The rollout has now been consumed
        self.storage.clear()

        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }

    def train_mode(
        self,
    ) -> None:
        """Set models to training mode"""

        self.actor.train()
        self.critic.train()

    def eval_mode(
        self,
    ) -> None:
        """Set models to evaluation mode"""

        self.actor.eval()
        self.critic.eval()

    def save(
        self,
    ) -> dict:
        """Return checkpoint state"""

        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load(
        self,
        loaded_dict: dict,
        load_cfg: dict | None,
        strict: bool,
    ) -> bool:
        """Load checkpoint state"""

        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
            }

        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(
                loaded_dict[
                    "actor_state_dict"
                ],
                strict=strict,
            )

        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(
                loaded_dict[
                    "critic_state_dict"
                ],
                strict=strict,
            )

        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(
                loaded_dict[
                    "optimizer_state_dict"
                ]
            )

            # Synchronise PPO learning rate with the loaded optimiser
            self.learning_rate = (
                self.optimizer.param_groups[
                    0
                ]["lr"]
            )

        return load_cfg.get(
            "iteration",
            False,
        )

    def get_policy(
        self,
    ) -> Actor:
        """Return the actor used for inference"""

        return self._raw_actor

    def compile(
        self,
        mode: str | None = None,
    ) -> None:
        """Compile actor and critic if requested"""

        if mode is None:
            return

        self.actor = torch.compile(
            self._raw_actor,
            mode=mode,
        )

        self.critic = torch.compile(
            self._raw_critic,
            mode=mode,
        )

    @staticmethod
    def construct_algorithm(
        obs: TensorDict,
        env: Any,
        cfg: dict,
        device: str,
    ) -> PPO:
        """Construct the project PPO stack"""

        if cfg.get("multi_gpu") is not None:
            raise NotImplementedError(
                "Distributed PPO is not currently supported"
            )

        # Observation dimensions
        observation_dim = obs[
            "policy"
        ].shape[-1]

        action_dim = env.num_actions

        # Actor configuration
        actor_cfg = (
            cfg["actor"].copy()
        )

        actor_cfg.pop(
            "class_name",
            None,
        )

        # The old config stores the latent dimension as the final hidden layer
        actor_hidden_dims = list(
            actor_cfg.pop(
                "hidden_dims"
            )
        )

        if len(actor_hidden_dims) < 1:
            raise ValueError(
                "Actor configuration requires at least one hidden dimension"
            )

        actor_latent_dim = (
            actor_hidden_dims.pop()
        )

        actor_activation = (
            actor_cfg.pop(
                "activation"
            )
        )

        actor_observation_normalisation = (
            actor_cfg.pop(
                "obs_normalization",
                False,
            )
        )

        distribution_cfg = (
            actor_cfg.pop(
                "distribution_cfg",
                None,
            )
            or {}
        )

        actor_initial_std = (
            distribution_cfg.get(
                "init_std",
                0.2,
            )
        )

        # Build actor explicitly
        actor = Actor(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims,
            latent_dim=actor_latent_dim,
            activation=actor_activation,
            initial_std=actor_initial_std,
            observation_normalisation=actor_observation_normalisation,
        ).to(device)

        print(
            f"Actor Model: {actor}"
        )

        # Critic configuration
        critic_cfg = (
            cfg["critic"].copy()
        )

        critic_cfg.pop(
            "class_name",
            None,
        )

        # Build critic explicitly
        critic = Critic(
            observation_dim=observation_dim,
            hidden_dims=critic_cfg[
                "hidden_dims"
            ],
            activation=critic_cfg[
                "activation"
            ],
            observation_normalisation=critic_cfg.get(
                "obs_normalization",
                False,
            ),
        ).to(device)

        print(
            f"Critic Model: {critic}"
        )

        # PPO configuration
        algorithm_cfg = (
            cfg["algorithm"].copy()
        )

        algorithm_cfg.pop(
            "class_name",
            None,
        )

        # Ignore unused fields still supplied by the old configuration schema
        algorithm_cfg.pop(
            "rnd_cfg",
            None,
        )

        algorithm_cfg.pop(
            "symmetry_cfg",
            None,
        )

        algorithm_cfg.pop(
            "share_cnn_encoders",
            None,
        )

        algorithm_cfg.pop(
            "use_mixed_precision",
            None,
        )

        # Initialise rollout storage
        storage = RolloutStorage(
            num_envs=env.num_envs,
            num_transitions_per_env=cfg[
                "num_steps_per_env"
            ],
            obs=obs,
            actions_shape=[
                action_dim
            ],
            device=device,
        )

        # Initialise PPO
        algorithm = PPO(
            actor=actor,
            critic=critic,
            storage=storage,
            device=device,
            **algorithm_cfg,
        )

        # Optionally compile the models
        algorithm.compile(
            cfg.get(
                "torch_compile_mode"
            )
        )

        return algorithm

    def broadcast_parameters(
        self,
    ) -> None:
        """Distributed training is not currently supported"""

        raise NotImplementedError(
            "Distributed PPO is not currently supported"
        )