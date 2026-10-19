# src/contact_prediction_rl/learning/algorithms/ppo.py

from __future__ import annotations

from itertools import chain
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from contact_prediction_rl.learning.models.actor import Actor
from contact_prediction_rl.learning.models.contact_prediction import ContactPredictionHead
from contact_prediction_rl.learning.models.critic import Critic
from contact_prediction_rl.learning.storage.rollout_storage import RolloutStorage


class PPO:
    """Proximal Policy Optimisation.

    Only ``obs["policy"]`` enters the actor and critic.
    Other TensorDict groups, including privileged contact labels, are preserved in rollout storage but never used as policy inputs.
    """

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
        contact_head: ContactPredictionHead | None = None,
        contact_loss_weight: float = 0.0,
    ) -> None:
        self.device = device

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Keep raw models for checkpointing and inference
        self._raw_actor = self.actor
        self._raw_critic = self.critic
        self.contact_head = contact_head.to(self.device) if contact_head is not None else None
        self.contact_loss_weight = float(contact_loss_weight)
        if self.contact_loss_weight < 0 or (self.contact_loss_weight > 0 and self.contact_head is None):
            raise ValueError("Nonzero contact BCE weight requires an auxiliary contact head")

        if optimizer != "adam":
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        self.optimizer = torch.optim.Adam(
            chain(
                self.actor.parameters(),
                self.critic.parameters(),
                self.contact_head.parameters() if self.contact_head is not None else (),
            ),
            lr=learning_rate,
        )

        self.storage = storage
        self.transition = RolloutStorage.Transition()

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

    def act(
        self,
        obs: TensorDict,
    ) -> torch.Tensor:
        """Sample actions and retain detached behaviour-policy statistics for PPO."""

        observations = obs["policy"]

        self.transition.actions = self.actor(
            observations,
            stochastic_output=True,
        ).detach()

        self.transition.values = self.critic(observations).detach()

        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()

        self.transition.distribution_params = tuple(parameter.detach() for parameter in self.actor.output_distribution_params)

        self.transition.observations = obs

        return self.transition.actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, Any],
    ) -> None:
        """Store rewards and done flags, adding value bootstrap for timeouts."""

        observations = obs["policy"]

        self.actor.update_normalisation(observations)

        self.critic.update_normalisation(observations)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrap episodes truncated by the time limit
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        self.storage.add_transition(self.transition)

        self.transition.clear()

    def compute_returns(
        self,
        obs: TensorDict,
    ) -> None:
        """Compute GAE over time-major rollouts, stopping recursion at resets."""

        storage = self.storage

        # Bootstrap V(s_{T}) after the final rollout transition
        last_values = self.critic(obs["policy"]).detach()

        advantage = 0

        for step in reversed(range(storage.num_transitions_per_env)):
            next_values = (
                last_values
                if step == storage.num_transitions_per_env - 1
                else storage.values[step + 1]
            )  # fmt: skip

            next_is_not_terminal = 1.0 - storage.dones[step].float()

            delta = storage.rewards[step] + next_is_not_terminal * self.gamma * next_values - storage.values[step]

            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage

            storage.returns[step] = advantage + storage.values[step]

        storage.advantages = storage.returns - storage.values

        # Normalise advantages over the full rollout
        if not self.normalize_advantage_per_mini_batch:
            storage.advantages = (storage.advantages - storage.advantages.mean()) / (storage.advantages.std() + 1e-8)

    def update(
        self,
    ) -> dict[str, float]:
        """Run PPO optimisation over the collected rollout"""

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_contact_loss = 0.0
        contact_metrics: dict[str, float] = {}

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
        )

        for batch in generator:
            # Optionally normalise advantages within each mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            observations = batch.observations["policy"]

            self.actor(
                observations,
                stochastic_output=True,
            )

            actions_log_prob = self.actor.get_output_log_prob(batch.actions)

            entropy = self.actor.output_entropy

            distribution_params = self.actor.output_distribution_params

            values = self.critic(observations)

            # Adaptive learning-rate schedule based on KL divergence
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params,
                        distribution_params,
                    )

                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(
                            1e-5,
                            self.learning_rate / 1.5,
                        )

                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(
                            1e-2,
                            self.learning_rate * 1.5,
                        )

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))

            surrogate = -torch.squeeze(batch.advantages) * ratio

            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )

            # PPO uses the more pessimistic objective
            surrogate_loss = torch.max(
                surrogate,
                surrogate_clipped,
            ).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param,
                    self.clip_param,
                )

                value_losses = (values - batch.returns).pow(2)

                value_losses_clipped = (value_clipped - batch.returns).pow(2)

                value_loss = torch.max(
                    value_losses,
                    value_losses_clipped,
                ).mean()

            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            if self.contact_head is not None and self.contact_loss_weight > 0:
                # Stored observations are pre-action o_t; labels are c_t.
                contact_loss, batch_metrics = self.contact_head.compute_loss(
                    self.actor.encode(observations), batch.observations["contact"]
                )
                loss = loss + self.contact_loss_weight * contact_loss
                mean_contact_loss += contact_loss.item()
                for name, value in batch_metrics.items():
                    contact_metrics[name] = contact_metrics.get(name, 0.0) + value

            self.optimizer.zero_grad()

            loss.backward()

            # Clip actor and critic gradients independently
            nn.utils.clip_grad_norm_(
                chain(self.actor.parameters(), self.contact_head.parameters() if self.contact_head is not None else ()),
                self.max_grad_norm,
            )

            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.max_grad_norm,
            )

            self.optimizer.step()

            mean_value_loss += value_loss.item()

            mean_surrogate_loss += surrogate_loss.item()

            mean_entropy += entropy.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_contact_loss /= num_updates

        self.storage.clear()

        result = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.contact_head is not None:
            result["contact_bce"] = mean_contact_loss
            result.update({name: value / num_updates for name, value in contact_metrics.items()})
        return result

    def train_mode(
        self,
    ) -> None:
        """Set actor and critic to training mode"""

        self.actor.train()
        self.critic.train()
        if self.contact_head is not None:
            self.contact_head.train()

    def eval_mode(
        self,
    ) -> None:
        """Set actor and critic to evaluation mode"""

        self.actor.eval()
        self.critic.eval()
        if self.contact_head is not None:
            self.contact_head.eval()

    def save(
        self,
    ) -> dict:
        """Return checkpoint state"""

        return {
            "actor_state_dict": (self._raw_actor.state_dict()),
            "critic_state_dict": (self._raw_critic.state_dict()),
            "optimizer_state_dict": (self.optimizer.state_dict()),
            "contact_head_state_dict": self.contact_head.state_dict() if self.contact_head is not None else None,
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
                "contact_prediction": True,
                "iteration": True,
            }

        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(
                loaded_dict["actor_state_dict"],
                strict=strict,
            )

        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(
                loaded_dict["critic_state_dict"],
                strict=strict,
            )

        if load_cfg.get("contact_prediction", True) and self.contact_head is not None:
            state = loaded_dict.get("contact_head_state_dict")
            if state is None:
                if strict:
                    raise RuntimeError("Checkpoint has no contact head. Load actor/critic only or use a new run.")
            else:
                self.contact_head.load_state_dict(state, strict=strict)

        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

            # Synchronise PPO learning rate with the loaded optimiser
            self.learning_rate = self.optimizer.param_groups[0]["lr"]

        return load_cfg.get(
            "iteration",
            False,
        )

    def get_policy(
        self,
    ) -> Actor:
        """Return the actor used for inference"""

        return self._raw_actor

    @staticmethod
    def construct_algorithm(
        obs: TensorDict,
        env: Any,
        cfg: dict,
        device: str,
    ) -> PPO:
        """Construct the PPO learning stack"""

        observation_dim = obs["policy"].shape[-1]

        action_dim = env.num_actions

        actor = Actor(
            observation_dim=observation_dim,
            action_dim=action_dim,
            **cfg["actor"],
        ).to(device)

        print(f"Actor Model: {actor}")

        critic = Critic(
            observation_dim=observation_dim,
            **cfg["critic"],
        ).to(device)

        print(f"Critic Model: {critic}")

        contact_cfg = cfg.get("contact_prediction")
        contact_head = None
        contact_loss_weight = 0.0
        if contact_cfg is not None:
            if "contact" not in obs:
                raise ValueError("Contact BCE requires a privileged 'contact' observation group")
            contact_loss_weight = float(contact_cfg["loss_weight"])
            contact_head = ContactPredictionHead(
                latent_dim=cfg["actor"]["latent_dim"],
                num_contacts=obs["contact"].shape[-1],
                hidden_dims=contact_cfg["hidden_dims"],
                activation=contact_cfg["activation"],
                pos_weight=float(contact_cfg["pos_weight"]),
            ).to(device)

        storage = RolloutStorage(
            num_envs=env.num_envs,
            num_transitions_per_env=cfg["num_steps_per_env"],
            obs=obs,
            actions_shape=[action_dim],
            device=device,
        )

        algorithm = PPO(
            actor=actor,
            critic=critic,
            storage=storage,
            device=device,
            contact_head=contact_head,
            contact_loss_weight=contact_loss_weight,
            **cfg["algorithm"],
        )

        return algorithm
