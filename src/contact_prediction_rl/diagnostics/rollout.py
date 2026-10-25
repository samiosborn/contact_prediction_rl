"""Episode statistics for batched auto-reset environments (pure Torch)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EpisodeRecord:
    """One finished episode. Terminal measurements must be taken before simulator reset."""

    env_id: int
    length: int
    reward: float
    timed_out: bool
    success_ever: bool
    success_at_end: bool
    longest_success_streak: int
    first_goal_error: float | None
    minimum_goal_error: float | None
    final_goal_error: float | None


class EpisodeTracker:
    """Aggregate one scalar reward and optional task-defined goal predicate per environment.

    Call update() with the transition reward/done and with *pre-reset* goal measurements.
    Isaac Lab auto-resets on step: reading scene tensors after a terminating step
    may instead measure the next episode. Callers must supply terminal data from
    a pre-reset hook or clearly label pre-step metrics as non-terminal estimates.
    """

    def __init__(self, num_envs: int, device: str | torch.device = "cpu") -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.records: list[EpisodeRecord] = []
        self._reward = torch.zeros(num_envs, device=device)
        self._length = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._at_end = torch.zeros_like(self._seen)
        self._streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._longest = torch.zeros_like(self._streak)
        self._first_error = torch.full((num_envs,), float("nan"), device=device)
        self._min_error = torch.full((num_envs,), float("inf"), device=device)
        self._last_error = torch.full((num_envs,), float("nan"), device=device)

    def update(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        *,
        time_outs: torch.Tensor | None = None,
        success: torch.Tensor | None = None,
        goal_error: torch.Tensor | None = None,
    ) -> list[EpisodeRecord]:
        """Process one transition. All arguments are length-N, never averaged first."""
        if rewards.shape != (self.num_envs,) or dones.shape != (self.num_envs,):
            raise ValueError("Rewards and done flags must have shape (num_envs,)")
        if not torch.isfinite(rewards).all():
            raise ValueError("Non-finite episode rewards")
        for name, value in (("time_outs", time_outs), ("success", success), ("goal_error", goal_error)):
            if value is not None and value.shape != (self.num_envs,):
                raise ValueError(f"{name} must have shape (num_envs,)")
        done = dones.to(device=self.device, dtype=torch.bool)
        timeout = torch.zeros_like(done) if time_outs is None else time_outs.to(self.device).bool()
        self._reward += rewards.to(self.device)
        self._length += 1
        if goal_error is not None:
            error = goal_error.to(self.device)
            if not torch.isfinite(error).all():
                raise ValueError("Non-finite goal error")
            first = torch.isnan(self._first_error)
            self._first_error = torch.where(first, error, self._first_error)
            self._min_error = torch.minimum(self._min_error, error)
            self._last_error = error
        if success is not None:
            hit = success.to(self.device).bool()
            self._seen |= hit
            self._at_end = hit
            self._streak = torch.where(hit, self._streak + 1, 0)
            self._longest = torch.maximum(self._longest, self._streak)
        finished: list[EpisodeRecord] = []
        for idx in done.nonzero(as_tuple=False).flatten().tolist():
            def optional(value: torch.Tensor, index: int = idx) -> float | None:
                result = float(value[index].item())
                return result if bool(torch.isfinite(value[index])) else None

            record = EpisodeRecord(
                env_id=idx,
                length=int(self._length[idx]),
                reward=float(self._reward[idx]),
                timed_out=bool(timeout[idx]),
                success_ever=bool(self._seen[idx]),
                success_at_end=bool(self._at_end[idx]),
                longest_success_streak=int(self._longest[idx]),
                first_goal_error=optional(self._first_error),
                minimum_goal_error=optional(self._min_error),
                final_goal_error=optional(self._last_error),
            )
            finished.append(record)
            self.records.append(record)
        if bool(done.any()):
            self._reward[done] = 0
            self._length[done] = 0
            self._seen[done] = False
            self._at_end[done] = False
            self._streak[done] = 0
            self._longest[done] = 0
            self._first_error[done] = float("nan")
            self._min_error[done] = float("inf")
            self._last_error[done] = float("nan")
        return finished

    def summary(self) -> dict[str, float]:
        """Return completed-episode summaries only; empty result means no episodes ended."""
        if not self.records:
            return {}
        n = len(self.records)
        result = {
            "episodes": float(n),
            "mean_return": sum(r.reward for r in self.records) / n,
            "mean_episode_length": sum(r.length for r in self.records) / n,
            "timeout_rate": sum(r.timed_out for r in self.records) / n,
            "success_ever_rate": sum(r.success_ever for r in self.records) / n,
            "success_at_end_rate": sum(r.success_at_end for r in self.records) / n,
            "mean_longest_success_streak": sum(r.longest_success_streak for r in self.records) / n,
        }
        for name in ("first_goal_error", "minimum_goal_error", "final_goal_error"):
            values = [getattr(r, name) for r in self.records if getattr(r, name) is not None]
            if values:
                result[f"mean_{name}"] = sum(values) / len(values)
        return result
