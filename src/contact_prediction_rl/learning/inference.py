# src/contact_prediction_rl/learning/inference.py

"""Load a trained PPO actor for evaluation and playback"""

from __future__ import annotations

from typing import Any

from contact_prediction_rl.learning.models.actor import Actor
from contact_prediction_rl.learning.runners.on_policy_runner import (
    OnPolicyRunner,
)


def load_policy(
    env: Any,
    agent_cfg: Any,
    checkpoint: str,
) -> Actor:
    """Build the actor from the agent configuration and load a checkpoint

    The returned actor maps the "policy" observation group to deterministic actions
    """

    runner = OnPolicyRunner(
        env=env,
        train_cfg=agent_cfg.to_dict(),
        log_dir=None,
        device=env.device,
    )

    runner.load(
        checkpoint,
        map_location=env.device,
    )

    return runner.get_inference_policy(
        device=env.device
    )
