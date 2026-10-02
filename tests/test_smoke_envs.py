# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test: every registered task builds and steps. Needs a CUDA GPU; run with `pytest -m gpu`."""

import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import pytest

import contact_prediction_rl.tasks  # noqa: F401

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_envs.py"
TASKS = sorted(spec.id for spec in gym.registry.values() if spec.id.startswith("ContactPredictionRl-"))


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.smoke
@pytest.mark.parametrize("task", TASKS)
def test_task_builds_and_steps(task):
    # One process per task: the simulator can only be launched once per process.
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--task", task, "--num_envs", "4", "--steps", "5"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert f"SMOKE_OK {task}" in proc.stdout, proc.stdout[-3000:] + proc.stderr[-3000:]
