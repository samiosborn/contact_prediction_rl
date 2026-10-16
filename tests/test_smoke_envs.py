# tests/test_smoke_envs.py

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


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.smoke
def test_pregrasp_initialises_kit_before_assets():
    """Exercise the PhysX pregrasp entry point through one short pose solve."""
    script = SCRIPT.parent / "franka_hx5d20" / "find_franka_hx5d20_pregrasp.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--iterations", "2", "--hovers", "0.15"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = proc.stdout + proc.stderr
    # Kit can return zero after a startup exception, so require a completed solve
    assert proc.returncode == 0 and "hover 0.15 m: position error" in output, output[-4000:]
    assert "Traceback (most recent call last)" not in output, output[-4000:]
