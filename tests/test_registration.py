# tests/test_registration.py

"""Tests for the generated task registrations."""

import gymnasium as gym

import contact_prediction_rl.tasks  # noqa: F401


def test_task_registrations():
    """The generated tasks must expose valid environment and agent entry points."""
    expected = {
        "ContactPredictionRl-Reorientation-HX5-D20-Right": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": ("contact_prediction_rl.tasks.reorientation.config.hx5_d20_right.env_cfg:ReorientationEnvCfg"),
            "default_agent": "ppo",
            "ppo_cfg_entry_point": ("contact_prediction_rl.tasks.reorientation.config.hx5_d20_right.agents.ppo_cfg:PPORunnerCfg"),
        },
        "ContactPredictionRl-Reorientation-Train-HX5-D20-Right": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": ("contact_prediction_rl.tasks.reorientation.config.hx5_d20_right.train_env_cfg:ReorientationTrainEnvCfg"),
            "default_agent": "ppo",
            "ppo_cfg_entry_point": ("contact_prediction_rl.tasks.reorientation.config.hx5_d20_right.train_ppo_cfg:ReorientationTrainPPORunnerCfg"),
        },
        "ContactPredictionRl-Reach-Franka-HX5-D20-Right": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": ("contact_prediction_rl.tasks.reach.config.franka_hx5d20.env_cfg:FrankaHX5D20ReachEnvCfg"),
            "ppo_cfg_entry_point": ("contact_prediction_rl.tasks.reach.config.franka_hx5d20.agents.ppo_cfg:FrankaHX5D20ReachPPOCfg"),
            "default_agent": "ppo",
        },
        "ContactPredictionRl-Lift-Block-Franka-HX5-D20-Right": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": ("contact_prediction_rl.tasks.lift.config.franka_hx5d20.env_cfg:FrankaHX5D20BlockLiftEnvCfg"),
            "ppo_cfg_entry_point": ("contact_prediction_rl.tasks.lift.config.franka_hx5d20.agents.ppo_cfg:FrankaHX5D20BlockLiftPPOCfg"),
            "default_agent": "ppo",
        },
        "ContactPredictionRl-Lift-Block-Franka-HX5-D20-Curl-Right": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": ("contact_prediction_rl.tasks.lift.config.franka_hx5d20.env_cfg:FrankaHX5D20BlockLiftCurlEnvCfg"),
            "ppo_cfg_entry_point": ("contact_prediction_rl.tasks.lift.config.franka_hx5d20.agents.ppo_cfg:FrankaHX5D20BlockLiftCurlPPOCfg"),
            "default_agent": "ppo",
        },
        "ContactPredictionRl-Lift-Block-Franka-HX5-D20-AllJoints-Right": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": (
                "contact_prediction_rl.tasks.lift.config.franka_hx5d20.env_cfg:"
                "FrankaHX5D20BlockLiftAllJointsEnvCfg"
            ),
            "ppo_cfg_entry_point": (
                "contact_prediction_rl.tasks.lift.config.franka_hx5d20.agents.ppo_cfg:"
                "FrankaHX5D20BlockLiftAllJointsPPOCfg"
            ),
            "default_agent": "ppo",
        },
        "ContactPredictionRl-Lift-Block-Franka-HX5-D20-Synergy-Right": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": ("contact_prediction_rl.tasks.lift.config.franka_hx5d20.env_cfg:FrankaHX5D20BlockLiftSynergyEnvCfg"),
            "ppo_cfg_entry_point": ("contact_prediction_rl.tasks.lift.config.franka_hx5d20.agents.ppo_cfg:FrankaHX5D20BlockLiftSynergyPPOCfg"),
            "default_agent": "ppo",
        },
    }

    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)

        assert spec.entry_point == expected_values["entry_point"]

        assert spec.kwargs["env_cfg_entry_point"] == expected_values["env_cfg_entry_point"]

        if "ppo_cfg_entry_point" in expected_values:
            assert spec.kwargs["ppo_cfg_entry_point"] == expected_values["ppo_cfg_entry_point"]

        if "default_agent" in expected_values:
            assert spec.kwargs["default_agent"] == expected_values["default_agent"]


def test_training_registration_does_not_require_grasp_snapshot():
    """The task must be discoverable before loading or exporting a local snapshot."""
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env.pop("REORIENTATION_GRASP_SNAPSHOT", None)
    command = [
        sys.executable,
        "-c",
        (
            "import gymnasium as gym; "
            "import contact_prediction_rl.tasks; "
            "spec = gym.spec('ContactPredictionRl-Reorientation-Train-HX5-D20-Right'); "
            "assert spec.kwargs['env_cfg_entry_point'].endswith('train_env_cfg:ReorientationTrainEnvCfg')"
        ),
    ]
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
