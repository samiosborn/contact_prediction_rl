"""CPU checks for Reorientation's hand-only control and local PPO configuration."""

import re

import gymnasium as gym
from pxr import Usd, UsdPhysics

from contact_prediction_rl.robots.franka_hx5d20 import FRANKA_HX5D20_CFG, FRANKA_HX5D20_USD_PATH
from contact_prediction_rl.tasks.reorientation.config.hx5_d20_right.agents.ppo_cfg import PPORunnerCfg
from contact_prediction_rl.tasks.reorientation.config.hx5_d20_right.env_cfg import ReorientationEnvCfg


def test_hand_only_control_and_sensor_paths():
    cfg = ReorientationEnvCfg()
    assert cfg.events.reset_scene_to_default.params["reset_joint_targets"] is True
    stage = Usd.Stage.Open(str(FRANKA_HX5D20_USD_PATH))
    joints = [prim.GetName() for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
    expected = {f"finger_r_joint{i}" for i in range(1, 21)}
    selections = [cfg.actions.joint_pos.joint_names]
    for term in (cfg.observations.policy.joint_pos, cfg.observations.policy.joint_vel):
        selections.append(term.params["asset_cfg"].joint_names)
    for expressions in selections:
        selected = {name for name in joints if any(re.fullmatch(expr, name) for expr in expressions)}
        assert selected == expected
    assert cfg.scene.robot.spawn.usd_path == str(FRANKA_HX5D20_USD_PATH)
    for name in ("panda_shoulder", "panda_forearm"):
        assert cfg.scene.robot.actuators[name] == FRANKA_HX5D20_CFG.actuators[name]
    # Reorientation's hand gains must not mutate the shared robot config used by Reach/Lift.
    assert FRANKA_HX5D20_CFG.actuators["hx5_d20_fingers"].stiffness == 20.0
    for name in ("palm", "thumb", "index", "middle", "ring", "little"):
        sensor = getattr(cfg.scene, f"{name}_contact")
        path = sensor.prim_path.replace("{ENV_REGEX_NS}/Robot", "/World")
        assert stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI)


def test_local_ppo_registration_and_preserved_settings():
    spec = gym.spec("ContactPredictionRl-Reorientation-HX5-D20-Right")
    assert spec.kwargs["default_agent"] == "ppo"
    assert "rsl_rl_cfg_entry_point" not in spec.kwargs
    cfg = PPORunnerCfg()
    assert (cfg.num_steps_per_env, cfg.max_iterations, cfg.save_interval) == (24, 1000, 100)
    assert cfg.actor["hidden_dims"] == [128, 128]
    assert cfg.actor["latent_dim"] == 32
    assert cfg.actor["initial_std"] == 1.0
    assert cfg.actor["observation_normalisation"] is True
    assert cfg.critic["hidden_dims"] == [128, 128, 32]
    assert cfg.critic["observation_normalisation"] is True
    assert cfg.algorithm["learning_rate"] == 1.0e-3
    assert cfg.algorithm["entropy_coef"] == 0.005
