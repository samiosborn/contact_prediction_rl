# tests/test_tactile_env_cfg.py

"""Tests for the observation set-up of the history and tactile lift tasks (configuration only, no simulator)."""

from contact_prediction_rl.robots.hx5_d20_contacts import FINGERTIP_SENSOR_NAMES
from contact_prediction_rl.tasks.lift.config.franka_hx5d20.env_cfg import (
    OBSERVATION_HISTORY,
    FrankaHX5D20BlockLiftPriorResidualEnvCfg,
    FrankaHX5D20BlockLiftPriorResidualHistoryEnvCfg,
    FrankaHX5D20BlockLiftPriorResidualTactileEnvCfg,
)

PROPRIOCEPTIVE_TERMS = ["joint_pos", "joint_vel", "object_position", "target_object_position", "actions"]


def policy_terms(cfg) -> list[str]:
    return [name for name, value in cfg.observations.policy.__dict__.items() if hasattr(value, "func")]


def sensor_names(cfg) -> list[str]:
    return [name for name in cfg.scene.__dict__ if name.endswith("_contact")]


def test_the_prior_residual_task_is_unchanged():
    cfg = FrankaHX5D20BlockLiftPriorResidualEnvCfg()

    assert not cfg.observations.policy.history_length
    assert policy_terms(cfg) == PROPRIOCEPTIVE_TERMS
    assert not hasattr(cfg.observations, "contact")
    assert sensor_names(cfg) == []


def test_the_history_task_stacks_the_proprioceptive_observation_only():
    cfg = FrankaHX5D20BlockLiftPriorResidualHistoryEnvCfg()

    assert cfg.observations.policy.history_length == OBSERVATION_HISTORY
    assert cfg.observations.policy.flatten_history_dim
    assert policy_terms(cfg) == PROPRIOCEPTIVE_TERMS
    assert not hasattr(cfg.observations, "contact")
    assert sensor_names(cfg) == []


def test_the_tactile_task_adds_contacts_to_the_policy_input_and_a_separate_label_group():
    cfg = FrankaHX5D20BlockLiftPriorResidualTactileEnvCfg()

    # contacts are stacked like the other terms, as the last term of the policy group
    assert cfg.observations.policy.history_length == OBSERVATION_HISTORY
    assert policy_terms(cfg) == [*PROPRIOCEPTIVE_TERMS, "fingertip_contacts"]

    # current labels, without history, in their own group (the target of the auxiliary loss)
    assert not cfg.observations.contact.history_length
    assert [n for n, v in cfg.observations.contact.__dict__.items() if hasattr(v, "func")] == ["fingertip_contacts"]


def test_the_tactile_task_has_five_sensors_and_enables_contact_reporting():
    cfg = FrankaHX5D20BlockLiftPriorResidualTactileEnvCfg()

    assert sensor_names(cfg) == list(FINGERTIP_SENSOR_NAMES)
    assert cfg.scene.robot.spawn.activate_contact_sensors

    for name in FINGERTIP_SENSOR_NAMES:
        assert cfg.scene.__dict__[name].filter_prim_paths_expr == ["{ENV_REGEX_NS}/Object"]
