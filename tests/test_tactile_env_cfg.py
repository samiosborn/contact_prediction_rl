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


def test_the_tr_agent_decodes_exactly_the_tactile_part_of_the_observation():
    from contact_prediction_rl.tasks.lift.config.franka_hx5d20.agents.ppo_cfg import (
        TACTILE_DIM,
        FrankaHX5D20BlockLiftTactileTRPaperPPOCfg,
    )

    cfg = FrankaHX5D20BlockLiftTactileTRPaperPPOCfg().to_dict()

    # 5 fingertips over the stacked history: the last entries of the policy observation
    assert TACTILE_DIM == OBSERVATION_HISTORY * len(FINGERTIP_SENSOR_NAMES)
    assert cfg["auxiliary"]["tactile_dim"] == TACTILE_DIM
    assert cfg["auxiliary"]["pos_weight"] == 10.0
    assert cfg["auxiliary"]["hidden_dims"] == [512, 512]
    assert cfg["architecture"] == "shared"


# --- grasp-start variants: the open hand starts at the grasp pose and the closed-hand reward is narrower ---


def arm_start(cfg) -> list[float]:
    return [cfg.scene.robot.init_state.joint_pos[f"panda_joint{i}"] for i in range(1, 8)]


def test_the_grasp_start_tasks_change_only_the_start_pose_and_the_closing_kernel():
    from contact_prediction_rl.tasks.lift.config.franka_hx5d20.env_cfg import (
        GRASP_START_ARM_POS,
        GRASP_START_CLOSED_NEAR_STD,
        START_ARM_POS,
        CLOSED_NEAR_STD,
        FrankaHX5D20BlockLiftGraspStartHistoryEnvCfg,
        FrankaHX5D20BlockLiftGraspStartTactileEnvCfg,
    )

    pairs = [
        (FrankaHX5D20BlockLiftPriorResidualHistoryEnvCfg, FrankaHX5D20BlockLiftGraspStartHistoryEnvCfg),
        (FrankaHX5D20BlockLiftPriorResidualTactileEnvCfg, FrankaHX5D20BlockLiftGraspStartTactileEnvCfg),
    ]

    for before_cls, after_cls in pairs:
        before, after = before_cls(), after_cls()

        # the existing tasks keep the 6 cm approach and the wide kernel
        assert arm_start(before) == START_ARM_POS
        assert before.rewards.hand_closed_near_object.params["std"] == CLOSED_NEAR_STD

        # the variants start at the grasp pose and narrow the kernel
        assert arm_start(after) == GRASP_START_ARM_POS
        assert after.rewards.hand_closed_near_object.params["std"] == GRASP_START_CLOSED_NEAR_STD < CLOSED_NEAR_STD

        # everything else is the same: observations, sensors, actions, other rewards
        assert policy_terms(after) == policy_terms(before)
        assert after.observations.policy.history_length == before.observations.policy.history_length
        assert sensor_names(after) == sensor_names(before)
        assert type(after.actions.gripper_action) is type(before.actions.gripper_action)
        assert after.rewards.lifting_object.weight == before.rewards.lifting_object.weight


def test_the_grasp_start_agents_have_their_own_experiment_names():
    from contact_prediction_rl.tasks.lift.config.franka_hx5d20.agents import ppo_cfg

    names = {
        ppo_cfg.FrankaHX5D20GraspStartHistoryPaperPPOCfg().experiment_name,
        ppo_cfg.FrankaHX5D20GraspStartTactilePaperPPOCfg().experiment_name,
        ppo_cfg.FrankaHX5D20GraspStartTactileTRPaperPPOCfg().experiment_name,
        ppo_cfg.FrankaHX5D20GraspStartTactileFRPaperPPOCfg().experiment_name,
    }

    assert len(names) == 4 and all("grasp_start" in n for n in names)
    assert ppo_cfg.FrankaHX5D20GraspStartTactileTRPaperPPOCfg().to_dict()["auxiliary"]["objective"] == "tactile_reconstruction"
    assert ppo_cfg.FrankaHX5D20GraspStartTactileFRPaperPPOCfg().to_dict()["auxiliary"]["objective"] == "full_reconstruction"


def test_the_raw_contact_agents_feed_exactly_the_tactile_part_unnormalised():
    from contact_prediction_rl.tasks.lift.config.franka_hx5d20.agents import ppo_cfg

    for cls in (
        ppo_cfg.FrankaHX5D20GraspStartTactileRawPaperPPOCfg,
        ppo_cfg.FrankaHX5D20GraspStartTactileRawTRPaperPPOCfg,
        ppo_cfg.FrankaHX5D20GraspStartTactileRawFRPaperPPOCfg,
    ):
        cfg = cls().to_dict()
        assert cfg["encoder"]["raw_tail_dim"] == ppo_cfg.TACTILE_DIM
        assert "raw" in cfg["experiment_name"]

    # the other agents keep the normalised contacts
    assert "raw_tail_dim" not in ppo_cfg.FrankaHX5D20GraspStartTactilePaperPPOCfg().to_dict()["encoder"]
