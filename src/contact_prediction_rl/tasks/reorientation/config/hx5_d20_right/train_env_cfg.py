"""Opt-in Reorientation training task requiring a validated Newton grasp snapshot."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from contact_prediction_rl.tasks.reorientation import mdp
from contact_prediction_rl.tasks.reorientation.grasp_snapshot import grasp_snapshot_sha256, load_grasp_snapshot
from contact_prediction_rl.tasks.reorientation.orientation import goal_orientation

from ..paths import REFERENCE_GRASP_PATH
from .env_cfg import ObservationsCfg, ReorientationEnvCfg


@configclass
class TrainPolicyCfg(ObservationsCfg.PolicyCfg):
    object_state = ObsTerm(func=mdp.object_state_palm)
    goal_error = ObsTerm(func=mdp.goal_error_quat, params={"goal_xyzw": (0.0, 0.0, 0.0, 1.0)})
    time_remaining = ObsTerm(func=mdp.remaining_time)


@configclass
class TrainObservationsCfg(ObservationsCfg):
    policy: TrainPolicyCfg = TrainPolicyCfg()


@configclass
class TrainRewardsCfg:
    orientation = RewTerm(func=mdp.orientation_reward, weight=1.0, params={"goal_xyzw": (0.0, 0.0, 0.0, 1.0)})
    success = RewTerm(func=mdp.success_reward, weight=2.0, params={"goal_xyzw": (0.0, 0.0, 0.0, 1.0), "start_pos_w": (0.0, 0.0, 0.0)})
    dropped = RewTerm(func=mdp.drop_penalty, weight=-2.0, params={"start_pos_w": (0.0, 0.0, 0.0)})


@configclass
class TrainTerminationsCfg:
    """Always run the entire six-second episode, even after goal entry or grasp loss."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ReorientationTrainEnvCfg(ReorientationEnvCfg):
    """Fixed +15-degree goal; original 40D diagnostic environment is unchanged."""

    observations: TrainObservationsCfg = TrainObservationsCfg()
    rewards: TrainRewardsCfg = TrainRewardsCfg()
    terminations: TrainTerminationsCfg = TrainTerminationsCfg()
    grasp_snapshot_path: str = str(REFERENCE_GRASP_PATH)
    grasp_snapshot_sha256: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        snapshot = load_grasp_snapshot(self.grasp_snapshot_path)
        self.grasp_snapshot_sha256 = grasp_snapshot_sha256(self.grasp_snapshot_path)
        # Default joints are commanded grip targets; measured joints are restored in an event.
        self.scene.robot.init_state.joint_pos = {**snapshot["arm_joint_pos"], **snapshot["finger_target_pos"]}
        self.scene.object.init_state.pos = tuple(snapshot["cube_pos_w"])
        self.scene.object.init_state.rot = tuple(snapshot["cube_quat_w"])
        for name in ("panda_shoulder", "panda_forearm"):
            self.scene.robot.actuators[name].stiffness = 4000.0
            self.scene.robot.actuators[name].damping = 160.0
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.use_default_offset = True
        self.events.restore_grasp = EventTerm(func=mdp.restore_grasp_joint_state, mode="reset", params={"finger_joint_pos": snapshot["finger_joint_pos"]})
        self.episode_length_s = 6.0
        self.is_finite_horizon = True
        initial = torch.tensor([snapshot["cube_quat_w"]], dtype=torch.float64)
        axis = torch.tensor([snapshot["axis_world"]], dtype=torch.float64)
        goal = tuple(float(x) for x in goal_orientation(initial, axis, torch.tensor([math.radians(15.0)]))[0])
        start = tuple(snapshot["cube_pos_w"])
        self.observations.policy.goal_error.params["goal_xyzw"] = goal
        for term in (self.rewards.orientation, self.rewards.success):
            term.params["goal_xyzw"] = goal
        for term in (self.rewards.success, self.rewards.dropped):
            term.params["start_pos_w"] = start
