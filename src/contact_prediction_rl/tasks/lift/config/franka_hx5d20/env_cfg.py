# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka + HX5-D20 cube-lift environment."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from isaaclab_tasks.contrib.lift import mdp
from isaaclab_tasks.contrib.lift.lift_env_cfg import LiftEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from contact_prediction_rl.robots.franka_hx5d20 import (  # isort: skip
    FRANKA_HX5D20_CFG,
    FRANKA_HX5D20_HAND_CLOSED,
    FRANKA_HX5D20_HAND_CLOSED_THUMB_OPEN,
    FRANKA_HX5D20_HAND_OPEN,
    FRANKA_HX5D20_HAND_SYNERGIES,
)

from ...mdp import rewards
from ...mdp.actions_cfg import RateLimitedBinaryJointPositionActionCfg, SynergyJointPositionActionCfg

CUBE_SIZE = 0.06
"""Edge length [m] of the cube-sized part of the block that the hand holds, which is also the block's footprint."""

BLOCK_HEIGHT = 0.12
"""Height [m] of the block. The hand holds its upper ``CUBE_SIZE`` part.

On the bare table a 6 cm cube cannot be grasped with this hand: the open fingers reach 10-12 cm along the finger axis
from the grasp centre, so with the centre 3 cm above the table the fingertips end up in the table. A block twice as
tall puts the grasp zone 6 cm above the table, where the scripted grasp lifts it (see
``scripts/franka_hx5d20/search_franka_hx5d20_table_grasp.py --block_height 0.12``).
"""

REST_HEIGHT = BLOCK_HEIGHT / 2
"""Height [m] of the block centre while it rests on the table."""

BLOCK_X = 0.45
"""Forward position [m] of the block in the robot base frame; the start pose is solved for it."""

LIFT_HEIGHT = 0.10
"""Height [m] above its resting height to which the block is lifted and held (the fixed goal)."""

GRASP_OFFSET_Z = (BLOCK_HEIGHT - CUBE_SIZE) / 2
"""Height [m] of the held part above the block centre."""

GRASP_CENTER = (0.04, 0.0, 0.09)
"""Position [m] in the ``panda_hand`` frame where the held part sits in the closed hand.

Closer to the palm than ``FRANKA_HX5D20_GRASP_CENTER`` (0.05, 0, 0.10), which was fitted for a cube held in the air:
on the table the block is lifted most reliably from here.
"""

APPROACH_DISTANCE = 0.06
"""Distance [m] the open hand starts back from the grasp pose, along the palm normal (hand x)."""

START_ARM_POS = [0.660, 0.802, -0.197, -1.950, -0.488, 1.280, 2.048]
"""Arm joint values [rad] for the start of an episode: the open hand ``APPROACH_DISTANCE`` back from the grasp pose in
the side-grasp orientation (``HOLDING_GRAVITY``), so the policy approaches the block's face horizontally.

The side grasp has several arm solutions. A first one kept joint 7 within 0.15 rad of its limit: the trained policy
grasped with the wrist pinned there and could not then lift the block straight up, so it held it 20-40 cm from the
goal. This one (joint 1 about +0.6 rad) keeps every joint at least 0.85 rad from its limits at the start, grasp and goal
poses. Found with ``scripts/franka_hx5d20/find_franka_hx5d20_pregrasp.py --cube 0.45,0,0.091 --center 0.10,0,0.09
--gravity 0,-1,0 --hovers 0.0 --start 0.617 0.848 -0.239 -1.87 -0.6 1.305 1.965 --posture_gain 0.3``, where the start
is the grasp solution of that branch. In the scripted check (approach, 2 s close, lift to the goal) it lifts the block
10 times out of 10 with a steady arm, but it is sensitive to arm noise (3-7 of 10 at 0.015-0.04 rad, 0 at 0.07 rad).
"""

GOAL_STD = 0.1
"""Length scale [m] of the goal-tracking reward. A first policy trained with 0.3 m learned to follow the goal sideways
but ignored its forward and height components, whose spread is only about 3-4 cm, so the kernel is narrower here."""

GOAL_BROAD_STD = 0.3
"""Length scale [m] of the broad goal-tracking reward. With only the narrow terms, a policy that overshot the goal while
lifting carried the block 40 cm away and kept it there: beyond about 25 cm the narrow terms pay nothing, so no action
leads back."""

LIFT_WEIGHT = 5.0
"""Weight of the reward for holding the block lifted anywhere (15 in the base task). Lower, so that holding it at the
goal pays clearly more than holding it elsewhere."""

GOAL_FINE_STD = 0.025
"""Length scale [m] of the fine-grained goal-tracking reward."""

HAND_CLOSING_TIME = 2.0
"""Time [s] the hand takes to close. Closing all joints together over about 3 s lifted the block 5 times out of 5 in
the scripted check; closing within 0.1-0.4 s pushed it over or missed it."""

REACHING_WEIGHT = 0.5
REACHING_STD = 0.03
CLOSED_NEAR_WEIGHT = 4.0
CLOSED_NEAR_STD = 0.03
"""Weights and length scales [m] of the reaching and closed-hand rewards.

These two terms have to be balanced against each other; each of the following failed in training:

* reaching 1.0 / 0.1 m with closing 4.0 / 0.05 m: the hand started closing at the first step, 3.5 cm short of the
  grasp point, so the fingertips caught the block and rolled it over;
* reaching 1.0 / 0.03 m with closing 4.0 / 0.015 m: an open hand parked 2 cm from the grasp point paid more than a
  closed one, and the policy never closed the hand.

With these values a closed hand 2 cm from the grasp point pays about eight times as much as an open one there, and a
hand closed 5 cm away pays about a sixth of one closed in place.
"""

LIFT_PROGRESS_WEIGHT = 5.0
"""Weight of the gradual lift reward. The lift reward is on/off at 2 cm, so a policy that had learned to close the
hand on the block kept holding it on the table: a small lift earned nothing, and with little exploration noise left it
never found a 2 cm one."""

ARM_ACTION_SCALE = 0.2
"""Scale [rad] of the arm joint-position actions; smaller than the Franka lift (0.5) so random exploration does not
slam the arm into the cube."""

HOLDING_GRAVITY = (0.0, -1.0, 0.0)
"""Direction of gravity in the hand frame of the side grasp: fingers horizontal, palm facing the block, thumb up.

The tilted palm-down grasp (gravity (0.82, 0, 0.57)) lifts the block about as often in a scripted approach, but it
fails under arm noise of the size a training policy explores with (0 of 10 at 0.07 rad per step), while the side grasp
keeps lifting it (6 of 10). Checked with ``scripts/franka_hx5d20/search_franka_hx5d20_table_grasp.py --approach 0.06
--arm_noise 0.07``.
"""


@configclass
class FrankaHX5D20BlockLiftEnvCfg(LiftEnvCfg):
    """Lift a 6x6x12 cm block from the table and hold it ``LIFT_HEIGHT`` higher, with the arm and hand.

    The policy commands the seven arm joints and one binary open/close action for the whole hand. Closing drives the
    20 finger joints to ``FRANKA_HX5D20_HAND_CLOSED``. The block is grasped from the side (``HOLDING_GRAVITY``):
    episodes start with the open hand beside the block, palm facing it, and end if the block falls off the table.
    """

    def __post_init__(self):
        super().__post_init__()

        # the slow hand closing needs more time than the base task's 5 s episode
        self.episode_length_s = 8.0

        # robot: start with the open hand beside the block, ready to approach it horizontally. Gravity is off for the
        # robot, as in Isaac Lab's Franka high-PD configuration: under the arm's PD gains it would otherwise sag by
        # several centimetres and push the block.
        self.scene.robot = FRANKA_HX5D20_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.rigid_props = PhysxRigidBodyCfg(disable_gravity=True)
        self.scene.robot.init_state.joint_pos = {
            **{f"panda_joint{i + 1}": value for i, value in enumerate(START_ARM_POS)},
            "finger_r_joint.*": 0.0,
        }

        # actions: seven arm joints and one open/close action for the hand
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=ARM_ACTION_SCALE, use_default_offset=True
        )
        self.actions.gripper_action = RateLimitedBinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_r_joint.*"],
            open_command_expr=FRANKA_HX5D20_HAND_OPEN,
            close_command_expr=FRANKA_HX5D20_HAND_CLOSED,
            closing_time=HAND_CLOSING_TIME,
        )

        # table and goal: a local table (no download) whose top is at z = 0, like the reach environment
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
            spawn=sim_utils.CuboidCfg(size=(0.9, 1.3, 1.0), collision_props=sim_utils.UsdPhysicsCollisionCfg()),
        )
        self.commands.object_pose.body_name = "panda_hand"
        # goal: a fixed point straight above where the block starts, so the task is "lift it and hold it there"
        self.commands.object_pose.ranges.pos_x = (BLOCK_X, BLOCK_X)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.ranges.pos_z = (REST_HEIGHT + LIFT_HEIGHT, REST_HEIGHT + LIFT_HEIGHT)

        # object: a block with high friction, standing on the table
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[BLOCK_X, 0.0, REST_HEIGHT + 0.001], rot=[0, 0, 0, 1]),
            spawn=sim_utils.CuboidCfg(
                size=(CUBE_SIZE, CUBE_SIZE, BLOCK_HEIGHT),
                rigid_props=PhysxRigidBodyCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassCfg(mass=0.1),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
            ),
        )
        # small spawn jitter, as the grasp was validated with offsets of 1 cm
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (0.0, 0.0),
        }
        # a toppled block does not end the episode: a training run that terminated on toppling learned to never close
        # the hand, because a failed attempt forfeited the reaching reward of the rest of the episode. Only a block
        # that falls off the table ends it (the base task's threshold).
        self.terminations.object_dropping.params["minimum_height"] = -0.05

        # end-effector frame: the grasp centre of the hand
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=list(GRASP_CENTER)),
                ),
            ],
        )

        # camera: a close view of the first robot and cube, for videos and the viewport
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(1.1, 0.9, 0.7), lookat=(0.45, 0.0, 0.15))

        # no penalty curriculum: raising the action-rate and joint-velocity penalties after 10000 steps (about 400
        # iterations) disturbed every run of this task; the small constant penalties of the base task are kept
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None

        # rewards: count the block as lifted when it is a few centimetres above its resting height. Distances to the
        # object are measured to the held (upper) part of the block, not to its centre.
        lifted = REST_HEIGHT + 0.02
        held = {"grasp_offset_z": GRASP_OFFSET_Z}
        self.rewards.reaching_object = RewTerm(
            func=rewards.grasp_point_reaching, params={"std": REACHING_STD, **held}, weight=REACHING_WEIGHT
        )
        # the lift and goal terms only count while the block is in the hand, so knocking it into the air pays nothing
        in_hand = 0.08
        self.rewards.lifting_object = RewTerm(
            func=rewards.object_lifted_in_hand,
            params={"minimal_height": lifted, "max_distance": in_hand, **held},
            weight=LIFT_WEIGHT,
        )
        self.rewards.object_goal_tracking = RewTerm(
            func=rewards.object_goal_distance_in_hand,
            params={
                "std": GOAL_STD,
                "minimal_height": lifted,
                "command_name": "object_pose",
                "max_distance": in_hand,
                "success_threshold": 0.05,
                **held,
            },
            weight=16.0,
        )
        # gradual lift reward: pays for every millimetre the held block rises, up to the goal height
        self.rewards.lift_progress = RewTerm(
            func=rewards.object_lift_progress_in_hand,
            params={
                "rest_height": REST_HEIGHT,
                "target_height": REST_HEIGHT + LIFT_HEIGHT,
                "max_distance": in_hand,
                **held,
            },
            weight=LIFT_PROGRESS_WEIGHT,
        )
        # broad goal term: still pays far from the goal, so a block carried away is pulled back toward it
        self.rewards.object_goal_tracking_broad = RewTerm(
            func=rewards.object_goal_distance_in_hand,
            params={
                "std": GOAL_BROAD_STD,
                "minimal_height": lifted,
                "command_name": "object_pose",
                "max_distance": in_hand,
                **held,
            },
            weight=8.0,
        )
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=rewards.object_goal_distance_in_hand,
            params={
                "std": GOAL_FINE_STD,
                "minimal_height": lifted,
                "command_name": "object_pose",
                "max_distance": in_hand,
                **held,
            },
            weight=5.0,
        )
        self.rewards.hand_closed_near_object = RewTerm(
            func=rewards.hand_closed_near_object, params={"std": CLOSED_NEAR_STD, **held}, weight=CLOSED_NEAR_WEIGHT
        )
        self.rewards.hand_orientation = RewTerm(
            func=rewards.hand_gravity_alignment,
            params={
                "std": 0.3,
                "target_gravity": HOLDING_GRAVITY,
                "near_std": 0.1,
                **held,
                "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
            },
            weight=0.5,
        )


@configclass
class FrankaHX5D20BlockLiftSynergyEnvCfg(FrankaHX5D20BlockLiftEnvCfg):
    """The block-lift task with a hand that blends hand-designed finger shapes instead of one open/close value.

    The policy commands the seven arm joints and one continuous value per synergy of
    ``FRANKA_HX5D20_HAND_SYNERGIES`` (finger curl, thumb, finger spread). Each synergy closes at the same limited
    rate as the single-closure hand (``HAND_CLOSING_TIME``), and the curl closure is what the closed-hand reward
    pays for. Everything else is as in the single-closure task, so the two can be compared directly.
    """

    def __post_init__(self):
        super().__post_init__()

        self.actions.gripper_action = SynergyJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_r_joint.*"],
            open_command_expr=FRANKA_HX5D20_HAND_OPEN,
            synergies=FRANKA_HX5D20_HAND_SYNERGIES,
            closing_time=HAND_CLOSING_TIME,
        )


@configclass
class FrankaHX5D20BlockLiftCurlEnvCfg(FrankaHX5D20BlockLiftEnvCfg):
    """The block-lift task with the single open/close hand, closing the fingers only and keeping the thumb open.

    This is the control of the synergy task: the policy still has one open/close value and no choice of finger
    shape, and the only difference from :class:`FrankaHX5D20BlockLiftEnvCfg` is the closed posture
    (``FRANKA_HX5D20_HAND_CLOSED_THUMB_OPEN`` instead of ``FRANKA_HX5D20_HAND_CLOSED``).
    """

    def __post_init__(self):
        super().__post_init__()

        self.actions.gripper_action.close_command_expr = FRANKA_HX5D20_HAND_CLOSED_THUMB_OPEN
