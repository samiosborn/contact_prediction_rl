# src/contact_prediction_rl/tasks/reorientation/config/hx5_d20_right/env_cfg.py

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

from contact_prediction_rl.robots.franka_hx5d20 import FRANKA_HX5D20_CFG
from contact_prediction_rl.robots.hx5_d20_contacts import fingertip_contact_sensor_cfg, fingertip_object_contact

from ... import mdp


@configclass
class ReorientationPhysicsCfg(PresetCfg):
    """Physics backends supported by this task"""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()

    physx: PhysxAutoCfg = PhysxAutoCfg(
        isaacsim_physx=isaacsim_physx,
        ovphysx=ovphysx,
    )

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=128,
            nconmax=128,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        # Resolve fast fingertip motion at 480 Hz while retaining the 120 Hz task timestep
        num_substeps=4,
        debug_mode=False,
        use_cuda_graph=True,
    )

    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoPADMMSolverCfg(
            sparse_jacobian=True,
        ),
        debug_mode=False,
        use_cuda_graph=True,
    )

    default: NewtonCfg = newton_mjwarp


@configclass
class ReorientationSceneCfg(InteractiveSceneCfg):
    """Minimal Franka + HX5-D20 scene"""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
    )

    robot = FRANKA_HX5D20_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Retain the hand's bring-up gains; the shared arm PD holds its default posture.
    robot.actuators["hx5_d20_fingers"].stiffness = 1.0
    robot.actuators["hx5_d20_fingers"].damping = 0.05

    # Catch surface 220 mm below the default palm; shifted forward to clear the arm base.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 1.2, 0.04),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.4, 0.45)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.85, 0.0, 0.237824)),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=[
                # Free rigid body: a kinematic cube reports contact but cannot be pushed
                sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
            ],
            collision_props=[
                sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            ],
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Preserve the old cube offset in the palm frame: (-0.049, 0, 0.075) m.
            pos=(0.408795, -0.002175, 0.370399),
        ),
    )

    palm_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ConvertedAsset/Geometry/hx5_d20_right_base",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        force_threshold=0.1,
    )

    thumb_contact = fingertip_contact_sensor_cfg("thumb")
    index_contact = fingertip_contact_sensor_cfg("index")
    middle_contact = fingertip_contact_sensor_cfg("middle")
    ring_contact = fingertip_contact_sensor_cfg("ring")
    little_contact = fingertip_contact_sensor_cfg("little")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=500.0,
        ),
    )


@configclass
class ActionsCfg:
    """Twenty joint-position actions"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_r_joint.*"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Public proprioception and separate privileged fingertip labels."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["finger_r_joint.*"])})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["finger_r_joint.*"])})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class ContactCfg(ObsGroup):
        fingertip_object = ObsTerm(func=fingertip_object_contact)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # A distinct group ensures PPO never treats privileged labels as policy input.
    contact: ContactCfg = ContactCfg()


@configclass
class RewardsCfg:
    """Dummy reward while validating the environment"""

    alive = RewTerm(
        func=mdp.is_alive,
        weight=1.0,
    )


@configclass
class TerminationsCfg:
    """Only time-limit termination for now"""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )


@configclass
class ReorientationEnvCfg(ManagerBasedRLEnvCfg):
    """Minimal Franka + HX5-D20 environment"""

    scene: ReorientationSceneCfg = ReorientationSceneCfg(
        num_envs=1,
        env_spacing=2.0,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        # The policy only writes finger targets; hold the arm at its initial PD targets.
        self.events.reset_scene_to_default.params["reset_joint_targets"] = True

        self.decimation = 4
        self.episode_length_s = 100.0

        self.sim.gravity = (0.0, 0.0, -9.81)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = ReorientationPhysicsCfg()
