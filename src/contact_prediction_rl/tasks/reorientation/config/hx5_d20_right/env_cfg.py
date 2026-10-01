from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

from ... import mdp


ASSET_DIR = (
    Path(__file__).resolve().parents[4]
    / "assets"
    / "data"
    / "hx5_d20"
    / "right"
)

HX5_USD_PATH = (
    ASSET_DIR
    / "usd"
    / "hx5_d20_right"
    / "hx5_d20_right.usda"
)


@configclass
class ReorientationPhysicsCfg(PresetCfg):
    """Physics backends supported by this task."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()

    physx: PhysxAutoCfg = PhysxAutoCfg(
        isaacsim_physx=isaacsim_physx,
        ovphysx=ovphysx,
    )

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=64,
            nconmax=128,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
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
    """Minimal HX5-D20 scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(HX5_USD_PATH),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={
                "finger_r_joint.*": 0.0,
            },
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["finger_r_joint.*"],
                stiffness=100.0,
                damping=1.0,
            ),
        },
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.07, 0.0, 0.61),
        ),
    )

    thumb_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/hx5_d20_right_base/finger_r_link1/finger_r_link2/finger_r_link3/finger_r_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        force_threshold=0.1,
    )

    index_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/hx5_d20_right_base/finger_r_link5/finger_r_link6/finger_r_link7/finger_r_link8",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        force_threshold=0.1,
    )

    middle_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/hx5_d20_right_base/finger_r_link9/finger_r_link10/finger_r_link11/finger_r_link12",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        force_threshold=0.1,
    )

    ring_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/hx5_d20_right_base/finger_r_link13/finger_r_link14/finger_r_link15/finger_r_link16",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        force_threshold=0.1,
    )

    little_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/hx5_d20_right_base/finger_r_link17/finger_r_link18/finger_r_link19/finger_r_link20",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        force_threshold=0.1,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=500.0,
        ),
    )


@configclass
class ActionsCfg:
    """Twenty joint-position actions."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_r_joint.*"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """For now expose only hand proprioception."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Dummy reward while validating the environment."""

    alive = RewTerm(
        func=mdp.is_alive,
        weight=1.0,
    )


@configclass
class TerminationsCfg:
    """Only time-limit termination for now."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )


@configclass
class ReorientationEnvCfg(ManagerBasedRLEnvCfg):
    """Minimal HX5-D20 environment."""

    scene: ReorientationSceneCfg = ReorientationSceneCfg(
        num_envs=1,
        env_spacing=2.0,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 100.0

        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = ReorientationPhysicsCfg()
