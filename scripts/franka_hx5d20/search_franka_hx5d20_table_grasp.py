# scripts/franka_hx5d20/search_franka_hx5d20_table_grasp.py

"""Search for a hand pose and closing posture that lifts a cube lying on a bare table, with scripted motions.

Each candidate is a combination of

* a **gravity direction** in the hand frame (this sets the hand orientation; the rotation about the vertical axis is
  left free),
* a **grasp centre**: where the cube centre sits in the hand frame, and
* a **flexion scale** applied to the finger flexion joints of ``FRANKA_HX5D20_HAND_CLOSED`` (1 = as stored), or
  instead a **synergy closure** ``curl,thumb,spread`` that blends the hand synergies of
  ``FRANKA_HX5D20_HAND_SYNERGIES`` (``--synergies``; the flexion scale and ``--thumbs`` are then ignored).

For every candidate the script solves the arm pose that puts the grasp centre on the cube and a second pose 12 cm
higher (damped least squares inside the simulation), closes the hand on the cube with the cube at several small
offsets, lifts the arm, and counts how many trials keep the cube. The table is the one of the lift task, with its top
at z = 0.

.. code-block:: bash

    uv run python scripts/franka_hx5d20/search_franka_hx5d20_table_grasp.py
    uv run python scripts/franka_hx5d20/search_franka_hx5d20_table_grasp.py \
        --gravities "0,-0.99,0.14" --centers "0.05,-0.03,0.10" --flex_scales "1.0"
    uv run python scripts/franka_hx5d20/search_franka_hx5d20_table_grasp.py \
        --gravities "0,-1,0" --centers "0.04,0,0.09" --block_height 0.12 --approach 0.06 \
        --synergies "1,1,0;1,1,0.5;1,1,1;0.8,1,0.5"

"""


import argparse
import itertools
import time
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject

parser = argparse.ArgumentParser(description="Search a table grasp for the Franka + HX5-D20.")
parser.add_argument(
    "--gravities",
    type=str,
    default="0.82,0,0.57;0.94,0,0.34;0,-0.99,0.14;0,-0.90,0.40;0.5,-0.8,0.3",
    help="Candidate directions of gravity in the hand frame 'gx,gy,gz', separated by ';'.",
)
parser.add_argument(
    "--centers",
    type=str,
    default="0.05,0,0.10",
    help="Candidate cube centres in the hand frame 'x,y,z' [m], separated by ';'.",
)
parser.add_argument(
    "--flex_scales",
    type=str,
    default="1.0",
    help="Scales of the finger flexion of the closed posture, separated by ';'.",
)
parser.add_argument("--finger_velocity", type=float, default=None, help="Hard limit [rad/s] on the speed of the finger joints.")
parser.add_argument(
    "--approach",
    type=float,
    default=0.0,
    help="Start this far [m] back along the palm normal (hand x) and move in before closing, as the policy must.",
)
parser.add_argument(
    "--arm_noise",
    type=float,
    default=0.0,
    help="Standard deviation [rad] of Gaussian noise added to every arm joint target at every step, like the exploration noise of a training policy (action std times the arm action scale).",
)
parser.add_argument(
    "--ik_start",
    type=float,
    nargs=7,
    default=None,
    help="Arm posture [rad] to solve from first, to stay on one branch of the arm's inverse kinematics.",
)
parser.add_argument("--lift_height", type=float, default=0.12, help="Height [m] the arm lifts the object by.")
parser.add_argument(
    "--thumbs",
    type=str,
    default=None,
    help="Candidate thumb joint values 'j1,j2,j3,j4' [rad] replacing those of the closed posture, separated by ';'.",
)
parser.add_argument(
    "--synergies",
    type=str,
    default=None,
    help="Candidate synergy closures 'curl,thumb,spread' in [0, 1], separated by ';', replacing the closed posture (1,1,0 is FRANKA_HX5D20_HAND_CLOSED).",
)
parser.add_argument(
    "--lift_after",
    type=int,
    default=None,
    help="Start lifting this many steps after the hand starts closing, before it has finished closing.",
)
parser.add_argument("--block_yaw", type=float, default=0.0, help="Rotation [deg] of the object about the vertical.")
parser.add_argument("--cube_size", type=float, default=0.06, help="Cube edge length [m].")
parser.add_argument(
    "--block_height",
    type=float,
    default=None,
    help="Height [m] of a taller block with the cube's footprint; the hand grasps its upper cube-sized part.",
)
parser.add_argument("--cube_xy", type=float, nargs=2, default=[0.45, 0.0], help="Cube position [m].")
parser.add_argument("--offsets", type=str, default="0,0;0.01,0;-0.01,0;0,0.01;0,-0.01", help="Cube offsets 'dx,dy' [m].")
parser.add_argument("--close_steps", type=int, default=90, help="Steps used to close the hand.")
parser.add_argument("--ik_iterations", type=int, default=100, help="Solver iterations per pose.")
add_launcher_args(parser)
args_cli = parser.parse_args()
# The USD joints are authored for PhysX, so use Isaac Sim's PhysX backend (which needs Kit).
args_cli.require_kit = True
args_cli.physics = "isaacsim_physx"


import math
import re

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.physics import PhysicsCfg
from isaaclab.sim import SimulationContext
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from contact_prediction_rl.robots.franka_hx5d20 import FRANKA_HX5D20_CFG  # isort:skip
from contact_prediction_rl.robots.franka_hx5d20 import FRANKA_HX5D20_HAND_CLOSED  # isort:skip
from contact_prediction_rl.robots.franka_hx5d20 import FRANKA_HX5D20_HAND_SYNERGIES, resolve_joint_posture  # isort:skip

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
READY_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
EXTRA_STARTS = [
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 2.2],
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, -0.8],
    [0.0, -0.4, 0.0, -2.0, 1.5, 1.5, 0.8],
    [0.0, -0.4, 0.0, -2.0, -1.5, 1.5, 0.8],
    [0.5, -0.4, 0.5, -2.0, 0.0, 1.8, 1.5],
    [-0.5, -0.4, -0.5, -2.0, 0.0, 1.8, 0.0],
    [0.0, 0.4, 0.0, -1.8, 0.0, 2.2, 2.0],
]
"""Alternative starting postures tried when the first solve fails; the hand orientations of interest can be on a
different branch of the arm's inverse kinematics."""
FLEXION_JOINTS = {f"finger_r_joint{i}" for i in (6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20)}
SETTLE_STEPS, HOLD_STEPS, LIFT_STEPS = 60, 30, 90


def parse_list(text: str) -> list[tuple[float, ...]]:
    return [tuple(float(v) for v in item.split(",")) for item in text.split(";")]


class Rig:
    """The robot, cube and table, with helpers to solve arm poses and run one scripted grasp."""

    def __init__(self, sim: SimulationContext, robot: "Articulation", cube: "RigidObject"):
        self.sim, self.robot, self.cube = sim, robot, cube
        self.dt = sim.get_physics_dt()
        self.device = sim.device
        self.hand_id = robot.body_names.index("hx5_d20_right_base")
        self.arm_ids = [robot.joint_names.index(n) for n in ARM_JOINTS]
        self.limits = robot.data.joint_pos_limits.torch[0][self.arm_ids]
        self.height = args_cli.block_height or args_cli.cube_size
        self.rest_z = self.height / 2 + 0.001
        # the grasp centre sits on the upper cube-sized part of the block
        self.grasp_z = self.rest_z + (self.height - args_cli.cube_size) / 2
        self.down = torch.tensor([0.0, 0.0, -1.0], device=self.device)

    # -- low level ---------------------------------------------------------------------------------------------

    def step(self, target: torch.Tensor, cube_pose: torch.Tensor | None = None, noisy: bool = False) -> None:
        if noisy and args_cli.arm_noise > 0.0:
            target = target.clone()
            target[0, self.arm_ids] += args_cli.arm_noise * torch.randn(len(self.arm_ids), device=self.device)
        self.robot.actuators.target_command.set_position_index(value=target, full_data=True)
        if cube_pose is not None:
            self.cube.write_root_pose_to_sim_index(root_pose=cube_pose)
            self.cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.device))
        self.robot.write_data_to_sim()
        self.cube.write_data_to_sim()
        self.sim.step()
        self.robot.update(self.dt)
        self.cube.update(self.dt)

    def hand_pose(self) -> torch.Tensor:
        return self.robot.data.body_pose_w.torch[0, self.hand_id]

    def centre_world(self, centre: torch.Tensor) -> torch.Tensor:
        pose = self.hand_pose()
        return pose[:3] + math_utils.quat_apply(pose[3:7], centre)

    # -- inverse kinematics ------------------------------------------------------------------------------------

    def solve(self, goal: torch.Tensor, gravity: torch.Tensor, centre: torch.Tensor, start: torch.Tensor):
        """Arm joint values that put ``centre`` (hand frame) on ``goal`` with gravity along ``gravity`` (hand frame).

        Returns the 7 arm joint values and the position [m] and gravity-direction errors.
        """
        far_away = torch.tensor([0.0, 3.0, 0.5, 0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0)
        q = self.robot.data.default_joint_pos.torch.clone()
        q[0, self.arm_ids] = start
        reference = start.clone()

        def error(q_full: torch.Tensor) -> torch.Tensor:
            self.robot.write_joint_position_to_sim_index(position=q_full)
            self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q_full))
            self.step(q_full, far_away)
            pose = self.hand_pose()
            g = math_utils.quat_apply_inverse(pose[3:7], self.down)
            return torch.cat([goal - (pose[:3] + math_utils.quat_apply(pose[3:7], centre)), gravity - g])

        for _ in range(args_cli.ik_iterations):
            err = error(q)
            jac = torch.zeros(6, 7, device=self.device)
            for k, j in enumerate(self.arm_ids):
                q_pert = q.clone()
                q_pert[0, j] += 1e-3
                jac[:, k] = -(error(q_pert) - err) / 1e-3
            pinv = jac.T @ torch.linalg.inv(jac @ jac.T + 0.02**2 * torch.eye(6, device=self.device))
            posture = (torch.eye(7, device=self.device) - pinv @ jac) @ (reference - q[0, self.arm_ids])
            q[0, self.arm_ids] = torch.clamp(q[0, self.arm_ids] + 0.7 * (pinv @ err + 0.1 * posture), self.limits[:, 0], self.limits[:, 1])
        err = error(q)
        return q[0, self.arm_ids].clone(), err[:3].norm().item(), err[3:].norm().item()

    def solve_best(self, goal, gravity, centre, starts):
        """Try the starting postures in order and return the first good solution, or the best one found."""
        best = None
        for start in starts:
            result = self.solve(goal, gravity, centre, start)
            if best is None or result[1] + result[2] < best[1] + best[2]:
                best = result
            if result[1] < 0.005 and result[2] < 0.05:
                break
        return best

    # -- one scripted grasp ------------------------------------------------------------------------------------

    def closed_target(self, arm: torch.Tensor, flex_scale: float, thumb=None, synergy=None) -> torch.Tensor:
        target = self.robot.data.default_joint_pos.torch.clone()
        target[0, self.arm_ids] = arm
        if synergy is not None:
            for closure, posture in zip(synergy, FRANKA_HX5D20_HAND_SYNERGIES.values(), strict=True):
                target[0] += closure * torch.tensor(resolve_joint_posture(posture, self.robot.joint_names), device=self.device)
            return target
        for pattern, value in FRANKA_HX5D20_HAND_CLOSED.items():
            for j, name in enumerate(self.robot.joint_names):
                if re.fullmatch(pattern, name):
                    target[0, j] = value * (flex_scale if name in FLEXION_JOINTS else 1.0)
        if thumb is not None:
            for k, value in enumerate(thumb):
                target[0, self.robot.joint_names.index(f"finger_r_joint{k + 1}")] = value
        return target

    def trial(
        self,
        grasp_arm,
        lift_arm,
        flex_scale: float,
        centre: torch.Tensor,
        offset,
        approach_arm=None,
        thumb=None,
        synergy=None,
    ) -> dict:
        open_pose = self.robot.data.default_joint_pos.torch.clone()
        open_pose[0, self.arm_ids] = grasp_arm
        start_pose = open_pose.clone()
        if approach_arm is not None:
            start_pose[0, self.arm_ids] = approach_arm
        close = self.closed_target(grasp_arm, flex_scale, thumb, synergy)
        lift = self.closed_target(lift_arm, flex_scale, thumb, synergy)
        cube_pos = torch.tensor([args_cli.cube_xy[0] + offset[0], args_cli.cube_xy[1] + offset[1], self.rest_z], device=self.device)
        half_yaw = math.radians(args_cli.block_yaw) / 2
        yaw_quat = torch.tensor([0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)], device=self.device)  # x, y, z, w
        cube_pose = torch.cat([cube_pos, yaw_quat]).unsqueeze(0)

        self.robot.write_joint_position_to_sim_index(position=start_pose)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(start_pose))
        self.robot.reset()
        self.cube.write_root_pose_to_sim_index(root_pose=cube_pose)
        self.cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.device))
        self.cube.reset()
        for _ in range(SETTLE_STEPS):
            self.step(start_pose, noisy=True)
        # move the open hand in from the approach pose
        for k in range(SETTLE_STEPS):
            self.step(start_pose + min(1.0, (k + 1) / (0.8 * SETTLE_STEPS)) * (open_pose - start_pose), noisy=True)
        start = self.cube.data.root_pos_w.torch[0].clone()
        if args_cli.lift_after is None:
            for k in range(args_cli.close_steps):
                self.step(open_pose + min(1.0, (k + 1) / (0.8 * args_cli.close_steps)) * (close - open_pose), noisy=True)
            for _ in range(HOLD_STEPS):
                self.step(close, noisy=True)
            after_close = self.cube.data.root_pos_w.torch[0].clone()
            for k in range(LIFT_STEPS):
                self.step(close + min(1.0, (k + 1) / (0.8 * LIFT_STEPS)) * (lift - close), noisy=True)
        else:
            # the hand keeps closing while the arm already lifts
            hand_ids = [j for j in range(open_pose.shape[1]) if j not in self.arm_ids]
            total = max(args_cli.close_steps, args_cli.lift_after + LIFT_STEPS)
            after_close = None
            for k in range(total):
                closure = min(1.0, (k + 1) / (0.8 * args_cli.close_steps))
                lifting = 0.0 if k < args_cli.lift_after else min(1.0, (k + 1 - args_cli.lift_after) / (0.8 * LIFT_STEPS))
                target = open_pose.clone()
                target[0, hand_ids] = (open_pose + closure * (close - open_pose))[0, hand_ids]
                target[0, self.arm_ids] = (close + lifting * (lift - close))[0, self.arm_ids]
                self.step(target, noisy=True)
                if k == args_cli.lift_after:
                    after_close = self.cube.data.root_pos_w.torch[0].clone()
        for _ in range(HOLD_STEPS):
            self.step(lift, noisy=True)
        final = self.cube.data.root_pos_w.torch[0]
        rise = 1000 * (final[2] - start[2]).item()
        to_hand = 1000 * (final - self.centre_world(centre)).norm().item() - 1000 * (self.grasp_z - self.rest_z)
        up = math_utils.quat_apply(self.cube.data.root_quat_w.torch[0], torch.tensor([0.0, 0.0, 1.0], device=self.device))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, up[2].item()))))
        return {
            "pushed": 1000 * (after_close - start).norm().item(),
            "rise": rise,
            "lifted": rise > 50 and to_hand < 50,
            "tilt": tilt,
        }

    def palm_yaw(self, arm: torch.Tensor) -> float:
        """Angle [deg] between the horizontal part of the palm normal and the world x axis, at the given arm pose."""
        q = self.robot.data.default_joint_pos.torch.clone()
        q[0, self.arm_ids] = arm
        self.robot.write_joint_position_to_sim_index(position=q)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        self.step(q)
        normal = math_utils.quat_apply(self.hand_pose()[3:7], torch.tensor([1.0, 0.0, 0.0], device=self.device))
        return math.degrees(math.atan2(normal[1].item(), normal[0].item()))


def main():
    gravities = parse_list(args_cli.gravities)
    centres = parse_list(args_cli.centers)
    flex_scales = [v[0] for v in parse_list(args_cli.flex_scales)]
    offsets = parse_list(args_cli.offsets)

    with launch_simulation(PhysicsCfg(), args_cli) as physics_cfg:
        # imported after the app has started: importing these classes earlier breaks Kit's startup
        from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg

        sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, physics=physics_cfg))
        sim.set_camera_view([1.2, 0.9, 0.7], [0.45, 0.0, 0.1])
        light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)
        robot_cfg = FRANKA_HX5D20_CFG.copy()
        robot_cfg.prim_path = "/World/Robot"
        robot_cfg.spawn.rigid_props = PhysxRigidBodyCfg(disable_gravity=True)  # as in the lift task
        if args_cli.finger_velocity is not None:
            robot_cfg.actuators["hx5_d20_fingers"].joint_velocity_limit = args_cli.finger_velocity
        robot = Articulation(cfg=robot_cfg)
        cube = RigidObject(
            cfg=RigidObjectCfg(
                prim_path="/World/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(args_cli.cube_size, args_cli.cube_size, args_cli.block_height or args_cli.cube_size),
                    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                    mass_props=sim_utils.MassCfg(mass=0.1),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(args_cli.cube_xy[0], args_cli.cube_xy[1], 0.031)),
            )
        )
        table_cfg = sim_utils.CuboidCfg(size=(0.9, 1.3, 1.0), collision_props=sim_utils.UsdPhysicsCollisionCfg())
        table_cfg.func("/World/Table", table_cfg, translation=(0.5, 0.0, -0.5))
        sim.reset()
        rig = Rig(sim, robot, cube)

        cube_centre = torch.tensor([args_cli.cube_xy[0], args_cli.cube_xy[1], rig.grasp_z], device=sim.device)
        lift_goal = cube_centre + torch.tensor([0.0, 0.0, args_cli.lift_height], device=sim.device)
        ready = torch.tensor(READY_POSE, device=sim.device)

        print(f"\n{'gravity (hand frame)':<22}{'cube centre (hand)':<22}{'flex':>5}  {'IK err':>14}  {'lifted':>7} {'pushed [mm]':>12} {'rise [mm]':>10}")
        results = []
        thumbs = [None] if args_cli.thumbs is None else parse_list(args_cli.thumbs)
        synergies = [None] if args_cli.synergies is None else parse_list(args_cli.synergies)
        if args_cli.synergies is not None:
            flex_scales, thumbs = [1.0], [None]
        for gravity, centre, flex_scale, thumb, synergy in itertools.product(gravities, centres, flex_scales, thumbs, synergies):
            started = time.time()
            g = torch.tensor(gravity, device=sim.device)
            g = g / g.norm()
            c = torch.tensor(centre, device=sim.device)
            starts = [ready] + [torch.tensor(q, device=sim.device) for q in EXTRA_STARTS]
            if args_cli.ik_start is not None:
                starts = [torch.tensor(args_cli.ik_start, device=sim.device)]
            grasp_arm, e1, a1 = rig.solve_best(cube_centre, g, c, starts)
            lift_arm, e2, a2 = rig.solve_best(lift_goal, g, c, [grasp_arm] + starts)
            ik = f"{1000 * max(e1, e2):.1f}mm/{max(a1, a2):.2f}"
            if max(e1, e2) > 0.005 or max(a1, a2) > 0.05:
                print(f"{str(gravity):<22}{str(centre):<22}{flex_scale:>5}  {ik:>14}  no IK solution")
                continue
            approach_arm = None
            if args_cli.approach > 0.0:
                back = c + torch.tensor([args_cli.approach, 0.0, 0.0], device=sim.device)
                approach_arm, e3, a3 = rig.solve_best(cube_centre, g, back, [grasp_arm] + starts)
                if e3 > 0.005 or a3 > 0.05:
                    print(f"{str(gravity):<22}{str(centre):<22}{flex_scale:>5}  no IK solution for the approach pose")
                    continue
            yaw = rig.palm_yaw(grasp_arm)
            trials = [rig.trial(grasp_arm, lift_arm, flex_scale, c, o, approach_arm, thumb, synergy) for o in offsets]
            lifted = sum(t["lifted"] for t in trials)
            upright = sum(t["lifted"] and t["tilt"] < 10.0 for t in trials)
            tilts = sorted(t["tilt"] for t in trials if t["lifted"])
            tilt_txt = f"tilt median {tilts[len(tilts) // 2]:.0f} deg" if tilts else "tilt n/a"
            pushed = sum(t["pushed"] for t in trials) / len(trials)
            rise = sum(t["rise"] for t in trials) / len(trials)
            results.append((lifted, gravity, centre, flex_scale, grasp_arm.tolist(), lift_arm.tolist()))
            print(
                f"{str(gravity):<22}{str(centre):<22}{flex_scale:>5}  {ik:>14}  "
                f"{lifted:>3}/{len(trials)}  {pushed:>12.1f} {rise:>10.1f}"
                f"   upright {upright}/{len(trials)}, {tilt_txt}, palm yaw {yaw:.0f} deg"
                f"{'' if thumb is None else f', thumb {thumb}'}"
                f"{'' if synergy is None else f', synergy {synergy}'}"
                f"   ({time.time() - started:.0f} s)"
            )

        if results:
            best = max(results, key=lambda r: r[0])
            print(f"\nbest: {best[0]}/{len(offsets)} lifted with gravity {best[1]}, cube centre {best[2]}, flex scale {best[3]}")
            print(f"  grasp arm [rad]: {[round(v, 3) for v in best[4]]}")
            print(f"  lift arm  [rad]: {[round(v, 3) for v in best[5]]}")


if __name__ == "__main__":
    main()
