# scripts/franka_hx5d20/find_franka_hx5d20_pregrasp.py

"""Find arm joint values that put the grasp centre of the HX5-D20 hand on a cube.

The grasp centre (``FRANKA_HX5D20_GRASP_CENTER`` in the ``panda_hand`` frame) is moved to the cube position plus a
hover height. The hand orientation is constrained through the direction of gravity in the hand frame (the fingers point
``--tilt`` degrees below the horizontal and the palm faces down), which is what decides whether the cube stays held.
The rotation about the vertical axis is left free.
The solver is damped least squares with a finite-difference Jacobian evaluated in the simulation itself, so no
kinematic model is duplicated here.

.. code-block:: bash

    uv run python scripts/franka_hx5d20/find_franka_hx5d20_pregrasp.py

"""

"""Parse the command-line arguments first."""

import argparse
import math

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Inverse kinematics for the grasp pose of the Franka + HX5-D20.")
parser.add_argument("--cube", type=str, default="0.45,0.0,0.03", help="Cube centre 'x,y,z' [m] in the robot base frame.")
parser.add_argument("--hovers", type=float, nargs="+", default=[0.15, 0.0], help="Heights [m] above the cube to solve for.")
parser.add_argument("--tilt", type=float, default=35.0, help="Angle [deg] of the finger axis below the horizontal.")
parser.add_argument(
    "--gravity",
    type=str,
    default=None,
    help="Target direction 'gx,gy,gz' of gravity in the hand frame; overrides --tilt (e.g. '0,-1,0' for a side grasp).",
)
parser.add_argument(
    "--start",
    type=float,
    nargs=7,
    default=[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    help="Arm joint values [rad] the solver starts from.",
)
parser.add_argument(
    "--center",
    type=str,
    default=None,
    help="Point 'x,y,z' [m] in the hand frame to place on the target; defaults to FRANKA_HX5D20_GRASP_CENTER.",
)
parser.add_argument(
    "--reference",
    type=float,
    nargs=7,
    default=None,
    help="Arm posture [rad] the spare freedom of the arm is pulled toward; defaults to --start.",
)
parser.add_argument("--posture_gain", type=float, default=0.1, help="Strength of the pull toward the reference posture.")
parser.add_argument("--iterations", type=int, default=80, help="Solver iterations per pose.")
add_launcher_args(parser)
args_cli = parser.parse_args()
# The USD joints are authored for PhysX, so use Isaac Sim's PhysX backend (which needs Kit).
args_cli.require_kit = True
args_cli.physics = "isaacsim_physx"

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.physics import PhysicsCfg
from isaaclab.sim import SimulationContext

from contact_prediction_rl.robots.franka_hx5d20 import FRANKA_HX5D20_CFG  # isort:skip
from contact_prediction_rl.robots.franka_hx5d20 import FRANKA_HX5D20_GRASP_CENTER  # isort:skip

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def main():
    cube = torch.tensor([float(v) for v in args_cli.cube.split(",")])
    with launch_simulation(PhysicsCfg(), args_cli) as physics_cfg:
        sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, physics=physics_cfg))
        cfg = FRANKA_HX5D20_CFG.copy()
        cfg.prim_path = "/World/Robot"
        robot = Articulation(cfg=cfg)
        sim.reset()
        dt = sim.get_physics_dt()
        device = sim.device
        hand_id = robot.body_names.index("hx5_d20_right_base")
        arm_ids = [robot.joint_names.index(n) for n in ARM_JOINTS]
        limits = robot.data.joint_pos_limits.torch[0][arm_ids]
        centre = [float(v) for v in args_cli.center.split(",")] if args_cli.center else FRANKA_HX5D20_GRASP_CENTER
        grasp_centre = torch.tensor(centre, device=device)
        cube = cube.to(device)

        # target gravity direction in the hand frame: fingers (hand z) tilted down, palm (hand x) facing down
        tilt = math.radians(args_cli.tilt)
        down = torch.tensor([0.0, 0.0, -1.0], device=device)
        target_gravity = torch.tensor([math.cos(tilt), 0.0, math.sin(tilt)], device=device)
        if args_cli.gravity:
            target_gravity = torch.tensor([float(v) for v in args_cli.gravity.split(",")], device=device)
            target_gravity = target_gravity / target_gravity.norm()

        q = robot.data.default_joint_pos.torch.clone()
        for j, value in zip(arm_ids, args_cli.start, strict=True):
            q[0, j] = value

        def evaluate(q_full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Grasp centre (world) and hand orientation for the given joint values."""
            robot.write_joint_position_to_sim_index(position=q_full)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q_full))
            robot.actuators.target_command.set_position_index(value=q_full, full_data=True)
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
            pose = robot.data.body_pose_w.torch[0, hand_id]
            return pose[:3] + math_utils.quat_apply(pose[3:7], grasp_centre), pose[3:7]

        def error(q_full: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
            centre, quat = evaluate(q_full)
            gravity = math_utils.quat_apply_inverse(quat, down)
            return torch.cat([goal - centre, target_gravity - gravity])

        reference = q[0, arm_ids].clone()
        if args_cli.reference is not None:
            reference = torch.tensor(args_cli.reference, device=device)
        for hover in args_cli.hovers:
            goal = cube + torch.tensor([0.0, 0.0, hover], device=device)
            for i in range(args_cli.iterations):
                err = error(q, goal)
                if i % 20 == 0:
                    print(f"  iter {i}: position error {err[:3].norm() * 1000:.1f} mm, gravity-direction error {err[3:].norm():.3f} rad")
                jac = torch.zeros(6, 7, device=device)
                for k, j in enumerate(arm_ids):
                    q_pert = q.clone()
                    q_pert[0, j] += 1e-3
                    # the error decreases as the pose moves toward the goal, so d(pose)/dq = -d(error)/dq
                    jac[:, k] = -(error(q_pert, goal) - err) / 1e-3
                pinv = jac.T @ torch.linalg.inv(jac @ jac.T + 0.02**2 * torch.eye(6, device=device))
                # spare freedom of the 7-DoF arm: pull toward the start posture without disturbing the pose
                posture = (torch.eye(7, device=device) - pinv @ jac) @ (reference - q[0, arm_ids])
                dq = pinv @ err + args_cli.posture_gain * posture
                q[0, arm_ids] = torch.clamp(q[0, arm_ids] + 0.7 * dq, limits[:, 0], limits[:, 1])
            err = error(q, goal)
            pose = robot.data.body_pose_w.torch[0]
            gravity = math_utils.quat_apply_inverse(pose[hand_id, 3:7], torch.tensor([0.0, 0.0, -1.0], device=device))
            names = robot.body_names
            lowest = min((pose[i, 2].item(), names[i]) for i in range(len(names)))
            print(f"\nhover {hover:.2f} m: position error {err[:3].norm() * 1000:.1f} mm, gravity-direction error {err[3:].norm():.3f} rad")
            print(f"  arm joints [rad]: {[round(v, 3) for v in q[0, arm_ids].tolist()]}")
            margins = torch.minimum(q[0, arm_ids] - limits[:, 0], limits[:, 1] - q[0, arm_ids])
            print(f"  margin to joint limits [rad]: {[round(v, 2) for v in margins.tolist()]}")
            print(f"  gravity in hand frame: {[round(v, 2) for v in gravity.tolist()]}")
            print(f"  lowest body origin: z = {lowest[0]:.3f} m ({lowest[1]})")


if __name__ == "__main__":
    main()
