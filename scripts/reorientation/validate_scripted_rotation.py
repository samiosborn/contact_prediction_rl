# scripts/reorientation/validate_scripted_rotation.py
"""Search for an unsupported free-cube grasp, then test finger-only rotation.

Standalone Newton MJWarp feasibility experiment, NOT an RL task or training
reset. A one-time IK solve configures the Franka with a flat, palm-up hand.
The free 5-cm cube starts in the palm. Optional --use-pedestal can temporarily
support the dynamic cube during closure; without that flag no support is used.
Only fingers move during grasp/rotation; the cube is never moved after setup.

Run:
    uv run python scripts/reorientation/validate_scripted_rotation.py --visualizer newton_gl
    uv run python scripts/reorientation/validate_scripted_rotation.py --max-grasps 6 --max-rotation-trials 8

A FEASIBILITY_INCOMPLETE result is useful: it reports failure without
misrepresenting joint commands as actual object rotation.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import contact_prediction_rl.tasks  # noqa: E402, F401
from contact_prediction_rl.tasks.reorientation.feasibility import (  # noqa: E402
    ARM_IK_SEED,
    FINGERS_FORWARD_WORLD,
    PALM_NORMAL_WORLD,
    GRASP_CENTRES,
    ROLL_PATTERNS,
    GraspCandidate,
    angle_error_deg,
    apply_rotation_pattern,
    axis_angle_xyzw,
    grasp_targets,
    quat_mul_xyzw,
    rotate_vector_xyzw,
    signed_twist_deg,
)

TASK = "ContactPredictionRl-Reorientation-HX5-D20-Right"
CUBE_SIDE_M = 0.05
SUPPORT_WIDTH_M = 0.025
SUPPORT_HEIGHT_M = 0.04
SUPPORT_CLEARANCE_M = 0.0005
# Default rotation goal is around the stationary hand's +X (palm normal).
LOCAL_ROTATION_AXIS = (1.0, 0.0, 0.0)


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def _xyz(values):
    return tuple(float(x) for x in values)


def _solve_palm_up_pose(env):
    """Find a reachable flat-hand arm pose using compiled Newton MuJoCo FK.

    Optimisation is offline in scratch MjData. No stepping a simulated object,
    and no robot-arm actions during grasp or rotation. Update the arm's reset
    targets once; subsequent env.reset() restores the same fixed posture.
    """
    import mujoco
    import numpy as np
    from scipy.optimize import least_squares

    from isaaclab_newton.physics import NewtonManager

    robot = env.scene["robot"]
    solver = NewtonManager._solver
    if solver is None or not hasattr(solver, "mj_model"):
        raise RuntimeError("Palm-up IK requires the Newton MJWarp solver")

    model = solver.mj_model
    labels = NewtonManager.get_model().body_label
    mj_to_newton = solver.mjc_body_to_newton.numpy().reshape(-1)
    hand_matches = [i for i, label in enumerate(labels) if label.endswith("/hx5_d20_right_base")]
    if len(hand_matches) != 1:
        raise RuntimeError(f"Could not identify the single HX5-D20 palm body: {hand_matches}")
    mujoco_hands = np.flatnonzero(mj_to_newton == hand_matches[0])
    if len(mujoco_hands) != 1:
        raise RuntimeError(f"Ambiguous compiled palm geometry: {mujoco_hands.tolist()}")
    hand_id = int(mujoco_hands[0])

    names = [f"panda_joint{i}" for i in range(1, 8)]
    mj_joints = []
    art_joints = []
    for i, name in enumerate(names, start=1):
        matching = [j for j, label in enumerate(labels) if label.endswith(f"/panda_link{i}")]
        if len(matching) != 1:
            raise RuntimeError(f"Cannot map arm link panda_link{i}: {matching}")
        bodies = np.flatnonzero(mj_to_newton == matching[0])
        candidates = [
            j for b in bodies
            for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b] + model.body_jntnum[b]))
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
        ]
        if len(candidates) != 1 or name not in robot.joint_names:
            raise RuntimeError(f"Cannot identify single arm joint {name}: {candidates}")
        mj_joints.append(candidates[0])
        art_joints.append(robot.joint_names.index(name))

    q_start = _tensor(robot.data.default_joint_pos)[0, art_joints].detach().cpu().numpy().astype(float)
    limits = _tensor(robot.data.joint_pos_limits)[0, art_joints].detach().cpu().numpy()
    lower, upper = limits[:, 0] + 0.005, limits[:, 1] - 0.005
    qadr = np.array([model.jnt_qposadr[j] for j in mj_joints], dtype=int)
    scratch = mujoco.MjData(model)
    # Initialise finger joints from the reset pose; avoid solving a different
    # kinematic chain than the actual open-hand simulation.
    for j in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if joint_name is None:
            continue
        short = joint_name.split("/")[-1]
        if short.startswith("finger_r_joint") and short in robot.joint_names:
            scratch.qpos[int(model.jnt_qposadr[j])] = float(
                _tensor(robot.data.default_joint_pos)[0, robot.joint_names.index(short)]
            )

    def fk(q):
        scratch.qpos[qadr] = q
        mujoco.mj_forward(model, scratch)
        rotation = np.asarray(scratch.xmat[hand_id]).reshape(3, 3)
        return np.asarray(scratch.xpos[hand_id]).copy(), rotation[:, 0].copy(), rotation[:, 2].copy()

    # Verify that scratch kinematics and live Isaac Lab use the same frame.
    _, initial_normal, initial_forward = fk(q_start)
    link_indices = [i for i, name in enumerate(robot.body_names) if name.split("/")[-1] == "hx5_d20_right_base"]
    if len(link_indices) != 1:
        raise RuntimeError("Cannot resolve HX5-D20 palm link in live articulation")
    live = _tensor(robot.data.body_link_pose_w)[0, link_indices[0]]
    live_rotation = torch.stack((
        quat_apply(live[3:7][None, :], torch.tensor([[1., 0., 0.]], device=env.device))[0],
        quat_apply(live[3:7][None, :], torch.tensor([[0., 0., 1.]], device=env.device))[0],
    )).detach().cpu().numpy()
    fk_error = max(float(np.linalg.norm(initial_normal - live_rotation[0])),
                   float(np.linalg.norm(initial_forward - live_rotation[1])))
    print(f"PALM_UP_FK_CHECK initial_frame_error={fk_error:.5f}", flush=True)
    if fk_error > 0.15:
        raise RuntimeError("Compiled Newton FK disagrees with live palm frame; refusing unverified IK")

    palm_up = np.array(PALM_NORMAL_WORLD, dtype=float)
    fingers_forward = np.array(FINGERS_FORWARD_WORLD, dtype=float)
    # Target the same reachable working volume as the original experiment,
    # keeping the palm away from the table and the fingers pointing horizontally.
    positions = (np.array((0.475, 0.0, 0.50)),
                 np.array((0.475, 0.0, 0.55)),
                 np.array((0.425, 0.0, 0.55)))
    seeds = [q_start, np.array((0.660, 0.802, -0.197, -1.950, -0.488, 1.280, 2.048))]
    for delta in (-0.6, 0.6):
        seed = q_start.copy()
        seed[6] = np.clip(seed[6] + delta, lower[6], upper[6])
        seeds.append(seed)

    def robot_penetration():
        for contact in scratch.contact[:scratch.ncon]:
            bodies = mj_to_newton[model.geom_bodyid[contact.geom]]
            if all("/Robot/" in labels[int(b)] for b in bodies) and contact.dist < -0.001:
                return True
        return False

    best = None
    for position_goal in positions:
        def residual(q):
            pos, normal, forward = fk(q)
            return np.concatenate((
                12.0 * (pos - position_goal),
                5.0 * (normal - palm_up),
                5.0 * (forward - fingers_forward),
                0.04 * (q - q_start),
            ))
        for seed in seeds:
            result = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                                   max_nfev=250, ftol=1e-9, xtol=1e-9)
            pos, normal, forward = fk(result.x)
            normal_alignment = float(normal @ palm_up)
            finger_alignment = float(forward @ fingers_forward)
            position_error = float(np.linalg.norm(pos - position_goal))
            penetrates = robot_penetration()
            valid = (normal_alignment > 0.98 and finger_alignment > 0.95
                     and position_error < 0.035 and not penetrates)
            score = min(normal_alignment, finger_alignment) - 3 * position_error
            print(f"PALM_UP_IK_CANDIDATE palm_dot={normal_alignment:.4f} "
                  f"fingers_dot={finger_alignment:.4f} pos_error_m={position_error:.4f} "
                  f"self_collision={penetrates} accepted={valid}", flush=True)
            if best is None or (valid, score) > best[0]:
                best = ((valid, score), result.x.copy(), (normal_alignment, finger_alignment, position_error))
            if valid:
                break
        if best is not None and best[0][0]:
            break

    if best is None or not best[0][0]:
        raise RuntimeError("No collision-free palm-up Franka pose found; do not substitute a fingers-down pose")

    q = best[1]
    normal_alignment, finger_alignment, position_error = best[2]
    _tensor(robot.data.default_joint_pos)[0, art_joints] = torch.tensor(
        q, device=env.device, dtype=_tensor(robot.data.default_joint_pos).dtype,
    )
    print(f"PALM_UP_IK_SELECTED arm_joints={[round(float(x), 4) for x in q]} "
          f"palm_dot={normal_alignment:.5f} fingers_dot={finger_alignment:.5f} "
          f"position_error_m={position_error:.5f}", flush=True)
    return tuple(float(x) for x in q)


@dataclass
class GraspResult:
    candidate: GraspCandidate
    valid: bool
    drift_m: float
    joint_drift_rad: float
    opposing_fraction: float
    contact_rate: tuple[float, ...]
    start_quat: tuple[float, ...]
    axis_world: tuple[float, ...]
    centre_world: tuple[float, ...]
    closing_target: torch.Tensor


def _action_info(env):
    term = env.action_manager.get_term("joint_pos")
    names = list(term._joint_names)
    if len(names) != 20 or set(names) != {f"finger_r_joint{i}" for i in range(1, 21)}:
        raise RuntimeError(f"Reorientation no longer exposes exactly 20 finger actions: {names}")
    robot = env.scene["robot"]
    indices = [robot.joint_names.index(name) for name in names]
    limits = _tensor(robot.data.joint_pos_limits)[0, indices]
    return names, limits[:, 0].clone(), limits[:, 1].clone()


class FeasibilityExperiment:
    def __init__(self, env, args):
        self.env, self.args = env, args
        self.cube = env.scene["object"]
        self.support = env.scene["support"]
        self.robot = env.scene["robot"]
        self.names, self.lower, self.upper = _action_info(env)
        self.open = torch.zeros((1, 20), device=env.device)
        self.arm_ids = [self.robot.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
        self.hand_id = next(i for i, name in enumerate(self.robot.body_names) if name.split("/")[-1] == "panda_hand")
        self.arm_reference = _tensor(self.robot.data.default_joint_pos)[0, self.arm_ids].detach().clone()

    def step(self, action):
        with torch.inference_mode():
            obs, _, terminated, truncated, _ = self.env.step(action)
        if bool(torch.any(terminated | truncated)):
            raise RuntimeError("Unexpected timeout/reset during feasibility trial")
        if _tensor(obs["policy"]).shape != (1, 40) or _tensor(obs["contact"]).shape != (1, 5):
            raise RuntimeError("Task policy/contact observation contract changed")
        labels = _tensor(obs["contact"])[0].detach().cpu().tolist()
        if any(value not in (0, 1) for value in labels):
            raise RuntimeError(f"Nonbinary contact labels: {labels}")
        return labels

    def cube_state(self):
        pos = _xyz(_tensor(self.cube.data.root_pos_w)[0].detach().cpu().tolist())
        quat = _xyz(_tensor(self.cube.data.root_quat_w)[0].detach().cpu().tolist())
        return pos, quat

    def hand_pose(self):
        pose = _tensor(self.robot.data.body_link_pose_w)[0, self.hand_id].detach()
        return _xyz(pose[:3].cpu().tolist()), _xyz(pose[3:7].cpu().tolist())

    def arm_drift(self):
        positions = _tensor(self.robot.data.joint_pos)[0, self.arm_ids]
        return float(torch.max(torch.abs(positions - self.arm_reference)).item())

    def check_arm_orientation(self, stage):
        robot = self.robot
        ids = [i for i, name in enumerate(robot.body_names) if name.split("/")[-1] == "hx5_d20_right_base"]
        if len(ids) != 1:
            raise RuntimeError("Cannot identify exactly one live palm link")
        pose = _tensor(robot.data.body_link_pose_w)[0, ids[0]]
        axes = torch.tensor([[1., 0., 0.], [0., 0., 1.]], device=self.env.device)
        actual = [quat_apply(pose[3:7][None, :], axis[None, :])[0] for axis in axes]
        palm_dot = float(torch.dot(actual[0], torch.tensor(PALM_NORMAL_WORLD, device=self.env.device)))
        forward_dot = float(torch.dot(actual[1], torch.tensor(FINGERS_FORWARD_WORLD, device=self.env.device)))
        print(f"PALM_UP_ALIGNMENT stage={stage} palm_dot={palm_dot:.4f} "
              f"fingers_dot={forward_dot:.4f} arm_drift_rad={self.arm_drift():.4f}", flush=True)
        if palm_dot < 0.95 or forward_dot < 0.90:
            raise RuntimeError("Live palm is not facing up with fingers horizontal")

    def clip(self, targets):
        return torch.maximum(torch.minimum(targets, self.upper[None, :]), self.lower[None, :])

    def initialise(self, candidate: GraspCandidate):
        """Reset and place the cube/pedestal once; no cube pose writes afterwards."""
        self.env.reset()
        self.check_arm_orientation("initial")
        palm_pos, palm_quat = self.hand_pose()
        local = torch.tensor([candidate.centre_hand_m], device=self.env.device)
        rotation = torch.tensor([palm_quat], device=self.env.device)
        world_centre = torch.tensor([palm_pos], device=self.env.device) + quat_apply(rotation, local)
        if self.args.use_pedestal:
            centre_z = float(world_centre[0, 2])
            table = self.env.cfg.scene.table
            table_top = float(table.init_state.pos[2]) + float(table.spawn.size[2]) / 2
            pedestal_bottom = centre_z - CUBE_SIDE_M / 2 - SUPPORT_CLEARANCE_M - SUPPORT_HEIGHT_M
            if pedestal_bottom <= max(table_top, 0.0) + 0.005:
                raise ValueError(
                    f"Candidate pedestal would hit table/ground: bottom={pedestal_bottom:.3f} m; "
                    f"table_top={table_top:.3f} m"
                )

        cube_pose = _tensor(self.cube.data.default_root_pose).clone()
        cube_pose[:, :3] = world_centre
        self.cube.write_root_pose_to_sim(cube_pose)
        self.cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.env.device))

        support_pose = _tensor(self.support.data.default_root_pose).clone()
        if self.args.use_pedestal:
            support_pose[:, :3] = world_centre
            support_pose[:, 2] -= CUBE_SIDE_M / 2 + SUPPORT_CLEARANCE_M + SUPPORT_HEIGHT_M / 2
        else:
            support_pose[:, :3] = torch.tensor([[1.4, 1.4, 1.4]], device=self.env.device)
        self.support.write_root_pose_to_sim(support_pose)
        self.support_pose = support_pose
        self.support_removed = not self.args.use_pedestal

        return _xyz(world_centre[0].cpu().tolist())

    def remove_support(self):
        """Move only the temporary pedestal, never the free cube."""
        pose = self.support_pose.clone()
        pose[:, :3] = torch.tensor([[1.4, 1.4, 1.4]], device=self.env.device)
        self.support.write_root_pose_to_sim(pose)
        self.support_removed = True

    def grasp(self, candidate: GraspCandidate) -> GraspResult:
        cube_centre = self.initialise(candidate)
        close = self.clip(torch.tensor([grasp_targets(self.names, candidate.posture)], device=self.env.device))
        for _ in range(self.args.settle_steps):
            self.step(self.open)
        for t in range(self.args.close_steps):
            fraction = min((t + 1) / self.args.close_ramp_steps, 1.0)
            self.step(torch.lerp(self.open, close, fraction))
        if not self.support_removed:
            self.remove_support()
        self.check_arm_orientation("after_closure")

        initial_pos, initial_quat = self.cube_state()
        palm_pos, palm_quat = self.hand_pose()
        axis_world = rotate_vector_xyzw(palm_quat, LOCAL_ROTATION_AXIS)
        totals = [0.0] * 5
        opposing = 0
        max_drift = 0.0
        max_arm_drift = 0.0
        for _ in range(self.args.hold_steps):
            bits = self.step(close)
            current_pos, _ = self.cube_state()
            max_drift = max(max_drift, math.dist(initial_pos, current_pos))
            max_arm_drift = max(max_arm_drift, self.arm_drift())
            opposing += bool(bits[0] and any(bits[1:]))
            for i in range(5):
                totals[i] += bits[i]
        fractions = tuple(value / self.args.hold_steps for value in totals)
        opposing_fraction = opposing / self.args.hold_steps
        current_pos, _ = self.cube_state()
        supported = not self.support_removed
        # Both stability and contact are required: a cube merely falling onto
        # the table or touching a single finger is NOT a valid grasp.
        valid = (not supported and max_drift < self.args.max_drift_m
                 and math.dist(cube_centre, current_pos) < self.args.max_drift_m
                 and max_arm_drift < self.args.max_arm_drift_rad
                 and opposing_fraction >= self.args.min_opposing_fraction)
        result = GraspResult(
            candidate=candidate, valid=valid, drift_m=max_drift,
            joint_drift_rad=max_arm_drift, opposing_fraction=opposing_fraction,
            contact_rate=fractions, start_quat=initial_quat, axis_world=axis_world,
            centre_world=cube_centre, closing_target=close,
        )
        print(
            f"GRASP_TRIAL candidate={candidate.label()} valid={valid} "
            f"drift_m={max_drift:.4f} total_displacement_m={math.dist(cube_centre, current_pos):.4f} "
            f"arm_drift_rad={max_arm_drift:.4f} opposing_fraction={opposing_fraction:.3f} "
            f"contact_fraction={[round(x, 3) for x in fractions]}",
            flush=True,
        )
        return result

    def rotation_trial(self, candidate: GraspCandidate, pattern: str, signed_amplitude: float):
        grip = self.grasp(candidate)  # same physics setup and independent reset per trial
        if not grip.valid:
            print(f"ROTATION_SKIPPED pattern={pattern} amplitude={signed_amplitude:.3f} reason=grasp_not_reproduced", flush=True)
            return None

        # Use the observed cube orientation *after* the two-second hold.
        start_position, start_quat = self.cube_state()
        _, palm_quat = self.hand_pose()
        axis = rotate_vector_xyzw(palm_quat, LOCAL_ROTATION_AXIS)
        goals = {
            deg: quat_mul_xyzw(axis_angle_xyzw(axis, deg), start_quat)
            for deg in (15, 45)
        }

        desired = apply_rotation_pattern(
            grip.closing_target[0].tolist(), self.names, pattern, signed_amplitude
        )
        commanded = self.clip(torch.tensor([desired], device=self.env.device))
        errors = {deg: math.inf for deg in goals}
        best_twist = 0.0
        max_drift = 0.0
        opposing_steps = 0
        retained_steps = 0
        success = {deg: False for deg in goals}
        for t in range(self.args.rotate_steps):
            fraction = min((t + 1) / self.args.rotate_ramp_steps, 1.0)
            action = torch.lerp(grip.closing_target, commanded, fraction)
            labels = self.step(action)
            position, quat = self.cube_state()
            drift = math.dist(start_position, position)
            max_drift = max(max_drift, drift)
            retained = drift < self.args.max_drift_m and self.arm_drift() < self.args.max_arm_drift_rad
            retained_steps += int(retained)
            opposing_steps += int(labels[0] and any(labels[1:]))
            twist = signed_twist_deg(start_quat, quat, axis)
            if retained and abs(twist) > abs(best_twist):
                best_twist = twist
            for deg, target in goals.items():
                error = angle_error_deg(quat, target)
                if retained and labels[0] and any(labels[1:]):
                    errors[deg] = min(errors[deg], error)
                    success[deg] |= error <= self.args.goal_tolerance_deg

        # A fleeting orientation match followed by dropping the object isn't success.
        final_pos, _ = self.cube_state()
        final_retained = (
            math.dist(start_position, final_pos) < self.args.max_drift_m
            and retained_steps >= int(0.8 * self.args.rotate_steps)
            and opposing_steps >= int(self.args.min_opposing_fraction * self.args.rotate_steps)
        )
        success = {deg: hit and final_retained for deg, hit in success.items()}
        print(
            f"ROTATION_TRIAL pattern={pattern} amplitude_rad={signed_amplitude:+.3f} "
            f"twist_deg={best_twist:+.2f} error15_deg={errors[15]:.2f} error45_deg={errors[45]:.2f} "
            f"max_drift_m={max_drift:.4f} opposing_fraction={opposing_steps/self.args.rotate_steps:.3f} "
            f"retained={final_retained} success15={success[15]} success45={success[45]}",
            flush=True,
        )
        return {"pattern": pattern, "amplitude": signed_amplitude, "success15": success[15],
                "success45": success[45], "twist": best_twist, "error45": errors[45]}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-pedestal", action="store_true",
                        help="Optional temporary cube support during closure; default is cube resting in palm")
    parser.add_argument("--max-grasps", type=int, default=12, help="Number of candidate cube-centre/posture trials")
    parser.add_argument("--max-rotation-trials", type=int, default=12, help="Number of finger-only trajectory trials after a grasp works")
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--close-steps", type=int, default=150)
    parser.add_argument("--close-ramp-steps", type=int, default=90)
    parser.add_argument("--hold-steps", type=int, default=60, help="Post-support hold: 60 steps = 2 s at 30 Hz")
    parser.add_argument("--rotate-steps", type=int, default=150)
    parser.add_argument("--rotate-ramp-steps", type=int, default=90)
    parser.add_argument("--max-drift-m", type=float, default=0.04)
    parser.add_argument("--max-arm-drift-rad", type=float, default=0.05)
    parser.add_argument("--min-opposing-fraction", type=float, default=0.15)
    parser.add_argument("--goal-tolerance-deg", type=float, default=5.0)
    add_launcher_args(parser)
    args = parser.parse_args()
    if min(args.max_grasps, args.max_rotation_trials, args.settle_steps, args.close_steps,
           args.hold_steps, args.rotate_steps, args.rotate_ramp_steps, args.close_ramp_steps) <= 0:
        parser.error("Trial counts and durations must be positive")
    if args.close_ramp_steps > args.close_steps or args.rotate_ramp_steps > args.rotate_steps:
        parser.error("Ramp steps cannot exceed their phase durations")
    return args


def main() -> int:
    args = _parse_args()
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.seed = args.seed
    # This only changes the standalone experiment. The production task is
    # still a free-cube contact-validation scaffold.
    cfg.scene.robot.init_state.joint_pos = {
        **{f"panda_joint{i + 1}": q for i, q in enumerate(ARM_IK_SEED)},
        "finger_r_joint.*": 0.0,
    }
    for actuator in ("panda_shoulder", "panda_forearm"):
        cfg.scene.robot.actuators[actuator].stiffness = 4000.0
        cfg.scene.robot.actuators[actuator].damping = 160.0

    # Kinematic pedestal only. The original cube remains a DYNAMIC rigid
    # object, mass 50 g, with gravity and contact physics unchanged.
    cfg.scene.support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TemporaryGraspPedestal",
        spawn=sim_utils.CuboidCfg(
            size=(SUPPORT_WIDTH_M, SUPPORT_WIDTH_M, SUPPORT_HEIGHT_M),
            rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)],
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.7, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.4, 1.4, 1.4)),
    )
    print(f"FIXTURE physics=newton_mjwarp cube=dynamic gravity=on "
          f"pedestal={'temporary' if args.use_pedestal else 'parked'} "
          f"palm=up arm=fixed fingers=20", flush=True)

    with launch_simulation(cfg, args):
        from isaaclab.envs import ManagerBasedRLEnv

        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            env.reset()
            _solve_palm_up_pose(env)
            env.reset()  # apply solved arm pose to both joints and PD reset targets
            exp = FeasibilityExperiment(env, args)
            exp.check_arm_orientation("solved")
            candidates = [
                GraspCandidate(point, posture)
                for point in GRASP_CENTRES
                for posture in ("lift", "diagnostic")
            ][:args.max_grasps]
            viable = []
            for candidate in candidates:
                try:
                    result = exp.grasp(candidate)
                    if result.valid:
                        viable.append(result)
                        # A confirmed first grasp is enough to start controller search.
                        break
                except ValueError as exc:
                    print(f"GRASP_REJECTED candidate={candidate.label()} reason={exc}", flush=True)

            if not viable:
                print("FEASIBILITY_SUMMARY grasp=False rotation15=False rotation45=False", flush=True)
                print("FEASIBILITY_INCOMPLETE no_unsupported_opposing_grasp_found", flush=True)
                return 2

            selected = max(viable, key=lambda x: (x.opposing_fraction, -x.drift_m))
            print(f"GRASP_SELECTED candidate={selected.candidate.label()} "
                  f"centre_world_m={[round(v,4) for v in selected.centre_world]} "
                  f"opposing_fraction={selected.opposing_fraction:.3f}", flush=True)

            trajectories = [
                (pattern, direction * amplitude)
                for amplitude in (0.2, 0.4, 0.6)
                for pattern in ROLL_PATTERNS
                for direction in (1, -1)
            ][:args.max_rotation_trials]
            results = [
                exp.rotation_trial(selected.candidate, pattern, amplitude)
                for pattern, amplitude in trajectories
            ]
            completed = [result for result in results if result is not None]
            hit15 = any(r["success15"] for r in completed)
            hit45 = any(r["success45"] for r in completed)
            best = max(completed, key=lambda r: abs(r["twist"])) if completed else None
            print(f"FEASIBILITY_SUMMARY grasp=True rotation15={hit15} rotation45={hit45} "
                  f"completed_trials={len(completed)} "
                  f"best_twist_deg={best['twist'] if best else None}", flush=True)
            if hit45:
                print("FEASIBILITY_ROTATION_45_OK", flush=True)
                return 0
            print("FEASIBILITY_INCOMPLETE free_cube_rotation_not_yet_demonstrated", flush=True)
            return 2
        finally:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
