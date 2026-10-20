# scripts/validate_fingertip_contacts.py

"""Validate real Newton fingertip contacts against a stationary diagnostic cube.

Local +Z of the hand is solved towards world -Z. Collision geometry from the
compiled MuJoCo model selects a collision-free cube placement in the selected
finger's swept workspace. Only real simulator forces generate contact labels.
Use --cube-mode free for the separate gravity-enabled grasp experiment.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import AssetBaseCfg

from isaaclab_tasks.utils import parse_env_cfg

import contact_prediction_rl.tasks  # noqa: F401
from contact_prediction_rl.robots.hx5_d20_contacts import (
    CONTACT_FORCE_THRESHOLD_N,
    FINGERTIP_NAMES,
    FINGERTIP_SENSOR_NAMES,
)

# HX5_PAD_DIAGNOSTIC_INTEGRATION_V1 -- purely optional debug functionality.
from contact_prediction_rl.robots.hx5_d20_patches import (
    NewtonPatchDiagnostic, PatchSelection, candidate_patches_from_compiled_mjcf,
    patch_surface_samples, wxyz_to_rotation,
)

TASK = "ContactPredictionRl-Reorientation-HX5-D20-Right"

# Original placement is retained for the separate free-cube experiment.
# Fixed sensor trials search the compiled collision meshes at reset instead.
DEFAULT_CUBE_X = 0.475
DEFAULT_CUBE_Y = 0.0
DEFAULT_CUBE_Z = 0.370
DEFAULT_PEDESTAL_WIDTH = 0.035
SUPPORT_CLEARANCE_M = 0.0005  # initial cube bottom is this far above support

# These values reproduce the physical closing motion used by scripts/test_hx5.py.
THUMB_CLOSED = (0.4, -0.8, 1.0, 1.0)
FINGER_CLOSED = (0.0, 1.8, 1.4, 1.4)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse diagnostic and simulator options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--finger",
        choices=("grasp", "all", *FINGERTIP_NAMES),
        default="index",
        help="Isolate index (default), another finger, or test all five separately.",
    )
    parser.add_argument("--cube-mode", choices=("fixed", "free"), default="fixed", help="Stationary kinematic sensor fixture or separate gravity-enabled free-cube experiment.")
    parser.add_argument("--settle-steps", type=int, default=120, help="Open-hand steps before contact testing.")
    parser.add_argument("--close-steps", type=int, default=240, help="Steps for closure and hold.")
    parser.add_argument("--release-steps", type=int, default=180, help="Steps for reopening and hold.")
    parser.add_argument("--ramp-steps", type=int, default=90, help="Action steps over which joints interpolate to their targets.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument("--cube-x", type=float, default=DEFAULT_CUBE_X, help="Cube centre world x [m].")
    parser.add_argument("--cube-y", type=float, default=DEFAULT_CUBE_Y, help="Cube centre world y [m].")
    parser.add_argument("--cube-z", type=float, default=DEFAULT_CUBE_Z, help="Cube centre world z [m]; pedestal is sized to its bottom.")
    parser.add_argument(
        "--pedestal-width", type=float, default=DEFAULT_PEDESTAL_WIDTH,
        help="Square raised-support width [m], must be smaller than the cube footprint.",
    )
    parser.add_argument("--report-positions", action="store_true", help="Print cube, palm and fingertip world positions at key phases.")
    parser.add_argument("--placement-region", choices=("tip", "proximal"), default="tip", help="Target fingertip-patch mesh surface, or proximal mesh as a negative control.")
    parser.add_argument("--placement-rank", type=int, default=0, help="Select a ranked collision-free swept-mesh cube placement for negative-control exploration.")
    parser.add_argument("--measure-patches", action="store_true", help="Classify real active Newton cube/distal contact events in candidate regions.")
    parser.add_argument("--show-patches", action="store_true", help="Overlay diagnostic green regions and live yellow/red collision points (implies --measure-patches).")
    parser.add_argument("--patch-tolerance-mm", type=float, default=1.5, help="Candidate box inclusion tolerance in mm.")
    parser.add_argument("--contact-distance-mm", type=float, default=2.0, help="Largest active Newton constraint signed distance in mm.")
    parser.add_argument("--ordinary-x-tail", type=float, default=.38, help="Fraction of high +X ordinary-finger collision bounds.")
    parser.add_argument("--ordinary-z-tail", type=float, default=.38, help="Fraction of distal +Z ordinary-finger collision bounds.")
    parser.add_argument("--ordinary-y-central", type=float, default=.84, help="Fraction of central Y ordinary-finger width.")
    parser.add_argument("--thumb-y-tail", type=float, default=.35, help="Fraction of distal +Y thumb collision bounds.")
    parser.add_argument("--thumb-x-central", type=float, default=.80, help="Fraction of central X thumb width.")
    parser.add_argument("--thumb-z-central", type=float, default=.80, help="Fraction of central Z thumb width.")
    parser.add_argument("--inspect-only", action="store_true", help="Keep the hand open in the viewer; do not run closing/contact trials.")
    parser.add_argument("--inspect-steps", type=int, default=9000, help="Maximum open-hand steps in --inspect-only mode; Ctrl+C exits early.")
    add_launcher_args(parser)
    return parser.parse_known_args(argv)


def _solve_fingers_down_pose(env) -> None:
    """Solve a fingers-down posture using the *compiled Newton/MJWarp* kinematics.

    The MuJoCo model is used only for forward kinematics of a scratch MjData.
    Actual physics, gravity, sensor forces, and task observations are unchanged.
    The result is installed as this diagnostic's reset joint posture. The
    fingertip contact trial still moves only the 20 hand joints.
    """
    import mujoco
    import numpy as np
    from scipy.optimize import least_squares

    from isaaclab_newton.physics import NewtonManager

    robot = env.scene["robot"]
    solver = NewtonManager._solver
    mj_model = solver.mj_model
    newton = NewtonManager.get_model()

    hand_body = next((i for i, label in enumerate(newton.body_label)
                      if label.endswith("/hx5_d20_right_base")), None)
    if hand_body is None:
        raise RuntimeError("The Newton model has no HX5-D20 palm body.")
    mj_to_newton = solver.mjc_body_to_newton.numpy().reshape(-1)
    mj_body_ids = np.flatnonzero(mj_to_newton == hand_body)
    if len(mj_body_ids) != 1:
        raise RuntimeError(f"Expected one MJWarp palm body, got {mj_body_ids.tolist()}.")
    mj_hand_id = int(mj_body_ids[0])

    # The compiled MJWarp/MuJoCo joint names are not guaranteed to match the
    # names exposed by Isaac Lab (and can be None). Map each joint through its
    # *owning rigid body* instead: panda_jointN belongs to panda_linkN.
    # Newton supplies a stable MJ-body -> Newton-body mapping even when the
    # compiler renames or drops joint names.
    names = [f"panda_joint{i}" for i in range(1, 8)]
    joint_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in range(mj_model.njnt)]
    mj_joint_ids = []
    articulation_joint_ids = []
    for number, name in enumerate(names, start=1):
        link_name = f"panda_link{number}"
        newton_matches = [i for i, label in enumerate(newton.body_label)
                          if label == link_name or label.endswith("/" + link_name)]
        if len(newton_matches) != 1:
            raise RuntimeError(
                f"Cannot uniquely resolve Newton body {link_name}: {newton_matches}; "
                f"Franka bodies={[label for label in newton.body_label if 'panda_' in label]}"
            )
        mj_body_matches = np.flatnonzero(mj_to_newton == newton_matches[0])
        # Fixed/welded bodies can map to the same Newton body. Only the link
        # body carrying exactly one hinge joint identifies the correct dof.
        candidate_joints = []
        for mj_body_id in mj_body_matches:
            start_joint = int(mj_model.body_jntadr[mj_body_id])
            count = int(mj_model.body_jntnum[mj_body_id])
            for j in range(start_joint, start_joint + count):
                if int(mj_model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                    candidate_joints.append(j)
        if len(candidate_joints) != 1 or name not in robot.joint_names:
            raise RuntimeError(
                f"Cannot uniquely map {name} via {link_name}: "
                f"mj_bodies={mj_body_matches.tolist()}, hinge_joints={candidate_joints}, "
                f"IsaacLab_has_joint={name in robot.joint_names}"
            )
        mj_joint_ids.append(candidate_joints[0])
        articulation_joint_ids.append(robot.joint_names.index(name))

    print(
        "DIAGNOSTIC_ARM_MAPPING "
        + ", ".join(f"{name}->mj_joint[{joint_id}]({joint_names[joint_id]!r})"
                    for name, joint_id in zip(names, mj_joint_ids, strict=True)),
        flush=True,
    )

    # Take the actual reset values; the source USD's qpos0 may differ.
    start = robot.data.default_joint_pos.torch[0, articulation_joint_ids].detach().cpu().numpy().astype(float)
    limits = robot.data.joint_pos_limits.torch[0, articulation_joint_ids].detach().cpu().numpy()
    lb, ub = limits[:, 0].astype(float) + 0.005, limits[:, 1].astype(float) - 0.005
    scratch = mujoco.MjData(mj_model)
    qpos = np.asarray([mj_model.jnt_qposadr[i] for i in mj_joint_ids], dtype=int)
    # Set all 20 hand joints to their current default too, so frame poses match the scene.
    for j, joint_name in enumerate(joint_names):
        if joint_name and joint_name.split("/")[-1].startswith("finger_r_joint"):
            short = joint_name.split("/")[-1]
            if short in robot.joint_names:
                scratch.qpos[int(mj_model.jnt_qposadr[j])] = float(
                    robot.data.default_joint_pos.torch[0, robot.joint_names.index(short)]
                )

    def fk(q):
        scratch.qpos[qpos] = q
        mujoco.mj_forward(mj_model, scratch)
        p = np.asarray(scratch.xpos[mj_hand_id]).copy()
        rot = np.asarray(scratch.xmat[mj_hand_id]).reshape(3, 3).copy()
        return p, rot[:, 0], rot[:, 2]  # palm +X, finger +Z in world

    before_pos, before_normal, before_finger = fk(start)
    # Confirm scratch MJ kinematics matches the live Isaac Lab/Newton articulation.
    # Otherwise the optimized pose would be based on the wrong coordinate frame.
    from isaaclab.utils.math import quat_apply
    link_matches = [i for i, name in enumerate(robot.body_names)
                    if name == "hx5_d20_right_base" or name.endswith("/hx5_d20_right_base")]
    if len(link_matches) != 1:
        raise RuntimeError("Cannot resolve the live HX5-D20 palm link.")
    pose = _tensor(robot.data.body_link_pose_w)[0, link_matches[0]]
    actual_normal = quat_apply(
        pose[3:7].unsqueeze(0),
        torch.tensor([[1.0, 0.0, 0.0]], device=env.device),
    )[0].detach().cpu().numpy()
    normal_mismatch = float(np.linalg.norm(before_normal - actual_normal))
    print(f"DIAGNOSTIC_FK_CHECK initial_normal_error={normal_mismatch:.4f}", flush=True)
    if normal_mismatch > 0.15:
        raise RuntimeError(
            "Scratch MuJoCo forward kinematics disagree with live Newton palm pose; "
            "cannot safely compute the wrist orientation."
        )
    normal_goal = np.array([-1.0, 0.0, 0.0])
    finger_goal = np.array([0.0, 0.0, -1.0])
    # Leave room for the 180-mm open fingers above the existing table.
    position_goal = np.array([0.475, 0.0, 0.56])

    def residual(q):
        p, normal, finger = fk(q)
        return np.concatenate([
            15.0 * (p - position_goal),
            2.5 * (normal - normal_goal),
            8.0 * (finger - finger_goal),
            0.05 * (q - start),
        ])

    # Solve offline on scratch kinematics, no physical simulation is stepped.
    seeds = [start.copy()]
    for delta in [-0.5, 0.5]:
        seed = start.copy()
        seed[5] = np.clip(seed[5] + delta, lb[5], ub[5])
        seeds.append(seed)
    results = [least_squares(residual, np.clip(seed, lb, ub), bounds=(lb, ub),
                             max_nfev=220, ftol=1e-9, xtol=1e-9)
               for seed in seeds]
    scored = []
    for result in results:
        p, normal, finger = fk(result.x)
        alignment = float(np.dot(finger, np.array([0.0, 0.0, -1.0])))
        pose_error = float(np.linalg.norm(p - position_goal))
        self_penetrations = []
        for contact in scratch.contact[:scratch.ncon]:
            bodies = mj_to_newton[mj_model.geom_bodyid[contact.geom]]
            if all("/Robot/" in newton.body_label[int(body)] for body in bodies) and contact.dist < -0.001:
                self_penetrations.append(float(contact.dist))
        print(f"IK_CANDIDATE finger_alignment={alignment:.6f} position_error_m={pose_error:.6f} self_penetrations_gt_1mm={len(self_penetrations)}", flush=True)
        scored.append((alignment > 0.95 and pose_error < 0.035 and not self_penetrations,
                       alignment - 2 * pose_error - 0.001 * float(np.linalg.norm(result.x-start)), result.x, alignment, pose_error))
    solved, _, target, alignment, pose_error = max(scored, key=lambda entry: (entry[0], entry[1]))
    print(f"FINGER_DOWN_ALIGNMENT scratch_cosine={alignment:.4f} "
          f"palm_shift={pose_error*1000:.1f} mm "
          f"arm_joints={[round(float(v), 4) for v in target]}", flush=True)
    if not solved:
        raise RuntimeError(
            "Could not find a sufficiently fingers-down pose near the original position "
            "within the Franka joint limits; do not continue with invalid geometry."
        )

    # This is diagnostic-only state; reset_scene_to_default reads this tensor.
    # reset_joint_targets=True then resets the arm PD targets to the same pose.
    robot.data.default_joint_pos.torch[0, articulation_joint_ids] = torch.tensor(
        target, device=env.device, dtype=robot.data.default_joint_pos.torch.dtype
    )


def _check_palm_direction(env, stage: str) -> None:
    """Verify local +Z points down using the live hand link pose."""
    from isaaclab.utils.math import quat_apply

    robot = env.scene["robot"]
    matches = [i for i, name in enumerate(robot.body_names)
               if name == "hx5_d20_right_base" or name.endswith("/hx5_d20_right_base")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one HX5 palm body, found {matches}.")
    pose = _tensor(robot.data.body_link_pose_w)[0, matches[0]]
    axes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=env.device)
    # quat_apply supports broadcasting a single quaternion to two vectors.
    oriented = quat_apply(pose[3:7].expand(2, 4), axes)
    palm_normal = oriented[0]
    finger_axis = oriented[1]
    def fmt(vec):
        return [round(float(value), 4) for value in vec]
    print(
        f"  PALM_DIRECTION {stage}: palm_normal_world={fmt(palm_normal)} "
        f"fingers_forward_world={fmt(finger_axis)} "
        f"finger_down_alignment={-float(finger_axis[2]):.4f}",
        flush=True,
    )
    if -float(finger_axis[2]) <= 0.95:
        raise RuntimeError("FINGER_DOWN_ALIGNMENT FAIL: live local +Z alignment must exceed 0.95.")
    print(f"FINGER_DOWN_ALIGNMENT PASS stage={stage} cosine={-float(finger_axis[2]):.6f}", flush=True)


def _configure_tabletop_fixture(env_cfg, args_cli) -> None:
    """Configure a stationary sensor fixture or a separate gravity-enabled cube.

    The existing broad table stays in place beneath the pedestal. A wide
    raised tabletop would collide with the long fingers before they can grasp
    a 5-cm cube; the support is narrower than the cube so fingers can pass
    either side. This config is local to this script, not the RL environment.
    """
    cube_cfg = env_cfg.scene.object
    cube_size = cube_cfg.spawn.size
    if len(cube_size) != 3 or max(cube_size) - min(cube_size) > 1e-6:
        raise ValueError(f"Expected a cubic object, got dimensions {cube_size}.")
    side = float(cube_size[0])
    if not 0 < args_cli.pedestal_width < side:
        raise ValueError(f"Pedestal width must lie strictly between 0 and cube side ({side:g} m).")

    table_cfg = env_cfg.scene.table
    table_top = float(table_cfg.init_state.pos[2]) + float(table_cfg.spawn.size[2]) / 2
    support_top = float(args_cli.cube_z) - side / 2 - SUPPORT_CLEARANCE_M
    height = support_top - table_top
    if height <= 0:
        raise ValueError(f"Cube bottom must sit above the existing tabletop at z={table_top:.4f} m.")

    if args_cli.cube_mode == "fixed":
        cube_cfg.spawn.rigid_props = [sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)]
        cube_cfg.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.9))
        cube_cfg.init_state.pos = (args_cli.cube_x, args_cli.cube_y, args_cli.cube_z)
        print("DIAGNOSTIC_FIXTURE mode=fixed kinematic=True cyan cube; no pedestal; sensor support must be measured", flush=True)
        return
    cube_cfg.init_state.pos = (args_cli.cube_x, args_cli.cube_y, args_cli.cube_z)
    # Keep the cube dynamics, gravity, mass, collision shapes and filtering from
    # the registered task. Only its initial pose is changed here.
    # InteractiveScene parses scene config instance attributes, including the
    # diagnostic-only extra below, before the Newton model is built.
    env_cfg.scene.support = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ContactValidationSupport",
        spawn=sim_utils.CuboidCfg(
            size=(args_cli.pedestal_width, args_cli.pedestal_width, height),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.55, 0.5)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(args_cli.cube_x, args_cli.cube_y, table_top + height / 2),
        ),
    )

    print(
        "DIAGNOSTIC_FIXTURE "
        f"cube_center=({args_cli.cube_x:.4f}, {args_cli.cube_y:.4f}, {args_cli.cube_z:.4f}) m "
        f"cube_side={side:.3f} m "
        f"table_top={table_top:.4f} m support_top={support_top:.4f} m "
        f"pedestal_width={args_cli.pedestal_width:.3f} m "
        "gravity=unchanged arm_actions=none",
        flush=True,
    )


def _cube_stability(env, expected_xyz: torch.Tensor) -> tuple[bool, str]:
    """Check that the cube is still resting near its intended initial pose."""
    cube = env.scene["object"]
    position = _tensor(cube.data.root_pos_w)[0]
    error = torch.linalg.vector_norm(position - expected_xyz).item()
    velocity = torch.linalg.vector_norm(_tensor(cube.data.root_lin_vel_w)[0]).item()
    print(
        f"  CUBE_SETTLED position={[round(float(x), 4) for x in position]} "
        f"initial_offset={error * 1000:.1f} mm speed={velocity:.4f} m/s",
        flush=True,
    )
    kinematic = any(getattr(prop, "kinematic_enabled", False) for prop in env.cfg.scene.object.spawn.rigid_props)
    if kinematic and (error > 0.0001 or velocity > 0.0001):
        return False, "Kinematic fixture moved; stationary contact validation is invalid."
    if error > 0.025:
        return False, "Cube moved more than 25 mm during settling; likely fell or tipped off its support."
    if velocity > 0.06:
        return False, "Cube is still moving during the baseline; use more --settle-steps."
    return True, "Cube remained approximately stationary on the support."


def _tensor(value):
    """Access an Isaac Lab TensorData wrapper or an ordinary Torch tensor."""
    return value.torch if hasattr(value, "torch") else value


class NewtonContactProbe:
    """Read actual cube/fingertip collision pairs from Newton MJWarp.

    These solver internals are diagnostic only; they are not part of the
    learning observation or contact-label implementation. Their API may
    change with Isaac Lab/Newton versions.
    """

    def __init__(self) -> None:
        from isaaclab_newton.physics import NewtonManager

        solver = NewtonManager._solver
        if solver is None or not hasattr(solver, "mjw_data"):
            raise RuntimeError("This diagnostic requires the Newton MJWarp physics backend.")

        self.solver = solver
        model = NewtonManager.get_model()
        labels = model.body_label
        self.robot_bodies = {i for i, label in enumerate(labels) if "/Robot/" in label}
        self.cube_robot_count = 0
        self.object_bodies = {i for i, label in enumerate(labels) if label.endswith("/Object")}
        self.tip_bodies = {
            name: {i for i, label in enumerate(labels) if label.endswith(f"/finger_r_link{4 * (index + 1)}")}
            for index, name in enumerate(FINGERTIP_NAMES)
        }
        self.geometry_bodies = solver.mjc_body_to_newton.numpy().reshape(-1)[solver.mj_model.geom_bodyid]

        if not self.object_bodies:
            raise RuntimeError("Newton model has no cube body named */Object.")
        missing = [name for name, bodies in self.tip_bodies.items() if not bodies]
        if missing:
            raise RuntimeError(f"Newton model is missing fingertip bodies: {missing}.")

    def count_contacts(self) -> dict[str, int]:
        """Count active physical cube/fingertip contact points by finger."""
        data = self.solver.mjw_data
        count = int(data.nacon.numpy()[0])  # This script deliberately uses one environment.
        pairs = data.contact.geom.numpy()[:count]
        addresses = data.contact.efc_address.numpy()[:count]
        counts = dict.fromkeys(FINGERTIP_NAMES, 0)
        self.cube_robot_count = 0

        for pair, address in zip(pairs, addresses, strict=True):
            if (pair < 0).any() or not (address >= 0).any():
                continue
            bodies = set(self.geometry_bodies[pair].tolist())
            if not bodies.intersection(self.object_bodies):
                continue
            if bodies.intersection(self.robot_bodies):
                self.cube_robot_count += 1
            for name, tip_bodies in self.tip_bodies.items():
                if bodies.intersection(tip_bodies):
                    counts[name] += 1
        return counts


    def report_points(self, env, phase, names):
        """Read solver contact positions and express them in the live distal frame."""
        import numpy as np

        from isaaclab.utils.math import quat_apply_inverse

        data = self.solver.mjw_data
        count = int(data.nacon.numpy()[0])
        pairs = data.contact.geom.numpy()[:count]
        addresses = data.contact.efc_address.numpy()[:count]
        positions = data.contact.pos.numpy()[:count]
        depths = data.contact.dist.numpy()[:count]
        robot = env.scene["robot"]
        poses = _tensor(robot.data.body_link_pose_w)[0]
        points = []
        for pair, address, position, depth in zip(pairs, addresses, positions, depths, strict=True):
            if (pair < 0).any() or not (address >= 0).any():
                continue
            bodies = set(self.geometry_bodies[pair].tolist())
            if not bodies.intersection(self.object_bodies):
                continue
            for name in names:
                if not bodies.intersection(self.tip_bodies[name]):
                    continue
                link = f"finger_r_link{4 * (FINGERTIP_NAMES.index(name) + 1)}"
                idx = next(i for i, value in enumerate(robot.body_names) if value.split("/")[-1] == link)
                point = torch.tensor(position, device=env.device).unsqueeze(0)
                local = quat_apply_inverse(poses[idx, 3:7].unsqueeze(0), point - poses[idx, :3])
                print(f"CONTACT_POINT phase={phase} finger={name} world_m={position.tolist()} distal_local_m={local[0].tolist()} signed_distance_m={float(depth):.6f}", flush=True)
                points.append(position)
        if points and not getattr(self, "pad_diagnostic", None):
            if not hasattr(self, "markers"):
                from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
                self.markers = VisualizationMarkers(VisualizationMarkersCfg(
                    prim_path="/Visuals/MeasuredFingertipContacts",
                    markers={"contact": sim_utils.SphereCfg(radius=0.003,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1., 1., 0.)))},
                ))
            self.markers.visualize(translations=torch.tensor(np.asarray(points), device=env.device))
            print("CONTACT_POINT yellow markers show first measured points; pad attribution requires comparing local coordinates with the unmodified distal mesh", flush=True)


def _plan_cube_placement(env, probe, finger, placement_rank=0, placement_region="tip"):
    """Search compiled collision surfaces, without advancing or editing physics.

    Each isolated trial may use a different reset placement. MuJoCo's convex
    mesh distance is a geometry prediction; Newton events remain the proof.
    """
    import mujoco
    import numpy as np

    model = probe.solver.mj_model
    scratch = mujoco.MjData(model)
    robot = env.scene["robot"]
    body_map = probe.solver.mjc_body_to_newton.numpy().reshape(-1)
    from isaaclab_newton.physics import NewtonManager
    labels = NewtonManager.get_model().body_label
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        body = labels[int(body_map[model.jnt_bodyid[j]])].split("/")[-1]
        name = body.replace("panda_link", "panda_joint").replace("finger_r_link", "finger_r_joint")
        if name in robot.joint_names:
            scratch.qpos[model.jnt_qposadr[j]] = float(_tensor(robot.data.default_joint_pos)[0, robot.joint_names.index(name)])
    base_q = scratch.qpos.copy()
    selected = "index" if finger == "grasp" else finger
    cube_geoms = [g for g, body in enumerate(probe.geometry_bodies) if body in probe.object_bodies]
    tip_geoms = {name: [g for g, body in enumerate(probe.geometry_bodies) if body in bodies
                        and (model.geom_contype[g] or model.geom_conaffinity[g])]
                 for name, bodies in probe.tip_bodies.items()}
    robot_geoms = [g for g in range(model.ngeom) if "Robot/" in labels[int(probe.geometry_bodies[g])]
                   and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not robot_geoms:
        raise RuntimeError("HAND_CUBE_ALIGNMENT FAIL: cannot identify robot collision geometry")
    if len(cube_geoms) != 1 or not tip_geoms[selected]:
        raise RuntimeError("Expected one cube collision geom and distal collision meshes")
    cube_geom = cube_geoms[0]
    action = _closed_action(env, selected)[0].cpu().numpy()
    action_names = env.action_manager.get_term("joint_pos")._joint_names
    hand_q = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            body = labels[int(body_map[model.jnt_bodyid[j]])].split("/")[-1]
            name = body.replace("finger_r_link", "finger_r_joint")
            if name in action_names:
                hand_q[j] = action[list(action_names).index(name)]

    def pose(alpha):
        scratch.qpos[:] = base_q
        for j, target in hand_q.items():
            adr = model.jnt_qposadr[j]
            scratch.qpos[adr] = (1-alpha)*base_q[adr] + alpha*np.clip(target, *model.jnt_range[j])
        mujoco.mj_forward(model, scratch)

    def distances(center, geoms):
        # Scratch geom transforms only: never write simulated state or contacts.
        scratch.geom_xpos[cube_geom] = center
        scratch.geom_xmat[cube_geom] = np.eye(3).ravel()
        return min(mujoco.mj_geomDistance(model, scratch, cube_geom, g, 1.0, None) for g in geoms)

    table = env.cfg.scene.table
    table_top = float(table.init_state.pos[2]) + float(table.spawn.size[2])/2
    patches = candidate_patches_from_compiled_mjcf(
        model, probe.geometry_bodies, probe.tip_bodies,
        getattr(probe, "placement_patch_cfg", PatchSelection()),
    )
    patch = patches[selected]
    candidates = []
    for alpha in np.linspace(0.15, 0.8, 14):
        pose(alpha)
        for g in tip_geoms[selected]:
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                raise RuntimeError("Expected existing distal mesh geometry")
            mesh = int(model.geom_dataid[g])
            first, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = model.mesh_vert[first:first + count]
            body_vertices = vertices @ wxyz_to_rotation(model.geom_quat[g]).T + model.geom_pos[g]
            if placement_region == "tip":
                face_first, face_count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
                points = patch_surface_samples(
                    body_vertices, model.mesh_face[face_first:face_first + face_count], patch,
                )
            else:
                # Explicit negative control: collide with proximal distal mesh.
                z = body_vertices[:, 2]
                points = body_vertices[z <= z.min() + .45 * np.ptp(z)]
            body = int(model.geom_bodyid[g])
            world = points @ scratch.xmat[body].reshape(3, 3).T + scratch.xpos[body]
            for direction in (np.array([1., 0., 0.]), np.array([-1., 0., 0.]),
                              np.array([0., 1., 0.]), np.array([0., -1., 0.])):
                for surface in world:
                    center = surface + direction * .024
                    if center[2] - .025 <= table_top + .005:
                        continue
                    if distances(center, tip_geoms[selected]) < -.0001:
                        candidates.append((center.copy(), float(alpha)))

    def first_touch_inside_patch(center, alpha):
        """Locate the first *predicted* collision on scratch kinematics."""
        previous = 0.
        for fraction in np.linspace(alpha / 12, alpha, 12):
            pose(fraction)
            if distances(center, tip_geoms[selected]) > 0:
                previous = fraction
                continue
            low, high = previous, fraction
            for _ in range(8):
                mid = (low + high) / 2
                pose(mid)
                if distances(center, tip_geoms[selected]) > 0:
                    low = mid
                else:
                    high = mid
            pose(high)
            nearest_geom = min(
                tip_geoms[selected],
                key=lambda g: mujoco.mj_geomDistance(model, scratch, cube_geom, g, 1., None),
            )
            fromto = np.empty(6)
            distances(center, [nearest_geom])
            mujoco.mj_geomDistance(model, scratch, cube_geom, nearest_geom, 1., fromto)
            body = int(model.geom_bodyid[nearest_geom])
            local = (fromto[3:] - scratch.xpos[body]) @ scratch.xmat[body].reshape(3, 3)
            return patch.contains_local(local), local
        return False, None

    feasible = []
    rejected_first_touch = 0
    pose(0)
    for center, alpha in candidates:
        clearance = distances(center, robot_geoms)
        if clearance <= .003:
            continue
        if placement_region == "tip":
            inside, _ = first_touch_inside_patch(center, alpha)
            if not inside:
                rejected_first_touch += 1
                continue
        feasible.append((alpha + np.linalg.norm(center - np.array([0.475, 0., 0.370])), center, clearance, alpha))
    if not feasible:
        raise RuntimeError(
            f"HAND_CUBE_ALIGNMENT FAIL: no separated {selected} {placement_region} placement; "
            f"candidates={len(candidates)} rejected_first_touch_outside_patch={rejected_first_touch}. "
            "Do not change collider geometry or widen the patch merely to force a pass."
        )
    ranked = sorted(feasible, key=lambda item: item[0])
    if not 0 <= placement_rank < len(ranked):
        raise ValueError(f"placement-rank must be in [0, {len(ranked)-1}]")
    _, center, clearance, alpha = ranked[placement_rank]
    pose(0)
    _tensor(env.scene["object"].data.default_root_pose)[0, :3] = torch.tensor(center, device=env.device)
    print(f"HAND_CUBE_ALIGNMENT PASS finger={selected} placement_region={placement_region} cube_world={center.tolist()} open_mesh_clearance_m={clearance:.6f} predicted_contact_closure={alpha:.3f} candidates={len(candidates)} rejected_first_touch_outside_patch={rejected_first_touch}", flush=True)
    for name, geoms in tip_geoms.items():
        print(f"GEOMETRY_DISTANCE {name} cube_to_distal_m={distances(center, geoms):.6f}", flush=True)
    print("OPPOSITION_UNVERIFIED: isolated surface reach does not establish thumb opposition or a grasp", flush=True)


@dataclass
class PhaseResult:
    """Per-finger observations gathered over a phase."""

    name: str
    label_steps: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FINGERTIP_NAMES, 0))
    raw_steps: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FINGERTIP_NAMES, 0))
    confirmed_steps: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FINGERTIP_NAMES, 0))
    peak_force_n: dict[str, float] = field(default_factory=lambda: dict.fromkeys(FINGERTIP_NAMES, 0.0))
    other_body_steps: int = 0
    pad_steps: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FINGERTIP_NAMES, 0))
    pad_inside_events: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FINGERTIP_NAMES, 0))
    pad_outside_events: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FINGERTIP_NAMES, 0))
    patch_examples: set[tuple[str, bool]] = field(default_factory=set)
    last_10_pad: list[list[int]] = field(default_factory=list)
    last_action: torch.Tensor | None = None
    last_10_raw: list[bool] = field(default_factory=list)
    last_bits: list[int] = field(default_factory=list)
    last_10_bits: list[list[int]] = field(default_factory=list)

    def sample(self, env, observations: dict, probe: NewtonContactProbe) -> None:
        labels = observations["contact"]
        if labels.shape != (1, 5):
            raise AssertionError(f"Expected obs['contact'] shape (1, 5), got {tuple(labels.shape)}")
        if not bool(torch.all((labels == 0) | (labels == 1))):
            raise AssertionError("Contact observations must contain only binary labels.")
        bits = [int(value) for value in labels[0].tolist()]
        pad_bits = [0] * len(FINGERTIP_NAMES)
        if getattr(probe, "pad_diagnostic", None) is not None:
            candidate_bits, pad_events = probe.pad_diagnostic.sample(env)
            pad_bits = [int(v) for v in candidate_bits]
            for event in pad_events:
                finger, inside = event["finger"], event["inside"]
                target = self.pad_inside_events if inside else self.pad_outside_events
                target[finger] += 1
                key = (finger, inside)
                if key not in self.patch_examples:
                    self.patch_examples.add(key)
                    print(f"PATCH_CONTACT_EXAMPLE phase={self.name} finger={finger} "
                          f"inside={inside} local_m={event['local_m']} "
                          f"distance_m={event['signed_distance_m']:.6f}", flush=True)
            self.last_10_pad = (self.last_10_pad + [pad_bits])[-10:]
            for i, finger in enumerate(FINGERTIP_NAMES):
                self.pad_steps[finger] += pad_bits[i]
        raw_contacts = probe.count_contacts()
        self.other_body_steps += int(probe.cube_robot_count > sum(raw_contacts.values()))
        first = [name for name in FINGERTIP_NAMES if raw_contacts[name] and not self.raw_steps[name]]
        if first:
            probe.report_points(env, self.name, first)
        self.last_10_raw = (self.last_10_raw + [any(raw_contacts.values())])[-10:]

        for index, (name, sensor_name) in enumerate(zip(FINGERTIP_NAMES, FINGERTIP_SENSOR_NAMES, strict=True)):
            forces = env.scene[sensor_name].data.force_matrix_w
            if forces is None:
                raise RuntimeError(f"{sensor_name} has no cube-filtered contact force matrix.")
            forces = _tensor(forces)
            if tuple(forces.shape) != (1, 1, 1, 3):
                raise RuntimeError(f"Unexpected {sensor_name} force shape: {tuple(forces.shape)}")
            force_n = float(torch.linalg.vector_norm(forces[0, 0, 0, :]))
            if not bool(torch.isfinite(forces).all()):
                raise RuntimeError(f"Non-finite contact force in {sensor_name}.")
            self.peak_force_n[name] = max(self.peak_force_n[name], force_n)
            self.label_steps[name] += bits[index]
            self.raw_steps[name] += int(raw_contacts[name] > 0)
            self.confirmed_steps[name] += int(bool(bits[index]) and raw_contacts[name] > 0)

        self.last_bits = bits
        self.last_10_bits.append(bits)
        self.last_10_bits = self.last_10_bits[-10:]

    def print_report(self) -> None:
        print(f"  {self.name}: final_bits={self.last_bits} non_distal_cube_collision_steps={self.other_body_steps}", flush=True)
        for name in FINGERTIP_NAMES:
            print(
                f"    {name:<6} label_steps={self.label_steps[name]:>3} "
                f"raw_collision_steps={self.raw_steps[name]:>3} "
                f"confirmed_steps={self.confirmed_steps[name]:>3} "
                f"peak_filtered_force={self.peak_force_n[name]:.4f} N "
                f"pad_steps={self.pad_steps[name]:>3} "
                f"pad_inside_events={self.pad_inside_events[name]:>3} "
                f"pad_outside_events={self.pad_outside_events[name]:>3}",
                flush=True,
            )

    def clear_at_end(self) -> bool:
        return bool(self.last_10_bits) and all(not any(bits) for bits in self.last_10_bits)

    def pad_clear_at_end(self) -> bool:
        return bool(self.last_10_pad) and all(not any(bits) for bits in self.last_10_pad)


def _closed_action(env, finger: str):
    """Build 20 hand targets in the action term's resolved joint order.

    The robot is a 27-DoF Franka + HX5-D20 articulation, but Reorientation
    exposes *only* the 20 hand joints as actions. Never index the 20D action
    tensor using positions in the full articulation's joint-name list.
    """
    action_term = env.action_manager.get_term("joint_pos")
    # JointAction resolves its regex against the asset, preserving this exact
    # order when applying the 20D action. This is a diagnostic-only read of
    # Isaac Lab's internal field, not a modification of the task action space.
    names = list(action_term._joint_names)
    expected = {f"finger_r_joint{i}" for i in range(1, 21)}
    if (
        len(names) != 20
        or set(names) != expected
        or action_term.action_dim != 20
        or env.action_manager.total_action_dim != 20
    ):
        raise RuntimeError(
            f"Expected a 20D hand-only joint_pos action term; got "
            f"action_dim={action_term.action_dim}, resolved_joints={names!r}."
        )
    if not expected.issubset(set(env.scene["robot"].joint_names)):
        raise RuntimeError("The full Franka + HX5-D20 articulation is missing expected hand joints.")

    action = torch.zeros((1, 20), device=env.device)
    for i, name in enumerate(FINGERTIP_NAMES):
        if finger not in ("grasp", name):
            continue
        values = THUMB_CLOSED if name == "thumb" else FINGER_CLOSED
        for offset, value in enumerate(values):
            joint_name = f"finger_r_joint{4 * i + offset + 1}"
            action[0, names.index(joint_name)] = value
    return action



def _report_positions(env, phase: str) -> None:
    """Show live cube/palm/tip link origins to diagnose missed grasps.

    Distances are to link frame origins, not signed distances between
    collision surfaces; interpret them only as approximate geometry clues.
    """
    robot = env.scene["robot"]
    cube = env.scene["object"]
    cube_pos = _tensor(cube.data.root_pos_w)[0]
    poses = _tensor(robot.data.body_link_pose_w)[0]
    names = list(robot.body_names)

    def body_position(link_name: str):
        matches = [i for i, name in enumerate(names) if name == link_name or name.endswith("/" + link_name)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one body named {link_name!r}; found {matches} in {names!r}")
        return poses[matches[0], :3]

    def xyz(value):
        return [round(float(component), 4) for component in value]

    palm = body_position("hx5_d20_right_base")
    print(f"  POSITIONS {phase}: cube_world_m={xyz(cube_pos)} palm_world_m={xyz(palm)} cube_minus_palm_m={xyz(cube_pos - palm)}", flush=True)
    for finger_index, name in enumerate(FINGERTIP_NAMES):
        link = f"finger_r_link{4 * (finger_index + 1)}"
        tip = body_position(link)
        dist = float(torch.linalg.vector_norm(tip - cube_pos))
        print(f"    {name:<6} tip_link_origin_world_m={xyz(tip)} tip_to_cube_center_m={dist:.4f}", flush=True)


def _run_phase(env, probe: NewtonContactProbe, name: str, target, steps: int, *, initial=None, ramp_steps: int = 1, stop_on_contact: bool = False) -> PhaseResult:
    """Step real physics using position targets and log collisions and labels."""
    result = PhaseResult(name=name)
    held = None
    for step in range(steps):
        if held is not None:
            action = held
        elif initial is None:
            action = target
        else:
            alpha = min((step + 1) / ramp_steps, 1.0)
            action = torch.lerp(initial, target, alpha)
        observations, _, _, _, _ = env.step(action)
        result.sample(env, observations, probe)
        result.last_action = action.clone()
        if stop_on_contact and held is None and any(result.raw_steps.values()):
            held = action.clone()
            print(f"FIRST_COLLISION step={step} closure motion stopped; hold current commanded target", flush=True)
    return result


def _interpret(finger: str, baseline: PhaseResult, closing: PhaseResult, released: PhaseResult) -> tuple[bool, str]:
    """Distinguish a missed physical collision from incorrect contact labels."""
    if baseline.other_body_steps or any(baseline.raw_steps.values()) or any(baseline.label_steps.values()):
        return False, "Open-hand baseline already touched the cube; initial separation must be contact-free."
    if not baseline.clear_at_end():
        return False, "Unexpected fingertip contact at the end of open-hand settling."
    if not released.clear_at_end() or any(released.last_10_raw):
        return False, "Fingertip contact did not clear after reopening."

    selected = FINGERTIP_NAMES if finger == "grasp" else (finger,)
    detected = [name for name in selected if closing.confirmed_steps[name] > 0]
    if not detected:
        if any(closing.pad_outside_events[name] for name in selected):
            return False, "Newton registered real distal contact, but all contact points fell outside the geometric fingertip patch; inspect cube placement."
        if not any(closing.raw_steps[name] for name in selected):
            return False, "No selected fingertip physically contacted the cube in this pose; adjust geometry/trajectory before judging sensors."
        if any(closing.label_steps[name] for name in selected):
            return False, "Contact labels and Newton collision pairs did not overlap in the sampled physics steps; inspect timing and filters."
        return False, "Newton registered distal contact without a positive observation; inspect active constraints, geometry filtering and sampling; the force readout is independent."

    if finger == "grasp":
        return True, f"Physical closing contact and binary labels observed for: {', '.join(detected)}."
    unexpected = [name for name in FINGERTIP_NAMES if name != finger and (closing.label_steps[name] > 0 or closing.raw_steps[name] > 0)]
    if unexpected:
        return False, f"{finger} contact registered, but other finger labels also fired: {unexpected}."
    return True, f"Isolated {finger} contact produced a physical collision and a correctly attributed binary label."


def run(argv: list[str]) -> int:
    """Execute isolated physics trials and return a nonzero code on failed validation."""
    args_cli, hydra_overrides = _parse_args(argv)
    if args_cli.inspect_steps <= 0:
        raise ValueError("--inspect-steps must be positive.")
    if min(args_cli.settle_steps, args_cli.close_steps, args_cli.release_steps, args_cli.ramp_steps) <= 0:
        raise ValueError("All phase durations must be positive.")
    if args_cli.close_steps < args_cli.ramp_steps or args_cli.release_steps < args_cli.ramp_steps:
        raise ValueError("Closure and release phases must be at least as long as --ramp-steps.")
    if args_cli.settle_steps < 10 or args_cli.release_steps < args_cli.ramp_steps + 10:
        raise ValueError("Allow at least 10 settled observation steps before and after contact.")

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1, overrides=hydra_overrides)
    env_cfg.seed = args_cli.seed
    # Diagnostic-only holding gains reduce gravity sag without teleporting the arm.
    for actuator in ("panda_shoulder", "panda_forearm"):
        env_cfg.scene.robot.actuators[actuator].stiffness = 4000.0
        env_cfg.scene.robot.actuators[actuator].damping = 160.0
    _configure_tabletop_fixture(env_cfg, args_cli)
    args_cli.device = env_cfg.sim.device
    trials = ("index", "thumb", "middle", "ring", "little") if args_cli.finger == "all" else (args_cli.finger,)
    successes = 0

    print(f"Validating {TASK} on {args_cli.device}; force threshold={CONTACT_FORCE_THRESHOLD_N:g} N", flush=True)
    print("A failure to make physical contact is NOT proof that a contact sensor is broken.", flush=True)

    with launch_simulation(env_cfg, args_cli):
        from isaaclab.envs import ManagerBasedRLEnv

        env = ManagerBasedRLEnv(cfg=env_cfg)
        try:
            # Solve the 7 Franka joints once. The actual contact trials still
            # command only the existing 20D hand action group.
            env.reset()  # initialise live poses before comparing against scratch MuJoCo FK
            _solve_fingers_down_pose(env)
            probe = NewtonContactProbe()
            probe.pad_diagnostic = None
            patch_cfg = PatchSelection(
                    inclusion_tolerance_m=args_cli.patch_tolerance_mm / 1000,
                    max_signed_distance_m=args_cli.contact_distance_mm / 1000,
                    ordinary_x_tail=args_cli.ordinary_x_tail,
                    ordinary_z_tail=args_cli.ordinary_z_tail,
                    ordinary_y_central=args_cli.ordinary_y_central,
                    thumb_y_tail=args_cli.thumb_y_tail,
                    thumb_x_central=args_cli.thumb_x_central,
                    thumb_z_central=args_cli.thumb_z_central,
                )
            probe.placement_patch_cfg = patch_cfg
            if args_cli.measure_patches or args_cli.show_patches:
                probe.pad_diagnostic = NewtonPatchDiagnostic(
                    probe, env, cfg=patch_cfg, show_patches=args_cli.show_patches)
            for finger in trials:
                print(f"\n=== Trial: {finger} ===", flush=True)
                if args_cli.cube_mode == "fixed":
                    _plan_cube_placement(env, probe, finger, args_cli.placement_rank, args_cli.placement_region)
                observations, _ = env.reset()
                _check_palm_direction(env, "initial")
                if observations["policy"].shape != (1, 40) or observations["contact"].shape != (1, 5):
                    raise AssertionError("The Reorientation 40D policy / 5D contact observation contract changed.")

                if args_cli.report_positions or args_cli.inspect_only:
                    _report_positions(env, "initial")

                if args_cli.inspect_only:
                    print(
                        "INSPECT_ONLY hand stays fully open in fingers-down orientation; "
                        "press Ctrl+C to stop (no contact pass/fail test).",
                        flush=True,
                    )
                    opened = torch.zeros((1, 20), device=env.device)
                    try:
                        for _ in range(args_cli.inspect_steps):
                            env.step(opened)
                            if probe.pad_diagnostic is not None:
                                probe.pad_diagnostic.sample(env)
                    except KeyboardInterrupt:
                        print("INSPECT_ONLY stopped by user.", flush=True)
                    _check_palm_direction(env, "inspect final")
                    _report_positions(env, "inspect final")
                    return 0

                opened = torch.zeros((1, 20), device=env.device)
                closed = _closed_action(env, finger)
                baseline = _run_phase(env, probe, "open / settle", opened, args_cli.settle_steps)
                expected_xyz = _tensor(env.scene["object"].data.default_root_pose)[0, :3].clone()
                stable, stability_reason = _cube_stability(env, expected_xyz)
                if args_cli.report_positions:
                    _report_positions(env, "after settling")
                initial_free = not baseline.other_body_steps and not any(baseline.raw_steps.values()) and not any(baseline.label_steps.values())
                print(f"INITIAL_COLLISION_FREE {'PASS' if initial_free else 'FAIL'}", flush=True)
                if not initial_free or not stable:
                    raise RuntimeError("Initial separation/stationarity failed; refusing to close.")
                closing = _run_phase(
                    env, probe, "close / hold", closed, args_cli.close_steps,
                    initial=opened, ramp_steps=args_cli.ramp_steps,
                    stop_on_contact=args_cli.cube_mode == "fixed" and finger != "grasp",
                )
                if args_cli.report_positions:
                    _report_positions(env, "after closing")
                released = _run_phase(env, probe, "reopen / hold", opened, args_cli.release_steps, initial=closing.last_action, ramp_steps=args_cli.ramp_steps)

                baseline.print_report()
                closing.print_report()
                released.print_report()

                _check_palm_direction(env, "released")
                prefix = finger.upper()
                if finger != "grasp":
                    print(f"{prefix}_RAW_CONTACT {'PASS' if closing.raw_steps[finger] else 'FAIL'} steps={closing.raw_steps[finger]}", flush=True)
                    print(f"{prefix}_FILTERED_FORCE {'PASS' if closing.peak_force_n[finger] > 0 else 'FAIL'} peak_N={closing.peak_force_n[finger]:.6f}", flush=True)
                    print(f"{prefix}_LABEL {'PASS' if closing.confirmed_steps[finger] else 'FAIL'} confirmed_steps={closing.confirmed_steps[finger]}", flush=True)
                    print(f"{prefix}_RELEASED {'PASS' if released.clear_at_end() and not any(released.last_10_raw) else 'FAIL'}", flush=True)
                stationary, _ = _cube_stability(env, expected_xyz)
                passed, reason = _interpret(finger, baseline, closing, released)
                if not stable or (args_cli.cube_mode == "fixed" and not stationary):
                    passed = False
                    reason = f"{stability_reason} {reason}"
                print(f"RESULT {finger}: {'PASS' if passed else 'FAIL'} — {reason}", flush=True)
                if probe.pad_diagnostic is not None:
                    selected_fingers = FINGERTIP_NAMES if finger == "grasp" else (finger,)
                    for name in selected_fingers:
                        print(
                            f"PATCH_RESULT finger={name} "
                            f"link_force_label_steps={closing.label_steps[name]} "
                            f"pad_active_steps={closing.pad_steps[name]} "
                            f"active_inside_events={closing.pad_inside_events[name]} "
                            f"active_outside_events={closing.pad_outside_events[name]} "
                            f"pad_release_clear={released.pad_clear_at_end()} "
                            f"link_release_clear={released.clear_at_end()}",
                            flush=True,
                        )
                successes += int(passed)
        finally:
            env.close()

    print(f"\nSUMMARY: {successes}/{len(trials)} physical-contact trials passed.", flush=True)
    if successes == len(trials):
        print("FINGERTIP_CONTACT_VALIDATION_OK", flush=True)
        return 0
    print("FINGERTIP_CONTACT_VALIDATION_INCOMPLETE", flush=True)
    return 1


if __name__ == "__main__":
    with torch.inference_mode():
        sys.exit(run(sys.argv[1:]))
