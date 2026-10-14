# scripts/validate_fingertip_contacts.py

"""Validate the HX5-D20 closing onto a gravity-supported cube without moving the arm.

The script creates a diagnostic-only tabletop fixture. A small raised support
(pedestal) sits on the existing table and holds a 5-cm cube inside the open
hand's *closing workspace*. The hand is already tilted down towards the table
in this Reorientation pose, so only the 20 hand joints are commanded.

A small pedestal is intentional: with this hand's long fingers, a full-width
raised tabletop would intersect the open fingertips. The pedestal supports the
cube under normal gravity while leaving room for the fingers to curl around it.
The fixture is *not* a training scene and never changes the task source code.

The diagnostic compares real Newton cube/fingertip collisions with the task's
binary 5D contact labels. The sensor tensors and labels are never modified.
Failure to make physical contact is reported separately from sensor failure.

Examples:
    uv run python scripts/validate_fingertip_contacts.py --report-positions
    uv run python scripts/validate_fingertip_contacts.py --finger index
    uv run python scripts/validate_fingertip_contacts.py --finger all
    uv run python scripts/validate_fingertip_contacts.py --visualizer newton
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

TASK = "ContactPredictionRl-Reorientation-HX5-D20-Right"

# Starting point selected from measured open- and closed-finger link positions
# in the Reorientation scene. The cube is 5 cm on a side; its centre is placed
# near the swept fingertips while the hand is open, and its bottom is supported
# by a thin, raised plinth on the existing table.
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
        default="grasp",
        help="Grasp with all fingers (default), isolate one finger, or test all five separately.",
    )
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
    add_launcher_args(parser)
    return parser.parse_known_args(argv)


def _configure_tabletop_fixture(env_cfg, args_cli) -> None:
    """Place a *free*, gravity-enabled cube on a small raised support.

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

        for pair, address in zip(pairs, addresses, strict=True):
            if (pair < 0).any() or not (address >= 0).any():
                continue
            bodies = set(self.geometry_bodies[pair].tolist())
            if not bodies.intersection(self.object_bodies):
                continue
            for name, tip_bodies in self.tip_bodies.items():
                if bodies.intersection(tip_bodies):
                    counts[name] += 1
        return counts


@dataclass
class PhaseResult:
    """Per-finger observations gathered over a phase."""

    name: str
    label_steps: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FINGERTIP_NAMES, 0))
    raw_steps: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FINGERTIP_NAMES, 0))
    confirmed_steps: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FINGERTIP_NAMES, 0))
    peak_force_n: dict[str, float] = field(default_factory=lambda: dict.fromkeys(FINGERTIP_NAMES, 0.0))
    last_bits: list[int] = field(default_factory=list)
    last_10_bits: list[list[int]] = field(default_factory=list)

    def sample(self, env, observations: dict, probe: NewtonContactProbe) -> None:
        labels = observations["contact"]
        if labels.shape != (1, 5):
            raise AssertionError(f"Expected obs['contact'] shape (1, 5), got {tuple(labels.shape)}")
        if not bool(torch.all((labels == 0) | (labels == 1))):
            raise AssertionError("Contact observations must contain only binary labels.")
        bits = [int(value) for value in labels[0].tolist()]
        raw_contacts = probe.count_contacts()

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
        print(f"  {self.name}: final_bits={self.last_bits}", flush=True)
        for name in FINGERTIP_NAMES:
            print(
                f"    {name:<6} label_steps={self.label_steps[name]:>3} "
                f"raw_collision_steps={self.raw_steps[name]:>3} "
                f"confirmed_steps={self.confirmed_steps[name]:>3} "
                f"peak_filtered_force={self.peak_force_n[name]:.4f} N",
                flush=True,
            )

    def clear_at_end(self) -> bool:
        return bool(self.last_10_bits) and all(not any(bits) for bits in self.last_10_bits)


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


def _run_phase(env, probe: NewtonContactProbe, name: str, target, steps: int, *, initial=None, ramp_steps: int = 1) -> PhaseResult:
    """Step real physics using position targets and log collisions and labels."""
    result = PhaseResult(name=name)
    for step in range(steps):
        if initial is None:
            action = target
        else:
            alpha = min((step + 1) / ramp_steps, 1.0)
            action = torch.lerp(initial, target, alpha)
        observations, _, _, _, _ = env.step(action)
        result.sample(env, observations, probe)
    return result


def _interpret(finger: str, baseline: PhaseResult, closing: PhaseResult, released: PhaseResult) -> tuple[bool, str]:
    """Distinguish a missed physical collision from incorrect contact labels."""
    if any(baseline.raw_steps.values()) or any(baseline.label_steps.values()):
        return False, "Open-hand baseline already touched the cube; initial separation must be contact-free."
    if not baseline.clear_at_end():
        return False, "Unexpected fingertip contact at the end of open-hand settling."
    if not released.clear_at_end():
        return False, "Fingertip contact did not clear after reopening."

    selected = FINGERTIP_NAMES if finger == "grasp" else (finger,)
    detected = [name for name in selected if closing.confirmed_steps[name] > 0]
    if not detected:
        if not any(closing.raw_steps[name] for name in selected):
            return False, "No selected fingertip physically contacted the cube in this pose; adjust geometry/trajectory before judging sensors."
        if any(closing.label_steps[name] for name in selected):
            return False, "Contact labels and Newton collision pairs did not overlap in the sampled physics steps; inspect timing and filters."
        return False, f"Newton registered physical contact but the {CONTACT_FORCE_THRESHOLD_N:g} N binary threshold was never exceeded; inspect filtered sensor forces."

    if finger == "grasp":
        return True, f"Physical closing contact and binary labels observed for: {', '.join(detected)}."
    unexpected = [name for name in FINGERTIP_NAMES if name != finger and (closing.label_steps[name] > 0 or closing.raw_steps[name] > 0)]
    if unexpected:
        return False, f"{finger} contact registered, but other finger labels also fired: {unexpected}."
    return True, f"Isolated {finger} contact produced a physical collision and a correctly attributed binary label."


def run(argv: list[str]) -> int:
    """Execute isolated physics trials and return a nonzero code on failed validation."""
    args_cli, hydra_overrides = _parse_args(argv)
    if min(args_cli.settle_steps, args_cli.close_steps, args_cli.release_steps, args_cli.ramp_steps) <= 0:
        raise ValueError("All phase durations must be positive.")
    if args_cli.close_steps < args_cli.ramp_steps or args_cli.release_steps < args_cli.ramp_steps:
        raise ValueError("Closure and release phases must be at least as long as --ramp-steps.")
    if args_cli.settle_steps < 10 or args_cli.release_steps < args_cli.ramp_steps + 10:
        raise ValueError("Allow at least 10 settled observation steps before and after contact.")

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1, overrides=hydra_overrides)
    env_cfg.seed = args_cli.seed
    _configure_tabletop_fixture(env_cfg, args_cli)
    args_cli.device = env_cfg.sim.device
    trials = FINGERTIP_NAMES if args_cli.finger == "all" else (args_cli.finger,)
    successes = 0

    print(f"Validating {TASK} on {args_cli.device}; force threshold={CONTACT_FORCE_THRESHOLD_N:g} N", flush=True)
    print("A failure to make physical contact is NOT proof that a contact sensor is broken.", flush=True)

    with launch_simulation(env_cfg, args_cli):
        from isaaclab.envs import ManagerBasedRLEnv

        env = ManagerBasedRLEnv(cfg=env_cfg)
        try:
            probe = NewtonContactProbe()
            for finger in trials:
                print(f"\n=== Trial: {finger} ===", flush=True)
                observations, _ = env.reset()
                if observations["policy"].shape != (1, 40) or observations["contact"].shape != (1, 5):
                    raise AssertionError("The Reorientation 40D policy / 5D contact observation contract changed.")

                if args_cli.report_positions:
                    _report_positions(env, "initial")

                opened = torch.zeros((1, 20), device=env.device)
                closed = _closed_action(env, finger)
                baseline = _run_phase(env, probe, "open / settle", opened, args_cli.settle_steps)
                expected_xyz = torch.tensor(
                    [args_cli.cube_x, args_cli.cube_y, args_cli.cube_z],
                    device=env.device,
                )
                stable, stability_reason = _cube_stability(env, expected_xyz)
                if args_cli.report_positions:
                    _report_positions(env, "after settling")
                closing = _run_phase(env, probe, "close / hold", closed, args_cli.close_steps, initial=opened, ramp_steps=args_cli.ramp_steps)
                if args_cli.report_positions:
                    _report_positions(env, "after closing")
                released = _run_phase(env, probe, "reopen / hold", opened, args_cli.release_steps, initial=closed, ramp_steps=args_cli.ramp_steps)

                baseline.print_report()
                closing.print_report()
                released.print_report()

                passed, reason = _interpret(finger, baseline, closing, released)
                if not stable:
                    passed = False
                    reason = f"{stability_reason} {reason}"
                print(f"RESULT {finger}: {'PASS' if passed else 'FAIL'} — {reason}", flush=True)
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
