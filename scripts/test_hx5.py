import argparse
import json
from pathlib import Path

import numpy as np

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Palm-up HX5 contact and stability diagnostic (Newton/MJWarp)")
parser.add_argument("--num_envs", type=int, default=1, choices=[1])
parser.add_argument("--random-steps", type=int, default=1000)
parser.add_argument("--physics-substeps", type=int, choices=[1, 4], help="Compare solver substep counts")
parser.add_argument(
    "--contact-profile", choices=["configured", "default", "stiff", "soft", "balanced"], default="configured"
)
parser.add_argument("--capture-dir", type=Path, help="Save newton_gl frames for visual inspection")

add_launcher_args(parser)

args_cli, hydra_overrides = parser.parse_known_args()

import torch

from isaaclab_tasks.utils import parse_env_cfg

import contact_prediction_rl.tasks  # noqa: F401

TASK = "ContactPredictionRl-Reorientation-HX5-D20-Right"


# Target joint positions in radians, keyed by URDF joint name
OPEN_POSE = {
    "finger_r_joint1": 0.0,
    "finger_r_joint2": 0.0,
    "finger_r_joint3": 0.0,
    "finger_r_joint4": 0.0,
    "finger_r_joint5": 0.0,
    "finger_r_joint6": 0.0,
    "finger_r_joint7": 0.0,
    "finger_r_joint8": 0.0,
    "finger_r_joint9": 0.0,
    "finger_r_joint10": 0.0,
    "finger_r_joint11": 0.0,
    "finger_r_joint12": 0.0,
    "finger_r_joint13": 0.0,
    "finger_r_joint14": 0.0,
    "finger_r_joint15": 0.0,
    "finger_r_joint16": 0.0,
    "finger_r_joint17": 0.0,
    "finger_r_joint18": 0.0,
    "finger_r_joint19": 0.0,
    "finger_r_joint20": 0.0,
}

HALF_CLOSED_POSE = {
    "finger_r_joint1": 0.2,
    "finger_r_joint2": -0.4,
    "finger_r_joint3": 0.5,
    "finger_r_joint4": 0.5,
    "finger_r_joint5": 0.0,
    "finger_r_joint6": 0.7,
    "finger_r_joint7": 0.6,
    "finger_r_joint8": 0.6,
    "finger_r_joint9": 0.0,
    "finger_r_joint10": 0.7,
    "finger_r_joint11": 0.6,
    "finger_r_joint12": 0.6,
    "finger_r_joint13": 0.0,
    "finger_r_joint14": 0.7,
    "finger_r_joint15": 0.6,
    "finger_r_joint16": 0.6,
    "finger_r_joint17": 0.0,
    "finger_r_joint18": 0.7,
    "finger_r_joint19": 0.6,
    "finger_r_joint20": 0.6,
}

CLOSED_POSE = {
    "finger_r_joint1": 0.4,
    "finger_r_joint2": -0.8,
    "finger_r_joint3": 1.0,
    "finger_r_joint4": 1.0,
    "finger_r_joint5": 0.0,
    "finger_r_joint6": 1.8,
    "finger_r_joint7": 1.4,
    "finger_r_joint8": 1.4,
    "finger_r_joint9": 0.0,
    "finger_r_joint10": 1.8,
    "finger_r_joint11": 1.4,
    "finger_r_joint12": 1.4,
    "finger_r_joint13": 0.0,
    "finger_r_joint14": 1.8,
    "finger_r_joint15": 1.4,
    "finger_r_joint16": 1.4,
    "finger_r_joint17": 0.0,
    "finger_r_joint18": 1.8,
    "finger_r_joint19": 1.4,
    "finger_r_joint20": 1.4,
}


def pose_to_action(robot, pose, device):
    action = torch.zeros((1, len(robot.joint_names)), device=device)

    for action_idx, joint_name in enumerate(robot.joint_names):
        action[0, action_idx] = pose[joint_name]

    return action


CONTACT_BODY_NAMES = {
    "thumb": "finger_r_link4",
    "index": "finger_r_link8",
    "middle": "finger_r_link12",
    "ring": "finger_r_link16",
    "little": "finger_r_link20",
    "palm": "hx5_d20_right_base",
}


class ContactReport:
    """Sample object-specific sensor forces and MJWarp contact points each action step"""

    def __init__(self, env):
        from isaaclab_newton.physics import NewtonManager

        self.env = env
        # MJWarp has no public solver accessor; use its data for point counts and signed distances
        self.solver = NewtonManager._solver
        self.geom_bodies = self.solver.mj_model.geom_bodyid
        model = NewtonManager.get_model()
        self.body_labels = model.body_label
        self.geom_newton_bodies = self.solver.mjc_body_to_newton.numpy().reshape(-1)[self.geom_bodies]
        self.object_bodies = {i for i, label in enumerate(self.body_labels) if label.endswith("/Object")}
        self.tip_bodies = {
            name: {i for i, label in enumerate(self.body_labels) if label.endswith("/" + link)}
            for name, link in CONTACT_BODY_NAMES.items()
        }
        self.scene_update = env.scene.update
        env.scene.update = self.update_and_sample
        self.reset()

    def update_and_sample(self, dt):
        """Observe every physics step after refreshing asset and sensor data"""
        self.scene_update(dt)
        self.sample()

    def reset(self):
        self.peak_force = dict.fromkeys(CONTACT_BODY_NAMES, 0.0)
        self.contact_samples = dict.fromkeys(CONTACT_BODY_NAMES, 0)
        self.peak_points = dict.fromkeys(CONTACT_BODY_NAMES, 0)
        self.max_penetration = 0.0
        self.peak_cube_speed = 0.0
        self.peak_cube_angular_speed = 0.0
        self.max_cube_height = 0.0
        self.peak_upward_speed = 0.0
        self.peak_point_force = 0.0
        self.max_table_penetration = 0.0
        self.peak_constraints = 0

    def sample(self):
        obj = self.env.scene["object"]
        for asset in (obj, self.env.scene["robot"]):
            for state in (asset.data.root_link_pose_w.torch, asset.data.root_com_vel_w.torch):
                assert torch.isfinite(state).all(), "Non-finite rigid-body state"
        robot = self.env.scene["robot"]
        assert torch.isfinite(robot.data.body_link_pose_w.torch).all(), "Non-finite finger pose"
        assert torch.isfinite(robot.data.body_com_vel_w.torch).all(), "Non-finite finger velocity"
        assert torch.isfinite(robot.data.joint_pos.torch).all(), "Non-finite joints"
        assert torch.isfinite(robot.data.joint_vel.torch).all(), "Non-finite joint velocities"
        self.peak_cube_speed = max(self.peak_cube_speed, float(obj.data.root_lin_vel_w.torch.norm()))
        self.peak_cube_angular_speed = max(self.peak_cube_angular_speed, float(obj.data.root_ang_vel_w.torch.norm()))
        self.max_cube_height = max(self.max_cube_height, float(obj.data.root_pos_w.torch[0, 2]))
        self.peak_upward_speed = max(self.peak_upward_speed, float(obj.data.root_lin_vel_w.torch[0, 2]))
        data = self.solver.mjw_data
        self.peak_constraints = max(self.peak_constraints, int(data.nefc.numpy().max()))
        count = int(data.nacon.numpy()[0])
        pairs = data.contact.geom.numpy()[:count]
        distances = data.contact.dist.numpy()[:count]
        addresses = data.contact.efc_address.numpy()[:count]
        forces = data.efc.force.numpy()[0]
        active = (addresses >= 0).any(axis=-1)
        points = dict.fromkeys(CONTACT_BODY_NAMES, 0)
        for pair, distance, is_active, address in zip(pairs, distances, active, addresses, strict=True):
            if not is_active or (pair < 0).any():
                continue
            bodies = set(self.geom_newton_bodies[pair].tolist())
            if not (bodies & self.object_bodies):
                continue
            # Pyramidal normal force is the sum of the nonnegative edge forces
            self.peak_point_force = max(self.peak_point_force, float(forces[address[address >= 0]].sum()))
            self.max_table_penetration = max(
                self.max_table_penetration,
                float(-distance) if not any(b >= 0 and "/Robot/" in self.body_labels[b] for b in bodies) else 0.0,
            )
            # All hand-object pairs, not just tips, contribute to the penetration bound
            if any(b >= 0 and "/Robot/" in self.body_labels[b] for b in bodies):
                self.max_penetration = max(self.max_penetration, float(-distance))
            for name, tip_bodies in self.tip_bodies.items():
                if bodies & tip_bodies:
                    points[name] += 1
        for name in CONTACT_BODY_NAMES:
            force = self.env.scene[name + "_contact"].data.normal_force_matrix_w.torch
            assert torch.isfinite(force).all(), "Non-finite contact force"
            self.peak_force[name] = max(self.peak_force[name], float(force.norm(dim=-1).max()))
            self.peak_points[name] = max(self.peak_points[name], points[name])
            self.contact_samples[name] += points[name]

    def result(self):
        return {
            "sampling_interval_s": self.env.physics_dt,
            "peak_normal_force_N": self.peak_force,
            "peak_contact_point_normal_force_N": self.peak_point_force,
            "peak_sampled_normal_impulse_N_s": {
                name: force * self.env.physics_dt / self.env.cfg.sim.physics.num_substeps
                for name, force in self.peak_force.items()
            },
            "peak_simultaneous_contact_points": self.peak_points,
            "contact_point_samples": self.contact_samples,
            "max_hand_cube_penetration_mm": 1000 * self.max_penetration,
            "max_cube_table_penetration_mm": 1000 * self.max_table_penetration,
            "peak_cube_speed_m_s": self.peak_cube_speed,
            "peak_cube_angular_speed_rad_s": self.peak_cube_angular_speed,
            "max_cube_height_m": self.max_cube_height,
            # An explicit diagnostic threshold, never a modification of simulator state
            "obvious_launch": self.peak_upward_speed > 2.0 or self.peak_cube_speed > 5.0 or self.max_cube_height > 0.8,
            "peak_constraint_rows": self.peak_constraints,
        }


def initial_contacts(report):
    """Check the compiled initial geometry before integrating gravity"""
    import mujoco

    model = report.solver.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    penetration = {"hand_cube_mm": 0.0, "cube_table_mm": 0.0}
    for contact in data.contact:
        bodies = set(report.geom_newton_bodies[contact.geom].tolist())
        if not bodies & report.object_bodies:
            continue
        key = "hand_cube_mm" if any(b >= 0 and "/Robot/" in report.body_labels[b] for b in bodies) else "cube_table_mm"
        penetration[key] = max(penetration[key], -1000 * float(contact.dist))
    assert max(penetration.values()) < 0.001, "Initial geometry penetrates before settling"
    return penetration


def capture(env, label):
    if args_cli.capture_dir is None:
        return
    from PIL import Image

    visualizer = next(v for v in env.sim._visualizers if v.cfg.visualizer_type == "newton_gl")
    args_cli.capture_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visualizer._viewer.get_frame().numpy()).save(args_cli.capture_dir / f"{label}.png")


def run_pose(env, action, steps, report, start_action=None, phase=None):
    first_contact = False
    for step in range(steps):
        target = action if start_action is None else torch.lerp(start_action, action, min((step + 1) / 90, 1.0))
        env.step(target)
        if phase and step in (30, 60, 90):
            capture(env, f"{phase}_{step}")
        if phase == "closure" and not first_contact and any(report.peak_points.values()):
            first_contact = True
            capture(env, "first_contact")


def orientation_change(start, end):
    dot = float(torch.abs((start[:, 3:] * end[:, 3:]).sum()).clamp(max=1.0))
    return float(2 * np.arccos(dot))


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1, overrides=hydra_overrides)
    env_cfg.seed = 0
    if args_cli.physics_substeps is not None:
        env_cfg.sim.physics.num_substeps = args_cli.physics_substeps
    assert torch.cuda.is_available(), "CUDA is required for this simulation diagnostic"
    assert str(args_cli.device).startswith("cuda"), "Pass --device cuda:0"
    print("CUDA_AVAILABLE", torch.cuda.is_available(), "Device", args_cli.device)
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg

    if args_cli.contact_profile == "default":
        env_cfg.scene.object.spawn.collision_props = env_cfg.scene.object.spawn.collision_props[:1]
    elif args_cli.contact_profile in ("soft", "stiff", "balanced"):
        stiff = args_cli.contact_profile == "stiff"
        env_cfg.scene.object.spawn.collision_props = env_cfg.scene.object.spawn.collision_props[:1] + [
            MujocoCollisionCfg(
                priority=1,
                solref=(0.005 if stiff else 0.01 if args_cli.contact_profile == "balanced" else 0.02, 1.0),
                solimp=(0.99, 0.999, 0.001, 0.5, 2.0) if stiff else (0.9, 0.95, 0.001, 0.5, 2.0),
            )
        ]
    with launch_simulation(env_cfg, args_cli):
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_newton.physics import NewtonManager

        env = ManagerBasedRLEnv(cfg=env_cfg)
        try:
            env.reset()
            env.sim.set_camera_view((0.25, -0.35, 0.8), (-0.08, 0.0, 0.46))
            robot, obj = env.scene["robot"], env.scene["object"]
            report = ContactReport(env)
            base_start = robot.data.root_link_pose_w.torch.clone()
            assert len(robot.joint_names) == 20
            assert np.allclose(report.solver.mj_model.opt.gravity, (0, 0, -9.81))
            assert not report.solver.mj_model.body_gravcomp.any()
            print("INITIAL_GEOMETRY", json.dumps(initial_contacts(report)))
            model = NewtonManager.get_model()
            shape_bodies = model.shape_body.numpy()
            cube_shapes = np.flatnonzero(np.isin(shape_bodies, list(report.object_bodies)))
            tip_shapes = np.flatnonzero(
                np.isin(shape_bodies, list(set.union(*(b for n, b in report.tip_bodies.items() if n != "palm"))))
            )
            pairs = np.array([(a, b) for a in tip_shapes for b in cube_shapes])
            print(
                "COLLISION_AUDIT",
                json.dumps(
                    {
                        "tip_shape_count_including_visuals": len(tip_shapes),
                        "cube_shape_count": len(cube_shapes),
                        "filtered_tip_cube_pairs": int(model.shape_collision_filter_mask(pairs).sum()),
                        "cube_mass_kg": model.body_mass.numpy()[list(report.object_bodies)].tolist(),
                        "cube_flags": model.body_flags.numpy()[list(report.object_bodies)].tolist(),
                        "cube_gravcomp": report.solver.mj_model.body_gravcomp[
                            np.isin(report.solver.mjc_body_to_newton.numpy().reshape(-1), list(report.object_bodies))
                        ].tolist(),
                        "use_mujoco_contacts": env.cfg.sim.physics.solver_cfg.use_mujoco_contacts,
                        "disable_contacts": env.cfg.sim.physics.solver_cfg.disable_contacts,
                        "njmax": env.cfg.sim.physics.solver_cfg.njmax,
                        "geom_solref": report.solver.mj_model.geom_solref.tolist(),
                        "geom_solimp": report.solver.mj_model.geom_solimp.tolist(),
                        "geom_friction": report.solver.mj_model.geom_friction.tolist(),
                        "geom_contype": report.solver.mj_model.geom_contype.tolist(),
                        "geom_conaffinity": report.solver.mj_model.geom_conaffinity.tolist(),
                    }
                ),
            )
            open_action = pose_to_action(robot, OPEN_POSE, env.device)
            closed_action = pose_to_action(robot, CLOSED_POSE, env.device)
            start = obj.data.root_pose_w.torch.clone()
            from isaaclab.utils.math import quat_apply

            palm_normal = quat_apply(base_start[:, 3:], torch.tensor([[1.0, 0.0, 0.0]], device=env.device))
            assert torch.allclose(palm_normal, torch.tensor([[0.0, 0.0, 1.0]], device=env.device), atol=1e-6)
            table_top = env.cfg.scene.table.init_state.pos[2] + env.cfg.scene.table.spawn.size[2] / 2
            print(
                "LAYOUT",
                json.dumps(
                    {
                        "hand_base_pose_xyzw": robot.data.root_pose_w.torch.tolist(),
                        "palm_normal_w": palm_normal.tolist(),
                        "palm_to_table_clearance_m": float(base_start[0, 2]) + 0.021 - table_top,
                        "table_top_m": env.cfg.scene.table.init_state.pos[2] + env.cfg.scene.table.spawn.size[2] / 2,
                        "cube_initial_pose_xyzw": start.tolist(),
                        "gravity_m_s2": env.cfg.sim.gravity,
                        "substeps": env.cfg.sim.physics.num_substeps,
                        "contact_profile": args_cli.contact_profile,
                        "initial_cube_table_gap_m": float(start[0, 2])
                        - env.cfg.scene.object.spawn.size[2] / 2
                        - env.cfg.scene.table.init_state.pos[2]
                        - env.cfg.scene.table.spawn.size[2] / 2,
                    }
                ),
            )
            env.step(open_action)
            print("INITIAL_CONTACT", json.dumps(report.result()))
            capture(env, "initial")
            run_pose(env, open_action, 120, report)
            settled = obj.data.root_pose_w.torch.clone()
            idle_drift = float((settled[:, :3] - start[:, :3]).norm())
            print(
                "SETTLING",
                json.dumps(
                    report.result()
                    | {
                        "cube_pose": settled.tolist(),
                        "settled_on_palm": report.peak_points["palm"] > 0,
                        "cube_displacement_m": idle_drift,
                        "final_speed_m_s": float(obj.data.root_lin_vel_w.torch.norm()),
                    }
                ),
            )
            assert idle_drift < 0.035, "Cube did not settle on palm"
            assert report.peak_points["palm"] > 0, "No physical palm support"
            assert float(settled[0, 2]) > table_top + 0.15, "Cube settled on table instead of palm"
            assert float(obj.data.root_lin_vel_w.torch.norm()) < 0.02, "Cube is not resting"
            assert not any(report.peak_points[n] for n in CONTACT_BODY_NAMES if n != "palm"), (
                "Tips contacted before closure"
            )
            capture(env, "settled")
            report.reset()
            run_pose(env, closed_action, 240, report, start_action=open_action, phase="closure")
            end = obj.data.root_pose_w.torch.clone()
            displacement = float((end[:, :3] - settled[:, :3]).norm())
            print(
                "CLOSURE",
                json.dumps(
                    report.result()
                    | {
                        "cube_start_pose": settled.tolist(),
                        "cube_end_pose": end.tolist(),
                        "cube_displacement_m": displacement,
                        "cube_orientation_change_rad": orientation_change(settled, end),
                    }
                ),
            )
            capture(env, "closed")
            assert any(report.peak_points[n] for n in CONTACT_BODY_NAMES if n != "palm"), (
                "No fingertip-object contact points detected"
            )
            assert max(report.peak_force.values()) > 0.1, "No significant fingertip-object force"
            assert displacement > 0.001, "Contact did not move the cube"
            assert report.max_penetration < 0.005, "Hand penetrated cube by more than 5 mm"
            assert not report.result()["obvious_launch"], "Cube launched during closure"
            report.reset()
            run_pose(env, open_action, 180, report, start_action=closed_action, phase="release")
            print(
                "RELEASE",
                json.dumps(
                    report.result()
                    | {
                        "cube_end_pose": obj.data.root_pose_w.torch.tolist(),
                        "final_speed_m_s": float(obj.data.root_lin_vel_w.torch.norm()),
                    }
                ),
            )
            capture(env, "released")

            # Opening alone leaves a palm-supported cube at rest; slide it off with a measured lateral force
            report.reset()
            release_start = obj.data.root_pos_w.torch.clone()
            force = torch.tensor([[[0.0, -2.0, 0.0]]], device=env.device)
            push_steps = 0
            if float(obj.data.root_pos_w.torch[0, 2]) > table_top + 0.1:
                obj.permanent_wrench_composer.set_forces_and_torques_index(forces=force, is_global=True)
                for _ in range(30):
                    env.step(open_action)
                    push_steps += 1
                    if float(obj.data.root_pos_w.torch[0, 1]) < -0.10:
                        break
                obj.permanent_wrench_composer.set_forces_and_torques_index(
                    forces=torch.zeros_like(force), is_global=True
                )
            run_pose(env, open_action, 120, report, phase="catch")
            release_end = obj.data.root_pos_w.torch.clone()
            landed = abs(float(release_end[0, 2]) - (table_top + 0.025)) < 0.003
            print(
                "PALM_RELEASE",
                json.dumps(
                    report.result()
                    | {
                        "release_start_position": release_start.tolist(),
                        "release_end_position": release_end.tolist(),
                        "lateral_release_force_N": 2.0 if push_steps else 0.0,
                        "force_duration_s": push_steps * env.step_dt,
                        "landed_on_table": landed,
                        "final_speed_m_s": float(obj.data.root_lin_vel_w.torch.norm()),
                    }
                ),
            )
            assert landed, "Released cube did not fall to the catch table"
            assert not report.result()["obvious_launch"], "Cube launched during release"
            capture(env, "caught")

            # A separate initial-condition drop verifies free fall without any supporting hand contact
            env.reset()
            drop_pose = obj.data.root_link_pose_w.torch.clone()
            drop_pose[0, :3] = torch.tensor((0.25, 0.0, 0.549), device=env.device)
            obj.write_root_pose_to_sim(drop_pose)
            obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
            report.reset()
            env.step(open_action)
            fall_speed = float(obj.data.root_lin_vel_w.torch[0, 2])
            capture(env, "gravity_drop")
            assert -0.34 < fall_speed < -0.31, "Free fall does not match Earth gravity"
            run_pose(env, open_action, 90, report)
            print(
                "GRAVITY_DROP",
                json.dumps(
                    report.result()
                    | {
                        "vertical_speed_after_1_30_s": fall_speed,
                        "final_cube_pose": obj.data.root_link_pose_w.torch.tolist(),
                        "final_speed_m_s": float(obj.data.root_lin_vel_w.torch.norm()),
                    }
                ),
            )
            assert (
                abs(float(obj.data.root_pos_w.torch[0, 2]) - (table_top + env.cfg.scene.object.spawn.size[2] / 2))
                < 0.002
            )
            assert float(obj.data.root_lin_vel_w.torch.norm()) < 0.001
            assert not any(report.peak_points.values()), "Drop probe contacted hand"
            capture(env, "gravity_settled")

            env.reset()
            report.reset()
            torch.manual_seed(0)
            # Bounded random joint targets using the actual URDF limits
            limits = robot.data.joint_pos_limits.torch
            for _ in range(args_cli.random_steps):
                action = limits[..., 0] + torch.rand((1, env.action_manager.total_action_dim), device=env.device) * (
                    limits[..., 1] - limits[..., 0]
                )
                env.step(action)
            print(
                "STABILITY",
                json.dumps(
                    report.result()
                    | {
                        "finite_steps": args_cli.random_steps,
                        "physics_steps": args_cli.random_steps * env.cfg.decimation,
                    }
                ),
            )
            assert torch.allclose(robot.data.root_link_pose_w.torch, base_start), "Suspended hand base moved"
            assert not report.result()["obvious_launch"], "Cube launched during random actions"
            print("CONTACT_DIAGNOSTIC_OK")
        finally:
            env.close()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
