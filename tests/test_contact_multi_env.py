# tests/test_contact_multi_env.py

"""GPU integration test for live Newton fingertip contacts in six parallel environments.

This test uses five *different* stationary cube placements from the successful
2026-10-20 five-finger diagnostic, plus a no-contact control. It checks the real
production obs["contact"] against an independent CPU reading of active Newton
constraints in each environment. No synthetic solver contacts are injected.

Run: uv run pytest -m gpu tests/test_contact_multi_env.py -v
"""

import subprocess
import sys
from pathlib import Path

import pytest

TASK = "ContactPredictionRl-Reorientation-HX5-D20-Right"
FINGERS = ("thumb", "index", "middle", "ring", "little")
NUM_ENVS = 6
# Cube positions in the local coordinates of each replicated environment.
# Captured with the default fingertip patches, fixed cube and solved arm pose.
CUBE_POSITIONS = (
    (0.4072893752, 0.1343374099, 0.4971052126),  # thumb
    (0.3801284042, 0.0382206323, 0.3512572255),  # index
    (0.3722101290, 0.0092303607, 0.3570347194),  # middle
    (0.3793780750, -0.0191558937, 0.3461935763),  # ring
    (0.3721428325, -0.0447687882, 0.3818164835),  # little
    (0.8, 0.6, 0.6),  # no-contact control
)
# From FINGER_DOWN_ALIGNMENT of the same Newton validation run, rounded to 4 dp.
ARM_JOINTS = (0.3697, -0.2897, -0.3216, -2.0356, -0.0921, 1.7594, -2.277)


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.smoke
def test_live_multi_env_contact_attribution_and_release():
    """Launch Newton in a child process: Isaac Lab's simulation is process-global."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    output = result.stdout + "\n" + result.stderr
    assert result.returncode == 0 and "MULTI_ENV_CONTACT_OK" in result.stdout, output[-10000:]
    assert "Traceback (most recent call last)" not in output, output[-10000:]


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def _make_reference(env):
    """Independent NumPy/CPU oracle based on raw *active* solver contacts."""
    import numpy as np

    from isaaclab_newton.physics import NewtonManager

    from contact_prediction_rl.robots.hx5_d20_patches import (
        PatchSelection,
        active_contact_is_inside,
        candidate_patches_from_compiled_mjcf,
    )

    solver = NewtonManager._solver
    assert solver is not None and hasattr(solver, "mjw_data"), "Newton MJWarp required"
    model = solver.mj_model
    body_names = NewtonManager.get_model().body_label
    geom_body = solver.mjc_body_to_newton.numpy().reshape(-1)[model.geom_bodyid]
    objects = {i for i, name in enumerate(body_names) if name.endswith("/Object")}
    tips = {
        name: {i for i, label in enumerate(body_names) if label.endswith(f"/finger_r_link{4 * (j + 1)}")}
        for j, name in enumerate(FINGERS)
    }
    assert objects and all(tips.values()), "Missing compiled object or distal links"
    patches = candidate_patches_from_compiled_mjcf(model, geom_body, tips, PatchSelection())
    robot_names = env.scene["robot"].body_names
    tip_indices = [
        next(i for i, body in enumerate(robot_names) if body.split("/")[-1] == f"finger_r_link{4 * (j + 1)}")
        for j in range(5)
    ]

    def sample():
        data = solver.mjw_data
        count = int(data.nacon.numpy()[0])
        pairs = data.contact.geom.numpy()[:count]
        addresses = data.contact.efc_address.numpy()[:count]
        positions = data.contact.pos.numpy()[:count]
        distances = data.contact.dist.numpy()[:count]
        world_ids = data.contact.worldid.numpy()[:count]
        poses = _tensor(env.scene["robot"].data.body_link_pose_w).detach().cpu().numpy()

        expected = np.zeros((NUM_ENVS, 5), dtype=np.float32)
        raw = np.zeros((NUM_ENVS, 5), dtype=np.int32)
        for pair, address, position, distance, world in zip(
            pairs, addresses, positions, distances, world_ids, strict=True,
        ):
            if np.any(pair < 0) or np.any(pair >= len(geom_body)) or not np.any(address >= 0):
                continue
            world = int(world)
            if not 0 <= world < NUM_ENVS:
                raise AssertionError(f"Invalid Newton contact world ID: {world}")
            body0, body1 = int(geom_body[pair[0]]), int(geom_body[pair[1]])
            if (body0 in objects) == (body1 in objects):
                continue
            for j, finger in enumerate(FINGERS):
                if (body0 in tips[finger]) + (body1 in tips[finger]) != 1:
                    continue
                raw[world, j] += 1
                pose = poses[world, tip_indices[j]]
                accepted, _ = active_contact_is_inside(
                    patches[finger], position, pose[:3], pose[3:7],
                    actual_pair=True, active_constraint=True,
                    signed_distance_m=float(distance),
                    max_signed_distance_m=PatchSelection().max_signed_distance_m,
                )
                expected[world, j] = max(expected[world, j], float(accepted))
        return expected, raw

    return sample


def _run_child():
    import argparse

    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.app import add_launcher_args, launch_simulation

    from isaaclab_tasks.utils import parse_env_cfg

    import contact_prediction_rl.tasks  # noqa: F401

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    launcher = parser.parse_args([])
    launcher.device = "cuda:0"

    env_cfg = parse_env_cfg(TASK, device=launcher.device, num_envs=NUM_ENVS)
    env_cfg.seed = 0
    env_cfg.scene.object.spawn.rigid_props = [sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)]
    # Replicate the proven diagnostic posture and holding gains, not the
    # production free-cube training conditions.
    env_cfg.scene.robot.init_state.joint_pos = {
        **{f"panda_joint{i + 1}": angle for i, angle in enumerate(ARM_JOINTS)},
        "finger_r_joint.*": 0.0,
    }
    for actuator in ("panda_shoulder", "panda_forearm"):
        env_cfg.scene.robot.actuators[actuator].stiffness = 4000.0
        env_cfg.scene.robot.actuators[actuator].damping = 160.0

    with launch_simulation(env_cfg, launcher):
        from isaaclab.envs import ManagerBasedRLEnv

        # Reuse the existing hand-action mapping; do not assume joint indices.
        from scripts.validate_fingertip_contacts import _closed_action

        env = ManagerBasedRLEnv(cfg=env_cfg)
        try:
            env.reset()
            origins = _tensor(env.scene.env_origins)
            cube = env.scene["object"]
            pose = _tensor(cube.data.default_root_pose).clone()
            positions = torch.tensor(CUBE_POSITIONS, device=env.device, dtype=pose.dtype)
            pose[:, :3] = positions + origins
            cube.write_root_pose_to_sim(pose)
            cube.write_root_velocity_to_sim(torch.zeros(NUM_ENVS, 6, device=env.device, dtype=pose.dtype))
            actual = _tensor(cube.data.root_pos_w) - origins
            torch.testing.assert_close(actual, positions, atol=1e-4, rtol=0)

            targets = torch.cat([_closed_action(env, finger) for finger in FINGERS], dim=0)
            targets = torch.cat([targets, torch.zeros((1, 20), device=env.device)], dim=0)
            opened = torch.zeros_like(targets)
            assert targets.shape == (NUM_ENVS, 20)
            sample_reference = _make_reference(env)

            def step(actions):
                with torch.inference_mode():
                    observations, _, terminated, truncated, _ = env.step(actions)
                contact = _tensor(observations["contact"]).detach().cpu().numpy()
                assert _tensor(observations["policy"]).shape == (NUM_ENVS, 40)
                assert contact.shape == (NUM_ENVS, 5)
                assert np.all((contact == 0) | (contact == 1)), contact
                expected, raw = sample_reference()
                np.testing.assert_array_equal(contact, expected, err_msg="GPU production contact labels differ from independent Newton contact classification")
                assert not bool(torch.any(terminated | truncated)), "Unexpected reset in contact-validation fixture"
                return contact, raw

            # No contacts in the open-hand baseline, including the control.
            for _ in range(120):
                bits, raw = step(opened)
                assert not bits.any(), f"Initial fixture is already contacting the cube: {bits}"

            positive_steps = np.zeros(NUM_ENVS, dtype=np.int32)
            held = torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.device)
            held_targets = opened.clone()
            for t in range(240):
                alpha = min((t + 1) / 90, 1.0)
                candidate = torch.lerp(opened, targets, alpha)
                commands = torch.where(held[:, None], held_targets, candidate)
                bits, raw = step(commands)
                positive_steps += bits.sum(axis=1).astype(np.int32)
                for env_id in range(5):
                    if not held[env_id] and raw[env_id, env_id] > 0:
                        held[env_id] = True
                        held_targets[env_id] = commands[env_id]
                for env_id in range(5):
                    assert bits[env_id].sum() == bits[env_id, env_id], (
                        f"Other finger unexpectedly labelled in environment {env_id}: {bits[env_id]}"
                    )
                assert not bits[5].any(), f"Contact leaked into no-contact environment: {bits[5]}"

            assert np.all(positive_steps[:5] >= 5), f"Missing live positive labels (thumb/index/middle/ring/little): {positive_steps.tolist()}"
            assert positive_steps[5] == 0, "No-contact control produced a positive label"

            # Reopen each hand and demand the final 10 sampled frames clear.
            final = []
            for t in range(180):
                alpha = min((t + 1) / 90, 1.0)
                actions = torch.lerp(held_targets, opened, alpha)
                bits, _ = step(actions)
                if t >= 170:
                    final.append(bits)
            assert not np.stack(final).any(), f"Contacts did not clear on release: {final[-1]}"
            print("MULTI_ENV_CONTACT_OK fingers=thumb,index,middle,ring,little control=clear release=clear", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    if sys.argv[1:] != ["--child"]:
        raise SystemExit("Run with pytest: uv run pytest -m gpu tests/test_contact_multi_env.py -v")
    _run_child()
