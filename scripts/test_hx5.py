import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Manual motion test for the HX5-D20 right hand.")
parser.add_argument("--num_envs", type=int, default=1)

add_launcher_args(parser)

args_cli, hydra_overrides = parser.parse_known_args()

import torch

import contact_prediction_rl.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


TASK = "ContactPredictionRl-Reorientation-HX5-D20-Right"


# Target joint positions in radians, keyed by URDF joint name.
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
    "finger_r_joint6": 1.4,
    "finger_r_joint7": 1.1,
    "finger_r_joint8": 1.1,

    "finger_r_joint9": 0.0,
    "finger_r_joint10": 1.4,
    "finger_r_joint11": 1.1,
    "finger_r_joint12": 1.1,

    "finger_r_joint13": 0.0,
    "finger_r_joint14": 1.4,
    "finger_r_joint15": 1.1,
    "finger_r_joint16": 1.1,

    "finger_r_joint17": 0.0,
    "finger_r_joint18": 1.4,
    "finger_r_joint19": 1.1,
    "finger_r_joint20": 1.1,
}


def pose_to_action(robot, pose, device):
    action = torch.zeros((1, len(robot.joint_names)), device=device)

    for action_idx, joint_name in enumerate(robot.joint_names):
        action[0, action_idx] = pose[joint_name]

    return action



TIP_BODY_NAMES = {
    "thumb": "finger_r_link4",
    "index": "finger_r_link8",
    "middle": "finger_r_link12",
    "ring": "finger_r_link16",
    "little": "finger_r_link20",
}


def run_pose(env, robot, obj, action, steps, contacts):
    max_filtered_forces = {name: 0.0 for name in contacts}
    max_net_forces = {name: 0.0 for name in contacts}

    for _ in range(steps):
        with torch.inference_mode():
            env.step(action)

        for name, sensor in contacts.items():
            filtered = sensor.data.normal_force_matrix_w.torch
            filtered_mag = torch.linalg.vector_norm(filtered, dim=-1)

            net = sensor.data.net_forces_w.torch
            net_mag = torch.linalg.vector_norm(net, dim=-1)

            max_filtered_forces[name] = max(
                max_filtered_forces[name],
                float(filtered_mag.max().item()),
            )

            max_net_forces[name] = max(
                max_net_forces[name],
                float(net_mag.max().item()),
            )

    print("Max filtered cube forces [N]:", max_filtered_forces)
    print("Max total fingertip forces [N]:", max_net_forces)

    cube_pos = obj.data.root_pos_w.torch[0].cpu()

    print("Cube position:", cube_pos.numpy())
    print("Fingertip positions:")

    for name, body_name in TIP_BODY_NAMES.items():
        body_idx = robot.body_names.index(body_name)
        tip_pos = robot.data.body_pos_w.torch[0, body_idx].cpu()

        distance = torch.linalg.vector_norm(tip_pos - cube_pos)

        print(
            f"  {name:6s}: "
            f"pos={tip_pos.numpy()} "
            f"distance_to_cube={distance.item():.4f} m"
        )



def main():
    env_cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        overrides=hydra_overrides,
    )

    with launch_simulation(env_cfg, args_cli):
        from isaaclab.envs import ManagerBasedRLEnv

        env = ManagerBasedRLEnv(cfg=env_cfg)
        robot = env.scene["robot"]
        obj = env.scene["object"]

        contacts = {
            "thumb": env.scene["thumb_contact"],
            "index": env.scene["index_contact"],
            "middle": env.scene["middle_contact"],
            "ring": env.scene["ring_contact"],
            "little": env.scene["little_contact"],
        }

        print("\nResolved body names:")
        for i, name in enumerate(robot.body_names):
            print(f"{i:2d}: {name}")

        print("\nResolved joint names:")
        for i, name in enumerate(robot.joint_names):
            print(f"{i:2d}: {name}")

        env.reset()

        open_action = pose_to_action(robot, OPEN_POSE, env.device)
        half_action = pose_to_action(robot, HALF_CLOSED_POSE, env.device)
        closed_action = pose_to_action(robot, CLOSED_POSE, env.device)

        print("\nOpen")
        run_pose(env, robot, obj, open_action, 120, contacts)

        print("Half closed")
        run_pose(env, robot, obj, half_action, 120, contacts)

        print("Closed")
        run_pose(env, robot, obj, closed_action, 180, contacts)

        print("Open")
        run_pose(env, robot, obj, open_action, 180, contacts)

        env.close()


if __name__ == "__main__":
    main()
