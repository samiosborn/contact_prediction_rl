# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Instantiate one registered task with a few envs and step it; used by tests/test_smoke_envs.py."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=5)
add_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()

import torch  # noqa: E402

import contact_prediction_rl.tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, overrides=hydra_overrides)
    with launch_simulation(env_cfg, args_cli):
        from isaaclab.envs import ManagerBasedRLEnv

        env = ManagerBasedRLEnv(cfg=env_cfg)
        env.reset()
        for _ in range(args_cli.steps):
            with torch.inference_mode():
                env.step(torch.zeros(env.action_space.shape, device=env.device))
        env.close()
    print(f"SMOKE_OK {args_cli.task}")


if __name__ == "__main__":
    main()
