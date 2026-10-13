# scripts/smoke_envs.py

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

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import contact_prediction_rl.tasks  # noqa: E402, F401


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, overrides=hydra_overrides)
    with launch_simulation(env_cfg, args_cli):
        from isaaclab.envs import ManagerBasedRLEnv

        env = ManagerBasedRLEnv(cfg=env_cfg)
        obs, _ = env.reset()
        contact_samples = []
        check_contact = args_cli.task == "ContactPredictionRl-Reorientation-HX5-D20-Right"
        if check_contact:
            assert obs["policy"].shape == (args_cli.num_envs, 40), obs["policy"].shape
            assert obs["contact"].shape == (args_cli.num_envs, 5), obs["contact"].shape
            assert env.action_space.shape[-1] == 20, env.action_space.shape
        for _ in range(args_cli.steps):
            with torch.inference_mode():
                obs, _, _, _, _ = env.step(torch.zeros(env.action_space.shape, device=env.device))
                if check_contact:
                    contact = obs["contact"]
                    assert contact.shape == (args_cli.num_envs, 5), contact.shape
                    assert torch.all((contact == 0) | (contact == 1))
                    contact_samples.append(contact)
        if contact_samples:
            rates = torch.cat(contact_samples, dim=0).float().mean(dim=0).tolist()
            finger_names = ("thumb", "index", "middle", "ring", "little")
            print(
                f"CONTACT_RATE overall={sum(rates) / len(rates):.4f} " + " ".join(f"{name}={rate:.4f}" for name, rate in zip(finger_names, rates, strict=True)),
                flush=True,
            )
        env.close()
        # print before leaving the block: closing Isaac Sim (Kit-based tasks) ends the process
        print(f"SMOKE_OK {args_cli.task}", flush=True)


if __name__ == "__main__":
    main()
