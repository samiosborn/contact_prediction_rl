"""GPU Newton hold/reset smoke test for snapshot-backed Reorientation."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import torch

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import parse_env_cfg

import contact_prediction_rl.tasks  # noqa: F401

TASK = "ContactPredictionRl-Reorientation-Train-HX5-D20-Right"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hold-steps", type=int, default=60)
    add_launcher_args(parser)
    args, _ = parser.parse_known_args()
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    with launch_simulation(cfg, args):
        from isaaclab.envs import ManagerBasedRLEnv

        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            retention_rates = []
            for trial in range(args.repeats):
                obs, _ = env.reset()
                assert obs["policy"].shape == (args.num_envs, 58), obs["policy"].shape
                assert obs["contact"].shape == (args.num_envs, 5), obs["contact"].shape
                retained = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
                positives = torch.zeros(5, device=env.device)
                for _ in range(args.hold_steps):
                    with torch.inference_mode():
                        obs, rewards, terminated, truncated, _ = env.step(torch.zeros((args.num_envs, 20), device=env.device))
                    assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rewards).all()
                    assert obs["contact"].shape == (args.num_envs, 5)
                    assert torch.all((obs["contact"] == 0) | (obs["contact"] == 1))
                    retained &= ~terminated & ~truncated
                    positives += obs["contact"].float().mean(dim=0)
                retention = float(retained.float().mean().item())
                retention_rates.append(retention)
                rates = positives / args.hold_steps
                if rates[0] < 0.05 or rates[1:].max() < 0.05:
                    raise RuntimeError("No sustained thumb plus opposing fingertip contacts in reset hold")
                print(f"RESET_HOLD trial={trial} retained={retention:.3f} "
                      f"finger_contact_rates={(positives / args.hold_steps).tolist()}", flush=True)
            print(f"RESET_SUMMARY average_retained={sum(retention_rates) / len(retention_rates):.3f} "
                  f"min_retained={min(retention_rates):.3f}", flush=True)
            if min(retention_rates) < 0.95:
                raise RuntimeError("Stable-grasp reset retention below 95%; do not start PPO training")
            print("REORIENTATION_RESET_OK", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
