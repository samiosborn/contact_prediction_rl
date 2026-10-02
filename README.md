# contact_prediction_rl

- Reinforcement-learning experiments for studying whether auxiliary contact prediction improves dexterous manipulation.
- The initial task uses a ROBOTIS HX5-D20 right hand in Isaac Lab to manipulate an object in simulation.
- The policy is trained with PPO while an auxiliary head predicts binary fingertip-object contact from the same learned representation.
- The main comparison is baseline PPO against PPO with auxiliary fingertip contact prediction.
- Contact is used as privileged supervision during training but is not provided as an input to the deployed policy.

## Environment

- The registered Isaac Lab environment is: ContactPredictionRl-Reorientation-HX5-D20-Right

- The simulated hand has 20 actuated joints.
- Five object-specific contact signals are defined for:
  - thumb
  - index
  - middle
  - ring
  - little
- These signals are derived from simulator contact forces at the corresponding fingertip links.

## Setup

### Prerequisites

- Linux x86_64 with an NVIDIA GPU and driver 580.65.06 or later (required by the CUDA 13 torch builds; the locked torch is the CUDA 12.8 build for x86_64 Linux).
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Internet access on first install (PyPI, `pypi.nvidia.com`, `download.pytorch.org`).
- No Isaac Lab checkout is needed. The default configuration uses the Newton backend and does not require Isaac Sim.

### Install

```bash
git clone https://github.com/samiosborn/contact_prediction_rl.git
cd contact_prediction_rl
uv sync
```

`uv sync` installs the locked environment from `pyproject.toml` and `uv.lock`, including Isaac Lab from the NVIDIA package index.

Optional Isaac Sim (large download, needed only for PhysX/Kit features): `uv sync --extra isaacsim`

### Tested versions

| Component | Version |
|---|---|
| Python | 3.12 |
| Isaac Lab | 3.0.0rc1 (NVIDIA wheel; code identical to the `v3.0.0-EA` tag) |
| Isaac Sim | not installed by default (`--extra isaacsim` pins 6.1.0.0) |
| torch | 2.11.0+cu128 |
| Newton / mujoco-warp | 1.5.2 / 3.11.0 |
| rsl-rl-lib | 5.4.1 |
| NVIDIA driver / GPU | 595.91 / RTX 5080 Laptop |

Check the installed versions against the pins in `pyproject.toml`: `uv run python scripts/check_versions.py`

Isaac Lab 3.0.0 final is not yet published to a package index. When it is, change the `isaaclab[rsl-rl]==3.0.0rc1` pins in `pyproject.toml` and re-run `uv lock`.

### Reading Isaac Lab source (optional)

A separate clone is only useful for reading or debugging Isaac Lab source. This project never imports from it:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git ~/Apps/IsaacLab
git -C ~/Apps/IsaacLab checkout v3.0.0-EA
```

## Run

- List the registered environments: `uv run python scripts/list_envs.py --show_presets`

- Train (headless is the default in Isaac Lab 3.0):
  `uv run python scripts/train.py --task ContactPredictionRl-Reorientation-HX5-D20-Right --num_envs 4096`

- Short check: add `--num_envs 64 --max_iterations 3`

- Play back a checkpoint:
  `uv run python scripts/play.py --task ContactPredictionRl-Reorientation-HX5-D20-Right --num_envs 16 --checkpoint logs/rsl_rl/<experiment>/<run>/model_<n>.pt`

- Run the HX5-D20 development test:
  `uv run python scripts/test_hx5.py --num_envs 1 --viz newton_gl physics=newton_mjwarp`

`scripts/train.py` and `scripts/play.py` import `contact_prediction_rl.tasks` (which registers the environments) and then call Isaac Lab's unified `isaaclab_rl.entrypoints` CLI, as Isaac Lab's own scripts do.

## Project structure

contact_prediction_rl/
├── scripts/
│   ├── check_versions.py
│   ├── list_envs.py
│   ├── play.py
│   ├── smoke_envs.py
│   ├── test_hx5.py
│   └── train.py
├── src/
│   └── contact_prediction_rl/
│       ├── assets/
│       │   └── data/
│       │       └── hx5_d20/
│       │           ├── left/
│       │           └── right/
│       └── tasks/
│           └── reorientation/
│               ├── mdp/
│               └── config/
│                   └── hx5_d20_right/
├── tests/
├── pyproject.toml
└── uv.lock

- `tasks/reorientation/mdp` contains task-level MDP components.
- `tasks/reorientation/config/hx5_d20_right` contains the HX5-D20 scene, environment configuration, task registration and PPO configuration.
- Robot assets are stored under `assets/data/hx5_d20`.
- The right-hand model currently includes the generated URDF and USD used by Isaac Lab.

## Development

- Run tests without the simulator: `uv run pytest -m "not gpu"`

- Run the GPU smoke test (builds every registered task with 4 envs and steps it): `uv run pytest -m gpu`

- Run everything: `uv run pytest`

- Run formatting and code-quality hooks with: uv run pre-commit run --all-files

- Local editor configuration, virtual environments, logs, training outputs and generated caches are excluded from version control.
